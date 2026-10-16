"""Prospectively locked, staged frozen-policy breadth study in Linux.

This successor imports unchanged pilot helpers. Source learning, known-domain
evaluation and reserved collection are separate exclusive stages. No stage can
resume or replace an earlier attempt, and no gradient adaptation is implemented.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import time

import generalization_pretraining as helper
from generalization_study_manifest import manifest, validate_manifest

PROJECT = helper.PROJECT
NATIVE_COMMAND = helper.NATIVE_COMMAND
NEW_FILES = (
    "training/generalization_study.py", "training/generalization_study_manifest.py",
    "docs/GENERALIZATION_STUDY_EXPERIMENT.md",
    "tests/check_generalization_study_manifest.py", "tests/check_generalization_study_driver.py",
)
canonical = helper.canonical
copy_json = helper.copy_json
sha256_file = helper.sha256_file
write_new = helper.write_new


class StudySourceSampler:
    """Source-only independent shuffled cycles; starts and experience differ."""
    def __init__(self, study, seed, arm):
        validate_manifest(study)
        if seed not in study["source_seeds"] or arm not in study["source_arms"]:
            raise ValueError("Source arm/seed is outside the prospective declaration")
        self.seed, self.arm = seed, arm
        self.curriculum_seed = seed + study["sampling"]["seed_offset"]
        self.random = random.Random(self.curriculum_seed)
        self.cases = copy_json(study["source_cases"][arm])
        self.order, self.position, self.draws = [], 0, 0
        self.counts = [0] * len(self.cases)
        self.digest = hashlib.sha256()
        self.latest_index = None

    def __call__(self):
        if self.position == len(self.order):
            self.order = list(range(len(self.cases)))
            self.random.shuffle(self.order)
            self.position = 0
        index = self.order[self.position]
        self.position += 1
        self.draws += 1
        self.latest_index = index
        self.counts[index] += 1
        self.digest.update(f"{index}\n".encode("ascii"))
        return copy_json(self.cases[index])

    def exposure_report(self):
        return {"schema": "shuffled_source_cycles_v1", "source_arm": self.arm,
                "source_seed": self.seed, "curriculum_seed": self.curriculum_seed,
                "source_count": len(self.cases), "draws": self.draws,
                "completed_cycles": self.draws // len(self.cases),
                "current_cycle_position": self.draws % len(self.cases),
                "draw_counts": self.counts.copy(), "draw_index_sha256": self.digest.hexdigest()}


def source_hashes():
    hashes = helper.source_hashes()
    hashes.update({name: sha256_file(PROJECT / name) for name in NEW_FILES})
    return dict(sorted(hashes.items()))


def safe_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Retained evidence path escapes its study directory")
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def bound_json(root, reference):
    path = safe_path(root, reference["path"])
    if sha256_file(path) != reference["sha256"]:
        raise ValueError(f"Retained evidence changed: {reference['path']}")
    return read_json(path)


def reference(root, path):
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def lock(output):
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(f"Refusing existing output: {root}")
    helper.require_container()
    study = manifest()
    validate_manifest(study)
    hashes = source_hashes()
    protocol = {"schema": "generalization_study_lock_protocol_v1", "study": study,
                "status": "locked_before_source_learning", "locked_at_utc": datetime.now(timezone.utc).isoformat(),
                "runtime": helper.runtime_record(), "source_sha256": hashes,
                "rhythm_semantics": {"rhythm_2s": {"period_decisions": 20, "sign_phase_decisions": 10},
                                     "rhythm_4s": {"period_decisions": 40, "sign_phase_decisions": 20},
                                     "starts": "negative", "initialization": "Native invalid-sensor governor emits zero; time is counted"},
                "release": "All6 exact source releases sealed, matching initial tensors per seed, then global source competence before reserved construction",
                "recording": "Native request tapes/response commitments and compact local telemetry; no default full physical snapshots",
                "independent_auditor": "Post-collection tool version is bound in its own receipt, outside this action-generating lock"}
    root.mkdir(parents=True, exist_ok=False)
    for name in hashes:
        target = root / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((PROJECT / name).read_bytes())
    write_new(root / "protocol.json", protocol)
    write_new(root / "lock.json", {"schema": "generalization_study_lock_v1",
              "protocol_sha256": sha256_file(root / "protocol.json"),
              "source_sha256": hashes, "runtime": protocol["runtime"],
              "physical_interactions": 0, "reserved_case_construction_requests": 0})
    print(canonical({"locked": str(root), "protocol_sha256": sha256_file(root / "protocol.json")}), flush=True)


def read_lock(output):
    helper.require_container()
    root = Path(output).resolve()
    protocol, receipt = read_json(root / "protocol.json"), read_json(root / "lock.json")
    if protocol.get("schema") != "generalization_study_lock_protocol_v1":
        raise ValueError("Unexpected study lock schema")
    digest = sha256_file(root / "protocol.json")
    if digest != receipt["protocol_sha256"]:
        raise ValueError("Prospective protocol changed")
    validate_manifest(protocol["study"])
    if source_hashes() != protocol["source_sha256"] or receipt["source_sha256"] != protocol["source_sha256"]:
        raise ValueError("Action-generating source changed since study lock")
    for name, expected in protocol["source_sha256"].items():
        if sha256_file(root / "source_snapshot" / name) != expected:
            raise ValueError(f"Archived source changed: {name}")
    if canonical(helper.runtime_record()) != canonical(protocol["runtime"]):
        raise ValueError("Runtime differs from prospective study lock")
    if canonical(receipt["runtime"]) != canonical(protocol["runtime"]):
        raise ValueError("Runtime lock binding differs")
    return root, protocol, digest


def new_stage(root, name, protocol_hash):
    folder = root / name
    folder.mkdir(exist_ok=False)
    write_new(folder / "started.json", {"schema": "generalization_study_stage_start_v1",
              "stage": name, "protocol_sha256": protocol_hash,
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "reserved_case_construction_requests": 0})
    return folder


def stage_record(root, folder, protocol, protocol_hash, status, **fields):
    hashes, runtime = source_hashes(), helper.runtime_record()
    unchanged = hashes == protocol["source_sha256"] and canonical(runtime) == canonical(protocol["runtime"])
    record = {"schema": "generalization_study_stage_v1", "stage": folder.name,
              "protocol_sha256": protocol_hash, "status": status if unchanged else "integrity_failed",
              "provenance_unchanged": unchanged, "source_sha256_after_collection": hashes,
              "runtime_after_collection": runtime, **fields}
    write_new(folder / "stage.json", record)
    write_new(folder / "receipt.json", {"schema": "generalization_study_stage_receipt_v1",
              "protocol_sha256": protocol_hash, "stage": reference(root, folder / "stage.json")})
    print(canonical({"stage": folder.name, "status": record["status"],
                     "stage_sha256": sha256_file(folder / "stage.json")}), flush=True)
    return record


def read_stage(root, name, protocol_hash):
    receipt = read_json(root / name / "receipt.json")
    if receipt["protocol_sha256"] != protocol_hash:
        raise ValueError("Stage receipt/protocol binding differs")
    record = bound_json(root, receipt["stage"])
    if record["protocol_sha256"] != protocol_hash or not record["provenance_unchanged"]:
        raise ValueError("Stage provenance failed")
    return record


def bootstrap_cases(cases, slots):
    if not cases or slots <= 0:
        raise ValueError("Nonempty source cases and positive vector size required")
    return [copy_json(cases[index % len(cases)]) for index in range(slots)]


def train_source(root, folder, study, protocol_hash, seed, arm, dependencies):
    torch, np, _, BaseCallback, _, base, _ = dependencies
    budget, ppo = study["source_budget"], study["ppo"]
    attempt = folder / f"source-{arm}-{seed}"
    attempt.mkdir(exist_ok=False)
    (attempt / "traces").mkdir()
    write_new(attempt / "started.json", {"source_arm": arm, "seed": seed,
              "protocol_sha256": protocol_hash, "reserved_case_construction_requests": 0})
    log = (attempt / "learning.jsonl").open("x", encoding="utf-8")
    start = time.monotonic()
    class LearningLog(BaseCallback):
        def _on_step(self):
            return True
        def _on_rollout_end(self):
            row = {"event": "rollout_collected", "decisions": self.num_timesteps,
                   "preceding_epochs": self.model._n_updates, "wall_s": time.monotonic() - start}
            log.write(canonical(row) + "\n")
            log.flush()
            print(canonical({"source_progress": {"arm": arm, "seed": seed, **row}}), flush=True)
        def _write_update(self, final=False):
            log.write(canonical({"event": "update_completed", "decisions": self.num_timesteps,
                      "optimization_epochs": self.model._n_updates,
                      "metrics": helper.scalar_metrics(self.model, np), "final": final}) + "\n")
            log.flush()
        def _on_rollout_start(self):
            if self.model._n_updates:
                self._write_update()
        def _on_training_end(self):
            self._write_update(True)
    sampler = StudySourceSampler(study, seed, arm)
    env = model = None
    initial = None
    error = None
    cancelled = False
    learning_start = None
    initialization_s = learning_s = 0.
    try:
        with helper.wall_deadline(study["wall_caps"]["source"]):
            env_type = helper.make_instrumented_env_class(base)
            env = env_type(bootstrap_cases(study["source_cases"][arm], budget["n_envs"]),
                           source_cases=study["source_cases"][arm], case_sampler=sampler,
                           horizon=budget["horizon"], record=False, command=NATIVE_COMMAND)
            if env.num_envs != budget["n_envs"]:
                raise ValueError("Actual vector size differs from equal-budget declaration")
            model = helper.build_model(env, ppo, seed, dependencies=dependencies)
            if model.policy.optimizer.state:
                raise ValueError("Source optimizer is not fresh")
            model.save(attempt / "policy_initial.zip")
            initial = {"policy_path": "policy_initial.zip", "policy_sha256": sha256_file(attempt / "policy_initial.zip"),
                       "parameter_sha256": helper.parameter_digest(model.policy), "weight_updates": 0, "source_decisions": 0}
            write_new(attempt / "initial-checkpoint.json", initial)
            initialization_s = time.monotonic() - start
            learning_start = time.monotonic()
            model.learn(total_timesteps=budget["decisions"], callback=LearningLog())
    except (Exception, KeyboardInterrupt) as failure:
        error = f"{type(failure).__name__}: {failure}"
        cancelled = isinstance(failure, KeyboardInterrupt)
    finally:
        if learning_start is not None:
            learning_s = time.monotonic() - learning_start
        close_error = helper.close_env(env)
        log.close()
    wall_s = time.monotonic() - start
    actual = env.total_interactions if env else 0
    exact = (env is not None and env.bridge.last_error_response is None and env.unresolved_request is None
             and sum(item["decisions"] for item in env.source_exposure) == actual
             and sum(item["native_steps"] for item in env.source_exposure) == env.total_native_steps)
    finite = model is not None and all(bool(torch.isfinite(value).all()) for value in model.policy.parameters())
    steps = ([int(value["step"].item()) for value in model.policy.optimizer.state.values() if "step" in value]
             if model is not None else [])
    expected_updates = budget["decisions"] // (budget["n_steps"] * budget["n_envs"])
    expected_epochs = expected_updates * ppo["n_epochs"]
    expected_steps = budget["decisions"] // ppo["batch_size"] * ppo["n_epochs"]
    updates_exact = (model is not None and model._n_updates == expected_epochs
                     and min(steps, default=0) == max(steps, default=0) == expected_steps)
    complete = (error is None and close_error is None and exact and finite and initial is not None
                and actual == budget["decisions"] and model.num_timesteps == actual and updates_exact
                and wall_s <= study["wall_caps"]["source"])
    result = {"schema": "generalization_study_source_result_v1", "protocol_sha256": protocol_hash,
              "source_arm": arm, "seed": seed, "complete": complete, "error": error, "close_error": close_error,
              "cancelled": cancelled, "requested_decisions": budget["decisions"], "actual_decisions": actual,
              "native_steps": env.total_native_steps if env else 0, "n_envs": env.num_envs if env else 0,
              "initialization_wall_s": initialization_s, "learning_wall_s": learning_s, "source_wall_s": wall_s,
              "wall_cap_exceeded": wall_s > study["wall_caps"]["source"], "parameters_finite": finite,
              "exposure_exact": exact, "update_counts_exact": updates_exact,
              "native_failure": env.bridge.last_error_response if env else None,
              "unresolved_request": env.unresolved_request if env else None,
              "safety_stops": env.safety_stops if env else 0, "episodes": env.episode_count if env else 0,
              "source_schedule": sampler.exposure_report(),
              "per_source_case_exposure": env.source_exposure if env else [], "initial_checkpoint": initial,
              "expected_rollout_updates": expected_updates, "expected_optimization_epochs": expected_epochs,
              "expected_optimizer_steps": expected_steps, "reserved_case_construction_requests": 0}
    if env is not None:
        result["request_tape"] = helper.tape_receipt(env, attempt / "traces" / "source-requests.json.gz")
    if model is not None:
        policy_name = "policy.zip" if complete else "policy_partial.zip"
        model.save(attempt / policy_name)
        result.update({"policy_path": policy_name, "policy_sha256": sha256_file(attempt / policy_name),
                       "parameter_sha256": helper.parameter_digest(model.policy),
                       "parameters": sum(value.numel() for value in model.policy.parameters()),
                       "optimization_epochs": model._n_updates, "rollout_updates": model._n_updates / ppo["n_epochs"],
                       "optimizer_steps_min": min(steps, default=0), "optimizer_steps_max": max(steps, default=0),
                       "final_update_metrics": helper.scalar_metrics(model, np)})
    write_new(attempt / "result.json", result)
    print(canonical({"source_result": {"arm": arm, "seed": seed, "complete": complete,
                     "actual_decisions": actual, "error": error}}), flush=True)
    return result, reference(root, attempt / "result.json")


def sources(output):
    root, protocol, protocol_hash = read_lock(output)
    folder = new_stage(root, "sources", protocol_hash)
    study = protocol["study"]
    dependencies = helper.learning_dependencies()
    results, references = [], []
    for seed in study["source_seeds"]:
        for arm in study["source_arms"]:
            try:
                result, ref = train_source(root, folder, study, protocol_hash, seed, arm, dependencies)
            except Exception as failure:
                # A finalization failure must not silently replace its prefix or
                # prevent the remaining independent declared source attempts.
                attempt = folder / f"source-{arm}-{seed}"
                attempt.mkdir(exist_ok=True)
                prefix = [reference(root, path) for path in sorted(attempt.rglob("*")) if path.is_file()]
                result = {"schema": "generalization_study_source_failure_v1", "protocol_sha256": protocol_hash,
                          "source_arm": arm, "seed": seed, "complete": False,
                          "error": f"{type(failure).__name__}: {failure}", "cancelled": False,
                          "actual_decisions": 0, "native_steps": 0, "exposure_exact": False,
                          "counts_interpretation": "Zero is the known lower bound; native advancement may have occurred before finalization failed",
                          "unknown_exposure": True, "initial_checkpoint": None,
                          "retained_prefix": prefix, "reserved_case_construction_requests": 0}
                failure_path = attempt / "source-failure.json"
                write_new(failure_path, result)
                ref = reference(root, failure_path)
            results.append(result)
            references.append(ref)
            if result["cancelled"]:
                break
        if results[-1]["cancelled"]:
            break
    initial_matches = []
    for seed in study["source_seeds"]:
        pair = [result for result in results if result["seed"] == seed]
        matches = (len(pair) == 2 and all(item["initial_checkpoint"] is not None for item in pair)
                   and pair[0]["initial_checkpoint"]["parameter_sha256"] == pair[1]["initial_checkpoint"]["parameter_sha256"])
        initial_matches.append({"seed": seed, "matches": matches})
    complete = len(results) == 6 and all(item["complete"] for item in results) and all(item["matches"] for item in initial_matches)
    seal = {"schema": "generalization_study_source_seal_v1", "protocol_sha256": protocol_hash,
            "operational_complete": complete, "source_results": references,
            "initial_identity_by_seed": initial_matches,
            "sealed_before_any_evaluation": True, "reserved_case_construction_requests": 0}
    write_new(folder / "sealed-checkpoints.json", seal)
    return stage_record(root, folder, protocol, protocol_hash, "complete" if complete else "incomplete",
                        source_results=references, seal=reference(root, folder / "sealed-checkpoints.json"),
                        initial_identity_by_seed=initial_matches, operational_complete=complete,
                        actual_decisions=sum(item["actual_decisions"] for item in results),
                        native_steps=sum(item["native_steps"] for item in results),
                        exposure_exact=all(item.get("exposure_exact", False) for item in results),
                        declared_source_attempts=6, collected_source_attempts=len(results),
                        reserved_case_construction_requests=0)


def source_release(root, protocol_hash):
    stage = read_stage(root, "sources", protocol_hash)
    seal = bound_json(root, stage["seal"])
    if stage["status"] != "complete" or not stage["operational_complete"] or not seal["operational_complete"]:
        return None, "At least one source attempt is incomplete or initial parameters differ"
    if not seal["sealed_before_any_evaluation"] or len(seal["source_results"]) != 6:
        return None, "Six source releases were not sealed before evaluation"
    released = {}
    for ref in seal["source_results"]:
        result = bound_json(root, ref)
        if not result["complete"] or result["protocol_sha256"] != protocol_hash:
            return None, "Source result is incomplete or belongs to another protocol"
        attempt = safe_path(root, ref["path"]).parent
        for checkpoint in (result, result["initial_checkpoint"]):
            if sha256_file(attempt / checkpoint["policy_path"]) != checkpoint["policy_sha256"]:
                raise ValueError("Sealed checkpoint bytes changed")
        tape = result["request_tape"]
        if sha256_file(attempt / "traces" / tape["path"]) != tape["sha256"]:
            raise ValueError("Source request tape changed")
        key = (result["source_arm"], result["seed"])
        if key in released:
            raise ValueError("Duplicate sealed source")
        released[key] = {"result": result, "folder": attempt, "reference": ref}
    for seed in (101, 202, 303):
        pair = [released.get((arm, seed)) for arm in ("pooled", "narrow")]
        if any(item is None for item in pair) or pair[0]["result"]["initial_checkpoint"]["parameter_sha256"] != pair[1]["result"]["initial_checkpoint"]["parameter_sha256"]:
            return None, "Per-seed pooled/narrow initial tensor identity failed"
    return released, None


def fixed_action(mode, decision):
    if mode in ("quiet", "negative", "positive"):
        return {"quiet": 1, "negative": 0, "positive": 2}[mode]
    if mode not in ("rhythm_2s", "rhythm_4s"):
        raise ValueError("Unknown prospective fixed controller")
    phase = {"rhythm_2s": 10, "rhythm_4s": 20}[mode]
    return 0 if (decision // phase) % 2 == 0 else 2


def trial(root, folder, study, protocol_hash, released, partition, index, controller, seed, dependencies):
    _, np, PPO, _, _, base, _ = dependencies
    cases_key = {"development": "development_cases", "narrow_development": "narrow_development_cases", "reserved": "reserved_cases"}[partition]
    case_id = study["case_ids"][partition][index]
    trial_id = f"{partition}--{case_id}--{controller}--{seed if seed is not None else 'fixed'}"
    if re.fullmatch(r"[A-Za-z0-9._-]+", trial_id) is None:
        raise ValueError("Unsafe trial identifier")
    attempt = folder / "trials" / trial_id
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "traces").mkdir()
    write_new(attempt / "started.json", {"protocol_sha256": protocol_hash, "trial_id": trial_id,
              "partition": partition, "controller": controller, "seed": seed,
              "case_index": index, "held_out": partition == "reserved"})
    env = model = None
    checkpoint = None
    checkpoint_ref = None
    initial_hash = final_hash = None
    initial_local = None
    error = None
    cancelled = False
    action_seed = (study["action_rng"]["base"] + seed * study["action_rng"]["seed_multiplier"]
                   + study["action_rng"]["offsets"][partition] + index if seed is not None else None)
    start = time.monotonic()
    try:
        with helper.wall_deadline(study["wall_caps"]["trial"]):
            env_type = helper.make_instrumented_env_class(base)
            env = env_type([copy_json(study[cases_key][index])], record=False, stop_on_safety=True,
                           horizon=study["evaluation_budget"]["decisions"] + 1, command=NATIVE_COMMAND)
            if controller in ("pooled", "narrow", "untrained"):
                source = released[("pooled" if controller == "untrained" else controller, seed)]
                source_result = source["result"]
                checkpoint = source_result["initial_checkpoint"] if controller == "untrained" else source_result
                checkpoint_path = source["folder"] / checkpoint["policy_path"]
                if sha256_file(checkpoint_path) != checkpoint["policy_sha256"]:
                    raise ValueError("Frozen checkpoint changed")
                checkpoint_ref = reference(root, checkpoint_path)
                model = helper.build_model(env, study["ppo"], seed, target=True, dependencies=dependencies)
                saved = PPO.load(checkpoint_path, device="cpu")
                model.policy.load_state_dict(saved.policy.state_dict(), strict=True)
                del saved
                model.set_random_seed(action_seed)
                if model.policy.optimizer.state:
                    raise ValueError("Evaluation optimizer is not fresh")
                initial_hash = helper.parameter_digest(model.policy)
                if initial_hash != checkpoint["parameter_sha256"]:
                    raise ValueError("Frozen parameter identity differs from sealed checkpoint")
            observation = env.reset()
            initial_local = copy_json(env.latest[0]["observation"])
            for decision in range(study["evaluation_budget"]["decisions"]):
                action = (model.predict(observation, deterministic=False)[0] if model is not None else
                          np.asarray([fixed_action(controller, decision)], dtype=np.int64))
                observation, _, _, _ = env.step(action)
    except (Exception, KeyboardInterrupt) as failure:
        error = f"{type(failure).__name__}: {failure}"
        cancelled = isinstance(failure, KeyboardInterrupt)
    finally:
        close_error = helper.close_env(env)
    rows = env.telemetry if env else []
    metrics = helper.metrics_from_results(rows)
    exact = env is not None and env.bridge.last_error_response is None and env.unresolved_request is None
    if model is not None:
        final_hash = helper.parameter_digest(model.policy)
    unchanged = model is None if checkpoint is None else initial_hash == final_hash == checkpoint["parameter_sha256"]
    wall_s = time.monotonic() - start
    clock = wall_s > study["wall_caps"]["trial"]
    complete = (error is None and close_error is None and exact and unchanged and not clock
                and len(rows) == study["evaluation_budget"]["decisions"]
                and metrics["native_steps"] == study["evaluation_budget"]["decisions"] * 5
                and metrics["safety_stops"] == 0)
    requests = env.bridge.request_tape if env else []
    construction_requests = sum(len(request["cases"]) for request in requests if request["op"] == "configure")
    if env is not None and env.unresolved_request is not None and env.unresolved_request.get("op") == "configure":
        construction_requests += len(env.unresolved_request["cases"])
    result = {"schema": "generalization_study_trial_v1", "protocol_sha256": protocol_hash,
              "trial_id": trial_id, "partition": partition, "case_id": case_id, "case_index": index,
              "controller": controller, "seed": seed, "action_seed": action_seed,
              "held_out": partition == "reserved", "weight_updates": 0,
              "complete": complete, "safe": exact and metrics["safety_stops"] == 0 and error is None and close_error is None,
              "error": error, "close_error": close_error, "cancelled": cancelled,
              "exposure_exact": exact, "native_failure": env.bridge.last_error_response if env else None,
              "unresolved_request": env.unresolved_request if env else None,
              "metrics": metrics, "wall_s": wall_s, "wall_cap_exceeded": clock,
              "checkpoint": checkpoint_ref, "checkpoint_parameter_sha256": checkpoint["parameter_sha256"] if checkpoint else None,
              "initial_parameter_sha256": initial_hash, "final_parameter_sha256": final_hash,
              "frozen_weights_unchanged": unchanged, "initial_local_observation": initial_local,
              "windows": [{"requested_decisions": count, "complete": len(rows) >= count,
                           "metrics": helper.metrics_from_results(rows[:count])} for count in study["evaluation_budget"]["windows"]],
              "case_construction_requests": construction_requests,
              "reserved_case_construction_requests": construction_requests if partition == "reserved" else 0}
    if controller in ("rhythm_2s", "rhythm_4s"):
        phase = {"rhythm_2s": 10, "rhythm_4s": 20}[controller]
        result["fixed_schedule"] = {"period_decisions": 2 * phase, "sign_phase_decisions": phase,
                                    "starts": "negative", "formula": "action=0 if floor(zero_based_decision/sign_phase_decisions)%2==0 else2"}
    if env is not None:
        result["request_tape"] = helper.tape_receipt(env, attempt / "traces" / "requests.json.gz")
        helper.save_gzip_new(attempt / "traces" / "telemetry.json.gz",
                             {"schema": "generalization_development_telemetry_v1", "rows": rows})
        result["telemetry"] = {"path": "telemetry.json.gz", "sha256": sha256_file(attempt / "traces" / "telemetry.json.gz")}
    write_new(attempt / "result.json", result)
    print(canonical({"trial": trial_id, "complete": complete, "reward": metrics["native_reward"], "error": error}), flush=True)
    return result, reference(root, attempt / "result.json")


def trial_index(trials):
    result = {}
    for value in trials:
        key = (value["partition"], value["controller"], value["seed"], value["case_index"])
        if key in result:
            raise ValueError("Duplicate recorded trial condition")
        result[key] = value
    return result


def eligible(value):
    return value is not None and value["complete"] and value["safe"] and value["frozen_weights_unchanged"]


def contrast(a, b):
    return a["metrics"]["native_reward"] - b["metrics"]["native_reward"] if a is not None and b is not None else None


def evaluate_source_gate(study, trials):
    lookup = trial_index(trials)
    comparisons = []
    for partition, controller in (("development", "pooled"), ("narrow_development", "narrow")):
        for seed in study["source_seeds"]:
            for index, case_id in enumerate(study["case_ids"][partition]):
                trained = lookup.get((partition, controller, seed, index))
                quiet = lookup.get((partition, "quiet", None, index))
                gain = contrast(trained, quiet)
                valid = eligible(trained) and eligible(quiet)
                comparisons.append({"partition": partition, "case_id": case_id, "seed": seed,
                                    "controller": controller, "complete_safe_pair": valid,
                                    "trained_minus_quiet": gain, "passes": valid and gain > study["gates"]["source_margin"]})
    return {"schema": "generalization_study_source_gate_v1", "passes": len(comparisons) == 48 and all(row["passes"] for row in comparisons),
            "required_pairs": 48, "margin_strictly_greater_than": study["gates"]["source_margin"], "comparisons": comparisons,
            "untrained_controls_gate_source_competence": False, "reserved_case_construction_requests": 0}


def learning_diagnostics(study, trials, partition, controllers):
    lookup = trial_index(trials)
    rows = []
    for controller in controllers:
        for seed in study["source_seeds"]:
            for index, case_id in enumerate(study["case_ids"][partition]):
                trained = lookup.get((partition, controller, seed, index))
                initial = lookup.get((partition, "untrained", seed, index))
                rows.append({"controller": controller, "seed": seed, "case_id": case_id,
                             "trained_minus_untrained": contrast(trained, initial),
                             "complete_safe_pair": eligible(trained) and eligible(initial)})
    def reduction(group):
        all_values = [row["trained_minus_untrained"] for row in group if row["trained_minus_untrained"] is not None]
        complete_values = [row["trained_minus_untrained"] for row in group if row["complete_safe_pair"]]
        return {"realized_pairs": len(all_values), "planned_pairs": len(group),
                "mean_realized_trained_minus_untrained": statistics.mean(all_values) if len(all_values) == len(group) else None,
                "complete_safe_pairs": len(complete_values),
                "mean_complete_safe_trained_minus_untrained": statistics.mean(complete_values) if complete_values else None,
                "complete_safe_trained_wins": sum(value > 0. for value in complete_values)}
    return {"partition": partition, "comparisons": rows, **reduction(rows),
            "by_controller": {controller: reduction([row for row in rows if row["controller"] == controller])
                              for controller in controllers},
            "per_seed_controller": [{"seed": seed, "controller": controller,
                                     **reduction([row for row in rows if row["seed"] == seed and row["controller"] == controller])}
                                    for seed in study["source_seeds"] for controller in controllers],
            "interpretation": "All-case realized means retain unequal stopped prefixes and cannot establish competence; eligible-only means are labeled separately"}


def blocked_stage(root, folder, protocol, protocol_hash, reason):
    return stage_record(root, folder, protocol, protocol_hash, "blocked", reason=reason,
                        actual_decisions=0, native_steps=0, trial_results=[],
                        reserved_case_construction_requests=0, reserved_worlds_constructed=0)


def development(output):
    root, protocol, protocol_hash = read_lock(output)
    folder = new_stage(root, "development", protocol_hash)
    try:
        released, reason = source_release(root, protocol_hash)
    except (OSError, ValueError, KeyError) as error:
        return blocked_stage(root, folder, protocol, protocol_hash, f"Source release unavailable: {error}")
    if reason:
        return blocked_stage(root, folder, protocol, protocol_hash, reason)
    study = protocol["study"]
    dependencies = helper.learning_dependencies()
    trials, references = [], []
    cancelled = False
    for partition, controllers in (("development", ("pooled", "narrow", "untrained")),
                                   ("narrow_development", ("narrow", "untrained"))):
        # Shared quiet is fixed first for the whole partition; subsequent model
        # order is seed, case, then arm, as prospectively declared.
        for index in range(len(study["case_ids"][partition])):
            result, ref = trial(root, folder, study, protocol_hash, released, partition, index, "quiet", None, dependencies)
            trials.append(result)
            references.append(ref)
            if result["cancelled"]:
                cancelled = True
                break
        if cancelled:
            break
        for seed in study["source_seeds"]:
            for index in range(len(study["case_ids"][partition])):
                for controller in controllers:
                    result, ref = trial(root, folder, study, protocol_hash, released, partition, index, controller, seed, dependencies)
                    trials.append(result)
                    references.append(ref)
                    if result["cancelled"]:
                        cancelled = True
                        break
                if cancelled:
                    break
            if cancelled:
                break
        if cancelled:
            break
    gate = evaluate_source_gate(study, trials)
    if len(trials) != 148:
        gate["passes"] = False
    write_new(folder / "gate.json", {"protocol_sha256": protocol_hash, **gate})
    return stage_record(root, folder, protocol, protocol_hash, "complete" if len(trials) == 148 else "incomplete",
                        source_seal_sha256=sha256_file(root / "sources" / "sealed-checkpoints.json"),
                        trial_results=references, collected_trials=len(trials), planned_trials=148,
                        gate=reference(root, folder / "gate.json"), source_competence=gate["passes"],
                        learning_diagnostics=[learning_diagnostics(study, trials, "development", ("pooled", "narrow")),
                                              learning_diagnostics(study, trials, "narrow_development", ("narrow",))],
                        actual_decisions=sum(item["metrics"]["decisions"] for item in trials),
                        native_steps=sum(item["metrics"]["native_steps"] for item in trials),
                        reserved_case_construction_requests=0)


def challenge_release(root, protocol_hash, study):
    released, reason = source_release(root, protocol_hash)
    if reason:
        return None, reason
    stage = read_stage(root, "development", protocol_hash)
    if stage["status"] != "complete" or stage["collected_trials"] != 148:
        return None, "Known-domain collection is incomplete"
    if stage["source_seal_sha256"] != sha256_file(root / "sources" / "sealed-checkpoints.json"):
        return None, "Source seal changed after development"
    gate = bound_json(root, stage["gate"])
    trials = []
    for ref in stage["trial_results"]:
        result = bound_json(root, ref)
        if result["protocol_sha256"] != protocol_hash or result["weight_updates"] != 0:
            raise ValueError("Development trial protocol or frozen-weight boundary differs")
        attempt = safe_path(root, ref["path"]).parent
        for name in ("request_tape", "telemetry"):
            if name in result:
                path = safe_path(root, (attempt / "traces" / result[name]["path"]).relative_to(root))
                if sha256_file(path) != result[name]["sha256"]:
                    raise ValueError(f"Development {name} changed before reserved release")
            elif result["complete"]:
                raise ValueError("Complete development trial lacks its retained evidence")
        if result["checkpoint"] is not None:
            checkpoint_path = safe_path(root, result["checkpoint"]["path"])
            if sha256_file(checkpoint_path) != result["checkpoint"]["sha256"]:
                raise ValueError("Development checkpoint binding changed")
        trials.append(result)
    recomputed = evaluate_source_gate(study, trials)
    if not recomputed["passes"] or not gate["passes"] or not stage["source_competence"]:
        return None, "Global own-domain competence gate failed; reserved cases remain unexposed"
    if canonical({key: gate[key] for key in recomputed}) != canonical(recomputed):
        return None, "Recorded and recomputed source gate differ"
    return released, None


def evaluate_challenge(study, trials):
    lookup = trial_index(trials)
    comparisons, seed_means = [], []
    breadth_eligible = True
    useful_rows = []
    for seed in study["source_seeds"]:
        seed_rows = []
        for index, case_id in enumerate(study["case_ids"]["reserved"]):
            pooled = lookup.get(("reserved", "pooled", seed, index))
            narrow = lookup.get(("reserved", "narrow", seed, index))
            quiet = lookup.get(("reserved", "quiet", None, index))
            valid = eligible(pooled) and eligible(narrow) and eligible(quiet)
            breadth_eligible = breadth_eligible and valid
            gain = contrast(pooled, narrow)
            case = study["reserved_cases"][index]
            row = {"seed": seed, "case_id": case_id, "case_index": index,
                   "powered_hinge": case["assembly"]["powered_hinge"], "lamp_side": -1 if case["sun"]["position_m"][0] < 0 else 1,
                   "pooled_minus_narrow": gain, "complete_safe_pair_and_quiet": valid}
            comparisons.append(row)
            seed_rows.append(row)
            quiet_gain = contrast(pooled, quiet)
            useful_rows.append({"seed": seed, "case_id": case_id, "pooled_minus_quiet": quiet_gain,
                                "complete_safe_pair": eligible(pooled) and eligible(quiet),
                                "passes": eligible(pooled) and eligible(quiet) and quiet_gain > study["gates"]["useful_requires_all_pooled_gain_above"]})
        values = [row["pooled_minus_narrow"] for row in seed_rows if row["pooled_minus_narrow"] is not None]
        seed_means.append({"seed": seed, "realized_pairs": len(values),
                           "mean_pooled_minus_narrow": statistics.mean(values) if len(values) == 12 else None,
                           "all_pairs_and_quiet_complete_safe": all(row["complete_safe_pair_and_quiet"] for row in seed_rows)})
    means = [row["mean_pooled_minus_narrow"] for row in seed_means if row["mean_pooled_minus_narrow"] is not None]
    median = statistics.median(means) if len(means) == 3 else None
    positive = sum(value > 0. for value in means)
    strata = []
    for seed in study["source_seeds"]:
        for hinge in (0, 1):
            for side in (-1, 1):
                values = [row["pooled_minus_narrow"] for row in comparisons
                          if row["seed"] == seed and row["powered_hinge"] == hinge and row["lamp_side"] == side]
                valid = len(values) == 3 and all(value is not None for value in values)
                strata.append({"seed": seed, "powered_hinge": hinge, "lamp_side": side,
                               "mean_realized_pooled_minus_narrow": statistics.mean(values) if valid else None})
    witnesses = []
    for index, case_id in enumerate(study["case_ids"]["reserved"]):
        quiet = lookup.get(("reserved", "quiet", None, index))
        rows = []
        for controller in study["witness_modes"]:
            if controller == "quiet":
                continue
            fixed = lookup.get(("reserved", controller, None, index))
            rows.append({"controller": controller, "fixed_minus_quiet": contrast(fixed, quiet),
                         "complete_safe_pair": eligible(fixed) and eligible(quiet)})
        safe_rows = [row for row in rows if row["complete_safe_pair"]]
        best = max(safe_rows, key=lambda row: row["fixed_minus_quiet"]) if safe_rows else None
        witnesses.append({"case_id": case_id, "comparisons": rows, "best_safe_fixed": best,
                          "attainable_margin": best is not None and best["fixed_minus_quiet"] > study["gates"]["source_margin"]})
    return {"schema": "generalization_study_challenge_analysis_v1", "comparisons": comparisons,
            "seed_means": seed_means, "lamp_motor_strata": strata,
            "breadth": {"passes": breadth_eligible and median is not None and median > 0. and positive >= study["gates"]["breadth_min_positive_seeds"],
                        "all_pooled_narrow_and_quiet_complete_safe": breadth_eligible,
                        "median_seed_mean": median, "positive_seed_means": positive,
                        "untrained_controls_gate_breadth": False},
            "useful_first_use": {"passes": len(useful_rows) == 36 and all(row["passes"] for row in useful_rows),
                                 "comparisons": useful_rows, "margin_strictly_greater_than": study["gates"]["useful_requires_all_pooled_gain_above"]},
            "target_headroom": {"all_cases_attainable_margin": all(row["attainable_margin"] for row in witnesses),
                                "cases": witnesses, "selection": "Exploratory safe fixed witness; no optimality claim, tuning or case selection"},
            "offline_learning": learning_diagnostics(study, trials, "reserved", ("pooled", "narrow")),
            "raw_prefix_interpretation": "All-case realized contrasts preserve stopped prefixes; unequal exposure cannot establish competence"}


def challenge(output):
    root, protocol, protocol_hash = read_lock(output)
    folder = new_stage(root, "challenge", protocol_hash)
    study = protocol["study"]
    try:
        released, reason = challenge_release(root, protocol_hash, study)
    except (OSError, ValueError, KeyError) as error:
        return blocked_stage(root, folder, protocol, protocol_hash, f"Release unavailable: {error}")
    if reason:
        return blocked_stage(root, folder, protocol, protocol_hash, reason)
    write_new(folder / "release.json", {"schema": "generalization_study_challenge_release_v1",
              "protocol_sha256": protocol_hash, "source_seal_sha256": sha256_file(root / "sources" / "sealed-checkpoints.json"),
              "development_stage_sha256": sha256_file(root / "development" / "stage.json"),
              "global_source_competence": True, "released_before_first_reserved_call": True})
    dependencies = helper.learning_dependencies()
    trials, references = [], []
    cancelled = False
    # Witnesses are collected after source release, with no outcome-based branches.
    for index in range(len(study["reserved_cases"])):
        for controller in study["witness_modes"]:
            result, ref = trial(root, folder, study, protocol_hash, released, "reserved", index, controller, None, dependencies)
            trials.append(result)
            references.append(ref)
            if result["cancelled"]:
                cancelled = True
                break
        if cancelled:
            break
    if not cancelled:
        for seed in study["source_seeds"]:
            for index in range(len(study["reserved_cases"])):
                for controller in study["challenge_arms"]:
                    result, ref = trial(root, folder, study, protocol_hash, released, "reserved", index, controller, seed, dependencies)
                    trials.append(result)
                    references.append(ref)
                    if result["cancelled"]:
                        cancelled = True
                        break
                if cancelled:
                    break
            if cancelled:
                break
    analysis = evaluate_challenge(study, trials)
    if len(trials) != 168:
        analysis["breadth"]["passes"] = analysis["useful_first_use"]["passes"] = False
    return stage_record(root, folder, protocol, protocol_hash, "complete" if len(trials) == 168 else "incomplete",
                        release=reference(root, folder / "release.json"),
                        trial_results=references, collected_trials=len(trials), planned_trials=168, analysis=analysis,
                        actual_decisions=sum(item["metrics"]["decisions"] for item in trials),
                        native_steps=sum(item["metrics"]["native_steps"] for item in trials),
                        reserved_case_construction_requests=sum(item["reserved_case_construction_requests"] for item in trials))


def summary(output):
    root, protocol, protocol_hash = read_lock(output)
    phases = {}
    for name in ("sources", "development", "challenge"):
        if (root / name / "receipt.json").is_file():
            phases[name] = read_stage(root, name, protocol_hash)
        elif (root / name).exists():
            phases[name] = {"status": "unfinished", "reason": "Exclusive stage has no final receipt; evidence preserved, no automatic resume"}
        else:
            phases[name] = {"status": "not_started"}
    report = {"schema": "generalization_study_summary_v1", "protocol_sha256": protocol_hash,
              "phases": phases, "actual_decisions": sum(value.get("actual_decisions", 0) for value in phases.values()),
              "native_steps": sum(value.get("native_steps", 0) for value in phases.values()),
              "reserved_case_construction_requests": phases["challenge"].get("reserved_case_construction_requests", 0),
              "adaptation_weight_updates": 0, "scope": protocol["study"]["scope"]}
    print(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("lock", "sources", "development", "challenge", "summary"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    helper.require_container()
    result = {"lock": lock, "sources": sources, "development": development,
              "challenge": challenge, "summary": summary}[args.command](args.output)
    if args.command == "sources" and result["status"] != "complete":
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as error:
        print(canonical({"error": f"{type(error).__name__}: {error}", "existing_evidence_preserved": True}), file=sys.stderr)
        raise SystemExit(1)
