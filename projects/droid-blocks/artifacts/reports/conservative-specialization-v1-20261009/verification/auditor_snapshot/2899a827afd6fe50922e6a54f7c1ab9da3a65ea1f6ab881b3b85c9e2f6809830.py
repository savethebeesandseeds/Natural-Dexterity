#!/usr/bin/env python3
"""Independent conservative-specialization accounting, checkpoints and replay.

No training is replayed. With --replay, retained native requests and categorical
actions are regenerated from bound checkpoint tensors, local history and Torch
RNG. A new exclusive receipt records integrity separately from scientific gates.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import zipfile

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
HELPER_PATH = PROJECT / "tools/verify-generalization-pretraining.py"
spec = importlib.util.spec_from_file_location("independent_native_audit", HELPER_PATH)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
Audit, load, digest, canonical, safe_path = old.Audit, old.load, old.digest, old.canonical, old.safe_path
SEEDS, ARMS = (101, 202, 303), ("frozen", "conservative", "standard", "scratch")
PREFIX = 384
UPDATE_COUNTS = tuple(range(64, PREFIX + 1, 64))
SETTINGS = {"conservative": (.0001, .1), "standard": (.0003, .2), "scratch": (.0003, .2), "frozen": (.0003, .2)}
SHAPES = {
    "mlp_extractor.policy_net.0.weight": (64, 204), "mlp_extractor.policy_net.0.bias": (64,),
    "mlp_extractor.policy_net.2.weight": (64, 64), "mlp_extractor.policy_net.2.bias": (64,),
    "mlp_extractor.value_net.0.weight": (64, 204), "mlp_extractor.value_net.0.bias": (64,),
    "mlp_extractor.value_net.2.weight": (64, 64), "mlp_extractor.value_net.2.bias": (64,),
    "action_net.weight": (3, 64), "action_net.bias": (3,), "value_net.weight": (1, 64), "value_net.bias": (1,),
}
WINDOWS = (*UPDATE_COUNTS, 640)


def reference(audit, record, label):
    if not isinstance(record, dict) or not {"path", "sha256"} <= set(record):
        raise ValueError(f"{label}: malformed evidence reference")
    path = audit.file(PROJECT, {"path": record["path"], "sha256": record["sha256"]})
    return path


def bound(audit, record, label):
    return load(reference(audit, record, label))


def partition(name):
    for prefix, label in (("mlp_extractor.policy_net.", "actor_trunk"), ("mlp_extractor.value_net.", "value_trunk"),
                          ("action_net.", "actor_head"), ("value_net.", "value_head")):
        if name.startswith(prefix):
            return label
    raise ValueError("Unexpected neural parameter")


def tensors_digest(items):
    result = hashlib.sha256()
    for name, tensor in sorted(items):
        value = tensor.detach().cpu().contiguous().numpy()
        result.update(name.encode() + b"\0" + str(value.shape).encode() + b"\0" + str(value.dtype).encode() + b"\0")
        result.update(value.tobytes())
    return result.hexdigest()


def parameter_summary(audit, state, torch, label):
    audit.equal(set(state), set(SHAPES), label + ".tensor names")
    for name, shape in SHAPES.items():
        value = state[name]
        audit.check(isinstance(value, torch.Tensor) and tuple(value.shape) == shape and value.dtype == torch.float32,
                    label + ".tensor shape/dtype " + name)
        audit.check(bool(torch.isfinite(value).all()), label + ".nonfinite " + name)
    groups = {}
    for part in ("actor_trunk", "value_trunk", "actor_head", "value_head"):
        selected = [(name, value) for name, value in state.items() if partition(name) == part]
        groups[part] = {"sha256": tensors_digest(selected), "tensors": len(selected),
                        "elements": sum(v.numel() for _, v in selected)}
    return {"parameter_sha256": tensors_digest(state.items()), "partitions": groups,
            "parameters_finite": all(bool(torch.isfinite(v).all()) for v in state.values()),
            "parameter_elements": sum(v.numel() for v in state.values()), "parameter_tensors": len(state)}


def optimizer_summary(audit, saved, state, names, label):
    if saved is None:
        return {"enabled": False, "fresh": True, "param_groups": [], "state": {}, "state_tensors": 0, "step_min": 0, "step_max": 0}
    audit.equal(set(saved), {"state", "param_groups"}, label + ".optimizer fields")
    audit.equal(len(saved["param_groups"]), 1, label + ".optimizer group count")
    group = saved["param_groups"][0]
    ids = group["params"]
    audit.check(len(ids) == len(set(ids)) == len(names), label + ".optimizer membership count")
    audit.check(set(saved["state"]) <= set(ids), label + ".unknown optimizer state")
    named, steps = {}, []
    for identifier, name in zip(ids, names):
        if identifier not in saved["state"]:
            continue
        entry = saved["state"][identifier]
        audit.equal(set(entry), {"step", "exp_avg", "exp_avg_sq"}, label + ".Adam fields")
        number = float(entry["step"].item())
        audit.check(math.isfinite(number) and number >= 0 and number.is_integer(), label + ".Adam counter")
        steps.append(int(number))
        for key in ("exp_avg", "exp_avg_sq"):
            audit.check(tuple(entry[key].shape) == tuple(state[name].shape) and bool(entry[key].isfinite().all()),
                        label + ".Adam moment shape/finite")
        named[name] = {"step": int(number), "exp_avg_sha256": tensors_digest([("exp_avg", entry["exp_avg"])]),
                       "exp_avg_sq_sha256": tensors_digest([("exp_avg_sq", entry["exp_avg_sq"])])}
    return {"enabled": True, "fresh": not named,
            "param_groups": [{"parameter_names": list(names), "learning_rate": float(group["lr"]), "eps": float(group["eps"]),
                              "betas": list(group["betas"]), "weight_decay": float(group["weight_decay"]), "amsgrad": bool(group["amsgrad"])}],
            "state": named, "state_tensors": len(named), "step_min": min(steps, default=0), "step_max": max(steps, default=0)}


def checkpoint(audit, record, arm, count, torch, loader, label):
    path = reference(audit, record, label)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        audit.check(len(names) == len(set(names)), label + ".duplicate ZIP entries")
        raw = old.parse(archive.read("data").decode("utf-8"))
    data, saved, variables = loader(path, load_data=False, device="cpu", verbose=0)
    audit.equal(data, None, label + ".no cloudpickle metadata")
    audit.check(variables in (None, {}), label + ".unexpected PyTorch variables")
    audit.check(set(saved) in ({"policy"}, {"policy", "policy.optimizer"}), label + ".checkpoint dictionaries")
    state = saved["policy"]
    summary = parameter_summary(audit, state, torch, label)
    trainable = list(state) if arm != "frozen" else []
    opt = optimizer_summary(audit, saved.get("policy.optimizer"), state, trainable, label)
    if arm == "frozen":
        audit.equal(opt["enabled"], False, label + ".disabled optimizer")
    else:
        audit.check(opt["enabled"] and opt["step_min"] == opt["step_max"] == count // 64 * 4,
                    label + ".exact optimizer steps")
        audit.equal(opt["fresh"], count == 0, label + ".fresh optimizer")
        audit.equal(opt["state_tensors"], 0 if count == 0 else len(trainable), label + ".initialized optimizer parameters")
        audit.equal(opt["param_groups"][0]["learning_rate"], SETTINGS[arm][0], label + ".actual constant Adam learning rate")
        for key, wanted in (("eps", 1e-5), ("betas", [.9, .999]), ("weight_decay", 0.), ("amsgrad", False)):
            audit.equal(opt["param_groups"][0][key], wanted, label + ".Adam " + key)
    for key, value in (("num_timesteps", count if arm != "frozen" else 0), ("_n_updates", count // 64 * 4 if arm != "frozen" else 0),
                       ("n_envs", 1), ("n_steps", 64), ("batch_size", 64), ("n_epochs", 4), ("use_sde", False)):
        audit.equal(raw[key], value, label + ".metadata " + key)
    audit.equal(raw["learning_rate"], SETTINGS[arm][0], label + ".metadata learning rate")
    audit.equal(raw["target_kl"], None, label + ".no optional KL early stop")
    if record.get("parameter_sha256") is not None:
        audit.equal(record["parameter_sha256"], summary["parameter_sha256"], label + ".tensor commitment")
    return {"state": state, "summary": summary, "optimizer": opt, "raw": raw, "path": path}


def declaration(audit, study):
    audit.equal(study["schema"], "conservative_specialization_manifest_v1", "manifest schema")
    audit.equal(study["source_seeds"], list(SEEDS), "source seeds")
    audit.equal(study["arms"], list(ARMS), "arms")
    target = [old.fixture(lengths, weighted, hinge, side)
              for lengths, weighted in (((5, 3), False), ((3, 5), False), ((5, 5), True))
              for hinge in (0, 1) for side in (-1, 1)]
    guard = [old.fixture((4, 3), False, 0, -1), old.fixture((4, 4), True, 1, 1)]
    for partition_name, expected in (("target_cases", target), ("guard_cases", guard)):
        audit.equal(len(study[partition_name]), len(expected), partition_name + ".count")
        for i, (entry, wanted) in enumerate(zip(study[partition_name], expected)):
            audit.equal(old.mechanics(entry["case"]), old.mechanics(wanted), f"{partition_name}[{i}].mechanics")
            audit.equal(entry["case"]["sun"], wanted["sun"], f"{partition_name}[{i}].sun")
    target_ids = [f"exposed-{body}-{'base' if hinge == 0 else 'elbow'}-{'left' if side < 0 else 'right'}"
                  for body in ("long-upper", "long-lower", "weighted") for hinge in (0, 1) for side in (-1, 1)]
    audit.equal([entry["id"] for entry in study["target_cases"]], target_ids, "target IDs")
    audit.equal([entry["id"] for entry in study["guard_cases"]], ["development-long-upper-base-left", "development-weighted-elbow-right"], "guard IDs")
    audit.equal(study["budget"], {"prefix_decisions": PREFIX, "total_decisions": 640, "continuation_decisions": 256,
                "horizon_decisions": 641, "rollouts": 6, "adam_steps": 24, "update_decisions": list(UPDATE_COUNTS)}, "budgets")
    audit.equal(study["trainable"], {"conservative": {"tensors": 12, "parameters": 34820},
                "standard": {"tensors": 12, "parameters": 34820}, "scratch": {"tensors": 12, "parameters": 34820}, "frozen": {"parameters": 0}}, "trainable scope")
    audit.equal(study["gates"], {"primary_arm": "conservative", "secondary_arm": "standard",
                "comparison": "integrated native return over the whole640 opportunity, including384 learning decisions",
                "requires_all_36_corresponding_pairs_safe_complete": True,
                "median_seed_mean_gain_strictly_greater_than": 3.2, "minimum_positive_seed_means": 2,
                "scratch_contrast_median_strictly_greater_than": 0., "regression_diagnostic_whole_loss_floor": -3.2,
                "regression_diagnostic_prefix_loss_floor": -1.92,
                "sampled_retention": {"each_seed_mean_gain_at_least": -3.2, "requires_all_corresponding_probes_safe_complete": True},
                "reliability": {"target_each_pair_gain_at_least": -3.2, "probe_each_pair_gain_at_least": -3.2,
                                "requires_all_corresponding_pairs_safe_complete": True},
                "combined_readiness": "For each declared method separately: useful-benefit AND target reliability AND sampled retention AND probe reliability. Do not select the better arm afterward. No automatic promotion even if passed."}, "scientific gates")
    audit.equal(study["maximum_collection"], {"target_opportunities": 144, "target_decisions": 92160,
                "learning_decisions_included": 41472, "guard_opportunities": 150, "guard_decisions": 96000,
                "total_opportunities": 294, "total_decisions": 188160, "native_controls": 940800,
                "physics_steps": 9408000, "aggregate_simulated_seconds": 18816}, "maximum collection")
    for key, wanted in (("target_n_steps", 64), ("target_n_envs", 1), ("batch_size", 64), ("n_epochs", 4),
                         ("learning_rate", .0003), ("optimizer_eps", 1e-5)):
        audit.equal(study["ppo"][key], wanted, "PPO." + key)
    audit.equal(study["per_arm_ppo"], {arm: {"learning_rate": lr, "clip_range": clip}
                                     for arm, (lr, clip) in SETTINGS.items()}, "arm schedules")
    audit.equal(study["action_rng"]["target_base"], 810000, "target RNG base")
    audit.equal(study["action_rng"]["guard_base"], 910000, "guard RNG base")
    audit.equal(study["action_rng"]["seed_multiplier"], 1000, "RNG multiplier")
    audit.equal(study["decision_dt_s"], .1, "decision interval")
    audit.equal(study["native_dt_s"], .02, "native interval")


def load_source(audit, record, torch, loader, label):
    path = reference(audit, record, label)
    _, saved, variables = loader(path, load_data=False, device="cpu", verbose=0)
    audit.check(variables in (None, {}), label + ".unexpected variables")
    return parameter_summary(audit, saved["policy"], torch, label)


def historical_inputs(audit, protocol, torch, loader):
    prior = protocol["inputs"]["prior_specialization"]
    commitments = {
        "protocol": "5b5dbda3123c20a030021c141e4812ec0db1c8f66fd9030eca78b8265ec002e4",
        "audit": "65b297f720b0b1fb8e1975d71073e707d759eef947d789e6f1f3f4b07f6d56b2",
        "final_provenance": "62aa877921e97edd1a57a0d29564251099502c5fb9c4ddc1f304aa57d2a29d72",
        "targets_stage": "6b7c09622a1c0615609071f997922a0a4b8fce0fe843193ea99e4d105f8c1a78",
        "targets_seal": "186cb490c899bfe8ec01c42b691bdbaa894d1f0f9af76822932094bd7cd70209"}
    records = {}
    for key, sha in commitments.items():
        records[key] = bound(audit, prior[key], "prior specialization " + key)
        audit.equal(prior[key]["sha256"], sha, "immutable earlier specialization " + key)
    previous_audit, stage, seal = records["audit"], records["targets_stage"], records["targets_seal"]
    audit.check(previous_audit["integrity_passed"] and previous_audit["native_replay_verified"]
                and previous_audit["operationally_complete"], "earlier full audit passed")
    audit.equal(previous_audit["analysis"]["primary_passed"], False, "earlier primary failed gate preserved")
    audit.equal(previous_audit["analysis"]["secondary_passed"], False, "earlier secondary failed gate preserved")
    audit.equal(stage["seal"], prior["targets_seal"], "old stage points to exact bound seal")
    audit.equal(stage["trial_results"], seal["trial_results"], "old stage and seal result membership")
    sealed = {r["path"]: r["sha256"] for r in seal["bound_evidence"]}
    audited = {r["trial_id"]: r for r in previous_audit["target_trials"]}
    audit.equal(len(prior["targets"]), 72, "all earlier36 frozen/full conditions")
    result = {}
    expected_order = [(seed, entry["id"], arm) for seed in SEEDS for entry in protocol["study"]["target_cases"]
                      for arm in ("frozen", "full")]
    audit.equal([(r["seed"], r["case_id"], r["arm"]) for r in prior["targets"]], expected_order,
                "prospective historical parity order")
    for item in prior["targets"]:
        raw = bound(audit, item["result"], "prior target")
        index = next(i for i, c in enumerate(protocol["study"]["target_cases"]) if c["id"] == item["case_id"])
        key = (item["seed"], index, item["arm"])
        audit.check(key not in result, "duplicate earlier target identity")
        audit.equal((raw["seed"], raw["case_index"], raw["arm"]), key, "earlier target labels")
        audit.equal(raw["case"], protocol["study"]["target_cases"][index]["case"], "same historical native case")
        audit.check(raw["complete"] and raw["safe"] and raw["exposure_exact"], "historical parity complete-safe prefix")
        audit.equal(item["result"]["sha256"], sealed[item["result"]["path"]], "historical result sealed bytes")
        audit.equal(audited[raw["trial_id"]]["reference"], item["result"], "earlier independent audit covers exact result")
        for field in ("request_tape", "telemetry"):
            audit.equal(raw[field]["sha256"], sealed[raw[field]["path"]], "earlier " + field + " seal binding")
        tape = load(reference(audit, raw["request_tape"], "earlier request tape"), True)
        telemetry = load(reference(audit, raw["telemetry"], "earlier telemetry"), True)
        audit.equal(tape["native_stream_sha256"], raw["request_tape"]["native_stream_sha256"], "earlier stream commitment")
        audit.equal(len(telemetry["rows"]), 640, "earlier full telemetry")
        states = {}
        for name, count in (("initial", 0), ("after64", 64), ("after128", 128)):
            if name in raw["checkpoints"]:
                ref = raw["checkpoints"][name]
                audit.equal(ref["sha256"], sealed[ref["path"]], "earlier actual checkpoint sealed")
                states[count] = checkpoint(audit, ref, "standard" if item["arm"] == "full" else "frozen",
                                           count, torch, loader, "earlier " + raw["trial_id"] + "." + name)
        audit.equal(set(states), {0, 64, 128} if item["arm"] == "full" else {0}, "earlier parity checkpoint matrix")
        result[key] = {"raw": raw, "rows": telemetry["rows"], "tape": tape, "states": states, "reference": item["result"]}
    return result


def prerequisites(root, audit, expected_sha, torch, loader):
    protocol, lock = load(root / "protocol.json"), load(root / "lock.json")
    sha = digest(root / "protocol.json")
    audit.equal(protocol["schema"], "conservative_specialization_protocol_v1", "protocol schema")
    audit.equal(protocol["status"], "locked_before_collection", "prospective protocol status")
    audit.equal(lock, {"schema": "conservative_specialization_lock_v1", "protocol_sha256": sha,
                      "native_worlds_constructed": 0, "decisions": 0}, "lock")
    if expected_sha:
        audit.equal(sha, expected_sha, "external protocol commitment")
    declaration(audit, protocol["study"])
    hashes = protocol["source_sha256"]
    audit.check(bool(hashes), "empty experimental source archive")
    for name, wanted in hashes.items():
        for base in (PROJECT, root / "source_snapshot"):
            audit.equal(digest(safe_path(base, name)), wanted, "source bytes " + str(base / name))
    actual = {p.relative_to(root / "source_snapshot").as_posix()
              for p in (root / "source_snapshot").rglob("*") if p.is_file()}
    audit.equal(actual, set(hashes), "complete source snapshot inventory")
    historical = {
        "protocol.json": "4cf6fe175dfad1150c9264e2ab074f8df7fe575a978724bc7d3ffddb16915727",
        "sources/sealed-checkpoints.json": "6404315c901c19972c7b190f797026431fc88ecd9006c6d0ebaa740feed18508",
        "verification/native-replay-v1.json": "92839768519ec6c3df1a7dc56f9ca25adcf563375886f26e3057fe679a19bf20",
        "verification/tensors-v1.json": "35cbaf63da3a84ade402efc486573fb2b0dde520865b04407ad02faab14e5544"}
    receipts = protocol["inputs"]["receipts"]
    for name, wanted in historical.items():
        value = bound(audit, receipts[name], "historical " + name)
        audit.equal(receipts[name]["sha256"], wanted, "trusted historical commitment " + name)
        if name.endswith("native-replay-v1.json"):
            audit.check(value["integrity_passed"] and value["native_replay_verified"], "historical native audit passed")
        if name.endswith("tensors-v1.json"):
            audit.check(value["checks_passed"], "historical tensor audit passed")
    seal = bound(audit, receipts["sources/sealed-checkpoints.json"], "historical seal")
    audit.check(seal["operational_complete"] and seal["sealed_before_any_evaluation"], "historical operational source seal")
    sealed = {item["path"]: item["sha256"] for item in seal["source_results"]}
    baseline = bound(audit, receipts["preservation_baseline"], "preservation baseline")
    audit.check(bool(baseline), "preservation baseline exists")
    preflight = bound(audit, receipts["synthetic_preflight"], "synthetic preflight")
    audit.check(preflight["passed"] and preflight["native_worlds_constructed"] == 0, "prospective synthetic preflight")
    for item in preflight["test_sources"]:
        reference(audit, item, "preflight source")
    releases = {}
    for item in protocol["inputs"]["releases"]:
        seed = item["seed"]
        audit.check(seed in SEEDS and seed not in releases, "source seed matrix")
        result = bound(audit, item["result"], f"source {seed}")
        expected_suffix = f"sources/source-pooled-{seed}/result.json"
        audit.check(item["result"]["path"].endswith("/" + expected_suffix), "canonical pooled source path")
        audit.equal(item["result"]["sha256"], sealed[expected_suffix], "sealed source result")
        audit.check(result["complete"] and result["source_arm"] == "pooled" and result["seed"] == seed,
                    "operational complete pooled source")
        audit.equal(result["actual_decisions"], 98304, "source learning budget")
        for kind, key in (("initial", "initial_parameter_sha256"), ("final", "final_parameter_sha256")):
            summary = load_source(audit, item[kind], torch, loader, f"source {seed} {kind}")
            audit.equal(summary["parameter_sha256"], item[key], "actual source tensor commitment")
            expected_bytes = result["initial_checkpoint"]["policy_sha256"] if kind == "initial" else result["policy_sha256"]
            audit.equal(item[kind]["sha256"], expected_bytes, "source checkpoint byte commitment")
            audit.equal(item[key], result["initial_checkpoint"]["parameter_sha256"] if kind == "initial" else result["parameter_sha256"],
                        "source tensor label")
        releases[seed] = item
    audit.equal(set(releases), set(SEEDS), "all three source releases")
    runtime = protocol["runtime"]
    audit.equal(runtime["python"], sys.version, "Python runtime")
    audit.equal(runtime["python_executable"], sys.executable, "Python executable")
    audit.equal(runtime["platform"], platform.platform(), "platform runtime")
    for package, version in runtime["packages"].items():
        audit.equal(metadata.version(package), version, "runtime package " + package)
    audit.check(runtime["cpu_only"] and runtime["torch_threads"] == runtime["torch_interop_threads"] == 1
                and runtime["torch_deterministic_algorithms"], "deterministic CPU contract")
    audit.equal(protocol["native_binary"]["path"], "/workspace/build/native/droid-transfer-env", "native command")
    audit.equal(digest(Path(protocol["native_binary"]["path"])), protocol["native_binary"]["sha256"], "native binary bytes")
    audit.equal(runtime["native_binary_sha256"], protocol["native_binary"]["sha256"], "native runtime commitment")
    import stable_baselines3
    dependency_root = Path(stable_baselines3.__file__).resolve().parent
    for name, wanted in protocol["dependency_sha256"].items():
        audit.equal(digest(safe_path(dependency_root, name)), wanted, "SB3 dependency bytes " + name)
    return protocol, sha, releases


def local_frame(observation, executed, previous, reward, np):
    if observation["schema_version"] != "construction_observation_v2":
        raise ValueError("Wrong local observation schema")
    sensors = {entry["family_id"]: entry for entry in observation["reward_sensors"]}
    imu, light = sensors["imu_6axis_v0"], sensors["ambient_light_v0"]
    motor = observation["actuator_feedback"][0]["feedback"]
    force = imu["observations"]["specific_force_m_s2"] if imu["valid"] else [0.] * 3
    gyro = imu["observations"]["angular_velocity_rad_s"] if imu["valid"] else [0.] * 3
    lux = light["observations"]["illuminance_lux"] if light["valid"] else 0.
    values = [executed / .15, math.sin(motor["position_rad"]), math.cos(motor["position_rad"]),
              motor["velocity_rad_s"] / 10., motor["current_a"] / .5,
              *(v / 50. for v in force), *(v / 10. for v in gyro), lux / 1000., previous / .15,
              reward / .2, float(imu["valid"]), float(light["valid"]), 1.]
    frame = np.asarray(values, dtype=np.float32)
    if frame.shape != (17,) or not np.isfinite(frame).all():
        raise ValueError("Invalid float32 local frame")
    return np.clip(frame, -5., 5.)


def input_digest(frames, np):
    return hashlib.sha256(np.concatenate(frames).astype("<f4", copy=False).tobytes()).hexdigest()


def action_replay(audit, trial, rows, states, torch, np, label):
    initial = trial.get("initial_local_observation")
    if initial is None:
        audit.check(not rows, label + ".missing initial observation")
        return {}, None
    frames = deque([np.zeros(17, np.float32) for _ in range(11)], maxlen=12)
    frames.append(local_frame(initial, 0., 0., 0., np))
    boundary = {0: input_digest(frames, np)}
    torch.manual_seed(trial["action_seed"])
    rng = lambda: hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest()
    rng_boundaries = {0: rng()}
    sequence = rows + ([trial["pending_action_audit"]] if trial.get("pending_action_audit") else [])
    with torch.no_grad():
        for ordinal, row in enumerate(sequence, 1):
            position = min(((ordinal - 1) // 64) * 64, PREFIX)
            if position not in states:
                position = 0
            state = states[position]["state"]
            value = torch.from_numpy(np.concatenate(frames).reshape(1, 204))
            for layer in ("mlp_extractor.policy_net.0", "mlp_extractor.policy_net.2"):
                value = torch.tanh(torch.nn.functional.linear(value, state[layer + ".weight"], state[layer + ".bias"]))
            logits = torch.nn.functional.linear(value, state["action_net.weight"], state["action_net.bias"])
            before = rng()
            chosen = int(torch.distributions.Categorical(logits=logits).sample().item())
            after = rng()
            audit.equal(row["input_float32_le_sha256"], input_digest(frames, np), label + f".input bytes {ordinal}")
            audit.equal(row["torch_rng_before_sha256"], before, label + f".RNG before {ordinal}")
            audit.equal(row["torch_rng_after_sha256"], after, label + f".RNG after {ordinal}")
            audit.equal(row["sampled_action"], chosen, label + f".categorical action {ordinal}")
            if ordinal <= len(rows):
                audit.equal(row["action"], chosen, label + f".native requested action {ordinal}")
                value = row["native"]
                observation = {"schema_version": "construction_observation_v2", **value["sampled_local"]}
                frames.append(local_frame(observation, value["executed_effort"], value["previous_request"], value["reward"], np))
                boundary[ordinal], rng_boundaries[ordinal] = input_digest(frames, np), after
    return boundary, rng_boundaries


def snapshot(audit, saved, actual, active_arm, count, boundary, rng_boundaries, label, frozen_suffix=False):
    for key, expected in actual["summary"].items():
        audit.equal(saved[key], expected, label + "." + key)
    audit.equal(saved["optimizer"], actual["optimizer"], label + ".actual Adam state")
    audit.equal(saved["ppo_settings"], {"learning_rate": SETTINGS[active_arm][0], "clip_range": SETTINGS[active_arm][1],
                                       "target_kl": None, "use_sde": False}, label + ".actual PPO settings")
    learning = active_arm != "frozen"
    counter = min(count, PREFIX) if learning else 0
    audit.equal(saved["num_timesteps"], counter, label + ".learning decisions")
    audit.equal(saved["optimization_epochs"], counter // 64 * 4, label + ".optimization epochs")
    trainable = {name: learning and not frozen_suffix for name in SHAPES}
    audit.equal(saved["requires_grad"], trainable, label + ".frozen scope")
    audit.equal(saved["trainable_tensors"], sum(trainable.values()), label + ".trainable tensors")
    audit.equal(saved["trainable_elements"], sum(math.prod(SHAPES[n]) for n in SHAPES if trainable[n]), label + ".trainable elements")
    audit.equal(saved["rollout_buffer"], {"full": learning and count >= 64, "position": 64 if learning and count >= 64 else 0,
                "buffer_size": 64, "n_envs": 1}, label + ".rollout buffer")
    if count in boundary:
        audit.equal(saved.get("input_float32_le_sha256"), boundary[count], label + ".history bytes")
    if rng_boundaries and count in rng_boundaries:
        audit.equal(saved["torch_rng_sha256"], rng_boundaries[count], label + ".action RNG")


def valid_post384(result):
    candidate = result.get("checkpoints", {}).get("after384")
    events = result.get("training_updates", [])
    snap = result.get("snapshots", {}).get("after384", {})
    return bool(candidate and [e.get("decisions") for e in events] == list(UPDATE_COUNTS)
                and all(e.get("passed") and e.get("error") is None for e in events)
                and snap.get("parameters_finite") and snap.get("num_timesteps") == PREFIX
                and snap.get("optimization_epochs") == 24 and snap.get("optimizer", {}).get("step_min") == 24
                and snap.get("optimizer", {}).get("step_max") == 24 and candidate.get("decisions") == PREFIX
                and candidate.get("parameter_sha256") == snap.get("parameter_sha256"))


def single_world_schedule(audit, tape, case, label):
    draws, actions = old.request_schedule(audit, tape, [case], 1, False, label)
    audit.check(len(draws) <= 1 and len(actions) <= 640, label + ".single world bounded budget")
    return draws, actions


def tape_check(audit, trial, case, do_replay, label):
    record = trial.get("request_tape")
    if record is None:
        return None, [], None, None
    path = reference(audit, record, label + ".request tape")
    tape = load(path, True)
    audit.equal(tape["schema"], "generalization_native_request_tape_v1", label + ".tape schema")
    audit.equal(tape["record"], False, label + ".native recording mode")
    audit.equal(record["request_count"], len(tape["requests"]), label + ".request count")
    audit.equal(record["native_stream_sha256"], tape["native_stream_sha256"], label + ".native stream label")
    audit.equal(trial["native_failure"], tape["native_failure"], label + ".native failure label")
    audit.equal(trial["unresolved_request"], tape["unresolved_request"], label + ".unresolved request label")
    draws, actions = single_world_schedule(audit, tape, case, label)
    initial, replayed = None, None
    if do_replay:
        if audit.failed:
            raise ValueError("Native replay refused after static evidence validation failed")
        # Capture the original configure response during the same independent
        # replay; no second construction is needed to check initial history.
        original = old.NativeReplay
        class Capturing(original):
            def call(self, request):
                nonlocal initial
                response = super().call(request)
                if request["op"] == "configure" and response.get("ok"):
                    initial = response["result"][0]["observation"]
                return response
        old.NativeReplay = Capturing
        try:
            replayed = old.replay(audit, tape, [case], 1, False, label)
        finally:
            old.NativeReplay = original
    return tape, actions, replayed, initial


def checkpoint_scope(audit, record, attempt, label):
    path = reference(audit, record, label)
    audit.check(path.parent == attempt, label + ".checkpoint outside its trial directory")
    return path


def audit_trial(audit, record, expected, protocol, sha, torch, np, loader, do_replay):
    before_failures = audit.failed
    path = reference(audit, record, "trial result")
    trial, label = load(path), expected["trial_id"]
    attempt = path.parent
    audit.equal(attempt.name, label, label + ".directory identity")
    for key in ("trial_id", "partition", "seed", "arm", "case_id", "case_index", "guard_source"):
        audit.equal(trial.get(key), expected[key], label + "." + key)
    emergency = trial["schema"] == "conservative_specialization_trial_failure_v1"
    if not emergency:
        audit.equal(trial.get("checkpoint"), expected["checkpoint"], label + ".checkpoint")
    audit.equal(trial["protocol_sha256"], sha, label + ".protocol")
    if (attempt / "started.json").is_file():
        start = load(attempt / "started.json")
        audit.equal(start, {"schema": "conservative_specialization_trial_start_v1", "protocol_sha256": sha,
                           **{key: expected[key] for key in ("trial_id", "partition", "seed", "arm", "case_id",
                                                            "case_index", "checkpoint", "guard_source")}}, label + ".start")
    else:
        audit.check(emergency, label + ".missing start only allowed for archival failure")
    if emergency:
        audit.check(not trial["complete"] and not trial["safe"] and trial["unknown_exposure"]
                    and not trial["exposure_exact"], label + ".honest archival failure")
        audit.equal(trial["metrics"], old.metrics([]), label + ".archival known lower bound")
        audit.equal(trial["case_construction_requests"], 0, label + ".archival configuration known lower bound")
        for item in trial["retained_prefix"]:
            retained = reference(audit, item, label + ".failure prefix")
            audit.check(retained.is_relative_to(attempt), label + ".failure prefix outside attempt")
        return {"raw": trial, "reference": record, "rows": [], "trial_id": label, "seed": expected["seed"],
                "arm": expected["arm"], "case_index": expected["case_index"], "guard_source": expected["guard_source"],
                "complete": False, "safe": False, "blocked": False, "exposure_exact": False,
                "metrics": old.metrics([]), "prefix_metrics": old.metrics([]), "continuation_metrics": old.metrics([]),
                "request_tapes": 0, "checkpoint_count": 0, "action_count": 0, "integrity_consistent": audit.failed == before_failures,
                "unknown_exposure": True, "valid_post384": False}
    audit.equal(trial["schema"], "conservative_specialization_trial_v1", label + ".schema")
    audit.equal(trial["case"], expected["case"], label + ".native case")
    seed = (910000 if expected["partition"] == "guard" else 810000) + expected["seed"] * 1000 + expected["case_index"]
    audit.equal(trial["action_seed"], seed, label + ".action seed")
    if expected["checkpoint"] is None:
        audit.check(trial["blocked"] and trial["status"] == "blocked" and not trial["complete"] and not trial["safe"],
                    label + ".missing checkpoint blocks probe")
        for key, wanted in (("metrics", old.metrics([])), ("case_construction_requests", 0), ("exposure_exact", True),
                            ("reset_count", 0), ("request_tape", None), ("telemetry", None), ("learning_log", None),
                            ("snapshots", {}), ("checkpoints", {}), ("training_updates", [])):
            audit.equal(trial[key], wanted, label + ".blocked " + key)
        audit.equal({p.name for p in attempt.rglob("*") if p.is_file()}, {"started.json", "result.json"}, label + ".blocked file inventory")
        return {"raw": trial, "reference": record, "rows": [], "trial_id": label, "seed": expected["seed"],
                "arm": expected["arm"], "case_index": expected["case_index"], "guard_source": expected["guard_source"],
                "complete": False, "safe": False, "blocked": True, "exposure_exact": True,
                "metrics": old.metrics([]), "prefix_metrics": old.metrics([]), "continuation_metrics": old.metrics([]),
                "request_tapes": 0, "checkpoint_count": 0, "action_count": 0,
                "integrity_consistent": audit.failed == before_failures, "valid_post384": False}
    reference(audit, expected["checkpoint"], label + ".initial weights input")
    audit.check(not trial["blocked"], label + ".executable opportunity marked blocked")
    tape, actions, replayed, initial = tape_check(audit, trial, expected["case"], do_replay, label)
    rows = []
    if trial.get("telemetry"):
        telemetry = load(reference(audit, trial["telemetry"], label + ".telemetry"), True)
        audit.equal(telemetry["schema"], "conservative_specialization_telemetry_v1", label + ".telemetry schema")
        rows = telemetry["rows"]
    audit.check(len(rows) <= 640, label + ".640-decision maximum")
    audit.check(len(actions) == len(rows) or len(actions) == len(rows) + 1 and bool(trial.get("native_failure") or trial.get("unresolved_request")),
                label + ".request/telemetry prefix accounting")
    previous = 0.
    for ordinal, row in enumerate(rows, 1):
        audit.equal(row["decision"], ordinal, label + ".decision ordinal")
        if ordinal <= len(actions):
            audit.equal(row["action"], actions[ordinal - 1][1][0], label + ".tape action")
        old.native_check(audit, row["native"], label + f".decision {ordinal}", previous)
        previous = row["native"]["time_s"]
        audit.check(not row["native"]["terminated"] or ordinal == len(rows), label + ".continued after safety stop")
    if replayed:
        audit.equal([{key: row[key] for key in ("decision", "action", "native")} for row in rows],
                    replayed["rows"], label + ".all native compact observations versus full-response replay")
        audit.equal(initial, trial["initial_local_observation"], label + ".initial physical observation")
    computed = old.metrics(rows)
    audit.equal(trial["metrics"], computed, label + ".metrics")
    audit.equal(trial["prefix_metrics"], old.metrics(rows[:PREFIX]), label + ".learning-prefix return")
    audit.equal(trial["continuation_metrics"], old.metrics(rows[PREFIX:]), label + ".fixed-continuation return")
    audit.equal(trial["windows"], [{"requested_decisions": n, "complete": len(rows) >= n,
                                  "metrics": old.metrics(rows[:n])} for n in WINDOWS], label + ".windows")
    exact = tape is not None and tape["native_failure"] is None and tape["unresolved_request"] is None
    audit.equal(trial["exposure_exact"], exact, label + ".exposure exactness")
    constructs = sum(len(req["cases"]) for req in
                     (tape["requests"] + ([tape["unresolved_request"]] if tape["unresolved_request"] else []))
                     if req["op"] == "configure") if tape else 0
    audit.equal(trial["case_construction_requests"], constructs, label + ".configuration calls")
    audit.check(constructs <= 1 and trial["reset_count"] in (0, 1), label + ".single continuous world")
    active = "frozen" if expected["partition"] == "guard" else expected["arm"]
    states = {}
    for checkpoint_label, item in trial["checkpoints"].items():
        count = 0 if checkpoint_label == "initial" else int(checkpoint_label.removeprefix("after"))
        audit.check(count in (0, *UPDATE_COUNTS) and (count == 0 or active != "frozen"), label + ".undeclared checkpoint")
        checkpoint_scope(audit, item, attempt, label + "." + checkpoint_label)
        audit.equal(item["decisions"], count, label + ".checkpoint decisions")
        states[count] = checkpoint(audit, item, active, count, torch, loader, label + "." + checkpoint_label)
    audit.check(not rows or 0 in states, label + ".fresh initial checkpoint retained")
    if 0 in states:
        audit.equal(states[0]["summary"]["parameter_sha256"], expected["parameter_sha256"], label + ".source copied weights")
        audit.equal(trial["checkpoint_parameter_sha256"], expected["parameter_sha256"], label + ".source weight label")
    boundary, rng_boundaries = action_replay(audit, trial, rows, states, torch, np, label) if states else ({}, None)
    snaps = trial["snapshots"]
    for name, saved in snaps.items():
        count = 0 if name == "initial" else int(name.removeprefix("after"))
        audit.check(count in (0, *UPDATE_COUNTS, 640) and count <= len(rows), label + ".snapshot ordinal")
        actual_count = 0 if active == "frozen" else min(count, PREFIX)
        if actual_count in states:
            snapshot(audit, saved, states[actual_count], active, count, boundary, rng_boundaries, label + "." + name, count == 640)
    if "initial" in snaps:
        audit.equal(trial["reset_count"], 1, label + ".one initialization reset")
    events = trial["training_updates"]
    if trial.get("learning_log"):
        log = reference(audit, trial["learning_log"], label + ".learning log")
        if log.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Learning log exceeds bounded size")
        recorded = [old.parse(line) for line in log.read_text(encoding="utf-8").splitlines()]
        audit.equal(recorded, events, label + ".bound optimizer event log")
    audit.check(len(events) <= 6 and (not events or active != "frozen"), label + ".declared update budget")
    rng_unchanged = True
    for event_index, event in enumerate(events):
        count = (event_index + 1) * 64
        audit.equal(event["event"], "ppo_update", label + ".optimizer event")
        audit.equal(event["decisions"], count, label + ".optimizer event timing")
        rng_same = event["before"]["torch_rng_sha256"] == event["after"]["torch_rng_sha256"]
        if "torch_rng_unchanged" in event:
            audit.equal(event["torch_rng_unchanged"], rng_same, label + ".optimizer RNG accounting")
            rng_unchanged &= rng_same
        previous_state = states.get(count - 64)
        if previous_state:
            for key, value in previous_state["summary"].items():
                audit.equal(event["before"][key], value, label + ".actual preupdate parameters")
            audit.equal(event["before"]["optimizer"], previous_state["optimizer"], label + ".actual preupdate optimizer")
        if count in states:
            audit.equal(event["after"], snaps["after" + str(count)], label + ".postupdate snapshot")
        passed = (event["error"] is None and "torch_rng_unchanged" in event
                  and rng_same and event["after"]["parameters_finite"]
                  and event["after"]["optimization_epochs"] == count // 64 * 4
                  and event["after"]["optimizer"]["step_min"] == count // 64 * 4
                  and event["after"]["optimizer"]["step_max"] == count // 64 * 4)
        audit.equal(event["passed"], bool(passed), label + ".optimizer update validation")
    # A failed update can lack the boolean because the driver never reached
    # its normal comparison; the aggregate starts true and changes only then.
    if trial["snapshots"] or events:
        audit.equal(trial["training_rng_unchanged"], rng_unchanged, label + ".training RNG aggregate")
    else:
        audit.check(type(trial["training_rng_unchanged"]) is bool, label + ".no-update RNG aggregate")
    if "after640" in snaps and "after384" in snaps:
        fixed = all(snaps["after640"][key] == snaps["after384"][key]
                    for key in ("parameter_sha256", "optimizer", "num_timesteps", "optimization_epochs"))
    else:
        fixed = False
    audit.equal(trial["continuation_weights_and_optimizer_unchanged"], fixed, label + ".frozen continuation")
    safe = exact and trial["error"] is None and trial["close_error"] is None and computed["safety_stops"] == 0
    wall = trial["wall_s"] > 180
    audit.equal(trial["wall_cap_exceeded"], wall, label + ".wall cap")
    complete = safe and not wall and fixed and len(rows) == 640 and computed["native_steps"] == 3200 and trial["reset_count"] == 1
    audit.equal(trial["safe"], safe, label + ".safety accounting")
    audit.equal(trial["complete"], complete, label + ".completion accounting")
    audit.equal(trial["status"], "complete" if complete else "incomplete", label + ".operational status")
    if complete:
        audit.equal(set(snaps), {"initial", *(f"after{n}" for n in UPDATE_COUNTS), "after640"}, label + ".complete snapshot matrix")
        audit.equal(set(states), {0} if active == "frozen" else {0, *UPDATE_COUNTS}, label + ".complete checkpoint matrix")
        audit.equal(len(events), 0 if active == "frozen" else 6, label + ".complete optimizer updates")
        audit.check(all(e["passed"] and e["error"] is None for e in events), label + ".completed updates")
    final = trial.get("retained_final_snapshot")
    if final and complete:
        audit.equal(final, {key: value for key, value in snaps["after640"].items() if key != "input_float32_le_sha256"},
                    label + ".retained terminal policy")
    return {"raw": trial, "reference": record, "rows": rows, "states": states, "tape": tape,
            "trial_id": label, "seed": expected["seed"],
            "arm": expected["arm"], "case_index": expected["case_index"], "guard_source": expected["guard_source"],
            "complete": complete, "safe": safe, "blocked": False, "exposure_exact": exact, "metrics": computed,
            "prefix_metrics": old.metrics(rows[:PREFIX]), "continuation_metrics": old.metrics(rows[PREFIX:]),
            "request_tapes": int(tape is not None), "checkpoint_count": len(states),
            "action_count": len(rows) + int(trial.get("pending_action_audit") is not None),
            "integrity_consistent": audit.failed == before_failures, "valid_post384": valid_post384(trial),
            "native_failure_completed_steps": replayed["failure_completed_native_steps"] if replayed else None,
            "native_failure_uncertain_steps_upper_bound": replayed["failure_uncertain_native_steps_upper_bound"] if replayed else None}


def expected_trial(study, releases, seed, arm, index, target=None):
    guard = target is not None
    entry = study["guard_cases" if guard else "target_cases"][index]
    suffix, source = "", None
    if guard and arm == "frozen":
        source = {"kind": "original_pooled_baseline"}
        weights = releases[seed]["final"]
        parameter = releases[seed]["final_parameter_sha256"]
    elif guard:
        source = {"kind": "post384_specialization", "case_id": target["raw"]["case_id"],
                  "case_index": target["case_index"], "arm": arm, "target_result": target["reference"]}
        suffix = "--" + target["raw"]["case_id"]
        weights = target["raw"].get("checkpoints", {}).get("after384") if target["valid_post384"] else None
        parameter = weights["parameter_sha256"] if weights else None
    else:
        kind = "initial" if arm == "scratch" else "final"
        weights, parameter = releases[seed][kind], releases[seed][kind + "_parameter_sha256"]
    name = "guard" if guard else "target"
    return {"trial_id": f"{name}--{entry['id']}--{seed}--{arm}{suffix}", "partition": name,
            "seed": seed, "arm": arm, "case_id": entry["id"], "case_index": index,
            "case": entry["case"], "checkpoint": weights, "parameter_sha256": parameter, "guard_source": source}


def stage_check(audit, root, name, sha, planned, expected, protocol, torch, np, loader, do_replay):
    folder = root / name
    stage, receipt = load(folder / "stage.json"), load(folder / "receipt.json")
    audit.equal(stage["schema"], "conservative_specialization_stage_v1", name + ".stage schema")
    audit.equal(stage["stage"], name, name + ".stage name")
    audit.equal(stage["protocol_sha256"], sha, name + ".stage protocol")
    audit.equal(receipt["schema"], "conservative_specialization_stage_receipt_v1", name + ".receipt schema")
    audit.equal(receipt["protocol_sha256"], sha, name + ".receipt protocol")
    audit.equal(reference(audit, receipt["stage"], name + ".stage receipt"), folder / "stage.json", name + ".stage receipt path")
    audit.equal(stage["planned_trials"], planned, name + ".planned opportunities")
    refs = stage["trial_results"]
    audit.check(len(refs) <= planned, name + ".opportunity budget")
    audited = []
    if stage["status"] == "blocked":
        audit.check(stage["blocked_reason"] is not None and not refs, name + ".global block receipt")
        audit.check(not (folder / "trials").exists() or not any((folder / "trials").rglob("*")),
                    name + ".blocked phase zero trial files")
    else:
        audit.check(stage["blocked_reason"] is None, name + ".active phase contradictory block")
        for index, record in enumerate(refs):
            audited.append(audit_trial(audit, record, expected[index], protocol, sha, torch, np, loader, do_replay))
        audit.equal(stage["status"], "complete" if len(refs) == planned else "incomplete", name + ".attempt completion")
        if len(refs) < planned:
            audit.check(bool(audited) and audited[-1]["raw"]["cancelled"], name + ".only cancellation truncates schedule")
    audit.equal(stage["collected_trials"], len(audited), name + ".collected attempts")
    for key, value in (("actual_decisions", sum(r["metrics"]["decisions"] for r in audited)),
                       ("native_steps", sum(r["metrics"]["native_steps"] for r in audited)),
                       ("case_construction_requests", sum(r["raw"]["case_construction_requests"] for r in audited)),
                       ("exposure_exact", all(r["exposure_exact"] for r in audited)),
                       ("all_trials_complete_safe", len(audited) == planned and all(r["complete"] and r["safe"] for r in audited))):
        audit.equal(stage[key], value, name + "." + key)
    found = {p.resolve() for p in (folder / "trials").glob("*/result.json")} | {
        p.resolve() for p in (folder / "trials").glob("*/archival-failure.json")}
    listed = {reference(audit, r, name + ".listed result").resolve() for r in refs}
    # An emergency receipt can retain an already written result.json. It must
    # be bound as a failed prefix, rather than counted as a second opportunity.
    retained = {reference(audit, p, "retained emergency result").resolve() for r in audited
                for p in r["raw"].get("retained_prefix", []) if p["path"].endswith("/result.json")}
    audit.equal(found, listed | retained, name + ".exact retained attempt matrix")
    return stage, audited


def target_seal_check(audit, stage, targets, sha):
    if stage["seal"] is None:
        audit.check(stage["status"] == "blocked", "active target phase missing seal")
        return None
    seal = bound(audit, stage["seal"], "target seal")
    audit.equal(seal["schema"], "conservative_specialization_target_seal_v1", "target seal schema")
    audit.equal(seal["protocol_sha256"], sha, "target seal protocol")
    for key, wanted in (("planned_trials", 144), ("collected_trials", len(targets)),
                        ("all_results_collected", len(targets) == 144), ("sealed_before_guard_worlds", True),
                        ("trial_results", stage["trial_results"]), ("guard_case_construction_requests", 0)):
        audit.equal(seal[key], wanted, "target seal." + key)
    unique = {}
    for trial in targets:
        records = [trial["reference"]]
        raw = trial["raw"]
        records += [raw[key] for key in ("checkpoint", "request_tape", "telemetry", "learning_log") if raw.get(key)]
        records += list(raw.get("checkpoints", {}).values()) + raw.get("retained_prefix", [])
        for item in records:
            reference(audit, item, "sealed evidence")
            audit.check(item["path"] not in unique or unique[item["path"]] == item["sha256"], "conflicting seal hashes")
            unique[item["path"]] = item["sha256"]
    audit.equal(seal["bound_evidence"], [{"path": name, "sha256": sha} for name, sha in sorted(unique.items())],
                "exact target evidence seal")
    return seal


def paired_randomness(audit, targets):
    indexed = {(r["seed"], r["case_index"], r["arm"]): r for r in targets}
    for seed in SEEDS:
        for index in range(12):
            pretrained = [indexed.get((seed, index, arm)) for arm in ("frozen", "conservative", "standard")]
            if all(pretrained):
                limit = min(64, *(len(r["rows"]) for r in pretrained))
                for row in range(limit):
                    base = pretrained[0]["rows"][row]
                    for other in pretrained[1:]:
                        audit.equal({key: other["rows"][row][key] for key in ("action", "native")},
                                    {key: base[key] for key in ("action", "native")}, f"paired pretrained prefix {seed}/{index}/{row}")
            group = [indexed.get((seed, index, arm)) for arm in ARMS]
            if all(group):
                limit = min(len(r["rows"]) for r in group)
                for row in range(limit):
                    for other in group[1:]:
                        for key in ("torch_rng_before_sha256", "torch_rng_after_sha256"):
                            audit.equal(other["rows"][row][key], group[0]["rows"][row][key],
                                        f"matched whole-opportunity action RNG {seed}/{index}/{row}.{key}")


def scientific_analysis(targets, guards, study):
    index = {(r["seed"], r["case_index"], r["arm"]): r for r in targets}
    contrasts = []
    for arm in ("conservative", "standard"):
        for comparator in ("frozen", "scratch"):
            pairs, means = [], []
            for seed in SEEDS:
                for case in range(12):
                    first, second = index.get((seed, case, arm)), index.get((seed, case, comparator))
                    present = first is not None and second is not None
                    realized = present and not first.get("unknown_exposure") and not second.get("unknown_exposure")
                    eligible = present and all(r["complete"] and r["safe"] and r["integrity_consistent"] for r in (first, second))
                    def difference(key):
                        return first[key]["native_reward"] - second[key]["native_reward"] if realized else None
                    pairs.append({"seed": seed, "case_index": case, "case_id": study["target_cases"][case]["id"],
                                  "eligible": bool(eligible), "whole_gain": difference("metrics"),
                                  "prefix_gain": difference("prefix_metrics"), "continuation_gain": difference("continuation_metrics"),
                                  "equal_complete_safe_exposure": bool(eligible),
                                  "first_decisions": first["metrics"]["decisions"] if first else None,
                                  "comparator_decisions": second["metrics"]["decisions"] if second else None})
                group = [p for p in pairs if p["seed"] == seed]
                raw = [p["whole_gain"] for p in group if p["whole_gain"] is not None]
                safe = [p["whole_gain"] for p in group if p["eligible"]]
                means.append({"seed": seed, "all_realized_mean": statistics.mean(raw) if len(raw) == 12 else None,
                              "complete_safe_pair_mean": statistics.mean(safe) if safe else None,
                              "complete_safe_pairs": len(safe)})
            raw_means = [r["all_realized_mean"] for r in means]
            median = statistics.median(raw_means) if all(v is not None for v in raw_means) else None
            positives = sum(v is not None and v > 0 for v in raw_means)
            all_safe = len(pairs) == 36 and all(p["eligible"] for p in pairs)
            threshold = 3.2 if comparator == "frozen" else 0.
            whole = [p["whole_gain"] for p in pairs if p["whole_gain"] is not None]
            prefix = [p["prefix_gain"] for p in pairs if p["prefix_gain"] is not None]
            contrasts.append({"arm": arm, "comparator": comparator, "pairs": pairs, "seed_means": means,
                              "median_seed_mean": median, "positive_seed_means": positives,
                              "all_pairs_eligible": all_safe, "threshold_strictly_greater_than": threshold,
                              "passed": bool(all_safe and median is not None and median > threshold and positives >= 2),
                              "target_reliability": {
                                  "each_pair_gain_at_least": -3.2,
                                  "below_floor_count": sum(v < -3.2 for v in whole),
                                  "worst_gain": min(whole, default=None),
                                  "passed": bool(comparator == "frozen" and all_safe and len(whole) == 36
                                                 and all(v >= -3.2 for v in whole))},
                              "regression_diagnostic": {"whole_loss_floor": -3.2, "prefix_loss_floor": -1.92,
                                  "whole_below_floor_count": sum(v < -3.2 for v in whole),
                                  "prefix_below_floor_count": sum(v < -1.92 for v in prefix),
                                  "worst_whole_gain": min(whole, default=None), "worst_prefix_gain": min(prefix, default=None)}})
    baseline = {(r["seed"], r["case_index"]): r for r in guards if r["arm"] == "frozen"}
    retention = []
    for arm in ("conservative", "standard"):
        summaries, all_pairs = [], []
        for seed in SEEDS:
            pairs = []
            for target in range(12):
                for guard in range(2):
                    current = next((r for r in guards if r["seed"] == seed and r["arm"] == arm and r["case_index"] == guard
                                    and r["guard_source"]["case_index"] == target), None)
                    base = baseline.get((seed, guard))
                    eligible = current is not None and base is not None and all(r["complete"] and r["safe"] and r["integrity_consistent"] for r in (current, base))
                    realized = current is not None and base is not None and not current["blocked"] and not current.get("unknown_exposure") and not base.get("unknown_exposure")
                    gain = current["metrics"]["native_reward"] - base["metrics"]["native_reward"] if realized else None
                    pairs.append({"seed": seed, "target_case_index": target, "guard_case_index": guard,
                                  "eligible": bool(eligible), "post_minus_base": gain,
                                  "blocked": bool(current and current["blocked"])})
            raw = [p["post_minus_base"] for p in pairs if p["post_minus_base"] is not None]
            safe = [p["post_minus_base"] for p in pairs if p["eligible"]]
            mean = statistics.mean(raw) if len(raw) == 24 else None
            all_safe = all(p["eligible"] for p in pairs)
            summaries.append({"seed": seed, "all_realized_mean": mean,
                              "complete_safe_pair_mean": statistics.mean(safe) if safe else None,
                              "eligible_pairs": len(safe), "worst_gain": min(raw, default=None),
                              "below_minus3_2_count": sum(v < -3.2 for v in raw),
                              "passed": bool(all_safe and mean is not None and mean >= -3.2)})
            all_pairs.extend(pairs)
        retention.append({"arm": arm, "pairs": all_pairs, "seed_summaries": summaries,
                          "passed": all(r["passed"] for r in summaries),
                          "probe_reliability": {
                              "each_pair_gain_at_least": -3.2,
                              "worst_gain": min((p["post_minus_base"] for p in all_pairs if p["post_minus_base"] is not None), default=None),
                              "below_floor_count": sum(p["post_minus_base"] is not None and p["post_minus_base"] < -3.2 for p in all_pairs),
                              "passed": bool(len(all_pairs) == 72 and all(p["eligible"] and p["post_minus_base"] is not None
                                                                         and p["post_minus_base"] >= -3.2 for p in all_pairs))},
                          "scope": "Two prospectively selected old conditions only"})
    combined = []
    for arm in ("conservative", "standard"):
        benefit = next(c for c in contrasts if c["arm"] == arm and c["comparator"] == "frozen")
        probes = next(r for r in retention if r["arm"] == arm)
        components = {"useful_benefit": benefit["passed"], "target_reliability": benefit["target_reliability"]["passed"],
                      "sampled_retention": probes["passed"], "probe_reliability": probes["probe_reliability"]["passed"]}
        combined.append({"arm": arm, **components, "passed": all(components.values())})
    return {"target_contrasts": contrasts, "sampled_retention": retention,
            "combined_readiness": combined, "primary_benefit_passed": contrasts[0]["passed"],
            "secondary_benefit_passed": contrasts[2]["passed"],
            "primary_passed": combined[0]["passed"], "secondary_passed": combined[1]["passed"],
            "interpretation": "Whole-opportunity realized return includes the384-decision learning cost. Failed/stopped prefixes remain in all-case records; complete-safe means are descriptive and do not replace all-pair eligibility. These12 target conditions were previously exposed; no new body-generalization claim."}


def public_trial(trial):
    return {key: value for key, value in trial.items() if key not in ("raw", "rows", "states", "tape")}


def historical_repeatability(audit, targets, historical):
    records = []
    for trial in targets:
        if trial["arm"] not in ("frozen", "standard"):
            continue
        previous_arm = "full" if trial["arm"] == "standard" else "frozen"
        previous = historical[(trial["seed"], trial["case_index"], previous_arm)]
        label = trial["trial_id"] + ".historical integrity repeat"
        before = audit.failed
        limit = min(len(trial["rows"]), 128 if trial["arm"] == "standard" else 640)
        audit.equal(trial["rows"][:limit], previous["rows"][:limit], label + ".actions/RNG/observations")
        if trial["raw"].get("initial_local_observation") is not None:
            audit.equal(trial["raw"]["initial_local_observation"], previous["raw"]["initial_local_observation"],
                        label + ".initial observation")
        else:
            audit.check(not trial["rows"], label + ".missing initial observation cannot accompany recorded actions")
        states = trial.get("states", {})
        for count in ((0, 64, 128) if trial["arm"] == "standard" else (0,)):
            if count in states:
                audit.equal(states[count]["summary"], previous["states"][count]["summary"], label + f".actual tensors {count}")
                audit.equal(states[count]["optimizer"], previous["states"][count]["optimizer"], label + f".actual Adam {count}")
        for name in (("initial", "after64", "after128") if trial["arm"] == "standard"
                     else ("initial", "after64", "after128", "after640")):
            current_snapshot = trial["raw"].get("snapshots", {}).get(name)
            previous_snapshot = previous["raw"]["snapshots"].get(name)
            if current_snapshot is not None and previous_snapshot is not None:
                keys = ("parameter_sha256", "partitions", "optimizer", "num_timesteps", "optimization_epochs",
                        "torch_rng_sha256", "input_float32_le_sha256", "requires_grad", "trainable_elements",
                        "trainable_tensors", "rollout_buffer")
                audit.equal({k: current_snapshot[k] for k in keys}, {k: previous_snapshot[k] for k in keys},
                            label + "." + name + ".state/RNG/history")
        full_stream = trial["arm"] == "frozen" and trial["complete"] and trial.get("tape") is not None
        if full_stream:
            audit.equal(trial["tape"]["requests"], previous["tape"]["requests"], label + ".all retained requests")
            audit.equal(trial["tape"]["native_stream_sha256"], previous["tape"]["native_stream_sha256"],
                        label + ".full native stream digest")
        records.append({"trial_id": trial["trial_id"], "previous_trial_reference": previous["reference"],
                        "retained_prefix_decisions_compared": limit, "full_frozen_stream_compared": bool(full_stream),
                        "standard128_complete": trial["arm"] == "standard" and limit == 128 and {0, 64, 128} <= set(states),
                        "integrity_consistent": audit.failed == before,
                        "scope": "Exact repeat on already exposed conditions, not an independent scientific replicate"})
    return {"records": records, "old_primary_failed_gate_preserved": True, "old_secondary_failed_gate_preserved": True,
            "full_frozen_streams_compared": sum(r["full_frozen_stream_compared"] for r in records),
            "standard128_prefixes_compared": sum(r["standard128_complete"] for r in records)}


def verify(root, do_replay, audit, expected_sha):
    if sys.platform != "linux" or os.name == "nt" or PROJECT != Path("/workspace") or not Path("/.dockerenv").is_file():
        raise ValueError("Run checkpoint/accounting audit in the documented /workspace Linux CPU container")
    import numpy as np
    import torch
    from stable_baselines3.common.save_util import load_from_zip_file
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol, sha, releases = prerequisites(root, audit, expected_sha, torch, load_from_zip_file)
    historical = historical_inputs(audit, protocol, torch, load_from_zip_file)
    if audit.failed:
        raise ValueError("Audit refused before native replay: prospectively bound protocol, source, dependency or release validation failed")
    study = protocol["study"]
    expected_targets = [expected_trial(study, releases, seed, arm, index)
                        for seed in SEEDS for index in range(12) for arm in ARMS]
    target_stage, targets = stage_check(audit, root, "targets", sha, 144, expected_targets, protocol,
                                       torch, np, load_from_zip_file, do_replay)
    seal = target_seal_check(audit, target_stage, targets, sha)
    paired_randomness(audit, targets)
    repeatability = historical_repeatability(audit, targets, historical)
    indexed = {(r["seed"], r["case_index"], r["arm"]): r for r in targets}
    expected_guards = [expected_trial(study, releases, seed, "frozen", guard, target={})
                       for seed in SEEDS for guard in range(2)]
    if len(targets) == 144:
        expected_guards += [expected_trial(study, releases, seed, arm, guard, indexed[(seed, target, arm)])
                            for seed in SEEDS for target in range(12) for arm in ("conservative", "standard") for guard in range(2)]
    guard_stage_path = root / "guards" / "stage.json"
    if not guard_stage_path.is_file():
        raise ValueError("Declared guard stage receipt is missing; audit cannot infer a zero-exposure release decision")
    declared_guard = load(guard_stage_path)
    if declared_guard["status"] != "blocked":
        audit.check(target_stage["status"] == "complete" and len(targets) == 144 and seal is not None,
                    "all target attempts sealed before any guard")
        if audit.failed:
            raise ValueError("Guard replay refused because target sealing or accounting failed")
        release = load(root / "guards" / "release.json")
        audit.equal(release, {"schema": "conservative_specialization_guard_release_v1", "protocol_sha256": sha,
                             "target_seal": target_stage["seal"], "released_before_first_guard_world": True},
                    "guard release receipt")
    else:
        audit.check(not (root / "guards" / "release.json").exists(), "blocked global guard phase has no release")
    guard_stage, guards = stage_check(audit, root, "guards", sha, 150, expected_guards, protocol,
                                     torch, np, load_from_zip_file, do_replay)
    if guard_stage["status"] != "blocked":
        audit.equal(guard_stage["seal"], target_stage["seal"], "guard release bound target seal")
    else:
        audit.equal(guard_stage["seal"], None, "blocked global guard phase has no selected seal")
    analysis = scientific_analysis(targets, guards, study)
    all_trials = targets + guards
    decisions = sum(r["metrics"]["decisions"] for r in all_trials)
    native_steps = sum(r["metrics"]["native_steps"] for r in all_trials)
    audit.check(decisions <= 188160 and native_steps <= 940800, "aggregate opportunity cap")
    # The only configure membership allowed above is exact target12 / guard2.
    # This study intentionally exposes no original rover-v3 challenge world.
    return {"protocol_sha256": sha, "lock_sha256": digest(root / "lock.json"),
            "target_stage_sha256": digest(root / "targets" / "stage.json"),
            "guard_stage_sha256": digest(root / "guards" / "stage.json"),
            "target_seal_sha256": digest(reference(audit, target_stage["seal"], "target seal")) if target_stage["seal"] else None,
            "source_files_verified": len(protocol["source_sha256"]),
            "historical_sources_verified": 3, "target_trials": [public_trial(r) for r in targets],
            "guard_trials": [public_trial(r) for r in guards], "analysis": analysis, "historical_repeatability": repeatability,
            "collected_opportunities": len(all_trials), "completed_safe_opportunities": sum(r["complete"] and r["safe"] for r in all_trials),
            "blocked_guard_opportunities": sum(r["blocked"] for r in guards),
            "request_tapes_verified": sum(r["request_tapes"] for r in all_trials),
            "fresh_and_prefix_checkpoints_verified": sum(r["checkpoint_count"] for r in all_trials),
            "categorical_actions_verified": sum(r["action_count"] for r in all_trials),
            "native_actions_regenerated": True,
            "native_request_streams_replayed": sum(r["request_tapes"] for r in all_trials) if do_replay else 0,
            "actual_decisions": decisions, "recorded_successful_native_controls": native_steps,
            "actual_case_construction_requests": sum(r["raw"]["case_construction_requests"] for r in all_trials),
            "all_exposure_exact": all(r["exposure_exact"] for r in all_trials),
            "original_rover_v3_configuration_calls": 0, "new_geometry_configuration_calls": 0,
            "unresolved_requests_replayed": False,
            "operationally_complete": target_stage["status"] == guard_stage["status"] == "complete"
                and all(r["complete"] and r["safe"] for r in all_trials),
            "scope": "Full record=False native request/response digest replay, local observations/reward accounting, functional Torch actor actions and RNG/history bytes, actual checkpoint tensors/Adam states and frozen scopes are independently checked. PPO gradients/updates are not rerun; wall timestamps do not establish external execution chronology. Scientific gates are distinct from integrity. Previously exposed targets only, no new held-out generalization or broad retention claim."}


def self_test():
    tests = []
    def expect(name, condition):
        tests.append({"name": name, "passed": bool(condition)})
    case = old.fixture((5, 3), False, 0, -1)
    tape = {"requests": [{"op": "spec"}, {"op": "configure", "cases": [case], "record": False},
                         {"op": "step", "actions": [1], "record": False}], "unresolved_request": None}
    audit = Audit()
    draws, actions = single_world_schedule(audit, tape, case, "synthetic")
    expect("valid single-case tape", audit.failed == 0 and draws == [0] and len(actions) == 1)
    for name, request in (
        ("reject other geometry", {"op": "configure", "cases": [old.fixture((3, 3), False, 0, -1)], "record": False}),
        ("reject reset rescue", {"op": "reset", "slots": [{"index": 0, "case": case}], "record": False}),
        ("reject invalid action", {"op": "step", "actions": [3], "record": False}),
        ("reject record True", {"op": "step", "actions": [1], "record": True})):
        changed = {**tape, "requests": tape["requests"][:2] + [request]}
        try:
            single_world_schedule(Audit(), changed, case, name)
            expect(name, False)
        except ValueError:
            expect(name, True)
    for name in ("../outside", "/outside", "a\\b"):
        try:
            safe_path(PROJECT, name)
            expect("reject path " + name, False)
        except ValueError:
            expect("reject path " + name, True)
    for name, requests in (
        ("641st attempted action rejected", tape["requests"][:2] + [tape["requests"][2]] * 641),
        ("second configure rejected", tape["requests"] + [tape["requests"][1]])):
        check = Audit()
        single_world_schedule(check, {**tape, "requests": requests}, case, name)
        expect(name, check.failed > 0)
    expect("384 decisions six rollouts and24 steps", PREFIX == 384 and UPDATE_COUNTS == (64, 128, 192, 256, 320, 384)
           and len(UPDATE_COUNTS) * 4 == 24 and PREFIX + 256 == 640)
    expect("942 checkpoints at full294 opportunity cap", 294 + 3 * 12 * 3 * len(UPDATE_COUNTS) == 942)
    candidate = {"checkpoints": {"after384": {"decisions": 384, "parameter_sha256": "synthetic"}},
                 "training_updates": [{"decisions": n, "passed": True, "error": None} for n in UPDATE_COUNTS],
                 "snapshots": {"after384": {"parameters_finite": True, "num_timesteps": 384, "optimization_epochs": 24,
                     "parameter_sha256": "synthetic", "optimizer": {"step_min": 24, "step_max": 24}}}}
    expect("six successful updates release384 candidate", valid_post384(candidate))
    candidate["training_updates"][2]["error"] = "synthetic archival failure"
    expect("failed update blocks candidate despite24-step label", not valid_post384(candidate))
    study = {"target_cases": [{"id": "synthetic-" + str(i)} for i in range(12)]}
    targets = []
    for seed in SEEDS:
        for i in range(12):
            for arm in ARMS:
                reward = 4. if arm in ("conservative", "standard") else 0.
                targets.append({"seed": seed, "case_index": i, "arm": arm, "complete": True, "safe": True,
                                "integrity_consistent": True, "metrics": {"decisions": 640, "native_reward": reward},
                                "prefix_metrics": {"native_reward": reward * .6}, "continuation_metrics": {"native_reward": reward * .4}})
    guards = []
    for seed in SEEDS:
        for i in range(2):
            guards.append({"seed": seed, "case_index": i, "arm": "frozen", "complete": True, "safe": True,
                           "blocked": False, "integrity_consistent": True, "guard_source": {"kind": "original_pooled_baseline"},
                           "metrics": {"native_reward": 0.}})
        for arm in ("conservative", "standard"):
            for target in range(12):
                for i in range(2):
                    guards.append({"seed": seed, "case_index": i, "arm": arm, "complete": True, "safe": True,
                                   "blocked": False, "integrity_consistent": True, "guard_source": {"case_index": target},
                                   "metrics": {"native_reward": 0.}})
    result = scientific_analysis(targets, guards, study)
    expect("strict whole-cost benefit positive fixture", result["primary_passed"] and result["secondary_passed"])
    first = next(r for r in targets if r["arm"] == "conservative")
    first["safe"] = False
    expect("one failed pair blocks primary only", not scientific_analysis(targets, guards, study)["primary_passed"]
           and scientific_analysis(targets, guards, study)["secondary_passed"])
    first["safe"] = True
    for r in targets:
        if r["arm"] == "conservative":
            r["metrics"]["native_reward"] = 3.2
    expect("strict3.2 benefit boundary fails", not scientific_analysis(targets, guards, study)["primary_passed"])
    for r in targets:
        if r["arm"] == "conservative":
            r["metrics"]["native_reward"] = 4.
    first["metrics"]["native_reward"] = -3.2
    expect("inclusive target floor exactly-3.2 passes reliability", scientific_analysis(targets, guards, study)["target_contrasts"][0]["target_reliability"]["passed"])
    first["metrics"]["native_reward"] = math.nextafter(-3.2, -math.inf)
    derived = scientific_analysis(targets, guards, study)
    expect("one target below floor blocks readiness despite positive mean", derived["primary_benefit_passed"]
           and not derived["primary_passed"] and derived["secondary_passed"])
    first["metrics"]["native_reward"] = 4.
    probe = next(g for g in guards if g["arm"] == "conservative")
    probe["metrics"]["native_reward"] = -3.2
    expect("inclusive probe floor exactly-3.2 passes", scientific_analysis(targets, guards, study)["primary_passed"])
    probe["metrics"]["native_reward"] = math.nextafter(-3.2, -math.inf)
    derived = scientific_analysis(targets, guards, study)
    expect("one probe regression hidden by acceptable seed mean blocks readiness", derived["sampled_retention"][0]["passed"]
           and not derived["sampled_retention"][0]["probe_reliability"]["passed"] and not derived["primary_passed"])
    probe["metrics"]["native_reward"] = 0.
    probe["blocked"] = True
    probe["safe"] = probe["complete"] = False
    expect("blocked guard invalidates readiness without rescue", not scientific_analysis(targets, guards, study)["primary_passed"])
    probe["blocked"] = False
    probe["safe"] = probe["complete"] = True
    scratch = next(t for t in targets if t["arm"] == "scratch")
    scratch["safe"] = False
    derived = scientific_analysis(targets, guards, study)
    expect("scratch failure affects separate scratch contrast", derived["primary_passed"]
           and not derived["target_contrasts"][1]["passed"])
    expect("missing probes fail sampled retention", not scientific_analysis(targets, [], study)["sampled_retention"][0]["passed"])
    return {"schema": "conservative_specialization_auditor_synthetic_checks_v1", "native_worlds_constructed": 0,
            "training_updates": 0, "checks": tests, "passed": all(t["passed"] for t in tests)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="existing specialization report directory")
    parser.add_argument("--receipt", type=Path, help="exclusive new verification JSON path")
    parser.add_argument("--source-archive", type=Path, help="existing retained exact copy of this auditor")
    parser.add_argument("--expected-protocol-sha256", help="owner's external prospective lock commitment")
    parser.add_argument("--replay", action="store_true", help="replay all retained native request/full response streams")
    parser.add_argument("--self-test", action="store_true", help="pure synthetic negative guards; constructs no native worlds")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
        result.update({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                       "verifier_source_path": str(Path(__file__).resolve()),
                       "verifier_source_sha256": digest(Path(__file__)),
                       "stable_native_helper_path": str(HELPER_PATH),
                       "stable_native_helper_sha256": digest(HELPER_PATH),
                       "observed_invocation": list(sys.orig_argv),
                       "python_version": sys.version,
                       "scope": "Pure synthetic guards and reductions; no collection results read, no native process or training"})
        if args.receipt:
            with args.receipt.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
        print(canonical(result))
        return 0 if result["passed"] else 1
    if args.output is None or args.receipt is None:
        parser.error("--output and --receipt are required except with --self-test")
    root, receipt = args.output.resolve(), args.receipt.resolve()
    if not root.is_dir() or not receipt.parent.is_dir() or receipt.is_symlink():
        raise ValueError("Existing report directory and receipt parent required; symlink receipts forbidden")
    audit = Audit()
    with receipt.open("x", encoding="utf-8") as stream:
        own_path, own_sha = Path(__file__).resolve(), digest(Path(__file__))
        result = {"schema": "conservative_specialization_independent_verification_v1", "input": str(root),
                  "created_at_utc": datetime.now(timezone.utc).isoformat(), "native_replay_requested": args.replay,
                  "verifier_source_path": str(own_path), "verifier_source_sha256": own_sha,
                  "stable_native_helper_path": str(HELPER_PATH), "stable_native_helper_sha256": digest(HELPER_PATH)}
        try:
            if args.source_archive:
                archive = args.source_archive.resolve()
                audit.equal(digest(archive), own_sha, "archived auditor bytes")
            else:
                archive = receipt.parent / "auditor_snapshot" / (own_sha + ".py")
                archive.parent.mkdir(exist_ok=True)
                if archive.exists():
                    audit.equal(digest(archive), own_sha, "existing exact auditor archive")
                else:
                    with archive.open("xb") as source:
                        source.write(own_path.read_bytes())
            result["verifier_source_archive_path"] = str(archive)
            helper_archive = receipt.parent / "auditor_snapshot" / (result["stable_native_helper_sha256"] + ".py")
            helper_archive.parent.mkdir(exist_ok=True)
            if helper_archive.exists():
                audit.equal(digest(helper_archive), result["stable_native_helper_sha256"], "existing exact stable helper archive")
            else:
                with helper_archive.open("xb") as source:
                    source.write(HELPER_PATH.read_bytes())
            result["stable_native_helper_archive_path"] = str(helper_archive)
            result.update(verify(root, args.replay, audit, args.expected_protocol_sha256))
            audit.equal(digest(own_path), own_sha, "auditor source unchanged during verification")
            audit.equal(digest(HELPER_PATH), result["stable_native_helper_sha256"], "stable helper unchanged during verification")
        except Exception as error:
            audit.check(False, f"{type(error).__name__}: {error}")
        result.update({"integrity_passed": audit.failed == 0, "checks": audit.checks, "failed_checks": audit.failed,
                       "issues": audit.issues, "issues_truncated": audit.failed > len(audit.issues),
                       "native_replay_verified": args.replay and audit.failed == 0})
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(canonical({"receipt": str(receipt), "integrity_passed": audit.failed == 0, "failed_checks": audit.failed}))
    return 0 if audit.failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(canonical({"error": f"{type(error).__name__}: {error}", "existing_evidence_preserved": True}), file=sys.stderr)
        raise SystemExit(1)
