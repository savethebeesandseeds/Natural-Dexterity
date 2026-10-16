#!/usr/bin/env python3
"""Independent standard-library audit of the prospective breadth study.

The completed pilot verifier supplies immutable native replay/metric helpers.
Neither learner nor adapter is imported. Policy bytes and parameter labels are
bound; neural inference, tensor deserialization and optimizer replay are outside
this audit. Every verification receipt is written exclusively to a new path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import statistics
import sys


PROJECT = Path(__file__).resolve().parents[1]
HELPER_PATH = PROJECT / "tools" / "verify-generalization-pretraining.py"
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("independent_pilot_audit", HELPER_PATH)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
Audit, canonical, digest, load, safe_path = old.Audit, old.canonical, old.digest, old.load, old.safe_path
SEEDS = (101, 202, 303)
SOURCE_ARMS = ("pooled", "narrow")
POLICY_ARMS = ("pooled", "narrow", "untrained")
WITNESSES = ("quiet", "negative", "positive", "rhythm_2s", "rhythm_4s")
WINDOWS = (64, 128, 320, 640)


def declaration(audit, study):
    """Independent physical matrix and gate constants; no world construction."""
    source = [old.fixture(lengths, weighted, hinge, side)
              for lengths in ((3, 3), (4, 3), (3, 4), (4, 4))
              for weighted in (False, True) for hinge in (0, 1) for side in (-1, 1)]
    known = [old.fixture(lengths, weighted, hinge, side)
             for lengths, weighted in (((4, 3), False), ((3, 4), False), ((4, 4), True))
             for hinge in (0, 1) for side in (-1, 1)]
    reserved = [old.fixture(lengths, weighted, hinge, side)
                for lengths, weighted in (((5, 3), False), ((3, 5), False), ((5, 5), True))
                for hinge in (0, 1) for side in (-1, 1)]
    audit.equal(study["schema"], "generalization_study_v1", "study schema")
    matrices = (("pooled source", study["source_cases"]["pooled"], source),
                ("narrow source", study["source_cases"]["narrow"], source[:4]),
                ("development", study["development_cases"], known),
                ("narrow development", study["narrow_development_cases"], source[:4]),
                ("reserved", study["reserved_cases"], reserved))
    for label, actual, expected in matrices:
        audit.equal(len(actual), len(expected), f"{label}.count")
        for index, (a, b) in enumerate(zip(actual, expected)):
            audit.equal(old.mechanics(a), old.mechanics(b), f"{label}[{index}].mechanics")
            audit.equal(a["sun"], b["sun"], f"{label}[{index}].sun")
    sources = {old.mechanics(c) for cases in study["source_cases"].values() for c in cases}
    audit.check(not sources & {old.mechanics(c) for c in study["reserved_cases"]},
                "source/reserved whole-assembly mechanics overlap")
    source_ids = [f"source-L{a}-{b}-{'distal-block' if weight else 'plain'}-"
                  f"{'base' if hinge == 0 else 'elbow'}-{'left' if side < 0 else 'right'}"
                  for a, b in ((3, 3), (4, 3), (3, 4), (4, 4)) for weight in (False, True)
                  for hinge in (0, 1) for side in (-1, 1)]
    def other_ids(prefix):
        return [f"{prefix}-{name}-{'base' if hinge == 0 else 'elbow'}-{'left' if side < 0 else 'right'}"
                for name in ("long-upper", "long-lower", "weighted") for hinge in (0, 1) for side in (-1, 1)]
    audit.equal(study["case_ids"], {"source": {"pooled": source_ids,
                "narrow": [s.replace("source-", "narrow-", 1) for s in source_ids[:4]]},
                "development": other_ids("development"),
                "narrow_development": [s.replace("source-", "narrow-own-", 1) for s in source_ids[:4]],
                "reserved": other_ids("reserved")}, "case IDs")
    audit.equal(study["source_seeds"], list(SEEDS), "source seeds")
    audit.equal(study["source_arms"], list(SOURCE_ARMS), "source arms")
    audit.equal(study["source_budget"], {"decisions": 98304, "n_envs": 8, "n_steps": 256, "horizon": 640}, "source budget")
    audit.equal(study["evaluation_budget"], {"decisions": 640, "windows": list(WINDOWS)}, "evaluation budget")
    audit.equal(study["decision_dt_s"], .1, "decision interval")
    audit.equal(study["native_dt_s"], .02, "native interval")
    audit.equal(study["sampling"]["schema"], "shuffled_source_cycles_v1", "sampler schema")
    audit.equal(study["sampling"]["seed_offset"], 500000, "sampler RNG offset")
    audit.equal(study["action_rng"]["base"], 100000, "action RNG base")
    audit.equal(study["action_rng"]["seed_multiplier"], 10, "action RNG seed multiplier")
    audit.equal(study["action_rng"]["offsets"], {"development": 0, "narrow_development": 100, "reserved": 200}, "action RNG offsets")
    audit.equal(study["challenge_arms"], list(POLICY_ARMS), "challenge arms")
    audit.equal(study["witness_modes"], list(WITNESSES), "witness modes")
    audit.equal(study["gates"], {"source_margin": 3.2, "source_requires_all_pooled_known_and_narrow_own": True,
                "breadth_median_seed_mean_positive": True, "breadth_min_positive_seeds": 2,
                "breadth_requires_all_pooled_narrow_and_quiet_safe": True,
                "useful_requires_all_pooled_gain_above": 3.2}, "prospective gates")
    for key, value in {"source_n_steps": 256, "source_n_envs": 8, "n_epochs": 4, "batch_size": 64}.items():
        audit.equal(study["ppo"][key], value, f"PPO.{key}")
    audit.equal(study["maximum_collection"], {"source_decisions": 589824, "development_decisions": 94720,
                "challenge_decisions_including_witnesses": 107520, "total_decisions": 792064,
                "native_controls": 3960320, "physics_steps": 39603200}, "maximum collection")


def schedule(audit, draws, seed, arm, count, reported, label):
    rng, expected = random.Random(seed + 500000), []
    while len(expected) < len(draws):
        cycle = list(range(count))
        rng.shuffle(cycle)
        expected.extend(cycle)
    audit.equal(draws, expected[:len(draws)], f"{label}.independent source schedule")
    recomputed = {"schema": "shuffled_source_cycles_v1", "source_seed": seed, "source_arm": arm,
                  "curriculum_seed": seed + 500000, "draws": len(draws),
                  "completed_cycles": len(draws) // count, "current_cycle_position": len(draws) % count,
                  "source_count": count, "draw_counts": [draws.count(i) for i in range(count)],
                  "draw_index_sha256": hashlib.sha256("".join(f"{i}\n" for i in draws).encode("ascii")).hexdigest()}
    audit.equal(reported, recomputed, f"{label}.source draw accounting")
    return recomputed


def checkpoints(audit, folder, result, label):
    if "policy_path" in result:
        audit.file(folder, result, "policy_sha256", "policy_path")
        audit.check(bool(old.HEX.fullmatch(result["parameter_sha256"])), f"{label}: invalid final parameter label")
    initial = result.get("initial_checkpoint")
    if initial is not None:
        audit.file(folder, initial, "policy_sha256", "policy_path")
        audit.equal(load(folder / "initial-checkpoint.json"), initial, f"{label}.initial checkpoint receipt")
        audit.equal(initial["weight_updates"], 0, f"{label}.initial updates")
        audit.equal(initial["source_decisions"], 0, f"{label}.initial experience")
        audit.check(bool(old.HEX.fullmatch(initial["parameter_sha256"])), f"{label}: invalid initial parameter label")
    return initial


def tape(audit, folder, result, cases, slots, source, replay, label):
    """Use independently tested request/replay code, with exact partition binding."""
    # Validate partition membership before launching a native process. Capture the
    # actual configure response during the same replay, never with an extra probe.
    values = old.verify_tape(audit, folder, result, cases, slots, source, False, label)
    if not replay:
        return values
    if audit.failed:
        raise ValueError("native replay refused after failed static integrity check")
    original, observations = old.NativeReplay, []
    class CapturingReplay(original):
        def call(self, request):
            response = super().call(request)
            if request.get("op") == "configure" and response.get("ok") is True:
                observations.extend(value["observation"] for value in response["result"])
            return response
    old.NativeReplay = CapturingReplay
    try:
        replayed = old.replay(audit, values[0], cases, slots, source, label)
    finally:
        old.NativeReplay = original
    replayed["configure_observations"] = observations
    return values[:3] + (replayed,)


def fixed_action(mode, ordinal):
    if mode == "quiet":
        return 1
    if mode == "negative":
        return 0
    if mode == "positive":
        return 2
    phase = 10 if mode == "rhythm_2s" else 20 if mode == "rhythm_4s" else None
    if phase is None:
        raise ValueError(f"unknown fixed controller {mode}")
    return 0 if ((ordinal - 1) // phase) % 2 == 0 else 2


def bound(audit, root, ref, label):
    path = audit.file(root, ref)
    return path, load(path)


def source_result(audit, root, path, result, protocol_hash, study, replay):
    arm, seed = result["source_arm"], result["seed"]
    label = f"source-{arm}-{seed}"
    audit.check(arm in SOURCE_ARMS and seed in SEEDS, f"{label}: undeclared source condition")
    folder = path.parent
    audit.equal(folder.relative_to(root).as_posix(), f"sources/{label}", f"{label}.location")
    audit.equal(result["protocol_sha256"], protocol_hash, f"{label}.protocol")
    audit.equal(result["reserved_case_construction_requests"], 0, f"{label}.reserved requests")
    if result["schema"] == "generalization_study_source_failure_v1":
        for key, expected in (("complete", False), ("unknown_exposure", True), ("exposure_exact", False),
                              ("actual_decisions", 0), ("native_steps", 0), ("initial_checkpoint", None)):
            audit.equal(result[key], expected, f"{label}.{key}")
        audit.check(bool(result["error"]), f"{label}: finalization failure has no error")
        audit.equal(result["counts_interpretation"],
                    "Zero is the known lower bound; native advancement may have occurred before finalization failed",
                    f"{label}.lower bound interpretation")
        retained = set()
        for ref in result["retained_prefix"]:
            target = audit.file(root, ref)
            target.relative_to(folder)
            retained.add(target.resolve())
        actual = {p.resolve() for p in folder.rglob("*") if p.is_file() and p != path}
        audit.equal(retained, actual, f"{label}.complete retained prefix manifest")
        recovered = None
        # A separately retained successful result can be audited even though the
        # seal correctly retains the finalization failure and blocks evaluation.
        if folder.joinpath("result.json").is_file():
            prefix = load(folder / "result.json")
            if prefix.get("schema") == "generalization_study_source_result_v1":
                recovered = source_result(audit, root, folder / "result.json", prefix,
                                          protocol_hash, study, replay)
        elif folder.joinpath("traces/source-requests.json.gz").is_file():
            raw = load(folder / "traces/source-requests.json.gz", True)
            old.request_schedule(audit, raw, study["source_cases"][arm], 8, True, label + ".prefix")
            if replay and not audit.failed:
                recovered = old.replay(audit, raw, study["source_cases"][arm], 8, True, label + ".prefix")
        return {"source_arm": arm, "seed": seed, "complete": False, "initial_checkpoint": None,
                "actual_decisions": 0, "native_steps": 0, "exposure_exact": False,
                "unknown_exposure": True, "request_tapes_verified": int(recovered is not None),
                "retained_prefix_recovery": recovered}
    audit.equal(result["schema"], "generalization_study_source_result_v1", f"{label}.schema")
    audit.equal(result["requested_decisions"], 98304, f"{label}.budget")
    audit.check(type(result["actual_decisions"]) is int and 0 <= result["actual_decisions"] <= 98304,
                f"{label}: source decision budget exceeded")
    initial = checkpoints(audit, folder, result, label)
    replayed, raw, draw_indices = None, None, []
    if "request_tape" in result:
        raw, draw_indices, actions, replayed = tape(audit, folder, result,
                study["source_cases"][arm], 8, True, replay, label)
        schedule(audit, draw_indices, seed, arm, len(study["source_cases"][arm]), result["source_schedule"], label)
        audit.check(sum(len(a) for _, a in actions) <= 98312, f"{label}: retained action opportunity exceeded")
        audit.equal(result["n_envs"], 8, f"{label}.vector size")
    else:
        audit.check(result["actual_decisions"] == 0 and not result["complete"], f"{label}: missing tape conceals experience")
    audit.equal({p.name for p in (folder / "traces").glob("*.gz")},
                {result["request_tape"]["path"]} if raw is not None else set(), f"{label}.exact trace inventory")
    exposure = result["per_source_case_exposure"]
    audit.check(len(exposure) in (0, len(study["source_cases"][arm])), f"{label}: incomplete case exposure array")
    for index, item in enumerate(exposure):
        audit.equal(item["case_index"], index, f"{label}.exposure index")
        audit.check(all(type(item[k]) is int and item[k] >= 0 for k in ("decisions", "native_steps", "safety_stops")),
                    f"{label}: invalid exposure count")
        audit.equal(item["duration_s"], .02 * item["native_steps"], f"{label}.exposure duration")
    for item_key, result_key in (("decisions", "actual_decisions"), ("native_steps", "native_steps"), ("safety_stops", "safety_stops")):
        audit.equal(sum(e[item_key] for e in exposure), result[result_key], f"{label}.{result_key} sum")
    if replayed is not None:
        audit.equal(exposure, replayed["exposure"], f"{label}.replayed per-case exposure")
        audit.equal(result["episodes"], replayed["episodes"], f"{label}.replayed episodes")
    exact = raw is not None and raw["native_failure"] is None and raw["unresolved_request"] is None and bool(exposure)
    audit.equal(result["exposure_exact"], exact, f"{label}.exact exposure")
    audit.equal(result["wall_cap_exceeded"], result["source_wall_s"] > 600, f"{label}.wall cap")
    for key, value in (("expected_rollout_updates", 48), ("expected_optimization_epochs", 192), ("expected_optimizer_steps", 6144)):
        audit.equal(result[key], value, f"{label}.{key}")
    updates_exact = (result.get("optimization_epochs") == 192 and
                     result.get("optimizer_steps_min") == result.get("optimizer_steps_max") == 6144)
    audit.equal(result["update_counts_exact"], updates_exact, f"{label}.optimizer labels")
    if "optimization_epochs" in result:
        audit.equal(result["rollout_updates"], result["optimization_epochs"] / 4., f"{label}.rollout labels")
    complete = (result["error"] is None and result["close_error"] is None and exact and result["parameters_finite"]
                and initial is not None and result["actual_decisions"] == 98304 and updates_exact and not result["wall_cap_exceeded"])
    audit.equal(result["complete"], complete, f"{label}.operational completion")
    return {"source_arm": arm, "seed": seed, "complete": complete, "initial_checkpoint": initial,
            "actual_decisions": result["actual_decisions"], "native_steps": result["native_steps"],
            "exposure_exact": exact, "unknown_exposure": False, "request_tapes_verified": int(raw is not None),
            "source_exposure": replayed["exposure"] if replayed else exposure,
            "source_schedule": result["source_schedule"], "parameter_sha256": result.get("parameter_sha256"),
            "failure_completed_native_steps": replayed["failure_completed_native_steps"] if replayed else None,
            "failure_uncertain_native_steps_upper_bound": replayed["failure_uncertain_native_steps_upper_bound"] if replayed else None,
            "failure_completed_prefixes": replayed["failure_completed_prefixes"] if replayed else None}


def trial_result(audit, root, path, value, protocol_hash, study, sources, replay):
    partition, controller, seed, index = value["partition"], value["controller"], value["seed"], value["case_index"]
    cases_key = {"development": "development_cases", "narrow_development": "narrow_development_cases", "reserved": "reserved_cases"}[partition]
    case, case_id = study[cases_key][index], study["case_ids"][partition][index]
    trial_id = f"{partition}--{case_id}--{controller}--{seed if seed is not None else 'fixed'}"
    stage = "challenge" if partition == "reserved" else "development"
    audit.equal(path.relative_to(root).as_posix(), f"{stage}/trials/{trial_id}/result.json", f"{trial_id}.location")
    for key, expected in (("schema", "generalization_study_trial_v1"), ("trial_id", trial_id), ("case_id", case_id),
                          ("protocol_sha256", protocol_hash), ("held_out", partition == "reserved"), ("weight_updates", 0)):
        audit.equal(value[key], expected, f"{trial_id}.{key}")
    fixed = controller in WITNESSES
    audit.check(controller in POLICY_ARMS or fixed, f"{trial_id}: undeclared controller")
    audit.check(seed is None if fixed else seed in SEEDS, f"{trial_id}: undeclared seed")
    action_seed = None if fixed else 100000 + seed * 10 + {"development": 0, "narrow_development": 100, "reserved": 200}[partition] + index
    audit.equal(value["action_seed"], action_seed, f"{trial_id}.action seed")
    folder, raw, replayed, actions = path.parent, None, None, []
    if "request_tape" in value:
        raw, _, actions, replayed = tape(audit, folder, value, [case], 1, False, replay, trial_id)
    else:
        audit.check(not value["complete"] and value["error"] is not None and value["metrics"]["decisions"] == 0,
                    f"{trial_id}: absent tape conceals experience")
    if "telemetry" in value:
        telemetry = load(audit.file(folder / "traces", value["telemetry"]), True)
        audit.equal(telemetry["schema"], "generalization_development_telemetry_v1", f"{trial_id}.telemetry schema")
        rows = telemetry["rows"]
    else:
        rows = []
        audit.check(raw is None and not value["complete"] and value["metrics"]["decisions"] == 0,
                    f"{trial_id}: absent telemetry conceals experience")
    expected_traces = {value[key]["path"] for key in ("request_tape", "telemetry") if key in value}
    audit.equal({p.name for p in (folder / "traces").glob("*.gz")}, expected_traces, f"{trial_id}.exact trace inventory")
    audit.check(len(rows) <= 640 and len(actions) <= 641, f"{trial_id}: evaluation budget exceeded")
    previous, stopped = 0., False
    for ordinal, row in enumerate(rows, 1):
        audit.check(not stopped, f"{trial_id}: control continued after native stop")
        audit.equal(row["decision"], ordinal, f"{trial_id}.telemetry ordinal")
        audit.check(ordinal <= len(actions), f"{trial_id}: telemetry without saved action")
        if ordinal <= len(actions):
            audit.equal(row["action"], actions[ordinal - 1][1][0], f"{trial_id}.retained action")
        if fixed:
            audit.equal(row["action"], fixed_action(controller, ordinal), f"{trial_id}.fixed policy")
        old.native_check(audit, row["native"], f"{trial_id}.decision{ordinal}", previous)
        previous, stopped = row["native"]["time_s"], row["native"]["terminated"]
    if replayed is not None:
        audit.equal(rows, replayed["rows"], f"{trial_id}.telemetry versus full native response")
        rows = replayed["rows"]
        if value["initial_local_observation"] is not None:
            audit.check(len(replayed["configure_observations"]) == 1, f"{trial_id}: missing initial native observation")
            if replayed["configure_observations"]:
                audit.equal(value["initial_local_observation"], replayed["configure_observations"][0], f"{trial_id}.initial observation")
    computed = old.metrics(rows)
    audit.equal(value["metrics"], computed, f"{trial_id}.recomputed metrics")
    audit.equal(value["windows"], [{"requested_decisions": n, "complete": len(rows) >= n, "metrics": old.metrics(rows[:n])}
                                 for n in WINDOWS], f"{trial_id}.recomputed windows")
    constructions = sum(len(r["cases"]) for r in raw["requests"] if r["op"] == "configure") if raw else 0
    if raw and raw["unresolved_request"] and raw["unresolved_request"]["op"] == "configure":
        constructions += len(raw["unresolved_request"]["cases"])
    audit.check(constructions <= 1, f"{trial_id}: repeated construction during evaluation")
    audit.equal(value["case_construction_requests"], constructions, f"{trial_id}.construction requests")
    audit.equal(value["reserved_case_construction_requests"], constructions if partition == "reserved" else 0,
                f"{trial_id}.reserved construction requests")
    expected_parameter = None
    if fixed:
        for key in ("checkpoint", "checkpoint_parameter_sha256", "initial_parameter_sha256", "final_parameter_sha256"):
            audit.equal(value[key], None, f"{trial_id}.{key}")
        unchanged = True
    elif value["checkpoint"] is not None:
        source_path, source = sources[("pooled" if controller == "untrained" else controller, seed)]
        checkpoint = source["initial_checkpoint"] if controller == "untrained" else source
        expected_path = (source_path.parent / checkpoint["policy_path"]).relative_to(root).as_posix()
        expected_parameter = checkpoint["parameter_sha256"]
        audit.equal(value["checkpoint"], {"path": expected_path, "sha256": checkpoint["policy_sha256"]}, f"{trial_id}.sealed checkpoint")
        audit.file(root, value["checkpoint"])
        audit.equal(value["checkpoint_parameter_sha256"], expected_parameter, f"{trial_id}.checkpoint parameter label")
        unchanged = value["initial_parameter_sha256"] == value["final_parameter_sha256"] == expected_parameter
    else:
        audit.check(not value["complete"] and value["error"] is not None and not rows,
                    f"{trial_id}: policy acted without sealed checkpoint")
        # Initialization may fail before or after model creation. These labels
        # bind the reported state without pretending to deserialize tensors.
        unchanged = value["initial_parameter_sha256"] is None and value["final_parameter_sha256"] is None
    audit.equal(value["frozen_weights_unchanged"], unchanged, f"{trial_id}.unchanged parameter labels")
    if controller.startswith("rhythm_"):
        phase = 10 if controller == "rhythm_2s" else 20
        audit.equal(value["fixed_schedule"], {"period_decisions": phase * 2, "sign_phase_decisions": phase,
                    "starts": "negative", "formula": "action=0 if floor(zero_based_decision/sign_phase_decisions)%2==0 else2"}, f"{trial_id}.rhythm")
    exact = raw is not None and raw["native_failure"] is None and raw["unresolved_request"] is None
    safe = exact and computed["safety_stops"] == 0 and value["error"] is None and value["close_error"] is None
    audit.equal(value["wall_cap_exceeded"], value["wall_s"] > 180, f"{trial_id}.wall cap")
    complete = safe and unchanged and not value["wall_cap_exceeded"] and len(rows) == 640 and computed["native_steps"] == 3200
    for key, expected in (("exposure_exact", exact), ("safe", safe), ("complete", complete)):
        audit.equal(value[key], expected, f"{trial_id}.{key}")
    return {"trial_id": trial_id, "partition": partition, "controller": controller, "seed": seed,
            "case_id": case_id, "case_index": index, "complete": complete, "safe": safe,
            "frozen_weights_unchanged": unchanged, "metrics": computed,
            "case_construction_requests": constructions,
            "reserved_case_construction_requests": constructions if partition == "reserved" else 0,
            "request_tapes_verified": int(raw is not None), "exposure_exact": exact,
            "failure_completed_native_steps": replayed["failure_completed_native_steps"] if replayed else None,
            "failure_uncertain_native_steps_upper_bound": replayed["failure_uncertain_native_steps_upper_bound"] if replayed else None}


def eligible(value):
    return value is not None and value["complete"] and value["safe"] and value["frozen_weights_unchanged"]


def lookup(trials):
    result = {}
    for value in trials:
        key = (value["partition"], value["controller"], value["seed"], value["case_index"])
        if key in result:
            raise ValueError("duplicate physical evaluation condition")
        result[key] = value
    return result


def difference(trained, other):
    return trained["metrics"]["native_reward"] - other["metrics"]["native_reward"] if trained is not None and other is not None else None


def source_gate(study, trials):
    index, comparisons = lookup(trials), []
    for partition, arm in (("development", "pooled"), ("narrow_development", "narrow")):
        for seed in SEEDS:
            for i, case_id in enumerate(study["case_ids"][partition]):
                trained, quiet = index.get((partition, arm, seed, i)), index.get((partition, "quiet", None, i))
                valid, gain = eligible(trained) and eligible(quiet), difference(trained, quiet)
                comparisons.append({"partition": partition, "case_id": case_id, "seed": seed, "controller": arm,
                                    "complete_safe_pair": valid, "trained_minus_quiet": gain, "passes": valid and gain > 3.2})
    return {"schema": "generalization_study_source_gate_v1", "passes": len(trials) == 148 and all(c["passes"] for c in comparisons),
            "required_pairs": 48, "margin_strictly_greater_than": 3.2, "comparisons": comparisons,
            "untrained_controls_gate_source_competence": False, "reserved_case_construction_requests": 0}


def learning(study, trials, partition, arms):
    index, rows = lookup(trials), []
    for arm in arms:
        for seed in SEEDS:
            for i, case_id in enumerate(study["case_ids"][partition]):
                trained, initial = index.get((partition, arm, seed, i)), index.get((partition, "untrained", seed, i))
                rows.append({"controller": arm, "seed": seed, "case_id": case_id,
                             "trained_minus_untrained": difference(trained, initial),
                             "complete_safe_pair": eligible(trained) and eligible(initial)})
    def reduce(group):
        realized = [r["trained_minus_untrained"] for r in group if r["trained_minus_untrained"] is not None]
        valid = [r["trained_minus_untrained"] for r in group if r["complete_safe_pair"]]
        return {"realized_pairs": len(realized), "planned_pairs": len(group),
                "mean_realized_trained_minus_untrained": statistics.mean(realized) if len(realized) == len(group) else None,
                "complete_safe_pairs": len(valid), "mean_complete_safe_trained_minus_untrained": statistics.mean(valid) if valid else None,
                "complete_safe_trained_wins": sum(x > 0. for x in valid)}
    return {"partition": partition, "comparisons": rows, **reduce(rows),
            "by_controller": {arm: reduce([r for r in rows if r["controller"] == arm]) for arm in arms},
            "per_seed_controller": [{"seed": seed, "controller": arm,
                                     **reduce([r for r in rows if r["controller"] == arm and r["seed"] == seed])}
                                    for seed in SEEDS for arm in arms],
            "interpretation": "All-case realized means retain unequal stopped prefixes and cannot establish competence; eligible-only means are labeled separately"}


def challenge_analysis(study, trials):
    index, comparisons, seed_means, useful, strata, witness_cases = lookup(trials), [], [], [], [], [], []
    for seed in SEEDS:
        seed_rows = []
        for i, case_id in enumerate(study["case_ids"]["reserved"]):
            pooled, narrow, quiet = (index.get(("reserved", arm, seed if arm != "quiet" else None, i))
                                     for arm in ("pooled", "narrow", "quiet"))
            case = study["reserved_cases"][i]
            valid = eligible(pooled) and eligible(narrow) and eligible(quiet)
            row = {"seed": seed, "case_id": case_id, "case_index": i,
                   "powered_hinge": case["assembly"]["powered_hinge"], "lamp_side": -1 if case["sun"]["position_m"][0] < 0 else 1,
                   "pooled_minus_narrow": difference(pooled, narrow), "complete_safe_pair_and_quiet": valid}
            comparisons.append(row)
            seed_rows.append(row)
            gain, valid_quiet = difference(pooled, quiet), eligible(pooled) and eligible(quiet)
            useful.append({"seed": seed, "case_id": case_id, "pooled_minus_quiet": gain,
                           "complete_safe_pair": valid_quiet, "passes": valid_quiet and gain > 3.2})
        values = [r["pooled_minus_narrow"] for r in seed_rows if r["pooled_minus_narrow"] is not None]
        seed_means.append({"seed": seed, "realized_pairs": len(values),
                           "mean_pooled_minus_narrow": statistics.mean(values) if len(values) == 12 else None,
                           "all_pairs_and_quiet_complete_safe": all(r["complete_safe_pair_and_quiet"] for r in seed_rows)})
        for hinge in (0, 1):
            for side in (-1, 1):
                values = [r["pooled_minus_narrow"] for r in seed_rows if r["powered_hinge"] == hinge and r["lamp_side"] == side]
                strata.append({"seed": seed, "powered_hinge": hinge, "lamp_side": side,
                               "mean_realized_pooled_minus_narrow": statistics.mean(values) if len(values) == 3 and all(x is not None for x in values) else None})
    means = [r["mean_pooled_minus_narrow"] for r in seed_means if r["mean_pooled_minus_narrow"] is not None]
    median, positives = statistics.median(means) if len(means) == 3 else None, sum(x > 0. for x in means)
    breadth_safe = all(r["complete_safe_pair_and_quiet"] for r in comparisons)
    for i, case_id in enumerate(study["case_ids"]["reserved"]):
        quiet, rows = index.get(("reserved", "quiet", None, i)), []
        for arm in WITNESSES[1:]:
            fixed = index.get(("reserved", arm, None, i))
            rows.append({"controller": arm, "fixed_minus_quiet": difference(fixed, quiet),
                         "complete_safe_pair": eligible(fixed) and eligible(quiet)})
        valid = [r for r in rows if r["complete_safe_pair"]]
        best = max(valid, key=lambda r: r["fixed_minus_quiet"]) if valid else None
        witness_cases.append({"case_id": case_id, "comparisons": rows, "best_safe_fixed": best,
                              "attainable_margin": best is not None and best["fixed_minus_quiet"] > 3.2})
    return {"schema": "generalization_study_challenge_analysis_v1", "comparisons": comparisons, "seed_means": seed_means,
            "lamp_motor_strata": strata,
            "breadth": {"passes": len(trials) == 168 and breadth_safe and median is not None and median > 0. and positives >= 2,
                        "all_pooled_narrow_and_quiet_complete_safe": breadth_safe,
                        "median_seed_mean": median, "positive_seed_means": positives, "untrained_controls_gate_breadth": False},
            "useful_first_use": {"passes": len(trials) == 168 and all(r["passes"] for r in useful), "comparisons": useful,
                                 "margin_strictly_greater_than": 3.2},
            "target_headroom": {"all_cases_attainable_margin": all(r["attainable_margin"] for r in witness_cases),
                                "cases": witness_cases, "selection": "Exploratory safe fixed witness; no optimality claim, tuning or case selection"},
            "offline_learning": learning(study, trials, "reserved", ("pooled", "narrow")),
            "raw_prefix_interpretation": "All-case realized contrasts preserve stopped prefixes; unequal exposure cannot establish competence"}


def expected_trials(study, stage):
    conditions = []
    if stage == "development":
        for partition, arms in (("development", POLICY_ARMS), ("narrow_development", ("narrow", "untrained"))):
            count = len(study["case_ids"][partition])
            conditions.extend((partition, "quiet", None, i) for i in range(count))
            conditions.extend((partition, arm, seed, i) for seed in SEEDS for i in range(count) for arm in arms)
    else:
        conditions.extend(("reserved", arm, None, i) for i in range(12) for arm in WITNESSES)
        conditions.extend(("reserved", arm, seed, i) for seed in SEEDS for i in range(12) for arm in POLICY_ARMS)
    return conditions


def stage_record(audit, root, name, protocol, protocol_hash):
    folder = safe_path(root, name)
    if not folder.exists():
        return None
    if not (folder / "receipt.json").is_file():
        raise ValueError(f"{name}: unfinished exclusive stage; exposure cannot be certified")
    receipt = load(folder / "receipt.json")
    audit.equal(receipt["schema"], "generalization_study_stage_receipt_v1", f"{name}.receipt schema")
    audit.equal(receipt["protocol_sha256"], protocol_hash, f"{name}.receipt protocol")
    path, record = bound(audit, root, receipt["stage"], name + ".stage")
    audit.equal(path, folder / "stage.json", f"{name}.stage path")
    for key, value in (("schema", "generalization_study_stage_v1"), ("stage", name), ("protocol_sha256", protocol_hash),
                       ("provenance_unchanged", True), ("source_sha256_after_collection", protocol["source_sha256"]),
                       ("runtime_after_collection", protocol["runtime"])):
        audit.equal(record[key], value, f"{name}.{key}")
    started = load(folder / "started.json")
    for key, value in (("schema", "generalization_study_stage_start_v1"), ("stage", name),
                       ("protocol_sha256", protocol_hash), ("reserved_case_construction_requests", 0)):
        audit.equal(started[key], value, f"{name}.started.{key}")
    audit.check(record["status"] in ("complete", "incomplete", "blocked"), f"{name}: unexpected status")
    return record


def blocked(audit, root, name, record, prerequisite):
    audit.check(not prerequisite, f"{name}: blocked despite satisfied release prerequisite")
    audit.check(bool(record.get("reason")), f"{name}: blocked stage has no reason")
    for key, value in (("actual_decisions", 0), ("native_steps", 0), ("trial_results", []),
                       ("reserved_case_construction_requests", 0), ("reserved_worlds_constructed", 0)):
        audit.equal(record[key], value, f"{name}.blocked.{key}")
    files = [p for p in safe_path(root, name).rglob("*") if p.is_file() and p.name not in ("stage.json", "receipt.json", "started.json")]
    audit.equal(files, [], f"{name}: blocked stage contains orphan experiment evidence")


def evaluate_stage(audit, root, name, record, protocol_hash, study, sources, replay):
    refs, values, audited, expected = record["trial_results"], [], [], expected_trials(study, name)
    audit.check(len(refs) <= len(expected), f"{name}: excess evaluation trials")
    paths = set()
    for ordinal, ref in enumerate(refs):
        path, value = bound(audit, root, ref, f"{name}.result{ordinal}")
        paths.add(path.resolve())
        condition = (value["partition"], value["controller"], value["seed"], value["case_index"])
        audit.equal(condition, expected[ordinal], f"{name}.prospective condition order[{ordinal}]")
        values.append(value)
        audited.append(trial_result(audit, root, path, value, protocol_hash, study, sources, replay))
    found = {p.resolve() for p in safe_path(root, name).glob("trials/*/result.json")}
    audit.equal(found, paths, f"{name}.complete trial result references")
    # A started attempt without a final result has unknown exposure. Do not
    # silently certify it as a zero-length stopped prefix.
    trial_dirs = {p.parent.resolve() for p in safe_path(root, name).glob("trials/*/started.json")}
    audit.equal(trial_dirs, {p.parent for p in paths}, f"{name}.no unfinished trial prefixes")
    audit.equal(record["collected_trials"], len(values), f"{name}.collected count")
    audit.equal(record["planned_trials"], len(expected), f"{name}.planned count")
    audit.equal(record["status"], "complete" if len(values) == len(expected) else "incomplete", f"{name}.operational stage status")
    if len(values) < len(expected):
        audit.check(values and values[-1]["cancelled"], f"{name}: missing declared attempts without owner interruption")
    audit.equal(record["actual_decisions"], sum(v["metrics"]["decisions"] for v in audited), f"{name}.decisions")
    audit.equal(record["native_steps"], sum(v["metrics"]["native_steps"] for v in audited), f"{name}.native controls")
    audit.equal(record["reserved_case_construction_requests"], sum(v["reserved_case_construction_requests"] for v in audited), f"{name}.reserved requests")
    return audited


def verify(root, replay, audit, expected_protocol_sha256=None):
    protocol_path, lock_path = safe_path(root, "protocol.json"), safe_path(root, "lock.json")
    protocol, lock, protocol_hash = load(protocol_path), load(lock_path), digest(protocol_path)
    if expected_protocol_sha256 is not None:
        audit.equal(protocol_hash, expected_protocol_sha256, "external prospective protocol commitment")
    for key, value in (("schema", "generalization_study_lock_protocol_v1"), ("status", "locked_before_source_learning")):
        audit.equal(protocol[key], value, f"protocol.{key}")
    for key, value in (("schema", "generalization_study_lock_v1"), ("protocol_sha256", protocol_hash),
                       ("source_sha256", protocol["source_sha256"]), ("runtime", protocol["runtime"]),
                       ("physical_interactions", 0), ("reserved_case_construction_requests", 0)):
        audit.equal(lock[key], value, f"lock.{key}")
    study = protocol["study"]
    declaration(audit, study)
    audit.equal(protocol["rhythm_semantics"], {"rhythm_2s": {"period_decisions": 20, "sign_phase_decisions": 10},
                "rhythm_4s": {"period_decisions": 40, "sign_phase_decisions": 20}, "starts": "negative",
                "initialization": "Native invalid-sensor governor emits zero; time is counted"}, "rhythm semantics")
    audit.equal(study["wall_caps"], {"source": 600, "trial": 180}, "wall caps")
    source_hashes = protocol["source_sha256"]
    for name, sha in source_hashes.items():
        if name.startswith(("artifacts/", "build/")) or "rover-v3" in name:
            raise ValueError("source archive must not reference experiment bundles or frozen challenges")
        audit.check(bool(old.HEX.fullmatch(sha)), f"{name}: source SHA format")
        audit.equal(digest(safe_path(root / "source_snapshot", name)), sha, f"{name}.archived source")
        audit.equal(digest(safe_path(PROJECT, name)), sha, f"{name}.live source")
    archived = {p.relative_to(root / "source_snapshot").as_posix() for p in (root / "source_snapshot").rglob("*") if p.is_file()}
    audit.equal(archived, set(source_hashes), "complete archived source manifest")
    if replay:
        if sys.platform != "linux" or os.name == "nt" or PROJECT != Path("/workspace") or not Path("/.dockerenv").is_file():
            raise ValueError("native replay requires the documented /workspace Linux development container")
        audit.equal(digest(old.NATIVE), protocol["runtime"]["native_binary_sha256"], "prospective native executable")
        if audit.failed:
            raise ValueError("native replay refused after failed prospective integrity checks")
    source_stage = stage_record(audit, root, "sources", protocol, protocol_hash)
    if source_stage is None:
        raise ValueError("source stage has not been sealed")
    source_refs, audited_sources, sources = source_stage["source_results"], [], {}
    declared_sources = [(arm, seed) for seed in SEEDS for arm in SOURCE_ARMS]
    audit.check(len(source_refs) <= 6, "excess source attempts")
    source_paths, raw_sources = set(), []
    for ordinal, ref in enumerate(source_refs):
        path, result = bound(audit, root, ref, f"source reference {ordinal}")
        source_paths.add(path.resolve())
        condition = (result["source_arm"], result["seed"])
        audit.equal(condition, declared_sources[ordinal], f"source prospective order {ordinal}")
        audit.check(condition not in sources, "duplicate source condition")
        sources[condition] = (path, result)
        raw_sources.append(result)
        audited_sources.append(source_result(audit, root, path, result, protocol_hash, study, replay))
    found_source_folders = {p.resolve() for p in safe_path(root, "sources").glob("source-*") if p.is_dir()}
    audit.equal(found_source_folders, {p.parent for p in source_paths}, "complete source attempt directories")
    if len(source_refs) < 6:
        audit.check(raw_sources and raw_sources[-1]["cancelled"], "missing remaining independent source attempts without owner interruption")
    identity = []
    for seed in SEEDS:
        pair = [v for v in audited_sources if v["seed"] == seed]
        matches = (len(pair) == 2 and all(v["initial_checkpoint"] is not None for v in pair)
                   and pair[0]["initial_checkpoint"]["parameter_sha256"] == pair[1]["initial_checkpoint"]["parameter_sha256"])
        identity.append({"seed": seed, "matches": matches})
    operational = len(audited_sources) == 6 and all(v["complete"] for v in audited_sources) and all(v["matches"] for v in identity)
    seal_path, seal = bound(audit, root, source_stage["seal"], "source seal")
    audit.equal(seal_path, root / "sources" / "sealed-checkpoints.json", "source seal path")
    audit.equal(seal, {"schema": "generalization_study_source_seal_v1", "protocol_sha256": protocol_hash,
                "operational_complete": operational, "source_results": source_refs,
                "initial_identity_by_seed": identity, "sealed_before_any_evaluation": True,
                "reserved_case_construction_requests": 0}, "source seal independent reconstruction")
    for key, value in (("operational_complete", operational), ("initial_identity_by_seed", identity),
                       ("actual_decisions", sum(v["actual_decisions"] for v in audited_sources)),
                       ("native_steps", sum(v["native_steps"] for v in audited_sources)),
                       ("exposure_exact", all(v["exposure_exact"] for v in audited_sources)),
                       ("declared_source_attempts", 6), ("collected_source_attempts", len(audited_sources)),
                       ("reserved_case_construction_requests", 0), ("status", "complete" if operational else "incomplete")):
        audit.equal(source_stage[key], value, f"sources.{key}")
    development_stage = stage_record(audit, root, "development", protocol, protocol_hash)
    development_trials, known_gate = [], source_gate(study, [])
    if development_stage is not None:
        if development_stage["status"] == "blocked":
            blocked(audit, root, "development", development_stage, operational)
        else:
            if not operational:
                raise ValueError("development requests exist despite operationally blocked source seal")
            audit.equal(development_stage["source_seal_sha256"], digest(seal_path), "development sealed release")
            development_trials = evaluate_stage(audit, root, "development", development_stage, protocol_hash, study, sources, replay)
            known_gate = source_gate(study, development_trials)
            gate_path, recorded_gate = bound(audit, root, development_stage["gate"], "known source gate")
            audit.equal(gate_path, root / "development" / "gate.json", "known gate path")
            audit.equal(recorded_gate, {"protocol_sha256": protocol_hash, **known_gate}, "independent known source gate")
            audit.equal(development_stage["source_competence"], known_gate["passes"], "development source competence")
            audit.equal(development_stage["learning_diagnostics"],
                        [learning(study, development_trials, "development", ("pooled", "narrow")),
                         learning(study, development_trials, "narrow_development", ("narrow",))],
                        "independent known learning diagnostics")
    challenge_prerequisite = (operational and development_stage is not None and
                              development_stage["status"] == "complete" and known_gate["passes"])
    challenge_stage = stage_record(audit, root, "challenge", protocol, protocol_hash)
    challenge_trials, analysis = [], None
    if challenge_stage is not None:
        if challenge_stage["status"] == "blocked":
            blocked(audit, root, "challenge", challenge_stage, challenge_prerequisite)
        else:
            # This is an explicit guard before any held-back configure request is
            # sent to native replay. Static source gate labels alone do not release it.
            if not challenge_prerequisite or audit.failed:
                raise ValueError("reserved replay refused: independent source prerequisite or integrity failed")
            release_path, release = bound(audit, root, challenge_stage["release"], "challenge release")
            audit.equal(release_path, root / "challenge" / "release.json", "challenge release path")
            audit.equal(release, {"schema": "generalization_study_challenge_release_v1", "protocol_sha256": protocol_hash,
                        "source_seal_sha256": digest(seal_path), "development_stage_sha256": digest(root / "development" / "stage.json"),
                        "global_source_competence": True, "released_before_first_reserved_call": True}, "challenge release binding")
            challenge_trials = evaluate_stage(audit, root, "challenge", challenge_stage, protocol_hash, study, sources, replay)
            analysis = challenge_analysis(study, challenge_trials)
            audit.equal(challenge_stage["analysis"], analysis, "independent reserved contrasts and all gates")
    # Absence of an unstarted downstream stage is honest, not a fabricated
    # blocked receipt. A complete study accounting claim remains separate.
    all_trials = development_trials + challenge_trials
    decisions = sum(v["actual_decisions"] for v in audited_sources) + sum(v["metrics"]["decisions"] for v in all_trials)
    native_steps = sum(v["native_steps"] for v in audited_sources) + sum(v["metrics"]["native_steps"] for v in all_trials)
    reserved_requests = sum(v["reserved_case_construction_requests"] for v in challenge_trials)
    audit.check(decisions <= 792064 and native_steps <= 3960320, "total declared study opportunities exceeded")
    if not challenge_prerequisite:
        audit.equal(reserved_requests, 0, "blocked challenge has zero reserved calls")
        if safe_path(root, "challenge").exists() and challenge_stage is None:
            raise ValueError("unknown unfinished challenge exposure")
    complete_records = (len(audited_sources) == 6 and development_stage is not None and challenge_stage is not None)
    if challenge_prerequisite:
        complete_records = complete_records and len(development_trials) == 148 and len(challenge_trials) == 168
    unknown = any(v["unknown_exposure"] or not v["exposure_exact"] for v in audited_sources) or any(not v["exposure_exact"] for v in all_trials)
    return {"protocol_sha256": protocol_hash, "lock_sha256": digest(lock_path), "source_files_verified": len(source_hashes),
            "source_seal_sha256": digest(seal_path), "source_results": audited_sources,
            "source_operational_complete": operational, "initial_identity_by_seed": identity,
            "development_trials": development_trials, "development_gate": known_gate,
            "challenge_release_eligible": challenge_prerequisite, "challenge_trials": challenge_trials,
            "challenge_analysis": analysis, "record_accounting_complete": complete_records,
            "source_status": source_stage["status"],
            "development_status": development_stage["status"] if development_stage else "not_started",
            "challenge_status": challenge_stage["status"] if challenge_stage else "not_started",
            "request_tapes_verified": sum(v["request_tapes_verified"] for v in audited_sources + all_trials),
            "actual_recorded_decisions": decisions, "recorded_successful_native_steps": native_steps,
            "exposure_exact": not unknown, "counts_are_lower_bounds_if_exposure_inexact": unknown,
            "reserved_native_configuration_calls": reserved_requests, "adaptation_weight_updates": 0,
            "unresolved_requests_replayed": False,
            "reserved_context": "These twelve length-five geometries become exposed development evidence after this test; future claims require a fresh holdout. The unrelated rover-v3 challenge is untouched.",
            "scope": "Native replay hashes every retained record=False request/full response, checks exact partition membership and recomputes physical returns, counts, windows and gates. ZIP bytes and parameter/update labels are bound; neural inference, tensor deserialization, optimizer execution, true chronological ordering and arbitrary topology generalization are not independently proven."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="existing locked report directory")
    parser.add_argument("--receipt", type=Path, required=True, help="exclusive NEW verification JSON path")
    parser.add_argument("--source-archive", type=Path, help="existing retained copy of this verifier")
    parser.add_argument("--helper-archive", type=Path, help="existing retained copy of the immutable pilot audit helper")
    parser.add_argument("--expected-protocol-sha256", help="optional externally retained prospective commitment")
    parser.add_argument("--replay", action="store_true", help="direct native replay in the managed Linux container")
    args = parser.parse_args()
    root, receipt = args.output.resolve(), args.receipt.resolve()
    if not root.is_dir() or not receipt.parent.is_dir():
        raise ValueError("report directory and receipt parent must already exist")
    if args.receipt.is_symlink() or args.output.is_symlink():
        raise ValueError("receipt and report directory must not be symlinks")
    audit = Audit()
    with receipt.open("x", encoding="utf-8") as stream:
        result = {"schema": "generalization_study_independent_verification_v1", "input": str(root),
                  "created_at_utc": datetime.now(timezone.utc).isoformat(), "native_replay_requested": args.replay,
                  "verifier_source_path": str(Path(__file__).resolve()), "verifier_source_sha256": digest(Path(__file__)),
                  "verifier_source_archive_path": str(args.source_archive.resolve()) if args.source_archive else None,
                  "helper_source_path": str(HELPER_PATH), "helper_source_sha256": digest(HELPER_PATH),
                  "helper_source_archive_path": str(args.helper_archive.resolve()) if args.helper_archive else None}
        try:
            for key, path in (("verifier", args.source_archive), ("helper", args.helper_archive)):
                if path:
                    audit.equal(digest(path.resolve()), result[f"{key}_source_sha256"], f"retained {key} source archive")
            if args.expected_protocol_sha256 is not None and not old.HEX.fullmatch(args.expected_protocol_sha256):
                raise ValueError("external protocol SHA256 must be 64 lowercase hexadecimal characters")
            result["expected_protocol_sha256"] = args.expected_protocol_sha256
            result.update(verify(root, args.replay, audit, args.expected_protocol_sha256))
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
    except (OSError, ValueError) as error:
        print(f"Audit setup refused: {error}", file=sys.stderr)
        raise SystemExit(2)
