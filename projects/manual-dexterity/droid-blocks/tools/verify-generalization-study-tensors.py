"""Independent, read-only tensor/optimizer audit of trusted study checkpoints.

Run only in the managed Linux container. No runner, native adapter, policy model
or world is constructed: SB3's ZIP reader loads state dictionaries on CPU with
load_data=False; scalar counters are read from the archive's plain JSON. This
deliberately skips cloudpickle metadata. The auditor is outside the prospective
action-generating lock and archives its own exact bytes in its exclusive receipt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import zipfile

SCHEMA = "generalization_study_tensor_audit_v1"
SEEDS = (101, 202, 303)
ARMS = ("pooled", "narrow")
DECISIONS, EPOCHS, ADAM_STEPS = 98304, 192, 6144
EXPECTED_SHAPES = {
    "mlp_extractor.policy_net.0.weight": (64, 204),
    "mlp_extractor.policy_net.0.bias": (64,),
    "mlp_extractor.policy_net.2.weight": (64, 64),
    "mlp_extractor.policy_net.2.bias": (64,),
    "mlp_extractor.value_net.0.weight": (64, 204),
    "mlp_extractor.value_net.0.bias": (64,),
    "mlp_extractor.value_net.2.weight": (64, 64),
    "mlp_extractor.value_net.2.bias": (64,),
    "action_net.weight": (3, 64), "action_net.bias": (3,),
    "value_net.weight": (1, 64), "value_net.bias": (1,),
}
FORBIDDEN_IMPORTS = {
    "native_transfer", "generalization_pretraining", "generalization_study",
    "generalization_curriculum", "generalization_study_manifest", "actuation_transfer",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_path(root, relative):
    value = Path(relative)
    if value.is_absolute():
        raise ValueError("Evidence references must be relative to the study")
    result = (root / value).resolve()
    if not result.is_relative_to(root):
        raise ValueError("Evidence reference escapes its study directory")
    return result


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_path(root, ref):
    require(isinstance(ref, dict), "Malformed retained evidence reference")
    path = safe_path(root, ref["path"])
    require(sha256_file(path) == ref["sha256"], f"Changed retained evidence: {ref['path']}")
    return path


def bound_json(root, ref):
    return read_json(bound_path(root, ref))


def scalar_integer(value, name):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"Checkpoint {name} is not a nonnegative integer")
    return value


def tensor_digest(state, torch):
    """Independently reproduce the declared sorted tensor-byte commitment."""
    require(isinstance(state, dict) and set(state) == set(EXPECTED_SHAPES),
            "Policy state_dict does not have the declared twelve actor/critic tensors")
    digest, rows, count, finite = hashlib.sha256(), [], 0, True
    for name, value in sorted(state.items()):
        require(isinstance(value, torch.Tensor), f"Non-tensor policy state: {name}")
        require(tuple(value.shape) == EXPECTED_SHAPES[name], f"Unexpected policy shape: {name}")
        require(value.dtype == torch.float32, f"Unexpected policy dtype: {name}")
        array = value.detach().cpu().contiguous().numpy()
        valid = bool(torch.isfinite(value).all().item())
        finite = finite and valid
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.shape).encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("utf-8") + b"\0")
        digest.update(array.tobytes())
        count += value.numel()
        rows.append({"name": name, "shape": list(value.shape), "dtype": str(array.dtype),
                     "elements": value.numel(), "finite": valid})
    require(count == 34820, "Unexpected actor/critic parameter count")
    return {"parameter_sha256": digest.hexdigest(), "parameter_count": count,
            "all_parameters_finite": finite, "tensors": rows}


def optimizer_summary(optimizer, policy, torch):
    require(isinstance(optimizer, dict) and set(optimizer) == {"state", "param_groups"},
            "Malformed saved optimizer state")
    groups, state = optimizer["param_groups"], optimizer["state"]
    require(isinstance(groups, list) and len(groups) == 1 and isinstance(state, dict),
            "Expected the declared single Adam parameter group")
    ids = groups[0]["params"]
    require(isinstance(ids, list) and len(ids) == len(set(ids)) == len(policy),
            "Optimizer does not cover each policy parameter exactly once")
    require(set(state).issubset(set(ids)), "Optimizer has an unknown parameter state")
    # State-dict registration order, rather than digest sorting, determines IDs.
    policy_values = list(policy.values())
    steps, rows, finite = [], [], True
    for index, identifier in enumerate(ids):
        if identifier not in state:
            rows.append({"parameter_id": identifier, "initialized": False})
            continue
        value = state[identifier]
        require(isinstance(value, dict) and set(value) == {"step", "exp_avg", "exp_avg_sq"},
                "Unexpected Adam state entries")
        step = value["step"]
        require(isinstance(step, torch.Tensor) and step.numel() == 1, "Adam step is not a scalar tensor")
        number = float(step.item())
        require(math.isfinite(number) and number >= 0 and number.is_integer(), "Invalid Adam step")
        steps.append(int(number))
        parameter_shape = tuple(policy_values[index].shape)
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = value[name]
            require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == parameter_shape,
                    f"Adam {name} does not match its policy parameter")
            finite = finite and bool(torch.isfinite(tensor).all().item())
        rows.append({"parameter_id": identifier, "initialized": True, "step": int(number)})
    return {"parameter_groups": len(groups), "parameter_ids": ids,
            "state_entries": len(state), "all_moments_finite": finite,
            "steps_min": min(steps, default=0), "steps_max": max(steps, default=0),
            "steps": steps, "states": rows,
            "group": {key: groups[0].get(key) for key in
                      ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize")}}


def checkpoint_audit(root, path, digest, label, kind, source_complete, result, torch, loader):
    row = {"path": path.relative_to(root).as_posix(), "sha256": digest,
           "kind": kind, "label_available": label is not None, "checks_passed": False}
    try:
        require(sha256_file(path) == digest, "Checkpoint ZIP changed before deserialization")
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), "Duplicate checkpoint ZIP entries")
            require(set(names) == {"data", "policy.pth", "policy.optimizer.pth", "pytorch_variables.pth",
                                   "_stable_baselines3_version", "system_info.txt"},
                    "Unexpected checkpoint ZIP entries")
            raw = json.loads(archive.read("data").decode("utf-8"))
            require(isinstance(raw, dict), "Checkpoint metadata is not a dictionary")
            saved_version = archive.read("_stable_baselines3_version").decode("utf-8").strip()
        require(saved_version == "2.9.0", "Checkpoint SB3 version differs from the declared runtime")
        # This intentionally does not deserialize model/policy classes or spaces.
        data, parameters, variables = loader(path, load_data=False, device="cpu", verbose=0)
        require(data is None and set(parameters) == {"policy", "policy.optimizer"},
                "Unexpected loaded checkpoint state dictionaries")
        require(variables in (None, {}), "Unexpected additional PyTorch variables")
        policy = parameters["policy"]
        tensors = tensor_digest(policy, torch)
        optimizer = optimizer_summary(parameters["policy.optimizer"], policy, torch)
        counters = {key: scalar_integer(raw.get(key), key) for key in
                    ("num_timesteps", "_n_updates", "_total_timesteps", "_num_timesteps_at_start", "_episode_num")}
        row.update({"saved_sb3_version": saved_version, "counters": counters,
                    "policy": tensors, "optimizer": optimizer,
                    "complete_source_count_requirements_applied": source_complete and kind != "initial"})
        require(raw.get("seed") == result["seed"], "Checkpoint seed differs from its sealed source")
        require(raw.get("n_envs") == 8 and raw.get("n_steps") == 256 and raw.get("batch_size") == 64
                and raw.get("n_epochs") == 4, "Checkpoint PPO rollout/update dimensions differ")
        require(tensors["all_parameters_finite"], "Policy contains nonfinite tensors")
        require(optimizer["all_moments_finite"], "Optimizer contains nonfinite moments")
        if label is not None:
            require(tensors["parameter_sha256"] == label, "Recomputed policy tensor digest differs from its retained label")
        if kind == "initial":
            require(not parameters["policy.optimizer"]["state"], "Initial optimizer already contains experience")
            require(all(value == 0 for value in counters.values()), "Initial checkpoint has nonzero training/episode counters")
            require(all(raw.get(key) is None for key in
                        ("_last_obs", "_last_episode_starts", "_last_original_obs", "ep_info_buffer", "ep_success_buffer")),
                    "Initial checkpoint already stores rollout/episode observations")
        elif source_complete:
            require(counters["num_timesteps"] == DECISIONS and counters["_total_timesteps"] == DECISIONS,
                    "Final checkpoint did not retain exactly98304 source decisions")
            require(counters["_n_updates"] == EPOCHS, "Final checkpoint did not retain192 optimization epochs")
            require(optimizer["state_entries"] == len(policy) and len(optimizer["steps"]) == len(policy)
                    and optimizer["steps_min"] == optimizer["steps_max"] == ADAM_STEPS,
                    "Final Adam state did not retain6144 steps for every parameter")
        if kind != "initial" and result.get("schema") == "generalization_study_source_result_v1":
            require(counters["num_timesteps"] == result["actual_decisions"], "Saved/source actual decision counts differ")
            require(counters["_n_updates"] == result["optimization_epochs"], "Saved/source epoch counts differ")
            require(optimizer["steps_min"] == result["optimizer_steps_min"]
                    and optimizer["steps_max"] == result["optimizer_steps_max"], "Saved/source Adam step counts differ")
            require(tensors["parameter_count"] == result["parameters"], "Saved/source parameter counts differ")
        row["checks_passed"] = True
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def checkpoint_references(root, result_ref, result):
    """Only deserialize ZIPs committed by the seal or a retained failure prefix."""
    attempt = safe_path(root, result_ref["path"]).parent
    candidates = {}
    if result.get("schema") == "generalization_study_source_result_v1":
        for kind, value in (("initial", result.get("initial_checkpoint")), ("final", result)):
            if value and "policy_path" in value:
                path = safe_path(root, (attempt.relative_to(root) / value["policy_path"]).as_posix())
                candidates[path] = (value["policy_sha256"], value.get("parameter_sha256"), kind)
    else:
        require(result.get("schema") == "generalization_study_source_failure_v1", "Unexpected sealed source schema")
        prefix = result.get("retained_prefix", [])
        bindings = {safe_path(root, ref["path"]): ref for ref in prefix}
        initial = None
        initial_receipt = attempt / "initial-checkpoint.json"
        if initial_receipt in bindings:
            initial = bound_json(root, bindings[initial_receipt])
        for path, ref in bindings.items():
            if path.parent == attempt and path.name in ("policy_initial.zip", "policy.zip", "policy_partial.zip"):
                label = initial.get("parameter_sha256") if path.name == "policy_initial.zip" and initial else None
                if path.name == "policy_initial.zip" and initial:
                    require(initial["policy_path"] == path.name and initial["policy_sha256"] == ref["sha256"],
                            "Failure prefix initial-checkpoint binding differs")
                candidates[path] = (ref["sha256"], label, "initial" if path.name == "policy_initial.zip" else "final")
    unexpected = [path.name for path in attempt.glob("policy*.zip") if path not in candidates]
    require(not unexpected, f"Unbound available checkpoint ZIPs: {unexpected}")
    if result["complete"]:
        require(len(candidates) == 2 and {value[2] for value in candidates.values()} == {"initial", "final"},
                "Complete source is missing one of its two committed checkpoints")
    return candidates


def audit(root, torch, loader):
    protocol = read_json(root / "protocol.json")
    protocol_hash = sha256_file(root / "protocol.json")
    lock = read_json(root / "lock.json")
    require(protocol.get("schema") == "generalization_study_lock_protocol_v1"
            and lock.get("schema") == "generalization_study_lock_v1", "Unexpected study lock schema")
    require(lock["protocol_sha256"] == protocol_hash, "Study protocol changed after lock")
    study = protocol["study"]
    require(study["source_seeds"] == list(SEEDS) and study["source_arms"] == list(ARMS),
            "Unexpected declared source matrix")
    require(study["source_budget"] == {"decisions": DECISIONS, "n_envs": 8, "n_steps": 256, "horizon": 640},
            "Unexpected source budget")
    require(study["ppo"]["n_epochs"] == 4 and study["ppo"]["batch_size"] == 64, "Unexpected declared update counts")
    stage_receipt = read_json(root / "sources" / "receipt.json")
    require(stage_receipt["protocol_sha256"] == protocol_hash, "Source stage belongs to another protocol")
    stage = bound_json(root, stage_receipt["stage"])
    require(stage.get("schema") == "generalization_study_stage_v1" and stage["stage"] == "sources"
            and stage["protocol_sha256"] == protocol_hash and stage["provenance_unchanged"],
            "Source stage binding/provenance failed")
    require(stage["status"] in ("complete", "incomplete"), "Source stage has no completed collection receipt")
    seal = bound_json(root, stage["seal"])
    require(seal.get("schema") == "generalization_study_source_seal_v1"
            and seal["protocol_sha256"] == protocol_hash and seal["sealed_before_any_evaluation"],
            "Unexpected source seal")
    require(seal["source_results"] == stage["source_results"], "Seal/stage source references differ")
    require(seal["operational_complete"] == stage["operational_complete"] == (stage["status"] == "complete"),
            "Operational completeness differs between stage and seal")
    require(metadata.version("torch") == protocol["runtime"]["packages"]["torch"]
            and metadata.version("stable-baselines3") == protocol["runtime"]["packages"]["stable-baselines3"],
            "Tensor deserialization runtime differs from the locked runtime")
    results, seen = [], set()
    for result_ref in seal["source_results"]:
        row = {"reference": result_ref, "checks_passed": False}
        try:
            value = bound_json(root, result_ref)
            key = (value["source_arm"], value["seed"])
            require(key[0] in ARMS and key[1] in SEEDS and key not in seen, "Duplicate or undeclared source attempt")
            seen.add(key)
            require(value["protocol_sha256"] == protocol_hash, "Sealed result belongs to another protocol")
            row.update({"source_arm": key[0], "seed": key[1], "source_complete": value["complete"],
                        "source_schema": value["schema"], "source_error": value.get("error"),
                        "unknown_exposure": value.get("unknown_exposure", False)})
            candidates = checkpoint_references(root, result_ref, value)
            row["checkpoints"] = [checkpoint_audit(root, path, digest, label, kind, value["complete"], value, torch, loader)
                                  for path, (digest, label, kind) in sorted(candidates.items())]
            row["checks_passed"] = all(item["checks_passed"] for item in row["checkpoints"])
            row["checkpoint_count"] = len(row["checkpoints"])
            row["missing_initial_checkpoint"] = not any(item["kind"] == "initial" for item in row["checkpoints"])
            row["missing_final_checkpoint"] = not any(item["kind"] == "final" for item in row["checkpoints"])
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        results.append(row)
    identity = []
    for seed in SEEDS:
        pair = []
        for arm in ARMS:
            source = next((value for value in results if value.get("seed") == seed and value.get("source_arm") == arm), None)
            initial = next((item for item in source.get("checkpoints", []) if item["kind"] == "initial" and item["checks_passed"]), None) if source else None
            pair.append(initial["policy"]["parameter_sha256"] if initial else None)
        matches = pair[0] == pair[1] if all(pair) else None
        identity.append({"seed": seed, "pooled_initial_parameter_sha256": pair[0],
                         "narrow_initial_parameter_sha256": pair[1], "independently_matches": matches,
                         "status": "checked" if all(pair) else "unavailable_in_retained_prefix"})
    complete = seal["operational_complete"]
    count = sum(value.get("checkpoint_count", 0) for value in results)
    all_checked = all(value["checks_passed"] for value in results)
    complete_valid = (len(results) == 6 and seen == {(arm, seed) for seed in SEEDS for arm in ARMS}
                      and count == 12 and all(item.get("source_complete", False) for item in results)
                      and all(item["independently_matches"] is True for item in identity))
    available_initial_identity_valid = all(item["independently_matches"] is not False for item in identity)
    return {"protocol_sha256": protocol_hash, "source_stage": stage_receipt["stage"], "source_seal": stage["seal"],
            "operational_complete": complete, "source_attempts": results, "checked_checkpoints": count,
            "declared_complete_checkpoint_count": 12, "initial_identity_by_seed": identity,
            "all_available_checkpoint_checks_passed": all_checked,
            "all_available_initial_pairs_match": available_initial_identity_valid,
            "complete_source_tensor_contract_passed": complete_valid and all_checked if complete else None,
            "checks_passed": all_checked and available_initial_identity_valid and (complete_valid if complete else True),
            "partial_interpretation": "Incomplete sources are not required to have final checkpoints or full update counts; unavailable initial identity stays unknown. Tensor checks do not establish native exposure or competence."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Existing locked study directory")
    parser.add_argument("--receipt", required=True, help="New exclusive JSON file beneath that study's verification/")
    args = parser.parse_args()
    require(os.name != "nt" and sys.platform == "linux" and Path("/.dockerenv").is_file(),
            "Checkpoint audit runs only in the managed Linux container")
    root = Path(args.output).resolve()
    require(root.is_dir(), "Study directory does not exist")
    receipt_path = Path(args.receipt).resolve()
    require(receipt_path.is_relative_to(root / "verification") and receipt_path.suffix == ".json",
            "Receipt must be a JSON file below this study's verification directory")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the receipt before importing the tensor runtime or reading ZIPs.
    with receipt_path.open("x", encoding="utf-8") as stream:
        record = {"schema": SCHEMA, "audited_at_utc": datetime.now(timezone.utc).isoformat(),
                  "study_directory": str(root), "checks_passed": False,
                  "no_model_environment_or_native_process_constructed": True,
                  "loader": "stable_baselines3.common.save_util.load_from_zip_file(load_data=False, device='cpu'); raw JSON counters; tensor-only weights_only=True reader",
                  "scope": "Trusted generated checkpoint tensors/Adam counters only; separate native-tape auditor checks physical exposure/reward. No training, inference, policy/model construction, world setup or native execution."}
        try:
            auditor = Path(__file__).resolve()
            auditor_hash = sha256_file(auditor)
            archive = root / "verification" / "auditor_snapshot" / f"verify-generalization-study-tensors-{auditor_hash}.py"
            archive.parent.mkdir(parents=True, exist_ok=True)
            if archive.exists():
                require(sha256_file(archive) == auditor_hash, "Existing auditor snapshot bytes changed")
            else:
                with archive.open("xb") as target:
                    target.write(auditor.read_bytes())
            record["auditor"] = {"path": auditor.as_posix(), "sha256": auditor_hash,
                                 "snapshot_path": archive.relative_to(root).as_posix(),
                                 "snapshot_sha256": sha256_file(archive)}
            import torch
            from stable_baselines3.common.save_util import load_from_zip_file
            forbidden = sorted(FORBIDDEN_IMPORTS.intersection(sys.modules))
            require(not forbidden, f"Unexpected project learner/adapter imports: {forbidden}")
            torch.set_num_threads(1)
            record["runtime"] = {"python": sys.version, "python_executable": sys.executable,
                                 "platform": platform.platform(), "machine": platform.machine(),
                                 "packages": {name: metadata.version(name) for name in
                                              ("torch", "stable-baselines3", "numpy", "gymnasium", "cloudpickle")},
                                 "torch_threads": torch.get_num_threads(), "cpu_deserialization": True,
                                 "os_release": Path("/etc/os-release").read_text(encoding="utf-8")}
            require(record["runtime"]["packages"]["torch"] == "2.8.0+cpu"
                    and record["runtime"]["packages"]["stable-baselines3"] == "2.9.0",
                    "Unexpected checkpoint audit runtime")
            record.update(audit(root, torch, load_from_zip_file))
            record["project_learner_or_adapter_imports"] = sorted(FORBIDDEN_IMPORTS.intersection(sys.modules))
            require(not record["project_learner_or_adapter_imports"], "Tensor deserialization imported project action-generating code")
        except Exception as error:
            record["checks_passed"] = False
            record["error"] = f"{type(error).__name__}: {error}"
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"receipt": str(receipt_path), "checks_passed": record["checks_passed"],
                      "checked_checkpoints": record.get("checked_checkpoints", 0),
                      "error": record.get("error")}, sort_keys=True), flush=True)
    return 0 if record["checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
