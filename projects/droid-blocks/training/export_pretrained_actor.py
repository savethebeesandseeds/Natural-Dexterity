#!/usr/bin/env python3
"""Export the canonical first declared pooled policy without training or worlds.

Run in the existing Linux training environment. Every output is exclusive; old
checkpoint, study, source archive and verification bytes are read-only inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
STUDY = "artifacts/reports/generalization-study-v1-20261006"
PROTOCOL_SHA = "4cf6fe175dfad1150c9264e2ab074f8df7fe575a978724bc7d3ffddb16915727"
ZIP_SHA = "4c7ce76f644dd07d41340ef311d5f797578fa58f0470e2b89aa49c0d3b93082c"
POLICY_SHA = "30cb3f0efad3b0a6755cecafac73366aef1b925e2ee30f62b1bab989a04023dc"
SHAPES = {
    "mlp_extractor.policy_net.0.weight": (64, 204), "mlp_extractor.policy_net.0.bias": (64,),
    "mlp_extractor.policy_net.2.weight": (64, 64), "mlp_extractor.policy_net.2.bias": (64,),
    "mlp_extractor.value_net.0.weight": (64, 204), "mlp_extractor.value_net.0.bias": (64,),
    "mlp_extractor.value_net.2.weight": (64, 64), "mlp_extractor.value_net.2.bias": (64,),
    "action_net.weight": (3, 64), "action_net.bias": (3,),
    "value_net.weight": (1, 64), "value_net.bias": (1,),
}
ACTOR_NAMES = tuple(sorted(name for name in SHAPES if ".policy_net." in name or name.startswith("action_net.")))


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(root, name):
    require(isinstance(name, str) and "\\" not in name, "Expected a relative POSIX evidence path")
    part = Path(name)
    require(not part.is_absolute() and ".." not in part.parts and "." not in part.parts, "Unsafe evidence path")
    path = (root / part).resolve()
    require(path.is_relative_to(root.resolve()), "Evidence path escapes root")
    return path


def ref(path):
    path = path.resolve()
    return {"path": path.relative_to(PROJECT).as_posix(), "sha256": sha(path)}


def runtime():
    require(sys.platform == "linux" and PROJECT == Path("/workspace") and Path("/.dockerenv").is_file(),
            "Use the existing /workspace Linux development container")
    require(sys.prefix == "/opt/droid-blocks/generalization-venv", "Use the established CPU training environment")
    require(metadata.version("torch") == "2.8.0+cpu" and metadata.version("stable-baselines3") == "2.9.0",
            "Unexpected checkpoint deserialization runtime")


def tensor_digest(state, names):
    result = hashlib.sha256()
    for name in sorted(names):
        value = state[name].detach().cpu().contiguous().numpy()
        result.update(name.encode("utf-8") + b"\0")
        result.update(str(value.shape).encode("utf-8") + b"\0float32\0")
        result.update(value.astype("<f4", copy=False).tobytes())
    return result.hexdigest()


def trusted_checkpoint(study, torch, loader):
    protocol_path, seal_path = study / "protocol.json", study / "sources/sealed-checkpoints.json"
    result_path = study / "sources/source-pooled-101/result.json"
    protocol, lock, seal, result = read(protocol_path), read(study / "lock.json"), read(seal_path), read(result_path)
    require(sha(protocol_path) == lock["protocol_sha256"] == PROTOCOL_SHA, "Changed prospective protocol")
    require(protocol["study"]["source_seeds"][0] == 101 and protocol["study"]["source_arms"][0] == "pooled",
            "Canonical policy must be first declared seed and pooled arm")
    stage = read(study / "sources/stage.json")
    source_receipt = read(study / "sources/receipt.json")
    require(source_receipt["stage"]["sha256"] == sha(study / "sources/stage.json"), "Changed source stage")
    require(stage["seal"]["sha256"] == sha(seal_path) and seal["operational_complete"]
            and seal["sealed_before_any_evaluation"], "Unsealed source checkpoints")
    selected_refs = [r for r in seal["source_results"] if r["path"] == "sources/source-pooled-101/result.json"]
    require(len(selected_refs) == 1 and selected_refs[0]["sha256"] == sha(result_path), "Changed canonical source result")
    require(result["complete"] and result["source_arm"] == "pooled" and result["seed"] == 101
            and result["actual_decisions"] == 98304 and result["parameter_sha256"] == POLICY_SHA,
            "Canonical policy is not the declared complete frozen release")
    checkpoint = relative(result_path.parent, result["policy_path"])
    require(sha(checkpoint) == result["policy_sha256"] == ZIP_SHA, "Changed canonical checkpoint ZIP")
    native_path, tensor_path = study / "verification/native-replay-v1.json", study / "verification/tensors-v1.json"
    native, tensors = read(native_path), read(tensor_path)
    require(native["protocol_sha256"] == tensors["protocol_sha256"] == PROTOCOL_SHA,
            "Independent audit protocol binding differs")
    require(native["integrity_passed"] and native["native_replay_verified"] and native["failed_checks"] == 0
            and tensors["checks_passed"], "Independent native/tensor audit did not pass")
    # All original experimental source bytes are read from the retained archive.
    # Current live native sources may now contain the separately authorized demo.
    for name, expected in protocol["source_sha256"].items():
        require(sha(relative(study / "source_snapshot", name)) == expected, f"Changed archived source: {name}")
    data, saved, variables = loader(checkpoint, load_data=False, device="cpu", verbose=0)
    require(data is None and set(saved) == {"policy", "policy.optimizer"} and variables in (None, {}),
            "Unexpected checkpoint dictionaries")
    state = saved["policy"]
    require(set(state) == set(SHAPES), "Unexpected policy tensors")
    for name, shape in SHAPES.items():
        value = state[name]
        require(tuple(value.shape) == shape and value.dtype == torch.float32 and bool(torch.isfinite(value).all()),
                f"Unexpected or nonfinite tensor: {name}")
    require(tensor_digest(state, SHAPES) == POLICY_SHA, "Recomputed full policy digest differs")
    return state, {"selection_rule": "first declared source seed, pooled arm; no target-score selection",
                   "source_arm": "pooled", "source_seed": 101, "source_decisions": 98304,
                   "protocol": ref(protocol_path), "source_seal": ref(seal_path), "source_result": ref(result_path),
                   "checkpoint": ref(checkpoint), "native_audit": ref(native_path), "tensor_audit": ref(tensor_path),
                   "full_policy_parameter_sha256": POLICY_SHA, "exporter_source_sha256": sha(Path(__file__))}


def make_artifact(state, provenance):
    byte_digest, tensors = hashlib.sha256(), []
    for name in ACTOR_NAMES:
        array = state[name].detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        raw = array.tobytes()
        byte_digest.update(raw)
        tensors.append({"name": name, "shape": list(array.shape), "data": array.reshape(-1).tolist(),
                        "sha256": hashlib.sha256(raw).hexdigest()})
    require(sum(len(t["data"]) for t in tensors) == 17475, "Unexpected actor parameter count")
    return {"schema": "pretrained_actor_v1", "version": 1, "policy_id": "pooled-seed101-v1",
            "dtype": "float32", "byte_order": "little", "dimensions": [204, 64, 64, 3], "activation": "tanh",
            "action_requests": [-.15, 0., .15], "decision_dt_s": .1, "native_dt_s": .02,
            "frame_count": 12, "frame_size": 17, "input_clip": [-5., 5.],
            "actor_tensor_sha256": tensor_digest(state, ACTOR_NAMES), "actor_bytes_sha256": byte_digest.hexdigest(),
            "tensors": tensors, "provenance": provenance,
            "stochastic_contract": {"law": "categorical_softmax", "native_rng": "versioned portable sampler; independent of Torch RNG",
                                    "torch_seeded_trajectory_equivalence": False}}


def write_exclusive(path, value, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=None if compact else 2,
                  separators=(",", ":") if compact else None)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=PROJECT / STUDY)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    runtime()
    artifact_path, receipt_path = args.artifact.resolve(), args.receipt.resolve()
    require(artifact_path.is_relative_to(PROJECT) and receipt_path.is_relative_to(PROJECT), "Outputs must stay in project")
    require(not artifact_path.exists() and not receipt_path.exists(), "Refuse to overwrite retained evidence")
    import torch
    from stable_baselines3.common.save_util import load_from_zip_file
    torch.set_num_threads(1)
    state, provenance = trusted_checkpoint(args.study.resolve(), torch, load_from_zip_file)
    artifact = make_artifact(state, provenance)
    write_exclusive(artifact_path, artifact, compact=True)
    receipt = {"schema": "pretrained_actor_export_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "artifact": ref(artifact_path), "actor_tensor_sha256": artifact["actor_tensor_sha256"],
               "actor_bytes_sha256": artifact["actor_bytes_sha256"], "actor_parameters": 17475,
               "provenance": provenance, "exporter_source": ref(Path(__file__)),
               "weight_updates": 0, "native_world_calls": 0, "reserved_world_calls": 0,
               "scope": "Lossless frozen actor export; forward and frame parity are separately verified before demo use."}
    write_exclusive(receipt_path, receipt)
    print(json.dumps({"artifact": receipt["artifact"], "actor_tensor_sha256": artifact["actor_tensor_sha256"],
                      "actor_bytes_sha256": artifact["actor_bytes_sha256"], "receipt": ref(receipt_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
