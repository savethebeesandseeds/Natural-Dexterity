"""Prospective bounded specialization declaration; locking constructs no worlds."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from generalization_curriculum import manifest as curriculum, validate_case

PROJECT = Path(__file__).resolve().parents[1]
OLD = PROJECT / "artifacts/reports/generalization-study-v1-20261006"
PRESERVATION = PROJECT / "artifacts/reports/specialization-preservation-v1-20261007/baseline.json"
SOURCE_FILES = (
    "training/specialization_manifest.py", "training/specialization_study.py",
    "training/generalization_curriculum.py", "training/generalization_pretraining.py",
    "training/native_transfer.py", "training/requirements-generalization.txt",
    "training/setup-generalization.sh", "tests/check_specialization_manifest.py",
    "tests/check_specialization_driver.py", "tools/preserve-specialization-evidence.py",
    "tools/check-specialization-preflight.py",
    "docs/SPECIALIZATION_EXPERIMENT.md", "CMakeLists.txt", "config/module_catalog.json",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def bounded(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT / path
    resolved = path.resolve()
    if not resolved.is_relative_to(PROJECT.resolve()) or resolved == PROJECT.resolve():
        raise ValueError("Evidence path must remain inside the project")
    relative = path.absolute().relative_to(PROJECT.absolute())
    current = PROJECT
    for part in relative.parts:
        if part in ("..", "."):
            raise ValueError("Evidence path must not traverse directories")
        current = current / part
        if current.is_symlink():
            raise ValueError("Evidence path must not traverse symlinks")
    return resolved


def ref(path):
    path = bounded(path)
    if not path.is_file():
        raise ValueError(f"Missing evidence file: {path}")
    return {"path": path.relative_to(PROJECT.resolve()).as_posix(), "sha256": digest(path)}


def read_bound(reference):
    if type(reference) is not dict or set(reference) != {"path", "sha256"}:
        raise ValueError("Expected an exact path/hash evidence reference")
    if type(reference["path"]) is not str or Path(reference["path"]).is_absolute():
        raise ValueError("Expected project-relative evidence path")
    path = bounded(reference["path"])
    if not path.is_file() or digest(path) != reference["sha256"]:
        raise ValueError(f"Evidence bytes differ: {reference['path']}")
    return path


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def manifest():
    old = curriculum()
    ppo = copy.deepcopy(old["ppo"])
    ppo["transfer"] = "Weights-only pooled actor+critic or saved pre-source initialization; fresh target optimizer; two updates during128 decisions then frozen512."
    cases = [{"id": name.replace("reserved-", "exposed-", 1), "case": copy.deepcopy(case)}
             for name, case in zip(old["case_ids"]["reserved"], old["reserved_cases"])]
    guards = [{"id": old["case_ids"]["development"][i],
               "case": copy.deepcopy(old["development_cases"][i])} for i in (0, 11)]
    return {
        "schema": "specialization_manifest_v1",
        "scope": "Bounded specialization on twelve already exposed length-5 conditions; three layouts, six mechanics. No fresh generalization or hardware claim.",
        "source_seeds": [101, 202, 303],
        "arms": ["frozen", "head_only", "full", "scratch"],
        "target_cases": cases, "guard_cases": guards,
        "budget": {"prefix_decisions": 128, "total_decisions": 640,
                   "continuation_decisions": 512, "horizon_decisions": 641,
                   "rollouts": 2, "adam_steps": 8, "update_decisions": [64, 128]},
        "ppo": ppo,
        "policy_input": copy.deepcopy(old["policy_input"]),
        "decision_dt_s": .1, "native_dt_s": .02,
        "action_rng": {"target_base": 810000, "guard_base": 910000,
                       "seed_multiplier": 1000,
                       "rule": "base + source_seed*1000 + ordered case index. Seed once after fresh construction and weights-only copy; no reseed at128. Stochastic Torch categorical; use_sde=False; audit train leaves Torch RNG unchanged."},
        "wall_caps_s": {"target": 180, "guard": 180},
        "trainable": {"head_only": {"tensors": ["action_net.weight", "action_net.bias", "value_net.weight", "value_net.bias"], "parameters": 260},
                      "full": {"tensors": 12, "parameters": 34820},
                      "scratch": {"tensors": 12, "parameters": 34820},
                      "frozen": {"parameters": 0}},
        "initialization": "Every target uses a fresh optimizer, rollout buffer, counters, native world and 11 absent history frames. Head/full/frozen copy exact pooled actor+critic; scratch copies saved pre-source initial actor+critic. Never restore source Adam state.",
        "continuity": "One world and sensor history for all640. Learn128 then continue512 from model._last_obs; no reset, set_env, rescue, second learn or action RNG reseed. Weights and optimizer fixed after128.",
        "gates": {"primary_arm": "head_only", "secondary_arm": "full",
                  "comparison": "integrated native return over the entire640-decision opportunity, including128 learning decisions",
                  "requires_all_36_corresponding_pairs_safe_complete": True,
                  "median_seed_mean_gain_strictly_greater_than": 3.2,
                  "minimum_positive_seed_means": 2,
                  "scratch_contrast_median_strictly_greater_than": 0.,
                  "regression_diagnostic_whole_loss_floor": -3.2,
                  "regression_diagnostic_prefix_loss_floor": -.64,
                  "sampled_retention": {"each_seed_mean_gain_at_least": -3.2,
                                        "requires_all_corresponding_probes_safe_complete": True}},
        "guard_rule": "After all144 target attempts and checkpoint refs are sealed: six frozen original baselines then144 frozen probes of72 head/full after128 checkpoints on two old development conditions, selected as first/last ordinal before outcomes. Fresh world/history and matched per-seed guard RNG. Missing128 checkpoint writes two blocked zero-exposure receipts; no retry. Sampled retention only, not whole-domain forgetting.",
        "failure_rule": "Record raw stopped prefixes and incomplete/faulted attempts; never rescue, retry, impute a completed return, remove cases, or select a winning arm/seed/checkpoint. Invalid source/lock globally blocks new worlds. Each benefit contrast and sampled retention eligibility are reported separately.",
        "maximum_collection": {"target_opportunities": 144, "target_decisions": 92160,
                               "learning_decisions_included": 13824, "guard_opportunities": 150,
                               "guard_decisions": 96000, "total_opportunities": 294,
                               "total_decisions": 188160, "native_controls": 940800,
                               "physics_steps": 9408000, "aggregate_simulated_seconds": 18816},
        "promotion": "None. Adapted checkpoints are local experiment evidence; immutable pretrained base and live workshop stay unchanged. Original rover-v3 challenge remains unexposed.",
    }


def validate_manifest(value):
    expected = manifest()
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("Unexpected specialization declaration fields")
    for partition in ("target_cases", "guard_cases"):
        if type(value[partition]) is not list or len(value[partition]) != len(expected[partition]):
            raise ValueError("Incorrect specialization condition count")
        for entry in value[partition]:
            if type(entry) is not dict or set(entry) != {"id", "case"}:
                raise ValueError("Unexpected case entry")
            validate_case(entry["case"])
    if canonical(value) != canonical(expected):
        raise ValueError("Specialization cases, arms, budgets or gates changed")


def source_hashes():
    files = {PROJECT / name for name in SOURCE_FILES}
    files.update((PROJECT / "src").glob("*.cpp"))
    files.update((PROJECT / "include/droid").glob("*.hpp"))
    return {path.relative_to(PROJECT).as_posix(): digest(path) for path in sorted(files)}


def input_refs():
    # The historical study and its source release are immutable inputs, not a
    # request to run its old live-source guards against the newer workshop.
    expected = {"protocol.json": "4cf6fe175dfad1150c9264e2ab074f8df7fe575a978724bc7d3ffddb16915727",
                "sources/sealed-checkpoints.json": "6404315c901c19972c7b190f797026431fc88ecd9006c6d0ebaa740feed18508",
                "verification/native-replay-v1.json": "92839768519ec6c3df1a7dc56f9ca25adcf563375886f26e3057fe679a19bf20",
                "verification/tensors-v1.json": "35cbaf63da3a84ade402efc486573fb2b0dde520865b04407ad02faab14e5544"}
    receipts = {}
    for name, sha in expected.items():
        reference = ref(OLD / name)
        if reference["sha256"] != sha:
            raise ValueError(f"Historical input differs: {name}")
        receipts[name] = reference
    releases = []
    seal = json.loads((OLD / "sources/sealed-checkpoints.json").read_text())
    if not seal["operational_complete"] or not seal["sealed_before_any_evaluation"]:
        raise ValueError("Source release was not operationally sealed")
    sealed = {entry["path"]: entry["sha256"] for entry in seal["source_results"]}
    for seed in (101, 202, 303):
        folder = OLD / f"sources/source-pooled-{seed}"
        result_ref = ref(folder / "result.json")
        if result_ref["sha256"] != sealed[f"sources/source-pooled-{seed}/result.json"]:
            raise ValueError("Source result differs from historical seal")
        record = json.loads((folder / "result.json").read_text())
        if not record["complete"] or record["seed"] != seed or record["source_arm"] != "pooled":
            raise ValueError("Invalid pooled release")
        final, initial = ref(folder / "policy.zip"), ref(folder / "policy_initial.zip")
        if final["sha256"] != record["policy_sha256"] or initial["sha256"] != record["initial_checkpoint"]["policy_sha256"]:
            raise ValueError("Release checkpoint byte mismatch")
        releases.append({"seed": seed, "result": result_ref, "final": final,
                         "initial": initial, "final_parameter_sha256": record["parameter_sha256"],
                         "initial_parameter_sha256": record["initial_checkpoint"]["parameter_sha256"]})
    receipts["preservation_baseline"] = ref(PRESERVATION)
    preflight = PRESERVATION.parent / "preflight.json"
    preflight_record = json.loads(preflight.read_text())
    if not preflight_record["passed"] or preflight_record["native_worlds_constructed"] != 0:
        raise ValueError("Synthetic contract preflight did not pass")
    for reference in preflight_record["test_sources"]:
        read_bound(reference)
    receipts["synthetic_preflight"] = ref(preflight)
    return {"releases": releases, "receipts": receipts}


def dependency_hashes():
    import stable_baselines3
    root = Path(stable_baselines3.__file__).resolve().parent
    names = ("ppo/ppo.py", "common/on_policy_algorithm.py", "common/base_class.py",
             "common/policies.py", "common/distributions.py", "common/buffers.py", "common/utils.py")
    return {name: digest(root / name) for name in names}


def lock(output):
    from generalization_pretraining import runtime_record
    output = bounded(output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite a previous study attempt")
    study = manifest()
    validate_manifest(study)
    if digest("/workspace/build/native/droid-transfer-env") != "377b754985b2977c947e37e41cbbff286595fe70b08888c9fc32ece1db2d9777":
        raise ValueError("Expected the retained, independently audited native simulator")
    protocol = {"schema": "specialization_protocol_v1", "status": "locked_before_collection",
                "locked_at_utc": datetime.now(timezone.utc).isoformat(), "study": study,
                "inputs": input_refs(), "source_sha256": source_hashes(),
                "runtime": runtime_record(), "dependency_sha256": dependency_hashes(),
                "native_binary": {"path": "/workspace/build/native/droid-transfer-env",
                                  "sha256": digest("/workspace/build/native/droid-transfer-env")}}
    output.mkdir(parents=True, exist_ok=False)
    for name, sha in protocol["source_sha256"].items():
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((PROJECT / name).read_bytes())
        if digest(target) != sha:
            raise ValueError("Source changed during snapshot")
    write_new(output / "protocol.json", protocol)
    write_new(output / "lock.json", {"schema": "specialization_lock_v1",
                                    "protocol_sha256": digest(output / "protocol.json"),
                                    "native_worlds_constructed": 0, "decisions": 0})
    print(canonical({"locked": str(output), "protocol_sha256": digest(output / "protocol.json")}))


def read_lock(output):
    from generalization_pretraining import runtime_record
    output = bounded(output)
    protocol = json.loads((output / "protocol.json").read_text())
    receipt = json.loads((output / "lock.json").read_text())
    sha = digest(output / "protocol.json")
    if protocol.get("schema") != "specialization_protocol_v1" or receipt.get("schema") != "specialization_lock_v1" or receipt["protocol_sha256"] != sha:
        raise ValueError("Invalid specialization protocol lock")
    validate_manifest(protocol["study"])
    if source_hashes() != protocol["source_sha256"]:
        raise ValueError("Action-generating sources changed since lock")
    for name, expected in protocol["source_sha256"].items():
        if digest(output / "source_snapshot" / name) != expected:
            raise ValueError("Archived action-generating source changed")
    if canonical(input_refs()) != canonical(protocol["inputs"]):
        raise ValueError("Historical input bytes changed")
    if canonical(runtime_record()) != canonical(protocol["runtime"]) or dependency_hashes() != protocol["dependency_sha256"]:
        raise ValueError("Pinned runtime changed")
    if digest(protocol["native_binary"]["path"]) != protocol["native_binary"]["sha256"]:
        raise ValueError("Native simulator changed")
    return protocol, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("lock", "verify"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "lock":
        lock(args.output)
    else:
        _, sha = read_lock(args.output)
        print(canonical({"valid": True, "protocol_sha256": sha, "native_worlds_constructed": 0}))


if __name__ == "__main__":
    main()
