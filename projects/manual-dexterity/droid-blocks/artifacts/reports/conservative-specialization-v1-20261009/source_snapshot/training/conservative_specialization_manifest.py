"""Prospective longer-budget specialization; locking constructs no worlds."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json

from generalization_curriculum import validate_case
from specialization_manifest import (
    PROJECT, bounded, canonical, dependency_hashes, digest, ref, read_bound,
    write_new, manifest as previous_manifest, input_refs as previous_inputs,
)

PRIOR = PROJECT / "artifacts/reports/specialization-v1-20261007"
PRESERVATION = PROJECT / "artifacts/reports/conservative-specialization-preservation-v1-20261009/baseline.json"
SOURCE_FILES = (
    "training/conservative_specialization_manifest.py",
    "training/conservative_specialization_study.py",
    "training/specialization_manifest.py", "training/specialization_study.py",
    "training/generalization_curriculum.py",
    "training/generalization_pretraining.py", "training/native_transfer.py",
    "training/requirements-generalization.txt", "training/setup-generalization.sh",
    "tests/check_conservative_specialization_manifest.py",
    "tests/check_conservative_specialization_driver.py",
    "tools/preserve-conservative-specialization-evidence.py",
    "tools/check-conservative-specialization-preflight.py",
    "docs/CONSERVATIVE_SPECIALIZATION_EXPERIMENT.md",
    "CMakeLists.txt", "config/module_catalog.json",
)


def manifest():
    value = copy.deepcopy(previous_manifest())
    value.update({
        "schema": "conservative_specialization_manifest_v1",
        "scope": "Longer-budget specialization on twelve already exposed length-5 conditions. Conservative settings are an experimental package, not a safety guarantee or fresh generalization claim.",
        "arms": ["frozen", "conservative", "standard", "scratch"],
        "budget": {"prefix_decisions": 384, "total_decisions": 640,
                   "continuation_decisions": 256, "horizon_decisions": 641,
                   "rollouts": 6, "adam_steps": 24,
                   "update_decisions": [64, 128, 192, 256, 320, 384]},
        "per_arm_ppo": {arm: {"learning_rate": .0001 if arm == "conservative" else .0003,
                              "clip_range": .1 if arm == "conservative" else .2}
                        for arm in ("frozen", "conservative", "standard", "scratch")},
        "trainable": {arm: {"tensors": 12, "parameters": 34820}
                      for arm in ("conservative", "standard", "scratch")},
        "initialization": "Fresh optimizer, buffer, counters, world and empty history per opportunity. Frozen/conservative/standard copy exact pooled actor+critic; scratch copies saved pre-source initial tensors. No source optimizer state.",
        "continuity": "One world/history/RNG for640: learn384 then frozen256 from model._last_obs. No reset, set_env, rescue, second learn or action reseed. Weights and optimizer fixed after384.",
        "guard_rule": "After all144 target attempts and refs are sealed: six original frozen baselines then144 frozen probes of72 conservative/standard after384 checkpoints on the same two fixed old conditions. Missing384 checkpoints produce blocked zero-exposure receipts, no retry. This tests two sampled conditions only.",
        "repeatability": "Same action RNG as October7. Repeated frozen640 and standard prefix128 are integrity checks on exposed cases, not new independent evidence. Compare native/actions and initial/64/128 policy and Adam tensors against bound earlier trials.",
        "maximum_collection": {"target_opportunities": 144, "target_decisions": 92160,
                               "learning_decisions_included": 41472,
                               "guard_opportunities": 150, "guard_decisions": 96000,
                               "total_opportunities": 294, "total_decisions": 188160,
                               "native_controls": 940800, "physics_steps": 9408000,
                               "aggregate_simulated_seconds": 18816},
    })
    value["trainable"]["frozen"] = {"parameters": 0}
    value["ppo"]["transfer"] = "Weights-only copy; six64-decision updates during384 then frozen256. Per-arm learning rate and PPO clip override base before model construction. Constant schedules, no KL early stop."
    value["action_rng"]["rule"] = "base + source_seed*1000 + ordered case index, matching October7. Seed once after fresh construction and weights-only copy; no reseed at384. Torch categorical, use_sde=False; optimizer updates leave Torch RNG unchanged."
    value["gates"].update({
        "primary_arm": "conservative", "secondary_arm": "standard",
        "comparison": "integrated native return over the whole640 opportunity, including384 learning decisions",
        "regression_diagnostic_prefix_loss_floor": -1.92,
        "reliability": {"target_each_pair_gain_at_least": -3.2,
                        "probe_each_pair_gain_at_least": -3.2,
                        "requires_all_corresponding_pairs_safe_complete": True},
        "combined_readiness": "For each declared method separately: useful-benefit AND target reliability AND sampled retention AND probe reliability. Do not select the better arm afterward. No automatic promotion even if passed.",
    })
    return value


def validate_manifest(value):
    expected = manifest()
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("Unexpected conservative specialization declaration fields")
    for partition in ("target_cases", "guard_cases"):
        if type(value[partition]) is not list or len(value[partition]) != len(expected[partition]):
            raise ValueError("Incorrect condition count")
        for entry in value[partition]:
            if type(entry) is not dict or set(entry) != {"id", "case"}:
                raise ValueError("Unexpected case entry")
            validate_case(entry["case"])
    if canonical(value) != canonical(expected):
        raise ValueError("Cases, arms, settings, budgets or prospective gates changed")


def source_hashes():
    files = {PROJECT / name for name in SOURCE_FILES}
    files.update((PROJECT / "src").glob("*.cpp"))
    files.update((PROJECT / "include/droid").glob("*.hpp"))
    return {path.relative_to(PROJECT).as_posix(): digest(path) for path in sorted(files)}


def input_refs():
    value = previous_inputs()
    # Keep the old releases/audits, replace only this study's prospective baseline.
    value["receipts"]["preservation_baseline"] = ref(PRESERVATION)
    preflight = PRESERVATION.parent / "preflight.json"
    record = json.loads(preflight.read_text())
    if not record["passed"] or record["native_worlds_constructed"] != 0:
        raise ValueError("New synthetic preflight did not pass")
    for reference in record["test_sources"]:
        read_bound(reference)
    value["receipts"]["synthetic_preflight"] = ref(preflight)
    expected = {
        "protocol": ("protocol.json", "5b5dbda3123c20a030021c141e4812ec0db1c8f66fd9030eca78b8265ec002e4"),
        "audit": ("verification/audit-v1.json", "65b297f720b0b1fb8e1975d71073e707d759eef947d789e6f1f3f4b07f6d56b2"),
        "final_provenance": ("verification/final-provenance.json", "62aa877921e97edd1a57a0d29564251099502c5fb9c4ddc1f304aa57d2a29d72"),
        "targets_stage": ("targets/stage.json", "6b7c09622a1c0615609071f997922a0a4b8fce0fe843193ea99e4d105f8c1a78"),
        "targets_seal": ("targets/seal.json", "186cb490c899bfe8ec01c42b691bdbaa894d1f0f9af76822932094bd7cd70209"),
    }
    prior = {}
    for key, (name, sha) in expected.items():
        reference = ref(PRIOR / name)
        if reference["sha256"] != sha:
            raise ValueError(f"Earlier specialization input differs: {name}")
        prior[key] = reference
    audit = json.loads(read_bound(prior["audit"]).read_text())
    if not (audit["integrity_passed"] and audit["native_replay_verified"] and audit["operationally_complete"]):
        raise ValueError("Earlier specialization was not independently verified")
    targets = json.loads(read_bound(prior["targets_stage"]).read_text())
    prior["targets"] = []
    for reference in targets["trial_results"]:
        result = json.loads(read_bound(reference).read_text())
        if result["arm"] in ("frozen", "full"):
            prior["targets"].append({"seed": result["seed"], "case_id": result["case_id"],
                                     "arm": result["arm"], "result": reference})
    if len(prior["targets"]) != 72:
        raise ValueError("Expected all36 earlier frozen and full targets")
    value["prior_specialization"] = prior
    return value


def lock(output):
    from generalization_pretraining import runtime_record
    output = bounded(output)
    if output.exists():
        raise FileExistsError("Never overwrite or resume an earlier study attempt")
    study = manifest()
    validate_manifest(study)
    native = "/workspace/build/native/droid-transfer-env"
    if digest(native) != "377b754985b2977c947e37e41cbbff286595fe70b08888c9fc32ece1db2d9777":
        raise ValueError("Native simulator differs from audited runtime")
    protocol = {"schema": "conservative_specialization_protocol_v1", "status": "locked_before_collection",
                "locked_at_utc": datetime.now(timezone.utc).isoformat(), "study": study,
                "inputs": input_refs(), "source_sha256": source_hashes(), "runtime": runtime_record(),
                "dependency_sha256": dependency_hashes(),
                "native_binary": {"path": native, "sha256": digest(native)}}
    output.mkdir(parents=True, exist_ok=False)
    for name, sha in protocol["source_sha256"].items():
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((PROJECT / name).read_bytes())
        if digest(target) != sha:
            raise ValueError("Source changed while capturing lock")
    write_new(output / "protocol.json", protocol)
    write_new(output / "lock.json", {"schema": "conservative_specialization_lock_v1",
                                    "protocol_sha256": digest(output / "protocol.json"),
                                    "native_worlds_constructed": 0, "decisions": 0})
    print(canonical({"locked": str(output), "protocol_sha256": digest(output / "protocol.json")}))


def read_lock(output):
    from generalization_pretraining import runtime_record
    output = bounded(output)
    protocol = json.loads((output / "protocol.json").read_text())
    receipt = json.loads((output / "lock.json").read_text())
    sha = digest(output / "protocol.json")
    if (protocol.get("schema") != "conservative_specialization_protocol_v1"
            or receipt.get("schema") != "conservative_specialization_lock_v1"
            or receipt["protocol_sha256"] != sha):
        raise ValueError("Invalid conservative specialization lock")
    validate_manifest(protocol["study"])
    if source_hashes() != protocol["source_sha256"]:
        raise ValueError("Action-generating sources changed after lock")
    for name, expected in protocol["source_sha256"].items():
        if digest(output / "source_snapshot" / name) != expected:
            raise ValueError("Archived action-generating bytes changed")
    if canonical(input_refs()) != canonical(protocol["inputs"]):
        raise ValueError("Bound inputs changed after lock")
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
