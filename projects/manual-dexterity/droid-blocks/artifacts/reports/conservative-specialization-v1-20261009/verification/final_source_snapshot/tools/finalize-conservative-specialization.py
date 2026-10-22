#!/usr/bin/env python3
"""Bind completed evidence and documentation without overwriting a receipt."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "training"))
from generalization_pretraining import require_container
from conservative_specialization_manifest import digest, read_lock, ref, write_new


def main():
    require_container()
    root = PROJECT / "artifacts/reports/conservative-specialization-v1-20261009"
    protocol, sha = read_lock(root)
    output = root / "verification/final-provenance.json"
    if output.exists():
        raise FileExistsError("Never overwrite final provenance")
    audit_path = root / "verification/audit-v2.json"
    preservation_path = PROJECT / "artifacts/reports/conservative-specialization-preservation-v1-20261009/after-v2.json"
    audit = json.loads(audit_path.read_text())
    preservation = json.loads(preservation_path.read_text())
    if not (audit["integrity_passed"] and audit["native_replay_verified"] and audit["protocol_sha256"] == sha):
        raise ValueError("Independent replay of this study has not passed")
    if not (preservation["passed"] and preservation["baseline_sha256"] == protocol["inputs"]["receipts"]["preservation_baseline"]["sha256"]):
        raise ValueError("Earlier evidence preservation has not passed")
    report_inputs_path = root / "verification/report-inputs-v1.json"
    report_inputs = json.loads(report_inputs_path.read_text())
    expected_report_inputs = {
        "audit": ref(audit_path), "preservation": ref(preservation_path),
        "audit_interruption": ref(root / "verification/audit-interruption-v1.json"),
        "renderer": ref(PROJECT / "tools/write-conservative-specialization-report.py"),
        "report": ref(PROJECT / "docs/CONSERVATIVE_SPECIALIZATION_RESULTS.md"),
    }
    if (report_inputs["protocol_sha256"] != sha
            or any(report_inputs[name] != value for name, value in expected_report_inputs.items())):
        raise ValueError("Final report inputs changed after rendering")
    if (audit["verifier_source_sha256"] != digest(PROJECT / "tools/verify-conservative-specialization.py")
            or audit["stable_native_helper_sha256"] != digest(PROJECT / "tools/verify-generalization-pretraining.py")):
        raise ValueError("Final verifier sources differ from the audited versions")
    crosscheck_path = root / "verification/reduction-crosscheck-v1.json"
    crosscheck = json.loads(crosscheck_path.read_text())
    if (not crosscheck["passed"] or crosscheck["failed_checks"] != 0
            or crosscheck["protocol_sha256"] != sha or crosscheck["audit"] != ref(audit_path)
            or crosscheck["recorded_reduction"] != ref(root / "verification/final-reduction-v1.json")):
        raise ValueError("Independent recorded reductions do not match the final audit")
    targets = json.loads((root / "targets/stage.json").read_text())
    guards = json.loads((root / "guards/stage.json").read_text())
    if targets["collected_trials"] != 144 or guards["collected_trials"] != 150:
        raise ValueError("All declared attempts need retained results")
    if (digest(root / "targets/stage.json") != audit["target_stage_sha256"]
            or digest(root / "guards/stage.json") != audit["guard_stage_sha256"]):
        raise ValueError("Collection ledgers differ from audited records")
    documents = ("docs/CONSERVATIVE_SPECIALIZATION_RESULTS.md", "docs/CONTROL_MODEL.md",
                 "docs/OPEN_HANDOFF.md", "README.md", "training/README.md")
    tool_names = ("tools/verify-conservative-specialization.py",
                  "tools/write-conservative-specialization-report.py",
                  "tools/finalize-conservative-specialization.py",
                  "tools/verify-generalization-pretraining.py")
    snapshots = []
    for name in documents + tool_names:
        source = PROJECT / name
        target = root / "verification/final_source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        if digest(target) != digest(source):
            raise ValueError("Source changed while capturing final provenance")
        snapshots.append({"source": ref(source), "snapshot": ref(target)})
    inventory = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Evidence must not traverse symlinks")
        if path.is_file():
            inventory.append({**ref(path), "size": path.stat().st_size})
    value = {"schema": "conservative_specialization_final_provenance_v1",
             "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "protocol_sha256": sha, "integrity_passed": True,
             "audit": ref(audit_path), "preservation": ref(preservation_path),
             "audit_interruption": ref(root / "verification/audit-interruption-v1.json"),
             "recovered_container": ref(preservation_path.parent / "container-reuse-v2.json"),
             "report_inputs": ref(report_inputs_path),
             "reduction_crosscheck": ref(crosscheck_path),
             "final_source_snapshots": snapshots, "retained_files": inventory,
             "retained_file_count": len(inventory), "retained_bytes": sum(r["size"] for r in inventory),
             "collection": {"target_trials": targets["collected_trials"], "guard_trials": guards["collected_trials"],
                            "completed_safe_opportunities": audit["completed_safe_opportunities"],
                            "operationally_complete": audit["operationally_complete"],
                            "decisions": targets["actual_decisions"] + guards["actual_decisions"],
                            "native_controls": targets["native_steps"] + guards["native_steps"]},
             "scientific_readiness": audit["analysis"]["combined_readiness"],
             "base_actor": ref(PROJECT / "artifacts/pretrained/pooled-seed101-v1.json"),
             "bulk_local_only": "New traces and policy ZIPs are ignored retained evidence; exact refs bind bytes needed for replay.",
             "promotion": "None; immutable base actor remains the live workshop policy",
             "original_rover_v3_challenge_constructed": False}
    write_new(output, value)
    print(json.dumps({"passed": True, "files": len(inventory), "provenance": ref(output)}))


if __name__ == "__main__":
    main()
