#!/usr/bin/env python3
"""Bind finished specialization evidence and current documentation, exclusively."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "training"))
from generalization_pretraining import require_container
from specialization_manifest import digest, read_lock, ref, write_new


def main():
    require_container()
    root = PROJECT / "artifacts/reports/specialization-v1-20261007"
    protocol, sha = read_lock(root)
    output = root / "verification/final-provenance.json"
    if output.exists():
        raise FileExistsError("Never overwrite final provenance")
    audit_path = root / "verification/audit-v1.json"
    preservation_path = PROJECT / "artifacts/reports/specialization-preservation-v1-20261007/after-v2.json"
    audit = json.loads(audit_path.read_text())
    preservation = json.loads(preservation_path.read_text())
    if not (audit["integrity_passed"] and audit["native_replay_verified"] and audit["operationally_complete"]
            and audit["protocol_sha256"] == sha):
        raise ValueError("Independent replay of this complete study has not passed")
    # Preservation utility reports a separate byte-level pass.
    if not (preservation["passed"] and preservation["baseline_sha256"] == protocol["inputs"]["receipts"]["preservation_baseline"]["sha256"]):
        raise ValueError("Preexisting artifact preservation has not passed")
    documents = ("docs/SPECIALIZATION_RESULTS.md", "docs/CONTROL_MODEL.md", "docs/OPEN_HANDOFF.md",
                 "README.md", "training/README.md", "artifacts/reports/README.md")
    tools = ("tools/verify-specialization-study.py", "tools/plot-specialization-results.py",
             "tools/write-specialization-report.py",
             "tools/finalize-specialization-study.py", "tools/verify-generalization-pretraining.py")
    snapshot_refs = []
    for name in documents + tools:
        source = PROJECT / name
        target = root / "verification/final_source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
        if digest(target) != digest(source):
            raise ValueError("Source changed while capturing final provenance")
        snapshot_refs.append({"source": ref(source), "snapshot": ref(target)})
    inventory = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Do not traverse evidence symlinks")
        if path.is_file():
            inventory.append({**ref(path), "size": path.stat().st_size})
    targets = json.loads((root / "targets/stage.json").read_text())
    guards = json.loads((root / "guards/stage.json").read_text())
    value = {"schema": "specialization_final_provenance_v1",
             "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "protocol_sha256": sha, "integrity_passed": True,
             "audit": ref(audit_path), "preservation": ref(preservation_path),
             "final_source_snapshots": snapshot_refs, "retained_files": inventory,
             "retained_file_count": len(inventory), "retained_bytes": sum(r["size"] for r in inventory),
             "collection": {"target_trials": targets["collected_trials"], "guard_trials": guards["collected_trials"],
                            "decisions": targets["actual_decisions"] + guards["actual_decisions"],
                            "native_controls": targets["native_steps"] + guards["native_steps"]},
             "base_actor": ref(PROJECT / "artifacts/pretrained/pooled-seed101-v1.json"),
             "bulk_local_only": "Target/guard traces and all new policy ZIPs are ignored retained evidence; refs bind their exact bytes. Fresh checkout needs these local assets for full replay.",
             "promotion": "None; original source release and live base actor preserved",
             "original_rover_v3_challenge_constructed": False}
    write_new(output, value)
    print(json.dumps({"passed": True, "files": len(inventory), "provenance": ref(output)}))


if __name__ == "__main__":
    main()
