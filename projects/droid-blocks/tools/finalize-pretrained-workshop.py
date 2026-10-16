#!/usr/bin/env python3
"""Retain the exact runtime sources and compact, independently bound receipts."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/reports/pretrained-workshop-v1-20261006"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reference(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path), "bytes": path.stat().st_size}


def main():
    destination = REPORT / "final-provenance-v1.json"
    if destination.exists():
        raise ValueError("Final provenance already exists")
    receipts = ("export-v1.json", "reference-v1.json", "native-workshop-v1.json", "native-workshop-v2.json",
                "http-service-v1.json", "http-service-v2.json", "browser-qa-v1.json", "preservation-baseline.json",
                "preservation-interruption-v1.json", "preservation-after-v1.json", "legacy-idle-before.json",
                "native-tests-v1.log", "container-after-v1.json")
    for name in ("reference-v1.json", "native-workshop-v2.json", "http-service-v2.json",
                 "browser-qa-v1.json", "preservation-after-v1.json"):
        if json.loads((REPORT / name).read_text())["passed"] is not True:
            raise ValueError(f"Required check did not pass: {name}")
    names = ["CMakeLists.txt", "Dockerfile", "run.sh", "setup.sh", ".gitignore", ".gitattributes",
             "config/module_catalog.json", "models/droid.xml", "web/pretrained.html", "web/pretrained.css",
             "web/pretrained.js", "web/construction.css", "web/index.html", "web/construction.html",
             "training/native_transfer.py", "training/export_pretrained_actor.py", "training/verify_pretrained_actor.py",
             "training/verify_pretrained_workshop.py", "tools/preserve-pretrained-workshop.py",
             "tools/check-pretrained-service.py", "tools/finalize-pretrained-workshop.py",
             "docs/PRETRAINED_WORKSHOP.md", "README.md", "training/README.md", "docs/OPEN_HANDOFF.md"]
    for pattern in ("include/droid/*.hpp", "src/*.cpp", "tests/native*_tests.cpp"):
        names.extend(str(path.relative_to(ROOT)) for path in ROOT.glob(pattern))
    source_records = []
    archive = REPORT / "source_snapshot"
    for name in sorted(set(names)):
        source = ROOT / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Invalid runtime source: {name}")
        target = archive / name
        if target.exists():
            raise ValueError("Source snapshot already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if digest(source) != digest(target):
            raise ValueError("Source snapshot mismatch")
        source_records.append({"source": reference(source), "snapshot": reference(target)})
    container = json.loads((REPORT / "container-after-v1.json").read_text())
    if container["id"] != "a3eabc3183e054aafd42304857b6a10b2c3a7cf15787fcf177149335e6aeaaaf" or container["health"] != "healthy":
        raise ValueError("The original managed container is not verified healthy")
    payload = {"schema": "pretrained_workshop_final_provenance_v1", "passed": True,
        "milestone": "Visible frozen pretrained build-wake-rebuild loop within the evaluated two-link kit",
        "actor": reference(ROOT / "artifacts/pretrained/pooled-seed101-v1.json"),
        "selection_rule": "Canonical first declared pooled source seed 101; no target-score selection",
        "weight_updates": 0,
        "sources": source_records, "receipts": [reference(REPORT / name) for name in receipts],
        "binaries": [reference(ROOT / "build/native" / name) for name in
            ("droid-blocks-server", "droid-pretrained-actor", "droid-pretrained-session", "droid-transfer-env")],
        "managed_container": container,
        "scope": "Already exposed geometries; no new generalization experiment, topology change or hardware result",
        "local_ignored_assets": {"native_binary": "runtime_snapshot/377b754985b2977c947e37e41cbbff286595fe70b08888c9fc32ece1db2d9777/droid-transfer-env",
                                 "run_traces": "traces/*.json.gz"},
        "historical_live_source_guard": "Expected to reject evolved current CMake/server; archived sources and original receipts stay immutable"}
    with destination.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": True, "sources": len(source_records), "receipts": len(receipts),
                      "final_provenance_sha256": digest(destination)}))


if __name__ == "__main__":
    main()
