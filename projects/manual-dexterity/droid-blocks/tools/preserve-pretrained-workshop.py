#!/usr/bin/env python3
"""Preserve existing experiment bytes before evolving the interactive runtime."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/reports/pretrained-workshop-v1-20261006"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["preserve", "verify"])
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    baseline = REPORT / "preservation-baseline.json"
    if args.operation == "preserve":
        if baseline.exists():
            raise ValueError("Baseline already exists; never replace retained evidence")
        records = []
        for path in sorted((ROOT / "artifacts").rglob("*")):
            if REPORT in path.parents or (ROOT / "artifacts/pretrained") in path.parents:
                continue
            if path.is_symlink():
                raise ValueError(f"Refusing evidence symlink: {path}")
            if path.is_file():
                records.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                                "sha256": digest(path)})
        binary = ROOT / "build/native/droid-transfer-env"
        sha = digest(binary)
        if sha != "377b754985b2977c947e37e41cbbff286595fe70b08888c9fc32ece1db2d9777":
            raise ValueError("The historical native replay binary changed before preservation")
        target = REPORT / "runtime_snapshot" / sha / "droid-transfer-env"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError("Runtime snapshot already exists")
        shutil.copy2(binary, target)
        if digest(target) != sha:
            raise ValueError("Runtime snapshot copy mismatch")
        payload = {"schema": "pretrained_workshop_preservation_v1", "existing_files": records,
                   "historical_native_binary": {"path": str(target.relative_to(ROOT)),
                       "sha256": sha, "bytes": target.stat().st_size},
                   "scope": "Existing artifact bytes remain immutable; live source evolution is separately versioned."}
        baseline.write_text(json.dumps(payload, indent=2) + "\n")
    payload = json.loads(baseline.read_text())
    for item in payload["existing_files"] + [payload["historical_native_binary"]]:
        path = ROOT / item["path"]
        if not path.is_file() or path.is_symlink() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(f"Protected evidence changed: {item['path']}")
    result = {"schema": "pretrained_workshop_preservation_check_v1", "passed": True,
              "protected_files": len(payload["existing_files"]), "baseline_sha256": digest(baseline),
              "historical_native_binary_sha256": payload["historical_native_binary"]["sha256"]}
    if args.receipt:
        destination = args.receipt.resolve()
        if not destination.is_relative_to(REPORT) or destination.exists():
            raise ValueError("Receipt must be a new file inside the workshop report")
        with destination.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
