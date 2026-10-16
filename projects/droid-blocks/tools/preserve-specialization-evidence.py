#!/usr/bin/env python3
"""Hash existing artifact bytes before specialization; never copy or delete them.

Run only inside the managed Linux development container. Baseline creation and
every verification receipt are exclusive. New artifact files are permitted by a
check; each baseline file must still have exactly the recorded size and SHA-256.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NEW_REPORT = "artifacts/reports/specialization-v1-20261007"
PRESERVATION_REPORT = "artifacts/reports/specialization-preservation-v1-20261007"
BASELINE_SCHEMA = "specialization_evidence_preservation_baseline_v1"


def require_container():
    if sys.platform != "linux" or not Path("/.dockerenv").is_file():
        raise RuntimeError("Run this utility inside the existing managed Linux container")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def exclusive_json(path, value, compact=False):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical(value) if compact else json.dumps(value, indent=2, allow_nan=False))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def file_digest(path):
    """Refuse symlink substitution and any observed mutation during hashing."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Expected a regular evidence file: {path}")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if signature(before) != signature(opened):
            raise ValueError(f"Evidence changed before hashing: {path}")
        digest = hashlib.sha256()
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
        after = os.fstat(stream.fileno())
    if signature(before) != signature(after) or signature(before) != signature(path.lstat()):
        raise ValueError(f"Evidence changed while hashing: {path}")
    return digest.hexdigest(), before


def validate_directory_chain(root, path):
    """Do not traverse an existing symlink in an output or input directory."""
    if not path.is_relative_to(root) or ".." in path.parts:
        raise ValueError("Path must stay inside the project root")
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Refusing directory symlink: {current}")
        if current.exists() and not current.is_dir():
            raise ValueError(f"Expected a directory: {current}")


def inventory(root):
    artifacts = root / "artifacts"
    validate_directory_chain(root, artifacts)
    if not artifacts.is_dir():
        raise ValueError("The project artifacts directory must already exist")
    exclusions = {root / NEW_REPORT, root / PRESERVATION_REPORT}
    paths = []
    for directory, directories, filenames in os.walk(artifacts, followlinks=False):
        parent = Path(directory)
        retained = []
        for name in sorted(directories):
            path = parent / name
            # Never inspect a .git subtree, including one nested in artifacts.
            if name == ".git" or path in exclusions:
                continue
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError(f"Refusing evidence directory symlink or special file: {path}")
            retained.append(name)
        directories[:] = retained
        for name in sorted(filenames):
            path = parent / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ValueError(f"Refusing evidence symlink or special file: {path}")
            paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def validate_baseline(value):
    if type(value) is not dict or set(value) != {"schema", "scan_root", "excluded_roots", "files"}:
        raise ValueError("Unexpected preservation baseline fields")
    if value["schema"] != BASELINE_SCHEMA or value["scan_root"] != "artifacts":
        raise ValueError("Unexpected preservation baseline schema or scan root")
    if value["excluded_roots"] != [NEW_REPORT, PRESERVATION_REPORT, "**/.git"]:
        raise ValueError("Unexpected preservation exclusions")
    if type(value["files"]) is not dict or not value["files"]:
        raise ValueError("A preservation baseline must contain evidence files")
    for relative, record in value["files"].items():
        if type(relative) is not str:
            raise ValueError("Baseline paths must be strings")
        path = PurePosixPath(relative)
        if ("\\" in relative or path.is_absolute()
                or path.as_posix() != relative or ".." in path.parts or ".git" in path.parts
                or not path.parts or path.parts[0] != "artifacts"
                or any(path.is_relative_to(PurePosixPath(excluded))
                       for excluded in (NEW_REPORT, PRESERVATION_REPORT))):
            raise ValueError(f"Unsafe or excluded baseline path: {relative}")
        if (type(record) is not dict or set(record) != {"size", "sha256"}
                or type(record["size"]) is not int or record["size"] < 0
                or type(record["sha256"]) is not str or len(record["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in record["sha256"])):
            raise ValueError(f"Invalid size or SHA-256 declaration: {relative}")


def baseline(root):
    destination = root / PRESERVATION_REPORT
    validate_directory_chain(root, destination.parent)
    # Retain an incomplete directory if a failure occurs; do not overwrite it.
    destination.mkdir()
    paths = inventory(root)
    files = {}
    signatures = {}
    for index, path in enumerate(paths, 1):
        sha256, info = file_digest(path)
        relative = path.relative_to(root).as_posix()
        files[relative] = {"size": info.st_size, "sha256": sha256}
        signatures[relative] = signature(info)
        if index % 256 == 0:
            print(f"Hashed {index}/{len(paths)} protected files", file=sys.stderr, flush=True)
    if [path.relative_to(root).as_posix() for path in inventory(root)] != list(files):
        raise ValueError("Artifact inventory changed during baseline collection")
    for relative, expected in signatures.items():
        if signature((root / relative).lstat()) != expected:
            raise ValueError(f"Evidence changed during baseline collection: {relative}")
    value = {"schema": BASELINE_SCHEMA, "scan_root": "artifacts",
             "excluded_roots": [NEW_REPORT, PRESERVATION_REPORT, "**/.git"], "files": files}
    validate_baseline(value)
    baseline_path = destination / "baseline.json"
    exclusive_json(baseline_path, value, compact=True)
    baseline_sha256, _ = file_digest(baseline_path)
    script_sha256, _ = file_digest(Path(__file__).resolve())
    summary = {"schema": "specialization_evidence_preservation_summary_v1",
               "completed": True, "created_at_utc": utc_now(),
               "baseline_path": baseline_path.relative_to(root).as_posix(),
               "baseline_sha256": baseline_sha256, "protected_files": len(files),
               "protected_bytes": sum(record["size"] for record in files.values()),
               "utility_sha256": script_sha256,
               "scope": "All preexisting artifact files, including local ignored runtime and checkpoints; byte preservation only, not scientific validity. No source/data copies or deletions."}
    exclusive_json(destination / "summary.json", summary)
    return summary


def check(root, baseline_path, output, expected_sha256=None):
    expected_path = root / PRESERVATION_REPORT / "baseline.json"
    if baseline_path != expected_path:
        raise ValueError("Use this experiment's separate preservation baseline")
    validate_directory_chain(root, baseline_path.parent)
    baseline_sha256, _ = file_digest(baseline_path)
    if expected_sha256 is not None and baseline_sha256 != expected_sha256:
        raise ValueError("Baseline SHA-256 differs from the externally bound receipt")
    value = json.loads(baseline_path.read_text(encoding="utf-8"))
    validate_baseline(value)
    if not output.is_relative_to(root / PRESERVATION_REPORT) or output == baseline_path:
        raise ValueError("Write a new check receipt inside the preservation report")
    validate_directory_chain(root, output.parent)
    if output.exists() or output.is_symlink():
        raise ValueError("Never overwrite a retained verification receipt")
    current = {path.relative_to(root).as_posix(): path for path in inventory(root)}
    failures = []
    checked_bytes = 0
    for index, (relative, expected) in enumerate(value["files"].items(), 1):
        if relative not in current:
            failures.append({"path": relative, "issue": "missing"})
            continue
        actual_sha256, info = file_digest(current[relative])
        checked_bytes += info.st_size
        if info.st_size != expected["size"] or actual_sha256 != expected["sha256"]:
            failures.append({"path": relative, "issue": "changed", "expected": expected,
                             "actual": {"size": info.st_size, "sha256": actual_sha256}})
        if index % 256 == 0:
            print(f"Checked {index}/{len(value['files'])} protected files", file=sys.stderr, flush=True)
    extras = sorted(set(current) - set(value["files"]))
    result = {"schema": "specialization_evidence_preservation_check_v1",
              "passed": not failures, "checked_at_utc": utc_now(),
              "baseline_path": baseline_path.relative_to(root).as_posix(),
              "baseline_sha256": baseline_sha256, "protected_files": len(value["files"]),
              "protected_bytes": sum(record["size"] for record in value["files"].values()),
              "checked_present_bytes": checked_bytes, "extra_files_permitted": True,
              "extra_file_count": len(extras), "extra_files": extras, "failures": failures}
    exclusive_json(output, result)
    return result


def main():
    require_container()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["baseline", "check"])
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT,
                        help="Existing project root; test fixtures must also be inside the container")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, help="Exclusive JSON receipt for check")
    parser.add_argument("--expected-baseline-sha256")
    args = parser.parse_args()
    root = args.project_root.absolute()
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ValueError("Project root must be an existing absolute directory without symlink traversal")
    if args.operation == "baseline":
        if args.baseline or args.output or args.expected_baseline_sha256:
            parser.error("baseline uses the fixed exclusive preservation report directory")
        result = baseline(root)
    else:
        if args.output is None:
            parser.error("check requires an exclusive --output receipt")
        baseline_path = args.baseline or root / PRESERVATION_REPORT / "baseline.json"
        if not baseline_path.is_absolute():
            baseline_path = root / baseline_path
        output = args.output if args.output.is_absolute() else root / args.output
        result = check(root, baseline_path, output, args.expected_baseline_sha256)
    print(canonical(result), flush=True)
    return 0 if result.get("passed", result.get("completed", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
