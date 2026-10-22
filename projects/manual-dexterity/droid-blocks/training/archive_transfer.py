"""Freeze reproducibility inputs in the new experiment directory before learning."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if ROOT not in output.parents or not (output / "protocol.json").is_file():
        raise ValueError("Expected a new locked experiment under this project")
    source = output / "source"
    source.mkdir(exist_ok=False)
    files = []
    for directory in ["src", "include", "config", "training", "tests"]:
        files += [path for path in (ROOT / directory).rglob("*")
                  if path.is_file() and path.suffix in [".cpp", ".hpp", ".py", ".json", ".txt", ".ps1"]
                  and "__pycache__" not in path.parts]
    files += [ROOT / name for name in ["CMakeLists.txt", "setup.sh", "run.sh",
               "docs/ACTUATION_TRANSFER_EXPERIMENT.md", "docs/MODULE_CONTRACT.md",
               "build/actuation-transfer-reference/cartpole-v1/protocol.json",
               "build/actuation-transfer-reference/cartpole-v1/report.json",
               "build/actuation-transfer-reference/setup/installed-packages.json",
               "build/actuation-transfer-reference/setup/torch-install.json",
               "build/actuation-transfer-reference/setup/pypi-install.json",
               "build/actuation-transfer-qa/native-ctest.log",
               "build/actuation-transfer-qa/adapter-tests.json",
               "build/actuation-transfer-qa/bridge-tests.json",
               "build/actuation-transfer-qa/prior-artifact-hashes.json",
               "build/native/droid-transfer-env"]]
    provenance = []
    for original in sorted(set(files)):
        relative = original.relative_to(ROOT)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        provenance.append({"path": relative.as_posix(), "sha256": sha(target), "bytes": target.stat().st_size})
    with (output / "source-manifest.json").open("x", encoding="utf-8") as stream:
        json.dump({"schema": "actuation_transfer_source_v1", "files": provenance,
                   "protocol_sha256": sha(output / "protocol.json")}, stream, indent=2)
    print(json.dumps({"archived_files": len(provenance), "protocol_sha256": sha(output / "protocol.json")}))


if __name__ == "__main__":
    main()
