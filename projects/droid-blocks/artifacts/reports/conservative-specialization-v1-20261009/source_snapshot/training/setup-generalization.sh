#!/bin/sh
# Optional training dependencies in the existing managed development container.
# This script does not create, start, stop, replace or build Docker objects.
set -eu

test -f /.dockerenv || { echo 'Run inside the documented development container.' >&2; exit 1; }
test "$(id -u)" = 0 || { echo 'APT setup requires the container root user.' >&2; exit 1; }
test -f /workspace/training/requirements-generalization.txt

training_environment=/opt/droid-blocks/generalization-venv
training_scratch=/tmp/droid-generalization-task
training_receipts=/opt/droid-blocks/generalization-setup
mkdir -p "$training_scratch" "$training_receipts"
export TMPDIR="$training_scratch"
export PYTHONDONTWRITEBYTECODE=1
export DEBIAN_FRONTEND=noninteractive

if ! command -v python3 >/dev/null 2>&1 || ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends python3 python3-venv
fi
if [ ! -e "$training_environment" ]; then
    python3 -m venv "$training_environment"
fi
test -x "$training_environment/bin/python" || {
    echo 'Existing training environment is incomplete; it has been preserved.' >&2; exit 1;
}
"$training_environment/bin/python" -I -c '
import sys
assert sys.prefix == "/opt/droid-blocks/generalization-venv", sys.prefix
assert sys.version_info >= (3, 11), sys.version
'

# Keep each installation attempt receipt. Never replace an earlier receipt.
training_attempt=$(date -u +%Y%m%dT%H%M%SZ)-$$
"$training_environment/bin/python" -I -m pip --isolated --disable-pip-version-check install \
    --timeout 60 --retries 2 --no-cache-dir --only-binary=:all: --no-deps \
    --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0+cpu' \
    --report "$training_receipts/$training_attempt-torch.json"
"$training_environment/bin/python" -I -m pip --isolated --disable-pip-version-check install \
    --timeout 60 --retries 2 --no-cache-dir --only-binary=:all: \
    --index-url https://pypi.org/simple \
    --requirement /workspace/training/requirements-generalization.txt \
    --report "$training_receipts/$training_attempt-pypi.json"
"$training_environment/bin/python" -I -m pip check
"$training_environment/bin/python" -I - "$training_receipts/$training_attempt-runtime.json" <<'PY'
import importlib.metadata as md
import json
import platform
import sys
import torch
expected = {"stable-baselines3": "2.9.0", "gymnasium": "1.2.3", "torch": "2.8.0+cpu", "numpy": "2.2.6"}
for package, version in expected.items():
    assert md.version(package) == version, (package, md.version(package), version)
assert torch.version.cuda is None
torch.set_num_threads(1)
assert (torch.ones(2) + 1).tolist() == [2.0, 2.0]
receipt = {"python": sys.version, "platform": platform.platform(), "prefix": sys.prefix,
           "cpu_only": True, "torch_threads": torch.get_num_threads(),
           "packages": sorted([{"name": d.metadata["Name"], "version": d.version}
                               for d in md.distributions()], key=lambda item: item["name"].lower())}
with open(sys.argv[1], "x", encoding="utf-8") as stream:
    json.dump(receipt, stream, indent=2)
    stream.write("\n")
print(json.dumps({"training_python": sys.executable, "receipt": sys.argv[1]}))
PY
