#!/bin/bash
# Dependencies and environment only; lifecycle belongs to radio.ps1/run.sh.
set -euo pipefail
[ "$#" -eq 0 ] || { echo 'setup.sh accepts no arguments' >&2; exit 2; }
export DEBIAN_FRONTEND=noninteractive
export PIP_CACHE_DIR=/opt/web-radio/cache/pip
mkdir -p /opt/web-radio /workspace/.runtime/environment
cat > /etc/apt/sources.list.d/debian.sources <<'EOF'
Types: deb
URIs: http://snapshot.debian.org/archive/debian/20261001T000000Z
Suites: trixie trixie-updates
Components: main
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
Check-Valid-Until: no

Types: deb
URIs: http://snapshot.debian.org/archive/debian-security/20261001T000000Z
Suites: trixie-security
Components: main
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
Check-Valid-Until: no
EOF
if [ ! -x /opt/web-radio/venv/bin/python ]; then
  apt-get update
  apt-get install -y --no-install-recommends python3 python3-venv ca-certificates curl ffmpeg
  python3 -m venv /opt/web-radio/venv
fi
if [ ! -f /workspace/requirements.lock ]; then
  /opt/web-radio/venv/bin/python -m pip install 'pip==25.2' 'pip-tools==7.5.1'
  /opt/web-radio/venv/bin/pip-compile --generate-hashes --resolver=backtracking --output-file=/workspace/requirements.lock /workspace/requirements.in
fi
current_hash=$(sha256sum /workspace/requirements.lock | cut -d' ' -f1)
previous_hash=$(cat /opt/web-radio/requirements.sha256 2>/dev/null || true)
if [ "$current_hash" != "$previous_hash" ]; then
  /opt/web-radio/venv/bin/python -m pip install --require-hashes -r /workspace/requirements.lock
  printf '%s\n' "$current_hash" > /opt/web-radio/requirements.sha256
fi
/opt/web-radio/venv/bin/python /workspace/infra/vendor.py
dpkg-query -W > /workspace/.runtime/environment/debian-packages.tsv
/opt/web-radio/venv/bin/python -m pip freeze > /workspace/.runtime/environment/python-packages.txt

