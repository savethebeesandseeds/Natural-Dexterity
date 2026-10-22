#!/bin/bash
# Dependencies/environment only. Container lifecycle lives in container.ps1.
set -euo pipefail
if (($#)); then
  printf 'setup.sh takes no project/lifecycle commands. See container.ps1 and run.sh.\n' >&2
  exit 2
fi
export DEBIAN_FRONTEND=noninteractive
# Immutable Debian package indices; package signatures remain mandatory.
# HTTP permits bootstrapping ca-certificates from this signed archive.
cat > /etc/apt/sources.list.d/debian.sources <<'DEBIAN_SOURCES'
Types: deb
URIs: http://snapshot.debian.org/archive/debian/20260901T000000Z/
Suites: bookworm
Components: main
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
Check-Valid-Until: no

Types: deb
URIs: http://snapshot.debian.org/archive/debian-security/20260901T000000Z/
Suites: bookworm-security
Components: main
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
Check-Valid-Until: no
DEBIAN_SOURCES
apt-get update
apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build ca-certificates curl unzip \
  libegl1 libgl1 libosmesa6-dev libjpeg62-turbo-dev libyaml-cpp-dev libssl-dev
mkdir -p /opt/humanoid /opt/mujoco /opt/nlohmann-json/include/nlohmann /opt/cpp-httplib/include
dpkg-query -W -f='${Package}=${Version}\n' > /opt/humanoid/debian-packages.txt

cd /opt/humanoid/bootstrap
sha256sum --check --strict dependencies.sha256
tar -xzf mujoco-3.3.2-linux-x86_64.tar.gz --directory /opt/mujoco --strip-components=1
unzip -q -o libtorch-cxx11-abi-shared-with-deps-2.7.1+cu126.zip -d /opt
printf '%s\n' /opt/libtorch/lib /opt/mujoco/lib > /etc/ld.so.conf.d/humanoid.conf
ldconfig

download_and_verify() {
  curl --fail --location --retry 5 --show-error --silent --output "$2" "$1"
  printf '%s  %s\n' "$3" "$2" | sha256sum --check --strict -
}
download_and_verify \
  https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp \
  /opt/nlohmann-json/include/nlohmann/json.hpp \
  aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63
download_and_verify \
  https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.51.0/httplib.h \
  /opt/cpp-httplib/include/httplib.h \
  dfbaccb76432ed6d56ddd9983fd9d262b61ba6ba0958f6b00db35c802607bd35

# Third-party model/controller assets are a dependency, pinned to one revision.
unitree_archive=/opt/humanoid/bootstrap/unitree_rl_gym-276801e.tar.gz
printf '%s  %s\n' 545ead52777901578fb645be4eba9dbe51b46dedb67f12545f6fa98f4cbf8d1e "$unitree_archive" | sha256sum --check --strict -
mkdir -p /opt/humanoid/unitree_rl_gym
tar -xzf "$unitree_archive" --directory /opt/humanoid/unitree_rl_gym \
  --strip-components=1 \
  unitree_rl_gym-276801e46c5d433564f24658bac64f254b7d2d4b/LICENSE \
  unitree_rl_gym-276801e46c5d433564f24658bac64f254b7d2d4b/README.md \
  unitree_rl_gym-276801e46c5d433564f24658bac64f254b7d2d4b/deploy/deploy_mujoco/configs/g1.yaml \
  unitree_rl_gym-276801e46c5d433564f24658bac64f254b7d2d4b/deploy/pre_train/g1 \
  unitree_rl_gym-276801e46c5d433564f24658bac64f254b7d2d4b/resources/robots/g1_description
printf '%s\n' 276801e46c5d433564f24658bac64f254b7d2d4b > /opt/humanoid/unitree_rl_gym/REVISION
if command -v python || command -v python3; then
  printf 'Unexpected Python interpreter in native environment.\n' >&2
  exit 1
fi
