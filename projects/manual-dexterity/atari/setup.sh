#!/bin/sh
# Install native dependencies only. Project builds and container lifecycle live
# in the project launcher and run.sh, never in this file.
set -eu

if [ "$#" -ne 0 ]; then
    printf '%s\n' 'setup.sh accepts no arguments; use atari.ps1 for project operations.' >&2
    exit 2
fi
if [ "$(id -u)" -ne 0 ]; then
    printf '%s\n' 'Run setup.sh as root inside the Debian container.' >&2
    exit 1
fi
if [ "$(dpkg --print-architecture)" != amd64 ]; then
    printf '%s\n' 'The pinned LibTorch distribution requires Linux amd64.' >&2
    exit 1
fi

bootstrap_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -r "$bootstrap_dir/dependencies.lock" ]; then
    printf '%s\n' 'dependencies.lock must be next to setup.sh.' >&2
    exit 1
fi
. "$bootstrap_dir/dependencies.lock"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    libgomp1 \
    libpng-dev \
    libssl-dev \
    ninja-build \
    unzip \
    zlib1g-dev
rm -rf /var/lib/apt/lists/*

dependency_tmp=$(mktemp -d /tmp/atari-dependencies.XXXXXX)
trap 'rm -rf "$dependency_tmp"' EXIT HUP INT TERM
download_cache=${ATARI_DOWNLOAD_CACHE:-/opt/atari/downloads}

# The image build can bind a local cache read-only to avoid downloading the
# large LibTorch archive again or retaining its ZIP as a Docker image layer.
fetch_verified() {
    dep_name=$1
    dep_url=$2
    dep_sha256=$3
    if [ -f "$download_cache/$dep_name" ]; then
        dep_path="$download_cache/$dep_name"
    else
        dep_path="$dependency_tmp/$dep_name"
        curl --fail --location --retry 3 --output "$dep_path" "$dep_url"
    fi
    if ! printf '%s  %s\n' "$dep_sha256" "$dep_path" | sha256sum --check --status; then
        printf 'Dependency checksum mismatch: %s\n' "$dep_name" >&2
        exit 1
    fi
    printf '%s\n' "$dep_path"
}

ale_archive=$(fetch_verified "$ALE_ARCHIVE" "$ALE_URL" "$ALE_SHA256")
libtorch_archive=$(fetch_verified "$LIBTORCH_ARCHIVE" "$LIBTORCH_URL" "$LIBTORCH_SHA256")
json_header=$(fetch_verified "$JSON_ARCHIVE" "$JSON_URL" "$JSON_SHA256")
httplib_header=$(fetch_verified "$HTTPLIB_ARCHIVE" "$HTTPLIB_URL" "$HTTPLIB_SHA256")
roms_archive=$(fetch_verified "$ROMS_ARCHIVE" "$ROMS_URL" "$ROMS_SHA256")

mkdir -p "$dependency_tmp/ale-source" "$dependency_tmp/roms" \
    /opt/nlohmann-json/include/nlohmann /opt/cpp-httplib/include /opt/ale/roms
tar -xzf "$ale_archive" -C "$dependency_tmp/ale-source" --strip-components=1
cmake -S "$dependency_tmp/ale-source" -B "$dependency_tmp/ale-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/ale \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_CPP_LIB=ON \
    -DBUILD_PYTHON_LIB=OFF \
    -DBUILD_VECTOR_LIB=OFF \
    -DBUILD_VECTOR_XLA_LIB=OFF \
    -DSDL_SUPPORT=OFF
cmake --build "$dependency_tmp/ale-build" --parallel 4
cmake --install "$dependency_tmp/ale-build"
mkdir -p /opt/ale/share/licenses/ale
install -m 0644 "$dependency_tmp/ale-source/LICENSE.md" /opt/ale/share/licenses/ale/LICENSE.md

unzip -q "$libtorch_archive" -d /opt
test -r /opt/libtorch/include/torch/csrc/api/include/torch/torch.h
test -r /opt/libtorch/lib/libtorch_cuda.so
install -m 0644 "$json_header" /opt/nlohmann-json/include/nlohmann/json.hpp
install -m 0644 "$httplib_header" /opt/cpp-httplib/include/httplib.h

# Reuse the exact ROM archive and checksum pinned by ALE's own release script.
# Decoding an archive needs no interpreter or Python package installation.
base64 --decode "$roms_archive" > "$dependency_tmp/roms.tar.gz"
tar -xzf "$dependency_tmp/roms.tar.gz" -C "$dependency_tmp/roms"
for rom_path in "$dependency_tmp"/roms/ROM/*/*.bin; do
    install -m 0644 "$rom_path" /opt/ale/roms/
done
printf '%s  %s\n' "$PONG_ROM_SHA256" /opt/ale/roms/pong.bin | sha256sum --check

printf '%s\n' /opt/libtorch/lib /opt/ale/lib > /etc/ld.so.conf.d/atari-native.conf
ldconfig
install -m 0644 "$bootstrap_dir/dependencies.lock" /opt/ale/share/atari-dependencies.lock
dpkg-query -W -f='${binary:Package}\t${Version}\n' > /opt/ale/share/atari-debian-packages.txt
