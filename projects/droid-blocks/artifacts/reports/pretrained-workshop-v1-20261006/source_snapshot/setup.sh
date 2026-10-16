#!/bin/sh
set -eu

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    gzip \
    libegl1 \
    libgl1 \
    libglfw3 \
    ninja-build \
    tar
rm -rf /var/lib/apt/lists/*

download_and_verify() {
    dependency_url="$1"
    dependency_file="$2"
    dependency_sha256="$3"

    curl \
        --fail \
        --location \
        --retry 5 \
        --retry-all-errors \
        --show-error \
        --silent \
        --output "$dependency_file" \
        "$dependency_url"
    printf '%s  %s\n' "$dependency_sha256" "$dependency_file" | sha256sum --check --strict -
}

mkdir -p \
    /opt/mujoco \
    /opt/nlohmann-json/include/nlohmann \
    /opt/cpp-httplib/include

mujoco_archive=/tmp/mujoco-3.12.0-linux-x86_64.tar.gz
download_and_verify \
    https://github.com/google-deepmind/mujoco/releases/download/3.12.0/mujoco-3.12.0-linux-x86_64.tar.gz \
    "$mujoco_archive" \
    a9367911e6d5eaeade17c2197304687421c1fc932cdf7bcd4cb8cfaf0374dcb2
tar --extract --gzip --file "$mujoco_archive" --directory /opt/mujoco --strip-components=1
rm -f "$mujoco_archive"

download_and_verify \
    https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp \
    /opt/nlohmann-json/include/nlohmann/json.hpp \
    aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63

download_and_verify \
    https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.51.0/httplib.h \
    /opt/cpp-httplib/include/httplib.h \
    dfbaccb76432ed6d56ddd9983fd9d262b61ba6ba0958f6b00db35c802607bd35
