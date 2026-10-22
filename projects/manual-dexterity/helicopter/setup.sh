#!/bin/sh
# Dependencies and environment only. Container lifecycle belongs to helicopter.ps1.
set -eu
[ "$#" -eq 0 ] || { echo 'setup.sh accepts no commands or arguments' >&2; exit 2; }
export DEBIAN_FRONTEND=noninteractive

# Freeze the Debian dependency universe, including transitive packages.
cat > /etc/apt/sources.list.d/debian.sources <<'EOF'
Types: deb
URIs: http://snapshot.debian.org/archive/debian/20260901T000000Z
Suites: bookworm bookworm-updates
Components: main
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
Check-Valid-Until: no

Types: deb
URIs: http://snapshot.debian.org/archive/debian-security/20260901T000000Z
Suites: bookworm-security
Components: main
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
Check-Valid-Until: no
EOF
apt-get update
apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl gzip ninja-build tar unzip
rm -rf /var/lib/apt/lists/*

download() {
    curl --fail --location --retry 5 --retry-all-errors --silent --show-error \
        "$1" --output "$2"
    printf '%s  %s\n' "$3" "$2" | sha256sum --check --strict -
}
mkdir -p /opt/mujoco /opt/nlohmann-json/include/nlohmann \
    /opt/cpp-httplib/include /opt/helicopter/web/vendor /opt/helicopter/environment
download https://github.com/google-deepmind/mujoco/releases/download/3.12.0/mujoco-3.12.0-linux-x86_64.tar.gz \
    /tmp/helicopter-mujoco.tar.gz a9367911e6d5eaeade17c2197304687421c1fc932cdf7bcd4cb8cfaf0374dcb2
tar -xzf /tmp/helicopter-mujoco.tar.gz -C /opt/mujoco --strip-components=1
rm -f /tmp/helicopter-mujoco.tar.gz
download https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp \
    /opt/nlohmann-json/include/nlohmann/json.hpp aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63
download https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.51.0/httplib.h \
    /opt/cpp-httplib/include/httplib.h dfbaccb76432ed6d56ddd9983fd9d262b61ba6ba0958f6b00db35c802607bd35
download https://raw.githubusercontent.com/mrdoob/three.js/r170/build/three.module.js \
    /opt/helicopter/web/vendor/three.module.js ce1fa418de16a19495a9f72495580e3015d7745c296d3ce0485897f902ddedfb
download https://raw.githubusercontent.com/mrdoob/three.js/r170/LICENSE \
    /opt/helicopter/web/vendor/THREE-LICENSE.txt 4c40a1ef62450b857c3b2aaf294936304cd552d965fbcd9d32d4c5bcf4ba4454

# Official CasADi distribution also contains its native C++ SDK and IPOPT.
# Extract only C++ headers, licenses, CMake metadata and the required native
# libraries. No MATLAB or Python interface/runtime is installed or executed.
mkdir -p /opt/casadi
download https://github.com/casadi/casadi/releases/download/3.7.2/casadi-3.7.2-linux64-matlab2018b.zip \
    /tmp/helicopter-casadi.zip 931b7719f028f4545cf05801f1af49a18f30269e15911f0f1b1642b853b89d25
unzip -q -o /tmp/helicopter-casadi.zip \
    'include/*' 'cmake/casadi*' \
    'libcasadi.so*' 'libcasadi_nlpsol_ipopt.so*' 'libcasadi_importer_shell.so*' \
    'libipopt.so*' 'libcoinmumps.so*' 'libcoinmetis.so*' \
    'libcasadi-tp-openblas.so*' 'libgfortran*.so*' 'libquadmath*.so*' \
    -d /opt/casadi
rm -f /tmp/helicopter-casadi.zip
dpkg-query -W > /opt/helicopter/environment/debian-packages.tsv
