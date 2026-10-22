#!/bin/sh
# Project build/run operations are deliberately separate from setup.sh.
set -eu
cd /workspace
cmake -S . -B build/native -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/ale -DBUILD_TESTING=ON
cmake --build build/native --parallel 2
if [ "$#" -eq 0 ]; then
    exec build/native/atari serve --root /workspace --rom /opt/ale/roms/pong.bin --host 0.0.0.0 --port 8080
fi
exec build/native/atari "$@"
