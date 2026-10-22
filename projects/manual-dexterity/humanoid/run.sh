#!/bin/bash
set -euo pipefail
cd /workspace
cmake -S . -B build/native -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build/native --parallel 2
ctest --test-dir build/native --output-on-failure
exec ./build/native/humanoid-viewer --host 0.0.0.0 --port 8080 --device cuda
