#!/bin/bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
cd /workspace
cmake -S . -B build/native -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/native --parallel 2
ctest --test-dir build/native -R 'thermodynamic-balances|causal-mass-estimator|minimum-entropy-hinf' --output-on-failure
# Controller synthesis is offline and source-bound. Full paired flight
# evaluation is an explicit launcher task, not a startup side effect.
cd /workspace/build/native
exec /workspace/build/native/helicopter-server
