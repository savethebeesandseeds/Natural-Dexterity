#!/bin/sh
set -eu

project_root=/workspace
build_directory=/workspace/build/native

cmake \
    -S "$project_root" \
    -B "$build_directory" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DBUILD_TESTING=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$build_directory" --target droid-native-validation
ctest --test-dir "$build_directory" --output-on-failure

exec "$build_directory/droid-blocks-server" \
    --host 0.0.0.0 \
    --port 8080 \
    --model /workspace/models/droid.xml \
    --web-root /workspace/web
