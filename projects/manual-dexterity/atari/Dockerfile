FROM debian:12-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171 AS native-runtime

LABEL io.waajacu.managed="true" \
      io.waajacu.project="atari" \
      io.waajacu.config="native-v1"

ENV OMP_NUM_THREADS="1" \
    MKL_NUM_THREADS="1" \
    CMAKE_PREFIX_PATH="/opt/ale" \
    ATARI_ROM_DIR="/opt/ale/roms" \
    NVIDIA_DRIVER_CAPABILITIES="compute,utility"

COPY setup.sh dependencies.lock /opt/atari/bootstrap/
RUN --mount=type=bind,source=.runtime/dependencies,target=/opt/atari/downloads,readonly \
    chmod 0755 /opt/atari/bootstrap/setup.sh \
    && /opt/atari/bootstrap/setup.sh

# Source code and run artifacts arrive through the project bind mount.
WORKDIR /workspace
EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=180s --retries=3 \
    CMD ["curl", "--fail", "--silent", "--show-error", "--max-time", "2", "http://127.0.0.1:8080/healthz"]

CMD ["sh", "/workspace/run.sh"]

# Compile and exercise the real emulator without starting a service container.
# GPU forward/backward and training still require the managed GPU container.
FROM native-runtime AS native-validation
WORKDIR /opt/atari/validation
COPY CMakeLists.txt ./
COPY games.json ./
COPY include/ include/
COPY src/ src/
COPY tests/ tests/
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
    && cmake --build build --parallel 2 \
    && ctest --test-dir build --output-on-failure \
    && ! command -v python \
    && ! command -v python3
RUN ./build/atari benchmark --root /opt/atari/validation --envs 1 --steps 50000 \
    && ./build/atari benchmark --root /opt/atari/validation --envs 8 --steps 50000

FROM native-runtime AS development
