FROM debian:12-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171 AS native-base

LABEL io.waajacu.managed="true" \
      io.waajacu.project="droid-blocks"

ENV MUJOCO_GL=egl \
    LD_LIBRARY_PATH="/opt/mujoco/lib"

COPY setup.sh /opt/droid-blocks/bootstrap/setup.sh
RUN chmod 0755 /opt/droid-blocks/bootstrap/setup.sh \
    && /opt/droid-blocks/bootstrap/setup.sh \
    && ! command -v python \
    && ! command -v python3

FROM native-base AS native-validation

WORKDIR /opt/droid-blocks/build-check
COPY CMakeLists.txt ./
COPY include/ include/
COPY src/ src/
COPY config/ config/
COPY models/ models/
COPY tests/ tests/
COPY artifacts/light-search-policy-v3-seed0.json artifacts/light-search-policy-v3-seed0.json
RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=ON \
    && cmake --build build --target droid-native-validation \
    && ctest --test-dir build --output-on-failure \
    && cmake -E touch /opt/droid-blocks/native-validation-passed

FROM native-base AS development

COPY --from=native-validation \
    /opt/droid-blocks/native-validation-passed \
    /opt/droid-blocks/native-validation-passed

WORKDIR /workspace
EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD ["curl", "--fail", "--silent", "--show-error", "--max-time", "2", "http://127.0.0.1:8080/healthz"]

CMD ["/workspace/run.sh"]
