FROM debian:12-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171 AS native-base
LABEL io.waajacu.managed="true" io.waajacu.project="humanoid"
ENV HUMANOID_ASSET_ROOT=/opt/humanoid/unitree_rl_gym \
    UNITREE_COMMIT=276801e46c5d433564f24658bac64f254b7d2d4b \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    LD_LIBRARY_PATH=/opt/libtorch/lib:/opt/mujoco/lib
COPY setup.sh /opt/humanoid/bootstrap/
COPY runtime/dependencies.sha256 runtime/unitree_rl_gym-276801e.tar.gz runtime/mujoco-3.3.2-linux-x86_64.tar.gz runtime/libtorch-cxx11-abi-shared-with-deps-2.7.1+cu126.zip /opt/humanoid/bootstrap/
RUN bash /opt/humanoid/bootstrap/setup.sh

FROM native-base AS native-validation
WORKDIR /opt/humanoid/validation
COPY CMakeLists.txt ./
COPY include/ include/
COPY src/ src/
COPY training/ training/
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
    && cmake --build build --parallel 2 \
    && ctest --test-dir build --output-on-failure \
    && touch /opt/humanoid/native-validation-passed

FROM native-validation AS native-evidence
RUN ./build/humanoid-evaluate --device cpu --trials 5 --seconds 20 --output /opt/humanoid/validation/evaluation-cpu.json

FROM scratch AS evidence-export
COPY --from=native-evidence /opt/humanoid/validation/evaluation-cpu.json /evaluation-cpu.json

FROM native-base AS development
COPY --from=native-validation /opt/humanoid/native-validation-passed /opt/humanoid/native-validation-passed
WORKDIR /workspace
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=4s --start-period=180s --retries=3 \
  CMD ["curl", "--fail", "--silent", "--max-time", "3", "http://127.0.0.1:8080/healthz"]
CMD ["bash", "/workspace/run.sh"]
