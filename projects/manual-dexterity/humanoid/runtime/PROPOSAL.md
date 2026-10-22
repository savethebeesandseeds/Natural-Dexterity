# Native walking runtime definition

The existing brief is in `doc/`. This runtime executes the official
Unitree G1 pretrained walking policy in actual MuJoCo physics, with a live
browser viewer and measured displacement, falls, and foot contacts. This is
an inference baseline; it is not evidence that we trained a policy ourselves.
A separate C++ LibTorch PPO implementation tests learning from scratch on the
same G1 model, and a native imitation learner trains a new student from expert
action labels. All paths use native MuJoCo and require no Python interpreter.

## Approved native container definition

The user approved a separate port after the first container `humanoid-mujoco-dev`
(ID `cd11cd8447238a9db4abba3914b96b941effddbf42d92652a6d69729503c8920`)
could not start after `helicopter-dev` acquired port 43118 during the build.
Both are preserved. The additional container below is now running, with the
viewer available at [127.0.0.1:43871](http://127.0.0.1:43871/).

- Container: `humanoid-mujoco-native`
- Container ID: `0ee55590b050c97bea3b2b37288813f9edcc0d597471c73e1b947280e18a2ebc`
- Image: `humanoid-mujoco:dev`
- Inspected image ID: `sha256:603dfd694bbd1eea2002b555a737b5289165cd10f5654d76b57f10a8905d823c`
- Base: `debian:12-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171`
- Command: `["bash", "/workspace/run.sh"]`
- Bind mount: `C:\Work\Natural-Dexterity\projects\manual-dexterity\humanoid` to `/workspace`, read-write
- Named volumes: none; experiment outputs persist under the bind mount
- Port: `127.0.0.1:43871:8080`
- Restart policy: `no`
- GPU: `--gpus all`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`
- Init: enabled; shared memory: 1 GiB; stop timeout: 20 seconds
- Labels: `io.waajacu.managed=true`, `io.waajacu.project=humanoid`, plus a configuration digest
- Dependencies: native C++17 compiler, CMake, Ninja, CA certificates, curl, unzip,
  EGL/OSMesa, JPEG, yaml-cpp and OpenSSL; MuJoCo 3.3.2, LibTorch 2.7.1 CUDA 12.6,
  nlohmann-json 3.12.0 and cpp-httplib 0.51.0. No Python interpreter is installed.
- Debian package indices are pinned to signed archive snapshots dated 2026-09-01.
- Native dependency archives are SHA-256 locked in `dependencies.sha256`.
- `setup.sh` only installs dependencies/configures the environment, using
  `apt-get install --no-install-recommends`. It has no lifecycle or project commands.
- `container.ps1` owns inspection/build/create/start/stop/exec. It preserves and
  reports a same-named mismatched container. It never removes containers or volumes.
- `run.sh` builds native targets, runs their checks and starts the viewer;
  separate C++ executables own evaluation/training.

## Measured status on 2026-09-16

All native targets compile in Debian and CTest passes. The C++ CUDA computation
check returned a dot product of 16. MuJoCo physics and OSMesa rendering run on
the CPU; LibTorch inference and PPO network updates use CUDA.

The [CUDA baseline evaluation](validation/evaluation-cuda.json) passed all five 20-second
pretrained trials: 9.2255–9.2306 m forward, 48 alternating foot contacts per trial,
and no falls. All five passive zero-torque controls fell within 0.56–0.58 seconds.
The earlier [CPU baseline evaluation](validation/evaluation-cpu.json) produced
matching results. These are finite flat-ground results for Unitree's policy;
the CUDA run verifies GPU policy inference with CPU MuJoCo physics.

The [native PPO smoke report](../artifacts/training/native-smoke/run.json)
records 8,192 transitions, 128 optimizer updates, 12.075 seconds of CUDA training,
and a parameter-change L2 norm of 4.5562. Final and held-out policies fell after
1.38–1.40 seconds, so this verifies the optimization/checkpoint pipeline but
does not demonstrate a walking policy trained by this project.

The longer PPO run completed 1,001,472 transitions and 14,419 CUDA updates in
1,663.17 seconds. Its final policy stayed upright for five 20-second trials,
but achieved only 0.055–0.269 m displacement and zero alternating touchdowns;
it failed all five walking trials. See [the measured PPO result](validation/evaluation-ppo-million.json).
The separate native supervised-imitation experiment uses the same container
and fixed local checkpoint paths, with a distinct controller option in the viewer.
Its eight-frame-history student completed 65,800 CUDA updates on 80,000 labeled
simulation steps in 464.54 seconds. The selected export passed four of five
independent 20-second trials, covering 8.815–9.130 m with 49 alternating contacts
in each successful trial. The fifth fell at 16.66 seconds. See the
[imitation evaluation](validation/evaluation-imitation-history8.json).

Sources: https://github.com/unitreerobotics/unitree_rl_gym
and https://docs.pytorch.org/cppdocs/ .
