# Native humanoid walking experiment

This project puts the [technical brief](doc/README.md) into a measurable MuJoCo
experiment. The runtime, policy inference, evaluation and PPO trainer are C++17.
There is no Python interpreter, pip environment or Python conversion step.
The browser uses HTML/CSS/JavaScript to display frames from the native simulator.

The first baseline runs Unitree's published G1 walking policy through LibTorch's
C++ TorchScript loader. It is a real floating-base simulation with twelve leg
motors and contact dynamics. The upper body joints are fixed. A separate native
PPO trainer starts from random network weights on that same model.

**Validated on 2026-09-16:** all native targets compile in Debian; CTest passes.
The published controller completed five 20-second MuJoCo trials with CUDA policy
inference, moving 9.2255–9.2306 m with 48 alternating foot contacts and no falls in
every trial. All five zero-torque controls fell in 0.56–0.58 seconds. See the
[CUDA evaluation](runtime/validation/evaluation-cuda.json) and the earlier matching
[CPU evaluation](runtime/validation/evaluation-cpu.json). This establishes native
pretrained walking on the measured flat-ground trials with either inference device.

Native CUDA execution also passed a C++ tensor check (dot product: 16). The
[8,192-transition PPO smoke run](artifacts/training/native-smoke/run.json)
completed its training phase on CUDA in 12.075 seconds with 128 optimizer updates
and a parameter-change L2 norm of 4.5562. Its learned policy fell after 1.38 and
1.40 seconds in the final and held-out evaluations. Optimization, checkpoint
export and evaluation ran successfully; **our smoke policy does not walk**.

The longer PPO run completed **1,001,472 transitions and 14,419 CUDA updates in
27.7 minutes**, starting from random weights with sixteen environments. Its final
policy remained upright for all five 20-second trials, moving only 0.055–0.269 m
with zero alternating touchdowns: **0/5 walking passes**. See the
[PPO evaluation](runtime/validation/evaluation-ppo-million.json) and
[training record](runtime/validation/ppo-million-run.json).

A separate [native imitation experiment](training/DISTILLATION.md) learns from
the published controller's action labels, starting with random student weights.
It completed 80,000 labeled simulation steps and 65,800 CUDA updates in 7.74
minutes. Its selected eight-frame-history policy passed **4/5 independent
20-second trials**, moving 8.815–9.130 m with 49 alternating contacts in each
successful trial. The fifth trial fell at 16.66 seconds. This demonstrates
locally learned, expert-assisted walking, with an observed stability failure.
See its [evaluation](runtime/validation/evaluation-imitation-history8.json).
The viewer offers **Our imitation policy** separately from **Our PPO policy**
and the published controller. The [evidence record](runtime/VALIDATION.md)
distinguishes every controller and experiment.

**The [live viewer](http://127.0.0.1:43871/) is running.** The user approved the
separate port after `helicopter-dev` acquired the originally chosen port 43118
during the image build. `humanoid-mujoco-native` now serves port 43871; the
original failed `humanoid-mujoco-dev` container and `helicopter-dev` remain
preserved. See [the preserved conflict](runtime/port-conflict.json).

## Container and dependency setup

| Setting | Value |
| --- | --- |
| Container / image | `humanoid-mujoco-native` / `humanoid-mujoco:dev` |
| Base image | `debian:12-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171` |
| Command | `bash /workspace/run.sh` |
| Bind mount | `C:\Work\Natural-Dexterity\projects\manual-dexterity\humanoid` → `/workspace`, read-write |
| Named volumes | None |
| Viewer | `127.0.0.1:43871` → container `8080` |
| Restart policy | `no` |
| GPU | All NVIDIA GPUs; compute, utility and graphics |
| Init / shared memory / stop timeout | Enabled / 1 GiB / 20 seconds |
| Management labels | Ownership, project root and configuration digest |

Running container ID:
`0ee55590b050c97bea3b2b37288813f9edcc0d597471c73e1b947280e18a2ebc`.
Inspected image ID:
`sha256:603dfd694bbd1eea2002b555a737b5289165cd10f5654d76b57f10a8905d823c`.

`setup.sh` only installs reproducible dependencies/configures the environment.
It uses signed Debian snapshots dated 2026-09-01 and APT
`--no-install-recommends`. Native archives are checked against
`runtime/dependencies.sha256`, and header-only libraries have embedded SHA-256
checks. Installed APT versions are saved in `/opt/humanoid/debian-packages.txt`.
No project build, test, service, evaluation, training, or lifecycle commands live
in `setup.sh`.

`container.ps1` owns lifecycle. It inspects the exact name first and reuses only
a matching managed container, including a stopped one. A mismatch fails and
preserves the conflicting container. It contains no remove or prune command.
`run.sh` incrementally builds the native targets, runs CTest, and starts the
viewer. Model/checkpoint data and experiment outputs persist on the bind mount.

From PowerShell 7 in this directory, inspect or reuse the existing container:

```powershell
.\container.ps1 up
.\container.ps1 status
.\container.ps1 logs
```

Open [the local viewer](http://127.0.0.1:43871/). It shows live MuJoCo camera
frames, forward distance, speed, pelvis height and foot contacts. Pause preserves
the current state. Restart and controller/speed changes start a fresh trial.
A fall or 30-second horizon pauses the simulation; it never hides falls behind
automatic resets. The passive comparison applies zero torque to all motors.

The host has an RTX A2000 Laptop GPU with 8 GB VRAM. MuJoCo physics runs on CPU;
CUDA executes LibTorch inference and PPO network updates. OSMesa renders camera
frames on CPU. The GPU computation check is a native tensor operation:

```powershell
.\container.ps1 exec nvidia-smi
.\container.ps1 exec ./build/native/humanoid-gpu-check
Invoke-RestMethod http://127.0.0.1:43871/healthz
```

## Evaluate a walking claim

```powershell
.\container.ps1 exec ./build/native/humanoid-evaluate --device cuda --trials 5 --seconds 20 --output runs/evaluation.json
```

The evaluator runs five initial conditions of the published policy and five
matching zero-torque controls. Seed 0 is the upstream initial pose; other seeds
perturb the leg joints uniformly by ±0.005 rad. It records source/model/policy
hashes, runtime versions, distance, falls, speed, contacts and flight fraction.
These are initial-condition trials of one policy, not independent training seeds.

Declared finite-trial criteria: finish 20 seconds without a fall, move at least
half the commanded forward distance, achieve five touchdowns per foot and eight
alternations, and spend at most 10% of sampled time airborne. A fall means
pelvis height below 0.45 m or pelvis up-axis cosine below 0.5. Contact transitions
require 60 ms without contact. These measurements support visual gait inspection;
they do not establish absence of slipping, general stability or hardware readiness.

## Train our own policy

The user reduced this milestone to a bounded feasibility attempt: one visible
20-second walk, with at most two further training invocations and 1,200 trainer
seconds. **That attempt has stopped without a convincing walking proof.** The
selected from-scratch PPO checkpoint stayed upright for 20 seconds and advanced
0.849 m, with alternating foot lifts, but visual inspection showed a wide stance,
swivelling and substantial slipping. The two final invocations used 827.046
trainer seconds; no further training is scheduled or authorized.

See the [bounded outcome](runtime/PPO_BOUNDED_RESULT.md),
[saved replay](artifacts/ppo-milestone/bounded-proof-offline-render/preview.html)
and [scope and stopping rule](runtime/PPO_GOAL.md). The three-seed, 45-trial study
is deferred. The [history PPO trainer](training/WALK_PPO.md) includes causal
observations, a privileged critic and concurrent physics rollouts. At the user's
subsequent request, the live viewer's **Our PPO policy** now loads selected v8
checkpoint 6,291,456 for inspection, with its matching evaluation report. This
preview does not claim walking success. The earlier viewer policy and report
are preserved in `artifacts/ppo-milestone/viewer-before-bounded-preview`.

See [native PPO training](training/README.md). `humanoid-train` performs actual
LibTorch PPO updates using MuJoCo rollouts. It does not load Unitree's weights.
The first short run checks collection, GPU optimization, checkpointing and
evaluation. These checks passed in the recorded 8,192-transition CUDA smoke run;
its walking screen failed in both final trials. The completed million-transition
run learned standing, with 0/5 walking passes in its final evaluation.

The viewer's **Our PPO policy** option becomes available when an exported
native PPO checkpoint is published to `runs/trained-policy.pt`. It always loads
that fixed local artifact. Its presence only makes replay available; use the
training run's measured evaluations to assess the gait. **Our imitation policy**
loads `runs/distilled-policy.pt`. Both measured checkpoints are published locally.
The viewer's evaluation link follows the selected controller.

## Sources

- [Unitree RL GYM, pinned revision](https://github.com/unitreerobotics/unitree_rl_gym/tree/276801e46c5d433564f24658bac64f254b7d2d4b), BSD-3-Clause: model, pretrained policy and deployment contract.
- [G1 settings](https://github.com/unitreerobotics/unitree_rl_gym/blob/276801e46c5d433564f24658bac64f254b7d2d4b/deploy/deploy_mujoco/configs/g1.yaml): 500 Hz physics/PD, 50 Hz policy, 47 observations, 12 position offsets.
- [MuJoCo 3.3.2](https://github.com/google-deepmind/mujoco/releases/tag/3.3.2): native physics engine.
- [LibTorch C++ distribution](https://docs.pytorch.org/cppdocs/installing.html): native inference, autograd and CUDA tensor operations.

The original brief's Transformer, randomized terrain and hardware transfer remain
future milestones. The imitation student implements a short causal observation
history. The local viewer is marked `noindex` and is
intended for development review.
