# Native C++ walking-policy training

`humanoid-train` trains a new policy for the same physical Unitree G1 model used
by the browser viewer. Training and inference use C++17, MuJoCo 3.3.2 and the
LibTorch 2.7.1 CUDA 12.6 distribution. This directory contains no Python runtime,
Python training code, or pip dependencies.

Every normal invocation starts from random network weights. Each training
simulation is explicitly constructed without loading the published policy and
uses external action control. The pretrained demonstration remains a separate
baseline. A successful optimization smoke test is evidence that learning runs;
it does not establish a walking gait.

## Build and launch

The root `setup.sh` installs pinned native dependencies only, with Debian APT
`--no-install-recommends`. Root CMake builds the trainer and its math tests.
Container lifecycle stays in `container.ps1`; root `run.sh` owns build/check/start.
After the existing project container has built successfully, from PowerShell in
the `projects/manual-dexterity/humanoid` directory:

```powershell
# First bounded optimization smoke: four independent CPU physics worlds,
# with batched inference and PPO gradients on the CUDA GPU.
.\container.ps1 exec /workspace/build/native/humanoid-train --steps 8192 --envs 4 --max-seconds 600 --output /workspace/artifacts/training/native-smoke

# First longer experiment; each --output directory must be new.
.\container.ps1 exec /workspace/build/native/humanoid-train --steps 1000000 --envs 16 --seed 1 --max-seconds 1800 --output /workspace/artifacts/training/native-million-seed1
```

CUDA is required by default; there is no silent CPU fallback. `--device cpu`
is an explicit diagnostic option. MuJoCo physics runs on the CPU; the neural
network's forward pass, value estimates, gradients, and Adam updates run on the
selected device. Physics worlds are stepped sequentially in this first version.
Start with four environments to measure actual throughput and memory before
increasing the count. The default is 16; the CLI permits at most 64.

`--max-seconds` bounds the training phase and is checked between complete
rollouts. Initial/final evaluation and serialization take additional time. A
rollout is 128 control steps per environment, so requested transitions round
up to a whole rollout. SIGINT/SIGTERM requests finish the current rollout/update
and save the checkpoint; no simulator or container objects are removed.

## What the algorithm does

The actor and critic each have two 128-unit tanh hidden layers. The actor
produces a Gaussian over 12 latent position-offset actions. Sampling uses a
learned diagonal standard deviation; tanh maps the actions into [-1, 1]. The
simulator maps them into nominal joint positions plus 0.25 radians per action
unit and applies the official PD controller at 50 Hz (ten 2 ms physics steps).
The policy consumes the simulator's existing 47 scaled observations.

The implementation uses the PPO clipped surrogate with clipping 0.2, discount
0.99, GAE lambda 0.95, four shuffled epochs, minibatches of 256, Adam learning
rate 0.0003, and gradient norm clipping at 0.5. Approximate KL above 0.03 stops
the remaining minibatch updates for that rollout. Exploration regularization
uses the analytic entropy of the Gaussian before tanh, explicitly as a proxy.
The policy ratio itself uses the exact tanh-Gaussian density with a stable
Jacobian correction and the retained latent samples.

A fall is a terminal state. The 1,000-control-step (20-second) episode limit is
a truncation: its value target bootstraps from the actual final observation
before reset. Both endings cut the GAE recursion so returns from the next
episode cannot leak backward. The pure C++ math test covers this distinction,
episode isolation, independent batched environments, and rollout boundaries.

The first reward is intentionally simple and needs experimental tuning:

| Term | Per-control-step contribution |
| --- | --- |
| Alive | +0.5 |
| Forward velocity near 0.5 m/s | +1.5 × exp(-(vx - 0.5)^2 / 0.25) |
| Torso upright | +0.5 × clamp(upright cosine, 0, 1) |
| Alternating single-foot support | +0.2 on an opposite support transition, separated by at least 0.16 s |
| Lateral velocity | -0.2 × vy^2 |
| Action magnitude | -0.02 × mean(action^2) |
| Action slew | -0.05 × mean((action - previous action)^2) |
| Fall / invalid physical state | -5, then terminate the episode |

Action penalties regularize joint targets. They are not measured torque,
mechanical work, or power. The reward can favor standing or exploit contacts;
it is a starting hypothesis, not an upstream validated training configuration.

## Outputs and replay

Each run writes:

- `run.json`: configuration, selected compute device, completed transitions,
  optimizer-update count, parameter change, and initial/final evaluation.
- `progress.csv`: reward, falls, timeouts, losses, KL, and measured throughput.
- `checkpoints/<steps>/model.pt`: actor, critic, and distribution parameters.
- `checkpoints/<steps>/optimizer.pt`: Adam state, with accompanying metadata.
- `policy.pt`: a deterministic **TorchScript** actor for the native viewer;
  exported and reloaded entirely in C++. This is distinct from `model.pt`.
- `initial-`, `final-`, and `heldout-evaluation.json`, plus corresponding CSV
  trajectories of continuous physics measurements.

The export checks inference equivalence on a nonzero probe and reloads the
saved TorchScript file before declaring success. The shared native simulator
can load this `policy.pt` through `load_policy()` and the `trained` mode; it has
the same 47-input/12-output action interface as the published walking policy.

To remeasure one of our checkpoints without training:

```powershell
.\container.ps1 exec /workspace/build/native/humanoid-train --eval-only /workspace/artifacts/training/native-smoke/checkpoints/8192/model.pt --output /workspace/artifacts/training/native-smoke-replay
```

Evaluation never resets after a fall. It compares the random initial policy
with the learned policy on the same small initial-joint perturbation, then
uses another held-out perturbation seed. The provisional candidate gate needs
20 uninterrupted seconds, no fall, at least 3 m forward displacement, at least
90% frames with base height >=0.55 m and upright cosine >=0.8, and at least
three touchdowns per foot in both final trials. Inspect live movement for
stepping rather than sliding or hopping. Further seeds and perturbations are
needed before claiming repeatable or robust walking. The trainer's raw contact
edges are only a preliminary screen. Use the shared evaluator's debounced
touchdowns, alternating support and paired zero-torque controls for a walking
claim:

```powershell
.\container.ps1 exec /workspace/build/native/humanoid-evaluate --device cuda --policy-path /workspace/artifacts/training/native-million-seed1/policy.pt --trials 5 --seconds 20 --output /workspace/artifacts/training/native-million-seed1/evaluation.json
```

To inspect an exported checkpoint in the browser, publish it atomically to the
viewer's fixed local path, then choose **Our PPO policy**:

```powershell
New-Item -ItemType Directory -Force runs | Out-Null
Copy-Item artifacts/training/native-million-seed1/policy.pt runs/trained-policy.pt.tmp
Move-Item -Force runs/trained-policy.pt.tmp runs/trained-policy.pt
```

This copies the export; it preserves the experiment's original checkpoint.

## Validation status and sources

The GAE math test passed locally with Clang and in the actual Debian image
with GCC 12. All native targets, including the trainer, compiled successfully.
The published walking controller passed five CPU and five CUDA MuJoCo trials.
Native CUDA computation is verified on the RTX A2000 Laptop GPU. The first
8,192-transition PPO run completed 128 optimizer updates in 12.075 seconds,
changed parameters by L2 norm 4.5562, and exported/reloaded its native policy.
It fell after 1.38–1.40 seconds in continuous evaluation: optimization works,
but that checkpoint does not walk. Detailed experiment results are retained
under `artifacts/training/`; measured walking evidence is under
`runtime/validation/` in the project root.

The longer `native-million-seed1` PPO run completed 1,001,472 transitions and
14,419 optimizer updates in 1,663.17 seconds. Its exported policy stayed upright
for 20 seconds in all five independent reset trials, but moved only 0.055–0.269 m
and recorded zero debounced footsteps or alternations. It passed zero of five
walking screens: this PPO run learned to stand, not walk. The shared result is
retained in `runtime/validation/evaluation-ppo-million.json`.

A separate [native imitation experiment](DISTILLATION.md) learns from Unitree
controller labels. It has a distinct checkpoint and viewer option, **Our
imitation policy**. Its eight-frame student passed four of five independent
20-second walking trials, covering 8.81–9.13 m with 49 alternating touchdowns
in each pass; the remaining trial fell at 16.66 seconds. This demonstrates
locally learned walking through supervised imitation. It is not evidence of
learning by PPO from scratch, and one failed trial limits the robustness claim.

- [MuJoCo simulation documentation](https://mujoco.readthedocs.io/en/3.3.2/programming/simulation.html).
- [PyTorch C++ frontend](https://docs.pytorch.org/cppdocs/frontend.html) and
  [LibTorch serialization](https://docs.pytorch.org/cppdocs/notes/serialization.html).
- [PPO paper](https://arxiv.org/abs/1707.06347) and
  [GAE paper](https://arxiv.org/abs/1506.02438).
- [Handling termination versus time limits](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).
- [Official Unitree RL GYM source](https://github.com/unitreerobotics/unitree_rl_gym/tree/276801e46c5d433564f24658bac64f254b7d2d4b).
