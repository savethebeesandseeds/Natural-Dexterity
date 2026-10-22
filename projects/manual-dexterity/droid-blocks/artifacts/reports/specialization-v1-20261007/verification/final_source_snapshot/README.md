# Droid Blocks

Droid Blocks is a Python-free C++20 MuJoCo development service for exploring modular robots. The project directory is bind-mounted into its Debian container, and `run.sh` incrementally configures, builds, tests, and starts the native server.

The September 7 [neural transfer experiment](docs/ACTUATION_TRANSFER_RESULTS.md)
historically used an isolated host Python environment with Stable-Baselines3
PPO. Current development instructions require all new training inside the
managed Linux container; the native service and live learner remain separate. Open the
[recorded motor-move comparison](http://127.0.0.1:43117/transfer.html) to watch
frozen transfer, continued learning and scratch share the same timeline.

New collaborators should begin with the living
[`open project brief and handoff`](docs/OPEN_HANDOFF.md). It separates the
current evidence and preserved contracts from the questions we actively invite
others to challenge.

## Pretrained competence on the first build

The owner's October 6 clarification is that a child's first assembly should
already benefit from broad simulated training, then need little new experience
to specialize to its body. Retaining knowledge after rebuilding builds on that
starting competence. The
[prospective pooled-pretraining pilot](docs/GENERALIZATION_PRETRAINING_EXPERIMENT.md)
uses the existing native two-link environment and established PPO implementation
to prepare a first bounded comparison. Known development performance and
genuinely new assembly performance are separate measurements; neither the old
transfer study nor the fresh-memory headroom result proves this product goal.

Optional CPU training dependencies are installed by
`training/setup-generalization.sh` into the existing managed container's
filesystem. Training and dependency caches stay in Linux; the base image and
native launcher still build a C++ service. Follow the prospective protocol for
its source manifest, declared budgets and execution commands.

The [completed first pilot](docs/GENERALIZATION_PRETRAINING_RESULTS.md) trained
one pooled policy on 32 conditions. With its weights frozen, it cleared the
useful-behavior margin over quiet in all 12 known cases and beat the matched
untrained policy in 10 of 12. All 36 development trials completed safely and
37 native request tapes replayed exactly. Gains were uneven and two cases
worsened relative to initialization. That pilot remains a known-domain record.

The [completed three-seed breadth study](docs/GENERALIZATION_STUDY_RESULTS.md)
then compared broad and narrow pretraining under equal budgets. All 36 broad
policy runs safely completed across the twelve reserved body/light conditions, beat
quiet by the declared margin and beat their saved untrained controls, with no
weight updates. Broader training improved the mean in every seed but won only
21 of 36 individual comparisons; narrow training also produced useful behavior
on every target. Sixteen fixed-witness safety stops remain in the evidence.
This is first-use geometry generalization within one anchored two-link graph.
Useful online specialization, arbitrary connections, multiple motors and hardware
remain separate milestones. The pretrained policy now powers the visible
build/wake loop, and the bounded adaptation comparison below measures the next step.

The [pretrained build-and-wake workshop](http://127.0.0.1:43117/pretrained.html)
now integrates pooled seed 101's frozen actor into native real-time physics.
Choose one of three evaluated bodies, change motor placement and lamp side,
then Rebuild and Wake. Rebuilding clears old-body sensor history while retaining
the same brain; Pause/Resume preserves an unfinished action. The
[runtime contract and evidence](docs/PRETRAINED_WORKSHOP.md) explain the actor
export, portable stochastic sampler, numerical checks and historical replay.
This completes the visible pretrained reuse milestone in the bounded kit.

The [October 7 specialization results](docs/SPECIALIZATION_RESULTS.md) compare
128 learning decisions followed by 512 fixed-weight decisions against frozen
reuse and matched scratch. All 294 target/retention opportunities safely completed,
and independent replay passed 21,227,993 checks. Head-only tuning's median seed
mean gain was −0.337379; full tuning's was +1.410498. Both missed the declared
practical +3.2 gate. Both beat scratch in every whole target comparison, but
frozen reuse also did so. Two-condition retention averages passed their gate;
full tuning still lost 10.037916 in one probe. Keep the unchanged pretrained
actor as the default while investigating useful, reliable specialization.

## Is there room for useful movement?

The [new native development study](docs/LIGHT_HEADROOM_EXPERIMENT.md) separates
two questions raised by the failed motor-transfer experiment: whether movement
can improve sensor reward over a quiet motor, and whether the unchanged learner
finds that improvement within 64 simulated seconds. Three bodies, two motor
positions and four lamps are compared under a protocol locked before collection.
Every fixed control, sensor-feedback witness and learner seed remains in the
record. The study uses the existing Debian container and requires no new
dependencies.

The [completed results](docs/LIGHT_HEADROOM_RESULTS.md) pass both declared
lower-lamp criteria: all 12 cases have a safe fixed witness, and all 36 native
learner runs beat quiet by more than +3.2 integrated reward. Historical upper
lamps remain difficult, and all 11 feedback-stopped trials are retained. This
demonstrates learning from fresh memory on known conditions; transfer remains
open.

The [recorded comparison](http://127.0.0.1:43117/headroom.html) shows quiet and
any declared controller together, with all 216 trials in the complete case
table. Results retain native replay, independent accounting and preservation
receipts. Builds and experiment execution use the managed Debian container.

## Try the sun-seeker

With the managed service running according to [Safe first launch](#safe-first-launch),
open the [local workshop](http://127.0.0.1:43117/). The saved learned controller
starts ready with sunshine on the left. Choose a side, press Play, and watch
the light at its eye during a 20-second simulated run. Pause preserves the run;
Reset or a side choice prepares a fresh one. Machine details offers an explicit
**Switch to cruise demo** and **Prepare the sun-seeker**. Airship view changes
only the presentation.

This playground uses two published training worlds and the preserved v3
artifact. It does not train during play or supply new held-out evidence. See
[The sun-seeker playground](docs/SUN_SEEKER_PLAYGROUND.md) for its controls,
native lifecycle, provenance, limitations, and verification record.

## Change its body

Choose **Change its body** in the same workshop to try a normal, reversed, or
disconnected transmission. **Discover each motor** measures a separate light
response for each motor; **Try motors together** measures one shared response.
Each 40-second experiment starts with fresh memory. The motor cards show what
was measured, including unclear responses. This successor uses a handwritten
probing controller and a delayed light sensor, with the original sun-seeker
available separately. See [I changed its body](docs/BODY_DISCOVERY_EXPERIMENT.md)
for the mechanical model, development comparison, preserved failures, and limits.

## Native architecture

The executable module vocabulary is stored in `config/module_catalog.json` and implemented by `include/droid/module_catalog.hpp` and `src/module_catalog.cpp`. Its rationale and equations are documented in `docs/MODULE_CONTRACT.md`. Version `0.1.0` contains one current-commanded rotary motor SKU, `rotary_dc_gearmotor_v0`, and five optional reward-sensor families:

- `power_v0`
- `imu_6axis_v0`
- `touch_force_v0`
- `proximity_tof_v0`
- `ambient_light_v0`

Plastic blocks, wheels, gears, and axles are passive mechanics. Only reward-sensor instances that are registered and present on an assembly contribute to its reward sum. A catalogued but absent family contributes nothing; a registered instance that later becomes stale or invalid contributes its family's `missing_reward`. The environment objective is literally `dt * sum(sensor reward rates)`. Hard safety vetoes are separate and cannot be cancelled by positive reward.

Motor encoder position and velocity, current, voltage, temperature, fault flags, load impedance, and stuck detection are actuator feedback. They are policy observations and safety data, have `reward_vote: false`, and therefore have no reward authority. The demo mounts one `ambient_light_v0` and two `touch_force_v0` instances. The other three reward-sensor families remain catalogued end targets.

The native service preserves the browser API:

- `GET /healthz`
- `GET /api/state`
- `GET /api/catalog`
- `POST /api/control` with `play`, `pause`, or `reset`
- `POST /api/playground/control` with `play`, `pause`, `reset`, `body`, or `demo`

The default playground uses frozen learned inference. The separately selected
cruise demo uses privileged position and orientation and is not part of the
learning-agent contract. Browser play/pause/reset preserve the selected mode.

## Native learning environment

`droid::DroidEnvironment` is the synchronous C++ learning boundary. A default,
worker-free instance advances MuJoCo as quickly as the CPU permits and never
sleeps to match wall-clock time. The visualization server instead owns one
real-time instance coordinated by `droid::PlaygroundSession`. Learned playback
steps it through the synchronous observation boundary. An explicit agent reset
takes external ownership; browser controls cannot silently take that ownership
away. Selecting the cruise demo is a separate explicit operation.

The fixed v0 assembly, `demo_rover_v0`, accepts exactly one finite effort for
each of `motor-0001` through `motor-0004`. The nominal 0.02-second control step
contains ten 0.002-second physics steps. Its observation contains only the
three mounted reward-sensor samples and four actuator-feedback samples; world
pose, task geometry, authored part roles, and demo-controller state remain on
the separate visualization surface and are not policy inputs. See
[`docs/AGENT_ENVIRONMENT.md`](docs/AGENT_ENVIRONMENT.md) for the complete
contract.

`fixed_demo_v0`, the default episode profile, keeps the light at `x = +16 m`
for the visual demo and legacy baseline. Learning uses the distinct seeded
`light_search_1d_v1` profile. After a tagged SplitMix64 finalizer, bit 63 chooses
negative or positive x and the low 24 bits choose an exact distance
`12 + bucket * 2^-21` metres in `[12, 20)`. The seed, side, distance, and
realization key are hidden from the policy; only their physical sensor effect
is observable.

The HTTP evaluation adapter adds:

- `GET /api/agent/spec`
- `POST /api/agent/reset` with optional `assembly_id`, unsigned `seed`, Boolean
  `record`, and `episode_profile` fields
- `POST /api/agent/step` with an `actions` object and optional
  `control_dt_s`
- `GET /api/agent/trace`, available when the current episode was reset with
  recording enabled

For example, after a reset, one nominal external-control transition is:

```json
{
  "actions": {
    "motor-0001": 0.25,
    "motor-0002": 0.25,
    "motor-0003": 0.25,
    "motor-0004": 0.25
  },
  "control_dt_s": 0.02
}
```

Use the in-process CLI for headless benchmarking and deterministic trace work:

```sh
./build/native/droid-agent benchmark --steps 20000 --seed 0 --control-dt 0.02
./build/native/droid-agent record --output build/episode-0001.json --steps 64 --seed 7
./build/native/droid-agent replay --input build/episode-0001.json
```

Recording is optional and disabled by the benchmark. A trace contains the
reset result and every ordered action, interval, and returned transition.
Saving uses an exclusive, same-filesystem installation and refuses to
overwrite an existing destination. Replay validates the trace schema, creates
a fresh worker-free environment, and requires exact JSON equality at reset and
at every transition; it reports the first mismatch.

## Non-learning control baseline

The project retains two reproducible non-learning controls: a zero-effort
policy that measures passive reward, and a seeded sample-and-hold random policy
that probes action sensitivity and naive exploration. Both declare
`uses_observation: false`; neither reads a
sensor, actuator feedback, reward, task geometry, or privileged simulation
state when selecting actions. They are not RL policies and their results are
not evidence of learning.

Run both controls through the worker-free native environment with:

```sh
./build/native/droid-agent evaluate
```

The complete interface is:

```text
droid-agent evaluate [--policy zero|random|both]
                     [--episodes N] [--steps N] [--seed N]
                     [--control-dt SECONDS]
                     [--random-effort MAGNITUDE] [--hold-steps N]
                     [--model PATH]
```

Defaults are both policies, eight episodes per policy, 1,000 control
transitions per episode, seed 0, a 0.02-second control interval, random effort
bounded to `[-0.6, +0.6)`, and a five-transition hold. Environment episode
seeds are paired across policies while random-policy seeds use a separate
deterministic stream.

Episode return is the raw, undiscounted sum of transition rewards; it is not
integrated a second time. Results also retain actual simulated time, reward
rate, safety termination, environment truncation, and runner-owned horizon
completion. Aggregate return dispersion is the population standard deviation.
Runtime and process-memory measurements are reported separately from policy
quality. See [`docs/BASELINE_EVALUATION.md`](docs/BASELINE_EVALUATION.md) for
the policy definitions, seed contract, metrics, interpretation limits, and
success criteria.

The validated default run completed 8 episodes per policy and 16,000 total
environment transitions. `zero_v0` had mean return `13.0253600000`;
`seeded_random_sample_hold_v0` had mean return `13.0378242925` with population
standard deviation `0.0577909571`. The descriptive random-minus-zero delta was
`+0.0124642925`, with zero safety terminations, zero environment truncations,
zero safety vetoes, and zero invalid samples. Both policies reached all eight
runner-owned horizons. This slight random-policy advantage is not evidence of
learning: reward was overwhelmingly ambient-light reward, and both policies
are open-loop non-learning controls.

One native validation run completed 20,000 environment transitions (200,000
physics transitions) in 5.1003 seconds: 3,921 environment transitions/s,
39,214 physics transitions/s, and 78.43 simulated seconds per wall-clock
second, with peak resident memory of 11,669,504 bytes. This is a local
throughput baseline, not an RL quality score or a cross-machine guarantee.

## Native learning milestone

The first learned controller is deliberately compact and Python-free:
`shared_linear_memory_v2` uses one shared vector of 33 binary64 parameters for
every motor. Its module-set encoder pools the currently present ambient-light,
touch, and actuator-feedback samples without treating array order or opaque IDs
as geometry. Per-motor inference combines those pooled values with local
actuator feedback, one-step sensor deltas, previous effort, a bias, and two
light-change interaction features, then applies `tanh`, a `0.6` effort bound,
and a `0.1` per-control-step slew bound. Missing valid reward sensing or
missing, invalid, or faulted local feedback fails that motor to zero effort.
Its raw parameter payload is `33 * 8 = 264` bytes.

The official v3 optimizer is `cem_masked_best_sample_v1`. It searches only
physical parameter indices `[29, 30, 32]`—previous effort, bias, and scaled
ambient-delta × local velocity—with respective physical scales
`[2.0, 0.25, 64.0]`; every inactive parameter is positive zero. CEM begins at
latent mean `0.0`, with latent standard deviation `1.0` and floor `0.1`.
Deterministic candidate noise is keyed by physical parameter index. Candidates
are ranked on a balanced train batch by a safety-first, paired-zero,
side-robust reward-rate objective. Each generation's best train-ranked sampled
candidate is then evaluated on the complete validation split; the updated
distribution mean is not a validation checkpoint.

Train and freeze a policy with the official defaults:

```sh
./build/native/droid-agent train \
  --output-policy build/light-search-policy-v3.json
```

The optional train-only CEM controls are `--generations`,
`--population-size`, `--elite-count`, `--train-batch-size`,
`--initial-stddev`, `--minimum-stddev`, `--update-rate`, and
`--optimizer-seed`. Overrides create an exploratory manifest and artifact; they
do not silently claim the official manifest digest. Output installation is
exclusive and never overwrites an existing artifact. `--workers N` controls
only parallel candidate evaluation (default `1`); it is execution metadata and
does not change the policy, effective manifest, or official protocol identity.

Before any held-out run, inspect a frozen checkpoint on its exact validation
split with the non-official diagnostic:

```sh
./build/native/droid-agent validate-policy \
  --input-policy build/light-search-policy-v3.json
```

This compares learned and zero control on validation only, reports overall and
per-side paired deltas and wins, emits no acceptance decision, and performs no
test rollout.

The official pre-held-out run froze
[`artifacts/light-search-policy-v3-seed0.json`](artifacts/light-search-policy-v3-seed0.json)
at generation 6. Its embedded canonical-payload `self_sha256` is
`4dd4d6ccadecb0b3b69d3b3d412af83eff15208b19058701538dbdf83a4c32b5`;
the SHA-256 of the exact serialized file bytes is separately
`6f7d5069bdb884d75f1241629d437c039a2f7a486633d9ef2d7f6b70db0a5582`.
The frozen protocol-manifest SHA-256 is
`f06d689351ddf10c6e987661d7a1f935df4cd8a3c911a76dd13395a6d4b49e93`.
Training took `715.225449445 s` wall time with peak RSS `30,363,648` bytes.

An independent validation-only reload gained
`+0.09730391078745496 reward/s` over paired zero control and won `16/16`
scenarios: negative x gained `+0.09791938469996925` with `8/8` wins, and
positive x gained `+0.09668843687494068` with `8/8` wins. It had zero
terminations, safety vetoes, invalid reward-sensor samples, and invalid
actuator-feedback samples. Mean absolute effort was
`0.31318514771991324`, maximum effort `0.5999999976413114`, peak current
`1.4999999759926383 A`, and peak temperature `27.840471196252533 C`.
This is official pre-held-out evidence, not an acceptance result.

That artifact is awaiting explicit approval for the preregistered procedure's
intended one held-out comparison:

```sh
./build/native/droid-agent evaluate-policy \
  --input-policy build/light-search-policy-v3.json \
  --output-report build/light-search-heldout-v3.json
```

`--output-report PATH` is required. The command installs that report
exclusively and refuses to overwrite an existing path. This protects the named
record; it does **not** make held-out execution technically or cryptographically
one-shot. A different path or copied workspace can run the command again.
Single execution is therefore an experiment-governance rule for this release.
Until the v3 decision is fixed, its held-out split and outcomes cannot influence
the v3 artifact, optimizer, checkpoint, acceptance rule, or claim. After that
decision is recorded, the evaluated worlds, topologies, and outcomes may be
explicitly relabeled and promoted to curriculum, replay, or diagnostics for a
future protocol version. They are then known data—not unseen evidence—for every
descendant that uses them; each such release needs a fresh, versioned challenge
split that remained outside its own development loop.

The command verifies the artifact checksum and exact runtime compatibility,
then compares the frozen policy with zero effort and four independently seeded
matched random-control streams on the untouched 64-scenario test split. The
decision uses side-stratified paired bootstrap intervals with Bonferroni
control, practical-gain and win-count gates, per-side gates, and strict safety
gates. A completed experiment that yields `accepted: false` still exits
successfully; it is a scientific result, while schema, provenance, or runtime
failures are command errors.

The executable and checked-in `config/learning_experiment_v3.json` are
protocol-fixed by canonical SHA-256. Here, protocol-fixed means mismatches are
rejected; it does not prevent another execution. Frozen artifacts also bind
the model bytes, module catalog, environment specification, feature schema,
motor topology, API version, and control interval. These hashes detect mismatch
or mutation; they are integrity checks, not signatures or proofs of authorship.
Exact rollout reproduction is limited to the same validated runtime because
MuJoCo and floating-point trajectories are not promised bit-identical across
platforms.

The official v3 held-out result is **pending**. The official artifact has been
trained and independently checked on validation, but its validation output
states `test_rollouts_executed: false`; no held-out report exists, and held-out
execution awaits explicit approval. Existing v1 and v2 policy files are
exploratory, are ineligible for the v3 held-out protocol, and contain no test
outcomes. See
[`docs/LEARNING_EXPERIMENT.md`](docs/LEARNING_EXPERIMENT.md) for the complete
protocol, leakage controls, acceptance conjunction, and pending results record;
see [`docs/LEARNING_DESIGN_HISTORY.md`](docs/LEARNING_DESIGN_HISTORY.md) for the
pre-held-out evidence that motivated v3.

## Build and dependency contract

CMake separates two native libraries and builds the service, experiment runners
and validation executables. The foundational targets are:

| Target | Purpose |
| --- | --- |
| `droid_core` | Module contract, motor model, reward calculation, synchronous environment, trace/replay, and MuJoCo simulation |
| `droid_agent_core` | Generic rollout evaluator, baseline adapters, frozen policy, artifact handling, native learner, and playground session |
| `droid-blocks-server` | HTTP API and static playground server |
| `droid-agent` | Worker-free benchmark, trace/replay, baseline evaluation, training, and frozen held-out evaluation CLI |
| `droid-native-tests` | Deterministic module-contract and motor-model tests |
| `droid-native-simulation-tests` | Golden-trajectory parity test for the complete simulator |
| `droid-native-environment-tests` | Policy-boundary, reward aggregation, validation, trace, replay, and mode tests |
| `droid-native-baseline-tests` | Observation-free policy, seed isolation, reward accounting, lifecycle, aggregate-statistics, and determinism tests |
| `droid-native-policy-tests` | Feature encoding, topology invariance, artifact integrity, compatibility, and frozen inference tests |
| `droid-native-training-tests` | Seed splits, deterministic CEM, manifest lock, held-out statistics, and protocol rejection tests |
| `droid-native-playground-tests` | Known-training-world frozen inference, playback lifecycle, ownership isolation, compatibility rejection, and failure recovery |
| `droid-native-body-tests` | Signed and disconnected mechanical transmission, causal sensor timing, original-model restoration, and replay |
| `droid-native-body-controller-tests` | Paired response discovery, local speed governor, observation boundaries, causal sample use, and per-run memory |
| `droid-body-experiment` | Retained 18-episode development comparison on the three declared bodies and two known training worlds |

The image uses the immutable Debian 12 digest in `Dockerfile`. `setup.sh` only installs reproducible build/runtime dependencies; it does not manage the container or run project operations. It installs `build-essential`, CMake, Ninja, curl, certificates, and the required MuJoCo graphics libraries with `--no-install-recommends`, then downloads these verified artifacts under `/opt`:

| Dependency | Version | SHA-256 |
| --- | --- | --- |
| MuJoCo Linux x86_64 SDK | 3.12.0 | `a9367911e6d5eaeade17c2197304687421c1fc932cdf7bcd4cb8cfaf0374dcb2` |
| nlohmann JSON single header | 3.12.0 | `aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63` |
| cpp-httplib single header | 0.51.0 | `dfbaccb76432ed6d56ddd9983fd9d262b61ba6ba0958f6b00db35c802607bd35` |

`docker build` copies CMake/native sources, configuration, model, native
fixtures, and the one preserved v3 policy artifact into an isolated validation
stage. It builds the single `droid-native-validation` aggregate target, then
passes every registered CTest suite
before the development image is produced. The final image retains the
toolchain for fast incremental builds from the bind mount. Both `python` and
`python3` are explicitly checked to be absent.

For a direct build inside the configured Debian environment:

```sh
cmake -S . -B build/native -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
cmake --build build/native --target droid-native-validation
ctest --test-dir build/native --output-on-failure
./build/native/droid-blocks-server --host 0.0.0.0 --port 8080 --model models/droid.xml --web-root web
```

## Fixed container contract

| Setting | Value |
| --- | --- |
| Image | `droid-blocks:dev` |
| Container | `droid-blocks-dev` |
| Command | `/workspace/run.sh` |
| Bind mount | This project directory to `/workspace`, read-write |
| Published port | `127.0.0.1:43117:8080` |
| Restart policy | `no` |
| Named volumes | None |
| GPU/device access | All NVIDIA GPUs requested; compute verified, graphics capability requested |
| Init process | Docker init enabled |
| Shared memory | 1 GiB |
| Stop timeout | 15 seconds |
| Labels | `io.waajacu.managed=true`, `io.waajacu.project=droid-blocks` |

## Safe first launch

Run these commands in PowerShell from this project directory. First inspect the requested name, even when the container is expected to be absent:

```powershell
docker container inspect droid-blocks-dev
```

If the container exists, do not create, delete, rename, or replace it. Confirm its immutable ID and full configuration:

```powershell
docker container inspect droid-blocks-dev --format 'ID={{.Id}} Image={{.Config.Image}} ImageID={{.Image}} Command={{json .Config.Cmd}} Init={{json .HostConfig.Init}} StopTimeout={{json .Config.StopTimeout}} Restart={{.HostConfig.RestartPolicy.Name}} ShmSize={{.HostConfig.ShmSize}} Env={{json .Config.Env}} Labels={{json .Config.Labels}} Mounts={{json .Mounts}} Ports={{json .HostConfig.PortBindings}} Devices={{json .HostConfig.Devices}} DeviceRequests={{json .HostConfig.DeviceRequests}}'
```

Reuse it only when every setting matches the table. A stopped matching container can be started in place; `docker start` also leaves an already-running matching container intact:

```powershell
docker container start droid-blocks-dev
docker container logs --follow droid-blocks-dev
```

If the name is absent, build the image and create the one managed container. The explicit source path avoids mounting the wrong directory:

```powershell
docker build --tag droid-blocks:dev .

docker container create `
  --name droid-blocks-dev `
  --label io.waajacu.managed=true `
  --label io.waajacu.project=droid-blocks `
  --mount 'type=bind,source=C:\Work\Universalis-Materiae-Dexteritas\projects\droid-blocks,target=/workspace' `
  --publish 127.0.0.1:43117:8080 `
  --gpus all `
  --env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics `
  --init `
  --shm-size 1g `
  --stop-timeout 15 `
  --restart no `
  droid-blocks:dev

docker container start droid-blocks-dev
```

Immediately record and verify the immutable image and container IDs and effective configuration:

```powershell
docker image inspect droid-blocks:dev --format 'ImageID={{.Id}} RepoDigests={{json .RepoDigests}}'
docker container inspect droid-blocks-dev --format 'ID={{.Id}} Image={{.Config.Image}} ImageID={{.Image}} Command={{json .Config.Cmd}} Init={{json .HostConfig.Init}} StopTimeout={{json .Config.StopTimeout}} Restart={{.HostConfig.RestartPolicy.Name}} ShmSize={{.HostConfig.ShmSize}} Env={{json .Config.Env}} Labels={{json .Config.Labels}} Mounts={{json .Mounts}} Ports={{json .HostConfig.PortBindings}} Devices={{json .HostConfig.Devices}} DeviceRequests={{json .HostConfig.DeviceRequests}} Health={{json .State.Health}}'
```

Open <http://127.0.0.1:43117/>. Docker checks <http://127.0.0.1:8080/healthz> inside the container every ten seconds. The uncommon host port is the only published endpoint.

The NVIDIA runtime supplies the host driver interface. The current `mj_step` simulation loop remains CPU-based. On this Docker Desktop/WSL setup, an earlier MuJoCo EGL probe selected Mesa `llvmpipe`; container-side camera rendering is therefore not yet NVIDIA-accelerated even though graphics capability is requested.

## Conflicts and rebuilds

A same-named container with a missing label or mismatched setting is unmanaged or incompatible. Preserve it and report its immutable ID and differences; do not stop, delete, or replace it. Use another name only after the revised configuration is explicitly approved.

Rebuilding the image does not update an existing container. Replacing a container is a separate, explicitly authorized operation: inspect the exact managed container by immutable ID first and preserve every host-mounted or named-volume data source. This project normally has no named volumes.

## Build a creature

The [construction bench](http://127.0.0.1:43117/construction.html) is the third
playground experiment. Extend either beam, snap on weight blocks, move the
motion sensor, and apply the body before exploring its motions. Each native
exploration compares five six-second patterns from rest. Recorded motion cards
can be replayed; pause preserves progress. Save/load preserves the assembly as
JSON. Save also keeps a browser-local copy with Restore saved body. Rebuilding clears evidence from the old body; returning to rest keeps
the current body's motion cards.

The first kit has an anchored motor and a genuinely passive second hinge.
Its geometry and mass distribution affect native MuJoCo physics. It is a
collisionless, planar mechanism with a movable physical IMU, not yet an
arbitrary assembly graph or a ground-contact robot. Exploration records
responses without optimizing the IMU's reward. See the
[broader adopted goal](docs/BUILD_AND_DISCOVER_GOAL.md) and the
[construction experiment record](docs/CONSTRUCTION_EXPERIMENT.md).

## A little sunshine: continuous learning with a constructed body

Open `http://127.0.0.1:43117/light.html`, or follow **Give it a little sunshine**
from the construction bench. **Use my workshop body** brings a separate copy of
your assembly and adds a physical light sensor at an empty socket. Move its
receiving face, choose **Learn from light**, and let it try. Moving the sun
preserves physical motion, delayed readings and learner memory. A fresh start
or a rebuild clears memory; pause/resume preserves it.

The native `history_expected_sarsa_v2` learner remembers recent sensor/action
history and updates values for positive effort, negative effort and coasting.
Its only inputs are local sensor samples, actuator feedback and summed sensor
reward; sun position and body pose stay outside the policy. Quiet, fixed-rhythm
and random controls are available for comparison. See
[the light-learning experiment](docs/LIGHT_LEARNING_EXPERIMENT.md) for the
protocol, first unsuccessful learner, measured correction, full results and
limits. This remains an anchored collisionless two-link development kit.

The final 16-trial comparison completed safely and replayed exactly; every
learning seed exceeded its body's comparison returns. All fifteen native
suites passed. The earlier 24 construction observation traces and commands
were preserved exactly, as were the frozen original v3 inputs and pending
challenge. Reports live in the versioned `artifacts/reports/light-learning-*`
directories, with source snapshots. To create fresh development evidence:

```sh
./build/native/droid-light-experiment --output artifacts/reports/NEW_DIRECTORY
```

The runner refuses an existing output directory. This command is development
learning, not the untouched frozen-v3 challenge.

## Does continued learning help when the sun moves?

The [paired result viewer](http://127.0.0.1:43117/adaptation.html) shows the next
engine experiment. Six pairs share exactly the same first minute; after the
sun moves, one keeps updating its values while the other uses frozen values.
Both continue sensing, remembering and controlling motion. Continuing updates
improved post-move sensor reward in all six known development pairs. Twelve
complete native trials and exact replays passed, as did all 15 native suites.
See the [full protocol, results and limits](docs/LIGHT_ADAPTATION_EXPERIMENT.md).

```sh
./build/native/droid-light-adaptation --output artifacts/reports/NEW_DIRECTORY
node tools/summarize-light-adaptation.mjs artifacts/reports/NEW_DIRECTORY/report.json
node tools/export-light-adaptation-view.mjs artifacts/reports/NEW_DIRECTORY/report.json
```

The native report is immutable evidence with source snapshots; the viewer is
a read-only derived display. The runner refuses existing output directories.
This experiment leaves the original rover-v3 challenge pending and does not
control or reset the live sunshine/workshop sessions.

## Can first-visit values help when the sun returns?

The [retention viewer](http://127.0.0.1:43117/retention.html) compares three paths
through A-to-B-to-A: keep learning, freeze the latest values, or restore and
freeze values saved on the first visit. All share the first 120 seconds.
Saved values improved the first 20 seconds of return reward in all six known
development cases; within-body median paired gains were +1.0812 and +1.0844.
All 18 native trials and independent replays passed, as did all 15 native suites.

The experiment selects the appropriate checkpoint; autonomous memory retrieval
is still future work. The [locked protocol](docs/LIGHT_RETENTION_EXPERIMENT.md)
and [results and next engine decision](docs/LIGHT_RETENTION_RESULTS.md) are
separate records. The live creature is preserved and the viewer is read-only.

```sh
./build/native/droid-light-retention --output artifacts/reports/NEW_DIRECTORY
node tools/summarize-light-retention.mjs artifacts/reports/NEW_DIRECTORY/report.json
node tools/export-light-retention-view.mjs artifacts/reports/NEW_DIRECTORY/report.json
```

## Can its senses distinguish the two light conditions?

The [context viewer](http://127.0.0.1:43117/context.html) compares current readings
with one second of local sensor/action history under identical random motor
probes. Entire action sequences were withheld from training. History reached
93.75% and 95.45% accuracy on the two development bodies, compared with 73.86%
and 72.16% for current readings. History improved every withheld probe; removing
light gave exactly 50%. All 48 native trials and complete replays passed, and
all 16 native suites passed.

This adds a frozen native decoder and a supervised information benchmark.
Its narrow neighbor votes are not calibrated confidence, and it does not yet
select a memory during control. The [locked protocol](docs/LIGHT_CONTEXT_EXPERIMENT.md)
and [results and next engine decision](docs/LIGHT_CONTEXT_RESULTS.md) explain
why selection among supplied memories is now worth testing, including controls
for the different movements a remembered policy will produce.

```sh
./build/native/droid-light-context --output artifacts/reports/NEW_DIRECTORY
node tools/summarize-light-context.mjs artifacts/reports/NEW_DIRECTORY/report.json
node tools/export-light-context-view.mjs artifacts/reports/NEW_DIRECTORY/report.json
```

The runner refuses an existing directory. The report archives 78 source/provenance
files; the independently validated viewer is read-only. Both live creatures,
earlier evidence and the original frozen-v3 challenge remain preserved.

After the owner's computer/Docker restart, the
[normal context viewer](http://127.0.0.1:43117/context.html) is available again.
The same container is healthy and retained evidence still verifies. The temporary
preview is no longer needed; the results record preserves the availability history.

## The mathematical controller

[The implemented control model](docs/CONTROL_MODEL.md) defines the plant state,
measurement functions, history features, motor control law, reward and Expected
SARSA updates. It distinguishes the existing model-free learner from the
separate sampled prediction model; a physical observer and model-based feedback
design remain future work.

## Can an explicit state predict the next motion?

The [prediction viewer](http://127.0.0.1:43117/prediction.html) shows the actual
A/B/C/D/c matrices and recursive sensor forecasts. The
[findings index](http://127.0.0.1:43117/discoveries.html) connects all earlier
experiments. The light page links to both.

The [locked identification protocol](docs/LIGHT_PREDICTION_EXPERIMENT.md) compares
12-dimensional current-output, 142-dimensional output/request-history, and
132-dimensional output-history states, plus persistence. Each body is fitted
separately, including one with a longer beam. The future input is the known
requested-effort tape; future actual governed efforts are predicted outputs.
Entire command sequences are withheld from fitting. These matrices predict
consequences; the existing RL controller still selects live actions.

```sh
./build/native/droid-light-prediction --output artifacts/reports/NEW_DIRECTORY
node tools/summarize-light-prediction.mjs artifacts/reports/NEW_DIRECTORY/report.json
node tools/export-light-prediction-view.mjs artifacts/reports/NEW_DIRECTORY/report.json
```

The runner refuses an existing output directory. The independent checker
reconstructs the inputs and sensor histories from full traces, refits the
declared regression, and verifies every recursive forecast and error metric.

The [completed results](docs/LIGHT_PREDICTION_RESULTS.md) contain 72 native trials,
72 exact replays and nine frozen models. Both input-using models passed the
prospective prediction gate on all three bodies. History reduced one-second
primary error by 47.2%, 55.1% and 63.6% versus persistence. The 12-state model also
passed, so longer history is not established as necessary. Light prediction
improved less than the composite motion/light score; better closed-loop control
remains to be tested. All 17 native suites and 13 evidence-integrity checks passed.
