# Droid Blocks agent environment v0

This contract turns the visual MuJoCo exhibit into a deterministic learning
environment without making the exhibit itself the learning API. The native C++
environment is authoritative; HTTP is a slower inspection and evaluation
adapter over the same boundary.

## Modes

- **Demo** runs the privileged cruise controller in real time so the browser
  always has a moving reference mechanism.
- **Evaluation** pauses the real-time worker and applies externally supplied
  motor efforts to the same visible mechanism one control transition at a
  time.
- **Training** uses worker-free environment instances and never sleeps for wall
  clock time. The native trainer can evaluate independent population candidates
  concurrently, while each environment instance remains a simple synchronous
  boundary rather than hiding vectorization inside the environment contract.

Sending `play` from the browser restores demo mode. An agent reset switches its
environment instance to external control.

## Reset and step

The v0 boundary is logically:

```text
reset(assembly_id, seed, record, episode_profile) -> observation, info
step({motor_instance_id: effort}, control_dt_s)
  -> observation, reward, reward_components, safety,
     terminated, truncated, info
```

Only `demo_rover_v0` is compiled in this milestone. Keeping the assembly ID in
the boundary is intentional: a later assembly compiler can replace this fixed
case without changing the learning loop.

The action map must contain each of `motor-0001` through `motor-0004` exactly
once and no unknown IDs. Each value is a finite normalized winding-current
request. The motor driver remains responsible for clamping and safety.

The nominal control period is 0.02 seconds. MuJoCo advances at 0.002 seconds,
so one control transition normally contains ten physics transitions.
`control_dt_s` must be a positive integer multiple of the physics timestep.
Reward returned for a control transition is the sum of every constituent
physics-transition reward, not merely the final reward rate multiplied by the
control period.

## Episode profiles

The profile is an explicit, versioned reset input and is repeated in reset,
step, trace, and evaluator provenance. There are two profiles:

- `fixed_demo_v0` is the default and keeps the light source at `x = +16 m`
  for every seed. It preserves the visual demo and legacy trajectory.
- `light_search_1d_v1` is the learning task. It maps the unsigned 64-bit
  environment seed to a hidden side and distance. First compute
  `mixed = splitmix64_finalizer(seed XOR 0x4c49474854563100)`. Bit 63 selects
  the side (`0` is negative x, `1` is positive x); the low 24 bits are
  `bucket = mixed & 0x00ffffff`. The absolute distance is exactly
  `12 + bucket * 2^-21` metres, a 24-bit grid over `[12, 20)`, and
  `source_x_m = (side_bit ? +1 : -1) * distance`. The compact realization key
  is `(side_bit << 24) | bucket`.

The search realization affects the ambient-light sensor but is not policy
visible. Neither the environment seed, realization key, side, bucket, source
position, nor distance appears in the policy observation. They may appear in
evaluation output after actions have been selected so the experiment can be
audited. Training and official held-out evaluation use `light_search_1d_v1`;
the visual demo and the original baseline CLI retain `fixed_demo_v0`.

## Policy observation

The policy-safe observation contains a variable-length collection of mounted
reward-sensor samples and a variable-length collection of actuator-feedback
samples. Entries carry opaque module IDs and versioned family/SKU IDs so order
is not semantic.

The current body exposes:

- `sensor-0001`: ambient illuminance and saturation.
- `sensor-0002` and `sensor-0003`: contact and normal force.
- `motor-0001` through `motor-0004`: encoder position and velocity, winding
  current, bus voltage, temperature, estimated torque and load impedance,
  stuck state, and fault flags.

Actuator feedback remains observation and safety data and has no reward field.
Reward components are returned beside the observation so their provenance can
be audited.

The following evaluation data is forbidden from the policy observation:

- MuJoCo world position or quaternion.
- Chassis speed, travelled distance, or global contact count.
- Light-source position or distance.
- Environment seed or episode-profile realization key.
- Authored front, rear, left, right, wheel, chassis, or task roles.
- The privileged demo-controller target or direction.

The browser may continue to receive those fields from `/api/state`; that route
is a visualization and diagnostics surface, not a policy input.

The generic native evaluator makes the inference boundary structural. A
policy callback receives only the current policy-safe observation, opaque
motor IDs, the zero-based control-step index, and an independently
domain-separated policy seed. It cannot receive reset or step `info`, the
environment seed, reward, reward components, safety results, lifecycle state,
or visualization state. Metrics are collected by the evaluator after the
callback returns.

## Reward and termination

For each physics transition:

```text
reward = physics_dt_s * sum(each registered sensor instance's reward_rate)
```

The control-transition reward is the sum across all physics transitions in the
step. It is neither averaged nor clipped centrally. Only the three mounted
sensor instances vote. Hard safety is evaluated separately; a hard veto marks
the episode terminated and cannot be offset by positive sensor reward.
`truncated` is reserved for an external episode horizon and is false in v0.

## Deterministic traces

Recording is optional. A versioned trace stores the assembly ID, seed, reset
result, ordered action maps, control intervals, and returned transitions.
Replay constructs a fresh worker-free environment and requires exact JSON
equality at every reset and step. A mismatch identifies its first transition
rather than silently accepting a tolerance.

The current schema identifier is `droid-blocks.environment-trace.v2`; it adds
the episode profile to top-level, reset, and transition provenance. The reader
accepts a legacy v1 trace as `fixed_demo_v0`, while every newly written trace
is v2. Saving a
trace is non-destructive: the destination directory must already exist, the
destination filename must not exist, and installation from a temporary sibling
is exclusive so a concurrent writer cannot be overwritten. Recording is kept
out of benchmarks and normal training unless explicitly requested, avoiding
the memory and serialization cost of retaining every transition.

The seed has no realization effect in `fixed_demo_v0`; it deterministically
selects the hidden light realization in `light_search_1d_v1`. Environment and
policy random streams remain separately domain-tagged so adding policy draws
cannot perturb the world and adding world randomization cannot perturb policy
actions.

## HTTP evaluation adapter

The local service exposes:

- `GET /api/agent/spec`
- `POST /api/agent/reset`, accepting only optional `assembly_id`, unsigned
  `seed`, Boolean `record`, and string `episode_profile` fields. Defaults are
  `demo_rover_v0`, `0`, `false`, and `fixed_demo_v0`.
- `POST /api/agent/step`, accepting only an `actions` object and optional
  `control_dt_s` (default `0.02`). The action object must contain all four motor
  IDs exactly once.
- `GET /api/agent/trace`, available only after a reset with `record: true`.

These routes are for inspecting the contract and showing evaluation transitions
in the browser. High-throughput learning should link the C++ environment
directly and avoid HTTP and JSON transport overhead where possible.

Malformed values, unknown fields, invalid action sets, and invalid control
intervals return HTTP 400. Stepping before reset, stepping after an episode has
ended, or requesting a disabled trace returns HTTP 409.

## Native CLI

The `droid-agent` executable exercises the same policy-safe boundary without
the HTTP server:

```text
droid-agent benchmark [--model PATH] [--steps N] [--seed N]
                      [--control-dt SECONDS]
droid-agent record --output PATH [--model PATH] [--steps N] [--seed N]
                   [--control-dt SECONDS]
droid-agent replay --input PATH [--model PATH]
droid-agent evaluate [--policy zero|random|both] [--episodes N]
                     [--steps N] [--seed N] [--control-dt SECONDS]
                     [--random-effort MAGNITUDE] [--hold-steps N]
                     [--model PATH]
droid-agent train --output-policy PATH [--model PATH]
                  [--generations N] [--population-size N]
                  [--elite-count N] [--train-batch-size N]
                  [--initial-stddev X] [--minimum-stddev X]
                  [--update-rate X] [--optimizer-seed N] [--workers N]
droid-agent validate-policy --input-policy PATH [--model PATH]
droid-agent evaluate-policy --input-policy PATH --output-report PATH
                            [--model PATH]
```

`benchmark` and `record` drive a deterministic, public-observation-independent
action sequence. `benchmark` disables trace recording and reports requested and
executed environment transitions separately, because a safety termination may
end an episode early. `record` refuses to replace an existing path. `replay`
returns success only when reset and every recorded transition match exactly.
`evaluate` runs the versioned zero-action and seeded sample-and-hold random
controls through only `spec`, `reset`, and `step`, then reports deterministic
episode and aggregate reward, safety, actuator, health, and lifecycle metrics.
Its wall-clock and memory measurements are kept in a separate performance
section. See `BASELINE_EVALUATION.md` for the exact policy and seed contract.

`train` uses the compiled `light_search_1d_v1` train and validation splits and
writes a versioned, self-checksummed frozen policy without replacing an
existing file. Optimizer flags create explicitly identified exploratory
manifests; the official defaults come from `config/learning_experiment_v3.json`.
`--workers` changes only concurrent candidate execution, defaults to one, and
is recorded as execution metadata rather than protocol or artifact identity.
The 264-byte `shared_linear_memory_v2` payload has 33 binary64 parameters, but
`cem_masked_best_sample_v1` searches only physical indices `[29, 30, 32]` with
scales `[2.0, 0.25, 64.0]`; all other parameters are positive zero. Its latent
mean starts at `0.0`, its standard deviation starts at `1.0` with floor `0.1`,
and deterministic sampler draws are keyed by physical parameter index.
Candidates use a paired-zero, side-robust train-batch objective. The best
train-ranked sampled controller from each generation—not the updated
distribution mean—is evaluated under the locked full-validation rule and may
become the frozen checkpoint.

`validate-policy` reconstructs and runs only the exact validation split stored
in a compatible frozen artifact. It compares learned and zero control, reports
paired overall and per-side gains and wins, declares itself diagnostic and
non-official, emits no acceptance field, and never executes the test split.
`evaluate-policy` requires `--output-report PATH`, then reloads and revalidates
a frozen artifact against the current model, catalog, environment
specification, feature schema, motor topology, and control interval before
running the protocol-fixed held-out learned-versus-zero-versus-random
experiment. Its required report is installed exclusively and cannot
overwrite an existing destination. That local path guarantee is not a global
one-shot lock: another path or workspace copy can execute the split again.
The protocol therefore relies on scientific governance. Until the v3 decision
is fixed, its preregistered split and outcomes cannot influence the v3 policy,
checkpoint, acceptance rule, or claim. Afterward, those evaluated worlds,
topologies, and outcomes may be explicitly relabeled for curriculum, replay,
diagnostics, or training in a future protocol version. A descendant that uses
them must treat them as known development data and establish a fresh, versioned
challenge split for its own unseen evaluation. The
learning commands do not accept `--steps`, `--seed`, or `--control-dt`, because
those values are fixed by the protocol. See
[`LEARNING_EXPERIMENT.md`](LEARNING_EXPERIMENT.md) for the complete experiment
and acceptance rules.

The official v3 **held-out decision** is pending; the pre-held-out policy is
already frozen. The official artifact,
[`light-search-policy-v3-seed0.json`](../artifacts/light-search-policy-v3-seed0.json),
selected generation 6 under the exact v3 manifest. A separate
`validate-policy` reload gained `+0.09730391078745496 reward/s` over paired
zero control with `16/16` validation wins and zero terminations, safety vetoes,
or invalid samples. That diagnostic explicitly reported
`test_rollouts_executed: false`, so it is validation-only evidence and not an
acceptance result. The v1, v2, and reduced-budget v3 artifacts whose names
begin with `exploratory-` remain ineligible for the official held-out protocol;
their design evidence is separated by split in
[`LEARNING_DESIGN_HISTORY.md`](LEARNING_DESIGN_HISTORY.md).

## Performance gate

The native agent CLI measures environment transitions per second, physics
transitions per second, simulated seconds per wall-clock second, peak resident
memory, and a reward checksum with recording disabled. This benchmark is a
baseline for later vectorization and is not an RL quality score.

The current local baseline ran 20,000 environment transitions, comprising
200,000 physics transitions, in 5.1003 seconds. That is 3,921 environment
transitions/s, 39,214 physics transitions/s, and 78.43 simulated seconds per
wall-clock second. Peak resident memory was 11,669,504 bytes. These figures are
an observed container run, not a portable performance guarantee.

## Reproducibility boundary

Seed derivation, policy parameter encoding, canonical JSON hashes, and rollout
reduction order are specified and tested. Exact rollout equality is still a
same-runtime claim: floating-point math, MuJoCo, compiler, CPU architecture,
and linked-library changes may alter trajectories. The frozen-policy
compatibility check fails closed on the model bytes, canonical module catalog,
canonical environment specification, feature schema, action topology, API
version, and control interval, but it is not a proof that arbitrary platforms
are bit-identical. Cross-runtime results require a new validation, not an
assumption of portability.

## Build targets

The CMake graph separates `droid_core` (environment, catalog, and simulation)
from `droid_agent_core` (generic evaluation, baselines, policy, and learner).
It builds the server and agent plus contract, simulation, environment,
baseline, policy, and training test executables. The
`droid-native-validation` aggregate target is the single executable inventory
used by both Docker validation and `run.sh`; CTest then runs all six registered
native suites before the development server starts.
