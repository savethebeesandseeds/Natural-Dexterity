# Non-learning baseline evaluation v0

This milestone establishes honest control measurements before choosing or
implementing a reinforcement-learning algorithm. It evaluates two deliberately
simple policies through the same policy-safe `DroidEnvironment` boundary that a
learned policy now uses. The baseline adapters and learned policy share the
generic native evaluator and its metrics rather than maintaining separate
rollout logic.

Neither policy learns. Neither policy adapts to a sensor, actuator-feedback
sample, reward, reward component, or previous outcome. Their results are
controls for interpreting future agents, not evidence that learning has taken
place.

## Goal

The baseline answers four narrow questions:

1. How much reward is available when the robot requests no motor effort?
2. Does bounded, unstructured actuation make the reward or safety outcomes
   measurably different from doing nothing?
3. Can repeated native evaluations be reproduced from one base seed without
   using privileged simulator state?
4. What wall time, simulated time, transition throughput, and process memory
   does this evaluation path require before any learned policy is added?

The baseline does **not** answer whether the task is learnable, whether one RL
algorithm is suitable, whether an agent generalizes, or whether a controller
will transfer to fabricated hardware.

## Policies

Both policy descriptions declare `uses_observation: false`.

### `zero` (`zero_v0`)

The CLI selector is `zero`; its versioned output policy ID is `zero_v0`. For
every control transition it emits an action map containing every motor ID
declared by `spec().action.motor_ids`, with exactly `0.0` effort for each motor.
It has no policy RNG and no internal adaptive state.

This is the passive-reward control. A nonzero return is expected when mounted
sensors can vote positively without motion; it must not be described as skill.

### `random` (`seeded_random_sample_hold_v0`)

The CLI selector is `random`; its versioned output policy ID is
`seeded_random_sample_hold_v0`. It is a seeded, piecewise-constant open-loop
controller. At the beginning of each hold interval it deterministically maps
the policy seed, hold-block index, and each opaque motor ID to an independent
effort in the half-open symmetric range
`[-random_effort, +random_effort)`. It reuses that action map for `hold_steps`
control transitions, then constructs another map. A final partial hold ends at
the episode horizon or an environment termination.

The motor ID is hashed with FNV-1a, combined with the policy seed and hold-block
index, finalized by the evaluator's specified SplitMix64-style integer mixer,
and mapped from its high 53 bits directly into the effort interval. There is no
shared sequential draw cursor: reordering the input motor IDs or adding a new
ID cannot shift an existing motor's action stream.

The default magnitude is `0.6` normalized current effort and the default hold
is five control transitions. The magnitude must be finite and within `[0, 1]`;
the hold must be a positive integer. The policy never uses an observation,
feedback, reward, safety value, or task geometry to choose an action. Stopping
after an environment terminal signal is runner lifecycle handling, not policy
feedback.

This is an action-sensitivity and naive-exploration control. If it scores above
`zero`, the result only shows that this particular seeded open-loop action
sequence received more reward. If it scores below `zero`, it may expose the
cost or danger of indiscriminate actuation. Neither result is learning.

## Seeds and paired episodes

The command accepts one unsigned 64-bit base seed. Seed derivation is stable,
tagged, and independent of execution order:

- An environment seed is derived from the base seed and zero-based episode
  index. Both policies receive the same environment seed at a given episode
  index, making their environment schedules paired.
- A distinct policy seed is derived from the base seed, the `random` policy
  namespace, and the episode index. It drives only that episode's random action
  stream and never the environment.
- The integer mixer and integer-to-effort mapping are explicitly implemented by
  the evaluator rather than delegated to `std::uniform_real_distribution`, so
  a supported compiler-library change cannot silently redefine the action
  sequence.
- Each episode record exposes its actual derived environment seed and, for
  `random`, its numeric policy seed. `zero` records `policy_seed: null` because
  it consumes no policy RNG stream.

Environment and policy streams must stay separate when environment
randomization is added. Adding random draws inside the world must not shift the
policy's actions, and adding motor modules must not change the environment seed
schedule.

The baseline CLI uses `fixed_demo_v0`, which always places the light source at
`x = +16 m`; consequently, distinct environment seeds do not create distinct
physical worlds in this historical control. The separate learning profile,
`light_search_1d_v1`, now maps its seed to a hidden signed distance and is
documented in [`AGENT_ENVIRONMENT.md`](AGENT_ENVIRONMENT.md). The paired
schedule remains part of both contracts.

## Episode lifecycle

Every episode resets `demo_rover_v0` with recording disabled, then performs at
most `steps` control transitions. The default is 1,000 transitions at a
0.02-second control interval. An environment hard-safety termination ends the
episode immediately.

The environment owns `terminated` and `truncated`. In environment v0,
`truncated` remains `false`. Reaching the command's step limit is recorded as
runner-owned `horizon_reached: true`; the runner must not rewrite that event as
an environment truncation. A safety-ended episode has `terminated: true` and
`horizon_reached: false`.

The default evaluation runs eight episodes per selected policy. Eight episodes
are useful for a deterministic integration and regression baseline, but they
are not a statistically powered experiment and must not be presented with
claims of significance.

## Reward accounting

Episode return is the plain sum of the environment's returned transition
`reward` values:

```text
episode_return = sum(step.reward)
```

There is no discounting, extra timestep multiplication, clipping,
normalization, shaping, or reward taken from actuator feedback. Each transition
reward already integrates the literal sum of mounted reward-sensor votes over
its constituent physics transitions. The evaluator must not integrate it a
second time.

Raw return can be confounded by exposure time when safety terminates an episode
early. Each episode therefore also reports:

```text
mean_reward_rate = episode_return / simulated_time_s
```

The rate must be finite; it is defined as zero only when no simulated time was
executed. The aggregate includes both the mean of episode reward rates and an
exposure-weighted rate, `total_return / total_simulated_time_s`. Raw return,
executed duration, and termination counts remain beside those rates so a rate
cannot conceal safety failures.

Return comparisons are valid only between runs using the same assembly,
control interval, horizon, seed schedule, reward contract, and software/model
revision.

## Metrics and output contract

Each selected policy produces one object with
`schema_version: "droid-blocks.baseline-evaluation.v2"`, an `environment`
provenance object, and four principal sections:

- `policy` identifies the versioned `zero_v0` or
  `seeded_random_sample_hold_v0` policy, declares `uses_observation: false`,
  and records the random magnitude and hold interval when applicable. The
  shorter `zero` and `random` names are CLI selectors, not schema policy IDs.
- `environment` records the API version, assembly, `fixed_demo_v0` episode
  profile, opaque motor IDs, and that the policy-safe boundary was used.
- `configuration` records the base seed, episode count, maximum transitions
  per episode, episode profile, control interval, physics timestep,
  random-policy parameters, and recording state.
- `episodes` records the zero-based index, derived seeds, `return`,
  `mean_reward_rate`, `control_steps`, `physics_steps`, `simulated_time_s`,
  `environment_terminated`, environment-owned `environment_truncated`, and
  runner-owned `horizon_reached`. Nested `reward`, `safety`, `health`,
  `actuators`, and `actions` metrics preserve public accounting and diagnostic
  provenance without becoming policy input.
- `aggregate` records `episode_count`, `total_return`, `mean_return`,
  `min_return`, `max_return`, return dispersion as the **population**
  `population_stddev`, `total_control_steps`, `total_physics_steps`,
  `total_simulated_time_s`, `exposure_weighted_reward_rate`,
  `mean_episode_reward_rate`, `terminated_episodes`, `truncated_episodes`, and
  `horizon_reached_episodes`. Nested cumulative reward, safety, health,
  actuator, and action metrics retain their distinct meanings.

Population standard deviation divides the sum of squared deviations by `N`,
not `N - 1`. This output describes the complete requested deterministic batch,
not a sample estimate of a larger stochastic population.

When both policies are requested, the CLI wrapper keeps evaluations in fixed
`zero`, then `random` order and adds a `comparison` containing:

- `random_minus_zero_mean_return`
- `random_minus_zero_termination_count`

Positive return delta means random received more reward; positive termination
delta means random suffered more safety terminations. These are descriptive
deltas over the paired schedule, not confidence intervals, causal estimates,
or evidence of learning.

Runtime measurements are kept in a separate `performance` object: wall time,
executed environment and physics transitions, transition throughput, simulated
seconds per wall-clock second, and peak resident set size. Wall time and peak
RSS are machine- and run-specific diagnostics and do not participate in policy
comparison. They cover the end-to-end `evaluate` command, including environment
and model construction; a `both` run constructs one worker-free environment per
policy. They are not isolated policy-inference timings.

## Policy-data boundary

The evaluator may use only:

- `spec()` for opaque motor IDs and public timing metadata;
- `reset()` to begin an episode;
- `step()` to submit actions and collect public reward, lifecycle, safety, and
  accounting data.

The policies themselves discard the reset and step observations. They do not
call the visualization surface, inspect MuJoCo state, use world pose, light
location, chassis velocity, global contacts, authored front/rear/left/right
roles, or read the privileged demo controller. They do not use sensor reward
components or actuator feedback to alter later actions. The evaluator must not
call `visualization_state()` or internal simulation APIs.

At action time the generic callback receives only the current policy-safe
observation, opaque motor IDs, zero-based control-step index, and independently
domain-separated policy seed. Reset/step `info`, environment seed, reward,
safety, lifecycle, and visualization state cannot enter that callback. The two
baseline adapters deliberately ignore the observation; the learned controller
uses it under the same type-level boundary.

Evaluation output is summary and provenance data, not a hidden observation
channel. The learned experiment continues to use only the policy-safe
observation documented in [`AGENT_ENVIRONMENT.md`](AGENT_ENVIRONMENT.md); see
[`LEARNING_EXPERIMENT.md`](LEARNING_EXPERIMENT.md) for its leakage controls.

## Native CLI

```text
droid-agent evaluate [--policy zero|random|both]
                     [--episodes N] [--steps N] [--seed N]
                     [--control-dt SECONDS]
                     [--random-effort MAGNITUDE] [--hold-steps N]
                     [--model PATH]
```

Defaults are `--policy both`, `--episodes 8`, `--steps 1000`, `--seed 0`,
`--control-dt 0.02`, `--random-effort 0.6`, and `--hold-steps 5`. The command
prints one JSON object. Its top level contains `command: "evaluate"`,
`assembly_id`, the requested `policy` selector, base `seed`, an `evaluations`
array, the two-policy `comparison` when applicable, and separate `performance`
measurements. Each evaluation is embedded unchanged and, for `both`, appears in
fixed `zero_v0`, then `seeded_random_sample_hold_v0` order.

Examples:

```sh
# The complete fixed-seed control baseline.
./build/native/droid-agent evaluate

# A short deterministic integration run.
./build/native/droid-agent evaluate --episodes 2 --steps 100 --seed 7

# Inspect the random control alone with a gentler action range.
./build/native/droid-agent evaluate --policy random --random-effort 0.25 \
  --hold-steps 10 --episodes 8 --steps 1000 --seed 7
```

Changing `--seed` changes the paired episode schedule and the random policy
stream. The command does not write traces; trace recording and exact replay
remain separate explicit operations.

## Success criteria for this milestone

The milestone succeeds when all of the following hold:

1. Both policies act through the public action map and declare
   `uses_observation: false`.
2. `zero` emits exactly zero for every declared motor, while `random` remains
   within its configured symmetric bound and holds actions for exactly the
   configured interval unless the episode ends.
3. Environment seeds are paired across policies, policy randomness is isolated,
   and repeating a run with the same inputs reproduces all non-performance
   results exactly.
4. Episode return exactly sums transition rewards; reward-rate denominators use
   actual simulated time; population dispersion, safety termination, horizon,
   and environment truncation accounting are tested.
5. Horizon completion never masquerades as environment truncation, and a safety
   termination is never hidden by return normalization.
6. No policy path reads privileged visualization or simulator data.
7. CLI validation rejects invalid policies, zero episodes or steps, invalid
   control intervals, random magnitudes outside `[0, 1]`, and zero hold length.
8. The native test suites, deterministic replay tests, container build, and a
   measured baseline run pass without Python.

No ordering between `zero` and `random` return is a success criterion. Equal,
better, worse, constant, or unexpectedly high passive reward are all useful
diagnostics. They become inputs to reward and task design, not facts to hide.

## Edge-device relevance and limits

Both controls require negligible policy compute and memory. That makes them a
clean measurement of the native environment/runner overhead, but it does not
predict the footprint of a learned model. The current MuJoCo stepping loop is
CPU-based; container GPU visibility does not accelerate this evaluation and is
not part of its scientific result.

Peak RSS is the process high-water mark for the evaluator, MuJoCo model, native
runtime, and libraries together. Before hardware deployment, a learned agent
must be measured separately for parameter storage, working memory, inference
latency and jitter at the control rate, power, thermal behavior, numerical
precision, and safe fallback behavior on the target edge device.

The historical baseline evaluation document records its schema, episode
profile, and environment API version, but it does not embed model,
module-catalog, binary, or image digests. Exact non-performance reproduction is
therefore a same-runtime guarantee. The newer frozen-policy artifact does bind
model, catalog, environment-specification, and feature-schema digests, as
documented in [`LEARNING_EXPERIMENT.md`](LEARNING_EXPERIMENT.md), but even those
integrity fields are not a cross-platform floating-point guarantee.

## Measured baseline

The validated default control run used this explicit command:

```sh
./build/native/droid-agent evaluate --policy both --episodes 8 --steps 1000 \
  --seed 0 --control-dt 0.02 --random-effort 0.6 --hold-steps 5 \
  --model models/droid.xml
```

Each episode reached the runner-owned 1,000-transition horizon: 20 simulated
seconds and 10,000 physics transitions. No episode was environment-terminated
or environment-truncated. Reward reconciliation error was zero, with no safety
vetoes and no invalid reward-sensor or actuator-feedback samples.

| Policy ID | Episodes | Mean return | Population return SD | Exposure-weighted reward rate | Terminated | Truncated | Horizon reached |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `zero_v0` | 8 | 13.0253600000 | 0.0000000000 | 0.6512680000 | 0 | 0 | 8 |
| `seeded_random_sample_hold_v0` | 8 | 13.0378242925 | 0.0577909571 | 0.6518912146 | 0 | 0 | 8 |

The descriptive random-minus-zero mean-return delta was `+0.0124642925`; the
termination-count delta was `0`. The reward was overwhelmingly the ambient
light sensor's vote, while touch contribution was numerically negligible in
this run. The small positive random-policy delta therefore shows only that this
fixed batch of open-loop actions changed ambient exposure slightly. It is not
an RL result, an estimate of statistical significance, or evidence of learned
behavior.

Across both policies, the command executed 16,000 environment transitions and
160,000 physics transitions in one observed 4.543836177-second run. That is
3,521.25 environment transitions/s, 35,212.54 physics transitions/s, and 70.43
simulated seconds per wall-clock second. Peak process RSS was 11,669,504 bytes.
These timing and memory figures are a nondeterministic end-to-end local
observation, are excluded from each deterministic evaluation document, and are
not an edge-device guarantee.

The random run reached 1.499200 A peak current, 27.714991 degrees C peak
temperature, and 0.346290 peak stuck score across its actuator feedback. It
reported no stuck samples or actuator faults. These are diagnostics for the
bounded control rollout, not policy inputs or measures of learning.

The final rebuilt validation image for the measured checkout was
`sha256:eabe48053a78bcb334a947e7a5c0c009d6ff174a3b5e57e2493593d423b91068`.
Exact deterministic metrics are guaranteed only with the same validated
artifacts; future cross-build comparisons need model and catalog content
digests plus an explicit build revision.

Both measured policies are non-learning controls and contain no trained
parameters.
