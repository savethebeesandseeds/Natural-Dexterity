# Native light-search learning experiment v3

This document preregisters the first learned-control milestone for Droid
Blocks. It describes an intentionally small C++ controller and a fixed
train/validation/test experiment. It is a learnability test for the current
simulated rover, not a claim that the final modular-robot problem is solved and
not yet a sample-efficiency benchmark for physical hardware.

The authoritative machine-readable protocol is
[`config/learning_experiment_v3.json`](../config/learning_experiment_v3.json).
The native executable constructs the same document and the test suite requires
exact JSON equality. Its canonical compact JSON SHA-256 is embedded in a frozen
policy artifact.

## Question

Can one small, topology-respecting policy use only mounted sensor observations
and actuator feedback to move `demo_rover_v0` toward a light whose side and
distance are hidden, while remaining safe and outperforming passive and
matched random controls on untouched realizations?

This milestone measures generalization only over the declared
`light_search_1d_v1` distribution. It does not establish transfer to other
assemblies, sensor families, dynamics, materials, manufacturing tolerances, or
real hardware.

## Environment and task

Every episode uses:

| Field | Locked value |
| --- | --- |
| Assembly | `demo_rover_v0` |
| Episode profile | `light_search_1d_v1` |
| Control interval | `0.02 s` |
| Maximum control steps | `1,000` |
| Maximum simulated duration | `20 s` |
| Trace recording | Disabled during training and evaluation |

`fixed_demo_v0` remains the default environment profile for the visual demo
and historical baseline. It always places the light at `x = +16 m`; its seed
does not change the world. It is deliberately not the learning distribution.

For `light_search_1d_v1`, the environment converts its unsigned 64-bit seed to
one exact hidden realization:

```text
mixed      = splitmix64_finalizer(seed XOR 0x4c49474854563100)
side_bit   = mixed >> 63
bucket     = mixed & 0x00ffffff
distance_m = 12 + bucket * 2^-21
source_x_m = (side_bit ? +1 : -1) * distance_m
key        = (side_bit << 24) | bucket
```

Side bit `0` means negative x and `1` means positive x. The absolute distance
lies on a 24-bit binary grid over `[12, 20)` metres. The realization key is 25
bits and identifies both side and distance bucket. The ambient-light sensor
uses the physical source position; the policy never receives the seed, side,
bucket, key, source position, or distance.

Reward remains the environment contract, not a learner invention. At each
physics transition it is `physics_dt_s * sum(reward_rate)` over the mounted
reward-sensor instances, and control-step reward is the sum of those physics
rewards. Actuator feedback never votes on reward. No learned-policy bonus,
distance shaping, terminal bonus, discount, or second timestep integration is
added.

## Policy-safe inference boundary

The generic evaluator calls a policy with exactly four values:

1. the current policy-safe observation;
2. the current opaque motor IDs;
3. the zero-based control-step index; and
4. an independently domain-separated policy seed.

The callback type cannot carry reset or step `info`, environment seed, reward,
reward components, safety results, termination state, visualization state, or
MuJoCo internals. The learned controller does not consume the policy seed.
Environment and policy streams are nevertheless derived in separate domains
so the random controls cannot perturb the world schedule.

The observation contains only variable-length registered reward-sensor samples
and actuator-feedback samples. Collection order and opaque instance IDs do not
encode chassis position, left/right, front/rear, wheel role, or task role. The
evaluator collects reward, safety, lifecycle, actuator, and realization metrics
after action selection; their presence in the report is not a feedback channel.

### Leakage controls

| Potential leak | Control |
| --- | --- |
| Hidden side or distance | Absent from the observation and callback; visible only in post-action evaluation provenance |
| Environment seed | Owned by the evaluator/environment and absent from the callback |
| Reward or reward components | Read only by the evaluator after the policy returns its action |
| Safety or termination | Used by runner lifecycle and metrics, never as an action input |
| World pose or privileged demo controller | Available only on the visualization surface, which the evaluator does not call |
| Authored part roles | Motor and module IDs are opaque; IDs route samples and actions but are not numeric policy features |
| Test-set tuning | Test seeds are derived and hashed when the artifact is frozen but are not evaluated during training or validation |

## Controller

The frozen architecture is `shared_linear_memory_v2`, with
`module_set_encoder_v2` and `tanh_v1`. One vector of 33 binary64 parameters is
shared across all motors. It is not four separately fitted wheel controllers.

For each motor, the 33 features comprise:

- nine order-invariant ambient-light and touch summaries, including explicit
  instance-count and validity masks;
- three one-step changes in light/contact summaries;
- six pooled actuator-feedback summaries;
- eleven local actuator fields and flags;
- the motor's previous commanded effort;
- a bias;
- scaled ambient-light delta times previous local effort; and
- scaled ambient-light delta times local shaft velocity.

Numeric collection summaries are value-sorted compensated means, and module
IDs are never features. Missing optional sensors map to zeros with explicit
count and validity features. The complete ordered and normalized feature
definition is executable, versioned, and hashed by
`policy_feature_schema_sha256()`.

For each motor, inference computes the shared dot product, applies `tanh`,
scales to the locked effort limit `0.6`, limits change to `0.1` per control
step, then clamps again to the effort envelope. If no reward sensor is valid,
or the local actuator feedback is missing, invalid, or faulted, that motor's
effort is exactly zero. Sensor-delta and previous-effort memory is cleared at
every episode's first callback.

The ambient-light delta is multiplied by the exact power-of-two scale `8192`
before clipping. This makes the small one-control-step sensor change usable
without exposing privileged direction. The two interaction features let the
shared controller correlate whether motion improved the light reading with its
own recent effort and measured velocity.

The parameter payload is only `33 * 8 = 264` bytes before artifact metadata.
That makes inference plausibly edge-oriented, but it is not an edge deployment
measurement: target latency, jitter, working memory, numeric precision, power,
thermal behavior, and safe fallback still require hardware-specific tests.

## Seed splits

The split derivation is `balanced_light_side_splitmix64_v1` with base seed `0`
and separate train, validation, and test domain tags. It rejects duplicate
64-bit seeds and duplicate 25-bit realization keys across all splits. Each
split contains an equal number of negative- and positive-x scenarios, stored
as deterministic negative/positive pairs.

| Split | Pairs | Scenarios | Use |
| --- | ---: | ---: | --- |
| Train | 16 | 32 | Candidate fitting only |
| Validation | 8 | 16 | Initial checkpoint and generation selection |
| Test | 32 | 64 | Reserved for the intended single preregistered held-out execution |

The artifact stores the base seed, counts, and ordered SHA-256 digest of each
split. The official evaluator derives the splits again and fails closed unless
the artifact and the complete test set agree. Reordering supplied test seeds
cannot alter reduction order; after exact-set validation the evaluator restores
the canonical order.

## Training protocol

The optimizer is the deterministic masked cross-entropy method
`cem_masked_best_sample_v1`. It searches three latent coordinates and maps
them into the otherwise 33-dimensional physical policy:

| Latent coordinate | Physical index | Feature | Physical scale |
| ---: | ---: | --- | ---: |
| 0 | 29 | Previous local effort | `2.0` |
| 1 | 30 | Bias | `0.25` |
| 2 | 32 | Scaled ambient delta × local velocity | `64.0` |

Each physical value is its scale times the latent value. Every inactive
parameter, including index 31, has the positive-zero bit pattern. The mask and
mapping are the registered `ambient_velocity_extremum_3d_v1` search space and
are part of protocol identity, not runtime tuning knobs.

The official optimizer values are:

| Field | Official value |
| --- | ---: |
| Generations | 40 |
| Population per generation | 32 |
| Elite count | 8 |
| Train batch size | 8 scenarios |
| Initial latent mean | 0.0 |
| Initial latent standard deviation | 1.0 |
| Minimum latent standard deviation | 0.1 |
| Update rate | 0.25 |
| Optimizer seed | 0 |

The 32-scenario train split is partitioned into four fixed balanced batches of
eight, and generations cycle through those batches. Every candidate in a
generation sees the same batch, providing common random numbers. Candidate
noise uses the specified SplitMix64-based `splitmix64_irwin_hall_12_v1`
sampler rather than a standard-library distribution. A draw is keyed by its
physical parameter index (`29`, `30`, or `32`), not by its latent array
position, so a coordinate's deterministic stream remains stable and auditable.

The zero-parameter controller is evaluated on the validation split before
generation one. Within each generation, candidates are ranked on that
generation's balanced train batch by the paired-zero, side-robust objective.
Candidate and checkpoint scores are ordered lexicographically by:

1. fewer invalid samples;
2. fewer terminated episodes;
3. fewer safety vetoes;
4. greater worst-side mean reward-rate gain versus the paired zero controller;
5. greater fixed-half-per-side balanced mean reward-rate gain versus zero; then
6. greater mean return.

The best train-ranked sampled candidate is copied before the elite distribution
update, then that exact sample is evaluated on all 16 validation scenarios.
The updated latent distribution mean is recorded for diagnostics but is never
a validation checkpoint. Across generation zero and sampled checkpoints, the
frozen policy is the best protocol-fixed validation score, with the earliest
generation retained on an exact tie. Validation scores do not update the search
distribution or future samples. Training never evaluates the test split.

The CLI exposes optimizer overrides for development. An override produces and
hashes a complete effective manifest distinct from the official manifest. Such
an artifact is exploratory and the official `evaluate-policy` command rejects
it; changing an optimizer field cannot silently claim the preregistered hash.

Candidate evaluation is serial by default. `--workers N` may execute population
candidate rollouts concurrently after all candidate parameter vectors have
been generated. Initialization, validation checkpoints, sorting, updates, and
reductions remain serial and stable. Worker count is recorded under execution
metadata and is deliberately absent from the artifact and protocol-manifest
hash: it changes scheduling, not the declared experiment. The planned 24-thread
host run may use `--workers 16` without becoming a different protocol.

## Frozen artifact and compatibility

`droid-blocks.policy-artifact.v1` stores:

- architecture IDs, effort/slew limits, and all 33 parameters using exact
  `ieee754-binary64-hex-v1` encoding;
- optimizer and selected-generation provenance;
- scenario profile, horizon, split derivation, base seed, counts, and split
  digests;
- the effective protocol-manifest digest; and
- exact compatibility hashes for the model file bytes, canonical module
  catalog, canonical environment specification, and policy feature schema,
  plus environment API version, assembly, motor topology, and control interval.

Loading is strict: missing or unknown fields, malformed binary64 values,
checksum mismatch, topology mismatch, timing mismatch, or content-hash mismatch
fails before inference. Saving is exclusive through a same-directory temporary
file and refuses to overwrite an existing destination.

The artifact's `self_sha256` is a canonical-document integrity checksum. It can
detect accidental corruption and post-freeze mutation, but it is **not
authentication**: it is not a digital signature, does not establish who
trained the policy, and can be recomputed by an attacker who can replace the
artifact. Authentic deployment will require a trusted signature or equivalent
release mechanism outside this milestone.

## CLI procedure

Train with the official manifest:

```sh
./build/native/droid-agent train \
  --model models/droid.xml \
  --output-policy build/light-search-policy-v3.json
```

The artifact destination's parent directory must already exist and the file
must not exist. Train-only optimizer and execution options are:

```text
--generations N       --population-size N  --elite-count N
--train-batch-size N  --initial-stddev X   --minimum-stddev X
--update-rate X       --optimizer-seed N   --workers N
```

`--workers` must be positive and is execution-only; unlike the optimizer
options beside it, it does not alter the effective manifest digest.

The command prints machine-readable JSON containing the official and effective
manifests and digests, effective optimizer values, split digests, runtime
compatibility, artifact metadata, deterministic history, and a separate
nondeterministic performance section.

Before touching the held-out split, diagnose a frozen artifact on exactly its
reconstructed validation split with:

```sh
./build/native/droid-agent validate-policy \
  --model models/droid.xml \
  --input-policy build/light-search-policy-v3.json
```

`validate-policy` reloads and compatibility-checks the artifact, then evaluates
the learned and zero policies on validation only. It reports paired overall and
per-side deltas and wins, emits no acceptance Boolean, explicitly identifies
itself as non-official, and executes no test rollout.

The preregistered procedure intends one evaluation of an eligible frozen
artifact on the official held-out set:

```sh
./build/native/droid-agent evaluate-policy \
  --model models/droid.xml \
  --input-policy build/light-search-policy-v3.json \
  --output-report build/light-search-heldout-v3.json
```

`--output-report PATH` is required. The report is installed exclusively and an
existing destination is never overwritten; the full versioned report is also
present in JSON stdout. `train`, `validate-policy`, and `evaluate-policy`
reject `--steps`, `--seed`, and `--control-dt`; these are protocol fields, not
experimenter choices. `evaluate-policy` rejects artifacts whose official
manifest, exact optimizer values, profile, horizon, split base/counts/digests, effort/slew
envelope, or runtime compatibility differs. A valid completed evaluation exits
zero even when its scientific decision is `accepted: false`; operational,
schema, integrity, provenance, and compatibility failures are nonzero.

Exclusive installation protects only the specified report path. The software
does not and cannot enforce global single execution: another filename, copied
workspace, or equivalent runtime can execute the same split again. The
one-evaluation rule is scientific governance, not a cryptographic or technical
one-shot mechanism. Until the v3 decision and report are fixed, the v3 split or
outcomes cannot feed back into the v3 artifact, optimizer, checkpoint,
acceptance rule, or claim. Re-execution may diagnose operational reproduction
only after the scientific decision is fixed; it cannot create a newly selected
v3 result.

This restriction is per release, not a ban on transfer learning. Once the v3
decision is fixed, its evaluated worlds, topologies, traces, and outcomes may be
explicitly relabeled as known data and promoted to curriculum, replay,
diagnostics, or training for a future protocol version. Any descendant that
uses or inspects them must treat them as development data, never as unseen
generalization evidence. That descendant requires a fresh, preregistered,
versioned challenge split that remains unavailable to its own training,
selection, and design loop until its decision is fixed.

## Held-out controls

Every one of the 64 canonical test scenarios uses the same environment seed
for the learned, zero, and random controllers.

- `zero_v0` requests exactly zero effort for every motor.
- `seeded_random_sample_hold_slew_v1` is evaluated in four independently
  domain-separated policy streams per scenario. Each stream has effort limit
  `0.6`, holds a target for five control steps, and applies the same `0.1`
  per-step slew limit as the learned policy. Its fixed base seed is
  `5927104639690624049` (`0x52414e4443544c31`). It uses no observation.

The `slew_v1` control is distinct from the historical unslewed
`seeded_random_sample_hold_v0` integration baseline. Matching the learned
policy's effort and slew envelope is part of this experiment's fairness rule.

The four random returns and reward rates are arithmetically averaged within a
scenario before any learned-versus-random comparison. The environment scenario,
not an individual random stream or control step, is the statistical unit.

## Exact acceptance rule

Acceptance is one conjunction. All of the following must be true:

1. The learned policy has zero environment terminations, zero safety vetoes,
   and zero invalid sensor or actuator-feedback samples across all 64 scenarios.
2. The paired, side-stratified bootstrap lower bound for learned-minus-zero
   mean reward rate is strictly greater than zero.
3. The corresponding lower bound for learned-minus-random-mean reward rate is
   strictly greater than zero.
4. The overall mean reward-rate gain is at least `0.01` against zero and at
   least `0.01` against random mean.
5. Learned reward rate wins in at least 48 of 64 scenarios against zero and in
   at least 48 of 64 against random mean.
6. Within negative-x scenarios and within positive-x scenarios separately, the
   mean reward-rate delta is strictly positive against both controls.
7. Within each 32-scenario side, learned wins at least 24 scenarios against
   zero and at least 24 against random mean.

The bootstrap is `paired_side_stratified_percentile_v1`: it resamples the 32
negative and 32 positive paired deltas independently, computes each stratum
mean, then gives each side fixed weight `0.5`. It uses 10,000 deterministic
resamples and seed `5786927923740624214` (`0x504f4c4943594556`). Each of the two
primary comparisons uses a 97.5% one-sided lower bound (the 2.5th percentile).
Bonferroni allocation of one-sided alpha `0.025` to each comparison provides the
declared 95% joint familywise confidence. Return metrics remain in the report
for diagnosis, but the statistical, practical, win, and per-side gates use
mean reward rate so early termination cannot look beneficial merely by reducing
exposure.

## Determinism and interpretation limits

Integer seed schedules, candidate sampling, parameter encoding, JSON hashing,
episode order, stable reductions, and bootstrap draws are specified. Repeating
an operation with the same validated runtime is expected to reproduce all
non-performance output exactly. Wall time, throughput, and peak RSS are kept in
separate performance sections and are never hashed into the scientific result.

This is not a cross-platform bitwise guarantee. MuJoCo trajectories and C++
floating-point evaluation may vary with CPU architecture, compiler, math
library, MuJoCo build, or other runtime details. Compatibility hashes prevent a
known artifact from silently running against changed model/catalog/spec/feature
inputs, but they do not prove that two arbitrary binaries execute floating
point identically. Results from a different runtime require fresh validation
and must not be merged with the official report as if they were the same run.

The held-out decision is descriptive of this one preregistered simulation
distribution. Passing does not demonstrate causality of a particular sensor,
optimality, robustness to assembly changes, sample efficiency, hardware safety,
or sim-to-real transfer. Failing is also useful: the report preserves paired
scenario, side, reward, safety, actuator, and lifecycle evidence for the next
design iteration.

## Results

**Official v3 held-out decision pending; the test split remains untouched.**

### Official frozen pre-held-out artifact

The official training run produced
[`artifacts/light-search-policy-v3-seed0.json`](../artifacts/light-search-policy-v3-seed0.json)
and selected generation 6.

| Identity | SHA-256 |
| --- | --- |
| Artifact canonical-payload `self_sha256` | `4dd4d6ccadecb0b3b69d3b3d412af83eff15208b19058701538dbdf83a4c32b5` |
| Exact serialized file bytes | `6f7d5069bdb884d75f1241629d437c039a2f7a486633d9ef2d7f6b70db0a5582` |
| Official protocol manifest | `f06d689351ddf10c6e987661d7a1f935df4cd8a3c911a76dd13395a6d4b49e93` |

These hashes have different domains. `self_sha256` is stored inside the
artifact and hashes its compact canonical JSON payload before the self field is
added. The file hash covers the exact bytes on disk, including serialization
formatting and the stored self field. Neither value substitutes for the
protocol-manifest hash, which identifies the canonical experiment definition.
All are integrity/provenance checks, not signatures.

The official training command took `715.225449445 s` wall time and reached peak
RSS `30,363,648` bytes. These are execution-resource measurements and are not
part of policy scoring or artifact identity.

### Independent validation-only reload

The frozen artifact was independently reloaded through `validate-policy`. The
following figures are from all 16 validation scenarios against their paired
zero controls; they are not train metrics and are not held-out metrics.

| Validation stratum | Mean reward-rate gain vs zero | Wins |
| --- | ---: | ---: |
| Negative x | `+0.09791938469996925` | `8/8` |
| Positive x | `+0.09668843687494068` | `8/8` |
| Overall | `+0.09730391078745496` | `16/16` |

The learned validation episodes had zero environment terminations, zero safety
vetoes, zero invalid reward-sensor samples, and zero invalid actuator-feedback
samples. Aggregate actuator/action diagnostics were:

| Diagnostic | Value |
| --- | ---: |
| Mean absolute effort | `0.31318514771991324` |
| Maximum absolute effort | `0.5999999976413114` |
| Peak absolute current | `1.4999999759926383 A` |
| Peak temperature | `27.840471196252533 C` |

The diagnostic explicitly reported `test_rollouts_executed: false` and emitted
no acceptance decision. No official held-out report exists under
`artifacts/reports/`; execution of `evaluate-policy` is awaiting explicit
approval. Therefore these strong validation results do not establish held-out
success, and no acceptance claim is made.

The v1 and v2 files under `artifacts/` remain exploratory and ineligible for
the v3 held-out command. Their pre-held-out evidence is recorded in
[`LEARNING_DESIGN_HISTORY.md`](LEARNING_DESIGN_HISTORY.md). The historical
values in [`BASELINE_EVALUATION.md`](BASELINE_EVALUATION.md) are fixed-demo
non-learning integration controls and must not be copied here as learned-policy
results.

After the explicitly approved held-out execution, append the exact command,
held-out report path and hashes, runtime/image identity, acceptance Boolean,
every acceptance criterion, and separate execution-performance measurements.
Do not replace the pre-held-out record with selected scenarios or an
unregistered rerun.
