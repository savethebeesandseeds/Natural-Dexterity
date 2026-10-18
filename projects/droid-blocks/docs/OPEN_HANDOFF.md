# Droid Blocks: open project brief and handoff

**Historical brief captured:** 2026-09-05; latest completed study: 2026-10-09;
owner clarification: 2026-10-06

**Status:** active research project; this document is intentionally open to revision  
**Start here:** the simulation is normally visible at <http://127.0.0.1:43117/>

**Current product direction, 2026-10-06:** the owner clarified that the very
first child-built creature should already carry substantial simulated training
exposure and generalized competence. Offline multi-body/environment pretraining
is the starting point. Inference and little new experience specialize it to an
assembly; retaining personal experience after rebuilding adds to that base.
Scratch learning remains an experimental comparator.

The [three-seed breadth study](GENERALIZATION_STUDY_RESULTS.md) now provides a
bounded first-use result on previously uninstantiated length-5 assemblies.
All 36 pooled trials completed safely with frozen weights, exceeded quiet's
useful-behavior margin and beat their matched saved initial policies. Equal-budget
broader training improved the mean over narrow training in all three seeds,
with median seed gain +1.723678, but won only 21/36 individual comparisons.
Narrow training also cleared the quiet margin everywhere. These are three
training replicates with correlated case outcomes, not a universal breadth
guarantee. All 16 fixed-witness safety stops remain in the record.

The [earlier pooled-pretraining development pilot](GENERALIZATION_PRETRAINING_EXPERIMENT.md)
is [complete](GENERALIZATION_PRETRAINING_RESULTS.md). One seed trained on 32
body/light conditions, then controlled 12 known cases with fixed weights. All
36 trained/untrained/quiet trials completed safely; trained beat quiet's margin
in all 12 and beat its matched initial policy in 10, with mean gain +7.987401.
Two cases worsened relative to initialization, the weakest quiet-margin pass
was narrow, and the benefit is strongly lamp-side dependent. Independent native
replay verified all 37 tapes with zero failed checks. Its reserve remained
unconstructed until the successor study's six-source seal and 48-pair
competence gate passed. Length-5 outcomes are now known development data.

The [bounded specialization study](SPECIALIZATION_RESULTS.md) now measures that
next step on the twelve exposed conditions. All 294 target/retention opportunities
safely completed and independent replay passed 21,227,993 checks. With 128 learning
decisions inside each 640-decision target budget, head-only tuning's median seed
mean gain over frozen was −0.337379 and full tuning's was +1.410498. Both missed
the practical +3.2 gate. Both beat matched scratch in every whole target run;
this supports inherited pretraining competence, without demonstrating useful
incremental specialization. Both methods' sampled retention gates passed, but full
tuning lost 10.037916 in one earlier-condition probe. No adapted brain was promoted.

The [longer-budget study](CONSERVATIVE_SPECIALIZATION_RESULTS.md) now completes
that comparison: 384 learning decisions and 256 frozen continuation decisions
within each 640-decision opportunity. Conservative updates gained a median
+1.213250 over frozen, below the +3.2 usefulness threshold, and failed individual
reliability with a 5.008546 target loss and a 34.334899 earlier-condition probe
loss. Its mean retention gate passed, showing why an average is insufficient.
Standard updates had two native safety stops, failed sampled mean retention,
and lost 41.550677 in one probe. All 150 guard opportunities safely completed; 292 of the
294 total opportunities did so. Neither method met combined readiness.

Keep frozen reuse as the product default. The next research milestone is to
diagnose when the retained update checkpoints change behavior and damage
inherited competence, then declare an adaptation recipe with explicit checks
for preserving that competence and test it prospectively. Increasing the budget
and lowering learning rate and clipping together did not meet the tested requirements. Personal memory should
build on a reliable specialization process.
Before extending the first-build claim, declare a fresh reserve and a supported
body family. Persistent per-creature knowledge, richer topology/motors/contact
and physical hardware need separate evidence. The [native pretrained workshop](PRETRAINED_WORKSHOP.md) brings the
canonical pooled seed 101 actor into the visible build/wake loop, with immutable
weights, fresh local history on rebuild and preserved pending actions on pause.
This fixture remains one anchored two-link topology,
so even successful new geometry results cannot establish arbitrary modular-body
generalization. Define the supported first-build family and child responsiveness
budget before making a product claim.

**Completed question, 2026-10-04:** the owner authorized continuing work
after reviewing the failed [actuation-transfer result](ACTUATION_TRANSFER_RESULTS.md).
The [locked native headroom study](LIGHT_HEADROOM_EXPERIMENT.md) now asks whether
the existing bodies have a useful advantage over quiet control, separately from
whether the unchanged learner finds it within 64 simulated seconds. The full
24-case matrix includes historical above-body lighting and new below-body
lighting. No earlier result or frozen rover-v3 input is replaced.

The [completed result](LIGHT_HEADROOM_RESULTS.md) passes both lower-lamp
criteria: fixed witnesses clear the margin in all 12 cases, and all 36 fresh
native learner runs also pass. The 216-trial record includes 11 feedback stops
and weak historical upper-lamp outcomes. The
[new recorded comparison](http://127.0.0.1:43117/headroom.html) exposes every
case, controller and seed. The
[earlier motor-move comparison](http://127.0.0.1:43117/transfer.html) remains
available, including its failed improvement gate.

The October 4 next-step proposal compared frozen reuse, adaptation and scratch
under equal budgets on the exposed lower-light curriculum. The October 6
clarification brings broader pretraining and first-use frozen competence ahead
of that comparison. Check source competence before exposing new conditions.
Fresh-memory acquisition on the older known cases and frozen first-use geometry
generalization in the bounded kit now have evidence. Useful incremental specialization,
variable connections, multiple motors, contact, physical hardware and retained
personal experience across rebuilds remain open.

## An invitation

This is a handoff, not a finished specification. We would like the next
collaborator to understand why the current choices were made, preserve the
evidence that is already useful, and challenge anything that prevents the
larger idea from working.

The project asks a playful but serious question: what happens if people build a
body from LEGO-like modules, connect sensors and motors, and do not program its
behavior? The assembled artifact should discover what is connected, act through
the available motors, learn from the preferences declared by its own sensors,
and try to maintain and improve its situation. “Trying to be alive” is the game
and design metaphor; it is not a claim of sentience.

The first software milestone is deliberately much smaller than that idea. It
establishes a trustworthy C++ simulation and one compact learned behavior before
we attempt assembly discovery, topology transfer, continual learning, or real
hardware.

Please treat **Current commitments behind the existing evidence** as starting
guardrails, not permanent doctrine. Everything under **Open work** and
**Questions worth challenging** is an invitation to propose something better.

## The idea in its current form

An assembly is a graph of active modules and passive mechanics:

- Passive plastic blocks, axles, gears, and wheels define geometry and
  transmission but emit no observations, actions, or rewards.
- Reward sensors are optional modules. Each sensor family owns a local,
  versioned reward function. Only registered instances that are actually
  attached vote, and their reward rates are summed without a hidden central
  objective.
- A duplicate physical sensor is deliberately a duplicate vote. Missing,
  stale, saturated, and invalid samples therefore need explicit semantics.
- Motors accept normalized effort. Encoder, velocity, current, voltage,
  temperature, torque, impedance, stuck estimates, and faults are actuator
  feedback for control and safety. They do **not** vote on reward.
- Implemented simulator safety vetoes remain outside reward. Positive sensor
  votes cannot cancel an overtemperature, current, joint, timeout, or
  stale-feedback stop. Independent, hardware-qualified interlocks remain future
  work.
- The current policy receives opaque module identities, family/SKU identifiers,
  and physical observation channels, but no connector graph or local transforms.
  Supplying measurable connector relationships and transforms—without authored
  labels such as “front wheel,” “left leg,” or “chassis”—is a hypothesis for a
  future topology learner. Useful roles should still be discovered.
- Not every assembly has every sensor or motor. Absence must be ordinary input,
  not a malformed special case.

The policy chooses actions from public sensor and actuator-feedback
observations. World pose, light position, task geometry, environment seed,
authored part roles, reward values, and the privileged visualization controller
are not action inputs. The trainer may observe the returned total reward after
an action in order to learn; the action callback itself does not receive it.

The detailed current contracts are in
[`MODULE_CONTRACT.md`](MODULE_CONTRACT.md) and
[`AGENT_ENVIRONMENT.md`](AGENT_ENVIRONMENT.md).

## Working north star

One proposed framing for the larger program is:

> Build a safe, sample-efficient continual-learning system that accumulates
> and transfers competence across modular robot topologies and worlds, while
> preserving immutable, exposure-aware checkpoints and using fresh,
> preregistered challenge sets for every new generalization claim.

This is broader than producing one successful frozen controller. The durable
artifact we ultimately care about is a learning process with memory, lineage,
and safe adaptation—not merely a single policy file.

The frozen v3 experiment remains a useful immediate sub-goal: complete its
already-defined evaluation without adapting v3, retain the result whether it
passes or fails, and use it as the historical parent of later work.

## What exists today

### Runtime and visualization

- The native engine and learning stack are Python-free C++20 using MuJoCo
  3.12.0. The browser playground is HTML, CSS, and JavaScript.
- It runs in the managed Debian 12 container `droid-blocks-dev` from the bind
  mount at `/workspace`.
- The uncommon host mapping is `127.0.0.1:43117 -> 8080`.
- The browser provides a live playground, state inspection, play/pause/reset,
  and an HTTP adapter for the learning boundary.
- The visible cruise animation uses a privileged demo controller. It is useful
  for presentation but must not be mistaken for the learned policy.
- An NVIDIA GPU device was visible inside the current container during setup.
  The probe output was not retained in the repository, so reverify it before
  relying on that capability. Current MuJoCo stepping is CPU-based; an earlier
  ad hoc EGL check selected Mesa `llvmpipe`, but that probe was not retained
  either. Camera rendering is therefore not known to be GPU accelerated.
- `run.sh` configures, incrementally builds, runs the six native CTest suites,
  and starts the service. `setup.sh` only installs declared dependencies and
  does not manage container lifecycle. Downloaded SDK/header artifacts are
  versioned and hash-pinned; Debian APT package versions are not pinned.

The principal routes are:

```text
GET  /healthz
GET  /api/state
GET  /api/catalog
POST /api/control
GET  /api/agent/spec
POST /api/agent/reset
POST /api/agent/step
GET  /api/agent/trace
```

The C++ API is authoritative. HTTP and JSON are for inspection and visible
evaluation; high-throughput training links the same environment directly.

### Current body and task

Only one assembly is compiled today: `demo_rover_v0`, with four motors, one
ambient-light sensor, and two touch/force sensors. The catalog also defines
power, six-axis IMU, and time-of-flight proximity families as future targets,
but they are not mounted on this body.

The visual profile, `fixed_demo_v0`, always places the light at `x = +16 m`.
Learning uses `light_search_1d_v1`: a deterministic 64-bit seed chooses a hidden
left-or-right light position at a distance in `[12, 20) m`. The policy sees only
the physical sensor consequence, never the seed or source position.

The environment advances at a 0.002-second physics step and a nominal
0.02-second control interval. Each training rollout uses a synchronous
environment with no real-time simulation worker and runs as fast as the CPU
allows; independent candidate rollouts may execute concurrently through
`--workers`. Trace recording and exact same-runtime replay are available.

### Current learner

The frozen policy architecture is `shared_linear_memory_v2`:

- 33 binary64 parameters, or 264 raw parameter bytes.
- One shared rule is applied to every motor.
- A module-set encoder pools present sensor and feedback samples without using
  array order or opaque ID spelling as geometry.
- Inference uses current observations, one-step sensor changes, local motor
  feedback, previous effort, and a small number of interactions.
- Output uses `tanh`, an effort bound of `0.6`, and a per-control-step slew bound
  of `0.1`.
- Missing required sensing, or missing, invalid, or faulted local feedback,
  fails the affected action safely to zero.

The official v3 optimizer, `cem_masked_best_sample_v1`, searches only three of
the 33 parameters: index 29 (previous effort) at physical scale `2.0`, index 30
(bias) at scale `0.25`, and index 32 (ambient-light change multiplied by local
motor velocity) at scale `64.0`. Every other parameter has the exact IEEE-754
binary64 positive-zero encoding. It uses deterministic sampling, safety-first
ranking, paired-zero comparisons, balanced left/right training batches, and a
disjoint validation split. Validation chooses among train-selected sampled
candidates; it does not feed back into the optimizer distribution or future
candidate samples.

The compact payload makes edge inference plausible. It does **not** yet prove
edge-device training speed, latency, memory use, power use, or sample
efficiency.

### Tests and reproducibility machinery

The native validation target covers six suites:

1. Module contracts and motor dynamics.
2. Golden simulator trajectories.
3. Policy-safe environment, reward aggregation, traces, replay, and modes.
4. Zero/random baselines, seed isolation, accounting, lifecycle, and
   determinism.
5. Feature encoding, module-order invariance, artifact integrity,
   compatibility, and frozen inference.
6. Seed splits, deterministic CEM, validation isolation, manifest locking,
   held-out statistics, and negative acceptance gates.

On 2026-09-05, the current bind-mounted source passed all six suites with
`ctest --test-dir /workspace/build/native --output-on-failure`. The frozen v3
core had also passed a clean Docker build before the later report-governance
wording change. Re-run the clean build before treating a new release as
validated. Determinism is intentionally a same-validated-runtime promise, not a
claim of bit-identical MuJoCo trajectories across arbitrary CPUs, compilers, or
math libraries.

Point-in-time evidence for that bind-mounted test run:

| Item | Captured value |
| --- | --- |
| Result | `6/6 CTest suites passed` |
| Native validation input digest | `61a1789a340f7c41fa7168a6fdde07d87044e7d87a841fc20bad92c75e15c08d` |
| Input file count | `32` |
| Managed container ID | `a3eabc3183e054aafd42304857b6a10b2c3a7cf15787fcf177149335e6aeaaaf` |
| Runtime image ID | `sha256:eabe48053a78bcb334a947e7a5c0c009d6ff174a3b5e57e2493593d423b91068` |

The input digest is SHA-256 over UTF-8 lines containing the sorted relative
path, a tab, the lowercase file SHA-256, and a newline for `CMakeLists.txt`,
`Dockerfile`, `setup.sh`, `run.sh`, and all files under `include`, `src`,
`config`, `models`, and `tests`. It is a point-in-time consistency aid, not a
signature. The image ID identifies the toolchain/runtime but cannot bind the
source because the source is mounted from the host. The raw CTest log was not
retained in the repository, so rerunning the command remains the authoritative
check.

## Scientific state: what we actually know

The official frozen pre-heldout artifact is
[`../artifacts/light-search-policy-v3-seed0.json`](../artifacts/light-search-policy-v3-seed0.json).
It selected generation 6.

| Identity domain | SHA-256 |
| --- | --- |
| Canonical artifact payload | `4dd4d6ccadecb0b3b69d3b3d412af83eff15208b19058701538dbdf83a4c32b5` |
| Exact serialized file | `6f7d5069bdb884d75f1241629d437c039a2f7a486633d9ef2d7f6b70db0a5582` |
| Frozen v3 protocol manifest | `f06d689351ddf10c6e987661d7a1f935df4cd8a3c911a76dd13395a6d4b49e93` |

These hashes detect mutation and compatibility mismatches. They are not
signatures and do not authenticate authorship or ownership.

A separate invocation reloaded the artifact and compared the learned policy
with paired zero control on all 16 validation worlds. This was not an
independent implementation, team, or runtime:

| Validation stratum | Mean reward-rate gain | Wins |
| --- | ---: | ---: |
| Light at negative x | `+0.09791938469996925` | `8/8` |
| Light at positive x | `+0.09668843687494068` | `8/8` |
| Overall | `+0.09730391078745496` | `16/16` |

Those learned episodes had zero environment terminations, safety vetoes,
invalid reward-sensor samples, and invalid actuator-feedback samples. Peak
current was about `1.5 A`, peak temperature about `27.84 degC`, maximum effort
about `0.6`, and mean absolute effort about `0.3132`.

This supports a deliberately narrow statement: a tiny shared controller learned
promising, bidirectional light-seeking behavior on separate validation
realizations of one declared 1-D task and one fixed body, with zero violations
of the implemented simulated safety gates during those episodes.

It is **not** an official success result. The validation command recorded
`test_rollouts_executed: false` and emitted no held-out acceptance decision. A
compact, explicitly reduced validation record is in
[`../artifacts/reports/light-search-policy-v3-seed0-validation-summary.json`](../artifacts/reports/light-search-policy-v3-seed0-validation-summary.json).
That file honestly records `raw_stdout_retained: false`: it is a convenient
machine-readable reduction, not the original command output, and cannot by
itself authenticate that reduction.

The project record declares 64 v3 challenge worlds untouched. Its
preregistered comparison runs the learned policy, zero effort, and four matched
seeded-random controls—384 episodes total—and then applies statistical,
practical-gain, side-balance, win-count, and strict safety gates. The retained
validation evidence says no test rollout occurred and no official held-out
report exists. Because another path or copied workspace could bypass local
exclusivity, the software cannot prove global non-execution; the untouched
status is an experiment-governance assertion. No held-out success is claimed.
See [`LEARNING_EXPERIMENT.md`](LEARNING_EXPERIMENT.md).

One operational weakness is worth preserving in the handoff: the evaluator
runs the rollouts before installing the final report. A late filesystem failure
could therefore expose the challenge split without leaving the intended durable
report. Exclusive output prevents overwriting a named report, but the system is
not a transactional or globally enforced one-shot evaluation service.

## Exposure and transfer-learning rule

Seen worlds and seen topologies are not a defect here. Reuse, replay, adaptation,
and transfer are part of the charm. The honesty constraint is release-scoped:

1. While a release is being trained and selected, its declared challenge set
   stays outside that development loop.
2. Freeze the policy, protocol, acceptance rule, lineage, and relevant hashes.
3. Evaluate once for that release and preserve the result, including failures.
4. After the decision is fixed, the evaluated worlds, topology, traces, and
   outcomes may be explicitly relabeled as known curriculum, replay, or
   diagnostic data for a later version.
5. Any descendant that uses them must not call them unseen evidence. A new
   generalization claim receives a fresh, versioned challenge set that stayed
   outside that descendant's development loop.

In short:

```text
sealed challenge -> frozen release -> evaluation -> preserved report
        -> relabel as known experience -> train descendant
        -> new sealed challenge for the descendant's next claim
```

This allows lifelong learning without rewriting history. The v3 artifact,
manifest, validation evidence, and eventual report should remain immutable; a
transfer-learning successor should use a new version and record v3 as its
parent.

## What is not built or proven

- There is no assembly compiler or physical connector discovery yet.
- Only one fixed motor topology is executable, and artifacts deliberately lock
  that topology. Current module-order invariance is not arbitrary-topology
  transfer.
- There is no learned representation of connector graphs, passive mechanics,
  gear ratios, or morphology.
- Optional sensor families exist in the contract, but the learning result uses
  one fixed mounted sensor set. General behavior with missing or newly attached
  sensors is unproven.
- There is no continual replay store, exposure ledger, checkpoint lineage
  graph, adaptation protocol, or catastrophic-forgetting measurement.
- Sample efficiency has not been benchmarked against modern alternatives.
- The small policy proves compact storage, not an edge training system.
- The reference motor constants are simulator values, not measurements from a
  selected physical motor.
- There is no hardware-in-the-loop test, fabrication calibration, system
  identification, battery model qualification, or sim-to-real result.
- No result demonstrates transfer across bodies, topology classes, materials,
  dynamics, manufacturing tolerances, or sensor families.
- The current task is a useful instrumented foothold, not an adequate model of
  open-ended viability or life.

## Open work: useful ways to help

These are candidate workstreams, not an imposed sequence.

### 1. Finish and preserve the v3 baseline

With explicit project-owner approval, execute the already-frozen v3 challenge
once, preserve the complete report and hashes, and make no post-result change to
the v3 artifact or claim. A failure is still valuable evidence and should not be
hidden or rerun into a preferred answer.

### 2. Add an exposure ledger and lineage

Design a small versioned format that records:

- Parent artifacts and algorithms.
- Which worlds, seeds, topologies, traces, and outcomes were available during
  training or selection.
- Which challenge sets were sealed for which claims.
- When evaluated data was promoted into curriculum.
- Runtime and hardware compatibility evidence.

The ledger should make transfer legitimate and auditable without pretending
that learned experience never happened.

### 3. Generate topology families

Move from one rover to parameterized families of assemblies. Start with small,
interpretable changes—motor count, axle orientation, wheel size, gear ratio,
sensor placement, missing sensors, mass, and friction—before attempting arbitrary
LEGO geometry. Keep some instances and, eventually, entire topology classes
outside development when measuring zero-shot transfer.

### 4. Generalize the policy boundary

Explore permutation-equivariant or graph-based encoders that accept variable
module sets and connector graphs while preserving opaque identities and local
frames. Decide what knowledge is shared globally, per module family, per
instance, and per assembly. Compact adapters or recurrent state may be more
edge-friendly than a large universal network.

### 5. Establish simple continual-learning baselines

Before a sophisticated method, measure:

- Fine-tuning from the parent policy.
- Rehearsal/replay with a bounded memory.
- Frozen shared core plus new topology adapter.
- Distillation or regularization for retention.
- Learning from scratch under the same sample budget.

Report adaptation samples and time, forward transfer, retention, backward
transfer or forgetting, safety events, and final task performance. A transfer
method should earn its complexity by beating the scratch baseline.

### 6. Preserve non-learned safety during adaptation

Current, temperature, speed, stall/impedance, joint, feedback-staleness, and
command-timeout enforcement should remain outside learned reward. Qualify new
adaptation in simulation, then hardware-in-the-loop, before allowing motor-side
online learning on a fabricated robot.

### 7. Prepare sim-to-real deliberately

Select real parts, bench-measure motor and sensor behavior, and give each
fabricated module a calibration record. Add controlled randomization for mass,
friction, backlash, compliance, voltage sag, latency, noise, thermal behavior,
and manufacturing tolerance only when its range is tied to measurements.

### 8. Make learning visible

The browser currently shows the engine and privileged demo animation. A useful
next visualization would distinguish demo, training replay, validation, and
frozen-policy evaluation; show which modules are present; and expose sensor
votes, actuator safety state, lineage, and experience status without leaking
privileged values into policy input.

## Questions worth challenging

- What should transfer: one universal policy, a shared module rule, a world
  model, reusable skills, per-topology adapters, or some combination?
- What graph information can a fabricated connector system measure reliably?
- How should a learner distinguish a missing sensor from a temporarily stale or
  failed one?
- Since sensor rewards literally sum, how do calibration and duplicated sensors
  shape behavior? Which effects are intended votes, and which are accidental
  scale advantages?
- What is the smallest topology family that can falsify the current shared-rule
  approach?
- Which metrics best separate memorization, rapid adaptation, retention, and
  zero-shot topology generalization?
- What bounded memory and compute budget represents the intended edge device?
- How can exploration remain informative without turning hardware protection
  into a reward-hacking problem?
- Which world variations should become curriculum, and which should remain a
  fresh challenge for a particular claim?
- What evidence would make us comfortable moving an online learner from
  simulation to one real actuator, then to a complete assembly?

Please add better questions. Negative results and simpler alternatives are
welcome.

## Getting oriented

Recommended reading order:

1. This file.
2. [`../README.md`](../README.md) for commands and deployment state.
3. [`MODULE_CONTRACT.md`](MODULE_CONTRACT.md) for sensors, rewards, motors, and
   safety.
4. [`AGENT_ENVIRONMENT.md`](AGENT_ENVIRONMENT.md) for the opaque learning API.
5. [`LEARNING_EXPERIMENT.md`](LEARNING_EXPERIMENT.md) for frozen v3 science.
6. [`LEARNING_DESIGN_HISTORY.md`](LEARNING_DESIGN_HISTORY.md) for why v3 looks
   the way it does.
7. [`../config/module_catalog.json`](../config/module_catalog.json),
   [`../config/learning_experiment_v3.json`](../config/learning_experiment_v3.json),
   and [`../models/droid.xml`](../models/droid.xml) for executable definitions.

The workspace is:

```text
C:\Work\Natural-Dexterity\projects\droid-blocks
```

Before touching Docker, inspect the existing managed container and compare its
complete effective configuration with the contract in the README. Start it in
place only if every setting matches; do not replace or delete it merely to
obtain a clean start:

```powershell
docker container inspect droid-blocks-dev
# Compare image, command, labels, mount, port, GPU request, init, shared memory,
# restart policy, and stop timeout with the README before continuing.
docker container start droid-blocks-dev
Invoke-RestMethod http://127.0.0.1:43117/healthz
```

The stopped container `droid-blocks-dev-python-backup` is intentionally
preserved historical data. Do not delete, rename, or repurpose it without an
explicit owner decision. The complete verified container contract and safe
creation procedure are in [`../README.md`](../README.md).

For a native build inside the configured Debian environment:

```sh
cmake -S . -B build/native -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
cmake --build build/native --target droid-native-validation
ctest --test-dir build/native --output-on-failure
```

## Current commitments behind the existing evidence

- Native C++ is the project language for simulation, learning, inference, and
  edge-facing artifacts; the current container intentionally has no Python.
- Sensor instances own reward functions, and present sensor rewards sum.
- Actuator feedback has no reward authority.
- Hard safety is noncompensable and outside learned reward.
- Policy observations exclude privileged simulation and authored functional
  roles.
- Modules and protocols are versioned; frozen evidence is not overwritten.
- Transfer learning is welcome, but exposure and lineage must be explicit.
- Existing managed Docker objects are preserved unless their owner explicitly
  authorizes replacement or deletion.

These starting guardrails can be revisited. Changing one should be an explicit
design decision with migration and evidence—not an accidental side effect.

## Collaborator notes and proposals

### Codex / 2026-09-05: let rebuilding lead the roadmap

[Play-first roadmap proposal](PLAY_FIRST_PROPOSAL.md): bring the child's
build-observe-rebuild loop forward with a visible learned sun-seeker, a tiny
family of genuinely different motor transmissions, and temporal sensor
modeling. Challenge shared control with a reversed motor before increasing
policy complexity; then try the same light preference in a stationary,
sensor-turning body. The note also discusses preference design, per-module
memory, and ways to make sensing and adaptation understandable through play.
This is a discussion proposal, not an adopted plan. The review used documents,
source, and a read-only playground inspection; it ran no learning or challenge
experiments and made no runtime or frozen-evidence changes.

Please extend this section rather than waiting for a perfect plan. A useful
entry can be as small as:

```text
Name / date:
Assumption being challenged:
Proposed smallest experiment:
Expected observation:
Failure or safety boundary:
Worlds/topologies/data that would become exposed:
Files or artifacts produced:
Open questions:
```

There is room here for controls, alternative algorithms, mechanical insight,
electronics constraints, learning theory, visualization ideas, and critiques of
the premise itself. The goal of the handoff is not to make the next person obey
the current design. It is to let them change it knowingly.

### Codex / 2026-09-05: adopted next goal — “It noticed the sunshine”

Following the project owner's invitation to define and pursue the next goal,
the first play-first milestone is now adopted: wake the existing learned rover,
show the sunshine and its sensor response, pause and reset it, and compare the
two published training-world sides. The implementation adds native frozen
inference playback and explicit controller ownership alongside a larger
workbench view. See [The sun-seeker playground](SUN_SEEKER_PLAYGROUND.md) for
the complete milestone, API, exposure record, and verification status.

This is an additive update to the context captured above. It does not revise
the original v3 evidence, change its simulator/model/catalog/policy, or execute
its pending challenge evaluation. The playground loads the preserved v3 artifact
and selects only the first published negative/positive training pair. All seven
native suites passed, and both browser runs completed their 20-second horizon
with higher light readings; HTTP ownership and pause/resume checks also passed.
The linked milestone record preserves the results and their development-only scope.
The next proposed question is whether a reversed or disconnected transmission
makes rebuilding understandable and informative in a separately versioned
development experiment.

### Codex / 2026-09-05: adopted next goal — “I changed its body”

The owner asked to continue after the first visible movements. The workshop now
has an explicit successor body experiment with a normal, reversed, or
disconnected transmission on one independent motor shaft, delayed light
samples, and a per-motor probing controller with run-local memory. A shared
probing baseline and zero control make its costs and benefits inspectable.
See [I changed its body](BODY_DISCOVERY_EXPERIMENT.md) for the protocol, exposure,
mechanical scope, controller versions, preserved failures, and results.

The first unloaded-shaft probe hit voltage limiting and stopped. That result
is preserved. A second controller adds a local speed governor without relaxing
feedback rejection or native safety. All nine native suites pass; all 18 final
development episodes complete without feedback flags or safety vetoes. Local
probing finds the reversed motor's different command and leaves an unclear
disconnected motor idle. Shared probing still does better on the normal and
disconnected bodies under the same total time budget. All three bodies and both
published training worlds are known development data. The original learned
sun-seeker remains available; its frozen evidence and pending v3 challenge are
unchanged. Rechecking preferences after changes during a run is the next open
question, not a capability already claimed by this milestone.

### Codex / 2026-09-05: adopted broader goal — build a body, discover its motions

The owner emphasized that rebuilding means adding blocks and changing the
artifact's shape, and that passive mechanics and underactuated motion belong
in the eventual exploration space. They explicitly asked us to define and
pursue that broader goal. [Build and discover](BUILD_AND_DISCOVER_GOAL.md)
records the adopted direction, the first bounded construction milestone, and
an experimental roadmap toward adaptive exploration, contact, sensor-guided
motion selection, and transfer across rebuilds.

This brings actual construction ahead of the previously proposed moving-light
recheck. The first kit is an anchored two-link creature with one motor, one
passive hinge, editable beam units, weight blocks, and a movable IMU. It receives
its own native world, assembly format, sensing boundary, and development
record. See [the construction experiment](CONSTRUCTION_EXPERIMENT.md) for
implementation details, verification, and limits. The earlier experiments and
frozen-v3 challenge remain separately preserved.

The first construction milestone is complete: all twelve native suites passed,
all 24 declared development trials completed without feedback flags or safety
vetoes, and every emitted observation trace replayed exactly. Moving the same
weight changed measured responses at equal total mass. Browser checks covered
actual part edits, save/restore and JSON import, native pause/resume, five
completed motion cards, exact trail replay, and narrow/wide layouts. The bench
remains an anchored collisionless two-link kit with a fixed-pattern exploration
baseline. Adaptive selection of new probes, contact mechanics, and broader
connector graphs remain future experiments, as recorded in the adopted goal.

### Codex / 2026-09-05: adopted next experiment — find the light, then stay with it

The owner asked to give the constructed creature a light sensor and a sun,
then authorized implementation. The new sunshine bench (`/light.html`) uses a
physical receiving face, delayed native sensing, and a continuous online
learning loop. It can explicitly copy the owner's current assembly from the
construction bench. Sun moves preserve physics, queued samples and memory;
rebuild/reset begins a fresh try. The old workshop remains available.

[The light-learning experiment](LIGHT_LEARNING_EXPERIMENT.md) records the
engine boundary and the declared two-body, 120-second comparison with a sun
move at 60 seconds. The first learner updated values correctly but lost to
quiet and fixed-rhythm controls. That report and source are preserved. A
single measured initialization correction, with the same bodies, sun, seeds,
reward and other parameters, produced higher total return than every baseline
for all six learning runs. Median late light after moving the source exceeded
quiet control by 26.4% and 18.2% on the two known development bodies. All sixteen
final trials completed without feedback flags or native vetoes and replayed
exactly. This is bounded development evidence, not a claim of arbitrary-body,
contact or hardware control.

All fifteen native suites passed after the correction. The original 24
construction command sequences and observation hashes remained identical;
pre-extension world source/header are archived. The original model, catalog,
v3 experiment and learned policy remain byte-for-byte unchanged and the v3
challenge remains unexposed. A broader moving-source protocol and a comparison
of continued learning against frozen learned values are proposed next; these
have not been run or silently adopted as held-out evidence.

## 2026-09-06: continued-learning ablation completed

The next goal was to isolate continued parameter updates after the sun moves.
[The adaptation experiment](LIGHT_ADAPTATION_EXPERIMENT.md) records six matched
pairs: identical 60-second prefixes, then continued learning versus frozen
values while both branches retain sensing, memory, exploration and feedback.
All twelve trials and full-transition replays passed. Continued learning won
all six paired post-move reward comparisons; median paired gains were +2.7022
(default) and +2.5756 (side weight). This is known development evidence, not
optimality, arbitrary-body transfer or the pending rover-v3 challenge.

The native learner now supports a tested one-way diagnostic freeze and exact
parameter/full-state fingerprints. The browser's controller API is unchanged.
The read-only result viewer is `/adaptation.html`; its data is derived from
`artifacts/reports/light-adaptation-v1-development-20260906/report.json`.
Its source/provenance manifest has 74 files. All 15 native suites passed.
The original v2 continuing trajectories, frozen-v3 inputs and both completed
live construction/light sessions were preserved exactly; no restart was needed.

Keep v2 as a measured baseline. The proposed next memory experiment is a
return to a familiar light condition (A-to-B-to-A) with a controlled comparison
against saved first-A values. This retention question is not yet executed.
State estimation, predictive control and broader constructions remain future
engine work; do not confuse the new ablation mechanism with those algorithms.

## 2026-09-06: the retention goal is complete

The [locked A-to-B-to-A protocol](LIGHT_RETENTION_EXPERIMENT.md) has now run.
[The separate results record](LIGHT_RETENTION_RESULTS.md) supersedes the earlier
pending status. First-visit frozen values beat current frozen values in all six
primary 120-140-second reward comparisons, with within-body median paired
advantages +1.081243 (default) and +1.084410 (side weight). They also beat both
current frozen values and continued learning over the full return in all six
cases. Continued learning still beat current frozen values. Recall increased
energy in four cases and reduced it in two; it is not an efficiency result.

The engine now has a tested immutable value-checkpoint boundary. Capture is
read-only; restore changes only the value weights and preserves all other
controller state. Both frozen branches continue observation history, feedback,
exploration and bookkeeping. All 18 trials and complete replays passed; all
15 native suites passed. Exact historical controller fingerprints matched at
60 and 120 seconds. The read-only viewer is `/retention.html`, derived from
`artifacts/reports/light-retention-v1-development-20260906/report.json` with
75 archived source/provenance files. The protocol hash is locked: do not append
outcomes to LIGHT_RETENTION_EXPERIMENT.md; keep them in the results document.

Both live completed sessions, earlier reports, catalog/model and frozen-v3
inputs remained unchanged. There was no server restart or challenge exposure.
The appropriate saved checkpoint was chosen externally: these results show
useful older values, not autonomous memory retrieval. The next proposed engine
step is a matched-action sensor-history decodability experiment, followed by
observation-driven selection among supplied value candidates if warranted.
Declare that follow-up separately before execution; source identity and switch
time must not become policy inputs. Keep v2 as the measured baseline.

## 2026-09-06: the sensor-context goal is complete

The [locked context protocol](LIGHT_CONTEXT_EXPERIMENT.md) has run once without
retuning. [The separate results](LIGHT_CONTEXT_RESULTS.md) record 48 native
trials and 48 complete exact replays. With whole random-probe sequences withheld,
one second of local sensor/action history achieved 93.75% accuracy on the default
body and 95.45% on the side-weight body. Current readings achieved 73.86% and
72.16%; history improved all eight body/probe comparisons. The strict no-light
control produced p(A)=0.5 everywhere and exactly 50% accuracy. History alone
cleared the predeclared per-body/per-probe readiness rule.

The native addition is an immutable five-nearest-neighbor decoder with fixed
feature projection, causal windows and tested input boundaries. Labels enter
only offline training and evaluation. A separate Node verifier reconstructed
all windows, training membership, votes, predictions and readiness decisions.
All 16 native suites passed; nine deliberately corrupted evidence fixtures
were rejected and an untouched copy accepted. The report and 78 source/provenance
files are in artifacts/reports/light-context-v1-development-20260906. The
read-only viewer is /context.html. No live controller API was added or changed.

The context protocol SHA-256 is
3e1218d1e0000cca5020962aeb73ba773f92bdcc1012a178ba4d238389d87dff.
Keep it locked and record outcomes separately. The report SHA-256 is
f3f729120df6b13f79845ecbd4ee72409f111f76b1b525037d3c059e41f7c6ee.
Earlier reports, frozen inputs, and both completed live sessions remained
byte-identical. The existing container stayed healthy without a restart.

This result supports testing observation-driven selection among supplied frozen
value memories. It does not demonstrate autonomous context discovery or useful
runtime retrieval yet. Most history votes were narrow 3:2 neighbor majorities;
87.5% fell within the declared ambiguity band. Do not interpret votes as calibrated
confidence or tune an abstention cutoff on these withheld results. The next
protocol should compare a frozen selector with always-current memory and an
externally chosen recall reference, including constant-source controls and
undisclosed changed switch times. Memory-controlled actions change the input
distribution; switched sources add mixed-history windows. Declare switching,
uncertainty and fallback rules before running that control experiment. Keep
v2 as the measured baseline and the original rover-v3 challenge unexposed.

Handoff availability: after successful native/browser/preservation checks,
Docker's Linux-engine pipe became unavailable and port 43117 refused connections.
The cause is unknown. No restart was performed. A separate read-only review is
available at http://127.0.0.1:43118/context.html, served by
build/light-context-qa/serve-context-review.mjs. It exposes only the retained
context-view assets, has no simulator API, and marks the native simulator offline.
The report/protocol/source hashes remain intact. The recorded live-state
preservation applies to the completed check before this runtime interruption.

## 2026-09-06: native viewer available again

The owner confirmed a computer restart and restarted Docker. The original
container is running and healthy; /context.html is restored on port 43117,
and the existing report tab has been moved back from the temporary preview.
The retained report, protocol, models and 78 archived source files verify;
no prior-goal scientific work remains and no new trials were run. The old
preview port has no listener. The recorded pre-restart live-session preservation
is historical and must not be interpreted as in-memory persistence across reboot.

The owner clarified that starting Docker and reusing the existing project
container is authorized when needed for this work; do not stop the task merely
because Docker is off. Follow the documented launcher and verify/reuse the
existing container. This does not authorize replacing containers, deleting
Docker data, or changing the established configuration.

## 2026-09-06: establish a shared control-theory description

The owner said the experimental language obscured states, transfer matrices,
control functions and the reinforcement-learning method. The new
[implemented control model](CONTROL_MODEL.md) maps these directly to code:
nonlinear underactuated plant, delayed observations, manually encoded 56-feature
history state, three linear action values with 168 learned coefficients,
Expected SARSA-based option updates/traces/replay, reward and effort governor.
It also distinguishes the separate 132-feature supervised context classifier.

An identified transition model, physical-state observer, A/B/C/D design,
controllability/observability analysis and stability guarantees are absent from
the current light engine. Do not suggest that these are implicit accomplishments
of classification accuracy or native replay. Saved 'memories' in the retention
experiment are value-weight matrices, not physical states or transition models.

Before pursuing another memory-selector experiment, the revised recommendation
is to agree on the engine's state/model architecture and evaluate prediction
of sensor consequences under withheld inputs. This is proposed work, not a new
adopted goal or completed system-identification result. Retain the present RL
controller as an experimental baseline. Use explicit mathematical objects in
future discussions, with approachable explanations alongside them.

## 2026-09-06: explicit state-space identification is complete

The owner adopted the revised model question as a goal. The
[locked prediction protocol](LIGHT_PREDICTION_EXPERIMENT.md) ran once with no
post-result tuning. [Separate results](LIGHT_PREDICTION_RESULTS.md) record
72 native trials and 72 exact full replays, across the default body, side-weight
body, and a longer first beam. Each body is fitted separately; this is not transfer.

The new frozen native predictor learns the affine transition
z_next=A*z+B*v+c and supplies C/D output maps at .1 s sampling. The primary
142-dimensional state stacks eleven local output frames and ten past requests.
Future normalized motor requests are known; future governed efforts remain
predicted outputs. No future sensor readings, source labels, or hidden physical
state enter a forecast. The independent checker rebuilt every training state,
refitted the declared ridge regression, checked every companion-matrix entry,
and recomputed all recursive forecasts, metrics and readiness gates.

History reduced one-second primary RMSE versus persistence by 47.17%, 55.15%
and 63.57%. The 12-dimensional current-output model also passed all prospective
gates, with reductions of 48.55%, 54.52% and 62.99%. Longer history improves the
shorter horizons but is not established as necessary. Light-specific prediction
benefits are smaller: one-step light prediction is worse than persistence on
the default and weighted bodies. Do not translate composite prediction gains
into equal light gains or a claim of better control.

Artifacts are in artifacts/reports/light-prediction-v1-development-20260906.
Report SHA-256: 3d9e70141c7fd73da226c4b5715678205c9181623a308f8f93e9d7a59494b0e0.
Models SHA-256: 5421f975d958d831e58414eef661eabfd852e78734036acb293d057fd7bb81ee.
Protocol SHA-256: 7cc508c8cf1bc88e0b0f417b49df93022e895f8f90c9f1c91e528372696ef6be.
There are 70 archived source/provenance files, 17 passing native suites, and
13 passing evidence checks (untouched copy accepted; twelve corruptions rejected).
All 19 protected prior files and both live sessions remain byte-identical.
The same existing container remains healthy without a restart.

The read-only /prediction.html page exposes the matrices, channel forecasts,
errors and mathematical interpretation. /discoveries.html connects the earlier
findings, with links from /light.html. CONTROL_MODEL.md now distinguishes this
identified sampled surrogate from the still unchanged model-free RL controller.
No physical-state observer, stability certificate or model-based control law
has been implemented. The next justified experiment is a bounded receding-horizon
control comparison using both passing models against the unchanged RL baseline.
Declare its horizon, action cadence, native reward approximation, constraints,
fallback, whole-run comparisons and distribution-shift checks before execution.
The original frozen rover-v3 challenge remains unexposed.

## 2026-09-06: current product orientation after the voice discussion

Read [the design discussion](DESIGN_DISCUSSION_2026_09_06.md) before interpreting
an earlier next-step recommendation as an adopted roadmap. The owner authorized
UI cleanup with animation central and emphasized useful reward-driven behavior
with minimal new experience after children rearrange parts. Configurable reward
and eventual contact/support/recovery behavior remain part of the direction.

A reference duck/T-Rex and a withheld T-Rex-to-snake rearrangement are proposals,
not a selected build or completed transfer result. A single-body pretrained
policy may not transfer to that new topology. Assess proven open methods before
bespoke algorithms. No MPC, neural architecture, RL optimizer, training stack
or simulator migration was committed by this conceptual discussion.

The live model-free light learner remains unchanged; the completed affine
predictor remains a separate prediction instrument. Preserve all earlier
protocols, reports and frozen inputs. The new note records requirements and
open choices without starting another research goal or training run.
