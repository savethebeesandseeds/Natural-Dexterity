# First assembly: simulation pretraining development pilot

**Prospective pilot protocol, 2026-10-06.** The lock command must archive this
document, its matching machine-readable manifest, sources and runtime receipts
before the first source-learning decision. Outcomes belong in a separate result
record. The later generalization study is proposed below; it is neither locked
nor executable through this pilot.

## Product requirement and the question this pilot answers

The owner clarified that the **first creature a child builds must arrive with
broadly pretrained competence across simulation environments**. Subsequent
online learning should make modest adjustments to that particular assembly.
Training substantially from scratch after the first assembly, or after each
rebuild, is not the intended product experience. The acceptable first-response
latency and specialization allowance still need an explicit product benchmark.

The immediate question is smaller: can one pooled PPO controller be trained
through the existing native interface on a declared development curriculum,
then earn useful reward with fixed parameters on twelve known constructions
from its first interaction? This pilot checks the Linux training path, exposure
accounting, checkpoint integrity and source competence, and compares the trained
policy with its original initialized policy to describe the benefit of offline
updates. It does not measure unseen-body generalization or establish the
first-creature product requirement.

The current scope is one anchored, collisionless, planar two-link connection
graph, with one motor, one passive hinge, one distal IMU and one light sensor.
"Broad" in this pilot means a broader distribution inside that fixed class:
beam lengths, a distal block, powered hinge and lamp side. It does not mean
arbitrary connector graphs, variable motor/sensor counts, ground contact,
walking or hardware transfer. Physics, native reward, acquisition queues,
governor and the live workshop controller remain unchanged.

## Why this is a successor experiment

The [September actuation-transfer study](ACTUATION_TRANSFER_RESULTS.md) failed
its improvement gate on upper-lamp conditions, with weak competence above quiet
before transfer. Its PPO checkpoints, failed reference gain and result stay
unchanged. The [October headroom study](LIGHT_HEADROOM_RESULTS.md) subsequently
found a useful movement margin and successful fresh-memory Expected SARSA
acquisition on all twelve lower-lamp cases within 64 simulated seconds.
It transferred no parameters and did not establish stable holding in every
layout. Its eleven stopped trials and historical upper-lamp failures remain
part of the record.

This pilot uses that exposed lower-lamp curriculum to assess a pooled pretrained
policy. Neither preceding result is promoted into a generalization success.
All inspected prior worlds, tapes, models, outcomes and design choices are
known development exposure for this successor, even where a historical
experiment correctly called its own probe sequences withheld.

## Curriculum and exposure boundary

The canonical curriculum uses schema `generalization_curriculum_v1`, with
`source_cases`, `development_cases`, `reserved_cases` and corresponding
`case_ids.source`, `case_ids.development`, `case_ids.reserved` arrays. Cases
contain only the declared assembly and sun configuration. IDs and split labels
are evaluator metadata and never policy inputs.

Every source is stationary at `[x, 0, -0.30]` metres, with x=-0.30 or +0.30 and
intensity 1000 lux. These are the existing headroom study's lower lamps, not its
historical upper lamps at z=+0.20. The IMU remains at segment 1, slot 2, side -1;
the light sensor at segment 1, slot 2, side +1, receiving face +1. An added
block is at segment 1, slot 0, side -1, with the existing mass and dimensions.

| Partition | Bodies and variation | Cases | Meaning in this pilot |
| --- | --- | ---: | --- |
| Source | Lengths [3,3], [4,3], [3,4], [4,4] × block absent/present × base/elbow motor × left/right lower lamp | 32 | Permitted source training |
| Development | [4,3] plain, [3,4] plain, [4,4] with block × base/elbow × left/right lower lamp | 12 | Known headroom cases, all inside the source domain |
| Reserved | [5,3] plain, [3,5] plain, [5,5] with block × base/elbow × left/right lower lamp | 12 | Manifest only; no native exposure in this pilot |

Source ordering is lengths as listed, block absent then present, base then
elbow, and left then right. Source IDs are
`source-L{a}-{b}-{plain|distal-block}-{base|elbow}-{left|right}`.
Development and reserved ordering is long upper, long lower, weighted, each
base/elbow and left/right. IDs use
`development-{long-upper|long-lower|weighted}-{base|elbow}-{left|right}` and
the corresponding `reserved-` prefix.

The source sampler draws shuffled round-robin cycles using an independent
`random.Random(500404)`. Each completed 32-draw cycle covers every source case
once. This balances reset-task draws, not transition counts: safety stops and
the final partially experienced episodes can give different physical exposure
to different cases. Retain the exact draw schedule and realized per-case
steps/resets. Policy sampling and optimizer randomness use seed 404 and cannot
advance the source sampler.

Reserved configurations may be constructed as JSON and checked statically for
manifest disjointness. No reserved case may be configured in a native world,
reset, stepped, rendered through new simulation, probed by a baseline or used
for normalization, debugging or checkpoint selection. The pilot has no command
for executing reserved cases. A future prospective lock and established source
competence are prerequisites for any reserved evaluation; there is no added
owner-approval requirement for this new study.

The original frozen rover-v3 challenge remains a separate pending experiment
with its existing explicit owner-approval requirement. This pilot does not use
the rover environment or execute that challenge. If a later curriculum includes
that profile, exclude its reserved realization keys as well as its seeds until
its fixed decision is preserved. Keep its artifact, manifest and acceptance
rule unchanged.

## Controller, observations and reward

Reuse the original PPO interface and its 204-dimensional causal stack:
twelve 17-scalar frames at 0.1-second decision endpoints, oldest to newest.
The frames contain local motor angle sine/cosine, shaft velocity/current,
previous emitted effort, local IMU specific force/angular velocity, delivered
light, previous request/reward and validity/presence masks, with the existing
fixed scales and [-5,5] clipping. Actor and critic see the same stack. No
fitted observation or reward normalization is added.

Geometry, mass, powered-hinge labels, lamp coordinates, world pose,
passive-joint state, split/case IDs, seeds and absolute clock/countdown remain
outside that input. Native validation metadata stays outside neural features.
This excludes explicit morphology conditioning from the pilot; it also limits
the generalization question the representation can answer.

Each action is a categorical request -0.15, zero or +0.15 held for 0.1 seconds.
Five native 0.02-second controls refresh the unchanged local governor and safety,
each containing ten 0.002-second physics steps. Retain sampled action, requested
effort and terminal emitted effort in compact telemetry. All five native emitted
efforts remain recoverable through retained requests and selected native replay
with recording enabled. Reward is the exact sum of the five already integrated
native transition rewards, combining the existing light and IMU votes. Do not
integrate twice, add a motion bonus, shape a target angle,
normalize return, or train on an evaluator's failure-completion score.

Native initialization is `rebuild -> set_sun -> reset`, with the existing
hanging pose, zero velocities, fresh motor state and fresh acquisition queues.
Initialize the causal stack with eleven absent zero frames and the current
reset observation under its validity masks. Acquisition and any initial quiet
requests count from wake-up inside the opportunity; there is no free physical
warmup or extra native probing.

## Fixed pilot training and development budgets

| Setting | Pilot value |
| --- | --- |
| Independent source seed | 404 only |
| Vector environments | 8 |
| Rollout steps per environment | 256 |
| Source aggregate decisions | Exactly 16,384: eight complete rollouts of 2,048 |
| Episode limit | 640 decisions / 64 simulated seconds |
| Actor and critic | Separate two-layer 64 × 64 tanh networks; categorical three-action output |
| Learning rate / Adam epsilon | 0.0003 / 0.00001 |
| Epochs / minibatch | 4 / 64 |
| Gamma / GAE lambda | 0.92^(0.1/0.4), approximately 0.9793703613355593 / 0.95 |
| Policy clip / value clip | 0.2 / none |
| Entropy / value coefficients | 0.01 / 0.5 |
| Gradient norm / target KL | 0.5 / none |
| Initialization | Orthogonal; existing fixed network configuration |
| Normalization | Advantage normalization only; no VecNormalize |
| Runtime | Pinned Stable-Baselines3 2.9.0, Linux CPU, one Torch thread |
| Operational caps | Source initialization plus learning 600 wall seconds; each development trial 180 wall seconds |

The final source checkpoint is the sole trained checkpoint used for development.
Before the first source-learning decision, preserve the seed-404 initialized policy
checkpoint and its parameter hash. This one fixed initial policy supplies the
untrained arm on every development case; never resample its initialization.
Record all eight rollout updates, actual decisions, finite-parameter checks,
timing and final parameter hashes. There is no reward-based early checkpoint
selection, resumed attempt, seed replacement or pilot hyperparameter sweep.
The exact budget is divisible by the rollout size; library overshoot must not
be relabeled as exact completion. A clock cap preserves an incomplete attempt.
Report initialization, learning and total source wall time separately. The source
deadline covers initialization and learning; final tape/hash archival and integrity
replay occur outside it and their times are reported separately. A blocked
native call or update may overrun a cap and must be identified.

Source training may reset at its episode limit or native failure, within the
aggregate budget. Preserve every reset, stopped prefix and terminated versus
time-limit-truncated flag. Bootstrap a time limit from the actual terminal
observation, not an automatically reset world's first observation. Clear the
frame stack only at an actual reset; not at PPO updates. Parallel environments
reduce elapsed computation, not physical experience.

After source weights are frozen, run trained frozen PPO (`frozen`), the preserved
untrained frozen PPO (`untrained`) and quiet once on each of the twelve
development cases: **36 trials**, each with at most 640 decisions from wake-up.
Both neural arms sample categorically (`deterministic=False`). Pair their action
RNG seeds at `10000 + 404*10 + case_index`, or 14040 through 14051, set after
fresh model construction and loading the appropriate weights. Quiet uses no
action RNG. Copy actor/critic weights only; no source optimizer, rollout buffer,
history, task schedule or observations cross into a development trial. The
trained parameter hash must match the final source checkpoint, and the untrained
hash the preserved initial checkpoint, before and after every respective run.

Development trials receive no optimizer updates, rescue resets or experience
from other cases. The evaluation loop prevents automatic reset after its final
budgeted transition. Stop at the first native termination, veto, feedback flag
or controller exception and retain its actual exposure and error.

The planned maximum is **39,424 aggregate decisions**, **197,120 native
controls**, **1,971,200 physics steps**, or **3,942.4 simulated seconds** across
training and development. Safety stops can reduce realized exposure. Integrity
replays are additional computation and repeated simulation, not independent
learning trials. CPU optimization pauses simulated time; this does not measure
responsiveness of an unpausable physical toy.

## Development diagnostic and interpretation

The source-competence diagnostic requires all twelve matched pairs to be
complete and safe, with trained frozen PPO's integrated native reward strictly more
than +3.2 above that case's quiet return over 64 seconds. This retains the
headroom engineering margin of +0.05 reward/s. It is a development diagnostic,
not a statistical significance rule, generalization gate or product threshold.

Separately report trained-minus-untrained native reward for every case. Its
primary descriptive mean uses all twelve realized raw-return differences,
including stopped prefixes explicitly identified as unequal exposure. Also
report the mean over complete-safe pairs, that subset's eligible count and
trained-policy wins with their eligibility rule. Neither the subset mean nor
the all-case mean containing stopped prefixes establishes matched-budget
learning benefit. Do not present a survivors-only reduction as the whole
comparison. This diagnostic has no additional pass threshold:
trained-versus-quiet competence can pass while trained-minus-untrained is
negative. Useful motion from arbitrary initial weights alone does not
demonstrate that offline learning helped.

Report every case and every failure, realized steps/duration, raw reward and
light/IMU components, gains versus quiet and untrained, delivered light, electrical energy,
native flags/vetoes, invalid samples and available telemetry. Report cumulative
windows at 64, 128, 320 and 640 decisions: 6.4, 12.8, 32 and 64 seconds.
Window differences are descriptive; none replaces the whole-run diagnostic.
A favorable stopped prefix minus a complete quiet run has unequal exposure and
cannot pass. Quiet can already satisfy another lighting task; this lower-lamp
margin is not a universal requirement to move every possible creature.

Separate two conclusions: plumbing integrity and behavioral source competence.
An operationally valid pilot may miss the behavioral margin. A pass does not
justify selecting seed 404, treating development cases as withheld, or asserting
generalization from one source checkpoint. A failure stays preserved and does
not authorize another seed, bigger budget or architecture change within this
attempt. Any later revised pilot is a new, prospectively declared development
version, with its accumulated exposure recorded.

## Evidence, execution and storage

The new optional training path runs only inside the existing managed Linux
container. It invokes `/workspace/build/native/droid-transfer-env` directly
through batched stdin, without Docker calls inside the bridge or the live HTTP
control API. It rejects Windows execution. The installation script only
configures reproducible optional dependencies; it does not create, replace,
start or stop containers or change the ordinary native launcher.

From the host, optional dependency setup is container orchestration:

```powershell
docker exec droid-blocks-dev /bin/sh /workspace/training/setup-generalization.sh
```

The pinned Linux CPU requirements are in
[`training/requirements-generalization.txt`](../training/requirements-generalization.txt).
The runtime is `/opt/droid-blocks/generalization-venv/bin/python`; setup receipts
live under `/opt/droid-blocks/generalization-setup`. Copy their exact bytes into
the new report's retained source/runtime record before learning. Dependencies,
wheels and package caches stay in container storage. No host Python/Java or
Windows compiler is part of this experiment.

Run these commands **inside the managed container from `/workspace`**:

```sh
/opt/droid-blocks/generalization-venv/bin/python \
  training/generalization_pretraining.py lock \
  --output artifacts/reports/generalization-pretraining-v1-development-pilot-20261006

/opt/droid-blocks/generalization-venv/bin/python \
  training/generalization_pretraining.py pilot \
  --output artifacts/reports/generalization-pretraining-v1-development-pilot-20261006
```

`lock` exclusively creates the new output directory and canonical protocol,
with compact source/runtime provenance before learning. `pilot` requires that
unchanged lock and creates one exclusive `pilot-404` attempt. It cannot resume
or overwrite it. Validate the implementation and canonical manifest against
this document before installing the lock. No reserved-execution command is
provided by this stage.

Retain initial and trained policy parameters/checkpoint digests, source sampler/reset schedule,
requested actions, a native request/response stream commitment, compact
per-decision telemetry, actual exposures, source/development hashes, runtime
versions and immutable container identity. Record every intended and actual
training update. Default records need not include full physical snapshot
streams; any later motion export must use retained native data and declare its
sampling/provenance rather than invent an animation.

Reconstruct native responses from retained requests and resets in an isolated
fresh world. Command replay checks physical reproducibility, not PPO optimizer
replay. Check frozen checkpoint identity separately. Independently recompute
development gains, budget counts, safety and reserved-case exclusion, and
report integrity failures separately from behavioral failure. The independent
post-collection audit tool receives its own source hash and exact byte archive
in the verification receipts; it is outside the action-generating source lock.
Bind those receipts to the locked protocol and collected evidence hashes, so
the validator version remains identifiable without changing frozen collection
sources. Protect earlier
reports, frozen rover-v3 inputs and live sessions by before/after checks.

New transition tapes, snapshot streams and bulk playback stay local and ignored.
Commit source, this protocol, compact summaries and provenance; document assets
a fresh checkout needs for replay. Preserve partial reports and failed receipts.
Do not recursively duplicate complete bundles for QA, force-add ignored bulk,
or delete retained evidence as scratch cleanup. Disposable task files use one
ignored scratch directory or container `/tmp` task directory, with bounded,
verified cleanup under the repository hygiene rules.

## Later generalization study: proposal, not a locked execution

The proposed next scientific study uses three independent source seeds
101, 202 and 303, each with 98,304 aggregate decisions, plus an equal-budget
narrower-source pretraining ablation. Its primary question is frozen first-use
competence on reserved bodies, not recovery after a long new training period.
Compare broader versus narrower training under the same architecture, action
interface, offline budget and declared observation boundary; specify exactly
which variation the ablation withholds and stratify its consequences.

Secondary comparisons can separate bounded head-only adaptation, full-parameter
adaptation and matched scratch learning. They need equal new-interaction
allowances, fresh independent branches and declared early measurement windows.
Frozen weights with ongoing sensor/history feedback and gradient adaptation are
different mechanisms. A useful initial frozen policy and specialization gain
are separate outcomes; neither an adaptation-minus-scratch advantage nor a
large later return can substitute for first-use competence.

The full study still needs a final source-competence rule, source/development
selection procedure, first-response and specialization budgets, all arms,
narrow-source manifest, action/reset RNG schedule, safety/failure handling,
seed-level reductions and primary acceptance rule locked before reserved
exposure. Source competence must be established first. Do not use the current
pilot to execute the proposed three-seed study or its reserved cases.

Length-5 cases would test limited geometry extrapolation within this same
two-link class. Passing them would not establish topology transfer. Broader
curricula should later address sensor placement, preference descriptors,
multiple motors, measurable connector relationships, contact/support and
recovery in separately implemented and versioned families. Feasibility and
available sensing must be assessed independently of learning quality; a
disconnected or unobservable assembly cannot be promised a requested motion.

## Primary method references

The pilot selects pooled frame-stacked PPO as the smallest established-method
comparison using the current boundary. [Official Stable-Baselines3 PPO
documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
recommends first considering frame stacking before its recurrent extension.
The pinned installed version and frozen manifest, rather than changing master
documentation, define the executable settings above.

Recurrence remains a later comparison. [Official RecurrentPPO
documentation](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
describes LSTM state and episode-start reset handling. [RL²](https://arxiv.org/abs/1611.02779)
illustrates task adaptation through changing recurrent activations under
pretrained parameters; that is distinct from updating a final head. Neither
mechanism is implemented by this pilot's finite frame stack.

[RMA](https://arxiv.org/abs/2107.04034) is a relevant simulation-trained base-policy
and adaptation-module reference for a fixed quadruped under varied conditions;
it does not establish arbitrary construction topology transfer.
[MetaMorph's author implementation](https://github.com/agrimgupta92/metamorph) and
[ModuMorph's paper](https://proceedings.mlr.press/v202/xiong23a.html) are relevant
later references for controllers spanning morphologies. They motivate explicit
questions about structural information and parameter sharing; no graph,
transformer, hypernetwork, simulator migration or external pretrained model is
selected by this pilot.
