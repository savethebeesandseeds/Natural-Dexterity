# Moving the motor: prior learning did not reliably help

2026-09-07. This is the outcome of the [locked actuation-transfer
experiment](ACTUATION_TRANSFER_EXPERIMENT.md), stored in
`artifacts/reports/actuation-transfer-v1-development-20260907`.
The [recorded comparison](http://127.0.0.1:43117/transfer.html) lets the reader
watch all three branches with a shared clock and inspect every case and seed.

The experiment's improvement gate failed. Pretraining followed by 64 seconds
of new simulated interaction did not reliably outperform learning from scratch
after the motor moved from the base hinge to the elbow. New updates repaired
some transferred behavior, but that recovery did not establish useful rapid
adaptation. This is an exploratory result on one anchored mechanism, not a
verdict on PPO, model-free control, or modular robotics in general.

## What changed in the engine

The native assembly now supports `powered_hinge: 0` or `1`. The actuator,
encoder, motor dynamics and reflected armature follow the selected hinge; the
other hinge remains passive. Omitted/zero selection preserves the earlier base
layout. The two-link connection graph and collisionless planar physics remain
the same. This is actuation-layout transfer, not a topology change or walking.

A separate host-side Stable-Baselines3 2.9.0 PPO learner controls isolated
native worlds over a batched stdin bridge. Its actor and critic each use two
64-unit tanh layers, together containing 34,820 parameters. Both see the same
204-element local history. Neither receives body geometry, hinge labels, sun
coordinates or a privileged state. The categorical policy requests negative,
quiet or positive effort every 0.1 seconds; native governing and safety still
run every 0.02 seconds. The light and IMU reward is unchanged.

The transferred object is the complete actor/critic weight set. Adaptation
updates all those weights with a fresh optimizer; no frozen embedding, learned
world model or recurrent adaptation module was added. The earlier live Expected
SARSA controller and the separate affine predictor remain available. See the
[control equations](CONTROL_MODEL.md) and [training boundary](../training/README.md).

## The declared comparison

Three independent source seeds each received 98,304 decisions on 16 base-motor
conditions varying beam lengths, a block and left/right illumination. All
source checkpoints were completed and sealed before target collection.

Each checkpoint then faced six matched conditions with the motor at the elbow.
Target geometries and lighting had appeared in the source domain; the actuation
layout had not. Frozen transfer, adapting transfer and scratch each received
exactly 640 new decisions, or 64 simulated seconds. A fourth branch evaluated
the saved policy on its original motor layout. Fixed quiet and motion probes
provided separate controls. An explicitly elbow-exposed development model used
its own body and seed; its weights and data never entered transfer branches.

The sole primary contrast is adapting minus scratch native reward, averaged
over six cases within each source seed, then summarized by the median of those
three seed means. A pass required positive median, at least two positive seed
means, and complete safe target runs. No checkpoint, seed or budget was changed
in response to outcomes.

| Source seed | Adapt − scratch | Frozen − scratch | Adapt − frozen | Adapt − quiet | Source − source quiet |
| --- | ---: | ---: | ---: | ---: | ---: |
| 101 | +1.351 | −0.089 | +1.440 | −0.819 | +0.214 |
| 202 | −0.311 | −3.440 | +3.129 | −2.442 | +0.617 |
| 303 | −4.802 | −19.026 | +14.224 | −6.891 | +0.124 |

These are differences in raw reward accumulated over 64 seconds, not
percentages. The primary median is **−0.310903**, with **one of three** positive
seed means. The other contrasts describe the result; they do not replace the
failed primary gate. With only three training seeds, this is a small controlled
experiment and does not establish a population-level significance claim.

The predeclared early windows also do not establish an early advantage:

| New simulated experience | Median seed-mean adapt − scratch |
| --- | ---: |
| 6.4 seconds | −0.211 |
| 12.8 seconds | −0.527 |
| 32 seconds | −0.797 |
| 64 seconds — sole primary | −0.311 |

All 18 frozen/adapting pairs have identical first 64 sampled actions and full
native prefixes. Those are the first 6.4 seconds, before the first update.
There are ten target PPO updates, but only the first nine can affect reward
inside the measured interaction window. No extra evaluation after the final
update was added to the reported score.

## What the controls tell us

New learning improves the seed-mean reward relative to frozen transfer for all
three seeds. In seed 303 the improvement is large, but much of it recovers
behavior that the transferred weights made worse. Recovery from negative
transfer and an advantage over scratch are different findings.

The source policy beats its matched quiet control in only three of six cases
for every seed. Its average advantage is small. On the rebuilt body, adaptation
beats quiet in four of eighteen case/seed pairs and is below quiet on average
for every seed. None of the eighteen pairs meets both the matched source-quiet
and target-quiet useful-behavior conditions.

Every fixed target probe suite is led by quiet: continuous negative, continuous
positive and the two declared rhythms all earn less. These probes therefore
provide no witness of a substantial improvement margin above rest. This does
not prove quiet optimal; the learned branches occasionally exceed it. The
isolated elbow development policy likewise beats quiet on one light side and
loses on the other.

As a posthoc description, quiet requests account for 64.21% of frozen actions,
71.45% of adapting actions and 33.36% of scratch actions. A quiet request does
not mean the mechanism stops: inertia, gravity, passive dynamics and residual
motor state remain. These counts do not establish why a policy earned its
return, but they reinforce the need to distinguish useful motion from high
reward in an already illuminated hanging pose.

## Cost and evidence

The three source runs use 294,912 decisions: 8.192 aggregate simulated hours,
or 2.731 hours per checkpoint. Their learning loops took 159.344, 123.000 and
132.187 seconds on one host CPU thread. The separate elbow development run
adds 32,768 decisions and 49.234 learning-loop seconds.

All 112 trial/control runs completed their 640-decision budgets with no recorded
safety stops or execution errors. Their exact native replays all passed. The
main collection totals **399,360 decisions**, **1,996,800 native control steps**
and **19,968,000 physics substeps**, including development, source learning and
all trials. Integrity replays, the separate CartPole reference and non-learning
preflights are additional work and are not counted as independent trials.

Simulator time pauses during optimizer updates. This measures interaction
efficiency, not real-time adaptation on an unpausable physical toy. Native
replay reconstructs physical responses to a saved action/reset tape; it does
not replay the neural optimizer. Net electrical energy is a signed electrical
integral and may decrease during regenerative intervals; it is not a reward
term in this experiment.

The separate CartPole reference retains its failed predeclared improvement
gate: trained mean 409.2 versus untrained mean 333.2, a gain of 76 against a
required 100. Its absolute-return, finite-parameter and checkpoint-reload
checks passed. There was no retuning or replacement seed. The native study was
explicitly locked as exploratory after that result and before native learning.

Before collection, all 18 native CTest suites, eight bridge test groups and
five adapter checks passed. The original base-layout fixtures reproduced their
pre-change hashes. The protocol and 89 source/dependency records were archived
before learning. The independent descriptive analysis verifies all 112 trace
hashes, native reward sums/counts, matched prefixes and parameter invariance.

All four training action/reset tapes also replay exactly, covering 41,031 bridge
requests. The independent export audit passes with zero issues and exports 72
playable traces; its scientific transfer gate remains false. Browser checks
verified synchronized play/pause, end-of-run values, case/seed changes, the
optional original-layout view and a 390-pixel phone layout without horizontal
overflow. Playback starts paused. The browser reported no errors during the
real-data checks, and the temporary viewport override was reset.

The completed audit, independent comparison and UI/preservation records are
retained in the batch's `verification/` directory. The source archive and all
raw learning/trial records remain unchanged.

Protocol SHA-256:
`82d83fe5004304bba9f8b17ac8710a8212fc9c5ec729908f05e14f98b08fcef1`.
Independent comparison SHA-256:
`6120e0b2f339ff6dc320d751b7f6512c10482e07180018fb52c1862c042ff077`.
Full audit SHA-256:
`a7705aee977e158a150ebee8379e7e669687ca0a8520bdd8efc14bd14258d7e1`.

The 466 protected earlier artifact files remain unchanged. The live light
response is byte-identical to its pre-goal snapshot. The construction response
differs only in `session_id` (2 to 3); every remaining byte, including the
completed physical/learner state, matches. Read-only GET requests cannot cause
that identity increment; its origin was not established. Both the strict
byte-comparison failure and its exact one-field explanation are retained in
`build/actuation-transfer-qa/preservation-check*.json`. No session was reset to
hide that difference, and the existing container was not restarted.

## The next justified question

Before selecting a larger network or a new optimizer, establish a stronger
behavioral benchmark: **with the same native reward and permitted sensors, can
the original body reliably learn a useful advantage over rest across declared
conditions?** Separately verify an attainable advantage on the rebuilt body.
Choose any new lighting/reset distribution prospectively and retain this
completed batch unchanged. This changes the experimental conditions rather
than adding a reward for motion the owner did not request.

Once there is dependable source competence and a measurable target improvement
margin, compare plain weight reuse with a controller trained to adapt across
body variations. Shared structure and memory remain candidates; this experiment
does not identify whether optimizer behavior, insufficient information or the
limited training distribution caused the failure. Ground contact, standing,
recovery and the T-Rex-to-snake challenge still require new physical capabilities
and their own held-out tests.
