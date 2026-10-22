# Build a body, discover its motions

## Adopted goal

On 2026-09-05 the project owner asked us to define and pursue the broader goal:

> A child can assemble a creature from passive parts, motors, and sensors; wake
> it to discover motions its body supports; and rebuild it to discover new
> possibilities, including motion through passive joints and contact.

The body is an essential part of the behavior. Geometry, mass distribution,
connections, compliance, friction, and sensor placement can all matter. A
construction need not travel, and not every construction can perform every
motion. Rocking, swinging, reaching, rolling, and remaining still belong in the
possible vocabulary.

### Owner clarification, 2026-10-06

The first creature a child assembles should already benefit from substantial
training across simulated bodies and environments. Broadly pretrained
generalization is the starting competence, before the child supplies any new
experience. Assembly-specific sensing and a small amount of online adaptation
should specialize that competence; a rebuild should also retain useful personal
experience. Learning each new creature from scratch is a research control, not
the intended first-use experience.

This requirement does not imply that every arrangement can perform every task.
Evaluate frozen pretrained behavior on a new feasible assembly first, then
measure the additional benefit and cost of bounded adaptation separately. The
acceptable child's waiting time, supported assembly family and eventual device
compute budget remain to be measured. See the
[prospective pretraining pilot](GENERALIZATION_PRETRAINING_EXPERIMENT.md).

## First deliverable: build a swinging creature

A small native construction bench brings actual shape editing and passive
mechanics forward. Its first kit has an anchored motor, two articulated links
built from beam units, a free hinge, snap-on weight blocks, and a movable IMU.
The two hinges move in a vertical plane. One has a motor; the other moves only
through gravity and physical coupling. This is a bounded kit, not yet a general
connector-graph editor or a free-moving robot.

The child can extend either link, add/remove/relocate weights, move the sensor,
undo edits, and save/load the assembly. Applying a construction regenerates its
native geometry, masses, and inertias. The first fixture excludes self-contact
and ground contact; those mechanics are a subsequent milestone.

The creature tries five bounded six-second motor patterns from the same initial
state, and records local motion-sensor summaries and replayable trials. The
patterns vary timing, including sways, taps, and push/coast. This is a declared
exploration experiment with a handwritten controller, not learned optimal
control. It does not maximize movement or rewrite the attached sensor's reward.

A motion repertoire initially means an observed collection of trials with
summaries of observed sensor samples and replayable motor sequences. Several patterns may produce similar movements;
there is no claim that each card is a distinct skill. Later exploration can
use differences between those histories to choose informative next trials.

The controller receives physical IMU measurements (local angular velocity and
specific force), plus motor feedback. Rendered poses, construction labels,
passive-joint coordinates, and tip trails are diagnostic displays. They do not
enter action selection. Sensor preferences and noncompensable safety remain
separate, including the existing IMU preference against extreme motion.

## Success criteria

- Saved assemblies reproduce the same native construction.
- Adding a beam unit or relocating a block changes mass distribution and motion.
- The passive joint has no actuator and moves through the simulated mechanics.
- The same bounded exploration routine can be compared across at least three
  actual constructions, including the same weight near a hinge and at the tip.
- A zero-effort comparison measures movement due to gravity alone.
- Native observations and exact motor sequences replay deterministically.
- Pause preserves a run. Rebuilding clears old-body evidence. Returning to rest
  preserves the current body's recorded motions.
- Browser interaction, native tests, and recorded development comparisons are
  verified before claiming the milestone complete.

See [the implementation and evidence record](CONSTRUCTION_EXPERIMENT.md).

## Broader roadmap

### 1. Construction and physical consequences — this milestone

Establish the build / wake / observe / rebuild loop with the two-link kit. Make
added pieces mechanically meaningful and keep the assembly format saveable.
The central question is whether construction differences are visible and
measurable under the same motion probes.

### 2. Explore using experience

Use per-module temporal histories to recognize repeated outcomes, choose an
informative next probe, and retain a small repertoire of reproducible motions.
Compare adaptive exploration against this fixed-pattern baseline under equal
time and energy exposure. Measure repertoire coverage and reliability without
quietly making either a hidden sensor reward. Introduce explicit, versioned
exploration objectives if needed. Recheck behavior when evidence deteriorates.

### 3. Let the creature meet the ground

Extend the assembly model with useful connector choices, rounded feet/wheels,
contact, and eventually compliant elements. Release the anchored base in a
separate experiment and investigate rocking, rolling, and crawling. Start with
one tightly controlled contact mechanism, then expand the editable family.
Record failed constructions and stability limits as part of the evidence.

### 4. Choose motions through its senses

Investigate how an attached sensor preference selects or combines discovered
motions. A desired outcome may require an initially unhelpful action; longer
histories and horizons matter. Compare repertoire-based control with direct
control. Do not assume a useful motor sign stays useful after a rebuild.

### 5. Carry experience across rebuilds

Distinguish restarting the world, changing the assembly, and forgetting
experience. Transfer only knowledge whose module identity and exposure are
known. Test small geometry changes, removed parts, new parts, and new sensor
placements before claiming topology transfer. Fresh evaluation families need
explicit manifests and untouched evidence; this work does not execute the
pending frozen-v3 challenge.

### 6. Bring the kit toward physical toys

Replace reference parameters with measurements, characterize connectors and
sensors, and test simulation-to-hardware transfer within measured limits.
Observe children building and interpreting consequences; no usability or
hardware safety claims follow from a successful simulator demonstration.

These stages organize experiments, not a rigid product plan. Evidence and play
should change their order. The immediate implementation remains deliberately
bounded so there is a complete, inspectable construction experience to try.

## 2026-09-05: light-guided motion comes next

The owner asked to give the constructed creature a light sensor and a sun,
then authorized the experiment. We are bringing sensor-guided selection ahead
of contact mechanics, combining stages 2 and 4 in a continuous native loop.
[Find the light, then learn to stay with it](LIGHT_LEARNING_EXPERIMENT.md)
records the declared development protocol, observation boundary, learning
method and results. The construction workshop remains the place to change the
body; the new sunshine bench uses a separate light-enabled copy. Ground
contact, broader connector graphs and transfer across rebuilds remain later
questions. A moved sun preserves memory within a try; rebuilding currently
starts fresh memory.

## 2026-09-06: separate continuing learning from remembered feedback

The next adopted goal is to measure the contribution of continued value
updates after the sun moves. Six paired experiments use the existing two
bodies and seeds, reconstruct identical 60-second prefixes, and then compare
continued updates with frozen learned values. Both branches continue sensing,
history, exploration and feedback control. The pre-execution protocol is
[Does continued learning help after the sun moves?](LIGHT_ADAPTATION_EXPERIMENT.md).
This is an engine experiment within stages 2 and 4. It does not add contact or
expand the assembly family, and does not imply that frozen feedback is static
motion. Results will determine the next controller experiment.

The paired experiment is now complete: continued updates improved post-move
reward in all six pairs, with exact prior-v2 compatibility and complete replay.
Keep v2 as the measured baseline. The next proposed memory milestone is to
measure retention when the source returns to a familiar condition, with a
controlled saved-value comparison; it has not yet been run. See the adaptation
record for the raw pair differences, quiet-control context and limits.

## 2026-09-06: retention becomes the next goal

The owner authorized measuring whether adapting to a new light condition
preserves useful earlier values. The [retention protocol](LIGHT_RETENTION_EXPERIMENT.md)
uses a continuous A-to-B-to-A sequence. At the return, compare frozen current
values with frozen first-visit values from the same reconstructed physical and
sensory history, while a third branch continues learning. Value checkpointing
is a diagnostic tool here; automatic recognition and retrieval are still
future engine capabilities. The goal includes an honest result, exact replay,
provenance and a read-only comparison page.

The retention goal is now complete. In all six declared cases, saved first-A
values improved early return reward compared with values left after adapting
to B; median paired gains were +1.081243 and +1.084410 on the two bodies.
All 18 trials/replays and 15 native suites passed. See the
[separate retention results](LIGHT_RETENTION_RESULTS.md) for secondary outcomes,
exact evidence and limits. The locked protocol remains unchanged.

This supports investigating how a creature can recognize when earlier values
are useful. The proposed next bounded step is to test whether local sensor
history can distinguish conditions under identical motor probes, with entire
probe trials held out and no source/time cues as inputs. Observation-driven
selection among supplied memories would follow only if that test warrants it.
Useful saved values and autonomous retrieval are distinct milestones; the
current engine has completed the former diagnostic, not the latter capability.

## 2026-09-06: can its senses distinguish a familiar condition?

The owner adopted the next question as a goal: establish whether permitted
local sensor/action history distinguishes A from B under identical motor
probes, before building a memory selector. The
[locked context-decoding protocol](LIGHT_CONTEXT_EXPERIMENT.md) declares twelve
probe seeds, a split by whole probe sequence, two known bodies, instantaneous
and one-second history comparisons, brightness-only and no-light controls,
exact replay and a read-only result viewer. This is a supervised information
benchmark; source labels are never predictor inputs. The outcome will determine
whether to test selection among supplied memories or revisit the sensing/model
representation first. Results will be recorded separately from the protocol.

The sensor-context goal is now complete. History reached 93.75% and 95.45%
withheld accuracy, improving every probe comparison with current readings;
without light the result was exactly 50%. All 48 trials and exact replays passed,
as did all 16 native suites. The [results record](LIGHT_CONTEXT_RESULTS.md)
contains complete counts, probe comparisons, uncertainty limits and evidence.
The native decoder is frozen and supervised; it does not yet retrieve values
for the live controller.

The justified next question is whether observation-driven selection among two
supplied value memories improves closed-loop sensor reward. Keep the decoder
and candidate memories fixed, hide source identity and switch times from the
selector, preserve controller/physical history, and declare switching behavior
before execution. Compare with always-current and externally chosen recall,
including stationary controls because the selected policies change the movement
distribution. The high classification accuracy comes with narrow neighbor votes;
confidence, mixed-condition history and selection delay require explicit control
experiments. This proposal has not been executed. Autonomous memory discovery,
broader assemblies and contact remain later questions.

## 2026-09-06: identify a predictive state-space model

The owner adopted the next goal after requesting explicit control-theory
language. Identify a sensor-based transition model and measure whether its state
predicts future sensory responses under new command sequences. The
[locked identification protocol](LIGHT_PREDICTION_EXPERIMENT.md) defines output
coordinates, requested input, observable delay states, fitted A/B/C/D/c matrices,
regression, whole-trajectory splits and recursive forecast horizons. Three
assemblies are fitted separately, including a longer first beam. The existing
RL controller remains the measured control baseline; this experiment evaluates
prediction and does not claim a new model-based controller.

The first state encoding is deliberately explicit delay coordinates, with its
transition learned from local data. Compare current output, history with requests,
history without explicit requests, and persistence. Future actual observations
or governor-corrected efforts cannot enter a forecast. Results will determine
whether a bounded closed-loop model-based-control experiment is justified.

This goal is complete. The [prediction results](LIGHT_PREDICTION_RESULTS.md)
record 72 native probes and 72 exact replays, immutable numerical A/B/C/D/c
matrices, independent regression refits, and 552 withheld forecast origins.
Both the 142-state history model and 12-state current-output model passed every
prospective gate on all three separately fitted bodies. History helps at .1/.4 s;
its necessity is not established, and the current-output model has slightly
lower one-second error on the default body. Light-specific errors remain material.

The justified next question is whether a bounded receding-horizon policy using
these frozen models earns better actual sensor reward than the unchanged RL
baseline. Compare both passing representations, declare the objective, horizon,
decision cadence, action constraints and fallback before running, and measure
reward, energy, safety and action-ranking error on the new closed-loop movement
distribution. This is a proposed next experiment, not a completed control result.
Do not replace the RL baseline or infer stability from prediction accuracy.
