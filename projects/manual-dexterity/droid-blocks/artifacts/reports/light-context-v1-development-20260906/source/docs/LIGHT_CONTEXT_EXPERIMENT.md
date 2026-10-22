# Can local sensor history distinguish familiar light conditions?

## Goal and protocol declared before collection

Adopted 2026-09-06. Determine whether a small decoder can distinguish the two
familiar light conditions using only permitted local sensor/action information,
under matched motor probes. Compare current readings with one second of causal
history, keeping entire probe sequences out of the training examples. This is
a supervised decodability benchmark for the next memory-selection decision;
it is not autonomous recognition, retrieval or a proof of state observability.

This protocol is locked before any benchmark trial is collected or scored.
Record outcomes in LIGHT_CONTEXT_RESULTS.md, not by changing this file.

## Physical trials and splits

Use the exact two known development assemblies from the archived light-learning
v2 report: the default sun-seeker and its side-weight variant. Keep native
MuJoCo physics, module catalog, rewards and protection unchanged. The fixture
is still anchored, planar, underactuated and collisionless. The existing live
light and construction sessions are separate and remain preserved.

A has source position [0.30,0,0.20] m; B has [-0.30,0,0.20] m. Both have intensity
1000 lux. Each trial has a constant source and lasts 12 seconds: 600 control
steps of .02 seconds, each with ten .002-second physics steps. Initialize every
trial as rebuild(assembly), set_sun(A or B), then reset(). Reset reacquires the
t=0 light sample under the chosen source; merely moving the source would retain
the prior queued sample. There are no in-trial moves or oracle cues.

Use the existing LightControl random baseline without alteration. Seeds 101
through 112 prescribe twelve probes: .10 seconds quiet, then requests from
{-0.15,0,+0.15} selected every .40 seconds with the existing mt19937_64 output
modulo-three mapping. The existing local governor applies validity, effort,
slew and soft shaft-speed feedback. Requests and all emitted effort bits must
match exactly between A and B for the same body and seed. Light intensity is
not an input to the baseline's random choices or mechanical governor.

Training seeds are 101-108. Withheld seeds are 109-112. Both bodies and both
conditions for a seed belong to the same split; no windows from a withheld
probe enter training. Each body has its own decoder training examples: this
experiment does not evaluate body transfer. Each seed has an A/B pair, so
source conditions are balanced at every sequence position. These are new
probe sequences on exposed development bodies, not a fresh unseen-world claim.

Run all 48 trials (2 bodies x 12 probes x 2 conditions) and independently replay
each entire trial. Do not add seeds, tune the decoder, select windows or rerun
trials to improve an outcome. Preserve failed trials as failed evidence.

## Sensor boundary and examples

Record a frame after steps 1+5k, k=0..119: t=.02+.10k seconds. At those endpoints
the newly delivered light sample has age .02 seconds. A history window contains
eleven such frames, spaced .10 seconds apart, covering one second of distinct
light acquisitions. Never interpolate future observations or include a sample
before its delivery. Both IMU and light must be valid, with their native
relative acquisition/delivery timing checked for consistency.

Examples use frame indices 10+5j, j=0..21: t=1.02,1.52,...,11.52 seconds. There
are 22 examples per trial, 352 training examples per body and 176 withheld
examples per body. Overlapping histories are not independent observations;
summary dispersion is reported across whole withheld probe pairs.

The frame has exactly twelve finite numbers in this fixed order:

| Index | Permitted quantity | Fixed transformation |
| ---: | --- | --- |
| 0 | Last executed effort | effort / .15 |
| 1 | Local motor encoder | sin(position_rad) |
| 2 | Local motor encoder | cos(position_rad) |
| 3 | Motor velocity | velocity_rad_s / 10 |
| 4 | Motor current | current_a / .5 |
| 5-7 | IMU specific force x,y,z | each / 50 m/s^2 |
| 8-10 | IMU angular velocity x,y,z | each / 10 rad/s |
| 11 | Delivered light reading | illuminance_lux / 1000 |

Scales are declared physical constants, not fitted using any data. Do not clip
these features. Invalid or nonfinite frames fail validation. Module identifiers
may validate the observation contract but never enter a feature. Source fields,
body/seed/trial identifiers, labels, absolute timestamps, sequence counters,
phase, rewards, world pose, passive joint state and simulator diagnostics are
excluded. Timing serves delivery validation and causal window selection only.

Require exact A/B equality of all non-light observation content, including IMU,
actuator feedback and light timing metadata. Only the light observation channels
may differ. Require canonical equality of mechanical public-state projections
at every step after removing sun, global reward, instantaneous-light diagnostic
and delivered light observation channels. Keep sensor position/direction,
geometry, joint diagnostics, motor, IMU, energy, safety and time in that check.
This tests that the lamp cannot reveal its identity by changing the mechanics.

## Fixed decoder and controls

Use a dependency-free native five-nearest-neighbor decoder with immutable copied
training examples and the same distance rule for four declared feature views:

1. `instantaneous`: the latest twelve-number frame.
2. `history`: all eleven frames concatenated latest to oldest, 132 numbers.
3. `light_only`: only the latest normalized light reading, one number.
4. `no_light_history`: the history with index 11 removed in every frame,
   121 numbers. This is the strict negative control.

Distance is the squared Euclidean distance divided by feature dimension. Find
the fifth-nearest distance and include every training example at exactly that
distance or closer, so file order does not break equal-distance ties. Return
p(A)=(number of A neighbors+1)/(number of included neighbors+2), and predict A
when p(A)>=.5. This is a smoothed neighbor vote, not calibrated confidence.
Source labels are supplied only as offline training targets and evaluator truth;
the prediction method receives only the numeric feature vector. There is no
parameter search, fitted scaling or use of withheld examples for model choice.

After collection, construct the four models using only the declared training
examples for each body. Save immutable training-models.json with provenance
metadata outside the numeric predictor, and record its SHA-256 before scoring
any withheld examples. The native report and independent validator must prove
all model examples and labels originate from the training split. No prediction
updates the models. Archive the complete models and all resulting predictions.

The no-light control must emit identical p(A) for paired A/B examples and have
exactly .5 balanced accuracy for every withheld probe and body. Failure of this
control invalidates evidence; it is not a negative classification result.
The brightness-only control reveals whether a simple current light level
already suffices under these probes. History has more coordinates than the
instantaneous view, so a gain does not alone isolate why that representation
works better or identify the minimal required memory.

## Measures and predeclared decision rule

Primary: absolute withheld balanced accuracy of the history decoder separately
for each body, with the predeclared paired history-minus-instantaneous balanced
accuracy comparison. A/B are balanced in each probe, so ordinary accuracy and
balanced accuracy coincide. Report every withheld probe pair and the equally
weighted mean across the four probe pairs. Also report their min/max range;
do not attach independent-window confidence intervals or treat replays as new
trials. The four probes provide limited information about broader uncertainty.

Secondary: confusion counts, mean Brier score (p(A)-target)^2, fraction of votes
in the inclusive [.4,.6] ambiguity band, all four decoders' accuracies, per-probe
history-minus-instantaneous differences, and prediction curves. Report native
reward, energy, current/temperature/speed maxima, vetoes and feedback flags as
measurements of the probes, not classifier optimization targets.

A decoder clears the proposed next-control-experiment threshold only if its
mean balanced accuracy is at least .75 on EACH body and its accuracy is at
least .65 on EVERY withheld probe pair on each body, with all integrity checks
passed. The declared primary readiness check applies to history. Also report
whether instantaneous/light-only clear the same rule: if simpler current
readings suffice, do not claim history is necessary. These thresholds are an
engineering decision for a subsequent bounded experiment, not significance
levels or a guarantee of reliable control.

Passing supports testing observation-driven selection among supplied value
memories under undisclosed and changed switch times, with stationary controls.
It does not implement that selection or independent memory creation. Failing
to pass limits this fixed decoder under these probes; it cannot establish that
the sensor interface contains no usable information. Arbitrary geometries,
contact, noise, moved sensors and hardware remain outside this result.

## Integrity and completion

Use a new exclusive output directory; refuse an existing directory. Archive
this locked protocol, native sources/headers/tests, independent validator,
build/runtime definitions, original v2 assembly reference and its source, module
catalog and frozen original inputs with SHA-256 checks. Preserve traces and
report source/trace/model provenance. Failed collection or scoring must leave
its partial evidence and errors, without certifying completion or replay.

Replay must match exact effort bits, full emitted transitions, observation
traces and final public physical state. The independent validator reconstructs
feature projection, causal windows, training membership, nearest-neighbor
votes, predictions, aggregate accounting, controls and threshold decisions.
Recorded native mechanics hashes are compared, not used to reconstruct hidden
MuJoCo internals. The original frozen rover-v3 challenge remains unexposed.

Success is an interpretable result, tested native sensor/decoder boundaries,
reproducible evidence, and a read-only visual report with a justified next
engine decision. High classification accuracy is an empirical outcome, not a
condition for completing this goal.
