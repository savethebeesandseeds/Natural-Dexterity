# Does adapting preserve earlier useful experience?

## Adopted goal and protocol, declared before execution

Adopted 2026-09-06 after the owner asked to define and pursue the retention
goal. Determine whether values learned during a first light condition remain
useful after adapting to another condition and returning. Preserve a measured,
reproducible answer whether older or newer values work better. This experiment
measures the usefulness of stored value parameters, not automatic memory
retrieval or transfer to a different body.

### World, experience and branches

Use the same two known development assemblies and learner seeds 1, 2 and 3
as the archived light-learning v2 report. Six cases, each with three branches:
18 trials of 180 simulated seconds, each independently replayed. No additional
seeds, bodies, tuning or retries to improve outcomes. The frozen original
rover-v3 challenge remains outside this work.

The sun has intensity 1000 lux throughout. A is [0.30,0,0.20] m and B is
[-0.30,0,0.20] m. Start in A, switch to B after 3,000 control steps (60 s),
and return to A after 6,000 steps (120 s). Stop after 9,000 steps (180 s).
The existing .002 s physics, .02 s control, .4 s options, quiet initialization,
56 features, discounts, updates, exploration, motor envelope, rewards and
protection all stay fixed. The creature is still the anchored two-link kit
without contact. The policy receives no sun coordinates, identity or change cue.

At 60 s, immediately before moving the source to B, capture a typed immutable
checkpoint of the exact learned value weights. Capture must not change any
controller state. Continue learning through B. At 120 s, immediately before
returning to A, reconstruct three identical complete prefixes with matching
commands, full emitted transitions, public physical snapshots, learner/driver
state fingerprints, histories and current weights. The first 120 s must also
reproduce the original v2 learner commands, observation hash and phase metrics
exactly in this runtime. A public snapshot is not a full native MuJoCo save;
deterministic reconstruction, identical complete prefixes and replay provide
the corresponding evidence without implementing a world restore mechanism.

The branches at the return are:

1. `continued`: retain current values and keep updating throughout the return.
2. `current_frozen`: retain the values reached after B and freeze their writes.
3. `recalled_frozen`: replace only the value weights with the saved first-A
   checkpoint, then freeze their writes.

Both frozen branches still sense, update observation/reward histories, select
actions, explore, apply local feedback, and process eligibility/replay
bookkeeping. A nonparameter-state fingerprint must remain exactly unchanged
by checkpoint restoration. The checkpoint does not replace physics, pending
sensor packets, RNG state, clocks, replay, traces, reward history or an
unfinished option. No option is interrupted at either sun move. Freezing both
value comparisons prevents additional learning during the return from masking
the retention question; the continuing branch separately shows practical
readaptation. Restore metadata and checkpoint selection stay in the experiment
runner, not the deployed policy or the browser's control API.

### Measures fixed before observing the outcome

Primary: integrated SUM sensor reward in the first 20 s after the return
(120-140 s), `recalled_frozen minus current_frozen`. Positive means the saved
first-visit weights produce a better return from this same starting state;
negative means the newer weights do better. Report every case and the median
paired difference within each body. Do not replace the median of differences
with a difference of medians, or select a different window after the run.

Secondary: reward over the full return (120-180 s), early (120-140 s) and late
(160-180 s) received light, electrical energy, sensor reward components,
maximum current/temperature/speed, native vetoes and feedback flags. Partition
the return into 120-140, 140-160 and 160-180 for accounting. Report continuing
minus each frozen branch separately; these contrasts include online updating
and are not the primary frozen-value retention comparison. Curves at .1 s
spacing show delivered light and cumulative reward, not independent trials.
Energy remains a measurement, with no additional reward vote.

These six exposed development cases provide no population-level significance
or unseen-world claim. A saved-weight advantage indicates lost usefulness of
the newer parameter set on these return trajectories, not that every form of
memory was erased. Absence of an advantage does not prove retention in every
state. First and second visits enter with different physical and sensory
histories, so faster second visits alone cannot establish remembering. The
experiment deliberately gives the recalled branch the appropriate checkpoint;
it does not demonstrate that a creature can recognize when to retrieve it.

### Integrity and completion

Use a new output directory and refuse overwrite. Archive the exact protocol,
native sources, tests, validator, build/runtime definitions, module catalog,
frozen inputs and original v2 reference evidence with SHA-256 checks. Compare
commands by exact binary64 bits and complete replay transitions/public states
by canonical serialization, preserving signed zero. Verify saved checkpoint
identity at capture and restore, identical nonparameter state across restore,
and unchanged frozen weights after every post-return transition. Independently
validate report accounting, prefixes, reference compatibility and provenance.

All trials, including failures, remain in evidence. If a correction becomes
necessary after running, preserve that report and its source rather than
silently replacing it. Success is an interpretable retention result, a tested
value-checkpoint boundary, and a reviewable result page with a justified next
engine decision. Successful retention is an empirical outcome, not a required
acceptance condition. The live creature and earlier reports remain preserved.
