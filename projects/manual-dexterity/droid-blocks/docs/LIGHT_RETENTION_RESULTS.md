# Retention results: when the sunshine returns

Completed 2026-09-06. Saved first-visit value weights outperformed the values
left after adapting to the other light condition in all six declared return
comparisons. The appropriate checkpoint was supplied by the experiment. The
creature has not yet learned when to retrieve a memory.

The [protocol declared before execution](LIGHT_RETENTION_EXPERIMENT.md) remains
byte-for-byte unchanged. This file records outcomes separately so the locked
protocol and its archived hash remain reproducible. The
[interactive record](http://127.0.0.1:43117/retention.html) shows all six cases,
three received-light curves, the return zoom, measurements and provenance.

## What was implemented

The native learner and its control wrapper now expose a typed, immutable value
checkpoint. Capture copies its 168 value weights and compatibility metadata;
restore changes only those weights. This is a diagnostic boundary for the
existing `history_expected_sarsa_v2` controller, not a new learning algorithm.

Tests cover capture without mutation, pending-observation and initialization
preconditions, incompatible module identities, exact restoration, unchanged
nonparameter state, continuation of an unfinished option, and subsequent
frozen-weight behavior. Restore preserves RNG, clocks, physical state, sensor
queues, reward/action history, replay, eligibility traces, local feedback and
the freeze flag. The HTTP control API has no checkpoint or recall action.
Checkpoint compatibility checks do not establish transfer across body geometry;
the runner separately verifies assembly identity for this experiment.

Each of six known development cases (two bodies, seeds 1-3) ran three branches
for 180 simulated seconds. The sun moved A-to-B at 60 seconds and B-to-A at
120 seconds. All branches learned identically through 120 seconds. The return
compared frozen current weights W120, frozen first-visit weights W60, and
uninterrupted learning. Both frozen branches still sensed, explored and used
feedback. The experiment changed no learner tuning, reward, body or seed.

## Declared primary result

Integrated SUM sensor reward over 120-140 seconds. Delta is first-visit frozen
minus current frozen; each row is one paired case. Rounding is display only.

| Body | Seed | Current frozen | First-visit frozen | Paired delta |
| --- | ---: | ---: | ---: | ---: |
| Default | 1 | 16.9406 | 18.0406 | +1.1001 |
| Default | 2 | 17.0516 | 18.1328 | +1.0812 |
| Default | 3 | 16.9573 | 17.9754 | +1.0181 |
| Side weight | 1 | 17.0145 | 18.0544 | +1.0399 |
| Side weight | 2 | 17.0037 | 18.2184 | +1.2147 |
| Side weight | 3 | 17.0907 | 18.1751 | +1.0844 |

The predeclared within-body median paired differences are **+1.0812430379**
(default) and **+1.0844103026** (side weight). All six differences are positive.
For descriptive scale, the median of each pair's delta divided by its current
frozen return is +6.3410% and +6.3450%, respectively. Percentages are secondary;
the declared measure is the raw paired integrated reward difference.

The early gain comes predominantly from the light sensor's reward vote.
The light/IMU contributions to each primary paired difference are:

| Body | Seed | Light reward delta | IMU reward delta |
| --- | ---: | ---: | ---: |
| Default | 1 | +1.137024 | -0.036929 |
| Default | 2 | +1.127524 | -0.046281 |
| Default | 3 | +1.024493 | -0.006377 |
| Side weight | 1 | +1.011855 | +0.028018 |
| Side weight | 2 | +1.273387 | -0.058712 |
| Side weight | 3 | +1.112794 | -0.028384 |

## Secondary outcomes and continuing-learning context

All entries below are medians of paired differences within that body.

| Paired measure | Default | Side weight |
| --- | ---: | ---: |
| First-visit minus current: full-return reward, 120-180 s | +3.548211 | +3.605855 |
| First-visit minus current: early mean light, 120-140 s | +168.963 lx | +167.700 lx |
| First-visit minus current: late mean light, 160-180 s | +183.543 lx | +209.042 lx |
| First-visit minus current: return electrical energy | +0.739429 J | +0.679995 J |
| Continuing minus current: full-return reward | +1.777658 | +1.462263 |
| Continuing minus first-visit: full-return reward | -1.770553 | -1.998340 |

First-visit values beat current values over the full return in all six cases
(deltas +3.4095 to +3.6934). Continuing learning beat current frozen values in
all six full-return comparisons, but remained below recalled frozen values
in every case (deficits 1.2526 to 2.7825 reward units). These are secondary
contrasts and do not replace the primary frozen-versus-frozen comparison.
In default seed 3, continued learning eventually exceeded recalled values in
late mean light (554.540 versus 524.377 lx). This result does not justify
permanently freezing a useful memory.

First-visit recall used more return energy in four cases and less in two.
It is therefore a reward improvement, not a demonstrated efficiency gain.
Energy is measured separately and has no added reward vote.

Across all 18 trials, maximum measured current was 0.374999998 A, temperature
25.644887 C, and motor speed 6.958852 rad/s. There were zero native veto steps
and zero feedback-flag steps over the entire runs. The 3 rad/s feedback target
is soft and can be exceeded through backdriving; it is not a hard speed limit.
These are simulator measurements, not a hardware safety certification.

## What this tells us about the engine

In this fixture, adapting a shared value function to B left weights that were
less useful on returning to A than the saved W60 weights. The comparison
isolates the consequence of replacing values from the same B-derived physical
and sensory starting history. It is evidence of local interference in the
usefulness of learned values; it does not establish global catastrophic
forgetting, identify which features caused it, or prove that a bank of memories
is the best solution. Changes in feature representation, replay or state
estimation remain other possible approaches.

Saving useful values has measurable potential here. Selecting the appropriate
memory is still an unsolved part of the controller. The runner knew which
checkpoint came from A; the policy received no source position, identity or
change cue. Comparing the speed of first and second visits alone would not
answer this question because their starting physical and sensory states differ.

Keep v2 as the measured baseline. The proposed next bounded engine experiment
is to establish whether the allowed sensor interface can distinguish the two
conditions under matched actions, before implementing a retrieval selector:

1. Declare paired A/B probe trials with identical initialization and identical
   bounded motor sequences. Balance conditions at each sequence position.
2. Compare a small decoder using instantaneous local observations with one
   using recent sensor/action history. Exclude absolute time, phase, step,
   source fields and candidate identity; retain relative sensor ages.
3. Hold out entire probe sequences/trials, not adjacent time windows. Report
   errors and uncertainty on each body, not just pooled accuracy. Source labels
   can be training targets and evaluator truth, never controller inputs.
4. If this supervised decodability test succeeds, separately test selecting
   among supplied opaque frozen value candidates using local observations,
   under changed and undisclosed switch times. Compare latest-only, uniform
   selection and externally selected recall. Measure transition reward and
   inappropriate switches, including stationary-source controls.

Decodability would show available information, not successful autonomous
control. Selection from supplied candidates would still not demonstrate
independent memory creation. These are proposed follow-ups, not completed
results or an adopted held-out protocol. The current six cases are exposed
development evidence; no unseen-world, arbitrary-body, contact or hardware
claim follows. The original frozen rover-v3 challenge remains unexposed.

## Evidence and verification

Native report:
`artifacts/reports/light-retention-v1-development-20260906/report.json`

SHA-256:
`aff06275e4380d9363bd61a722f5e806b40d6e5f04178748d900ef7e1d93478b`

The report archives 75 source/provenance files, including the locked protocol,
native code and tests, validator, runtime/build definitions and original v2
reference evidence. All 18 trials and 18 independent full replays completed;
zero cases failed. Full transition records, exact commands, final controller
state and public physical snapshots matched on replay. All branches reproduced
the original v2 120-second commands, observation hashes and phase metrics.

At 60 and 120 seconds, all 36 full-controller fingerprint comparisons and all
36 corresponding parameter comparisons matched the prior adaptation report.
Restore preserved the nonparameter fingerprint exactly, and both frozen
branches verified unchanged weights after all 3,000 return transitions. The
first five return steps finished the same existing option in every branch.
Public snapshots and reconstruction are not a direct save/restore comparison
of hidden MuJoCo internals. The independent JS validator checks recorded native
fingerprints; it does not reconstruct private learner state from a hash.

The aggregate native target built, the final runner rebuilt, and all 15 CTest
suites passed in 11.16 seconds. The validator accepted an untouched owned copy
and rejected eight altered fixtures: wrong primary difference, changed frozen
weights, changed nonparameter restore state, failed replay, incomplete report,
unavailable final snapshot, inconsistent reward components, and modified
archived source. Signed zero was preserved while serializing test fixtures.
The runner refused an occupied owned output fixture and preserved its sentinel.

Browser checks covered missing-data recovery, body/seed selection, whole/return
plot ranges, within-body medians, provenance, and normal/390-pixel layouts.
No browser warning/error logs were present after reload. Both completed live
light and construction session responses remained byte-for-byte unchanged.
The same container remained running and healthy without a server restart.
The original model, catalog, frozen-v3 inputs, prior v2/adaptation reports,
adaptation viewer data and locked retention protocol retained their hashes.
QA logs and verification records are under `build/light-retention-qa/`.

To reproduce in the documented existing development environment, choose a new
output directory. Never change the locked protocol or overwrite prior evidence.

```sh
cmake --build build/native --target droid-native-validation -j 4
ctest --test-dir build/native --output-on-failure
./build/native/droid-light-retention --output artifacts/reports/NEW_DIRECTORY
node tools/summarize-light-retention.mjs artifacts/reports/NEW_DIRECTORY/report.json
node tools/export-light-retention-view.mjs artifacts/reports/NEW_DIRECTORY/report.json
```

The last command derives `web/retention-data.json` after independent validation;
it does not modify the native report. Reproduction requires the exact source,
reference and runtime archived with this result.
