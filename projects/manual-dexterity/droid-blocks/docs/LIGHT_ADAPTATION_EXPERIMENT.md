# Does continued learning help after the sun moves?

## Adopted goal and protocol, before execution

Adopted 2026-09-06 after the owner asked to define and pursue the next goal.
Determine the contribution of continued action-value learning to adaptation
in the existing constructed creature. Preserve a complete reproducible result,
whether continued learning helps, hurts, or has inconsistent effects. This is
a bounded engine experiment before a new controller or more elaborate body.

### Fixed comparison

Use the same two exposed development bodies as light-learning v2: the default
light-enabled [3,3] assembly and that assembly with a .012 kg block at segment
1, slot 0, side +1. Use learner seeds 1, 2 and 3 on each. Six pairs, twelve
120-second runs, each independently replayed. No parameter selection, extra
seeds, retries to improve outcomes, or new physical scenarios in this protocol.
The frozen original rover-v3 challenge remains outside this work.

Each branch starts from the same deterministic world initialization and seed,
then executes the unmodified v2 learner for 3,000 control steps (60 seconds).
Compare the complete command/observation prefix, public physical state, full
controller-state fingerprint and learned-parameter fingerprint. Reconstructing
the same prefix avoids adding a simulator save/restore mechanism. Public state
is not a complete native MuJoCo checkpoint; deterministic reconstruction,
identical full prefixes and independent replay provide the corresponding
reproducibility evidence. Require these checks before interpreting a pair.

Immediately before action 3,001, move the sun from [0.30,0,0.20] to
[-0.30,0,0.20] m, intensity 1000 lux, in both branches. One continues learning;
the other freezes its current value parameters. Freeze occurs between a
completed act/observe pair, and can fall within a .4-second option. Do not end
or restart that option. Its accumulated reward, current motor command, world,
sensor delivery queues and all history continue across the intervention.

Freezing suppresses both online and replay writes to value parameters. It
preserves sensing, feature computation, reward history, eligibility/replay
bookkeeping, seeded random draws, exploration schedule, action selection,
local feedback governor, and native protection. Thus frozen values can still
produce changing actions in response to new measurements. Reset enables
learning again; the experiment uses a one-way freeze within a run. The
controller receives no sun coordinates or sun-move notification.

The existing v2 initialization, 56 features, .92 option discount, .4-second
options, .02-second control and .002-second physics, .15 effort requests,
.35 cap, .1 slew and 3 rad/s soft governor stay fixed. No contact is added.
Require every continued branch to reproduce the corresponding preserved v2
commands, complete observation hash and summed return exactly in this runtime.

### Measures declared before reading the new outcome

Primary: paired difference (continued minus frozen) in integrated summed
sensor reward during seconds 60-120. Report all six differences and their
median within each body. Do not substitute a difference of medians for the
median of paired differences. The report horizon is undiscounted; the online
learner retains its existing discounted continuing objective.

Secondary: mean delivered light over 60-80 and 100-120 seconds, post-move
electrical energy, reward-component sums, maximum current/temperature/speed,
native vetoes and feedback flags. Report every run, including failures. Energy
is a cost measurement, not an additional reward vote. Curves at .1-second
spacing are descriptive displays of delivered readings, not independent trials.

No population-level significance or universal learning claim follows from
three known seeds on each of two known bodies. This intervention estimates
the total effect of continuing value updates in this particular moving-sun
scenario, including consequent changes in actions and visited states. It does
not separately identify the value of learning specifically caused by moving
the sun versus the effect of further training in an unchanged world; that
would need an additional stationary-sun control.

### Integrity and completion

The output directory must be new and must refuse overwrite. Archive exact
source, protocol, relevant build/runtime definitions, frozen inputs and the
reference v2 report with SHA-256 checks. Replay each entire controller and
observation trace with its same intervention. Require frozen parameter hashes
to remain identical after every post-freeze transition, and confirm that
sensing and control continue. Preserve failed reports and implementation
versions if a correction becomes necessary; do not silently replace evidence.

Success means the freeze contract is tested, all pairs have interpretable
integrity checks, results and limits are documented, and the result directs
the next engine decision. Improvement is an empirical outcome, not a required
acceptance condition. A read-only local result page will expose paired light
curves and metrics without controlling the owner's active creature.

## Implemented intervention and validation

`LightLearner::freeze_learning()` and `LightControl::freeze_learning()` are
one-way native diagnostic interventions. They reject incomplete initialization
or a pending action without changing state; repeated valid calls are harmless.
Reset enables learning again. No freeze action was added to the browser API.
The only change to the ordinary learning path is a false-by-default guard on
parameter writes; continued trajectories reproduced the prior v2 evidence.

Parameter fingerprints encode exact IEEE-754 binary64 bits. The complete
learner-state fingerprint includes histories, replay, traces, identities,
validation state, counters, RNG streams and unfinished options; the common
driver includes its own wrapper state. These are diagnostic hashes, not an
export/restore interface or policy inputs. Existing update counters count
scheduled processing; suppressed-update counters identify processing whose
parameter writes were disabled. Frozen values still act on changing features.

The native aggregate build and all 15 CTest suites passed (10.04 seconds).
New tests exercise rejection/reset, identical prefixes, mid-option freezing,
exact unchanged weights under opposite reward histories, continuing samples,
actions, exploration and replay bookkeeping, and browser API isolation.
An independent review tightened effort comparisons to exact bits and complete
transition/public-state comparisons to canonical JSON before the first run.
An occupied output fixture was rejected and its pre-recorded hash preserved.

## Measured result, 2026-09-06

The single declared report is
`artifacts/reports/light-adaptation-v1-development-20260906/report.json`.
Its exact SHA-256 is
`531faefe51ed3044cc10d09b194a6341487dfe4f6f203035365407369572b054`.
All six pairs passed: twelve complete trials and twelve exact full-transition
replays, including reward. Every continuing branch reproduced archived v2
commands, observation hash, return and corresponding phase metrics exactly.
The independent summarizer verified 74 archived source/provenance files,
prefix equality, metric additions, replay evidence and frozen-value checks.

The primary measure below is integrated summed sensor reward during seconds
60-120. Each difference is computed within its matched pair.

| Body | Seed | Continue updates | Freeze values | Difference |
| --- | ---: | ---: | ---: | ---: |
| Default | 1 | 51.983170 | 49.280960 | +2.702210 |
| Default | 2 | 51.468387 | 48.668865 | +2.799522 |
| Default | 3 | 51.885721 | 49.489352 | +2.396370 |
| Side weight | 1 | 51.278809 | 49.193787 | +2.085022 |
| Side weight | 2 | 50.943766 | 47.430046 | +3.513720 |
| Side weight | 3 | 50.757393 | 48.181835 | +2.575559 |

Median paired reward gains were +2.702210 for the default body and +2.575559
for the side-weight body. As an additional descriptive calculation, first
compute `100 * (continued-frozen)/frozen` in each pair, then take its median:
+5.48% and +5.35% respectively, or +5.41% across all six pairs. This percentage
is relative to frozen values, not quiet control, whole-run return or lux.

Median paired late-light gains (seconds 100-120) were +164.46 lux and +149.22
lux. Post-move electrical energy fell in four pairs and rose in two; median
paired changes were -1.75 J and -2.30 J. Energy is measured separately and
remains outside the reward sum for these assemblies. All twelve branches had
zero feedback flags or native vetoes. Observed maxima were .375 A, 25.570 C
and 6.928 rad/s. The latter remains above the soft 3 rad/s governor target;
these observations are not a physical safety certificate.

### Interpretation and remaining limits

Continued parameter updates improve this controller's outcome after this
specific first minute of experience in all six pairs. Frozen branches retain
active sensing, reward history, memory features, exploration and feedback,
so their lower scores cannot be attributed merely to disabling feedback or
stopping the creature. It does not follow that every possible frozen policy
needs online updates: one trained across several conditions might handle
them without modifying its parameters during a try.

An additional descriptive comparison with archived quiet runs puts the effect
in context. All frozen branches earned less post-move reward than quiet.
Continuing branches exceeded quiet in five of six cases; weighted seed 3
returned 50.757393 versus quiet's 50.826192, despite greater late light and
better whole-run return. Those quiet comparisons have unmatched first-minute
physical histories, so they are not the matched intervention above. The result
establishes a useful contribution from continued learning, not optimal control.

The evidence remains six known development prefixes, without a stationary-sun
factorial control or new body/topology exposure. No tuning or extra rollout
was added after seeing the outcome. Frozen original rover-v3 files, the prior
light-learning reports and the pending challenge remain unchanged.

### Engine decision and proposed next milestone

Keep online v2 unchanged as the measured baseline: the new evidence gives
continued updates an empirical role in this fixture. Next, test whether
adaptation retains earlier useful experience when a condition returns.
A separately declared A-to-B-to-A source sequence can expose forgetting before
we choose a more elaborate memory architecture. Compare current values with
saved first-A values from an identically reconstructed physical/history state
at the return. Checkpoint selection is evaluation machinery; the deployed
learner still receives no source identity, coordinates or change cue.
Faster second visits alone would not establish retention because physical
entry states differ. This next milestone is proposed, not executed here.

Timestamp-aware state estimation, identified predictive models, stronger
control baselines, larger assembly graphs and transfer across rebuilds remain
broader engine work. This experiment does not claim those capabilities.

## Inspect and reproduce

Open the read-only [paired result page](http://127.0.0.1:43117/adaptation.html)
to choose a body and seed, compare delivered-light curves, and inspect reward,
energy and provenance. Its compact data is derived from the validated native
report, carrying that report's path and exact hash. It has no session-control
API. Browser checks covered missing data and retry, body/seed selection,
provenance disclosure, normal width and a 390-pixel viewport; the normal
viewport was restored, and browser warning/error logs were empty. Both of the
owner's completed live sessions remained byte-for-byte identical to their
pre-experiment API responses. No server restart was required.

From the configured project directory:

```sh
./build/native/droid-light-adaptation --output artifacts/reports/NEW_DIRECTORY
node tools/summarize-light-adaptation.mjs artifacts/reports/NEW_DIRECTORY/report.json
node tools/export-light-adaptation-view.mjs artifacts/reports/NEW_DIRECTORY/report.json
```

The report runner reserves a new directory and archives the sources before
running. The exporter independently validates the report before writing viewer
data; its optional output is limited to web or the owned adaptation QA tree.
Exact native reproduction is scoped to the validated runtime, not promised
across different compilers or MuJoCo versions. This run reused the existing
`droid-blocks-dev` container, GCC 12.2.0, CMake 3.25.1, MuJoCo 3.12.0, x86_64,
and the RelWithDebInfo build. Its recorded image identity is
`sha256:eabe48053a78bcb334a947e7a5c0c009d6ff174a3b5e57e2493593d423b91068`.

Additional validator checks used an owned copy of the report/source tree:
valid evidence passed, while incorrect paired reward, changed frozen values,
a failed replay flag, an unfinished report and modified archived source each
rejected. The initial fixture serializer dropped signed zero and was itself
correctly rejected; the fixture writer was corrected to retain those bits.
The real report and source snapshots were never rewritten. Logs, runtime,
preservation hashes and QA fixtures are under `build/light-adaptation-qa/`.
