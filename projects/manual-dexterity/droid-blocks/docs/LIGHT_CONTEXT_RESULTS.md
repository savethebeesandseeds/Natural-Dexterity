# Can its senses tell the difference? Results

Completed 2026-09-06 under the separately locked
[context-decoding protocol](LIGHT_CONTEXT_EXPERIMENT.md).
The [read-only viewer](http://127.0.0.1:43117/context.html) shows the complete
withheld-sequence comparison and individual prediction curves.

## Outcome and engine decision

One second of permitted sensor/action history distinguished A from B with
93.75% accuracy on the default body and 95.45% on the side-weight body.
History improved every withheld probe comparison with instantaneous readings.
It passed the predeclared readiness rule on both bodies; the other three
feature views did not. Removing light gave exactly 50% accuracy and a 0.5 vote
for every evaluated example, as the paired negative control requires.

This supports a bounded closed-loop experiment selecting among supplied saved
value memories. It does not establish that the creature already retrieves a
memory, discovers contexts, or knows when its prediction will be correct.
The present addition is a tested, frozen native sensor-history decoder and an
offline experiment. It does not change the live controller or add a browser
memory-selection action. Keep history_expected_sarsa_v2 as the measured baseline.

## What was measured

Two known development assemblies, twelve random motor probes per body, and
stationary A/B light conditions produced 48 twelve-second trials. Each A/B pair
used identical requested and executed motor efforts. Eight complete sequences
(seeds 101-108) supplied training examples; four (109-112) were withheld.
No window from a withheld sequence entered training. Separate models were
trained for each body, with 352 training and 176 evaluation examples per body.

A native five-nearest-neighbor decoder used the protocol's fixed transformations,
mean squared coordinate distance, and all exact ties at the fifth neighbor.
Predictions receive only numeric local sensor/action features. Source labels
are supervised training targets and evaluator truth, never prediction inputs.
Models were saved and hashed before withheld evaluation and remained unchanged.

Each withheld probe contributes 44 evaluation examples, equally split between A and B.
Accuracy equals balanced accuracy here. Overall values weight the four probes
equally. The histories overlap: the 176 windows are not independent trials,
and the 48 replays are verification, not additional experimental observations.

### Primary comparison and controls

| Body | One-second history | Current readings | History minus current | Brightness only | History without light |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default | 165/176 = 93.75% | 130/176 = 73.86% | +19.89 percentage points | 100/176 = 56.82% | 88/176 = 50.00% |
| Side weight | 168/176 = 95.45% | 127/176 = 72.16% | +23.30 percentage points | 122/176 = 69.32% | 88/176 = 50.00% |

| Body | Withheld probe | History | Current | Brightness only | Without light | History minus current |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 109 | 95.45% | 65.91% | 43.18% | 50.00% | +29.55 pp |
| Default | 110 | 97.73% | 77.27% | 63.64% | 50.00% | +20.45 pp |
| Default | 111 | 90.91% | 81.82% | 59.09% | 50.00% | +9.09 pp |
| Default | 112 | 90.91% | 70.45% | 61.36% | 50.00% | +20.45 pp |
| Side weight | 109 | 100.00% | 79.55% | 59.09% | 50.00% | +20.45 pp |
| Side weight | 110 | 100.00% | 72.73% | 79.55% | 50.00% | +27.27 pp |
| Side weight | 111 | 95.45% | 75.00% | 77.27% | 50.00% | +20.45 pp |
| Side weight | 112 | 86.36% | 61.36% | 61.36% | 50.00% | +25.00 pp |

History ranges were 90.91%-97.73% and 86.36%-100.00%. Both exceed the declared
75% per-body mean and 65% on every withheld probe. Current readings and
brightness alone do not clear that same rule. This is an engineering gate for
another experiment, not a significance test or guarantee of reliable control.
Four withheld probes per body give limited evidence about generalization.

### Errors, vote quality and ambiguity

Confusion entries below are counts, with 88 examples of each true class per row.
A tie predicts A, explaining the no-light control's asymmetric confusion counts.

| Body | View | True A, predict A | True A, predict B | True B, predict A | True B, predict B |
| --- | --- | ---: | ---: | ---: | ---: |
| Default | History | 83 | 5 | 6 | 82 |
| Default | Current | 68 | 20 | 26 | 62 |
| Default | Brightness only | 52 | 36 | 40 | 48 |
| Default | Without light | 88 | 0 | 88 | 0 |
| Side weight | History | 86 | 2 | 6 | 82 |
| Side weight | Current | 65 | 23 | 26 | 62 |
| Side weight | Brightness only | 69 | 19 | 35 | 53 |
| Side weight | Without light | 88 | 0 | 88 | 0 |

Brier score is the mean squared difference between the smoothed vote p(A) and
the binary A target; lower is better. Ambiguity uses the predeclared inclusive
[0.4,0.6] band. It is a diagnostic, not a deployed abstention rule.

| Body | View | Brier score | Votes in ambiguity band |
| --- | --- | ---: | ---: |
| Default | History | 0.180572 | 156/176 = 88.64% |
| Default | Current | 0.186638 | 127/176 = 72.16% |
| Default | Brightness only | 0.263815 | 61/176 = 34.66% |
| Default | Without light | 0.250000 | 176/176 = 100.00% |
| Side weight | History | 0.175499 | 152/176 = 86.36% |
| Side weight | Current | 0.178898 | 113/176 = 64.20% |
| Side weight | Brightness only | 0.187500 | 59/176 = 33.52% |
| Side weight | Without light | 0.250000 | 176/176 = 100.00% |

The high classification accuracy coexists with narrow neighbor votes. Across
both bodies, 305/352 history predictions were ordinary 3:2 five-neighbor votes,
which become 3/7 or 4/7 after the declared smoothing. Of these, 288/305 were
correct. Another three history predictions had p(A)=0.5. Thus 308/352 (87.5%)
were in the ambiguity band. A vote of 0.57 is not evidence of 57% calibrated
reliability. Abstaining throughout that band would discard most predictions;
these outcomes must not be used to tune a confidence threshold after evaluation.

For supplemental pooled accounting only, history was 333/352 = 94.60%, current
257/352 = 73.01%, brightness 222/352 = 63.07%, and no-light 176/352 = 50.00%.
The per-body results above remain primary.

### Physical probe measurements

These describe the fixed probes, not objectives optimized by the classifier.
Ranges cover the 24 original trials for each body.

| Body | Integrated sensor reward range | Electrical energy range (J) | Maximum current (A) | Maximum temperature (C) | Maximum motor speed (rad/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default | 10.048130-10.611504 | 1.380886-2.173207 | 0.375000 | 25.052295 | 6.486016 |
| Side weight | 9.884933-10.653776 | 1.413829-2.303846 | 0.374999 | 25.062445 | 6.887534 |

There were zero native-veto steps and zero feedback-flag steps. The existing
shaft-speed governor has a soft target; passive dynamics can backdrive the
motor above that target. No new mechanical safety or hardware claim follows.

## Verification and preserved evidence

All 48 trials and all 48 independently reconstructed full trials completed.
Every replay matched requested and executed effort traces, full transitions,
observations and final public physical state. All 24 A/B pairs had identical
requests, commands, non-light observations and mechanical public projections.
Within these trials, changing the source did not change the recorded mechanics.
The projection checks cover the declared public state; they are not a saved
checkpoint of every private MuJoCo field.

The independent Node verifier reconstructed the causal frames and windows,
training membership, numeric model artifacts, neighbor votes, every prediction,
aggregate measures, negative control and readiness decision. Its result matched
the native result. No probe, representation or decoder was retuned after seeing
withheld outcomes; the first collected report is the final report.

The aggregate native build passed and all 16 CTest suites passed in 11.98 seconds.
Six new native unit-test groups exercise input contracts and excluded cues,
causal history projection, frozen training copies, exact cutoff ties, paired
negative controls, and malformed/nonfinite inputs. The validator accepted an
untouched copy and rejected nine altered fixtures: wrong neighbor vote, future
frame substitution, wrong aggregate accuracy, failed replay, incomplete evidence,
unfinished scoring, withheld training membership even with a consistent changed
model digest, changed negative-control vote, and changed archived source.
The runner also refused an occupied output directory without changing its sentinel.

Browser verification covered the initial missing-data/retry state, both bodies,
all four feature views, withheld-sequence selection, the flat 0.5 negative-control
plot, provenance, and wide and 390x844 layouts. The finished page reloaded with
no browser warnings or errors. It displays derived evidence and has no control
write path. Final display is default body, history view, probe 109.

Preservation checks verified eleven prior/locked files byte-for-byte, including
the locked context and retention protocols, original model/catalog, frozen-v3
configuration and policy, earlier reports, and adaptation/retention viewer data.
Both completed live light/construction HTTP state responses remained byte-for-byte
identical. The same existing container stayed running and healthy; no restart,
container replacement, or original frozen-v3 challenge execution occurred.

Authoritative artifacts:

- [Native report](../artifacts/reports/light-context-v1-development-20260906/report.json),
  SHA-256 `f3f729120df6b13f79845ecbd4ee72409f111f76b1b525037d3c059e41f7c6ee`.
- [Locked protocol](LIGHT_CONTEXT_EXPERIMENT.md),
  SHA-256 `3e1218d1e0000cca5020962aeb73ba773f92bdcc1012a178ba4d238389d87dff`.
- [Frozen training models](../artifacts/reports/light-context-v1-development-20260906/training-models.json),
  SHA-256 `8b81255e3712017234deb8023f2a1b6291d30548e06cde3cf55810c1d480d3b5`.
- The report directory contains 78 archived source/provenance files and all 48
  full trace sidecars, with their SHA-256 manifest in the report.
- [Independent summary](../build/light-context-qa/summary.json),
  [native test log](../build/light-context-qa/native-ctest.log),
  [validator rejection checks](../build/light-context-qa/validator-negative-results.json),
  and [preservation verification](../build/light-context-qa/preserved-after.json).
- [Viewer data](../web/context-data.json), independently verified before export.

Reproduce within the established development environment and project root.
The native runner requires a new output directory and refuses an existing one:

```sh
cmake --build build/native --target droid-native-validation
ctest --test-dir build/native --output-on-failure
./build/native/droid-light-context --output artifacts/reports/NEW_DIRECTORY
node tools/summarize-light-context.mjs artifacts/reports/NEW_DIRECTORY/report.json
node tools/export-light-context-view.mjs artifacts/reports/NEW_DIRECTORY/report.json
```

Revalidate the existing evidence without collecting another trial:

```sh
node tools/summarize-light-context.mjs artifacts/reports/light-context-v1-development-20260906/report.json
```

Do not append outcomes to or edit the locked protocol. Keep the archived source,
report and training models intact when developing the next engine capability.

## Limits and the next question

This is supervised discrimination on two familiar, anchored, collisionless,
planar underactuated bodies under stationary sources and random motor probes.
It does not establish arbitrary-body transfer, contact behavior, robustness to
noise, moved sensors or hardware, autonomous context discovery, minimal required
memory, or general state observability. History has more coordinates than the
instantaneous view; this comparison does not isolate why that representation
works better. Four withheld sequences per body cannot support broad reliability
claims, and overlapping windows cannot supply independent-window confidence
intervals.

The next proposed goal is: can local observations select a useful supplied
memory during closed-loop control when the source changes at undisclosed times?
Use the present decoder and two supplied value memories as frozen components,
with their A/B associations declared as experimental scaffolding. The selector
must not receive source identity, phase or switch time. Preserve physical state,
sensor queues, controller history and unfinished motor options; change value
memory only at ordinary option boundaries. Declare one rule to prevent rapid
switching, including invalid/unavailable-history behavior, before execution.
Do not derive a confidence cutoff from this report's withheld vote distribution.

Compare the selector with always-current frozen values and an externally chosen
recall reference, using changed switch times plus constant-A and constant-B
controls. Constant-source controls are essential: memory-controlled actions
change the input distribution from the random probes used here. Source switches
also introduce mixed-condition history windows and delayed observations. If
uncertainty handling, bounded probing or a fallback to the existing controller
is introduced, specify it prospectively and count its time, reward and energy.

Measure integrated sensor reward as the primary control outcome, including any
selection delay or probing. Record context-inconsistent memory dwell, selection latency, repeated
switching, distance to training examples, energy and safety diagnostics as
secondary measures. Successful classification is not sufficient: selection must
improve behavior under these controls. Such a result would establish runtime
selection from supplied memories; discovering, creating and associating those
memories autonomously would remain a separate engine question.

## Handoff availability note

After the completed native run, preservation check and normal-server browser
QA, the Docker Desktop Linux-engine pipe became unavailable and port 43117
refused connections. The cause was not established. The healthy-container and
live-response checks above describe their recorded verification time, not a
claim that the simulator remained available afterward. No Docker restart or
container mutation was performed to recover the viewer.

A separate [read-only preview](http://127.0.0.1:43118/context.html) now serves the
retained chart files on loopback. Its header explicitly identifies the native
simulator as offline. It serves only the context page, its two stylesheets,
script and derived data, accepts GET/HEAD only, and exposes no simulator API.
The preview data matches web/context-data.json byte-for-byte; the report and
locked protocol remain unchanged. Its page loaded correctly with no browser
warnings or errors. This is a review helper, separate from the archived native
experiment sources. Relaunch it from the project root if needed:

```sh
node build/light-context-qa/serve-context-review.mjs
```

The native viewer route remains /context.html when the established Docker
environment is available again. Restoring the native runtime is outside this
completed experiment; the retained evidence can be validated without it.

## Native viewer restored after the user's restart

On 2026-09-06 the owner explained the computer restart and restarted Docker.
The same container (a3eabc3183e054aafd42304857b6a10b2c3a7cf15787fcf177149335e6aeaaaf)
is running and healthy with the original project bind and loopback port 43117.
The [normal viewer](http://127.0.0.1:43117/context.html) is restored and the
existing report tab now uses it. Its page, styles, script and data match the
workspace bytes, and the browser reports no warnings or errors. No listener
remains on the temporary preview port 43118.

The retained report, protocol, frozen models and all 78 source files passed
another read-only integrity check, including the independent validator. No
new trials were required or collected. This supersedes the availability issue
above; the sensor-context goal remains complete. The earlier live-session
preservation captures remain historical evidence from before the restart, not
a claim that in-memory sessions survived it. Runtime checks are recorded in
[restored-runtime.json](../build/light-context-qa/restored-runtime.json).
