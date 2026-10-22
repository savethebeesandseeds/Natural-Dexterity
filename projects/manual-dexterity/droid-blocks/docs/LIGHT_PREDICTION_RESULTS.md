# Sensor-based state-space identification: results

Completed 2026-09-06 under the separately locked
[prediction protocol](LIGHT_PREDICTION_EXPERIMENT.md).
The [retained report](../artifacts/reports/light-prediction-v1-development-20260906/report.json),
[frozen models](../artifacts/reports/light-prediction-v1-development-20260906/models.json),
and [independent summary](../build/light-prediction-qa/independent-summary.json)
contain the full precision results, individual trials, forecasts and matrices.

Both input-using models passed every predeclared readiness check on all three
bodies. This justifies a bounded model-based-control experiment. It does not
establish improved control: these models predicted recorded probes and did not
choose the motor commands in this experiment.

## What was identified

The fitted system is the composition of the existing local motor governor,
motor dynamics, mechanics and delivered sensor samples. Its output has twelve
fixed normalized coordinates: last executed effort, encoder sine/cosine, motor
velocity/current, local IMU specific force and angular velocity, and light.
The known input is the requested effort divided by .15, held for five .02-second
controls. Future emitted efforts depend on future feedback and were predicted
as outputs; forecasts never received the future emitted-effort trace.

The saved models export the affine discrete-time realization

z_(n+1) = A z_n + B v_n + c,    y_n = C z_n + D v_n.

The state coordinates and shift structure were designed; the next-output
coefficients were fitted. C selects the current output and D is zero by
construction. These are sensor-delay input-output matrices, not a physical
state observer, a mechanical inertia matrix or a global plant linearization.
See [the control model](CONTROL_MODEL.md) for the physical/controller boundary.

- `current_input`: the current twelve outputs and current requested input;
  12 state coordinates.
- `history_input`: eleven output frames spanning one second, ten past requests,
  and the current request; 142 state coordinates.
- `history_no_input`: eleven output frames, with no explicit requested inputs;
  132 state coordinates. Past emitted effort remains in its output history.
- `persistence`: keep the origin output unchanged at every forecast horizon.

Each body received a separate affine ridge fit with fixed lambda .001,
mean-loss normalization and an unpenalized intercept. The two known bodies
were joined by a declared construction change: extending the default first
beam from three to four cells. The longer body was fitted using its own data;
this is not transfer to an unseen body.

There were 72 stationary-source trials: three bodies, twelve command seeds,
and both source conditions. Each ran for 24.02 seconds and supplied 241 frames.
Seeds 201-208 supplied 3,680 training rows per body; entire seeds 209-212 were
withheld. Each withheld trial contributed 23 forecast origins, giving 184
origins per body and 552 overall. Models were saved and hashed before evaluation.
Every ten-step forecast advanced its own predicted state without intermediate
measurements. Source labels, absolute time, body identity, geometry, reward and
hidden joint state were excluded from the model rows.

## Prediction error and the predeclared decision

The primary metric is RMSE over seven normalized coordinates: encoder sine and
cosine, motor velocity, force x/z, gyro y and light. Origins are averaged within
each trial, then the eight withheld trials are weighted equally within a body.
The score excludes effort/current and the constant planar axes. It is
unitless; its value is not an angular error or a light error.

| Body | Model | 0.1 s RMSE | 0.4 s RMSE | 1.0 s RMSE |
| --- | --- | ---: | ---: | ---: |
| Default | current_input | 0.0823 | 0.1919 | 0.2351 |
| Default | history_input | 0.0424 | 0.1511 | 0.2414 |
| Default | history_no_input | 0.0636 | 0.2151 | 0.3365 |
| Default | persistence | 0.1197 | 0.3355 | 0.4569 |
| Side weight | current_input | 0.0749 | 0.1746 | 0.2031 |
| Side weight | history_input | 0.0413 | 0.1441 | 0.2003 |
| Side weight | history_no_input | 0.0628 | 0.2075 | 0.3193 |
| Side weight | persistence | 0.1190 | 0.3291 | 0.4465 |
| Longer first beam | current_input | 0.0598 | 0.1214 | 0.1537 |
| Longer first beam | history_input | 0.0325 | 0.1061 | 0.1513 |
| Longer first beam | history_no_input | 0.0562 | 0.1810 | 0.2886 |
| Longer first beam | persistence | 0.1170 | 0.3123 | 0.4153 |

History input reduced the one-second composite RMSE relative to persistence by
47.2%, 55.1% and 63.6% on the default, weighted and longer bodies respectively.
It outperformed current input at .1 and .4 seconds on all three bodies. At one
second, current input was slightly better on the default body; the two models
were close on the other bodies. The success of the 12-coordinate model means
this experiment does not establish that 142 coordinates are necessary.

Both candidates had zero failed forecasts; beat .8 times persistence RMSE at
.4 and 1 second on each body; were no worse than persistence on every individual
withheld trial at one second; and beat .9 times the no-input model at one second,
both overall and on the predeclared changed-request subset.

That subset contains origins where at least one future request after the first
differs from the first. There were 164 eligible origins per body. Its RMSE first
averages eligible origins within each trial, then eligible trials within a body.

| Body | Eligible origins | Current input, 1 s | History input, 1 s | No input, 1 s | History reduction versus no input |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default | 164 | 0.2151 | 0.2179 | 0.3221 | 32.4% |
| Side weight | 164 | 0.1851 | 0.1768 | 0.3033 | 41.7% |
| Longer first beam | 164 | 0.1429 | 0.1390 | 0.2731 | 49.1% |

This supports an observed contribution from explicit command information.
The ablation removes past and future requested inputs together, so it does not
isolate the effect of the future plan alone.

The report retains every trial and all four A/B seed-pair aggregates. For example,
the history model's one-second seed-pair RMSE ranges were 0.1681-0.3218,
0.1585-0.2554 and 0.1369-0.1728 across the three bodies. Four withheld command
seeds give limited evidence about other excitation. Overlapping forecast
windows are not independent replications; deterministic replays are integrity
checks, not extra observations. The thresholds are engineering gates, not
significance levels or stability guarantees.

## Light remains a separate limitation

Composite improvement does not imply comparable accuracy in every channel.
Light-only physical errors show why an objective-seeking controller still needs
a direct test. The table reports lux RMSE; history means `history_input`.

| Body | Horizon | Current input | History input | No input | Persistence |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default | 0.1 s | 37.6 | 38.6 | 38.7 | 33.5 |
| Default | 0.4 s | 73.7 | 71.4 | 71.5 | 85.4 |
| Default | 1.0 s | 104.5 | 104.1 | 104.4 | 138.6 |
| Side weight | 0.1 s | 32.3 | 33.7 | 33.8 | 30.3 |
| Side weight | 0.4 s | 78.5 | 75.1 | 75.3 | 81.8 |
| Side weight | 1.0 s | 95.4 | 94.0 | 94.6 | 114.8 |
| Longer first beam | 0.1 s | 31.8 | 31.7 | 31.7 | 33.0 |
| Longer first beam | 0.4 s | 79.4 | 71.0 | 70.9 | 80.7 |
| Longer first beam | 1.0 s | 95.9 | 95.5 | 96.0 | 117.9 |

At .1 second all fitted models predicted light worse than persistence on the
default and weighted bodies. At one second the history model still had about
104, 94 and 95 lux RMSE. Its light error was close to the no-input model's,
despite a substantial composite advantage. Good average motion prediction does
not yet demonstrate reliable ranking of actions by future sensory reward.

## Verification and physical measurements

All 72 trials completed and all 72 independently reconstructed full replays
matched requested commands, emitted effort bits, serialized transitions and
public mechanical state. This is not a comparison of every private MuJoCo
variable. There were no native veto steps or motor feedback flags, and no
failed recursive forecasts.

Across the probes, maximum absolute motor current was 0.375 A, maximum motor
temperature 25.158 C, and maximum absolute motor speed 7.491 rad/s. Electrical
energy ranged from 2.893 to 4.609 J per trial. The governor's 3 rad/s target is a
soft effort rule, not an absolute speed limit. These are excitation measurements,
not evidence of reduced energy or improved reward under a new controller.

[Precollection checks](../build/light-prediction-qa/precollection-validation.json)
record 17 passing native suites, including six predictor mathematics/boundary
test groups, zero compiler warnings and refusal to overwrite an occupied output.
The independent checker verified 70 archived source files, reconstructed local
outputs and training membership, refitted the regressions, checked all matrices,
recomputed recursive forecasts, and recomputed metrics and readiness.

The [evidence harness results](../build/light-prediction-qa/evidence-tests-v1/results.json)
record 13 passing checks: an untouched exact-byte fixture was accepted and twelve
tampered fixtures were rejected. These included renewed-hash coefficient/matrix
changes, withheld training membership, injected future actual outputs, command
and sampled-effort discrepancies, and corrupted aggregate/readiness fields.
The original evidence and all 142 linked fixture files were rehashed unchanged.

| Artifact | SHA-256 |
| --- | --- |
| Locked protocol | `7cc508c8cf1bc88e0b0f417b49df93022e895f8f90c9f1c91e528372696ef6be` |
| Retained report | `3d9e70141c7fd73da226c4b5715678205c9181623a308f8f93e9d7a59494b0e0` |
| Frozen models | `5421f975d958d831e58414eef661eabfd852e78734036acb293d057fd7bb81ee` |
| Independent checker | `48c4fb3f290a323ef06420004760a81c5df98483e820cd90eb7be989a32c0d39` |

The completed [before/after preservation audit](../build/light-prediction-qa/preservation-after.json)
verified all 19 protected files unchanged, including prior reports, RL/plant
sources, locked protocols and original frozen inputs. Both live light and
construction API snapshots also retained identical bytes and hashes. This is
separate from the evidence harness's check of the new prediction archive.

The [read-only prediction viewer](http://127.0.0.1:43117/prediction.html) exposes
all three bodies, four predictors, physical-unit forecasts and matrix entries.
[Browser checks](../build/light-prediction-qa/browser-qa.json) verified the real
export, controls, exact A/B delay shifts, keyboard matrix navigation and narrow
layout. The [findings index](http://127.0.0.1:43117/discoveries.html) connects
the earlier results and is linked from the live light page. The original
container remained healthy; no server restart or live API mutation was required.

## Next engine decision

The justified next experiment is bounded receding-horizon control using the
frozen identified models, the same local motor governor and the native summed
sensor-reward objective. Compare controllers using BOTH current-input and
history-input representations against the unchanged online RL controller.
Their prediction results justify testing them; they do not select a winner.

Declare the planning horizon, control cadence, candidate requested-action
sequences, fallback behavior and any action-cost choice before running that
comparison. A .4-second initial planning horizon is a recommendation based on
the smaller observed prediction errors, not a new locked protocol or a claim
that .4 seconds is optimal. If adding an effort/energy penalty, declare it as an
explicit planning choice and continue to report the original native reward
separately.

Measure actual integrated native reward, energy and safety, together with errors
in the predicted ranking of candidate actions. The controller will visit states
and command sequences that differ from random probes; test those errors under
that controller-induced distribution. Keep evaluation and any action-ranking
truth separate from the deployed predictor's permitted inputs. Light and IMU
prediction errors, model uncertainty and out-of-range recursive values must not
be hidden by a favorable composite score.

This result supplies inspectable predictive matrices and a justified next test.
It provides no proof of closed-loop stability, full physical-state observability,
accurate prediction after an unannounced source change, transfer to an unseen
body, or control improvement over RL.

