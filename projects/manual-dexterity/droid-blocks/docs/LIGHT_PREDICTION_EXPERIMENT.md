# Sensor-based state-space identification: locked experiment

Adopted 2026-09-06. This protocol is fixed before collecting or scoring any new
prediction trial. Put outcomes in LIGHT_PREDICTION_RESULTS.md; do not edit this
file after collection. The goal is an explicit, independently verifiable
predictive state model and an honest decision about a next control experiment.
Passing an accuracy threshold is not required to complete the goal.

## State, input, output and scope

For the current anchored, planar, collisionless two-hinge fixture, the physical
state includes both joint positions/velocities and motor current/temperature,
plus delay queues and hybrid safety state. The deployable predictor receives
none of the simulator's hidden joint/body/source state. It models the composition
of the unchanged local motor governor, motor dynamics, mechanics and sampling.

We choose measured delay coordinates for this first state-space identification:
the state encoding is explicitly designed, while its transition coefficients
are learned from data. This is not a learned latent physical-state observer.
Define y_n as the following twelve-number local output, sampled at t_n=.02+.1n:

| Index | Output | Fixed normalization |
| ---: | --- | --- |
| 0 | Last executed motor effort | /.15 |
| 1-2 | Local motor encoder | sin(position), cos(position) |
| 3 | Motor velocity | /10 rad/s |
| 4 | Motor current | /.5 A |
| 5-7 | IMU specific force x,y,z | /50 m/s^2 |
| 8-10 | IMU angular velocity x,y,z | /10 rad/s |
| 11 | Delivered light | /1000 lux |

Use the existing strict light-context frame projector, including valid native
sensor delivery checks. Do not clip features, fit normalization scales, project
predicted sin/cos back to the unit circle, or supply IDs, source condition,
absolute time, seed, reward, geometry, world pose or passive-joint state.
Metadata validates causality and membership; it is not a numerical feature.

The explicit input v_n is the requested effort divided by .15, held constant
for the five .02-second control intervals from t_n to t_(n+1). The existing
LightLearner::bounded_effort governor refreshes actual effort using the local
observations every .02 s. At a forecast origin the entire future REQUEST tape
is known. Future executed efforts are not: they depend on future feedback and
must be predicted as y[0], never read from the future trace. Motor feedback and
queued sensor timing remain the unchanged native contracts.

For an input-using model with p taps, define the observable state

z_n = [y_n,...,y_(n-p+1),v_(n-1),...,v_(n-p+1)]^T.

Its learned top-row equation and exact shift realization are

y_hat_(n+1) = c_y + F z_n + G v_n,
z_(n+1) = A z_n + B v_n + c,
y_n = C z_n + D v_n,   D=0.

C selects the first twelve state coordinates. A includes learned top rows,
identity shifts of outputs and past inputs; B inserts the current request in
the past-input state as well as affecting the learned output; c is zero outside
the learned output rows. Export all matrices with their coordinate ordering.
These are identified discrete-time input-output matrices at .1 s sampling,
not the mechanical inertia matrix or a global physical linearization.

## Models fixed before collection

Fit separately for each body, pooling its stationary positive-x and negative-x
source trials. Neither source identity nor body identity enters any model row.

1. current_input: p=1, state dimension 12, affine current-output model with v_n.
2. history_input: p=11, state dimension 142, one second of output history and ten
   past requested inputs. This is the primary predictive model.
3. history_no_input: p=11, state dimension 132, output histories alone; its B
   matrix is zero. Past executed efforts remain legitimate output-history data.
   This ablates explicit past and future request information jointly, rather
   than isolating only future-plan information.
4. persistence: no fitting; hold every output at its origin value throughout
   the forecast. It is a prediction baseline, not a physical controller.

Fit each learned next-output model by minimizing

(1/N) sum_i ||y_(i+1)-Theta*[1,z_i,v_i]||^2 + .001 ||Theta_nonbias||_F^2.

Omit v_i and past-request coordinates for history_no_input. Penalize every
nonbias coefficient; do not penalize the intercept. Compute Gram and right-hand
side divided by N, add .001 on nonbias diagonal, solve in native C++ using
Cholesky, and report normal-equation residuals. There is no hyperparameter
search, fitted scaling, model-order search or post-result tuning.

Save coefficient/matrix artifacts and exact training-row membership and their
SHA-256 before the first withheld forecast is evaluated. Prediction never fits
or mutates the model. The independent validator rebuilds training rows from
retained traces, refits the declared regression, and checks coefficients and
companion matrices without relying on native predictions as truth.

## Assemblies, source and excitation

Use the two exact assemblies in the completed context report, whose SHA-256 is
f3f729120df6b13f79845ecbd4ee72409f111f76b1b525037d3c059e41f7c6ee.
Add a third assembly copied from its default body, with segments changed from
[3,3] to [4,3] and name 'Sun creature with a longer first beam'. All mounts,
blocks and other fields remain as in that default body. This tests identification
after a declared shape change, with separate fitting for each body; it is not
zero-shot transfer. No live user assembly is changed or substituted.

Each source is stationary during a trial: condition A=[.30,0,.20] m and
B=[-.30,0,.20] m, both intensity 1000 lux. The source label is evaluator metadata.
Initialize every trial as rebuild(assembly), set_sun(condition), reset().
No claim concerns prediction of an unannounced external source change.

Use seeds 201 through 212. A SplitMix64 stream starts with state=seed, then for
each block draws duration from {1,2,4,8} sample intervals by next()%4 and requested
normalized amplitude from {-1,0,+1} by next()%3, in that order. Use the same
SplitMix64 increment/mix constants as the existing LightLearner random stream.
Repeat/truncate blocks to exactly 240 requested intervals. Generate this tape
before simulation and independently reconstruct it for verification. The same
seed uses the same request tape across both sources and all bodies; actual
efforts may differ across bodies because the local governor observes speed.

After reset, execute one initial zero-effort .02-second control interval to
reach t_0=.02 with delivered sensors, then 240 intervals of five controls each.
Total per trial: 1201 controls, 24.02 s, 241 sampled frames n=0..240. Native hard
safety stays active. Do not tune amplitude/length or discard failed trajectories.
A/B mechanics and executed efforts must match within a given body/seed pair.

Training seeds: 201-208. Withheld seeds: 209-212. Both conditions and all bodies
for a seed stay in the same split. There are 72 trials total: 3 bodies*12 seeds*
2 conditions. Independently reconstruct and replay all 72 full trials with exact
requests, efforts, complete transitions and final public physical state.

## Fit and forecast membership

Every learned model uses the same training origins n=10..239, predicting n+1.
There are 230 rows/trial and 3680 training rows/body. No withheld trace contributes
a row, scale, parameter or model-selection decision.

On every withheld trial, forecast from n=10,20,...,230: 23 origins, starting at
1.02 s and ending at 23.02 s. Seed state from permitted actual output/request
history through that origin only. Supply v_n,...,v_(n+9) from the already known
request tape. Advance the predictor ten times without injecting an intermediate
actual observation, current, effort, light, hidden state or recalculated true
future governor value. Actual future outputs are targets only.

Evaluate horizons 1,4,10 sample steps (.1,.4,1.0 s). Archive all ten predicted
outputs at every origin, including raw out-of-range values. A nonfinite recursive
forecast is retained as a model failure and disqualifies readiness; do not hide it
by clamping, omitting a case or returning a finite aggregate from surviving cases.

## Measures and prospective decision

Primary at each horizon: equally weighted mean squared error over normalized
channels [1,2,3,5,7,9,11] (encoder sin/cos, velocity, force x/z, gyro y, light).
This excludes constant planar axes and the easier effort/current channels from
the headline metric. Average origins within each trial, then average the eight
withheld trials within each body; RMSE is the square root of that mean MSE.
All model/persistence comparisons are paired on the same origins and targets.

Report each trial and the four seed-pair aggregates, their ranges, every body's
results, and every channel's RMSE in its normalized and physical units. For
sin/cos the units remain dimensionless; do not silently convert vector error to
an angular error. Relative improvement is 1-model_RMSE/reference_RMSE; if the
reference RMSE is zero, show unavailable and use exact absolute inequalities.
The overlapping windows are not independent experimental replications; no IID
window confidence interval or replay-as-replication claim is permitted.

The primary history_input model is a candidate for a subsequent bounded
model-based-control experiment only if it has no failed forecasts and, on EACH
body, all of these hold:

- RMSE at .4 s and 1.0 s is at most .8 times persistence RMSE.
- At 1.0 s every withheld trial's RMSE is at most its persistence RMSE.
- At 1.0 s aggregate RMSE is at most .9 times history_no_input RMSE, so explicit
  command information has an observed contribution. The same .9 comparison must
  also hold on the prospectively defined forecast subset where at least one of
  v_(n+1)..v_(n+9) differs from v_n. Report this subset's origin counts and RMSE
  separately, averaging eligible origins per trial then eligible trials per
  body. If that subset is empty for a body, readiness is not established.

Apply and report the same engineering gate to current_input as a simpler
candidate, using the same no-input comparison. A simpler passing model prevents
claiming that longer history is necessary. The no-input model and persistence
are controls, not candidates passing this gate. These thresholds are prospective
engineering decisions, not significance levels, stability proofs or guarantees
of effective control. Even a pass requires a separately declared closed-loop
comparison with the unchanged RL controller before claiming improved control.

Record probe reward, energy, maximum current/temperature/speed and native safety
flags separately. They are measurements of excitation, not fitted objectives.
Do not expose the original frozen-v3 challenge or modify the live RL controller.

## Integrity and deliverables

Use an exclusive output directory and refuse existing directories. Preserve
partial trials and errors on failure; incomplete evidence cannot be certified.
Archive native sources/headers/tests, locked protocol, independent validator,
module catalog, build/runtime definitions and the reference assembly report
with SHA-256 checks. Preserve full traces, training membership, frozen models
and forecasts. Source labels/timestamps/IDs remain metadata outside prediction.

Deliver native predictor/runner/tests, independent validation, and a read-only
page showing actual versus recursive predicted responses, exact state/input
meaning, model matrices, per-horizon/per-channel errors and provenance. Connect
it to a discoveries index so the project's findings are navigable. Record the
result and the next state/model/control decision separately from this protocol.
Earlier reports, the existing RL source and frozen original inputs are preserved.
