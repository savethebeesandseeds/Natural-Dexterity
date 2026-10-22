# Is useful motion attainable? Locked development protocol

**Locked before collection: 2026-10-04.** The owner authorized continuing the
project after reviewing the failed September 7 actuation-transfer result.
The first batch is `artifacts/reports/light-headroom-v1-development-20261004`.
Results belong in a separate document. Preserve this protocol, its source
archive and every outcome rather than changing conditions to obtain a pass.

## Question

Does the existing two-link mechanism have a useful reward advantage over a
quiet motor under prospectively declared lower lighting, and does its unchanged
native Expected SARSA learner discover that advantage within 64 simulated
seconds? This separates attainable behavior from learner competence before
another transfer comparison. It neither reruns nor rehabilitates the earlier
PPO experiment.

The earlier transfer study often favored quiet control. At the hanging reset
pose the light sensor's receiving face points upward. A source below the body
therefore offers a different incentive, without changing the light sensor,
its reward, the IMU preference, motor physics or safety. Whether movement
actually improves summed native reward remains an experimental question.

## Fixed physical conditions

Use the three existing transfer body configurations, separately with the motor
at the base (`powered_hinge: 0`) and elbow (`powered_hinge: 1`):

| Body | Beam lengths | Added block |
| --- | --- | --- |
| Long upper | [4, 3] | None |
| Long lower | [3, 4] | None |
| Weighted | [4, 4] | Segment 1, slot 0, side -1 |

The IMU is at segment 1, slot 2, side -1; light sensor at segment 1, slot 2,
side +1, receiving face +1. Use the original light-enabled assembly schema,
block parameters and sensor specifications. Part names and identities are
metadata, never controller inputs.

Cross every body and powered hinge with four stationary sources, each at
intensity 1000 lux and y=0:

| Source group | x | z | Role |
| --- | --- | --- | --- |
| Historical above-left | -0.30 m | +0.20 m | Diagnostic control |
| Historical above-right | +0.30 m | +0.20 m | Diagnostic control |
| New below-left | -0.30 m | -0.30 m | Primary development condition |
| New below-right | +0.30 m | -0.30 m | Primary development condition |

All 24 cases are known development data. No source condition, body or seed is
removed after inspecting results. Every trial uses an isolated worker-free
`ConstructionWorld` initialized by `rebuild -> set_sun -> reset`, with the
hanging pose, zero velocities, fresh motor state and fresh acquisition queues.
Reset after selecting the source is mandatory because `set_sun` preserves
already acquired samples. There is no source change or rescue reset in a trial.

## Controllers and exposure

Each trial lasts 3,200 controls at 0.02 seconds, or 64 simulated seconds;
each complete control contains ten 0.002-second physics steps. Startup,
searching and learning count inside that allowance. Stop and retain any native
termination, veto, feedback fault or controller exception. An incomplete run
cannot pass a gate.

Every manual controller requests quiet for the first five controls (0.1 s).
Subsequent fixed schedules are relative to the end of that startup:

1. Quiet: request zero throughout.
2. Constant negative: request -0.15.
3. Constant positive: request +0.15.
4. Square 2 s: start at -0.15 and alternate sign every 1 s.
5. Square 4 s: start at -0.15 and alternate sign every 2 s.

Every requested effort passes through the existing
`LightLearner::bounded_effort` at every native control. Its speed governor,
slew limit, invalid-sample handling and safety remain unchanged. Preserve both
requested and emitted effort. Fixed schedules receive no source position,
body geometry, hinge label or privileged physical state.

A sixth, explicitly handwritten **sensor search-and-hold witness** uses only
the local motor encoder, local shaft velocity and realized summed native
reward. Following the common startup it visits encoder targets
`[0, +pi/2, -pi/2, pi]` for exactly four seconds each. Its request is
`clamp(0.30 * atan2(sin(target-q), cos(target-q)) - 0.05*v, -0.15, +0.15)`,
then passed through the same native governor. Score each target by summed raw
native reward during the final one second of its dwell. Select the largest
score, resolving exact ties by the first target, and hold that target from
16.1 s through 64 s. All search costs count. The targets are local encoder
coordinates, never a desired world pose. This baseline has continuous feedback
and a different action vocabulary from the learner; it is an attainable
behavior witness, not an architecture-matched claim of superior learning.

Finally run the unchanged `LightControl` learner with fresh seeds
`101, 202, 303` in every case, with its existing initialization, option cadence,
history, replay, discount and update settings. Do not add a separate startup
that bypasses its act/observe contract. No controller memory or experience
passes between trials. There is no parameter search, checkpoint selection,
PPO update, new reward, model-based policy or transfer in this batch.

The schedule is 24 cases times nine arms = **216 trials**, or at most
691,200 native controls, 6,912,000 physics steps and 13,824 simulated seconds.
Integrity replays are additional computation and are not independent trials.
Simulator time may pause during computation; this is not a real-time toy test.

## Primary decisions

Use whole-run summed native reward. Each trial's paired gain is its reward
minus the same case's quiet reward. The engineering readiness margin is
**+3.2 integrated reward**, equivalent to +0.05 reward/s over 64 s, or five
percent of the maximum light vote rate. This is a declared threshold, not a
statistical significance criterion or a hardware/product acceptance claim.

- **Attainable headroom:** every one of the 12 lower-source cases has at least
  one complete, safe manual witness with gain strictly greater than +3.2.
  Show fixed-schedule and sensor-feedback witnesses separately. Choosing the
  best fixed witness per case is an exposed evaluation selection, not a
  deployable controller or proof of optimality.
- **Unchanged-learner competence:** every one of the three seeds in every one
  of those 12 cases is complete and safe, with gain strictly greater than +3.2.
  This conservative all-case rule is independent of the headroom gate.

Report historical above-source cases regardless of the primary decisions.
Retain case-level gains and learner medians, worst seeds and comparisons with
all manual witnesses. A near miss, a safety failure and failure despite a
witnessed improvement margin are different findings. Do not retune this batch.

## Evidence and verification

Refuse an existing output directory before collecting transitions. Archive
the protocol, native source/headers, CMake and runtime setup, catalog and model
inputs before the first trial; record exact SHA-256 values and the inspected
managed-container identity. Preserve partial reports and failed prefixes.

Retain per-trial full transition tapes, requested/emitted efforts and sparse
physical snapshots for playback. Recompute raw total reward, light and IMU
components, elapsed time and sample accounting from those tapes. Native
transition reward already integrates physics substeps: do not multiply it by
time again. Report cumulative reward at 6.4, 12.8, 32 and 64 s, a late window,
delivered light, energy and explicitly sampled telemetry maxima. Periodic light
exposure need not imply a stable light-seeking posture.

Fresh-world replay must reproduce every full recorded transition from the
emitted command tape, with matching initialization, metrics and final public
state. Verify retained tape/snapshot hashes and archived sources independently
of the report's stated gate. A command replay verifies native responses; it
does not establish unseen-world success or hidden-state identity. State clearly
whether learner command generation is additionally reconstructed.

Record native test results and independently verify aggregates and both gates.
Rehash preexisting artifacts and the frozen rover-v3 inputs before and after.
Do not write to the live workshop/light control APIs during collection.
The original held-out rover-v3 evaluation remains pending and untouched.

## Interpretation and next decision

Passing headroom supports this newly declared lighting distribution, not all
assemblies. Passing native learner competence supports that existing controller
on these known conditions, not the earlier PPO pipeline or cross-body transfer.
If headroom passes and learning fails, measure the failure before selecting a
new method. If no witness passes, improve the feasibility diagnostic in a new
declared batch without asserting that the body is incapable. Contact mechanics,
walking, arbitrary topology and hardware calibration remain separate work.
