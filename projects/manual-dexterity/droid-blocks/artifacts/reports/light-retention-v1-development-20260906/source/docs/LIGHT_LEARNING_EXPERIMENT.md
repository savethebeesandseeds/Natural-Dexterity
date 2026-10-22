# Find the light, then learn to stay with it

Adopted 2026-09-05 after the owner asked for a light sensor and a sun, then said “let’s do it.” This milestone develops the inner engine: observation, recent memory, action, sensed consequence, and learning while physical motion continues.

The existing two-link workshop stays available. A separately versioned light-enabled assembly adds a real sensing block and a controllable light source. The controller receives delayed mounted sensor samples and local actuator feedback. The sun position, body geometry, and passive joint state are diagnostic information only. The unchanged sensor catalog supplies the sum of sensor reward rates; the motor has no reward vote, and hard safety stays independent of preferences.

## Initial development protocol (declared before the first run)

- Native MuJoCo with the existing .002-second physics and .02-second control steps; no contact in this fixture.
- Two known development assemblies: the default light-enabled [3,3] body, then the same body with one .012 kg block at segment 1, slot 0, side +1. Fresh memory for each body, no claim of transfer across rebuilds.
- Controllers: online learner, zero effort, a fixed .5 Hz sinusoidal rhythm, and seeded random positive/negative/coast pulses. All share the same effort, slew and local speed governor. Equal 120-second simulated horizon per run; no resets at action boundaries.
- Seeds 1, 2 and 3 for the learner and random controller; deterministic controls run once per assembly.
- Sun starts at [0.30,0,0.20] m with intensity 1000 lux. After 60 seconds it moves to [-0.30,0,0.20] m. This is the sole scheduled intervention. The world, learner, and pending sensor deliveries continue unchanged at that moment.
- Record the whole-run and 0–60 / 60–120 second integrated sensor reward, plus the final 20 seconds of each phase, unique light samples, electrical energy, maximum current/temperature/speed, feedback flags and safety vetoes. The learner is assessed against every baseline, including a quiet body that may already see substantial light.
- Preserve commands, source hashes, protocol constants and observation-trace hashes. Replay each command trace with the same sun intervention to verify exact emitted observations. Reports reserve a new output directory and refuse overwrite.
- All results are development evidence. If implementation or parameters change after observing a run, preserve that report and exact source, describe the change, and use a new report directory. These are not independent held-out tests. The frozen original v3 model, catalog, policy, experiment and pending challenge remain untouched.

Success means a working, inspectable online learning experiment with causal sensing, continuous dynamics, honest comparisons and safe controls. Higher return and recovery after moving the sun are empirical questions. A working learning update alone does not establish that it controls this body well.

Implementation, checks and observed results will be appended after validation.

## Implemented engine boundary

`ConstructionWorld` accepts a separate `construction_kit_v2` assembly. It
retains the two native hinges and original IMU and adds a 20 g light block at an
unoccupied socket. Its receiving face points along either direction of the
mounted segment's local Z axis. The site is on that actual cube face. Received
light is `I * max(0, dot(normal, direction_to_sun)) / (1 + (distance / .5m)^2)`.
This bounded point-source model has no occlusion or noise. The light sensor
samples every .1 s and delivers .02 s later. Moving the source changes future
acquisitions, including when an old sample remains queued. The sun position is
visible to the scene renderer, never to the policy.

`LightLearner` is `history_expected_sarsa_v1`: linear Expected SARSA with
eligibility traces and a small buffer of recent option transitions. It chooses
negative effort, coast, or positive effort for .4 s at a time, after .1 s of
initial quiet. Its target includes the discounted integral of all attached
sensor rewards during the option plus expected subsequent value. The discount
is .92 per .4 s, also applied within an option. It stores 64 control transitions
and 32 completed options, using four recent replay updates per new option.
There are 56 bounded features derived from local encoder, motor feedback, IMU,
light, and recent sensor/action/reward history. No feature uses geometry,
world orientation, sun coordinates, a target angle, or a desired motor sign.
The first three options are tried in seeded shuffled order; exploration then
continues with an epsilon floor of .12. This is an approximate learning method,
not a convergence guarantee.

`LightControl` is the common driver for the browser session and development
runner. All four modes use the same actuator envelope: requested signed effort
is capped at .35, limited using local shaft feedback toward a soft 3 rad/s
target with gain .05, and slewed by at most .1 each control step. Gravity can
backdrive the shaft above that soft target; the native hard motor and IMU
safety rules remain active. Actual energy can differ between modes and is
reported. This protocol gives equal time and actuator limits, not equal energy
consumption.

`LightSession` owns a separate native world and controller. It steps
continuously for 120 simulated seconds. Pause preserves physical state and
memory. Moving the source preserves the entire try. Reset, a fresh start, or a
rebuild clears memory and resets the body while retaining the selected sun.
Rebuilding a body currently does not transfer learned parameters. The session
records the initial sun and later source moves, and emits a .1 s light history
for the browser.

The sunshine bench is `/light.html`; it can explicitly copy the assembly from
`/construction.html` and find a free light socket. The construction workshop's
v1 schema and fixed probes remain supported. Its original world source and
header were archived before extension at
`artifacts/source-archives/construction-v1/`. Frozen-v3 inputs remain separate.

Native APIs: `GET /api/light/spec`, `GET /api/light/state`, and
`POST /api/light/control`. Actions are `start` with a controller `mode`,
`pause`, `play`, `reset`, `build` with a v2 `assembly`, and `sun` with its
`position_m` and `intensity_lux`. Invalid controls reject without replacing
valid state. The learner rejects malformed or stale inputs, out-of-order sensor timestamps/sequences,
mutated held sensor packets, undeclared fields, and motor/native fault flags.

## First measured result and a bounded follow-up

The original `history_expected_sarsa_v1` result is preserved at
`artifacts/reports/light-learning-v1-development-20260905/report.json`, with
exact source snapshots and hashes. All 16 declared trials completed without
feedback flags or native safety vetoes, and their controllers and emitted
observation traces replayed exactly. All 15 native suites passed. All 24 old
construction command sequences and observation hashes also matched the
original construction report exactly after this optional light extension.

The first learner did **not** outperform the controls. Median whole-run sensor
return across its three seeds was 102.557 on the default body and 102.330 with
the added side weight. Zero effort achieved 104.005 and 103.828 respectively;
the fixed rhythm achieved 104.013 and 103.843. The learner's median last-20-second
light after moving the sun was 352.8 / 334.5 lux, versus 363.9 / 346.9 lux for
zero effort. It used about 19.5 J per run; zero effort used no electrical energy.
These results demonstrate the learning machinery and expose a control failure,
not successful light tracking.

The diagnostic action counts suggest an initialization problem: positive
ambient reward raises the initially selected action's value, while rarely
selected alternatives retain low values. The same seeded habits recur on both
bodies. The next development version will change only the equal initial action
value: estimate a continuing value from the measured quiet-start reward and
initialize all action biases equally. This uses received rewards, not a sun
position or desired motor direction. The raw summed-reward TD objective,
physical bodies, source schedule, seeds, horizon and all other parameters stay
fixed. The same 16 trials will be repeated once in a new report directory; both
reports remain exposed development evidence.

The follow-up is `history_expected_sarsa_v2`. Its only behavior change is the
common initial action value. With per-control-step discount
`g = .92^(1/20)`, let `mean_r` be the mean actual summed transition reward over
[.02,.04], [.04,.06], [.06,.08] and [.08,.10] seconds of quiet startup. Each
action's bias starts at `mean_r / (1-g)`; all other weights and traces remain
zero. Those are four fully delivered reward intervals, sharing the first held
light acquisition, not four independent light measurements. The first
[0,.02] interval is excluded because it contains initial missing readings.
The estimate remains fixed after a sun move. Initialization has its own
recorded diagnostics and does not count as a learned update. Native tests
check equal starting values, zero TD residual for unchanged background
reward, and the exact residual when subsequent raw summed reward decreases.

## Final development result

The repeated report is
`artifacts/reports/light-learning-v2-development-20260905/report.json`.
Its 16 trials all completed and replayed exactly, without feedback flags or
native safety vetoes. The source snapshots match the recorded hashes. The
following are medians over three seeds for learning/random modes; zero and
rhythm are deterministic single runs. Return is the full 120-second integral
of summed sensor reward. Light is the mean over seconds 100–120, after the
source moved at second 60.

| Body | Control | Sensor return | Final light (lux) | Electrical energy (J) |
| --- | --- | ---: | ---: | ---: |
| Default | Learner v2 | 105.663 | 459.8 | 21.64 |
| Default | Quiet | 104.005 | 363.9 | 0 |
| Default | Fixed rhythm | 104.013 | 367.9 | 16.25 |
| Default | Random | 103.542 | 362.0 | 17.85 |
| Side weight | Learner v2 | 105.075 | 410.0 | 22.51 |
| Side weight | Quiet | 103.828 | 346.9 | 0 |
| Side weight | Fixed rhythm | 103.843 | 350.0 | 16.64 |
| Side weight | Random | 103.569 | 344.4 | 18.25 |

Every learner seed earned greater total return than every comparison run on
its corresponding body. The learner's median final light exceeded quiet
control by 26.4% / 18.2%; that is a light-reading comparison, not the percentage
increase in the log-scaled reward. This used more electrical energy, which is
measured but has no reward vote because no power sensor is attached.

A descriptive check added after the declared analysis also compared received
light at seconds 61–65 with seconds 100–120: all six learner runs rose from
266–306 lux shortly after the source move to 407–470 lux late in the try.
This is consistent with recovery after moving the sun. It is not a separately
held-out recovery test. The initial weak result and this corrected result are
both preserved; the new evidence supports a useful learning loop in this
bounded fixture, not arbitrary-body or hardware capability.

Across the final 16 runs, maximum current was .375 A, maximum motor temperature
25.469°C, and maximum shaft speed 6.930 rad/s. The latter illustrates why the
3 rad/s governor is a soft control target rather than a physical speed limit.
All 15 native suites passed again after the correction, with six focused
learner test groups. The report summarizer verifies source snapshots, phase
reward/energy addition and sensor-vote addition:
`node tools/summarize-light-report.mjs artifacts/reports/light-learning-v2-development-20260905/report.json`.

## What this suggests next

Keep the engine interfaces and make the next experiment about memory and
adaptation: several source moves, light temporarily disappearing, and changing
sensor placement, with a separate protocol that has not yet been used to tune
this learner. Compare continuing learning with frozen learned values to
separate remembered control from online adaptation. Before adding contact,
extend observation histories and action durations only when this evidence
shows they are needed. Broader body graphs and experience transfer across
rebuilds still need their own experiments.

## Live workbench verification

Browser QA covered the native light scene and chart at 390 px and 1280 px,
then restored the normal viewport. The owner's current body ([2,6] beams,
weight at lower slot 3 right, IMU at lower slot 3 left; native zero-based slot
indices) was copied explicitly and given a light sensor at lower slot 4 right.
It completed a full 120-second v2 run with 300 choices, 299 completed-option
updates and no safety flags. This third user construction is additional known
interactive development exposure, not part of the declared two-body report.

The first live run also verified pause/resume and a moved sun at second 67:
while paused, elapsed time, full learner diagnostics, sensor observations and
history remained exactly unchanged after moving the source. The separate
headless comparisons verify uninterrupted physical learning across the
scheduled source move. Browser checks rejected an occupied mount, exercised a
face flip and quiet control, and observed zero lux from the wrong hemisphere
then about 816 lux with the source below the downward-facing sensor. Invalid
HTTP controls returned 400 without changing paused state. No browser warning
or error logs were recorded.

The owner's original construction and all five existing motion cards were
restored exactly after the server restart. The final sunshine page retains
that body's completed v2 try, ready to start another. Report overwrite refusal
was checked using a disposable occupied output fixture with a pre-test hash;
its sentinel and the real preserved report remained unchanged. QA snapshots
and logs are in `build/light-qa/`.
