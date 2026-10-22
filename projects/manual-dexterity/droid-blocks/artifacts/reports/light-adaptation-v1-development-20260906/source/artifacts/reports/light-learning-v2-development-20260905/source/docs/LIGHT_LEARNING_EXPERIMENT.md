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
