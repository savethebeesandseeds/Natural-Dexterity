# I changed its body

**Adopted:** 2026-09-05, following the owner's request to continue after the
first visible sun-seeker movements. This is a development experiment.

The question is whether a rover can learn a useful local motor command after
one physical transmission changes. The child chooses a body, wakes it, watches
it try its motors, and sees the preferences it measured. The controller is a
small handwritten probing algorithm with memory for the current run. It does
not load the frozen v3 parameters, identify arbitrary LEGO assemblies, or
retain experience between rebuilds.

## Play

Open the [workshop](http://127.0.0.1:43117/) and choose **Change its body**.
Choose an ordinary, reversed, or disconnected front-left transmission, then
choose **Discover each motor** or **Try motors together**. Changing a body,
strategy, or sunshine side prepares a fresh run and clears its measured
preferences. Pause preserves the physical state and memory. Each run finishes
after 40 simulated seconds; it never silently repeats.

The motor cards display the current probe, measured positive or negative effort
preference, and light-trend evidence. A preference is local evidence from the
probe, not a named physical role. **No clear preference** can mean weak motion,
an unhelpful light gradient, contradictory evidence, or a disconnected drive.
It is not a diagnosis of the connection. Together mode measures the combined
effect and cannot identify an individual motor's contribution.

The drawing shows actual wheel angles and the changed motor's actual shaft
angle. A marked coupling shows the selected physical connection. These display
details are privileged visualization and never enter the controller.

## Three physical bodies, one declared family

The successor profile is `body_discovery_1d_v1`. It requires an explicit matching
assembly:

| Assembly | Changed transmission |
| --- | --- |
| `transmission_normal_v1` | Independent motor output shaft coupled to the wheel with ratio +1. |
| `transmission_reversed_v1` | Same shaft and wheel, coupled with ratio −1. |
| `transmission_disconnected_v1` | Same shaft and wheel, coupling released. |

MuJoCo compiles this successor from the original model in memory. The changed
motor's command acts on a separate shaft degree of freedom. A compliant joint
equality transmits movement to the wheel when connected. The encoder measures
the shaft; the disconnected wheel remains part of the contact simulation and
can freewheel. The added shaft mass and inertia are present in all three
successor variants. This is an idealized transmission family, without gear
teeth, backlash, mechanical breakage, or hardware calibration.

The remaining body retains free translation and rotation. The profile name
describes the source's left/right placement; it does not constrain the rover
to a rail. A solo wheel can turn the rover, and its sensor's movement can change
the reading even when the chassis barely translates. That makes identifying a
stable motor preference a hypothesis to test.

The original `models/droid.xml`, catalog, v3 artifact, v3 manifest, and original
environment specification stay unchanged. Returning to the original learned
sun-seeker restores the original native model and its original sensing behavior.
The successor descriptor is exposed separately through the native
`body_experiment_spec()` method.

## The eye takes time

Only in this experiment, ambient light is sampled every 100 ms and delivered
40 ms after acquisition. The initial sample is acquired at time zero and is
invalid until its delivery. Between deliveries, the controller receives the
same held sample with the same sequence and acquisition timestamp. The reward
sensor also votes from its delivered sample. Touch sensing and actuator
feedback keep their existing behavior.

The sequence, `sample_time_s`, `delivered_time_s`, and `age_s` make causality
inspectable. The timing is an explicit experimental override, not a claim to
match the catalog's nominal sensor. The underlying analytic radial light field
still has no noise or occlusion. Waiting for a delivery does not recalculate
the reading at the rover's newer position.

## A small learning rule

`paired_light_probe_v2` receives only public physical observations and opaque
motor IDs. It receives no body choice, source side or location, seed, robot
pose, reward, connector graph, or UI state.

It starts with 0.3 seconds at zero effort. It then gives each motor a one-second
positive pulse and a one-second negative pulse, with 0.6-second zero-effort
washouts. Effort is bounded to 0.35 and changes by at most 0.1 per 20 ms control
step. A local shaft-speed governor reduces this requested effort as the shaft
speeds up; it applies identically to every motor and both strategies. It uses
only actuator feedback and never the declared connection. Only fresh,
unsaturated light samples acquired in the latter 0.6 seconds
of a pulse contribute; delayed arrivals are retained during washout.

A least-squares light slope is measured for each polarity. The controller
retains a direction only if the two slopes have opposite signs and both exceed
the declared threshold. Otherwise that motor's preference is zero. The signed
evidence displayed in lux/second is half the positive slope minus the negative
slope. After probing, motors hold their retained directions for the remainder
of the run, still subject to the speed governor. This measures a local response and does not continually relearn as
the rover turns or passes the source. All constants and semantics are available
in `body_controller_spec()` and captured in the report.

**Discover each motor** probes in the enumeration supplied at reset. ID spelling
does not supply geometry; renaming IDs while preserving enumeration leaves the
schedule unchanged. Observation-array reordering does not affect actions.
Changing the physical probe order can change outcomes and is not claimed to be
invariant. **Try motors together** uses the same procedure with all motors
commanded together, then retains one shared direction. It finishes its probes
sooner; both strategies get the same total 40-second interaction budget.

Missing, stale, malformed, or faulted required feedback stops the experiment.
An error preserves the physical state, prevents further autonomous ticks, and
requires an explicit reset. Existing simulator safety vetoes remain outside
reward and cannot be canceled by a favorable light reading.

## Exposure and retained evidence

The [development protocol](../config/body_experiment_v1.json) was written before
the first comparison. Every body in this family is exposed development data.
The only worlds are the two already published v3 training seeds:
`6106049681611768608` (left) and `6990021049862345368` (right). The browser sends
side names, avoiding JavaScript's integer precision limit. No v3 validation or
held-out rollout is needed or run by this experiment.

The runner compares all three bodies and both worlds under individual probing,
shared probing, and zero effort: 18 episodes. It records light readings, sensor
reward integral, effort, simulated safety events, controller evidence through
time, and final privileged state for diagnosis. The v3 artifact is recorded as
historical context; its weights are not used by either successor controller.

Run the authoritative build using the existing managed container and README
procedure, then invoke the native executable from the project directory:

```sh
./build/native/droid-body-experiment --output-dir NEW_DIRECTORY
```

The output directory must be new and its parent must exist. It is reserved
before any rollout. The runner retains a running report before starting and
updates it after every completed episode, including source hashes, runtime,
body descriptor, controller specification, and exposure. Failure retains an
error record and all completed episodes. A controller stop retains that
episode's observation, physical state, and measured evidence, then the runner
continues with the next fresh episode; it never resumes a failed controller.
Existing result directories are never
overwritten. Interrupted partial episodes are not fully replayable from this
summary; these reports do not pretend to be transactional challenge evaluation.

This small development comparison has no official acceptance decision and
supports no claim about unseen bodies or educational benefit. Any future
generalization claim needs a fresh, separately versioned challenge family kept
outside that future development loop. The original pending v3 challenge remains
separate.

## Native and HTTP integration

`POST /api/playground/control` with
`{"action":"body","side":"left","variant":"reversed","strategy":"discover"}`
prepares the explicit successor in `body_lab` mode. Side, variant, and strategy
are fixed enumerations; the browser cannot supply a numeric seed. Omitted body
options preserve the last choice. `play` and `pause` preserve selected mode;
legacy `/api/control` reset also preserves it. Playground `reset` explicitly
returns to the original learned sun-seeker, while `demo` selects the cruise
demo. External agent ownership still requires explicit takeover and blocks
autonomous play until an explicit playground mode selection.

`GET /api/state` retains the base visualization and adds controller evidence in
`playground.body_experiment`, along with physical shaft/coupling details in
`body_experiment_physics`. `playground.artifact_sha256` is null in this mode;
its `policy_id` identifies the probing controller. Missing initial light samples
are explicitly marked invalid and shown as waiting. Failed model preparation
preserves the previous scene and mode with an error, preventing old evidence
from being mislabeled as a newly built body.

The native `reset` API accepts the explicit successor assembly/profile pair.
The original `spec()` stays byte-for-byte compatible with v3; the separate
`body_experiment_spec()` describes the successor. Successor resets reject
physics timesteps that cannot represent the sensor's period and latency as
integral steps. The shipped physics timestep is 2 ms.

## Verification and outcome

The first controller, `paired_light_probe_v1`, stopped after 0.38 seconds on a
disconnected motor's `voltage_limited` feedback flag. It was not an overspeed,
stuck, overtemperature, or native safety-veto event: the freely spinning shaft
reached about 6.64 rad/s and the requested current exceeded available voltage
headroom. The conservative controller refused further actions.

The [initial partial comparison](../artifacts/reports/body-discovery-v1-development-20260905/report.json)
retains its 12 completed normal/reversed episodes and error. The
[diagnostic comparison](../artifacts/reports/body-discovery-v1-diagnostics-20260905/report.json)
retains all 18 conditions, including all four disconnected controller stops and
their full final observations. The first runner's incomplete stopped episode
motivated the more detailed per-episode failure recording described above.

The second controller adds the prospective local speed governor. It continues
to reject every reported fault flag, and does not change the motor model,
catalog, hard safety rules, probe timing, response estimator, or development
worlds. This is an explicitly exposed development revision; the earlier failure
has not been erased or reinterpreted as a successful run.

The final governor uses a soft 3 rad/s target, with magnitude
`min(0.35, 0.05 * max(0, 3 - direction * shaft_velocity))`, followed by the
unchanged slew bound. At rest this permits 0.15 effort. It is a sampled control
envelope, not a guaranteed speed limit. An intermediate target of 4 rad/s
overshot the declared development test envelope during the negative pulse;
the target was lowered without changing the test threshold or tuning reward.
Both outcomes are retained in
[governor development checks](../build/body-qa/governor-development.log).
The diagnostic v1 source and header are preserved in
[`artifacts/body-probe-v1-source`](../artifacts/body-probe-v1-source/).

**Final native results, 2026-09-05:** all nine suites passed in 7.04 seconds;
see the [complete CTest output](../build/body-qa/final-native-ctest.log).
The [governed development report](../artifacts/reports/body-discovery-v1-governed-20260905/report.json)
contains all 18 complete 2,000-step episodes. No episode had a feedback flag,
native safety veto, controller stop, or environment termination. Across those
episodes, the largest recorded absolute shaft speed was 4.416992 rad/s, current
was 0.496495 A, and temperature was 26.216081 degrees C. These are simulated
development observations, not hardware-qualified bounds.

| Body | Sunshine | Discover: final lux | Together: final lux | Zero: final lux | Individual preferences, M01–M04 |
| --- | --- | ---: | ---: | ---: | --- |
| Normal | Left | 95.90 | 113.02 | 64.89 | −, −, −, − |
| Normal | Right | 123.15 | 146.09 | 80.56 | +, +, +, + |
| Reversed | Left | 97.43 | 82.85 | 66.19 | +, −, −, − |
| Reversed | Right | 121.00 | 102.62 | 78.83 | −, +, +, + |
| Disconnected | Left | 90.32 | 102.72 | 65.54 | unclear, −, −, − |
| Disconnected | Right | 113.42 | 130.80 | 79.69 | unclear, +, +, + |

Initial delivered readings were 65.535858 lux on the left and 79.688643 lux
on the right. Individual probing found a changed sign for the reversed motor
and assigned zero preference to the disconnected motor. Both strategies
improved light over zero control in all six body/world pairs. Individual
probing beat shared control on the reversed body in both final light and
integrated sensor reward. Shared control did better on the normal and
disconnected bodies within this budget: individual probing spends longer
collecting evidence. This supports the usefulness of local motor memory in
this small reversed-transmission family, not a general advantage of more
probing or a claim of physical disconnection identification.

HTTP checks rejected eight malformed requests without changing session identity,
preserved body selection on legacy reset, enforced external ownership, restored
the original frozen policy, and prepared a fresh body run. The retained record
is [HTTP checks](../build/body-qa/http-checks.json). Browser checks verified
initial waiting-state sensing, the actual v2 identity in details, body/side/
strategy switching, visible opposite motor preferences, and pause/resume through
desktop/narrow viewport changes. The viewport was restored. The reversed-right
run paused at step 482, resumed, and completed at exactly 2,000 steps and
120.996691 lux, matching the native comparison. The disconnected-left run also
completed at exactly 2,000 steps and 90.323289 lux, showing zero preference and
zero effort for M01. The unchanged completed state survived switching to
airship framing. Retained live states:
[pause](../build/body-qa/browser-reversed-paused.json),
[reversed completion](../build/body-qa/browser-reversed-completed.json), and
[disconnected completion](../build/body-qa/browser-disconnected-completed.json).
The runner's existing-directory rejection was also checked: it refused before
rollouts and left the final report's hash unchanged. Every input hash listed in
the final report matched the current workspace after verification.

The existing container was restarted in place with ID
`a3eabc3183e054aafd42304857b6a10b2c3a7cf15787fcf177149335e6aeaaaf`.
Its image, bind mount, device access, and published port were preserved.
No new Docker image was built for this local milestone. Before/after hashes
confirmed the original model, catalog, v3 manifest, and frozen policy bytes
were unchanged. No official held-out evaluation or new parameter-training
experiment was run. No child usability study has yet been performed.

The next useful question is whether a measured preference remains useful after
the body or light changes *during* a run. Rechecking uncertain or deteriorating
responses is a better next test than assuming these initial four signs remain
correct indefinitely.
