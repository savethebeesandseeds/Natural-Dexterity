# The construction and motion experiment

Status: first construction milestone implemented and verified on 2026-09-05.

## Scope and entry point

Open `/construction.html` on the existing local Droid Blocks service. This is a
separate native successor world, exposed through `GET /api/construction/state`,
`GET /api/construction/spec`, and `POST /api/construction/control`. The older
sun-seeker and transmission experiments keep their own scene and controller.

The adopted broader direction is [Build and discover](BUILD_AND_DISCOVER_GOAL.md).
The first deliverable is an anchored two-link construction kit with one powered
hinge and one passive hinge. It is a vertical planar mechanism simulated in
MuJoCo, with no ground or self-contact in this version. It does not implement
arbitrary connector graphs, locomotion, compliance, or transfer across rebuilds.

## Saveable assembly

```json
{
  "schema": "construction_kit_v1",
  "name": "My first creature",
  "segments": [3, 3],
  "blocks": [],
  "sensor": {"segment": 1, "slot": 2, "side": -1}
}
```

Each of the two links contains 2–6 beam units, each 40 mm long and 15 g.
Snap-on 40 mm cubes weigh 12 g; the 36 mm IMU block weighs 20 g. A socket is
identified by segment (0 or 1), zero-based slot, and side (-1 or +1). At most
12 weight blocks may be attached; occupied sockets are unique, including the
sensor. Exactly one IMU is present in this bounded kit.

A weight has an ID and a socket, for example
`{"id":"block-1","segment":1,"slot":0,"side":1}`. The construction editor
supports extending/shortening the links, adding, moving and removing weights,
relocating the sensor, undo, starter shapes, and saving/loading an assembly JSON
file. Starter shapes are ordinary editable assemblies.

Each beam unit, weight, and sensor contributes native geometry, mass, and
inertia. Native compilation derives each articulated body's inertial tensor
from these parts. The motor uses the existing current/thermal/gearbox/friction
model and reflected rotor inertia. The passive hinge has no actuator, with
0.006 N m s/rad damping and 0.00002 kg m² armature.

A rebuild validates and compiles the replacement before replacing the active
world. Malformed fields, out-of-range sizes, and duplicate occupied sockets
are rejected. Rendering uses compiled native geometry during a run. A draft
is explicitly different from the applied body until its build succeeds.

## Sensing and exploration boundary

Physics runs every 2 ms, with a 20 ms control step. The attached IMU emits
local specific force and angular velocity every 10 ms. Its catalog nominal
5 ms latency is rounded up to 6 ms on the physics clock. The initial sample is
invalid until delivered. A control endpoint receives the newest delivered
sample, with sequence, acquisition time, delivery time, and age. Intermediate
samples are not reconstructed by the explorer.

Specific force is accelerometer output and includes the support force measured
at rest in gravity. It is not world-frame linear acceleration. World position,
orientation, tip trails, passive-joint angle, and construction labels remain
on the privileged display/diagnostic surface. The action routine receives only
the declared physical observation object.

Five fixed six-second trials compare three sway frequencies, paired taps, and
push-and-coast. Each begins from the same assembly at rest with fresh motor
state. The initial quiet interval and final coast are included in the six
seconds. A local shaft-feedback governor and effort slew limiter bound the
requests; they are sampled envelopes, not guaranteed physical speed limits.
All feedback flags stop exploration. Native motor safety remains independent;
the construction fixture also vetoes drive at its declared severe IMU limits.

The explorer records unique delivered samples: local angular-speed RMS and
peak, and specific-force magnitude RMS, peak, and variation. The cards are an
observed repertoire, not a ranking of trained skills. Several patterns may
produce similar outcomes. There is no hidden movement reward or parameter
training. The unchanged attached IMU reward is still computed and reported;
exploration does not optimize it.

## Lifecycle and replay

- `build` with `assembly`: requires a paused/nonrunning scene; validates and
  rebuilds, then clears the previous body's observations and motion cards.
- `explore`: starts five independent six-second trials, replacing this body's
  prior exploration cards. It requires a nonrunning scene.
- `pause` / `play`: preserve and resume the current trial, including sensor
  history and the motor sequence.
- `reset`: return the current assembly to rest and keep its motion cards.
- `replay` with `pattern_id`: replay a completed card from rest on the same
  assembly. Each computed motor request is checked against the original stored
  sequence. It preserves the recorded repertoire.

The API rejects unknown fields and invalid lifecycle requests before changing
session identity. Rebuilding increments the assembly revision. A new run,
replay, reset, or accepted rebuild increments session identity. The construction
session owns its own native world; it does not borrow the original rover's
external-agent ownership or observation space.

## Development protocol and evidence

`droid-construction-experiment --output NEW_DIRECTORY` compares the short body,
a longer body, and a single weight near the passive hinge versus the same
weight at the tip. Each body receives the same five motion patterns and a
zero-effort baseline, all for six seconds. Moving the weight keeps its mass
constant. The zero trial measures gravitational motion without motor effort.

The runner records assembly and source hashes, sensor descriptors, physical
coupling diagnostics, maximum motor/current/temperature/IMU readings, rewards,
feedback flags, stops, command traces, and exact native-observation replay.
Diagnostic poses do not influence action choice. The output directory must be
new; prior evidence is never overwritten.

All constructions, waveforms, and results here are exposed development data.
No frozen-v3 challenge is run. The old model, catalog, v3 manifest, and preserved
policy artifact must retain their original byte hashes. No child usability
study or hardware qualification is claimed.

## Verification results

All twelve native CTest suites passed in 7.62 seconds. The first focused run
also passed all three added suites. The final build had no warnings.
[Native CTest log](../build/construction-qa/final-native-ctest.log),
[build log](../build/construction-qa/final-native-build.log).

The [development report](../artifacts/reports/construction-v1-development-20260905/report.json)
contains 24 complete 300-step trials: four constructions, each with five
patterns and zero control. All had zero feedback flags and native safety
vetoes. Every trial's emitted observation sequence replayed exactly. Each of
the twenty driven trials recorded 300 unique delivered IMU samples.

| Construction | Moving mass (kg) | Quick sway: gyro RMS (rad/s) | Slow sway: peak passive-joint speed (rad/s) | Zero effort: peak gyro (rad/s) |
| --- | ---: | ---: | ---: | ---: |
| Short arm | 0.110 | 2.167 | 6.251 | 0.745 |
| Long arm | 0.140 | 2.348 | 8.012 | 0.456 |
| Weight near hinge | 0.152 | 2.347 | 7.185 | 0.165 |
| Same weight at tip | 0.152 | 2.493 | 7.427 | 0.141 |

These selected columns illustrate different physical responses; they do not
rank motion quality. All patterns and metrics are retained in the report.
Bodies share requested waveforms and the same feedback governor, while actual
motor efforts may differ with shaft feedback. The native mechanical tests
also compare actual construction responses to common effort sequences.
The initial condition is the same authored hanging pose with zero velocities,
not necessarily gravitational equilibrium: an offset IMU or weight can start
moving without motor effort.

The maximum observed motor speed was 6.277352 rad/s and passive-joint speed
8.012067 rad/s. This exceeds the governor's soft 3 rad/s target and illustrates
why it is not a physical speed guarantee. Maximum current was 0.320104 A,
temperature 25.043496 C, IMU angular speed 4.339796 rad/s, and specific force
15.274916 m/s². No declared hard-limit event occurred. These are simulator
development observations, not hardware-qualified limits.

The runner's existing-directory refusal was verified and left the report
byte-for-byte unchanged. All source hashes recorded in the report matched the
workspace after the run. No frozen input was changed.

[HTTP checks](../build/construction-qa/http-checks.json) verified nine rejected
requests without state mutation, running-build rejection, accepted assembly
roundtrip, exact pause preservation, resume progress, reset geometry, static
routes, and unchanged original rover state throughout construction actions.
Browser checks in the in-app workshop added a block, extended the lower beam
from three to four units, moved the same weight to its tip, moved the IMU to
the opposite tip socket, removed the weight, and undid that removal. Waking
compiled exactly that edited assembly. The run paused at 7.84 s overall
(second trial, step 92); subsequent draft edits, JSON file import, local saved
body restore, page reloads, and viewport changes preserved the paused native
run. It then resumed to exactly 30 s with all five 300-sample cards.

The quick-sway card replay completed at 6 s with exactly the retained tip
trail and unchanged repertoire. Normal browser sizing was restored after
390×844 and 1280×900 checks; neither layout had horizontal overflow. The
bench now names the current rhythm and trial, including while paused.
Browser console checks reported no warnings or errors. Retained live states:
[pause](../build/construction-qa/browser-custom-paused.json),
[after file/reload/viewport checks](../build/construction-qa/browser-after-file-and-viewport.json),
[completion](../build/construction-qa/browser-custom-completed.json), and
[replay](../build/construction-qa/browser-custom-replay.json).

Saving keeps a versioned copy in this browser and also prepares a JSON export.
Restore was verified after an edit and a reload. Actual JSON file import was
verified through the browser file chooser and changed only the draft. The
in-app download-event observer timed out, so JSON download completion is not
claimed for this browser; browser-local persistence provides the verified
save/restore path. The interface does not claim that an export finished.

The final bench is ready with the edited three/four-unit body, its tip weight
and sensor, and all five replayable cards. The original rover remains available
through the workshop link. The existing managed container was restarted in
place with immutable ID
`a3eabc3183e054aafd42304857b6a10b2c3a7cf15787fcf177149335e6aeaaaf`;
image, bind mount, port, device access, and storage were preserved. No container
or image was created, replaced, or deleted for this milestone.
