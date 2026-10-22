# The sun-seeker playground

**Milestone adopted:** 2026-09-05  
**Validation record:** completed locally on 2026-09-05; seven native suites and the HTTP/browser checks below passed.

The completed software milestone is **“It noticed the sunshine.”** A child can wake the existing
rover, see the light it senses, pause to inspect it, and try the sunshine on
the other side. This makes the current learned behavior available for play
before we introduce new bodies or a more complicated learner.

This milestone implements the first experience in the
[play-first proposal](PLAY_FIRST_PROPOSAL.md). It does not adopt that entire
proposal as a fixed roadmap. The controller is the preserved v3 policy, running
native inference in two already published training worlds. No parameters are
trained or updated by playing, and the experience makes no new generalization
claim.

## Trying the experience

Use the existing managed service and the
[README launch procedure](../README.md#safe-first-launch), then open the
[local workshop](http://127.0.0.1:43117/).

1. The sun-seeker starts ready, with the sunshine on the left. Press **Wake the rover** to
   wake it, or choose the right side to prepare that world first.
2. Watch the light source, its world map, the rover's movement, and the
   illuminance reading at its eye. The sensor glow follows measured light.
3. Pause and resume the same run. Reset starts that side again with fresh
   physical state and cleared controller memory. Choosing a side also prepares
   a fresh run and leaves it ready.
4. A learned run completes after 20 simulated seconds. Reset prepares another
   attempt; completion never silently loops into a new episode.

The larger workbench keeps the invention in view. Airship view changes the
framing. **Peek inside** opens machine details with the controller and provenance;
**Switch to cruise demo** explicitly prepares the old scripted demo, paused.
**Prepare the sun-seeker** returns to learned control. Changing the view alone
does not change the controller.

## Worlds and preserved policy

The playground accepts a side name, never a browser-supplied numeric seed.
These are the first negative/positive pair in the published v3 training split:

| Playground side | Native `uint64_t` seed, written here as a decimal string |
| --- | --- |
| `left` | `"6106049681611768608"` |
| `right` | `"6990021049862345368"` |

Both values exceed JavaScript's exact integer range. They are native integer
constants; browser requests use only `left` or `right`. The runtime and tests
check their membership in the training split, exclusion from validation and
test splits, and expected physical light side. Deriving split metadata does
not instantiate or run any of those other worlds. The pair was already
published in [Learning design history](LEARNING_DESIGN_HISTORY.md).

The episode profile is `light_search_1d_v1`, the assembly is `demo_rover_v0`,
and the controller is `shared_linear_memory_v2`. Its source artifact is
[`artifacts/light-search-policy-v3-seed0.json`](../artifacts/light-search-policy-v3-seed0.json).

| Integrity identity | SHA-256 |
| --- | --- |
| Exact preserved file bytes | `6f7d5069bdb884d75f1241629d437c039a2f7a486633d9ef2d7f6b70db0a5582` |
| Canonical artifact `self_sha256` | `4dd4d6ccadecb0b3b69d3b3d412af83eff15208b19058701538dbdf83a4c32b5` |

Startup and each learned reset validate both identities and compatibility with
the actual runtime: model bytes, canonical module catalog, full environment
specification, feature schema, motor IDs, assembly, API version, and control
interval. These are integrity and compatibility checks, not signatures or
proofs of authorship. Inference uses the loaded, validated artifact in memory.

`exposure: "known_training_world"` means precisely that these worlds are known
training data. Their visible behavior is a demonstration and a development
check. It is not held-out evidence, an evaluation of arbitrary assemblies, or
evidence of educational benefit. The v3 challenge decision remains separate
and pending; this playground does not invoke its evaluation procedure.

## Native control and observation boundary

[`PlaygroundSession`](../include/droid/playground.hpp) coordinates the existing
`DroidEnvironment`. Its worker schedules native inference at the nominal
0.02-second control interval. Each control step advances ten existing
0.002-second physics steps; 1,000 control steps complete the run. The displayed
20 seconds measure simulated time, so wall-clock duration can be longer on a
busy machine.

The only arguments supplied to `SharedLinearMemoryPolicy::act` are the current
policy-safe observation and opaque motor IDs. The policy receives the mounted
sensor observations and actuator feedback already defined by the frozen
environment. It receives no side, seed, source position, robot pose, reward,
termination metadata, browser state, or map information. Light-source position
is read separately for visualization. The policy's one-step memory survives
pause/resume and is cleared on every learned reset.

All HTTP access to the environment and every learned tick pass through the
session's mutex. A reset or controller handoff cannot interleave with a learned
step. The native demo worker is paused while the learned controller or an
external agent owns physics. The server has one shared session, so multiple
browser tabs operate the same robot.

This work adds a session coordinator and browser presentation. It changes no
simulator dynamics, model, catalog, reward rule, policy encoder, learned
parameters, frozen environment specification, or experiment protocol.

## HTTP controls and metadata

`POST /api/playground/control` accepts an object with `action` and an optional
`side`. Unknown fields, invalid sides, and sides supplied with another action
are rejected.

| Request | Effect |
| --- | --- |
| `{"action":"play"}` | Run or resume the selected learned/demo controller; a completed learned run requires reset. |
| `{"action":"pause"}` | Pause the selected controller without clearing memory. |
| `{"action":"reset","side":"right"}` | Prepare that learned world, clear memory, and leave it ready. Omitted side preserves the last learned side. |
| `{"action":"demo"}` | Explicitly prepare the fixed scripted demo, paused. |

The legacy `POST /api/control` still accepts `play`, `pause`, and `reset`, but
now preserves learned/demo selection. Its reset resets the selected mode.
Play, pause, and legacy reset reject external ownership with HTTP 409 rather
than silently changing controllers. An explicit playground reset or demo
selection takes ownership back.

`POST /api/agent/reset` explicitly gives an external agent ownership and retains
the existing generic reset API. `POST /api/agent/step` requires that ownership.
Agent reset/step results, policy observations, and the agent specification do
not acquire playground metadata. Existing generic agent endpoints are not a
playground world selector.

`GET /api/state` appends a `playground` object alongside the unchanged native
visualization fields:

| Fields | Meaning |
| --- | --- |
| `mode` | `learned`, `demo`, or `external`. |
| `status` | `ready`, `running`, `paused`, `completed`, `stopped`, or `error`. External stepping does not schedule autonomous ticks. |
| `side` | Last successfully prepared learned side, `left` or `right`. |
| `elapsed_s`, `duration_s`, `step`, `max_steps` | Learned progress, with a 20-second/1,000-step horizon. The horizon does not constrain the generic external API or cruise demo. |
| `session_id` | Monotonically increasing identity for successful learned, demo, or external resets. |
| `light_position_m` | Actual light-source position for display only. |
| `initial_illuminance_lux`, `illuminance_lux` | Initial and current light reading. Learned values come from policy-safe sensor observations. |
| `policy_id`, `artifact_sha256` | Selected controller and, when a learned policy has loaded, its canonical artifact identity; the hash is otherwise null. |
| `exposure` | `known_training_world` for learned mode; null for demo/external. |
| `error` | Present when the selected session has failed. |

Use `playground.mode` and `playground.status` for playback UI. The unchanged
base `running` and `controller` fields describe the native simulator path;
synchronous learned steps use its paused external-effort path. A prepared,
paused demo activates the native demo controller on Play.

Missing, altered, or incompatible artifacts leave the server available with a
visible learned error and no automatic motion or fallback. An inference,
observation, or native-step failure also stops the session and preserves its
physical state for inspection. Play cannot resume potentially inconsistent
memory; an explicit learned reset revalidates and rebuilds the session. Native
demo failures are also surfaced. `GET /healthz` includes playground status and
returns HTTP 503 when the native environment or selected session has failed.
`GET /api/state` remains inspectable. A failed learned-reset response contains
the error state, so clients must inspect `playground.status` even when the
control request returns HTTP 200.

## Verification and next question

The new `native-playground` suite compares the serialized visualization state
at every step with a separately instantiated observation-only frozen controller
for both approved worlds. These display values include rounded physical fields;
this is an integration comparison, not a claim of exact unrounded physics replay.
The suite also checks
pause/resume, reset repeatability, completion, ownership handoffs, artifact and
runtime rejection, failure recovery, and realtime worker isolation. Its two
full runs check for a higher final light reading. These are development checks
on known data. Existing native suites remain required; some exercise synthetic
or small training-library fixtures, independently of this inference-only
playground.

**Validation results, 2026-09-05:**

- The authoritative CMake build of `droid-native-validation` passed in the
  existing managed container. All seven CTest suites passed in 5.31 seconds.
  The complete output is retained in
  [`build/playground-qa/native-ctest.log`](../build/playground-qa/native-ctest.log).
- Five malformed playground-control requests returned HTTP 400 without
  changing controller ownership. HTTP checks also verified mode-preserving
  legacy controls, HTTP 409 on agent stepping without ownership, explicit
  external ownership, and deliberate return to learned-ready state. Records:
  [rejections](../build/playground-qa/http-rejections.json) and
  [ownership](../build/playground-qa/http-ownership.json).
- In-app browser checks exercised both learned runs, completion, side reset,
  pause/resume, controller details, explicit cruise-demo selection, and both
  workbench and airship views. The right run remained at step 189 while paused
  during view changes and a 390-pixel mobile layout check, then resumed to
  completion. The temporary viewport was restored. JavaScript syntax checking
  passed. These checks do not establish child usability.
- The existing container was restarted in place, preserving its immutable ID
  `a3eabc3183e054aafd42304857b6a10b2c3a7cf15787fcf177149335e6aeaaaf`,
  image, mounts, devices, and port configuration. The modified Dockerfile adds
  only the frozen artifact to the clean validation-stage inputs; a new Docker
  image build was not performed for this local milestone.
- Before/after SHA-256 checks confirmed the policy file, experiment manifest,
  module catalog, and model bytes were unchanged. Runtime compatibility checks
  also accepted the original environment and feature schemas. No official
  held-out evaluator or new training experiment was run.

The observed browser runs are **known-training-world development evidence**:

| Side | Initial light | Final light | Final x | Completion |
| --- | ---: | ---: | ---: | --- |
| Left | 65.535858 lx | 222.732936 lx | -9.540082 m | 1000 steps / 20 s |
| Right | 79.688643 lx | 323.821376 lx | +9.766625 m | 1000 steps / 20 s |

Full endpoint snapshots are retained for [left completion](../build/playground-qa/left-completed.json),
[right pause](../build/playground-qa/right-paused.json), and
[right completion](../build/playground-qa/right-completed.json). These files
live under the local build directory; archive them explicitly if moving this
milestone to another machine. They are not held-out acceptance reports.

The dotted comparison trail is the previous visible run on that side in the
current browser page. It is sampled for display, may cover only part of a run,
and disappears on page reload. Native simulation state is server-owned and
survives a page reload; a running episode can continue while a tab is hidden.

The present body is still the fixed four-motor rover. There is no assembly
editor, discovery of arbitrary transmissions, online parameter learning, or
physical hardware qualification. The light sensor still uses the original
analytic readings at control time; catalogued sampling delay has not been
introduced into the preserved experiment. The displayed lamp and map make a
stimulus understandable but do not model a new optical system. Child usability
has not yet been tested.

The next proposed experiment asks **“I changed its body.”** Declare a small
successor development family with a normal transmission, one reversed
transmission, and one disconnected transmission. Compare what a simple shared
controller can discover through physical feedback, make the consequences
visible, and exercise realistic sensor timing before adding substantial
adaptation machinery. Version those bodies and observations separately from
v3 and reserve a fresh challenge set for any future generalization claim.
