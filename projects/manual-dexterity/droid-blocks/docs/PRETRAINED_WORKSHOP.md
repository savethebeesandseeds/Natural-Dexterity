# Build it. Wake what it already knows.

The live [pretrained workshop](http://127.0.0.1:43117/pretrained.html) carries
one frozen brain into the supported first-build family. Choose a long upper
beam `[5,3]`, a long lower beam `[3,5]`, or a weighted `[5,5]` body, put the
single motor at the anchor or elbow, and place the lamp on either side.
Rebuild applies these choices; Wake begins a 64 simulated-second opportunity.
Pause and Resume continue the same opportunity.

This is the build/wake product milestone following the
[completed breadth study](GENERALIZATION_STUDY_RESULTS.md). It demonstrates
reuse on twelve previously evaluated conditions. The study's first-use geometry
result belongs to its original sealed evaluations; this viewer does not create
another unseen-body result. The fixture remains an anchored planar two-link
body with one motor, one IMU and one light sensor.

## What is retained

The actor is pooled seed **101**, the first seed in the declared study order,
selected without ranking target scores. Its six float32 actor tensors contain
17,475 parameters: `204 → 64 tanh → 64 tanh → 3 logits`. The critic and optimizer
are absent from the runtime. The exported weights and trusted provenance are
checked independently at load, against pinned tensor commitments.

- Source checkpoint SHA-256: `4c7ce76f644dd07d41340ef311d5f797578fa58f0470e2b89aa49c0d3b93082c`.
- Actor byte SHA-256: `73db0a5e562fe052c81894844fac08bdc35676da356b8acefd7e3602f4bb5e2b`.
- Export: [pooled-seed101-v1.json](../artifacts/pretrained/pooled-seed101-v1.json).
- Selection and export: [export receipt](../artifacts/reports/pretrained-workshop-v1-20261006/export-v1.json).

Rebuild creates fresh physics, delayed sensor queues, actuator state, sampler
state and local observation history. It inserts eleven absent frames and one
initial local frame. It keeps the same immutable policy object and weight
commitments. Pause preserves all of those runtime states, including the held
request and partial reward of an unfinished action.

The actor receives twelve frames of seventeen causal local channels: governed
effort, encoder sine/cosine and velocity, current, IMU force and gyro, delivered
light, previous request, completed action reward, validity bits and frame-present
marker. It receives no body choice, geometry, lamp position, absolute time or
passive joint state. Frame construction matches the training adapter's double
scalar calculation, float32 conversion and fixed clipping to `[-5,5]`.

## Native action and safety contract

The actor samples its three requests `[-.15,0,.15]` every `.1 s`. The existing
native effort governor refreshes every `.02 s`; each control advances ten `.002 s`
physics steps. Five native rewards form one completed action frame. A native
termination, safety veto or motor feedback fault ends the opportunity and holds
its prefix. There is no automatic restart or rescue.

Sampling remains categorical, with a declared portable SplitMix64 top-53-bit
uniform stream reset to seed 1 on rebuild. It preserves the actor probability
law within the checked numerical tolerance; it **does not reproduce Torch's
seeded sampled trajectory**. Study returns cannot be copied onto this new stream.
The runtime records input byte hashes, supplied uniforms, probabilities, selected
requests, emitted efforts and native transition commitments. The diagnostic
panel exposes the brain identity, and `/api/pretrained/trace` returns the current
run's trace, including an unfinished action when paused.

## Verification and evidence preservation

Verification is performed in the existing managed Debian container. Reference
checks compare lossless actor tensor bytes, exact local frame/history bytes,
raw logits and softmax probabilities against CPU Torch, using synthetic inputs
and retained known-case telemetry. Native lifecycle checks exercise atomic
rejected controls, stale session IDs, pause inside an action, rebuild history
clearing, immutable weights, and the exact run budget across every supported
configuration. An independent implementation check replays all new action
streams through the retained historical native engine, comparing each governed
effort, reward, transition hash and final physical state.

Completed checks:

| Check | Result |
| --- | --- |
| Authoritative native CTest suites | 21 passed, 0 failed |
| Independent actor reference | 3,977 checks passed; 647 actor inputs, 174 histories, 175 categorical checks |
| Maximum reference error | Logits `2.325e-6`; probabilities `7.749e-7`, below `1e-4` / `1e-5` gates |
| Strengthened native replay | 12/12 complete cases; 7,680 decisions and 38,400 native controls exact |
| Portable sampler | All 7,680 uniforms independently regenerated exactly from SplitMix64 seed 1 |
| Live HTTP service | 61 checks passed, all 12 choices, real-time pause/resume and exact legacy idle states |
| Browser | Native motion, Wake/Pause/Resume/Rebuild, keyboard selection, desktop and 390 px layout passed |

Receipts: [native tests](../artifacts/reports/pretrained-workshop-v1-20261006/native-tests-v1.log),
[actor reference](../artifacts/reports/pretrained-workshop-v1-20261006/reference-v1.json),
[strengthened native replay](../artifacts/reports/pretrained-workshop-v1-20261006/native-workshop-v2.json),
[live HTTP service](../artifacts/reports/pretrained-workshop-v1-20261006/http-service-v2.json),
[browser verification](../artifacts/reports/pretrained-workshop-v1-20261006/browser-qa-v1.json).
The earlier replay receipt is retained together with its exact auditor source;
v2 adds independent RNG, returned-state and total-reward checks. Both runs used
the same actor and identical action/physical streams, without parameter updates.

The [preservation baseline](../artifacts/reports/pretrained-workshop-v1-20261006/preservation-baseline.json)
binds 3,153 existing artifact files and retains the original native bridge with
SHA-256 `377b754985b2977c947e37e41cbbff286595fe70b08888c9fc32ece1db2d9777`.
Historical checkpoint, protocol, source snapshot and verification bytes remain
immutable. Current `CMakeLists.txt` and server source evolve for this feature;
the old study's strict **live-source** guard consequently refuses a current-tree
rerun. Do not weaken that guard or rewrite its receipts. A future archival
reverification tool must explicitly bind the original archived sources and
retained executable, rather than claim current-tree equivalence.

The completed [preservation check](../artifacts/reports/pretrained-workshop-v1-20261006/preservation-after-v1.json)
matches all 3,153 earlier artifact files. An activation restart interrupted
one background scan; its interruption record is retained, and this completed
check was run afterward.

The frozen actor is a compact tracked asset, so a fresh checkout can run this
page using the documented container launcher without Python. Full reference
reverification needs the optional existing Linux training environment and
ignored historical telemetry. New bulk traces and the preserved native binary
remain local and ignored under the workshop report's `traces/` and
`runtime_snapshot/` directories. Compact receipts and source snapshots are
retained evidence.

The [final provenance](../artifacts/reports/pretrained-workshop-v1-20261006/final-provenance-v1.json)
binds the runtime source snapshot, actor, binaries, receipts and unchanged managed
container identity. Earlier UI bytes are retained alongside the first HTTP
receipt; the second receipt verifies the corrected final page.

## Next milestone

Measure a small, explicitly charged online specialization budget, comparing
the frozen brain with head-only and full adaptation and matched scratch
controls. Include the adaptation period and regressions in the result. Lock a
fresh reserve before any new generalization claim. Variable topology, multiple
motors, contact/support behavior, persistent personal experience and hardware
remain later milestones.
