# Pooled pretraining: first development pilot

**Completed:** 2026-10-06. **Batch:**
`artifacts/reports/generalization-pretraining-v1-pilot-20261006`.
The [prospective protocol](GENERALIZATION_PRETRAINING_EXPERIMENT.md) and exact
source/runtime provenance were locked before learning. This is a single-seed
development result; none of the twelve reserved new-body cases was constructed
or evaluated.

## Result and its meaning

One pooled PPO policy trained across 32 declared body/light conditions and then
controlled all twelve known development cases with fixed weights. Every trained
run cleared the declared useful-behavior margin over quiet, and all 36 trained,
untrained and quiet trials completed safely. The pretrained policy beat its
matched, saved initial policy in 10 of 12 cases, with mean integrated native
reward gain **+7.987401** over 64 simulated seconds.
The median case gain was only **+0.950337**. The untrained policy itself cleared
the quiet margin in 10 of 12 cases; pretraining narrowly brought the two
long-lower/base cases across it. Useful motion alone is therefore insufficient
evidence of learned knowledge.

This establishes this checkpoint's useful competence in the known training
family and a bounded offline-learning advantage over that initialization. It
does not establish competence on a child's unseen first assembly. Development
cases are in the source distribution, the fixture has one anchored two-link
graph, and only one training seed was used. Stable light holding, arbitrary
assemblies, multiple motors, contact and hardware remain untested here.

The benefit is uneven. Mean gain over the untrained policy was **-0.562843** on
left lamps and **+16.537645** on right lamps. Two left-lamp elbow cases worsened.
Most of the overall gain comes from the three elbow/right conditions; base
motor improvements over initialization are small positives in every case.
The weakest trained-minus-quiet result cleared its margin by only **+0.042061**.
The pooled average therefore cannot stand in for reliable behavior across bodies
and worlds.

## Collection and controls

- Source seed 404 collected exactly **16,384 decisions**, or **81,920 native
  controls**, with no safety stops. Eight PPO rollout updates executed 32
  optimization epochs and 1,024 Adam steps. Parameters remained finite.
- The independent shuffled schedule started every source condition once. Actual
  exposure was **128 to 640 decisions per condition**; balanced starts did not
  give equal physical exposure. Final partial episodes are retained in the
  per-case accounting.
- The 34,820-parameter policy's initial checkpoint was saved before learning.
  The trained and initial checkpoints were independently copied into each
  development branch with fresh observation history and matched action RNG
  seeds. Neither branch updated parameters during evaluation.
- Each of 36 development trials received exactly 640 decisions / 64 simulated
  seconds from wake-up, including sensor acquisition. No free warmup, rescue
  reset or additional post-training episode was used.
- Total collection was **39,424 decisions**, **197,120 native controls** and
  **1,971,200 physics steps**. Replay repeats that physical exposure for
  integrity checking; it is not another independent learning trial.

Source learning took 15.95 wall seconds on this Linux CPU runtime. That is an
offline computation measurement, not a child's waiting time or a device
responsiveness result. The separate development trials totaled 37.57 wall
seconds. Neither timing establishes a physical toy's latency.

## Every development comparison

Gains are raw integrated native light-plus-IMU reward over the same 64-second
opportunity. The competence diagnostic requires a safe, complete trained/quiet
pair and a gain **strictly greater than +3.2**. All pairs below qualify.

| Body | Motor | Lamp | Trained minus quiet | Trained minus untrained |
| --- | --- | --- | ---: | ---: |
| Long upper [4,3] | Base | Left | +7.636787 | +0.599467 |
| Long upper [4,3] | Base | Right | +9.954199 | +1.345419 |
| Long upper [4,3] | Elbow | Left | +15.052160 | +2.705278 |
| Long upper [4,3] | Elbow | Right | +43.694908 | +23.662761 |
| Long lower [3,4] | Base | Left | +3.449170 | +1.198591 |
| Long lower [3,4] | Base | Right | +3.242061 | +0.702083 |
| Long lower [3,4] | Elbow | Left | +8.831667 | -1.395572 |
| Long lower [3,4] | Elbow | Right | +41.255469 | +35.659225 |
| Weighted [4,4] | Base | Left | +11.987594 | +0.545454 |
| Weighted [4,4] | Base | Right | +6.913127 | +0.573823 |
| Weighted [4,4] | Elbow | Left | +9.696896 | -7.030274 |
| Weighted [4,4] | Elbow | Right | +49.110097 | +37.282562 |

Mean trained-minus-untrained gain was +0.827473 for base motors and +15.147330
for elbow motors. These are descriptive reductions of the complete cases,
not population estimates or a selected subset.

| From wake-up | Mean trained minus untrained | Cases with positive gain |
| --- | ---: | ---: |
| 6.4 simulated seconds | +0.692030 | 8/12 |
| 12.8 simulated seconds | +1.700093 | 9/12 |
| 32 simulated seconds | +4.050690 | 10/12 |
| 64 simulated seconds | +7.987401 | 10/12 |

The early windows show that useful offline knowledge can influence initial
behavior in these known cases. They do not define an acceptable child wake-up
budget or guarantee that every case improves immediately.

## Verification and retained evidence

All 19 native CTest suites passed at container startup, followed by seven
curriculum contract checks and five source-only native adapter integration
checks. PPO initialization/inference passed without updates before collection.

The independent standard-library verifier replayed **all 37 native request
tapes** through a fresh native process. It matched complete response-stream
digests, recomputed source exposure and development rewards/windows, verified
checkpoint/telemetry bytes and frozen parameter labels, and reproduced both
comparison reductions with **zero failed checks**. All 53 archived experimental
source files, runtime/native binary bindings and post-collection provenance
checks agreed. No reserved configuration calls occurred.

Native action replay verifies physics and accounting. It does not independently
rerun neural action sampling or PPO optimization, and parameter-label checks
are distinct from independently deserializing every tensor. The verification
receipt states this scope.

- [Locked protocol](../artifacts/reports/generalization-pretraining-v1-pilot-20261006/protocol.json)
- [Complete source and development summary](../artifacts/reports/generalization-pretraining-v1-pilot-20261006/pilot-404/summary.json)
- [Independent native replay receipt](../artifacts/reports/generalization-pretraining-v1-pilot-20261006/verification/native-replay-v1.json)
- [Preflight and installation evidence](../artifacts/reports/generalization-pretraining-v1-pilot-20261006/verification/preflight.json)

The exact source snapshot, initial/final checkpoints, protocol, compact result
receipts and verification are retained. The 37 compressed request tapes and 36
telemetry streams stay local and ignored in `pilot-404/traces/`. Their hashes are
recorded in source/trial receipts; a fresh checkout needs those exact local
assets for full replay. No earlier study or frozen rover-v3 artifact was changed.
The live workshop controller was not replaced with this pilot checkpoint.

## Next evidence to obtain

The pilot justifies a prospectively declared multi-seed pretraining comparison,
not promotion of seed 404 into a product release. Train independent pooled and
narrow-source controls under matched offline budgets, inspect known-case
competence and worst-case robustness by lamp side and motor position, then
freeze checkpoints before executing
new whole-assembly cases. Keep frozen first-use behavior primary and report
bounded specialization separately. The twelve length-5 configurations remain
manifest-only and available for that next study.

Before treating that as a child-facing capability, specify a supported assembly
domain and real responsiveness budget. Geometry transfer within this graph is
a prerequisite; variable module sets, changed connectivity, configurable
preferences, contact and hardware need their own mechanisms and evidence.
