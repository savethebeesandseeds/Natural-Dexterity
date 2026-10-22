# Useful motion is available, and the unchanged learner finds it

2026-10-04. Outcome of the [locked native headroom development
study](LIGHT_HEADROOM_EXPERIMENT.md), retained in
`artifacts/reports/light-headroom-v1-development-20261004`.
The [recorded comparison](http://127.0.0.1:43117/headroom.html) displays quiet
and any declared controller together, with all cases and seeds in the table.

Both primary criteria pass on the newly declared **lower-lamp conditions**:
all 12 cases have a complete, safe handwritten witness exceeding +3.2 integrated
sensor reward over matched quiet, and all **36 unchanged native learner runs**
also exceed that margin. Fixed schedules alone supply a witness in every case.
This establishes attainable reward and acquisition from fresh memory within
64 simulated seconds on these known conditions. No learned parameters were
transferred between bodies or trials.

Historical upper-lamp conditions remain difficult: none of their completed
learner runs clears the same +3.2 margin. Eleven feedback-stopped trials remain
in the record. The new result does not replace the
[failed September motor-transfer result](ACTUATION_TRANSFER_RESULTS.md).

## What was measured

The existing collisionless, anchored two-link mechanism was evaluated with
three bodies, the motor at either the base or elbow, and four stationary lamps.
Twelve lower-lamp cases were primary; twelve historical upper-lamp cases were
diagnostic. All 24 are exposed development data.

Each case contains quiet, constant negative/positive effort, two square waves,
one handwritten sensor search-and-hold witness, and the unchanged native
Expected SARSA learner with fresh seeds 101, 202 and 303. Every requested action
uses the existing motor governor and feedback checks. Search and startup count
inside the same 64-second allowance. The light and IMU reward, physics, sensor
queues, learner settings and safety were unchanged.

The two independent primary rules require a gain **strictly above +3.2** in
every lower-lamp case. Learner competence additionally requires every seed to
pass. This is an engineering margin of +0.05 reward/s, not a significance test
or hardware acceptance standard. Choosing the best fixed schedule separately
for each case is an exposed evaluation selection, not one deployed controller.

## Every primary case

All entries are integrated reward gains over the same case's quiet run.
They are raw differences, not percentages; lower-lamp quiet light exposure is
zero. Learner columns use the three declared seeds. Rounded values below do
not replace the full-precision report used for decisions.

| Body / powered hinge / lower lamp | Best safe fixed witness | Fixed gain | Search-and-hold gain | Learner 101 | Learner 202 | Learner 303 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Long upper / base / left | Square 2 s | 8.122 | 0.244 | 7.825 | 9.750 | 10.439 |
| Long upper / base / right | Square 2 s | 8.909 | 0.266 | 8.081 | 13.516 | 8.378 |
| Long upper / elbow / left | Constant negative | 39.345 | 48.690 | 37.816 | 27.666 | 39.188 |
| Long upper / elbow / right | Constant positive | 39.904 | 50.125 | 35.604 | 38.208 | 26.458 |
| Long lower / base / left | Constant negative | 4.436 | 0.114 | 12.257 | 4.060 | 5.538 |
| Long lower / base / right | Constant positive | 4.529 | 0.055 | 5.277 | 3.656 | 3.538 |
| Long lower / elbow / left | Constant negative | 40.425 | 48.747 | 38.676 | 34.892 | 39.095 |
| Long lower / elbow / right | Constant positive | 41.369 | 47.003 | 35.060 | 34.459 | 25.926 |
| Weighted / base / left | Square 2 s | 8.870 | 0.360 | 10.287 | 10.845 | 17.600 |
| Weighted / base / right | Square 2 s | 6.824 | 0.194 | 7.560 | 9.636 | 7.236 |
| Weighted / elbow / left | Constant negative | 41.516 | 49.403 | 39.828 | 22.709 | 36.259 |
| Weighted / elbow / right | Constant positive | 41.811 | 49.556 | 38.155 | 39.233 | 26.493 |

The weakest learner gain is **+3.538115**, long lower / base / below right,
seed 303: only **0.338115 above the declared margin**. Mean learner gains across
the twelve primary cases are 23.0356, 20.7192 and 20.5123 for seeds 101, 202 and
303. These secondary averages do not relax the all-case rule. The pooled
36-run median is 20.1546; it is descriptive, not 36 independent body samples.

## Movement and holding are different outcomes

Lower-lamp learner gains are smaller with base actuation: median **8.2297**,
versus **35.9312** with elbow actuation. During the final sixteen seconds,
base learner runs average 5.30–64.64 lux at delivered control endpoints, while
elbow learner runs average 136.73–358.63 lux. Both layouts continue moving;
sampled late shaft-speed maxima range 2.99–5.11 and 3.46–7.33 rad/s respectively.

The sensor search-and-hold witness passes **6/12 cases**, all at the elbow.
There it demonstrates settled bright behavior: late-window light is
377.88–545.77 lux and sampled shaft speed is essentially zero. Its 81 sparse
physical snapshots from 48–64 seconds show nearly constant light and passive
joint speed ≤2×10⁻⁹ rad/s. At the base, the same controller settles dark.

An explicitly posthoc inspection of base learner seed 303 finds 45–69 dark
frames among those 81 late snapshots, interspersed with light peaks and passive
motion. The base gains therefore support intermittent dynamic exposure;
stable light tracking is not established. Snapshot counts are descriptive,
subject to sampling aliasing, and add no new acceptance criterion.

One plausible mechanical explanation is that elbow actuation can fold and
hold the distal receiving face toward the lower source, while a held base
encoder leaves the distal hinge passive under gravity. That explanation is an
inference from these behaviors, not a proof of reachability or optimal control.
The handwritten witness also has continuous feedback and a different action
vocabulary from the learner, so their comparison does not establish superiority
between matched learning architectures.

## Preserved failures and historical controls

There are **216 retained trials, 205 complete safe trials and 11 stops**:

- Eight weighted/base constant-effort trials, both signs under all four lamps,
  stop after 389 controls, or **7.78 seconds**.
- Weighted/base/above-left learner seeds 202 and 303 stop at **16.34** and
  **47.56 seconds**.
- Weighted/base/above-right learner seed 101 stops at **26.00 seconds**.

All eleven stopped endpoints report actuator `voltage_limited` and the matching
safety flag. There are zero native veto steps. This protocol also stops on
feedback flags; the stops do not imply a hardware damage event. Observed peak
shaft speeds are 8.305–8.412 rad/s, currents 0.358–0.375 A and temperatures
25.147–25.546 °C. These are sampled simulation telemetry, not hardware bounds.

Every stopped emitted-command tape reproduces its full recorded native prefix.
The three stopped learner request replays raise the same public feedback
exception and are recorded as unsuccessful controller replays. Their prefixes
are preserved rather than replaced. Prefix return minus a 64-second quiet
return has unequal exposure and cannot be treated as full-duration performance.

Of eighteen historical base learner runs, fifteen complete safely; their
safe-run median gain is +1.6257. All eighteen historical elbow learner runs
complete safely, with median gain −0.9252. None of these completed historical
learner runs exceeds +3.2. Lower-lamp success therefore cannot be generalized
to the previous lighting distribution or used to rehabilitate PPO transfer.

Actual collection contains **663,607 controls**, **6,636,070 physics steps**
and **13,272.14 simulated seconds**, excluding integrity replays. No rescue
reset, replacement seed or parameter search was added after seeing outcomes.

## Evidence and verification

The retained batch contains the locked protocol and 74 source/configuration
fingerprints, 216 raw transition tapes, 216 sparse snapshot files, native
replay receipts, full reward components, cumulative windows and a late window.
Whole transitions include requested/emitted efforts and public feedback.
Playback uses saved physical snapshots, with no new simulation or learning.
Its default case and constant-negative controller follow the declared order;
they were not selected for their outcome. A stopped recording holds its last
saved snapshot, which may precede the stopped endpoint by less than 0.2 seconds.

The post-collection native verifier passes for all **216 retained trials**;
full emitted-command transition replay matches every trial, and separate
controller replay matches all **205 complete safe trials**. The corrected
independent audit reports **zero failed checks**, agreeing with both gates and
accounting for the eleven stopped trials. All **19 registered native CTest
suites** and the focused audit issue-cap regression check pass.
Post-collection hashing confirms
all 1,000 preexisting artifacts, every tape and every snapshot. All 74 inputs
remained unchanged through collection and their archived copies still match;
the live auditor was subsequently corrected as recorded below. Frozen rover-v3
inputs remain unchanged and its held-out
evaluation was not executed. Paused live light and construction API responses
are byte-for-byte identical before and after collection.

The original independent auditor incorrectly expected observation schema v1
instead of the existing light-enabled v2. Its original failed receipt and
source remain archived. After collection, the verification tool was corrected
to require v2 and count failures independently of its 1,000-message display cap.
The correction changes no simulation, controller, protocol or trial. The
corrected source, source hashes, original receipt and focused issue-cap
regression check are retained under `verification/`. The failed original
audit's behavioral counts and gates are unusable.

Native full-transition/controller replay and independent JSONL accounting have
different roles: replay checks native reproducibility, while the independent
tool recomputes metrics, manual requests, endpoint consistency and both gates
without linking the simulator or learner. Neither provides unseen-condition
or hardware evidence. Their receipts are bound to the unchanged report digest
`60e03c8fe9d6c80bb6b91757b7054759e2a9d36152d372481564c3ed90755569`.

All builds, tests, collection and audit execution use the existing managed
Debian development container. Its immutable identity and preserved mounts,
ports and image are recorded in `verification/runtime.json`. No host compiler,
Python environment, Java installation or new container was introduced.

The native exporter validated and wrote all playback assets, but the existing
Docker Desktop shared Windows mount rejected its final Linux
`renameat2(RENAME_NOREPLACE)` publication. The complete staged index is archived
under `verification/unpublished-index.json`. The small host filesystem helper
`tools/publish-headroom.ps1` verified all 216 playback fingerprints and the
report/audit bindings, then published the identical bytes using Windows'
same-directory, no-overwrite `System.IO.File.Move`. This was filesystem
integration; no project build or runtime ran on Windows.

The separate export inspection passes every published hash, all frame counts,
first/last physical snapshot equality, complete/stopped exposure and complete
endpoint rewards: **216 playbacks and 66,568 saved frames**. Repeat export
refuses the existing destination. Browser checks cover all learner seeds,
selection from the full table, play/pause/restart, seeking, honest stopped-run
durations, notebook navigation and a contained 375-pixel phone layout. The
published index digest is
`90c3b080396e8417762799a93404cb77b604603c627256eb5fb351f7a691def1`.

To inspect retained evidence again, work in `/workspace` inside the managed
container. The isolated native build used for this study is:

```sh
cmake -S . -B /tmp/droid-headroom-native -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON
cmake --build /tmp/droid-headroom-native --target droid-native-validation -j2
/tmp/droid-headroom-native/droid-light-headroom --verify \
  artifacts/reports/light-headroom-v1-development-20261004

c++ -std=c++20 -O2 -Wall -Wextra -Wpedantic \
  -I/opt/nlohmann-json/include tools/audit-light-headroom.cpp \
  -o /tmp/droid-headroom-audit
/tmp/droid-headroom-audit \
  --input artifacts/reports/light-headroom-v1-development-20261004 \
  --output /tmp/light-headroom-audit-new.json
```

The audit output must be a new file. Collection and export commands also refuse
existing destinations; the retained study is not a request to collect it again.

## What this makes possible next

This study provides a demonstrated scratch-learning reference and useful
movement margin before another transfer comparison. A separately locked
study can hold this now-exposed lower-light curriculum fixed, establish source
competence, then compare frozen reuse, adaptation and scratch under equal
target interaction budgets. It must preserve the historical controls and
voltage-limit failures. Any unseen-condition claim needs a fresh challenge
split; these 24 cases can no longer serve as unseen evaluation.

Fast acquisition from fresh memory is now demonstrated on these declared
conditions. Reuse of learned experience after rebuilding, arbitrary topology,
contact and walking, continual learning and physical hardware remain open.
