# Native validation

## Current evidence — 2026-09-17 12:27 UTC

Pong passed its declared held-out criterion. The native game registry and
independent training/evaluation paths now cover Pong, Breakout and Space Invaders.
Both one-million-decision scratch baselines missed their targets: Breakout mean
22.9 with two truncations; Space Invaders mean 329 with 20 natural endings and
no truncations. Breakout transfer finished with mean 21.8 and zero truncations;
the validated capped-episode difference from scratch is -1.1. Space Invaders
transfer finished at mean 391.5, a +62.5 paired difference over scratch across
20 natural game pairs. All four independent runs and both comparisons are
published. The replacement two-game shared run completed exactly 2M new decisions
and passed independent result verification. Pong retained its criterion:
20/20 wins before and after, mean 11.05 → 12.8, no truncations. Breakout mean
0 → 22.15 is capped acquisition evidence, with 20 baseline truncations and none
afterward; its mean-100 milestone was missed. The three-game stage completed all
before evaluations but failed at 11:37:44 UTC on a denied atomic replacement of
`progress.json`. Its saved checkpoint records 2,356,224 new decisions; no after
evaluations completed, and the attempt is excluded from final retention evidence.
File-write fixes passed all five native test suites and actual bind-mount checks.
A fresh three-game-only recovery completed exactly 3M new decisions, 1M/game,
in 1,439.496447156 training seconds. Its full-game paired means were Pong
13.5 → 15.0, Breakout 22.2 → 36.1 and Space Invaders 82.25 → 358.5. All before
and after evaluations completed 20 natural games per game with zero truncations.
Pong retained 20/20 wins; the other two games missed their original milestones.
The API reports complete with no active learner. Final independent verification
passed, including the completed source; all earlier stopped/failed attempts and
queues remain preserved. Final browser review passed and the documented
experiment scope is complete, with Breakout and Space Invaders targets unmet.
The active protocol is recorded in [GOAL.md](GOAL.md).

### Pong held-out confirmation — 2026-09-16

A checkpoint frozen at **3,031,808 cumulative training decisions** completed
**20/20 full games**, winning **19**, with **mean raw return +10.75** and **zero
truncations**. Evaluation used fresh seeds **3000100–3000119** and took
189.325 seconds. This passes the declared requirement of at least 16 wins,
positive mean raw return and no truncations in 20 complete games.

- Evidence directory:
  `runs/pong-20260916T144112Z-seed9-820/verification-3m/`.
- `evaluation.json` contains all 20 scores, seeds, frame/decision counts and
  termination flags; its verdict is `target_met` and `passed` is `true`.
- Frozen checkpoint SHA-256, rechecked from the preserved file on 2026-09-17:
  `ea3106fbead715ef54fe144121934a0b10787962a735ac68b84fc7680f0867c8`.
- `checkpoint-check.json` records finite updated parameters and the checkpoint's
  decision count. The adjacent `config.json` preserves its training settings.
- `runs/proofs.json` points to this exact frozen checkpoint and evaluation.
- The learner subsequently stopped gracefully at 3,582,880 cumulative decisions.
  Its later `final.pt` is preserved separately and is not the confirmed policy.

The confirmation seeds were separate from the earlier development seeds. The
original automatic final range 1000100–1000119 was not used. This establishes
the criterion for one frozen policy under the declared protocol; it does not
establish performance across independent training seeds or Atari57.

### Registry and current native build

`games.json` pins each game's ROM SHA-256, minimal action count, default budget,
evaluation seeds and criterion. Training and evaluation check the ROM and game
identity; evaluation also checks the checkpoint's adjacent configuration.

| Game | Actions | Automatic evaluation seeds | Criterion over 20 full games |
| --- | ---: | --- | --- |
| Pong | 6 | 1000100–1000119 | At least 16 wins, mean raw return >0, no truncations |
| Breakout | 4 | 1100100–1100119 | Mean raw return >=100, no truncations |
| Space Invaders | 6 | 1200100–1200119 | Mean raw return >=1000, no truncations |

The two score thresholds are declared engineering milestones, not human
baselines or literal game-completion claims. Non-Pong positive scores are
reported separately from wins.

At **2026-09-17 08:57 UTC**, the live-container build passed **3/3 CTest suites
in 1.77 seconds**: registry parsing/score boundaries, real-ROM environment
behavior, and learning math/configuration/evaluation identity. At that point,
additional transfer tests were under audit and shared training was not validated.

At **2026-09-17 09:13 UTC**, the rebuilt live-container binary passed **4/4 CTest
suites in 2.83 seconds**: registry, environment, learning and shared tests.
Added coverage includes exact feature copying with
fresh heads/Adam and an unchanged RNG stream; malformed feature rejection;
transfer lineage through continuation; evaluation network/preprocessing
compatibility, including legacy Pong; balanced shared gradients and GAE;
updates to both game heads; exact logits/values from per-game exports; and joint
model/Adam archive round trips. These tests validate implementation invariants,
not a transfer advantage or full-budget retention outcome.

The subsequent three-game native build also succeeded, followed by **4/4 CTest
suites in 8.26 seconds**. Evidence is preserved in
`.runtime/native-tests-20260917-three-game.txt`. Extended shared tests cover
restoration of the two-game source and its evidence, the fresh third-game head,
balanced gradients and compatible policy exports. No three-game GPU experiment
or retention outcome is established by these CPU checks.

The runtime container remains `atari-dev`, immutable ID
`14238f24e9de26fccfe122ad53c1fd65e313738e76f66eac489418ed42c3f5c7`, using the
same image, project bind mount, all-GPU access and local port 43260 documented
below. It was stopped and started with the authoritative launcher to load the
new server binary, reusing that exact immutable ID. No replacement container or
volume was created for this extension.
The preserved validation image predates these changes; current native results
come from builds inside the managed container.

A **read-only runtime audit on 2026-09-17 at 10:42–10:43 UTC** passed the
authoritative `atari.ps1 status` contract check. It verified the same container
ID above, runtime image
`sha256:3defd02c090da3927832fb5d7d1a368e303ec4f4ca8f87c8b9170b6b18964855`, and
the pinned Debian 12 base layer. Command, project bind mount, local port 43260,
absence of named volumes, restart policy, GPU access, ownership/configuration
labels, init, shared memory and stop timeout all matched the documented contract.

Docker reported the container healthy; `/healthz` returned HTTP 200 and the C++
runtime marker. Neither `python` nor `python3` was installed. The actual ELF
executable linked ALE, LibTorch and CUDA with no missing libraries. `setup.sh`
remained dependency/environment-only, used APT `--no-install-recommends`, and
built ALE with `BUILD_PYTHON_LIB=OFF`. The native server and shared learner were
live during inspection. This audit made no mutation, restart or evaluation;
it verifies runtime integrity and did not evaluate shared-learning results.

The authoritative `atari.ps1 status` check passed again at **12:18 UTC** during
final evaluation: the same immutable container, image, mount, port,
GPU and restart contract; healthy with zero health-check failures; Debian 12,
neither Python interpreter installed, and RTX A2000 visible. No container
restart occurred. The five native suites and actual bind-mount checks remain
the latest code validation, with no subsequent native changes.

The dashboard's native `flock` liveness probe reported `learner_active: false`
after the shared smoke run and `true` during Space Invaders training. A progress
file left at `training`, `evaluating` or `initializing` is exposed as interrupted
when no process holds the experiment lock; the historical file is preserved.

### Independent pilots and completed comparisons

The earlier Breakout pilot `breakout-20260916T152520Z-seed21-73` was interrupted.
Its preserved checkpoint contains **522,240 decisions**; its last progress file
reports 523,264 decisions and still says `training`. A `docker top` inspection
on 2026-09-17 found no learner process, so that stale status is not evidence of a
running experiment. The partial run and its artifacts remain preserved; it has
no completed held-out result.

A fresh scratch baseline, **`breakout-20260917T085805Z-seed21-282`**, started on
2026-09-17 with seed 21, eight environments, a 1,000,000-decision budget and a
one-hour training cap. Its config records CUDA, `cold_start`, zero previous
decisions and an empty initial Adam state. At 08:58 UTC this run was in progress.
It subsequently completed **1,000,000 decisions**, **977 PPO updates** and **2,327
training games** in **477.444715545 training seconds** (2,094.48 decisions/second).

Its frozen final policy evaluated seeds **1100100–1100119** in **100.421428924
seconds**. All 20 requested episodes finalized: **18 ended naturally**, and
**two reached the 108,000-frame limit**. Mean raw return across all 20 was
**22.9**. The result is **`target_not_met` / `passed: false`**, both below the
declared mean-100 target and outside the zero-truncation requirement. It is not
a result from 20 naturally completed games.

- Evidence: `runs/breakout-20260917T085805Z-seed21-282/evaluation.json` and
  `progress.json`.
- Final checkpoint SHA-256:
  `1e738ca035b9ffd12c66ac1e28819a7a514c8381f2e8f2f441b9a9c8f2e5ef93`.
- Truncated seeds: 1100101 (return 23) and 1100113 (return 18).

**`space_invaders-20260917T091419Z-seed31-192`** was the active scratch run at
09:14 UTC. It completed **1,000,000 decisions**, **977 PPO updates** and **1,777
training games**, using seed 31 and eight environments, in **640.766713133
training seconds** (1,560.63 decisions/second). Its frozen final policy produced
**mean raw return 329** over **20 naturally completed games**, with **zero
truncations**, on seeds **1200100–1200119**. Evaluation took **37.294135401 seconds**.
The result is **`target_not_met` / `passed: false`**, below the mean-1000 milestone.

- Evidence: `runs/space_invaders-20260917T091419Z-seed31-192/evaluation.json` and
  `progress.json`.
- Final checkpoint SHA-256:
  `7301dcbc3d8b2757105cd9f666e3a8082196bf347fb31ff2f9dcbffb35450fa3`.
- Both completed scratch baselines are indexed in `runs/experiments.json` with
  checkpoint/config/progress/evaluation hashes.

Matched feature-transfer runs will use each scratch run's seed/settings and
evaluation seeds, loading only the proven Pong convolution/hidden features.
Target actor/critic heads and Adam state start fresh. Source pretraining cost
is recorded separately from target decisions. The CLI exposes `-Game` and
`-TransferFeatures`; the native transfer implementation and its invariant tests
are now validated. At 09:31 UTC the first transfer run was still training; the
completed result follows below.

The suite `runs/pilot-suite-20260917T092034Z-465/` recorded its first transfer
phase starting at **09:25:39 UTC**. That run was
**`breakout-transfer-20260917T092034Z-seed21-465`**, with seed 21 and a one-million
target-decision budget. It has now completed. At **09:41:09 UTC**, the suite
advanced to **`space_invaders-transfer-20260917T092034Z-seed31-465`**, with seed
31 and the same target budget. Shared Pong/Breakout training remains next.
The suite's manifest, status and runner log are
preserved; the declared three-game extension follows separately.

### First matched transfer result: Breakout

The transfer run completed **1,000,000 target decisions**, **977 updates** and
**2,101 training games** in **870.744969366 training seconds**. Final evaluation
used the same seeds 1100100–1100119 as scratch and took **51.845802995 seconds**.
All **20 games ended naturally**, with **mean raw return 21.8**, **zero
truncations** and **`passed: false`** against the mean-100 milestone.

| Measurement | Scratch | Pong feature transfer |
| --- | ---: | ---: |
| Target decisions | 1,000,000 | 1,000,000 |
| Mean return over 20 finalized episodes | 22.9 | 21.8 |
| Natural endings / truncations | 18 / 2 | 20 / 0 |
| Target training seconds | 477.444715545 | 870.744969366 |
| Additional source pretraining decisions | 0 | 3,031,808 |

The paired transfer-minus-scratch mean is **-1.1**, with 8 higher, 1 equal and
11 lower transfer scores. Both runs share the 108,000-frame cap; because scratch
truncated twice, this is a **capped-episode comparison**, not 20 full-game pairs.
The report sets `bounded_episode_comparison: true` and
`full_game_comparison: false`. Only 18 pairs had two natural endings; conditioning
on those pairs would select outcomes and is not the declared comparison.

The target elapsed-time difference is **+393.300253821 seconds** for transfer.
Elapsed time includes host scheduling and concurrent work, and the source Pong
training cost is additional. One training seed and these capped returns do not
establish a general transfer effect or full-game superiority.

- Immutable report: `runs/comparisons/breakout-20260917T094300Z-eb3e2974.json`.
- Report SHA-256:
  `6f90b8814f462a10e61e9457f6996820fc806f32b116ce3e09c7ef7e394becdd`.
- Transfer final checkpoint SHA-256:
  `ceeb17ea04c9edb5741e8f10601c1abee427306b1e6ebfd311f90269db3fcd75`.
- `runs/comparisons/index.json` points to this report. Validation checked matched
  settings/seeds, identical fresh heads, fresh Adam, eight exactly copied feature
  tensors, artifact hashes, and returns recomputed from episode records.

### Comparison evidence checks

The second immutable result is
`runs/comparisons/space_invaders-20260917T100045Z-9460e921.json`, SHA-256
`f17e6c0adb62255ca7ecd2b36cbde61ecb8f78f979494fa554d19960635e52a7`.
Transfer run `space_invaders-transfer-20260917T092034Z-seed31-465` completed
**1,000,000 target decisions**, **977 updates** and **1,672 training games** in
**1,064.647475039 training seconds**. Its final checkpoint SHA-256 is
`f0127b9cad10e0f521d0fd311c7f0f81a4a374d1fac71351bdff716de106ca31`.

Evaluation took **69.711941725 seconds** on seeds **1200100–1200119**, producing
**mean 391.5**, **20 natural endings** and **zero truncations**. It failed the
mean-1000 milestone. Scratch scored 329 with no truncations; the matched
**+62.5** difference is therefore a full-game comparison, with 14 higher and
6 lower transfer scores. Target training took **423.880761906 seconds longer**,
with source Pong pretraining additional. Timing remains observational and
affected by concurrent host workloads. This single-seed positive difference is
not a robust claim of transfer advantage.

`compare-experiments.ps1` validates scratch/transfer artifacts, hashes, matched
budgets/settings/seeds, fresh heads, copied features and per-episode summaries
before writing a report. Its integration tests passed valid natural-game and
capped-episode cases plus **17 rejection cases**. A capped comparison remains
explicitly distinct from a full-game comparison; reports preserve each run's
truncation count and cannot turn a truncated result into a milestone pass.
These were synthetic verifier tests, not measured transfer outcomes.

The following command validates completed run evidence before updating
`runs/experiments.json`:

```powershell
.\publish-results.ps1 -SuiteRun 'runs/pilot-suite-20260917T092034Z-465' -BreakoutScratchRun 'runs/breakout-20260917T085805Z-seed21-282'
```

It creates or verifies
scratch/transfer comparison reports only when both runs are complete and updates
`runs/comparisons/index.json`. In-progress or missing runs are skipped; the
publisher performs no training or container lifecycle operation. At the current
timestamp, all four independent runs are published and both immutable comparison
reports are indexed. The dashboard labels these preserved independent results
separately from the current shared-policy measurements.

Browser verification at **10:02:53 UTC** showed both real comparison rows and
active Pong/Breakout initial evaluation. A stale-proof fallback in the current
final-evaluation panel was corrected: it now shows **Pong final evaluation:
Pending**, seed **4,000,100**, until this run has final evidence. The preserved
Pong PASS remains in the separately captioned recorded-results section.
Dashboard JavaScript syntax validation passed before the browser reload.

### Shared GPU smoke test

`runs/shared-pong-breakout-smoke-20260917/` completed **4,096 total new decisions**,
exactly **2,048 per game**, and four PPO updates in **3.07771253 training seconds**.
The run used CUDA and the proven Pong source. Joint checkpoints and per-game
exports were saved; native CPU checkpoint checks found finite parameters and
recorded **3,033,856 Pong decisions** (source plus new Pong decisions) and
**2,048 Breakout decisions**. The joint model's total new budget is separate
from each export's per-game decision count.

The smoke run's status is **`complete_without_evaluation`**. It validates the
GPU update/save/export path and provides no measured retention result.
The full run **`shared-pong-breakout-20260917T092034Z-seed41-465`** was launched
at **10:00:14 UTC**. At 10:03 UTC its progress reported
`evaluating_before` and zero new training decisions. It uses the proven
checkpoint/evaluation, `-Steps 2000000` and `-Seed 41`, and evaluates both
games before and after training on paired seeds and preserves `retention.json`.
At **10:01:42 UTC**, the actual native process was verified live (container PID
2093 / host PID 42334) with the fixed 2M/seed-41/eight-environment/one-hour
arguments. The extension runner and waiter were also still live (host PIDs
36762 and 36781), waiting for the two-game suite to finish.
The third game follows successful two-game integration validation and the full
two-game retention measurement. As fixed in `GOAL.md`, the next stage restores
the two-game encoder and learned Pong/Breakout heads, adds fresh Space Invaders
heads, resets Adam, and trains **3,000,000 total new decisions**, one million per
game. It uses seed **51**, **12 environments**, minibatches of **384** and equal
optimization weights. Fresh paired evaluation ranges are 5000100–5000119 for
Pong, 5100100–5100119 for Breakout and 5200100–5200119 for Space Invaders.

This extension compiled and passed its CPU tests and completed its before
evaluations, but its first training attempt failed on file replacement.
The replacement completed the full budget and measured
retention of learned Pong and Breakout while Space Invaders learned;
the original two-game Breakout before/after delta measures acquisition from a
fresh head. Source training cost and changed worker/batch settings remain
explicit, so the three-game stage is not treated as an otherwise matched
comparison with the earlier two-game stage. Both declared shared experiments
are now complete and passed independent artifact verification.

### Three-game extension queued — 2026-09-17 09:49 UTC

`run-extension.sh` passed shell syntax checking and was dispatched inside the
same immutable `atari-dev` container at **09:49:35 UTC**. The preserved queue is
`runs/extension-suite-20260917T094935Z-1657/`; its manifest fixes the 3M total/1M
per-game budget, seed 51, twelve environments and one-hour training cap. Its
planned run is `shared-pong-breakout-space-invaders-20260917T094935Z-seed51-1657`.

At **09:49:50 UTC**, `docker top` verified the actual runner (container PID 1657,
host PID 36762) and lock waiter (container PID 1676, host PID 36781). Status was
`waiting_for_pilot_suite_lease`; Space Invaders transfer remained the only native
learner. This verifies a live queued process, not a started three-game experiment.

A second dispatch was rejected with “An extension queue is already active;
preserving it.” The sole extension-suite directory remained unchanged; no
duplicate experiment was created.

The runner is separate from dependency setup and container lifecycle. It waits
at most 10,800 seconds, then requires a completed pilot suite, completed two-game
progress/retention, all 2M source decisions, matching configuration and checkpoint
hashes, and no STOP request. It preserves the source and refuses duplicate
extension queues through a separate kernel lease. Inspect this queue before any
manual extension dispatch; do not dispatch another while it remains pending.

### Intentional shared stop and revised time guards — 2026-09-17 10:08 UTC

The first untrained Breakout baseline episode reached the unchanged 108,000-frame
cap with score zero in **66.827968758 seconds**. Twenty such episodes project
approximately **1,337 seconds**, longer than the original 900-second per-game
evaluation budget. The run was therefore stopped intentionally before any
training; it did **not** reach an actual 900-second timeout.

Preserved run `shared-pong-breakout-20260917T092034Z-seed41-465` reports
`evaluation_incomplete`, `baseline_evaluation_incomplete`, and **zero new
decisions**, with its STOP file retained. Its Pong before evaluation completed
**20/20 wins**, **mean 11.05**, **zero truncations**, in **221.061897551 seconds**.
Its final partial Breakout evaluation records **two capped score-zero episodes**
and status `incomplete`; that partial measurement is excluded from retention
claims. No training or after-policy measurement occurred.

The pilot suite and extension suite both recorded `stopped` at **10:08:01 UTC**.
At **10:09:45 UTC**, `docker top` confirmed their processes and the learner were
absent, with only the native server remaining. The scheduled three-game run had
never started. Both old queue directories, manifests and runner procedures remain
preserved as historical evidence.

Before launching replacements, the declared guards are **two training hours per
stage** and **3,600 seconds per game's evaluation**, independently for before
and after measurements. The native option is `--evaluation-seconds`, exposed as
`-EvaluationSeconds` by the launcher and recorded in
`evaluation_max_seconds_per_game`. The 2M/3M decision budgets, seeds 41/51,
environments 8/12, observation/sampling protocol, 108,000-frame cap, 20-episode
measurements and scoring criteria are unchanged. Concurrent CPU workloads also
motivate the larger training guard; these are not new learning hyperparameters.

At 10:11 UTC, `run-shared-pilots.sh` and launcher action `shared-pilots` were
pending validation and dispatch. All four independent runs and both transfer
comparisons remained valid and were not rerun.

### Replacement shared suite verified — 2026-09-17 10:13 UTC

The evaluation-budget update passed **4/4 CTest suites in 3.75 seconds**; the log
is `.runtime/native-tests-20260917-evaluation-budget.txt`. Native
`--evaluation-seconds 7201` was rejected before run creation. PowerShell AST and
`sh -n` syntax checks passed. The authoritative `shared-pilots` launcher reused
the same immutable container ID; it performs no container replacement.

Suite `runs/shared-suite-20260917T101321Z-2644/` was dispatched at **10:13:21 UTC**
and entered `shared_two` at **10:13:22 UTC**. At **10:13:41 UTC**, `docker top`
confirmed the supervisor (container PID 2644 / host PID 49285) and sole native
learner (container PID 2712 / host PID 49383); the old queues were absent.
The manifest fixes fresh run `shared-pong-breakout-20260917T101321Z-seed41-2644`
at 2M decisions, seed 41, eight environments, minibatch 256, a 7,200-second
training guard and 3,600 seconds per game's before/after evaluation. It entered
`training` around **10:38 UTC** and completed exactly **2M new decisions**, **1M
per game**, in **1,240.039908332 training seconds**. Source pretraining remains
separate at **3,031,808 decisions**, for **5,031,808 source-plus-new decisions**.

| Game | Before mean | After mean | Paired delta | Before / after truncations | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Pong | 11.05 | 12.8 | +1.75 | 0 / 0 | 20/20 wins both stages; criterion retained |
| Breakout | 0 | 22.15 | +22.15 | 20 / 0 | Capped acquisition from fresh head; mean-100 target missed |

Each stage finalized 20 episodes per game on paired seeds. Pong scores summed
to **221 before / 256 after**, with all games natural. Breakout scores summed
to **0 before / 443 after**: every baseline episode hit the unchanged 108,000-frame
cap, while all after games ended naturally. Criterion retention is not applicable
to Breakout because its initial policy did not meet the milestone; its positive
delta measures acquisition and is a capped comparison. The overall report is
`bounded_episode_comparison: true`, `full_game_comparison: false`. This remains
a one-training-seed descriptive result, not a generalization claim.

Before/after evaluations are preserved under `before/<game>/evaluation.json`
and `<game>/evaluation.json`. Before evaluation times were 148.690584279 seconds
for Pong and 1,338.360023098 for Breakout; after times were 196.885432546 and
62.701564115 seconds, respectively.

The independent verifier passed its **first positive integration** on this
completed result without modifying evidence. It rehashed artifacts and recomputed
all paired scores, episode sums/termination flags, criteria and source/new costs;
it does not deserialize or replay LibTorch policies. Preserved identities:

- Verification report: `.runtime/shared-two-verification-20260917T110450Z-2affd38c.json`.
- Verification SHA-256: `0c3c9a93e4b184685d005447fd5e76ce8e0cb6257d10438155aec5722d4a7689`.
- Joint checkpoint SHA-256: `f60d6638d0ea5f9a8a0d75d2ab895ad588c31b2c9bcf59152db5c417103864ff`.
- Retention SHA-256: `ba382a63ca60d23218f780a9d8d8e1d24c04c864c0cdb2bd15f91c9bbd4a300f`.

The supervisor transitioned to `shared_three` at **11:03:17 UTC**, launching
`shared-pong-breakout-space-invaders-20260917T101321Z-seed51-2644`. Its learner
was verified live (container PID **4606** / host PID **73146**); the prior learner
(host PID 49383) had exited. All three before evaluations completed before
training started. The run had a **3M total / 1M-per-game** target, seed **51**, **12 environments**,
minibatch **384**, and **zero initial Adam state entries**. Native source loading
validated the completed two-game evidence, joint checkpoint and exports,
restored its encoder and both learned heads, and initialized fresh Space Invaders
heads. Its before measurements are:

| Game | Mean raw return | Natural endings / truncations | Evaluation seconds |
| --- | ---: | ---: | ---: |
| Pong | 13.5 (20 wins) | 20 / 0 | 206.920820827 |
| Breakout | 22.2 | 20 / 0 | 40.15993422 |
| Space Invaders | 82.25 | 20 / 0 | 29.458264968 |

These are baseline measurements on the declared fresh paired seeds, not after
results. The attempt failed during training and completed no after evaluations;
it is excluded from final retention evidence.

At **11:37:44 UTC**, the learner failed with `filesystem error: cannot rename:
Permission denied` while replacing `progress.json.tmp` with `progress.json`.
The suite recorded `failed` in `shared_three`, child exit code 1. The saved
checkpoint manifest records **2,356,224 new decisions**, **785,408/game**;
the last training-log row records **2,354,688**. The learner (host PID 73146)
and supervisor (host PID 49285) were verified absent, with only the server
remaining and the original container healthy.

Snapshot `.runtime/shared-three-failure-20260917T113744Z.json` binds the config,
progress, checkpoint manifest, saved joint checkpoint and all three baseline
evaluations. Its SHA-256 is
`de318254b2c396ad977e2897998ad26e19f62efcbae4bdf1bd2fddfea0d636bb`.
The failed run, suite and artifacts are preserved unchanged.

A controlled Windows read handle without `FileShare.Delete` reproduced the
rename denial; releasing the handle allowed the same rename and preserved the
new contents. Evidence is
`.runtime/atomic-rename-sharing-20260917T1142-result.json`. This establishes a
possible failure mechanism; the specific reader at the actual failure is unknown.
The bounded native atomic-replacement retries and launcher's read/write/delete
sharing passed the authoritative native build and **5/5 CTest suites in 4.89
seconds**, logged in `.runtime/native-tests-20260917-atomic-io.txt`.
Actual Windows bind-mount checks also passed, preserved in
`.runtime/atomic-bind-verification-20260917T114936Z-066d36e2.json`:

- A reader released after 1.5 seconds allowed replacement on attempt 11, after
  1,679 ms.
- A persistent reader exhausted the 200 ms test deadline after 201 ms, reporting
  the error and preserving the exact destination and temporary contents.
- A reader sharing read/write/delete access allowed first-attempt replacement
  in 6 ms.

These changes affect file I/O only; model, sampling and evaluation protocol are
unchanged. They validate the fix, not a recovered three-game training result.

The recovery is a fresh **three-game-only** `train-shared-extend` run
from the same verified two-game `shared-final.pt`, with **3M new decisions**,
seed **51**, **12 environments**, minibatch **384**, fresh Adam, a **7,200-second**
training guard and **3,600 seconds per game's evaluation**. The before/after
seeds, observation/sampling protocol, frame cap and score criteria are unchanged.
It does not resume the partial three-game checkpoint or repeat the completed
two-game/independent measurements.

Replacement `shared-pong-breakout-space-invaders-20260917T115030Z-seed51-6445`
launched at **11:50:30 UTC**. At **11:50:48 UTC**, its sole learner was verified
(container PID **6445** / host PID **95762**). It is now complete, with the
unchanged 3M/seed-51/12-environment/minibatch-384 configuration. All before
evaluations are complete: Pong mean **13.5** and **20/20 wins**, Breakout **22.2**,
Space Invaders **82.25**. Each finalized **20 natural games**, with **zero
truncations**; all episode records exactly match the failed attempt's before
measurements. All after results have also completed, as recorded below.
The old supervisor and learner were absent; the same server remained. This
direct authoritative launcher invocation has no queue or supervisor and required
no container restart or replacement.

The failed attempt's `recovery.json` records the replacement, protocol and hashes;
its SHA-256 is `64166820209fd64c5b5cb00b0b8e7949074c9c09bd90e116880cde66c34b0527`.
The source joint checkpoint is unchanged at
`f60d6638d0ea5f9a8a0d75d2ab895ad588c31b2c9bcf59152db5c417103864ff`.
All three initial policy checkpoints are byte-identical to the failed attempt:

- Pong: `44d55bc325f0c31dbb337f4fbf93bcd4796f0e52e2c6a539968c9b7ab1e6ce21`.
- Breakout: `9864616f24b89cd5e4c8d80282077027ff7b857ea415cb4f2535cee62bef27e6`.
- Space Invaders: `0a1a8473e47dbf9ec0db00dfa13b4ae1b8f5a3ccdad298e61dc5a38a563f3ca9`.

The replacement native binary SHA-256 is
`d1daab111e6ad94040bc5ce1f3d7e86a060d7ec8c3cd26dc8dbf5207bcd94030`.
Live host text reads use read/write/delete sharing, or monitoring uses the local
API. The completed run contains **3,000,000 new decisions**, exactly **1,000,000
per game**, **1,954 updates** and **3,188 training episodes**. Training took
**1,439.496447156 seconds**; total experiment time was **1,849.0609379 seconds**.

| Game | Before mean | After mean | Paired difference | Natural before / after | Criterion interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Pong | 13.5 | 15.0 | +1.5 | 20 / 20 | 20/20 wins both; criterion retained |
| Breakout | 22.2 | 36.1 | +13.9 | 20 / 20 | Mean-100 target missed; criterion retention not applicable |
| Space Invaders | 82.25 | 358.5 | +276.25 | 20 / 20 | Fresh-head acquisition; mean-1000 target missed |

All six evaluations have zero truncations. `retention.json` reports
`full_game_comparison: true`; before/after seeds are paired within each game.
Breakout and Space Invaders did not meet their thresholds before training, so
their report's `criterion_retention_applicable: false` is not a lost criterion.
These measurements describe one training seed and do not establish general
transfer advantage or literal completion of either score-based game.

The final joint checkpoint SHA-256 is
`0d14691fa8783d4fbe4683a6cbfc3763105df8eae2bde6635662979048ebe5ed`.
The **5,031,808 source decisions** are additional to the 3M new budget, totaling
**8,031,808 decisions in this model's lineage**. The failed three-game attempt
spent **2,356,224 saved decisions**, excluded from this lineage and final paired
comparison but preserved as additional experiment cost. This lineage total is
not the sum of every pilot and failed experiment in the project.

The API reports `complete` and `learner_active: false`. Independent verification
passed without verifier changes. Report
`.runtime/shared-three-verification-20260917T122310Z-97b2f1a9.json` has SHA-256
`c75cc5b62dfd5d920dabaa7b3ec9803c22682a064cefa83919d7c5ad6ccfda24`;
the final `retention.json` SHA-256 is
`6eac9c390dd90f3cce2a2704836b8227617bbdafc8d3469fae162814f0d4ea2a`.
The audit rehashed all checkpoint, config, evaluation, manifest and retention
artifacts; recomputed all 120 episode records' totals, natural endings, paired
deltas and criteria; and recursively reverified the unchanged completed two-game
source. Before/after score sums were **270/300** for Pong, **444/722** for
Breakout and **1,645/7,170** for Space Invaders. Source/new decision accounting
and the excluded failed attempt were checked separately. The host verifier does
not deserialize or replay LibTorch models. These are paired development
measurements for one training seed, not held-out generalization evidence.

Final browser review passed after reload at **12:27 UTC**: `Complete`, **3M/3M**
decisions, **1M per game**, **1,954 updates**, all final means and zero truncations
matched the artifacts. Pong displayed **Retained**; Breakout and Space Invaders
displayed **Target missed**. The preserved Pong proof, independent runs and
comparison tables were correct. The final label change displayed **Training
elapsed / time cap: 00:23:59 / 02:00:00**. Process inspection showed only init
(host PID 7945) and the native server (host PID 7987); the learner and supervisor
had exited, and `/healthz` returned C++ `ok`. This completes review of the declared
experiments and engineering viewer without changing any game criterion or
claiming all games were beaten.

The stopped and replacement **two-game** runs' initial checkpoint bytes match
exactly, independently rehashed:

- Pong: `fe45142a2612460a3e24c390b3130dfdc3598dc404d62cb689d7af0f6440e158`.
- Breakout: `03b3c2ae4b87441927ab7af5df2a5faf8034cb163ba515caf9b374ff6ab54dfb`.

The stopped two-game attempt's `recovery.json` preserves these identities, the
explicit wall-time-only correction, exclusion from retention claims and replacement
pointers. These identities verify unchanged initial two-game policies; the
completed results are established by the separate evaluations and audit above.

Browser recovery verification at **10:15:55 UTC** displayed the exact fresh
two-game run, **02:00:00** training guard, **four environments per game**, and
**01:00:00 evaluation cap per game**. The current Pong final evaluation remained
pending, separate from the preserved Pong proof. The latest dashboard JavaScript
passed its syntax check; no native code changed after the 3.75-second test run.

## Historical record — 2026-09-16

The entries below preserve the sequence of earlier checks and development
results. Statements about pending proof or the then-active run describe their
recorded stage, not the current state above.

The Python scaffold was replaced with C++20. The GPU service and checkpoint
pipeline were verified at this initial stage. A winning Pong policy had not
yet been established.

### Initial dependency and native checks

- Native dependencies built on pinned Debian 12; downloaded archives passed
  SHA-256 verification, including LibTorch 2.7.1 CUDA 12.8 and Pong's actual ROM.
- Complete C++ application compiled and linked with GCC 12.2.0.
- Native environment tests passed: deterministic seeded replay/reseed,
  observation layout, RGB rendering, frame stacking, action validation,
  seven-frame time-limit truncation, and a naturally completed Pong game.
- Learning tests passed: terminal masking, time-limit bootstrapping, separate
  parallel GAE traces, and false-success rejection.
- Neither `python` nor `python3` is installed in the validation image.

Commands were executed in the documented `native-validation` image stage.
Both CTest suites passed (2/2; 1.61 seconds). This includes real ROM execution;
it does not include a learned winning policy.

### Measured emulator concurrency

Each measurement performs 50,000 total agent decisions using a random policy,
including native preprocessing and ordinary game resets. One decision repeats
up to four emulator frames. Timing excludes initial emulator construction.

| Native workers | Elapsed seconds | Decisions/second |
| --- | ---: | ---: |
| 1 | 13.098779 | 3,817.15 |
| 8 | 2.982609 | 16,763.85 |

Eight workers delivered approximately 4.39× the single-worker throughput in
this short Docker Desktop measurement. GPU inference and gradient updates are
excluded. These measurements cannot establish end-to-end training speed or
an afternoon completion time.

### Image and container state

- Runtime image: `atari:dev`, immutable ID
  `sha256:3defd02c090da3927832fb5d7d1a368e303ec4f4ca8f87c8b9170b6b18964855`.
- Validation image: `atari:native-validation`, immutable ID
  `sha256:7ea3271348e3b7de7bf59b743a4ff2ff67c4bb7ffdbea2da35ec35751d975f4a`.
- Original never-started container: `atari-dev`, immutable ID
  `1e707cb2ffe9e0859b76904576a1849fed91fe3dd57cda7af127b2d9627d253a`.
- Docker rejected its host port because
  `helicopter-dev` started using `127.0.0.1:43118` during the image build.
- Its sole mount is the project bind at `/workspace`; it has no named volumes.
  The requested all-GPU device configuration was present.
- Original inspection is preserved in `.runtime/failed-container-inspect.json`.

### Approved port correction

After explicit user approval, the original container's immutable ID, never-started
state, mount and configuration were checked again. Its inspection is preserved
in `.runtime/approved-replacement-before.json`. Only that exact container was
removed, without volume-removal flags. Other containers were left intact.

The replacement `atari-dev` is running and healthy, immutable ID
`14238f24e9de26fccfe122ad53c1fd65e313738e76f66eac489418ed42c3f5c7`.
It uses the same runtime image and project bind mount, no named volumes,
`127.0.0.1:43260:8080`, all-GPU access and `unless-stopped`. The launcher verified
the full contract and saved `.runtime/container-inspect.json`.

### GPU training and playback

- `atari check` detected one CUDA device and passed actual convolution forward
  and backward computation with finite gradients. LibTorch reports 2.7.1.
- Both CTest suites passed in the live container (2/2; 1.84 seconds).
- Neither `python` nor `python3` is installed in the live container.
- GPU: NVIDIA RTX A2000 8GB Laptop GPU; training used CUDA.
- Smoke run `pong-native-smoke-20260916`: 34,976 agent decisions, 35 PPO updates,
  35 completed training games, 19.43 seconds of training, approximately 1,800
  decisions/second. It stopped gracefully via `STOP` and saved final/latest
  checkpoints. Finite optimization loss was observed before stopping.
- `atari-checkpoint-check` reloaded the final checkpoint on CPU. Actor and critic
  biases changed from zero initialization to maximum absolute values
  0.0074669584 and 0.0044184872, respectively; both are finite. This proves saved
  parameter updates, not playing skill.
- A CPU reload completed a fresh-seed full game (seed 3000000): raw score -21,
  no truncation. This is expected early failure, not a success result. Evidence:
  `runs/pong-native-smoke-20260916-reload.json`.
- Smoke checkpoint SHA-256:
  `5d124deba058d1f5c76854ce036375b6368da1853392f8ffb651c793337d29e8`.
- Browser verification at `http://127.0.0.1:43260/` showed actual Pong frames,
  live training metrics and `latest.pt` playback after the first saved update.

The first full run began as `pong-20260916T143304Z-seed1-461`: seed 1, eight
workers, five million decisions or three hours, followed by a bounded 20-game
held-out evaluation. Its config, progress, episode log and checkpoints persist
under `runs/`. At 162,816 decisions the main run measured 1,100 decisions/second,
lower than the smoke run. At that rate the decision budget takes roughly
75 minutes. Throughput may vary; this is not a completion or victory claim.

Calling `atari.ps1 start` again reused the same running immutable container ID;
no replacement was performed.

### Checkpoint continuation

To avoid discarding learned state if more training is needed, native continuation
was added and compiled. Both suites passed again (2/2; 6.67 seconds while the
learner was active), including rejection of incompatible continuation configs.

The first run was gracefully stopped at 513,792 decisions. The same model and
Adam state resumed on CUDA in `pong-20260916T144112Z-seed9-820`. The remaining
budget is 4,486,208 decisions and 10,333 training seconds, preserving the original
five-million-decision and three-hour caps across the two sessions. Emulator and
random-generator state restart with seed 9; this is weight/optimizer continuation.

- Parent checkpoint SHA-256:
  `29501c621b76d147cb570711e94dc7dd0459f500034913b14cfa22a801a90bfd`.
- Parent config SHA-256:
  `3ff99fc6a9dc43d660ac3cfbb5d8fae5d3b3d7638ebe6accafe4bf624e61486a`.
- Loaded CUDA model and Adam tensors, update counters, and settings passed native
  validation. The recorded parent checkpoint hash matches the preserved file.
- Resumed optimization produced finite losses and new checkpoints. CPU inspection
  of the first resumed checkpoint reported 514,816 cumulative trained decisions
  and finite, nonzero actor/critic parameters.
- At 560,896 cumulative decisions, the resumed run had completed 46 PPO updates
  and 25 new training games, with a recent raw mean of -13.36. These early
  training scores are not held-out evaluation evidence.
- The browser was reloaded and verified to display cumulative progress and
  playback from the resumed saved policy. The 20-game final evaluation remains
  automatic when the remaining budget ends. No victory is claimed yet.

### Development evaluation at 1.16M decisions

While CUDA training continued, `latest.pt` was copied to a unique development
directory and evaluated on CPU. This frozen checkpoint contains 1,157,888
cumulative training decisions. Five complete games used seeds 200000–200004,
separate from training and the reserved final evaluation seeds.

| Seed | Raw return | Win | Truncated |
| ---: | ---: | :---: | :---: |
| 200000 | -8 | No | No |
| 200001 | -14 | No | No |
| 200002 | +6 | Yes | No |
| 200003 | +3 | Yes | No |
| 200004 | +1 | Yes | No |

Result: **3/5 wins, mean -2.4**, 42.85 evaluation seconds. These are real full-game
wins, but this small development sample does not meet the final 20-game criterion.
The GPU learner was not stopped or altered by this evaluation.

- Evidence: `runs/pong-20260916T144112Z-seed9-820/development-1m/evaluation.json`.
- Checkpoint SHA-256:
  `08bced555ce9e4f2bab2d304c32cb812bd453b6a86b498110a21ecf847763f9e`.
- CPU checkpoint inspection confirmed finite updated parameters and the recorded
  cumulative step count.

The viewer was simplified to engineering values: compact playback, decision
budgets, throughput, updates, losses, time estimates, numeric PPO/environment
configuration and evaluation results. Static assets reload without restarting
the container or interrupting its learner.

### Repeatable development evaluation

`atari.ps1 evaluate` now freezes a checkpoint/config into a unique development
directory, checks the saved parameter values and cumulative decision count,
records a manifest, and runs bounded CPU evaluation. Its seed range and 1–19
game limit keep this operation separate from the final 20-game evaluation.

The launcher was tested end to end while the original GPU learner stayed active.
Its 1,427,200-decision snapshot produced the following comparison on the same
five development seeds:

| Cumulative decisions | Returns, seeds 200000–200004 | Wins | Mean return |
| ---: | --- | ---: | ---: |
| 1,157,888 | -8, -14, +6, +3, +1 | 3/5 | -2.4 |
| 1,427,200 | -7, -15, +7, +10, +2 | 3/5 | -0.6 |

All games completed naturally. The second evaluation took 72.75 seconds.
Repeated development seeds support progress comparison, not final proof.
The separate final seed range 1000100–1000119 remains unused at this point.

- Second checkpoint SHA-256:
  `e108cdbfe0e338f7a2b2785e6991742179db94e2ecf5769112a6c3fc216e5214`.
- Evidence directory:
  `runs/pong-20260916T144112Z-seed9-820/development-20260916T145619Z-5ac1c1b0/`.
- `runs/current.json` still points to the original active continuation run.
- Launcher PowerShell syntax and dashboard JavaScript syntax checks passed.
- Browser review confirmed live frames, populated numeric metrics/configuration,
  collapsible checkpoint hashes, and a pending final-evaluation state.
