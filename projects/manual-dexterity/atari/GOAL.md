# Atari engineering goal

Objective: validate native C++ learning across Pong, Breakout and Space
Invaders, then measure feature transfer and shared-encoder retention. Preserve
all runs and publish measured outcomes, including failed targets.

## Protocol fixed before the runs

| Game | Independent target | Evaluation |
| --- | --- | --- |
| Pong | At least 16 wins; mean return >0 | 20 full games, no truncations |
| Breakout | Mean raw return >=100 | 20 full games, no truncations |
| Space Invaders | Mean raw return >=1000 | 20 full games, no truncations |

Breakout and Space Invaders thresholds are engineering milestones, not published
human baselines or claims of literal game completion. ROM identities and full
criteria are fixed in `games.json`.

1. **Pong proof: passed.** Frozen checkpoint at 3,031,808 decisions: 19/20 wins,
   mean +10.75, no truncations, seeds 3000100–3000119. Evidence pointer and SHA-256
   are in `runs/proofs.json`. The later learner checkpoint is preserved separately.
2. **Registry and independent pilots.** Run Breakout (seed 21) and Space Invaders
   (seed 31) for 1,000,000 target decisions each, eight environments, unchanged PPO.
   Evaluate each on its registry's 20-game seed range. This first budget establishes
   a measured baseline; missing a score target is recorded as a miss.
   Breakout completed 1,000,000 decisions on 2026-09-17 in 477.44 seconds:
   evaluation mean 22.9, 20 finalized episodes, two frame-limit truncations.
   Its milestone failed. The earlier interrupted 522,240-decision checkpoint is
   preserved and excluded from the matched comparison. Space Invaders scratch
   completed as `space_invaders-20260917T091419Z-seed31-192`: mean 329 over 20
   natural games, no truncations, milestone missed. Both baselines are published
   in `runs/experiments.json` after artifact and episode validation.
3. **Matched transfer pilots.** Repeat both budgets/seeds/settings, initializing
   only conv1/2/3 and hidden features from the proven Pong checkpoint. Actor,
   critic and Adam start fresh. Record source training cost separately. Compare
   per-game returns and elapsed training time on matched evaluation seeds. One
   training seed is a pilot, not a robust claim of transfer advantage.
   Compare all finalized episodes under the identical 108,000-frame cap. If
   either policy truncates, label the comparison as capped-episode returns;
   it does not establish full-game superiority. The success criteria above
   retain their zero-truncation requirement. `compare-experiments.ps1` verifies
   artifacts, paired settings and initial heads before writing the report.
   Breakout transfer completed 1,000,000 target decisions: mean 21.8 across 20
   natural games, zero truncations, milestone missed. The validated paired
   difference from scratch is -1.1 under the fixed episode cap; scratch's two
   truncations prevent a full-game comparison. Training took 870.74 seconds
   versus scratch's 477.44, plus 3,031,808 source Pong decisions. This one-seed
   result is indexed in `runs/comparisons/index.json`; it establishes neither a
   general transfer advantage nor full-game superiority. Space Invaders transfer
   also completed: mean 391.5 versus scratch 329 across 20 natural game pairs,
   zero truncations, paired difference +62.5, milestone missed. Training took
   1,064.65 seconds versus scratch's 640.77; timing includes concurrent host load.
   All four independent runs and both immutable comparisons are published.
   This single-seed positive difference does not establish robust transfer advantage.
4. **Shared-learning retention.** First mix Pong and Breakout using a shared
   encoder and separate actor/critic heads, balanced environment counts and
   optimization contributions. Initialize the Pong encoder/head from the proven
   checkpoint. Measure each game's before/after returns and Pong retention.
   Extend to the third game only after the two-game integration is validated.
   Native shared-policy tests and a 4,096-decision GPU smoke run pass. The smoke
   has no skill evaluation and is excluded from retention claims.
   The subsequent three-game stage restored the two-game encoder and both
   learned heads, added fresh Space Invaders heads, reset Adam and allocated equal
   optimization weight and decisions across all three. The declared pilot is
   3,000,000 new decisions (1,000,000/game), seed 51, twelve environments and
   minibatches of 384. Paired evaluation ranges are 5000100–5000119 (Pong),
   5100100–5100119 (Breakout), and 5200100–5200119 (Space Invaders). Source compute
   is recorded separately. This stage tests Pong/Breakout retention while Space
   Invaders learns; changed batch/worker settings prevent treating it as an
   otherwise matched comparison to the earlier two-game stage.
   The three-game native build and CPU tests pass (4/4 suites, 8.26 seconds;
   `.runtime/native-tests-20260917-three-game.txt`). The full two-game result now
   passes independent verification: exactly 2M new decisions, 1M/game, in
   1,240.039908332 training seconds. Pong retained its criterion (20/20 wins
   before and after, mean 11.05 → 12.8, no truncations). Breakout mean 0 → 22.15
   is capped acquisition evidence from a fresh head: 20 baseline truncations,
   no after truncations, mean-100 milestone missed; criterion retention is not
   applicable. Source pretraining adds 3,031,808 decisions, total 5,031,808.
   The replacement three-game run completed 3M new decisions, 1M/game, in
   1,439.496447156 training seconds. Pong retained 20/20 wins and mean return
   improved 13.5 → 15.0. Breakout mean 22.2 → 36.1 and Space Invaders mean
   82.25 → 358.5 remain below their original targets. All six before/after
   evaluations completed 20 natural games without truncations. This is a
   full-game paired comparison for one training seed; independent verification
   passed, including the unchanged two-game source.
   The first full two-game attempt was intentionally stopped before training:
   a 66.83-second capped Breakout baseline episode projected about 1,337 seconds
   for 20 episodes, exceeding the old 900-second evaluation guard. This was not
   an actual timeout. Its incomplete baseline is preserved and excluded from
   final retention. Before replacement launch, declare two-hour training guards
   for both stages and 3,600 seconds per game's before/after evaluation. Decision
   budgets, seeds, environments, frame cap, scoring and sampling remain unchanged.

Keep one GPU learner active at a time. CPU evaluation may run beside it. Use
fresh directories for every run and frozen copies for evaluations. All runtime
code remains native C++; lifecycle stays in `atari.ps1`, dependency installation
stays in `setup.sh`, and the existing container/mount/port contract is preserved.

The engineering dashboard must distinguish training values, development tests,
and frozen-policy confirmations, show game-specific thresholds and checkpoint
identities, and never call a positive non-Pong score a win.

Current execution at 2026-09-17 12:27 UTC:
All four independent runs and both comparisons are complete and preserved.
Shared attempt `shared-pong-breakout-20260917T092034Z-seed41-465` has
`evaluation_incomplete`, zero new decisions and `baseline_evaluation_incomplete`;
its STOP was intentional. Pong's before measurement completed 20/20 wins, mean
11.05, no truncations; Breakout's partial baseline finalized two capped score-zero
episodes and is excluded from retention claims.

The old pilot and extension suites (`pilot-suite-20260917T092034Z-465` and
`extension-suite-20260917T094935Z-1657`) stopped at 10:08:01 UTC. At 10:09:45 UTC,
their processes and the learner were absent; only the server remained. No
three-game learner started. Keep both suites and the shared attempt unchanged.

The evaluation-budget update passed 4/4 native tests in 3.75 seconds; malformed
budget rejection and launcher/shell syntax checks passed. Replacement suite
`runs/shared-suite-20260917T101321Z-2644/` was dispatched through `shared-pilots`
at 10:13:21 UTC. Its supervisor and sole learner were verified live at 10:13:41
UTC (container PIDs 2644/2712; host PIDs 49285/49383), with the old queues absent.
The fresh two-game run `shared-pong-breakout-20260917T101321Z-seed41-2644` is
complete. Its independent verification passed without evidence changes:
`.runtime/shared-two-verification-20260917T110450Z-2affd38c.json`, SHA-256
`0c3c9a93e4b184685d005447fd5e76ce8e0cb6257d10438155aec5722d4a7689`.
Both initial policies are byte-identical to the stopped attempt; its
`recovery.json` records the hashes, rationale and replacement pointers.
The supervisor transitioned to `shared_three` at 11:03:17 UTC. Its learner
was container PID 4606 / host PID 73146. Run
`shared-pong-breakout-space-invaders-20260917T101321Z-seed51-2644` completed all
before evaluations: Pong mean 13.5 (20/20 wins), Breakout 22.2 and Space Invaders
82.25, each over 20 natural games with no caps. Native source loading
verified the completed two-game evidence/joint/exports, restored its encoder and
both learned heads, and initialized fresh Space Invaders heads.

That three-game attempt failed at 11:37:44 UTC on a denied rename of
`progress.json.tmp` to `progress.json`. The saved checkpoint records 2,356,224
new decisions (785,408/game); the last training-log row records 2,354,688.
No after evaluations completed, so the attempt is excluded from final retention.
Its learner and supervisor are absent, and the same container remains healthy.
Failure evidence and hashes are preserved in
`.runtime/shared-three-failure-20260917T113744Z.json`.

A Windows reader without delete sharing reproduced this rename failure, but
the reader responsible for the actual incident is unknown. Bounded native
replacement retries and host readers sharing read/write/delete access passed
all five native test suites in 4.89 seconds and actual bind-mount checks.
The fresh three-game-only recovery launched via `train-shared-extend` at
11:50:30 UTC as `shared-pong-breakout-space-invaders-20260917T115030Z-seed51-6445`,
from the same verified two-game `shared-final.pt`, with
3M new decisions (1M/game), seed 51, twelve environments, minibatch 384, fresh
Adam, two training hours and 3,600 seconds per game's evaluation. Evaluation
seeds, frame cap, sampling and result criteria remain unchanged. The native
`--evaluation-seconds` / host `-EvaluationSeconds` value is recorded as
`evaluation_max_seconds_per_game`.

At 11:50:48 UTC its sole native learner was verified (container PID 6445 / host
PID 95762). It is now complete; all three before evaluations reproduced the
failed attempt's episode records exactly: Pong mean 13.5 (20/20 wins), Breakout
22.2 and Space Invaders 82.25, each with 20 natural endings and no truncations.
All three initial checkpoints are byte-identical to the failed attempt. The old
learner and supervisor are absent; this direct launch has no queue or supervisor.
The failed attempt's `recovery.json` records hashes, unchanged protocol and replacement.
Preserve that run and suite; no partial checkpoint was resumed, no completed
experiment repeated, and no container restart/replacement performed. The run
completed exactly 3M new decisions, 1M/game, and 1,954 updates in
1,439.496447156 training seconds (1,849.0609379 total experiment seconds).
Source cost is 5,031,808 decisions, separate from the new budget; the successful
lineage totals 8,031,808. The excluded failed three-game attempt's 2,356,224 saved
decisions remain separately recorded spent compute. The API reports complete
with no active learner. Independent final verification passed without verifier
changes, including recursive verification of the source:
`.runtime/shared-three-verification-20260917T122310Z-97b2f1a9.json`, SHA-256
`c75cc5b62dfd5d920dabaa7b3ec9803c22682a064cefa83919d7c5ad6ccfda24`.
Final browser review passed at 12:27 UTC: complete budgets, final scores and
truncations, Pong retention, both missed score targets, preserved proof and
independent comparison tables. Only init and the native server remained; no
learner or supervisor was running. The documented experiment scope is complete.

## Requirement-to-evidence summary

| Declared requirement | Evidence and outcome |
| --- | --- |
| Native Atari engine, concurrency and GPU | C++20 ALE/LibTorch; 12-environment CUDA shared run completed; no Python runtime |
| Debian container and dependency-only setup | Same immutable container/contract; `setup.sh` dependencies only; 12:18 UTC runtime check healthy |
| Engineering viewer | Local port 43260, numeric budgets/results, saved-policy playback and native learner liveness; final browser review passed 12:27 UTC |
| Frozen winning Pong policy | `runs/proofs.json`: 19/20 held-out wins, mean 10.75, no truncations |
| Two independent scratch/transfer pairs | Four completed runs and two verified reports in `runs/experiments.json` and `runs/comparisons/index.json`; targets missed |
| Shared two-game experiment | Complete 2M budget; verified Pong retention, capped Breakout acquisition |
| Shared three-game extension | Complete 3M budget and independent audit; full-game paired results 15.0/36.1/358.5; Pong criterion retained, other targets missed |
| Preserved evidence and honest cost accounting | Frozen hashes, excluded failed/stopped attempts, separate source/new/failed-attempt decisions |

The declared experiments are complete. This does not establish that Breakout
or Space Invaders were beaten, or that transfer generally improves learning.
