# Native Atari Lab

C++20 Atari 2600 emulation, parallel environments, CUDA PPO training, and live
browser playback. ALE (built on Stella) drives real Atari ROMs; LibTorch supplies
native neural networks, autograd, optimizers, and CUDA. No Python interpreter,
bindings, or scripts are required. The Python scaffold was removed at the user's
request. The research draft in `doc/` is preserved.

**Pong passed its declared criterion:** a frozen checkpoint at 3,031,808 decisions
won **19/20 held-out complete games**, with mean raw return **+10.75** and **zero
truncations**. The checkpoint, full episode results and SHA-256 are preserved in
`runs/pong-20260916T144112Z-seed9-820/verification-3m/` and `runs/proofs.json`.

**Status at 2026-09-17 12:27 UTC:** the declared independent, transfer and shared
training experiments are complete. The same Debian GPU container serves live
playback at **http://127.0.0.1:43260/**. The game registry supports Pong, Breakout
and Space Invaders, with separate ROM identities, action counts and evaluation
criteria. The atomic file-write update passed all five native test suites in
4.89 seconds, including shared-policy and replacement-retry tests. No learner
is active; the viewer continues playing the saved policy. Final browser review
confirmed the complete budget, game results and target-missed labels.

The final three-game experiment completed **3M new decisions**, 1M per game, in
**1,439.50 training seconds**. All before/after measurements used 20 natural games
per game with zero truncations. Pong retained **20/20 wins**, mean **13.5 → 15.0**;
Breakout improved **22.2 → 36.1**; Space Invaders learned **82.25 → 358.5** from
fresh heads. **Breakout's mean-100 and Space Invaders' mean-1000 targets remain
unmet.** This is one training seed, not a general transfer-advantage claim.
Both shared results passed independent artifact verification, and both independent
transfer comparisons are preserved. The failed three-game attempt remains
excluded, with its spent decisions recorded separately. See
[GOAL.md](GOAL.md) for the fixed experiment protocol and
[VALIDATION.md](VALIDATION.md) for results and validation scope.

## Container and setup

The exact native container contract is in [CONTAINER-PROPOSAL.md](CONTAINER-PROPOSAL.md).
`setup.sh` accepts no arguments and installs dependencies/environment only.
APT uses `--no-install-recommends`. Downloaded native dependencies are pinned
by URL and SHA-256 in `dependencies.lock`. Project build and execution belong
to `run.sh`; Docker lifecycle and project commands belong to `atari.ps1`.

```powershell
.\atari.ps1 start
.\atari.ps1 games
.\atari.ps1 check -Game breakout
.\atari.ps1 train -Game breakout -Steps 1000000 -Hours 1 -Envs 8 -Seed 21
.\atari.ps1 status
```

Run commands from this project directory. `-Game` accepts `pong`, `breakout` or
`space_invaders`; a new training run defaults to Pong when omitted. The launcher
permits one GPU learner at a time. The command above is the Breakout pilot budget;
the registry's ordinary training default remains five million decisions.

The viewer uses port 43260. The launcher reuses a matching container,
including a stopped one. Conflicts are preserved and rejected. It has no
prune, delete, replacement, or volume-reset operation. `build` refuses to
retag the image while a matching container exists.

The container uses restart policy `no`: starting Docker does not start Atari.
Run `.\atari.ps1 start` when needed and `.\atari.ps1 stop` when finished.

The browser watches a separate native ALE instance running the active run's
latest saved checkpoint on CPU, selecting that run's registered game. Training
is not paced by the viewer. Random/untrained play is labeled. Read-only endpoints:
`/healthz`, `/api/status`, `/frame.png`.

The dashboard shows numeric decision budgets, throughput, PPO updates/loss,
episode counts, elapsed time, estimated training time remaining, checkpoint
steps, hyperparameters, game-specific thresholds and preserved proof results.
The train-return value is the current episode-block mean (reset after at least
100 completed games), not a sliding last-100 average. An unavailable measurement
is shown as `—`. Learner activity comes from the native experiment's kernel lock;
a stale active progress file is presented as interrupted when no learner holds it.

Use `atari.ps1 stop-training` for a graceful learner stop, or `atari.ps1 stop`
to stop the container. A file lock prevents overlapping experiments. Closing
the browser does not stop training. Restarts preserve files but do not silently
resume interrupted experiments.

Continue a saved model and its Adam optimizer in a fresh run directory:

```powershell
.\atari.ps1 train -Resume 'runs/EXPERIMENT/final.pt' -Steps 5000000 -Hours 2 -Seed 9
```

`Steps` and `Hours` are additional budgets for that invocation. Checkpoints and
the viewer count cumulative decisions. The new config records the parent
checkpoint/config hashes; incompatible ROMs or model settings are rejected.
The source checkpoint must retain its adjacent `config.json`. Environments and
random generators restart with the requested seed; this preserves learned
weights and optimizer state, not exact emulator or random-generator state.

Feature transfer is a separate initialization mode:

```powershell
.\atari.ps1 train -Game breakout -Steps 1000000 -Hours 1 -Envs 8 -Seed 21 -TransferFeatures 'runs/pong-20260916T144112Z-seed9-820/verification-3m/checkpoint.pt'
```

This copies the three convolution layers and the hidden layer from the proven
Pong checkpoint. The target actor, critic and Adam optimizer start fresh; target
decision accounting starts at zero, with the source's training cost and tensor
fingerprints recorded separately. `-TransferFeatures` and `-Resume` are mutually
exclusive. Native tests verify exact feature copying, fresh target heads/Adam,
unchanged random sampling state and preserved transfer lineage. The first matched
Breakout pilot is complete: transfer scored 21.8 against scratch's 22.9 at one
million target decisions each. Space Invaders transfer also completed at that
budget, scoring 391.5 against scratch's 329. Both games missed their declared
milestones; these single-seed pilots do not establish a general transfer advantage.

Shared training uses one encoder with separate Pong and Breakout heads:

```powershell
.\atari.ps1 train-shared -Checkpoint 'runs/pong-20260916T144112Z-seed9-820/verification-3m/checkpoint.pt' -SourceEvaluation 'runs/pong-20260916T144112Z-seed9-820/verification-3m/evaluation.json' -Steps 2000000 -Seed 41 -Hours 2 -EvaluationSeconds 3600
```

`Steps` is the total new budget across both games: two million allocates one
million decisions to each. Four of the default eight environments serve each
game, with balanced optimization contributions. Pong's encoder and heads come
from the proven policy; Breakout's heads and Adam start fresh. The run evaluates
both games before and after training on paired fresh seeds and writes
`retention.json`. Shared checkpoints preserve the joint model and optimizer;
per-game exports are for inference. The completed smoke test validates GPU
updates and exports, but provides no full-budget retention result.

The completed 2M-decision experiment retained Pong's criterion: 20/20 wins
before and after, mean 11.05 → 12.8, no truncations. Breakout mean 0 → 22.15
measures acquisition from a fresh head; all 20 baseline episodes were capped,
and all 20 after games ended naturally. Its mean-100 target was missed. This
two-game result passed independent artifact verification.

After the verified two-game result, the completed third stage added Space Invaders
to the shared model. Its source loader validated the completed source evidence,
joint checkpoint and exports, restored the learned encoder and Pong/Breakout
heads, and initialized fresh Space Invaders heads and Adam. It allocated one
million new decisions to each game. The three-game run used
seed 51, twelve environments and minibatches of 384, with equal game weights.
The first attempt completed its before evaluations but failed during training
on an atomic file replacement. Its artifacts remain preserved. The replacement
completed paired before/after evaluations, retaining Pong's criterion and
increasing measured Breakout and Space Invaders returns. The changed worker/batch
settings are recorded explicitly.

The shared attempt `shared-pong-breakout-20260917T092034Z-seed41-465` stopped at
zero new decisions. Its first capped Breakout baseline episode took 66.83 seconds;
20 such episodes would take about 1,337 seconds, exceeding the former 900-second
evaluation budget. The final partial Breakout baseline is excluded from retention
claims. This was an intentional stop, not an actual 900-second timeout.

The pilot queue `pilot-suite-20260917T092034Z-465` and extension queue
`extension-suite-20260917T094935Z-1657` stopped at 10:08:01 UTC; their processes
were absent at 10:09:45 UTC. Their artifacts and original runner procedures remain
preserved. The three-game learner never started.

The validated replacement `run-shared-pilots.sh` sequence creates fresh two-game
and three-game runs with the same 2M/3M decision budgets, seeds
41/51, environment counts 8/12, and unchanged scoring, 20-game evaluations and
108,000-frame episode cap. Declared before launch: **two-hour training guards**
and **3,600 seconds per game's evaluation**, for both before and after stages.
The native option is `--evaluation-seconds`; the host option is
`-EvaluationSeconds`, recorded as `evaluation_max_seconds_per_game` in configs.
These are wall-clock guards, not changes to the observation or sampling protocol.

The preserved replacement suite was dispatched with the following command;
it has now failed in its three-game stage and must not be dispatched again:

```powershell
.\atari.ps1 shared-pilots -Checkpoint 'runs/pong-20260916T144112Z-seed9-820/verification-3m/checkpoint.pt' -SourceEvaluation 'runs/pong-20260916T144112Z-seed9-820/verification-3m/evaluation.json'
```

This runner is separate from dependency setup and container lifecycle. It
preserves the four completed independent runs and their comparisons; they do
not need retraining. Preserved suite: `runs/shared-suite-20260917T101321Z-2644/`,
dispatched at 10:13:21 UTC. Its supervisor and sole native learner were verified
live at 10:13:41 UTC. Two-game run
`shared-pong-breakout-20260917T101321Z-seed41-2644` completed in 1,240.04 training
seconds; its 2M new decisions are separate from 3,031,808 source decisions
(5,031,808 combined). Its initial checkpoints were byte-identical to the stopped
attempt. The supervisor advanced to three-game run
`shared-pong-breakout-space-invaders-20260917T101321Z-seed51-2644` at 11:03:17 UTC;
that learner and supervisor exited after the file-replacement failure at
11:37:44 UTC. Its saved 2,356,224 new decisions (785,408/game) are additional
spent compute, excluded from the successful model's lineage and final retention
comparison. The original container remains healthy.

A Windows reader without delete sharing reproduced the replacement failure;
releasing the reader allowed the rename. The specific reader at the actual
failure is unknown. Bounded native replacement retries and the launcher's
read/write/delete sharing passed native tests and actual bind-mount checks.
The recovery command runs only the three-game stage
from the verified two-game joint checkpoint:

```powershell
.\atari.ps1 train-shared-extend -Checkpoint 'runs/shared-pong-breakout-20260917T101321Z-seed41-2644/shared-final.pt' -Steps 3000000 -Envs 12 -Seed 51 -Hours 2 -EvaluationSeconds 3600
```

Fresh run `shared-pong-breakout-space-invaders-20260917T115030Z-seed51-6445`
launched at 11:50:30 UTC; its sole native learner was verified at 11:50:48 UTC
(container PID 6445 / host PID 95762). It completed 3M new decisions after its before
evaluations exactly reproduced Pong mean 13.5 (20/20 wins), Breakout 22.2 and
Space Invaders 82.25, all from 20 natural games with no truncations. It retains
the original decision budget, seed, minibatch 384, evaluation seeds, frame cap
and criteria. All three initial checkpoints are byte-identical to the failed
attempt. The recovery started from the verified two-game source, preserving
completed experiments and the failed partial run. This direct launcher invocation
has no queue or supervisor; the container was neither restarted nor replaced.
The failed attempt's `recovery.json` records identities and replacement pointers.

Evaluate progress without stopping training:

```powershell
.\atari.ps1 evaluate
.\atari.ps1 evaluate -Checkpoint 'runs/EXPERIMENT/latest.pt' -Episodes 5 -EvalSeed 200000 -Seconds 180
```

This freezes the chosen checkpoint and config in a unique `development-*`
directory, records its cumulative steps, and runs CPU evaluation. The default
is five full games with seeds 200000–200004 and a 180-second cap. Development
evaluation accepts 1–19 games and seeds below 1000000, keeping it separate from
the reserved final 20-game evaluation. Results, hashes and manifests are
preserved; the active run and viewer selection remain unchanged.

## Parallel training

Each worker owns one ALE instance. Eight workers produce observations for one
batched CUDA policy. A 128-step rollout feeds PPO updates with minibatches of 256,
4 epochs, gamma 0.99, GAE lambda 0.95, clipping 0.1, entropy 0.01, and learning
rate 0.00025. The CNN uses three convolution layers and a 512-unit hidden layer.
Weights start random unless continuation or feature transfer is selected. The
default budget is 5 million agent decisions or 3 training hours, followed by
bounded held-out evaluation; success is not promised.

The native environment contract is shared by training, evaluation, and playback:

- Emulator frame skip 1; repeat each action for up to 4 raw frames.
- Sticky-action probability 0.25, applied once inside ALE.
- Max-pool the final two grayscale screens, bilinearly resize to 84x84, stack 4.
- Pixels only; no RAM or object-state policy inputs.
- Seeded 0-30 no-op reset prefix, then FIRE when supported.
- Complete-game episodes, with no artificial life-loss termination.
- Train with clipped rewards but record original full-game scores.
- Limit 108000 emulator frames, distinguishing time limits from natural endings.
- Bootstrap a truncated final observation's value without crossing reset boundaries.

This is a declared native protocol, not claimed byte-identical to Gymnasium.
`games.json` pins each supported game's ROM hash, minimal action count and score
criterion. Training and evaluation reject a mismatched ROM or checkpoint game.
Evaluation also requires the checkpoint's recorded network and complete
preprocessing protocol to match the evaluator. Shared-encoder balancing,
per-game heads and exports have native test coverage. Both completed shared
experiments measured Pong criterion retention; Breakout and Space Invaders
remained below their milestones.

## Evidence

Every criterion requires 20 full games with no time-limit truncations:

| Game | Declared criterion | Established result |
| --- | --- | --- |
| Pong | At least 16 wins and mean raw return >0 | Passed: 19/20 wins, mean +10.75 |
| Breakout | Mean raw return >=100 | Unmet: final shared mean 36.1, no truncations |
| Space Invaders | Mean raw return >=1000 | Unmet: final shared mean 358.5; transfer pilot 391.5, no truncations |

The completed three-game shared experiment has 20 natural before/after game
pairs for each game:

| Game | Before mean | After mean | Paired difference | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Pong | 13.5 | 15.0 | +1.5 | 20/20 wins both stages; criterion retained |
| Breakout | 22.2 | 36.1 | +13.9 | Measured return increased; mean-100 target missed |
| Space Invaders | 82.25 | 358.5 | +276.25 | Fresh-head acquisition; mean-1000 target missed |

Its `retention.json` reports `full_game_comparison: true`. Criterion retention
does not apply to Breakout or Space Invaders because neither baseline met its
threshold. The 3M new decisions add to 5,031,808 source decisions, producing
8,031,808 decisions in the successful model's lineage. The excluded failed
three-game attempt spent another 2,356,224 saved decisions; these are separate
from the lineage total. Results are descriptive for one training seed.

The completed Breakout transfer pilot also missed the mean-100 target: mean
21.8 over 20 natural games, no truncations. Its paired comparison with scratch
is explicitly a **capped-episode comparison**, because scratch truncated twice.
The observed score difference is -1.1; target training took 870.74 seconds with
transfer versus 477.44 seconds from scratch. Source Pong pretraining adds
3,031,808 decisions. Host scheduling and concurrent work affect wall-clock time;
this single-seed pilot does not establish a general transfer effect or full-game
superiority. The immutable report is indexed in `runs/comparisons/index.json`.

Space Invaders transfer scored 391.5 against scratch's 329 on the same 20 seeds,
with no truncations in either run: a +62.5 full-game paired difference. Target
training took 1,064.65 versus 640.77 seconds, with source pretraining additional.
This also missed the mean-1000 milestone. The measured positive difference is
from one training seed, not a robust transfer-advantage claim. The dashboard
separates these preserved independent pilots and the Pong proof from the current
shared run's retention measurements.

Breakout and Space Invaders thresholds are engineering milestones, not human
baselines or claims of literal game completion. Positive scores in these games
are not counted as wins. A short evaluation or a training score cannot pass.
Evaluation records ROM and checkpoint SHA-256 values. The Pong confirmation used
fresh seeds 3000100–3000119; its result applies to that frozen policy and protocol.
Broader performance across training seeds and games needs separate evidence.

To compare a completed matched pair, use PowerShell 7:

```powershell
.\compare-experiments.ps1 -ScratchRun 'runs/breakout-20260917T085805Z-seed21-282' -TransferRun 'runs/TRANSFER_RUN' -Output 'runs/comparisons/breakout.json'
```

The comparison checks hashes, target budgets, seeds, settings, fresh heads and
episode summaries. It distinguishes complete natural games from episodes capped
at 108,000 frames, reports truncation counts, and never relaxes a milestone's
zero-truncation requirement. One matched training seed is a pilot comparison,
not a robust claim of transfer advantage.

Publish verified completed results to the dashboard indexes with:

```powershell
.\publish-results.ps1 -SuiteRun 'runs/pilot-suite-20260917T092034Z-465' -BreakoutScratchRun 'runs/breakout-20260917T085805Z-seed21-282'
```

The publisher validates completed run artifacts and updates `runs/experiments.json`.
When a scratch/transfer pair is ready, it creates or verifies its comparison
report and updates `runs/comparisons/index.json`. In-progress runs are skipped;
publication does not start or change training. All four independent runs and
both validated immutable comparison reports are published.

Each invocation uses a fresh run directory; nonempty directories are refused.
Runs live on the host bind mount and are excluded from Git:

| Artifact | Purpose |
| --- | --- |
| `runs/current.json` | Run followed by the viewer |
| `runs/proofs.json` | Preserved frozen-policy confirmation pointers and hashes |
| `runs/experiments.json` | Validated completed experiment results for the dashboard |
| `runs/<run>.log` | Persistent learner output |
| `config.json` | Algorithm, seed, budget, and environment settings |
| `progress.json` | Stage, throughput, and recent raw scores |
| `episodes.jsonl` | Full-game training outcomes |
| `latest.pt` / `final.pt` | Native LibTorch checkpoints |
| `evaluation.json` | Frozen-policy held-out results and verdict |
| `shared-final.pt` / `shared-latest.pt` | Joint shared model and Adam checkpoints |
| `retention.json` | Shared before/after per-game measurements and retention |
| `runs/comparisons/*.json` | Validated scratch/transfer comparison reports |

Place `STOP` in an active run directory for a graceful checkpoint/stop.
Explicitly stopped runs skip final evaluation. Failed runs remain preserved.

After a shared experiment completes, independently verify its recorded results:

```powershell
.\verify-shared-results.ps1 -Run 'runs/COMPLETED_SHARED_RUN'
```

This read-only check rehashes artifacts and recomputes paired scores, truncations,
criteria, decision counts and source costs. For the three-game stage it also
checks the completed two-game source. It rejects stopped or incomplete runs and
prints JSON; it does not deserialize or replay LibTorch policies. Syntax and
rejection checks pass. The first positive integration verification passed on
the completed two-game result without modifying its evidence. The verified
report is `.runtime/shared-two-verification-20260917T110450Z-2affd38c.json`.
The three-game result also passed, including recursive verification of its
unchanged two-game source:
`.runtime/shared-three-verification-20260917T122310Z-97b2f1a9.json`, SHA-256
`c75cc5b62dfd5d920dabaa7b3ec9803c22682a064cefa83919d7c5ad6ccfda24`.
All 120 three-game before/after episode records ended naturally. These paired
development measurements establish the recorded result, not held-out
generalization across training seeds.

## Native commands

Inside the managed container:

```sh
build/native/atari games
build/native/atari check --game breakout
ctest --test-dir build/native --output-on-failure
build/native/atari-checkpoint-check runs/EXPERIMENT/final.pt
build/native/atari benchmark --game pong --envs 8 --steps 50000
build/native/atari train --game space_invaders --envs 8 --steps 1000000 --hours 1 --seed 31
build/native/atari evaluate --game space_invaders --checkpoint runs/EXPERIMENT/final.pt --output runs/EXPERIMENT/recheck.json --seed 3200100 --episodes 20
```

The environment benchmark excludes GPU optimization; learner throughput is a
separate measurement. CUDA forward/backward checks, actual-ROM tests, GAE and
success-rule tests, and saved-policy evaluations provide distinct evidence.
The native evaluator requires a matching adjacent `config.json`; use a fresh
output location and unused seeds for a new confirmation.

Native compilation and CPU tests can also run in the image build:

```powershell
docker build --target native-validation --tag atari:native-validation .
```

This does not start, replace, or change the managed service container. The
preserved validation image was built before the game registry and transfer work;
the latest four-suite test result comes from the live container build. CUDA
checks still require the GPU-enabled service container.

BuildKit reads pinned archives from ignored `.runtime/dependencies` without
retaining compressed archives in permanent image layers. Missing archives are
fetched from pinned URLs. The initial official CUDA LibTorch download is about
3.8 GB. The ALE ROM bundle uses the exact upstream packaging source and checksum.

## Upstream projects

- [ALE C++ interface](https://ale.farama.org/cpp-interface/): load ROMs, send
  joystick actions, read pixels/rewards, reset, and detect episode endings.
- [Stella](https://stella-emu.github.io/): open-source Atari 2600 emulator.
- [LibTorch](https://docs.pytorch.org/cppdocs/frontend): native C++ training/CUDA.
- [ALE parallel environments](https://ale.farama.org/vector-environment/):
  native parallel Atari support. This project uses its own small ALE worker pool.

The local viewer is `noindex`. Its CSP allows style elements for Codex's page
selector while keeping script execution restricted to external same-origin code.
