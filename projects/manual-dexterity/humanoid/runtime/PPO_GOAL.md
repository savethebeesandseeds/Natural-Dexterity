# PPO walking proof — bounded scope

**Final bounded outcome, 2026-09-17: training stopped; walking not convincingly
proved.** Both allowed optimizer invocations have finished, using 827.046 seconds
combined. The candidate survives 20 seconds and makes alternating foot lifts,
but the actual replay shows a wide stance, swivelling and substantial slipping.
No more training, diagnostics or expanding milestones are planned. Further
experiments require a new explicit user instruction. See
[the outcome and saved replay](PPO_BOUNDED_RESULT.md). Historical statements
below about an active investigation or future experiments are superseded.

## Current scope, revised by the user on 2026-09-17

The user explicitly asked to avoid an expanding, expensive or infinite goal:
we need a proof that the humanoid can walk, without requiring a polished gait or
a full reliability study. This section replaces the earlier acceptance plan
below for the current pursuit. The previous experiments remain historical evidence.

The target is **one from-scratch native C++ PPO checkpoint visibly walking for
20 continuous seconds on flat ground, with alternating steps and no fall**.
Use one command (0.5 m/s) and one fresh reset. Require forward movement across
the floor and visually confirm real foot lifting and alternation; sliding,
standing, a short initial stumble or externally supported motion do not qualify.
Retain the original random-weight provenance, no expert/imitation assistance,
and the existing physical fall thresholds. Save the checkpoint, a concise
evaluation report and the browser demonstration. This is a feasibility proof,
with no claim of reliability across commands, training seeds or hardware.

The remaining budget is **at most two training attempts, each capped at 600
trainer seconds, at most 1,200 trainer seconds combined**. Count every new
optimizer-running invocation against these limits, including short training
checks. Record elapsed time and outcomes in `artifacts/ppo-milestone/bounded-proof-budget.json`.
Use existing tests and evaluation machinery. The proposed detailed tracing,
model-sharing optimization, multi-seed study, 45-trial gate, command-tracking
thresholds and subsequent research milestones are deferred.

Stop immediately on a verified walking demonstration. If both attempts fail or
the time allowance is consumed, stop training and report that the bounded
attempt did not prove walking. Do not create another experiment, widen the
acceptance criteria, restart training, or extend the budget automatically.
Further work then requires a new explicit user instruction. Budget exhaustion
is not a claim that a PPO walking controller has been achieved.

## Historical research protocol — deferred, not current acceptance

The earlier milestone was a credible controller learned by PPO from random
weights, with native C++17 MuJoCo/LibTorch in the existing Debian GPU container.
Expert labels, pretrained weights, imitation initialization, scripted joint
trajectories and external support are excluded. Earlier runs and the separate
published/imitation controllers remain preserved.

## Acceptance

Three independently trained seeds each face five held-out resets at commands
0.25, 0.50 and 0.75 m/s: 45 continuous 60-second trials. At least 41/45 must pass,
and each training-seed/command group must pass at least 4/5. A passing trial has
no fall, reaches at least 70% of commanded forward distance, records five
debounced touchdowns per foot and eight alternating touchdowns, and spends at
most 10% of sampled time airborne. There are no resets within an evaluation.
The existing fall thresholds remain pelvis height <0.45 m or up-axis cosine <0.5.

Final initial-condition seeds are **9000000000–9000000004**, reserved before
training. Development screens instead use 8000000000–8000000001. Each seed
specifies independent uniform ±0.005 rad initial joint offsets. Different reset
seeds are not substitutes for three independent training seeds.

The numerical gate is necessary but not sufficient: actual alternating stepping
must be visually inspected, foot-contact slip and speed error reported, and
training provenance verified. These finite flat-ground tests establish neither
general stability nor hardware transfer. Quantitative evaluation never marks the
goal complete by itself.

Command tracking is reviewed separately from the approved numeric walking gate:
mean speed must be within 70–130% of the requested speed, with tracking RMSE no
greater than max(0.15 m/s, 30% of the command). This review threshold is recorded
before the new experiments. It prevents a fixed-speed gait from being presented
as command-following just because it exceeds all three minimum distances.

## Native evaluator

`humanoid-goal-evaluate` preserves the earlier evaluator and its original gates.
Its output path must be new. Each candidate supplies a training directory with
`run.json` and `policy.pt`; an explicit `--policy FILE` for each `--run DIR` may
select an immutable checkpoint instead. It hashes policies before and after
trials, binds each hash to that run's final-export or checkpoint metadata,
snapshots training manifests and records every failed trial. A changing
checkpoint invalidates any evaluation; a changing training manifest invalidates
final evaluation. Use completed or frozen runs for the final gate. Development
screens may inspect an immutable checkpoint while its parent run continues;
they snapshot the starting manifest, record its ending hash, and require the
checkpoint and its binding metadata to remain unchanged.
The evaluator reports its executable hash and the current source-file hash
separately; the latter does not claim that unbuilt source edits are in the binary.

```powershell
.\container.ps1 exec ./build/native/humanoid-goal-evaluate --screen --run artifacts/training/RUN --output artifacts/training/RUN/development-screen.json
.\container.ps1 exec ./build/native/humanoid-goal-evaluate --run artifacts/training/SEED1 --run artifacts/training/SEED2 --run artifacts/training/SEED3 --output runs/ppo-goal-evaluation.json
```

The development mode tests one policy at 0.5 m/s for two 20-second trials and
cannot qualify as milestone completion. Full mode requires three distinct
training seeds and checkpoint hashes. Neither mode modifies the viewer checkpoint.

## First improvement experiment

The original million-transition run learned standing. Its observations, action
range, reward and training budget differ from the successful imitation student;
that comparison does not establish that PPO is inferior. A separate native
trainer will test causal history, an asymmetric critic, wider action targets,
stronger command incentives and concurrent rollouts. These changes test a new
training recipe; they are not a controlled ablation of any one component.

The new [trainer specification](../training/WALK_PPO.md) records that recipe.
On 2026-09-17, the added torque/contact-slip measurements reproduced all ten
prior CPU baseline/control trials exactly in time, distance, fall state and
alternating counts. The published controller still passed 5/5, and all passive
controls fell with zero measured actuator effort. See the
[regression report](../artifacts/ppo-milestone/metrics-regression.json).
The GAE and milestone acceptance checks both passed in the existing Debian
container. No new PPO walking result has yet been established.

## Recorded development runs

| Run | Transitions | Updates | Training/export time | Result |
| --- | ---: | ---: | ---: | --- |
| `history-ppo-smoke32-seed1` | 8,192 | 80 | 4.77 s | Export/reload passed; 1,856 transitions/s by final rollout |
| `history-ppo-smoke64-seed1` | 32,768 | 160 | 11.32 s | Export/reload passed; 2,973 transitions/s by final rollout |
| `history-ppo-resume-smoke-seed1` | 4,096 additional; 36,864 cumulative | 180 cumulative | See run report | Resumed 32,768-transition checkpoint, preserved seed and optimizer accounting |

All are under `artifacts/ppo-milestone/`. The 32,768-transition export failed
both separate 20-second development trials, falling at 1.22–1.24 seconds with
zero alternating touchdowns. Short optimization checks do not establish walking.

The next run, `history-ppo-v2-seed1-10m`, starts from random weights with commands
sampled uniformly from 0.25–0.75 m/s, 64 worlds, eight workers, a 64-step rollout,
five epochs and 1,024-sample minibatches. It requests 10 million transitions,
saves every 524,288 transitions and has a 7,200-second wall-time cap. This is the
first sustained test of the new recipe; its status and results are in its
`run.json` and flushed `progress.csv`. Its selection screens use development
seeds, leaving the fixed final evaluation seeds untouched.

## Reward diagnosis and bounded continuation

The v2 run was stopped deliberately after 3,526,656 transitions (1,295.86 s,
17,220 optimizer updates), preserving its final model, policy, optimizer and RNG
with verified hashes. Its requested ten-million-transition budget was not
completed. The final export hash is
`91f9717dc79628b33083720de8a76eab47c1ad61de3d4341ff9ee861c0114210`.
The original executable is retained in the run directory as
`humanoid-walk-train-v2`. Interrupted shutdown screens contain no elapsed trial;
use completed checkpoint screens or a separate replay for performance evidence.

The development screens show why higher return does not establish walking:

| Checkpoint | Command | Duration | Forward distance | Alternations | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 524,288 | 0.50 m/s | 1.12 s | 0.73 m | 0 | Fell |
| 2,097,152 | 0.50 m/s | 20 s | 0.57 m | 2 | Upright, substantial slip and turning |
| 2,621,440 | 0.50 m/s | 20 s | 7.89 m | 0 | Displacement without alternating steps |
| 3,145,728 | 0.75 m/s | 20 s | 13.09 m | 1 | Displacement without sustained steps |

At checkpoint 2,621,440 and 0.50 m/s, mean supporting-foot slip was 0.324 m/s
(RMS 0.512 m/s). Its speed RMSE was only 0.130 m/s, demonstrating why tracking
and distance alone are insufficient. These sampled contact measurements support
a sliding diagnosis; visual gait review remains required for any claimed success.

A load-only native diagnostic reproduced the reported trajectory measurements
exactly for nine trials across checkpoints 524,288, 2,097,152 and 2,621,440.
It computed each reward on identical actions and states, without optimizer
updates. At 524,288 and 0.50 m/s, the short fall earned discounted v2 return
1.247; v3's fall/heading correction reduced it to -0.496. Ideal stationary
balance has an analytical v2 infinite-horizon upper bound of 0.2 before effort
costs, so the short fall exposes an incentive problem. This analytical reference
is not a measured standing-policy trial.

V3 alone still rewards straight sliding: the 2,621,440 checkpoint's 0.50 m/s
trial has discounted return 7.003 under v2 and 8.187 under v3. These are finite
rollout sums with gamma 0.99 and no value bootstrap, not evidence that v3 trains
better. See `artifacts/ppo-milestone/v2-reward-diagnostic-verification.json`
and the associated immutable diagnostic directories.

The next comparison is recorded before training in
`artifacts/ppo-milestone/reward-v3-v4-experiment-plan.json`. Both arms restore
the same 2,621,440-transition seed-1 checkpoint, including optimizer and random
state, then restart physics episodes using the same saved reset counters and
command generators. Both receive at most 1,048,576 additional transitions, with
development screens every 262,144. V3 is the correction described above; v4
changes only its stance-slip penalty to
`4 * min(sum_support_contact_speed_squared / command_squared, 25)`.

Selection requires sustained alternating touchdowns, useful commanded distance,
no fall and lower measured slip. Replacing sliding with standing, creeping or
higher reward is insufficient. This is an exploratory comparison on one training
seed; it cannot meet the three-seed milestone. Its reward switch retains the
parent critic and optimizer, and absolute heading is not directly observed by
the current actor or critic. The reserved final reset seeds remain unused.

Both bounded arms completed their 1,048,576 additional transitions. Native tests,
replay equivalence, explicit v4 resume and inherited-v4 resume passed before
launch. Their initial reported trajectories, saved model/optimizer/RNG,
reset counters, command generators and parent provenance were compared in
`artifacts/ppo-milestone/reward-v3-v4-start-verification.json`.
The compiled trainer is preserved as `artifacts/ppo-milestone/humanoid-walk-train-v4`;
its SHA256 is `b305fd470a21948129595daf64e3e38ac08595c6aec92c74f90482c4c6c5a2a1`.
The initial optimizer files have different raw hashes because LibTorch writes
process-specific tensor pointer IDs and unordered state entries. The separate
native `humanoid-compare-checkpoints` tool verified exact model values, all 13
complete and finite Adam parameter states, ordered parameter mappings and group
settings. An updated-checkpoint negative control correctly reported differences.
The proof is `artifacts/ppo-milestone/optimizer-audit/trained-state-proof.json`.

| Final arm | Command | Duration | Distance | Alternations | Mean support slip |
| --- | ---: | ---: | ---: | ---: | ---: |
| v3 | 0.25 m/s | 20 s | 2.44 m | 0 | 0.189 m/s |
| v3 | 0.50 m/s | 20 s | 8.76 m | 1 | 0.231 m/s |
| v3 | 0.75 m/s | 20 s | 5.08 m | 2 | 0.382 m/s |
| v4 | 0.25 m/s | 15.32 s, fell | -0.22 m | 0 | 0.011 m/s |
| v4 | 0.50 m/s | 20 s | 0.34 m | 1 | 0.009 m/s |
| v4 | 0.75 m/s | 7.50 s, fell | -0.25 m | 1 | 0.017 m/s |

Neither arm records any touchdowns after five seconds in its final screens.
V3 preserves movement without sustained steps; v4 greatly reduces slip but
largely stands by the final checkpoint. V4 had learned short forward falls
earlier in this budget, so that late change must inform the next decision.
These finite screens do not establish a general causal effect or walking.
Both cumulative checkpoints are at 3,670,016 transitions; invocation times are
750.82 s for v3 and 725.44 s for v4, with concurrent execution and different
initialization times. Raw results and verified final checkpoint hashes are in
`artifacts/ppo-milestone/reward-v3-v4-result.json`.

Use `runtime/summarize-walk-runs.ps1` for a read-only development snapshot. The
viewer continues to show the earlier controller until a new PPO policy meets
the milestone. No final-goal trial has been run and walking remains unproven.

## Unchanged v4 continuation

The final change toward low-slip standing warrants a short observation block,
not another reward change yet. `reward-v4-stability-extension-seed1` restores
the final v4 checkpoint and inherits v4 unchanged for 524,288 additional
transitions, with a midpoint screen at 262,144 and a 1,200-second cap. Further
extension requires repeated touchdowns after five seconds and increasing
sustained commanded distance; longer standing or higher return alone is
insufficient. The plan and build verification are recorded in
`artifacts/ppo-milestone/reward-v4-extension-plan.json` and
`artifacts/ppo-milestone/reward-v4-extension-build-record.json`.

The newly compiled trainer also contains an unselected v5 recipe (v4 with fall
penalty 10), but the continuation trains with v4. Before launch, all prior v2,
v3 and v4 finite returns and reported physical trajectories were reproduced
exactly on the final v4 checkpoint. On those trajectories, v5 would change
discounted return by only -0.0023, zero and -0.1166 across the three commands.
This explains why the earlier short-fall motivation does not establish that
doubling the penalty addresses the final standing behavior.

This continuation completed its 524,288 transitions in 185.20 s, reaching
4,194,304 cumulative transitions. At 0.25/0.50/0.75 m/s, its final screens
lasted 14.38/20/20 seconds and moved -0.109/-0.069/0.314 metres. All three had
zero alternating touchdowns and no touchdown after five seconds. Supporting-foot
slip stayed low (mean 0.009–0.013 m/s), but longer standing did not satisfy the
recorded continuation criterion. No further unchanged-v4 extension is selected.
The next investigation concerns incentives for discovering steps. The complete
result and verified final checkpoint hashes are in
`artifacts/ppo-milestone/reward-v4-extension-result.json`. V5 remains unselected;
the milestone is active and unmet.

The next experiment is recorded in
`artifacts/ppo-milestone/step-discovery-experiment-plan.json`. Relative to v4,
it changes only the airtime and alternation event
bonuses from speed-times-upright gating to upright gating, giving stationary
exploratory steps some credit. Its budget is 524,288 transitions from checkpoint
4,194,304, with screens every 131,072. Peak swing sole clearance and late
touchdowns will distinguish lifting from chatter; a declared 1 cm clearance
threshold is diagnostic only. Marching alone still does not establish walking.

V6 has been implemented and launched as `reward-v6-step-discovery-seed1` after
native PPO, acceptance-criteria and swing-diagnostic tests passed. The new
observer reproduced the saved parent's physical trajectories and all v2–v5
finite reward sums exactly. A separate 4,096-transition resume check completed
20 optimizer updates and passed export/reload validation. The compiled trainer
is preserved as `artifacts/ppo-milestone/humanoid-walk-train-v6`, SHA256
`45a90de05179450519db6a4a5f90dccb1f5ffdfddabb99364bee17859036ab6c`.
See `artifacts/ppo-milestone/reward-v6-build-record.json` for source hashes and
validation. The run completed its 524,288 additional transitions in 202.95 s,
reaching 4,718,592 cumulative transitions. All three final development screens
survived 20 seconds but moved only 0.389–0.412 metres. Each had one early
qualifying left-foot lift and no qualifying lift or alternating touchdown after
five seconds. Mean support slip was 0.010–0.013 m/s. This is stable standing;
the registered criterion rules out another unchanged-v6 extension.
The result and verified checkpoint hashes are preserved in
`artifacts/ppo-milestone/reward-v6-step-discovery-result.json`.

The next registered investigation is load-only stochastic playback, with five
paired sampling trials at each command: deterministic mean, learned noise,
twice learned noise, and learned noise held for four control intervals while
the policy mean still updates at 50 Hz. The physical reset stays fixed. Late
qualifying lifts, survival, post-five-second movement/slip and actual joint-target
perturbation sizes will inform the next experiment. This diagnostic performs no
optimizer updates and cannot establish that a newly learned policy walks. The
plan is `artifacts/ppo-milestone/v6-exploration-probe-plan.json`.

The probe completed all 60 planned playbacks. Default playback reproduced all
111 original non-timing fields across three saved screens exactly. All 15 mean
references matched, and a repeated native-noise trial at each command reproduced
the physical, reward and noise diagnostics exactly. Checkpoint hashes and build
sources remained unchanged; no optimization occurred.

| Playback | Falls / 15 | Trials with late qualifying lifts | Late lifts | Late alternations |
| --- | ---: | ---: | ---: | ---: |
| Mean | 0 | 0 | 0 | 0 |
| Learned noise, 20 ms | 6 | 8 | 18 | 0 |
| Twice learned noise, 20 ms | 15 | 0 | 0 | 0 |
| Learned noise held 80 ms | 15 | 3 | 15 | 0 |

All 33 late qualifying lifts were right-foot lifts. Native noise also produced
35 brief left-foot support losses, but none reached both the declared airtime
and clearance thresholds. At 0.5 m/s, native playback survived all five sampling
trials while the mean policy remained in double support at every endpoint after
five seconds. Mean-policy actions reached 95% of the actual action limit only
0.25–0.28% of joint samples, so the older `abs(action) >= 0.95` telemetry does
not establish widespread saturation for this policy's ±3 action range.

These limited fixed-reset probes favor investigating weight transfer and reward
credit over blindly increasing action noise. The next proposed diagnostic is a
paired trace of mean playback and native sampling trial 5 at 0.5 m/s, with
bilateral foot loads/clearance, joint targets, lateral motion, reward components
and local critic signals. That native trial survived 20 seconds and made three
late right-foot lifts. Its from-reset discounted return alone cannot establish
how PPO credits individual late actions; local bootstrapped signals are needed.
No further training change has been selected. Results and verification are in
`artifacts/ppo-milestone/v6-exploration-probe-result.json` and
`artifacts/ppo-milestone/v6-exploration-probe-analysis.json`. Final held-out reset
seeds remain unused, the viewer checkpoint is unchanged, and the goal remains
active and unmet.

Reported trainer seconds cover training/export and screens after model/world
initialization; they exclude initialization and compilation. The upstream
controller also has materially different reward and recurrent-policy settings,
recorded separately in [the pinned source audit](UPSTREAM_RECIPE_AUDIT.md).
