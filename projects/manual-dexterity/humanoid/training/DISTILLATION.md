# Expert-guided native imitation learning

`humanoid-distill` trains a new C++/LibTorch student using action labels from the
official Unitree G1 TorchScript policy. This is **supervised imitation with
DAgger-style data aggregation**, not PPO or independent reinforcement learning.
The student starts with random parameters; teacher weights are not copied.
The existing PPO experiment is unchanged.

The eight-frame student now demonstrates locally learned walking: its selected
checkpoint passed the independent 20-second gait screen in **four of five
initial-condition trials**. The other trial fell after 16.66 seconds. This is
evidence for supervised imitation in the tested flat-ground simulation; it does
not establish robust walking across resets or learning through PPO.

After building the `humanoid-distill` CMake target in the existing container:

```powershell
.\container.ps1 exec /workspace/build/native/humanoid-distill --rounds 20 --steps-per-round 4000 --epochs 20 --max-seconds 600 --output /workspace/artifacts/imitation/g1-history8
```

The output directory must be new. CUDA is the default; `--device cpu` is an
explicit alternative. Other options are `--seed` and `--asset-root`. The default
budget covers collection and optimization for up to 600 wall-clock seconds;
finishing the current round's continuous evaluation and saving artifacts can
take additional time. SIGINT/SIGTERM requests stop collection/optimization and
retain available results. No container lifecycle operation is performed.

The student receives eight causal observation frames: the current 47 values and
up to seven preceding frames, flattened oldest to newest into 376 inputs. At
50 Hz this is a 0.16-second window, with 0.14 seconds between the oldest and
current samples. Missing frames at each episode reset are zero. Two 256-unit
tanh hidden layers produce `3*tanh(...)` for 12 position-target actions. The
public viewer interface remains one current 47-value observation per call; its
exported policy maintains the history internally. The shared controller
converts each normalized action into a nominal joint position
plus `0.25*action` radians. The wider ±3 output covers the measured published
policy's ±1 crossings; the learner records teacher labels outside ±3 and does
not silently clip them.

The expert weight decreases linearly from 1 to 0 across the requested rounds.
Five default rounds therefore use weights 1, 0.75, 0.5, 0.25, and 0; the command
above requests 20 rounds. Each step follows this order:

1. Read the current observation, including the previously applied action.
2. Append it to the causal student history, then run the teacher on only the
   current 47 values and save the 376-value history with that teacher action.
3. Apply the weighted expert/student action blend through real MuJoCo physics.

The teacher is reloaded and student history is zeroed at every episode reset.
This order prevents a future-action label from leaking into its own previous-action
input. The final round visits student-controlled states while the teacher still
provides labels. A reservoir buffer retains at most 100,000 observation/label
pairs, each holding 376 history values and 12 action labels. Adam minimizes raw
action MSE over shuffled minibatches of 256; each round trains on the aggregated
buffer. The CUDA teacher uses ATen recurrent
operators with cuDNN disabled, matching the measured native inference fix.

Every round retains `model.pt`, an independently usable `policy.pt`, and a
20-second continuous evaluation that stops on a fall without resets. The
export is constructed entirely in C++. It accepts shape `[1, 47]`, shifts its
registered `[1, 376]` history buffer, appends the current observation, and runs
the student. Export verification compares a 12-frame nonzero sequence against
the C++ caller-managed history, including rollover beyond eight frames. The
buffer is explicitly zeroed before saving; a fresh reload must start at zero
and reproduce the complete sequence. Native playback reloads the policy at
every episode reset, so no history crosses episodes. Candidates passing the
shared gait screen rank first; remaining ties use survival time + capped forward
distance + capped alternating
touchdowns. All candidates remain available. The selected candidate is copied
to the run directory's `policy.pt`.

`run.json` records the teacher hash, model/asset and source provenance, random
seed, architecture and history window, expert blend schedule, losses, actual
updates, candidate hashes and evaluations. `progress.csv` is flushed each round,
and `dataset.pt` saves the
final history/label buffer. Each screen records the external student's inference
device separately from the CPU physics wrapper and labels the policy origin as
supervised imitation. The pinned teacher SHA256 is
`cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d`.

One evaluation seed per round is used for selection, so it is not held-out
validation. Before presenting a walking result, evaluate the selected exported
policy independently with the shared native evaluator across at least five
initial-condition seeds and inspect actual gait. Publish it to the viewer's
**Our imitation policy** option only after that review. The training program
does not replace `/workspace/runs/distilled-policy.pt` or any viewer artifact.

Validation records from 2026-09-16:

- `artifacts/imitation/native-dagger-seed1`: the original single-frame student
  completed five rounds and 2,370 updates in 59.75 seconds. Its selected policy
  made six alternating touchdowns, then fell at 1.60 seconds. It did not pass
  the walking screen; the exact v1 source and results are retained.
- `artifacts/imitation/native-dagger-history8-seed1`: the first history version
  failed during export because the native TorchScript parser could not resolve
  `RuntimeError`. Its source and `failure.json` are preserved.
- `artifacts/imitation/native-history-export-smoke`: after replacing that guard
  with a TorchScript assertion, the one-update smoke completed, including the
  12-frame export comparison, zero-before-save check, and fresh reload check.
  This verifies the history/export path; it is not evidence of walking.
- `artifacts/imitation/native-dagger-history8-seed1-r2`: the corrected history
  student completed 20 rounds, 80,000 collected action labels and 65,800 Adam
  updates in 464.539 seconds. Each round requested 20 epochs over the aggregate
  buffer. Round 19 was selected by the training screen; its exact source,
  checkpoint, training report and subsequent `evaluation-five-seeds.json` are
  preserved in that run directory.

The independent evaluation used the selected student alone with CUDA inference,
CPU MuJoCo physics, a 0.5 m/s command and no resets after a fall. Each new trial
reloaded the exported checkpoint with zero observation history. Seed 0 uses the
canonical initial pose; seeds 1–4 add up to ±0.005 radians of initial leg-joint
noise. These are five initial conditions of one learned policy, not five
independently trained policies.

| Reset seed | Continuous time | Forward distance | Alternating touchdowns | Gait screen |
| --- | ---: | ---: | ---: | --- |
| 0 | 20.00 s | 9.1247 m | 49 | Pass |
| 1 | 16.66 s | 6.8466 m | 41 | Fall |
| 2 | 20.00 s | 9.1193 m | 49 | Pass |
| 3 | 20.00 s | 9.1304 m | 49 | Pass |
| 4 | 20.00 s | 8.8148 m | 49 | Pass |

Each passing trial recorded 25 debounced touchdowns per foot and no airborne
samples. Paired zero-torque controls all fell within 0.56–0.58 seconds and passed
zero of five trials. The selected `round-19/policy.pt` SHA256 is
`a77e672689271cf91907be2a2d058bb3e3cd2b865629d440041ba531e91ed1bc`.

The original `run.json` retains `walking_proven: false`: the training program
only selects candidates using one reset seed and leaves certification to a
separate evaluation. Its `candidate_walking: true` is that selection result;
the later `evaluation-five-seeds.json` supplies the independent **4/5** result.
These records remain separate so the training report's provenance is preserved.
The viewer labels this controller **Our imitation policy**.

On exceptions, the trainer writes `failure.json` and updates `run.json` to
`status: "failed"` with the error, retaining previous round results and
provenance. A failure must not leave a valid run report marked as collecting.

Sources: [DAgger paper](https://proceedings.mlr.press/v15/ross11a.html),
[Unitree's pinned G1 deployment source](https://github.com/unitreerobotics/unitree_rl_gym/tree/276801e46c5d433564f24658bac64f254b7d2d4b),
and [PyTorch C++ frontend](https://docs.pytorch.org/cppdocs/frontend.html).
