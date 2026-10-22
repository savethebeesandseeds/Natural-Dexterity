# Bounded PPO feasibility result

The bounded attempt has stopped without a convincing walking proof. This is
the recorded outcome, not a reason to extend training automatically.

The user requested a small proof rather than a polished gait or an expanding
research programme. The revised target was one native C++ PPO controller,
learned from random weights without expert assistance, visibly walking for
20 continuous seconds with alternating steps, forward movement and no fall.
The cap was two further optimizer invocations, each at most 600 trainer seconds.

Both invocations are finished: 314.370 seconds for v7 and 512.676 seconds for v8,
827.046 seconds combined (13 minutes 47 seconds). These are the additional costs
after the scope reduction, not the total cost of the earlier investigation.
Initialization, compilation and final replay are outside the trainer timer.
No optimizer invocations remain and no further experiment is planned.

## Observed result

The selected v8 checkpoint, at 6,291,456 cumulative transitions, ran on CUDA
through the native TorchScript loader. A continuous 20-second replay from reset
8000000000 advanced 0.848591 m without falling. It recorded 40 left and 43 right
touchdowns, 49 contact alternations and 1.6% airborne samples. Mean forward speed
was 0.042430 m/s; supporting-foot slip averaged 0.072227 m/s (RMS 0.120109 m/s).
The 0.5 m/s command was poorly followed: speed RMSE was 0.505188 m/s.

The saved frames show genuine small foot lifts, but also an extremely wide
stance, largely straight legs, swivelling and foot pivots. The agent alternately
faces different directions and moves back and forth: forward displacement at
5, 10, 15 and 20 seconds is 0.333, 0.719, 0.431 and 0.849 m. There is movement
beyond the initial steps, but these images and substantial slip do not provide
a convincing demonstration of forward walking. This judgment does not impose
the deferred command-tracking or reliability thresholds.

A second fresh reset survived 20 seconds and advanced 0.904133 m with 52
contact alternations. Neither trial establishes general reliability. A separate
native-actor diagnostic recorded completed lifts of both feet above 1 cm with
at least 60 ms without force support; it is supporting evidence from a different
rollout, not the exact exported-policy replay. Its copied statement that 1 cm is
"never a reward gate" is stale: versions v7/v8 use that threshold for reward
credit. The immutable raw diagnostic remains preserved.

## Evidence and reproduction

- [Saved 20-second replay](../artifacts/ppo-milestone/bounded-proof-offline-render/preview.html): 201 actual MuJoCo JPEGs at 10 Hz, with playback and seeking.
- [Render measurements and provenance](../artifacts/ppo-milestone/bounded-proof-offline-render/report.json) and [per-frame states](../artifacts/ppo-milestone/bounded-proof-offline-render/frames.json).
- [Two fresh CUDA evaluations](../artifacts/ppo-milestone/bounded-proof-candidate-evaluation.json), retaining the original evaluator's deferred gate results.
- [Budget](../artifacts/ppo-milestone/bounded-proof-budget.json) and [machine-readable closure](../artifacts/ppo-milestone/bounded-proof-result.json).
- [Selected checkpoint](../artifacts/ppo-milestone/bounded-proof-v8-attempt2/checkpoints/6291456/policy.pt), SHA256 `4b035a7f749286cbb88bd9dd90cb05e063225003b243189fc7ab417b12150dd6`.

Training provenance preserves random initialization, no expert labels or copied
teacher weights. Native PPO, acceptance and swing-diagnostic tests passed for
the training build, with older recorded physical trajectories reproduced.
The new offline C++ renderer compiled and reproduced the evaluator's distance,
contacts and no-fall result for the same checkpoint and reset. Physics and
rendering run on CPU; neural inference and training use CUDA. No Python was used.

The offline renderer can reproduce this artifact without training or changing
the live viewer; its output directory must be new:

```sh
./build/native/humanoid-render-proof \
  --policy artifacts/ppo-milestone/bounded-proof-v8-attempt2/checkpoints/6291456/policy.pt \
  --output artifacts/ppo-milestone/another-offline-replay \
  --seed 8000000000
```

Automatic approval review initially rejected replacing the live viewer's policy
with an unverified candidate. Offline inspection was completed, and the candidate
was not promoted as a walking success. The user subsequently explicitly asked to
see this trained policy in the existing webpage. The selected v8 checkpoint and
matching evaluation report are now available under **Our PPO policy** at port
43871 as an experimental preview. The earlier viewer policy/report are backed up
in `artifacts/ppo-milestone/viewer-before-bounded-preview`; the separate published
and imitation controllers are unchanged. The page labels the gait's limitations.
The final result JSON records the state at the earlier closure; the subsequent
live preview's hash and measurements are recorded in
`artifacts/ppo-milestone/viewer-bounded-preview-state.json`. No new training was
performed, and the paper's claims are unchanged.

The three-seed study, robustness, speed tracking, more reward recipes and
subsequent research milestones are deferred. Further work requires a new
explicit user instruction.
