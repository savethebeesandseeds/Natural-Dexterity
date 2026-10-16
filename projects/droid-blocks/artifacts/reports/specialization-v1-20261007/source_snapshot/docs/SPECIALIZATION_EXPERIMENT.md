# Bounded specialization experiment — 7 October 2026

The child-facing goal is a broadly pretrained first creature that can then
specialize through small amounts of experience. This experiment measures that
second step on already exposed conditions. It does not establish new body
generalization, arbitrary topology, physical hardware readiness or child waiting
time. The live pretrained workshop continues to use its immutable base actor.

The source releases are the three pooled seeds 101/202/303 from the sealed
6 October breadth study. Each was pretrained for 98,304 decisions. Use all three
and all twelve exposed length-5 conditions: three layouts, six mechanics, two
lamp positions. These conditions were previously called reserved; their earlier
exposure means they are development conditions for this study. The original
rover-v3 challenge remains unexposed.

Four arms start each 640-decision opportunity in a fresh native world with fresh
local sensor history and matched categorical action RNG: pretrained unchanged;
output layers tuned; all weights tuned; and matched scratch. Scratch copies the
exact saved pre-source actor and critic. All models start with empty Adam state,
fresh rollout buffers and counters. No source optimizer is restored. Each target
and probe saves a fresh `policy_initial.zip` so independent inspection can verify
initial tensor identity, empty optimizer state and zero counters.

Learning arms spend the first 128 decisions learning in two 64-step rollouts,
with four PPO epochs per rollout and eight Adam steps total. Updates occur after
decisions 64 and 128: 6.4 and 12.8 simulated seconds. They then continue 512 decisions
with weights and optimizer fixed, in the same world, from the same local history
and continuing RNG stream. There is no reset, rescue, second learn call or reseed
at the boundary. The unchanged arm spends all 640 decisions without updates.
Horizon 641 prevents a 640-step auto-reset. Native faults terminate an opportunity;
the raw stopped prefix and any checkpoint already produced are retained.

Output tuning freezes eight actor/value embedding tensors and trains the four
output tensors: 195 actor plus 65 critic parameters, 260 total. Full and scratch
train 12 tensors containing 34,820 parameters. The deployed workshop actor has
17,475 parameters and does not contain the critic. Head-only saved Adam groups
contain four tensors; inspect/load their policy tensors into a fresh model for
frozen probes rather than using ordinary PPO.load with incompatible Adam groups.
All PPO hyperparameters, input scaling and native reward match the pinned source
study. The actor sees twelve 17-value local history frames; no body identity,
lamp coordinates, passive joint state or privileged critic observations.

The primary comparison is output tuning minus pretrained unchanged over the
**entire 640-decision opportunity**, including the 128 learning decisions. It passes
only if all 36 corresponding pairs safely complete, the median of three seed
mean gains is strictly above +3.2 raw native return, and at least two seed means
are positive. This is +.05 return per simulated second over 64 seconds. Full tuning
is a separately declared secondary comparison with the same gate; do not select
the better method after outcomes. Each tuned-arm versus scratch contrast has
separate complete-safe eligibility and requires median seed mean gain > 0 with at
least two positive seed means. A scratch failure does not erase an eligible
tuned-versus-frozen result.

Report all 36 paired differences for each contrast, seed means and their median,
worst regressions, signs, safety events, and actual exposure. Report0–128 prefix
and129–640 follow-up returns separately. Diagnostic regression floors are −.64
for the prefix and −3.2 for the whole opportunity. These are separate from the
primary average-benefit gate. The follow-up begins in states produced by each
arm's own behavior; it is not a same-state isolated weight comparison. Return is
unmodified native mounted light plus IMU reward, not PPO bootstrap value, lux,
or signed electrical energy. Three source seeds are the training replicates;
the36 case pairs are not 36 independent pretrained brains.

After all 144 target attempts and checkpoint references have been sealed, measure
sampled retention on two fixed old development conditions: first and last in the
old ordered list, `development-long-upper-base-left` and
`development-weighted-elbow-right`. Collect six original frozen baselines, then
two 640-decision frozen probes for each of 72 head/full after 128 checkpoints. Each
probe starts a fresh world/history and matched per-seed guard action RNG. Missing
after 128 checkpoints produce blocked zero-exposure receipts. Unfinished or
tampered target seals block the whole guard stage before any native world starts.
Sampled retention is separately eligible only when all corresponding probes
safely complete; every seed mean post-minus-base must be at least −3.2. Report
every probe and worst loss. Two cases cannot establish whole-domain retention.

Target action seed is810000+source_seed*1000+ordered target index. Guard action
seed is910000+source_seed*1000+ordered guard index. Seed once after model creation
and weights-only copy. Record each categorical draw's Torch RNG hashes and verify
PPO training leaves Torch RNG unchanged. Pretrained arms must match through their
first64 actions and native responses. Audit exact native request replay,
local observations, raw reductions, trainable tensor scope, checkpoints at64/128,
Adam steps, and fixed continuation weights separately. Native replay alone does
not establish neural inference or optimization correctness.

The maximum target collection is 144×640 = 92,160 decisions, including 13,824 learning
decisions. The separate retention collection is 150×640 = 96,000 decisions. Total
maximum is 294 opportunities, 188,160 decisions, 940,800 native controls and 9,408,000
physics steps, representing 18,816 aggregate simulated seconds. Fault stops reduce
realized exposure. Replay adds validation work, not independent training samples.
Wall-clock runtime is reported separately; every opportunity has a 180-second cap.

`training/specialization_manifest.py` locks the declaration, historical protocol,
release ZIP and tensor identities, source/native bytes, pinned dependencies and
preexisting artifact preservation baseline before collection. Source snapshots
and compact receipts remain retained evidence. Request/telemetry tapes and new
adapted ZIPs remain local and ignored; a fresh checkout needs those assets plus
the three source releases for complete replay. Never overwrite an attempt,
retry a failed opportunity, tune cases/hyperparameters from this batch, promote
an adapted checkpoint or replace the original pretrained base.
