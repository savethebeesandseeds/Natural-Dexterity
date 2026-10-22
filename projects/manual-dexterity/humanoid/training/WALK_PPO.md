# From-scratch history PPO

`humanoid-walk-train` is a separate native C++17/LibTorch trainer. It starts with
random weights, uses only MuJoCo observations and measured physical rewards,
and never loads an expert controller, action labels, joint trajectories or
external support. The original PPO and imitation experiments remain available
for comparison. This implementation has no walking result until measured.

Build the root CMake target, then run a bounded optimization/export smoke in
the existing container:

```powershell
.\container.ps1 exec /workspace/build/native/humanoid-walk-train --steps 8192 --envs 32 --workers 8 --horizon 64 --epochs 5 --minibatch 512 --command-min .5 --command-max .5 --max-seconds 180 --eval-seconds 0 --output /workspace/artifacts/walk-ppo/smoke-seed1
```

`--output` must be new. CUDA is the default and has no silent CPU fallback.
Persistent worker threads step independent CPU MuJoCo worlds; neural inference
and optimization are batched on the selected device. The default is 64 worlds
and eight workers. All draws for action noise and minibatch permutations use
the saved CPU Torch generator before transfer to CUDA. Load-only diagnostic
noise has a separate local generator described below.

The actor consumes eight causal 47-value observations, oldest to current, with
zero padding after reset. Its 376→256→128→12 network uses ELU hidden activations
and a learned diagonal Gaussian followed by `3*tanh`; the simulator maps each
action unit to 0.25 radians of position offset. Initial latent standard deviation
is exp(-1). The critic adds 16 privileged measurements: pelvis linear/angular
velocities, height and uprightness, foot clearance, contact support/slip,
command speed and measured torque RMS. Privileged inputs never reach the actor.
Existing observation scales are retained and inputs are clipped to ±10.

Training samples a command uniformly per episode in `--command-min .25` through
`--command-max .75`. Set both to `.5` for a first fixed-command experiment.
Changing that range on a later resume provides a recorded command curriculum.
Episodes end on the shared physical fall condition or truncate after
`--episode-seconds` (20 by default). GAE uses the actual final history and
privileged state for timeout bootstrap, and cuts recursion at every reset.

The default `--reward-version v2` combines rates multiplied by the 0.02-second
control interval with explicit contact events and a -1 fall penalty:

| Term | Reward rate or event |
| --- | --- |
| Forward tracking | 4 × upright × (Gaussian velocity tracking minus its stationary score); width 0.15 + 0.15 × command |
| Forward progress | 0.5 × clamp(body forward speed / command, -1, 1) |
| Upright survival | 0.1 × upright cosine |
| Body motion | -0.5 × lateral speed²; -0.5 × vertical speed²; -0.05 × roll/pitch rate²; -0.25 × yaw rate² |
| Tilt | -(1-upright)² |
| Stance slip | -0.4 × summed squared supporting-contact tangential speed, capped at 25 |
| Effort | -0.00001 × mean actual joint torque²; -0.0001 × absolute mechanical power |
| Action change | -0.015 × mean normalized action change² |
| Contact support | +0.25 × forward-motion fraction for single support; -0.5 when neither foot supports |
| Touchdown event | 0.35 × clamp(airtime-0.04, 0, 0.3) × forward-motion fraction, after at least 0.06 s absence |
| Alternation event | +0.1 × forward-motion fraction for an opposite-foot touchdown separated by at least 0.16 s; simultaneous landings excluded |

The forward-motion fraction is clamp(body forward speed / command, 0, 1).
Contact events use force-bearing floor contact and measured airtime; there is
no desired joint trajectory or phase-target reward. The stationary tracking
subtraction changes the absolute reward baseline; the much smaller survival
reward and measured progress/contact terms supply the preference for movement.

The `--reward-version v3` option changes only this reward recipe:

- Upright survival rate becomes 1.0; the discrete fall penalty becomes -5.
- Forward tracking, progress and the forward-motion fraction use world `vx`;
  the lateral velocity penalty uses world `vy`.
- The progress term and forward-motion fraction are additionally multiplied by
  upright cosine. This gates the single-support, airtime and alternation bonuses
  by uprightness as well. Tracking retains its existing upright multiplier.

All other reward coefficients, physical measurements, policy inputs, actions,
PPO calculations and success gates stay the same. The aim is to make short
forward falls less attractive while preserving a substantial tracking advantage
over standing. The original v2 experiment and subsequent recipe comparisons
retain their own results. The exact selected recipe is recorded in `run.json`
and every checkpoint's `metadata.json`.

The `--reward-version v4` option is exactly v3 except for the stance-slip cost:

```text
raw_slip_squared = sum of squared contact-point speeds for supporting feet
v3 slip cost rate = 0.4 * min(raw_slip_squared, 25)
v4 slip cost rate = 4.0 * min(raw_slip_squared / (command_x * command_x), 25)
```

The command is validated in the positive range 0.25–0.75 m/s, so no epsilon or
additional clipping is introduced in the denominator. The cap is applied after
command normalization in v4. This cost is still a rate multiplied by 0.02 seconds;
all other terms, contact bonuses, actions, architecture and PPO calculations are
unchanged. `mean_slip_squared` and the `progress.csv` column `slip` remain the
**raw**, unnormalized sum in m²/s²; the run marks this explicitly. The recipe
records the coefficient, normalization and cap order.

V4 is a hypothesis motivated by a v2 checkpoint that slid for 20 seconds without
alternating footsteps while receiving substantial v3 counterfactual return.
It has no established walking result. Continuing an existing random-initialized
PPO checkpoint under v4 retains its prior training history and is a continuation
experiment, not a clean comparison between independently initialized recipes.

The staged, **unlaunched** `--reward-version v5` is exactly v4 except that the discrete fall
penalty is -10 instead of -5. Normalized stance slip, world-frame tracking,
upright gates, survival, contact bonuses, actions, architecture and PPO are
unchanged. V2, v3 and v4 keep their original recipes.

The motivation is a measured v4 continuation checkpoint after 786,432 additional
transitions: its 0.25 and 0.5 m/s screens fell after 2.44 and 2.00 seconds while
earning discounted returns about 2.4255 and 2.3922. Both exceed the cost-free
standing reference `0.02 / (1-0.99) = 2`. For those same trajectories, an extra
-5 terminal penalty gives returns about 0.94 and 0.54, using the actual terminal
discount indices. Four-way diagnostics compute these sums directly. This is a
bounded incentive correction; it neither demonstrates walking nor rules out
rewarding all longer trajectories that eventually fall. V5 still needs its own
continuation and independent walking evaluation.

That motivation concerns the earlier checkpoint, not the final v4 behavior.
At cumulative checkpoint 3,670,016, the 0.5 m/s screen instead stayed upright for
20 seconds while moving only 0.339 m. The 0.25 and 0.75 m/s screens fell after
15.32 and 7.50 seconds with slightly negative forward displacement. All three
had only 0–1 alternations, no touchdowns after five seconds, and stance-slip
means around 0.009–0.017 m/s. V4 had moved toward low-slip standing. V5 remains
an unlaunched hypothesis, and this later evidence must inform whether it is
the appropriate next experiment; staging does not select or schedule that run.

The selected step-discovery experiment uses **v6**, relative to v4:

```text
v4 airtime and alternation event multiplier = clamp(world_vx/command, 0, 1) * upright
v6 airtime and alternation event multiplier = upright
```

Only those two event multipliers change. Single-support reward still uses
`moving`; tracking, normalized slip, survival, fall penalty **5**, controls,
observations, PPO and gamma remain v4. V6 does not inherit v5's penalty 10.
There is no clearance reward gate. The hypothesis is that exploratory leg
lifting can earn event credit before useful translation; marching or contact
chatter are possible failures, not walking successes.

The registered plan is
`artifacts/ppo-milestone/step-discovery-experiment-plan.json`. It continues seed 1
from `reward-v4-stability-extension-seed1/checkpoints/4194304` for 524,288
additional transitions, retaining checkpoints every 131,072 transitions. It
uses 64 worlds, eight workers, horizon 64, five epochs, minibatches of 1,024,
commands 0.25–0.75 m/s, 20-second episodes/screens and a 1,200-second cap. This
is a continuation with its existing optimizer and prior training history.
The bounded v6 run completed at checkpoint 4,718,592; its final development
screens mostly stood for 20 seconds, so they did not establish stepping or
walking. V5 remains unselected. Standing or chatter at the budget end rejects this recipe;
alternating lifts establish step discovery only. An extension needs late
alternating lifts, increasing forward distance, low slip and survival across
commands. Reserved goal reset seeds remain unused by these screens.

The first bounded attempt used **v7**, relative to v6. Its airtime coefficient
increases from 0.35 to 3.5, and its alternation bonus from 0.1 to 1.0. A v7
reward touchdown requires both at least 0.06 seconds without force support and
peak sampled sole clearance of at least 0.01 m during that unsupported interval.
Each foot's peak is updated only while unsupported, at the existing 50 Hz
endpoints, and reset on support and episode reset. A landing below either
threshold earns no airtime credit and is not recognized for reward alternation.

V7 keeps v6's upright-gated event bonuses, moving-gated single-support term,
all continuous rates, normalized slip cost and fall penalty 5. Physics,
architecture, PPO and the existing walking screen are unchanged. This heuristic
increases credit for actual foot lifts; unilateral flapping, marching or falling
still do not demonstrate walking. V2–v6 retain their original calculations and
contact histories.

The user narrowed the active target to **one visible 20-second walking proof**,
with at most two further optimizer invocations and 1,200 total trainer seconds.
No additional diagnostic tracing or broad evaluation study is planned. The
first attempt is registered from v6 checkpoint 4,718,592 at fixed command
0.5 m/s, at most 1,310,720 additional transitions and 560 loop seconds within
a 600-second total allowance; it uses 64 worlds, eight workers, horizon 64,
five epochs, minibatches of 1,024 and screens every 262,144 transitions.

Attempt 1 stopped gracefully after 786,432 transitions and 314.37 trainer
seconds. Two completed 0.5 m/s screens recorded 34 then 58 late qualifying
right-foot lifts, zero late alternations and roughly 0.5 m total displacement.
This was repeated unilateral lifting, not the requested walking proof.

The **final optimizer attempt** uses **v8**, exactly v7 except that its computed
airtime bonus is credited only when the current interval has `alternate == 1`.
The existing recognized touchdown must switch to the opposite foot with at
least 0.16 seconds separation. A first-foot, repeated same-foot or simultaneous
touchdown receives no airtime credit. A qualifying alternation receives both
its airtime credit and the existing 1.0 alternation bonus, each upright-gated.
The 1 cm peak-clearance and 60 ms airtime qualifications, all rates, normalized
slip cost, fall penalty 5, physics and PPO remain v7. V2–v7 retain their original
calculations and contact histories.

The coordinator plans this last attempt from the known-complete v7 checkpoint
5,242,880, with fixed command 0.5 m/s and at most 560 loop seconds within a
600-second total allowance. V8 is staged for build and replay verification;
this edit performs no optimizer invocation. No tracing or further experiments
are planned. Removing unilateral airtime credit is a bounded heuristic, not a
walking guarantee.

PPO uses clipping 0.2, gamma 0.99, GAE lambda 0.95, five shuffled epochs,
minibatches of 512, standardized advantages, value clipping 0.2, and gradient
norm clipping at 1. Adam starts at 0.0003 with epsilon 1e-5. Sampled approximate
KL adapts learning rate within 1e-5–1e-3 and stops further minibatches above
0.06. The default 0.003 entropy term is explicitly the unsquashed Gaussian
entropy proxy. The action density includes the full `3*tanh` Jacobian.

`--steps` means **additional transitions for this invocation**, rounded up to a
complete rollout. `--max-seconds` is checked between rollouts; checkpoint export
and optional screens add work. Interrupts finish the current rollout/update
and save its result. `--checkpoint-every` defaults to 262,144 transitions.
Each checkpoint retains `model.pt`, Adam `optimizer.pt`, CPU `rng.pt`, an
independently loadable `policy.pt`, metadata hashes and an optional screen.
The latest policy is also copied to the run directory; no viewer file is changed.

Resume into a new directory with `--resume PATH_TO_CHECKPOINT`. The trainer
verifies checkpoint hashes and provenance, preserves the original training seed,
and restores model, Adam state, learning rate, cumulative transitions/updates,
CPU random state and per-world command generators. Keep the same `--envs`.
Physics episodes restart with new reset counters and zero history: this is
optimizer/accounting continuation, not bitwise simulation continuation. Every
run records its parent checkpoint hash, source snapshots, executable hash and
parameters. Source-on-disk hashes are identified separately from the executable;
the build coordinator must verify those sources did not change before launch.
The CLI accepts `v2`, `v3`, `v4`, `v5`, `v6`, `v7` or `v8`, with v2 as the fresh-run default. Without an
explicit `--reward-version`, resume inherits the checkpoint's recipe;
legacy checkpoints without that field are identified as v2. An explicit recipe
change is allowed and recorded with the parent's version in both the run and
checkpoint provenance. Adam and the critic are retained, so a changed reward
needs fresh training before its value estimates can be trusted.

Exports accept `[1,47]` and own a `[1,376]` history buffer. Twelve nonzero frames
verify history ordering, rollover and numerical equivalence. The buffer is
zeroed before saving, and fresh loading must recover zero state and repeat the
sequence. Native evaluation resets by loading the file freshly each trial.

Optional screens use reset seed `8000000000 + training_seed` at 0.25, 0.5 and
0.75 m/s and stop on a fall without resets. Training excludes reserved reset
seeds `9000000000..9000000004`. The earlier 45-trial, three-training-seed study
is superseded by the user's narrower 20-second visible proof target above.
Standing, injected-noise motion or optimization progress alone is insufficient.

Every screen now records `counterfactual_returns.v2`, `.v3`, `.v4`, `.v5`, `.v6`, `.v7` and `.v8`, each with
`discounted` and `undiscounted` sums from the same actual physical rollout.
Each recipe has its own action-history and airtime/alternation state; rewards
are computed only after the chosen action is applied and never feed back into
control. Discounting uses gamma 0.99 starting at exponent zero. These are finite
rollout sums including any terminal fall penalty, with no value bootstrap or
assumed rewards after the observed end.

Screens additionally record `swing_diagnostics` using the existing simulation's
force-bearing support flags and lowest sole clearance. The independent observer
is reset from the initial physical state, then samples every completed control
endpoint at nominal 50 Hz. It records each support-loss interval's peak sole
clearance, observed air duration and touchdown time. A diagnostic qualifying
lift needs an observed, completed support-loss interval of at least 0.06 seconds
and peak clearance at least 0.01 m. Initial-airborne intervals and incomplete
swings are reported separately and do not count as qualifying completed lifts.

Per-foot late lift counts require touchdown after five seconds; a late
qualifying alternation requires both constituent qualifying touchdowns after
five seconds and at least 0.16 seconds separation. Simultaneous landings break
the alternation chain. These are sampled force-support events; they differ
from the simulator's legacy foot-contact touchdown counters and are not
substep-exact timing. The 1 cm threshold is an explicit diagnostic classification
only within that observer. Separately, v7/v8 rewards apply their own 1 cm
peak-clearance gate described above; the observer never changes reward state,
actions, termination or the walking screen. Its header is included in source snapshots.

To measure these returns for an immutable checkpoint without any optimization:

```powershell
.\container.ps1 exec /workspace/build/native/humanoid-walk-train --diagnose /workspace/artifacts/ppo-milestone/history-ppo-v2-seed1-10m/checkpoints/1048576 --device cuda --eval-seconds 20 --output /workspace/artifacts/ppo-milestone/reward-diagnostic-1048576
```

This load-only mode verifies model/policy hashes and no-expert provenance,
inherits the training seed, and rejects a conflicting explicit seed. It loads
the original `model.pt` and runs the three deterministic screens sequentially,
with only one simulation at a time. No optimizer, worker pool or training-world
batch is created; `--resume` and `--diagnose` are mutually exclusive. Output
contains `screen.json`, `run.json`, all seven counterfactual recipes and source/
executable hashes. `optimizer_updates` and `additional_steps` are zero;
`completed_steps` is explicitly labeled as the inherited source-checkpoint
count. No checkpoint or viewer file is overwritten.
The diagnostic also obtains model/asset hashes, physical metric definitions and
simulator provenance directly from its first existing simulation instance; no
extra world is created. It records the external actor's actual device and
source model checkpoint separately from the CPU simulation wrapper.

Load-only diagnostics also accept `--diagnostic-samples` (1–5, default 1),
`--diagnostic-noise-scale` (0–3, default 0), `--diagnostic-noise-hold` (1–8 control
steps, default 1), and `--diagnostic-seed` (unsigned 64-bit integer, default 1729).
Nondefault values require `--diagnose`; they cannot change training. Scale zero
uses the original deterministic mean-action expression. Each command receives
the requested number of trials at the same physical reset seed
`8000000000 + training_seed`; these are repeated paired exploration trials,
not independent physical reset seeds or the held-out walking gate.

For a positive noise scale, each step applies:

```text
action[t] = 3*tanh(mean(current_history[t]) + scale*model_stddev*epsilon[H*floor(t/H)])
```

The mean is recomputed at 50 Hz. Only standardized Gaussian noise is held for
`H` steps. The frozen model's actual per-joint latent standard deviation is
used, including its existing log-standard-deviation clamp. Every trial
pre-generates one 12-value Gaussian row per planned control step with a local
`std::mt19937_64` and `std::normal_distribution<float>`. Its seed is
`base + 1000003*command_index + 10007*trial_index`, with zero-based indices and
unsigned 64-bit wrap. Neither scale nor hold affects this seed. A held arm
uses the row at each hold boundary and discards intervening rows, so white and
held arms share boundary innovations. The trace is reproducible with the same
binary/standard library; the diagnostic draws do not use the training RNG.

Each screen records these settings, its exact derived seed, the 12 model and
effective latent standard deviations, and the actual post-tanh action-minus-
mean RMS from the same current state/history. Target-offset RMS multiplies
that difference by 0.25 radians, with per-joint values and a combined RMS over
all joints/observed steps. This measures effective perturbation after action
bounds. `near_action_bound_fraction` uses the explicit threshold 2.85 (95% of
the ±3 range); the older `near_one_fraction` field has a different meaning.

The `after_five_seconds` section reports observed exposure after nominal five
seconds, forward distance from that boundary, endpoint world-speed RMSE, raw
summed supporting-foot slip squared per interval, and supporting-foot speed
RMS over supported-foot endpoint observations. Exposure ends at a fall or
stop; it includes a terminal-fall interval and assumes no time after it. Missing
post-five-second observations yield null statistics, not zero errors. History,
physics, swing classifications and the seven independent counterfactual reward
histories otherwise follow the same screen path. Injected-noise motion is
**stochastic diagnostic playback**, not a new trained policy or learned gait.

The registered four-arm probe is
`artifacts/ppo-milestone/v6-exploration-probe-plan.json`: scale/hold pairs
`0/1`, `1/1`, `2/1`, `1/4`, five trials per command, 20 seconds each, base seed
1729000001. For example, stage the native-sigma held-noise arm with a new output:

```powershell
.\container.ps1 exec /workspace/build/native/humanoid-walk-train --diagnose /workspace/artifacts/ppo-milestone/reward-v6-step-discovery-seed1/checkpoints/4718592 --device cuda --eval-seconds 20 --diagnostic-samples 5 --diagnostic-noise-scale 1 --diagnostic-noise-hold 4 --diagnostic-seed 1729000001 --output /workspace/artifacts/ppo-milestone/v6-exploration-native-sigma-hold4
```

This command describes a diagnostic invocation; source preparation does not
execute it. All original checkpoint parameters and optimizer files are preserved.

The legacy pure C++ GAE tests cover terminal/time-limit semantics and are reused
unchanged. Native compile, optimization, export and resume results should be
recorded before promoting this implementation's runtime status.
