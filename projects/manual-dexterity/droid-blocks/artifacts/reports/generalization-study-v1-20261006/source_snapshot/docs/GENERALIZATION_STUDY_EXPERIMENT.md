# Frozen first-use behavior after broader pretraining

**Prospective protocol, 2026-10-06.** This is a new study, separate from the
completed seed-404 development pilot. The lock must archive this document,
its canonical manifest and collection sources before source learning. No
reserved length-5 construction may be instantiated before the declared release
gate passes.
Outcomes belong in a separate results document; this protocol is not amended
after collection begins.

## Question and limits

The owner's requirement is useful competence on the first assembly a child
builds, inherited from broad simulation pretraining, followed by modest
specialization. This study asks whether broader source training improves frozen
first-use behavior over an equally trained narrower source on previously
uninstantiated bodies. It also measures useful behavior against quiet and
offline learning benefit against the saved initial neural policy.

The [completed development pilot](GENERALIZATION_PRETRAINING_RESULTS.md) used
seed 404 and 16,384 source decisions. Its trained controller passed the known
case competence margin and improved over its initial policy in ten of twelve
cases, with uneven gains. Those data justify this next prospective comparison;
they are development exposure, not reserved-body evidence. Seed 404's weights,
optimizer, observations and action history are not reused in this study.

All cases remain anchored, collisionless planar two-link mechanisms with one
powered hinge, one passive hinge, a distal IMU and a light sensor. Reserved
lengths test limited geometry extrapolation within this graph, not arbitrary
connectivity, variable module counts, walking, contact or hardware transfer.
There are no parameter updates during any evaluation in this stage. Head-only
adaptation, full adaptation and matched scratch learning need a separate
prospective study and new-experience budget.

## Declared source and evaluation domains

All lamps have position `[x, 0, -0.30]` metres, x=-0.30 or +0.30, and intensity
1000 lux. Both source curricula include powered hinge 0 (base) and 1 (elbow).
The IMU is segment 1, slot 2, side -1. The light sensor is segment 1, slot 2,
side +1, receiving face +1. The optional distal block is segment 1, slot 0,
side -1, with the unchanged native mass and dimensions.

| Domain | Bodies and variations | Cases | Use |
| --- | --- | ---: | --- |
| Pooled source | [3,3], [4,3], [3,4], [4,4] × plain/distal block × base/elbow × left/right lower lamp | 32 | Broader pretraining |
| Narrow source | [3,3] plain × base/elbow × left/right lower lamp | 4 | Equal-budget source-breadth control |
| Known development | [4,3] plain, [3,4] plain, [4,4] with block × base/elbow × left/right lower lamp | 12 | Pooled source competence and descriptive narrow transfer |
| Narrow own-domain | The four narrow-source conditions | 4 | Narrow source competence, assessed separately |
| Reserved | [5,3] plain, [3,5] plain, [5,5] with block × base/elbow × left/right lower lamp | 12 | Conditional frozen first-use evaluation |

Known development cases are inside the pooled source domain. They are outside
the narrow source domain but already inspected development conditions for this
research program; their narrow-policy results cannot be called unseen evidence.
The narrow own-domain four are disjoint from the known development twelve.
Reserved whole-assembly mechanics are disjoint from both sources; changing a
label, block ID or lamp does not define a new assembly identity.

Source case order is the listed lengths, block absent then present, base then
elbow, left then right. Narrow order is base/elbow and left/right. Known and
reserved order is long upper, long lower, weighted, each base/elbow and
left/right. `training/generalization_study_manifest.py` owns the canonical
`generalization_study_v1` manifest, including these orders, physical identities,
budgets and split labels. Case IDs and seeds are evaluator metadata, never
neural inputs.

The twelve length-5 cases have been declared as JSON but not configured, reset,
stepped or newly simulated in earlier pilots. Before release, they may be
checked statically for well-formed assembly sockets and disjointness only.
They may not be probed through a baseline, preview, normalization procedure,
debugging run, checkpoint selection or policy execution.

Earlier headroom, actuation-transfer and pilot conditions, traces, models,
metrics and resulting design choices are known exposure for this successor.
The original rover-v3 challenge remains separate and subject to its existing
owner-approval requirement. This protocol does not execute it or add an owner
approval gate to the new length-5 study.

## Fixed neural boundary and training budget

Reuse the installed Linux CPU Stable-Baselines3 2.9.0 PPO path and unchanged
native adapter. Every source arm has eight vector environments, including the
narrow four-case arm, which repeats conditions across slots. Source sampling
uses independent `random.Random(500000 + seed)` shuffled round-robin schedules
for each curriculum and seed, never advanced by policy/optimizer RNG. Every
complete cycle covers that curriculum once; resets are balanced, not
necessarily physical exposure. Retain draw schedules and realized per-case
decisions, native controls, duration, safety stops and complete/partial episodes.

Actor and critic receive the same causal 204-dimensional input: twelve frames
of seventeen local sensor, motor feedback, previous effort/request/reward and
validity/presence values, oldest to newest, with the existing fixed scaling
and [-5,5] clipping. Geometry, powered-hinge labels, masses, lamp coordinates,
world pose, passive joint state, split IDs and absolute clock stay outside the
input. No fitted observation/reward normalization or privileged critic is added.

Actions remain categorical requests -0.15, zero or +0.15 held for 0.1 seconds.
Five 0.02-second native controls refresh the unchanged governor and safety,
each containing ten 0.002-second physics steps. Training reward is the exact
sum of the already integrated native light and IMU transition rewards. It is
not reintegrated, shaped, normalized or replaced by an evaluator score.

| Setting | Every source arm |
| --- | --- |
| Independent seeds | 101, 202, 303; paired across pooled/narrow |
| Aggregate source decisions | Exactly 98,304 per seed per curriculum |
| Vector environments / rollout steps | 8 / 256 |
| Rollout updates / optimization epochs / Adam steps | 48 / 192 / 6,144 |
| Episode limit | 640 decisions / 64 simulated seconds |
| Actor/value networks | Separate two-layer 64 × 64 tanh networks |
| Learning rate / Adam epsilon | 0.0003 / 0.00001 |
| Epochs / minibatch | 4 / 64 |
| Gamma / GAE lambda | 0.92^(0.1/0.4) / 0.95 |
| Policy clip / value clip | 0.2 / none |
| Entropy / value coefficients | 0.01 / 0.5 |
| Gradient norm / target KL | 0.5 / none |
| Initialization / normalization | Orthogonal; advantage normalization only; no VecNormalize |
| Runtime | Existing pinned Linux CPU environment, one Torch thread |
| Wall-time caps | 600 seconds per source, including initialization and learning; 180 seconds per evaluation trial |

All six attempts start from newly initialized parameters, a fresh optimizer
and empty source history. Before learning, preserve each seed's initialized
checkpoint and parameter hash. Pooled and narrow initial parameters must match
for that seed, so one shared untrained control can represent both arms. This
identity is checked, not assumed from a seed label. No checkpoint, source
optimizer, rollout data or history from the prior pilot crosses this boundary.

Each attempt uses its final checkpoint only. Preserve actual update counts and
finite-parameter checks. No favorable intermediate checkpoint, replacement
seed, outcome-dependent curriculum, hyperparameter adjustment or resumed
training is allowed. Exact source budgets are divisible by rollout size; a
library overshoot cannot be relabeled as exact completion.

Collect sources in seed order 101, 202, 303, pooled then narrow within each
seed. Attempt all six once even if an earlier attempt is operationally
incomplete; retain each failure without retry. Any incomplete source prevents
all evaluations and reserved calls. A cap preserves the incomplete attempt;
it does not permit another seed, larger cap or a resumed optimizer. Report
initialization, learning and total source wall times, with final archival and
integrity replay time separate from the source learning cap.

Source safety termination may reset the affected episode within the aggregate
decision budget, and every stopped prefix/reset is retained. Time-limit
bootstrap uses the actual terminal observation, not a reset world's first
observation. History clears only at a real reset, not at PPO updates. Equal
macrodecision budgets do not promise identical physics exposure where native
safety shortens a macrodecision; report that difference explicitly.

## Six-checkpoint seal and known-domain competence

All six source checkpoints and their six initial parameter identities must be
sealed with source, protocol, runtime and checkpoint hashes before any learned
or quiet evaluation begins. Evaluation results cannot select which checkpoint
is released. Source failure cannot be repaired by adding decisions or reusing
the incomplete optimizer.

On each of the known twelve cases, run pooled, narrow and shared saved initial
weights for all three seeds, plus one shared deterministic quiet run. This is
120 trials: 108 neural runs and twelve quiet runs. On each of narrow's own four
cases, separately run narrow and matched initial weights for all three seeds,
plus shared quiet: 28 trials. These domains are not combined into one
source-competence average. A shared quiet return is one deterministic baseline,
not three independent replications. Known collection order is all twelve quiet cases
first, then seed, case and arms pooled/narrow/untrained. Narrow own-domain order
is its four quiet cases first, then seed, case and narrow/untrained.

Every run gets at most 640 decisions from native wake-up. Reset is
`rebuild -> set_sun -> reset`, with the hanging pose, zero velocities, fresh
motor state and acquisition queues. The local stack starts with eleven absent
zero frames and the current reset frame under validity masks. No free warmup,
probing, rescue reset or history from another run is permitted. Neural inference
samples categorically (`deterministic=False`) with action seed
`100000 + training_seed*10 + case_index` on the known twelve, adding 100 for
narrow's own four and 200 for reserved cases. Within a domain, that seed is
paired across all neural arms and set after fresh model construction and weight
loading. Quiet uses no RNG. Only policy weights are loaded into a fresh
model/optimizer. Hashes before and after each run must match its sealed
checkpoint; no evaluation gradient update
occurs. The manual finite loop prevents automatic reset after the final
budgeted step. A native termination, veto, feedback flag or controller exception
ends the run, with its realized prefix preserved.

Before any reserved native call, all of these release requirements must pass:

1. All six source attempts are complete at their exact declared decision/update
   budgets, with finite weights, valid provenance and the required sealed hashes.
2. For pooled, each of all twelve known cases and all three seeds has a complete,
   safe trained/quiet pair and integrated native reward gain strictly >+3.2.
3. For narrow, each of its own four cases and all three seeds has a complete,
   safe trained/quiet pair and integrated native reward gain strictly >+3.2.
4. Required frozen-weight identities and exact evidence accounting remain intact.

Narrow's performance on pooled's known twelve is descriptive and is not a
release gate. The gate checks that both families learned competent source
policies before attributing a challenge difference to source breadth. Pooled
source competence alone cannot compensate for a failed narrow own-domain gate.
The >+3.2 margin retains the lower-lamp engineering threshold; it is not a
statistical significance test or a first-assembly product criterion.

If a requirement fails, retain every collected failure and write an explicit
no-release record. There must be no reserved configure/reset/step call and no
reserved feasibility witness. A later revised experiment requires a new
prospective lock and exposure record; failed seeds or cases cannot be dropped.

## Conditional reserved first-use comparison

After the complete release receipt and all checkpoint/evidence hashes are
frozen, run the reserved feasibility suite described below, including twelve
quiet runs. Then, in seed, case and pooled/narrow/untrained order, evaluate
the 108 neural trials: three seeds × twelve cases × three arms. Their shared
quiet baselines come from the already retained suite and are not rerun. The
neural-plus-quiet comparison contains 120 trials; the whole challenge contains
168 trials, including the other 48 fixed witnesses. Each gets at most 640
decisions / 64 seconds, under the same native boundary, from-wake accounting
and no rescue. All cases and seeds remain in the results, including stopped
or failed runs.

The primary breadth quantity for seed s is
`D_s = mean_over_all_12_cases(R_pooled,s - R_narrow,s)` at 640 decisions.
Report all three D_s values, their median and the number strictly positive.
The prospective breadth advantage criterion requires all 72 pooled/narrow runs
and the twelve shared quiet runs to safely complete with unchanged policy hashes,
a median D_s >0 and at least two of three D_s >0. Ties are not positive.
A favorable mean of stopped prefixes or a selected safe subset cannot
pass this criterion. Report raw failed-prefix differences separately, without
claiming matched-budget improvement. The 36 initialized controls are retained
with separate safety and learning-benefit eligibility. An initialized-policy
safety stop alone does not invalidate a complete pooled-versus-narrow contrast.

Useful reserved behavior is a separate criterion: all 36 pooled runs and twelve
quiet runs must safely complete, and every pooled case/seed must exceed its
shared quiet return by strictly >+3.2.
A breadth advantage may coexist with poor absolute behavior, and useful pooled
behavior may coexist with no advantage over narrow pretraining. Report both
outcomes; neither substitutes for the other.

For each curriculum, also report trained-minus-saved-initial reward per case
and seed. The primary descriptive mean includes all realized returns and marks
stopped prefixes as unequal exposure, alongside a separate complete-safe-pair
mean and eligibility count. Neither an eligible survivor subset nor stopped
prefixes establishes the whole matched-budget learning benefit. There is no
extra threshold or source-release gate on this diagnostic.

Cumulative windows are fixed at 64, 128, 320 and 640 decisions, or 6.4, 12.8,
32 and 64 seconds. The whole 64-second run is primary. Early returns, lamp side,
motor location and body layout are declared descriptive views; they cannot
select checkpoints or replace the whole-run criterion. Seed-level paired
reductions describe these three runs, not an established population guarantee.

## Fixed reserved feasibility suite

Only after release, execute five fixed controllers on every reserved case:
quiet, constant negative, constant positive, negative-first 2-second rhythm and
negative-first 4-second rhythm. Collect in case order, with those five modes in
the listed order within each case. Every run has the same 640-decision budget,
native request magnitudes and local governor as the policy trials. Quiet requests
zero, constants request -0.15 or +0.15, and rhythms alternate those signs every
10 or 20 macrodecisions respectively, starting negative at wake-up. Their full
periods are 2 or 4 seconds, with sign phases of 1 or 2 seconds. There is
no free sensor acquisition, position search or warmup. These sixty runs precede
the 108 neural trials, after all source and release hashes are frozen.
Initial invalid native sensor samples cause the unchanged governor to emit zero;
those controls remain inside each controller's declared from-wake budget.

The independent witness outcome requires every reserved case to have a
complete-safe quiet run and at least one complete-safe nonquiet fixed controller
whose raw integrated reward exceeds quiet by strictly >+3.2. It is a feasibility
diagnostic, separate from source competence, breadth advantage and useful pooled
first-use behavior. Failure of this fixed suite does not prove infeasibility;
the declared schedules need not find every attainable behavior.

Witness outcomes do not gate or change neural execution, select a policy,
change a parameter, adjust a lamp, exclude an unsuccessful case or license a
different controller. Collect all 108 declared neural runs even if witnesses
fail. A retrospective choice of which fixed schedule worked is reported as a
witness comparison, not supplied to any neural policy. Do not add the earlier
continuous-angle search/hold servo to this suite after observing outcomes.

## Aggregate experience accounting

| Stage | Opportunities | Maximum macrodecisions |
| --- | ---: | ---: |
| Offline pretraining | Six independent source attempts | 589,824 |
| Known and narrow own-domain evaluation | 120 + 28 trials | 94,720 |
| Conditional reserved witnesses and neural evaluation | 60 + 108 trials | 107,520 |
| Whole study, if released | Six source attempts + 316 evaluation trials | 792,064 |

The maximum is **3,960,320 native controls**, **39,603,200 physics steps**, or
**79,206.4 simulated seconds**. Quiet reused as the reserved baseline is counted
once. Stops and blocked stages reduce realized exposure; never describe planned
budget as actual collection. Retain decisions, controls, duration and resets
by source, case, seed, arm and stage, including failed prefixes and zero-exposure
blocked stages. Integrity replay is additional repeated simulation/computation,
not another independent training or evaluation sample. CPU pauses simulated
time; these timings do not establish a child's wait or physical device latency.

## Evidence and execution boundaries

The study uses the existing installed Linux CPU dependencies and direct native
executable in the managed `droid-blocks-dev` container. No dependency changes,
simulator migration, live API control or live workshop controller replacement
are included. Host PowerShell is limited to orchestration and file inspection;
all training, builds, tests and replay run in the managed Linux container.

The prospective lock archives this document, the canonical study manifest,
action-generating runner/adapter/native sources and runtime receipts before
learning. Source and stage seals bind the native binary, initial/final
checkpoint bytes, parameter hashes, request tapes, telemetry, actual exposures,
source schedules and immutable container identity. Stage transitions must
validate these hashes and refuse existing output or a resumed attempt.

Source attempts are retained under `sources/source-{arm}-{seed}/result.json`,
with the six-source release at `sources/sealed-checkpoints.json` and exclusive stage
receipts. Development and challenge have their own trial directories containing
case/controller/seed-or-fixed identities, results and tape hashes. Their
`stage.json` gate records bind the previous stage and its evidence, including
blocked-stage receipts. Archive the new manifest checks in
`tests/check_generalization_study_manifest.py` and the driver checks with the
collection source snapshot.

An independent post-collection auditor has its own exact source archive and
hash in verification receipts, outside the action-generating source lock. It
recomputes rewards/windows, budget counts, source membership, pre-release
reserve exclusion, checkpoint bindings and every gate, and replays retained
native requests in fresh isolated worlds. Native action replay verifies
physics/accounting rather than rerunning PPO optimization or proving neural
action sampling. Any neural-policy replay or independent tensor load check
must state its separate scope.

Retain failed and partial source/evaluation records, checkpoint bytes and
verification receipts. Bulk transition tapes, physical snapshots and playback
streams stay local and ignored according to repository hygiene; commit source,
protocol, compact summaries and provenance, with the exact local assets a
fresh checkout needs for replay documented. Do not recursively duplicate old
bundles for QA or delete retained evidence as scratch. Earlier pilot and
headroom bytes, original rover-v3 artifacts and live sessions stay preserved.

## Ordered Linux execution

The new driver is `training/generalization_study.py`. From `/workspace` inside
the existing managed container, use the installed runtime for each exclusive
stage against the same new report root:

```sh
/opt/droid-blocks/generalization-venv/bin/python training/generalization_study.py \
  lock --output artifacts/reports/generalization-study-v1-20261006
/opt/droid-blocks/generalization-venv/bin/python training/generalization_study.py \
  sources --output artifacts/reports/generalization-study-v1-20261006
/opt/droid-blocks/generalization-venv/bin/python training/generalization_study.py \
  development --output artifacts/reports/generalization-study-v1-20261006
/opt/droid-blocks/generalization-venv/bin/python training/generalization_study.py \
  challenge --output artifacts/reports/generalization-study-v1-20261006
/opt/droid-blocks/generalization-venv/bin/python training/generalization_study.py \
  summary --output artifacts/reports/generalization-study-v1-20261006
```

Validate the runner, manifest and stage guards against this document before
installing the lock. Stages refuse duplicate collection, overwrite and resume;
an existing incomplete stage stays preserved. `development` cannot instantiate
worlds without the six-source seal. `challenge` cannot instantiate reserved
worlds without the complete prospective release gate and evidence hashes; when
blocked, it records that condition with zero reserved native calls. No retries
or reserved-only convenience command are part of this version. The summary
keeps operational completion, source competence, witness feasibility, breadth
advantage and useful pooled behavior distinct.
