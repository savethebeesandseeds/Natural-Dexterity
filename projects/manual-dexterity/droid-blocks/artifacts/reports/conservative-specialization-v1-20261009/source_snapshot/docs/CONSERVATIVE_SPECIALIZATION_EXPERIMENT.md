# Longer budget and conservative specialization experiment

Declared 9 October 2026, before native collection. This successor tests whether
more assembly experience and smaller PPO updates can add useful, reliable
benefit to the inherited pretrained brain. The [October 7 study](SPECIALIZATION_RESULTS.md)
found strong benefits over scratch but missed the practical benefit gate over
frozen reuse, with substantial individual regressions. That result is retained.

This study uses the same twelve exposed length-5 conditions, three layouts,
six mechanics and two lamp positions, and the three sealed pooled releases
(seeds 101, 202 and 303). No new body generalization is claimed. The original
rover-v3 challenge remains unexposed. The visible workshop continues to use
its immutable pretrained actor; no outcome automatically promotes a checkpoint.

## Fixed comparison

Each arm receives one 640-decision opportunity in a fresh native world with
fresh sensor history. One decision covers 0.1 simulated seconds. The three
learning arms use the first 384 decisions (38.4 simulated seconds), then act
for 256 decisions with fixed weights, history and action RNG continuing in
the same world. The extra learning is charged inside the original total budget.
The continuation starts in states produced by each arm's earlier behavior;
its return does not isolate a weight effect at an identical physical state.

| Arm | Initialization | Learning rate | PPO clipping | Role |
| --- | --- | ---: | ---: | --- |
| frozen | Sealed pooled actor and critic | No updates | No updates | Benchmark |
| conservative | Same pooled tensors | 0.0001 | 0.1 | Primary method |
| standard | Same pooled tensors | 0.0003 | 0.2 | Secondary longer-budget method |
| scratch | Exact saved pre-source initial tensors | 0.0003 | 0.2 | Matched untrained comparator |

All learning arms train the full twelve tensors, 34,820 actor and critic
parameters. Each has a fresh Adam optimizer, rollout buffer and counters;
source optimizer state is never restored. Six 64-decision rollouts receive four
single-batch epochs each, for 24 Adam steps. All other PPO, architecture,
observation, reward, governor, native physics and safety settings match the
earlier study. Constant per-arm schedules are set before model construction.
There is no KL early stop, reward change, body metadata input or hidden-state
input. Checkpoints are recorded after 64, 128, 192, 256, 320 and 384 decisions.

Lower learning rate and tighter clipping are a combined experimental package;
this comparison cannot attribute an effect to either setting separately.
PPO clipping limits the surrogate objective but is not itself a hard bound on
policy change or safety. The [SB3 PPO documentation](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/ppo.html)
describes these controls and that limitation. The installed implementation is
pinned SB3 2.9.0, whose exact relevant source hashes are bound by this protocol.

## Continuity and repeatability

No reset, environment replacement, rescue, second learning call or action RNG
reseed is allowed at the 384-decision boundary. Horizon 641 prevents an automatic
reset at decision 640. Native stops and unresolved requests terminate the attempt;
the raw prefix is retained without retry. Each opportunity has a 180-second
wall cap. All updates must leave Torch action RNG unchanged.

Target action seed is 810000 + source seed times 1000 + ordered case index.
Guard seed is 910000 + source seed times 1000 + ordered guard index. These
match October 7 to permit controlled historical comparisons. The repeated
frozen opportunity and first 128 standard decisions, native records and policy/
Adam tensors are checked against the bound earlier trials. Such repeats verify
implementation consistency on known cases; they are not independent new samples.
The standard longer-budget contrast is descriptive evidence on the same exposed
conditions, with additional updates after the previously tested prefix.

## Benefit and reliability gates

The primary comparison is conservative minus frozen integrated native return
over all 640 decisions, including learning cost. The standard method is secondary;
neither replaces the other after seeing results. Each must be reported separately.

Useful benefit requires all 36 corresponding pairs safe and complete, the
median of three seed mean gains strictly greater than +3.2, and at least two
positive seed means. The three source seeds are training replicates; the twelve
case outcomes per seed are correlated. Scratch contrasts use the separate
median >0 threshold and do not establish benefit over frozen reuse.

Target reliability additionally requires every corresponding whole-opportunity
gain to be at least −3.2. This is a bound on the tested cases, not future-body or
hardware safety. Learning-prefix loss below −1.92 is a separate diagnostic,
scaled from the earlier study's prefix floor; it does not replace the whole gate.

## Two fixed retention guards

After all 144 target attempts and their evidence are sealed, collect six original
frozen baselines, then 144 frozen probes of the 72 conservative/standard
checkpoints after 384 decisions. The fixed guards remain
`development-long-upper-base-left` and `development-weighted-elbow-right`.
Each receives a fresh 640-decision world/history with the matched seed.

Sampled retention requires all corresponding probes safe and complete and
each seed mean post-minus-base return at least −3.2. Probe reliability additionally
requires every one of the 72 individual deltas per method to be at least −3.2.
Missing final-prefix checkpoints produce blocked zero-exposure receipts.
Unfinished or changed target seals block guard release. There are no retries.

Combined readiness for each method requires useful benefit, target reliability,
sampled retention and probe reliability together. Integrity passing is separate
from these scientific gates. The two guards cannot establish whole-domain
retention or personal knowledge surviving a rebuild.

## Exposure and evidence

The maximum is 294 opportunities: 144 targets and 150 guards, 188,160 decisions,
940,800 native controls, 9,408,000 physics steps and 18,816 aggregate simulated
seconds. The 41,472 learning decisions are already included in target exposure.
Record actual exposure and all blocked/failed opportunities. Validation replay
does not count as another sample. Linux CPU wall times do not establish child
waiting time or physical responsiveness.

Before collection, synthetic tests pass without native worlds, all prior
artifacts and the live actor receive a byte-level preservation baseline, and
the declaration, input releases, runtime, native binary and action-generating
sources are locked. No action-generating source changes after locking.

Retain all request/telemetry recordings, fresh initial and six prefix checkpoint
ZIPs, source snapshots, failures, manifests and independent receipts. New bulk
tapes and ZIPs remain local and ignored. A full batch needs 294 request tapes,
294 telemetry tapes, 294 fresh initial ZIPs and 648 prefix ZIPs, plus the original
releases, exact native runtime and archived auditor/helper sources for replay.
These are the full-completion counts; stopped or blocked attempts retain their
actual prefixes and receipts. Compact reports and provenance records describe
the required local assets.

## Container execution

Use the existing inspected `droid-blocks-dev` and its pinned
`/opt/droid-blocks/generalization-venv/bin/python`; no new container, host
toolchain or dependency installation is required. Run Python with `-B`.
The exclusive sequence is preservation baseline and synthetic preflight,
`conservative_specialization_manifest.py lock`,
`conservative_specialization_study.py targets`, then `guards`, then independent
checkpoint/action/native audit, final preservation and reporting.

The new batch is `artifacts/reports/conservative-specialization-v1-20261009`.
Never overwrite it, repeat failed trials, tune settings after outcomes or change
the declared thresholds. If the comparison fails, retain the negative result
and use it to choose a separately declared next question. Reliable incremental
specialization still precedes fresh body-family evaluation, persistent personal
memory, richer modular mechanics and hardware.
