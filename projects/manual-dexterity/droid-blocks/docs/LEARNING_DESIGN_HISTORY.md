# Learning design history before held-out evaluation

This note records the evidence that changed the light-search controller and
optimizer before any test or held-out rollout. It is deliberately a design
history, not an acceptance report. The train, validation, and test roles are
those locked in the [learning experiment protocol](LEARNING_EXPERIMENT.md):
training fits candidates, validation selects a frozen checkpoint, and the
64-scenario test split is reserved for one final held-out decision.

**The test/held-out split was untouched throughout every stage below.** The
artifacts contain the preregistered test count and digest as provenance, but
deriving or storing that digest is not an environment rollout and produced no
test outcome.

## Evidence ledger

| Stage | Evidence actually observed | Result | Test/held-out use |
| --- | --- | --- | --- |
| v1 exploratory fit | Train for fitting; 16 validation scenarios for checkpoint selection and diagnosis | One-sided validation failure | None |
| v2 robust pilot | Train for fitting; 16 validation scenarios for checkpoint selection and diagnosis | Generation-zero/zero controller retained; all validation comparisons tied | None |
| v2 capacity probe | Only the 32 train scenarios, plus a stated representative train pair | A fixed three-weight controller solved both sides; recurrence ablation failed one side | None |
| v3 reduced-budget pilot | Train for fitting; 16 validation scenarios for checkpoint selection | Masked best-sample run selected generation 1; no validation metric is preserved in the artifact | None |

The three exploratory policy files are immutable evidence under
[`artifacts/`](../artifacts/README.md). Their `exploratory-` prefix explicitly
marks them as ineligible for official held-out evaluation.

## v1: validation exposed a one-sided controller

The v1 exploratory artifact is
[`exploratory-light-search-g12-p16-seed0.json`](../artifacts/exploratory-light-search-g12-p16-seed0.json),
with canonical artifact checksum
`2e8ab4b996532317a99140c3c4d4a16b4fb29f149f04590fa12def0fc3cf8802`.
It used `shared_linear_memory_v1`, 31 shared parameters, 12 generations, a
population of 16, and selected generation 11.

The diagnostic numbers below are **validation**, not train and not test:

- negative-x validation: mean reward-rate gain versus zero
  `+0.1136583`, with `8/8` wins;
- positive-x validation: mean reward-rate gain versus zero
  `-0.0728613`, with `0/8` wins.

The aggregate improvement therefore hid a directional failure. A controller
could acquire and exploit a constant drive preference, but the v1 feature set
did not provide a clean shared parameter that estimated which direction made
light improve. Freezing or evaluating this artifact on the held-out split
would have converted a known design defect into test-set consumption, so no
test rollout was run.

This observation motivated the executable v2 feature contract now declared in
[`policy.hpp`](../include/droid/policy.hpp) and implemented in
[`policy.cpp`](../src/policy.cpp). In addition to scaling the one-step ambient
change by `8192`, v2 adds:

- feature 31: scaled ambient-light delta times previous local effort; and
- feature 32: scaled ambient-light delta times local shaft velocity.

Both are derived only from policy-visible sensor and actuator-feedback data.
They do not expose the light side, world pose, seed, reward, or simulator state.

## v2 pilot: representation improved, isotropic search did not reach it

The v2 robust pilot artifact is
[`exploratory-v2-light-search-g12-p16-seed0.json`](../artifacts/exploratory-v2-light-search-g12-p16-seed0.json),
with checksum
`68d9843691adb7ac85e417402cc13b5c034c5876acc958a28d7a410b82556385`.
It used the 33-parameter `shared_linear_memory_v2` architecture but retained
the same small exploratory CEM shape: 12 generations, population 16, and
isotropic initial standard deviation `0.35` from a zero mean.

The selected checkpoint remained generation 0, whose 33 parameters are all
zero. Consequently, its 16 **validation** comparisons against zero were 16
exact ties. This is evidence that the pilot optimizer did not find and retain
a robust controller; it is not evidence that v2 lacks representational
capacity. No test scenario was run.

## v2 train-only capacity probe

The focused native diagnostic
[`train_seed_v2_probe.cpp`](../tools/train_seed_v2_probe.cpp) supplied actions
through `SharedLinearMemoryPolicy` using only the public observation. Its
trajectory inspection was diagnostic-only and never became a policy input.
It reconstructed the fixed split but passed only the train member to MuJoCo;
its outputs explicitly reported zero validation and zero test scenarios
executed.

All parameters were zero except these four named positions:

| Parameter | Feature | Fixed weight |
| ---: | --- | ---: |
| 29 | previous local effort | `2.0` |
| 30 | bias | `0.03` |
| 31 | scaled ambient delta × previous effort | `0.0` |
| 32 | scaled ambient delta × local velocity | `64.0` |

On the first negative/positive pair in the **train split**:

| Train seed | Side | Gain vs zero (reward/s) | End displacement | Direction changes |
| ---: | --- | ---: | ---: | ---: |
| `6106049681611768608` | negative x | `+0.07911691839992885` | `-9.629365 m` | 1 |
| `6990021049862345368` | positive x | `+0.08981056190010117` | `+9.730605 m` | 0 |

The positive-side episode kept its initial positive probe. The negative-side
episode first probed positive x, detected worsening light through the
delta–velocity interaction, reversed once by about control step 20, and then
continued toward negative x. Neither episode terminated or produced a safety
veto.

The same fixed vector was then evaluated once on all 32 **train** scenarios:

| Train stratum | Mean gain vs zero (reward/s) | Minimum | Maximum | Wins | Mean toward-light travel | Minimum travel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Negative x (16) | `+0.0920287739625194` | `+0.07408417480007123` | `+0.11177698750009757` | `16/16` | `9.6359435625 m` | `9.628702 m` |
| Positive x (16) | `+0.09078761130628811` | `+0.07888949700008507` | `+0.11490585210005799` | `16/16` | `9.72941275 m` | `9.727397 m` |

The fixed vector achieved `32/32` train wins, balanced mean gain
`+0.09140819263440375 reward/s`, worst-side mean gain
`+0.09078761130628811 reward/s`, mean absolute action
`0.3110003405142064`, zero safety vetoes, and zero terminations.

This is a capacity result on training worlds only. It must not be described as
generalization or acceptance.

### Recurrence ablation

The only changed value in the ablation was parameter 29, from `2.0` to `0.0`.
On the same representative **train** pair:

- negative x: gain `-0.0014004228000699115 reward/s`; the rover ended
  `+0.200671 m` in the wrong direction;
- positive x: gain `+0.08226323630010024 reward/s`; the rover ended
  `+9.348519 m` toward the light;
- balanced pair gain: `+0.040431406750015164 reward/s`;
- zero safety vetoes and zero terminations.

Feature 32 alone therefore amplified the favorable initial probe but could not
sustain enough action–velocity signal to reverse the unfavorable probe.
Previous-effort recurrence is a necessary third active coordinate for this
simple two-sided search mechanism. Feature 31 was not needed by the capacity
vector and can oppose the desired correction while inertia temporarily makes
command and velocity disagree.

## Consequence: v3 optimizer requirements

The train-only evidence isolates a compact mechanism with three active
coordinates: previous-effort recurrence (29), startup bias (30), and the
delta–velocity directional estimate (32). It also explains why the v2 pilot's
33-dimensional isotropic search was poorly scaled:

- the useful feature-32 weight was about `64`, or roughly 183 times the pilot's
  initial standard deviation of `0.35` from zero;
- the useful bias was only `0.03`, so one global scale cannot resolve it while
  also reaching feature 32;
- perturbing the other 30 parameters adds variance and unsafe or misleading
  behavior without being required by the demonstrated mechanism.

The registered v3 optimizer therefore uses a **scaled three-dimensional masked
search** over parameters 29, 30, and 32. Their physical scales are respectively
`2.0`, `0.25`, and `64.0`; every non-mask parameter remains the positive-zero
bit pattern. The latent mean starts at `0.0`, with standard deviation `1.0` and
floor `0.1`. The deterministic sampler keys each draw by the physical parameter
index rather than its latent position. The mask, scales, latent defaults, and
keying rule are versioned protocol inputs so execution cannot silently revert
to a 33-dimensional isotropic search. The learner's split and safety-first,
paired-zero side-robust score contracts are defined in
[`learner.hpp`](../include/droid/learner.hpp) and implemented in
[`learner.cpp`](../src/learner.cpp); v3 must preserve the policy-safe boundary
and continue using train data for fitting and validation data only for model
selection.

V3 also uses a **best-sample checkpoint**. In the v2 CEM path, a rare sampled
controller can enter the useful nonlinear feedback regime but then be diluted
when elite parameters are averaged; validating only the updated population
mean can discard that controller. In each v3 generation, the best sample is
selected solely by its train-batch score and copied before the latent
distribution update. That exact sampled controller is then evaluated on the
full validation split. The updated distribution mean is recorded for diagnosis
but is not a validation candidate. Final selection uses only the locked
validation rule, preserves deterministic tie-breaking and full provenance, and
never consults test outcomes.

## v3 reduced-budget masked-search pilot

The reduced-budget v3 artifact is
[`exploratory-v3-light-search-g12-p16-seed0.json`](../artifacts/exploratory-v3-light-search-g12-p16-seed0.json).
Its canonical artifact checksum is
`cedd7381fc6efa5ee31bd06dbaee942b8e2fd674b38a1187f4c505510b7ed625`,
and the SHA-256 of its exact serialized file bytes is
`38d0d788e29b573bc49d517195d0d5ddeafdb3fb9a50dfef933a39258a651677`.
It used `cem_masked_best_sample_v1` with 12 generations, population 16, elite
count 4, train batch size 8, initial latent standard deviation `1.0`, floor
`0.1`, update rate `0.25`, and optimizer seed 0. It selected generation 1.
Only physical parameters 29, 30, and 32 are nonzero; every inactive parameter
has the exact positive-zero binary64 representation.

Its effective-manifest digest,
`8fc97e67585c0e5c8092215de48ead712bd815e37bbba08eb35fa67df37a2c03`,
differs from the official v3 digest because of the reduced generation,
population, and elite counts. The official held-out gate therefore rejects
this artifact. The artifact preserves its selected generation and protocol
provenance, but not the command's full validation metrics, so this history
makes no quantitative validation claim for the pilot. No test or held-out
rollout was executed for it.

These changes are justified by train-only capacity and pre-test validation
failures. They do not authorize any feedback from the pending v3 held-out split
into the v3 artifact, acceptance rule, or claim. At the end of this history,
the official 64-scenario test/held-out split remains completely untouched.

After the v3 decision is fixed, its evaluated worlds, topologies, traces, and
outcomes may be relabeled as known curriculum or replay data for a future
version. A descendant that uses them cannot count them as unseen evidence and
must reserve a fresh, preregistered, versioned challenge split outside its own
development loop.
