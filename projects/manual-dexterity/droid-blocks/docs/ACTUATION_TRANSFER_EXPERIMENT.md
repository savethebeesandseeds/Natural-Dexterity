# Actuation-layout transfer: locked exploratory experiment

Status: LOCKED before native learning, 2026-09-07, for artifacts/reports/actuation-transfer-v1-development-20260907. The owner adopted the goal of testing whether pretrained neural control helps a rebuilt body with little new experience. This is a bounded exploratory comparison. The implementation and non-learning native preflights passed, but the separate CartPole learning reference missed its predeclared improvement gate; that failure is retained below. No native learning or target outcomes informed this protocol. Archive this document with the machine-readable manifest and source hashes before the first native learning run. Outcomes belong in a separate results document; do not amend this locked protocol to accommodate them.

## Question and scope

Does a PPO policy pretrained with the motor on the anchored base hinge provide useful initial behavior or faster reward-driven adaptation when that same motor powers the elbow instead, under an equal 64-second budget of new simulated interaction?

The mechanism remains an anchored, collisionless, planar two-link chain with one motor, one passive hinge, one distal IMU and one distal light sensor. Switching hinge 0 to hinge 1 changes the actuation map and the meaning of the local motor encoder. It does not change the connection graph, introduce another actuator, demonstrate locomotion, or establish T-Rex-to-snake transfer. Previous mass/length experiments alone did not test this actuation change.

The existing Expected SARSA learner, affine predictor, historical protocols and evidence remain separate and preserved. This goal introduces an established neural PPO implementation; it does not use the affine model to choose actions, commit to a graph architecture, or claim that a frozen embedding plus final-layer updates is sufficient. The first comparison transfers and, where specified, updates the complete actor and critic.

## Native plant and observation boundary

Use the existing ConstructionWorld motor model, native sensor reward, sensor acquisition queues, .002-second physics and .02-second control steps. The powered_hinge assembly field selects the active hinge. Moving actuation must move the actuator mapping, reflected rotor armature, current/torque/friction model and local encoder/speed feedback together. The other hinge receives the existing passive armature and damping. Base-layout assemblies must reproduce their prior behavior. Geometry is available to the simulator and evidence renderer only.

Actor and critic receive exactly the same permitted local information. They receive neither source coordinates, joint-layout/assembly labels, link lengths/masses, world pose, passive-joint state, experiment identity, seed, absolute elapsed time nor termination countdown. There is no privileged critic. IDs, sequence counters and timestamps are validation metadata, not neural inputs. Reward comes from the native transition, never from a hidden target-angle or motor-sign rule.

The interface is a Discrete(3) requested action: -0.15, 0 or +0.15. Each request lasts .1 second, comprising five native controls. The unchanged local governor runs at every .02-second endpoint, using the latest measured powered-shaft velocity and previous emitted effort: magnitude limit .35, target speed 3 rad/s, speed gain .05, slew limit .1 per control. Safety remains authoritative and separate. The target speed is soft, not a bound on gravity-driven speed. Zero requests zero current; residual current and shaft friction remain.

Record sampled action, requested effort and all five emitted efforts separately. PPO probabilities and rollout actions refer to the sampled discrete request, not a substituted governed effort. Native initial undelivered samples command zero until valid. Any additional forced-quiet interval is inside the declared interaction budget and must be recorded; resetting the environment must not conceal uncounted physical warmup.

## Fixed causal representation

Use 12 frames at .1-second decision endpoints, flattened oldest to newest: 1.1 seconds between the oldest and newest available frame. Each frame has 17 scalars, yielding 204 inputs:

1. Previous emitted effort / .15.
2. Sine and cosine of the local motor angle (two scalars).
3. Local motor velocity / 10 and current / .5 (two scalars).
4. Three local IMU specific-force components / 50.
5. Three local IMU angular-velocity components / 10.
6. Received light / 1000.
7. Previous requested effort / .15.
8. Previous macro-transition reward / (2 * .1).
9. IMU-valid, light-valid and frame-present masks (three scalars).

Clip continuous normalized values to [-5,5] using fixed scaling. Masks are binary. No fitted observation or reward normalizer is used. Missing history is zero with frame-present=0; a real reset observation is present with invalid sensor masks. Invalid initial sensor values are zero under their masks. Freshness, sample holding, monotonic delivery, faults and non-finite values are validated before projection; stale or malformed data is not silently converted to missing data.

The first two physical feature groups are deliberately unchanged in form when the motor moves: they report the powered hinge, not a relabeled base angle. The IMU remains on the distal link. Its gyro combines joint motions; the active encoder supplies only one joint coordinate. Specific force mixes gravity and acceleration, and a single light reading can match multiple configurations. History can reduce this ambiguity but is not a demonstrated state observer or Markov state. A failure may reveal insufficient observability or control authority rather than a defect in neural optimization.

## Reward, discount and episode boundary

The PPO reward for one .1-second request is the exact sum of the five raw native transition rewards. The native calculation integrates the sum of IMU and light votes at physics substeps using the then-delivered samples. No energy penalty, artificial terminal cost, reward normalization or light-difference shaping is introduced. Preserve native components and each subtransition for independent checking. The masks and history preserve sample holding: light is sampled every .1 second with .02-second latency; IMU every .01 second with .006-second latency.

Use gamma = .92^(.1/.4), approximately .97937, to preserve the earlier learner's physical discount timescale. This is coarse-step discounting of the exact undiscounted native macro reward; it is not bit-for-bit the former micro-discounted option-return formula. GAE lambda=.95 is a separate estimator setting, not an additional physical-state assumption.

A native safety failure is a genuine terminal event. A normal 64-second limit is a truncation with the actual terminal observation available for correct PPO bootstrapping; never bootstrap from an automatically reset body's observation. Training may reset after a terminal event and must count every transition/reset. Target evaluation gets no rescue reset or extra physical interaction after failure.

Report raw realized native reward and actual exposure even when a target fails early. Success requires zero safety failures, so a short positive prefix cannot win by avoiding the remaining task. An optional conservative horizon-completed score adds -2 per unobserved remaining second after a failure, the lower bound from two sensor votes. Label it an evaluator-defined diagnostic, never native reward or a PPO objective. Numerical failures preserve evidence and count as failures, not discarded seeds.

## Bodies, conditions and exposure

All cases use the existing block masses and dimensions. The IMU is at segment 1, slot 2, side -1. The light is at segment 1, slot 2, side +1, face +1. The optional added block occupies segment 1, slot 0, side -1. Use an opaque stable part ID; names and IDs never become predictive features.

The source training sampler independently and uniformly selects each link length from {3,4}, optional added block from {absent,present}, and sun x from {-.30,+.30} metres. The sun always has y=0, z=.20 m and intensity=1000 lux. Every source episode has motor hinge 0. Save the sampled reset-task schedule, seed derivation and deterministic RNG implementation; compact source evidence must permit reconstruction of the exact task/action sequence. This is a finite 16-condition source domain, not arbitrary-body pretraining.

The six target cases below all have motor hinge 1. Their geometries and illumination conditions are inside the source domain; their actuation layout is withheld from all source pretraining. This isolates the changed actuation map rather than combining unseen geometry, a new objective and new actuation at once.

| Target body | Link lengths | Added block | Target conditions |
| --- | --- | --- | --- |
| long-upper-left / long-upper-right | [4,3] | absent | sun x=-.30 and +.30 |
| long-lower-left / long-lower-right | [3,4] | absent | sun x=-.30 and +.30 |
| weighted-left / weighted-right | [4,4] | present | sun x=-.30 and +.30 |

For each target task, use rebuild -> set_sun -> reset. This matters because a source change preserves already-acquired light readings; set_sun alone after rebuild can leave an initial sample from the wrong source. Reset pose is the authored hanging pose with zero joint velocities and fresh motor/queue state. All comparisons share it. Source position remains constant during each episode; source-switch adaptation is outside this question.

The dedicated elbow development reference uses [3,3], no added block, both source positions and hinge 1. Its exact body is outside the target table. Exposure to this development layout is explicit. Its trajectories, weights, optimizer and observations cannot enter source training or any target branch. It can demonstrate achievable neural behavior on that development body; it is not an optimum or proof that every target body is feasible.

## Established optimizer and exact budgets

Root selected Stable-Baselines3 PPO on host CPU in a project-local isolated Python environment. The native C++ world remains in the existing container, connected through one batched stdin bridge; training does not use the live browser session or reset its creature. Exact Python/package versions, installation hashes and runtime settings accompany the machine-readable lock and source archive. The library reference and bridge/safety checks were run before native training; their distinct outcomes are recorded in the preflight section below. SB3's [PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) defines rollout size as n_steps times n_envs; [vector-environment documentation](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html) describes terminal-observation handling. The pinned installed version, not a moving documentation URL, is the executable specification.

Fixed network: MlpPolicy, two 64-unit tanh layers for each of actor and critic, no recurrent hidden state, dropout or learned input normalizer. Fixed settings: learning rate 3e-4, entropy coefficient .01, GAE lambda .95, policy clip .2, value coefficient .5, max gradient norm .5, four optimization epochs and minibatch 64. Use the pinned library's advantage normalization, Adam epsilon 1e-5, orthogonal initialization, no value clipping and no target-KL early-stop threshold. Record the remaining effective settings with the model. No hyperparameter sweep or reward-based checkpoint selection is authorized by this protocol.

| Stage | Seeds / environments | Rollout and cap | Exposure meaning |
| --- | --- | --- | --- |
| Elbow development learning | seed 404, eight environments | 256 per environment; 16 rollouts = 32,768 decisions | Explicitly elbow-exposed reference, isolated from source learning |
| Source pretraining | seeds 101, 202, 303, eight environments per seed | 256 per environment; 48 rollouts = 98,304 decisions per seed | Base hinge only; final checkpoint from every seed retained |
| Target branches | one environment per branch | 64 per rollout; 10 rollouts = 640 decisions = 64 simulated seconds | Same new-interaction allowance for frozen, adaptation and scratch |

Training episodes last at most 640 decisions. Source and reference training may use their declared task samplers at resets, without selecting easy outcomes. No seed is replaced. The three source runs cost 294,912 decisions, at most 1,474,560 native controls and 14,745,600 physics substeps, before target experience. Parallel environments reduce elapsed computation, not the counted amount of experience.

Locked operational caps: 900 seconds for development reference learning, 1,800 seconds for each source seed, and 180 seconds for each target/control trial action loop. The non-learning throughput preflight confirmed these caps before collection. They apply to the learning/action loop; initialization, archival work and integrity replay are outside that timed loop. Report the measured loop time and available total wall time with their exact scope. There is no additional aggregate wall-time cap. A cap stops and retains an incomplete run; it does not select an earlier high-reward checkpoint or permit another attempt. Boundary checks cannot preempt a native call or optimizer update already underway, so report any overrun and never label an over-cap result as a fully completed budget. Counts of actual decisions, native controls, physics steps, resets, updates and CPU threads accompany the timing. Per-update metrics/counters are retained, but no separate profiler measurement of pure optimizer time is claimed.

These are simulated-interaction budgets. A native world paused while a CPU performs PPO updates does not demonstrate real-time learning on an unpausable physical toy. Report wall-clock time including updates separately. Sixty-four seconds is a first diagnostic budget, not a claim that the product requirement of minimal post-assembly experience has been met.

## Target comparison and what is transferred

For every source seed and each of the six target tasks, run the following three branches from identical fresh native initialization. This gives 18 matched triples, or 54 target trajectories. For case_index 0 through 5 in the table order, set the action-sampling seed to 10000 + source_seed * 10 + case_index after constructing the model and transferring any weights. Record optimizer and action RNG handling separately. Each branch is a continuous 640-decision opportunity with no trial reset or cross-task memory sharing.

- Frozen pretrained: load that seed's final actor and critic parameters, keep them fixed, and act stochastically from the same discrete policy family throughout.
- Brief adaptation: load the same actor and critic parameters and use the declared PPO updates every 64 decisions. Update all parameters; this is not a head-only adaptation experiment.
- From scratch: freshly initialize the identical actor/critic architecture and apply the identical target PPO schedule. The only intended difference from brief adaptation is the initial learned parameter values.

Create a fresh optimizer with the same settings for both updating branches; do not transfer Adam moments, a partially filled rollout buffer, training step counters, source RNG state or source history. Reset source-to-target frame history identically in all three branches. Use fixed hyperparameters rather than a training-progress learning-rate schedule. The frozen branch still receives current sensors, action/reward history and governor feedback. Its unchanged weights do not imply open-loop control or no within-episode information.

Keep pretrained weights read-only outside each branch's own copy. Save parameter hashes before and after runs. Frozen hashes must remain identical. Archive adaptation update locations and weight changes. No branch can import observations, gradients or target experience from another branch or body. The frame stack continues across PPO updates; it is cleared only at an actual task reset.

Measure cumulative realized reward at 6.4, 12.8, 32 and 64 seconds, and reward/lux in declared early and late windows, using only the trajectory collected under that budget. The first 64 decisions precede the first parameter update; paired frozen/adaptation behavior should match there under identical action RNG handling. The tenth update occurs after the final measured action and cannot explain any reported reward. Do not silently add a new post-training evaluation episode; any such exposure requires its own declared budget and question.

At a target safety failure, stop physical interaction and target updates for that branch. Do not let an auto-reset vector wrapper obtain another body's initial observation or populate an otherwise incomplete rollout with newly reset experience. Record how any incomplete rollout is discarded. Failure plots may display the separate conservative completion diagnostic, but artificial absorbing samples are not trained on as native transitions.

## Controls and attainable-behavior witnesses

For each final source checkpoint, run a frozen base-layout control on the six target body/sun combinations with hinge 0, using the same 64-second observation/action interface. Collect source_quiet once for each of those six cases, not once per training seed. These are 18 source_frozen and six source_quiet trajectories. A useful-behavior label additionally requires adaptation to beat the matched target quiet return and that seed's source_frozen control to beat the matched source_quiet return. These checks show whether source training learned anything useful before asking it to transfer.

After source checkpoints are frozen and hashed, collect this fixed five-policy suite on each of the six target cases: quiet; constant negative; constant positive; a square wave with a 2-second full period; and a square wave with a 4-second full period. Square waves start negative and alternate equal half-periods, matching action IDs 0 then 2 in the locked runner. Requested amplitudes are +/- .15; quiet is zero. Every control refreshes the same governor and follows native safety. Each policy has a 64-second opportunity. These requests depend on their declared schedule only, with no hidden angle, target light coordinate or simulator diagnostics.

The best safe probe in each case is an exposed attainable-behavior witness, not a deployable selector or a performance upper bound. Its result cannot be used to choose source checkpoints, tune PPO, choose a motor sign or change a target policy. Report all probes, not only the best. If none improves on quiet, label useful improvement in that target case as not established by these witnesses; do not conclude that the body is physically incapable or that the learning algorithm alone failed.

Evaluate the final elbow development reference on its two declared [3,3] tasks and their quiet controls. This reference is exposed to those tasks during training and is excluded from the transfer score. It has a larger budget and cannot be presented as a sample-matched competitor. A failed fixed-cap reference is a reported feasibility limit; there is no second training round, new reward or easier replacement task in this protocol.

The locked full schedule has an upper bound of 399,360 policy decisions: 294,912 source training, 32,768 reference training, 34,560 target triples, 11,520 frozen base controls, 3,840 base quiet, 19,200 target probes including quiet, and 2,560 development reference/quiet evaluation. Safety stops can reduce actual physical exposure. Integrity replays repeat already selected requests without producing new learning experience and must be counted separately as computation and simulation exposure.

## Metrics, interpretation and stop criteria

Primary transfer contrast: for each source seed, take the arithmetic mean across its six target cases of brief-adaptation minus from-scratch integrated native sensor reward through 64 seconds; then take the median of those three seed means. Report all 18 case-level differences and all failures. The declared improvement gate requires every frozen/adaptation/scratch target trajectory to complete its 640-decision budget without safety stops or errors, a positive median seed-mean difference, and at least two of three seed means positive. If any required trajectory fails, the gate fails; do not delete that row and compute a success from safe survivors. Partial raw-return contrasts may still be displayed with their exposure lengths. This groups the six cases sharing a pretrained checkpoint instead of treating them as six independent training replications. Three training seeds support a bounded development result, not broad statistical assurance.

Report frozen-minus-scratch and adaptation-minus-frozen separately. Positive frozen performance is evidence of useful prior parameters under feedback; extra improvement from updating is a different question. A positive adaptation-minus-scratch difference alone does not show a useful toy if both remain worse than quiet. Report every method's difference from quiet, difference from the safe probe witness, late-window received lux, sensor-reward components, energy, maximum local motor/IMU measurements, all flags/vetoes and actual steps before failure. Energy is an outcome, not a hidden training cost.

Use 0-12.8 seconds and 51.2-64 seconds as early/late observation windows. Summarize held light samples with duration weighting; do not count repeated held packets as independent fresh samples. Report safety and completed opportunities before reward rankings. Do not normalize by realized time alone: a short failed trajectory with a favorable average must remain visibly failed.

An incomplete runtime, failed observation boundary, nonfinite training state, unintended target exposure, replay mismatch or unsuccessful source/reference learning is retained and reported. No reward-driven tuning, enlarged budget, seed replacement or target redesign follows inside this declared batch. If source learning or target witnesses fail, distinguish an unestablished premise from evidence against transfer. If valid safe transfer gains are small, report their magnitude rather than calling any positive number product-ready.

The next decision after this bounded test is whether pretrained parameters offer useful new-interaction savings under a changed actuation map. It cannot establish arbitrary topology transfer, variable sensor/motor counts, contact locomotion, a universally sufficient state representation, or an optimal architecture. It also does not compare PPO against the unchanged live SARSA learner as a practical controller; that would be a separate question.

## Development, locking and evidence

Completed preflight, before native learning:

- All 18 native test suites passed, together with eight bridge test groups and five Python adapter checks. These cover implementation and native-boundary behavior, not success at the transfer objective.
- The non-learning throughput pilot completed 2,048 decisions in 2.031 seconds, approximately 1,008 decisions/second, with zero safety stops and zero parameter updates. This supports the locked operational caps; it is not a learning result or a withheld transfer trial.
- The separate pinned CartPole reference completed 32,768 training steps in 15.32 seconds. Its mean trained return was 409.2 versus 333.2 untrained. The absolute-return gate of at least 200 passed, but the predeclared gain gate of at least 100 failed: the observed gain was 76.0. The overall reference therefore remains failed. Finite-value and exact save/reload checks passed. Preserve the original reference artifact and criteria; do not rerun it, change its gate or call it a passed reference.

The owner-authorized native comparison proceeds as exploratory despite that reference learning-gain failure. It is not a structural/API failure, but it prevents claiming that all learning-reference validation succeeded. Passing numerical, serialization and native-boundary checks justifies collecting this bounded native evidence; it does not establish that PPO will learn or transfer here. Save the exact machine-readable task manifest, hyperparameters, seed derivations, action policy, wall limits and this final protocol hash before the first native development learning outcome.

The one fixed-cap elbow development run is explicitly a development pilot. It can justify a stop or a separately declared later experiment; it cannot justify silently tuning this batch. Preserve the candidate final source checkpoints independently of any target outcome. Seal all source checkpoints before running target probes or target comparisons. Source-domain evaluations may describe learning curves, but the checkpoint used for transfer is always the final budgeted checkpoint, never the best curve point.

Archive native/module specifications; all relevant source, tests and bridge code; pinned Python wheels/lock information; model and optimizer serialization; manifest/protocol/checkpoint hashes; reproducible RNG initialization and any continuation state; per-update diagnostics; complete/failed run markers; and the independent summarizer. Source/reference training retains compact action tapes, reset-task schedules and a SHA-256 stream commitment over native requests/responses rather than every large physical snapshot; the replay command verifies the retained native request stream. Every target/control trial retains the full five native transitions per decision, requested/emitted actions, raw native reward components, local observations, safety transitions and physical snapshots used only for replay verification. Exact action-tape native replay checks simulator evidence; it is not by itself proof that a policy computed those actions. Check the reported frozen before/after parameter hashes, source-policy parameter-hash match and the matched frozen/adaptation prefix before the first update. Complete neural-optimizer replay is not claimed. Do not call native action replay a validation of PPO updates.

Use exclusive new output directories and preserve failures. A resumed orchestration may skip an existing finished trial only after its protocol, trace, checkpoint dependencies and replay evidence validate. It must not rerun a recorded scientific failure, replace an occupied incomplete directory, or present a fresh training restart as continuation. Setup/save failures need explicit retained status; partial native-batch failures distinguish confirmed completed exposure from an uncertain remainder bounded by five native controls in the current slot. Mark that exposure inexact and do not claim successful complete evidence for a failed native batch. Keep historical reports, frozen inputs and the live browser's world untouched. A viewer may display hidden physical geometry to explain the creature while clearly showing the policy's smaller permitted input. The runtime/manifest decisions above are now fixed. Archive this document before native learning and put all outcomes, including negative or incomplete results, in a separate results document.
