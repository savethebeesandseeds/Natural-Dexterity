# Droid Blocks: design discussion, 2026-09-06

This records the voice discussion and the owner's current product direction.
It distinguishes implemented work, requirements and proposals. It does not
select a new learning architecture or turn a suggestion into an experiment.

## Owner clarification, 2026-10-06

The owner clarified that the **first** child-built creature should already have
substantial exposure to training environments and broadly generalized
competence. The product applies generalization learning to modular block
robotics: offline training supplies the starting knowledge, and the child's
assembly calls for inference and modest specialization rather than a new full
training session. Personal experience after rebuilding is useful in addition
to that pretrained starting point.

This supersedes a roadmap centered only on one learned body's checkpoint
transfer. It does not retroactively change the September discussion, select
a universal neural architecture or claim that existing results meet the goal.
The first bounded step is a
[pooled-pretraining development pilot](GENERALIZATION_PRETRAINING_EXPERIMENT.md),
with frozen first-use behavior measured separately from future adaptation.

## Product direction

Children should assemble a creature, rearrange its parts, and see purposeful
behavior driven by its senses and configured preferences. Very little new
experience after assembly is a product requirement: a child should not need to
wait through a substantial training session after each rebuild. The acceptable
time/sample budget is still to be defined and measured.

A small duck or T-Rex was proposed as an appealing reference build included with
the kit. It would demonstrate what the parts can do, while leaving children free
to build other bodies. This is a reference-build idea, not a committed mechanical
design. Not every arrangement will support walking or the same objective.

The owner authorized returning to development and simplifying the interface.
Animation remains central: seeing the creature move is the main experience.
Keep the scientific evidence accessible, with diagnostics and long explanations
secondary to the creature and clear controls. UI cleanup does not authorize a
silent change of controller, simulator, reward or research roadmap.

## The proposed T-Rex-to-snake stress test

Pretrain using a reference T-Rex, then rearrange the same motors and blocks into
a snake that was withheld from training. Ask whether prior knowledge makes
reward-driven behavior emerge with little new interaction.

This changes connectivity and contact mechanics, rather than merely varying the
mass or length of the same body. Reusing the parts does not preserve the action
semantics or guarantee that a learned gait remains useful. Pretraining on just
one T-Rex may not cover the features or behaviors required by a snake. A wider
simulation curriculum is one proposal; genuine snake/topology withholding must
remain explicit if it becomes an experiment. No such training or test has run.

The objective also needs to be feasible for the rebuilt body. Preferences such
as light seeking and movement should be configurable and distinguishable; a
walking demonstration is not a universal objective. Independent simulated safety
remains separate from reward. Locomotion will require collisions, ground support,
frictional contacts, balance, falls and recovery that the present anchored fixture
does not exercise. Contact/support sensing and useful recovery behavior remain
future design questions.

## What is actually implemented

The live light learner is still the model-free `history_expected_sarsa_v2`:
three bounded motor options, manually encoded local sensor/action history,
linear action values and online value updates. It is not a pretrained neural
foundation model. Its current construction fixture has one active joint and one
passive joint, with an anchored body and declared IMU/light sensing.

The completed affine predictor is separate. It identifies per-body sampled
sensor-delay dynamics and exports A/B/C/D/c matrices; it does not choose actions
in the live learner. Both current-output and longer-history predictors passed
their declared prediction gates. Those results justify considering a control
experiment, but have not demonstrated model-based control, locomotion or transfer
between topology classes. See [CONTROL_MODEL.md](CONTROL_MODEL.md) and
[LIGHT_PREDICTION_RESULTS.md](LIGHT_PREDICTION_RESULTS.md).

Earlier rover, learning, adaptation, retention, context and prediction findings
remain historical evidence. Their locked protocols, reports and frozen artifacts
must be preserved. A later method may reuse declared prior experience, but must
not relabel it as unseen evidence.

## Architecture questions still open

A neural network is a possible policy/value representation, not an RL algorithm
by itself. A shared controller across assemblies needs an explicit way to handle
variable motor/sensor sets and connectivity. Shared module processing, recurrence,
graphs and compact adapters were discussed as options; none was selected.

The owner proposed a reusable simulation-pretrained embedding with only a small
final head adapted after rebuilding. This may be efficient if the frozen features
retain the relevant information and the head can express the new behavior. An
unexpected topology can violate either assumption. Short-history context inference
with fixed weights and gradient updates to a head are distinct adaptation
mechanisms; a useful future comparison would separate their contributions under
matched new-experience budgets.

Actor-critic does not inherently double the needed physical interactions: actor
and critic can train from the same transitions. Shared versus separate networks,
and how much computation to spend on updates, are different choices from the
sampling/update algorithm. A2C belongs to the actor-critic family; UNREAL explores
auxiliary representation learning, not a guaranteed replacement for modern
robotics methods.

No commitment was made to MPC, a neural architecture, a particular RL optimizer,
a training stack, a T-Rex build or a simulator migration. The earlier
receding-horizon suggestion remains a proposal. Before inventing another bespoke
algorithm, assess established open implementations against the modular-body,
reward and low-experience requirements. A next experiment should state its
question and evaluation criteria before it runs.

## Research references, not selected dependencies

- [Disney's BD-X paper](https://arxiv.org/html/2501.05204v1) describes PPO with
  separate policy/critic networks, identified actuators and simulation
  randomization for a particular expressive biped. It is a useful reference for
  the in-box character, not evidence of arbitrary assembly transfer.
- [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) and
  [RSL-RL](https://github.com/leggedrobotics/rsl_rl) provide inspectable robot
  environments and established training implementations, including PPO. Their
  existence does not make a trained policy independent of its robot embodiment.
- [Newton](https://github.com/newton-physics/newton) is an open physics engine
  initiated by NVIDIA, Google DeepMind and Disney. It is simulation
  infrastructure, not a new learning algorithm or a decision to migrate.
- [Meta Motivo](https://github.com/facebookresearch/metamotivo) provides code,
  pretrained models and reward/goal-conditioned behavioral inference for the
  fixed simulated HumEnv humanoid. It is relevant to reusable behavior and
  configurable reward, but does not establish T-Rex-to-snake transfer. The
  repository uses CC BY-NC 4.0: this is a noncommercial research reference,
  not a selected dependency for a commercial toy.
- [UNREAL's original paper](https://arxiv.org/abs/1611.05397) studies unsupervised
  auxiliary tasks sharing a representation with reward learning. Its reported
  game benchmarks are not evidence that it solves modular robot transfer.
