# Droid Blocks: the implemented control model

## Current controller map — 9 October 2026

The [pretrained workshop](PRETRAINED_WORKSHOP.md) at `/pretrained.html` deploys a
frozen 17,475-parameter actor from pooled source seed 101: 204 local-history inputs,
two 64-unit tanh layers and three categorical effort choices. Its native runtime
contains no critic, optimizer or online training. Rebuilding clears world and
sensor history while preserving that same pretrained actor.

The research PPO model additionally contains a separate value network, for
34,820 actor-plus-critic parameters. The [bounded specialization protocol](SPECIALIZATION_EXPERIMENT.md)
compares tuning four output tensors (195 actor plus 65 critic parameters),
tuning all weights, and matched scratch against the unchanged pretrained model.
These isolated experimental worlds do not modify the workshop actor.

The [completed specialization result](SPECIALIZATION_RESULTS.md) misses the
declared practical gain gate for both tuning scopes. The immutable pretrained
actor remains the workshop default; sampled retention averages do not bound
individual losses.

The [longer-budget comparison](CONSERVATIVE_SPECIALIZATION_RESULTS.md) uses
384 learning decisions, six 64-decision rollouts and 24 Adam steps, followed by
256 decisions with fixed weights in the same world. Conservative updates lower
both learning rate and PPO clipping; standard updates retain the previous
settings. Both tune all 12 actor/critic tensors with fresh optimizer state.
Neither meets combined usefulness and reliability gates. Conservative mean
retention passes despite a 34.334899 individual probe loss; standard has two
native safety stops and a 41.550677 probe loss. Further work must establish how
adaptation preserves the pretrained behavior before adding it to the workshop.

The September analysis below describes the older Expected SARSA controller and
its manually designed features. Its historical parameter counts and recall
semantics belong to that controller. The later PPO appendix and the linked
October protocols describe the neural controller separately.

## Historical review — 6 September 2026

Reviewed against the native source on 2026-09-06. This document supplies a shared
mathematical vocabulary for the current light-enabled, anchored two-link kit.
It describes the implemented controller and explicitly separates proposed work.
The original rover experiment is a different controller/plant and is outside
this document. No controller or experiment parameters were changed by this review.

## The central distinction

The live controller is a model-free, history-conditioned reinforcement learner
acting on a nonlinear simulated plant. A separate prediction experiment now
implements affine input-output identification with explicit A/B/C/D/c matrices;
section 9 defines those objects. Those matrices do not choose live commands.
We still have no physical-state observer, A/B/C/D feedback design, or
model-predictive controller for this kit. MuJoCo supplies the world dynamics;
the deployable learner and predictor do not receive its hidden physical state.

Our experiment language hid several different mathematical objects:

| Earlier term | Precise object |
| --- | --- |
| Body | Assembly parameters sigma: geometry, masses, inertias, joints and sensor mounts |
| Senses | Delivered observation y, with different sample times and delays |
| Recent memory | Finite history of observations, executed effort and received reward |
| Controller state/features | A manually designed 56-dimensional encoding phi of that history |
| Learned experience / saved values | W in R^(3x56), the 168 action-value coefficients |
| Choose a move | Sample one of three effort options from the policy induced by W |
| Adaptation | Continue updating W after the illumination condition changes |
| Recall experiment | Restore an earlier W, leaving physical and other controller state intact |
| Context recognition | A separate supervised nearest-neighbor classifier of A/B; not a state observer |

## 1. Plant: what actually evolves

For a fixed assembly sigma, define the mechanical coordinates

$$q=\begin{bmatrix}q_1\\q_2\end{bmatrix},\qquad x_{\mathrm{mech}}=\begin{bmatrix}q_1&q_2&\dot q_1&\dot q_2\end{bmatrix}^{T}.$$

q1 is the powered hinge angle; q2 is the passive hinge angle relative to the
first link. Both are continuous revolute coordinates about the Y axis. The
mechanism moves in the XZ plane with an anchored base and no collisions.
The conceptual mechanical equation is

$$M_\sigma(q)\ddot q+C_\sigma(q,\dot q)\dot q+g_\sigma(q)+
\begin{bmatrix}0\\0.006\dot q_2\end{bmatrix}=S\tau,
\qquad S=\begin{bmatrix}1\\0\end{bmatrix}.$$

Here M includes the declared joint armatures, C qdot denotes the Coriolis and
centrifugal contribution, and g denotes gravity. Motor output friction is already
included in tau and must not be counted twice. The first armature is the reflected
rotor inertia 1.5e-6 * 50^2 = 0.00375 kg m^2; the passive armature is 0.00002.
MuJoCo assembles/integrates this nonlinear mechanics from the generated model;
we have not exported symbolic M/C/g functions or identified them from data.
This is the joint-coordinate form of [MuJoCo's dynamics](https://mujoco.readthedocs.io/en/latest/computation/#general-framework).

One input drives two degrees of freedom. The second coordinate can move through
mechanical coupling and gravity. Rank(S)=1 establishes underactuation; it does
not determine the rank of a local state-space controllability matrix.

The command u is dimensionless signed motor effort, not torque. At every
h=0.002-second physics step the native motor wrapper computes a thermally derated
and voltage-limited target current i_star, then

i_next = clip(i + (1-exp(-h/0.01)) (i_star-i), -2.5, 2.5),

tau_motor = 0.75 * 50 * 0.018 * i_next
            - 0.002 qdot1 - 0.02 tanh(qdot1/0.1),

T_next = T + h [2.4 i_next^2 - (T-25)/8] / 12.

Before voltage limiting, i_star = u * 2.5 * thermal_derate(T). Voltage limiting
uses back EMF 0.018 * 50 * qdot1 and the 7.4 V reference bus. Native safety can
veto drive, and the world limits the applied torque to +/-4 Nm. These equations
describe the implemented reference motor approximation; winding inductance in
the catalog is not a separately integrated electrical state in this wrapper.

Each 0.02-second control transition composes ten motor/physics substeps. Effort
is held during those ten steps; torque is recomputed as current and speed change.
The four mechanical coordinates alone are therefore not the full Markov state.
A fuller simulator state includes i, T, delivered/queued samples, sampling phase,
the stuck-estimator state and latched safety state. Source position/intensity ell(t) is an external input. For a
fixed or explicitly augmented source schedule, write the complete transition as

X_(k+1) = F_sigma(X_k, u_k, ell on this interval).

This F is implemented by simulation. It is not a learned predictor inside the
controller. Replay fingerprints include additional bookkeeping; a hash is not
a state estimate or a learned state representation.

Sources: [assembly and native world](../src/construction.cpp),
[motor equations and sensor rewards](../src/module_catalog.cpp),
[motor state](../include/droid/module_catalog.hpp).

## 2. Output: what the controller can measure

The essential numeric measurements are

y_k = [motor angle, motor speed, current,
       local IMU specific force (3), local IMU angular velocity (3), light].

Motor feedback is synchronous at the control endpoint in this construction world; the catalog's nominal motor-feedback latency is not separately simulated here. The observation contract also carries validity, acquisition/delivery times,
sample age and local actuator status. The policy has no direct q2, qdot2,
world pose, source coordinate or assembly-geometry input.

A schematic measurement equation is y_j = h_sigma,j(X at acquisition), delivered
later and held until the next delivery. This is not one instantaneous C x:
IMU sampling is every 0.010 s with 0.006 s delivery latency; light sampling is
every 0.100 s with 0.020 s latency. The controller runs every 0.020 s. Sensor noise and light occlusion are not modeled in this fixture.

For example, let p(q) and n(q) be the mounted light sensor position and receiving
normal, s the source position, d=||s-p|| and L0 its intensity. The implemented
instantaneous light model is

L(q,s) = clip(L0 max(0, n dot (s-p)/d) / [1+(d/0.5)^2], 0, 1000).

The implementation returns zero at d<=1e-9. The learner receives the delayed
sample of this value. It does not receive p, n or s. Changing link geometry or sensor mounts can change both dynamics and measurement
functions. Moving a passive weight changes dynamics and resulting readings
through motion; it need not change the optical function at fixed joint angles.

## 3. The learner's state representation

The learner retains up to 64 control transitions (1.28 s). Let H_k denote that
finite history together with the current delivered observation. Its decision
representation is

$$z_k=\phi(H_k)\in\mathbb{R}^{56}.$$

This is a manually designed information-state approximation. It is not an
estimated physical state x_hat, a probability distribution over hidden states,
or a representation shown to be Markov/sufficient for optimal control.

| Feature indices | Implemented meaning |
| --- | --- |
| 0 | Constant bias 1 |
| 1-2 | sin/cos of the local motor angle |
| 3-5 | Motor velocity/6, its absolute value, current/2.5 |
| 6-11 | Three IMU force axes/39.24 and three gyro axes/10 |
| 12-14 | log1p(light)/log1p(1000), valid-sensor mask, fresh normalized-log-light slope times 4 s |
| 15-20 | Gyro and force changes over at least .2 s, scaled by 40 rad/s^2 and 100 m/s^3 |
| 21-24 | Previous effort/.15, gyro magnitude/10, motor acceleration/40, light age/.5 |
| 25-27 | Recent .4 s means of total reward rate/2, effort/.15, motor velocity/6 |
| 28-31 | Products of motor-speed feature with encoder sin/cos, light and light slope |
| 32-55 | Eight circular angle bases multiplied by three motor-speed bases |

Most signed channels are clipped to [-1,1]; normalized light and age to [0,1].
Derivative features remain zero when the required older reading is unavailable.
For angle centers c_j=2*pi*j/8 and speed centers v_l in {-1,0,1}, the last 24
features are exp(3[cos(q1-c_j)-1]) * exp(-2[clip(qdot1/6)-v_l]^2).
Thus the value function is linear in its learned coefficients but nonlinear
in the measured angle, speed and history. A fixed 56-feature controller is not
yet an architecture for an arbitrary number of modules.

Source: [features()](../src/light_learner.cpp), lines 330-377.

## 4. Control law: what reaches the motor

The policy chooses a fixed-duration option a_m in {negative, coast, positive}
with requested effort v(a_m) in {-0.15,0,+0.15}. Decision index m advances every
20 control steps, or .4 s. It starts with .1 s quiet, then visits each option
once in seeded shuffled order. Thereafter it is epsilon-greedy with

epsilon_m = max(.12, .4 exp(-decision_count/150)).

There are three linear action-value functions:

$$Q_W(z,a)=w_a^{T}z,\qquad W\in\mathbb{R}^{3\times56}.$$

With probability 1-epsilon choose a maximal Q option; with probability epsilon
choose uniformly among all three. Near-exact greedy ties use the seeded tie
handler. W has 168 learned coefficients. W is not a dynamics or transfer matrix.

Every .02 s the shared local governor converts the requested effort to an
actual command. For sign s=sign(v), valid sensors, and v!=0:

target = s min(|clip(v,-.35,.35)|, .05 max(0,3-s*qdot1)),

u_k = clip(target, u_(k-1)-.1, u_(k-1)+.1).

The zero option requests zero current, with current decay and friction still present; it is not an ideal free hinge or a position hold. The slew limiter can make a transition to zero
non-instantaneous. Initial undelivered sensors return zero, and malformed/stale
observations fail validation. Speed feedback is refreshed throughout an option.
The 3 rad/s target is soft: it cannot guarantee the passive body will never
backdrive the motor faster. Separate native safety remains active.

This is dynamic output feedback with adaptive parameters and discrete options.
It is not a PID law, inverse-dynamics controller, LQR gain or MPC solver. Holding
an option does not remove the local feedback or freeze the body's motion.

Source: [envelope(), choose(), act()](../src/light_learner.cpp).

## 5. Objective and reinforcement-learning update

The implemented algorithm is history_expected_sarsa_v2: a linear Expected
SARSA-based controller with accumulating eligibility traces, fixed-duration
options and recent one-step replay. The learner fits action values; it does not
fit the next state or learn the sensor reward function.

The light reward rate is

rho_light(L) = clip(log(1+L)/log(1001),0,1).

With D(v;soft,hard)=clip((v-soft)/(hard-soft),0,1), the IMU rate is

rho_imu(f,omega) = -max(D(abs(||f||-9.81);4.905,29.43),
                       D(||omega||;8,20)).

Missing readings have the catalog's -1 rate. Each control reward r_k sums the
ten physics-substep integrals h*(rho_light+rho_imu), using the then-delivered
samples. It is not necessarily .02 times the final sample's rate. The control
objective is discounted integrated sensor reward. It is not a commanded-angle
tracking error, raw lux increase, or an explicit energy-minimization objective.
Hard safety is evaluated separately and cannot be offset by a favorable reward.

Let gamma=.92^(1/20), Gamma=gamma^20=.92 and

R_m = sum_(j=0..19) gamma^j r_(20m+j).

Indices here start after the quiet initialization. For the current feature z_m,
chosen option a_m and next feature z_(m+1), the main TD error is

$$\delta_m=R_m+\Gamma\sum_a\pi_W(a\mid z_{m+1})Q_W(z_{m+1},a)
-Q_W(z_m,a_m).$$

The expected bootstrap uses the current epsilon policy; during the initial
shuffled visits it uses the scheduled next option. This is an Expected SARSA
backup, rather than a Q-learning max-only backup. The implementation then uses

delta_bar = clip(delta_m,-2,2),
E_a <- clip(Gamma*.7*E_a + 1[a=a_m]*z_m, -5,5),
w_a <- clip(w_a + [.12/(1+||z_m||^2)]*delta_bar*E_a, -20,20).

After each option it stores (z_m,a_m,R_m,Gamma,z_(m+1)) in a FIFO buffer of the
most recent 32 options and makes four uniformly sampled one-step replay updates.
Replay uses the same current-policy expected target, step size .08/(1+||z||^2),
and TD/weight clipping, without eligibility traces or importance weighting.
Consequently this practical combination must not be presented as an unmodified
textbook on-policy algorithm with a supplied convergence theorem. Frozen runs
suppress parameter writes; sensing, history, exploration and bookkeeping continue.

In v2 all actions start with the same bias Q0=mean_quiet_transition_reward/(1-gamma),
using the four fully delivered quiet intervals .02-.10 s. Other weights and
traces start zero. This initialization correction and the unsuccessful v1
experiment are preserved in [the learning record](LIGHT_LEARNING_EXPERIMENT.md).

Source: [learn(), observe(), specification()](../src/light_learner.cpp).
General background: [Sutton and Barto, Reinforcement Learning](https://reinforcementlearning.pubpub.org/).
The exact formulas above describe this repository's implementation.

## 6. Where are the transition and transfer matrices?

Three different things must not be conflated:

1. M_sigma(q) is the configuration-dependent mechanical inertia matrix used
   inside MuJoCo. Its existence does not supply a learned state-space model.
2. A local linear model would be delta xdot=A delta x+B delta u,
   delta y=C delta x+D delta u, with Jacobians at a stated operating point.
   The corresponding LTI transfer function is G(s)=C(sI-A)^(-1)B+D, for zero
   initial perturbation and its specified input/output approximation. Delays,
   nonsmooth limits and large motions require explicit treatment. We have not
   derived or used a physical Jacobian model for the constructed creature.
   The separately identified sampled input-output surrogate is defined below.
3. An RL transition kernel P(s_next|s,a), or learned predictor F_hat(z,u),
   describes consequences of actions. The current model-free learner does not
   estimate either. It learns Q from observed transitions without storing a
   transition-probability matrix. Its history features are not proven Markov.

Before the prediction experiment, the light-engine experiments supplied no
numerical A/B/C/D matrices. The prediction addition supplies a sampled affine
input-output surrogate, not an identified nonlinear model, physical observer,
observability/controllability analysis, closed-loop stability certificate or
model-based controller.
Parameter/effort clipping and successful finite trials do not establish those
properties. A control-theory discussion must state this gap plainly.

## 7. What the later experiments mean mathematically

- Adaptation compared W continuing to update with W held constant after a
  source change. Both retained the dynamic feedback controller.
- Retention compared the then-current frozen W120 with an earlier frozen W60
  from the same learning trajectory. The experiment selected the checkpoint;
  it did not reset x or restore the controller's other internal variables.
- Context decoding fitted a separate supervised five-neighbor classifier.
  Its one-second representation is 11 twelve-number frames, or 132 features,
  with different scaling from the RL encoder. It predicts an A/B label under
  random probes. It is not the 56-feature RL representation, a physical-state
  estimator, a dynamics predictor or part of the live control law.

Thus context accuracy does not establish that the RL state encoding is
sufficient, that hidden physical state is observable, or that memory selection
will improve closed-loop behavior.

## 8. The recommendation that led to the prediction experiment

Before adding another layer of memory selection, agree on and test the state
and model we intend the general engine to learn. A concrete model question is:
can local sensor/action history predict future sensor responses to previously
withheld commands, for a declared family of constructed bodies?

Use the simulator's physical state/Jacobians only as a clearly labeled analysis
reference. Decide separately whether the deployable engine should use a physical
observer, a learned latent/information state, or a measured input-output model.
An identified local A/B/C/D model is one diagnostic baseline around a specified
operating condition; it is not a universal model for large underactuated swings.
Declare prediction horizons, excitation, withheld trajectories, uncertainty and
error criteria before fitting. Then assess a model-based controller against the
unchanged RL baseline on the same sensing and actuator boundary.

This recommendation was subsequently adopted as the separately declared
prediction experiment below; it is not a claim that RL requires a transition
matrix. Observation-driven memory selection remains
an available later experiment; it must not substitute for an agreed state/model
architecture. General connector graphs, contact and transfer across rebuilds
still require their own physical and control validation.

## 9. The explicit sampled predictor

The [locked prediction protocol](LIGHT_PREDICTION_EXPERIMENT.md) is the authority
for fitting and withheld evaluation. The native module is
[light_prediction.cpp](../src/light_prediction.cpp); the independent regression
and evidence checker is
[summarize-light-prediction.mjs](../tools/summarize-light-prediction.mjs).
The model is fitted separately for each body. Construction changes can alter
the learned coefficients; this experiment does not demonstrate model transfer.

At t_n=.02+.1n seconds, the twelve-dimensional y_n contains last emitted effort,
encoder sin/cos, motor speed/current, six IMU channels and delivered light, with
fixed normalization. The requested input v_n is normalized effort, held for
the next five .02-second controls. The unchanged local governor uses actual
feedback during those controls, so future emitted effort is an output to predict,
not a known input. The identified system includes this governor and sampling.

The main state and model are

\[
z_n=[y_n^T,\ldots,y_{n-10}^T,v_{n-1},\ldots,v_{n-10}]^T\in\mathbb R^{142},
\qquad z_{n+1}=Az_n+Bv_n+c,\quad y_n=Cz_n+Dv_n.
\]

The top twelve rows of A/B/c are fitted; the remaining rows shift prior outputs
and requests exactly. C selects the newest output block, and D=0. Thus A is
142x142, B is 142x1, C is 12x142, D is 12x1, and c has 142 entries. The simpler
current-output model has 12 state entries. The history model with explicit
requests removed has 132 entries and B=0; past emitted efforts remain in y.

Training learns Theta in y_next=Theta*[1,z,v], by mean squared residual plus
.001 times the sum of squared nonbias coefficients. The intercept is not
penalized. No reward or future hidden physical variable appears in this fit.
This is supervised system identification; Expected SARSA remains the separate
model-free reinforcement-learning strategy described in section 5.

At an evaluation origin, initialize z from available history and supply the
known future request sequence. Every subsequent state uses predicted outputs;
later real measurements only score the forecast. One-step accuracy alone is
insufficient: the protocol measures .1, .4 and 1.0-second recursive forecasts.
The representation is chosen delay coordinates, not a learned latent state or
a proof that the hidden mechanical state is observable from this history.

For differences between trajectories of this fixed affine surrogate, c cancels.
With zero initial difference, its discrete requested-input/output transfer matrix is

\[
G(\zeta)=C(\zeta I-A)^{-1}B+D.
\]

Here z is the state and zeta is the transform variable. This transfer matrix is
12x1 and belongs to the fitted .1-second surrogate. It is not a global transfer
function of the nonlinear mechanism. Forecast error, physical observability,
numerical model stability and closed-loop controller stability are distinct
questions. The separate results record decides whether prediction evidence
justifies a subsequent bounded control experiment.

The [completed prediction results](LIGHT_PREDICTION_RESULTS.md) now pass that
prospective gate for both the 142-state and 12-state models on all three bodies.
This supports testing a bounded receding-horizon control law. It does not prove
that 142 coordinates are necessary, that the learned state is Markov, or that
optimizing predicted reward improves actual light-seeking. The shorter-horizon
motion gains and weaker light-specific gains should both inform that next protocol.

## 2026-09-07: neural model-free transfer experiment

The owner subsequently adopted the actuation-transfer question in
[the locked protocol](ACTUATION_TRANSFER_EXPERIMENT.md). This adds a separate
Stable-Baselines3 PPO experiment. The earlier live learner and prediction models
remain available; the prediction gate has not become an implemented MPC loop.

Let h_k be the causal stack of 12 local feedback/action/reward frames, with
204 scalar inputs in total. Both networks use this same information:

\[
a_k \sim \pi_\theta(\cdot\mid h_k),\qquad
V_\phi(h_k)\approx \mathbb E_\pi\!\left[\sum_{j\ge0}\gamma^j r_{k+j}\mid h_k\right].
\]

The actor produces three action probabilities. A sampled action requests
-0.15, 0 or +0.15 for 0.1 seconds; the existing native governor still updates
the emitted effort every 0.02 seconds. The critic estimates discounted return,
not the next physical state. Both use two 64-unit tanh layers. A fixed history
stack supplies short memory; there is no recurrent network or learned latent
body representation in this baseline.

The recorded reward r_k is the sum of five native transition rewards, each
integrating the local light and IMU votes. The physical discount timescale is
preserved with gamma = .92^(.1/.4). PPO estimates advantages using temporal
differences and GAE, with lambda=.95. For an ordinary continuing transition:

\[
\delta_k=r_k+\gamma V_\phi(h_{k+1})-V_\phi(h_k),\qquad
\widehat A_k=\sum_{j\ge0}(\gamma\lambda)^j\delta_{k+j}.
\]

The sum is finite over a rollout. A genuine terminal event removes the future
value; a time-limit truncation bootstraps from its actual terminal observation.
The runner records raw physical reward independently of the library's
time-limit bootstrapping adjustment.

PPO reuses each collected rollout for four optimization epochs. Its clipped
actor objective uses the likelihood ratio
rho_k = pi_theta(a_k|h_k) / pi_old(a_k|h_k):

\[
\mathbb E_k\!\left[
\min\!\left(\rho_k\widehat A_k,
\operatorname{clip}(\rho_k,0.8,1.2)\widehat A_k\right)\right].
\]

The implementation also trains the critic against estimated returns and adds
an entropy term to the optimization objective. Entropy is not an extra physical
sensor reward. The two networks learn from the same transitions; two networks
do not imply two separate physical sample budgets. PPO by itself does not
establish fast adaptation or cross-body generalization.

Source training varies lengths, a block and illumination with the motor at
the base. The test moves that motor to the elbow. Frozen and adapting branches
receive identical pretrained actor/critic weights and fresh histories; the
adapting branch has a fresh optimizer and updates all weights. Scratch starts
from newly initialized weights. Each receives 640 new decisions, with no
post-budget evaluation used to improve its measured score. The affine model
does not participate. There is no explicit world/body predictor, transfer
matrix or model-based planning horizon in this controller.

The reward comparison must include quiet controls: receiving high reward
while already well illuminated is different from learning useful light-seeking
movement. The [completed transfer results](ACTUATION_TRANSFER_RESULTS.md)
distinguish the failed transfer contrast from that behavioral requirement.
Quiet remains a strong control, and no case passes both matched source/target
quiet checks.
