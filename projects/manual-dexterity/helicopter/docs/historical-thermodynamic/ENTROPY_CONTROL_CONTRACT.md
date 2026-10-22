# Entropy control contract

This project studies an explicitly simulated electric helicopter. Its parameters
define a reproducible reference vehicle; they are not measurements of an aircraft.
The purpose of the entropy controller is to solve a constrained nonlinear flight
problem using an explicit irreversible-loss model. Successful simulation does not
establish hardware performance or validate an aerodynamic model experimentally.

This document specifies the accounting and acceptance criteria. The native model
metadata supplies the actual numerical parameters, units, state layout, operating
limits, and source fingerprint. Evaluation artifacts supply results; this document
does not assert that any pending acceptance criterion has passed.

**Implementation status:** the C++ model, native CasADi/IPOPT NMPC, and independent
flight evaluator are implemented. The measured simulation results and limitations
are in the [project README](../README.md) and the source-verified
[evaluation report](../artifacts/evaluation.json). This contract specifies the
model and acceptance criteria; it does not establish real-time execution,
global optimality, or hardware readiness.
The runtime architecture is explicitly hybrid: entropy NMPC for airborne flight,
counted PID fallback for rejected solves, and planned PID support as the skids
approach the ground. Entropy-optimal contact control is not implemented.

## Boundary and assumptions

- The modeled system includes rigid-body flight, the main and tail rotors, the
  electric drive, its thermal storage, and the specified relaxation of the wake
  into an ambient reservoir at absolute temperature `T0`.
- Electrical input is work delivered at the modeled supply boundary. Any ideal
  battery or supply is explicitly lossless. Electrochemical battery entropy is
  excluded unless a separate identified battery model is introduced.
- Rotor speeds are governed at their declared values. Their kinetic energy is
  constant in this approximation. Varying rotor speed requires rotor-energy and
  governor states; simply changing a profile-power coefficient is insufficient.
- Pitch-actuator and flap-angle lags are modeled, but their internal mechanical
  storage and servo electrical losses are omitted. Inflow has explicit effective
  kinetic-energy storage. The assigned inflow masses are reference-model
  parameters, not measured wake masses.
- The operating envelope is near hover. Blade stall, vortex-ring descent,
  detailed rotor interference, and vehicle-specific inflow maps are not validated
  by this approximation. The controller must obey the declared envelope.
- The tail collective is nonnegative within its declared range, with available
  authority on either side of the yaw-trim requirement. Reverse tail thrust is
  outside this reference vehicle's allowed command set.
- Wind is a prescribed external mechanical-energy source or sink. It is not free
  propulsion supplied by the electrical system. Record its work separately.
- MuJoCo ground contact is an external mechanical port. Contact work is recorded
  in the plant energy ledger, but ground deformation, frictional heating, and
  ground-contact entropy are outside the entropy model. The NMPC predictor is a
  free-flight model and does not predict contact forces.
- Temperature is in kelvin, energy in joules, power in watts, entropy in joules
  per kelvin, and entropy-generation rate in watts per kelvin.

## State, initialization, and command timing

The model has 23 state components and four commanded pitch angles. These are
physical variables, not PID gains or a surrogate tracking-error objective.

| State indices | Quantity | Units / convention |
| --- | --- | --- |
| 0–2 | Position | m, world |
| 3–5 | Translational velocity | m/s, world |
| 6–9 | Attitude quaternion | wxyz, body to world |
| 10–12 | Body angular velocity | rad/s |
| 13–16 | Actual main collective, longitudinal cyclic, lateral cyclic, tail collective | rad |
| 17–18 | Main and tail induced-flow speeds | m/s |
| 19–20 | Longitudinal and lateral flap angles | rad |
| 21–22 | Main and tail motor temperatures | K |

All vehicle parameters are assumed reference values. The nominal vehicle has
mass `2 kg`, diagonal inertia `[0.060, 0.095, 0.090] kg m²`, main/tail radii
`0.55/0.11 m`, governed speeds `180/650 rad/s`, and ambient temperature `293.15 K`.
The model metadata exposes the remaining coefficients and limits, including
their numerical values and units.

Each paired trial starts from the same prepared state at `[0, 0, 0.22] m`: rotors
already governed and spooled, steady pitch/inflow/flap states, zero translational
and body rates, and a numerical still-air mechanical hover trim. The trim balances
gravity and all rotor body moments, including tail force and both reaction
torques. Body roll and pitch compensate the rotor assembly's lateral force.
Motors start at ambient temperature by assumption; this is not a simulated
spool-up history or thermal equilibrium. Initial wake energy is nonzero and is
subtracted in the completed-process accounting. Mass-perturbed trials recompute
this prepared state for the actual plant. Both controllers start from the nominal
mass prior and then use the causal estimate described below. Neither controller
is credited with discovering the initial trim.

Both enabled controllers hold those prepared pitch commands for the first
`0.1 s`, collecting ten causal velocity/force observations before feedback
starts. Mission time, energy and entropy include this interval. Its one
`trim_observation_hold` decision counts against NMPC participation; it is not
a hidden preflight identification period. The disabled negative control requests
zero pitch immediately, subject to the same command increments and actuator lag.

Accepted NMPC commands use a `0.1 s` zero-order hold. The discrete command bound is
`|u[k] - u[k-1]| <= input_rate_max × 0.1 s`, with rates
`[0.65, 1.2, 1.2, 1.2] rad/s`. This bounds changes between commanded setpoints;
it is not a continuous derivative limit on a discontinuous held command.
Actual pitch follows the separate continuous lag
`d(actual_pitch)/dt = (command - actual_pitch) / 0.08 s`.
Flap lag is `0.06 s`; the two inflow lags are `0.12/0.06 s`.

The plant advances every `0.01 s`. The PID baseline, PID fallback, and planned
PID landing support update on
that interval, using the same command ranges and corresponding
`input_rate_max × 0.01 s` increment bounds. An accepted NMPC action is not ramped
again in the plant: doing so would add input dynamics absent from its predictor.
Controller sampling rates therefore differ and must remain visible in a
comparison.

## Rotor power and mechanical work

For a rotor with thrust magnitude `F`, disk area `A`, radius `R`, angular speed
`Omega`, solidity `s`, and mean blade drag coefficient `Cd`, the declared hover
approximation is

```text
P_induced = kappa F^(3/2) / sqrt(2 rho A)
P_profile = rho A (Omega R)^3 s Cd / 8
```

The main and tail rotors each contribute. Any smooth approximation at zero thrust
must be disclosed and its effect quantified. Profile power remains positive at
zero collective when a rotor is spinning. NASA's rotor model distinguishes these
terms from propulsive/climb power and describes the need for inflow and
operating-condition corrections. This project does not claim to reproduce the
full NASA model. [NDARC theory, sections 11.5–11.6][ndarc]

The native positive-thrust approximation is
`positive(z) = [z + sqrt(z² + epsilon²)] / 2`, with `epsilon = 1e-6 N`.
Its excess over `max(z, 0)` is at most `epsilon/2 = 5e-7 N`.

For a reduced force/moment representation, useful mechanical power must include
both translation and rotation:

```text
v_air = v - wind
P_useful = F_rotor_world . v_air + tau_rotor_body . omega_body
E_inflow = sum(0.5 M_inflow v_inflow^2)
P_shaft = P_useful + sum(P_induced + P_profile) + dE_inflow/dt
```

The native transient model uses `P_induced = kappa F v_inflow`; the hover formula
above is its steady, zero-axial-speed limit. Its first-order inflow target is
`sqrt(V_axial^2 / 4 + F / (2 rho A) + 1e-12) - V_axial / 2`.
The small regularization and effective inflow-energy store are explicit modeling
choices. They do not make the approximation valid in vortex-ring flow.

At an offset rotor hub, with body-frame axis `n` and offset `r`, the implemented
power-consistent allocation is

```text
V_axial = n . [R_body_to_world^T (v - wind) + omega_body cross r]
P_aero = F V_axial + P_induced + P_profile + dE_inflow/dt
Q_aero = P_aero / (Omega + omega_body . n)
P_shaft = Omega Q_aero
F_body = F n
tau_body = r cross F_body - Q_aero n
```

The denominator stays away from zero throughout the declared body-rate limits.
The hub work is counted through the force moment exactly once. Cyclic-induced
airframe force and moment work enter this ledger; there is no independent free body torque. Lossless coordinate
changes preserve force/velocity and torque/rate dot products. Negative shaft work
uses the explicit non-regenerative braking loss below; silently clipping it
would destroy the power balance.

The equivalent mechanical-energy ledger in free flight is

```text
E_mech = 0.5 m |v|^2 + 0.5 omega_body^T I omega_body + m g z
dE_mech/dt = P_rotor_body - P_drag_diss - P_angular_diss + P_wind
P_drag_diss = -F_drag . (v - wind)
P_wind = F_drag . wind
```

Here `P_rotor_body = F_rotor . v + tau_rotor . omega`. If the ledger instead
uses air-relative `P_useful`, the rotor contribution `F_rotor . wind` also belongs
in `P_wind`. These are equivalent conventions; mixing them double counts wind.
Contact work belongs in a separate mechanical boundary term when contact occurs.
Airborne conservation tests must not accidentally include an unaccounted ground
constraint.

## Electrical losses and thermal storage

The exact native parameterization is exposed in model metadata. For each main or
tail motor, the quasi-static equivalent circuit and non-regenerative drive use

```text
positive(P) = 0.5 [P + sqrt(P^2 + epsilon_power^2)]
P_drive_loss = drive_loss_fraction positive(P_shaft)
I_motor = (P_shaft + P_drive_loss) / (Omega_motor k_t)
R(T) = R_ref [1 + alpha (T - T_ref)]
P_copper = I_motor^2 R(T)
P_braking = positive(-P_shaft)
L = P_drive_loss + P_copper + P_iron + P_braking
P_electrical = positive(P_shaft) + P_copper + P_iron + P_drive_loss
C dT/dt = L - Q_out
Q_out = h (T - T0)
```

`positive(P) - positive(-P) = P`, so `P_electrical - P_shaft = L` even through the
smooth braking transition. The default smoothing is `1e-6 W`. Both branches have
their own thermal state and are summed only after their balances are formed.
The constant iron loss and fractional drive loss are explicitly assumed; phase
and RMS-current conventions cannot be mixed with this equivalent motor.
Resistance, torque constants, and thermal parameters are declared reference
values, not manufacturer data. Do not add a second fixed-efficiency penalty for
losses already represented here.
[Motor electrical and thermal model][motor], [lumped motor thermal circuit][thermal]

## Entropy and wake accounting

For each thermal lump and its ambient heat path, irreversible generation is

```text
sigma_internal = L / T
sigma_heat_transfer = Q_out (1 / T0 - 1 / T)
```

The second expression is nonnegative for passive heat transfer. The lump's
entropy change is `C log(T_final / T_initial)`; it is not the same quantity as
entropy generated. These heat-transfer signs follow the hot and cold reservoir
entropy balance. [NASA second-law discussion][nasa-entropy]

The implemented boundary includes complete relaxation of outgoing aerodynamic
expenditure at `T0`, giving its eventual entropy as

```text
sigma_air = [sum(P_induced + P_profile)
             + P_drag_diss + P_angular_diss] / T0
sigma = sum(sigma_internal + sigma_heat_transfer) + sigma_air
```

That is an explicit ambient-relaxation approximation. It does not assert that
organized wake kinetic energy becomes entropy at the rotor disk. The near-rotor
inflow-energy state is retained separately and its derivative is already present
in shaft demand. It must not also be charged as dissipation. A different
finite-relaxation model for outgoing wake energy would need another explicit
energy balance, for example

```text
dE_wake/dt = P_wake_in - E_wake / tau_wake
sigma_wake = E_wake / (tau_wake T0)
```

That alternative is not the current outgoing-wake model. Do not add the
instantaneous wake term and its delayed counterpart together.
Shaft power divided by `T0` is not this model: shaft power contains useful work
and does not account for electrical losses or changes in stored energy.

## Finite horizons and terminal states

For a thermal lump initially at the declared temperature, complete passive
cooling after shutdown has the exact remaining entropy cost

```text
B_thermal(T) = C [(T - T0) - T0 log(T / T0)]
Phi_thermal(T) = B_thermal(T) / T0
```

The implemented terminal completion cost sums this term for both thermal lumps
and adds `E_inflow_final / T0`. Include corresponding
initial-state offsets when reporting an absolute completed-process comparison.
This prevents the optimizer from hiding future entropy in a hot motor or in wake
energy just beyond the horizon. A temperature limit alone is insufficient.

The reported completed-process quantity is
`S_generated + Phi_final - Phi_initial`. Here completion refers specifically to
passive motor cooling and relaxation of the modeled inflow store. It is not a
simulated motor shutdown, rotor spool-down, battery discharge cycle, or complete
aircraft-and-ground entropy balance. Governed rotor speeds remain fixed over
the mission; rotor rotational energy and its shutdown dissipation are excluded.

For one passive lump the independent accounting identity is

```text
integral(sigma_internal + sigma_heat_transfer) dt
  + Phi_thermal(T_final) - Phi_thermal(T_initial)
  = integral(L) dt / T0
```

This identity also explains an important limitation: with complete relaxation,
one ambient reservoir, and equal mechanical endpoints, entropy optimization can
be equivalent to dissipated-energy optimization. Temperature-dependent losses
and thermal limits still affect the solution. Do not claim a distinct benefit
over electrical-energy optimization merely because an objective is labeled
entropy. Separate energy and entropy balances are the basis for such control
problems. [Philipp et al., energy/entropy/exergy optimal control][ph]

## Constrained NMPC completion criteria

The native C++ controller uses CasADi 3.7.2 and IPOPT direct multiple shooting:
20 intervals of `0.1 s`, a `2 s` horizon, and a control decision every `0.1 s` of
simulated time. Airborne feedback decisions attempt an optimization; the initial
trim observation hold and planned landing support make no optimizer attempt. The predictor integrates the nonlinear
rotor/rigid-body/thermal ODE using RK4 with
five `0.02 s` substeps per interval and quaternion normalization. It applies the
first accepted command and replans from the current simulated state. Exact state
and current wind are available; the wind forecast persists the current value
over the horizon. No state estimator or hardware sensing performance is claimed.

### Causal mass identification shared by both controllers

The simulator supplies exact velocity, but the controller starts with the nominal
mass, not the scenario's actual mass. On each completed airborne step, the
observer evaluates rotor and drag force from the observed state and the nominal
constitutive force map. With gravity excluded from this force,

```text
q = (v_after - v_before) / dt + g e_z
F = m q
m_estimate = sum(q dot F) / sum(q dot q)
```

The fit accumulates only finite, informative, contact-free observations in the
declared mass range `[1, 3] kg`. Contact at either end rejects the whole interval.
It starts at `2 kg`; the first observation can identify mass in this noise-free
simulator because hover force is informative through gravity. This is not a
demonstration of identification with uncertain force maps or noisy sensors.

The same causal estimate feeds the PID baseline, PID fallback and landing
support, NMPC prediction, and its PID initial guess. Each optimization freezes
the current estimate over its horizon. One scalar runtime parameter scales mass
and all three principal inertias, matching the declared perturbation family;
inertia is not independently identified. A single compiled solver serves all
estimates. The symbolic acceleration transformation is checked against ordinary
numerical dynamics with directly scaled parameters. The physical entropy cost,
mission, actuator limits, and flight acceptance criteria remain unchanged.

The objective is only the integral of the stated physical entropy-generation
rate plus terminal thermal/inflow completion cost, in `J/K`. There is no
tracking-error cost, slack penalty, or control-effort term. Tracking and physical
requirements are hard constraints. A precomputed path, PID gain adjustment, or
relabeling of a tracking cost would not satisfy this contract.

The current predictor imposes the following bounds at future `0.1 s` knots.
The initial state is fixed to the current measurement, so a solver cannot repair
a violation that has already occurred by changing its initial decision state.

| Requirement | Bound |
| --- | --- |
| Internal predictor position corridor | ±`[0.40, 0.40, 0.20] m` about the common reference |
| Horizon endpoint position | ±`0.10 m` per coordinate before the landing phase |
| Horizon endpoint velocity | ±`0.25 m/s` per coordinate relative to reference |
| Horizon endpoint body rates / heading | ±`0.5 rad/s` per axis; `|qz| <= 0.10` |
| Free-flight altitude | `0.20–6.20 m` |
| Internal tilt / body rates / airspeed | `0.40 rad` (physical `0.52 rad`); ±`2.5 rad/s` per axis; `4.7 m/s` (physical `5 m/s`) |
| Main/tail motor temperatures | `293.15–353.15 K` |
| Electrical power | `0–650 W` |
| Absolute main/tail shaft power | `550/90 W` |
| Absolute main/tail current | `35/15 A` |
| Near-hover descent envelope | `V_main_axial >= -0.50 × v_main_inflow` |
| Command range | main `[0, 0.22]`, cyclic `[-0.18, 0.18]`, tail `[0, 0.35] rad` |
| Command increments | The sampled bounds specified above |

The independently audited outer mission corridor remains
±`[0.60, 0.60, 0.35] m` for both controllers. The tighter internal prediction tube
does not relax any executed-flight acceptance threshold.

Additional settling constraints apply at the horizon endpoint. The derivative
of the free-flight model is evaluated at the terminal state with the last held
command and the persisted wind forecast:

| Terminal derivative | Absolute component limit |
| --- | --- |
| World translational acceleration | `0.6 m/s²` |
| Body angular acceleration | `0.8 rad/s²` |
| Actual main, longitudinal, lateral, tail pitch rates | `[0.03, 0.15, 0.15, 0.08] rad/s` |
| Main and tail induced-flow rates | `0.4 m/s²` |
| Longitudinal and lateral flap rates | `0.15 rad/s` |

These bounds discourage ending a horizon with rapidly changing rotor states or
collapsing support. Translational acceleration is bounded in world coordinates,
not relative to the reference acceleration. The constraints are endpoint-rate
conditions; they do not establish an invariant terminal set, recursive
feasibility, closed-loop stability, future thermal viability, or contact
equilibrium. Their effect on flight still requires validation.

For predicted time `t >= 65 s`, the approach constraints require horizontal
position error at most `0.05 m` per axis, horizontal velocity error at most
`0.05 m/s` per axis, tilt at most `5°`, and body rates at most `0.10 rad/s` per
axis. These aim to prepare a settled handover to PID support.

For every predicted knot at mission time `t >= 68 s`, position tolerances tighten
to ±`[0.10, 0.10, 0.025] m`, velocity error to `0.10 m/s` per coordinate, body
rates to `0.20 rad/s` per axis, and tilt to `5°`. The final reference is
`[0, 0, 0.20] m` at zero velocity. Free-flight feasibility alone does not establish
touchdown: executed trials must additionally have ground contact and satisfy the
landing and terminal-state audits. The approach bounds also remain active at
these later predicted times. Executed-flight landing criteria remain unchanged:
the tighter internal approach constraints do not replace their independent audit.

During the final descending mission phase, at scheduled decisions with `t >= 58 s`,
the simulation checks the actual minimum skid clearance and reference clearance:

```text
c_skid = z - 0.175 Rzz - 0.025 |Rzz| - 0.28 |Rzx| - 0.185 |Rzy|
c_reference = z_reference - 0.20
margin = 0.15 + 0.5 max(0, -v_z)
handover when min(c_skid, c_reference) <= margin
```

Here `R` maps body coordinates to world coordinates, distances are metres, and
`0.5` is a closing-time allowance in seconds. The skid expression is the minimum
height of the two modeled box skids. Once the guard fires, PID landing support
is latched for the remaining mission. The free-flight optimizer is no longer
attempted or credited with controlling contact. The full mission's energy and
entropy accounting continues through these PID-controlled intervals.

Handover records include its time, clearances, margin, position, velocity, tilt,
body rates, and reason. `settled_at_handover` requires horizontal position error
and speed within `0.05 m` and `0.05 m/s` per axis, tilt within `5°`, and body rates
within `0.10 rad/s` per axis. The geometry guard does not wait for these conditions
if contact is approaching; an unsettled handover is reported explicitly. It is
not concealed by relabeling PID support as a successful solve. The unchanged
outer flight and landing criteria judge the resulting trajectory.

Every candidate is checked outside IPOPT in ordinary double precision. Only its
control sequence is reused; the nonlinear ODE is integrated again from the
current state. The checks compare shooting states, reevaluate original knot
constraints, recompute entropy and terminal storage, and check physical bounds
at `0.02 s` samples. The tighter internal `0.40 rad` tilt and `4.7 m/s` airspeed
limits are also constrained and independently checked at every future RK4 output,
separately from physical violations. These reserves provide headroom for plant
and changing-wind prediction errors; they are heuristic margins, not a robustness proof.
Physical bounds include actual pitch, nonnegative inflow,
temperature, clearance, tilt, rates, airspeed, currents, and powers. Corridor and
landing constraints remain knot checks in this predictor. These sampled checks
are not a proof of continuous-time constraint satisfaction.

Candidate acceptance requires finite cost, dimensionless constraint violation
at most `1e-4`, and scaled state/shooting discrepancy at most `1e-4`. State scales
are `0.1 rad` for pitch/flap, `5 m/s` for inflow, `20 K` for temperature, and one
in the remaining state units. The NLP decision coordinates use these same state
scales and a `0.1 rad` scale for all four command coordinates. The equations,
returned trajectories, and physical bounds are converted consistently between
scaled coordinates and SI variables; the objective remains in `J/K`. IPOPT's
additional automatic NLP scaling is disabled. The independently recomputed
objective discrepancy must not exceed `1e-4 J/K`; it is recorded and audited again
by flight evaluation. Rechecking the original constraints also rechecks the
terminal derivative bounds on the independently integrated endpoint.

The initial guess is a fresh PID rollout or a stored multiple-shooting plan,
with inputs clipped to command/increment bounds. Fresh PID states are integrated
from the current state. For a stored plan, the state nodes and commands are
shifted together, the initial node is replaced with the measured state, and only
new tail nodes are integrated. The initial guess can therefore contain shooting
defects for the nonlinear solver to correct; it is not an executable prediction.
Separately, a full rollout of the shifted controls from the current state checks
whether that plan is a feasible objective comparator. Each stored plan carries
its originating simulated timestamp. Its shift
is computed from the elapsed time on the `0.1 s` prediction grid: a repeated
same-time solve uses stage zero, and elapsed fallback intervals age the plan by
their full duration. Plans from a future time or plans at least one horizon old
are discarded. A valid shifted plan holds its final command for any newly added
tail intervals. Every accepted candidate still undergoes the full independent
rollout, physical checks, and shooting-defect checks specified above.

A finite rejected candidate may be retained solely as an initial guess for a
later solve if its independently reintegrated constraint violation is at most
`0.01`. Retention favors feasibility before objective value: an infeasible
candidate cannot displace a feasible available warm plan just because its cost
is smaller. Rejected plans never supply executable commands or displayed
accepted predictions, and their dual multipliers are discarded. The `0.01`
retention bound does not relax the `1e-4` executable acceptance tolerance or the
objective/dynamics checks. Diagnostics identify the warm-start source, plan
timestamp, shift, and whether a rejected candidate was retained.

Dual multipliers are reused only for an accepted plan in the same, unshifted
prediction-grid stage. With the regular `0.1 s` solve schedule this means a
repeated solve at the same timestamp; the grid-stage check also permits elapsed
times shorter than one interval. Shifted plans and retained rejected candidates supply primal initial
guesses only; their multiplier rows are not assumed to match the new horizon.
A candidate must not exceed the best feasible seed's entropy cost by more than
`1e-7 J/K`; if neither seed is feasible, this objective comparison cannot certify
improvement and feasibility is evaluated independently. Seed feasibility for this
comparison includes the original knot constraints and the same `0.02 s` physical
checks used for accepted candidates. A warm start is a numerical initial guess,
not evidence that PID is the executed NMPC action.

IPOPT currently uses at most 150 iterations and a `10 s` CPU budget per solve,
an exact Hessian, adaptive barrier updates, and primal/dual warm-start settings.
Its desired and acceptable constraint tolerances are both `1e-8`, tighter than
the executable `1e-4` independent audit to allow for shooting-defect amplification.
Its native JIT compilation uses `-Og -fno-inline`. This reduced measured cold
compilation time while preserving the same model and derivatives.
Its CPU limit is not a wall-time deadline and a final iteration can exceed it.
The synchronous simulation waits for the solve; no claim of running the
controller in real time follows from a successful simulated flight. Record total
solve wall time, whether it fits the `0.1 s` simulated control interval, solver
termination, objective discrepancy, constraint/dynamics residuals, and all
fallback activations. A feasible budget-limited iterate may be accepted while
`converged` remains false. `Solve_Succeeded` and `Solved_To_Acceptable_Level` are
reported separately as convergence outcomes. Neither proves a global optimum.

Rejected, infeasible, or failed solves explicitly activate the PID fallback and
clear the displayed accepted prediction. The fallback, accepted-solve count,
converged-solve count, and action difference from the PID seed must remain visible.
Failed solves cannot be counted as successful NMPC merely because fallback flies
the mission.

Planned landing support is a separate no-attempt decision with
`optimizer_attempted=false`, `applied_controller=pid_landing_support`, and
`status=landing_support`. It has no accepted prediction, objective, seed cost,
candidate residuals, or action-difference value; these diagnostics are null.
Its solve time is zero and it is neither accepted nor converged. Actual attempts
use `applied_controller=entropy_nmpc` when accepted or `pid_fallback` when rejected.
The single initial trim hold has `status=observation_hold` and
`applied_controller=trim_observation_hold`, with the same null solver fields and
zero solve time. Its separate counter is `observation_hold_decisions`.
Every transition from NMPC to PID resets the PID integral and initializes its
previous input from the actual command, including rejection fallback and the
planned landing handover.

## Baseline and fair comparisons

Both controllers use the same 70-second takeoff, hover, waypoint, gust-recovery,
return, and landing reference, physical plant, disturbance schedule, initial
state, and physical limits within a paired scenario. Model metadata publishes
the baseline coefficients: position `Kp=[1.45,1.45,6.0]`,
`Kd=[2.35,2.35,4.2]`, `Ki=[0.3,0.3,2.0]`, attitude
`Kp=[1.30,1.80,0.95]`, and `Kd=[0.40,0.58,0.40]`. The baseline includes
wind-drag, tail-force, gyro, and damping feedforward and inverse rotor allocation;
it is not an intentionally crippled comparator. Its position integral is bounded
by ±`2 m s`, horizontal acceleration by `3.5 m/s²`, and vertical acceleration by
`[-4,5] m/s²`.

The PID landing policy unloads up to 8% of commanded main thrust as the descending
reference moves from `0.22` to `0.20 m`, while actual altitude is below `0.30 m`.
This transfers support to the skids without changing the reference. The same
policy is present in the PID warm-start controller and fallback. It must be
disclosed because contact and terminal mechanical work affect comparisons.

Evaluate nominal flight, an alternate wind phase, added gusts, and ±20% plant
mass/inertia perturbations. Both controllers start with nominal parameters and
receive the same causal mass estimate; all other constitutive parameters remain
nominal. These trials evaluate adaptation within the declared common-scaling
family, not robustness to every unmodeled aerodynamic or inertial error.
The disabled-controller scenario is a separate negative control. For every pair
report generated and completed-process entropy, electrical energy, trajectory
error, peak torque/power/current/temperature, constraint violations, wind work,
contact work, terminal rigid-body state, motor temperatures, aggregate wake/thermal
storage, and optimizer participation. The current
participation criterion is at least 90% NMPC-applied decisions over **all**
scheduled control decisions, including the initial observation hold and planned PID landing support. A full
70-second run has 700 decisions. The identities are
`control_decisions = solves + landing_support_decisions + observation_hold_decisions` and
`solves = accepted_solves + fallback_solves`, where `fallback_solves` means actual
rejected optimization attempts. Overall non-NMPC time includes the trim hold,
rejected attempts, and planned support. `accepted_solves / solves` is also reported, but it is not
the participation denominator. Convergence fractions, solve-time averages, and
deadline misses count actual optimizer attempts only. Action changes and
objective reductions relative to a feasible seed are also required for each
enabled entropy trial to pass, as evidence that optimization influenced control.
PID and disabled-controller trials are exempt from that optimizer-contribution
requirement. Mission success alone is insufficient.

Savings can be claimed only when both paired trials pass the common mission,
physical, accounting, and applicable participation criteria. Their residual
mechanical endpoint-energy difference and contact-work difference enter the
comparison's conservative uncertainty allowance, as do numerical accounting
residuals and left-sum versus trapezoidal quadrature differences. A reported
savings must exceed that allowance. Ground-contact entropy is not included or
inferred by dividing contact work by ambient temperature. Wind work is recorded
separately; wind-assisted operation remains part of this prescribed scenario.
Passing the evaluation and demonstrating savings are separate report fields.

## Independent verification

`tests/thermodynamics.cpp` checks identities independently from controller gains
and mission success. Its five groups cover:

1. First- and second-law balances, wind work, stored mechanical/thermal/inflow
   energy, terminal offsets, and nonnegative irreversible terms in test fixtures.
2. Analytic passive cooling, thermal entropy change, and terminal completion cost.
3. Hover demand, positive profile power for spinning rotors, and nonzero cyclic
   mechanical work with consistent wind-port signs.
4. Finite-difference energy derivatives converging with timestep refinement.
5. Negative shaft work with explicit non-regenerative braking, and the increase
   in electrical demand caused by temperature-dependent winding resistance.

`tests/optimizer.cpp` separately requires a feasible hover action that differs
from an independently calculated PID action and reduces the physical entropy
objective beyond numerical discrepancy. It permits a fixed maximum of three
finite-budget nominal attempts without relaxing its acceptance thresholds. It
reintegrates the applied action, checks prediction bounds, requires rejection of
an impossible `500 K` motor-temperature fixture, and verifies reset/fallback
counters. A passing unit fixture would establish optimizer contribution for that
fixture, not a full mission or savings against the baseline.

`tests/evaluate.cpp` audits the executed plant every `0.01 s`, including actual
pitch, command increments, corridor, post-68-second landing bounds, and physical
limits. It recomputes energies, entropy, and stored-state changes instead of
trusting the displayed totals. Its ground-contact scope allows center altitude
down to `0.18 m` for the compliant floor and checks the main descent envelope only
while airborne. The contact exception does not excuse excessive current, power,
temperature, attitude, or other vehicle limits.

The flight evaluator audits each scheduled control decision once, recomputes
participation, checks decision timestamps and attempt counters, and distinguishes
rejection fallback, planned landing support, and convergence. The historical
`unique_solves` array is retained for compatibility but now contains one record
per control decision, with a decision index and `optimizer_attempted` flag.
Null no-attempt diagnostics are validated without treating them as zero-valued
optimizer results. It requires
accepted candidate objective discrepancies no greater than `1e-4 J/K` and checks
source fingerprints. Deterministic reset replay is tested for PID only: CPU-budget
optimization and timing are not assumed to be byte-identical across runs.

The tests establish neither correctness over every continuous state nor
experimental fidelity. Flight validation still requires complete matched
mission reports, disturbance/model perturbations, terminal-state audits, no
hidden energy storage, independent objective checks, and explicit handling of
infeasible optimizations. Source hashes bind evidence to the compiled model and
evaluator; they do not substitute for validation. Partial scenario runs must
remain labeled partial, and a stale PID-only artifact cannot validate this NMPC.

[ndarc]: https://rotorcraft.arc.nasa.gov/Publications/files/Johnson%20TP-2015-218751.pdf
[motor]: https://www.mathworks.com/help/sps/ref/dcmotor.html
[thermal]: https://www.mathworks.com/help/simscape/ug/motor-thermal-circuit.html
[nasa-entropy]: https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/second-law-entropy/
[ph]: https://arxiv.org/html/2306.08914v2
