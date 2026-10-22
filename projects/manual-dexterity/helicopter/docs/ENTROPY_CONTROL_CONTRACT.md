# Minimum-entropy H-infinity control contract

This project implements the minimum-entropy H-infinity method associated with
Mustafa and Glover for an explicitly simulated helicopter. This is the design
and acceptance contract. **Synthesis, flight performance, and comparative results
for the corrected method are pending.**

The previous thermodynamic-entropy NMPC study optimized a different quantity.
Its contract, manuscript, published PDF, and project README were copied into
[historical-thermodynamic](historical-thermodynamic/README.md) before this revision.
Successful historical simulation and its small electrical/thermal savings are
unrelated to whether the book's minimum-entropy criterion has been implemented.

## Objective and meaning

Let `w` denote exogenous disturbance/noise, `u` incremental pitch commands,
`y` controller measurements, and `z` weighted regulated outputs. Define

    xdot = A x + B1 w + B2 u
    z    = C1 x + D11 w + D12 u
    y    = C2 x + D21 w + D22 u
    u    = K(s) y
    Tzw  = P11 + P12 K (I - P22 K)^(-1) P21

For a prescribed feasible `gamma > 0`, require internal stability,
`||Tzw||_infinity < gamma`, and minimum entropy

    I_gamma(Tzw) = -gamma^2/(2 pi) integral_{-infinity}^{infinity}
                  log det[I - gamma^(-2) Tzw(jw)* Tzw(jw)] dw .

The star is conjugate transpose. This continuous-time integral uses a stable,
strictly proper `Tzw`. Nonzero closed-loop feedthrough generally makes the
unweighted integral divergent and cannot silently be ignored. Publish all signal
scales and weights so `Tzw/gamma` is consistently normalized. This is a
closed-loop frequency-response functional, not physical entropy in J/K or the
entropy of a tracking-error histogram.

The H-infinity norm limits worst-case induced L2 gain for the stated channels and
zero initial state. Entropy also accounts for response away from the worst
frequency: `I_gamma >= ||Tzw||_2^2`, with limit `||Tzw||_2^2` as
`gamma -> infinity`. Robustness to plant uncertainty additionally requires an
explicit uncertainty interconnection, normalization, and norm bound. Five
successful wind/mass simulations alone are not that certificate.
[Definition and centralized problem, pp. 1–4][lessard]

Some earlier papers maximize the log-determinant integral with the opposite
sign. The convention above is positive and minimized. Neither convention allows
motor heat, actuator effort alone, or an arbitrary tracking cost to replace
`I_gamma`. The connection to combined H-infinity/LQG concerns particular weights
and an auxiliary cost, not every mixed H2/H-infinity optimization.
[Original relation paper][mustafa1989]

## Plant, initialization, and linearization

The validation plant remains the native 23-state rotor/electrical/thermal model
with MuJoCo rigid-body/contact integration. No controller sets the pose directly.
The reference vehicle is uncalibrated: mass `2 kg`, inertia
`[0.060,0.095,0.090] kg m²`, main/tail radii `0.55/0.11 m`, governed rotor speeds
`180/650 rad/s`, ambient temperature `293.15 K`.

| Native indices | Quantity |
| --- | --- |
| 0–2 | World position, m |
| 3–5 | World velocity, m/s |
| 6–9 | Body-to-world unit quaternion, wxyz |
| 10–12 | Body angular velocity, rad/s |
| 13–16 | Actual collective, longitudinal/lateral cyclic, tail pitch, rad |
| 17–18 | Main/tail induced-flow speed, m/s |
| 19–20 | Longitudinal/lateral flap angle, rad |
| 21–22 | Main/tail motor temperature, K |

Synthesis uses the nominal `2 kg` free-flight hover point and local attitude error
coordinates. The mechanical synthesis state has 20 components:
position, velocity, three local attitude errors, body rates, four actual pitch
perturbations, two inflow perturbations, and two flap perturbations. Thermal
states remain in the nonlinear plant and its limits/accounting, but are outside
this flight-control linearization. Do not treat four unit-quaternion components
as four independent linear perturbations.

Three stable position-weight filters augment this state to **23 synthesis and
controller coordinates**: `eta_dot = position_error - 0.04 eta`, with `eta` in
metre-seconds and zero initialization. These are low-frequency weights, not exact
integral-action guarantees. The hover Jacobian is computed by centred finite
differences of the native mechanical dynamics; an independent directional
perturbation check includes actuator, inflow, and flap responses. The trim,
Jacobian, channel labels, and physical-coordinate scales are exported in
`model.hinf`.

The generalized plant has **18 disturbance inputs, 4 control inputs, 12 measured
outputs, and 27 regulated outputs**. Its normalized state `x` maps to physical
deviations through `diag(state_scales) x`. The first 23 regulated outputs are
the following physical quantities times the stated weights; the last four are
normalized incremental pitch commands.

| Coordinate group | Physical coordinate scales | Regulated-output weights |
| --- | --- | --- |
| Position xyz, m | `[0.5,0.5,0.5]` | `[3,3,5]` |
| Velocity xyz, m/s | `[1,1,1]` | `[1.5,1.5,2]` |
| Local attitude xyz, rad | `[0.1,0.1,0.1]` | `[3,3,2]` |
| Body rates xyz, rad/s | `[1,1,1]` | `[0.35,0.4,0.3]` |
| Collective, longitudinal/lateral cyclic, tail pitch, rad | `[0.05,0.05,0.05,0.1]` | `[1,1,1,1]` |
| Main/tail inflow, m/s | `[3,3]` | `[0.03,0.03]` |
| Longitudinal/lateral flap, rad | `[0.1,0.1]` | `[0.3,0.3]` |
| Position filters xyz, m s | `[0.5,0.5,0.5]` | `[1.2,1.2,2]` |

Weights have reciprocal physical units so the regulated channels are normalized.
These are illustrative design choices, not identified aircraft parameters.
Incremental control scales are `[0.05,0.08,0.08,0.10] rad`; thus the final four
outputs penalize those normalized commands. The first six disturbance channels
inject world translational accelerations with scales `[0.8,0.8,1.2] m/s²` and
body angular accelerations with scales `[1.5,1.5,1.0] rad/s²`. The remaining
twelve are measurement noise, with physical channel scales `0.005 m` for
position, `0.02 m/s` for velocity, `0.002 rad` for local attitude, and
`0.01 rad/s` for rates. `D21=[0 I12]` in normalized measurement coordinates;
the process-noise columns and sensor-noise columns are separate. These declared
acceleration channels are not a certified bound on nonlinear wind aerodynamics.

Expose every generalized-plant matrix, state/channel ordering, physical scaling,
tracking/actuator weights, and disturbance/noise amplitude. Intended measurements
are position, velocity, local attitude, and body rates. Disturbances include
force/moment-equivalent accelerations and sensor noise. Changing weights changes
the question; gamma/entropy comparisons require the same normalized channels.

Both controllers receive the same deterministic simulated sensor-noise process.
Neither command calculation receives privileged true wind. A synthesis noise
channel alone is not noisy-sensor validation: actual injected noise, seeds,
amplitudes, and sensor mapping must appear in flight evidence. Design mass is
fixed nominal, with no direct actual-plant mass input. The existing causal mass
estimator is diagnostic only in this comparison; its exact force/state assumptions
do not establish adaptive-control performance.

Each paired trial starts at `[0,0,0.22] m` with spooled rotors and the same prepared
state. The simulator computes mechanical trim for the actual plant mass. Motors
start at ambient by assumption. Neither controller is credited with discovering
trim, recovering from an unprepared unknown-mass start, or performing spool-up.
Any initial hold stays inside the 70-second clock and physical accounting;
initial wake storage remains explicitly recorded.

## Central output-feedback synthesis

Use the centralized DGKF solution at a specified feasible gamma, with its central
free parameter fixed to zero. Do not presume that every H-infinity solver, a
fixed-order fit, or a gamma-optimal search emits this minimum-entropy realization.
Check the chosen formula and assumptions directly. For normalized `D11=D22=0`:

    D12' C1 = 0,  D12' D12 = I
    B1 D21' = 0,  D21 D21' = I
    A' X + X A + C1' C1 + X(gamma^-2 B1 B1' - B2 B2')X = 0
    A Y + Y A' + B1 B1' + Y(gamma^-2 C1' C1 - C2' C2)Y = 0
    F = -B2' X
    L = -Y C2'
    Z = (I - gamma^-2 Y X)^(-1)
    Ak = A + B2 F + Z L C2 + gamma^-2 B1 B1' X
    Bk = -Z L,  Ck = F,  Dk = 0 .

Required stabilizing Riccati solutions are positive semidefinite and satisfy
`rho(XY) < gamma²`. Check stabilizability/detectability and channel/rank conditions.
Publish whitening or loop shifting and its inverse physical-command mapping.
Exact state availability does not remove this output-feedback formula's
nonsingular noise-channel assumption. [DGKF, pp. 834–835 and 840][dgkf]

ARE synthesis and gamma selection occur offline. A feasibility search saves its
bracket, tolerances, margin, and rejected cases. Report entropy minimization at
the chosen gamma; do not claim global gamma optimality without separate evidence.

The implementation uses native LAPACK ordered real Schur decompositions of the
two Hamiltonians. PBH singular-value checks cover imaginary-axis and unstable
modes for both stabilizability and detectability pairs. It rejects normalized
ARE residuals above `1e-8`, a Hamiltonian real-part separation below `1e-8`,
incorrect stabilizing branches, non-positive-semidefinite solutions beyond
`1e-8 (1+||X||_F)`, and coupling without a relative `1e-8` margin. The numerical
gamma search uses 30 bisection steps and selects `1.5` times its feasible upper
endpoint. An infeasible numerical-search endpoint is not a proven optimal lower
bound. The published gamma sweep uses the same plant and weights throughout.

Online execution uses fixed dynamic-controller matrices:

    xk_dot = Ak xk + Bk y
    u_delta = Ck xk + Dk y
    u_requested = u_trim + declared_reference_feedforward + u_delta .

It neither solves a nonlinear trajectory program each step nor integrates the
frequency-domain objective during flight. Disclose sampling, discretization,
reference injection, initialization/reset, and feedforward. Verify the sampled
controller separately: a continuous-time certificate is not a sampled-data
certificate. An Euler update is not automatically stable. The linear theorem
does not automatically cover added nonlinear feedforward or changing references.

The controller state is initialized to zero. At `100 Hz`, the runtime outputs
`Ck*xk` from the current state and then advances the state using the exact
zero-order-hold matrices `Ad,Bd` for the held measurement over `0.01 s`.
The independently assembled sampled plant/controller closed loop is checked
for stability. This is a sampled stability check, not a sampled H-infinity gain
certificate. The native test replays the exported `Ad,Bd,Ck` against unsaturated
runtime updates to check normalization and output/update ordering.

Reference handling subtracts position and velocity targets from the measured
channels. It defines `a_ff = reference_acceleration + linear_drag*reference_velocity/m_nominal`,
subtracts the small-angle attitude target
`R_trim' * [-a_ff_y/g, a_ff_x/g, 0]`, and adds nominal trim-command sensitivity
times `a_ff_z` to the physical input. These fixed nominal feedforward rules are
outside the regulator certificate. Physical magnitude and sampled slew clipping
are followed by a known applied-minus-requested input correction, integrated
through the controller's plant-input matrix. Its correction is zero in the
unsaturated certified realization. The held initial trim synchronizes limiter
history without initializing the controller from hidden plant states.

The central dynamic coordinates are risk-sensitive controller coordinates, not
unbiased state estimates. Runtime `measurement` is the twelve-channel physical
error before whitening. `controller_coordinate_scaled` and
`measurement_reconstruction` are explicitly labelled coordinate reconstructions;
`weighted_output` has a `weighted_output_kind` stating that it is reconstructed
as `C1*xk+D12*Ck*xk`, not the actual plant's regulated output `z`.

## Independent synthesis evidence

The native artifact must support independent recomputation:

1. Operating point, generalized-plant/controller matrices, gamma, weights,
   dimensions, dependencies, and source/configuration fingerprints.
2. Normalized ARE residuals, symmetry/PSD errors, stabilizing-branch eigenvalues,
   `rho(XY)/gamma²`, conditioning of `I-gamma^-2 YX`, and independently assembled
   closed-loop eigenvalues. Declare numerical tolerances.
3. A continuous-time gain upper bound checked through an independent bounded-real,
   Riccati, or equivalent state-space calculation. A frequency sweep is useful
   visualization, not a bound over all frequencies. Save residuals and strict margin.
4. Squared H2 norm from a Lyapunov equation and entropy from the applicable
   state-space formula, cross-checked by frequency quadrature with range,
   refinement, and tail treatment. Check `I_gamma >= H2_squared` and the
   large-gamma limit. Physical-entropy telemetry is not this cross-check.
5. Verification of the emitted central realization; tests for sign/order/
   normalization mistakes and infeasible-gamma rejection. A numerical certificate
   has finite tolerances and is not exact-arithmetic proof.

Reject nonfinite matrices, unstable closed loops, wrong Riccati branches,
violated coupling, or unverified gain bounds. Failed synthesis is unavailable;
a substitute PID command must never be labelled H-infinity.

For the implemented weights, the native synthesis check obtains
`gamma = 1.513435045`, a numerical bounded-real gain upper bound
`1.338888058`, sampled frequency peak `1.338887367`,
`I_gamma = 17.504639378`, and `H2 = 3.636691622`. The normalized X/Y ARE residuals
are about `1.51e-17` and `4.29e-14`; `rho(XY)/gamma² = 0.057857952`.
The continuous closed-loop spectral abscissa is `-0.04 /s`; the sampled
closed-loop spectral radius is `0.999600080`. These are floating-point numerical
results for the disclosed LTI model, not exact arithmetic or nonlinear guarantees.

The artifact exports `Pgamma`, its bounded-real residual, and the H2
observability Gramian for independent equation replay. A 2001-point logarithmic
quadrature from `1e-5` to `1e6 rad/s` gives entropy `17.505047288`, versus
`17.506451050` for its 1001-point subset. A conservative resolvent estimate bounds
the omitted high-frequency entropy tail by `5.13e-5`; the low-frequency omission
is explicitly diagnostic. The fixed closed-loop map's entropy at `100*gamma`
approaches its squared H2 norm. These checks support the state-space calculation;
neither the frequency grid nor its refinement certifies a global gain bound.

## Nonlinear execution and attribution

The plant and intended controller update every `0.01 s`. Inputs remain physical
pitch angles. Magnitude and sampled-increment limiters act outside the linear
controller and must record interventions:

| Quantity | Bound |
| --- | --- |
| Collective / tail pitch | `[0,0.22] / [0,0.35] rad` |
| Each cyclic pitch | `[-0.18,0.18] rad` |
| Command increment rate factors | `[0.65,1.2,1.2,1.2] rad/s` |
| Actual pitch lag | `0.08 s` |
| Airborne altitude / tilt / airspeed | `0.20–6.20 m / 0.52 rad / 5 m/s` |
| Body rate, each axis | `2.5 rad/s` |
| Main/tail temperature | `293.15–353.15 K` |
| Electrical power | `0–650 W` |
| Absolute main/tail shaft power | `550/90 W` |
| Absolute main/tail current | `35/15 A` |
| Near-hover descent | `V_main_axial >= -0.50 v_main_inflow` |

Saturation, slew limits, anti-windup, reference/feedforward logic, and contact
create a system beyond the unconstrained LTI theorem. Publish each extra rule.
A limiter is not proof of constrained optimality.

Geometry-triggered PID landing support remains explicit. Record handover time,
position/velocity, attitude/rates, and whether its settling condition was met.
Show hold, fallback, and support in state/history and evaluation. Require
H-infinity application for at least 90% of **all 100 Hz mission intervals**;
the denominator is the full 70 seconds. This is a contribution gate, not a theorem.

Count total intervals, H-infinity application, initial hold, PID fallback,
PID landing, and disabled control. Each interval has one applied-controller
category. Saturation/anti-windup are extra flags. Timing includes the controller
update and limiter work; report mean, percentile, maximum, and misses against
the 10 ms interval. Simulation timing is not hardware qualification.

## Nonlinear evaluation and comparisons

Preserve five paired 70-second scenarios: nominal, alternate wind, extra gusts,
mass minus 20%, and mass plus 20% with gust, plus the disabled negative control.
Both modes use the same reference, plant, prepared condition, disturbances/noise,
actuator bounds, and outer criteria. Disclose controller-specific support.

Audit every `0.01 s`: finite state/commands, input magnitude/increments, zero
violations above `1e-4` normalized tolerance, and position corridor
`±[0.60,0.60,0.35] m`. Contact allows the existing `0.18 m` center-height floor
and exempts only the airborne descent-envelope check.
After `68 s` require position errors `[0.10,0.10,0.025] m`, velocity errors
`0.10 m/s` per axis, tilt `5 degrees`, and body rates `0.20 rad/s` per axis.
Enabled missions complete with contact, RMS error below `0.55 m`, peak error
below `1.4 m`, endpoint error below `0.25 m`, and final speed below `0.25 m/s`.
The disabled trial must fail tracking under the native negative-control criteria.
Do not loosen thresholds to obtain a pass.

Keep these conclusions separate:

- **Linear synthesis:** the central realization meets the stated linear-model
  entropy functional and H-infinity bound.
- **Nonlinear mission:** its sampled implementation and disclosed support pass
  the actual reference-vehicle trials.
- **Comparative benefit:** fixed-weight, matched tests measure tracking,
  disturbance response, effort, timing, and physical energy against PID or H2.
  An H2-limit baseline is needed to isolate the robustness/performance tradeoff;
  PID alone does not do that.

Report all scenarios, including failures. These tests do not establish universal
superiority, hardware efficiency, global nonlinear stability, or robust contact
control. Unspecified delays, actuator errors, and unmodeled aerodynamics remain
outside the claim.

## Physical accounting remains a measurement

The plant retains electrical, motor-thermal, drag, wake, and contact ledgers.
They are useful measurements but are not `I_gamma`. Labels must distinguish
`physical_entropy_j_per_k` from `hinf_entropy`. The historical contract contains
the constitutive derivation. For its fixed-ambient boundary:

    S_generated + Phi_final - Phi_initial
     = (E_electric + W_wind + W_contact - Delta E_mechanical) / T0 .

Retain independent 100 Hz balances, terminal thermal/inflow offsets, and
numerical/contact/mechanical-endpoint comparison allowances. Equal imposed wind
does not imply equal wind work. Changing the controller adds no battery chemistry,
servo electrical losses, rotor-speed transient, stall, calibrated vortex-ring flow,
or detailed ground aerodynamics. Physical-energy savings are a separate measured
result, not an automatic consequence of minimum-entropy H-infinity.

## Evidence, API, and provenance

Metadata/reports identify `objective_kind=minimum_entropy_hinf` and
`controller_family=central_dgkf`, plus a schema version. Separate synthesis status
from flight status. Include plant/controller/test hashes and a synthesis fingerprint
covering matrices, trim, weights, gamma, algorithm, tolerances, discretization,
and dependency versions.

Expose synthesis states/channels, matrices or a matched downloadable artifact,
gain/entropy/H2 results, and limitations through the model endpoint. State/history
expose actual controller category, requested/applied pitch, controller-state norm,
measurement/reference input, clipping/rate limits, support counters, and update
duration. Inapplicable optimization-iteration, prediction, or physical-objective
fields are absent or null; never suggest an online nonlinear solve occurred.

Separate synthesis verification, nonlinear flight/accounting, attribution,
comparisons, and HTTP/browser checks. Reject historical objectives or mismatched
fingerprints as current evidence. The evaluator produces paired trials in one native process; filtered runs are marked incomplete. They cannot establish the full-suite result. The API and viewer compare source/model provenance before displaying current evidence.

The current native report passes all 15 trials: seven per enabled controller plus the disabled negative control. The paired table and numerical certificates are published in the project README and artifacts. Historical reports retain their original controller identity.

## Sources and reading scope

The requested reference is [Mustafa and Glover, *Minimum Entropy H-infinity
Control*, Springer, 1990][book]. Chapter listings identify entropy on pp. 7–14 and
synthesis on pp. 15–33. The complete subscription monograph has not been accessed
here. This contract is checked against accessible original papers and explicit
equations, not presented as a reading of inaccessible book pages.

[book]: https://link.springer.com/book/10.1007/BFb0008861
[dgkf]: https://www.doyle.caltech.edu/images/doyle/2/20/TAC1989.pdf
[lessard]: https://arxiv.org/pdf/1403.5020
[mustafa1989]: https://doi.org/10.1016/0167-6911(89)90050-9
