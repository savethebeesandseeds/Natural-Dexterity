# Helicopter flight laboratory

A native C++20 laboratory for **Mustafa–Glover minimum-entropy H-infinity
control** of an explicitly simulated, uncalibrated reference helicopter.
The intended controller is the central DGKF output-feedback realization,
synthesized offline around nominal hover and updated as a dynamic linear
controller during flight. MuJoCo and the existing 23-state rotor/electrical/
thermal model provide the nonlinear validation plant.

**Corrected-method synthesis and all 15 flight trials pass.** The earlier
physical-entropy NMPC study solved a different problem. Its passed missions and
small thermodynamic savings do not demonstrate the book's control method.
The old [README, contract, manuscript, and PDF](docs/historical-thermodynamic/README.md)
were preserved before this revision.

The [active design and acceptance contract](docs/ENTROPY_CONTROL_CONTRACT.md)
defines the frequency-domain objective, synthesis assumptions, physical limits,
controller attribution, and required evidence. The [one-page brief](doc/README.md)
summarizes it. No new performance, robustness, real-time, or hardware claim is made
until matched evidence exists.

## Use and inspect the laboratory

Start the environment below and open <http://127.0.0.1:43118/>. The reference
mission has 70 seconds of takeoff, hover, waypoints, wind recovery, return, and
landing. A visible flight is not itself a validation result.

The viewer is being updated to the corrected method. It must show the actual
applied controller: minimum-entropy H-infinity, PID baseline, initial hold,
explicit fallback, or PID landing support. Changing mode starts a new run;
ordinary restart preserves the selected mode. Page loading does not issue a
flight command. The body pose is native plant output; blade spin is cosmetic.

The corrected control panels must distinguish the offline synthesis from online
execution: gamma, certified gain bound, frequency-response entropy, H2 norm,
Riccati residuals, stability, and synthesis provenance; then requested/applied
commands, controller state, clipping/rate limits, support counts, and update time.
There is no receding-horizon predicted trajectory or per-step optimizer solve
for this controller. Inapplicable historical optimizer fields must not imply one.

Flight graphs retain position/reference, velocity, attitude, rates, commands,
actual actuators, thrust, wind, inflow/flaps, and measured sensor errors.
Electrical, temperature, physical-entropy, and energy-balance graphs remain
separate plant measurements. They are not the H-infinity entropy objective.

Native history spans the whole current run, sampled every 0.1 simulation seconds.
Reset clears the run and changes its identifier. State/reference timestamps,
applied-command interval timestamps, and controller-update timestamps must remain
explicit. World force excludes gravity/contact, which MuJoCo applies separately;
body rates are angular velocity, not Euler-angle derivatives.

## Environment setup on Windows

Prerequisites: Docker Desktop running Linux containers with its WSL2 backend,
a supported NVIDIA Windows driver, PowerShell, and an Internet connection for
initial provisioning. No host Python, Node, CUDA toolkit, or C++ installation is
required. See Docker's [Windows GPU prerequisites](https://docs.docker.com/desktop/features/gpu/).

1. Inspect the existing Docker environment and GPU:

   ```powershell
   docker context show
   docker version
   nvidia-smi
   docker container ls -a
   ```

2. Build/start the managed environment from the project directory:

   ```powershell
   Set-Location 'C:\Work\Natural-Dexterity\projects\helicopter'
   .\helicopter.ps1 up
   ```

   The launcher first inspects `helicopter-dev`. It reuses a matching container,
   including a stopped one. If absent, it builds `helicopter:dev`, creates the
   configuration below, verifies it, starts it, and waits for `/healthz`.
   A conflicting container is preserved and reported by immutable ID; it is not
   replaced. Build or startup-test failure preserves the environment and prevents
   the service from being treated as ready. Use `logs` to inspect progress.
   The launcher allows up to 30 minutes for a first native build. Build products
   remain in mounted `build/`. H-infinity synthesis is performed before flight;
   it does not require the historical NMPC derivative-code JIT during playback.

3. Verify the runtime and GPU, then open the viewer:

   ```powershell
   .\helicopter.ps1 status
   .\helicopter.ps1 gpu
   Start-Process 'http://127.0.0.1:43118/'
   ```

   The GPU check loads the injected NVIDIA driver, uploads two values, executes
   a CUDA addition kernel, and checks the downloaded result in the managed
   container. **The plant, synthesis, and controller run on CPU.** GPU device access
   does not establish CUDA acceleration or a real-time deadline. Three.js
   rendering runs in the browser, independently of container GPU access.

4. Run the native evaluation and service checks, or inspect logs:

   ```powershell
   .\helicopter.ps1 test
   .\helicopter.ps1 check
   .\helicopter.ps1 logs
   ```

   `test` runs the native unit and paired-flight evaluation targets and writes
   the saved report. Inspect the report's objective
   identifier and source provenance: old thermodynamic-NMPC tests do not validate
   minimum-entropy H-infinity control.
   `check` exercises the running HTTP service and resets the live mission.
   After changing native source, stop/up so the service uses the rebuilt code;
   a report from a different model/evaluator build is not current evidence.

   The independent thermodynamic executable can also be run directly after a
   successful build:

   ```powershell
   docker exec helicopter-dev /workspace/build/native/helicopter-thermodynamics
   ```

5. Stop without deleting the container or its data:

   ```powershell
   .\helicopter.ps1 stop
   ```

   Start again with `up`. Restart policy is `no`, so start explicitly after a
   Docker or Windows restart. A new server process begins a fresh mission.

### Existing-container dependency update

The existing managed container has already been provisioned with the updated
`setup.sh` after the native CasADi dependency was added. This installs dependencies
in its writable layer without replacing the container or changing its mounts.
A fresh image build runs the same updated script through the Dockerfile and
contains the SDK from the outset.

For an existing matching environment that has not yet received that dependency
update, the dependency-only provisioning command is:

```powershell
docker exec helicopter-dev /bin/sh /workspace/setup.sh
```

Run it only against the inspected managed container, then use the separate
launcher to stop/up and compile the source. This is environment provisioning,
not a `setup.sh` lifecycle subcommand. Rebuilding an image alone does not change
an existing container. The launcher does not delete or replace containers or
volumes; any replacement requires a separately reviewed operation against the
exact immutable ID, preserving host data.

Ordinary source changes are bind-mounted. `run.sh` builds the native targets and
runs **the thermodynamic and causal mass-estimator CTests** before starting the server. Full
paired flight evaluation is the explicit `helicopter.ps1 test` operation, not a
startup side effect. A healthy viewer does not establish that flight evaluation
passed. Static viewer changes are visible after a page refresh.

## Approved container configuration

| Setting | Value |
| --- | --- |
| Container | `helicopter-dev` |
| Image | `helicopter:dev` |
| Base | `debian:12-slim@sha256:88200866dfff7ea7f5cbcb6ec7c8a701889efe6fe859fe64d6990e4b07ea4171` |
| Command | `/bin/bash /workspace/run.sh` |
| Working directory | `/workspace` |
| Bind mount | This project directory → `/workspace`, read/write |
| Named volumes | None |
| Published port | `127.0.0.1:43118` → `8080/tcp` |
| GPU | `--gpus all`; `NVIDIA_DRIVER_CAPABILITIES=compute,utility` |
| Additional devices / privileged mode | None / disabled |
| Network | Docker's default `bridge` |
| Restart | `no` |
| Process init / stop timeout | `--init`, 15 seconds |
| Labels | `io.waajacu.managed=true`, `io.waajacu.project=helicopter`, `io.waajacu.configuration=1` |
| Health | HTTP `/healthz`, every 10 seconds; 60-second initial grace |

The published service is bound to localhost. POST controls validate local
Host/Origin and JSON input. This is a `noindex` local review/development viewer.
Its CSP permits inline **style elements** for the Codex review overlay while
blocking inline scripts and style attributes; it is not a production hosting
configuration.

## Reproducible dependencies and script responsibilities

[`setup.sh`](setup.sh) **only installs dependencies and configures the environment**.
It accepts no subcommands and performs no container lifecycle, project builds,
tests, service startup, or data migrations. Debian packages use the signed
2026-09-01 snapshot of bookworm, bookworm-updates, and bookworm-security, installed
with `apt-get install --no-install-recommends`. Downloads are version-pinned and
SHA-256 checked:

| Dependency | Version | Purpose |
| --- | --- | --- |
| GCC / CMake / Ninja | Debian snapshot packages | Native C++20 compilation |
| MuJoCo | 3.12.0, Linux x86_64 | Rigid-body integration and contact |
| CasADi / bundled native libraries | CasADi 3.7.2 C++ SDK | Existing native numerical environment; bundled LAPACK/OpenBLAS supports synthesis |
| nlohmann/json | 3.12.0 | State, metadata, and evaluation JSON |
| cpp-httplib | 0.51.0 | Local HTTP service |
| Three.js | r170 / 0.170.0 | Browser rendering, served locally |

The pinned official CasADi archive is named
`casadi-3.7.2-linux64-matlab2018b.zip`. `setup.sh` extracts its C++ headers,
CMake metadata, CasADi/IPOPT libraries, native linear-algebra dependencies, and
shell JIT support into `/opt/casadi`. **No MATLAB interface/runtime, Python,
Python package installer, or Python control process is installed or invoked by
this procedure.** The native controller uses the existing C++ and linear-algebra libraries;
the previous NMPC's generated code remains historical. The archive filename does not imply a MATLAB runtime dependency.

Exact URLs and hashes are recorded in `setup.sh`; CMake requires the pinned
CasADi package. The image records Debian packages in
`/opt/helicopter/environment/debian-packages.tsv`. Dependencies live under `/opt`,
separately from bind-mounted source. Three.js and its MIT license are served
locally; viewing a provisioned run does not require a CDN.

[`helicopter.ps1`](helicopter.ps1) owns host-side lifecycle and task commands.
[`run.sh`](run.sh) owns source configuration/build, startup verification gates,
and launching the server. Native build products stay in ignored `build/native/`.
Neither script delegates container operations to `setup.sh`.

## Minimum-entropy H-infinity method

For a stable, strictly proper closed-loop map `Tzw` from declared disturbance
and noise channels to weighted tracking/actuator outputs, choose a feasible
`gamma` and minimize

    I_gamma(Tzw) = -gamma²/(2 pi) integral log det(I - Tzw* Tzw/gamma²) dw
    subject to internal stability and ||Tzw||_infinity < gamma .

The integral covers all frequencies. The signal weights and scales define its
meaning. This entropy approaches the squared H2 norm as gamma increases; it
measures closed-loop response subject to an amplification bound. It is not
motor heat in J/K and does not automatically minimize electrical input.

The central DGKF realization uses two stabilizing algebraic Riccati solutions
and their coupling condition. Synthesis selects the central free parameter as
zero, after checking the required normalization/rank and
stabilizability/detectability conditions. An arbitrary H-infinity solution is not
automatically minimum-entropy. Native C++ performs synthesis before flight.
Online execution updates the resulting controller state and produces incremental
pitch commands; it does not repeatedly solve a nonlinear optimization problem.

The nominal design has **23 coordinates**: 20 local mechanical states plus three
position-weight filters `eta_dot=position_error-0.04 eta`. Three local attitude
errors replace the quaternion; temperatures remain in the nonlinear plant but
outside flight synthesis. The generalized plant has 18 disturbance/noise inputs,
4 incremental pitch inputs, 12 measured outputs, and 27 weighted regulated
outputs. The [contract's scaling table](docs/ENTROPY_CONTROL_CONTRACT.md#plant-initialization-and-linearization)
records every state weight, input scale, and disturbance/noise scale. Full matrices
and labelled channels are published in `GET /api/model`.

Both controllers receive the same deterministic simulated sensor noise in
position, velocity, attitude, and rate measurements. Neither command calculation
uses privileged true wind. Design mass is fixed at nominal 2 kg; the previous
causal mass observer is diagnostic only. The simulator separately prepares the
same spooled trim for the actual plant mass in each paired trial. Neither
controller is credited with discovering that trim or recovering from an
unprepared unknown-mass start.

Execution uses exact zero-order-hold controller matrices at 100 Hz, with output
computed before the state update. Position/velocity references are subtracted
from measurements; fixed nominal acceleration/drag attitude feedforward and
vertical trim-sensitivity feedforward assist tracking. Controller coordinates
start at zero. Pitch and slew clipping use known applied-minus-requested input
correction; every intervention is counted. Geometry-triggered PID landing
support is explicit. These extra rules and nonlinear contact fall outside the
linear theorem. H-infinity application must occupy at least 90% of all mission
intervals, including initial hold and PID support in the denominator.

The controller's dynamic coordinates are not presented as unbiased state
estimates. Its displayed measurement and weighted-output reconstructions are
labelled accordingly; actual position and sensor-error plots come from separately
recorded plant truth and measurements.

### What the linear certificate establishes

Native synthesis uses ordered-Schur Riccati solutions, explicit normalized DGKF
and PBH assumptions, coupling and stability checks, and an independent
bounded-real Riccati gain bound. For the implemented weights it produces
`gamma=1.513435045`, numerical gain upper bound `1.338888058`, entropy
`17.504639378`, and H2 norm `3.636691622`. Gamma is selected with a 1.5 margin
over a numerical feasible bracket; this is not a claim of global gamma optimality.

The artifact includes both synthesis Riccati solutions, the bounded-real solution,
H2 Gramian, and sampled controller matrices. Independent equation replay,
frequency-quadrature refinement and tail treatment, and a large-gamma limit check
support the calculation. A frequency sweep alone cannot prove a bound over all
frequencies. All certificates are numerical and tied to the disclosed linear
model, scales, and weights.

A bound on the declared linear disturbance-to-output channels does not by itself
establish robustness of the entire nonlinear helicopter to arbitrary uncertainty,
saturation, or contact. The sampled implementation needs its own checks.
Measured update times must meet the actual interval before any simulation
real-time claim; hardware remains unvalidated.

## Reference vehicle and physical accounting

The assumed electric main/tail-rotor helicopter has nominal mass 2 kg, main/tail
radii 0.55/0.11 m, and governed speeds 180/650 rad/s. Coordinates are right handed:
world +z up, body +x forward/+y left/+z up, quaternion order wxyz.

The plant retains blade-element thrust, induced/profile power, inflow/flap lags,
offset-hub forces and reaction torques, drag/damping, electrical losses, motor
thermal storage, heat rejection, and effective inflow kinetic storage.
Its physical parameters are not identified aircraft data. Current numerical
values and source provenance belong in `GET /api/model`.

Physical entropy is recorded independently from control entropy. The existing
fixed-ambient accounting gives

    S_generated + Phi_final - Phi_initial
      = (E_electric + W_wind + W_contact - Delta E_mechanical) / T0 .

Terminal commitment covers passive motor cooling and modeled inflow relaxation.
It does not simulate rotor shutdown, a battery cycle, or total ground-contact
entropy. Wind/contact work are separate external ports. Equal wind schedules can
produce different wind work along different trajectories.

No hardware-calibrated aerodynamics, pitch-servo electrical losses, battery
chemistry, rotor-speed transients, stall, validated vortex-ring descent, detailed
ground effect, or flexible structure is added by changing the controller.
Physical-energy savings, if observed, remain a separate measured outcome.

## Validation status and evidence

The source-matched native report passes **all 15 trials**: seven paired 70-second
cases for H-infinity and PID, plus the disabled negative control. The first five
wind/mass cases are retained; two add threefold sensor noise, with and without
extra gusts. Every enabled trial has zero independent physical/landing violations.
Reset replay is identical. The original tracking and physical thresholds are
unchanged; the [contract](docs/ENTROPY_CONTROL_CONTRACT.md) lists them.

| Paired case | H-infinity RMS error (m) | PID RMS error (m) |
| --- | ---: | ---: |
| Nominal | 0.0823 | 0.1389 |
| Alternate wind | 0.0855 | 0.1410 |
| Extra gusts | 0.1418 | 0.2388 |
| Mass minus 20% | 0.1029 | 0.1440 |
| Mass plus 20% and gust | 0.1022 | 0.2078 |
| Threefold sensor noise | 0.0857 | 0.1422 |
| Threefold noise and gusts | 0.1426 | 0.2394 |

H-infinity supplies **93.04–93.16% of all 7000 mission decisions**, with ten
initial trim-hold steps and 469–477 PID landing steps. There are no controller
failure fallbacks. Every limiter intervention is recorded; frequent slew limiting
in the threefold-noise cases is outside the linear guarantee. These results are
for the complete implemented controller including its limiter and landing support.

The PID comparison was retuned for the new noisy, nominal-parameter, no-wind-input
conditions. It uses stronger lateral/yaw feedback, original roll/pitch bandwidth,
and static tail allocation that preserves aerodynamic damping. Landing takeover
initializes the bounded integral from known applied pitch and measured state;
it does not read the true mass. Failed preliminary runs are preserved in `build/`
and summarized in the final provenance record. This comparison establishes lower
tracking error in these seven cases, not a generally optimal baseline or universal
superiority. Controller gains were not tested against a withheld scenario set.

The final report records mean, p95, p99 and maximum CPU update times against the
10 ms interval. In this final run, mean updates were 0.0058–0.0063 ms, the worst update was 0.205 ms, and none missed 10 ms. These are host measurements, not a hard-real-time guarantee. The report also records independent electrical/thermal/entropy balances,
source-bound synthesis certificates and sampled stability. HTTP and browser
checks establish display/provenance behavior separately from flight performance.
All cases, including failures, must remain visible. An H2-limit comparison is
needed to isolate the robustness/performance tradeoff; PID alone does not show
whether this criterion improves over H2. Changing weights between controllers
makes the raw entropy/gamma numbers incomparable.

The prior model fingerprint `076fba9b8d055319…` and its eleven passed trials
describe historical thermodynamic NMPC with PID support. Those results, earlier
unit wrappers, and HTTP/UI captures must not be reused as H-infinity validation.
The even earlier `artifacts/pid-original-evaluation.json` is also historical.
Preserved documentary results are in
[the historical archive](docs/historical-thermodynamic/README.md).

The active saved report must carry the corrected objective identifier, current
model/evaluator hashes, synthesis fingerprint, and scenario/controller identity.
The viewer's validation badge must reject stale method/schema/source evidence.
Native flight evidence, service checks, browser QA, and GPU access are separate
checks. GPU availability does not establish controller acceleration.

## Native service and provenance

The existing local service routes remain the interface:

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | Worker health and GPU visibility |
| `GET /api/state` | Native state, applied controller/command, runtime diagnostics, accounting |
| `GET /api/model` | Physical model, synthesis states/channels/weights, certificates and fingerprints |
| `GET /api/telemetry` | Whole current mission history |
| `GET /api/evaluation` | Saved evaluation and objective/source verification |
| `POST /api/control` | Play, pause, reset, gust, and controller selection |

The transition must label the actual controller family and objective explicitly
rather than inferring meaning from a legacy route name. Synthesis metadata must
identify `objective_kind=minimum_entropy_hinf` and `controller_family=central_dgkf`.
Inapplicable diagnostics are absent or null. Native integration/evaluation retain
double precision; rounded display telemetry is not the source of numerical
certification. The source manifest must cover the plant, synthesized realization,
weights, tests, service, and viewer used for the claimed result.
