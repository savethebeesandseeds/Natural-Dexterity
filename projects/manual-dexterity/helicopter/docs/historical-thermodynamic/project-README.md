# Helicopter flight laboratory

A native C++20 control laboratory for an **explicitly simulated, uncalibrated
reference helicopter**. The implementation now includes a shared rotor/electrical/
thermal model and constrained entropy-minimizing nonlinear model predictive
control (NMPC), with an explicit PID baseline, fallback, and landing support. MuJoCo integrates the
rigid-body/contact plant; native CasADi 3.7.2 and IPOPT solve the predictive control
problem. The browser displays native state and results; it does not calculate
flight dynamics or control commands.

The optimizer models airborne free flight and uses approach constraints intended
to settle the aircraft before a geometry-triggered handover to **PID landing
support**. Approaching skid contact triggers the handover even if the aircraft
has not settled. The report records the actual handover state and
`settled_at_handover` flag. This is a hybrid controller, not entropy-optimal
contact control.
The simulator supplies a spooled trim computed for each actual plant mass;
neither controller discovers that initial trim. Both enabled modes hold those
prepared trim commands for the first 0.1 seconds to
collect causal mass observations. That interval remains inside the mission's
time, energy and entropy accounting. Every non-NMPC interval, including this
observation hold, rejected solves and planned landing support, counts against the
required 90% NMPC participation across the complete mission. Actual IPOPT
attempts and scheduled control decisions have separate counters.

**Measured simulation result: all 11 native evaluation trials passed.** The five
paired 70-second missions met the flight, physical, accounting and optimizer
participation criteria, with small entropy and electrical-energy savings for the
hybrid controller. Tracking was less accurate than PID and every optimizer
attempt exceeded its 100 ms control interval. These results apply to the declared
uncalibrated simulation; they establish neither hardware performance nor
real-time operation. The measured results and limitations are recorded below.

The [entropy control contract](docs/ENTROPY_CONTROL_CONTRACT.md) defines the
physical boundary, accounting identities, assumptions, and acceptance criteria.
The original exploratory document remains in [`doc/`](doc/README.md).

## Use the laboratory

Start the environment below, then open <http://127.0.0.1:43118/>. The nominal
mission contains 70 seconds of simulation time: takeoff, hover, two waypoints,
wind recovery, return, and landing. Solver computation can make playback slower
than wall time. A rendered flight by itself is not a passed validation test.

- The selected mode is labelled **NMPC + PID landing** or **PID baseline**.
  **Switch & restart** starts a fresh run in the selected mode. Ordinary restart
  preserves the mode. A rejected optimization is visibly labelled **PID fallback**;
  fallback counts are retained. Planned landing support has its own applied
  controller label and count; it is not presented as optimized control.
- Pause/play, restart, and gust controls operate the native simulation. Opening
  or refreshing the page does not issue a control action. Drag to orbit; scroll
  to zoom. The body pose is actual plant output; blade spin is cosmetic. The
  purple prediction is shown only for an accepted NMPC candidate.
- The history tabs contain flight tracking, energy/entropy, and optimizer plots.
  Flight plots include reference/actual position, error, attitude, command/actuator
  response, thrust, wind, induced inflow, and flapping. Thermodynamic plots expose
  irreversible components, storage, temperature, input energy, and balance
  residuals. Optimizer plots expose candidate/seed objective, solve duration,
  residuals, iterations, action change from the PID seed, acceptance, and fallback.
- Graphs load the native service's full current-run history, sampled every 0.1
  simulation seconds. Late joining does not lose earlier samples. Reset changes
  the run ID and clears history; a server restart or backwards time change also
  clears the display. Use **Download data** for native telemetry JSON.
- Expand the model and optimization panels for equations, physical pitch inputs,
  inflow/flap states, motor variables, bounds, and solver/energy accounting.
  Absent or inapplicable values are not replaced with synthetic measurements.

State and reference vectors are at `state.time`; `solution.time` describes the
preceding 0.01-second integration interval. Optimizer diagnostics have their own
latest-control-decision timestamp, including decisions that select PID landing
support without attempting a solve. Thermodynamic power and entropy rates are instantaneous at
the current state. Force in world coordinates includes rotor force and drag but
excludes gravity/contact; MuJoCo applies those separately. Body rates are angular
velocity `[p,q,r]` in rad/s, not Euler-angle derivatives.

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
   Set-Location 'C:\Work\Universalis-Materiae-Dexteritas\projects\helicopter'
   .\helicopter.ps1 up
   ```

   The launcher first inspects `helicopter-dev`. It reuses a matching container,
   including a stopped one. If absent, it builds `helicopter:dev`, creates the
   configuration below, verifies it, starts it, and waits for `/healthz`.
   A conflicting container is preserved and reported by immutable ID; it is not
   replaced. Build or startup-test failure preserves the environment and prevents
   the service from being treated as ready. Use `logs` to inspect progress.
   First startup compiles native optimizer derivative code (about six minutes
   in the measured local build). The launcher allows up to 30 minutes;
   subsequent runs reuse a source-bound native cache in the mounted `build/`
   directory. Changing the mathematical/controller sources invalidates it.

3. Verify the runtime and GPU, then open the viewer:

   ```powershell
   .\helicopter.ps1 status
   .\helicopter.ps1 gpu
   Start-Process 'http://127.0.0.1:43118/'
   ```

   The GPU check loads the injected NVIDIA driver, uploads two values, executes
   a CUDA addition kernel, and checks the downloaded result in the managed
   container. **The plant and CasADi/IPOPT solver run on CPU.** GPU device access
   does not mean the optimizer uses CUDA or meets a real-time deadline. Three.js
   rendering runs in the browser, independently of container GPU access.

4. Run the native evaluation and service checks, or inspect logs:

   ```powershell
   .\helicopter.ps1 test
   .\helicopter.ps1 check
   .\helicopter.ps1 logs
   ```

   `test` runs thermodynamic, mass-estimator, optimizer, report-merger, and full paired flight checks, then
   writes the saved report. Full paired NMPC trials can take substantially longer
   than their simulated duration.
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
| CasADi / bundled IPOPT | CasADi 3.7.2 native C++ SDK | Symbolic derivatives and constrained nonlinear optimization |
| nlohmann/json | 3.12.0 | State, metadata, and evaluation JSON |
| cpp-httplib | 0.51.0 | Local HTTP service |
| Three.js | r170 / 0.170.0 | Browser rendering, served locally |

The pinned official CasADi archive is named
`casadi-3.7.2-linux64-matlab2018b.zip`. `setup.sh` extracts its C++ headers,
CMake metadata, CasADi/IPOPT libraries, native linear-algebra dependencies, and
shell JIT support into `/opt/casadi`. **No MATLAB interface/runtime, Python,
Python package installer, or Python control process is installed or invoked by
this procedure.** CasADi's generated native code is compiled by the container's
C++ toolchain. The archive filename does not imply a MATLAB runtime dependency.

Exact URLs and hashes are recorded in `setup.sh`; CMake requires the pinned
CasADi package. The image records Debian packages in
`/opt/helicopter/environment/debian-packages.tsv`. Dependencies live under `/opt`,
separately from bind-mounted source. Three.js and its MIT license are served
locally; viewing a provisioned run does not require a CDN.

[`helicopter.ps1`](helicopter.ps1) owns host-side lifecycle and task commands.
[`run.sh`](run.sh) owns source configuration/build, the thermodynamic startup
verification gate, and launching the server. Build products and native optimizer
JIT files stay in ignored `build/native/`.
Neither script delegates container operations to `setup.sh`.

## Reference vehicle and entropy objective

The declared vehicle is a 2 kg electric single-main-rotor/tail-rotor approximation.
Its 23 native states are position, velocity, quaternion, body rates, four actual
pitch angles, two induced inflow speeds, two flap angles, and two motor
temperatures. Commands are physical collective/cyclic pitch angles in radians;
the viewer also exposes normalized channels. Coordinates are right handed, world
+z up and body +x forward/+y left/+z up; quaternion order is `(w,x,y,z)`.

The model includes blade-element thrust, induced/profile power, first-order
inflow and flapping, offset-hub forces and reaction torques, drag/damping, and
explicit induced-flow kinetic storage. Main and tail motors have copper,
temperature-dependent resistance, iron, drive, and non-regenerative braking
losses, with separate thermal storage and heat rejection. Current parameter
values, physical limits, state layout, equations, and source fingerprints are
published by `GET /api/model` from the compiled implementation.

The optimized quantity is physical irreversible generation, in J/K:

```text
J = integral_over_horizon(sigma dt) + Phi_terminal
sigma = sum_motors[L/T + Q_out (1/T0 - 1/T)]
        + (wake + drag + angular dissipation) / T0
Phi_terminal = (sum_motors C[(T-T0) - T0 log(T/T0)] + E_inflow) / T0
```

`L` is motor/drive loss, `Q_out` is rejected heat, and `T0` is the declared ambient
reservoir temperature. The objective has **no tracking-error cost**. Tracking is
required by constraints. Terminal cooling and wake commitment prevent a horizon
from appearing cheaper merely by leaving heat or induced-flow energy unaccounted
for. Whole-run comparisons report generated entropy plus the change in terminal
commitment from the initial state, alongside electrical input energy.

During airborne control, native C++ CasADi/IPOPT repeatedly solves a direct multiple-shooting problem with
RK4 prediction, applies the first admissible pitch command, and replans from the
new simulated state. The compiled metadata gives the horizon and update interval.
Hard constraints cover the tracking corridor and terminal position/velocity,
command bounds and slew, altitude, attitude/rates, airspeed and descent/inflow
envelope, motor temperature, electrical/shaft power, and current.
The predictor uses internal tilt/airspeed limits of 0.40 rad and 4.7 m/s, inside
the independently audited physical limits of 0.52 rad and 5 m/s. These heuristic
margins allow for prediction errors; they do not prove robustness to arbitrary
wind or model error.

Every candidate is reintegrated outside IPOPT's state decision variables, and
its nonlinear constraints, dynamics consistency, and objective are checked.
Solver convergence is reported separately from acceptance: a finite-budget,
feasible candidate need not carry a convergence claim. A rejected, failed, or
infeasible solve uses the declared PID fallback, with counts and reasons visible.
The PID comparison uses the same plant, mission, limits, and disturbances.
Neither controller sets the aircraft pose directly.
Their trajectories need not be identical: NMPC can use the permitted corridor
to reduce its physical objective. Compare the reported tracking errors as well
as entropy and energy; a savings result does not imply equal tracking accuracy.

The nominal route climbs to 2.5 m, visits `(3,0,2.8)` and `(3,3,2.8)`, returns to
`(0,0,2.5)`, and lands. Seeded wind and a scheduled crosswind pulse are part of the
mission; the viewer can add a pulse. The predictor uses exact simulated state and
current-wind persistence. A causal least-squares fit estimates mass from observed
airborne velocity changes and the assumed rotor/drag force map. The same estimate
feeds PID, its fallback/landing support, and NMPC. All principal inertias scale
with estimated mass under the declared perturbation-family assumption; inertia
is not identified independently. The estimator starts from the nominal 2 kg
prior, skips contact intervals, and never reads the simulator's true mass.
Initialization separately supplies the same mass-specific spooled trim to both
controllers, so these trials do not demonstrate recovery from an unprepared,
unknown-mass start.
Exact state and force-map assumptions remain; this is not a noisy sensor loop.

### Boundary and limits of interpretation

Electrical input and imposed-wind work are external energy ports. Rigid-body,
motor thermal, and effective inflow storage are tracked; eventual outgoing-wake
relaxation is assigned to the ambient reservoir. MuJoCo contact work is reported
separately because the NMPC prediction model is free flight.

Rotor speed is governed at fixed declared values. No battery chemistry/state of
charge, identified hardware parameters, governor transient, stall, ground effect,
vortex-ring validation, detailed rotor interference, structural flexibility, or
physical sensor loop is claimed. Pitch-servo energy and internal mechanical
storage are omitted. The near-hover envelope and these assumed boundaries limit
the meaning of the results. Complete relaxation at a common ambient temperature
can make entropy and dissipated-energy objectives closely related; the project
does not claim an extra efficiency benefit merely from using the word entropy.
See the [contract](docs/ENTROPY_CONTROL_CONTRACT.md) for the full derivation and
independent acceptance identities.

## Validation and measured results

The complete [native report](artifacts/evaluation.json) passed all eleven
70-second trials: five paired hybrid-NMPC/PID scenarios and the disabled negative
control. The disabled trial passed by failing the mission as expected. All eleven
trials recorded **zero physical-constraint violations**. The paired trials also
passed landing, energy/entropy accounting, causal identification and applicable
optimizer-contribution checks. These results correspond to model fingerprint
`076fba9b8d055319…` and evaluator fingerprint `759fda0aa1c2a6f1…`; the report contains
the complete hashes and the merger's source provenance.

Savings below compare the complete hybrid mission with its matched PID mission.
Entropy includes the change in thermal/wake terminal commitment. Each NMPC trial
has 700 scheduled decisions, including one initial trim-observation hold.

| Scenario | NMPC applied / 700 | Rejected solves / PID landing decisions | RMS error, hybrid / PID (m) | Entropy saving | Electrical-energy saving | Hybrid wall time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nominal | 653 (93.29%) | 0 / 46 | 0.300 / 0.037 | 0.157% | 0.204% | 803.7 |
| Alternate wind | 653 (93.29%) | 0 / 46 | 0.297 / 0.037 | 0.161% | 0.212% | 813.2 |
| Extra gusts | 645 (92.14%) | 4 / 50 | 0.305 / 0.057 | 0.130% | 0.184% | 850.6 |
| Mass −20% | 648 (92.57%) | 1 / 50 | 0.298 / 0.044 | 0.204% | 0.268% | 834.5 |
| Mass +20% with gust | 634 (90.57%) | 15 / 50 | 0.293 / 0.043 | 0.122% | 0.146% | 901.5 |

The strict 90% requirement uses all 700 decisions, including the initial hold,
rejected solves and planned PID landing support. Only the nominal handover met
the stricter `settled_at_handover` condition; the other four handovers were
explicitly recorded as unsettled. All five subsequently passed the unchanged
executed-flight landing criteria with PID support. This result does not establish
entropy-optimal contact control.

The measured tradeoff is small modeled savings with larger tracking error:
hybrid RMS error was 0.293–0.305 m versus PID's 0.037–0.057 m. Every pair's energy
and entropy reductions exceeded its audited numerical, contact-work and
mechanical-endpoint uncertainty allowances. Those allowances do not cover
uncalibrated aerodynamics or omitted pitch-servo losses, which may outweigh gains
of this size. Savings describe these prescribed wind scenarios; wind work is
reported separately and no intrinsic drive-efficiency improvement is established.

Each 70-second hybrid trial took 803.7–901.5 seconds of wall time in the recorded
environment. Mean optimizer wall time was 1.225–1.383 seconds; **all 3,253 actual
optimizer attempts missed the 100 ms control interval**. Synchronous simulation
can wait for these solves. The controller has not demonstrated real-time operation.

Unit evidence paths are [thermodynamic balances](artifacts/thermodynamics-check.json),
[causal mass estimation](artifacts/mass-estimator-check.json),
[optimizer contribution and rejection](artifacts/optimizer-check.json), and
[report-merger integrity](artifacts/report-integrity-check.json). The merger tests
use synthetic structural fixtures and are separate from flight evidence.
The unit wrappers preserve native output and state how they are associated with
the validated build. The [HTTP checks](artifacts/http-check.json) cover both
controller modes, and the [browser checks](artifacts/ui-check.json) cover the
live view, 18 graphs, model inspector, and desktop/mobile layout.
The [source manifest](artifacts/source-manifest.json) records the source and
evidence hashes. These checks assess the
declared model and its implementation, without establishing aerodynamic accuracy
against measured aircraft data.

The earlier direct-torque PID demonstration remains preserved in
[`artifacts/pid-original-evaluation.json`](artifacts/pid-original-evaluation.json)
as historical evidence for that older model. The viewer reads the current
[`artifacts/evaluation.json`](artifacts/evaluation.json); its validation badge
requires matching compiled model/evaluator provenance. A live flight or stale
artifact cannot replace that saved evaluation. GPU evidence in
[`artifacts/gpu-check.json`](artifacts/gpu-check.json) is a separate execution check.
HTTP/UI checks and GPU access are separate from the native flight result;
GPU availability does not imply GPU acceleration of the optimizer.

To save a separate native report without replacing the viewer's saved report:

```powershell
docker exec -w /workspace/build/native `
  -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 helicopter-dev `
  /workspace/build/native/helicopter-evaluate --output /workspace/artifacts/evaluation-new.json
```

Independent scenarios can also run in separate processes with
`--mode pid` or `--scenario nominal --mode entropy`. Write each report to a
distinct file. After all five entropy scenarios and the six-row PID/disabled
report finish, combine them with the native evaluator's `--merge file1.json
file2.json ... --output report.json` mode. Merge requires exact coverage of all
11 trials, matching compiled sources and policies, and consistent counters;
it recomputes pass gates and comparisons. It preserves failed trials and never
converts partial coverage into a complete validation result. Prewarm the native
solver with `helicopter-optimizer-test` before launching simultaneous processes.

The live mission and saved evaluator results are separate. Do not infer the
current acceptance result from an old file or from the README's implementation
description; inspect the current report, flags, provenance, and comparison
eligibility.

## Native service API

| Endpoint | Data |
| --- | --- |
| `GET /healthz` | Worker health and GPU visibility |
| `GET /api/state` | Actual state, active controller, applied solution, solver/prediction, thermodynamics, metrics, run ID |
| `GET /api/model` | Native model/controller parameters, units, equations, limitations, and fingerprints |
| `GET /api/telemetry` | Whole current mission at 10 Hz; bounded to 800 samples; includes optimizer and thermodynamic history |
| `GET /api/evaluation` | Saved paired evaluation and provenance verification |
| `POST /api/control` | Exactly one JSON field `action`: `play`, `pause`, `reset`, `gust`, `entropy`, or `pid` |

`entropy` and `pid` start fresh runs in that mode. `reset` starts a fresh run in
the current mode. A finished mission must restart before playing again. Native
integration/evaluation retain double precision; telemetry transfer rounds display
values to six decimals. Unavailable optimizer diagnostics are JSON `null`.
CMake fingerprints the compiled plant, controller, headers, and evaluator so a
saved report can be checked against the executable serving the viewer.
