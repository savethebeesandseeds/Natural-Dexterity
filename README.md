# Natural-Dexterity

Repository: <https://github.com/savethebeesandseeds/Natural-Dexterity>.

Use `C:\Work\Natural-Dexterity` for the local checkout. The former path,
`C:\Work\Universalis-Materiae-Dexteritas`, remains available for existing Docker
mounts and saved workspace references. A compatibility junction makes both names
resolve to the same checkout. Keep both paths while those references are in use. Container
names, IDs, images, dependencies, and experiment data are preserved. Historical
reports, source snapshots, and verification records retain their original bytes.

On Windows, open Codex helpers can prevent moving the backing directory. While
it is locked, the new name is the junction to the original directory. After
closing Codex, run the following from a separate PowerShell window to move the
backing directory and leave the old name as the compatibility junction:

```powershell
pwsh -NoProfile -File C:\Work\Natural-Dexterity\tools\rename-local-checkout.ps1
```

Project experiments live under [`projects/`](projects/).

The [Electric Wind Instrument](projects/electric-wind-instrument/README.md) is a
concept for a silent, wearable wind instrument using wrist and neck radar,
optional precision mouthpiece sensing, and headphone sound. Its description is
recorded as an idea only; no development is requested.

Temporary files, previews, builds, and dependency caches are ignored. Remove
disposable task output when finished; ignoring it does not reclaim disk space.
See [repository hygiene](AGENTS.md) for the cleanup and staging rules.

Droid Blocks commits compact experiment reports, source snapshots, verification
records, and the headroom viewer index. Full transition streams, physical frame
snapshots, and generated `web/headroom-traces/` playback files stay local. A fresh
checkout can inspect the recorded summaries; full replay and animation require
the original local recordings. Restore those files from the retained experiment
batch and check their hashes before using them. The existing local service keeps
its complete data. Do not rerun a study to substitute for its original evidence.

## Helicopter environment

The [helicopter laboratory](projects/helicopter/README.md) implements the central
minimum-entropy H-infinity output-feedback controller associated with Mustafa and
Glover's *Minimum Entropy H∞ Control*. Here entropy is a log-determinant functional
of a weighted closed-loop transfer map. It is **not physical heat/entropy
minimization**. The previous thermodynamic NMPC experiment and its results are
preserved under the project's historical directories and do not validate this method.

The native C++20 application combines offline two-Riccati synthesis, a 100 Hz
dynamic feedback controller, a nonlinear MuJoCo helicopter simulation, and a local
viewer. The viewer separates the nominal linear design certificate, live sensor
and controller variables, physical telemetry, and paired flight evaluations.
PID landing support, actuator limiting, and fallback decisions are disclosed.
The model is an explicitly parameterized simulation with no hardware calibration.

All **15 final trials pass**: seven wind/mass/noise missions per controller and a
disabled negative control. H-infinity RMS tracking error is 0.082–0.143 m versus
0.139–0.239 m for the retuned PID. It supplies 93.04–93.16% of all mission
commands; the remainder are trim hold and PID landing support. No failure
fallback occurs. The nominal linear gain bound is 1.339 below gamma 1.513;
that certificate does not cover the nonlinear limits/contact behavior.

### Start the existing environment

Prerequisites: Docker Desktop using Linux containers/WSL2, an NVIDIA GPU and
compatible Windows driver, and PowerShell. From the repository root:

```powershell
Set-Location 'C:\Work\Natural-Dexterity\projects\helicopter'
.\helicopter.ps1 up
.\helicopter.ps1 gpu
```

The launcher inspects and reuses the managed `helicopter-dev` container, including
when stopped. Its `helicopter:dev` image uses pinned Debian 12 slim. The command is
`/bin/bash /workspace/run.sh`; the helicopter project is the sole read/write bind
mount at `/workspace`; there are no named volumes. The only published port is
`127.0.0.1:43118` to container port `8080`. Restart policy is `no`, with init and a
15-second stop timeout. All NVIDIA GPUs are accessible with `compute,utility`;
there are no additional devices or privileges.

Open <http://127.0.0.1:43118/> and select minimum-entropy H∞ or PID. The simulator
and controller run on CPU; GPU availability is verified independently. This is
not a hardware-flight or hard-real-time certification.

```powershell
.\helicopter.ps1 test   # native synthesis, physics, and full paired flight checks
.\helicopter.ps1 check  # HTTP, provenance, controls, and history checks
.\helicopter.ps1 logs
.\helicopter.ps1 stop   # stops without deleting data
```

`setup.sh` only installs reproducible dependencies and configures the environment;
APT uses `--no-install-recommends`. The separate launcher owns container operations,
and `run.sh` builds/tests/starts the application. No Python, pip, or MATLAB runtime
is needed. Native LAPACK is supplied by the already pinned CasADi 3.7.2 SDK bundle;
the current controller does not invoke CasADi or IPOPT. Do not replace or delete an
existing container to apply a source change: stop/start it through the launcher.

The [project README](projects/helicopter/README.md) records exact pins and setup.
The [control contract](projects/helicopter/docs/ENTROPY_CONTROL_CONTRACT.md) defines
the generalized plant, interpretation of the certificate, and acceptance gates.
Only the source-matched complete report in `projects/helicopter/artifacts/evaluation.json`
establishes the current flight results; historical reports concern other methods.
