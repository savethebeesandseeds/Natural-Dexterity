# Neural actuation-transfer experiment

## Current training procedure, 2026-10-09

All new builds, tests and training run inside the documented managed Linux
container. The Windows Python setup and commands below describe the preserved
September 7 experiment and must not be used for new work under the current
container-only instructions. Its locked reports and source snapshots remain
unchanged.

The [first pooled-pretraining pilot](../docs/GENERALIZATION_PRETRAINING_RESULTS.md)
and the [three-seed breadth study](../docs/GENERALIZATION_STUDY_RESULTS.md) are
complete. Substantial simulation exposure supplies the first creature's
competence; assembly-specific inference and small adaptation specialize it. The optional
dependency installer is `setup-generalization.sh`; it installs CPU training
packages into `/opt/droid-blocks/generalization-venv` inside the existing
`droid-blocks-dev`, without managing Docker objects or changing the native
launcher. Its receipts remain in `/opt/droid-blocks/generalization-setup`.

The [breadth protocol](../docs/GENERALIZATION_STUDY_EXPERIMENT.md) documents the
exclusive `lock`, `sources`, `development`, `challenge` and `summary` stages in
`generalization_study.py`, using the container interpreter and local
`/workspace/build/native/droid-transfer-env` bridge. Its recorded batch is
`artifacts/reports/generalization-study-v1-20261006`; do not resume or overwrite
it. The independent native and tensor auditors live in `tools/`, with exact
versions archived in that batch's verification folder. Native physics,
sensor timing, safety, action governor and live workshop controller remain the
same. All six releases passed source competence before the twelve length-5
assemblies were instantiated. Those evaluated assemblies are now exposed
development data. Future unseen-body claims need a fresh prospective reserve;
online specialization needs its own charged experience budget and controls.

## Bounded specialization

The [prospective protocol](../docs/SPECIALIZATION_EXPERIMENT.md) and
[completed result](../docs/SPECIALIZATION_RESULTS.md) cover the recorded
`artifacts/reports/specialization-v1-20261007` batch. `specialization_manifest.py`
locks source releases, initialization tensors, settings and preservation
baseline; `specialization_study.py` collects exclusive `targets` then `guards`
stages. Each target uses one continuous 128+512 decision opportunity, with fresh
Adam state and no reset or action reseed at the learning boundary. Head-only
Adam contains four tensors/260 parameters, including the value output layer.
Probe checkpoints load policy tensors only, avoiding filtered-optimizer group
incompatibility. The live actor remains unchanged.

The independent auditor is `tools/verify-specialization-study.py`. Its exact
source/helper copies accompany `verification/audit-v1.json`, which verifies
294 native streams, 188,160 actions and 510 fresh/prefix checkpoints. PPO
gradients are not rerun. All target/guard `traces/` and all new ZIPs remain local
and ignored; the report lists the assets a fresh checkout needs. Never resume
or overwrite the completed batch, or change its failed practical-benefit gate.
Use a new prospective protocol for subsequent research.

`tools/write-specialization-report.py` renders the audited Markdown tables;
`tools/finalize-specialization-study.py` binds the final provenance. The optional
plotting tool requires Matplotlib, which is absent from the pinned training
environment; this batch uses tables and does not change its runtime dependencies.

## Longer budget and conservative specialization

The [prospective protocol](../docs/CONSERVATIVE_SPECIALIZATION_EXPERIMENT.md) and
[results](../docs/CONSERVATIVE_SPECIALIZATION_RESULTS.md) cover the completed
`artifacts/reports/conservative-specialization-v1-20261009` batch. The manifest
locks source, original releases, historical comparisons, runtime and a new
preservation baseline. The driver collects exclusive `targets`, seals all
144 attempts, then releases the 150 `guards`. Never resume, overwrite or
recollect this completed batch.

The declared target budget contains 384 learning decisions followed by 256
fixed-weight decisions in one continuous world. Conservative uses learning rate 0.0001 and
clip 0.1; standard and scratch use 0.0003 and 0.2. All learning arms train all 12
actor/critic tensors with fresh Adam state. Constant learning-rate schedules
are set before model construction so later PPO updates use the declared rate.
Every 64-decision update is retained as a checkpoint. No source optimizer,
boundary reset, action reseed or rescue is used.

Inside the documented Linux container, the authoritative stages are
`conservative_specialization_manifest.py lock`, then
`conservative_specialization_study.py targets` and `guards`, sharing one new
`--output` study directory declared by its prospective protocol. The completed
October 9 protocol can be checked with:

```sh
/opt/droid-blocks/generalization-venv/bin/python -B \
  /workspace/training/conservative_specialization_manifest.py verify \
  --output /workspace/artifacts/reports/conservative-specialization-v1-20261009
```

`tools/verify-conservative-specialization.py` independently audits declared
settings, fresh optimizer state, checkpoint tensors, action RNG, continuous
histories and all retained native streams. Its exact source and stable helper
are archived with `verification/audit-v2.json`. The first audit was interrupted
by a container stop; its empty `audit-v1.json` and interruption receipt remain
retained. Only verification was retried. Historical frozen 640-decision streams
and standard 128-decision prefixes are controlled repeats of October 7, not new replicates.
PPO gradients are not independently rerun.

The batch retains 294 request tapes, 294 telemetry tapes and 942 ZIPs
(294 fresh initial plus 648 update checkpoints), all local and ignored. The two
standard late safety stops remain in the records and gates. Collection totals
187,919 decisions and 939,593 successful native controls; a partially completed
macro means controls cannot be inferred as five times decisions. No checkpoint
is promoted. `tools/write-conservative-specialization-report.py` renders the
audited tables and `tools/finalize-conservative-specialization.py` binds final
documentation and asset provenance. Full replay needs the original releases,
the pinned native/runtime dependencies and the exact locally retained assets.

## Frozen native workshop

The [pretrained workshop](../docs/PRETRAINED_WORKSHOP.md) uses the first declared
pooled seed, 101. `export_pretrained_actor.py` exports its six actor tensors with
sealed-source provenance; it never starts training. The tracked actor JSON is
sufficient for the native service, so opening `/pretrained.html` requires no
Python runtime or policy ZIP loading.

`verify_pretrained_actor.py` checks native inference against CPU Torch using
synthetic inputs and retained known-case telemetry. `verify_pretrained_workshop.py`
independently regenerates the portable sampler and replays the new runtime's
actions through the preserved historical engine. These are implementation checks
on exposed cases. Use new receipt versions, retain failures, and never overwrite
the completed workshop's receipts or recordings. Full audit prerequisites and
the distinction between categorical probability parity and Torch trajectory
identity are documented in the workshop note.

## Historical September procedure

The native C++ simulator remains the environment. This directory adds an
isolated host-side Stable-Baselines3 PPO learner; it does not replace the live
Expected SARSA controller or use the affine dynamics predictor to select actions.
Python is not installed in the existing development container.

Read the [locked experiment](../docs/ACTUATION_TRANSFER_EXPERIMENT.md) before
running anything. It defines the source/target split, observations, native
reward, seeds, budgets and interpretation rules. The first batch is exploratory:
the separate CartPole reference completed its integrity checks but missed its
predeclared reward-improvement gate. That result is retained, not repaired by
changing the gate or trying another seed.

## Learning boundary

`native_transfer.py` connects a host process to the existing container's
`droid-transfer-env` executable over stdin/stdout. Each bridge process owns its
own worlds and never takes ownership of the browser's creature. No additional
container, service port or GPU training process is required.

Both actor and critic receive 12 frames of local feedback, sensor measurements,
actions and rewards: 204 inputs. Two 64-unit tanh layers map this history to a
three-action categorical policy and a scalar value estimate. PPO uses the same
collected transitions to train both networks. There is no learned transition
model, privileged critic, body-layout input or geometry-conditioned policy.

One discrete request lasts 0.1 simulated seconds. The native motor governor
still runs every 0.02 seconds. The objective is the sum of the native light and
IMU rewards; geometry and full physical snapshots are recorded only for
validation and playback. A history stack is a practical partial-observation
baseline, not proof of a Markov state or a learned body model.

Transferred branches copy actor and critic weights into a fresh optimizer.
The frozen branch uses those weights unchanged. The adaptation branch updates
all of them; this experiment does not freeze an embedding and train only a head.

## Reproduce without overwriting evidence

Follow the project's normal launcher to inspect and reuse `droid-blocks-dev`.
Do not create or replace a container from this directory. Build the bridge and
native checks with the existing project configuration. In PowerShell:

```powershell
docker exec droid-blocks-dev cmake --build /workspace/build/native --target droid-native-validation -j 2
docker exec droid-blocks-dev ctest --test-dir /workspace/build/native --output-on-failure
./training/setup-transfer.ps1
```

The setup script creates `build/transfer-venv` and installs pinned dependencies
there. It does not manage Docker. The runner uses one CPU thread. Installation
receipts, package versions and the separate reference report accompany the
first batch's archived provenance.

The following shows the stage order, **not an instruction to rerun the completed
batch**. Choose a new output directory for a separately declared replication;
never reuse an incomplete attempt to conceal a failure.

```powershell
$transferPython = './build/transfer-venv/Scripts/python.exe'
$transferOutput = 'artifacts/reports/NEW_DECLARED_BATCH'
& $transferPython training/actuation_transfer.py lock --output $transferOutput
& $transferPython training/archive_transfer.py --output $transferOutput
& $transferPython training/actuation_transfer.py development --output $transferOutput
& $transferPython training/actuation_transfer.py development-eval --output $transferOutput
foreach ($transferSeed in @(101, 202, 303)) {
    & $transferPython training/actuation_transfer.py source --seed $transferSeed --output $transferOutput
    if ($LASTEXITCODE -ne 0) { throw 'Preserve and inspect the failed source run.' }
}
foreach ($transferSeed in @(101, 202, 303)) {
    & $transferPython training/actuation_transfer.py targets --seed $transferSeed --output $transferOutput
    if ($LASTEXITCODE -ne 0) { throw 'Preserve and inspect the failed target run.' }
}
& $transferPython training/actuation_transfer.py probes --output $transferOutput
& $transferPython training/actuation_transfer.py replay-sources --output $transferOutput
node tools/summarize-actuation-transfer.mjs --input $transferOutput --web-root web --output build/NEW_AUDIT
```

The runner retains final checkpoints, sampled source task/action tapes, target
traces, update metrics, hashes and failures. Native replay verifies the recorded
physical action sequence. It is not a second independent trial or a replay of
the neural optimizer. The auditor checks provenance, counts, reward components,
matched comparisons and trace consistency before exporting the read-only
`/transfer.html` animation.

All three source checkpoints must be sealed before target collection. The
development elbow model is explicitly exposed to the changed motor layout and
is never transferred into a target branch. Results for all declared source
seeds and target cases must be reported, including negative transfer.
