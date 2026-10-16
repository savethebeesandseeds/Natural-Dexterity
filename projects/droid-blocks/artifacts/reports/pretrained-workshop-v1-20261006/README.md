# Pretrained build/wake workshop v1

This is implementation evidence for the live frozen-actor milestone. Read
[the runtime note](../../../docs/PRETRAINED_WORKSHOP.md) and open
[the workshop](http://127.0.0.1:43117/pretrained.html).

The canonical pooled seed 101 checkpoint was selected by declared seed order,
without target-score ranking. Its exported actor is immutable during wake,
pause/resume and rebuild. Rebuild discards old-body physics and observation
history. The fixture supports three evaluated body layouts, two motor positions
and two lamp sides; no new generalization or hardware result is claimed.

Retained receipts include the actor export, 3,977 reference checks, 21 passing
native suites, twelve exact runtime replays, independent portable RNG checks,
61 final HTTP checks, desktop/mobile browser verification and preservation of
all 3,153 earlier artifact files. The initial runtime replay receipt is retained;
its successor adds independent RNG and browser-facing state checks. The first
HTTP receipt retains its original JS bytes, while the second checks the final
two UI corrections. A background preservation scan interrupted by service
activation is recorded separately from the successful completed scan.

`final-provenance-v1.json` binds these receipts, the actor, current binaries,
managed container identity and exact files in `source_snapshot/`. Nothing in
the completed research studies is amended to accommodate evolved live source.
Their strict live-source guards remain strict. Archived research sources and
checkpoints remain available, together with the original replay executable.

Local ignored evidence:

- `traces/*.json.gz`: twelve frozen runtime action, input and transition records.
- `runtime_snapshot/<sha256>/droid-transfer-env`: the original Linux replay
  engine, SHA-256 `377b754985b2977c947e37e41cbbff286595fe70b08888c9fc32ece1db2d9777`.

The compact actor JSON is tracked at `artifacts/pretrained/pooled-seed101-v1.json`;
a fresh checkout can run the native page with the documented container launcher
without Python. Full reference audit needs the existing optional Linux CPU
training environment and retained known-case telemetry. Full historical replay
needs the local original binary and its existing native MuJoCo libraries, or a
separately verified rebuild from the archived source. Do not recreate old
receipts, replace their assets, or treat ignored evidence as disposable scratch.
