# Verification checkpoints

- [Creative scene workflow, 23 October 2026](creative-scene-workflow-2026-10-23.json):
  1038 tests passed. Scene perform/resume/revision, quota/budget safeguards,
  sanitized diagnostics, preset allocations, playable exports, measured timing/stems,
  and source reuse are verified. The retained 19-line scene assembled offline with
  zero key reads/provider calls; 123 original files and existing rows across 11
  legacy tables stayed unchanged. Both listening ports remain read-only.
- [Timed semantic embeddings backend, 22 October 2026](backend-embeddings-2026-10-22.json):
  677 tests passed. Timed text/audio/joint Embedding 2 requests, exact release/source
  provenance, persistent cache, recovery/accounting, bounded vector sidecars and
  backup restoration are fixture-tested. Both radio ports passed live read-only
  route checks. No paid calls, frontend edits, or dimensionality reduction.
- [Characters and conversations backend, 22 October 2026](backend-characters-2026-10-22.json):
  442 tests passed. Description-only voice design, versioned characters, pinned
  larger casts, ordered audio assembly, accounting/recovery, and publication are
  fixture-tested end to end. Both listener ports reject production routes.
  No new paid calls or frontend edits; replication is excluded.
- [Private backend Gemini integration, 22 October 2026](backend-gemini-check-2026-10-22.json):
  one real two-speaker request succeeded, 9.32 seconds, estimated US$0.001410 under
  the US$0.25 total test cap. Key ignored/untracked and protected; no publication.
  Default backups now omit credentials; 19 focused privacy/recovery tests passed.
- [Radio-only interface, 22 October 2026](radio-only-interface-2026-10-22.json):
  232 tests passed. Both published ports serve the minimal radio and reject
  production routes; old studio assets removed. Backend code/data retained.
  Browser verifies the off-air page and keyboard volume with no console errors.
  Earlier checkpoints below describe the previous interface and remain historical.
- [Keyless refinement, 22 October 2026](keyless-refinement-2026-10-22.json):
  154 tests passed, zero failures/errors/skips; live studio and station reachable.
  Adds provider format/voice guards and character-first design notes.
- [Initial keyless foundation](current.json): prior 145-test checkpoint, preserved
  unchanged with its original source/log checksums.
- [Environment](environment.json): managed container identity, sole bind, matching
  ports, pinned image, and verified NVIDIA GPU access.

Ignored local logs, JUnit, source inventories, and live-check reports are retained
under `.runtime/verification/`. New test/check runs use unique directories.
Tests use labelled fixture audio and mocked provider responses. The private
credential is configured. In addition to the earlier native two-speaker check,
a separate creative session retained the liked 52.36-second three-character scene
“The Mouse Who Brought a Monster”, including two original prompted voices. The
scene workflow reuses those source recordings offline; new live generation still
requires explicit budgets and listening review. Fixture audio is not published
to the running library.
