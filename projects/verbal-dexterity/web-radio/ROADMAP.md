# Web Radio Roadmap

Updated: 23 October 2026. Scope corrected to a radio, with production in the backend.

## Product

The website plays the schedule. Keep one listening surface: station identity,
play/pause, now playing, volume, and a subtle audio-driven animation.
Content generation, scripts, characters, takes, budgets, publishing, and schedule
editing do not belong on this website. A separate production service may come later.

`Backend production + schedule -> immutable releases -> independent station -> HLS -> listener`

## Built foundation

- Isolated `radio-waajacu`, dependency-only `setup.sh`, GPU access, sole project mount.
  Keep matching loopback ports `43120:43120` and `43121:43121`.
- A shared minimal radio page on both listening addresses. The 43120 entrance
  forwards explicit read-only station routes; production APIs are not published.
- Validated immutable releases, strict hashes/schema, bounded media handling,
  station-owned audio, backend rundown/fallback, and changes at airing boundaries.
- Independent AAC/HLS playout, actual-airing journal, stream-aligned now playing,
  completed replay and transcripts retained as backend capabilities.
- Backend reusable/versioned characters, description-only original voice design,
  native/chunked/two-speaker dialogue and ordered larger casts, immutable cast
  snapshots, aggregate budgets, per-call accounting, and explicit operator CLI.
- Script/voice revisions, imports, take decisions, budgets and publication.
  Existing production records remain; production is not exposed on the website.
- Safe process control, retained verification receipts, offline backup and
  checksum-verified restore into fresh directories.

## Current acceptance loop

1. Backend prepares a programme and fallback ident, then publishes validated
   audio + transcript + manifest packages.
2. Backend supplies the ordered schedule; the station validates and retains it.
3. The radio plays the broadcast and shows its current programme.
4. Stop production while the station keeps advancing HLS and its journal.
5. Verify waiting, buffering, reconnect, pause/resume, and mobile listening.
6. The sound animation follows actual playing audio and respects reduced motion.

The credential is now configured privately. One short native Gemini two-speaker
check succeeded: 9.32 seconds, one request, estimated US$0.001410, within the approved
US$0.25 total test cap. No content was published or scheduled. Larger generation
jobs still need explicit budgets; recordings and fixtures remain useful.
Keep compatibility and fixture broadcasts out of the real library.

## Next

1. Finish the reliable listening loop and backend schedule handoff.
2. Leave interface visuals to the separate visuals session; make backend integration changes only as needed.
3. Improve backend production/scheduling jobs and operational recovery as needed.
4. Consider a separate production service only when requested.

## Retained constraints and later work

The station trusts only validated release manifests, not creative database state.
Corrections create new releases. Keep generation budgets and exact request/recipe
provenance, preserve original audio, and never promise identical regenerated voices.
Only prebuilt synthetic voices or original voices designed from descriptions;
voice cloning is excluded.

Reusable character records and backend cast binding are implemented.
Backend scene jobs now support per-line continuation, targeted line/recipe
replacement, explicit quota pacing/fallback, conservative allocation presets,
sanitized failure diagnostics, and playable exports with exact measured timing
and character stems. Cast/dialogue editors, episode catalogues, overlapping sound
design, and richer production interfaces remain later work.
[Character/Gemini notes](docs/CHARACTERS_AND_GEMINI.md) describe current behavior;
[backend commands](docs/BACKEND_PRODUCTION.md) provide the operator workflow.

The creative session produced the liked 52.36-second, 19-turn scene “The Mouse Who
Brought a Monster”, with two original prompted voices and one prebuilt voice.
Its retained recordings provide the offline integration proof for the new scene
workflow; mock tests cover continuation, quotas, budgets, and recovery. No further
paid calls were made in this engineering pass. New live jobs still require explicit
budgets. Versioned design allocations are local guards, not provider billing caps.
See the [scene workflow](docs/CREATIVE_WORKFLOW.md) for the preferred commands.

Transcript/replay pages, listener requests, low-latency HLS, adaptive bitrate, NFC,
advanced scheduling, and optional Icecast are deferred. No creator dashboard or
generation settings are exposed by the radio.

See [README.md](README.md) for operations and [verification](docs/verification/)
for historical and current checks.

Backend timed text/audio/joint source embeddings, reusable provider-result caches,
and separate bounded read-only sidecars are implemented; see [EMBEDDINGS.md](docs/EMBEDDINGS.md).
Projection, semantic trajectories, and all rendering remain owned by the visuals session.
