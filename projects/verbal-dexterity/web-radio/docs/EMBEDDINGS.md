# Timed source embeddings for visual performance

This backend prepares reusable high-dimensional semantic inputs. The visuals
session owns dimensionality reduction, shared 3D projection, interpolation,
smoothing, camera mapping, and rendering. No interface code or immutable v1 audio
release is changed by embedding preparation.

## Provider and application bounds

Verified against the official Gemini documentation on 22 October 2026:

- Model: stable gemini-embedding-2, all voices and modalities in one declared
  model/configuration space. Default 3072 dimensions; supported 128–3072.
- Provider context: 8192 input tokens across modalities. Provider audio supports
  WAV/MP3 up to 180 seconds. This application accepts PCM WAV sources, prepares
  mono 16 kHz PCM16 excerpts, and limits each window to 30 seconds.
- Application text limit: 4096 UTF-8 bytes per request. Text-only preflight uses a
  conservative byte/framing bound. Audio tokenization is not inferred from the
  text path or from other Gemini models.
- One Content with text, audio, or both produces one source vector. Combined input
  is a provider-produced joint embedding, not an average of two vectors.
- All modes use identical dimensionality/configuration, no task prefixes or
  character names injected into spoken text. autoTruncate=false; oversize or
  explicitly truncated results fail. No automatic retry or fallback calls occur.

Sources: [embedding guide](https://ai.google.dev/gemini-api/docs/embeddings),
[model specification](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2),
[REST contract](https://ai.google.dev/api/embeddings),
[pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-embedding-2).

## Exact release and timing provenance

Preparation requires a release already accepted by the station, with its exact
package digest and duration. It validates station-owned media before dispatch.

Each authored window has a stable window ID, passage ID, release-relative
start_ms/end_ms, explicit timing evidence and optional editorial character ID.
Matched text/audio/joint variants share the same passage and audible interval.

Text must match exact UTF-8 bytes at the supplied offset in the accepted release
transcript. The transcript digest, byte interval, and exact submitted text are
retained. A script/turn number by itself is not alignment evidence. Native TTS
does not supply reliable turn timing; this backend does not invent timings or
perform diarization/forced alignment.

Audio requires the immutable whole-source digest, exact native source-frame
interval, sample rate, extraction recipe, resulting WAV digest and alignment
evidence. Windows are checked against actual audio frame counts, including near
the end where a rounded manifest duration is insufficient.

- release_mix uses the exact accepted audio and matching release/frame offsets.
  Its character ID must be null, including during overlap.
- character_stem requires an explicitly identified known recording under
  content/ or retained .runtime/studio/takes/, or a scene export at
  .runtime/studio/scenes/<32-hex-job-id>/output/stems/<character-id>.wav,
  a character ID, source hash, and an editorial claim mapping the source interval
  to the accepted release. Other files under scene jobs are not accepted as stems.
  Bytes/crops are verified; the truth of the editorial source relationship must
  be established by the author. A mixed waveform cannot become a stem by
  changing a label.

Use [preparation.example.json](../content/embeddings/preparation.example.json) as a
structural template. Its placeholder hashes prevent accidental queuing.
For text-only work, remove audio and use modalities: ["text"]. Retain exact
transcript text/offsets. For mixed audio comparison, use character_id:null and
kind:"release_mix" without a source path.

## Backend commands

Run inside the existing radio-waajacu container, prefixing these arguments with:

~~~powershell
docker exec radio-waajacu /opt/web-radio/venv/bin/python -m web_radio.embedding_cli
~~~

~~~text
plan --file /workspace/PATH/preparation.json
enqueue --file /workspace/PATH/preparation.json --episode-budget USD_LIMIT --series-budget USD_LIMIT
show --job JOB_ID
run --job JOB_ID
recover --job JOB_ID
reconcile --job JOB_ID --request REQUEST_SHA256 --charged VERIFIED_USD --evidence VERIFIED_NOTE
install --job JOB_ID
~~~

Planning never reserves or dispatches. Queuing freezes exact prepared requests,
source provenance and all unique uncached allocations atomically. Run executes
only the chosen job; it never drains a queue, generates voices, accepts media,
changes a schedule, or installs a bundle automatically. Installation is a separate
explicit local operation.

Per-release embedding budgets use embedding:<release_id>; their series budget
is embeddings, separate from speech/design namespaces in the same ledger.
Checked serving allocations reserve all 8192 tokens at the highest enabled
modality rate: text-only US$0.001639, audio/joint US$0.053248 per unique uncached
request. These are conservative allocations, not measured charges. The stored
rates expire after 31 December 2026. Successful work initially consumes the
allocation; verified billing evidence can reconcile it. There is no claim of an
exact audio token count or a provider-account hard spending cap.

Completed work remains recorded on partial failure. Unsent requests release
their allocations; ambiguous dispatched requests retain them. Recovery uses
responses whose hashes match the authoritative ledger and never calls Gemini.
An unresolved equivalent dispatch prevents another paid attempt until reconciled.

## Cache and shared space

The canonical provider request binds model, dimensions, configuration, exact text,
and exact prepared audio bytes. Verified requests/results persist under
.runtime/studio/embeddings/, with authoritative hashes and paid-attempt identity
in the production ledger. The complete parsed provider response, returned model
information, and usage survive; comparison/reinstallation reuses those results.

Release/window bindings remain separate from content cache identity. Identical
spoken content can reuse a vector without falsely copying another character's
timeline or provenance. Cache hits do not read a key or reserve another request.

A bundle contains only one declared configuration. If Gemini reports a model
revision, it becomes part of config.provider_revision and space_id; known and
unknown revisions, or different reported revisions, cannot mix in one bundle.
A null revision means the provider did not expose one. The space ID identifies
the declared configuration/evidence; it cannot certify undisclosed provider weight
changes. Use the bundle's final space ID, not a pre-dispatch planning placeholder.

## Separate immutable sidecar contract v1

Embedding bundles do not extend performance v1's 8 MiB JSON document, and never
add files to the two-asset v1 release. They are installed separately at:

~~~text
.runtime/station/embeddings/<release_id>/<bundle_sha256>/
  manifest.json
  vectors-0000.f32
  ...
~~~

The versioned manifest binds release ID, package/audio/transcript digests,
duration, model/configuration, shared space ID, source windows, request/response
digests, and vector positions. Its kind is source_embeddings.
The bundle ID hashes canonical manifest metadata including shard digests.

Limits are explicit: manifest at most 1 MiB; at most 2000 vector rows; shards at
most 64 vectors / 786432 bytes. Application jobs accept at most 512 windows
(up to three modalities each) and 64 MiB of prepared provider request JSON.
Metadata is preflighted against the final manifest limit before any paid call.
Over-limit work must be split deliberately.

Vectors are unprojected IEEE-754 float32, little-endian, row-major. Each shard
descriptor supplies count, byte size and SHA-256. Each window supplies shard_id
and vector_index; offset is vector_index * dimensions * 4 bytes.
The backend performs no semantic normalization or dimensionality reduction.
Original provider JSON values remain in the private cache; public float32
encoding has the normal representational rounding. Nonfinite/out-of-range
values, corrupt shards and shape mismatches are rejected.

## Read-only consumption

Both listening ports expose only these prepared-data GET routes:

~~~text
/api/embeddings/<release_id>
/api/embeddings/<release_id>/<bundle_id>
/api/embeddings/<release_id>/<bundle_id>/vectors-0000.f32
~~~

They return catalog JSON, manifest JSON, and application/octet-stream vector
bytes respectively. The listener proxy has fixed upstream paths and explicit
per-response size/type bounds, forwards no browser credentials, and offers no
query parameters or write methods. Missing bundles return 404; stale/corrupt
data returns 409. Preparation controls and credentials remain inaccessible.

Pair a row with the actual audible release offset using the same calibrated
playhead/airing relationship described in [PERFORMANCE.md](PERFORMANCE.md).
Embedding timing is release-relative, not wall-clock time.

Offline backups retain the production ledger/cache, prepared bundles and
installed embedding/performance sidecars, while omitting credentials by default.
Preparation/installation hold the existing backup coordination lease.

## Verification status

Fixture tests cover text/audio/joint preparation, exact text/frame provenance,
per-character stem declarations, cache reuse, release validation, read-only
serving, accounting failures, crash recovery, and backup restoration. No live
embedding request, paid voice generation, release publication, or schedule
mutation was used for this implementation. Provider access and the visual
meaning/fluidity comparison require later separately authorized live work.
