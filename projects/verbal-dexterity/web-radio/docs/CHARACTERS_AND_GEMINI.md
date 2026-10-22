# Characters and Gemini production

Updated: 23 October 2026. These are backend services; the radio website exposes
listening only. Interface visuals belong to the separate visuals session.

## Implemented workflow

Create reusable characters -> select a prebuilt recipe or design an original voice
from a description -> write dialogue/directions -> inspect the plan -> queue with
explicit budgets -> run that selected job -> review -> publish an immutable release.

`CharacterStore` versions character identity, personality/background, pronunciation
notes and preferred recipes. A queued take pins character/recipe revisions and
stores the complete cast snapshot. Later edits create new revisions.

A voice recipe records prebuilt/original-design provenance, provider/configuration,
requested model and returned identity, exact design prompt when used, audition
audio/checksums, timestamps and expiry evidence. Original recipes require retained
description-only request and prompted result provenance. Raw custom IDs, replicated
results and reference-upload voice creation are excluded.

The legacy `design_prompt` field in a prebuilt speaker remains a creative brief.
The description-only design service submits an actual prompt to Gemini and records
the resulting ID/audition; it does not relabel an existing prebuilt voice as designed.

## Coherent scene jobs

The preferred [scene workflow](CREATIVE_WORKFLOW.md) uses one retained unary request
per line, including two-speaker scenes, to support precise continuation and revisions.
It keeps spoken text separate from delivery directions, pins effective recipes, and
records the actual model for every line. Known minute throttling may wait within the
explicit policy; daily exhaustion pauses or uses only a declared model fallback.
Unknown/uncertain dispatch results are retained for recovery and reconciliation,
without automatic repeat calls. Partial policy revisions preserve the other choices.
The native take strategy below remains available for deliberately selected jobs.

## Native take request strategy

One speaker uses the native unary shape. Two prebuilt speakers use native dialogue
when the full input fits. Longer prebuilt dialogue is split at existing turn
boundaries into bounded native chunks. Larger casts and designed-voice dialogue
use ordered unary turns inside the adapter, then one prepared PCM programme.

These choices follow the current documented paths: native dialogue supports up to
two prebuilt speakers; designed multi-character dialogue requires individual turns.
[Official TTS guide](https://ai.google.dev/gemini-api/docs/speech-generation),
checked 22 October 2026.

The application supports up to 16 characters and 200 turns per script, with a
28,000-character spoken-text cap. Full serialized requests, including repeated
directions/configuration, also face a conservative UTF-8-byte-plus-framing bound.
That bound is not an exact tokenizer. A single oversized turn is rejected clearly.

Spoken text and sustained delivery directions remain separate. Character identity
belongs in the recipe; emotion, pace and delivery vary by scene/turn. Inline pause
or vocalization markup can be explicit. No automatic director engine, overlapping
dialogue mixer, or complex editor is required.

## Original design and budgets

Description-only design uses `POST /v1beta/voices` with `type=prompted` and retains
the returned WAV audition. Provider voice IDs have finite retention; recipes and
audio cannot promise identical regeneration after provider/model changes.
[Official voice-design guide](https://ai.google.dev/gemini-api/docs/voice-design).

Every synthesis request is validated before dispatch. All request ceilings are
reserved atomically before a take queues, with separate accounting per call.
Confirmed work is retained; unsent calls release allocations; ambiguous results
hold reservations until evidenced reconciliation. Prices are checked again when
queued work runs. No automatic retry or queue drain is used by operator commands.

Voice creation accepts the versioned standard design preset or an explicit
operator allocation with evidence and expiry, plus explicit budgets. The standard
preset reserves US$0.25 locally; it is not a documented provider creation fee or
billing hard cap. Synthesis plans separate expected usage estimates, conservative
request allocations, held uncertain/rejected attempts, and verified invoice charges.
Every retry needs another allocation within the same total scene limit.

## Verification and remaining practical limits

Backend code and mocked acceptance tests cover original voice creation, reusable
characters, larger casts, ordered assembly, publication, and failure/recovery.
The earlier native two-speaker check succeeded with real Gemini audio. A separate
creative session produced the retained 52.36-second, 19-turn scene “The Mouse Who
Brought a Monster”: Nell uses prebuilt Leda; Rusk and Orro use original prompted
voices. Completed source lines span Flash and Flash-Lite. The user listened to and
liked the scene. This pass reuses it offline, makes no provider calls, and preserves
all original assets and failed evidence. Acting quality, long-form consistency and provider voice-ID
availability still require listening/live validation; code tests cannot certify
those qualities.

Use [backend operations](BACKEND_PRODUCTION.md) for commands and
[verification checkpoints](verification/) for exact evidence. The studio/station
boundary remains an immutable media package; creative state is never station input.
