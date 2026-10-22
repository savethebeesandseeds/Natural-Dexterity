# Backend production operations

All commands run in the existing `radio-waajacu` container. The radio website stays
listener-only. Nothing here starts a production web service or a background queue
worker. The key is read privately when an explicitly selected job dispatches.

Prefix the commands below with:

```powershell
docker exec radio-waajacu /opt/web-radio/venv/bin/python -m web_radio.production
```

Use `--help` for complete argument lists. JSON results are safe for operator logs:
the CLI never returns credentials or raw exception/validation input values.

## Scene workflow

The preferred scene CLI is described in [CREATIVE_WORKFLOW.md](CREATIVE_WORKFLOW.md).
It creates a private scene job, freezes the cast and per-line requests, enforces one
explicit total budget, and exports WAV/MP3, transcript, source provenance, and optional
character stems. Resume dispatches unfinished lines only; a deliberate revision can
replace one line or recipe while retaining the other recordings. Neither operation
publishes or schedules audio. Native take operations below remain available.

## Inspect without spending

```text
plan --script /workspace/content/series/first-light/programme.json --model gemini-3.8-flash-lite-tts
status
takes
```

`plan` reports the strategy, number of requests, conservative input bound, and
combined reservation ceiling. It does not reserve money or contact Gemini.
A long dialogue may need multiple bounded native chunks. A larger/designed cast
needs one request per turn, so its combined allocation can be much higher.

## Reusable characters

Create identities from the supplied fictional character inputs, then select
prebuilt recipes. These commands do not generate audio:

```text
character create --file /workspace/content/characters/moss/character.json
character create --file /workspace/content/characters/wren/character.json
character create --file /workspace/content/characters/ash/character.json
character prebuilt --id moss --voice Charon --model gemini-3.8-flash-lite-tts
character prebuilt --id wren --voice Kore --model gemini-3.8-flash-lite-tts
character prebuilt --id ash --voice Puck --model gemini-3.8-flash-lite-tts
character list
character recipes --id ash
plan --script /workspace/content/series/first-light/three-characters.example.json --model gemini-3.8-flash-lite-tts
```

Creation preserves conflicts instead of overwriting an existing character.
Updates require the current `--expected-revision`. Recipes are immutable.
A script can name an alias and `character_id`, with optional pinned
`character_revision` and `voice_recipe_id`. Queueing resolves and snapshots exact
identities/recipes; later character edits cannot alter an existing take.

## Original voices from descriptions

Voice design accepts text descriptions only. No person/reference-audio upload,
voice replication, or arbitrary custom-ID registration is implemented.

The scene CLI offers a versioned `standard` design preset with a US$0.25 local
allocation, retained rationale/expiry, and explicit character/series limits. It
queues only; `character design-run --job JOB_ID` is the separate dispatch operation.
See [the design commands](CREATIVE_WORKFLOW.md). The provider does not document a
separate guaranteed creation fee or output ceiling, so this preset is a conservative
local allocation rather than a provider price quote or billing guarantee. Usage
estimates and verified invoice charges stay separate.

For advanced overrides, `content/characters/voice-design.example.json` still accepts
operator-supplied allocation evidence and expiry. Its placeholder values cannot be
dispatched. Never treat mock-test allocations as real prices.

```text
voice-design enqueue --file /workspace/PATH/design-request.json --episode-budget USD_LIMIT --series-budget USD_LIMIT
voice-design show --job JOB_ID
voice-design run --job JOB_ID
voice-design recover --job JOB_ID
voice-design reconcile --job JOB_ID --charged VERIFIED_USD --evidence VERIFIED_BILLING_NOTE
```

`enqueue` reserves and queues only. `run` sends the selected job once.
Returned prompted provenance, exact inputs/result identity, and audition WAV are
retained before a recipe becomes usable. Replicated/unknown results are rejected.
Recovery uses verified retained responses or marks the outcome uncertain; it never
creates another voice automatically. Expired voice IDs require a new recipe;
retained audio preserves the original performance.

## Render, review, publish

```text
render --script /workspace/PATH/scene.json --model gemini-3.8-flash-lite-tts --episode-budget USD_LIMIT --series-budget USD_LIMIT
run --take TAKE_ID
takes --take TAKE_ID
decision --take TAKE_ID --decision accepted
publish --take TAKE_ID
```

`render` saves a revision and queues a take. All planned request allocations must
fit before any reservation/take transaction commits. `run` processes only that
take; it never drains older queued jobs, accepts, publishes, or schedules output.

Each request retains its original WAV/metadata and settles separately. On failure,
completed work stays recorded, undispatched calls are cancelled/released, and an
uncertain dispatched call retains its reservation. No automatic retry occurs.
Programmatic billing reconciliation accepts an explicit request index for a
multi-request take. Recovery cannot silently replace a frozen cast or bypass
expired prices.

Larger conversations are normalized into ordered PCM and prepared as one programme.
This is sequential dialogue assembly, with no promised simultaneous overlap or
custom sound-design editor. Complete output uses the existing acceptance and
immutable-release contract. Station scheduling remains a separate backend handoff.

For a finished recording, `import --file WAV --script JSON` preserves original bytes,
binds the exact script, and uses the same review/publication path without TTS spend.

## Persistence, recovery, and evidence

Character records, voice recipes, and design artifacts live under
`.runtime/studio/characters/`. Design job state shares the production SQLite
ledger under `.runtime/studio/` with takes, plans, per-request receipts, budgets,
and revisions.
Keys stay in the ignored, protected secrets directory. Backups exclude secrets
by default and reject active generation/design leases. Stop all production jobs
and authoring operations for an offline snapshot; direct Python writers must
coordinate with that maintenance window too.

No paid generation was used in this engineering pass. The creative session has
separately retained a 52.36-second, 19-turn scene with Nell (prebuilt Leda), Rusk and
Orro (original prompted voices), using recorded Flash and Flash-Lite turns. The user
listened to and liked that scene. Its rejected design and quota diagnostics remain
part of the evidence; no undocumented age rule is inferred from a rejection.

The scene workflow is verified with mocked failures/recovery and offline assembly
of those existing recordings. This establishes actual saved-source integration;
new live resume/fallback paths and future voices still need explicitly authorized
provider work and listening review. Original assets, old ledger rows and unresolved
holds are preserved; this pass does not claim to reconcile provider billing.
