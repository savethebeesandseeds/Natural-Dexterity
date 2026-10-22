# Creative scene workflow

Write a script, choose a budget, and run `perform` to hear a scene. It creates and renders only its new scene; other queued jobs stay untouched. Characters and immutable voice recipes use the existing character registry.

All work runs in the existing container. In PowerShell, define this short helper:

```powershell
function scene {
    docker exec radio-waajacu /opt/web-radio/venv/bin/python -m web_radio.scene_cli @args
}
scene --help
```

The default project root is `/workspace`, the mounted Web Radio folder. File arguments use container paths. No API key belongs in a command, script, policy, or source receipt; the renderer uses the already configured private credential.

## First scene

Load fictional character identities and choose prebuilt voices. These commands create registry records and recipes; they do not generate audio:

```powershell
scene character load --file /workspace/content/characters/moss/character.json
scene character load --file /workspace/content/characters/wren/character.json
scene character prebuilt --id moss --voice Charon --model gemini-3.8-flash-tts
scene character prebuilt --id wren --voice Kore --model gemini-3.8-flash-tts
scene character recipes --id moss
```

For characters already registered, use `character show --id moss` and `character recipes --id moss`. Loading preserves conflicting records instead of replacing them. Voice recipes stay in the same registry used by the existing production tools.

Save an authored script, for example `/workspace/content/series/my-story/scene-one.json`:

```json
{
  "programme_id": "scene-one",
  "series_id": "my-story",
  "kind": "programme",
  "title": "The Open Door",
  "language": "en",
  "speakers": [
    {"name": "Moss", "character_id": "moss"},
    {"name": "Wren", "character_id": "wren"}
  ],
  "turns": [
    {"speaker": "Moss", "text": "The door is open.", "style": "Quiet surprise."},
    {"speaker": "Wren", "text": "Then we should go inside.", "style": "Warm, but cautious."}
  ]
}
```

Speaker names match the names in `turns`; `character_id` selects the registered fictional character. An optional `voice_recipe_id` pins a particular recipe. Scene creation freezes the selected character and recipe revisions.

Inspect the plan before rendering:

```powershell
scene plan --file /workspace/content/series/my-story/scene-one.json --budget 0.25
scene perform --file /workspace/content/series/my-story/scene-one.json --budget 0.25
```

The USD amount is an example explicit scene limit; choose the approved amount after inspecting the plan. `plan` does not reserve money or contact the provider. `perform` freezes the script and cast, renders its newly created scene, and assembles completed lines. It does not publish or schedule the radio programme. Use the returned `id` to inspect it later:

```powershell
scene show --job SCENE_ID
```

The status response contains `lines`, `counts`, `wait`, `outputs`, and `budget`. Treat the returned `outputs` descriptors as authoritative. `outputs.scene_summary` identifies a compact summary at `.runtime/studio/scenes/SCENE_ID/scene-summary.json`, with its checksum, playable files, models used, reuse counts, and receipt path.

Finished media live under `.runtime/studio/scenes/SCENE_ID/output/`: `scene.wav`, `scene.mp3`, `transcript.txt`, `assembly.json`, and optional per-character WAV stems. The receipt retains timing and source provenance.

To inspect a frozen scene before generating, use the separate queue/render steps:

```powershell
scene create --file /workspace/content/series/my-story/scene-one.json --budget 0.25
scene show --job SCENE_ID
scene render --job SCENE_ID
```

`create` queues only; `render` selects that exact scene. Neither command drains a queue.

## Listen, change one line, listen again

A short audition creates a queued single-line scene using a registered character:

```powershell
scene audition --character moss --text "The door is open." --style "Quiet surprise." --budget 0.25
scene render --job AUDITION_ID
```

Add `--recipe RECIPE_ID` to audition a particular existing recipe. This does not design a new voice.

For an original voice, queue a text description through the same entrypoint:

```powershell
scene character design --id moss --description "An original fictional forest guide: gentle rasp, patient pace, dry warmth." --model gemini-3.8-flash-tts --budget 0.25
scene character design-show --job DESIGN_ID
scene character design-run --job DESIGN_ID
scene character recipes --id moss
```

The `standard` preset provides a versioned **$0.25 local allocation**, not a provider price quote or guaranteed billing cap. `design` queues only; `design-run` explicitly selects one job. Its returned recipe and sample are retained in the existing registry. Person voice replication is not part of this system.

`--series-budget` defaults to the supplied `--budget`; prior spent or held allocations may leave too little room for another design. The command never increases that cap automatically to get a job through. An explicit larger series budget is a separate caller choice. `character design-recover --job DESIGN_ID` uses retained evidence without generating another voice. Advanced billing reconciliation is described in the [voice-design operations](BACKEND_PRODUCTION.md#original-voices-from-descriptions).

Revise the second spoken line without replacing the original scene:

```powershell
scene revise --job SCENE_ID --line 2 --text "Wait. Someone is already inside." --style "Barely a whisper." --budget 0.25
scene render --job REVISION_ID
```

Line numbers start at **1**. A revision creates a new scene and preserves reusable unchanged performances. You may change text, direction, a line's `--recipe`, or a combination. `--style ""` clears a line's direction. The original scene and its evidence remain available.

## Resume a paused scene

```powershell
scene show --job SCENE_ID
scene recover --job SCENE_ID
scene resume --job SCENE_ID
```

Inspect the saved wait/diagnostic information first. `recover` only verifies retained local evidence; it never contacts the provider. Resume reuses retained completed work; it is an explicit operation, with no background queue drain. An unresolved dispatched request needs inspection rather than a blind paid retry.

The original scene budget covers successful work, planned requests, held allocations, and any permitted new attempts. Live `accounting` separates:

| Field | Meaning |
| --- | --- |
| `conservative_planned_reservations_microdollars` | Reserved for requests that have not dispatched. |
| `in_flight_reserved_microdollars` | A dispatch is currently in progress. |
| `unresolved_held_microdollars` | A dispatched outcome still needs accounting evidence. |
| `token_estimates_or_allocations_consumed_microdollars` | Token estimates or conservative allocations consumed; not an invoice. |
| `verified_invoice_microdollars` | Charges explicitly reconciled from external billing evidence. |

Imported legacy source accounting remains separate in `legacy_source_accounting`; reuse neither releases old holds nor charges them to the new scene.

Provider rejections, including quota rejections, do not automatically prove a zero invoice charge. To reconcile one held attempt after checking actual billing evidence:

```powershell
scene reconcile --job SCENE_ID --attempt ATTEMPT_ID --charged VERIFIED_USD --evidence "Verified invoice reference and charge"
```

Use the attempt ID from `show`. Reconciliation records accounting evidence; it does not render or retry. Successful cost estimates, unresolved allocations, and operator-verified invoice charges remain distinct.

## Rendering policy

Flash TTS is the default. Common choices have direct flags; no policy file is needed:

```powershell
scene perform --file /workspace/PATH/scene.json --budget 0.25 --model gemini-3.8-flash-tts --max-output-tokens 2048
scene plan --file /workspace/PATH/scene.json --budget 0.25 --model gemini-3.8-flash-tts --fallback-model gemini-3.8-flash-lite-tts
```

`--fallback-model` explicitly permits daily-quota fallback in the declared primary/secondary order. Lite is never selected silently. These model flags also work with `create` and `import-sources`; `--max-output-tokens` works with `plan`, `create`, and `perform`.

For additional controls, save policy JSON and pass `--policy /workspace/PATH/policy.json`. Flags and explicit policy fields must agree; contradictory model orders, fallback choices, or output caps are rejected:

```json
{
  "schema_version": 1,
  "models": ["gemini-3.8-flash-tts"],
  "fallback_on_daily": false,
  "max_output_tokens": 1024,
  "max_quota_retries": 2,
  "max_wait_seconds": 60,
  "min_interval_seconds": 0,
  "stems": true
}
```

Model fallback is a creative choice and can change delivery. With policy JSON, explicitly list Flash then Lite and set `fallback_on_daily` to `true`. `max_output_tokens` is an output cap, so longer lines may need a larger approved cap and budget.

To change only a selected line's output cap after review, put `{"max_output_tokens": 2048}` in a policy file and create a revision:

```powershell
scene revise --job SCENE_ID --line 2 --policy /workspace/PATH/larger-cap.json --budget 0.25
scene render --job REVISION_ID
```

Completed unchanged lines retain their audio; a larger cap never silently changes the original scene.

Optional `quota_limits` records observed limits for each model using `requests_per_minute`, `requests_per_day`, `observed_at`, and `evidence`. Use observed project information; missing or unknown limits do not establish available quota.

Structured diagnostics retain bounded, sanitized provider code/status, field violations, quota observations, and retry hints. Learned observations are private to the credential/model; changing the credential does not carry them over. The retained scene observed 3 RPM and 10 RPD for its Flash FreeTier context, which is evidence for that request scope rather than a universal model limit. Plans can show known observations without probing the provider.

`max_wait_seconds` limits each individual automatic wait. A longer hint pauses the job for an explicit resume. Known minute throttling can retry only under `max_quota_retries` and the remaining total budget. Unknown quota errors without a retry hint pause after the first call. Daily exhaustion follows the provider hint when present, otherwise the documented next Pacific-midnight reset; declared fallback is the only alternative model path. An uncertain transport outcome is never retried automatically.

The default 1024-token output ceiling is explicit in each retained request. Incomplete responses, reported usage at that ceiling, or explicitly exposed token-limit stop observations require review. Stop fields are retained defensively when the provider exposes them; their absence cannot certify that an acting performance is complete. Prices/presets have versioned evidence and expiry, and dispatch refuses expired pricing evidence.

## Reuse retained recordings

A previous partial take can supply verified completed lines without regenerating them. Pass the authored Script plus a source receipt:

```powershell
scene import-sources --file /workspace/PATH/scene.json --sources /workspace/PATH/source-receipt.json --budget 0.25
```

The receipt is an object containing a `sources` list. Each entry identifies a zero-based script `turn_index`, a retained `take_id`, and its `request_index`:

```json
{
  "sources": [
    {"turn_index": 0, "take_id": "RETAINED_TAKE_ID", "request_index": 0}
  ]
}
```

An existing receipt may include additional retained metadata; the service verifies it. Imports check the original frozen script, speaker, text, direction, completed request state, and retained source hashes. This is reuse of known production evidence, not an arbitrary audio upload. Only missing lines require allocations. If every line is supplied, the service can finalize locally; import itself never calls the provider.

## Timing and retained-source proof

Assembly measures every selected recording, converts it to 48 kHz mono PCM16, concatenates exact line boundaries, and applies one measured static gain. Original provider WAVs remain unchanged. Character stems occupy that same full timeline and contain silence outside their known line intervals; they are derived from selected sources, without speaker separation or inferred mixed-audio identities.

`assembly.json` includes a `performance_candidate` bound to the final scene WAV hash and frame count. It is private preparation data, not an installed release sidecar. Acceptance must verify and bind it to the accepted release. Any later processing that changes PCM or timing requires renewed alignment/hash checks. The embeddings backend accepts only the exact verified scene-stem paths and retains its existing release/hash/frame/editorial checks.

The retained “The Mouse Who Brought a Monster” scene was adopted offline as job `3a6459e05b81474082767b5f3b113274`: 19 lines, 52.36 seconds, Flash and Flash-Lite source turns. This proof made zero key reads or provider calls and preserved 123 original files and existing rows across 11 legacy tables. It did not publish or schedule the result. Local evidence is under `.runtime/verification/retained-scene-adoption-20261022T205758Z-80rzagbh/`.

Advanced radio acceptance, immutable publication, and scheduling remain in the [backend operations workflow](BACKEND_PRODUCTION.md).
