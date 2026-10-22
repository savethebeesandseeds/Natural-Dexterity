# Prepared visual performance

The public listener consumes optional, prepared character cues and features.
It never generates speech, infers character identities from a mixed waveform,
or exposes preparation controls. Missing data is normal: the radio still plays.

Performance data is separate from immutable v1 audio releases. Do not add
extra files to a release directory or alter its two-asset media contract.

## Contract v1

The JSON response has no envelope:

```json
{
  "schema_version": 1,
  "release_id": "approved-programme-release",
  "package_sha256": "64 lowercase hexadecimal characters from the accepted manifest",
  "duration_ms": 1000,
  "cast": [
    {"id": "first-character", "color": "#668877"},
    {"id": "second-character", "color": "#996677"}
  ],
  "cues": [
    {"speaker_id": "first-character", "start_ms": 0, "end_ms": 600},
    {"speaker_id": "second-character", "start_ms": 500, "end_ms": 1000}
  ]
}
```

Replace the illustrative digest string with the exact accepted release digest.
All fields are strict; unknown fields, duplicate JSON keys, non-finite numbers,
and Boolean values in integer/numeric fields are rejected.

- Cast: 1–12 unique, stable editorial character IDs. IDs use the release slug
  grammar (lowercase letters/digits and hyphens, at most 128 characters).
  Colors are exactly `#RRGGBB`. Cast length follows the programme, not a fixed
  request for three speakers. Do not use provider voice IDs as character identity.
- Cues: sorted by nondecreasing `start_ms`, nonempty half-open intervals
  `[start_ms, end_ms)` within the release duration. Different characters
  may overlap; a character's own cues may not. At most 100,000 cues.
- Identity: release ID, package SHA-256 and duration must all match a previously
  accepted, verified station-owned audio release.
- Total UTF-8 document size: at most 8 MiB.

An optional `features` object supplies independent per-character deformation:

```json
{
  "hop_ms": 1000,
  "tracks": {
    "first-character": {
      "energy": [0.0, 0.0],
      "low": [0.0, 0.0],
      "mid": [0.0, 0.0],
      "high": [0.0, 0.0]
    }
  }
}
```

This feature example fits the 1,000 ms document above and may be added as its
`features` field. `hop_ms` is an integer from 20 to 1,000. Each included
track requires all four arrays of finite normalized values from 0 to 1.
All four arrays have exactly `ceil(duration_ms / hop_ms) + 1` values.
Sample `i` represents time `i * hop_ms`; the final sample can extend
slightly past the release endpoint to support interpolation. At most 500,000
feature values across all tracks/bands are allowed. Track keys must be cast IDs;
a subset is permitted, and a missing track means independent features are absent.

Features must be prepared from known per-character recordings/stems or trustworthy
aligned source features. Timed cues must come from verified editorial/alignment
data. This module performs no diarization or automatic alignment. The mixed live
waveform alone cannot independently animate simultaneous voices. In cue-only
mode, the mixed analyser can be attributed only while exactly one character is
active; during overlap, independent character deformation must wait for tracks.

## Backend installation

Authored JSON is installed only after the release has been accepted by the
station. Inside the existing documented development container:

```sh
/opt/web-radio/venv/bin/python -m web_radio.performance install \
  /path/to/authored-performance.json --runtime /workspace/.runtime
```

This verifies the accepted journal digest and station-owned release, validates
the document, then writes atomically to:

```text
<runtime>/station/performance/<release_id>.json
```

Existing performance is preserved unless `--replace` is explicitly supplied.
Preparation does not import releases, change the rundown, modify media, or start
broadcasts. No install command was run against actual programme data for the
initial visualizer integration.

## Public consumption and playback timing

`GET /api/performance/{release_id}` returns the validated document directly.
Unknown releases/missing sidecars return 404. Invalid/stale performance returns
409. Query parameters and write methods are unsupported. The listening entrance
proxies only this fixed route to its fixed station; it forwards no browser
credentials and applies the 8 MiB response limit. Responses use `no-store`.

Find the airing with `/api/now?at=<actual playback timestamp>`. Subtract its
calibrated `stream_start` from that playback timestamp to obtain the release
offset, then sample cues/features at that offset. Account for buffering, pause,
seek, reconnect and HLS discontinuities. Wall clock and the audible playhead are
different. Native HLS live-window estimates must not be presented as exact
character timing. Without trustworthy timing, retain quiet unassigned spirits.

Local helix deformation and fixed scene position are separate. Text-derived
dimensions, narrative choreography and semantic movement remain deferred.


## Initial visual mapping

The player draws one continuous elongated spring per cast member using a 3D
projection, depth shading, translucent strands and a short fading trail.
Energy changes local radius/coil spacing; low/mid/high bands add small local
ripples. Attack/release smoothing avoids sample-level jitter. These are an
initial artistic mapping, not a physical or lossless audio representation.

The renderer keeps scene anchors independent of body deformation. Stable
editorial IDs and explicit colors preserve identity while speakers take turns.
Missing cast data shows one neutral unassigned helix; there is no invented
three-person cast. Without a reliable playhead timestamp, character-specific
deformation stays quiet. Inactive cast members remain visible. Motion is capped
at 30 fps, suspended in hidden tabs, and becomes static with reduced motion.

The current exact clock uses hls.js programme-date-time at the actual playhead.
Native HLS live-window estimates remain available for title metadata, but never
drive character identity. Pausing retains a quiet cast; reconnect/seek samples
the new playback position. No real audio generation or live schedule mutation is
part of this implementation.


## Verification fixture

The isolated browser fixture is in `tests/helix-browser.html` with its external
CSS/JavaScript siblings. It covers continuous geometry, smoothing, cue boundaries,
overlap isolation, inherited-key character IDs, exact-clock guards, request races,
and cast changes. It is not mounted by either radio service.

The Codex in-app browser disallows local `file:` URLs, so this standalone
JavaScript fixture was not executed during the initial integration. The actual
running player was inspected, with no console errors; backend cue/feature and
listener/station behavior passed focused tests.
