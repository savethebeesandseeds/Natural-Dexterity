# Radio Waajacu

A radio that plays the backend schedule. The web interface is for listening:
play/pause, now playing, volume, and a subtle animation driven by the audio.

`Backend content + schedule -> validated releases -> station -> HLS -> radio player`

## Listen

From this directory in PowerShell:

```powershell
.\radio.ps1 up
```

- Radio: <http://127.0.0.1:43120/>
- Direct station/player: <http://127.0.0.1:43121/>

Both addresses serve the same listening interface. Neither exposes content
generation, credentials, imports, take review, publication, or schedule editing.
The former studio web assets have been removed. Production remains backend work;
a separate production service can be considered later.

Until the backend publishes and schedules verified audio, the player shows
**Off air**. No demonstration broadcast is inserted into the live library.

## Listening canvas

The player uses a separate transparent botanical PNG and warm paper texture in
`web/player/`. The central `.visualizer-slot` contains a continuous 3D helix performance in the
`#sound-wave` canvas. Prepared cast/cues determine character identity and count;
per-character features support overlapping voices. Missing data leaves one
unassigned resting helix, which may follow mixed audio without identifying a
speaker. Local body deformation and optional prepared semantic centers remain separate. Playback and volume stay
small, programme details appear only for an airing, and the footer reads
“by waajacu.com”. Portrait layouts keep the plants at the outer edges.
Artwork generation prompts and hashes are recorded in
[visual asset provenance](docs/visuals/radio-botanical-v1.json). See
[prepared performance](docs/PERFORMANCE.md) for backend cue/feature installation
and the playback timing boundary. See [local audio and semantic audition](docs/LOCAL_PREVIEW.md) for the real recording preview and shared text projection into 3D position and spring rotation.

## Backend boundary

The production code includes reusable characters, original voice design from
descriptions, native/ordered multi-character rendering, script/voice revisions,
imports, takes, budgets, and immutable release publication. Backend jobs can use the
`Studio` Python service and validated release/control contracts. The explicit
production HTTP factory is retained for isolated internal tests; normal startup
runs only the listening entrance and independent station.

The station validates releases, copies their media into its own library, and
follows the backend's ordered rundown and fallback selection at airing boundaries.
It continues without a running content-generation process. Its journal, completed
programme replay, transcript, and backup capabilities remain backend services;
the listening page shows only the current broadcast.

The Gemini credential is configured privately; no key or generation setup appears
on the radio. Paid backend work still requires explicit episode/series limits.
One short native two-speaker check succeeded under a US$0.25 total cap; its
usage-based cost estimate was US$0.001410. It was not published or scheduled.
Use the [scene workflow](docs/CREATIVE_WORKFLOW.md) for one-command conversations,
character auditions, explicit continuation, and line revisions. Use
[backend production operations](docs/BACKEND_PRODUCTION.md) for release review,
publication, and advanced operations. See
[character/Gemini behavior](docs/CHARACTERS_AND_GEMINI.md) for implementation limits.
New feature checks use mocked providers; no additional paid generation is automatic.

## Container and operations

`radio-waajacu` uses the project's sole bind mount at `/workspace`, verified NVIDIA
GPU access, and matching loopback mappings `43120:43120` and `43121:43121`.
The pinned Debian container uses dependency-only `setup.sh`; lifecycle and service
operations remain in `radio.ps1`/`run.sh`. No new ports or volumes are needed.
See [container configuration](docs/CONTAINER.md).

```powershell
.\radio.ps1 status
.\radio.ps1 logs
.\radio.ps1 test
.\radio.ps1 check
.\radio.ps1 services-stop
.\radio.ps1 backup
.\radio.ps1 start
.\radio.ps1 stop
```

The legacy `studio-start`/`studio-stop` operation names control only the listener
entrance on 43120; they do not launch a production API. The station has its own
process. Launchers validate/reuse the managed container and preserve conflicts.

Persistent content, secrets, journals, releases, and reports stay in ignored
`.runtime/`. The credential is untracked and protected by the Windows secrets-folder
ACL. Never put it into source, command arguments, logs, or release packages.

Backups under `.runtime/backups/` require stopped services and omit credentials
by default. Restore into a fresh directory and provision credentials separately.
See [container operations](docs/CONTAINER.md) for explicit secret-inclusive backup
opt-in; such archives are unencrypted. Keep important backups on private,
independent storage. Existing recordings and production records are preserved.

## Organization

- `web/player/`: the sole web interface.
- `web_radio/listener.py`: read-only listening entrance; explicit station routes only.
- `web_radio/station.py`, `playout.py`: station, HLS, journal, and media.
- Other `web_radio/` modules: backend production, contracts, programming, recovery.
- `content/`: authored templates, voices, and provenance guidance.
- `packages/media-contract/`: release schema and validation documentation.
- `infra/`, `tests/`: environment contract and isolated verification.

See [ROADMAP.md](ROADMAP.md) and [verification](docs/verification/).
Development checks use labelled fixture audio and mocked Gemini responses.

## Software and content permissions

Original Waajacu software and its technical/development documentation use
[PolyForm Noncommercial 1.0.0](LICENSE). Preserve [NOTICE](NOTICE) when
redistributing as required by that license.

Stories, scripts, dialogue/transcripts, protected character material, and
finished recordings have [separate content permissions](CONTENT-PERMISSIONS.md).
Personal, noncommercial reading and listening are permitted through authorized
copies and services. Redistribution, adaptation, rebroadcast, and commercial
reuse need separate permission for rights Waajacu controls, except where law,
an existing grant, or an identified separate license permits the use. These
content permissions do not claim exclusive rights in unprotected AI output.

Keep authorship, creative-edit, source, and permission records as described in
[content records and release notices](content/README.md). That guide includes
a small public notice tied to the release ID and package digest.

Timed semantic source embeddings for visual performance are prepared separately in
the backend; see [embedding operations and contract](docs/EMBEDDINGS.md).
