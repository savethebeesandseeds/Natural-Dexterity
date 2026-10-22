# Web Radio development container

Authoritative configuration: [container.json](../infra/container.json).

- Name: `radio-waajacu`; pinned Debian 13 slim digest.
- Command: `/bin/bash /workspace/infra/container-entry.sh`; workdir `/workspace`.
- Sole read/write bind: this project's Windows directory -> `/workspace`.
- No named volumes, other workspace mounts, Docker socket, or extra devices.
- Matching loopback mappings: `127.0.0.1:43120:43120` (radio listening entrance),
  `127.0.0.1:43121:43121` (station/player).
- All NVIDIA GPUs, `compute,utility`; bridge networking; restart `no`; init;
  20-second stop timeout; no privileged mode.

`setup.sh` installs dependencies and configures the environment only. Python,
package caches, and tooling stay under `/opt/web-radio` inside the container.
The initial dependency resolution creates a hash-locked `requirements.lock`;
subsequent setup uses it. Debian APT sources use a dated snapshot and
`--no-install-recommends`. No Dockerfile or host development toolchain.

`radio.ps1` owns lifecycle and validates existing containers before reuse.
It never replaces or deletes a container, volume, or image. Conflicts are preserved.
`run.sh` owns service operations and tests. A stopped matching container is reused. `up` waits for dependency setup and starts
both services; normal `stop` stops managed processes gracefully before stopping
the container. Service PID records include kernel start ticks to guard reuse.

Persistent records, audio, secrets, journals, and HLS files live under ignored
`.runtime/` in this project. Back up that directory explicitly; Git is not a media
backup. Keep the listener entrance separate from the station process. Both web
ports are listener-only; production HTTP routes are not mounted by normal startup.
The internal process/launcher name `studio` is retained for PID compatibility and
now runs the listener facade. Neither port is exposed to the LAN by this launcher.

## Credential handling

The local Gemini key stays in ignored `.runtime/secrets/gemini-api-key`. The
directory has a protected Windows ACL allowing the workspace owner, SYSTEM, and
Administrators; ownership remains with the workspace owner. No container command,
environment argument, source snapshot, or release package contains its value.

Default offline backups omit `.runtime/secrets/` and all restored trees. Restore
credentials separately. If explicitly needed, opt in from the container:

```powershell
docker exec radio-waajacu /opt/web-radio/venv/bin/python -m web_radio.backup backup --include-secrets
```

This opt-in archive is unencrypted and contains credentials; keep it private.
The normal `radio.ps1 backup` action uses the credential-free default. Existing
archives remain unchanged and old credential-bearing archives can still restore.

A configured key does not authorize unlimited generation. Reserve explicit episode
and series budgets before dispatch; retain uncertain allocations and avoid automatic
retry. The first compatibility run uses its own retained database and US$0.25 total
series cap, with no background worker or publication.
