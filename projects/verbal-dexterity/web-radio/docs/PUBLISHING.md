# Publish Radio Waajacu

The public radio at https://radio.waajacu.com is a static GitHub Pages site.
Source remains under `Natural-Dexterity/projects/verbal-dexterity/web-radio/`.
The repository's `.github/workflows/radio-pages.yml` publishes on pushes to `main`
that change this project or its workflow. Other project pushes do not trigger it.
The workflow can also be run manually from GitHub Actions.

## What a push publishes

The workflow checks out the pushed player source, installs the hash-locked Python
dependencies inside a pinned Linux container, tests the publication contracts,
and assembles the site using the immutable media release named in `publication.json`.
It verifies both the archive SHA-256 and the public-file inventory before assembly.
The current HTML, CSS, JavaScript, images and notices replace those from the saved
snapshot, so player updates deploy automatically without uploading recordings again.
Only a verified Pages artifact can reach the deployment job.

Audio, programme metadata and prepared visual timelines come from the pinned public
release bundle. Production archives, credentials, raw embeddings, provider responses
and uncurated programmes are excluded. Neither GitHub Actions nor the public website
calls a generation provider. Listening works without the local computer running.
The separate local station and its live schedule remain independent.

## Local verification

Use the documented existing `radio-waajacu` container. Never run builds or project
Python on Windows. Choose a new destination for every export; existing outputs and
approved productions are preserved.

```powershell
docker exec -e PYTHONDONTWRITEBYTECODE=1 -w /workspace radio-waajacu /opt/web-radio/venv/bin/python tools/export_static.py --publication publication.json --output /workspace/.runtime/pages-RELEASE --platform github-pages
docker exec -e PYTHONDONTWRITEBYTECODE=1 -w /workspace radio-waajacu /opt/web-radio/venv/bin/python tools/export_static.py --verify /workspace/.runtime/pages-RELEASE
```

The GitHub Actions build uses the same assembler and verifier. GitHub Pages receives
an early strict meta Content Security Policy and its CNAME declaration; Cloudflare
response-header files are excluded. GitHub Pages itself controls HTTP cache headers.

## Publish another programme

1. Curate and prepare the approved production using the existing documented workflow.
2. Export all current curated programmes to a new public directory inside the container:

```powershell
docker exec -e PYTHONDONTWRITEBYTECODE=1 -w /workspace radio-waajacu /opt/web-radio/venv/bin/python tools/export_static.py --output /workspace/.runtime/media-RELEASE --platform github-pages
docker exec -e PYTHONDONTWRITEBYTECODE=1 -w /workspace radio-waajacu /opt/web-radio/venv/bin/python tools/package_public_media.py /workspace/.runtime/media-RELEASE --output /workspace/.runtime/radio-media-RELEASE.tar.gz
```

3. Publish that public archive as a uniquely versioned GitHub release asset. The host
GitHub CLI may upload this already-built file as an operating-system integration;
all generation, assembly and verification stay inside the managed Linux container.
4. Update `publication.json` with the versioned release URL, archive SHA-256 and public
inventory SHA-256 printed by the packaging command. Keep old release assets unchanged.
5. Commit and push the small source/catalogue pointer changes. Actions publishes the
new selection. Bulk media and private runtime files remain outside Git history.

The current static verifier retains a 25 MiB per-file ceiling. The three current WAVs
fit. A future larger programme needs a deliberate delivery-contract change before
publication; changing hosting alone does not bypass that check.

## Domain, checks and rollback

GitHub Pages is configured for `radio.waajacu.com`. Cloudflare provides DNS only for
this hostname, using a DNS-only CNAME to `savethebeesandseeds.github.io`.
No radio Worker, Tunnel or Cloudflare Pages project is required.

After deployment verify HTTPS, the default and every library selection, query links,
seek/play/pause/mute/volume, portrait layout and audio-driven visual synchronization.
Check audio Content-Type and byte-range responses, and verify private/API paths return
404. The site's public `site-manifest.json` records every deployed listening asset.

To roll back a player change, revert its source commit and push. To roll back the
programme selection, restore the preceding immutable media URL and checksums in
`publication.json` and push. No audio regeneration or station-schedule change is needed.
The Actions run and GitHub Pages environment record each deployment result.
