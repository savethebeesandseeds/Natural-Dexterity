# Media contract v1

The station accepts a release through this contract only. Production records stay
private to the studio and are not required for broadcasting.

A package contains `manifest.json`, one signed-16-bit PCM WAV programme/ident
(1 or 2 channels, 8–192 kHz), and one nonempty UTF-8 transcript. Every asset is
declared by role, relative path, MIME type, byte size, and SHA-256.

`package_sha256` is SHA-256 over RFC 8785 canonical JSON for the manifest with
that field omitted. Asset hashes bind exact media bytes; whitespace and object-key
order do not change identity. Receive the expected digest through a trusted
publication record: a matching self-reported hash does not authenticate a publisher.

`manifest.schema.json` is generated from strict Pydantic types. Schema validation
checks structure; Python package validation additionally checks paths, inventory,
symlinks, duplicate keys, byte sizes, checksums, transcripts, decodable audio, and
measured duration (100 ms tolerance).

## Python interface

```python
from web_radio.contract import create_release, import_release, validate_package

manifest = create_release(
    "/workspace/.runtime/studio/mix.wav",
    "Presenter: Welcome to Web Radio.",
    "/workspace/.runtime/studio/releases/my-release",
    programme_id="first-programme",
    title="Our first programme",
    release_id="my-release",
)
manifest, station_path = import_release(
    "/workspace/.runtime/studio/releases/my-release",
    "/workspace/.runtime/station/releases",
    manifest.package_sha256,
)
```

`output_dir` means the complete final package directory. Imports land at
`destination_root/release_id`. Existing releases are never overwritten.
Reimporting the same trusted release is idempotent. Both paths validate staged
bytes before a same-filesystem atomic rename. Playback uses the imported copy.

Regenerate the schema inside the managed development container:

```sh
/opt/web-radio/venv/bin/python -m web_radio.contract schema packages/media-contract/manifest.schema.json
```

Generate a valid synthetic example package (clearly labelled as a test tone):

```sh
PYTHONPATH=/workspace /opt/web-radio/venv/bin/python packages/media-contract/make_example.py
```

The retained example is written to ignored `.runtime/examples/contract-v1/`;
no bulk audio is stored in Git.

## Resource limits

Contract v1 accepts at most 128 MiB of audio, 1 MiB of UTF-8 transcript,
and 30 minutes of decoded audio. Declared sizes and duration are schema-bound;
actual bytes are bounded while copying, hashing, and reading transcripts.
Decoded duration is checked before full validation decode. Audio validation
forces the WAV demuxer with file/pipe protocols only, so a disguised playlist
cannot fetch remote media. Studio upload limits may be narrower.

## Content permissions

Successful package validation checks the release's integrity and media format;
it does not establish ownership or permission to publish or reuse its content.
The software license does not license the programme's creative works. See
[content permissions](../../CONTENT-PERMISSIONS.md) and
[content records and release notices](../../content/README.md).

Use [the public notice example](../../docs/release-permissions.example.json)
to identify the exact `release_id`, `package_sha256`, policy version/URL/hash,
public credits, third-party terms, and any specific additional grants. Use the
package digest from the validated manifest; an audio asset's hash/ETag is not
the package digest. Publish the notice alongside the release's distribution
page or announcement. Keep private authorship/permission evidence in the
ignored studio runtime, separate from the public notice.

V1 does not carry a content-permissions field, and the player/API does not
automatically show one. Do not add the notice as an extra manifest field or
package asset: v1 requires an exact inventory. Do not modify existing release
bytes. This documented notice workflow leaves the v1 contract unchanged.
