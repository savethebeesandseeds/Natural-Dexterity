# Creative content records and release notices

Stories, scripts, protected character material, and recordings use
[Web Radio's content permissions](../CONTENT-PERMISSIONS.md). The software
license does not automatically license these works.

## Keep a small private record

For each work and finished release, keep a record under
`.runtime/studio/content-permissions/<release_id>/`. This location is ignored
by Git and included with the studio's runtime backups. For drafts without a
release ID, use a working ID and add the release reference later.

Record the author/editor/performer and their actual contribution; the script
revision or source hash; substantive human writing, creative edits and
arrangements; imported sources and their license or permission evidence; and
any AI provider, model, generation/take reference, date, and applicable provider
terms. Note uncertain or unprotected elements instead of inventing an ownership
claim. Keep performer agreements and private evidence here, not in a public
notice. Do not put credentials in these records.

When a release is approved, record its `release_id` and `package_sha256` from
the validated manifest, the relevant credits and permissions, and the version
and hash of the content policy that was actually offered. Keep earlier policy
copies and grants; changing a later notice does not withdraw earlier rights.

## Attach a clear public reference

Copy [the public notice example](../docs/release-permissions.example.json),
replace its null identifiers with the validated release ID and package digest,
fill its policy URL with an accessible versioned or commit-pinned copy of
[the content policy](../CONTENT-PERMISSIONS.md), and set `example_only` to
`false` only after the notice describes that release. The template includes the
current policy's SHA-256; update that value if the offered policy bytes differ.
List only approved public credits, third-party terms, and specific additional
grants. A rights summary does not replace the full terms.

Place that approved notice and its link beside the release on the distribution
page or release announcement. A suitable short label is:

> Personal, noncommercial reading and listening. Redistribution, adaptation,
> rebroadcast, and commercial reuse need separate permission where Waajacu holds
> the relevant rights; see the linked content terms for scope and exceptions.

The current player and release API do not attach or display this notice
automatically. Keep it outside the immutable v1 package; do not add manifest
fields, extra package files, or edit a published transcript. See the
[media contract](../packages/media-contract/README.md#content-permissions).
Private provenance and permission documents are not the public notice.

## AI and rights records

Do not infer exclusive rights from a generator's output or from a notice.
[Gemini's terms](https://ai.google.dev/gemini-api/terms#use-of-generated-content)
say Google does not claim ownership of generated output and may generate similar
output for others. That does not establish copyright or clear third-party rights.
The [U.S. Copyright Office's 2025 summary](https://www.copyright.gov/newsnet/2025/1060.html)
distinguishes qualifying human expression or creative modifications/arrangements
from prompting alone under U.S. law; applicable rules can differ elsewhere.
