# Character templates

These authored descriptions are input templates for the backend character registry.
They are not provider voice-design results or audition recordings. Register a
template through the backend production CLI, then select a prebuilt voice or
explicitly enqueue a description-only original design.

Immutable runtime character revisions, voice recipes, exact provider provenance,
auditions, and design jobs live under ignored `.runtime/studio/characters/`.
Scripts pin character and voice-recipe revisions when queued. Prebuilt Charon/Kore
remain in the older starter scripts for compatibility; existing releases are not
rewritten when templates or preferred recipes change.

Only prebuilt synthetic voices and original voices designed from descriptions are
supported. There is no voice replication or reference-recording upload workflow.
