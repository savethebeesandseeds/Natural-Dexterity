# Learning artifacts

This directory stores immutable native policy artifacts and evaluation reports.
The Droid Blocks CLI creates files exclusively and refuses to overwrite an
existing path. Files whose names begin with `exploratory-` are train/validation
runs and are not eligible for the official held-out acceptance protocol.

The checked-in v1 artifact
`exploratory-light-search-g12-p16-seed0.json` and v2 artifact
`exploratory-v2-light-search-g12-p16-seed0.json` are pre-held-out design
evidence only. Neither is compatible with the official v3 manifest in
`config/learning_experiment_v3.json`, neither may be passed as an eligible v3
held-out policy, and neither contains a test outcome. Their validation and
train-only findings are documented in
[`../docs/LEARNING_DESIGN_HISTORY.md`](../docs/LEARNING_DESIGN_HISTORY.md).

The official frozen pre-held-out artifact is
[`light-search-policy-v3-seed0.json`](light-search-policy-v3-seed0.json). It
identifies `shared_linear_memory_v2` (33 binary64 parameters, 264-byte payload),
`cem_masked_best_sample_v1`, and selected generation 6.

| Identity | SHA-256 |
| --- | --- |
| Artifact canonical-payload `self_sha256` | `4dd4d6ccadecb0b3b69d3b3d412af83eff15208b19058701538dbdf83a4c32b5` |
| Exact serialized file bytes | `6f7d5069bdb884d75f1241629d437c039a2f7a486633d9ef2d7f6b70db0a5582` |
| Official protocol manifest | `f06d689351ddf10c6e987661d7a1f935df4cd8a3c911a76dd13395a6d4b49e93` |

The embedded self checksum covers canonical artifact payload content before
the self field is added. The file checksum covers the exact serialized bytes on
disk. The manifest checksum identifies the experiment protocol. These domains
are intentionally distinct.

Official training took `715.225449445 s` wall time with peak RSS `30,363,648`
bytes. A separate validation-only reload gained
`+0.09730391078745496 reward/s` with `16/16` wins against paired zero control:
negative x was `+0.09791938469996925` and `8/8`; positive x was
`+0.09668843687494068` and `8/8`. It recorded zero terminations, safety vetoes,
invalid reward-sensor samples, and invalid actuator-feedback samples. Mean
absolute effort was `0.31318514771991324`, maximum effort
`0.5999999976413114`, peak current `1.4999999759926383 A`, and peak temperature
`27.840471196252533 C`.

That `validate-policy` evidence is validation-only, diagnostic, and never an
official acceptance result. It explicitly reported
`test_rollouts_executed: false`. No held-out report exists; the official
held-out result remains pending and awaits explicit approval. Only an exclusive
report emitted by `evaluate-policy` after all manifest, checksum, and runtime
checks can contain the preregistered v3 held-out decision.

A compact machine-readable reduction of the validation stdout is preserved at
[`reports/light-search-policy-v3-seed0-validation-summary.json`](reports/light-search-policy-v3-seed0-validation-summary.json).
It deliberately records `raw_stdout_retained: false`, contains no acceptance
Boolean, and is not an official held-out report.

`evaluate-policy` requires `--output-report PATH` and installs that path
exclusively, refusing to overwrite an existing report. Exclusivity protects
only that local filename; it is not a cryptographic or technical one-shot
mechanism, because a different destination or copied workspace can execute the
same split again. The intended use is one preregistered held-out execution for
the v3 claim. Until that decision is fixed, the split and outcomes cannot
influence the v3 artifact, optimizer, checkpoint, acceptance rule, or claim.
Afterward, the evaluated worlds, topologies, traces, and outcomes may be
explicitly relabeled as known curriculum, replay, diagnostic, or training data
for a future protocol version. A descendant that uses them cannot present them
as unseen evidence and must reserve a fresh, preregistered, versioned challenge
split outside its own development loop. The present v3 held-out status remains
untouched and pending.
