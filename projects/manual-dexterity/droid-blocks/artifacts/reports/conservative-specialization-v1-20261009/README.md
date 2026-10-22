# Longer budget and conservative specialization — 9 October 2026

This completed known-condition study retained all 294 declared attempts:
144 targets and 150 retention guards. Of these, 292 safely completed. Two
standard targets stopped on the native `voltage_limited` flag after decisions
446 and 593; neither was retried. Both had valid checkpoints after the full
384-decision learning prefix, so their declared retention probes were released.

The [results](../../../docs/CONSERVATIVE_SPECIALIZATION_RESULTS.md) and
[prospective protocol](../../../docs/CONSERVATIVE_SPECIALIZATION_EXPERIMENT.md)
describe the fixed 384-learning plus 256-continuation budget, controls,
eligibility, individual-loss floors and interpretation. Conservative updates
gained a median +1.213250 over frozen, below the strict +3.2 gate, with a worst
target loss of 5.008546 and a worst old-condition probe loss of 34.334899.
Standard had a worst probe loss of 41.550677. Both methods failed combined
readiness. No adapted checkpoint was promoted; the original workshop actor
remains unchanged. The original rover-v3 challenge was not constructed here.

The [independent replay audit](verification/audit-v2.json) passed 24,391,284
checks with zero failures, verifying all 294 native streams, 187,919 categorical
actions and 942 fresh/prefix policy ZIPs. It checks exact tensors, Adam states,
declared settings, action RNG, local histories, native telemetry and reward
accounting. It does not rerun PPO gradients or establish external execution
chronology. Integrity passing is separate from scientific readiness and safe
completion. The audit also matched all 36 historical frozen 640-decision
streams and 36 standard 128-decision prefixes to October 7.

The first audit was interrupted by a container stop. Its empty
`verification/audit-v1.json` and the [interruption receipt](verification/audit-interruption-v1.json)
remain retained. The same inspected matching container was restarted; only
verification was repeated, into a new exclusive receipt. The
[recovery receipt](../conservative-specialization-preservation-v1-20261009/container-reuse-v2.json)
binds the unchanged container configuration, native binary, runtime and actor.
The [post-recovery preservation check](../conservative-specialization-preservation-v1-20261009/after-v2.json)
confirms all 5,360 earlier files / 2,023,061,685 bytes unchanged.

`protocol.json`, `lock.json` and `source_snapshot/` bind the prospective inputs
and action-generating sources. `targets/stage.json`, `targets/seal.json` and
`guards/stage.json` bind every attempt. The independent recorded
[reduction](verification/final-reduction-v1.json), audited
[report inputs](verification/report-inputs-v1.json) and final provenance in
`verification/final-provenance.json` retain the reductions and exact asset refs.
The independent [reduction crosscheck](verification/reduction-crosscheck-v1.json)
passed 1,275 exact comparisons against the final audit, with zero failures and
no native worlds or training.

Bulk assets stay local and ignored: 294 request tapes, 294 telemetry tapes,
294 fresh initial ZIPs and 648 update ZIPs, totaling 1,530 assets. The stopped
streams retain their actual prefixes. Reproducing this collection needs these
exact assets, the original sealed source releases, archived auditor/helper
sources and the bound Linux CPU runtime/native binary. A source checkout alone
does not provide full replay. Actual collection exposure was 187,919 decisions
and 939,593 successful native controls. Validation replay adds no scientific
samples. Never resume, overwrite or recollect this completed study.
