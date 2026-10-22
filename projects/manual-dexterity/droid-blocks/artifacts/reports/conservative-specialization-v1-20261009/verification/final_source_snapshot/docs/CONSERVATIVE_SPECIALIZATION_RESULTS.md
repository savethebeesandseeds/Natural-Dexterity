# Longer budget and conservative specialization results

Recorded 9 October 2026.

The conservative primary method did not meet combined readiness; the standard secondary method did not meet it. Readiness requires useful whole-opportunity benefit, acceptable individual target losses, sampled retention and acceptable individual probe losses together. No adapted checkpoint is promoted into the workshop.

Conservative updates' median seed mean gain over frozen was +1.213250, against a usefulness threshold strictly above +3.2. The worst target difference was -5.008546; the worst earlier-condition probe difference was -34.334899. Increasing experience and lowering both learning rate and clipping did not produce useful, reliable incremental specialization in this tested scope.

The [prospective protocol](CONSERVATIVE_SPECIALIZATION_EXPERIMENT.md) tests 384 learning decisions inside each 640-decision opportunity, followed by 256 fixed-weight decisions without resetting the world, history or RNG. It uses three original pooled source releases and twelve already exposed length-5 conditions. This is known-condition specialization; the original rover-v3 challenge remains unexposed.

## Whole opportunity benefit

The conservative package lowers learning rate from 0.0003 to 0.0001 and PPO clipping from 0.2 to 0.1. Standard and scratch retain the original settings. All three learning arms use fresh full actor/critic optimizers, six 64-decision rollouts and four epochs each, for 24 Adam steps. The package comparison does not isolate the two hyperparameters separately.

Useful benefit over frozen requires all 36 corresponding pairs safe/complete, median of the three seed mean gains strictly above +3.2, and at least two positive seed means. Scratch contrasts use their separate median >0 threshold. Three source seeds are training replicates; twelve cases per seed are correlated. For incomplete attempts, differences subtract recorded returns over unequal durations: the stopped attempt's prefix versus its completed comparator's full opportunity. They cannot establish 640-decision benefit or an equal-length comparison; eligibility and actual paired exposure are shown below.

| Method / comparator | Seed 101 mean | Seed 202 | Seed 303 | Median seed mean | Positive seeds | Eligible pairs | Benefit gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| conservative / frozen | +0.025341 | +1.213250 | +1.424058 | +1.213250 | 3/3 | 36/36 | FAIL |
| conservative / scratch | +18.275274 | +20.255306 | +22.267553 | +20.255306 | 3/3 | 36/36 | PASS |
| standard / frozen | +2.291397 | -0.019448 | -0.508006 | -0.019448 | 1/3 | 34/36 | FAIL |
| standard / scratch | +20.541330 | +19.022608 | +20.335489 | +20.335489 | 3/3 | 34/36 | FAIL |

## Individual reliability and combined readiness

Target reliability requires every whole target gain at least −3.2; probe reliability requires every sampled old-condition gain at least −3.2. These empirical bounds apply only to the tested cases. Physical safety stops and return-loss floors are different conditions.

| Method | Useful benefit | Target reliability | Sampled retention | Probe reliability | Combined readiness |
| --- | --- | --- | --- | --- | --- |
| conservative | FAIL | FAIL | PASS | FAIL | FAIL |
| standard | FAIL | FAIL | FAIL | FAIL | FAIL |

| Method / frozen | Recorded wins / ties / losses | Worst recorded return difference | Recorded losses below −3.2 | Prefix losses below −1.92 |
| --- | ---: | ---: | ---: | ---: |
| conservative | 24 / 0 / 12 | -5.008546 | 1 | 2 |
| standard | 28 / 0 / 8 | -20.384581 | 3 | 0 |

## Learning prefix and fixed continuation

| Method / frozen | Seed | Recorded total return difference | Prefix difference at 384 | Recorded continuation difference up to 256 |
| --- | ---: | ---: | ---: | ---: |
| conservative | 101 | +0.025341 | -0.660075 | +0.685416 |
| conservative | 202 | +1.213250 | +0.649816 | +0.563434 |
| conservative | 303 | +1.424058 | +0.543614 | +0.880444 |
| standard | 101 | +2.291397 | +0.786659 | +1.504737 |
| standard | 202 | -0.019448 | +1.428974 | -1.448422 |
| standard | 303 | -0.508006 | +0.538773 | -1.046779 |

Continuation states follow each arm's earlier actions, so these differences do not isolate weights at an identical state. Return is the native light-plus-IMU integral, separate from PPO values, lux and signed energy.

## Every paired target outcome

| Seed | Exposed condition | Conservative minus frozen | Standard minus frozen | Eligible conservative / standard | Decisions conservative / standard / frozen |
| ---: | --- | ---: | ---: | --- | --- |
| 101 | long-upper-base-left | +0.240383 | +0.211448 | yes / yes | 640 / 640 / 640 |
| 101 | long-upper-base-right | -0.374826 | -0.104800 | yes / yes | 640 / 640 / 640 |
| 101 | long-upper-elbow-left | +1.158977 | +4.169948 | yes / yes | 640 / 640 / 640 |
| 101 | long-upper-elbow-right | +0.398695 | +0.693057 | yes / yes | 640 / 640 / 640 |
| 101 | long-lower-base-left | -0.084049 | -0.094395 | yes / yes | 640 / 640 / 640 |
| 101 | long-lower-base-right | +0.505175 | +0.389159 | yes / yes | 640 / 640 / 640 |
| 101 | long-lower-elbow-left | +0.603033 | +0.704754 | yes / yes | 640 / 640 / 640 |
| 101 | long-lower-elbow-right | +4.095367 | +19.767985 | yes / yes | 640 / 640 / 640 |
| 101 | weighted-base-left | +0.253431 | +0.274830 | yes / yes | 640 / 640 / 640 |
| 101 | weighted-base-right | +0.372250 | +0.593049 | yes / yes | 640 / 640 / 640 |
| 101 | weighted-elbow-left | -5.008546 | +0.853158 | yes / yes | 640 / 640 / 640 |
| 101 | weighted-elbow-right | -1.855801 | +0.038570 | yes / yes | 640 / 640 / 640 |
| 202 | long-upper-base-left | -0.100437 | +0.229549 | yes / yes | 640 / 640 / 640 |
| 202 | long-upper-base-right | -0.052114 | -0.548056 | yes / yes | 640 / 640 / 640 |
| 202 | long-upper-elbow-left | +1.128399 | +1.977514 | yes / yes | 640 / 640 / 640 |
| 202 | long-upper-elbow-right | +0.204521 | +0.138874 | yes / yes | 640 / 640 / 640 |
| 202 | long-lower-base-left | +0.380337 | +0.284888 | yes / yes | 640 / 640 / 640 |
| 202 | long-lower-base-right | -0.071344 | +0.275680 | yes / yes | 640 / 640 / 640 |
| 202 | long-lower-elbow-left | +0.214602 | +1.074725 | yes / yes | 640 / 640 / 640 |
| 202 | long-lower-elbow-right | +7.073165 | +8.368621 | yes / yes | 640 / 640 / 640 |
| 202 | weighted-base-left | -0.035909 | +0.025140 | yes / yes | 640 / 640 / 640 |
| 202 | weighted-base-right | +0.507158 | +0.206965 | yes / yes | 640 / 640 / 640 |
| 202 | weighted-elbow-left | +5.822552 | +8.117303 | yes / yes | 640 / 640 / 640 |
| 202 | weighted-elbow-right | -0.511924 | -20.384581 | yes / no | 640 / 446 / 640 |
| 303 | long-upper-base-left | -0.014916 | -0.143927 | yes / yes | 640 / 640 / 640 |
| 303 | long-upper-base-right | -0.107932 | -0.034517 | yes / yes | 640 / 640 / 640 |
| 303 | long-upper-elbow-left | +0.121264 | +0.071477 | yes / yes | 640 / 640 / 640 |
| 303 | long-upper-elbow-right | +13.870293 | +6.422280 | yes / yes | 640 / 640 / 640 |
| 303 | long-lower-base-left | +0.691287 | -4.609755 | yes / yes | 640 / 640 / 640 |
| 303 | long-lower-base-right | +0.069276 | +0.056617 | yes / yes | 640 / 640 / 640 |
| 303 | long-lower-elbow-left | +0.937440 | +2.081730 | yes / yes | 640 / 640 / 640 |
| 303 | long-lower-elbow-right | +0.531457 | +0.892767 | yes / yes | 640 / 640 / 640 |
| 303 | weighted-base-left | +0.410981 | +1.281003 | yes / yes | 640 / 640 / 640 |
| 303 | weighted-base-right | -0.026202 | +0.763181 | yes / yes | 640 / 640 / 640 |
| 303 | weighted-elbow-left | +0.220504 | +0.160757 | yes / yes | 640 / 640 / 640 |
| 303 | weighted-elbow-right | +0.385237 | -13.037690 | yes / no | 640 / 593 / 640 |

## Sampled retention

The fixed plan contains six original frozen baselines and 144 frozen probes of 72 checkpoints after 384 decisions on two old conditions. The audit retained 150 guard attempts, with 0 blocked. Mean retention requires every seed mean at least −3.2 and all corresponding probes safe/complete. The separate individual probe gate prevents losses from being hidden by the mean.

| Method | Seed 101 mean | Seed 202 | Seed 303 | Worst individual gain | Losses below −3.2 | Mean gate | Individual gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| conservative | -0.021162 | -0.007801 | -1.936335 | -34.334899 | 2 | PASS | FAIL |
| standard | -0.730693 | -0.767690 | -4.841529 | -41.550677 | 11 | FAIL | FAIL |

Two sampled guards do not establish whole-domain retention, persistent personal memory, arbitrary topology or hardware behavior.

## Individual losses beyond the declared floor

| Method | Seed | Adaptation condition | Evaluation | Gain |
| --- | ---: | --- | --- | ---: |
| conservative | 101 | exposed-weighted-elbow-left | Same target | -5.008546 |
| standard | 202 | exposed-weighted-elbow-right | Same target | -20.384581 |
| standard | 303 | exposed-long-lower-base-left | Same target | -4.609755 |
| standard | 303 | exposed-weighted-elbow-right | Same target | -13.037690 |
| conservative | 303 | exposed-long-lower-elbow-right | development-weighted-elbow-right | -10.179013 |
| conservative | 303 | exposed-weighted-elbow-right | development-weighted-elbow-right | -34.334899 |
| standard | 101 | exposed-weighted-elbow-left | development-long-upper-base-left | -4.936922 |
| standard | 202 | exposed-long-lower-elbow-left | development-long-upper-base-left | -4.698074 |
| standard | 202 | exposed-long-lower-elbow-left | development-weighted-elbow-right | -3.279176 |
| standard | 202 | exposed-weighted-elbow-right | development-weighted-elbow-right | -3.225538 |
| standard | 303 | exposed-long-upper-elbow-left | development-weighted-elbow-right | -30.047290 |
| standard | 303 | exposed-long-lower-base-left | development-long-upper-base-left | -12.058593 |
| standard | 303 | exposed-long-lower-base-left | development-weighted-elbow-right | -3.889096 |
| standard | 303 | exposed-long-lower-elbow-left | development-long-upper-base-left | -4.178035 |
| standard | 303 | exposed-long-lower-elbow-right | development-weighted-elbow-right | -14.089802 |
| standard | 303 | exposed-weighted-base-right | development-long-upper-base-left | -4.800386 |
| standard | 303 | exposed-weighted-elbow-right | development-weighted-elbow-right | -41.550677 |

## Stopped and blocked attempts

Every stopped prefix and blocked receipt remains in the gates and accounting; no trial was retried.

| Partition | Method | Seed | Condition | Recorded decisions / controls | Terminal native flags | Native veto | Status or error |
| --- | --- | ---: | --- | ---: | --- | --- | --- |
| target | standard | 202 | exposed-weighted-elbow-right | 446 / 2230 | motor-0001:voltage_limited | false | SafetyStop: Native safety stop in target slot(s) [0]; trial ends |
| target | standard | 303 | exposed-weighted-elbow-right | 593 / 2963 | motor-0001:voltage_limited | false | SafetyStop: Native safety stop in target slot(s) [0]; trial ends |

A recorded native flag can terminate the trial even when the native veto is false. The receipt records simulator behavior; it does not establish hardware damage or a qualified physical safety interlock.

## Historical repeatability

The audit compared 36 complete frozen 640-decision streams and 36 standard 128-decision prefixes against the exact October 7 records, including actions, native telemetry, local histories, Torch RNG and actual policy/Adam states. These are controlled repeats on exposed cases, not fresh independent samples. The earlier failed gates remain unchanged.

## Integrity exposure and preservation

The [independent audit](../artifacts/reports/conservative-specialization-v1-20261009/verification/audit-v2.json) passed 24,391,284 checks with 0 failures, regenerated 187,919 actions, replayed 294 native streams and verified 942 new fresh/prefix checkpoints. It checks tensor bytes, Adam membership/counters, declared settings, continuous histories, request/reward accounting and frozen continuation. PPO gradients are not independently rerun; timestamps do not establish external execution chronology.

The first audit was interrupted when the existing container stopped with exit code 255; inspection did not establish the cause. Its empty `audit-v1.json` and the [interruption receipt](../artifacts/reports/conservative-specialization-v1-20261009/verification/audit-interruption-v1.json) remain retained. The same matching container was restarted and existing recordings were verified into `audit-v2.json`; no scientific trial was repeated. This infrastructure interruption is separate from the two recorded native safety stops.

Safe completed opportunities: 292/294. Actual collection: 187,919 decisions and 939,593 successful native controls. The maximum 41,472 learning decisions are included in target exposure, with 9,408,000 physics steps at full completion. Recorded target opportunity wall time totals 322.096s; guard time totals 305.606s including initialization, rollout, checkpoint capture and close, excluding stage imports/seals and validation replay. These CPU measurements do not establish child waiting time. Replay is validation, not an additional sample.

The [preservation receipt](../artifacts/reports/conservative-specialization-preservation-v1-20261009/after-v2.json) verified 5,360 earlier files (2,023,061,685 bytes) unchanged, including the October 7 bundle, original releases, native evidence and live pretrained asset.

Bulk recordings and ZIPs remain local and ignored. Reproducing this collection requires 294 request tapes, 294 telemetry tapes, 294 fresh initial ZIPs and 648 prefix ZIPs, exact original releases, the bound native runtime and archived auditor/helper sources. Stopped/blocked attempts retain their actual prefixes and receipts. Compact source, summaries and provenance records accompany them.

Protocol SHA-256: `cfe2dd83b378bc7fa50367aaffe058143c866d5d218804a4e32ae05b4924d38c`. Native binary SHA-256: `377b754985b2977c947e37e41cbbff286595fe70b08888c9fc32ece1db2d9777`. The pinned Linux CPU runtime uses Torch 2.8.0+cpu and SB3 2.9.0. The visible workshop uses its documented portable sampler; its trajectories are not claimed identical to research Torch sampling.

## Next milestone

Keep the immutable pretrained actor as the first-creature default. The primary method did not meet tested readiness. Use retained checkpoint/action changes and the full case outcomes to diagnose when updates help or damage behavior before declaring another adaptation recipe. A passing secondary would remain a separate finding, not a replacement primary chosen after results. Do not change a failed threshold, choose the better arm after outcomes or silently extend the experience budget. Useful and reliable incremental specialization remains the gate before online product integration. Fresh supported-body evaluation, personal memory across rebuilds, richer graphs/motors/contact and physical hardware are separate later milestones.
