# Bounded specialization results — 7 October 2026

The small online adaptation budget did **not** clear the declared practical improvement gate over the unchanged pretrained brain. Output-layer tuning is the declared primary method; full tuning is the declared secondary method. Both methods beat matched scratch on every whole target opportunity, supporting the inherited competence supplied by pretraining. That advantage does not establish a benefit from online updates over frozen reuse. No adapted brain is promoted into the workshop.

The [prospective protocol](SPECIALIZATION_EXPERIMENT.md) used all three pooled source seeds and twelve already exposed length-5 conditions: three layouts, six mechanics and two lamp positions. This is specialization on known conditions, not a fresh unseen-body test. The original rover-v3 challenge remains unexposed.

## Whole-opportunity benefit

Every target arm received 640 decisions (64 simulated seconds). Learning consumed the first 128 decisions: updates after 64 and 128, followed by 512 decisions with weights fixed in the same world/history/RNG stream. The primary threshold applies to the whole opportunity, including the learning cost. It requires all 36 corresponding pairs safe and complete, median of three seed mean gains strictly above +3.2, and at least two positive seed means.

| Method / comparator | Seed 101 mean gain | Seed 202 | Seed 303 | Median seed mean | Positive seeds | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| head_only / frozen | -1.068380 | -0.337379 | +1.648500 | -0.337379 | 1/3 | FAIL |
| head_only / scratch | +18.551809 | +20.554270 | +24.085135 | +20.554270 | 3/3 | PASS |
| full / frozen | +0.815676 | +1.410498 | +1.894412 | +1.410498 | 3/3 | FAIL |
| full / scratch | +20.435864 | +22.302147 | +24.331046 | +22.302147 | 3/3 | PASS |

All corresponding pairs are safe and complete. Scratch contrasts use their separately declared threshold of median gain >0. The three source seeds are the pretrained-brain replicates; the 36 body/light pairs are correlated case outcomes.

| Method / frozen | Whole wins / ties / losses | Worst whole gain | Whole losses below −3.2 | Prefix losses below −0.64 |
| --- | ---: | ---: | ---: | ---: |
| head_only | 14 / 4 / 18 | -13.105104 | 2 | 0 |
| full | 26 / 0 / 10 | -17.255287 | 1 | 1 |

The head-only worst regression is seed 101 on long-upper/elbow/left. Full tuning's worst regression is seed 101 on weighted/elbow/left. The positive average for full tuning therefore does not imply improvement on every creature.

## Learning prefix and fixed-weight follow-up

| Method / frozen | Seed | Whole 640 gain | Learning prefix 128 gain | Fixed follow-up 512 gain |
| --- | ---: | ---: | ---: | ---: |
| head_only | 101 | -1.068380 | +0.007446 | -1.075826 |
| head_only | 202 | -0.337379 | -0.000655 | -0.336724 |
| head_only | 303 | +1.648500 | -0.006301 | +1.654802 |
| full | 101 | +0.815676 | +0.113498 | +0.702178 |
| full | 202 | +1.410498 | +0.417750 | +0.992748 |
| full | 303 | +1.894412 | +0.025437 | +1.868975 |

Follow-up states are produced by each arm's own earlier behavior. These differences do not isolate a same-state weight effect. Return is the unmodified native light-plus-IMU integral; it is separate from PPO bootstrap values, light lux and signed electrical energy.

## Every paired target outcome

| Seed | Exposed condition | Head-only minus frozen | Full minus frozen |
| ---: | --- | ---: | ---: |
| 101 | long-upper-base-left | +0.200755 | +0.139660 |
| 101 | long-upper-base-right | -0.229246 | -0.260527 |
| 101 | long-upper-elbow-left | -13.105104 | +1.501602 |
| 101 | long-upper-elbow-right | +0.006999 | +0.434424 |
| 101 | long-lower-base-left | -0.003392 | -0.080934 |
| 101 | long-lower-base-right | +0.264820 | +0.208717 |
| 101 | long-lower-elbow-left | -0.257185 | +0.317947 |
| 101 | long-lower-elbow-right | -0.166764 | +24.457220 |
| 101 | weighted-base-left | +0.000000 | -0.207711 |
| 101 | weighted-base-right | +0.002396 | +0.302177 |
| 101 | weighted-elbow-left | +0.099108 | -17.255287 |
| 101 | weighted-elbow-right | +0.367059 | +0.230824 |
| 202 | long-upper-base-left | +0.003220 | +0.038035 |
| 202 | long-upper-base-right | +0.110651 | -0.119736 |
| 202 | long-upper-elbow-left | -1.667443 | +0.182336 |
| 202 | long-upper-elbow-right | -0.010071 | +0.208514 |
| 202 | long-lower-base-left | -0.000730 | +0.375076 |
| 202 | long-lower-base-right | -0.210377 | +0.187317 |
| 202 | long-lower-elbow-left | +0.652767 | -0.033090 |
| 202 | long-lower-elbow-right | +0.224667 | +9.057061 |
| 202 | weighted-base-left | +0.000000 | -0.058928 |
| 202 | weighted-base-right | -0.019900 | +0.617074 |
| 202 | weighted-elbow-left | -3.220623 | +5.665025 |
| 202 | weighted-elbow-right | +0.089291 | +0.807290 |
| 303 | long-upper-base-left | -0.014280 | +0.168816 |
| 303 | long-upper-base-right | -0.105661 | -0.181960 |
| 303 | long-upper-elbow-left | +0.104014 | +0.145309 |
| 303 | long-upper-elbow-right | +20.336224 | +20.440866 |
| 303 | long-lower-base-left | +0.000000 | +0.505774 |
| 303 | long-lower-base-right | -0.059969 | -0.149244 |
| 303 | long-lower-elbow-left | -0.033142 | +0.578584 |
| 303 | long-lower-elbow-right | -0.055437 | +0.026699 |
| 303 | weighted-base-left | +0.019651 | +0.743336 |
| 303 | weighted-base-right | +0.000000 | +0.321453 |
| 303 | weighted-elbow-left | -0.336543 | -0.098980 |
| 303 | weighted-elbow-right | -0.072852 | +0.232291 |

## Sampled retention

The two guards were fixed before collection: `development-long-upper-base-left` and `development-weighted-elbow-right`. Six unchanged baselines precede 144 frozen probes of 72 head/full checkpoints after 128 decisions. These probes consume a separate experience budget. The gate requires all corresponding probes safe/complete and every seed mean post-minus-base ≥−3.2.

| Method | Seed101 mean | Seed202 mean | Seed303 mean | Worst individual gain | Individual gains below −3.2 | Sampled gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| head_only | -0.078026 | -0.065879 | +0.021602 | -1.641959 | 0 | PASS |
| full | -0.085258 | -0.075153 | -0.498318 | -10.037916 | 1 | PASS |

Full tuning's worst probe loss is seed 303, adapted on long-upper/elbow/left and probed on weighted/elbow/right. Passing the mean gate does not bound individual losses. This establishes only the sampled two-condition result. It does not establish whole-domain retention, persistent personal memory across rebuilds, arbitrary topology, hardware safety or child waiting time.

## Integrity, exposure and preservation

The [independent audit](../artifacts/reports/specialization-v1-20261007/verification/audit-v1.json) passed 21,227,993 checks with 0 failures. It regenerated 188,160 categorical actions from actual checkpoint tensors, local float32 histories and Torch RNG, replayed all 294 exact native request/response streams, and inspected 510 fresh/prefix checkpoints. Head-only feature trunks stayed unchanged; fresh optimizer scopes, Adam steps, first 64 pretrained action pairing, continuous histories and fixed continuation commitments agree. PPO gradients are not independently rerun, and recorded timestamps alone do not prove external execution chronology.

All 294/294 opportunities safely completed. There were no blocked guards, fault stops or unresolved requests. Actual collection was 92,160 target plus 96,000 guard decisions: 188,160 decisions, 940,800 native controls and 9,408,000 physics steps. The 13,824 learning decisions are included in the target total. Aggregate simulated time is 18,816 seconds; Recorded per-opportunity elapsed wall time totals 272.089s for targets and 189.693s for guards (model initialization, rollout, checkpoint capture and close included; stage imports/seal checks and replay excluded). These Linux CPU timings do not establish real child responsiveness. Replay is validation work, not another independent sample.

The [preservation check](../artifacts/reports/specialization-preservation-v1-20261007/after-v1.json) verified all 3,280 preexisting artifact files (1,830,081,220 bytes) unchanged, including original checkpoints, source archives, earlier reports and the live pretrained asset. The protocol binds their prospective baseline.

New request/telemetry recordings and every new ZIP remain local and ignored. Compact results, protocol, source snapshots and receipts are retained. A fresh checkout needs all 294 request tapes, all 294 telemetry tapes, 294 fresh initial ZIPs and 216 ZIPs captured after 64/128 decisions, the exact native runtime, three original final/initial releases and archived auditor/helper bytes for full verification. Summary prose cannot substitute for these assets.

Protocol SHA-256: `5b5dbda3123c20a030021c141e4812ec0db1c8f66fd9030eca78b8265ec002e4`. Native binary SHA-256: `377b754985b2977c947e37e41cbbff286595fe70b08888c9fc32ece1db2d9777`. The pinned Linux CPU environment uses Torch 2.8.0+cpu and SB3 2.9.0. Research actions use Torch categorical RNG; the visible pretrained workshop uses its documented portable sampler, so identical trajectories are not claimed.

## Next milestone

Keep the unchanged pretrained actor as the default first creature. Investigate a longer explicitly charged specialization budget and conservative updates on these now-known conditions, with fresh prospective comparisons and regression/retention gates. Do not repair this failed practical-benefit result by changing its threshold or choosing the better seed/arm afterward. Before adding an online product mode, demonstrate useful incremental gain and bound worst-case regressions. After that, evaluate a supported first-build family on a fresh reserve, then persistent per-creature memory, richer graphs/motors/contact, and physical hardware separately.
