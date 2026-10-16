#!/usr/bin/env python3
"""Render retained, independently audited specialization results as repo prose."""
import json
from pathlib import Path
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "training"))
from generalization_pretraining import require_container
from specialization_manifest import read_bound, read_lock, ref, write_new


def number(value):
    return "unavailable" if value is None else f"{value:+.6f}"


def main():
    require_container()
    root = PROJECT / "artifacts/reports/specialization-v1-20261007"
    protocol, sha = read_lock(root)
    audit_path = root / "verification/audit-v1.json"
    audit = json.loads(audit_path.read_text())
    if not (audit["integrity_passed"] and audit["native_replay_verified"] and audit["operationally_complete"]):
        raise ValueError("Final report requires complete independent replay; retain any failed receipt")
    if audit["protocol_sha256"] != sha:
        raise ValueError("Audit belongs to another protocol")
    preservation_path = PROJECT / "artifacts/reports/specialization-preservation-v1-20261007/after-v1.json"
    preservation = json.loads(preservation_path.read_text())
    if not preservation["passed"] or preservation["baseline_sha256"] != protocol["inputs"]["receipts"]["preservation_baseline"]["sha256"]:
        raise ValueError("Preexisting artifact preservation has not passed")
    analysis = audit["analysis"]
    raw_trials = [json.loads(read_bound(row["reference"]).read_text())
                  for row in audit["target_trials"] + audit["guard_trials"]]
    target_wall_s = sum(row["wall_s"] for row in raw_trials if row["partition"] == "target")
    guard_wall_s = sum(row["wall_s"] for row in raw_trials if row["partition"] == "guard")
    contrasts = analysis["target_contrasts"]
    head, full = next(c for c in contrasts if c["arm"] == "head_only" and c["comparator"] == "frozen"), next(c for c in contrasts if c["arm"] == "full" and c["comparator"] == "frozen")
    lines = ["# Bounded specialization results — 7 October 2026", "",
        "The small online adaptation budget did **not** clear the declared practical improvement gate over the unchanged pretrained brain. "
        "Output-layer tuning is the declared primary method; full tuning is the declared secondary method. "
        "Both methods beat matched scratch on every whole target opportunity, supporting the inherited competence supplied by pretraining. "
        "That advantage does not establish a benefit from online updates over frozen reuse. No adapted brain is promoted into the workshop.", "",
        "The [prospective protocol](SPECIALIZATION_EXPERIMENT.md) used all three pooled source seeds and twelve already exposed length-5 conditions: "
        "three layouts, six mechanics and two lamp positions. This is specialization on known conditions, not a fresh unseen-body test. "
        "The original rover-v3 challenge remains unexposed.", "",
        "## Whole-opportunity benefit", "",
        "Every target arm received 640 decisions (64 simulated seconds). Learning consumed the first 128 decisions: "
        "updates after 64 and 128, followed by 512 decisions with weights fixed in the same world/history/RNG stream. "
        "The primary threshold applies to the whole opportunity, including the learning cost. It requires all 36 corresponding pairs "
        "safe and complete, median of three seed mean gains strictly above +3.2, and at least two positive seed means.", "",
        "| Method / comparator | Seed 101 mean gain | Seed 202 | Seed 303 | Median seed mean | Positive seeds | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for item in contrasts:
        means = [v["all_realized_mean"] for v in item["seed_means"]]
        lines.append(f"| {item['arm']} / {item['comparator']} | " + " | ".join(number(v) for v in means)
                     + f" | {number(item['median_seed_mean'])} | {item['positive_seed_means']}/3 | {'PASS' if item['passed'] else 'FAIL'} |")
    lines += ["", "All corresponding pairs are safe and complete. Scratch contrasts use their separately declared threshold of median gain >0. "
        "The three source seeds are the pretrained-brain replicates; the 36 body/light pairs are correlated case outcomes.", "",
        "| Method / frozen | Whole wins / ties / losses | Worst whole gain | Whole losses below −3.2 | Prefix losses below −0.64 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for item in (head, full):
        gains = [p["whole_gain"] for p in item["pairs"]]
        diagnostic = item["regression_diagnostic"]
        lines.append(f"| {item['arm']} | {sum(v > 0 for v in gains)} / {sum(v == 0 for v in gains)} / {sum(v < 0 for v in gains)} "
                     f"| {number(min(gains))} | {diagnostic['whole_below_floor_count']} | {diagnostic['prefix_below_floor_count']} |")
    lines += ["", "The head-only worst regression is seed 101 on long-upper/elbow/left. Full tuning's worst regression is "
        "seed 101 on weighted/elbow/left. The positive average for full tuning therefore does not imply improvement on every creature.", "",
        "## Learning prefix and fixed-weight follow-up", "",
        "| Method / frozen | Seed | Whole 640 gain | Learning prefix 128 gain | Fixed follow-up 512 gain |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for item in (head, full):
        for seed in (101, 202, 303):
            pairs = [p for p in item["pairs"] if p["seed"] == seed]
            lines.append(f"| {item['arm']} | {seed} | " + " | ".join(number(statistics.mean(p[key] for p in pairs)) for key in ("whole_gain", "prefix_gain", "continuation_gain")) + " |")
    lines += ["", "Follow-up states are produced by each arm's own earlier behavior. These differences do not isolate a same-state weight effect. "
        "Return is the unmodified native light-plus-IMU integral; it is separate from PPO bootstrap values, light lux and signed electrical energy.", "",
        "## Every paired target outcome", "",
        "| Seed | Exposed condition | Head-only minus frozen | Full minus frozen |",
        "| ---: | --- | ---: | ---: |"]
    secondary = {(p["seed"], p["case_index"]): p for p in full["pairs"]}
    for pair in head["pairs"]:
        other = secondary[(pair["seed"], pair["case_index"])]
        lines.append(f"| {pair['seed']} | {pair['case_id'].removeprefix('exposed-')} | {number(pair['whole_gain'])} | {number(other['whole_gain'])} |")
    lines += ["", "## Sampled retention", "",
        "The two guards were fixed before collection: `development-long-upper-base-left` and `development-weighted-elbow-right`. "
        "Six unchanged baselines precede 144 frozen probes of 72 head/full checkpoints after 128 decisions. These probes consume a separate experience budget. "
        "The gate requires all corresponding probes safe/complete and every seed mean post-minus-base ≥−3.2.", "",
        "| Method | Seed101 mean | Seed202 mean | Seed303 mean | Worst individual gain | Individual gains below −3.2 | Sampled gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for item in analysis["sampled_retention"]:
        gains = [p["post_minus_base"] for p in item["pairs"]]
        means = [s["all_realized_mean"] for s in item["seed_summaries"]]
        lines.append(f"| {item['arm']} | " + " | ".join(number(v) for v in means)
                     + f" | {number(min(gains))} | {sum(v < -3.2 for v in gains)} | {'PASS' if item['passed'] else 'FAIL'} |")
    lines += ["", "Full tuning's worst probe loss is seed 303, adapted on long-upper/elbow/left and probed on weighted/elbow/right. "
        "Passing the mean gate does not bound individual losses. This establishes only the sampled two-condition result. "
        "It does not establish whole-domain retention, persistent personal memory "
        "across rebuilds, arbitrary topology, hardware safety or child waiting time.", "",
        "## Integrity, exposure and preservation", "",
        f"The [independent audit](../artifacts/reports/specialization-v1-20261007/verification/audit-v1.json) passed {audit['checks']:,} checks with "
        f"{audit['failed_checks']} failures. It regenerated {audit['categorical_actions_verified']:,} categorical actions from actual checkpoint tensors, "
        f"local float32 histories and Torch RNG, replayed all {audit['native_request_streams_replayed']} exact native request/response streams, "
        f"and inspected {audit['fresh_and_prefix_checkpoints_verified']} fresh/prefix checkpoints. Head-only feature trunks stayed unchanged; "
        "fresh optimizer scopes, Adam steps, first 64 pretrained action pairing, continuous histories and fixed continuation commitments agree. "
        "PPO gradients are not independently rerun, and recorded timestamps alone do not prove external execution chronology.", "",
        f"All {audit['completed_safe_opportunities']}/{audit['collected_opportunities']} opportunities safely completed. There were no blocked guards, "
        "fault stops or unresolved requests. Actual collection was 92,160 target plus 96,000 guard decisions: "
        f"{audit['actual_decisions']:,} decisions, {audit['recorded_successful_native_controls']:,} native controls and 9,408,000 physics steps. "
        "The 13,824 learning decisions are included in the target total. Aggregate simulated time is 18,816 seconds; "
        f"Recorded per-opportunity elapsed wall time totals {target_wall_s:.3f}s for targets and {guard_wall_s:.3f}s for guards "
        "(model initialization, rollout, checkpoint capture and close included; stage imports/seal checks and replay excluded). "
        "These Linux CPU timings do not establish real child responsiveness. Replay is validation work, not another independent sample.", "",
        f"The [preservation check](../artifacts/reports/specialization-preservation-v1-20261007/after-v1.json) verified all "
        f"{preservation['protected_files']:,} preexisting artifact files ({preservation['protected_bytes']:,} bytes) unchanged, "
        "including original checkpoints, source archives, earlier reports and the live pretrained asset. The protocol binds their prospective baseline.", "",
        "New request/telemetry recordings and every new ZIP remain local and ignored. Compact results, protocol, source snapshots and receipts are retained. "
        "A fresh checkout needs all 294 request tapes, all 294 telemetry tapes, 294 fresh initial ZIPs and 216 ZIPs captured after 64/128 decisions, the exact native runtime, "
        "three original final/initial releases and archived auditor/helper bytes for full verification. Summary prose cannot substitute for these assets.", "",
        f"Protocol SHA-256: `{sha}`. Native binary SHA-256: `{protocol['native_binary']['sha256']}`. "
        "The pinned Linux CPU environment uses Torch 2.8.0+cpu and SB3 2.9.0. Research actions use Torch categorical RNG; "
        "the visible pretrained workshop uses its documented portable sampler, so identical trajectories are not claimed.", "",
        "## Next milestone", "",
        "Keep the unchanged pretrained actor as the default first creature. Investigate a longer explicitly charged specialization budget "
        "and conservative updates on these now-known conditions, with fresh prospective comparisons and regression/retention gates. "
        "Do not repair this failed practical-benefit result by changing its threshold or choosing the better seed/arm afterward. "
        "Before adding an online product mode, demonstrate useful incremental gain and bound worst-case regressions. "
        "After that, evaluate a supported first-build family on a fresh reserve, then persistent per-creature memory, richer graphs/motors/contact, "
        "and physical hardware separately.", ""]
    # Compact provenance binds the exact audited inputs used by this prose renderer.
    receipt = {"schema": "specialization_report_render_inputs_v1", "protocol_sha256": sha,
               "audit": ref(audit_path), "preservation": ref(preservation_path),
               "renderer": ref(Path(__file__)), "scientific_primary_passed": head["passed"],
               "scientific_secondary_passed": full["passed"]}
    output = PROJECT / "docs/SPECIALIZATION_RESULTS.md"
    with output.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    receipt["report"] = ref(output)
    write_new(root / "verification/report-inputs-v1.json", receipt)
    print(json.dumps({"report": str(output), "report_sha256": receipt["report"]["sha256"]}))


if __name__ == "__main__":
    main()
