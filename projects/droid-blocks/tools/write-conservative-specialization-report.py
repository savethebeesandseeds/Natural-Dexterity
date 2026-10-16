#!/usr/bin/env python3
"""Render exact independent reductions without changing declared gates."""
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import statistics
import sys

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "training"))
from generalization_pretraining import require_container
from conservative_specialization_manifest import read_bound, read_lock, ref, write_new


def number(value):
    return "unavailable" if value is None else f"{value:+.6f}"


def status(value):
    return "PASS" if value else "FAIL"


def average(values):
    values = list(values)
    return statistics.mean(values) if values and all(v is not None for v in values) else None


def main():
    require_container()
    root = PROJECT / "artifacts/reports/conservative-specialization-v1-20261009"
    protocol, sha = read_lock(root)
    audit_path = root / "verification/audit-v2.json"
    audit = json.loads(audit_path.read_text())
    if not (audit["integrity_passed"] and audit["native_replay_verified"] and audit["protocol_sha256"] == sha):
        raise ValueError("Report requires independent replay of this exact protocol")
    interruption_path = root / "verification/audit-interruption-v1.json"
    interruption = json.loads(interruption_path.read_text())
    if (not interruption["recovery"]["empty_attempt_preserved"]
            or interruption["recovery"]["scientific_trial_retries"] != 0
            or (root / "verification/audit-v1.json").stat().st_size != 0):
        raise ValueError("Interrupted first audit receipt must remain preserved")
    preservation_path = PROJECT / "artifacts/reports/conservative-specialization-preservation-v1-20261009/after-v2.json"
    preservation = json.loads(preservation_path.read_text())
    if not (preservation["passed"] and preservation["baseline_sha256"] == protocol["inputs"]["receipts"]["preservation_baseline"]["sha256"]):
        raise ValueError("Earlier artifact preservation has not passed")
    analysis = audit["analysis"]
    contrasts = analysis["target_contrasts"]
    methods = [next(c for c in contrasts if c["arm"] == arm and c["comparator"] == "frozen")
               for arm in ("conservative", "standard")]
    readiness = analysis["combined_readiness"]
    primary_guard = next(r for r in analysis["sampled_retention"] if r["arm"] == "conservative")
    raw = [json.loads(read_bound(row["reference"]).read_text())
           for row in audit["target_trials"] + audit["guard_trials"]]
    wall = {partition: sum(r.get("wall_s", 0.) for r in raw if r["partition"] == partition)
            for partition in ("target", "guard")}
    intro = (f"The conservative primary method {'met' if analysis['primary_passed'] else 'did not meet'} combined readiness; "
             f"the standard secondary method {'met' if analysis['secondary_passed'] else 'did not meet'} it. "
             "Readiness requires useful whole-opportunity benefit, acceptable individual target losses, sampled retention and acceptable individual probe losses together. "
             "No adapted checkpoint is promoted into the workshop.")
    next_step = (
        "The primary method met the tested readiness gates. Confirm the recipe prospectively on the declared scope and "
        "validate an explicit online product-mode contract, including latency, pause/rebuild behavior and update failure handling, before integration. "
        "This result does not automatically promote a checkpoint."
        if analysis["primary_passed"] else
        "The primary method did not meet tested readiness. Use retained checkpoint/action changes and the full case outcomes to diagnose "
        "when updates help or damage behavior before declaring another adaptation recipe. A passing secondary would remain a separate finding, "
        "not a replacement primary chosen after results."
    )
    lines = ["# Longer budget and conservative specialization results", "",
             "Recorded 9 October 2026.", "", intro, "",
             f"Conservative updates' median seed mean gain over frozen was {number(methods[0]['median_seed_mean'])}, "
             "against a usefulness threshold strictly above +3.2. "
             f"The worst target difference was {number(methods[0]['target_reliability']['worst_gain'])}; "
             f"the worst earlier-condition probe difference was {number(primary_guard['probe_reliability']['worst_gain'])}. "
             "Increasing experience and lowering both learning rate and clipping did not produce useful, reliable incremental specialization in this tested scope.", "",
             "The [prospective protocol](CONSERVATIVE_SPECIALIZATION_EXPERIMENT.md) tests 384 learning decisions inside each 640-decision opportunity, "
             "followed by 256 fixed-weight decisions without resetting the world, history or RNG. "
             "It uses three original pooled source releases and twelve already exposed length-5 conditions. "
             "This is known-condition specialization; the original rover-v3 challenge remains unexposed.", "",
             "## Whole opportunity benefit", "",
             "The conservative package lowers learning rate from 0.0003 to 0.0001 and PPO clipping from 0.2 to 0.1. "
             "Standard and scratch retain the original settings. All three learning arms use fresh full actor/critic optimizers, "
             "six 64-decision rollouts and four epochs each, for 24 Adam steps. "
             "The package comparison does not isolate the two hyperparameters separately.", "",
             "Useful benefit over frozen requires all 36 corresponding pairs safe/complete, median of the three seed mean gains strictly above +3.2, "
             "and at least two positive seed means. Scratch contrasts use their separate median >0 threshold. "
             "Three source seeds are training replicates; twelve cases per seed are correlated. "
             "For incomplete attempts, differences subtract recorded returns over unequal durations: the stopped attempt's prefix versus its completed comparator's full opportunity. "
             "They cannot establish 640-decision benefit or an equal-length comparison; eligibility and actual paired exposure are shown below.", "",
             "| Method / comparator | Seed 101 mean | Seed 202 | Seed 303 | Median seed mean | Positive seeds | Eligible pairs | Benefit gate |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for item in contrasts:
        lines.append(f"| {item['arm']} / {item['comparator']} | "
                     + " | ".join(number(s["all_realized_mean"]) for s in item["seed_means"])
                     + f" | {number(item['median_seed_mean'])} | {item['positive_seed_means']}/3 | {sum(p['eligible'] for p in item['pairs'])}/36 | {status(item['passed'])} |")
    lines += ["", "## Individual reliability and combined readiness", "",
              "Target reliability requires every whole target gain at least −3.2; probe reliability requires every sampled old-condition gain at least −3.2. "
              "These empirical bounds apply only to the tested cases. Physical safety stops and return-loss floors are different conditions.", "",
              "| Method | Useful benefit | Target reliability | Sampled retention | Probe reliability | Combined readiness |",
              "| --- | --- | --- | --- | --- | --- |"]
    for item in readiness:
        lines.append(f"| {item['arm']} | " + " | ".join(status(item[k]) for k in
                     ("useful_benefit", "target_reliability", "sampled_retention", "probe_reliability", "passed")) + " |")
    lines += ["", "| Method / frozen | Recorded wins / ties / losses | Worst recorded return difference | Recorded losses below −3.2 | Prefix losses below −1.92 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for item in methods:
        gains = [p["whole_gain"] for p in item["pairs"] if p["whole_gain"] is not None]
        diagnostic = item["regression_diagnostic"]
        lines.append(f"| {item['arm']} | {sum(v > 0 for v in gains)} / {sum(v == 0 for v in gains)} / {sum(v < 0 for v in gains)} "
                     f"| {number(min(gains, default=None))} | {diagnostic['whole_below_floor_count']} | {diagnostic['prefix_below_floor_count']} |")
    lines += ["", "## Learning prefix and fixed continuation", "",
              "| Method / frozen | Seed | Recorded total return difference | Prefix difference at 384 | Recorded continuation difference up to 256 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for item in methods:
        for seed in (101, 202, 303):
            pairs = [p for p in item["pairs"] if p["seed"] == seed]
            lines.append(f"| {item['arm']} | {seed} | "
                         + " | ".join(number(average(p[k] for p in pairs)) for k in
                                      ("whole_gain", "prefix_gain", "continuation_gain")) + " |")
    lines += ["", "Continuation states follow each arm's earlier actions, so these differences do not isolate weights at an identical state. "
              "Return is the native light-plus-IMU integral, separate from PPO values, lux and signed energy.", "",
              "## Every paired target outcome", "",
              "| Seed | Exposed condition | Conservative minus frozen | Standard minus frozen | Eligible conservative / standard | Decisions conservative / standard / frozen |",
              "| ---: | --- | ---: | ---: | --- | --- |"]
    secondary = {(p["seed"], p["case_index"]): p for p in methods[1]["pairs"]}
    for pair in methods[0]["pairs"]:
        other = secondary[(pair["seed"], pair["case_index"])]
        lines.append(f"| {pair['seed']} | {pair['case_id'].removeprefix('exposed-')} | {number(pair['whole_gain'])} | {number(other['whole_gain'])} "
                     f"| {'yes' if pair['eligible'] else 'no'} / {'yes' if other['eligible'] else 'no'} "
                     f"| {pair['first_decisions']} / {other['first_decisions']} / {pair['comparator_decisions']} |")
    lines += ["", "## Sampled retention", "",
              "The fixed plan contains six original frozen baselines and 144 frozen probes of 72 checkpoints after 384 decisions on two old conditions. "
              f"The audit retained {len(audit['guard_trials'])} guard attempts, with {sum(r['blocked'] for r in audit['guard_trials'])} blocked. "
              "Mean retention requires every seed mean at least −3.2 and all corresponding probes safe/complete. "
              "The separate individual probe gate prevents losses from being hidden by the mean.", "",
              "| Method | Seed 101 mean | Seed 202 | Seed 303 | Worst individual gain | Losses below −3.2 | Mean gate | Individual gate |",
              "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for item in analysis["sampled_retention"]:
        reliability = item["probe_reliability"]
        lines.append(f"| {item['arm']} | " + " | ".join(number(s["all_realized_mean"]) for s in item["seed_summaries"])
                     + f" | {number(reliability['worst_gain'])} | {reliability['below_floor_count']} | {status(item['passed'])} | {status(reliability['passed'])} |")
    lines += ["", "Two sampled guards do not establish whole-domain retention, persistent personal memory, arbitrary topology or hardware behavior.", "",
              "## Individual losses beyond the declared floor", "",
              "| Method | Seed | Adaptation condition | Evaluation | Gain |",
              "| --- | ---: | --- | --- | ---: |"]
    loss_count = 0
    for item in methods:
        for pair in item["pairs"]:
            if pair["whole_gain"] is not None and pair["whole_gain"] < -3.2:
                lines.append(f"| {item['arm']} | {pair['seed']} | {pair['case_id']} | Same target | {number(pair['whole_gain'])} |")
                loss_count += 1
    for item in analysis["sampled_retention"]:
        for pair in item["pairs"]:
            if pair["post_minus_base"] is not None and pair["post_minus_base"] < -3.2:
                target = protocol["study"]["target_cases"][pair["target_case_index"]]["id"]
                guard = protocol["study"]["guard_cases"][pair["guard_case_index"]]["id"]
                lines.append(f"| {item['arm']} | {pair['seed']} | {target} | {guard} | {number(pair['post_minus_base'])} |")
                loss_count += 1
    if not loss_count:
        lines.append("| Both methods | All | All | Target and sampled probes | No recorded loss below −3.2 |")
    incomplete = [r for r in raw if not r["complete"]]
    lines += ["", "## Stopped and blocked attempts", "",
              "Every stopped prefix and blocked receipt remains in the gates and accounting; no trial was retried.", "",
              "| Partition | Method | Seed | Condition | Recorded decisions / controls | Terminal native flags | Native veto | Status or error |",
              "| --- | --- | ---: | --- | ---: | --- | --- | --- |"]
    for row in incomplete:
        reason = str(row.get("error") or row.get("status", "incomplete")).replace("|", "/").replace("\n", " ")
        safety = {}
        if row.get("telemetry"):
            with gzip.open(read_bound(row["telemetry"]), "rt", encoding="utf-8") as stream:
                telemetry = json.load(stream)
            if telemetry.get("rows"):
                safety = telemetry["rows"][-1].get("native", {}).get("safety", {})
        flags = ", ".join(safety.get("flags", [])) or "unavailable"
        veto = "unavailable" if safety.get("veto") is None else str(safety["veto"]).lower()
        lines.append(f"| {row['partition']} | {row['arm']} | {row['seed']} | {row['case_id']} "
                     f"| {row['metrics']['decisions']} / {row['metrics']['native_steps']} | {flags} | {veto} | {reason} |")
    if not incomplete:
        lines.append("| Both | All | All | All | All 640 / 3200 | None | false | None |")
    lines += ["", "A recorded native flag can terminate the trial even when the native veto is false. "
              "The receipt records simulator behavior; it does not establish hardware damage or a qualified physical safety interlock."]
    repeat = audit["historical_repeatability"]
    lines += ["", "## Historical repeatability", "",
              f"The audit compared {repeat['full_frozen_streams_compared']} complete frozen 640-decision streams and "
              f"{repeat['standard128_prefixes_compared']} standard 128-decision prefixes against the exact October 7 records, "
              "including actions, native telemetry, local histories, Torch RNG and actual policy/Adam states. "
              "These are controlled repeats on exposed cases, not fresh independent samples. The earlier failed gates remain unchanged.", "",
              "## Integrity exposure and preservation", "",
              f"The [independent audit](../artifacts/reports/conservative-specialization-v1-20261009/verification/audit-v2.json) passed "
              f"{audit['checks']:,} checks with {audit['failed_checks']} failures, regenerated {audit['categorical_actions_verified']:,} actions, "
              f"replayed {audit['native_request_streams_replayed']} native streams and verified {audit['fresh_and_prefix_checkpoints_verified']} new fresh/prefix checkpoints. "
              "It checks tensor bytes, Adam membership/counters, declared settings, continuous histories, request/reward accounting and frozen continuation. "
              "PPO gradients are not independently rerun; timestamps do not establish external execution chronology.", "",
              "The first audit was interrupted when the existing container stopped with exit code 255; inspection did not establish the cause. "
              "Its empty `audit-v1.json` and the [interruption receipt](../artifacts/reports/conservative-specialization-v1-20261009/verification/audit-interruption-v1.json) remain retained. "
              "The same matching container was restarted and existing recordings were verified into `audit-v2.json`; no scientific trial was repeated. "
              "This infrastructure interruption is separate from the two recorded native safety stops.", "",
              f"Safe completed opportunities: {audit['completed_safe_opportunities']}/{audit['collected_opportunities']}. "
              f"Actual collection: {audit['actual_decisions']:,} decisions and {audit['recorded_successful_native_controls']:,} successful native controls. "
              "The maximum 41,472 learning decisions are included in target exposure, with 9,408,000 physics steps at full completion. "
              f"Recorded target opportunity wall time totals {wall['target']:.3f}s; guard time totals {wall['guard']:.3f}s "
              "including initialization, rollout, checkpoint capture and close, excluding stage imports/seals and validation replay. "
              "These CPU measurements do not establish child waiting time. Replay is validation, not an additional sample.", "",
              f"The [preservation receipt](../artifacts/reports/conservative-specialization-preservation-v1-20261009/after-v2.json) verified "
              f"{preservation['protected_files']:,} earlier files ({preservation['protected_bytes']:,} bytes) unchanged, "
              "including the October 7 bundle, original releases, native evidence and live pretrained asset.", "",
              "Bulk recordings and ZIPs remain local and ignored. Reproducing this collection requires 294 request tapes, 294 telemetry tapes, "
              "294 fresh initial ZIPs and 648 prefix ZIPs, exact original releases, the bound native runtime and archived auditor/helper sources. "
              "Stopped/blocked attempts retain their actual prefixes and receipts. Compact source, summaries and provenance records accompany them.", "",
              f"Protocol SHA-256: `{sha}`. Native binary SHA-256: `{protocol['native_binary']['sha256']}`. "
              "The pinned Linux CPU runtime uses Torch 2.8.0+cpu and SB3 2.9.0. The visible workshop uses its documented portable sampler; "
              "its trajectories are not claimed identical to research Torch sampling.", "",
              "## Next milestone", "",
              "Keep the immutable pretrained actor as the first-creature default. " + next_step + " "
              "Do not change a failed threshold, choose the better arm after outcomes or silently extend the experience budget. "
              "Useful and reliable incremental specialization remains the gate before online product integration. "
              "Fresh supported-body evaluation, personal memory across rebuilds, richer graphs/motors/contact and physical hardware are separate later milestones.", ""]
    output = PROJECT / "docs/CONSERVATIVE_SPECIALIZATION_RESULTS.md"
    with output.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    receipt = {"schema": "conservative_specialization_report_inputs_v1",
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": sha,
               "audit": ref(audit_path), "preservation": ref(preservation_path),
               "audit_interruption": ref(interruption_path),
               "renderer": ref(Path(__file__)), "report": ref(output),
               "primary_benefit_passed": analysis["primary_benefit_passed"],
               "secondary_benefit_passed": analysis["secondary_benefit_passed"],
               "primary_combined_passed": analysis["primary_passed"],
               "secondary_combined_passed": analysis["secondary_passed"]}
    write_new(root / "verification/report-inputs-v1.json", receipt)
    print(json.dumps({"report": str(output), "report_sha256": receipt["report"]["sha256"]}))


if __name__ == "__main__":
    main()
