#!/usr/bin/env python3
"""Plot the sealed specialization record after independent verification.

This analysis tool creates no native worlds and performs no learning. Run it
inside the managed Linux container only after the independent audit is complete.
All cases and seeds keep their prospective order; stopped prefixes stay visible.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA256 = "5b5dbda3123c20a030021c141e4812ec0db1c8f66fd9030eca78b8265ec002e4"
SEEDS = [101, 202, 303]
ARMS = ["head_only", "full"]
LABELS = {"head_only": "Output layers tuned − pretrained unchanged",
          "full": "All weights tuned − pretrained unchanged"}
COLORS = {101: "#2767a5", 202: "#a05213", 303: "#7b4c9f"}
MARKERS = {101: "o", 202: "s", 303: "^"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def bounded(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT / path
    if not path.is_relative_to(PROJECT) or ".." in path.parts:
        raise ValueError("Figure inputs and outputs must remain inside the project")
    current = PROJECT
    for part in path.relative_to(PROJECT).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Refusing symlink traversal: {current}")
    return path


class Inputs:
    def __init__(self):
        self.hashes = {}

    def read(self, path, expected=None):
        path = bounded(path)
        if not path.is_file():
            raise ValueError(f"Missing completed figure input: {path}")
        raw = path.read_bytes()
        sha256 = hashlib.sha256(raw).hexdigest()
        if expected is not None and sha256 != expected:
            raise ValueError(f"Figure input hash mismatch: {path}")
        relative = path.relative_to(PROJECT).as_posix()
        prior = self.hashes.get(relative)
        if prior is not None and prior != sha256:
            raise ValueError(f"Figure input changed during collection: {relative}")
        self.hashes[relative] = sha256
        return json.loads(raw)

    def reference(self, value):
        if type(value) is not dict or set(value) != {"path", "sha256"}:
            raise ValueError("Expected an exact path/SHA-256 input reference")
        if type(value["path"]) is not str or Path(value["path"]).is_absolute():
            raise ValueError("Expected a project-relative input reference")
        return self.read(value["path"], value["sha256"])

    def recheck(self):
        for relative, sha256 in self.hashes.items():
            if digest(bounded(relative)) != sha256:
                raise ValueError(f"Figure input changed before publication: {relative}")


def finite(value, name):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"Expected finite native return: {name}")
    return float(value)


def close(actual, expected, name):
    if actual is None or expected is None:
        if actual != expected:
            raise ValueError(f"Independent audit disagrees on {name}")
    elif type(actual) in (float, int) and type(expected) in (float, int):
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError(f"Independent audit disagrees on {name}: {actual} versus {expected}")
    elif actual != expected:
        raise ValueError(f"Independent audit disagrees on {name}")


def raw_return(trial, key):
    metrics = trial.get(key)
    if metrics is None or trial.get("unknown_exposure", False):
        return None
    return finite(metrics["native_reward"], trial["trial_id"] + "." + key)


def eligible(trial):
    return (trial.get("complete") is True and trial.get("safe") is True
            and trial.get("exposure_exact") is True and not trial.get("blocked", False))


def mean_complete(values):
    return statistics.mean(values) if values and all(value is not None for value in values) else None


def case_label(entry):
    assembly, sun = entry["case"]["assembly"], entry["case"]["sun"]
    lengths = assembly["segments"]
    body = "Weighted" if assembly["blocks"] else "Long upper" if lengths[0] > lengths[1] else "Long lower"
    motor = "base" if assembly["powered_hinge"] == 0 else "elbow"
    lamp = "left" if sun["position_m"][0] < 0 else "right"
    return f"{body} · {motor} · {lamp}"


def read_stage(inputs, root, name, sha256, count):
    receipt = inputs.read(root / name / "receipt.json")
    if receipt["protocol_sha256"] != sha256:
        raise ValueError(f"{name} receipt belongs to another protocol")
    stage = inputs.reference(receipt["stage"])
    if (stage["schema"] != "specialization_stage_v1" or stage["protocol_sha256"] != sha256
            or stage["stage"] != name or stage["status"] != "complete"
            or stage["planned_trials"] != count or stage["collected_trials"] != count
            or len(stage["trial_results"]) != count):
        raise ValueError(f"The {name} stage has not completed its full declared ledger")
    return stage


def derive_contrast(trials, cases, arm):
    pairs = []
    for seed in SEEDS:
        for index, entry in enumerate(cases):
            trained = trials[(seed, index, arm)]
            baseline = trials[(seed, index, "frozen")]
            row = {"seed": seed, "case_index": index, "case_id": entry["id"],
                   "eligible": eligible(trained) and eligible(baseline),
                   "trained_trial": trained["trial_id"], "baseline_trial": baseline["trial_id"],
                   "trained_decisions": trained["metrics"]["decisions"],
                   "baseline_decisions": baseline["metrics"]["decisions"],
                   "trained_complete_safe": eligible(trained), "baseline_complete_safe": eligible(baseline)}
            for key, name in (("metrics", "whole_gain"), ("prefix_metrics", "prefix_gain"),
                              ("continuation_metrics", "continuation_gain")):
                left, right = raw_return(trained, key), raw_return(baseline, key)
                row[name] = None if left is None or right is None else left - right
            pairs.append(row)
    seed_means = []
    for seed in SEEDS:
        rows = [row for row in pairs if row["seed"] == seed]
        safe = [row["whole_gain"] for row in rows if row["eligible"]]
        seed_means.append({"seed": seed, "all_realized_mean": mean_complete([row["whole_gain"] for row in rows]),
                           "complete_safe_pair_mean": statistics.mean(safe) if safe else None,
                           "complete_safe_pairs": len(safe),
                           "prefix_all_realized_mean": mean_complete([row["prefix_gain"] for row in rows]),
                           "continuation_all_realized_mean": mean_complete([row["continuation_gain"] for row in rows])})
    means = [row["all_realized_mean"] for row in seed_means]
    median = statistics.median(means) if all(value is not None for value in means) else None
    positive = sum(value is not None and value > 0 for value in means)
    all_eligible = all(row["eligible"] for row in pairs)
    return {"arm": arm, "baseline": "frozen", "pairs": pairs, "seed_means": seed_means,
            "median_seed_mean": median, "positive_seed_means": positive,
            "all_pairs_eligible": all_eligible,
            "passed": bool(all_eligible and median is not None and median > 3.2 and positive >= 2),
            "prefix_median_seed_mean": median_seed_window(seed_means, "prefix_all_realized_mean"),
            "continuation_median_seed_mean": median_seed_window(seed_means, "continuation_all_realized_mean")}


def median_seed_window(rows, key):
    values = [row[key] for row in rows]
    return statistics.median(values) if all(value is not None for value in values) else None


def audit_contrast(analysis, arm):
    """Accept the auditor's declared list, not a performance-selected subset."""
    rows = analysis["target_contrasts"]
    if type(rows) is not list:
        raise ValueError("Expected the independent auditor's ordered contrast list")
    matching = [row for row in rows if row["arm"] == arm and row["comparator"] == "frozen"]
    if len(matching) != 1:
        raise ValueError("Audit must contain one complete tuned-versus-frozen contrast per arm")
    return matching[0]


def compare_contrast(actual, expected):
    audit_pairs = {(row["seed"], row["case_index"]): row for row in expected["pairs"]}
    if set(audit_pairs) != {(seed, index) for seed in SEEDS for index in range(12)}:
        raise ValueError("Independent audit contrast matrix is incomplete")
    for row in actual["pairs"]:
        checked = audit_pairs[(row["seed"], row["case_index"])]
        for key in ("case_id", "eligible", "whole_gain", "prefix_gain", "continuation_gain"):
            close(row[key], checked[key], f"{actual['arm']}.{row['seed']}.{row['case_index']}.{key}")
    by_seed = {row["seed"]: row for row in expected["seed_means"]}
    if set(by_seed) != set(SEEDS):
        raise ValueError("Independent audit seed means are incomplete")
    for row in actual["seed_means"]:
        for key in ("all_realized_mean", "complete_safe_pair_mean", "complete_safe_pairs"):
            close(row[key], by_seed[row["seed"]][key], actual["arm"] + f".seed-{row['seed']}." + key)
    for key in ("median_seed_mean", "positive_seed_means", "all_pairs_eligible", "passed"):
        close(actual[key], expected[key], actual["arm"] + "." + key)


def collect(root, audit_path):
    inputs = Inputs()
    protocol = inputs.read(root / "protocol.json", PROTOCOL_SHA256)
    lock = inputs.read(root / "lock.json")
    if lock["protocol_sha256"] != PROTOCOL_SHA256 or protocol["schema"] != "specialization_protocol_v1":
        raise ValueError("Unexpected specialization lock")
    study = protocol["study"]
    if (study["source_seeds"] != SEEDS or study["arms"] != ["frozen", "head_only", "full", "scratch"]
            or len(study["target_cases"]) != 12 or study["budget"]["total_decisions"] != 640
            or study["budget"]["prefix_decisions"] != 128
            or study["gates"]["primary_arm"] != "head_only"
            or study["gates"]["median_seed_mean_gain_strictly_greater_than"] != 3.2):
        raise ValueError("Figure scope differs from the declared study")
    audit = inputs.read(audit_path)
    if (audit["schema"] != "specialization_independent_verification_v1"
            or audit["protocol_sha256"] != PROTOCOL_SHA256
            or audit["integrity_passed"] is not True or audit["native_replay_verified"] is not True
            or audit["issues"] or audit["failed_checks"] != 0):
        raise ValueError("Figures require a passing completed independent integrity audit")
    targets = read_stage(inputs, root, "targets", PROTOCOL_SHA256, 144)
    seal = inputs.reference(targets["seal"])
    if (seal["schema"] != "specialization_target_seal_v1" or not seal["all_results_collected"]
            or not seal["sealed_before_guard_worlds"] or seal["collected_trials"] != 144
            or seal["protocol_sha256"] != PROTOCOL_SHA256
            or seal["trial_results"] != targets["trial_results"]):
        raise ValueError("Target result ledger is not faithfully sealed")
    guards = read_stage(inputs, root, "guards", PROTOCOL_SHA256, 150)
    for path, key in ((root / "lock.json", "lock_sha256"),
                      (root / "targets/stage.json", "target_stage_sha256"),
                      (root / "guards/stage.json", "guard_stage_sha256")):
        close(inputs.hashes[path.relative_to(PROJECT).as_posix()], audit[key], key)
    close(targets["seal"]["sha256"], audit["target_seal_sha256"], "target seal SHA-256")
    trials = {}
    for reference in targets["trial_results"]:
        trial = inputs.reference(reference)
        key = (trial["seed"], trial["case_index"], trial["arm"])
        if (key in trials or trial["protocol_sha256"] != PROTOCOL_SHA256
                or trial["partition"] != "target" or not 0 <= key[1] < 12
                or trial["case_id"] != study["target_cases"][key[1]]["id"]):
            raise ValueError("Duplicate or mismatched target result")
        trials[key] = trial
    if set(trials) != {(seed, index, arm) for seed in SEEDS for index in range(12) for arm in study["arms"]}:
        raise ValueError("The full 144-attempt target matrix is required")
    audited_targets = {row["trial_id"]: row for row in audit["target_trials"]}
    if set(audited_targets) != {row["trial_id"] for row in trials.values()}:
        raise ValueError("Independent audit does not cover every target attempt")
    for trial in trials.values():
        checked = audited_targets[trial["trial_id"]]
        for key in ("seed", "arm", "case_index", "complete", "safe", "exposure_exact"):
            close(trial[key], checked[key], trial["trial_id"] + "." + key)
        for key in ("metrics", "prefix_metrics", "continuation_metrics"):
            if trial.get(key) is not None:
                close(trial[key]["native_reward"], checked[key]["native_reward"], trial["trial_id"] + "." + key)
    guard_trials = [inputs.reference(reference) for reference in guards["trial_results"]]
    if {row["trial_id"] for row in guard_trials} != {row["trial_id"] for row in audit["guard_trials"]}:
        raise ValueError("Independent audit does not cover every guard attempt")
    contrasts = [derive_contrast(trials, study["target_cases"], arm) for arm in ARMS]
    for contrast in contrasts:
        compare_contrast(contrast, audit_contrast(audit["analysis"], contrast["arm"]))
    return inputs, study, audit, targets, guards, contrasts


def limits(values):
    values = [value for value in values if value is not None]
    low, high = min([0., 3.2] + values), max([0., 3.2] + values)
    padding = max(1., (high - low) * .11)
    return low - padding, high + padding


def draw(study, contrasts, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.fonttype": "none"})
    figure, axes = plt.subplots(2, 2, figsize=(15, 10),
                                gridspec_kw={"height_ratios": [3.2, 1.5]}, layout="constrained")
    case_limits = limits([row["whole_gain"] for contrast in contrasts for row in contrast["pairs"]])
    summary_limits = limits([row["all_realized_mean"] for contrast in contrasts for row in contrast["seed_means"]])
    for column, contrast in enumerate(contrasts):
        panel, summary = axes[0, column], axes[1, column]
        panel.set_title(("Primary: " if column == 0 else "Secondary: ") + LABELS[contrast["arm"]], loc="left", fontsize=12)
        panel.axvline(0, color="#505b64", linewidth=1)
        for seed_position, seed in enumerate(SEEDS):
            for row in [value for value in contrast["pairs"] if value["seed"] == seed]:
                y = row["case_index"] + (seed_position - 1) * .21
                if row["whole_gain"] is None:
                    panel.text(.99, y, f"N/A {seed}", transform=panel.get_yaxis_transform(),
                               ha="right", va="center", fontsize=8, color=COLORS[seed],
                               bbox={"facecolor": "white", "edgecolor": "none", "pad": .5})
                    continue
                panel.scatter(row["whole_gain"], y, color=COLORS[seed], s=43,
                              marker=MARKERS[seed] if row["eligible"] else "x", zorder=3)
        panel.set_yticks(range(12), [case_label(entry) for entry in study["target_cases"]])
        panel.set_ylim(11.65, -.65)
        panel.set_xlim(case_limits)
        panel.grid(axis="x", color="#dfe4e8", linewidth=.6)
        panel.set_xlabel("Whole 640-decision native return difference")
        for boundary in (3.5, 7.5):
            panel.axhline(boundary, color="#c9d1d8", linewidth=.8)
        summary.axvline(0, color="#505b64", linewidth=1)
        summary.axvline(3.2, color="#a52e3b", linestyle="--", linewidth=1.2,
                        label="+3.2 threshold applies to seed median")
        for index, row in enumerate(contrast["seed_means"]):
            value = row["all_realized_mean"]
            if value is not None:
                summary.scatter(value, index, color=COLORS[row["seed"]], marker=MARKERS[row["seed"]], s=65, zorder=4)
                summary.annotate(f"{value:+.3f}", (value, index), xytext=(6, 7), textcoords="offset points", fontsize=9)
            else:
                summary.text(.99, index, "N/A: incomplete raw-return ledger", transform=summary.get_yaxis_transform(),
                             ha="right", va="center", fontsize=8, color=COLORS[row["seed"]])
        median = contrast["median_seed_mean"]
        if median is not None:
            summary.scatter(median, 3, color="#182c3b", marker="D", s=76, zorder=4)
            summary.annotate(f"{median:+.3f}", (median, 3), xytext=(6, 7), textcoords="offset points", fontsize=9)
        summary.set_yticks(range(4), ["Seed 101 mean", "Seed 202 mean", "Seed 303 mean", "Median of seed means"])
        summary.set_ylim(3.65, -.65)
        summary.set_xlim(summary_limits)
        summary.grid(axis="x", color="#dfe4e8", linewidth=.6)
        summary.set_xlabel("Mean of the 12 declared case differences")
        status = "All 36 pairs complete and safe" if contrast["all_pairs_eligible"] else "Stopped/ineligible pairs retained; exposure can differ"
        summary.set_title(status, loc="left", fontsize=10)
    handles = [Line2D([], [], color=COLORS[seed], marker=MARKERS[seed], linestyle="", label=f"Seed {seed}") for seed in SEEDS]
    if any(not row["eligible"] for contrast in contrasts for row in contrast["pairs"]):
        handles.append(Line2D([], [], color="#505b64", marker="x", linestyle="", label="Stopped / ineligible pair"))
    figure.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    figure.suptitle("Bounded specialization: every declared case, including the learning cost", fontsize=16, fontweight="bold")
    figure.supxlabel("128 updating decisions (12.8 s) + 512 frozen follow-up decisions (51.2 s).\n"
                       "Three source seeds are replicates; the 36 case pairs are correlated. +3.2 is a seed-median criterion, with separate safety and scratch conditions.", fontsize=10)
    png = destination / "whole-opportunity-contrasts.png"
    svg = destination / "whole-opportunity-contrasts.svg"
    figure.savefig(png, dpi=180, facecolor="white", metadata={"Software": "Droid Blocks specialization analysis"})
    figure.savefig(svg, facecolor="white", metadata={"Date": None, "Description": "All declared specialization contrasts; source hashes retained in figure-data.json"})
    plt.close(figure)
    return matplotlib.__version__, [png, svg]


def main():
    if sys.platform != "linux" or not Path("/.dockerenv").is_file():
        raise RuntimeError("Run this plotting tool inside the existing managed Linux container")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="artifacts/reports/specialization-v1-20261007")
    parser.add_argument("--audit", required=True, help="Completed independent audit JSON, inside this batch")
    args = parser.parse_args()
    root, audit_path = bounded(args.report), bounded(args.audit)
    if not audit_path.is_relative_to(root):
        raise ValueError("Use the completed audit belonging to this report")
    destination = root / "figures"
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Refusing to replace an existing figure attempt")
    inputs, study, audit, targets, guards, contrasts = collect(root, audit_path)
    inputs.recheck()
    destination.mkdir()
    version, images = draw(study, contrasts, destination)
    inputs.recheck()
    payload = {"schema": "specialization_figure_data_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "protocol_sha256": PROTOCOL_SHA256, "plotting_tool_sha256": digest(Path(__file__).resolve()),
               "audit_path": audit_path.relative_to(PROJECT).as_posix(), "input_sha256": inputs.hashes,
               "matplotlib_version": version, "python_version": sys.version,
               "native_worlds_constructed": 0, "training_calls": 0,
               "target_attempts": targets["collected_trials"], "guard_attempts": guards["collected_trials"],
               "case_order": [entry["id"] for entry in study["target_cases"]], "seed_order": SEEDS,
               "contrasts": contrasts, "all_target_contrasts_from_audit": audit["analysis"]["target_contrasts"],
               "sampled_retention_from_audit": audit["analysis"]["sampled_retention"],
               "interpretation": "Whole640 raw native return includes128 learning decisions. All36 realized pairs are retained; crosses denote incomplete/unsafe unequal exposure. Seed means require all12 raw returns, never survivor filtering. +3.2 applies to the median of3seed means, not each case; eligibility and scratch contrasts remain separate. Prefix and tail means are descriptive; tail worlds differ because earlier actions differ.",
               "figures": [{"path": path.relative_to(PROJECT).as_posix(), "size": path.stat().st_size,
                            "sha256": digest(path)} for path in images]}
    output = destination / "figure-data.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(canonical({"passed": True, "figure_data": output.relative_to(PROJECT).as_posix(),
                     "sha256": digest(output), "figures": payload["figures"]}))


if __name__ == "__main__":
    main()
