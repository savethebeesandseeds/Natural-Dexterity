"""Prospective equal-budget breadth study; JSON declarations start no worlds.

The pilot's locked curriculum implementation is reused without editing it.
Reserved assemblies can only be run by the separately gated study driver.
"""
from __future__ import annotations

import copy
import json

from generalization_curriculum import manifest as pilot_manifest
from generalization_curriculum import mechanics_identity, validate_case


def manifest():
    pilot = pilot_manifest()
    return {
        "schema": "generalization_study_v1",
        "status": "prospective_before_study_learning",
        "question": "Does broader equal-budget simulation pretraining improve frozen first-use behavior on new whole assemblies?",
        "scope": "Geometry extrapolation inside one anchored planar two-link graph; one motor, one IMU and one light; no gradient adaptation, topology or hardware claim.",
        "source_cases": {"pooled": copy.deepcopy(pilot["source_cases"]),
                         "narrow": copy.deepcopy(pilot["source_cases"][:4])},
        "development_cases": copy.deepcopy(pilot["development_cases"]),
        "narrow_development_cases": copy.deepcopy(pilot["source_cases"][:4]),
        "reserved_cases": copy.deepcopy(pilot["reserved_cases"]),
        "case_ids": {"source": {"pooled": copy.deepcopy(pilot["case_ids"]["source"]),
                                "narrow": [name.replace("source-", "narrow-", 1)
                                           for name in pilot["case_ids"]["source"][:4]]},
                     "development": copy.deepcopy(pilot["case_ids"]["development"]),
                     "narrow_development": [name.replace("source-", "narrow-own-", 1)
                                            for name in pilot["case_ids"]["source"][:4]],
                     "reserved": copy.deepcopy(pilot["case_ids"]["reserved"])},
        "source_seeds": [101, 202, 303],
        "source_arms": ["pooled", "narrow"],
        "source_budget": {"decisions": 98304, "n_envs": 8, "n_steps": 256, "horizon": 640},
        "evaluation_budget": {"decisions": 640, "windows": [64, 128, 320, 640]},
        "decision_dt_s": .1,
        "native_dt_s": .02,
        "ppo": copy.deepcopy(pilot["ppo"]),
        "policy_input": copy.deepcopy(pilot["policy_input"]),
        "sampling": {"schema": "shuffled_source_cycles_v1", "seed_offset": 500000,
                     "rule": "Per-arm independent random.Random(seed+500000); shuffled whole cycles; starts balanced within each curriculum, actual transitions recorded separately."},
        "action_rng": {"base": 100000, "seed_multiplier": 10,
                       "offsets": {"development": 0, "narrow_development": 100, "reserved": 200},
                       "rule": "Reset policy RNG after model construction and checkpoint copy; same seed/case across arms; stochastic categorical inference."},
        "initial_control": "Pooled and narrow initial parameter hashes must match for every seed before evaluation; use one saved pooled initial policy per seed for shared untrained controls.",
        "gates": {"source_margin": 3.2,
                  "source_requires_all_pooled_known_and_narrow_own": True,
                  "breadth_median_seed_mean_positive": True,
                  "breadth_min_positive_seeds": 2,
                  "breadth_requires_all_pooled_narrow_and_quiet_safe": True,
                  "useful_requires_all_pooled_gain_above": 3.2},
        "wall_caps": {"source": 600, "trial": 180},
        "source_failure": "Collect all six declared source attempts without retry. Any incomplete source or mismatched initial parameters blocks development and challenge; write exclusive blocked receipts with zero reserved calls.",
        "source_gate": "All 36 pooled known-case and 12 narrow own-case trained/quiet pairs must safely complete and gain strictly more than3.2; narrow need not pass pooled's twelve geometries.",
        "challenge_arms": ["pooled", "narrow", "untrained"],
        "witness_modes": ["quiet", "negative", "positive", "rhythm_2s", "rhythm_4s"],
        "witness_rule": "Only after all six releases are sealed and global source gate passes; same governor and640 decisions; negative-first rhythms; native invalid initial sensors command zero. Quiet is shared with challenge. Witness outcomes never tune/select/prune any challenge or controller.",
        "challenge_gate": "All72 pooled/narrow trials and12 shared quiet trials complete safely; median of3seed-mean pooled-minus-narrow returns is positive and at least2seedmeans positive. Useful first-use separately requires all36pooled-minus-quiet gains>3.2. Untrained controls and their failures are retained with separate learning-benefit eligibility. Witness attainability is a separate all12-case diagnostic, not an execution or selection gate.",
        "reporting": "Retain all trials and raw stopped prefixes. Report all-case realized contrasts, complete-safe eligibility separately, seed means and lamp/motor strata. No survivor-only success claim or checkpoint/seed/hyperparameter selection.",
        "maximum_collection": {"source_decisions": 589824, "development_decisions": 94720,
                               "challenge_decisions_including_witnesses": 107520,
                               "total_decisions": 792064, "native_controls": 3960320,
                               "physics_steps": 39603200},
        "preservation": "Pilot source/protocol/artifacts and original rover-v3 challenge stay unchanged. No new dependencies, live-controller replacement or container lifecycle change.",
    }


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_manifest(value):
    expected = manifest()
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("Unexpected study declaration fields")
    for arm, cases in value["source_cases"].items():
        if arm not in expected["source_cases"] or type(cases) is not list:
            raise ValueError("Unexpected source curriculum")
        for case in cases:
            validate_case(case)
    for partition in ("development_cases", "narrow_development_cases", "reserved_cases"):
        if type(value[partition]) is not list:
            raise ValueError("Expected an ordered case list")
        for case in value[partition]:
            validate_case(case)
    sources = {mechanics_identity(case["assembly"]) for cases in value["source_cases"].values() for case in cases}
    reserved = {mechanics_identity(case["assembly"]) for case in value["reserved_cases"]}
    if sources & reserved:
        raise ValueError("Source and reserved whole assemblies overlap")
    if canonical(value) != canonical(expected):
        raise ValueError("Study curriculum, settings or budgets differ from the prospective declaration")
