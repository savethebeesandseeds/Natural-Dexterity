"""Standard-library contract tests; never start native worlds or learning.

Run only in the documented Linux development container once its Python runtime
has been established. Reserved configurations are inspected as JSON data only.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from generalization_curriculum import (DeterministicSourceSampler, manifest,
                                       mechanics_identity, validate_case, validate_manifest)


class CurriculumContract(unittest.TestCase):
    def setUp(self):
        self.protocol = manifest()

    def test_exact_matrix_and_budgets(self):
        validate_manifest(self.protocol)
        self.assertEqual([len(self.protocol[key]) for key in
                          ("source_cases", "development_cases", "reserved_cases")], [32, 12, 12])
        source = self.protocol["source_cases"]
        combinations = {(tuple(c["assembly"]["segments"]), bool(c["assembly"]["blocks"]),
                         c["assembly"]["powered_hinge"], c["sun"]["position_m"][0]) for c in source}
        expected = {(lengths, weighted, hinge, side * .3)
                    for lengths in ((3, 3), (4, 3), (3, 4), (4, 4))
                    for weighted in (False, True) for hinge in (0, 1) for side in (-1, 1)}
        self.assertEqual(combinations, expected)
        for group in ("source_cases", "development_cases", "reserved_cases"):
            for case in self.protocol[group]:
                self.assertEqual(case["sun"]["position_m"][1:], [0., -.3])
                self.assertEqual(case["sun"]["intensity_lux"], 1000.)
        self.assertEqual(self.protocol["pilot"], {
            "seed": 404, "source_decisions": 16384, "n_envs": 8, "n_steps": 256,
            "horizon_decisions": 640, "development_decisions_per_arm": 640,
            "development_arms": ["frozen", "untrained", "quiet"]})
        maximum = self.protocol["pilot"]["source_decisions"] + 12 * 3 * self.protocol["pilot"]["development_decisions_per_arm"]
        self.assertEqual(maximum, 39424)
        self.assertEqual(self.protocol["full_study"]["source_seeds"], [101, 202, 303])
        self.assertEqual(self.protocol["full_study"]["source_decisions_per_seed"], 98304)
        self.assertEqual(self.protocol["pilot"]["source_decisions"] % (8 * 256), 0)
        self.assertEqual(self.protocol["development_gate"]["required_cases"], 12)
        self.assertEqual(self.protocol["development_gate"]["native_reward_gain_strictly_greater_than"], 3.2)

    def test_whole_body_split_and_cosmetic_identity(self):
        identities = {key: {mechanics_identity(c["assembly"]) for c in self.protocol[key]}
                      for key in ("source_cases", "development_cases", "reserved_cases")}
        self.assertEqual([len(identities[key]) for key in identities], [16, 6, 6])
        self.assertFalse(identities["source_cases"] & identities["reserved_cases"])
        self.assertLessEqual(identities["development_cases"], identities["source_cases"])
        body = copy.deepcopy(self.protocol["source_cases"][4]["assembly"])
        original = mechanics_identity(body)
        body["name"] = "A different cosmetic label"
        body["blocks"][0]["id"] = "another-block-label"
        self.assertEqual(mechanics_identity(body), original)
        body["blocks"][0]["slot"] = 1
        self.assertNotEqual(mechanics_identity(body), original)
        default_hinge = copy.deepcopy(self.protocol["source_cases"][0]["assembly"])
        explicit_identity = mechanics_identity(default_hinge)
        del default_hinge["powered_hinge"]
        self.assertEqual(mechanics_identity(default_hinge), explicit_identity)

    def test_overlap_cannot_be_hidden_by_labels_or_light(self):
        altered = copy.deepcopy(self.protocol)
        altered["reserved_cases"][0]["assembly"] = copy.deepcopy(altered["source_cases"][0]["assembly"])
        altered["reserved_cases"][0]["assembly"]["name"] = "Supposedly new creature"
        altered["reserved_cases"][0]["sun"]["position_m"][0] = .9
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_manifest(altered)

    def test_manifest_changes_are_rejected(self):
        for key in ("source_decisions", "n_envs", "n_steps", "horizon_decisions", "development_decisions_per_arm"):
            changed = copy.deepcopy(self.protocol)
            changed["pilot"][key] += 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_manifest(changed)
        for mutate in (
                lambda p: p["source_cases"].reverse(),
                lambda p: p["reserved_cases"].pop(),
                lambda p: p["case_ids"]["source"].__setitem__(0, "changed-id"),
                lambda p: p["source_cases"][0]["sun"].__setitem__("intensity_lux", 999.),
                lambda p: p["ppo"].__setitem__("n_epochs", 5),
                lambda p: p["development_gate"].__setitem__("native_reward_gain_strictly_greater_than", 0.),
                lambda p: p.__setitem__("unexpected", True)):
            changed = copy.deepcopy(self.protocol)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_manifest(changed)

    def test_structural_range_collision_and_name_validation(self):
        valid = self.protocol["source_cases"][4]
        for mutate in (
                lambda c: c["assembly"].__setitem__("segments", [3, 7]),
                lambda c: c["assembly"].__setitem__("segments", [3, 3.]),
                lambda c: c["assembly"].__setitem__("segments", [3]),
                lambda c: c["assembly"].__setitem__("powered_hinge", True),
                lambda c: c["assembly"]["sensor"].__setitem__("slot", 3),
                lambda c: c["assembly"]["sensor"].__setitem__("side", 0),
                lambda c: c["assembly"]["light"].__setitem__("face", 0),
                lambda c: c["assembly"]["light"].__setitem__("side", -1),
                lambda c: c["assembly"]["blocks"][0].__setitem__("slot", 2),
                lambda c: c["assembly"]["blocks"][0].__setitem__("id", "motor-0001"),
                lambda c: c["assembly"]["blocks"].append(copy.deepcopy(c["assembly"]["blocks"][0])),
                lambda c: c["assembly"].__setitem__("name", "x" * 61),
                lambda c: c["assembly"].__setitem__("name", "\ud800"),
                lambda c: c["sun"]["position_m"].__setitem__(1, .1),
                lambda c: c["sun"]["position_m"].__setitem__(0, float("nan")),
                lambda c: c["sun"].__setitem__("intensity_lux", 1001.),
                lambda c: c["sun"].__setitem__("intensity_lux", True)):
            changed = copy.deepcopy(valid)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_case(changed)

    def test_independent_copies_and_source_only_sampler(self):
        sampler = DeterministicSourceSampler(self.protocol, 404)
        self.protocol["source_cases"][0]["assembly"]["segments"][0] = 6
        allowed = {mechanics_identity(c["assembly"]) for c in manifest()["source_cases"]}
        reserved = {mechanics_identity(c["assembly"]) for c in manifest()["reserved_cases"]}
        for _ in range(96):
            case = sampler()
            self.assertIn(mechanics_identity(case["assembly"]), allowed)
            self.assertNotIn(mechanics_identity(case["assembly"]), reserved)
            case["assembly"]["segments"][0] = 6
        report = sampler.exposure_report()
        self.assertEqual(report["draw_counts"], [3] * 32)
        self.assertEqual(report["completed_cycles"], 3)
        self.assertEqual(report["current_cycle_position"], 0)
        report["draw_counts"][0] = 999
        self.assertEqual(sampler.exposure_report()["draw_counts"], [3] * 32)
        self.assertEqual(manifest()["source_cases"][0]["assembly"]["segments"], [3, 3])

    def test_determinism_cycle_coverage_and_digest(self):
        first = DeterministicSourceSampler(self.protocol, 404)
        second = DeterministicSourceSampler(self.protocol, 404)
        other = DeterministicSourceSampler(self.protocol, 405)
        indices, other_indices = [], []
        independent_digest = hashlib.sha256()
        for draw in range(101):
            random.seed(draw)  # Global policy/test RNG must not rewind the curriculum.
            self.assertEqual(first(), second())
            other()
            indices.append(first.latest_index)
            other_indices.append(other.latest_index)
            independent_digest.update(f"{first.latest_index}\n".encode("ascii"))
        for start in (0, 32, 64):
            self.assertEqual(set(indices[start:start + 32]), set(range(32)))
        self.assertNotEqual(indices, other_indices)
        report = first.exposure_report()
        self.assertEqual(report, second.exposure_report())
        self.assertEqual(report["curriculum_seed"], 500404)
        self.assertEqual(report["draw_index_sha256"], independent_digest.hexdigest())
        self.assertEqual(report["draws"], 101)
        self.assertEqual(report["completed_cycles"], 3)
        self.assertEqual(report["current_cycle_position"], 5)
        self.assertEqual(sum(report["draw_counts"]), 101)
        self.assertLessEqual(max(report["draw_counts"]) - min(report["draw_counts"]), 1)
        self.assertEqual(json.loads(json.dumps(report)), report)


if __name__ == "__main__":
    unittest.main()
