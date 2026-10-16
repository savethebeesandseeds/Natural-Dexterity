"""Synthetic prospective contracts; no learning or native worlds are constructed."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import conservative_specialization_manifest as subject
import specialization_manifest as shared
from generalization_curriculum import manifest as curriculum, mechanics_identity


PRIOR_HASHES = {
    "protocol.json": "5b5dbda3123c20a030021c141e4812ec0db1c8f66fd9030eca78b8265ec002e4",
    "verification/audit-v1.json": "65b297f720b0b1fb8e1975d71073e707d759eef947d789e6f1f3f4b07f6d56b2",
    "verification/final-provenance.json": "62aa877921e97edd1a57a0d29564251099502c5fb9c4ddc1f304aa57d2a29d72",
    "targets/stage.json": "6b7c09622a1c0615609071f997922a0a4b8fce0fe843193ea99e4d105f8c1a78",
    "targets/seal.json": "186cb490c899bfe8ec01c42b691bdbaa894d1f0f9af76822932094bd7cd70209",
}


def replace_nested(value, keys, replacement):
    for key in keys[:-1]:
        value = value[key]
    value[keys[-1]] = replacement


class DeclarationContracts(unittest.TestCase):
    def test_charged_longer_budget_and_all_declared_opportunities(self):
        value = subject.manifest()
        subject.validate_manifest(value)
        self.assertEqual(value["schema"], "conservative_specialization_manifest_v1")
        self.assertEqual(value["source_seeds"], [101, 202, 303])
        self.assertEqual(value["arms"], ["frozen", "conservative", "standard", "scratch"])
        self.assertEqual(value["budget"], {
            "prefix_decisions": 384, "total_decisions": 640,
            "continuation_decisions": 256, "horizon_decisions": 641,
            "rollouts": 6, "adam_steps": 24,
            "update_decisions": [64, 128, 192, 256, 320, 384]})
        maximum = value["maximum_collection"]
        self.assertEqual(3 * 12 * 4, maximum["target_opportunities"])
        self.assertEqual(3 * 12 * 4 * 640, maximum["target_decisions"])
        self.assertEqual(3 * 12 * 3 * 384, maximum["learning_decisions_included"])
        self.assertEqual(3 * 2 + 3 * 12 * 2 * 2, maximum["guard_opportunities"])
        self.assertEqual(maximum["guard_decisions"], 150 * 640)
        self.assertEqual(maximum["total_opportunities"], 294)
        self.assertEqual(maximum["total_decisions"], 188160)
        self.assertEqual(maximum["native_controls"], 188160 * 5)
        self.assertEqual(maximum["physics_steps"], 188160 * 5 * 10)
        self.assertEqual(maximum["aggregate_simulated_seconds"], 18816)

    def test_exposed_cases_sampled_guards_and_same_rng_are_fixed(self):
        value, previous, old = subject.manifest(), shared.manifest(), curriculum()
        self.assertEqual(value["target_cases"], previous["target_cases"])
        self.assertEqual([entry["case"] for entry in value["target_cases"]], old["reserved_cases"])
        self.assertEqual(value["guard_cases"], previous["guard_cases"])
        self.assertEqual([entry["case"] for entry in value["guard_cases"]],
                         [old["development_cases"][i] for i in (0, 11)])
        self.assertEqual(len({mechanics_identity(row["case"]["assembly"])
                              for row in value["target_cases"]}), 6)
        self.assertEqual(len({row["id"] for row in value["target_cases"]}), 12)
        for key in ("target_base", "guard_base", "seed_multiplier"):
            self.assertEqual(value["action_rng"][key], previous["action_rng"][key])
        self.assertIn("no reseed at384", value["action_rng"]["rule"])

    def test_full_trainability_and_two_predeclared_ppo_recipes(self):
        value = subject.manifest()
        for arm in ("conservative", "standard", "scratch"):
            self.assertEqual(value["trainable"][arm], {"tensors": 12, "parameters": 34820})
        self.assertEqual(value["trainable"]["frozen"], {"parameters": 0})
        self.assertEqual(value["per_arm_ppo"]["conservative"], {"learning_rate": .0001, "clip_range": .1})
        for arm in ("frozen", "standard", "scratch"):
            self.assertEqual(value["per_arm_ppo"][arm], {"learning_rate": .0003, "clip_range": .2})
        prior = shared.manifest()["ppo"]
        self.assertEqual({key: val for key, val in value["ppo"].items() if key != "transfer"},
                         {key: val for key, val in prior.items() if key != "transfer"})
        self.assertEqual(value["policy_input"], shared.manifest()["policy_input"])

    def test_benefit_and_individual_reliability_remain_separate(self):
        gates = subject.manifest()["gates"]
        self.assertEqual(gates["primary_arm"], "conservative")
        self.assertEqual(gates["secondary_arm"], "standard")
        self.assertIn("whole640", gates["comparison"])
        self.assertEqual(gates["median_seed_mean_gain_strictly_greater_than"], 3.2)
        self.assertEqual(gates["minimum_positive_seed_means"], 2)
        self.assertTrue(gates["requires_all_36_corresponding_pairs_safe_complete"])
        self.assertEqual(gates["scratch_contrast_median_strictly_greater_than"], 0.)
        self.assertEqual(gates["regression_diagnostic_prefix_loss_floor"], -1.92)
        self.assertEqual(gates["reliability"], {
            "target_each_pair_gain_at_least": -3.2, "probe_each_pair_gain_at_least": -3.2,
            "requires_all_corresponding_pairs_safe_complete": True})
        self.assertEqual(gates["sampled_retention"], {
            "each_seed_mean_gain_at_least": -3.2,
            "requires_all_corresponding_probes_safe_complete": True})
        self.assertIn("No automatic promotion", gates["combined_readiness"])

    def test_strict_budget_recipe_gate_and_identity_tamper_rejection(self):
        original = subject.manifest()
        changes = [
            (("budget", "prefix_decisions"), 128),
            (("budget", "total_decisions"), 768),
            (("budget", "continuation_decisions"), 512),
            (("budget", "horizon_decisions"), 640),
            (("budget", "adam_steps"), 8),
            (("budget", "rollouts"), 2),
            (("budget", "update_decisions"), [64, 128]),
            (("source_seeds",), [303, 202, 101]),
            (("arms",), ["frozen", "standard", "conservative", "scratch"]),
            (("per_arm_ppo", "conservative", "learning_rate"), .0003),
            (("per_arm_ppo", "conservative", "clip_range"), .2),
            (("per_arm_ppo", "standard", "learning_rate"), .0001),
            (("trainable", "conservative", "parameters"), 260),
            (("gates", "primary_arm"), "standard"),
            (("gates", "median_seed_mean_gain_strictly_greater_than"), 0.),
            (("gates", "minimum_positive_seed_means"), 1),
            (("gates", "comparison"), "256 frozen-tail return only"),
            (("gates", "reliability", "target_each_pair_gain_at_least"), -10.),
            (("gates", "reliability", "probe_each_pair_gain_at_least"), -10.),
            (("gates", "reliability", "requires_all_corresponding_pairs_safe_complete"), False),
            (("gates", "sampled_retention", "each_seed_mean_gain_at_least"), -10.),
            (("action_rng", "target_base"), 810001),
            (("maximum_collection", "learning_decisions_included"), 13824),
            (("target_cases", 0, "id"), "selected-winner"),
            (("target_cases", 0, "case", "assembly", "segments"), [6, 3]),
            (("guard_cases",), list(reversed(original["guard_cases"]))),
            (("budget", "total_decisions"), "640"),
            (("gates", "minimum_positive_seed_means"), 2.0),
        ]
        for keys, replacement in changes:
            with self.subTest(field=keys):
                altered = copy.deepcopy(original)
                replace_nested(altered, keys, replacement)
                with self.assertRaises(ValueError):
                    subject.validate_manifest(altered)
        self.assertEqual(subject.manifest(), original)
        altered = copy.deepcopy(original)
        altered["outcome_selected_arm"] = "conservative"
        with self.assertRaises(ValueError):
            subject.validate_manifest(altered)


class SyntheticInputBindings(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="droid-conservative-bindings-", dir="/tmp")
        self.root = Path(self.temporary.name).resolve()
        self.assertEqual(self.root.parent, Path("/tmp"))
        self.prior = self.root / "artifacts/reports/previous-specialization"
        self.prior.mkdir(parents=True)
        self.preservation = self.root / "artifacts/reports/new-preservation/baseline.json"
        self.preservation.parent.mkdir()
        self.preservation.write_text("{}\n")
        self.patches = [patch.object(shared, "PROJECT", self.root),
                        patch.object(subject, "PROJECT", self.root),
                        patch.object(subject, "PRIOR", self.prior),
                        patch.object(subject, "PRESERVATION", self.preservation),
                        patch.object(subject, "previous_inputs", return_value={
                            "releases": [{"seed": seed} for seed in (101, 202, 303)],
                            "receipts": {"preservation_baseline": {"path": "old", "sha256": "old"},
                                         "synthetic_preflight": {"path": "old", "sha256": "old"}}})]
        for context in self.patches:
            context.start()
        self.test_source = self.root / "tests/synthetic_contract.py"
        self.test_source.parent.mkdir()
        self.test_source.write_text("# synthetic contract bytes\n")
        self.preflight = self.preservation.parent / "preflight.json"
        self.preflight.write_text(json.dumps({"passed": True, "native_worlds_constructed": 0,
                                             "test_sources": [shared.ref(self.test_source)]}))
        for name in PRIOR_HASHES:
            path = self.prior / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        self.audit = self.prior / "verification/audit-v1.json"
        self.audit.write_text(json.dumps({"integrity_passed": True, "native_replay_verified": True,
                                         "operationally_complete": True}))
        self.result_paths, references = [], []
        for seed in (101, 202, 303):
            for index in range(12):
                for arm in ("frozen", "full"):
                    path = self.prior / f"targets/trials/{seed}-{index}-{arm}/result.json"
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps({"seed": seed, "case_id": f"exposed-{index}", "arm": arm}))
                    self.result_paths.append(path)
                    references.append(shared.ref(path))
        self.stage = self.prior / "targets/stage.json"
        self.stage.write_text(json.dumps({"trial_results": references}))

        def relative_prior(path):
            path = Path(path)
            return path.relative_to(self.prior).as_posix() if path.is_relative_to(self.prior) else None

        def synthetic_ref(path):
            name = relative_prior(path)
            if name in PRIOR_HASHES:
                return {"path": Path(path).relative_to(self.root).as_posix(), "sha256": PRIOR_HASHES[name]}
            return shared.ref(path)

        def synthetic_bound(reference):
            path = self.root / reference["path"]
            name = relative_prior(path)
            # Only the five external immutable receipts are mocked. Actual
            # synthetic trial and preflight-source bytes use real SHA-256.
            if name in PRIOR_HASHES:
                if reference["sha256"] != PRIOR_HASHES[name]:
                    raise ValueError("Synthetic historical receipt hash differs")
                return path
            return shared.read_bound(reference)

        self.binding_patches = [patch.object(subject, "ref", side_effect=synthetic_ref),
                                patch.object(subject, "read_bound", side_effect=synthetic_bound)]
        for context in self.binding_patches:
            context.start()

    def tearDown(self):
        for context in reversed(self.binding_patches + self.patches):
            context.stop()
        self.assertEqual(self.root.parent, Path("/tmp"))
        self.assertFalse(any(path.is_symlink() for path in self.root.rglob("*")))
        self.temporary.cleanup()

    def test_new_preflight_baseline_and_all72_historical_controls_bound(self):
        value = subject.input_refs()
        self.assertEqual([row["seed"] for row in value["releases"]], [101, 202, 303])
        self.assertEqual(shared.read_bound(value["receipts"]["preservation_baseline"]), self.preservation)
        self.assertEqual(shared.read_bound(value["receipts"]["synthetic_preflight"]), self.preflight)
        prior = value["prior_specialization"]
        self.assertEqual(len(prior["targets"]), 72)
        self.assertEqual({(row["seed"], row["case_id"], row["arm"]) for row in prior["targets"]},
                         {(seed, f"exposed-{index}", arm)
                          for seed in (101, 202, 303) for index in range(12) for arm in ("frozen", "full")})

    def test_changed_historical_trial_bytes_are_rejected(self):
        self.result_paths[0].write_text('{"swapped":"trial"}')
        with self.assertRaisesRegex(ValueError, "Evidence bytes differ"):
            subject.input_refs()

    def test_missing_historical_arm_cannot_pass_the72_control_contract(self):
        value = json.loads(self.stage.read_text())
        value["trial_results"].pop()
        self.stage.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "all36 earlier"):
            subject.input_refs()

    def test_prior_independent_verification_must_be_complete_and_passed(self):
        original = json.loads(self.audit.read_text())
        for key in original:
            altered = copy.deepcopy(original)
            altered[key] = False
            self.audit.write_text(json.dumps(altered))
            with self.subTest(flag=key), self.assertRaisesRegex(ValueError, "independently verified"):
                subject.input_refs()

    def test_new_preflight_must_pass_without_exposure_and_bind_source_bytes(self):
        original = json.loads(self.preflight.read_text())
        for key, replacement in (("passed", False), ("native_worlds_constructed", 1)):
            altered = copy.deepcopy(original)
            altered[key] = replacement
            self.preflight.write_text(json.dumps(altered))
            with self.subTest(flag=key), self.assertRaisesRegex(ValueError, "preflight did not pass"):
                subject.input_refs()
        self.preflight.write_text(json.dumps(original))
        self.test_source.write_text("# changed after preflight\n")
        with self.assertRaisesRegex(ValueError, "Evidence bytes differ"):
            subject.input_refs()


if __name__ == "__main__":
    if sys.platform != "linux" or not Path("/.dockerenv").is_file():
        raise SystemExit("Run these checks inside the existing managed Linux container")
    unittest.main(verbosity=2)
