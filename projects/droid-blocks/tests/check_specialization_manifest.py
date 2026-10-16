"""Synthetic specialization declarations and bindings; construct no native worlds."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
import specialization_manifest as subject
from generalization_curriculum import manifest as old_curriculum, mechanics_identity


HISTORICAL_RECEIPT_HASHES = {
    "protocol.json": "4cf6fe175dfad1150c9264e2ab074f8df7fe575a978724bc7d3ffddb16915727",
    "sources/sealed-checkpoints.json": "6404315c901c19972c7b190f797026431fc88ecd9006c6d0ebaa740feed18508",
    "verification/native-replay-v1.json": "92839768519ec6c3df1a7dc56f9ca25adcf563375886f26e3057fe679a19bf20",
    "verification/tensors-v1.json": "35cbaf63da3a84ade402efc486573fb2b0dde520865b04407ad02faab14e5544",
}


def set_nested(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


class ManifestContract(unittest.TestCase):
    def test_exact_opportunity_and_charged_update_accounting(self):
        value = subject.manifest()
        subject.validate_manifest(value)
        self.assertEqual(value["schema"], "specialization_manifest_v1")
        self.assertEqual(value["source_seeds"], [101, 202, 303])
        self.assertEqual(value["arms"], ["frozen", "head_only", "full", "scratch"])
        self.assertEqual(value["budget"], {
            "prefix_decisions": 128, "total_decisions": 640,
            "continuation_decisions": 512, "horizon_decisions": 641,
            "rollouts": 2, "adam_steps": 8, "update_decisions": [64, 128]})
        maximum = value["maximum_collection"]
        self.assertEqual(3 * 12 * 4, maximum["target_opportunities"])
        self.assertEqual(3 * 12 * 4 * 640, maximum["target_decisions"])
        self.assertEqual(3 * 12 * 3 * 128, maximum["learning_decisions_included"])
        self.assertEqual(3 * 2 + 3 * 12 * 2 * 2, maximum["guard_opportunities"])
        self.assertEqual(maximum["guard_opportunities"] * 640, 96000)
        self.assertEqual(maximum["total_opportunities"], 294)
        self.assertEqual(maximum["target_decisions"] + maximum["guard_decisions"], 188160)
        self.assertEqual(maximum["total_decisions"] * 5, maximum["native_controls"])
        self.assertEqual(maximum["native_controls"] * 10, maximum["physics_steps"])
        self.assertEqual(maximum["total_decisions"] / 10, maximum["aggregate_simulated_seconds"])

    def test_exposed_targets_and_preselected_sampled_guards(self):
        value, previous = subject.manifest(), old_curriculum()
        self.assertEqual([entry["case"] for entry in value["target_cases"]], previous["reserved_cases"])
        self.assertEqual([entry["id"] for entry in value["target_cases"]],
                         [name.replace("reserved-", "exposed-", 1)
                          for name in previous["case_ids"]["reserved"]])
        self.assertEqual([entry["case"] for entry in value["guard_cases"]],
                         [previous["development_cases"][index] for index in (0, 11)])
        self.assertEqual([entry["id"] for entry in value["guard_cases"]],
                         [previous["case_ids"]["development"][index] for index in (0, 11)])
        self.assertEqual(len({mechanics_identity(entry["case"]["assembly"])
                              for entry in value["target_cases"]}), 6)
        self.assertEqual(len({entry["id"] for entry in value["target_cases"]}), 12)

    def test_trainability_and_fixed_primary_gate(self):
        value = subject.manifest()
        self.assertEqual(value["trainable"]["head_only"]["parameters"], 3 * 64 + 3 + 64 + 1)
        self.assertEqual(value["trainable"]["head_only"]["tensors"],
                         ["action_net.weight", "action_net.bias", "value_net.weight", "value_net.bias"])
        self.assertEqual(value["trainable"]["frozen"]["parameters"], 0)
        for arm in ("full", "scratch"):
            self.assertEqual(value["trainable"][arm], {"tensors": 12, "parameters": 34820})
        gates = value["gates"]
        self.assertEqual(gates["primary_arm"], "head_only")
        self.assertEqual(gates["secondary_arm"], "full")
        self.assertIn("entire640", gates["comparison"])
        self.assertTrue(gates["requires_all_36_corresponding_pairs_safe_complete"])
        self.assertEqual(gates["median_seed_mean_gain_strictly_greater_than"], 3.2)
        self.assertEqual(gates["minimum_positive_seed_means"], 2)
        self.assertEqual(gates["scratch_contrast_median_strictly_greater_than"], 0.)
        self.assertEqual(gates["sampled_retention"], {
            "each_seed_mean_gain_at_least": -3.2,
            "requires_all_corresponding_probes_safe_complete": True})

    def test_rejects_budget_gate_arm_rng_and_trainability_tampering(self):
        original = subject.manifest()
        mutations = [
            (("budget", "prefix_decisions"), 64),
            (("budget", "continuation_decisions"), 640),
            (("budget", "total_decisions"), 768),
            (("budget", "horizon_decisions"), 640),
            (("budget", "adam_steps"), 16),
            (("budget", "update_decisions"), [64, 128, 192]),
            (("arms",), ["head_only", "frozen", "full", "scratch"]),
            (("source_seeds",), [101, 202, 404]),
            (("gates", "primary_arm"), "full"),
            (("gates", "median_seed_mean_gain_strictly_greater_than"), 0.),
            (("gates", "comparison"), "512-decision follow-up only"),
            (("gates", "requires_all_36_corresponding_pairs_safe_complete"), False),
            (("gates", "minimum_positive_seed_means"), 1),
            (("gates", "scratch_contrast_median_strictly_greater_than"), -1.),
            (("gates", "sampled_retention", "each_seed_mean_gain_at_least"), -10.),
            (("trainable", "head_only", "parameters"), 195),
            (("trainable", "head_only", "tensors"), ["action_net.weight", "action_net.bias"]),
            (("action_rng", "target_base"), 810001),
            (("wall_caps_s", "target"), 360),
            (("maximum_collection", "total_decisions"), 188161),
        ]
        for path, replacement in mutations:
            with self.subTest(path=path):
                altered = copy.deepcopy(original)
                set_nested(altered, path, replacement)
                with self.assertRaises(ValueError):
                    subject.validate_manifest(altered)
        self.assertEqual(subject.manifest(), original)

    def test_rejects_condition_scope_type_and_unknown_field_tampering(self):
        original = subject.manifest()
        mutations = [
            (("target_cases",), original["target_cases"][:-1]),
            (("guard_cases",), original["guard_cases"][:1]),
            (("target_cases", 0, "case", "assembly", "segments"), [6, 3]),
            (("target_cases", 0, "id"), "outcome-selected-case"),
            (("budget", "total_decisions"), "640"),
            (("gates", "minimum_positive_seed_means"), 2.0),
            (("gates", "requires_all_36_corresponding_pairs_safe_complete"), 1),
        ]
        for path, replacement in mutations:
            with self.subTest(path=path):
                altered = copy.deepcopy(original)
                set_nested(altered, path, replacement)
                with self.assertRaises(ValueError):
                    subject.validate_manifest(altered)
        altered = copy.deepcopy(original)
        altered["posthoc_gate"] = True
        with self.assertRaises(ValueError):
            subject.validate_manifest(altered)
        altered = copy.deepcopy(original)
        altered["target_cases"][0]["extra"] = "hidden context"
        with self.assertRaises(ValueError):
            subject.validate_manifest(altered)


class SyntheticBindings(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="droid-specialization-bindings-", dir="/tmp")
        self.root = Path(self.temporary.name).resolve()
        self.assertEqual(self.root.parent, Path("/tmp"))
        self.old = self.root / "artifacts/reports/old"
        self.old.mkdir(parents=True)
        self.preservation = self.root / "artifacts/reports/preservation/baseline.json"
        self.preservation.parent.mkdir()
        self.preservation.write_text("{}\n")
        self.patches = [patch.object(subject, "PROJECT", self.root),
                        patch.object(subject, "OLD", self.old),
                        patch.object(subject, "PRESERVATION", self.preservation)]
        for context in self.patches:
            context.start()
        self.test_source = self.root / "tests/synthetic_contract.py"
        self.test_source.parent.mkdir()
        self.test_source.write_text("# synthetic contract bytes\n")
        self.preflight = self.preservation.parent / "preflight.json"
        self.preflight.write_text(json.dumps({
            "passed": True, "native_worlds_constructed": 0,
            "test_sources": [subject.ref(self.test_source)]}))
        self.records = {}
        for seed in (101, 202, 303):
            folder = self.old / f"sources/source-pooled-{seed}"
            folder.mkdir(parents=True)
            final, initial = folder / "policy.zip", folder / "policy_initial.zip"
            final.write_bytes(f"trained-{seed}".encode())
            initial.write_bytes(f"initial-{seed}".encode())
            self.records[seed] = {
                "complete": True, "seed": seed, "source_arm": "pooled",
                "policy_sha256": subject.digest(final), "parameter_sha256": "a" * 64,
                "initial_checkpoint": {"policy_sha256": subject.digest(initial),
                                       "parameter_sha256": "b" * 64}}
        self.write_records_and_seal()
        real_ref = subject.ref

        def synthetic_receipt_ref(path):
            path = Path(path)
            relative = path.relative_to(self.old).as_posix() if path.is_relative_to(self.old) else None
            # Fixed old receipt bytes are mocked only here. Synthetic result and
            # policy files still use real SHA-256 and the actual binding code.
            if relative in HISTORICAL_RECEIPT_HASHES:
                return {"path": path.relative_to(self.root).as_posix(),
                        "sha256": HISTORICAL_RECEIPT_HASHES[relative]}
            return real_ref(path)

        self.ref_patch = patch.object(subject, "ref", side_effect=synthetic_receipt_ref)
        self.ref_patch.start()

    def tearDown(self):
        self.ref_patch.stop()
        for context in reversed(self.patches):
            context.stop()
        self.assertEqual(self.root.parent, Path("/tmp"))
        self.assertFalse(any(path.is_symlink() for path in self.root.rglob("*")))
        self.temporary.cleanup()

    def write_records_and_seal(self):
        source_results = []
        for seed, record in self.records.items():
            path = self.old / f"sources/source-pooled-{seed}/result.json"
            path.write_text(json.dumps(record))
            source_results.append({"path": path.relative_to(self.old).as_posix(),
                                   "sha256": subject.digest(path)})
        seal = {"operational_complete": True, "sealed_before_any_evaluation": True,
                "source_results": source_results}
        (self.old / "sources/sealed-checkpoints.json").write_text(json.dumps(seal))

    def test_exact_source_releases_initializations_and_preservation_reference(self):
        value = subject.input_refs()
        self.assertEqual([item["seed"] for item in value["releases"]], [101, 202, 303])
        for item in value["releases"]:
            self.assertTrue(subject.read_bound(item["result"]).is_file())
            self.assertTrue(subject.read_bound(item["final"]).is_file())
            self.assertTrue(subject.read_bound(item["initial"]).is_file())
            self.assertEqual(item["final_parameter_sha256"], "a" * 64)
            self.assertEqual(item["initial_parameter_sha256"], "b" * 64)
        self.assertEqual(subject.read_bound(value["receipts"]["preservation_baseline"]), self.preservation)
        self.assertEqual(subject.read_bound(value["receipts"]["synthetic_preflight"]), self.preflight)

    def test_swapped_self_consistent_result_rejected_by_original_seal(self):
        folder = self.old / "sources/source-pooled-101"
        (folder / "policy.zip").write_bytes(b"different-but-internally-consistent-policy")
        record = copy.deepcopy(self.records[101])
        record["policy_sha256"] = subject.digest(folder / "policy.zip")
        (folder / "result.json").write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "historical seal"):
            subject.input_refs()

    def test_checkpoint_bytes_must_match_the_bound_result(self):
        for name in ("policy.zip", "policy_initial.zip"):
            with self.subTest(checkpoint=name):
                path = self.old / "sources/source-pooled-101" / name
                original = path.read_bytes()
                path.write_bytes(b"swapped-checkpoint")
                with self.assertRaisesRegex(ValueError, "checkpoint byte mismatch"):
                    subject.input_refs()
                path.write_bytes(original)

    def test_source_record_must_be_completed_correct_seed_and_pooled(self):
        original = copy.deepcopy(self.records[101])
        for key, replacement in (("complete", False), ("seed", 404), ("source_arm", "narrow")):
            with self.subTest(field=key):
                self.records[101] = copy.deepcopy(original)
                self.records[101][key] = replacement
                self.write_records_and_seal()
                with self.assertRaisesRegex(ValueError, "Invalid pooled release"):
                    subject.input_refs()

    def test_release_requires_operational_seal_before_evaluation(self):
        path = self.old / "sources/sealed-checkpoints.json"
        original = json.loads(path.read_text())
        for key in ("operational_complete", "sealed_before_any_evaluation"):
            altered = copy.deepcopy(original)
            altered[key] = False
            path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "operationally sealed"):
                subject.input_refs()

    def test_preflight_must_pass_without_native_exposure_and_bind_test_bytes(self):
        original = json.loads(self.preflight.read_text())
        for key, replacement in (("passed", False), ("native_worlds_constructed", 1)):
            altered = copy.deepcopy(original)
            altered[key] = replacement
            self.preflight.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "preflight did not pass"):
                subject.input_refs()
        self.preflight.write_text(json.dumps(original))
        self.test_source.write_text("# altered after the preflight\n")
        with self.assertRaisesRegex(ValueError, "Evidence bytes differ"):
            subject.input_refs()

    def test_references_reject_changed_bytes_unknown_fields_absolute_and_traversal(self):
        reference = {"path": self.preservation.relative_to(self.root).as_posix(),
                     "sha256": subject.digest(self.preservation)}
        self.assertEqual(subject.read_bound(reference), self.preservation)
        for altered in ({**reference, "extra": 1}, {**reference, "sha256": "0" * 64},
                        {**reference, "path": str(self.preservation)},
                        {**reference, "path": "../outside.json"}):
            with self.subTest(reference=altered):
                with self.assertRaises(ValueError):
                    subject.read_bound(altered)
        link = self.root / "linked.json"
        link.symlink_to(self.preservation)
        try:
            with self.assertRaisesRegex(ValueError, "symlinks"):
                subject.read_bound({"path": "linked.json", "sha256": reference["sha256"]})
        finally:
            link.unlink()


if __name__ == "__main__":
    if sys.platform != "linux" or not Path("/.dockerenv").is_file():
        raise SystemExit("Run these checks inside the existing managed Linux container")
    unittest.main(verbosity=2)
