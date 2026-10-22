"""Static study checks; run in Linux and never construct reserved worlds."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from generalization_study_manifest import manifest, validate_manifest
from generalization_curriculum import mechanics_identity


class StudyManifestContract(unittest.TestCase):
    def test_equal_budget_and_exact_partitions(self):
        protocol = manifest()
        validate_manifest(protocol)
        self.assertEqual([len(protocol["source_cases"][a]) for a in ("pooled", "narrow")], [32, 4])
        self.assertEqual([len(protocol[k]) for k in
                          ("development_cases", "narrow_development_cases", "reserved_cases")], [12, 4, 12])
        self.assertEqual(protocol["source_budget"]["decisions"] % (8 * 256), 0)
        self.assertEqual(protocol["source_budget"]["decisions"] * 6, 589824)
        self.assertEqual((3 * 12 * 3 + 12 + 3 * 4 * 2 + 4) * 640, 94720)
        self.assertEqual((12 * 5 + 3 * 12 * 3) * 640, 107520)
        self.assertEqual(sum(protocol["maximum_collection"][k] for k in
                             ("source_decisions", "development_decisions", "challenge_decisions_including_witnesses")), 792064)
        for case in protocol["source_cases"]["narrow"]:
            self.assertEqual(case["assembly"]["segments"], [3, 3])
            self.assertEqual(case["assembly"]["blocks"], [])

    def test_whole_assembly_exclusion_and_known_own_domain(self):
        protocol = manifest()
        groups = {key: {mechanics_identity(c["assembly"]) for c in protocol[key]}
                  for key in ("development_cases", "narrow_development_cases", "reserved_cases")}
        pooled = {mechanics_identity(c["assembly"]) for c in protocol["source_cases"]["pooled"]}
        narrow = {mechanics_identity(c["assembly"]) for c in protocol["source_cases"]["narrow"]}
        self.assertLessEqual(narrow, pooled)
        self.assertLessEqual(groups["development_cases"], pooled)
        self.assertEqual(groups["narrow_development_cases"], narrow)
        self.assertFalse(pooled & groups["reserved_cases"])
        self.assertFalse(groups["development_cases"] & groups["narrow_development_cases"])

    def test_no_mutable_defaults_and_no_budget_or_gate_drift(self):
        original = manifest()
        altered = copy.deepcopy(original)
        altered["source_budget"]["decisions"] += 2048
        with self.assertRaises(ValueError):
            validate_manifest(altered)
        altered = copy.deepcopy(original)
        altered["gates"]["source_margin"] = 0.
        with self.assertRaises(ValueError):
            validate_manifest(altered)
        altered = copy.deepcopy(original)
        altered["reserved_cases"][0] = copy.deepcopy(altered["source_cases"]["pooled"][0])
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_manifest(altered)
        altered["source_cases"]["pooled"][0]["assembly"]["segments"] = [6, 6]
        self.assertEqual(manifest(), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
