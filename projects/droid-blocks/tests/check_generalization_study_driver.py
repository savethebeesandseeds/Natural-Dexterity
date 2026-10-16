"""Linux contract checks with mock trials; never configure a reserved world."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "training"))
import generalization_study as driver
from generalization_study_manifest import manifest
from generalization_curriculum import mechanics_identity


def trial(partition, controller, seed, index, reward):
    return {"partition": partition, "controller": controller, "seed": seed,
            "case_index": index, "metrics": {"native_reward": reward},
            "complete": True, "safe": True, "frozen_weights_unchanged": True}


def source_trials(study):
    rows = []
    for partition, controller in (("development", "pooled"), ("narrow_development", "narrow")):
        for index in range(len(study["case_ids"][partition])):
            rows.append(trial(partition, "quiet", None, index, 0.))
            for seed in study["source_seeds"]:
                rows.append(trial(partition, controller, seed, index, 4.))
    return rows


def reserved_mock_trials(study):
    rows = []
    for index in range(12):
        for fixed in study["witness_modes"]:
            rows.append(trial("reserved", fixed, None, index, 0. if fixed == "quiet" else 5.))
        for seed in study["source_seeds"]:
            for controller, reward in (("pooled", 6.), ("narrow", 4.), ("untrained", 1.)):
                rows.append(trial("reserved", controller, seed, index, reward))
    return rows


class StudyDriverContract(unittest.TestCase):
    def setUp(self):
        self.study = manifest()
        self.scratch = Path("/tmp/droid-generalization-study-task")
        self.scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="driver-contract-", dir=self.scratch)
        self.root = Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(self.scratch.resolve()))

    def tearDown(self):
        # These tests create only finite synthetic files, launch no processes,
        # and never write into the workspace or a dependency environment.
        self.assertTrue(self.root.is_relative_to(self.scratch.resolve()))
        self.assertFalse(any(p.is_symlink() for p in self.root.rglob("*")))
        self.temp.cleanup()

    def test_narrow_eight_slots_and_balanced_independent_sampling(self):
        cases = self.study["source_cases"]["narrow"]
        slots = driver.bootstrap_cases(cases, 8)
        self.assertEqual(len(slots), 8)
        self.assertEqual(slots, cases + cases)
        slots[0]["assembly"]["segments"][0] = 99
        self.assertEqual(cases[0]["assembly"]["segments"], [3, 3])
        reserved = {mechanics_identity(c["assembly"]) for c in self.study["reserved_cases"]}
        for arm, count in (("narrow", 4), ("pooled", 32)):
            sampler = driver.StudySourceSampler(self.study, 101, arm)
            second = driver.StudySourceSampler(self.study, 101, arm)
            draws = [sampler() for _ in range(count * 2)]
            self.assertEqual(draws, [second() for _ in draws])
            self.assertEqual(sampler.exposure_report()["draw_counts"], [2] * count)
            self.assertFalse(reserved & {mechanics_identity(c["assembly"]) for c in draws})

    def test_strict_source_margin_and_own_domain_separation(self):
        rows = source_trials(self.study)
        self.assertTrue(driver.evaluate_source_gate(self.study, rows)["passes"])
        # Narrow's transfer failure on pooled-known geometry is not a failure
        # of competence in narrow's own four source conditions.
        rows.append({**trial("development", "narrow", 101, 0, -100.), "safe": False})
        rows.append({**trial("development", "untrained", 101, 0, -100.), "safe": False})
        self.assertTrue(driver.evaluate_source_gate(self.study, rows)["passes"])
        rows[1]["metrics"]["native_reward"] = 3.2
        gate = driver.evaluate_source_gate(self.study, rows)
        self.assertFalse(gate["passes"])
        self.assertFalse(gate["comparisons"][0]["passes"])
        rows[1]["metrics"]["native_reward"] = 4.
        rows[1]["complete"] = False
        self.assertFalse(driver.evaluate_source_gate(self.study, rows)["passes"])
        self.assertFalse(driver.evaluate_source_gate(self.study, rows[1:])["passes"])

    def test_breadth_usefulness_and_untrained_eligibility_are_distinct(self):
        rows = reserved_mock_trials(self.study)
        analysis = driver.evaluate_challenge(self.study, rows)
        self.assertTrue(analysis["breadth"]["passes"])
        self.assertTrue(analysis["useful_first_use"]["passes"])
        untrained = next(r for r in rows if r["controller"] == "untrained")
        untrained["complete"] = untrained["safe"] = False
        untrained["metrics"]["native_reward"] = -100.
        analysis = driver.evaluate_challenge(self.study, rows)
        self.assertTrue(analysis["breadth"]["passes"])
        self.assertEqual(analysis["offline_learning"]["complete_safe_pairs"], 70)
        self.assertEqual(analysis["offline_learning"]["realized_pairs"], 72)
        self.assertNotEqual(analysis["offline_learning"]["mean_realized_trained_minus_untrained"],
                            analysis["offline_learning"]["mean_complete_safe_trained_minus_untrained"])
        quiet = next(r for r in rows if r["controller"] == "quiet")
        quiet["safe"] = False
        self.assertFalse(driver.evaluate_challenge(self.study, rows)["breadth"]["passes"])
        self.assertFalse(driver.evaluate_challenge(self.study, rows)["useful_first_use"]["passes"])
        quiet["safe"] = True
        for row in rows:
            if row["controller"] == "pooled":
                row["metrics"]["native_reward"] = 3.
            elif row["controller"] == "narrow":
                row["metrics"]["native_reward"] = 1.
        analysis = driver.evaluate_challenge(self.study, rows)
        self.assertTrue(analysis["breadth"]["passes"])
        self.assertFalse(analysis["useful_first_use"]["passes"])

    def test_breadth_ties_and_single_positive_seed_cannot_pass(self):
        rows = reserved_mock_trials(self.study)
        for row in rows:
            if row["controller"] == "pooled":
                row["metrics"]["native_reward"] = 6. if row["seed"] == 101 else 4.
        analysis = driver.evaluate_challenge(self.study, rows)
        self.assertEqual(analysis["breadth"]["median_seed_mean"], 0.)
        self.assertFalse(analysis["breadth"]["passes"])
        pooled = next(r for r in rows if r["controller"] == "pooled")
        pooled["frozen_weights_unchanged"] = False
        self.assertFalse(driver.evaluate_challenge(self.study, rows)["breadth"]["all_pooled_narrow_and_quiet_complete_safe"])

    def test_rhythms_use_full_period_and_start_negative(self):
        for mode, phase in (("rhythm_2s", 10), ("rhythm_4s", 20)):
            self.assertEqual([driver.fixed_action(mode, i) for i in range(phase * 3)],
                             [0] * phase + [2] * phase + [0] * phase)
        self.assertEqual([driver.fixed_action(mode, 0) for mode in ("negative", "quiet", "positive")], [0, 1, 2])
        with self.assertRaises(ValueError):
            driver.fixed_action("retrospectively_selected_servo", 0)

    def test_changed_evidence_and_escaping_paths_are_rejected(self):
        path = self.root / "bound.json"
        driver.write_new(path, {"passes": True})
        ref = driver.reference(self.root, path)
        path.write_text('{"passes":false}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            driver.bound_json(self.root, ref)
        with self.assertRaises(ValueError):
            driver.safe_path(self.root, "../outside.json")

    def test_failed_release_blocks_before_any_learning_or_native_dependency(self):
        protocol = {"study": self.study}
        for entry, release in ((driver.development, "source_release"), (driver.challenge, "challenge_release")):
            for failure in ("Source incomplete", "Initial parameter identities differ", ValueError("Evidence hash changed")):
                folder = self.root / f"{entry.__name__}-{str(failure).split()[0]}"
                folder.mkdir()
                with patch.object(driver, "read_lock", return_value=(folder, protocol, "a" * 64)), \
                     patch.object(driver, release, side_effect=failure if isinstance(failure, Exception) else None,
                                  return_value=(None, failure)), \
                     patch.object(driver, "blocked_stage", return_value={"status": "blocked", "actual_decisions": 0}) as blocked, \
                     patch.object(driver.helper, "learning_dependencies", side_effect=AssertionError("Native dependency reached")) as deps:
                    result = entry(folder)
                self.assertEqual(result["status"], "blocked")
                self.assertEqual(result["actual_decisions"], 0)
                blocked.assert_called_once()
                deps.assert_not_called()

    def test_sealed_initial_identity_is_checked_from_all_six_results(self):
        digest = "a" * 64
        refs = [{"path": f"sources/source-{arm}-{seed}/result.json", "sha256": digest}
                for seed in self.study["source_seeds"] for arm in self.study["source_arms"]]
        records = {ref["path"]: {"complete": True, "protocol_sha256": digest,
                   "source_arm": arm, "seed": seed, "policy_path": "policy.zip", "policy_sha256": digest,
                   "initial_checkpoint": {"policy_path": "policy_initial.zip", "policy_sha256": digest,
                                          "parameter_sha256": str(seed)},
                   "request_tape": {"path": "source-requests.json.gz", "sha256": digest}}
                   for ref, (seed, arm) in zip(refs, ((s, a) for s in self.study["source_seeds"] for a in self.study["source_arms"]))}
        records["seal.json"] = {"operational_complete": True, "sealed_before_any_evaluation": True, "source_results": refs}
        stage = {"status": "complete", "operational_complete": True, "seal": {"path": "seal.json"}}
        with patch.object(driver, "read_stage", return_value=stage), \
             patch.object(driver, "bound_json", side_effect=lambda root, ref: records[ref["path"]]), \
             patch.object(driver, "sha256_file", return_value=digest):
            releases, reason = driver.source_release(self.root, digest)
            self.assertIsNone(reason)
            self.assertEqual(len(releases), 6)
            records[refs[1]["path"]]["initial_checkpoint"]["parameter_sha256"] = "different tensors"
            releases, reason = driver.source_release(self.root, digest)
            self.assertIsNone(releases)
            self.assertIn("identity", reason)

    def test_unexpected_source_failure_still_attempts_remaining_sources(self):
        calls = []
        def attempt(root, folder, study, protocol_hash, seed, arm, deps):
            calls.append((seed, arm))
            if len(calls) == 1:
                raise RuntimeError("Synthetic archival failure")
            return ({"seed": seed, "source_arm": arm, "complete": False, "cancelled": False,
                     "initial_checkpoint": None, "actual_decisions": 0, "native_steps": 0},
                    {"path": f"synthetic-{arm}-{seed}", "sha256": "a" * 64})
        with patch.object(driver, "read_lock", return_value=(self.root, {"study": self.study}, "a" * 64)), \
             patch.object(driver.helper, "learning_dependencies", return_value=()), \
             patch.object(driver, "train_source", side_effect=attempt), \
             patch.object(driver, "stage_record", side_effect=lambda *args, **kw: kw) as record:
            result = driver.sources(self.root)
        self.assertEqual(calls, [(s, a) for s in (101, 202, 303) for a in ("pooled", "narrow")])
        self.assertEqual(result["collected_source_attempts"], 6)
        self.assertFalse(result["operational_complete"])
        record.assert_called_once()

    def test_source_initial_tensors_match_and_evaluation_horizon_has_no_rescue_reset(self):
        # This integration touches only source[0:4], with no gradient updates.
        import numpy as np
        deps = driver.helper.learning_dependencies()
        base = deps[5]
        env_type = driver.helper.make_instrumented_env_class(base)
        envs = []
        try:
            digests = []
            for arm in ("pooled", "narrow"):
                cases = self.study["source_cases"][arm]
                env = env_type(driver.bootstrap_cases(cases, 8), source_cases=cases,
                               case_sampler=driver.StudySourceSampler(self.study, 101, arm),
                               horizon=640, command=driver.NATIVE_COMMAND)
                envs.append(env)
                model = driver.helper.build_model(env, self.study["ppo"], 101, dependencies=deps)
                self.assertFalse(model.policy.optimizer.state)
                digests.append(driver.helper.parameter_digest(model.policy))
            self.assertEqual(digests[0], digests[1])
            env = env_type([self.study["source_cases"]["narrow"][0]], horizon=641,
                           stop_on_safety=True, command=driver.NATIVE_COMMAND)
            envs.append(env)
            observation = env.reset()
            self.assertEqual(observation.shape, (1, 204))
            for _ in range(640):
                observation, _, done, _ = env.step(np.asarray([1], dtype=np.int64))
                self.assertFalse(done[0])
            self.assertEqual(env.total_interactions, 640)
            ops = [request["op"] for request in env.bridge.request_tape]
            self.assertEqual(ops.count("configure"), 1)
            self.assertEqual(ops.count("reset"), 0)
            self.assertEqual(ops.count("step"), 640)
        finally:
            for env in envs:
                env.close()


if __name__ == "__main__":
    if sys.platform != "linux" or not Path("/.dockerenv").is_file():
        raise SystemExit("Run checks inside the managed Linux development container.")
    unittest.main(verbosity=2)
