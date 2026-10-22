"""Synthetic SB3 accounting/freeze tests. No native worlds or replay calls."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "training"))
if sys.platform != "linux" or not Path("/.dockerenv").is_file():
    raise SystemExit("Run these checks only inside the managed Linux development container.")

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import VecEnv
import conservative_specialization_study as driver
from conservative_specialization_manifest import manifest
import specialization_study as historical_driver
from specialization_manifest import manifest as historical_manifest
from native_transfer import SafetyStop

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)


class SyntheticEnv(VecEnv):
    """Deterministic vectors and scalar rewards, with no physical simulation."""
    def __init__(self, stop_at=None):
        self.render_mode = None
        self.horizon = 641
        self.stop_on_safety = True
        self.total_interactions = self.total_native_steps = self.resets = 0
        self.telemetry = []
        self.latest = []
        self.stop_at = stop_at
        self.pending_actions = None
        super().__init__(1, gym.spaces.Box(-5., 5., (204,), dtype=np.float32), gym.spaces.Discrete(3))

    def vector(self):
        result = np.zeros((1, 204), dtype=np.float32)
        result[0, -17:] = (self.total_interactions % 29) / 29.
        result[0, -1] = 1.
        return result

    def reset(self):
        self.resets += 1
        return self.vector()

    def step_async(self, actions):
        self.pending_actions = list(actions)

    def step_wait(self):
        action = int(self.pending_actions[0])
        self.total_interactions += 1
        self.total_native_steps += 5
        reward = (.12, .03, -.08)[action] + self.total_interactions * .0001
        self.telemetry.append({"decision": self.total_interactions, "action": action,
                               "native": {"reward": reward, "synthetic_endpoint": self.total_interactions}})
        if self.total_interactions == self.stop_at:
            raise SafetyStop("Synthetic safety stop; no reset rescue")
        return self.vector(), np.asarray([reward], np.float32), np.asarray([False]), [{}]

    def close(self):
        pass

    def get_attr(self, name, indices=None):
        return [getattr(self, name)]

    def set_attr(self, name, value, indices=None):
        raise NotImplementedError

    def env_method(self, name, *args, indices=None, **kwargs):
        return [None]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]


class ConservativeSpecializationDriverContract(unittest.TestCase):
    def setUp(self):
        self.study = manifest()
        self.scratch = Path("/tmp/droid-conservative-specialization-contract")
        self.scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="driver-", dir=self.scratch)
        self.root = Path(self.temp.name).resolve()
        self.dependencies = (torch, np, PPO, BaseCallback, None, SyntheticEnv, SafetyStop)

    def tearDown(self):
        self.assertTrue(self.root.is_relative_to(self.scratch.resolve()))
        self.assertFalse(any(path.is_symlink() for path in self.root.rglob("*")))
        self.temp.cleanup()

    def model(self, env, arm, *, state=None):
        ppo = driver.ppo_for_arm(self.study, arm)
        result = driver.helper.build_model(env, ppo, 101, target=True, dependencies=self.dependencies)
        if state is not None:
            result.policy.load_state_dict(state, strict=True)
        driver.configure_scope(result, arm, torch, ppo)
        result.set_random_seed(911007)
        return result

    def fake_ref(self, path):
        path = Path(path)
        return {"path": path.relative_to(self.root).as_posix(), "sha256": driver.digest(path)}

    def test_scoped_fresh_optimizers_and_explicit_categorical_contract(self):
        for arm, expected_tensors, expected_elements in (("frozen", 0, 0), ("conservative", 12, 34820),
                                                        ("standard", 12, 34820), ("scratch", 12, 34820)):
            env = SyntheticEnv()
            model = self.model(env, arm)
            value = driver.policy_snapshot(model, torch)
            self.assertEqual((value["trainable_tensors"], value["trainable_elements"]), (expected_tensors, expected_elements))
            self.assertTrue(value["optimizer"]["fresh"])
            self.assertEqual(value["optimizer"]["step_max"], 0)
            self.assertEqual(value["num_timesteps"], 0)
            self.assertEqual(value["optimization_epochs"], 0)
            self.assertFalse(value["rollout_buffer"]["full"])
            self.assertFalse(model.use_sde)
            expected_lr, expected_clip = (.0001, .1) if arm == "conservative" else (.0003, .2)
            self.assertEqual(value["ppo_settings"], {"learning_rate": expected_lr, "clip_range": expected_clip,
                                                    "target_kl": None, "use_sde": False})
            if arm != "frozen":
                self.assertEqual(len(value["optimizer"]["param_groups"][0]["parameter_names"]), 12)
                self.assertEqual(value["optimizer"]["param_groups"][0]["learning_rate"], expected_lr)
            self.assertEqual(env.resets, 0)

    def test_continuous_budget_six_updates_and_frozen_followup(self):
        env = SyntheticEnv()
        model = self.model(env, "standard")
        calls = []
        audit = driver.run_episode(model, env, "standard", self.study, torch, np,
                                   on_snapshot=lambda count, *_: calls.append(count))
        self.assertEqual(calls, [0, 64, 128, 192, 256, 320, 384, 640])
        self.assertEqual(env.resets, 1)
        self.assertEqual(env.total_interactions, 640)
        self.assertEqual(model.num_timesteps, 384)
        self.assertEqual(model._n_updates, 24)
        self.assertEqual([event["decisions"] for event in audit["training_updates"]], list(driver.UPDATE_DECISIONS))
        self.assertTrue(all(event["passed"] and event["torch_rng_unchanged"] for event in audit["training_updates"]))
        self.assertTrue(audit["continuation_weights_and_optimizer_unchanged"])
        self.assertEqual(audit["snapshots"]["after384"]["optimizer"], audit["snapshots"]["after640"]["optimizer"])
        self.assertEqual(audit["snapshots"]["after640"]["optimizer"]["step_min"], 24)
        self.assertEqual(audit["snapshots"]["after640"]["trainable_elements"], 0)
        self.assertEqual(audit["snapshots"]["after384"]["input_float32_le_sha256"],
                         env.telemetry[384]["input_float32_le_sha256"])
        self.assertEqual(len(env.telemetry), 640)

    def test_conservative_updates_full_network_with_its_locked_package(self):
        env = SyntheticEnv()
        model = self.model(env, "conservative")
        audit = driver.run_episode(model, env, "conservative", self.study, torch, np)
        first, final = audit["snapshots"]["initial"], audit["snapshots"]["after384"]
        for label in driver.PARTITIONS:
            self.assertNotEqual(first["partitions"][label]["sha256"], final["partitions"][label]["sha256"])
        self.assertEqual(final["optimizer"]["state_tensors"], 12)
        self.assertEqual(final["optimizer"]["step_min"], 24)
        self.assertEqual(final["optimizer"]["param_groups"][0]["learning_rate"], .0001)
        self.assertEqual(final["ppo_settings"]["clip_range"], .1)

    def test_all_arms_share_action_rng_stream_and_pretrained_first64_prefix(self):
        base_env = SyntheticEnv()
        base = self.model(base_env, "standard")
        initial = copy.deepcopy(base.policy.state_dict())
        trained = copy.deepcopy(initial)
        trained["action_net.bias"] += torch.tensor([.2, -.1, .05])
        rows, audits = {}, {}
        for arm in self.study["arms"]:
            env = SyntheticEnv()
            model = self.model(env, arm, state=initial if arm == "scratch" else trained)
            audits[arm] = driver.run_episode(model, env, arm, self.study, torch, np)
            rows[arm] = env.telemetry
        for arm in ("conservative", "standard"):
            self.assertEqual(rows["frozen"][:64], rows[arm][:64])
        states = lambda value: [(r["torch_rng_before_sha256"], r["torch_rng_after_sha256"]) for r in value]
        for arm in self.study["arms"]:
            self.assertEqual(states(rows["frozen"]), states(rows[arm]))
        frozen = audits["frozen"]["snapshots"]
        self.assertEqual(frozen["initial"]["parameter_sha256"], frozen["after640"]["parameter_sha256"])
        self.assertEqual(frozen["after640"]["optimization_epochs"], 0)

    def test_early_stop_records_real_endpoint_without_second_update_or_reset(self):
        env = SyntheticEnv(stop_at=65)
        model = self.model(env, "conservative")
        with self.assertRaises(SafetyStop):
            driver.run_episode(model, env, "conservative", self.study, torch, np)
        audit = env.specialization_episode_audit
        self.assertEqual((env.total_interactions, model.num_timesteps), (65, 64))
        self.assertEqual(env.resets, 1)
        self.assertEqual(len(env.telemetry), 65)
        self.assertEqual(len(audit["training_updates"]), 1)
        self.assertIn("after64", audit["snapshots"])
        self.assertNotIn("after128", audit["snapshots"])
        self.assertIsNone(audit["pending_action_audit"])

    def test_callback_early_exit_cannot_masquerade_as_sixth_update(self):
        class IncorrectBudgetStop(BaseCallback):
            def _on_step(self):
                return self.num_timesteps < 384
        env = SyntheticEnv()
        model = self.model(env, "standard")
        with self.assertRaisesRegex(ValueError, "six complete rollouts"):
            driver.run_episode(model, env, "standard", self.study, torch, np, callback=IncorrectBudgetStop())
        self.assertEqual(env.total_interactions, 384)
        self.assertEqual(model._n_updates, 20)
        self.assertNotIn("after384", env.specialization_episode_audit["snapshots"])

    def test_torch_rng_consumption_during_optimizer_is_rejected(self):
        env = SyntheticEnv()
        model = self.model(env, "conservative")
        original = model.train
        def consuming_train():
            original()
            torch.rand(1)
        model.train = consuming_train
        with self.assertRaisesRegex(ValueError, "changed RNG"):
            driver.run_episode(model, env, "conservative", self.study, torch, np)
        self.assertEqual(env.total_interactions, 64)
        self.assertFalse(env.specialization_episode_audit["training_updates"][0]["passed"])

    def test_zip_roundtrip_scoped_initial_and_update_tensors_no_closures(self):
        for arm in self.study["arms"]:
            env = SyntheticEnv()
            model = self.model(env, arm)
            saved = {}
            def save(count, current, snapshot):
                if count == 0 or (arm in driver.LEARNING_ARMS and count in driver.UPDATE_DECISIONS):
                    path = self.root / f"{arm}-{count}.zip"
                    driver.save_checkpoint(current, path)
                    saved[count] = (path, snapshot)
            audit = driver.run_episode(model, env, arm, self.study, torch, np, on_snapshot=save)
            for count, (path, snapshot) in saved.items():
                data, params, _ = load_from_zip_file(path, device="cpu", load_data=False)
                self.assertIsNone(data)
                self.assertEqual(set(params), {"policy"} if arm == "frozen" else {"policy", "policy.optimizer"})
                fresh = self.model(SyntheticEnv(), "frozen")
                driver.load_policy_weights(fresh, path, snapshot["parameter_sha256"])
                self.assertEqual(driver.helper.parameter_digest(fresh.policy), snapshot["parameter_sha256"])
                with zipfile.ZipFile(path) as archive:
                    metadata = json.loads(archive.read("data"))
                self.assertNotIn("train", metadata)
                self.assertNotIn("_get_torch_save_params", metadata)
                if arm != "frozen":
                    state = params["policy.optimizer"]
                    self.assertEqual(len(state["param_groups"][0]["params"]), 12)
                    self.assertEqual(len(state["state"]), 0 if count == 0 else 12)
                    self.assertEqual(state["param_groups"][0]["lr"], .0001 if arm == "conservative" else .0003)
                    self.assertEqual({int(value["step"].item()) for value in state["state"].values()},
                                     set() if count == 0 else {count // 64 * 4})
                with self.assertRaises(FileExistsError):
                    driver.save_checkpoint(model, path)
            self.assertTrue(audit["continuation_weights_and_optimizer_unchanged"])

    def test_guard_scope_has_six_baselines_and144_probes_and_requires_valid384(self):
        schedule = driver.guard_schedule(self.study)
        self.assertEqual(len(schedule), 150)
        self.assertEqual(sum(row[1] == "frozen" for row in schedule), 6)
        self.assertFalse(any(row[1] == "scratch" for row in schedule))
        value = {"checkpoints": {"after384": {"path": "checkpoint.zip", "sha256": "a" * 64,
                                             "parameter_sha256": "b" * 64, "decisions": 384}},
                 "training_updates": [{"decisions": count, "passed": True, "error": None} for count in driver.UPDATE_DECISIONS],
                 "snapshots": {"after384": {"parameters_finite": True, "num_timesteps": 384,
                    "optimization_epochs": 24, "optimizer": {"step_min": 24, "step_max": 24}, "parameter_sha256": "b" * 64}}}
        self.assertIsNotNone(driver.post384_checkpoint(value))
        value["complete"] = False  # A later continuation stop does not erase valid384.
        self.assertIsNotNone(driver.post384_checkpoint(value))
        value["training_updates"][1]["passed"] = False
        self.assertIsNone(driver.post384_checkpoint(value))
        self.assertEqual(driver.file_reference(value["checkpoints"]["after384"]),
                         {"path": "checkpoint.zip", "sha256": "a" * 64})

    def test_global_guard_release_failure_cannot_reach_native_dependencies(self):
        with patch.object(driver, "read_lock", return_value=({"study": self.study}, "a" * 64)), \
             patch.object(driver, "ref", side_effect=self.fake_ref), \
             patch.object(driver, "guard_release", side_effect=ValueError("Tampered target tape")), \
             patch.object(driver.helper, "learning_dependencies", side_effect=AssertionError("Native dependency reached")) as dependencies:
            result = driver.guards(self.root)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["actual_decisions"], 0)
        self.assertEqual(result["case_construction_requests"], 0)
        dependencies.assert_not_called()

    def test_missing384_writes_zero_world_receipt_and_existing_attempt_refuses(self):
        folder = self.root / "guards"
        folder.mkdir()
        source = {"kind": "post384_specialization", "case_id": self.study["target_cases"][0]["id"], "case_index": 0}
        with patch.object(driver, "ref", side_effect=self.fake_ref):
            value, _ = driver.trial(self.root, folder, {"study": self.study}, "a" * 64,
                                    101, "conservative", 0, None, None, (), guard_source=source)
            self.assertEqual(value["status"], "blocked")
            self.assertEqual(value["case_construction_requests"], 0)
            self.assertEqual(value["metrics"]["decisions"], 0)
            with self.assertRaises(FileExistsError):
                driver.trial(self.root, folder, {"study": self.study}, "a" * 64,
                             101, "conservative", 0, None, None, (), guard_source=source)

    def test_arm_settings_reach_constant_schedules_and_wrong_schedule_is_rejected(self):
        for arm in driver.LEARNING_ARMS:
            env = SyntheticEnv()
            model = self.model(env, arm)
            expected = self.study["per_arm_ppo"][arm]
            for progress in (0., .5, 1.):
                self.assertEqual(model.lr_schedule(progress), expected["learning_rate"])
                self.assertEqual(model.clip_range(progress), expected["clip_range"])
            model.lr_schedule = lambda _: .0003
            if arm == "conservative":
                with self.assertRaisesRegex(ValueError, "schedules"):
                    driver.configure_scope(model, arm, torch, driver.ppo_for_arm(self.study, arm))
            self.assertEqual(env.resets, 0)
        mutated = copy.deepcopy(self.study)
        mutated["per_arm_ppo"]["conservative"]["clip_range"] = .2
        with self.assertRaisesRegex(ValueError, "package"):
            driver.ppo_for_arm(mutated, "conservative")

    def test_standard_first128_and_frozen640_reproduce_old_synthetic_contract(self):
        old_study = historical_manifest()
        for old_arm, new_arm, count in (("full", "standard", 128), ("frozen", "frozen", 640)):
            old_env = SyntheticEnv()
            old_model = historical_driver.helper.build_model(old_env, old_study["ppo"], 101,
                                                             target=True, dependencies=self.dependencies)
            historical_driver.configure_scope(old_model, old_arm, torch, old_study["ppo"])
            old_model.set_random_seed(911007)
            old_audit = historical_driver.run_episode(old_model, old_env, old_arm, old_study, torch, np)
            new_env = SyntheticEnv()
            new_model = self.model(new_env, new_arm)
            new_audit = driver.run_episode(new_model, new_env, new_arm, self.study, torch, np)
            self.assertEqual(new_env.telemetry[:count], old_env.telemetry[:count])
            for label in ("initial", "after64", "after128"):
                for key in ("parameter_sha256", "partitions", "optimizer", "num_timesteps",
                            "optimization_epochs", "input_float32_le_sha256", "torch_rng_sha256"):
                    self.assertEqual(new_audit["snapshots"][label][key], old_audit["snapshots"][label][key])

    def test_stop_after384_preserves_valid_update_before_any_continuation_rescue(self):
        env = SyntheticEnv(stop_at=385)
        model = self.model(env, "conservative")
        checkpoints = {}
        def retain(count, current, snapshot):
            if count == 384:
                checkpoints["after384"] = {"path": "synthetic.zip", "sha256": "a" * 64,
                                            "parameter_sha256": snapshot["parameter_sha256"], "decisions": 384}
        with self.assertRaises(SafetyStop):
            driver.run_episode(model, env, "conservative", self.study, torch, np, on_snapshot=retain)
        audit = env.specialization_episode_audit
        self.assertEqual((env.total_interactions, model.num_timesteps, model._n_updates, env.resets), (385, 384, 24, 1))
        self.assertIsNotNone(driver.post384_checkpoint({**audit, "checkpoints": checkpoints, "complete": False}))
        self.assertNotIn("after640", audit["snapshots"])

    def test_archival_failure_is_retained_and_remaining_target_attempts_continue(self):
        calls = []
        releases = {seed: {"initial": {}, "final": {}, "initial_parameter_sha256": "a" * 64,
                           "final_parameter_sha256": "b" * 64} for seed in self.study["source_seeds"]}
        def collect(root, folder, protocol, sha, seed, arm, index, *unused):
            calls.append((seed, arm, index))
            if len(calls) == 1:
                raise OSError("Synthetic archive fault")
            name = f"target--{self.study['target_cases'][index]['id']}--{seed}--{arm}"
            attempt = folder / "trials" / name
            attempt.mkdir(parents=True)
            value = {"trial_id": name, "seed": seed, "arm": arm, "case_index": index,
                     "complete": True, "safe": True, "cancelled": False,
                     "metrics": driver.helper.metrics_from_results([]), "case_construction_requests": 0,
                     "exposure_exact": True}
            driver.write_new(attempt / "result.json", value)
            return value, self.fake_ref(attempt / "result.json")
        with patch.object(driver, "read_lock", return_value=({"study": self.study}, "a" * 64)), \
             patch.object(driver, "source_inputs", return_value=releases), \
             patch.object(driver, "ref", side_effect=self.fake_ref), \
             patch.object(driver, "read_bound", side_effect=lambda value: self.root / value["path"]), \
             patch.object(driver.helper, "learning_dependencies", return_value=()), \
             patch.object(driver, "trial", side_effect=collect):
            result = driver.targets(self.root)
        self.assertEqual(len(calls), 144)
        self.assertEqual(result["collected_trials"], 144)
        self.assertFalse(result["all_trials_complete_safe"])
        self.assertFalse(result["exposure_exact"])
        failure = json.loads((self.root / result["trial_results"][0]["path"]).read_text())
        self.assertTrue(failure["unknown_exposure"])
        self.assertIn("Synthetic archive fault", failure["error"])

    def test_existing_target_stage_refuses_before_dependencies(self):
        (self.root / "targets").mkdir()
        with patch.object(driver, "read_lock", return_value=({"study": self.study}, "a" * 64)), \
             patch.object(driver.helper, "learning_dependencies", side_effect=AssertionError("Dependency reached")) as dependencies:
            with self.assertRaises(FileExistsError):
                driver.targets(self.root)
        dependencies.assert_not_called()

    def test_invalid_checkpoint_ref_fails_before_environment_constructor(self):
        folder = self.root / "targets"
        folder.mkdir()
        with patch.object(driver, "ref", side_effect=self.fake_ref), \
             patch.object(driver, "read_bound", side_effect=ValueError("Synthetic bad checkpoint hash")), \
             patch.object(driver.helper, "make_instrumented_env_class", side_effect=AssertionError("World constructor reached")) as constructor:
            value, _ = driver.trial(self.root, folder, {"study": self.study}, "a" * 64,
                                    101, "conservative", 0, {"path": "absent.zip", "sha256": "a" * 64},
                                    "b" * 64, self.dependencies)
        constructor.assert_not_called()
        self.assertFalse(value["complete"])
        self.assertEqual(value["case_construction_requests"], 0)
        self.assertEqual(value["metrics"]["decisions"], 0)
        self.assertIn("Synthetic bad checkpoint hash", value["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
