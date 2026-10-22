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
import specialization_study as driver
from specialization_manifest import manifest
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


class SpecializationDriverContract(unittest.TestCase):
    def setUp(self):
        self.study = manifest()
        self.scratch = Path("/tmp/droid-specialization-contract")
        self.scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="driver-", dir=self.scratch)
        self.root = Path(self.temp.name).resolve()
        self.dependencies = (torch, np, PPO, BaseCallback, None, SyntheticEnv, SafetyStop)

    def tearDown(self):
        self.assertTrue(self.root.is_relative_to(self.scratch.resolve()))
        self.assertFalse(any(path.is_symlink() for path in self.root.rglob("*")))
        self.temp.cleanup()

    def model(self, env, arm, *, state=None):
        result = driver.helper.build_model(env, self.study["ppo"], 101, target=True, dependencies=self.dependencies)
        if state is not None:
            result.policy.load_state_dict(state, strict=True)
        driver.configure_scope(result, arm, torch, self.study["ppo"])
        result.set_random_seed(911007)
        return result

    def fake_ref(self, path):
        path = Path(path)
        return {"path": path.relative_to(self.root).as_posix(), "sha256": driver.digest(path)}

    def test_scoped_fresh_optimizers_and_explicit_categorical_contract(self):
        for arm, expected_tensors, expected_elements in (("frozen", 0, 0), ("head_only", 4, 260),
                                                        ("full", 12, 34820), ("scratch", 12, 34820)):
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
            if arm == "head_only":
                self.assertEqual(set(value["optimizer"]["param_groups"][0]["parameter_names"]), set(driver.HEAD_NAMES))
            self.assertEqual(env.resets, 0)

    def test_continuous_budget_two_updates_and_frozen_followup(self):
        env = SyntheticEnv()
        model = self.model(env, "full")
        calls = []
        audit = driver.run_episode(model, env, "full", self.study, torch, np,
                                   on_snapshot=lambda count, *_: calls.append(count))
        self.assertEqual(calls, [0, 64, 128, 640])
        self.assertEqual(env.resets, 1)
        self.assertEqual(env.total_interactions, 640)
        self.assertEqual(model.num_timesteps, 128)
        self.assertEqual(model._n_updates, 8)
        self.assertEqual([event["decisions"] for event in audit["training_updates"]], [64, 128])
        self.assertTrue(all(event["passed"] and event["torch_rng_unchanged"] for event in audit["training_updates"]))
        self.assertTrue(audit["continuation_weights_and_optimizer_unchanged"])
        self.assertEqual(audit["snapshots"]["after128"]["optimizer"], audit["snapshots"]["after640"]["optimizer"])
        self.assertEqual(audit["snapshots"]["after640"]["optimizer"]["step_min"], 8)
        self.assertEqual(audit["snapshots"]["after640"]["trainable_elements"], 0)
        self.assertEqual(audit["snapshots"]["after128"]["input_float32_le_sha256"],
                         env.telemetry[128]["input_float32_le_sha256"])
        self.assertEqual(len(env.telemetry), 640)

    def test_heads_change_but_both_trunks_stay_bit_identical(self):
        env = SyntheticEnv()
        model = self.model(env, "head_only")
        audit = driver.run_episode(model, env, "head_only", self.study, torch, np)
        first, final = audit["snapshots"]["initial"], audit["snapshots"]["after128"]
        for label in ("actor_trunk", "value_trunk"):
            self.assertEqual(first["partitions"][label], final["partitions"][label])
        for label in ("actor_head", "value_head"):
            self.assertNotEqual(first["partitions"][label]["sha256"], final["partitions"][label]["sha256"])
        self.assertEqual(final["optimizer"]["state_tensors"], 4)
        self.assertEqual(final["optimizer"]["step_min"], 8)

    def test_all_arms_share_action_rng_stream_and_pretrained_first64_prefix(self):
        base_env = SyntheticEnv()
        base = self.model(base_env, "full")
        initial = copy.deepcopy(base.policy.state_dict())
        trained = copy.deepcopy(initial)
        trained["action_net.bias"] += torch.tensor([.2, -.1, .05])
        rows, audits = {}, {}
        for arm in self.study["arms"]:
            env = SyntheticEnv()
            model = self.model(env, arm, state=initial if arm == "scratch" else trained)
            audits[arm] = driver.run_episode(model, env, arm, self.study, torch, np)
            rows[arm] = env.telemetry
        for arm in ("head_only", "full"):
            self.assertEqual(rows["frozen"][:64], rows[arm][:64])
        states = lambda value: [(r["torch_rng_before_sha256"], r["torch_rng_after_sha256"]) for r in value]
        for arm in self.study["arms"]:
            self.assertEqual(states(rows["frozen"]), states(rows[arm]))
        frozen = audits["frozen"]["snapshots"]
        self.assertEqual(frozen["initial"]["parameter_sha256"], frozen["after640"]["parameter_sha256"])
        self.assertEqual(frozen["after640"]["optimization_epochs"], 0)

    def test_early_stop_records_real_endpoint_without_second_update_or_reset(self):
        env = SyntheticEnv(stop_at=65)
        model = self.model(env, "head_only")
        with self.assertRaises(SafetyStop):
            driver.run_episode(model, env, "head_only", self.study, torch, np)
        audit = env.specialization_episode_audit
        self.assertEqual((env.total_interactions, model.num_timesteps), (65, 64))
        self.assertEqual(env.resets, 1)
        self.assertEqual(len(env.telemetry), 65)
        self.assertEqual(len(audit["training_updates"]), 1)
        self.assertIn("after64", audit["snapshots"])
        self.assertNotIn("after128", audit["snapshots"])
        self.assertIsNone(audit["pending_action_audit"])

    def test_callback_early_exit_cannot_masquerade_as_second_update(self):
        class IncorrectBudgetStop(BaseCallback):
            def _on_step(self):
                return self.num_timesteps < 128
        env = SyntheticEnv()
        model = self.model(env, "full")
        with self.assertRaisesRegex(ValueError, "two complete rollouts"):
            driver.run_episode(model, env, "full", self.study, torch, np, callback=IncorrectBudgetStop())
        self.assertEqual(env.total_interactions, 128)
        self.assertEqual(model._n_updates, 4)
        self.assertNotIn("after128", env.specialization_episode_audit["snapshots"])

    def test_torch_rng_consumption_during_optimizer_is_rejected(self):
        env = SyntheticEnv()
        model = self.model(env, "head_only")
        original = model.train
        def consuming_train():
            original()
            torch.rand(1)
        model.train = consuming_train
        with self.assertRaisesRegex(ValueError, "changed RNG"):
            driver.run_episode(model, env, "head_only", self.study, torch, np)
        self.assertEqual(env.total_interactions, 64)
        self.assertFalse(env.specialization_episode_audit["training_updates"][0]["passed"])

    def test_zip_roundtrip_scoped_initial_and_update_tensors_no_closures(self):
        for arm in ("frozen", "head_only", "full"):
            env = SyntheticEnv()
            model = self.model(env, arm)
            saved = {}
            def save(count, current, snapshot):
                if count in (0, 64, 128):
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
                    self.assertEqual(len(state["param_groups"][0]["params"]), 4 if arm == "head_only" else 12)
                    self.assertEqual(len(state["state"]), 0 if count == 0 else 4 if arm == "head_only" else 12)
                    self.assertEqual({int(value["step"].item()) for value in state["state"].values()},
                                     set() if count == 0 else {count // 64 * 4})
                with self.assertRaises(FileExistsError):
                    driver.save_checkpoint(model, path)
            self.assertTrue(audit["continuation_weights_and_optimizer_unchanged"])

    def test_guard_scope_has_six_baselines_and144_probes_and_requires_valid128(self):
        schedule = driver.guard_schedule(self.study)
        self.assertEqual(len(schedule), 150)
        self.assertEqual(sum(row[1] == "frozen" for row in schedule), 6)
        self.assertFalse(any(row[1] == "scratch" for row in schedule))
        value = {"checkpoints": {"after128": {"path": "checkpoint.zip", "sha256": "a" * 64,
                                             "parameter_sha256": "b" * 64, "decisions": 128}},
                 "training_updates": [{"passed": True}, {"passed": True}],
                 "snapshots": {"after128": {"parameters_finite": True, "num_timesteps": 128,
                    "optimization_epochs": 8, "optimizer": {"step_min": 8, "step_max": 8}, "parameter_sha256": "b" * 64}}}
        self.assertIsNotNone(driver.post128_checkpoint(value))
        value["complete"] = False  # A later continuation stop does not erase valid128.
        self.assertIsNotNone(driver.post128_checkpoint(value))
        value["training_updates"][1]["passed"] = False
        self.assertIsNone(driver.post128_checkpoint(value))
        self.assertEqual(driver.file_reference(value["checkpoints"]["after128"]),
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

    def test_missing128_writes_zero_world_receipt_and_existing_attempt_refuses(self):
        folder = self.root / "guards"
        folder.mkdir()
        source = {"kind": "post128_specialization", "case_id": self.study["target_cases"][0]["id"], "case_index": 0}
        with patch.object(driver, "ref", side_effect=self.fake_ref):
            value, _ = driver.trial(self.root, folder, {"study": self.study}, "a" * 64,
                                    101, "head_only", 0, None, None, (), guard_source=source)
            self.assertEqual(value["status"], "blocked")
            self.assertEqual(value["case_construction_requests"], 0)
            self.assertEqual(value["metrics"]["decisions"], 0)
            with self.assertRaises(FileExistsError):
                driver.trial(self.root, folder, {"study": self.study}, "a" * 64,
                             101, "head_only", 0, None, None, (), guard_source=source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
