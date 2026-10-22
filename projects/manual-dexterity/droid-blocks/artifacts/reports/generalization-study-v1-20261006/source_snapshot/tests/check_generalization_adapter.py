"""Source-only Linux integration checks; no training or reserved-world exposure."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "training"))
import native_transfer as adapter
from generalization_curriculum import DeterministicSourceSampler, manifest, mechanics_identity

COMMAND = [str(PROJECT / "build/native/droid-transfer-env")]


class GeneralizationAdapterContract(unittest.TestCase):
    def test_source_cycle_reset_and_exact_native_reward(self):
        protocol = manifest()
        sampler = DeterministicSourceSampler(protocol, 404)
        env = adapter.NativeTransferVecEnv(protocol["source_cases"][:8],
                                          case_sampler=sampler, horizon=4, command=COMMAND)
        try:
            observation = env.reset()
            self.assertEqual(observation.shape, (8, 204))
            self.assertEqual(observation.dtype, np.float32)
            self.assertFalse(np.count_nonzero(observation[:, :187]))
            self.assertTrue(np.all(observation[:, -1] == 1))
            self.assertTrue(np.all(observation[:, -3:-1] == 0))
            for batch in range(16):
                observation, rewards, done, infos = env.step(np.ones(8, dtype=np.int64))
                self.assertTrue(np.isfinite(observation).all())
                for slot, info in enumerate(infos):
                    self.assertEqual(info["native"]["native_steps"], 5)
                    self.assertAlmostEqual(float(rewards[slot]), info["native"]["reward"], places=7)
                    self.assertAlmostEqual(info["native"]["reward"],
                                           info["native"]["light_reward"] + info["native"]["imu_reward"], places=12)
                    self.assertEqual(bool(done[slot]), batch % 4 == 3)
                    if done[slot]:
                        self.assertTrue(info["TimeLimit.truncated"])
                        self.assertEqual(info["terminal_observation"][-3:-1].tolist(), [1., 1.])
                        self.assertEqual(observation[slot, -3:-1].tolist(), [0., 0.])
                        self.assertFalse(np.count_nonzero(observation[slot, :187]))
            self.assertEqual(env.total_interactions, 128)
            self.assertEqual(env.total_native_steps, 640)
            self.assertEqual(env.safety_stops, 0)
            self.assertTrue(all(count >= 1 for count in sampler.exposure_report()["draw_counts"]))
            source = {mechanics_identity(c["assembly"]) for c in protocol["source_cases"]}
            reserved = {mechanics_identity(c["assembly"]) for c in protocol["reserved_cases"]}
            for request in env.bridge.request_tape:
                cases = request.get("cases", []) + [entry["case"] for entry in request.get("slots", [])]
                for case in cases:
                    identity = mechanics_identity(case["assembly"])
                    self.assertIn(identity, source)
                    self.assertNotIn(identity, reserved)
        finally:
            env.close()

    def test_diagnostic_geometry_cannot_enter_policy_frame(self):
        bridge = adapter.NativeBridge(COMMAND)
        try:
            initial = bridge.call({"op": "configure", "cases": manifest()["source_cases"][:1], "record": True})[0]
            original = adapter.local_frame(initial)
            changed = copy.deepcopy(initial)
            changed["physics"] = {"world_pose": [999, 888, 777], "passive_joint": 100,
                                  "assembly": {"powered_hinge": 99}, "sun": [999, 0, -999]}
            changed["case_id"] = "a hidden target label"
            self.assertTrue(np.array_equal(original, adapter.local_frame(changed)))
        finally:
            bridge.close()

    def test_invalid_actions_do_not_advance_world(self):
        env = adapter.NativeTransferVecEnv(manifest()["source_cases"][:1], command=COMMAND)
        try:
            env.reset()
            initial_calls = len(env.bridge.request_tape)
            for invalid in (np.array([1.]), np.array([True]), np.array([3]), np.array([-1]), np.array([0, 1])):
                with self.assertRaises(ValueError):
                    env.step_async(invalid)
            self.assertEqual(len(env.bridge.request_tape), initial_calls)
            self.assertEqual(env.total_interactions, 0)
        finally:
            env.close()

    def test_identical_source_action_tape_replays_exactly(self):
        config = manifest()["source_cases"][2]
        tapes = []
        for repeat in range(2):
            bridge = adapter.NativeBridge(COMMAND)
            try:
                tape = [bridge.call({"op": "configure", "cases": [config], "record": True})]
                for action in (1, 2, 1, 0, 1):
                    tape.append(bridge.call({"op": "step", "actions": [action], "record": True}))
                tapes.append(tape)
            finally:
                bridge.close()
        self.assertEqual(tapes[0], tapes[1])
        self.assertTrue(all(not step[0]["terminated"] for step in tapes[0][1:]))

    def test_safety_stop_does_not_reset_rescue(self):
        bridge = adapter.NativeBridge(COMMAND)
        try:
            initial = bridge.call({"op": "configure", "cases": manifest()["source_cases"][:1], "record": False})[0]
            terminal = bridge.call({"op": "step", "actions": [1], "record": False})[0]
        finally:
            bridge.close()
        # A mocked terminal response checks wrapper behavior only. It is not a
        # measured physical fault and is never included in an experiment report.
        terminal["terminated"] = True

        class TerminalBridge:
            def __init__(self):
                self.calls = []
            def call(self, request):
                self.calls.append(copy.deepcopy(request))
                if request["op"] == "configure":
                    return [copy.deepcopy(initial)]
                if request["op"] == "step":
                    return [copy.deepcopy(terminal)]
                raise AssertionError("Safety-stopped opportunity was reset")
            def close(self):
                pass

        mocked = TerminalBridge()
        with patch.object(adapter, "NativeBridge", return_value=mocked):
            env = adapter.NativeTransferVecEnv(manifest()["source_cases"][:1],
                                              stop_on_safety=True, command=COMMAND)
            try:
                env.reset()
                with self.assertRaises(adapter.SafetyStop):
                    env.step(np.ones(1, dtype=np.int64))
                self.assertEqual([request["op"] for request in mocked.calls], ["configure", "step"])
                self.assertEqual(env.total_interactions, 1)
                self.assertEqual(env.total_native_steps, 5)
                self.assertEqual(env.safety_stops, 1)
            finally:
                env.close()


if __name__ == "__main__":
    if sys.platform != "linux" or not Path("/.dockerenv").is_file():
        raise SystemExit("Run these integration checks inside the managed Linux development container.")
    unittest.main(verbosity=2)
