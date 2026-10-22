"""Gym/SB3 adapter to isolated native worlds; no workshop HTTP control calls."""
from __future__ import annotations

import json
import hashlib
import math
import os
import subprocess
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

PROJECT = Path(__file__).resolve().parents[1]
NATIVE_COMMAND = ["docker", "exec", "-i", "droid-blocks-dev", "/workspace/build/native/droid-transfer-env"]
FRAME_COUNT = 12
FRAME_SIZE = 17
OBSERVATION_SIZE = FRAME_COUNT * FRAME_SIZE
DT = 0.1


class NativeBridge:
    def __init__(self, command=None):
        self.last_error_response = None
        self.request_tape = []
        self.stream_digest = hashlib.sha256()
        self.process = subprocess.Popen(
            command or NATIVE_COMMAND, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
            cwd=PROJECT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        self.spec = self.call({"op": "spec"})
        if self.spec["schema"] != "actuation_transfer_bridge_v1":
            self.close()
            raise RuntimeError("Native bridge version differs from experiment")

    def call(self, value):
        self.process.stdin.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            error = self.process.stderr.read() if self.process.poll() is not None else "no reply"
            raise RuntimeError(f"Native bridge ended unexpectedly: {error[:1000]}")
        response = json.loads(line)
        if value["op"] != "close":
            self.request_tape.append(copy_json(value))
            self.stream_digest.update(json.dumps({"request": value, "response": response},
                                                sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            self.stream_digest.update(b"\n")
        if not response.get("ok"):
            self.last_error_response = response
            raise RuntimeError(f"Native bridge rejected request: {response.get('error')}")
        return response.get("result")

    def close(self):
        if self.process.poll() is None:
            try:
                self.call({"op": "close"})
                self.process.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                # Only our own docker-exec client; never the container/service.
                self.process.terminate()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream:
                stream.close()


def local_frame(result):
    """Only the native local observation and completed action/reward enter here."""
    obs = result["observation"]
    if obs["schema_version"] != "construction_observation_v2":
        raise ValueError("Unexpected native observation schema")
    sensors = {entry["family_id"]: entry for entry in obs["reward_sensors"]}
    imu = sensors["imu_6axis_v0"]
    light = sensors["ambient_light_v0"]
    motor = obs["actuator_feedback"][0]["feedback"]
    force = imu["observations"]["specific_force_m_s2"] if imu["valid"] else [0.] * 3
    gyro = imu["observations"]["angular_velocity_rad_s"] if imu["valid"] else [0.] * 3
    lux = light["observations"]["illuminance_lux"] if light["valid"] else 0.
    values = [
        result["executed_effort"] / .15,
        math.sin(motor["position_rad"]), math.cos(motor["position_rad"]),
        motor["velocity_rad_s"] / 10., motor["current_a"] / .5,
        *(v / 50. for v in force), *(v / 10. for v in gyro), lux / 1000.,
        result["previous_request"] / .15,
        result["reward"] / (2. * DT),
        float(imu["valid"]), float(light["valid"]), 1.,
    ]
    frame = np.asarray(values, dtype=np.float32)
    if frame.shape != (FRAME_SIZE,) or not np.isfinite(frame).all():
        raise ValueError("Invalid local observation frame")
    return np.clip(frame, -5., 5.)


class SafetyStop(RuntimeError):
    """A held-out trial ends on a native stop and receives no reset rescue."""


def copy_json(value):
    return json.loads(json.dumps(value, allow_nan=False))


class NativeTransferVecEnv(VecEnv):
    def __init__(self, cases, *, case_sampler=None, horizon=640, record=False,
                 stop_on_safety=False, command=None):
        self.render_mode = None
        self.bridge = NativeBridge(command)
        self.cases = list(cases)
        self.case_sampler = case_sampler
        self.horizon = horizon
        self.record = record
        self.stop_on_safety = stop_on_safety
        self.frames = [deque(maxlen=FRAME_COUNT) for _ in cases]
        self.steps = np.zeros(len(cases), dtype=np.int64)
        self.returns = np.zeros(len(cases))
        self.pending_actions = None
        self.audit = []
        self.initial_physics = []
        self.latest = []
        self.total_interactions = 0
        self.total_native_steps = 0
        self.safety_stops = 0
        self.episode_count = 0
        self._configured = False
        super().__init__(len(cases), gym.spaces.Box(-5., 5., (OBSERVATION_SIZE,), np.float32),
                         gym.spaces.Discrete(3))

    def _stack(self, i, result, reset=False):
        if reset:
            self.frames[i].clear()
            self.frames[i].extend(np.zeros(FRAME_SIZE, np.float32) for _ in range(FRAME_COUNT - 1))
        self.frames[i].append(local_frame(result))
        return np.concatenate(self.frames[i])

    def reset(self):
        # Source sampling has its own protocol-owned Generator, seeded once by
        # the runner; policy/action RNG resets never rewind the case schedule.
        if self.case_sampler:
            self.cases = [self.case_sampler() for _ in self.cases]
        initial = self.bridge.call({"op": "configure", "cases": self.cases, "record": self.record})
        self._configured = True
        self.steps[:] = 0
        self.returns[:] = 0
        self.latest = initial
        self.initial_physics = [entry.get("physics") for entry in initial]
        self._reset_seeds()
        self._reset_options()
        return np.stack([self._stack(i, result, True) for i, result in enumerate(initial)])

    def step_async(self, actions):
        values = np.asarray(actions)
        if values.shape != (self.num_envs,) or values.dtype.kind not in "iu" or not np.isin(values, [0, 1, 2]).all():
            raise ValueError("Expected exactly one discrete action per native world")
        self.pending_actions = values.astype(np.int64).tolist()

    def step_wait(self):
        actions, self.pending_actions = self.pending_actions, None
        results = self.bridge.call({"op": "step", "actions": actions, "record": self.record})
        self.latest = results
        self.total_interactions += self.num_envs
        self.total_native_steps += sum(r["native_steps"] for r in results)
        observations, rewards, dones, infos, reset_slots = [], [], [], [], []
        safety_indices = []
        for i, result in enumerate(results):
            self.steps[i] += 1
            self.returns[i] += result["reward"]
            observation = self._stack(i, result)
            safety = result["terminated"]
            done = safety or self.steps[i] >= self.horizon
            info = {"TimeLimit.truncated": bool(done and not safety), "native": result}
            if self.record:
                self.audit.append({"slot": i, "decision": int(self.steps[i]),
                                   "action": actions[i], "result": result})
            if safety:
                self.safety_stops += 1
                safety_indices.append(i)
            if done:
                self.episode_count += 1
                info["terminal_observation"] = observation.copy()
                info["episode"] = {"r": float(self.returns[i]), "l": int(self.steps[i])}
                new_case = self.case_sampler() if self.case_sampler else self.cases[i]
                self.cases[i] = new_case
                reset_slots.append({"index": i, "case": new_case})
            observations.append(observation)
            rewards.append(result["reward"])
            dones.append(done)
            infos.append(info)
        if safety_indices and self.stop_on_safety:
            raise SafetyStop(f"Native safety stop in target slot(s) {safety_indices}; trial ends")
        if reset_slots:
            reset_results = self.bridge.call({"op": "reset", "slots": reset_slots, "record": False})
            for entry in reset_results:
                i = entry["index"]
                observations[i] = self._stack(i, entry["initial"], True)
                self.steps[i] = 0
                self.returns[i] = 0.
        return (np.stack(observations), np.asarray(rewards, np.float32),
                np.asarray(dones, bool), infos)

    def close(self):
        self.bridge.close()

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("Environment parameters are explicit immutable case manifests")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if method_name == "render":
            return [None for _ in self._get_indices(indices)]
        raise NotImplementedError(method_name)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
