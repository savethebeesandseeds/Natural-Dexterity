#!/usr/bin/env python3
"""Replay all portable live-actor cases through the retained historical engine.

This is an implementation check on already exposed layouts, not another
generalization experiment. It cannot update parameters or tune the sampler.
"""
import argparse
import gzip
import hashlib
import json
import math
import subprocess
from collections import deque
from pathlib import Path

import numpy as np
import torch
from native_transfer import local_frame

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/reports/pretrained-workshop-v1-20261006"
OLD_SHA = "377b754985b2977c947e37e41cbbff286595fe70b08888c9fc32ece1db2d9777"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Driver:
    def __init__(self, binary):
        self.process = subprocess.Popen([str(binary)], cwd=ROOT, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def call(self, request):
        self.process.stdin.write(json.dumps(request, separators=(",", ":"), allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise ValueError("Native process exited: " + self.process.stderr.read())
        result = json.loads(line)
        if not result.get("ok"):
            raise ValueError(result.get("error", "Native operation failed"))
        return result.get("result")

    def close(self):
        if self.process.poll() is None:
            self.call({"op": "close"})
        self.process.communicate(timeout=10)
        if self.process.returncode != 0:
            raise ValueError("Native process failed closing")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=2)
    args = parser.parse_args()
    if args.version < 2:
        raise ValueError("The strengthened verifier writes a new receipt version >=2")
    destination = REPORT / f"native-workshop-v{args.version}.json"
    if destination.exists():
        raise ValueError("Verification receipt already exists; never overwrite evidence")
    old = REPORT / "runtime_snapshot" / OLD_SHA / "droid-transfer-env"
    if digest(old) != OLD_SHA:
        raise ValueError("Retained historical binary changed")
    actor_path = ROOT / "artifacts/pretrained/pooled-seed101-v1.json"
    artifact = json.loads(actor_path.read_text())
    tensors = {item["name"]: torch.tensor(item["data"], dtype=torch.float32).reshape(item["shape"])
               for item in artifact["tensors"]}
    torch.set_num_threads(1)

    def probabilities(values):
        value = torch.from_numpy(values)
        for layer in (0, 2):
            prefix = f"mlp_extractor.policy_net.{layer}"
            value = torch.tanh(torch.nn.functional.linear(value, tensors[prefix + ".weight"], tensors[prefix + ".bias"]))
        logits = torch.nn.functional.linear(value, tensors["action_net.weight"], tensors["action_net.bias"])
        return torch.distributions.Categorical(logits=logits).probs.detach().numpy()

    # Reserve a new receipt before starting worlds. Failure records and obtained
    # prefixes are evidence, rather than silently discarded retry material.
    receipt_stream = destination.open("x")
    live = replay = None
    failures = []
    cases = []
    traces = REPORT / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    try:
        live = Driver(ROOT / "build/native/droid-pretrained-session")
        replay = Driver(old)
        state = live.call({"op": "state"})
        for body in ("long_upper", "long_lower", "weighted"):
            for motor in ("base", "elbow"):
                for lamp in ("left", "right"):
                    name = f"{body}-{motor}-{lamp}"
                    state = live.call({"op": "control", "request": {"action": "rebuild",
                        "expected_session_id": state["session_id"], "body": body, "motor": motor, "lamp": lamp}})
                    fresh = state
                    live.call({"op": "control", "request": {"action": "wake", "expected_session_id": state["session_id"]}})
                    state = live.call({"op": "advance", "steps": 3200})
                    trace = live.call({"op": "trace"})
                    path = traces / (name + ".json.gz")
                    if path.exists():
                        with gzip.open(path, "rt", encoding="utf-8") as stream:
                            if json.load(stream) != trace:
                                raise ValueError("Retained trace differs from this deterministic native run")
                    else:
                        with gzip.open(path, "wt", encoding="utf-8") as stream:
                            json.dump(trace, stream, separators=(",", ":"), allow_nan=False)
                    if state["status"] != "completed" or len(trace["decisions"]) != 640:
                        raise ValueError(f"Incomplete demo: {name}")
                    initial = replay.call({"op": "configure", "cases": [trace["case"]], "record": True})[0]
                    if initial["observation"] != fresh["observation"] or initial["physics"] != fresh["physics"]:
                        raise ValueError(f"Initial native reset mismatch: {name}")
                    frames = deque([np.zeros(17, dtype=np.float32) for _ in range(11)], maxlen=12)
                    frames.append(local_frame(initial))
                    rewards, max_prob_error = [], 0.
                    rng = 1
                    for decision in trace["decisions"]:
                        mask = (1 << 64) - 1
                        rng = (rng + 0x9E3779B97F4A7C15) & mask
                        random_bits = ((rng ^ (rng >> 30)) * 0xBF58476D1CE4E5B9) & mask
                        random_bits = ((random_bits ^ (random_bits >> 27)) * 0x94D049BB133111EB) & mask
                        random_bits ^= random_bits >> 31
                        expected_uniform = (random_bits >> 11) / (1 << 53)
                        if expected_uniform != decision["uniform"]:
                            raise ValueError(f"Portable RNG sequence mismatch: {name}:{decision['start_step']}")
                        values = np.concatenate(frames).astype(np.float32, copy=False)
                        if hashlib.sha256(values.astype("<f4").tobytes()).hexdigest() != decision["input_float32_le_sha256"]:
                            raise ValueError(f"Local frame/history mismatch: {name}:{decision['start_step']}")
                        expected = probabilities(values)
                        difference = float(np.max(np.abs(expected - np.asarray(decision["probabilities"], dtype=np.float32))))
                        max_prob_error = max(max_prob_error, difference)
                        if difference > 1e-5:
                            raise ValueError(f"Actor probability mismatch: {name}")
                        masses = np.asarray(decision["probabilities"], dtype=np.float32).astype(np.float64).tolist()
                        cumulative, total = 0., sum(masses)
                        action = max(i for i, mass in enumerate(masses) if mass > 0)
                        for i, mass in enumerate(masses):
                            cumulative += mass / total
                            if mass > 0 and decision["uniform"] < cumulative:
                                action = i
                                break
                        if action != decision["action"]:
                            raise ValueError(f"Portable categorical sampler mismatch: {name}")
                        actual = replay.call({"op": "step", "actions": [action], "record": True})[0]
                        for field in ("efforts", "reward", "native_steps", "elapsed_s", "terminated", "transition_sha256"):
                            if actual[field] != decision[field]:
                                raise ValueError(f"Historical native transition mismatch: {name}:{decision['start_step']}:{field}")
                        if actual["terminated"]:
                            raise ValueError(f"Unexpected stop: {name}")
                        rewards.append(actual["reward"])
                        frames.append(local_frame(actual))
                    if actual["observation"] != trace["final_observation"] or actual["physics"] != trace["final_physics"]:
                        raise ValueError(f"Final physical state mismatch: {name}")
                    if actual["observation"] != state["observation"] or actual["physics"] != state["physics"]:
                        raise ValueError(f"Browser-facing final state mismatch: {name}")
                    for total in (state["total_reward"], trace["total_reward"]):
                        if not math.isclose(total, sum(rewards), rel_tol=1e-12, abs_tol=1e-10):
                            raise ValueError(f"Browser-facing total reward mismatch: {name}")
                    final_values = np.concatenate(frames).astype("<f4", copy=False)
                    if hashlib.sha256(final_values.tobytes()).hexdigest() != state["brain_history"]["input_float32_le_sha256"]:
                        raise ValueError(f"Browser-facing final history mismatch: {name}")
                    cases.append({"case": name, "status": state["status"], "decisions": 640, "native_steps": 3200,
                        "native_transition_sha256_exact": True, "local_input_bytes_exact": True,
                        "final_physics_exact": True, "browser_state_exact": True,
                        "portable_rng_exact": True, "total_reward_checked": True,
                        "max_probability_abs_error": max_prob_error,
                        "return": sum(rewards), "actor_sha256": state["policy"]["actor_sha256"],
                        "trace": {"path": str(path.relative_to(ROOT)), "sha256": digest(path), "bytes": path.stat().st_size}})
                    print(json.dumps({"case": name, "passed": True, "return": sum(rewards)}), flush=True)
    except Exception as error:
        failures.append(f"{type(error).__name__}: {error}")
    finally:
        for driver in (live, replay):
            if driver is not None:
                try:
                    driver.close()
                except Exception as error:
                    failures.append(f"Native close: {type(error).__name__}: {error}")
    receipt = {"schema": "pretrained_workshop_native_verification_v1", "passed": not failures and len(cases) == 12,
        "scope": "Implementation parity on twelve previously exposed conditions; no learning or new research claim",
        "sampler": "splitmix64-top53-categorical-v1", "sampler_seed": 1,
        "torch_seeded_trajectory_identity": False, "actor_artifact_sha256": digest(actor_path),
        "historical_engine_sha256": OLD_SHA,
        "native_session_binary_sha256": digest(ROOT / "build/native/droid-pretrained-session"),
        "verifier_sha256": digest(Path(__file__)), "cases": cases,
        "decisions_checked": len(cases) * 640, "native_controls_checked": len(cases) * 3200,
        "errors": failures,
        "max_probability_abs_error": max((item["max_probability_abs_error"] for item in cases), default=None)}
    with receipt_stream:
        json.dump(receipt, receipt_stream, indent=2, allow_nan=False)
        receipt_stream.write("\n")
    print(json.dumps({"passed": receipt["passed"], "cases": len(cases), "decisions": len(cases)*640,
                      "receipt_sha256": digest(destination)}), flush=True)
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
