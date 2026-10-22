#!/usr/bin/env python3
"""Native frozen-actor parity against CPU Torch and archived local telemetry.

No learner, adapter, world or training is constructed. Native parity operations
only encode observations, evaluate the frozen actor and sample supplied uniforms.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import queue
import subprocess
import sys
import threading

sys.dont_write_bytecode = True
import export_pretrained_actor as export

PROJECT = export.PROJECT
SCHEMA = "pretrained_actor_parity_request_v1"
ACTOR_SHA = "f163453766fe2ca7f7bab75a0093238bbcfce682aaa8bf11529ddc1673d0454c"
BYTES_SHA = "73db0a5e562fe052c81894844fac08bdc35676da356b8acefd7e3602f4bb5e2b"
STEPS = (0, 1, 2, 10, 11, 12, 63, 64, 127, 128, 319, 320, 639, 640)


class Audit:
    def __init__(self):
        self.checks = self.failed = 0
        self.errors = []

    def check(self, condition, message):
        self.checks += 1
        if not condition:
            self.failed += 1
            if len(self.errors) < 100:
                self.errors.append(message)
        return bool(condition)


class Native:
    def __init__(self, binary, artifact):
        self.process = subprocess.Popen([str(binary), "--artifact", str(artifact), "--parity"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, encoding="utf-8", cwd=PROJECT, bufsize=1)
        self.queue, self.errors, self.calls = queue.Queue(), deque(maxlen=20), 0
        def stdout():
            try:
                while True:
                    line = self.process.stdout.readline(2 * 1024 * 1024 + 1)
                    self.queue.put(line or None)
                    if not line:
                        return
            except Exception as error:
                self.queue.put(error)
        def stderr():
            for line in self.process.stderr:
                self.errors.append(line[:1000])
        threading.Thread(target=stdout, daemon=True).start()
        threading.Thread(target=stderr, daemon=True).start()

    def call(self, request):
        self.calls += 1
        self.process.stdin.write(json.dumps({"schema": SCHEMA, **request}, allow_nan=False, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        line = self.queue.get(timeout=30)
        if isinstance(line, Exception):
            raise line
        if not line or len(line) > 2 * 1024 * 1024 or not line.endswith("\n"):
            raise ValueError("Native parity CLI failed: " + "".join(self.errors))
        return json.loads(line)

    def close(self):
        try:
            self.process.stdin.close()
            self.process.wait(timeout=10)
            if self.process.returncode:
                raise ValueError("Native parity CLI exited unsuccessfully: " + "".join(self.errors))
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                self.process.wait(timeout=10)
            self.process.stdout.close()
            self.process.stderr.close()


def frame(obs, executed=0., previous=0., reward=0.):
    """Independent transcription of the frozen native_transfer local contract."""
    import numpy as np
    if obs["schema_version"] != "construction_observation_v2":
        raise ValueError("Unexpected native observation schema")
    sensors = {v["family_id"]: v for v in obs["reward_sensors"]}
    imu, light = sensors["imu_6axis_v0"], sensors["ambient_light_v0"]
    motor = obs["actuator_feedback"][0]["feedback"]
    force = imu["observations"]["specific_force_m_s2"] if imu["valid"] else [0.] * 3
    gyro = imu["observations"]["angular_velocity_rad_s"] if imu["valid"] else [0.] * 3
    lux = light["observations"]["illuminance_lux"] if light["valid"] else 0.
    values = [executed / .15, math.sin(motor["position_rad"]), math.cos(motor["position_rad"]),
              motor["velocity_rad_s"] / 10., motor["current_a"] / .5,
              *(x / 50. for x in force), *(x / 10. for x in gyro), lux / 1000., previous / .15,
              reward / .2, float(imu["valid"]), float(light["valid"]), 1.]
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.asarray(values, dtype=np.float32)
    if value.shape != (17,) or not np.isfinite(value).all():
        raise ValueError("Invalid local observation frame")
    return np.clip(value, -5., 5.)


def history(initial, transitions):
    import numpy as np
    frames = deque([np.zeros(17, np.float32) for _ in range(11)], maxlen=12)
    frames.append(frame(initial))
    for value in transitions:
        frames.append(frame(value["observation"], value["executed_effort"], value["previous_request"], value["reward"]))
    return np.concatenate(frames)


def observation(imu=True, light=True, *, force=(3., -5., 9.), gyro=(.4, -1.1, 2.4),
                lux=570., position=.7, velocity=-2., current=.08):
    return {"schema_version": "construction_observation_v2",
            "reward_sensors": [{"family_id": "imu_6axis_v0", "valid": imu,
                                "observations": {"specific_force_m_s2": list(force), "angular_velocity_rad_s": list(gyro)}},
                               {"family_id": "ambient_light_v0", "valid": light,
                                "observations": {"illuminance_lux": lux}}],
            "actuator_feedback": [{"feedback": {"position_rad": position, "velocity_rad_s": velocity, "current_a": current}}]}


def categorical(probabilities, uniform):
    # Native API first converts masses to float32, then normalizes/accumulates in
    # double. Equality with a CDF threshold advances to the next positive mass.
    import numpy as np
    values = np.asarray(probabilities, dtype=np.float32).astype(np.float64).tolist()
    if len(values) != 3 or not all(math.isfinite(x) and x >= 0 for x in values) or sum(values) <= 0:
        raise ValueError("Invalid categorical masses")
    if not math.isfinite(uniform) or not 0 <= uniform < 1:
        raise ValueError("Invalid uniform")
    total, cumulative = sum(values), 0.
    for i, value in enumerate(values):
        cumulative += value / total
        if value > 0 and uniform < cumulative:
            return i
    return max(i for i, value in enumerate(values) if value > 0)


def artifact_state(audit, artifact, state, np):
    audit.check(artifact["schema"] == "pretrained_actor_v1" and artifact["version"] == 1,
                "Unexpected actor schema")
    for key, value in (("dimensions", [204, 64, 64, 3]), ("dtype", "float32"), ("byte_order", "little"),
                       ("activation", "tanh"), ("frame_count", 12), ("frame_size", 17), ("input_clip", [-5., 5.]),
                       ("action_requests", [-.15, 0., .15]), ("decision_dt_s", .1), ("native_dt_s", .02)):
        audit.check(artifact[key] == value, f"Actor contract differs: {key}")
    audit.check([v["name"] for v in artifact["tensors"]] == list(export.ACTOR_NAMES), "Actor tensor names/order differ")
    digest, framed = hashlib.sha256(), hashlib.sha256()
    for tensor in artifact["tensors"]:
        name = tensor["name"]
        value = np.asarray(tensor["data"], dtype="<f4").reshape(tuple(tensor["shape"]))
        raw, expected = value.tobytes(), state[name].detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        audit.check(tuple(tensor["shape"]) == export.SHAPES[name] and np.isfinite(value).all(), f"Malformed actor tensor: {name}")
        audit.check(raw == expected.tobytes(), f"Exported actor tensor bytes differ: {name}")
        audit.check(hashlib.sha256(raw).hexdigest() == tensor["sha256"], f"Actor tensor digest differs: {name}")
        digest.update(raw)
        framed.update(name.encode() + b"\0" + str(value.shape).encode() + b"\0float32\0" + raw)
    audit.check(digest.hexdigest() == artifact["actor_bytes_sha256"] == BYTES_SHA, "Actor byte digest differs")
    audit.check(framed.hexdigest() == artifact["actor_tensor_sha256"] == ACTOR_SHA, "Actor framed digest differs")


def verify(audit, native, artifact, state, study, np, torch):
    reference_digest = hashlib.sha256()
    maxima = {"logit": 0., "probability": 0.}
    counts = {"actor_inputs": 0, "frames": 0, "histories": 0, "known_telemetry_inputs": 0,
              "categorical_thresholds": 0, "negative_cases": 0}
    refs, probabilities_for_thresholds = [], []
    def actor(vector, label, response=None):
        vector = np.asarray(vector, dtype=np.float32)
        with torch.no_grad():
            x = torch.from_numpy(vector.copy()).reshape(1, 204)
            x = torch.tanh(torch.nn.functional.linear(x, state["mlp_extractor.policy_net.0.weight"], state["mlp_extractor.policy_net.0.bias"]))
            x = torch.tanh(torch.nn.functional.linear(x, state["mlp_extractor.policy_net.2.weight"], state["mlp_extractor.policy_net.2.bias"]))
            logits = torch.nn.functional.linear(x, state["action_net.weight"], state["action_net.bias"])[0]
            probs = torch.distributions.Categorical(logits=logits).probs
        expected_logits, expected_probs = logits.numpy(), probs.numpy()
        if response is None:
            response = native.call({"input": vector.tolist(), "uniform": .375})
        got_logits, got_probs = np.asarray(response["logits"], np.float32), np.asarray(response["probabilities"], np.float32)
        audit.check(got_logits.shape == got_probs.shape == (3,) and np.isfinite(got_logits).all() and np.isfinite(got_probs).all(), f"{label}: malformed native actor output")
        le, pe = float(np.max(np.abs(got_logits - expected_logits))), float(np.max(np.abs(got_probs - expected_probs)))
        maxima["logit"], maxima["probability"] = max(maxima["logit"], le), max(maxima["probability"], pe)
        audit.check(le <= 1e-4, f"{label}: logit gate exceeded ({le})")
        audit.check(pe <= 1e-5, f"{label}: probability gate exceeded ({pe})")
        audit.check((got_probs >= 0).all() and abs(float(got_probs.sum(dtype=np.float64)) - 1.) <= 1e-6, f"{label}: invalid probability law")
        audit.check(response["action"] == categorical(got_probs, .375), f"{label}: inferred categorical choice differs")
        reference_digest.update(vector.astype("<f4", copy=False).tobytes() + expected_logits.astype("<f4").tobytes() + expected_probs.astype("<f4").tobytes())
        counts["actor_inputs"] += 1
        if len(probabilities_for_thresholds) < 12:
            probabilities_for_thresholds.append(expected_probs.tolist())

    actor(np.zeros(204, np.float32), "zero input")
    for i in range(204):
        for sign in (-1., 1.):
            value = np.zeros(204, np.float32)
            value[i] = sign
            actor(value, f"coordinate {i}/{sign}")
    rng = np.random.default_rng(20261006)
    for i, value in enumerate(rng.uniform(-5., 5., size=(64, 204)).astype(np.float32)):
        actor(value, f"seeded synthetic {i}")

    observations = [observation(imu, light) for imu in (False, True) for light in (False, True)]
    observations += [observation(force=(250., -250., 251.), gyro=(50., -50., 51.), lux=5000., velocity=51., current=-3.),
                     observation(position=-math.pi, velocity=-50., current=2.5, lux=-5001.)]
    for i, obs in enumerate(observations):
        request = {"op": "frame", "observation": obs, "executed_effort": -.11, "previous_request": .15, "reward": .12}
        expected = frame(obs, -.11, .15, .12)
        response = native.call(request)
        actual = np.asarray(response["frame"], np.float32)
        audit.check(actual.shape == (17,) and actual.astype("<f4").tobytes() == expected.astype("<f4").tobytes(), f"synthetic frame {i}: input bytes differ")
        counts["frames"] += 1
        transitions = [{"observation": item, "executed_effort": -.11, "previous_request": .15, "reward": .12}
                       for item in observations]
        response = native.call({"op": "history", "initial": obs, "transitions": transitions, "uniform": .375})
        expected = history(obs, transitions)
        actual = np.asarray(response["input"], np.float32)
        audit.check(actual.shape == (204,) and actual.astype("<f4").tobytes() == expected.astype("<f4").tobytes(), f"synthetic history {i}: bytes/order differ")
        audit.check(response["count"] == min(1 + len(transitions), 12), f"synthetic history {i}: frame count differs")
        audit.check(response["diagnostics"]["input_float32_le_sha256"] == hashlib.sha256(expected.astype("<f4").tobytes()).hexdigest(),
                    f"synthetic history {i}: diagnostic input digest differs")
        counts["histories"] += 1
        actor(expected, f"synthetic history {i}", response)

    # All twelve known pooled seed101 cases; these are existing development
    # records. No reserved challenge files or world constructors are accessed.
    protocol = export.read(study / "protocol.json")
    stage = export.read(study / "development/stage.json")
    for index, case_id in enumerate(protocol["study"]["case_ids"]["development"]):
        trial_id = f"development--{case_id}--pooled--101"
        result_path = study / "development/trials" / trial_id / "result.json"
        relative_name = result_path.relative_to(study).as_posix()
        matching = [r for r in stage["trial_results"] if r["path"] == relative_name]
        export.require(len(matching) == 1 and matching[0]["sha256"] == export.sha(result_path), "Changed known telemetry trial result")
        value = export.read(result_path)
        export.require(value["complete"] and value["safe"] and value["controller"] == "pooled" and value["seed"] == 101
                       and value["partition"] == "development" and value["case_index"] == index,
                       "Unexpected known-case reference")
        telemetry_path = export.relative(result_path.parent / "traces", value["telemetry"]["path"])
        export.require(export.sha(telemetry_path) == value["telemetry"]["sha256"], "Changed known telemetry")
        with gzip.open(telemetry_path, "rt", encoding="utf-8") as stream:
            telemetry = json.load(stream)
        rows = telemetry["rows"]
        export.require(len(rows) == 640, "Incomplete known-case history")
        transitions = []
        for row in rows:
            data = row["native"]
            transitions.append({"observation": {"schema_version": "construction_observation_v2", **data["sampled_local"]},
                                "executed_effort": data["executed_effort"], "previous_request": data["previous_request"], "reward": data["reward"]})
        initial = value["initial_local_observation"]
        for step in STEPS:
            # A complete window needs at most twelve transitions. Encoding the
            # dropped earlier frames cannot change the frozen policy input.
            selected = transitions[max(0, step - 12):step]
            expected = history(initial, transitions[:step])
            response = native.call({"op": "history", "initial": initial, "transitions": selected, "uniform": .375})
            actual = np.asarray(response["input"], np.float32)
            audit.check(actual.shape == (204,) and actual.astype("<f4").tobytes() == expected.astype("<f4").tobytes(), f"{case_id}/{step}: real history input bytes differ")
            audit.check(response["count"] == min(1 + len(selected), 12), f"{case_id}/{step}: frame count differs")
            audit.check(response["diagnostics"]["input_float32_le_sha256"] == hashlib.sha256(expected.astype("<f4").tobytes()).hexdigest(),
                        f"{case_id}/{step}: diagnostic input digest differs")
            actor(expected, f"{case_id}/{step}", response)
            counts["histories"] += 1
            counts["known_telemetry_inputs"] += 1
        refs.append({"case_id": case_id, "source_result": export.ref(result_path), "telemetry": export.ref(telemetry_path),
                     "sampled_decisions": list(STEPS)})

    probabilities_for_thresholds += [[0., 1., 0.], [0., 0., 1.], [1., 0., 0.], [.1, .2, .7], [1., 1., 1.]]
    for i, probs in enumerate(probabilities_for_thresholds):
        masses = np.asarray(probs, np.float32).astype(np.float64)
        boundaries = np.cumsum(masses / masses.sum())[:-1]
        uniforms = [0., .25, .5, .75, math.nextafter(1., 0.)]
        for boundary in boundaries:
            if 0 <= boundary < 1:
                uniforms += [float(boundary)]
                if boundary > 0:
                    uniforms += [math.nextafter(float(boundary), 0.)]
                uniforms += [math.nextafter(float(boundary), 1.)]
        for uniform in uniforms:
            if not 0 <= uniform < 1:
                continue
            response = native.call({"op": "categorical", "probabilities": probs, "uniform": uniform})
            audit.check(response["action"] == categorical(probs, uniform), f"categorical vector{i}/uniform{uniform}: threshold differs")
            counts["categorical_thresholds"] += 1

    invalid = [{"input": [0.] * 203, "uniform": .5}, {"input": [0.] * 204, "uniform": 1.},
               {"input": [0.] * 204, "uniform": -.1},
               {"op": "categorical", "probabilities": [0., 0., 0.], "uniform": .5},
               {"op": "categorical", "probabilities": [-1., 1., 1.], "uniform": .5},
               {"op": "frame", "observation": observation(force=(1e300, 0., 0.)), "executed_effort": 0., "previous_request": 0., "reward": 0.}]
    for i, request in enumerate(invalid):
        response = native.call(request)
        audit.check(isinstance(response, dict) and "error" in response, f"negative case{i}: malformed request accepted")
        counts["negative_cases"] += 1
    return {"counts": counts, "max_abs_logit_error": maxima["logit"], "max_abs_probability_error": maxima["probability"],
            "reference_vector_stream_sha256": reference_digest.hexdigest(), "known_telemetry_references": refs,
            "reserved_telemetry_read": False, "native_world_calls": 0, "training_decisions": 0,
            "declared_gates": {"input_frame_bytes_exact": True, "max_abs_logit_error": 1e-4, "max_abs_probability_error": 1e-5,
                      "categorical_thresholds_exact": True},
            "scope": "Float32 frames/history, frozen actor forward/probability outputs and supplied-uniform categorical semantics. This does not regenerate Torch's seeded sampled trajectories or evaluate new physical behavior."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--native", type=Path, default=PROJECT / "build/native/droid-pretrained-actor")
    parser.add_argument("--study", type=Path, default=PROJECT / export.STUDY)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    export.runtime()
    artifact_path, receipt = args.artifact.resolve(), args.receipt.resolve()
    export.require(receipt.is_relative_to(PROJECT) and artifact_path.is_relative_to(PROJECT), "Outputs/actor must stay within project")
    export.require(receipt.parent.is_dir() and not receipt.exists(), "Receipt needs an existing parent and new path")
    audit, engine = Audit(), None
    with receipt.open("x", encoding="utf-8") as stream:
        report = {"schema": "pretrained_actor_reference_verification_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                  "artifact": export.ref(artifact_path), "verifier_source": export.ref(Path(__file__)),
                  "exporter_source": export.ref(Path(export.__file__)), "native_binary": export.ref(args.native.resolve()),
                  "weight_updates": 0, "new_reserved_world_calls": 0}
        try:
            import numpy as np
            import torch
            from stable_baselines3.common.save_util import load_from_zip_file
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            state, provenance = export.trusted_checkpoint(args.study.resolve(), torch, load_from_zip_file)
            artifact = export.read(artifact_path)
            audit.check(artifact["provenance"] == provenance, "Artifact provenance differs from sealed canonical source")
            artifact_state(audit, artifact, state, np)
            if audit.failed:
                raise ValueError("Native parity refused after actor/provenance failures")
            engine = Native(args.native.resolve(), artifact_path)
            report.update(verify(audit, engine, artifact, state, args.study.resolve(), np, torch))
            report["native_parity_calls"] = engine.calls
        except Exception as error:
            audit.check(False, f"{type(error).__name__}: {error}")
        finally:
            if engine is not None:
                report["native_parity_calls"] = engine.calls
                try:
                    engine.close()
                except Exception as error:
                    audit.check(False, f"Native close: {error}")
        report.update({"passed": audit.failed == 0, "checks": audit.checks, "failed_checks": audit.failed,
                       "errors": audit.errors, "errors_truncated": audit.failed > len(audit.errors)})
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"receipt": export.ref(receipt), "passed": audit.failed == 0, "failed_checks": audit.failed}, sort_keys=True))
    return 0 if audit.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
