"""Development-only pooled PPO pilot, executed in the managed Linux container.

The curriculum owns all cases. Reserved cases are manifest entries only: this
runner has no command that instantiates them. Native requests are retained for
later replay; full physical transition streams are deliberately not collected.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import sys
import time

from generalization_curriculum import (
    DeterministicSourceSampler, manifest, validate_manifest,
)

PROJECT = Path(__file__).resolve().parents[1]
NATIVE_COMMAND = ["/workspace/build/native/droid-transfer-env"]
EARLY_DECISIONS = (64, 128, 320, 640)
SOURCE_WALL_CAP_S = 600
TRIAL_WALL_CAP_S = 180
NEW_SOURCE_FILES = (
    "training/generalization_pretraining.py", "training/generalization_curriculum.py",
    "training/native_transfer.py",
    "training/setup-generalization.sh", "training/requirements-generalization.txt",
    "tests/check_generalization_curriculum.py", "tests/check_generalization_adapter.py",
    "docs/GENERALIZATION_PRETRAINING_EXPERIMENT.md",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def copy_json(value):
    return json.loads(canonical(value))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_gzip_new(path, value):
    with Path(path).open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write(canonical(value).encode("utf-8"))


def require_container():
    if os.name == "nt" or sys.platform != "linux" or not Path("/.dockerenv").is_file():
        raise RuntimeError("Run this pilot only inside the documented managed Linux development container")
    if PROJECT != Path("/workspace") or not Path(NATIVE_COMMAND[0]).is_file():
        raise RuntimeError("Expected the documented /workspace mount and native transfer executable")


def runtime_record():
    require_container()
    versions = {name: metadata.version(name) for name in
                ("torch", "stable-baselines3", "gymnasium", "numpy", "cloudpickle")}
    required = {"torch": "2.8.0+cpu", "stable-baselines3": "2.9.0", "gymnasium": "1.2.3"}
    for name, expected in required.items():
        if versions[name] != expected:
            raise RuntimeError(f"Expected {name}=={expected}, found {versions[name]}")
    return {"schema": "generalization_runtime_v1", "python": sys.version,
            "python_executable": sys.executable, "platform": platform.platform(),
            "machine": platform.machine(), "packages": versions,
            "native_command": NATIVE_COMMAND, "native_binary_sha256": sha256_file(NATIVE_COMMAND[0]),
            "cpu_only": True, "torch_threads": 1, "torch_interop_threads": 1,
            "torch_deterministic_algorithms": True,
            "os_release": Path("/etc/os-release").read_text(encoding="utf-8")}


def source_hashes():
    files = {PROJECT / name for name in NEW_SOURCE_FILES}
    files.update((PROJECT / "src").glob("*.cpp"))
    files.update((PROJECT / "include" / "droid").glob("*.hpp"))
    files.update(PROJECT / name for name in
                 ("CMakeLists.txt", "training/native_transfer.py", "config/module_catalog.json"))
    return {path.relative_to(PROJECT).as_posix(): sha256_file(path) for path in sorted(files)}


def lock(output):
    """Write prospective settings and byte provenance without constructing worlds."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing output: {output}")
    curriculum = manifest()
    validate_manifest(curriculum)
    runtime = runtime_record()
    hashes = source_hashes()
    protocol = {"schema": "generalization_pretraining_pilot_v1",
                "status": "locked_before_learning", "locked_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Development-only pooled PPO. No held-out execution, tuning or promotion.",
                "curriculum": curriculum, "runtime": runtime, "source_sha256": hashes,
                "wall_time_caps_s": {"source": SOURCE_WALL_CAP_S, "development_trial": TRIAL_WALL_CAP_S},
                "development_action_seed": "10000 + pilot_seed*10 + development_case_index",
                "development_policy_sampling": "stochastic categorical, deterministic=False",
                "untrained_control": "Seed404 initial actor+critic saved before source learning; frozen, same per-case action seed",
                "development_success_gate": {"all_cases": 12, "strict_gain_over_quiet": 3.2,
                    "requires": "Both paired arms complete 640 decisions safely; frozen weights unchanged"},
                "initialization": "Fresh optimizer and 11 zero absent history frames at every actual reset",
                "recording": "Gzipped requests and response stream digest; compact development telemetry; no physical snapshots",
                "reserved_execution": "Not implemented or authorized in this development pilot"}
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "source_snapshot"
    archive.mkdir()
    # Copy individual compact source files, never earlier report bundles or tapes.
    for name in hashes:
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((PROJECT / name).read_bytes())
    write_new(output / "protocol.json", protocol)
    write_new(output / "lock.json", {"schema": "generalization_lock_v1",
              "protocol_sha256": sha256_file(output / "protocol.json"),
              "source_sha256": hashes, "runtime": runtime,
              "reserved_worlds_constructed": 0, "physical_interactions": 0})
    print(canonical({"locked": str(output), "protocol_sha256": sha256_file(output / "protocol.json")}), flush=True)


def read_lock(output):
    output = Path(output).resolve()
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "lock.json").read_text(encoding="utf-8"))
    if protocol.get("schema") != "generalization_pretraining_pilot_v1":
        raise ValueError("Unexpected prospective protocol schema")
    if receipt["protocol_sha256"] != sha256_file(output / "protocol.json"):
        raise ValueError("Prospective protocol has changed since its lock")
    validate_manifest(protocol["curriculum"])
    if canonical(protocol["curriculum"]) != canonical(manifest()):
        raise ValueError("Locked curriculum differs from the current implementation")
    if source_hashes() != protocol["source_sha256"] or receipt["source_sha256"] != protocol["source_sha256"]:
        raise ValueError("Source bytes have changed since protocol lock; preserve this lock and create a new attempt")
    for name in protocol["source_sha256"]:
        if sha256_file(output / "source_snapshot" / name) != protocol["source_sha256"][name]:
            raise ValueError(f"Archived source changed: {name}")
    runtime = runtime_record()
    if canonical(runtime) != canonical(protocol["runtime"]):
        raise ValueError("Runtime differs from the locked runtime")
    return output, protocol, receipt["protocol_sha256"]


class WallTimeCap(RuntimeError):
    pass


@contextmanager
def wall_deadline(seconds):
    """Interrupt stalled native IPC too, retaining any uncertain request prefix."""
    def timeout(signum, frame):
        raise WallTimeCap("Predeclared wall-time cap")
    previous_handler = signal.signal(signal.SIGALRM, timeout)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] or previous_timer[1]:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def learning_dependencies():
    require_container()
    import torch
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from native_transfer import NativeBridge, NativeTransferVecEnv, SafetyStop
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    return torch, np, PPO, BaseCallback, NativeBridge, NativeTransferVecEnv, SafetyStop


def parameter_digest(policy):
    result = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(name.encode() + b"\0")
        result.update(str(array.shape).encode() + b"\0" + str(array.dtype).encode() + b"\0")
        result.update(array.tobytes())
    return result.hexdigest()


def metrics_from_results(rows):
    result = {"decisions": len(rows), "native_steps": 0, "duration_s": 0., "native_reward": 0.,
              "light_reward": 0., "imu_reward": 0., "energy_j": 0., "safety_stops": 0,
              "max_current_a": 0., "max_speed_rad_s": 0., "max_temperature_c": 0.,
              "sampled_invalid_imu": 0, "sampled_invalid_light": 0, "sampled_light_lux_sum": 0.,
              "sampled_valid_light_count": 0, "sampled_max_light_lux": 0., "sampled_feedback_flag_count": 0}
    for row in rows:
        value = row["native"]
        result["native_steps"] += value["native_steps"]
        result["duration_s"] += value["elapsed_s"]
        for name in ("light_reward", "imu_reward"):
            result[name] += value[name]
        result["native_reward"] += value["reward"]
        result["energy_j"] = value["energy_j"]
        result["safety_stops"] += int(value["terminated"])
        for name in ("max_current_a", "max_speed_rad_s", "max_temperature_c"):
            result[name] = max(result[name], value[name])
        sensors = {entry["family_id"]: entry for entry in value["sampled_local"]["reward_sensors"]}
        result["sampled_invalid_imu"] += int(not sensors["imu_6axis_v0"]["valid"])
        light = sensors["ambient_light_v0"]
        result["sampled_invalid_light"] += int(not light["valid"])
        if light["valid"]:
            lux = light["observations"]["illuminance_lux"]
            result["sampled_light_lux_sum"] += lux
            result["sampled_valid_light_count"] += 1
            result["sampled_max_light_lux"] = max(result["sampled_max_light_lux"], lux)
        flags = value["sampled_local"]["actuator_feedback"][0]["feedback"]["fault_flags"]
        result["sampled_feedback_flag_count"] += len(flags)
    result["sampled_mean_light_lux"] = (result["sampled_light_lux_sum"] / result["sampled_valid_light_count"]
                                        if result["sampled_valid_light_count"] else None)
    return result


def compact_native(result):
    names = ("reward", "light_reward", "imu_reward", "previous_request", "executed_effort",
             "terminated", "elapsed_s", "native_steps", "time_s", "energy_j",
             "max_current_a", "max_speed_rad_s", "max_temperature_c", "safety")
    value = {name: copy_json(result[name]) for name in names}
    value["sampled_local"] = {name: copy_json(result["observation"][name])
                              for name in ("reward_sensors", "actuator_feedback")}
    return value


def build_model(env, ppo, seed, *, target=False, dependencies):
    torch, _, PPO, *_ = dependencies
    return PPO("MlpPolicy", env, device="cpu", seed=seed, verbose=0,
               n_steps=ppo["target_n_steps"] if target else ppo["source_n_steps"],
               batch_size=ppo["batch_size"], n_epochs=ppo["n_epochs"],
               learning_rate=ppo["learning_rate"], gamma=ppo["gamma"], gae_lambda=ppo["gae_lambda"],
               clip_range=ppo["clip_range"], clip_range_vf=None, ent_coef=ppo["ent_coef"],
               vf_coef=ppo["vf_coef"], max_grad_norm=ppo["max_grad_norm"], target_kl=None,
               normalize_advantage=True,
               policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]},
                              "activation_fn": torch.nn.Tanh, "ortho_init": True,
                              "optimizer_kwargs": {"eps": ppo["optimizer_eps"]}})


def make_instrumented_env_class(base):
    class InstrumentedEnv(base):
        def __init__(self, *args, source_cases=None, **kwargs):
            self.telemetry = []
            self.source_indices = ({canonical(case): i for i, case in enumerate(source_cases)}
                                   if source_cases is not None else None)
            self.source_exposure = [{"case_index": i, "decisions": 0, "native_steps": 0,
                                     "duration_s": 0., "native_reward": 0., "safety_stops": 0}
                                    for i in range(len(source_cases or []))]
            self.unresolved_request = None
            super().__init__(*args, **kwargs)
            original_call = self.bridge.call
            def tracked_call(request):
                try:
                    return original_call(request)
                except Exception:
                    # Successful responses, including native errors, are in the
                    # retained tape. A missing response leaves advancement uncertain.
                    if self.bridge.last_error_response is None:
                        self.unresolved_request = copy_json(request)
                    raise
            self.bridge.call = tracked_call

        def step_wait(self):
            before = self.total_interactions
            actions = list(self.pending_actions)
            cases = list(self.cases)
            try:
                return super().step_wait()
            finally:
                if self.total_interactions != before:
                    for slot, result in enumerate(self.latest):
                        if self.source_indices is None:
                            self.telemetry.append({"decision": before + slot + 1,
                                                   "action": actions[slot], "native": compact_native(result)})
                        else:
                            index = self.source_indices[canonical(cases[slot])]
                            exposure = self.source_exposure[index]
                            exposure["decisions"] += 1
                            exposure["native_steps"] += result["native_steps"]
                            exposure["duration_s"] += result["elapsed_s"]
                            exposure["native_reward"] += result["reward"]
                            exposure["safety_stops"] += int(result["terminated"])
    return InstrumentedEnv


def tape_receipt(env, path):
    save_gzip_new(path, {"schema": "generalization_native_request_tape_v1",
                  "record": False, "requests": env.bridge.request_tape,
                  "native_stream_sha256": env.bridge.stream_digest.hexdigest(),
                  "native_failure": env.bridge.last_error_response,
                  "unresolved_request": env.unresolved_request,
                  "replay_note": "Same requests reproduce native responses; enable record for later selected physical replay"})
    return {"path": str(path.name), "sha256": sha256_file(path),
            "native_stream_sha256": env.bridge.stream_digest.hexdigest(),
            "request_count": len(env.bridge.request_tape)}


def close_env(env):
    if env is None:
        return None
    try:
        env.close()
        return None
    except Exception as error:
        return f"{type(error).__name__}: {error}"


def scalar_metrics(model, np):
    return {key: float(value) if math.isfinite(float(value)) else None
            for key, value in model.logger.name_to_value.items()
            if key.startswith("train/") and np.isscalar(value)}


def train_source(root, protocol, protocol_hash, dependencies):
    torch, np, _, BaseCallback, _, base, _ = dependencies
    curriculum = protocol["curriculum"]
    pilot = curriculum["pilot"]
    seed = pilot["seed"]
    folder = root / f"pilot-{seed}"
    folder.mkdir(exist_ok=False)
    (folder / "traces").mkdir()
    write_new(folder / "started.json", {"protocol_sha256": protocol_hash, "seed": seed,
              "source_only": True, "reserved_worlds_constructed": 0})
    updates = (folder / "learning.jsonl").open("x", encoding="utf-8")
    start = time.monotonic()
    class LearningLog(BaseCallback):
        def _on_step(self):
            return True
        def _on_rollout_end(self):
            updates.write(canonical({"event": "rollout_collected", "decisions": self.num_timesteps,
                           "wall_s": time.monotonic() - start, "preceding_epochs": self.model._n_updates}) + "\n")
            updates.flush()
        def _on_rollout_start(self):
            if self.model._n_updates:
                updates.write(canonical({"event": "update_completed", "decisions": self.num_timesteps,
                               "optimization_epochs": self.model._n_updates,
                               "metrics": scalar_metrics(self.model, np)}) + "\n")
                updates.flush()
        def _on_training_end(self):
            updates.write(canonical({"event": "update_completed", "decisions": self.num_timesteps,
                           "optimization_epochs": self.model._n_updates,
                           "metrics": scalar_metrics(self.model, np), "final": True}) + "\n")
            updates.flush()
    sampler = DeterministicSourceSampler(curriculum, seed)
    env = model = None
    initial_checkpoint = None
    initialization_wall_s = learning_wall_s = 0.
    learning_start = None
    error = None
    stopped_by_clock = False
    try:
        with wall_deadline(protocol["wall_time_caps_s"]["source"]):
            env_type = make_instrumented_env_class(base)
            env = env_type(curriculum["source_cases"][:pilot["n_envs"]], case_sampler=sampler,
                           source_cases=curriculum["source_cases"], horizon=pilot["horizon_decisions"],
                           record=False, command=NATIVE_COMMAND)
            model = build_model(env, curriculum["ppo"], seed, dependencies=dependencies)
            if model.policy.optimizer.state:
                raise RuntimeError("Source optimizer is not fresh")
            model.save(folder / "policy_initial.zip")
            initial_checkpoint = {"policy_path": "policy_initial.zip",
                                  "policy_sha256": sha256_file(folder / "policy_initial.zip"),
                                  "parameter_sha256": parameter_digest(model.policy),
                                  "weight_updates": 0, "source_decisions": 0}
            write_new(folder / "initial-checkpoint.json", initial_checkpoint)
            initialization_wall_s = time.monotonic() - start
            learning_start = time.monotonic()
            model.learn(total_timesteps=pilot["source_decisions"], callback=LearningLog())
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
        stopped_by_clock = isinstance(failure, WallTimeCap)
    finally:
        if learning_start is not None:
            learning_wall_s = time.monotonic() - learning_start
        close_error = close_env(env)
        updates.close()
    parameters_finite = model is not None and all(bool(torch.isfinite(p).all()) for p in model.policy.parameters())
    actual = env.total_interactions if env else 0
    source_wall_s = time.monotonic() - start
    stopped_by_clock = stopped_by_clock or source_wall_s > protocol["wall_time_caps_s"]["source"]
    exact = (env is not None and env.bridge.last_error_response is None and env.unresolved_request is None
             and sum(item["decisions"] for item in env.source_exposure) == actual
             and sum(item["native_steps"] for item in env.source_exposure) == env.total_native_steps)
    optimizer_steps = ([int(state["step"].item()) for state in model.policy.optimizer.state.values() if "step" in state]
                       if model is not None else [])
    expected_epochs = pilot["source_decisions"] // (pilot["n_envs"] * pilot["n_steps"]) * curriculum["ppo"]["n_epochs"]
    expected_optimizer_steps = pilot["source_decisions"] // curriculum["ppo"]["batch_size"] * curriculum["ppo"]["n_epochs"]
    update_counts_exact = (model is not None and model._n_updates == expected_epochs
                          and min(optimizer_steps, default=0) == max(optimizer_steps, default=0) == expected_optimizer_steps)
    complete = (error is None and close_error is None and actual == pilot["source_decisions"]
                and model.num_timesteps == actual and parameters_finite and exact
                and not stopped_by_clock and update_counts_exact and initial_checkpoint is not None)
    result = {"schema": "generalization_source_pilot_result_v1", "seed": seed, "protocol_sha256": protocol_hash,
              "requested_decisions": pilot["source_decisions"], "actual_decisions": actual,
              "native_steps": env.total_native_steps if env else 0,
              "complete": complete, "error": error, "close_error": close_error,
              "stopped_by_clock": stopped_by_clock, "initialization_wall_s": initialization_wall_s,
              "learning_wall_s": learning_wall_s, "source_wall_s": source_wall_s,
              "initial_checkpoint": initial_checkpoint,
              "parameters_finite": parameters_finite, "exposure_exact": exact,
              "native_failure": env.bridge.last_error_response if env else None,
              "unresolved_request": env.unresolved_request if env else None,
              "safety_stops": env.safety_stops if env else 0, "episodes": env.episode_count if env else 0,
              "source_schedule": sampler.exposure_report(),
              "per_source_case_exposure": env.source_exposure if env else [],
              "reserved_worlds_constructed": 0, "checkpoint_status": "development_only"}
    if env is not None:
        result["request_tape"] = tape_receipt(env, folder / "traces" / "source-requests.json.gz")
    if model is not None:
        policy_name = "policy.zip" if complete else "policy_partial.zip"
        model.save(folder / policy_name)
        result.update({"policy_path": policy_name, "policy_sha256": sha256_file(folder / policy_name),
                       "parameter_sha256": parameter_digest(model.policy),
                       "parameters": sum(p.numel() for p in model.policy.parameters()),
                       "optimization_epochs": model._n_updates,
                       "rollout_updates": model._n_updates / curriculum["ppo"]["n_epochs"],
                       "optimizer_steps_min": min(optimizer_steps, default=0),
                       "optimizer_steps_max": max(optimizer_steps, default=0),
                       "final_update_metrics": scalar_metrics(model, np)})
        result["expected_optimization_epochs"] = expected_epochs
        result["expected_optimizer_steps"] = expected_optimizer_steps
        result["update_counts_exact"] = update_counts_exact
    write_new(folder / "result.json", result)
    print(canonical({"source_pilot": result}), flush=True)
    return folder, result


def development_trial(folder, protocol, protocol_hash, source_result, index, arm, dependencies):
    torch, np, PPO, _, _, base, SafetyStop = dependencies
    curriculum = protocol["curriculum"]
    pilot = curriculum["pilot"]
    case_id = curriculum["case_ids"]["development"][index]
    trial_id = f"{case_id}--{arm}"
    trial_folder = folder / "trials" / trial_id
    trial_folder.mkdir(parents=True, exist_ok=False)
    write_new(trial_folder / "started.json", {"protocol_sha256": protocol_hash, "trial_id": trial_id,
              "case_index": index, "arm": arm, "held_out": False, "reserved_worlds_constructed": 0})
    env = model = None
    initial_hash = final_hash = None
    error = None
    stopped_by_clock = False
    action_seed = 10000 + pilot["seed"] * 10 + index
    start = time.monotonic()
    try:
        with wall_deadline(protocol["wall_time_caps_s"]["development_trial"]):
            env_type = make_instrumented_env_class(base)
            env = env_type([copy_json(curriculum["development_cases"][index])],
                           # A manual finite rollout avoids the adapter's extra reset at horizon.
                           horizon=pilot["development_decisions_per_arm"] + 1,
                           record=False, stop_on_safety=True, command=NATIVE_COMMAND)
            if arm in ("frozen", "untrained"):
                model = build_model(env, curriculum["ppo"], pilot["seed"], target=True, dependencies=dependencies)
                checkpoint = source_result if arm == "frozen" else source_result["initial_checkpoint"]
                if sha256_file(folder / checkpoint["policy_path"]) != checkpoint["policy_sha256"]:
                    raise RuntimeError("Frozen checkpoint bytes changed before development")
                source = PPO.load(folder / checkpoint["policy_path"], device="cpu")
                model.policy.load_state_dict(source.policy.state_dict(), strict=True)
                del source
                model.set_random_seed(action_seed)
                if model.policy.optimizer.state:
                    raise RuntimeError("Frozen development optimizer is not fresh")
                initial_hash = parameter_digest(model.policy)
                if initial_hash != checkpoint["parameter_sha256"]:
                    raise RuntimeError("Frozen development weights differ from the fixed source checkpoint")
            obs = env.reset()
            for _ in range(pilot["development_decisions_per_arm"]):
                action = model.predict(obs, deterministic=False)[0] if model else np.asarray([1], dtype=np.int64)
                obs, _, _, _ = env.step(action)
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
        stopped_by_clock = isinstance(failure, WallTimeCap)
    finally:
        close_error = close_env(env)
    rows = env.telemetry if env else []
    metrics = metrics_from_results(rows)
    exact = env is not None and env.bridge.last_error_response is None and env.unresolved_request is None
    trial_wall_s = time.monotonic() - start
    stopped_by_clock = stopped_by_clock or trial_wall_s > protocol["wall_time_caps_s"]["development_trial"]
    if model is not None:
        final_hash = parameter_digest(model.policy)
    expected_hash = (source_result["parameter_sha256"] if arm == "frozen" else
                     source_result["initial_checkpoint"]["parameter_sha256"] if arm == "untrained" else None)
    frozen_unchanged = arm == "quiet" or (initial_hash == final_hash == expected_hash)
    complete = (error is None and close_error is None and exact and frozen_unchanged and not stopped_by_clock
                and len(rows) == pilot["development_decisions_per_arm"]
                and metrics["native_steps"] == 3200 and metrics["safety_stops"] == 0)
    result = {"schema": "generalization_development_trial_v1", "trial_id": trial_id, "case_id": case_id,
              "case_index": index, "arm": arm, "protocol_sha256": protocol_hash,
              "held_out": False, "weight_updates": 0, "complete": complete,
              "safe": exact and metrics["safety_stops"] == 0 and error is None and close_error is None,
              "metrics": metrics, "error": error, "close_error": close_error,
              "stopped_by_clock": stopped_by_clock, "wall_s": trial_wall_s,
              "exposure_exact": exact, "native_failure": env.bridge.last_error_response if env else None,
              "unresolved_request": env.unresolved_request if env else None,
              "action_seed": action_seed if model else None,
              "initial_parameter_sha256": initial_hash, "final_parameter_sha256": final_hash,
              "frozen_weights_unchanged": frozen_unchanged,
              "windows": [{"requested_decisions": count, "complete": len(rows) >= count,
                           "metrics": metrics_from_results(rows[:count])} for count in EARLY_DECISIONS],
              "reserved_worlds_constructed": 0}
    if env is not None:
        tape_path = folder / "traces" / f"{trial_id}-requests.json.gz"
        result["request_tape"] = tape_receipt(env, tape_path)
        telemetry_path = folder / "traces" / f"{trial_id}-telemetry.json.gz"
        save_gzip_new(telemetry_path, {"schema": "generalization_development_telemetry_v1", "rows": rows})
        result["telemetry"] = {"path": telemetry_path.name, "sha256": sha256_file(telemetry_path)}
    write_new(trial_folder / "result.json", result)
    print(canonical({"development_trial": result}), flush=True)
    return result


def pilot(output):
    root, protocol, protocol_hash = read_lock(output)
    dependencies = learning_dependencies()
    folder, source_result = train_source(root, protocol, protocol_hash, dependencies)
    trials = []
    if source_result["complete"]:
        for index in range(len(protocol["curriculum"]["development_cases"])):
            for arm in ("quiet", "untrained", "frozen"):
                trials.append(development_trial(folder, protocol, protocol_hash, source_result,
                                                index, arm, dependencies))
    comparisons = []
    for index, case_id in enumerate(protocol["curriculum"]["case_ids"]["development"]):
        pair = {trial["arm"]: trial for trial in trials if trial["case_index"] == index}
        eligible = ("frozen" in pair and "quiet" in pair
                    and all(pair[arm]["complete"] and pair[arm]["safe"] for arm in ("frozen", "quiet")))
        gain = (pair["frozen"]["metrics"]["native_reward"] - pair["quiet"]["metrics"]["native_reward"]
                if "frozen" in pair and "quiet" in pair else None)
        learning_pair = ("frozen" in pair and "untrained" in pair
                         and all(pair[arm]["complete"] and pair[arm]["safe"] for arm in ("frozen", "untrained")))
        learning_gain = (pair["frozen"]["metrics"]["native_reward"] - pair["untrained"]["metrics"]["native_reward"]
                         if "frozen" in pair and "untrained" in pair else None)
        comparisons.append({"case_id": case_id, "complete_safe_pair": eligible,
                            "frozen_minus_quiet": gain, "passes": eligible and gain > 3.2,
                            "complete_safe_learning_pair": learning_pair, "trained_minus_untrained": learning_gain,
                            "trained_beats_untrained": learning_pair and learning_gain > 0.})
    learning_gains = [item["trained_minus_untrained"] for item in comparisons if item["complete_safe_learning_pair"]]
    realized_learning_gains = [item["trained_minus_untrained"] for item in comparisons
                               if item["trained_minus_untrained"] is not None]
    summary = {"schema": "generalization_pilot_summary_v1", "protocol_sha256": protocol_hash,
               "source": source_result, "development_trials": trials, "development_comparisons": comparisons,
               "development_competence": len(comparisons) == 12 and all(item["passes"] for item in comparisons),
               "complete": source_result["complete"] and len(trials) == 36 and all(trial["complete"] for trial in trials),
               "offline_learning_diagnostic": {"eligible_pairs": len(learning_gains),
                    "realized_pairs": len(realized_learning_gains),
                    "mean_trained_minus_untrained": (sum(realized_learning_gains) / len(realized_learning_gains)
                                                      if len(realized_learning_gains) == 12 else None),
                    "mean_complete_safe_pair_trained_minus_untrained": (sum(learning_gains) / len(learning_gains)
                                                                        if learning_gains else None),
                    "all_case_mean_has_equal_complete_safe_exposure": len(learning_gains) == 12,
                    "all_case_mean_interpretation": "Descriptive realized raw returns across all12 cases; stopped prefixes have unequal exposure and this mean cannot establish competence",
                    "trained_wins": sum(item["trained_beats_untrained"] for item in comparisons),
                    "has_prospective_success_threshold": False},
               "reserved_worlds_constructed": 0, "held_out_trials": 0, "unseen_body_claim": False,
               "promotion": False, "scope": "Single-seed development pilot; no tuning or held-out evaluation",
               "source_sha256_after_collection": source_hashes(), "runtime_after_collection": runtime_record()}
    summary["provenance_unchanged"] = (summary["source_sha256_after_collection"] == protocol["source_sha256"]
                                        and canonical(summary["runtime_after_collection"]) == canonical(protocol["runtime"]))
    if not summary["provenance_unchanged"]:
        summary["development_competence"] = False
        summary["complete"] = False
    write_new(folder / "summary.json", summary)
    print(canonical({"pilot_summary": str(folder / "summary.json"),
                     "complete": summary["complete"], "development_competence": summary["development_competence"],
                     "reserved_worlds_constructed": 0, "held_out_trials": 0}), flush=True)
    return 0 if source_result["complete"] else 1


def summary(output):
    require_container()
    root = Path(output).resolve()
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    folder = root / f"pilot-{protocol['curriculum']['pilot']['seed']}"
    report = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if report["protocol_sha256"] != sha256_file(root / "protocol.json"):
        raise ValueError("Summary/protocol binding differs")
    print(json.dumps(report, indent=2, allow_nan=False))


def native_preflight():
    """Probe only a known source case; no training or reserved construction."""
    dependencies = learning_dependencies()
    _, _, _, _, NativeBridge, _, _ = dependencies
    bridge = NativeBridge(NATIVE_COMMAND)
    try:
        spec = bridge.spec
        if spec["native_steps_per_decision"] != 5 or spec["requests"] != [-.15, 0., .15]:
            raise ValueError("Unexpected native control boundary")
        case = manifest()["source_cases"][0]
        bridge.call({"op": "configure", "cases": [case], "record": False})
        result = bridge.call({"op": "step", "actions": [1], "record": False})[0]
        if result["terminated"] or result["native_steps"] != 5:
            raise ValueError("Known source quiet preflight failed")
        print(canonical({"schema": "generalization_preflight_v1", "runtime": runtime_record(),
                         "spec": spec, "source_case_index": 0, "result": compact_native(result),
                         "reserved_worlds_constructed": 0}), flush=True)
    finally:
        bridge.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("lock", "pilot", "summary"):
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
    commands.add_parser("native-preflight")
    args = parser.parse_args()
    require_container()
    if args.command == "native-preflight":
        native_preflight()
    elif args.command == "pilot":
        return pilot(args.output)
    else:
        {"lock": lock, "summary": summary}[args.command](args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(canonical({"error": f"{type(error).__name__}: {error}",
                         "existing_evidence_preserved": True}), file=sys.stderr)
        raise SystemExit(1)
