"""Locked bounded specialization on previously exposed native kit conditions.

Each target is one continuous opportunity: 128 decisions for PPO, then 512
with fixed weights. This driver does not lock protocols, summarize research
gates, resume attempts, or replace the workshop's immutable pretrained actor.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import time

import generalization_pretraining as helper
from specialization_manifest import PROJECT, digest, ref, read_bound, read_lock, write_new

NATIVE_COMMAND = ["/workspace/build/native/droid-transfer-env"]
LEARNING_ARMS = ("head_only", "full", "scratch")
HEAD_NAMES = ("action_net.bias", "action_net.weight", "value_net.bias", "value_net.weight")
PARTITIONS = ("actor_trunk", "value_trunk", "actor_head", "value_head")
WINDOWS = (64, 128, 320, 640)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_reference(value):
    return {"path": value["path"], "sha256": value["sha256"]}


def save_checkpoint(model, path):
    """Serialize ordinary tensor dictionaries, then create bytes exclusively.

    A frozen trial intentionally has no optimizer. Its ZIP contains policy
    tensors only, so no absent optimizer or audit closure is serialized.
    """
    memory = io.BytesIO()
    custom = "_get_torch_save_params" in model.__dict__
    original = model._get_torch_save_params
    try:
        if model.policy.optimizer is None:
            model._get_torch_save_params = lambda: (["policy"], [])
        model.save(memory, exclude=["train", "_get_torch_save_params"])
    finally:
        if custom:
            model._get_torch_save_params = original
        else:
            model.__dict__.pop("_get_torch_save_params", None)
    with Path(path).open("xb") as stream:
        stream.write(memory.getvalue())


def rng_digest(torch):
    return hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest()


def observation_digest(value, np):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (1, 204) or not np.isfinite(array).all():
        raise ValueError("Expected one finite 204-channel local history")
    return hashlib.sha256(array.astype("<f4", copy=False).tobytes()).hexdigest()


def partition(name):
    if name.startswith("mlp_extractor.policy_net."):
        return "actor_trunk"
    if name.startswith("mlp_extractor.value_net."):
        return "value_trunk"
    if name.startswith("action_net."):
        return "actor_head"
    if name.startswith("value_net."):
        return "value_head"
    raise ValueError(f"Unexpected trainable policy parameter: {name}")


def tensor_digest(items):
    value = hashlib.sha256()
    for name, tensor in sorted(items):
        array = tensor.detach().cpu().contiguous().numpy()
        value.update(name.encode() + b"\0")
        value.update(str(array.shape).encode() + b"\0" + str(array.dtype).encode() + b"\0")
        value.update(array.tobytes())
    return value.hexdigest()


def optimizer_report(model):
    """Bind optimizer membership to parameter names rather than unstable IDs."""
    optimizer = model.policy.optimizer
    if optimizer is None:
        return {"enabled": False, "fresh": True, "param_groups": [], "state": {},
                "state_tensors": 0, "step_min": 0, "step_max": 0}
    names = {id(value): name for name, value in model.policy.named_parameters()}
    groups, state, steps = [], {}, []
    for group in optimizer.param_groups:
        group_names = [names[id(value)] for value in group["params"]]
        if len(group_names) != len(set(group_names)):
            raise ValueError("Duplicate optimizer parameter")
        groups.append({"parameter_names": group_names, "learning_rate": float(group["lr"]),
                       "eps": float(group["eps"]), "betas": list(group["betas"]),
                       "weight_decay": float(group["weight_decay"]), "amsgrad": bool(group["amsgrad"])})
        for value in group["params"]:
            name = names[id(value)]
            entry = optimizer.state.get(value)
            if not entry:
                continue
            step = int(entry["step"].item())
            steps.append(step)
            state[name] = {"step": step,
                           "exp_avg_sha256": tensor_digest([("exp_avg", entry["exp_avg"])]),
                           "exp_avg_sq_sha256": tensor_digest([("exp_avg_sq", entry["exp_avg_sq"])])}
    return {"enabled": True, "fresh": not state, "param_groups": groups, "state": state,
            "state_tensors": len(state), "step_min": min(steps, default=0), "step_max": max(steps, default=0)}


def policy_snapshot(model, torch, *, observation=None, np=None):
    parameters = list(model.policy.named_parameters())
    parts = {}
    for label in PARTITIONS:
        values = [(name, value) for name, value in parameters if partition(name) == label]
        parts[label] = {"sha256": tensor_digest(values), "tensors": len(values),
                        "elements": sum(value.numel() for _, value in values)}
    buffer = model.rollout_buffer
    result = {"parameter_sha256": helper.parameter_digest(model.policy), "partitions": parts,
              "parameters_finite": all(bool(torch.isfinite(value).all()) for _, value in parameters),
              "parameter_elements": sum(value.numel() for _, value in parameters),
              "parameter_tensors": len(parameters),
              "requires_grad": {name: bool(value.requires_grad) for name, value in parameters},
              "trainable_elements": sum(value.numel() for _, value in parameters if value.requires_grad),
              "trainable_tensors": sum(bool(value.requires_grad) for _, value in parameters),
              "optimizer": optimizer_report(model), "num_timesteps": int(model.num_timesteps),
              "optimization_epochs": int(model._n_updates), "torch_rng_sha256": rng_digest(torch),
              "rollout_buffer": {"full": bool(buffer.full), "position": int(buffer.pos),
                                 "buffer_size": int(buffer.buffer_size), "n_envs": int(buffer.n_envs)}}
    if observation is not None:
        result["input_float32_le_sha256"] = observation_digest(observation, np)
    return result


def configure_scope(model, arm, torch, ppo):
    if arm not in ("frozen", *LEARNING_ARMS):
        raise ValueError("Unknown declared specialization arm")
    parameters = dict(model.policy.named_parameters())
    if len(parameters) != 12 or sum(value.numel() for value in parameters.values()) != 34820:
        raise ValueError("The locked actor/critic architecture changed")
    for name, value in parameters.items():
        partition(name)
        value.requires_grad_(arm in ("full", "scratch") or (arm == "head_only" and name in HEAD_NAMES))
        value.grad = None
    selected = [value for value in parameters.values() if value.requires_grad]
    model.policy.optimizer = (torch.optim.Adam(selected, lr=ppo["learning_rate"], eps=ppo["optimizer_eps"])
                              if selected else None)
    expected = {"frozen": (0, 0), "head_only": (4, 260), "full": (12, 34820), "scratch": (12, 34820)}[arm]
    if (len(selected), sum(value.numel() for value in selected)) != expected:
        raise ValueError("Optimizer scope differs from the locked arm")
    if model.use_sde or model.policy.use_sde:
        raise ValueError("This categorical study requires use_sde=False")
    if model.n_steps != 64 or model.batch_size != 64 or model.n_epochs != 4 or model.n_envs != 1:
        raise ValueError("Expected two 64-decision rollouts with four single-batch epochs each")
    if model.num_timesteps or model._n_updates or model.rollout_buffer.pos or model.rollout_buffer.full:
        raise ValueError("Target model contains experience or optimization history")
    if not optimizer_report(model)["fresh"]:
        raise ValueError("Trial optimizer is not fresh")


def load_policy_weights(model, checkpoint, expected_hash):
    # PPO.load would restore source optimizer/counters, and cannot load a
    # filtered head-only optimizer into the standard all-parameter optimizer.
    from stable_baselines3.common.save_util import load_from_zip_file
    _, parameters, _ = load_from_zip_file(checkpoint, device="cpu", load_data=False)
    if not parameters or "policy" not in parameters:
        raise ValueError("Checkpoint has no policy state dictionary")
    model.policy.load_state_dict(parameters["policy"], strict=True)
    if helper.parameter_digest(model.policy) != expected_hash:
        raise ValueError("Loaded policy tensors differ from the bound checkpoint")


class ActionAudit:
    """Audit the same SB3 categorical calls used by rollout and inference."""
    def __init__(self, model, env, torch, np):
        self.model, self.env, self.torch, self.np = model, env, torch, np
        self.pending = None
        self.reset_count = 0
        self.initial_local = None
        self._forward = model.policy.forward
        self._predict = model.policy._predict
        self._step_wait, self._reset = env.step_wait, env.reset

    def _action(self, original, observation, *args, **kwargs):
        if self.pending is not None:
            raise ValueError("An earlier sampled action has not received its native response")
        before = rng_digest(self.torch)
        result = original(observation, *args, **kwargs)
        action = result[0] if isinstance(result, tuple) else result
        values = action.detach().cpu().numpy().reshape(-1).tolist()
        if len(values) != 1:
            raise ValueError("Expected exactly one categorical action")
        self.pending = {"torch_rng_before_sha256": before, "torch_rng_after_sha256": rng_digest(self.torch),
                        "input_float32_le_sha256": observation_digest(observation.detach().cpu().numpy(), self.np),
                        "sampled_action": int(values[0])}
        return result

    def install(self):
        self.model.policy.forward = lambda observation, *a, **kw: self._action(self._forward, observation, *a, **kw)
        self.model.policy._predict = lambda observation, *a, **kw: self._action(self._predict, observation, *a, **kw)
        def reset():
            if self.reset_count:
                raise ValueError("A target or guard opportunity cannot reset or restart")
            self.reset_count += 1
            observation = self._reset()
            if getattr(self.env, "latest", None):
                self.initial_local = helper.copy_json(self.env.latest[0]["observation"])
            return observation
        def step_wait():
            if self.pending is None:
                raise ValueError("Native step has no audited categorical action")
            before = len(self.env.telemetry)
            try:
                return self._step_wait()
            finally:
                if len(self.env.telemetry) == before + 1:
                    row = self.env.telemetry[-1]
                    if row["action"] != self.pending["sampled_action"]:
                        raise ValueError("Native request differs from the sampled action")
                    row.update(self.pending)
                    self.pending = None
        self.env.reset, self.env.step_wait = reset, step_wait

    def restore(self):
        self.model.policy.forward, self.model.policy._predict = self._forward, self._predict
        self.env.step_wait, self.env.reset = self._step_wait, self._reset


def run_episode(model, env, arm, study, torch, np, callback=None, on_snapshot=None, on_update=None):
    """One reset; standard learn(128), then inference from its exact last obs.

    The returned audit is available even after a raised safety/error through
    env.specialization_episode_audit. Synthetic tests can supply a VecEnv with
    telemetry without launching any native process.
    """
    budget = study["budget"]
    if (budget["prefix_decisions"] != 128 or budget["total_decisions"] != 640
            or budget["continuation_decisions"] != 512 or budget["horizon_decisions"] != 641
            or budget["rollouts"] != 2 or budget["adam_steps"] != 8 or budget["update_decisions"] != [64, 128]):
        raise ValueError("This driver implements only the declared 128+512 budget")
    if env.num_envs != 1 or env.horizon != 641 or not env.stop_on_safety:
        raise ValueError("One continuous opportunity requires one slot, horizon641, and safety-stop holding")
    audit = {"snapshots": {}, "training_updates": [], "reset_count": 0,
             "initial_local_observation": None, "pending_action_audit": None,
             "continuation_weights_and_optimizer_unchanged": False, "training_rng_unchanged": True}
    env.specialization_episode_audit = audit
    tracker = ActionAudit(model, env, torch, np)
    original_train = model.train
    tracker.install()
    def snapshot(count, observation):
        value = policy_snapshot(model, torch, observation=observation, np=np)
        audit["snapshots"]["initial" if count == 0 else f"after{count}"] = value
        if on_snapshot:
            on_snapshot(count, model, value)
        return value
    original_reset = env.reset
    def reset_with_snapshot():
        observation = original_reset()
        snapshot(0, observation)
        return observation
    env.reset = reset_with_snapshot
    def train():
        before = policy_snapshot(model, torch)
        event = {"event": "ppo_update", "decisions": int(model.num_timesteps),
                 "before": before, "after": None, "passed": False, "error": None}
        try:
            original_train()
            after = snapshot(int(model.num_timesteps), model._last_obs)
            event["after"] = after
            unchanged = before["torch_rng_sha256"] == after["torch_rng_sha256"]
            event["torch_rng_unchanged"] = unchanged
            audit["training_rng_unchanged"] &= unchanged
            expected_steps = int(model.num_timesteps) // 64 * 4
            correct = (unchanged and after["parameters_finite"]
                       and after["optimization_epochs"] == expected_steps
                       and after["optimizer"]["step_min"] == expected_steps
                       and after["optimizer"]["step_max"] == expected_steps)
            if arm == "head_only":
                correct &= all(after["partitions"][part]["sha256"] == before["partitions"][part]["sha256"]
                               for part in ("actor_trunk", "value_trunk"))
            event["passed"] = bool(correct)
            if not correct:
                raise ValueError("PPO update changed RNG, frozen trunks, finite parameters, or declared optimizer counters")
        except (Exception, KeyboardInterrupt) as failure:
            event["error"] = f"{type(failure).__name__}: {failure}"
            if event["after"] is None:
                event["after"] = policy_snapshot(model, torch)
            raise
        finally:
            audit["training_updates"].append(event)
            if on_update:
                on_update(event)
    try:
        if arm in LEARNING_ARMS:
            model.train = train
            model.learn(total_timesteps=128, callback=callback)
            if model.num_timesteps != 128 or env.total_interactions != 128 or len(audit["training_updates"]) != 2:
                raise ValueError("The learning prefix did not finish exactly two complete rollouts")
            observation = model._last_obs.copy()
            for value in model.policy.parameters():
                value.requires_grad_(False)
                value.grad = None
            start = 128
        elif arm == "frozen":
            observation = env.reset()
            start = 0
        else:
            raise ValueError("Unknown episode arm")
        for decision in range(start, 640):
            action, _ = model.predict(observation, deterministic=False)
            observation, _, done, _ = env.step(action)
            model._last_obs = observation
            if bool(done[0]):
                raise ValueError("The continuous opportunity unexpectedly ended or reset")
            if decision + 1 in (64, 128, 640):
                snapshot(decision + 1, observation)
        after = audit["snapshots"]["after640"]
        fixed = audit["snapshots"]["after128"]
        audit["continuation_weights_and_optimizer_unchanged"] = (
            after["parameter_sha256"] == fixed["parameter_sha256"]
            and after["optimizer"] == fixed["optimizer"]
            and after["optimization_epochs"] == fixed["optimization_epochs"]
            and after["num_timesteps"] == fixed["num_timesteps"])
        if not audit["continuation_weights_and_optimizer_unchanged"]:
            raise ValueError("Frozen continuation changed weights or optimizer history")
        return audit
    finally:
        audit["reset_count"] = tracker.reset_count
        audit["initial_local_observation"] = tracker.initial_local
        audit["pending_action_audit"] = tracker.pending
        tracker.restore()
        model.train = original_train


def stage_start(root, name, protocol_hash):
    folder = root / name
    folder.mkdir(exist_ok=False)
    write_new(folder / "started.json", {"schema": "specialization_stage_start_v1", "stage": name,
              "protocol_sha256": protocol_hash, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "case_construction_requests": 0})
    return folder


def checked_name(value):
    if re.fullmatch(r"[A-Za-z0-9._-]+", value) is None:
        raise ValueError("Unsafe retained trial identifier")
    return value


def seed_for(study, partition_name, seed, index):
    return study["action_rng"][partition_name + "_base"] + seed * study["action_rng"]["seed_multiplier"] + index


def source_inputs(protocol):
    released = {}
    for release in protocol["inputs"]["releases"]:
        if release["seed"] in released:
            raise ValueError("Duplicate pretrained source seed")
        for key in ("result", "initial", "final"):
            read_bound(release[key])
        released[release["seed"]] = release
    if set(released) != set(protocol["study"]["source_seeds"]):
        raise ValueError("Every declared source seed needs matched initial and final weights")
    return released


def post128_checkpoint(result):
    candidate = result.get("checkpoints", {}).get("after128")
    updates = result.get("training_updates", [])
    snapshot = result.get("snapshots", {}).get("after128", {})
    if (candidate is None or len(updates) != 2 or not all(row.get("passed", False) for row in updates)
            or not snapshot.get("parameters_finite", False)
            or snapshot.get("num_timesteps") != 128 or snapshot.get("optimization_epochs") != 8
            or snapshot.get("optimizer", {}).get("step_min") != 8
            or snapshot.get("optimizer", {}).get("step_max") != 8
            or candidate.get("decisions") != 128
            or candidate.get("parameter_sha256") != snapshot.get("parameter_sha256")):
        return None
    return candidate


def guard_schedule(study):
    result = [(seed, "frozen", index, None) for seed in study["source_seeds"] for index in range(2)]
    result.extend((seed, arm, guard_index, target_index) for seed in study["source_seeds"]
                  for target_index in range(12) for arm in ("head_only", "full") for guard_index in range(2))
    return result


def record_traces(attempt, env):
    if env is None:
        return {}
    tape_path = attempt / "traces" / "requests.json.gz"
    tape = helper.tape_receipt(env, tape_path)
    tape.update(ref(tape_path))
    telemetry_path = attempt / "traces" / "telemetry.json.gz"
    helper.save_gzip_new(telemetry_path, {"schema": "specialization_telemetry_v1", "rows": env.telemetry})
    return {"request_tape": tape, "telemetry": ref(telemetry_path)}


def construction_count(env):
    if env is None:
        return 0
    requests = list(env.bridge.request_tape)
    if env.unresolved_request is not None:
        requests.append(env.unresolved_request)
    return sum(len(value["cases"]) for value in requests if value["op"] == "configure")


def trial(root, folder, protocol, protocol_hash, seed, arm, case_index, checkpoint,
          expected_hash, dependencies, *, guard_source=None):
    study = protocol["study"]
    guard = guard_source is not None
    partition_name = "guard" if guard else "target"
    case_entry = study["guard_cases" if guard else "target_cases"][case_index]
    suffix = "" if not guard or arm == "frozen" else "--" + guard_source["case_id"]
    trial_id = checked_name(f"{partition_name}--{case_entry['id']}--{seed}--{arm}{suffix}")
    attempt = folder / "trials" / trial_id
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "traces").mkdir()
    started = {"schema": "specialization_trial_start_v1", "protocol_sha256": protocol_hash,
               "trial_id": trial_id, "partition": partition_name, "seed": seed, "arm": arm,
               "case_id": case_entry["id"], "case_index": case_index, "checkpoint": checkpoint,
               "guard_source": guard_source}
    write_new(attempt / "started.json", started)
    if checkpoint is None:
        blocked = {"schema": "specialization_trial_v1", **started, "status": "blocked", "complete": False,
                   "safe": False, "blocked": True, "cancelled": False,
                   "error": "The target has no valid sealed post128 checkpoint; no guard world constructed",
                   "metrics": helper.metrics_from_results([]), "case_construction_requests": 0,
                   "exposure_exact": True, "checkpoints": {}, "snapshots": {}, "training_updates": [],
                   "case": helper.copy_json(case_entry["case"]),
                   "action_seed": seed_for(study, partition_name, seed, case_index),
                   "checkpoint_parameter_sha256": None, "reset_count": 0,
                   "request_tape": None, "telemetry": None, "learning_log": None,
                   "training_rng_unchanged": None, "continuation_weights_and_optimizer_unchanged": False}
        # started has a different schema; the final result is always a trial.
        blocked["schema"] = "specialization_trial_v1"
        write_new(attempt / "result.json", blocked)
        return blocked, ref(attempt / "result.json")
    torch, np, _, BaseCallback, _, base, _ = dependencies
    env = model = None
    error = None
    cancelled = False
    checkpoints = {}
    update_stream = (attempt / "learning.jsonl").open("x", encoding="utf-8")
    start = time.monotonic()
    action_seed = seed_for(study, partition_name, seed, case_index)
    active_arm = "frozen" if guard else arm
    class Callback(BaseCallback):
        def _on_step(self):
            # Exact learn(128) exits normally after both complete rollouts.
            return True
    def on_snapshot(count, current, value):
        if count == 0 or (count in (64, 128) and active_arm in LEARNING_ARMS):
            label = "initial" if count == 0 else f"after{count}"
            path = attempt / ("policy_initial.zip" if count == 0 else f"policy_after{count}.zip")
            if path.exists():
                raise FileExistsError("Refusing checkpoint overwrite")
            save_checkpoint(current, path)
            checkpoints[label] = {**ref(path), "parameter_sha256": value["parameter_sha256"], "decisions": count}
    def on_update(event):
        update_stream.write(helper.canonical(event) + "\n")
        update_stream.flush()
    try:
        with helper.wall_deadline(study["wall_caps_s"][partition_name]):
            path = read_bound(file_reference(checkpoint))
            env_type = helper.make_instrumented_env_class(base)
            env = env_type([helper.copy_json(case_entry["case"])], record=False, stop_on_safety=True,
                           horizon=641, command=NATIVE_COMMAND)
            if env.bridge.spec["schema"] != "actuation_transfer_bridge_v1":
                raise ValueError("Unexpected native bridge specification")
            model = helper.build_model(env, study["ppo"], seed, target=True, dependencies=dependencies)
            load_policy_weights(model, path, expected_hash)
            configure_scope(model, active_arm, torch, study["ppo"])
            model.set_random_seed(action_seed)
            run_episode(model, env, active_arm, study, torch, np, callback=Callback(),
                        on_snapshot=on_snapshot, on_update=on_update)
    except (Exception, KeyboardInterrupt) as failure:
        error = f"{type(failure).__name__}: {failure}"
        cancelled = isinstance(failure, KeyboardInterrupt)
    finally:
        update_stream.close()
        close_error = helper.close_env(env)
    rows = env.telemetry if env is not None else []
    metrics = helper.metrics_from_results(rows)
    audit = getattr(env, "specialization_episode_audit", {}) if env is not None else {}
    exact = env is not None and env.bridge.last_error_response is None and env.unresolved_request is None
    wall_s = time.monotonic() - start
    clock = wall_s > study["wall_caps_s"][partition_name]
    snaps = audit.get("snapshots", {})
    frozen = audit.get("continuation_weights_and_optimizer_unchanged", False)
    complete = (error is None and close_error is None and exact and not clock and frozen
                and metrics["decisions"] == 640 and metrics["native_steps"] == 3200
                and metrics["safety_stops"] == 0 and audit.get("reset_count") == 1)
    result = {"schema": "specialization_trial_v1", **{key: value for key, value in started.items() if key != "schema"},
              "status": "complete" if complete else "incomplete", "blocked": False,
              "case": helper.copy_json(case_entry["case"]), "action_seed": action_seed,
              "checkpoint_parameter_sha256": expected_hash, "complete": bool(complete),
              "safe": bool(exact and error is None and close_error is None and metrics["safety_stops"] == 0),
              "error": error, "close_error": close_error, "cancelled": cancelled,
              "metrics": metrics, "windows": [{"requested_decisions": count, "complete": len(rows) >= count,
                  "metrics": helper.metrics_from_results(rows[:count])} for count in WINDOWS],
              "prefix_metrics": helper.metrics_from_results(rows[:128]),
              "continuation_metrics": helper.metrics_from_results(rows[128:]),
              "snapshots": snaps, "checkpoints": checkpoints,
              "training_updates": audit.get("training_updates", []),
              "training_rng_unchanged": audit.get("training_rng_unchanged", False),
              "continuation_weights_and_optimizer_unchanged": frozen,
              "initial_local_observation": audit.get("initial_local_observation"),
              "reset_count": audit.get("reset_count", 0), "pending_action_audit": audit.get("pending_action_audit"),
              "exposure_exact": exact, "native_failure": env.bridge.last_error_response if env else None,
              "unresolved_request": env.unresolved_request if env else None,
              "case_construction_requests": construction_count(env), "wall_s": wall_s,
              "wall_cap_exceeded": clock, "learning_log": ref(attempt / "learning.jsonl")}
    if model is not None:
        result["retained_final_snapshot"] = policy_snapshot(model, torch)
    result.update(record_traces(attempt, env))
    write_new(attempt / "result.json", result)
    print(helper.canonical({"trial": trial_id, "complete": complete, "decisions": metrics["decisions"],
                            "reward": metrics["native_reward"], "error": error}), flush=True)
    return result, ref(attempt / "result.json")


def stage_finish(folder, protocol_hash, results, references, planned, *, seal=None, blocked_reason=None):
    value = {"schema": "specialization_stage_v1", "stage": folder.name, "protocol_sha256": protocol_hash,
             "status": "blocked" if blocked_reason else "complete" if len(results) == planned else "incomplete",
             "all_trials_complete_safe": len(results) == planned and all(row["complete"] and row["safe"] for row in results),
             "trial_results": references, "planned_trials": planned, "collected_trials": len(results),
             "actual_decisions": sum(row["metrics"]["decisions"] for row in results),
             "native_steps": sum(row["metrics"]["native_steps"] for row in results),
             "case_construction_requests": sum(row["case_construction_requests"] for row in results),
             "exposure_exact": all(row["exposure_exact"] for row in results),
             "seal": seal, "blocked_reason": blocked_reason}
    write_new(folder / "stage.json", value)
    write_new(folder / "receipt.json", {"schema": "specialization_stage_receipt_v1",
              "protocol_sha256": protocol_hash, "stage": ref(folder / "stage.json")})
    return value


def emergency_result(root, folder, partition_name, case_entry, case_index, seed, arm, failure,
                     protocol_hash, guard_source=None):
    """An archival failure preserves its files and does not replace its attempt."""
    suffix = "" if guard_source is None or arm == "frozen" else "--" + guard_source["case_id"]
    trial_id = checked_name(f"{partition_name}--{case_entry['id']}--{seed}--{arm}{suffix}")
    attempt = folder / "trials" / trial_id
    attempt.mkdir(parents=True, exist_ok=True)
    prefix = [ref(path) for path in sorted(attempt.rglob("*")) if path.is_file()]
    result = {"schema": "specialization_trial_failure_v1", "protocol_sha256": protocol_hash,
              "trial_id": trial_id, "partition": partition_name, "seed": seed, "arm": arm,
              "case_id": case_entry["id"], "case_index": case_index, "guard_source": guard_source,
              "complete": False, "safe": False, "cancelled": isinstance(failure, KeyboardInterrupt),
              "error": f"{type(failure).__name__}: {failure}", "retained_prefix": prefix,
              "metrics": helper.metrics_from_results([]), "case_construction_requests": 0,
              "exposure_exact": False, "unknown_exposure": True, "checkpoints": {},
              "counts_interpretation": "Zero is a known lower bound; native advancement may precede archival failure"}
    path = attempt / "archival-failure.json"
    write_new(path, result)
    return result, ref(path)


def seal_targets(folder, protocol_hash, results, references):
    evidence = []
    for result in results:
        for key in ("checkpoint", "request_tape", "telemetry", "learning_log"):
            if result.get(key):
                evidence.append({"path": result[key]["path"], "sha256": result[key]["sha256"]})
        evidence.extend({"path": value["path"], "sha256": value["sha256"]}
                        for value in result.get("checkpoints", {}).values())
        evidence.extend(result.get("retained_prefix", []))
    unique = {}
    for item in references + evidence:
        read_bound(item)
        if item["path"] in unique and unique[item["path"]] != item["sha256"]:
            raise ValueError("Conflicting sealed evidence hashes")
        unique[item["path"]] = item["sha256"]
    seal = {"schema": "specialization_target_seal_v1", "protocol_sha256": protocol_hash,
            "planned_trials": 144, "collected_trials": len(results), "all_results_collected": len(results) == 144,
            "sealed_before_guard_worlds": True, "trial_results": references,
            "bound_evidence": [{"path": path, "sha256": value} for path, value in sorted(unique.items())],
            "guard_case_construction_requests": 0}
    write_new(folder / "seal.json", seal)
    return ref(folder / "seal.json")


def targets(output):
    root = Path(output).resolve()
    protocol, protocol_hash = read_lock(root)
    folder = stage_start(root, "targets", protocol_hash)
    try:
        releases = source_inputs(protocol)
    except (OSError, ValueError, KeyError) as failure:
        return stage_finish(folder, protocol_hash, [], [], 144,
                            blocked_reason=f"Source release unavailable: {failure}")
    dependencies = helper.learning_dependencies()
    results, references = [], []
    study = protocol["study"]
    for seed in study["source_seeds"]:
        source = releases[seed]
        for index, entry in enumerate(study["target_cases"]):
            for arm in study["arms"]:
                initial = arm == "scratch"
                checkpoint = source["initial" if initial else "final"]
                expected = source["initial_parameter_sha256" if initial else "final_parameter_sha256"]
                try:
                    result, reference = trial(root, folder, protocol, protocol_hash, seed, arm, index,
                                              checkpoint, expected, dependencies)
                except (Exception, KeyboardInterrupt) as failure:
                    result, reference = emergency_result(root, folder, "target", entry, index, seed, arm, failure, protocol_hash)
                results.append(result)
                references.append(reference)
                if result["cancelled"]:
                    break
            if results[-1]["cancelled"]:
                break
        if results[-1]["cancelled"]:
            break
    seal = seal_targets(folder, protocol_hash, results, references)
    _, final_protocol_hash = read_lock(root)
    if final_protocol_hash != protocol_hash:
        raise ValueError("Protocol changed during target collection")
    return stage_finish(folder, protocol_hash, results, references, 144, seal=seal)


def guard_release(root, protocol_hash, study):
    receipt = read_json(root / "targets" / "receipt.json")
    if receipt["protocol_sha256"] != protocol_hash:
        raise ValueError("Target stage belongs to another lock")
    stage = read_json(read_bound(receipt["stage"]))
    if stage["status"] != "complete" or stage["collected_trials"] != 144:
        raise ValueError("All 144 target attempts must finish before guard release")
    seal = read_json(read_bound(stage["seal"]))
    if (seal["protocol_sha256"] != protocol_hash or not seal["all_results_collected"]
            or not seal["sealed_before_guard_worlds"] or seal["collected_trials"] != 144
            or seal["trial_results"] != stage["trial_results"]):
        raise ValueError("Target ledger seal is incomplete or inconsistent")
    for evidence in seal["bound_evidence"]:
        read_bound(evidence)
    results = {}
    for reference in seal["trial_results"]:
        result = read_json(read_bound(reference))
        key = (result["seed"], result["case_index"], result["arm"])
        if key in results or result["protocol_sha256"] != protocol_hash or result["partition"] != "target":
            raise ValueError("Duplicate or mismatched sealed target result")
        if result["case_id"] != study["target_cases"][result["case_index"]]["id"]:
            raise ValueError("Sealed target case identity differs")
        results[key] = {"result": result, "reference": reference}
    expected = {(seed, index, arm) for seed in study["source_seeds"]
                for index in range(12) for arm in study["arms"]}
    if set(results) != expected:
        raise ValueError("Sealed target matrix differs from the declared four arms")
    return results, stage["seal"]


def guards(output):
    root = Path(output).resolve()
    protocol, protocol_hash = read_lock(root)
    folder = stage_start(root, "guards", protocol_hash)
    study = protocol["study"]
    try:
        sealed, seal = guard_release(root, protocol_hash, study)
        sources = source_inputs(protocol)
    except (OSError, ValueError, KeyError, IndexError) as failure:
        return stage_finish(folder, protocol_hash, [], [], 150,
                            blocked_reason=f"Guard release unavailable: {failure}")
    write_new(folder / "release.json", {"schema": "specialization_guard_release_v1", "protocol_sha256": protocol_hash,
              "target_seal": seal, "released_before_first_guard_world": True})
    dependencies = helper.learning_dependencies()
    results, references = [], []
    for seed, arm, index, target_index in guard_schedule(study):
        source = None
        if arm == "frozen":
            checkpoint = sources[seed]["final"]
            expected = sources[seed]["final_parameter_sha256"]
            source = {"kind": "original_pooled_baseline"}
        else:
            target = sealed[(seed, target_index, arm)]
            result = target["result"]
            checkpoint = post128_checkpoint(result)
            expected = checkpoint["parameter_sha256"] if checkpoint else None
            source = {"kind": "post128_specialization", "case_id": result["case_id"],
                      "case_index": target_index, "arm": arm, "target_result": target["reference"]}
        try:
            result, reference = trial(root, folder, protocol, protocol_hash, seed, arm, index,
                                      checkpoint, expected, dependencies, guard_source=source)
        except (Exception, KeyboardInterrupt) as failure:
            result, reference = emergency_result(root, folder, "guard", study["guard_cases"][index], index,
                                                  seed, arm, failure, protocol_hash, source)
        results.append(result)
        references.append(reference)
        if result["cancelled"]:
            break
    _, final_protocol_hash = read_lock(root)
    if final_protocol_hash != protocol_hash:
        raise ValueError("Protocol changed during guard collection")
    return stage_finish(folder, protocol_hash, results, references, 150, seal=seal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("targets", "guards"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    helper.require_container()
    result = {"targets": targets, "guards": guards}[args.command](args.output)
    print(helper.canonical({"stage": result["stage"], "status": result["status"],
                            "actual_decisions": result["actual_decisions"]}), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as error:
        print(helper.canonical({"error": f"{type(error).__name__}: {error}", "existing_evidence_preserved": True}), file=sys.stderr)
        raise SystemExit(1)
