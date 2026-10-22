"""Bounded native actuation-transfer experiment using unmodified SB3 PPO.

Commands are separate so the prospective manifest can be inspected and locked
before any learning, and interrupted work can resume without overwriting trials.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import itertools
import json
import os
import platform
import time
import traceback
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parents[1] / "build" / "transfer-matplotlib")

import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from native_transfer import NativeBridge, NativeTransferVecEnv, SafetyStop, PROJECT, DT

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameter_digest(policy):
    result = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        result.update(name.encode() + b"\0")
        result.update(str(array.shape).encode() + b"\0" + str(array.dtype).encode() + b"\0")
        result.update(array.tobytes())
    return result.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def save_gzip_new(path, value):
    with Path(path).open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            compressed.write(json.dumps(value, separators=(",", ":"), allow_nan=False).encode())


def already_finished(folder, root, *, training=False):
    if not folder.exists():
        return False
    path = folder / "result.json"
    if not path.is_file():
        raise RuntimeError(f"Incomplete attempt preserved at {folder}; inspect its failure before resuming")
    result = json.loads(path.read_text())
    if result["protocol_sha256"] != digest(root / "protocol.json"):
        raise RuntimeError("Existing attempt belongs to a different protocol")
    if training:
        valid = result["complete"] and digest(folder / "policy.zip") == result["policy_sha256"]
        valid = valid and digest(folder / "native-tape.json.gz") == result["native_tape_sha256"]
    else:
        valid = result["replay"]["exact"] and digest(folder / "trace.json.gz") == result["trace_sha256"]
    if not valid:
        raise RuntimeError(f"Incomplete or altered attempt preserved at {folder}; no automatic rerun")
    print(json.dumps({"preserved_completed_attempt": folder.name}), flush=True)
    return True


def body(lengths=(3, 3), weight=False, hinge=0):
    return {"schema": "construction_kit_v2", "name": "The same parts, rebuilt",
            "segments": list(lengths), "powered_hinge": hinge,
            "blocks": ([{"id": "block-1", "segment": 1, "slot": 0, "side": -1}] if weight else []),
            "sensor": {"segment": 1, "slot": 2, "side": -1},
            "light": {"segment": 1, "slot": 2, "side": 1, "face": 1}}


def case(lengths, weight, sign, hinge):
    return {"assembly": body(lengths, weight, hinge),
            "sun": {"position_m": [sign * .3, 0., .2], "intensity_lux": 1000.}}


def manifest():
    source = [case((a, b), w, sign, 0)
              for a, b, w, sign in itertools.product([3, 4], [3, 4], [False, True], [-1, 1])]
    target = []
    for name, lengths, weight in [("long-upper", (4, 3), False),
                                  ("long-lower", (3, 4), False),
                                  ("weighted", (4, 4), True)]:
        for sign in [-1, 1]:
            target.append({"id": f"{name}-{'left' if sign < 0 else 'right'}",
                           "case": case(lengths, weight, sign, 1)})
    return {
        "schema": "actuation_transfer_protocol_v1", "status": "locked_before_native_learning",
        "question": "Does original-layout pretraining improve early native light reward after the motor moves to the elbow?",
        "scope": "Same two-link graph and one motor. Actuation layout is withheld; target geometry and lights occur in the source distribution.",
        "evidence_status": "Exploratory native comparison; the separate CartPole reproduction missed its declared gain gate. Its gate remains failed, with no retuning or rerun.",
        "reference": {"report": "build/actuation-transfer-reference/cartpole-v1/report.json",
                      "report_sha256": "a0421bb5fa1b792af934986a3758fdb517081dceb6fdaaf68f1498116decd309",
                      "overall_passed": False, "trained_mean_return": 409.2,
                      "untrained_mean_return": 333.2, "gain": 76., "required_gain": 100.,
                      "absolute_performance_and_integrity_checks": True},
        "source_cases": source, "target_cases": target,
        "development_cases": [case((3, 3), False, sign, 1) for sign in [-1, 1]],
        "source_seeds": [101, 202, 303], "development_seed": 404,
        "target_action_seed_rule": "10000 + source_seed * 10 + case_index; reset after model construction/weight transfer",
        "source_decisions_per_seed": 98304, "development_decisions": 32768,
        "target_decisions": 640, "decision_dt_s": .1, "native_dt_s": .02,
        "early_windows_s": [6.4, 12.8, 32., 64.],
        "native_reward": "Unmodified sum of five native transitions; one light plus one IMU; no reward shaping or normalization",
        "policy_input": {"frames": 12, "frame_size": 17, "dimension": 204,
                         "order": "oldest to newest", "fixed_clip": [-5., 5.],
                         "frame": ["executed_effort/.15", "sin(motor_position)", "cos(motor_position)",
                                   "motor_velocity/10", "motor_current/.5", "imu_force_xyz/50 (3)",
                                   "imu_gyro_xyz/10 (3)", "light_lux/1000", "previous_request/.15",
                                   "previous_macro_reward/(2*.1)", "imu_valid", "light_valid", "frame_present"],
                         "initialization": "11 zero absent frames and current local frame; invalid sensor values zero under validity masks",
                         "excluded": ["assembly", "powered_hinge", "geometry", "passive_joint", "sun_position", "absolute_time", "case_id"],
                         "critic": "same local stack as actor; no privileged critic inputs"},
        "ppo": {"implementation": "stable-baselines3", "version": stable_baselines3.__version__,
                "network": "Separate two-layer 64x64 tanh actor/value networks; categorical 3-action output",
                "learning_rate": .0003, "n_epochs": 4, "batch_size": 64,
                "gamma": .92 ** (.1 / .4), "gae_lambda": .95, "clip_range": .2,
                "ent_coef": .01, "vf_coef": .5, "max_grad_norm": .5,
                "optimizer": "Adam", "optimizer_eps": 1e-5, "ortho_init": True,
                "clip_range_vf": None, "target_kl": None,
                "source_n_steps": 256, "source_n_envs": 8, "target_n_steps": 64,
                "target_n_envs": 1, "source_sampling": "uniform cases at each reset; independent numpy Generator(source_seed+500000), never rewound by policy RNG reset",
                "normalization": "advantage normalization only; fixed sensor scaling; no VecNormalize",
                "transfer": "actor+critic weights only; fresh optimizer, rollout buffer and episode history",
                "target_updates": "10 batches; last update is saved but earns no measured post-update reward"},
        "arms": ["frozen", "adapt", "scratch"],
        "controls": ["source_frozen", "source_quiet", "quiet", "negative", "positive", "rhythm_2s", "rhythm_4s"],
        "probe_scope": "Run only after source checkpoints fixed. Same governor and raw reward. Best probe is an exposed attainable witness, not a learned policy or upper bound.",
        "safety": "Native veto or any motor feedback flag ends target opportunity; no reset rescue. Report raw prefix return and actual duration; a stop fails the success gate.",
        "primary": "Paired adapt-minus-scratch integrated native return through64s, summarized per training seed then median across3 seeds",
        "success_gate": "All target arms complete without safety stops; median seed mean adapt-minus-scratch is positive and at least2 of3 seed means positive. Claim useful behavior only where adapt also beats quiet and source learning beats source quiet.",
        "limits": "Three training seeds/six declared cases are a bounded development comparison, not population generalization or a hardware sample-efficiency guarantee.",
        "wall_time_caps_s": {"reference_development": 900, "source_per_seed": 1800, "target_trial": 180},
        "no_tuning": "Final checkpoints only; no architecture/hyperparameter/seed selection after outcomes. Failed or incomplete budgeted runs remain evidence.",
    }


def load_protocol(root):
    path = root / "protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    expected = (root / "protocol.sha256").read_text().strip()
    if digest(path) != expected:
        raise RuntimeError("Locked protocol hash changed")
    return protocol


def build_model(env, protocol, seed, *, target=False):
    p = protocol["ppo"]
    model = PPO("MlpPolicy", env, device="cpu", seed=seed, verbose=0,
                n_steps=p["target_n_steps"] if target else p["source_n_steps"],
                batch_size=p["batch_size"], n_epochs=p["n_epochs"],
                learning_rate=p["learning_rate"], gamma=p["gamma"], gae_lambda=p["gae_lambda"],
                clip_range=p["clip_range"], ent_coef=p["ent_coef"], vf_coef=p["vf_coef"],
                max_grad_norm=p["max_grad_norm"],
                policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]},
                               "activation_fn": torch.nn.Tanh})
    return model


class BudgetCallback(BaseCallback):
    def __init__(self, path, wall_cap):
        super().__init__()
        self.path = Path(path)
        self.wall_cap = wall_cap
        self.began = time.monotonic()
        self.last_log = -1
        self.stopped_by_clock = False

    def _on_step(self):
        elapsed = time.monotonic() - self.began
        if elapsed >= self.wall_cap:
            self.stopped_by_clock = True
            return False
        return True

    def _on_rollout_end(self):
        row = {"decisions": self.num_timesteps, "wall_s": time.monotonic() - self.began,
               "mean_episode_return": (float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
                                       if self.model.ep_info_buffer else None),
               "preceding_update_epochs": self.model._n_updates,
               "preceding_update_metrics": {k: float(v) for k, v in self.model.logger.name_to_value.items()
                                            if k.startswith("train/") and np.isscalar(v)}}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        if row["wall_s"] - self.last_log >= 30:
            print(json.dumps({"progress": self.path.parent.name, **row}), flush=True)
            self.last_log = row["wall_s"]


def summarize_audit(audit, expected=640):
    rows = [entry["result"] for entry in audit]
    if not rows:
        return {"complete": False, "decisions": 0, "native_reward": 0.}
    result = {"complete": len(rows) == expected and not any(r["terminated"] for r in rows),
              "decisions": len(rows), "native_steps": sum(r["native_steps"] for r in rows),
              "duration_s": sum(r["elapsed_s"] for r in rows),
              "native_reward": sum(r["reward"] for r in rows),
              "light_reward": sum(r["light_reward"] for r in rows),
              "imu_reward": sum(r["imu_reward"] for r in rows),
              "energy_j": rows[-1]["energy_j"],
              "safety_stops": sum(bool(r["terminated"]) for r in rows),
              "max_speed_rad_s": max(r["max_speed_rad_s"] for r in rows),
              "max_current_a": max(r["max_current_a"] for r in rows),
              "max_temperature_c": max(r["max_temperature_c"] for r in rows)}
    result["windows"] = [{"seconds": n * DT, "native_reward": sum(r["reward"] for r in rows[:n]),
                          "available_decisions": min(len(rows), n)} for n in [64, 128, 320, 640]]
    return result


def replay_trace(case_config, audit):
    env = NativeTransferVecEnv([case_config], horizon=len(audit) + 1, record=True, stop_on_safety=True)
    error = None
    compared = 0
    try:
        env.reset()
        for entry in audit:
            try:
                env.step(np.asarray([entry["action"]]))
            except SafetyStop:
                if not entry["result"]["terminated"]:
                    raise
            actual = env.audit[-1]["result"]
            # Full JSON equality includes all five native-transition hashes,
            # unrounded raw reward, local observations and display geometry.
            if json.dumps(actual, sort_keys=True, separators=(",", ":")) != json.dumps(entry["result"], sort_keys=True, separators=(",", ":")):
                raise RuntimeError(f"Native replay differs at decision {entry['decision']}")
            compared += 1
    except Exception as exc:
        error = str(exc)
    finally:
        env.close()
    return {"exact": error is None and compared == len(audit), "decisions": compared, "error": error}


def save_trial(folder, env, record, initial):
    record["metrics"] = summarize_audit(env.audit)
    record["metrics"]["physical_horizon_complete"] = record["metrics"]["complete"]
    record["metrics"]["complete"] = record["metrics"]["complete"] and record["error"] is None
    trace_path = folder / "trace.json.gz"
    payload = {"case": record["case"], "initial_physics": initial,
               "steps": env.audit, "schema": "actuation_transfer_trace_v1"}
    save_gzip_new(trace_path, payload)
    record["trace_sha256"] = digest(trace_path)
    record["replay"] = replay_trace(record["case"], env.audit)
    write_new(folder / "result.json", record)
    print(json.dumps({"trial": folder.name, "metrics": record["metrics"],
                      "replay": record["replay"]}), flush=True)


def train(root, protocol, seed, development=False):
    name = "elbow-development" if development else f"source-{seed}"
    folder = root / name
    if already_finished(folder, root, training=True):
        return
    folder.mkdir(exist_ok=False)
    write_new(folder / "started.json", {"protocol_sha256": digest(root / "protocol.json"), "seed": seed})
    cases = protocol["development_cases"] if development else protocol["source_cases"]
    rng = np.random.default_rng(seed + 500000)
    sampler = lambda: copy.deepcopy(cases[int(rng.integers(len(cases)))])
    env = NativeTransferVecEnv([cases[0]] * 8, case_sampler=sampler)
    model = build_model(env, protocol, seed)
    count = protocol["development_decisions"] if development else protocol["source_decisions_per_seed"]
    cap = protocol["wall_time_caps_s"]["reference_development" if development else "source_per_seed"]
    callback = BudgetCallback(folder / "learning.jsonl", cap)
    start = time.monotonic()
    error = None
    try:
        model.learn(total_timesteps=count, callback=callback)
    except Exception as exc:
        error = str(exc)
    finally:
        learning_wall_s = time.monotonic() - start
        env.close()
    parameters_finite = all(bool(torch.isfinite(p).all()) for p in model.policy.parameters())
    if learning_wall_s > cap:
        callback.stopped_by_clock = True
    model.save(folder / "policy")
    save_gzip_new(folder / "native-tape.json.gz", {"requests": env.bridge.request_tape,
                  "native_stream_sha256": env.bridge.stream_digest.hexdigest()})
    report = {"seed": seed, "development_elbow_exposure": development, "requested_decisions": count,
              "actual_decisions": env.total_interactions, "native_steps": env.total_native_steps,
              "wall_s": time.monotonic() - start, "learning_wall_s": learning_wall_s,
              "stopped_by_clock": callback.stopped_by_clock, "parameters_finite": parameters_finite,
              "complete": error is None and env.total_interactions == count and not callback.stopped_by_clock and parameters_finite,
              "error": error, "native_failure": env.bridge.last_error_response,
              "exposure_exact": env.bridge.last_error_response is None,
              "safety_stops": env.safety_stops, "episodes": env.episode_count,
              "policy_sha256": digest(folder / "policy.zip"), "parameters": sum(p.numel() for p in model.policy.parameters()),
              "parameter_sha256": parameter_digest(model.policy),
              "native_tape_sha256": digest(folder / "native-tape.json.gz"),
              "native_stream_sha256": env.bridge.stream_digest.hexdigest(),
              "optimization_epochs": model._n_updates,
              "optimizer_steps": model._n_updates * 32,
              "final_update_metrics": {k: float(v) for k, v in model.logger.name_to_value.items()
                                       if k.startswith("train/") and np.isscalar(v)},
              "protocol_sha256": digest(root / "protocol.json")}
    write_new(folder / "result.json", report)
    print(json.dumps({"training_complete": name, **report}), flush=True)


def target_trial(root, protocol, item, case_index, seed, arm, *, development=False):
    name = f"{item['id']}--{seed}--{arm}"
    folder = root / "trials" / name
    if already_finished(folder, root):
        return
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder / "started.json", {"protocol_sha256": digest(root / "protocol.json"), "case_id": item["id"], "seed": seed, "arm": arm})
    config = copy.deepcopy(item["case"])
    if arm == "source_frozen":
        config["assembly"]["powered_hinge"] = 0
    elif arm == "source_quiet":
        config["assembly"]["powered_hinge"] = 0
    env = NativeTransferVecEnv([config], record=True, stop_on_safety=True)
    source_path = root / ("elbow-development" if development else f"source-{seed}") / "policy.zip"
    mode = "frozen" if arm in ["source_frozen", "development_frozen"] else arm
    learned = mode in ["frozen", "adapt", "scratch"]
    model = None
    initial_parameter_sha256 = None
    if learned:
        model = build_model(env, protocol, seed, target=True)
        if mode != "scratch":
            source = PPO.load(source_path, device="cpu")
            model.policy.load_state_dict(source.policy.state_dict(), strict=True)
            del source
        # RNG reset after initialization makes frozen/adapt identical until their
        # first parameter update, and avoids a random-initialization RNG confound.
        model.set_random_seed(10000 + seed * 10 + case_index)
        if model.policy.optimizer.state:
            raise RuntimeError("Target optimizer was not freshly initialized")
        initial_parameter_sha256 = parameter_digest(model.policy)
    error = None
    start = time.monotonic()
    initial = None
    try:
        if mode in ["adapt", "scratch"]:
            cb = BudgetCallback(folder / "learning.jsonl", protocol["wall_time_caps_s"]["target_trial"])
            model.learn(total_timesteps=protocol["target_decisions"], callback=cb)
            initial = env.initial_physics[0]
            model.save(folder / "policy_after")
            if cb.stopped_by_clock or time.monotonic() - cb.began > cb.wall_cap:
                error = "predeclared wall-time cap"
            if not all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()):
                error = "nonfinite policy parameters"
        else:
            obs = env.reset()
            initial = env.initial_physics[0]
            for step in range(protocol["target_decisions"]):
                if time.monotonic() - start > protocol["wall_time_caps_s"]["target_trial"]:
                    raise RuntimeError("predeclared wall-time cap")
                if mode == "frozen":
                    action, _ = model.predict(obs, deterministic=False)
                else:
                    action_id = {"quiet": 1, "source_quiet": 1, "negative": 0, "positive": 2}.get(mode)
                    if action_id is None:
                        period = 20 if mode == "rhythm_2s" else 40
                        action_id = 0 if step % period < period // 2 else 2
                    action = np.asarray([action_id])
                obs, _, _, _ = env.step(action)
    except Exception as exc:
        error = str(exc)
        initial = env.initial_physics[0] if env.initial_physics else None
    finally:
        env.close()
    record = {"schema": "actuation_transfer_trial_v1", "id": name, "case_id": item["id"],
              "case": config, "source_seed": seed, "arm": arm,
              "protocol_sha256": digest(root / "protocol.json"),
              "source_policy_sha256": digest(source_path) if learned and mode != "scratch" else None,
              "initial_parameter_sha256": initial_parameter_sha256,
              "final_parameter_sha256": parameter_digest(model.policy) if learned else None,
              "optimization_epochs": model._n_updates if learned else 0,
              "optimizer_steps": model._n_updates if learned else 0,
              "wall_s": time.monotonic() - start, "error": error,
              "native_failure": env.bridge.last_error_response,
              "new_interaction_budget": protocol["target_decisions"],
              "post_final_update_evaluation": False}
    save_trial(folder, env, record, initial)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["lock", "development", "source", "targets", "probes", "development-eval", "replay-sources"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    root = args.output.resolve()
    if args.command == "lock":
        root.mkdir(exist_ok=False, parents=True)
        write_new(root / "protocol.json", manifest())
        (root / "protocol.sha256").write_text(digest(root / "protocol.json") + "\n")
        write_new(root / "runtime.json", {"python": platform.python_version(), "torch": torch.__version__,
                                          "sb3": stable_baselines3.__version__, "numpy": np.__version__,
                                          "threads": torch.get_num_threads(), "device": "cpu"})
        reference = root / "reference-report.json"
        reference.write_bytes((PROJECT / manifest()["reference"]["report"]).read_bytes())
        if digest(reference) != manifest()["reference"]["report_sha256"]:
            raise RuntimeError("Reference evidence differs from declared dependency check")
        print(json.dumps({"locked": str(root), "sha256": digest(root / "protocol.json")}), flush=True)
        return
    protocol = load_protocol(root)
    if args.command == "development":
        train(root, protocol, protocol["development_seed"], True)
    elif args.command == "source":
        if args.seed not in protocol["source_seeds"]:
            raise ValueError("Seed is not in locked manifest")
        train(root, protocol, args.seed)
    elif args.command == "replay-sources":
        for name in ["elbow-development", *[f"source-{seed}" for seed in protocol["source_seeds"]]]:
            folder = root / name
            report_path = folder / "native-replay.json"
            if report_path.exists():
                existing = json.loads(report_path.read_text())
                if not existing["exact"] or existing["tape_sha256"] != digest(folder / "native-tape.json.gz"):
                    raise RuntimeError("Existing replay failed or its tape changed")
                continue
            with gzip.open(folder / "native-tape.json.gz", "rt", encoding="utf-8") as stream:
                tape = json.load(stream)
            bridge = NativeBridge()
            began = time.monotonic()
            error = None
            completed = 1
            try:
                if tape["requests"][0] != {"op": "spec"}:
                    raise RuntimeError("Native tape does not start with expected specification request")
                for request in tape["requests"][1:]:
                    bridge.call(request)
                    completed += 1
            except Exception as exc:
                error = str(exc)
            finally:
                bridge.close()
            report = {"exact": error is None and bridge.stream_digest.hexdigest() == tape["native_stream_sha256"],
                      "error": error, "requests_replayed": completed,
                      "expected_stream_sha256": tape["native_stream_sha256"],
                      "actual_stream_sha256": bridge.stream_digest.hexdigest(),
                      "tape_sha256": digest(folder / "native-tape.json.gz"), "wall_s": time.monotonic() - began}
            write_new(report_path, report)
            print(json.dumps({"native_source_replay": name, **report}), flush=True)
            if not report["exact"]:
                raise RuntimeError("Native source replay failed")
    elif args.command == "development-eval":
        status = json.loads((root / "elbow-development" / "result.json").read_text())
        if not status["complete"] or digest(root / "elbow-development" / "policy.zip") != status["policy_sha256"]:
            raise RuntimeError("Development reference incomplete or checkpoint altered")
        for i, config in enumerate(protocol["development_cases"]):
            item = {"id": f"elbow-development-{i}", "case": config}
            for arm in ["development_frozen", "quiet"]:
                target_trial(root, protocol, item, i, protocol["development_seed"], arm, development=True)
    else:
        # All final source checkpoints must exist before any target/probe run.
        for seed in protocol["source_seeds"]:
            status = json.loads((root / f"source-{seed}" / "result.json").read_text())
            if not status["complete"] or digest(root / f"source-{seed}" / "policy.zip") != status["policy_sha256"]:
                raise RuntimeError("Source training incomplete or checkpoint changed; do not open targets")
        if args.command == "targets":
            if args.seed not in protocol["source_seeds"]:
                raise ValueError("Seed is not in locked manifest")
            for i, item in enumerate(protocol["target_cases"]):
                for arm in ["frozen", "adapt", "scratch", "source_frozen"]:
                    target_trial(root, protocol, item, i, args.seed, arm)
        else:
            for i, item in enumerate(protocol["target_cases"]):
                for arm in ["quiet", "source_quiet", "negative", "positive", "rhythm_2s", "rhythm_4s"]:
                    target_trial(root, protocol, item, i, 0, arm)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Preserve setup/integration failures too, without overwriting any trial.
        import sys
        if "--output" in sys.argv:
            failure_root = Path(sys.argv[sys.argv.index("--output") + 1])
            if failure_root.is_dir():
                write_new(failure_root / f"execution-error-{time.time_ns()}.json",
                          {"arguments": sys.argv[1:], "traceback": traceback.format_exc()})
        raise
