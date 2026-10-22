"""Predeclared reproduction of the SB3 2.9.0 CartPole PPO example.

Reference: https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/ppo.html
Four CartPole-v1 environments, MlpPolicy, standard PPO hyperparameters, seed 0.
Request 25,000 transitions once; the 8,192-transition rollout size rounds this
to 32,768. Evaluate untrained/trained/reloaded policies deterministically on
the same twenty seeds 1000..1019. No tuning, early stopping, or extra training.
Engineering gate declared before execution: trained mean return >= 200, mean
gain over the untrained policy >= 100, finite parameters, and exact equality of
checkpoint parameters, policy outputs and seeded evaluation after save/reload.
This is dependency/reference validation, not evidence of Droid Blocks transfer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["MPLCONFIGDIR"] = str(Path(__file__).resolve().parents[1] / "build" / "actuation-transfer-reference" / "matplotlib-cache")

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT / "build" / "actuation-transfer-reference"
EXPECTED_VERSIONS = {
    "stable-baselines3": "2.9.0",
    "gymnasium": "1.2.3",
    "torch": "2.8.0+cpu",
    "numpy": "2.2.6",
}
PROTOCOL = {
    "schema": "sb3_cartpole_reference_protocol_v1",
    "reference": "https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/ppo.html",
    "scope": "Reference implementation and CPU runtime validation; no native Droid Blocks simulator involved",
    "environment": "CartPole-v1",
    "train_seed": 0,
    "evaluation_seeds": list(range(1000, 1020)),
    "n_envs": 4,
    "requested_timesteps": 25_000,
    "expected_actual_timesteps": 32_768,
    "torch_threads": 1,
    "torch_interop_threads": 1,
    "deterministic_training_operations": True,
    "deterministic_evaluation": True,
    "ppo": {
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "normalize_advantage": True,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_sde": False,
    },
    "policy": {"type": "MlpPolicy", "pi_hidden": [64, 64], "vf_hidden": [64, 64], "activation": "Tanh"},
    "gate": {"trained_mean_return_at_least": 200.0, "mean_gain_over_untrained_at_least": 100.0,
             "exact_reload_required": True, "finite_parameters_required": True},
    "outcome_policy": "One fixed run. Preserve failure or weak performance; no post-result tuning or extra training.",
    "versions": EXPECTED_VERSIONS,
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    target = path if exclusive else path.with_suffix(path.suffix + ".next")
    with target.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if not exclusive:
        target.replace(path)


def package_record() -> dict:
    packages = []
    for dist in sorted(metadata.distributions(), key=lambda entry: entry.metadata["Name"].lower()):
        item = dist.metadata
        packages.append({"name": item["Name"], "version": dist.version,
                         "license_expression": item.get("License-Expression"), "license": item.get("License"),
                         "license_files": item.get_all("License-File") or [],
                         "license_classifiers": [text for text in item.get_all("Classifier", []) if text.startswith("License ::")]})
    return {"executable": sys.executable, "prefix": sys.prefix, "python": sys.version,
            "platform": platform.platform(), "torch_cuda": torch.version.cuda,
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "packages": packages}


def parameter_hash(model: PPO) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        assert np.isfinite(array).all(), f"Nonfinite parameter {name}"
        hasher.update(name.encode("utf-8"))
        hasher.update(str(array.dtype).encode("ascii"))
        hasher.update(json.dumps(list(array.shape)).encode("ascii"))
        hasher.update(array.tobytes())
    return hasher.hexdigest()


def evaluate(model: PPO, label: str) -> tuple[dict, list]:
    episodes = []
    traces = []
    for seed in PROTOCOL["evaluation_seeds"]:
        environment = gym.make(PROTOCOL["environment"])
        try:
            observation, _ = environment.reset(seed=seed)
            episode = {"seed": seed, "return": 0.0, "steps": 0, "terminated": False, "truncated": False}
            trace = {"seed": seed, "initial_observation": observation.tolist(), "transitions": []}
            while not episode["terminated"] and not episode["truncated"]:
                assert np.isfinite(observation).all(), "Nonfinite reference observation"
                action, _ = model.predict(observation, deterministic=True)
                next_observation, reward, terminated, truncated, _ = environment.step(int(action))
                assert math.isfinite(float(reward)), "Nonfinite reference reward"
                trace["transitions"].append({"observation": observation.tolist(), "action": int(action),
                    "next_observation": next_observation.tolist(), "reward": float(reward),
                    "terminated": bool(terminated), "truncated": bool(truncated)})
                episode["return"] += float(reward)
                episode["steps"] += 1
                episode["terminated"] = bool(terminated)
                episode["truncated"] = bool(truncated)
                assert episode["steps"] <= 500, "CartPole-v1 time limit changed"
                observation = next_observation
            episodes.append(episode)
            traces.append(trace)
        finally:
            environment.close()
    returns = np.asarray([episode["return"] for episode in episodes], dtype=np.float64)
    return {"label": label, "episodes": episodes, "mean_return": float(returns.mean()),
            "population_std_return": float(returns.std()), "min_return": float(returns.min()),
            "max_return": float(returns.max())}, traces


class Progress(BaseCallback):
    def _on_step(self) -> bool:
        assert self.model.num_timesteps <= PROTOCOL["expected_actual_timesteps"], "Reference step budget exceeded"
        return True

    def _on_rollout_end(self) -> None:
        print(f"Reference PPO collected {self.model.num_timesteps}/{PROTOCOL['expected_actual_timesteps']} transitions", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "cartpole-v1")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(OUTPUT_ROOT.resolve()) or output == OUTPUT_ROOT.resolve():
        parser.error("Output must be a new subdirectory under build/actuation-transfer-reference")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = {"schema": "sb3_cartpole_reference_v1", "complete": False, "stage": "declared",
              "started_utc": datetime.now(timezone.utc).isoformat(), "protocol": PROTOCOL, "errors": []}
    training_env = None
    try:
        for name, expected in EXPECTED_VERSIONS.items():
            assert metadata.version(name) == expected, f"Pinned version differs: {name}"
        assert Path(sys.prefix).resolve() == (PROJECT / "build" / "transfer-venv").resolve(), "Use the isolated project training interpreter"
        assert torch.version.cuda is None, "The reference requires the CPU-only PyTorch build"
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        save_json(output / "protocol.json", PROTOCOL, exclusive=True)
        report["protocol_sha256"] = digest(output / "protocol.json")
        report["runtime"] = package_record()
        source = output / "source"
        source.mkdir()
        report["sources_sha256"] = {}
        for name in ("reference_ppo.py", "setup-transfer.ps1", "requirements-transfer.txt"):
            source_path = PROJECT / "training" / name
            shutil.copyfile(source_path, source / name)
            report["sources_sha256"][f"training/{name}"] = digest(source_path)
        save_json(output / "report.json", report)
        training_env = make_vec_env(PROTOCOL["environment"], n_envs=PROTOCOL["n_envs"], seed=PROTOCOL["train_seed"],
                                    monitor_dir=str(output / "training-monitor"))
        model = PPO("MlpPolicy", training_env, seed=PROTOCOL["train_seed"], device="cpu", verbose=0,
                    policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}, "activation_fn": torch.nn.Tanh},
                    **PROTOCOL["ppo"])
        model.set_logger(configure(str(output / "training-log"), ["csv"]))
        report["initial_parameter_sha256"] = parameter_hash(model)
        report["untrained"], traces = evaluate(model, "untrained")
        save_json(output / "evaluation-untrained.json", traces, exclusive=True)
        report["stage"] = "training"
        save_json(output / "report.json", report)
        train_start = time.perf_counter()
        model.learn(total_timesteps=PROTOCOL["requested_timesteps"], callback=Progress(), progress_bar=False)
        report["training_elapsed_s"] = time.perf_counter() - train_start
        report["actual_timesteps"] = model.num_timesteps
        assert model.num_timesteps == PROTOCOL["expected_actual_timesteps"], "Actual rollout rounding differs"
        report["trained_parameter_sha256"] = parameter_hash(model)
        report["trained"], trained_traces = evaluate(model, "trained")
        save_json(output / "evaluation-trained.json", trained_traces, exclusive=True)
        checkpoint = output / "ppo-cartpole.zip"
        model.save(checkpoint)
        report["checkpoint_sha256"] = digest(checkpoint)
        report["stage"] = "reload_check"
        save_json(output / "report.json", report)
        reloaded = PPO.load(checkpoint, device="cpu")
        exact_parameters = all(torch.equal(tensor, reloaded.policy.state_dict()[name]) for name, tensor in model.policy.state_dict().items())
        all_observations = np.asarray([step["observation"] for trace in trained_traces for step in trace["transitions"]], dtype=np.float32)
        actions_before, _ = model.predict(all_observations, deterministic=True)
        actions_after, _ = reloaded.predict(all_observations, deterministic=True)
        with torch.no_grad():
            tensor_observations = torch.as_tensor(all_observations)
            probabilities_before = model.policy.get_distribution(tensor_observations).distribution.probs
            probabilities_after = reloaded.policy.get_distribution(tensor_observations).distribution.probs
        report["reloaded"], reloaded_traces = evaluate(reloaded, "reloaded")
        save_json(output / "evaluation-reloaded.json", reloaded_traces, exclusive=True)
        report["checks"] = {
            "finite_parameters": True,
            "parameters_changed_during_training": report["initial_parameter_sha256"] != report["trained_parameter_sha256"],
            "exact_reloaded_parameters": exact_parameters,
            "exact_reloaded_parameter_hash": parameter_hash(reloaded) == report["trained_parameter_sha256"],
            "exact_reloaded_actions": bool(np.array_equal(actions_before, actions_after)),
            "exact_reloaded_action_probabilities": bool(torch.equal(probabilities_before, probabilities_after)),
            "exact_seeded_evaluation_traces": trained_traces == reloaded_traces,
            "exact_seeded_evaluation_file_bytes": digest(output / "evaluation-trained.json") == digest(output / "evaluation-reloaded.json"),
            "checkpoint_unchanged_after_reload": digest(checkpoint) == report["checkpoint_sha256"],
            "protocol_unchanged": digest(output / "protocol.json") == report["protocol_sha256"],
            "source_unchanged": all(digest(PROJECT / name) == expected for name, expected in report["sources_sha256"].items()),
        }
        report["reload_observations_checked"] = len(all_observations)
        report["mean_return_gain"] = report["trained"]["mean_return"] - report["untrained"]["mean_return"]
        report["gate"] = {
            "trained_mean_return_at_least_200": report["trained"]["mean_return"] >= 200,
            "mean_gain_at_least_100": report["mean_return_gain"] >= 100,
            "all_integrity_checks": all(report["checks"].values()),
        }
        report["passed"] = all(report["gate"].values())
        report["complete"] = True
        report["stage"] = "completed"
        report["elapsed_s"] = time.perf_counter() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_json(output / "report.json", report)
        print(json.dumps({"complete": True, "passed": report["passed"], "actual_timesteps": model.num_timesteps,
                          "untrained_mean_return": report["untrained"]["mean_return"], "trained_mean_return": report["trained"]["mean_return"],
                          "mean_return_gain": report["mean_return_gain"], "reload_observations_checked": len(all_observations),
                          "elapsed_s": report["elapsed_s"], "report": str(output / "report.json")}, indent=2), flush=True)
        return 0 if report["passed"] else 2
    except Exception as error:
        report["errors"].append({"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        report["elapsed_s"] = time.perf_counter() - started
        save_json(output / "report.json", report)
        raise
    finally:
        if training_env is not None:
            training_env.close()


if __name__ == "__main__":
    raise SystemExit(main())
