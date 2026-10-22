"""Non-learning integration checks and throughput pilot; no target exposure."""
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(ROOT / "build" / "transfer-matplotlib")
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(ROOT / "training"))
import numpy as np
from actuation_transfer import case, build_model, manifest
from native_transfer import NativeTransferVecEnv, local_frame


def main():
    checks = []
    config = case((3, 3), False, -1, 0)
    env = NativeTransferVecEnv([config], horizon=4, record=True)
    try:
        obs = env.reset()
        assert obs.shape == (1, 204) and obs.dtype == np.float32
        assert np.count_nonzero(obs[0, :187]) == 0
        assert obs[0, -1] == 1 and obs[0, -3:-1].tolist() == [0., 0.]
        checks.append("204-dimensional local stack with explicit absent and invalid masks")
        initial_frame = local_frame(env.latest[0])
        changed = copy.deepcopy(env.latest[0])
        changed["physics"] = {"powered_hinge": 99, "sun": [999, 999, 999], "hidden_state": "forbidden"}
        changed["case_id"] = "another-body"
        assert np.array_equal(initial_frame, local_frame(changed))
        checks.append("diagnostic geometry, source and identity cannot enter frame projection")
        for invalid in [np.array([1.]), np.array([True]), np.array([3]), np.array([-1])]:
            try:
                env.step_async(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid action dtype/value accepted")
        checks.append("Python and native discrete-action contracts agree")
        for i in range(4):
            obs, reward, done, infos = env.step(np.array([1], dtype=np.int64))
            assert infos[0]["native"]["native_steps"] == 5
            assert abs(float(reward[0]) - infos[0]["native"]["reward"]) < 1e-7
            assert bool(done[0]) == (i == 3)
        terminal = infos[0]["terminal_observation"]
        assert infos[0]["TimeLimit.truncated"] is True
        assert terminal[-3:-1].tolist() == [1., 1.]
        assert obs[0, -3:-1].tolist() == [0., 0.]
        assert np.count_nonzero(obs[0, :187]) == 0
        checks.append("timeout exposes actual terminal stack before resetting next episode history")
        model = build_model(env, manifest(), 999, target=True)
        actions, _ = model.predict(obs, deterministic=False)
        assert actions.dtype.kind in "iu" and actions.shape == (1,)
        assert all(p.isfinite().all() for p in model.policy.parameters())
        assert model.num_timesteps == 0
        checks.append("SB3 policy and value initialize and infer without any learning")
    finally:
        env.close()
    bench = NativeTransferVecEnv([config] * 8)
    try:
        bench.reset()
        rng = np.random.default_rng(777)
        began = time.monotonic()
        for _ in range(256):
            obs, _, _, _ = bench.step(rng.integers(0, 3, size=8, dtype=np.int64))
            assert obs.shape == (8, 204) and np.isfinite(obs).all()
        duration = time.monotonic() - began
        throughput = {"decisions": bench.total_interactions, "native_steps": bench.total_native_steps,
                      "wall_s": duration, "decisions_per_second": bench.total_interactions / duration,
                      "safety_stops": bench.safety_stops, "parameter_updates": 0,
                      "case_scope": "source-domain base-motor [3,3], no blocks, left source; separate pilot"}
    finally:
        bench.close()
    output = {"pass": True, "checks": checks, "throughput": throughput}
    path = ROOT / "build" / "actuation-transfer-qa" / "adapter-tests.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
