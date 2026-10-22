"""Independent stdlib-only contract tests for the batched native transfer bridge.

Uses isolated stdin/stdout worlds, never HTTP or a live workshop session.
Run with the project's existing Python runtime; no packages are installed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import traceback

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_COMMAND = ["docker", "exec", "-i", "droid-blocks-dev", "/workspace/build/native/droid-transfer-env"]
REQUESTS = [-.15, 0., .15]
DECODER = json.JSONDecoder()


def check(value, message):
    if not value:
        raise AssertionError(message)


def near(actual, expected, message, tolerance=1e-11):
    check(isinstance(actual, (int, float)) and not isinstance(actual, bool) and math.isfinite(actual), message)
    check(abs(actual - expected) <= tolerance * max(1., abs(expected)), f"{message}: {actual!r} != {expected!r}")


def digest(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode("utf-8")).hexdigest()


def raw_members(text):
    """Extract native JSON values without rewriting floats or signed zeros."""
    text = text.strip()
    is_object = text.startswith("{")
    check(is_object or text.startswith("["), "Expected native JSON container")
    end_char = "}" if is_object else "]"
    result = {} if is_object else []
    position = 1
    while True:
        while text[position].isspace():
            position += 1
        if text[position] == end_char:
            break
        if is_object:
            key, position = DECODER.raw_decode(text, position)
            while text[position].isspace():
                position += 1
            check(text[position] == ":", "Malformed native object")
            position += 1
            while text[position].isspace():
                position += 1
        start = position
        _, position = DECODER.raw_decode(text, position)
        if is_object:
            check(key not in result, "Repeated JSON field")
            result[key] = text[start:position]
        else:
            result.append(text[start:position])
        while text[position].isspace():
            position += 1
        if text[position] == end_char:
            break
        check(text[position] == ",", "Malformed native separator")
        position += 1
    return result


class Bridge:
    def __init__(self, command):
        self.command = command
        self.lines = queue.Queue()
        self.stderr = []
        self.reply_count = 0
        self.process = subprocess.Popen(command, cwd=PROJECT, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, encoding="utf-8", bufsize=1,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self):
        for line in self.process.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def _read_stderr(self):
        for line in self.process.stderr:
            self.stderr.append(line)

    def call(self, value, *, ok=True, closing=False):
        self.process.stdin.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
        self.process.stdin.flush()
        try:
            line = self.lines.get(timeout=30)
        except queue.Empty as error:
            raise AssertionError("Native bridge did not reply within 30 seconds") from error
        check(line is not None, "Native bridge ended without a reply: " + "".join(self.stderr))
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as error:
            raise AssertionError("Non-JSON diagnostic was written to stdout: " + line[:200]) from error
        self.reply_count += 1
        check(type(reply.get("ok")) is bool and reply["ok"] is ok, f"Unexpected native reply: {reply}")
        check(set(reply) == ({"ok"} if closing else ({"ok", "result"} if ok else
              {"ok", "error", "batch_failed", "completed_slots_before_failure"})),
              "Bridge response has unexpected fields")
        if not ok:
            check(isinstance(reply["error"], str) and reply["error"], "Rejected command has no error")
            check(reply["batch_failed"] is False and reply["completed_slots_before_failure"] == [],
                  "Malformed request advanced part of the batch")
            return reply, line.strip(), None
        raw = raw_members(line)
        parts = raw_members(raw["result"]) if not closing and isinstance(reply["result"], list) else None
        return reply.get("result"), line.strip(), parts

    def close(self):
        if self.process.poll() is None:
            self.call({"op": "close"}, closing=True)
        self.process.wait(timeout=10)
        check(self.process.returncode == 0, "Native bridge exited unsuccessfully")
        tail = self.lines.get(timeout=10)
        check(tail is None, "Unexpected extra stdout after close: " + str(tail))
        self.process.stdin.close()
        self.process.stdout.close()
        self.process.stderr.close()

    def abort(self):
        if self.process.poll() is None:
            self.process.terminate()  # Only this test's docker-exec client.
            self.process.wait(timeout=10)


def assembly(hinge):
    return {"schema": "construction_kit_v2", "name": "Bridge contract fixture",
            "segments": [3, 3], "blocks": [],
            "sensor": {"segment": 1, "slot": 2, "side": -1},
            "light": {"segment": 1, "slot": 2, "side": 1, "face": 1},
            "powered_hinge": hinge}


def cases():
    return [{"assembly": assembly(hinge),
             "sun": {"position_m": [side * .3, 0., .2], "intensity_lux": 1000.}}
            for hinge in (0, 1) for side in (1, -1)]


def local_observation(observation, time_s):
    check(set(observation) == {"schema_version", "reward_sensors", "actuator_feedback"}, "Observation top-level leakage")
    check(observation["schema_version"] == "construction_observation_v2", "Wrong observation version")
    check(len(observation["reward_sensors"]) == 2 and len(observation["actuator_feedback"]) == 1, "Local interface count changed")
    sensors = {sensor["family_id"]: sensor for sensor in observation["reward_sensors"]}
    check(set(sensors) == {"imu_6axis_v0", "ambient_light_v0"}, "Unknown or repeated sensor")
    for family, period, latency, channels in (("imu_6axis_v0", .01, .006, {"specific_force_m_s2", "angular_velocity_rad_s"}),
                                              ("ambient_light_v0", .1, .02, {"illuminance_lux", "saturated"})):
        sensor = sensors[family]
        check(set(sensor) == {"module_id", "family_id", "valid", "sequence", "sample_time_s", "delivered_time_s",
                              "age_s", "sample_period_s", "latency_s", "observations"}, "Sensor metadata boundary changed")
        check(set(sensor["observations"]) == channels, "Unexpected sensor channel or geometry")
        near(sensor["sample_period_s"], period, "Sensor period")
        near(sensor["latency_s"], latency, "Sensor latency")
        if time_s == 0:
            check(sensor["valid"] is False and sensor["sequence"] == 0, "Initial sensor is not undelivered")
            check(all(sensor[k] is None for k in ("sample_time_s", "delivered_time_s", "age_s")), "Initial sensor timestamps")
        else:
            sequence = math.floor((time_s - latency + 1e-8) / period) + 1
            check(sensor["valid"] is True and sensor["sequence"] == sequence, "Sensor delivery cadence changed")
            near(sensor["sample_time_s"], (sequence - 1) * period, "Sensor acquisition time", 2e-6)
            near(sensor["delivered_time_s"], sensor["sample_time_s"] + latency, "Sensor delivery time", 2e-6)
            near(sensor["age_s"], time_s - sensor["sample_time_s"], "Sensor age", 2e-6)
    motor = observation["actuator_feedback"][0]
    check(set(motor) == {"module_id", "sku_id", "valid", "feedback"}, "Motor metadata boundary changed")
    check(motor["module_id"] == "motor-0001" and motor["sku_id"] == "rotary_dc_gearmotor_v0" and motor["valid"], "Motor identity/address changed")
    check(set(motor["feedback"]) == {"position_rad", "velocity_rad_s", "current_a", "bus_voltage_v", "temperature_c",
                                       "output_torque_nm", "load_impedance_nm_s_per_rad", "stuck_score", "stuck", "fault_flags"}, "Motor feedback leakage")


def verify_initial(result, config):
    local_observation(result["observation"], 0.)
    for field in ("reward", "previous_request", "executed_effort", "elapsed_s"):
        near(result[field], 0., "Initial " + field)
    check(result["terminated"] is False, "Initial stopped flag")
    check(result["physics"]["assembly"] == config["assembly"] and result["physics"]["sun"] == config["sun"], "Configured physical case differs")
    near(result["physics"]["elapsed_s"], 0., "Initial physical time")


def verify_decision(result, raw, before, action):
    check(result["native_steps"] == 5, "A nonterminal .1-second action did not execute five native controls")
    near(result["elapsed_s"], .1, "Decision duration")
    near(result["time_s"], before["physics"]["elapsed_s"] + .1, "Native decision clock")
    check(result["terminated"] is False, "Small contract probe unexpectedly stopped")
    check(result["safety"]["veto"] is False and not result["safety"]["flags"], "Small contract probe raised safety flags")
    check(len(result["efforts"]) == len(result["native_transitions"]) == 5, "Missing per-control audit evidence")
    raw_fields = raw_members(raw)
    raw_efforts = raw_members(raw_fields["efforts"])
    raw_transitions = raw_members(raw_fields["native_transitions"])
    audit = "".join('{"effort":' + effort + ',"transition":' + transition + '}\n'
                    for effort, transition in zip(raw_efforts, raw_transitions))
    check(digest(audit) == result["transition_sha256"], "Native transition digest does not bind the recorded control audit")
    observation = before["observation"]
    previous = before["executed_effort"]
    total_reward = light_reward = imu_reward = 0.
    maximum_current = maximum_speed = maximum_temperature = 0.
    for i, (effort, transition) in enumerate(zip(result["efforts"], result["native_transitions"])):
        request = REQUESTS[action]
        if any(not sensor["valid"] for sensor in observation["reward_sensors"]):
            governed = 0.
        else:
            if request:
                sign = 1. if request > 0 else -1.
                speed = observation["actuator_feedback"][0]["feedback"]["velocity_rad_s"]
                target = sign * min(abs(request), .05 * max(0., 3. - sign * speed))
            else:
                target = 0.
            governed = max(previous - .1, min(previous + .1, target))
        near(effort, governed, "Independent governor must refresh from each native observation", 1e-12)
        near(transition["info"]["elapsed_control_dt_s"], .02, "Native control dt")
        check(transition["info"]["physics_steps_executed"] == 10, "Native control must contain ten physics steps")
        check(transition["terminated"] is False and transition["truncated"] is False, "Unexpected native terminal flag")
        local_observation(transition["observation"], before["physics"]["elapsed_s"] + (i + 1) * .02)
        components = {component["family_id"]: component["transition_reward"] for component in transition["reward_components"]}
        check(set(components) == {"imu_6axis_v0", "ambient_light_v0"}, "Reward voters changed")
        near(transition["reward"], sum(components.values()), "Every native reward equals exact sensor-component sum", 1e-13)
        total_reward += transition["reward"]
        light_reward += components["ambient_light_v0"]
        imu_reward += components["imu_6axis_v0"]
        feedback = transition["observation"]["actuator_feedback"][0]["feedback"]
        maximum_current = max(maximum_current, abs(feedback["current_a"]))
        maximum_speed = max(maximum_speed, abs(feedback["velocity_rad_s"]))
        maximum_temperature = max(maximum_temperature, feedback["temperature_c"])
        observation, previous = transition["observation"], effort
    check(result["observation"] == observation, "Bridge endpoint is not the last native local observation")
    for field, expected in (("reward", total_reward), ("light_reward", light_reward), ("imu_reward", imu_reward),
                            ("max_current_a", maximum_current), ("max_speed_rad_s", maximum_speed),
                            ("max_temperature_c", maximum_temperature), ("executed_effort", previous),
                            ("previous_request", REQUESTS[action])):
        near(result[field], expected, "Decision aggregate " + field, 1e-13)
    near(result["reward"], result["light_reward"] + result["imu_reward"], "Decision component sum", 1e-13)
    if before["physics"]["elapsed_s"] == 0.:
        delivered = next(sensor for sensor in result["native_transitions"][0]["observation"]["reward_sensors"]
                         if sensor["family_id"] == "ambient_light_v0")
        near(delivered["observations"]["illuminance_lux"],
             before["physics"]["diagnostics"]["instantaneous_light_lux"],
             "First delivered sample must be acquired under the configured sun", 2e-9)


def run(command):
    started = time.monotonic()
    bridges = [Bridge(command) for _ in range(3)]
    main, reference, fresh = bridges
    checks = []
    try:
        spec = main.call({"op": "spec"})[0]
        check(spec["schema"] == "actuation_transfer_bridge_v1", "Bridge schema")
        near(spec["native_dt_s"], .02, "Spec native dt")
        near(spec["decision_dt_s"], .1, "Spec decision dt")
        check(spec["native_steps_per_decision"] == 5 and spec["requests"] == REQUESTS, "Action spec")
        manifest = cases()
        initial, raw_initial, _ = main.call({"op": "configure", "cases": manifest, "record": True})
        ref_initial, ref_raw, _ = reference.call({"op": "configure", "cases": manifest, "record": True})
        check(raw_initial == ref_raw, "Independent configure bytes differ")
        for result, config in zip(initial, manifest):
            verify_initial(result, config)
        checks.append("four isolated hinge/source configurations and strict local observation boundary")
        before = initial
        decision_records = []
        action_tape = [[1, 1, 1, 1]] + [[(i + slot) % 3 for slot in range(4)] for i in range(12)]
        for actions in action_tape:
            results, raw, raw_results = main.call({"op": "step", "actions": actions, "record": True})
            ref_results, ref_raw, _ = reference.call({"op": "step", "actions": actions, "record": True})
            check(raw == ref_raw, "Independent full bridge replay bytes differ")
            for result, encoded, previous, action in zip(results, raw_results, before, actions):
                verify_decision(result, encoded, previous, action)
            decision_records.append(raw)
            before = results
        checks.append("five native controls per decision; governor, acquisition clocks, component rewards and audit digests independently checked")
        checks.append("fresh configured-sun sample and exact independent native replay for both powered hinges")

        def rejects_without_advancing(value, label):
            nonlocal before
            main.call(value, ok=False)
            actions = [1] * 4
            results, raw, raw_results = main.call({"op": "step", "actions": actions, "record": True})
            _, ref_raw, _ = reference.call({"op": "step", "actions": actions, "record": True})
            check(raw == ref_raw, "Rejected " + label + " changed at least one world or governor history")
            for result, encoded, previous in zip(results, raw_results, before):
                verify_decision(result, encoded, previous, 1)
            before = results

        malformed_actions = [[], [1], [1] * 5, [1, 1, 1, 3], [1, 1, 1, -1], [1, 1, 1, 1.0],
                             [1, 1, 1, True], [1, 1, 1, None], [1, 1, 1, "1"], [1, 1, 1, []],
                             [1, 1, 1, 4294967296], [1, 1, 1, 2**63]]
        for actions in malformed_actions:
            rejects_without_advancing({"op": "step", "actions": actions}, "action batch")
        checks.append(f"all {len(malformed_actions)} malformed action batches rejected before any world advances")
        malformed_resets = [None, {}, [{"index": 0}, {"index": 0}], [{"index": 0}, {"index": 4}],
                            [{"index": 0}, {"index": -1}], [{"index": 0}, {"index": True}],
                            [{"index": 0}, {"index": 1.0}], [{"index": 0}, {}]]
        invalid_case = copy.deepcopy(manifest[0])
        invalid_case["assembly"]["powered_hinge"] = 2
        malformed_resets.append([{"index": 0}, {"index": 1, "case": invalid_case}])
        bad_sun = copy.deepcopy(manifest[0])
        bad_sun["sun"]["position_m"][1] = .1
        malformed_resets.append([{"index": 0}, {"index": 1, "case": bad_sun}])
        for replacements in malformed_resets:
            rejects_without_advancing({"op": "reset", "slots": replacements}, "reset batch")
        for invalid in ([], [manifest[0], invalid_case], [manifest[0], bad_sun]):
            rejects_without_advancing({"op": "configure", "cases": invalid}, "configure batch")
        checks.append(f"all {len(malformed_resets)} malformed reset and three configure batches are transactional across every slot")

        replacement = copy.deepcopy(manifest[2])
        replacement["sun"] = {"position_m": [-.4, 0., .3], "intensity_lux": 800.}
        reset = main.call({"op": "reset", "slots": [{"index": 2, "case": replacement}], "record": True})[0]
        fresh_initial = fresh.call({"op": "configure", "cases": [replacement], "record": True})[0][0]
        check(len(reset) == 1 and reset[0]["index"] == 2 and reset[0]["initial"] == fresh_initial, "Single-slot reset not equivalent to independent fresh world")
        before[2] = fresh_initial
        for _ in range(3):
            actions = [1] * 4
            results, _, raw_results = main.call({"op": "step", "actions": actions, "record": True})
            references, _, ref_raw_results = reference.call({"op": "step", "actions": actions, "record": True})
            new_results, _, fresh_raw = fresh.call({"op": "step", "actions": [1], "record": True})
            for slot in (0, 1, 3):
                check(raw_results[slot] == ref_raw_results[slot], "Resetting one slot altered another world's exact continuation")
            check(raw_results[2] == fresh_raw[0], "Reset target does not match its independent fresh continuation")
            for result, encoded, previous in zip(results, raw_results, before):
                verify_decision(result, encoded, previous, 1)
            before = results
        checks.append("one-slot source reset leaves every other world and governor history byte-identical")

        _, repeated_initial, _ = main.call({"op": "configure", "cases": manifest, "record": True})
        check(repeated_initial == raw_initial, "Repeated configure retained prior physical or sensor state")
        for actions, expected_raw in zip(action_tape, decision_records):
            _, raw, _ = main.call({"op": "step", "actions": actions, "record": True})
            check(raw == expected_raw, "Reconfigure/replay did not reproduce the original exact native trace")
        checks.append("full reconfigure replay reproduces every observation, reward, physical state and transition digest")
        for bridge in bridges:
            bridge.close()
        checks.append("stdout contains exactly one JSON response per request and no trailing diagnostics")
        return {"schema": "transfer_bridge_test_results_v1", "complete": True, "passed": True,
                "checks": checks, "check_count": len(checks), "malformed_action_cases": len(malformed_actions),
                "malformed_reset_cases": len(malformed_resets), "malformed_configure_cases": 3,
                "independent_processes": 3, "reply_count": sum(b.reply_count for b in bridges),
                "stderr": ["".join(b.stderr) for b in bridges], "seconds": time.monotonic() - started}
    finally:
        for bridge in bridges:
            bridge.abort()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT / "build/actuation-transfer-qa/bridge-tests.json")
    parser.add_argument("--command", nargs="+", default=DEFAULT_COMMAND)
    args = parser.parse_args()
    try:
        result = run(args.command)
    except Exception as error:
        result = {"schema": "transfer_bridge_test_results_v1", "complete": False, "passed": False,
                  "error": str(error), "traceback": traceback.format_exc()}
    result["command"] = args.command
    result["python"] = sys.version
    result["source_sha256"] = {name: digest((PROJECT / name).read_bytes()) for name in
                               ("tests/check_transfer_bridge.py", "src/transfer_env_cli.cpp", "training/native_transfer.py")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        previous = args.output.read_bytes()
        preserved = args.output.with_name(args.output.stem + ".previous-" + digest(previous)[:12] + args.output.suffix)
        if not preserved.exists():
            preserved.write_bytes(previous)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
