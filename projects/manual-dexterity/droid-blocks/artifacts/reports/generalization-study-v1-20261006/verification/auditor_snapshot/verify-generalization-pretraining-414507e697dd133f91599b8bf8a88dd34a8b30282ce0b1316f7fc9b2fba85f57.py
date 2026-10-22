#!/usr/bin/env python3
"""Independent standard-library accounting and native replay audit.

--output names the existing report directory; --receipt must name a new file.
This never imports the learner, adapter, curriculum, Torch or SB3. Native replay
checks physical requests/responses, not neural inference or optimizer execution.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import queue
import random
import re
import subprocess
import sys
import threading


PROJECT = Path(__file__).resolve().parents[1]
NATIVE = Path("/workspace/build/native/droid-transfer-env")
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
ARMS = ("frozen", "untrained", "quiet")
WINDOWS = (64, 128, 320, 640)
HEX = re.compile(r"[0-9a-f]{64}\Z")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def parse(data):
    def invalid(value):
        raise ValueError(f"non-finite JSON constant {value}")
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            invalid(value)
        return number
    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite_float)


def load(path, compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, "rb") as stream:
        data = stream.read(MAX_JSON_BYTES + 1)
    if len(data) > MAX_JSON_BYTES:
        raise ValueError(f"JSON exceeds bounded parsing limit: {path.name}")
    return parse(data.decode("utf-8"))


def safe_path(base, name):
    if not isinstance(name, str) or "\\" in name:
        raise ValueError("evidence paths must use relative POSIX components")
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or any(part in (".", "..") for part in relative.parts):
        raise ValueError(f"unsafe evidence path: {name!r}")
    target = base.joinpath(*relative.parts)
    cursor = target
    while cursor != base:
        if cursor.is_symlink():
            raise ValueError(f"symlink in evidence path: {name}")
        cursor = cursor.parent
    target.resolve().relative_to(base.resolve())
    return target


def mechanics(case):
    if set(case) != {"assembly", "sun"}:
        raise ValueError("case must contain exactly assembly and sun")
    a = case["assembly"]
    if set(a) != {"schema", "name", "segments", "powered_hinge", "blocks", "sensor", "light"}:
        raise ValueError("unexpected declared assembly fields")
    if not isinstance(a["name"], str) or len(a["name"]) > 60:
        raise ValueError("invalid assembly label")
    a["name"].encode("utf-8")
    return canonical({"schema": a["schema"], "segments": a["segments"],
                      "powered_hinge": a["powered_hinge"],
                      "blocks": sorted([b["segment"], b["slot"], b["side"]] for b in a["blocks"]),
                      "sensor": a["sensor"], "light": a["light"]})


def fixture(lengths, weighted, hinge, side):
    return {"assembly": {"schema": "construction_kit_v2", "name": "",
                         "segments": list(lengths), "powered_hinge": hinge,
                         "blocks": ([{"id": "block-1", "segment": 1, "slot": 0, "side": -1}] if weighted else []),
                         "sensor": {"segment": 1, "slot": 2, "side": -1},
                         "light": {"segment": 1, "slot": 2, "side": 1, "face": 1}},
            "sun": {"position_m": [side * .3, 0., -.3], "intensity_lux": 1000.}}


class Audit:
    def __init__(self):
        self.checks = self.failed = 0
        self.issues = []

    def check(self, condition, message):
        self.checks += 1
        if not condition:
            self.failed += 1
            if len(self.issues) < 1000:
                self.issues.append(message)
        return bool(condition)

    def equal(self, actual, expected, label):
        if type(expected) is dict:
            if not self.check(type(actual) is dict, f"{label}: expected object"):
                return
            self.check(set(actual) == set(expected), f"{label}: fields differ")
            for key in set(actual) & set(expected):
                self.equal(actual[key], expected[key], f"{label}.{key}")
        elif type(expected) is list:
            if not self.check(type(actual) is list, f"{label}: expected array"):
                return
            self.check(len(actual) == len(expected), f"{label}: length differs")
            for index, (a, b) in enumerate(zip(actual, expected)):
                self.equal(a, b, f"{label}[{index}]")
        elif type(expected) is float:
            self.check(type(actual) in (int, float) and math.isfinite(actual)
                       and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), f"{label}: numeric value differs")
        else:
            self.check(type(actual) is type(expected) and actual == expected, f"{label}: value differs")

    def file(self, base, record, key="sha256", path_key="path"):
        path = safe_path(base, record[path_key])
        self.check(bool(HEX.fullmatch(str(record[key]))), f"{path.name}: invalid SHA-256 label")
        self.check(path.is_file(), f"{path.name}: missing retained file")
        if path.is_file():
            self.check(digest(path) == record[key], f"{path.name}: retained bytes differ")
        return path


def check_curriculum(audit, curriculum):
    source = [fixture(lengths, weighted, hinge, side)
              for lengths in ((3, 3), (4, 3), (3, 4), (4, 4))
              for weighted in (False, True) for hinge in (0, 1) for side in (-1, 1)]
    layouts = (((4, 3), False), ((3, 4), False), ((4, 4), True))
    development = [fixture(lengths, weighted, hinge, side) for lengths, weighted in layouts
                   for hinge in (0, 1) for side in (-1, 1)]
    reserved = [fixture(lengths, weighted, hinge, side)
                for lengths, weighted in (((5, 3), False), ((3, 5), False), ((5, 5), True))
                for hinge in (0, 1) for side in (-1, 1)]
    audit.equal(curriculum["schema"], "generalization_curriculum_v1", "curriculum.schema")
    for key, expected in (("source_cases", source), ("development_cases", development), ("reserved_cases", reserved)):
        actual = curriculum[key]
        audit.check(len(actual) == len(expected), f"{key}: incorrect declared condition count")
        for index, (case, wanted) in enumerate(zip(actual, expected)):
            audit.equal(mechanics(case), mechanics(wanted), f"{key}[{index}].mechanics")
            audit.equal(case["sun"], wanted["sun"], f"{key}[{index}].sun")
    source_bodies = {mechanics(case) for case in curriculum["source_cases"]}
    reserved_bodies = {mechanics(case) for case in curriculum["reserved_cases"]}
    audit.check(not source_bodies & reserved_bodies, "source/reserved whole-assembly mechanics overlap")
    audit.check({mechanics(c) for c in curriculum["development_cases"]} <= source_bodies,
                "development mechanics are not known source mechanics")
    source_ids = [f"source-L{a}-{b}-{'distal-block' if weighted else 'plain'}-"
                  f"{'base' if hinge == 0 else 'elbow'}-{'left' if side < 0 else 'right'}"
                  for a, b in ((3, 3), (4, 3), (3, 4), (4, 4)) for weighted in (False, True)
                  for hinge in (0, 1) for side in (-1, 1)]
    other_ids = lambda prefix: [f"{prefix}-{name}-{'base' if hinge == 0 else 'elbow'}-"
                               f"{'left' if side < 0 else 'right'}"
                               for name in ("long-upper", "long-lower", "weighted")
                               for hinge in (0, 1) for side in (-1, 1)]
    audit.equal(curriculum["case_ids"], {"source": source_ids, "development": other_ids("development"),
                                       "reserved": other_ids("reserved")}, "declared case identities")
    audit.equal(curriculum["pilot"], {"seed": 404, "source_decisions": 16384, "n_envs": 8, "n_steps": 256,
                "horizon_decisions": 640, "development_decisions_per_arm": 640, "development_arms": list(ARMS)}, "pilot")
    audit.equal(curriculum["development_gate"]["native_reward_gain_strictly_greater_than"], 3.2, "declared competence margin")
    audit.equal(curriculum["decision_dt_s"], .1, "decision interval")
    audit.equal(curriculum["native_dt_s"], .02, "native interval")
    audit.equal(curriculum["sampling"]["schema"], "shuffled_source_cycles_v1", "sampler schema")
    audit.equal(curriculum["sampling"]["curriculum_seed_offset"], 500000, "sampler seed offset")
    for key, value in {"source_n_steps": 256, "source_n_envs": 8, "n_epochs": 4, "batch_size": 64}.items():
        audit.equal(curriculum["ppo"][key], value, f"ppo.{key}")


def compact(value):
    names = ("reward", "light_reward", "imu_reward", "previous_request", "executed_effort", "terminated",
             "elapsed_s", "native_steps", "time_s", "energy_j", "max_current_a", "max_speed_rad_s", "max_temperature_c", "safety")
    return {**{key: value[key] for key in names},
            "sampled_local": {key: value["observation"][key] for key in ("reward_sensors", "actuator_feedback")}}


def metrics(rows):
    total = {"decisions": len(rows), "native_steps": 0, "duration_s": 0., "native_reward": 0.,
             "light_reward": 0., "imu_reward": 0., "energy_j": 0., "safety_stops": 0,
             "max_current_a": 0., "max_speed_rad_s": 0., "max_temperature_c": 0.,
             "sampled_invalid_imu": 0, "sampled_invalid_light": 0, "sampled_light_lux_sum": 0.,
             "sampled_valid_light_count": 0, "sampled_max_light_lux": 0., "sampled_feedback_flag_count": 0}
    for row in rows:
        value = row["native"]
        for result_key, native_key in (("native_steps", "native_steps"), ("duration_s", "elapsed_s"),
                                      ("native_reward", "reward"), ("light_reward", "light_reward"), ("imu_reward", "imu_reward")):
            total[result_key] += value[native_key]
        total["energy_j"] = value["energy_j"]
        total["safety_stops"] += int(value["terminated"])
        for key in ("max_current_a", "max_speed_rad_s", "max_temperature_c"):
            total[key] = max(total[key], value[key])
        sensors = {entry["family_id"]: entry for entry in value["sampled_local"]["reward_sensors"]}
        total["sampled_invalid_imu"] += int(not sensors["imu_6axis_v0"]["valid"])
        light = sensors["ambient_light_v0"]
        total["sampled_invalid_light"] += int(not light["valid"])
        if light["valid"]:
            lux = light["observations"]["illuminance_lux"]
            total["sampled_light_lux_sum"] += lux
            total["sampled_valid_light_count"] += 1
            total["sampled_max_light_lux"] = max(total["sampled_max_light_lux"], lux)
        total["sampled_feedback_flag_count"] += len(value["sampled_local"]["actuator_feedback"][0]["feedback"]["fault_flags"])
    total["sampled_mean_light_lux"] = (total["sampled_light_lux_sum"] / total["sampled_valid_light_count"]
                                     if total["sampled_valid_light_count"] else None)
    return total


def native_check(audit, value, label, previous_time=0.):
    audit.check(type(value["native_steps"]) is int and 1 <= value["native_steps"] <= 5, f"{label}: invalid native count")
    audit.equal(value["elapsed_s"], .02 * value["native_steps"], f"{label}.elapsed_s")
    audit.equal(value["time_s"], previous_time + value["elapsed_s"], f"{label}.time_s")
    audit.equal(value["reward"], value["light_reward"] + value["imu_reward"], f"{label}.component sum")
    flags = value["sampled_local"]["actuator_feedback"][0]["feedback"]["fault_flags"]
    audit.equal(value["terminated"], bool(value["safety"]["veto"] or flags), f"{label}.stop accounting")
    if not value["terminated"]:
        audit.equal(value["native_steps"], 5, f"{label}.complete decision")


def request_schedule(audit, tape, permitted, slots, source, label):
    """Validate all requests before any native replay; recover exact task draws."""
    current, draws, actions = [], [], []
    lookup = {canonical(case): index for index, case in enumerate(permitted)}
    requests = tape["requests"]
    audit.check(len(requests) <= 20000, f"{label}: too many retained requests")
    for number, request in enumerate(requests + ([tape["unresolved_request"]] if tape["unresolved_request"] else [])):
        if type(request) is not dict:
            raise ValueError(f"{label}: malformed request {number}")
        op = request.get("op")
        if request.get("record", False) is not False:
            raise ValueError(f"{label}: record=True differs from retained tape contract")
        if op == "spec":
            if request != {"op": "spec"} or number != 0:
                raise ValueError(f"{label}: unexpected spec request")
        elif op == "configure":
            if set(request) != {"op", "cases", "record"} or len(request["cases"]) != slots:
                raise ValueError(f"{label}: invalid configure batch")
            current = []
            for case in request["cases"]:
                identity = canonical(case)
                if identity not in lookup:
                    raise ValueError(f"{label}: configuration is outside the permitted declared partition")
                current.append(lookup[identity])
                draws.append(lookup[identity])
        elif op == "reset":
            if not source or set(request) != {"op", "slots", "record"} or not current:
                raise ValueError(f"{label}: unauthorized reset")
            seen = set()
            for item in request["slots"]:
                if set(item) != {"index", "case"} or type(item["index"]) is not int or item["index"] in seen or not 0 <= item["index"] < slots:
                    raise ValueError(f"{label}: malformed reset slot")
                seen.add(item["index"])
                identity = canonical(item["case"])
                if identity not in lookup:
                    raise ValueError(f"{label}: reset case is outside the permitted source partition")
                current[item["index"]] = lookup[identity]
                draws.append(lookup[identity])
        elif op == "step":
            if set(request) != {"op", "actions", "record"} or not current or len(request["actions"]) != slots:
                raise ValueError(f"{label}: malformed action batch")
            if any(type(action) is not int or action not in (0, 1, 2) for action in request["actions"]):
                raise ValueError(f"{label}: invalid discrete action")
            actions.append((list(current), request["actions"]))
        else:
            raise ValueError(f"{label}: unexpected operation {op!r}")
    audit.check(requests and requests[0] == {"op": "spec"}, f"{label}: missing initial specification")
    return draws, actions


class NativeReplay:
    def __init__(self):
        self.process = subprocess.Popen([str(NATIVE)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1, cwd=PROJECT)
        self.replies = queue.Queue()
        self.errors = deque(maxlen=20)
        def stdout():
            try:
                while True:
                    line = self.process.stdout.readline(MAX_RESPONSE_BYTES + 1)
                    if not line:
                        self.replies.put(None)
                        return
                    self.replies.put(line)
            except Exception as error:
                self.replies.put(error)
        def stderr():
            for line in self.process.stderr:
                self.errors.append(line[:1000])
        threading.Thread(target=stdout, daemon=True).start()
        threading.Thread(target=stderr, daemon=True).start()

    def call(self, request):
        self.process.stdin.write(canonical(request) + "\n")
        self.process.stdin.flush()
        line = self.replies.get(timeout=60)
        if isinstance(line, Exception):
            raise line
        if line is None or len(line.encode("utf-8")) > MAX_RESPONSE_BYTES or not line.endswith("\n"):
            raise ValueError("native replay ended or exceeded bounded response size: " + "".join(self.errors))
        return parse(line)

    def close(self):
        try:
            if self.process.poll() is None:
                response = self.call({"op": "close"})
                if response != {"ok": True}:
                    raise ValueError("native close receipt differs")
                self.process.wait(timeout=10)
            if self.process.returncode != 0:
                raise ValueError("native replay exited unsuccessfully")
        finally:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                stream.close()


def replay(audit, tape, permitted, slots, source, label):
    engine, stream = NativeReplay(), hashlib.sha256()
    current, counters, times, required_reset = [], [], [], set()
    exposure = [{"case_index": i, "decisions": 0, "native_steps": 0, "duration_s": 0.,
                 "native_reward": 0., "safety_stops": 0} for i in range(len(permitted))]
    lookup = {canonical(case): index for index, case in enumerate(permitted)}
    rows, episodes, failure, partial_steps, uncertain, failed_prefixes = [], 0, None, 0, 0, []
    try:
        for number, request in enumerate(tape["requests"]):
            response = engine.call(request)
            stream.update((canonical({"request": request, "response": response}) + "\n").encode("utf-8"))
            if response.get("ok") is not True:
                failure = response
                partial_steps += sum(value["native_steps"] for value in response.get("completed_slots_before_failure", []))
                uncertain += response.get("uncertain_current_slot_native_steps_upper_bound", 0)
                if request["op"] == "step":
                    for slot, value in enumerate(response.get("completed_slots_before_failure", [])):
                        c = compact(value)
                        native_check(audit, c, f"{label}.failed request{number}.completed slot{slot}", times[slot])
                        failed_prefixes.append({"request_index": number, "slot": slot, "case_index": current[slot],
                                                "native_steps": c["native_steps"], "duration_s": c["elapsed_s"],
                                                "native_reward": c["reward"], "light_reward": c["light_reward"],
                                                "imu_reward": c["imu_reward"], "stopped": c["terminated"]})
                continue
            op, result = request["op"], response["result"]
            if op == "spec":
                audit.equal(result["schema"], "actuation_transfer_bridge_v1", f"{label}.native schema")
                audit.equal(result["requests"], [-.15, 0., .15], f"{label}.native requests")
                audit.equal(result["native_steps_per_decision"], 5, f"{label}.native frequency")
            elif op == "configure":
                current = [lookup[canonical(case)] for case in request["cases"]]
                counters, times, required_reset = [0] * slots, [0.] * slots, set()
                audit.equal(len(result), slots, f"{label}.configure response count")
            elif op == "reset":
                indices = {item["index"] for item in request["slots"]}
                audit.equal(indices, required_reset, f"{label}.resets follow actual episode ends")
                for item in request["slots"]:
                    slot = item["index"]
                    current[slot] = lookup[canonical(item["case"])]
                    counters[slot], times[slot] = 0, 0.
                required_reset = set()
            elif op == "step":
                audit.check(not required_reset, f"{label}: step precedes required reset")
                audit.equal(len(result), slots, f"{label}.step response count")
                for slot, value in enumerate(result):
                    c = compact(value)
                    native_check(audit, c, f"{label}.request{number}.slot{slot}", times[slot])
                    times[slot] = c["time_s"]
                    counters[slot] += 1
                    e = exposure[current[slot]]
                    e["decisions"] += 1
                    e["native_steps"] += c["native_steps"]
                    e["duration_s"] += c["elapsed_s"]
                    e["native_reward"] += c["reward"]
                    e["safety_stops"] += int(c["terminated"])
                    if c["terminated"] or source and counters[slot] >= 640:
                        episodes += 1
                        required_reset.add(slot)
                    if not source:
                        rows.append({"decision": len(rows) + 1, "action": request["actions"][slot], "native": c})
    finally:
        engine.close()
    audit.equal(stream.hexdigest(), tape["native_stream_sha256"], f"{label}.replayed native stream")
    audit.equal(failure, tape["native_failure"], f"{label}.retained native failure")
    return {"native_stream_sha256": stream.hexdigest(), "request_count": len(tape["requests"]),
            "episodes": episodes, "exposure": exposure, "rows": rows,
            "failure_completed_native_steps": partial_steps, "failure_uncertain_native_steps_upper_bound": uncertain,
            "failure_completed_prefixes": failed_prefixes,
            "unresolved_request_replayed": False}


def verify_tape(audit, folder, result, permitted, slots, source, do_replay, label):
    record = result["request_tape"]
    path = audit.file(folder / "traces", record)
    tape = load(path, True)
    audit.equal(tape["schema"], "generalization_native_request_tape_v1", f"{label}.tape schema")
    audit.equal(tape["record"], False, f"{label}.record mode")
    audit.equal(record["request_count"], len(tape["requests"]), f"{label}.request count")
    audit.equal(record["native_stream_sha256"], tape["native_stream_sha256"], f"{label}.stream labels")
    audit.equal(result["native_failure"], tape["native_failure"], f"{label}.native failure labels")
    audit.equal(result["unresolved_request"], tape["unresolved_request"], f"{label}.unresolved labels")
    draws, actions = request_schedule(audit, tape, permitted, slots, source, label)
    result_replay = replay(audit, tape, permitted, slots, source, label) if do_replay else None
    return tape, draws, actions, result_replay


def verify(root, do_replay, audit):
    protocol_path, lock_path = safe_path(root, "protocol.json"), safe_path(root, "lock.json")
    protocol, lock = load(protocol_path), load(lock_path)
    protocol_hash = digest(protocol_path)
    audit.equal(protocol["schema"], "generalization_pretraining_pilot_v1", "protocol schema")
    audit.equal(protocol["status"], "locked_before_learning", "protocol status")
    audit.equal(lock["schema"], "generalization_lock_v1", "lock schema")
    audit.equal(lock["protocol_sha256"], protocol_hash, "prospective protocol hash")
    audit.equal(lock["source_sha256"], protocol["source_sha256"], "source commitments")
    audit.equal(lock["runtime"], protocol["runtime"], "runtime commitments")
    audit.equal(lock["physical_interactions"], 0, "lock interactions")
    audit.equal(lock["reserved_worlds_constructed"], 0, "lock reserved worlds")
    curriculum = protocol["curriculum"]
    check_curriculum(audit, curriculum)
    source_hashes = protocol["source_sha256"]
    for name, sha in source_hashes.items():
        if name.startswith("artifacts/") or name.startswith("build/"):
            raise ValueError("source provenance must not read artifact or build evidence")
        archived = safe_path(root / "source_snapshot", name)
        live = safe_path(PROJECT, name)
        audit.check(bool(HEX.fullmatch(sha)), f"{name}: source SHA format")
        audit.check(digest(archived) == sha, f"{name}: archived source differs")
        audit.check(digest(live) == sha, f"{name}: current source differs")
    archived_paths = {p.relative_to(root / "source_snapshot").as_posix()
                      for p in (root / "source_snapshot").rglob("*") if p.is_file()}
    audit.equal(archived_paths, set(source_hashes), "complete archived experimental source manifest")
    if do_replay:
        if sys.platform != "linux" or os.name == "nt" or PROJECT != Path("/workspace") or not Path("/.dockerenv").is_file():
            raise ValueError("native replay requires the documented /workspace Linux development container")
        audit.check(digest(NATIVE) == protocol["runtime"]["native_binary_sha256"], "native executable differs from prospective runtime")
        if audit.failed:
            raise ValueError("native replay refused because prospective protocol/source/runtime checks failed")
    folder = safe_path(root, "pilot-404")
    source, summary = load(folder / "result.json"), load(folder / "summary.json")
    audit.equal(source["schema"], "generalization_source_pilot_result_v1", "source schema")
    audit.equal(source["seed"], 404, "source seed")
    audit.equal(source["protocol_sha256"], protocol_hash, "source protocol")
    audit.equal(summary["schema"], "generalization_pilot_summary_v1", "summary schema")
    audit.equal(summary["protocol_sha256"], protocol_hash, "summary protocol")
    audit.equal(summary["source"], source, "embedded source receipt")
    audit.equal(summary["source_sha256_after_collection"], source_hashes, "post-collection source hashes")
    audit.equal(summary["runtime_after_collection"], protocol["runtime"], "post-collection runtime")
    audit.equal(summary["provenance_unchanged"], True, "post-collection provenance")
    for key in ("reserved_worlds_constructed", "held_out_trials"):
        audit.equal(summary[key], 0, f"summary.{key}")
    for key in ("unseen_body_claim", "promotion"):
        audit.equal(summary[key], False, f"summary.{key}")
    audit.equal(source["requested_decisions"], 16384, "source requested budget")
    audit.check(0 <= source["actual_decisions"] <= 16384, "source exposure exceeds declared budget")
    if "policy_path" in source:
        audit.file(folder, source, "policy_sha256", "policy_path")
        audit.check(bool(HEX.fullmatch(source["parameter_sha256"])), "source parameter label format")
    initial = source["initial_checkpoint"]
    if initial is not None:
        audit.file(folder, initial, "policy_sha256", "policy_path")
        audit.equal(load(folder / "initial-checkpoint.json"), initial, "initial checkpoint receipt")
        audit.equal(initial["weight_updates"], 0, "initial checkpoint weight updates")
        audit.equal(initial["source_decisions"], 0, "initial checkpoint physical decisions")
        audit.check(bool(HEX.fullmatch(initial["parameter_sha256"])), "initial parameter label format")
    source_replay, tapes = None, 0
    if "request_tape" in source:
        _, draws, _, source_replay = verify_tape(audit, folder, source, curriculum["source_cases"], 8, True, do_replay, "source")
        tapes += 1
        rng, order, expected_draws = random.Random(500404), [], []
        while len(expected_draws) < len(draws):
            order = list(range(32))
            rng.shuffle(order)
            expected_draws.extend(order)
        audit.equal(draws, expected_draws[:len(draws)], "independent source draw schedule")
        schedule = {"schema": "shuffled_source_cycles_v1", "source_seed": 404, "curriculum_seed": 500404,
                    "draws": len(draws), "completed_cycles": len(draws) // 32, "current_cycle_position": len(draws) % 32,
                    "source_count": 32, "draw_counts": [draws.count(i) for i in range(32)],
                    "draw_index_sha256": hashlib.sha256("".join(f"{i}\n" for i in draws).encode("ascii")).hexdigest()}
        audit.equal(source["source_schedule"], schedule, "source schedule receipt")
    else:
        audit.check(source["actual_decisions"] == 0 and not source["complete"], "missing source tape has claimed experience")
    exposure = source["per_source_case_exposure"]
    audit.check(len(exposure) in (0, 32), "source per-case exposure matrix incomplete")
    audit.equal(sum(e["decisions"] for e in exposure), source["actual_decisions"], "source decision sum")
    audit.equal(sum(e["native_steps"] for e in exposure), source["native_steps"], "source native-step sum")
    audit.equal(sum(e["safety_stops"] for e in exposure), source["safety_stops"], "source stop sum")
    if source_replay:
        audit.equal(exposure, source_replay["exposure"], "independently replayed per-source exposure")
        audit.equal(source["episodes"], source_replay["episodes"], "independently replayed source episodes")
    exact = source["native_failure"] is None and source["unresolved_request"] is None and bool(exposure)
    audit.equal(source["exposure_exact"], exact, "source exposure exactness")
    if source["complete"]:
        audit.equal(source["actual_decisions"], 16384, "complete source budget")
        for key, wanted in (("optimization_epochs", 32), ("expected_optimization_epochs", 32),
                            ("optimizer_steps_min", 1024), ("optimizer_steps_max", 1024),
                            ("expected_optimizer_steps", 1024), ("parameters_finite", True), ("update_counts_exact", True)):
            audit.equal(source[key], wanted, f"source.{key}")
        audit.check(exact and initial is not None and source["error"] is None and source["close_error"] is None
                    and not source["stopped_by_clock"], "source completion contradicts retained failures")
    trials, audited_trials = [], []
    by_case = {}
    expected_paths = set()
    for index, case_id in enumerate(curriculum["case_ids"]["development"]):
        by_case[index] = {}
        for arm in ARMS:
            trial_id = f"{case_id}--{arm}"
            path = safe_path(folder / "trials", f"{trial_id}/result.json")
            if not path.is_file():
                audit.check(not source["complete"], f"{trial_id}: missing declared development trial")
                continue
            expected_paths.add(path.resolve())
            trial = load(path)
            trials.append(trial)
            audit.equal(trial["schema"], "generalization_development_trial_v1", f"{trial_id}.schema")
            for key, wanted in (("trial_id", trial_id), ("case_id", case_id), ("case_index", index),
                                ("arm", arm), ("protocol_sha256", protocol_hash), ("held_out", False),
                                ("weight_updates", 0), ("reserved_worlds_constructed", 0)):
                audit.equal(trial[key], wanted, f"{trial_id}.{key}")
            tape = replayed = None
            if "request_tape" in trial:
                tape, _, actions, replayed = verify_tape(audit, folder, trial, [curriculum["development_cases"][index]], 1, False, do_replay, trial_id)
                tapes += 1
            else:
                actions = []
                audit.check(not trial["complete"] and trial["error"] is not None
                            and trial["metrics"]["decisions"] == 0, f"{trial_id}: missing tape conceals claimed experience")
            if "telemetry" in trial:
                telemetry = load(audit.file(folder / "traces", trial["telemetry"]), True)
                audit.equal(telemetry["schema"], "generalization_development_telemetry_v1", f"{trial_id}.telemetry schema")
                rows = telemetry["rows"]
            else:
                rows = []
                audit.check(tape is None and not trial["complete"] and trial["metrics"]["decisions"] == 0,
                            f"{trial_id}: missing telemetry conceals claimed experience")
            audit.check(len(rows) <= 640, f"{trial_id}: development budget exceeded")
            previous = 0.
            for decision, row in enumerate(rows, 1):
                audit.equal(row["decision"], decision, f"{trial_id}.telemetry ordinal")
                audit.check(decision <= len(actions), f"{trial_id}: telemetry has no retained action request")
                if decision <= len(actions):
                    audit.equal(row["action"], actions[decision - 1][1][0], f"{trial_id}.telemetry action versus tape")
                if arm == "quiet":
                    audit.equal(row["action"], 1, f"{trial_id}.quiet action")
                native_check(audit, row["native"], f"{trial_id}.decision{decision}", previous)
                previous = row["native"]["time_s"]
            if replayed:
                audit.equal(rows, replayed["rows"], f"{trial_id}.telemetry versus independent native replay")
                rows = replayed["rows"]
            computed = metrics(rows)
            audit.equal(trial["metrics"], computed, f"{trial_id}.metrics")
            windows = [{"requested_decisions": count, "complete": len(rows) >= count,
                        "metrics": metrics(rows[:count])} for count in WINDOWS]
            audit.equal(trial["windows"], windows, f"{trial_id}.windows")
            expected_parameter = (source["parameter_sha256"] if arm == "frozen" else
                                  initial["parameter_sha256"] if arm == "untrained" else None)
            labels_unchanged = (arm == "quiet" or trial["initial_parameter_sha256"] == trial["final_parameter_sha256"] == expected_parameter)
            audit.equal(trial["frozen_weights_unchanged"], labels_unchanged, f"{trial_id}.parameter label consistency")
            if arm == "quiet":
                audit.equal(trial["initial_parameter_sha256"], None, f"{trial_id}.quiet initial parameters")
                audit.equal(trial["final_parameter_sha256"], None, f"{trial_id}.quiet final parameters")
            if trial["action_seed"] is not None:
                audit.equal(trial["action_seed"], None if arm == "quiet" else 14040 + index, f"{trial_id}.action seed")
            else:
                audit.check(arm == "quiet" or not trial["complete"] and trial["error"] is not None,
                            f"{trial_id}: missing frozen action seed without a failed initialization")
            exact = tape is not None and tape["native_failure"] is None and tape["unresolved_request"] is None
            safe = exact and computed["safety_stops"] == 0 and trial["error"] is None and trial["close_error"] is None
            complete = safe and labels_unchanged and not trial["stopped_by_clock"] and len(rows) == 640 and computed["native_steps"] == 3200
            audit.equal(trial["exposure_exact"], exact, f"{trial_id}.exposure exactness")
            audit.equal(trial["safe"], safe, f"{trial_id}.safe accounting")
            audit.equal(trial["complete"], complete, f"{trial_id}.completion accounting")
            verified = {"trial_id": trial_id, "case_index": index, "arm": arm, "complete": complete, "safe": safe, "metrics": computed}
            by_case[index][arm] = verified
            audited_trials.append(verified)
    found = {p.resolve() for p in (folder / "trials").glob("*/result.json")}
    audit.equal(found, expected_paths, "unexpected development result directories")
    audit.check(source["complete"] or not trials, "development ran despite incomplete source checkpoint")
    audit.equal(sorted(summary["development_trials"], key=lambda t: t["trial_id"]),
                sorted(trials, key=lambda t: t["trial_id"]), "embedded development receipts")
    comparisons = []
    for index, case_id in enumerate(curriculum["case_ids"]["development"]):
        group = by_case[index]
        def pair(other):
            present = "frozen" in group and other in group
            eligible = present and all(group[arm]["complete"] and group[arm]["safe"] for arm in ("frozen", other))
            gain = group["frozen"]["metrics"]["native_reward"] - group[other]["metrics"]["native_reward"] if present else None
            return bool(eligible), gain
        eligible, gain = pair("quiet")
        learned, learning_gain = pair("untrained")
        comparisons.append({"case_id": case_id, "complete_safe_pair": eligible, "frozen_minus_quiet": gain,
                            "passes": eligible and gain > 3.2, "complete_safe_learning_pair": learned,
                            "trained_minus_untrained": learning_gain, "trained_beats_untrained": learned and learning_gain > 0.})
    audit.equal(summary["development_comparisons"], comparisons, "independent comparisons including stopped prefixes")
    gains = [c["trained_minus_untrained"] for c in comparisons if c["complete_safe_learning_pair"]]
    realized = [c["trained_minus_untrained"] for c in comparisons if c["trained_minus_untrained"] is not None]
    diagnostic = {"eligible_pairs": len(gains), "realized_pairs": len(realized),
                  "mean_trained_minus_untrained": sum(realized) / len(realized) if len(realized) == 12 else None,
                  "mean_complete_safe_pair_trained_minus_untrained": sum(gains) / len(gains) if gains else None,
                  "all_case_mean_has_equal_complete_safe_exposure": len(gains) == 12,
                  "all_case_mean_interpretation": "Descriptive realized raw returns across all12 cases; stopped prefixes have unequal exposure and this mean cannot establish competence",
                  "trained_wins": sum(c["trained_beats_untrained"] for c in comparisons), "has_prospective_success_threshold": False}
    audit.equal(summary["offline_learning_diagnostic"], diagnostic, "offline learning diagnostic")
    competence = all(c["passes"] for c in comparisons)
    complete = source["complete"] and len(trials) == 36 and all(t["complete"] for t in audited_trials)
    audit.equal(summary["development_competence"], competence, "independent competence gate")
    audit.equal(summary["complete"], complete, "overall operational completion")
    if source["complete"]:
        audit.equal(tapes, 37, "all source and development tapes")
    decisions = source["actual_decisions"] + sum(t["metrics"]["decisions"] for t in audited_trials)
    native_steps = source["native_steps"] + sum(t["metrics"]["native_steps"] for t in audited_trials)
    audit.check(decisions <= 39424 and native_steps <= 197120, "total declared pilot opportunity exceeded")
    return {"protocol_sha256": protocol_hash, "lock_sha256": digest(lock_path), "summary_sha256": digest(folder / "summary.json"),
            "source_result_sha256": digest(folder / "result.json"), "source_files_verified": len(source_hashes),
            "request_tapes_verified": tapes, "native_replay_requested": do_replay,
            "source_complete": source["complete"], "source_exposure_exact": source["exposure_exact"],
            "source_exposure": source_replay["exposure"] if source_replay else exposure,
            "source_failure_extra_completed_native_steps": source_replay["failure_completed_native_steps"] if source_replay else None,
            "source_failure_uncertain_native_steps_upper_bound": source_replay["failure_uncertain_native_steps_upper_bound"] if source_replay else None,
            "source_failure_completed_prefixes": source_replay["failure_completed_prefixes"] if source_replay else None,
            "unresolved_requests_replayed": False, "development_trials": audited_trials,
            "development_comparisons": comparisons, "development_competence": competence,
            "offline_learning_diagnostic": diagnostic, "operationally_complete": complete,
            "actual_decisions": decisions, "recorded_successful_native_steps": native_steps,
            "reserved_native_configuration_calls": 0,
            "scope": "Native replay verifies full record=False responses and recomputes physical accounting. Policy ZIP bytes and parameter label consistency are verified; neural inference, tensor hashes and optimizer execution are not independently replayed."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="existing prospective report directory")
    parser.add_argument("--receipt", type=Path, required=True, help="exclusive new verification JSON path")
    parser.add_argument("--source-archive", type=Path, help="existing retained copy of this verifier, written by the report owner")
    parser.add_argument("--replay", action="store_true", help="independently replay native requests in the managed Linux container")
    args = parser.parse_args()
    root, receipt = args.output.resolve(), args.receipt.resolve()
    if not root.is_dir() or not receipt.parent.is_dir():
        raise ValueError("report directory and receipt parent must already exist")
    audit = Audit()
    with receipt.open("x", encoding="utf-8") as stream:
        result = {"schema": "generalization_independent_verification_v1", "input": str(root),
                  "created_at_utc": datetime.now(timezone.utc).isoformat(), "native_replay_requested": args.replay,
                  "verifier_source_path": str(Path(__file__).resolve()), "verifier_source_sha256": digest(Path(__file__)),
                  "verifier_source_archive_path": str(args.source_archive.resolve()) if args.source_archive else None}
        try:
            if args.source_archive:
                audit.check(digest(args.source_archive.resolve()) == result["verifier_source_sha256"],
                            "retained verifier source differs from the executing verifier")
            result.update(verify(root, args.replay, audit))
        except Exception as error:
            audit.check(False, f"{type(error).__name__}: {error}")
        result.update({"integrity_passed": audit.failed == 0, "checks": audit.checks,
                       "failed_checks": audit.failed, "issues": audit.issues,
                       "issues_truncated": audit.failed > len(audit.issues),
                       "native_replay_verified": args.replay and audit.failed == 0})
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(canonical({"receipt": str(receipt), "integrity_passed": audit.failed == 0, "failed_checks": audit.failed}))
    return 0 if audit.failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(canonical({"error": f"{type(error).__name__}: {error}", "existing_evidence_preserved": True}), file=sys.stderr)
        raise SystemExit(1)
