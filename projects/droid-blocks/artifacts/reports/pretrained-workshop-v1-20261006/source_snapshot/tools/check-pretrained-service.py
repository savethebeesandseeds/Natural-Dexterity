#!/usr/bin/env python3
"""Capture idle legacy sessions, then verify the real HTTP build/wake service."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/reports/pretrained-workshop-v1-20261006"
URL = "http://127.0.0.1:8080"
OLD = ("/api/state", "/api/construction/state", "/api/light/state")


def call(path, payload=None, status=200):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(URL + path, data=data, headers={"Content-Type": "application/json"})
    try:
        response = urlopen(request, timeout=10)
    except HTTPError as error:
        response = error
    with response:
        if response.status != status:
            raise ValueError(f"Unexpected HTTP status {response.status}: {path}")
        return json.loads(response.read())


def save(name, payload):
    path = REPORT / name
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("capture", "check"))
    parser.add_argument("--version", type=int, default=2)
    args = parser.parse_args()
    if args.operation == "capture":
        states = {path: call(path) for path in OLD}
        if states["/api/state"]["running"] or states["/api/state"]["tick"] != 0:
            raise ValueError("Legacy playground contains live progress; preserve it before activation")
        for path in OLD[1:]:
            if states[path]["status"] != "ready" or states[path]["elapsed_s"] != 0:
                raise ValueError("Legacy session contains progress; preserve it before activation")
        sha = save("legacy-idle-before.json", {"schema": "pretrained_workshop_legacy_idle_v1", "states": states})
        print(json.dumps({"legacy_sessions_idle": True, "receipt_sha256": sha}))
        return
    before = json.loads((REPORT / "legacy-idle-before.json").read_text())["states"]
    checks = 0

    def check(value, description):
        nonlocal checks
        checks += 1
        if not value:
            raise ValueError(description)

    for path in OLD:
        check(call(path) == before[path], "Legacy idle session changed after activation")
    health = call("/healthz")
    check(health["status"] == "ok", "Unhealthy engine")
    spec = call("/api/pretrained/spec")
    check(spec["duration_s"] == 64 and spec["decision_dt_s"] == .1, "Runtime timing contract differs")
    state = call("/api/pretrained/state")
    actor = state["policy"]
    check(state["status"] == "ready" and state["elapsed_s"] == 0, "Demo not initially paused")
    call("/api/pretrained/control", {"action": "wake"}, 400)
    call("/api/pretrained/control", {"action": "rebuild", "expected_session_id": state["session_id"],
                                   "body": "wrong", "motor": "base", "lamp": "left"}, 400)
    check(call("/api/pretrained/state") == state, "Rejected controls mutated state")
    tested = []
    for body in ("long_upper", "long_lower", "weighted"):
        for motor in ("base", "elbow"):
            for lamp in ("left", "right"):
                old_id = state["session_id"]
                state = call("/api/pretrained/control", {"action": "rebuild", "expected_session_id": old_id,
                             "body": body, "motor": motor, "lamp": lamp})
                check(state["case"] == {"body": body, "motor": motor, "lamp": lamp}, "Wrong applied configuration")
                check(state["history_frames"] == 1 and not state["history"] and state["native_steps"] == 0,
                      "Rebuild kept old body history")
                check(state["policy"] == actor, "HTTP rebuild changed brain")
                call("/api/pretrained/control", {"action": "wake", "expected_session_id": old_id}, 409)
                tested.append(state["case"])
    # Native real-time worker, rather than the offline stepping test driver.
    running = call("/api/pretrained/control", {"action": "wake", "expected_session_id": state["session_id"]})
    time.sleep(.31)
    paused = call("/api/pretrained/control", {"action": "pause", "expected_session_id": running["session_id"]})
    check(paused["native_steps"] > 0 and paused["status"] == "paused", "Real-time worker did not advance")
    time.sleep(.08)
    check(call("/api/pretrained/state") == paused, "Paused HTTP session advanced")
    call("/api/pretrained/control", {"action": "wake", "expected_session_id": paused["session_id"]})
    time.sleep(.13)
    resumed = call("/api/pretrained/control", {"action": "pause", "expected_session_id": paused["session_id"]})
    check(resumed["session_id"] == paused["session_id"] and resumed["native_steps"] > paused["native_steps"],
          "Resume reset or failed advancing")
    check(resumed["policy"] == actor, "Real-time run changed brain")
    state = call("/api/pretrained/control", {"action": "rebuild", "expected_session_id": resumed["session_id"],
                 "body": "long_upper", "motor": "base", "lamp": "left"})
    check(state["status"] == "ready" and state["elapsed_s"] == 0, "Did not leave a fresh paused demo")
    for path in OLD:
        check(call(path) == before[path], "New demo controlled a legacy session")
    assets = {}
    for path in ("/pretrained.html", "/pretrained.css", "/pretrained.js", "/headroom.html", "/transfer.html"):
        with urlopen(URL + path, timeout=10) as response:
            check(response.status == 200, "Missing web asset")
            content = response.read()
            check(content == (ROOT / "web" / path.lstrip("/")).read_bytes(), "Web asset differs from saved source")
            assets[path] = {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content),
                            "content_security_policy": response.headers.get("Content-Security-Policy")}
    if args.version < 2:
        raise ValueError("Use a new HTTP receipt version >=2")
    sha = save(f"http-service-v{args.version}.json", {"schema": "pretrained_workshop_http_verification_v1", "passed": True,
        "checks": checks, "cases": tested, "legacy_sessions_exact": True, "actor": actor,
        "real_time_steps_before_pause": paused["native_steps"], "real_time_steps_after_resume": resumed["native_steps"],
        "final_status": state["status"], "assets": assets,
        "verifier_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({"passed": True, "checks": checks, "receipt_sha256": sha}))


if __name__ == "__main__":
    main()
