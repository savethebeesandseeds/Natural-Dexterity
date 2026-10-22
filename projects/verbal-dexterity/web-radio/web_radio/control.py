"""Operate only the explicitly recorded Web Radio service processes."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path("/workspace")
DATA = Path(os.environ.get("WEB_RADIO_DATA", ROOT / ".runtime")).resolve()
CONTROL = DATA / "control"
SERVICES = {"studio": ("web_radio.studio:app", 43120),
            "station": ("web_radio.station:app", 43121)}

def paths(name):
    return CONTROL / (name + ".json"), CONTROL / (name + ".log")

def _process_identity(pid):
    process = Path("/proc", str(pid))
    command = (process / "cmdline").read_bytes().split(b"\0")
    # starttime is field 22; comm may contain spaces or parentheses.
    ticks = int((process / "stat").read_text().rsplit(")", 1)[1].split()[19])
    return command, ticks

def managed_alive(name):
    record_path, _ = paths(name)
    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text())
    pid = record["pid"]
    try:
        command, ticks = _process_identity(pid)
    except FileNotFoundError:
        return None
    if not any(command):
        return None
    if record.get("start_ticks") != ticks:
        raise RuntimeError("Refusing stale PID identity for " + name)
    marker = SERVICES[name][0].encode()
    if marker not in command or b"uvicorn" not in command:
        raise RuntimeError("Refusing to signal a process that does not match " + name)
    return pid

def start(name):
    if managed_alive(name):
        return
    target, port = SERVICES[name]
    _, log_path = paths(name)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1", WEB_RADIO_DATA=str(DATA))
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", target, "--host", "0.0.0.0",
             "--port", str(port), "--no-access-log"],
            cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            start_new_session=True)
    record_path, _ = paths(name)
    record_path.write_text(json.dumps({"pid": process.pid, "target": target,
                                      "port": port, "started_at": time.time(),
                                      "start_ticks": _process_identity(process.pid)[1]}) + "\n")
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError(name + " exited; inspect " + str(log_path))
        try:
            with urlopen("http://127.0.0.1:" + str(port) + "/api/status", timeout=1) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(name + " did not become ready; process preserved; inspect " + str(log_path))

def stop(name):
    pid = managed_alive(name)
    if not pid:
        return
    os.killpg(pid, signal.SIGTERM)
    for _ in range(100):
        if not managed_alive(name):
            return
        status_path = Path("/proc", str(pid), "stat")
        if status_path.exists() and status_path.read_text().split()[2] == "Z":
            return
        time.sleep(0.1)
    raise RuntimeError(name + " did not stop gracefully; process preserved")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start","stop","status","studio-stop","studio-start"])
    args = parser.parse_args()
    CONTROL.mkdir(parents=True, exist_ok=True)
    with (CONTROL / "operation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.action == "start":
            start("station")
            start("studio")
        elif args.action == "stop":
            stop("studio")
            stop("station")
        elif args.action == "studio-stop":
            stop("studio")
        elif args.action == "studio-start":
            start("studio")
        result = {}
        for name, (_, port) in SERVICES.items():
            result[name] = {"pid": managed_alive(name), "url": "http://127.0.0.1:" + str(port)}
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

