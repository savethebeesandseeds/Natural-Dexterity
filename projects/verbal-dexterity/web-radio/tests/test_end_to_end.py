"""Process-level separation proof using synthetic audio and isolated test state.

No Gemini request is made. Fake credentials, budgets, generated tones, published
packages, and station journals exist only in pytest's private temporary project.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import textwrap
import time

import httpx
import pytest


SOURCE = Path(__file__).resolve().parents[1]


STUDIO_RUNNER = r"""
import io
import math
from pathlib import Path
import struct
import sys
import wave

import uvicorn
from web_radio import studio

def synthetic_generate(request, key, transport=None):
    # This stub is installed inside this fixture process only. No HTTP provider
    # client is constructed, and all supplied text must identify the fixture.
    texts = [part["text"] for part in request["input"][0]["content"]]
    assert all("TEST FIXTURE" in text for text in texts)
    seconds = 0.4 if "IDENT TEST FIXTURE" in texts[0] else 1.2
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"".join(
            struct.pack("<h", round(2500 * math.sin(2 * math.pi * 330 * index / 24000)))
            for index in range(round(seconds * 24000))
        ))
    return output.getvalue(), {
        "interaction_id": "synthetic-process-fixture",
        "status": "completed",
        "model": request["model"],
        "usage": {"total_input_tokens": 100, "total_output_tokens": 30},
        "fixture": "synthetic tone, no speech provider used",
    }

studio.gemini.generate = synthetic_generate
uvicorn.run(studio.create_app(Path(sys.argv[1])), host="127.0.0.1",
            port=int(sys.argv[2]), log_level="error", access_log=False)
"""


STATION_RUNNER = r"""
from pathlib import Path
import sys
import uvicorn
from web_radio import station

station.PLAYER = Path(sys.argv[1]) / "web" / "player"
uvicorn.run(station.create_app(Path(sys.argv[1]) / ".runtime"), host="127.0.0.1",
            port=int(sys.argv[2]), log_level="error", access_log=False)
"""


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def wait_for(predicate, *, timeout=12, description="condition"):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            result = predicate()
            if result:
                return result
        except (httpx.HTTPError, OSError, ValueError) as exc:
            last_error = exc
        time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for {description}; last error: {last_error}")


def get_json(client, url):
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def terminate_owned(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        # Every fixture child is its own session leader; only this owned group
        # may be signalled, including an encoder that failed to stop gracefully.
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def encoder_pid(station_pid):
    # Popen runs on the playout thread. Linux records child PIDs beneath that
    # thread's task entry rather than necessarily the process's main task.
    child_pids = set()
    for children in Path(f"/proc/{station_pid}/task").glob("*/children"):
        try:
            child_pids.update(children.read_text().split())
        except FileNotFoundError:
            continue
    for raw_pid in child_pids:
        command_path = Path(f"/proc/{raw_pid}/cmdline")
        try:
            command = command_path.read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        if b"hls" in command and b"pipe:0" in command:
            return int(raw_pid)
    return None


def latest_segment(client, station_url):
    response = client.get(station_url + "/hls/live.m3u8")
    response.raise_for_status()
    playlist = response.text
    assert "#EXT-X-PROGRAM-DATE-TIME:" in playlist
    assert "#EXT-X-DISCONTINUITY" in playlist
    segments = [line for line in playlist.splitlines() if line.endswith(".ts")]
    return segments[-1] if segments else None


def test_actual_station_keeps_broadcasting_after_actual_studio_process_ends(tmp_path):
    project = tmp_path / "isolated-project"
    shutil.copytree(SOURCE / "content", project / "content")
    shutil.copytree(SOURCE / "web", project / "web")
    runners = {
        "studio": textwrap.dedent(STUDIO_RUNNER),
        "station": textwrap.dedent(STATION_RUNNER),
    }
    studio_port, station_port = free_port(), free_port()
    while station_port == studio_port:
        station_port = free_port()
    # Ports are container-internal ephemeral loopback sockets, never Docker
    # publications and never the actual services' 43120/43121 endpoints.
    assert {studio_port, station_port}.isdisjoint({43120, 43121})
    urls = {"studio": f"http://127.0.0.1:{studio_port}",
            "station": f"http://127.0.0.1:{station_port}"}
    environment = os.environ.copy()
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        environment.pop(key, None)
    environment.update(
        PYTHONPATH=str(SOURCE), PYTHONDONTWRITEBYTECODE="1",
        WEB_RADIO_ROOT=str(project), WEB_RADIO_DATA=str(project / ".runtime"),
    )
    processes = {}
    log_paths = {}
    try:
        for name in ("station", "studio"):
            runner = tmp_path / f"{name}_fixture.py"
            runner.write_text(runners[name], encoding="utf-8")
            log_paths[name] = tmp_path / f"{name}.log"
            with log_paths[name].open("wb") as log:
                processes[name] = subprocess.Popen(
                    [sys.executable, str(runner), str(project),
                     str(studio_port if name == "studio" else station_port)],
                    cwd=SOURCE, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                )
        with httpx.Client(timeout=4, trust_env=False) as client:
            for name in ("station", "studio"):
                wait_for(
                    lambda name=name: get_json(client, urls[name] + "/healthz").get("service") == name,
                    description=f"{name} HTTP readiness",
                )
                assert processes[name].poll() is None

            headers = {"Origin": urls["studio"], "X-Web-Radio-Request": "studio"}
            def mutate(method, path, body=None):
                response = client.request(
                    method, urls["studio"] + path, headers=headers, json=body,
                )
                assert response.is_success, f"{path}: {response.status_code} {response.text}"
                return response.json()

            mutate("POST", "/api/credentials",
                   {"key": "fixture-key-never-an-actual-credential"})
            take_ids = {}
            for kind in ("programme", "ident"):
                script = get_json(client, urls["studio"] + f"/api/script?kind={kind}")
                script.update(
                    programme_id=f"process-fixture-{kind}",
                    series_id="process-fixture",
                    title=f"TEST FIXTURE synthetic {kind}; no generated speech",
                )
                label = "IDENT TEST FIXTURE" if kind == "ident" else "PROGRAMME TEST FIXTURE"
                script["turns"] = [
                    {"speaker": speaker["name"], "text": f"{label}: synthetic tone only.",
                     "style": "This is a process-separation fixture."}
                    for speaker in script["speakers"]
                ]
                mutate("PUT", "/api/script", script)
                mutate("POST", "/api/settings", {
                    "kind": kind, "episode_limit_usd": "1.00", "series_limit_usd": "3.00",
                })
                take = mutate("POST", "/api/generate", {
                    "kind": kind, "model": "gemini-3.8-flash-tts",
                })
                take_ids[kind] = take["id"]

            def completed_takes():
                takes = get_json(client, urls["studio"] + "/api/takes")["takes"]
                selected = {take["id"]: take for take in takes}
                for identifier in take_ids.values():
                    take = selected.get(identifier)
                    if take and take["status"] in {"failed", "uncertain"}:
                        raise AssertionError(f"Synthetic fixture generation failed: {take}")
                return selected if all(
                    selected.get(identifier, {}).get("status") == "ready"
                    for identifier in take_ids.values()
                ) else None

            ready = wait_for(completed_takes, description="both synthetic takes to finish")
            releases = {}
            for kind, identifier in take_ids.items():
                assert ready[identifier]["metadata"]["fixture"].startswith("synthetic tone")
                mutate("POST", f"/api/takes/{identifier}/decision", {"decision": "accepted"})
                releases[kind] = mutate("POST", f"/api/takes/{identifier}/publish")["release_id"]

            wait_for(
                lambda: get_json(client, urls["station"] + "/api/status")["stream_ready"],
                description="calibrated real HLS output",
            )
            first_segment = wait_for(
                lambda: latest_segment(client, urls["station"]),
                description="first complete HLS segment",
            )
            first_replay = wait_for(
                lambda: get_json(client, urls["station"] + "/api/replay").get("airing"),
                description="completed programme in the real playback journal",
            )
            assert first_replay["release_id"] == releases["programme"]
            assert first_replay["timing_calibrated"] is True
            before_events = get_json(client, urls["station"] + "/api/journal?limit=200")["events"]
            before_sequence = max(event["sequence"] for event in before_events)
            station_encoder = wait_for(
                lambda: encoder_pid(processes["station"].pid),
                description="the station-owned encoder process",
            )

            # End only the actual studio process. The separate station and its
            # encoder must continue producing both audio and completion records.
            terminate_owned(processes["studio"])
            assert processes["studio"].poll() is not None
            assert processes["station"].poll() is None
            with pytest.raises(httpx.ConnectError):
                client.get(urls["studio"] + "/healthz")

            def advanced_segment():
                current = latest_segment(client, urls["station"])
                return current if current != first_segment else None

            next_segment = wait_for(advanced_segment, description="HLS advancement after studio shutdown")
            segment_response = client.get(urls["station"] + "/hls/" + next_segment)
            assert segment_response.status_code == 200
            assert len(segment_response.content) > 1000
            assert encoder_pid(processes["station"].pid) == station_encoder

            def later_completion():
                events = get_json(client, urls["station"] + "/api/journal?limit=200")["events"]
                return events if any(
                    event["sequence"] > before_sequence and event["event"] == "completed"
                    and event["release_id"] == releases["programme"] for event in events
                ) else None

            wait_for(later_completion, description="new journal completion after studio shutdown")
            replay = get_json(client, urls["station"] + "/api/replay")
            assert replay["airing"]["release_id"] == releases["programme"]
            assert replay["release"]["release_id"] == releases["programme"]
            audio = client.get(
                urls["station"] + replay["release"]["audio_url"],
                headers={"Range": "bytes=0-43"},
            )
            assert audio.status_code == 206
            assert len(audio.content) == 44 and audio.content[:4] == b"RIFF"
            assert audio.headers["content-range"].startswith("bytes 0-43/")
            stream_start = datetime.fromisoformat(replay["airing"]["stream_start"].replace("Z", "+00:00"))
            at = (stream_start + timedelta(milliseconds=100)).isoformat()
            now = get_json(client, urls["station"] + "/api/now?at=" + at.replace("+", "%2B"))
            assert now["airing"]["id"] == replay["airing"]["id"]
            assert now["timing"] == "stream"
            assert get_json(client, urls["station"] + "/api/status")["state"] == "broadcasting"
    except BaseException as exc:
        logs = "\n".join(
            f"--- {name} fixture ---\n{path.read_text(errors='replace')[-5000:]}"
            for name, path in log_paths.items() if path.exists()
        )
        raise AssertionError(f"{exc}\n{logs}") from exc
    finally:
        for name in ("station", "studio"):
            if name in processes:
                terminate_owned(processes[name])



KEYLESS_STUDIO_RUNNER = r"""
from pathlib import Path
import sys
import uvicorn
from web_radio import studio

uvicorn.run(studio.create_app(Path(sys.argv[1])), host="127.0.0.1",
            port=int(sys.argv[2]), log_level="error", access_log=False)
"""


def local_wav_fixture(seconds):
    import io
    import math
    import struct
    import wave
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(24000)
        recording.writeframes(b"".join(
            struct.pack("<h", round(2400 * math.sin(2 * math.pi * 440 * index / 24000)))
            for index in range(round(seconds * 24000))
        ))
    return buffer.getvalue()


def test_keyless_wav_import_and_private_programming_survive_studio_exit(tmp_path):
    project = tmp_path / "keyless-project"
    shutil.copytree(SOURCE / "content", project / "content")
    shutil.copytree(SOURCE / "web", project / "web")
    ports = {"studio": free_port(), "station": free_port()}
    while ports["station"] == ports["studio"]:
        ports["station"] = free_port()
    assert set(ports.values()).isdisjoint({43120, 43121})
    urls = {name: f"http://127.0.0.1:{port}" for name, port in ports.items()}
    environment = os.environ.copy()
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        environment.pop(key, None)
    environment.update(
        PYTHONPATH=str(SOURCE), PYTHONDONTWRITEBYTECODE="1",
        WEB_RADIO_ROOT=str(project), WEB_RADIO_DATA=str(project / ".runtime"),
    )
    processes, log_paths = {}, {}
    try:
        for name, source in (("station", STATION_RUNNER), ("studio", KEYLESS_STUDIO_RUNNER)):
            runner = tmp_path / f"keyless_{name}_fixture.py"
            runner.write_text(textwrap.dedent(source), encoding="utf-8")
            log_paths[name] = tmp_path / f"keyless_{name}.log"
            with log_paths[name].open("wb") as log:
                processes[name] = subprocess.Popen(
                    [sys.executable, str(runner), str(project), str(ports[name])],
                    cwd=SOURCE, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                )
        with httpx.Client(timeout=5, trust_env=False) as client:
            for name in ("station", "studio"):
                wait_for(
                    lambda name=name: get_json(client, urls[name] + "/healthz").get("service") == name,
                    description=f"keyless {name} HTTP readiness",
                )
            headers = {"Origin": urls["studio"], "X-Web-Radio-Request": "studio"}
            def mutate(method, path, body=None):
                response = client.request(method, urls["studio"] + path, headers=headers, json=body)
                assert response.is_success, f"{path}: {response.status_code} {response.text}"
                return response.json()

            assert get_json(client, urls["studio"] + "/api/status")["key_configured"] is False
            releases = {}
            for kind, seconds in (("programme", 1.2), ("ident", 0.4)):
                script = get_json(client, urls["studio"] + f"/api/script?kind={kind}")
                script.update(
                    programme_id=f"keyless-fixture-{kind}", series_id="keyless-fixture",
                    title=f"TEST FIXTURE locally imported {kind}; synthetic tone",
                )
                script["turns"] = [{
                    "speaker": speaker["name"],
                    "text": f"TEST FIXTURE local {kind} recording; synthetic tone, no speech provider.",
                    "style": "",
                } for speaker in script["speakers"]]
                mutate("PUT", "/api/script", script)
                imported = client.post(
                    urls["studio"] + f"/api/import?kind={kind}",
                    content=local_wav_fixture(seconds),
                    headers={**headers, "Content-Type": "audio/wav",
                             "X-Recording-Filename": f"synthetic-fixture-{kind}.wav"},
                )
                assert imported.status_code == 201, imported.text
                take = imported.json()
                assert take["status"] == "ready" and take["source"] == "imported"
                assert take["reservation_microdollars"] == 0
                assert take["model"] == "local-recording"
                mutate("POST", f"/api/takes/{take['id']}/decision", {"decision": "accepted"})
                releases[kind] = mutate("POST", f"/api/takes/{take['id']}/publish")["release_id"]
            assert not (project / ".runtime" / "secrets" / "gemini-api-key").exists()
            budgets = get_json(client, urls["studio"] + "/api/status")["budgets"]
            for scopes in budgets.values():
                for summary in scopes.values():
                    assert summary["limit_microdollars"] is None
                    assert summary["reserved_microdollars"] == summary["spent_microdollars"] == 0

            programming = get_json(client, urls["studio"] + "/api/programming")
            selected = {entry["release_id"]: {
                "release_id": entry["release_id"], "package_sha256": entry["package_sha256"],
            } for entry in programming["releases"]}
            programme_ref = selected[releases["programme"]]
            ident_ref = selected[releases["ident"]]
            change = {
                "expected_revision": programming["revision"], "mode": "programmes",
                "programme_releases": [programme_ref], "fallback_release": ident_ref,
            }
            submitted = mutate("PUT", "/api/programming", change)
            first_revision = submitted["revision"]

            def acknowledged(revision):
                current = get_json(client, urls["studio"] + "/api/programming")
                ack = current.get("acknowledgement")
                return ack if ack and ack["accepted_revision"] == revision and ack["status"] == "accepted" else None

            wait_for(lambda: acknowledged(first_revision),
                     description="station acknowledgement of digest-bound programme rundown")
            wait_for(
                lambda: get_json(client, urls["station"] + "/api/status")["stream_ready"],
                description="keyless HLS playback",
            )
            wait_for(
                lambda: get_json(client, urls["station"] + "/api/replay").get("airing"),
                description="keyless programme completion",
            )
            assert get_json(client, urls["station"] + "/api/replay")["release"]["release_id"] == releases["programme"]

            # Explicitly select the validated ident through the private controls.
            fallback = mutate("PUT", "/api/programming", {
                "expected_revision": first_revision, "mode": "fallback",
                "programme_releases": [], "fallback_release": ident_ref,
            })
            wait_for(lambda: acknowledged(fallback["revision"]),
                     description="station acknowledgement of fallback mode")
            wait_for(
                lambda: get_json(client, urls["station"] + "/api/status")["state"] == "fallback",
                description="fallback taking effect at a natural airing boundary",
            )
            before_segment = wait_for(
                lambda: latest_segment(client, urls["station"]),
                description="fallback HLS segment",
            )
            before_sequence = max(event["sequence"] for event in get_json(
                client, urls["station"] + "/api/journal?limit=200")["events"])
            station_encoder = wait_for(
                lambda: encoder_pid(processes["station"].pid),
                description="keyless station encoder",
            )
            terminate_owned(processes["studio"])
            assert processes["studio"].poll() is not None
            assert processes["station"].poll() is None
            with pytest.raises(httpx.ConnectError):
                client.get(urls["studio"] + "/healthz")

            def next_segment():
                segment = latest_segment(client, urls["station"])
                return segment if segment != before_segment else None

            segment = wait_for(next_segment, description="fallback stream advancing without studio")
            response = client.get(urls["station"] + "/hls/" + segment)
            assert response.status_code == 200 and len(response.content) > 1000
            assert encoder_pid(processes["station"].pid) == station_encoder

            def continued_fallback():
                events = get_json(client, urls["station"] + "/api/journal?limit=200")["events"]
                return events if any(
                    event["sequence"] > before_sequence and event["event"] == "fallback"
                    and event["release_id"] == releases["ident"] for event in events
                ) else None
            wait_for(continued_fallback, description="new fallback airing after studio exit")
            assert get_json(client, urls["station"] + "/api/status")["state"] == "fallback"
            replay = get_json(client, urls["station"] + "/api/replay")
            assert replay["release"]["release_id"] == releases["programme"]
            audio = client.get(urls["station"] + replay["release"]["audio_url"],
                               headers={"Range": "bytes=0-43"})
            assert audio.status_code == 206 and audio.content[:4] == b"RIFF"
            assert len(audio.content) == 44
    except BaseException as exc:
        logs = "\n".join(
            f"--- keyless {name} fixture ---\n{path.read_text(errors='replace')[-5000:]}"
            for name, path in log_paths.items() if path.exists()
        )
        raise AssertionError(f"{exc}\n{logs}") from exc
    finally:
        for name in ("station", "studio"):
            if name in processes:
                terminate_owned(processes[name])
