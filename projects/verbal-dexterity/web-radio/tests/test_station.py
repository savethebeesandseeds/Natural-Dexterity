"""Station tests use labelled tone fixtures only; they do not call Gemini."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import time
import wave

import pytest
from fastapi.testclient import TestClient

from web_radio.contract import create_release
from web_radio.playout import Journal, Playout, iso, parse_time
from web_radio.station import create_app


def tone(path: Path, seconds=0.5):
    import math
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"".join(struct.pack("<h", int(5000 * math.sin(n * 2 * math.pi * 440 / 24000)))
                                  for n in range(int(seconds * 24000))))
    return path


def publish(runtime: Path, tmp_path: Path, *, ident=False, seconds=0.5, release_id=None):
    rid = release_id or ("fixture-ident-v1" if ident else "fixture-programme-v1")
    source = tone(tmp_path / (rid + ".wav"), seconds)
    destination = runtime / "library" / "releases" / rid
    manifest = create_release(source, "TEST TONE FIXTURE — not a voiced programme.",
                              destination, "fixture-ident" if ident else "fixture-programme",
                              "Test tone fixture", kind="ident" if ident else "programme",
                              release_id=rid)
    catalog = runtime / "library" / "catalog.json"
    existing = json.loads(catalog.read_text()) if catalog.exists() else {"schema_version": 1, "releases": []}
    existing["releases"].append({"release_id": rid, "package_sha256": manifest.package_sha256})
    catalog.write_text(json.dumps(existing))
    return manifest, destination


def wait_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError("condition not reached")


def test_journal_uses_calibrated_stream_time_and_only_completed_programmes_for_replay(tmp_path):
    runtime = tmp_path / "runtime"
    programme, _ = publish(runtime, tmp_path)
    ident, _ = publish(runtime, tmp_path, ident=True)
    journal = Journal(tmp_path / "journal.sqlite3")
    journal.session("session")
    origin = datetime(2026, 10, 22, tzinfo=timezone.utc)
    journal.calibrate("session", iso(origin))
    airing = journal.begin("session", programme, 0)
    journal.advance("session", 200)
    assert journal.now(origin + timedelta(milliseconds=100))["id"] == airing
    assert journal.now(origin + timedelta(milliseconds=300)) is None
    assert journal.last_completed() is None
    journal.finish(airing, 500)
    interlude = journal.begin("session", ident, 500)
    journal.finish(interlude, 1000)
    assert journal.last_completed()["id"] == airing
    assert journal.now(origin + timedelta(milliseconds=600))["kind"] == "ident"
    assert [e["event"] for e in journal.events()] == ["completed", "fallback", "completed", "started"]
    journal.session("another")
    interrupted = journal.begin("another", programme, 0)
    journal.finish(interrupted, 10, interrupted=True)
    assert journal.last_completed()["id"] == airing


def test_restart_marks_unfinished_airing_interrupted(tmp_path):
    runtime = tmp_path / "runtime"
    manifest, _ = publish(runtime, tmp_path)
    journal_path = tmp_path / "journal.sqlite3"
    first = Journal(journal_path)
    first.session("session")
    first.begin("session", manifest, 0)
    second = Journal(journal_path)
    # Constructing/importing another reader must not interrupt a live airing.
    assert second.last_completed() is None
    with second.db() as db:
        assert db.execute("SELECT status FROM airings").fetchone()["status"] == "playing"
    second.recover_interrupted()
    assert second.events()[0]["event"] == "interrupted"
    assert second.events()[0]["detail"] == "Previous station process ended before completion."
    with second.db() as db:
        assert db.execute("SELECT status FROM airings").fetchone()["status"] == "interrupted"


def test_station_import_is_independent_of_source_library(tmp_path):
    runtime = tmp_path / "runtime"
    manifest, source = publish(runtime, tmp_path)
    station = Playout(runtime)
    station.refresh()
    package = station.known[manifest.release_id][1]
    assert package != source
    assert station._select()[0].release_id == manifest.release_id
    # Removing a task-created source simulates an unavailable studio/library.
    import shutil
    shutil.rmtree(source)
    (runtime / "library" / "catalog.json").unlink()
    station.refresh()
    assert station._select()[0].release_id == manifest.release_id
    restarted = Playout(runtime)
    assert restarted._select()[0].release_id == manifest.release_id


def test_corrupt_release_is_rejected_and_verified_ident_is_selected(tmp_path):
    runtime = tmp_path / "runtime"
    programme, package = publish(runtime, tmp_path)
    ident, _ = publish(runtime, tmp_path, ident=True)
    audio = next(a for a in programme.assets if a.role == "audio")
    with (package / audio.path).open("ab") as handle:
        handle.write(b"corrupt")
    station = Playout(runtime)
    station.refresh()
    assert programme.release_id in station.rejections
    assert station._select()[0].release_id == ident.release_id
    assert any(e["event"] == "rejected" for e in station.journal.events())


@pytest.mark.parametrize("catalog", [
    '{"schema_version":1,"schema_version":1,"releases":[]}',
    '{"schema_version":true,"releases":[]}',
    '{"schema_version":2,"releases":[]}',
    '{"schema_version":1,"releases":[{"release_id":"../escape","package_sha256":"' + "a" * 64 + '"}]}',
])
def test_untrusted_catalogue_fields_do_not_replace_accepted_rundown(tmp_path, catalog):
    runtime = tmp_path / "runtime"
    manifest, _ = publish(runtime, tmp_path)
    station = Playout(runtime)
    station.refresh()
    (runtime / "library" / "catalog.json").write_text(catalog)
    station.refresh()
    assert station._select()[0].release_id == manifest.release_id


def test_local_media_corruption_never_gets_selected(tmp_path):
    runtime = tmp_path / "runtime"
    manifest, _ = publish(runtime, tmp_path)
    ident, _ = publish(runtime, tmp_path, ident=True)
    station = Playout(runtime)
    station.refresh()
    package = station.known[manifest.release_id][1]
    audio = next(a for a in manifest.assets if a.role == "audio")
    (package / audio.path).write_bytes(b"invalid")
    assert station._select()[0].release_id == ident.release_id


def test_public_api_serves_only_accepted_assets_with_range_replay(tmp_path):
    runtime = tmp_path / "runtime"
    manifest, _ = publish(runtime, tmp_path)
    app = create_app(runtime, start_engine=False)
    engine = app.state.playout
    engine.refresh()
    engine.journal.session("fixture")
    airing = engine.journal.begin("fixture", manifest, 0)
    engine.journal.finish(airing, manifest.duration_ms)
    with TestClient(app) as client:
        assert client.get("/healthz").json()["service"] == "station"
        saved = client.get("/api/replay").json()
        assert saved["release"]["release_id"] == manifest.release_id
        response = client.get(saved["release"]["audio_url"], headers={"Range": "bytes=0-43"})
        assert response.status_code == 206
        assert len(response.content) == 44
        assert response.content[:4] == b"RIFF"
        assert "immutable" in response.headers["cache-control"]
        assert client.get("/media/unknown/audio.wav").status_code == 404
        assert client.get(f"/media/{manifest.release_id}/manifest.json").status_code == 404
        assert client.get("/hls/not-a-segment").status_code == 404
        assert client.get("/api/now?at=not-a-date").status_code == 422
        assert client.get("/api/now?at=2026-10-22T00:00:00").status_code == 422
        assert client.get("/api/journal?limit=1000").status_code == 422
        assert client.get("/").status_code == 200
        assert "default-src 'self'" in client.get("/").headers["content-security-policy"]


def test_real_ffmpeg_hls_loops_fixture_and_records_completed_airings(tmp_path):
    runtime = tmp_path / "runtime"
    manifest, _ = publish(runtime, tmp_path, seconds=1.2)
    station = Playout(runtime)
    station.start()
    try:
        playlist = station.hls / "live.m3u8"
        wait_until(lambda: station.snapshot()["stream_ready"], timeout=12)
        text = playlist.read_text()
        assert "#EXT-X-PROGRAM-DATE-TIME:" in text
        assert "#EXT-X-DISCONTINUITY" in text
        assert "#EXTINF:" in text
        segments = [line for line in text.splitlines() if line.endswith(".ts")]
        assert segments
        segment = station.hls / segments[0]
        assert segment.is_file() and segment.stat().st_size > 1000
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(segment)],
                               check=True, capture_output=True, text=True)
        assert json.loads(probe.stdout)["streams"][0]["codec_name"] == "aac"
        saved = wait_until(lambda: station.journal.last_completed())
        assert saved["release_id"] == manifest.release_id
        assert saved["timing_calibrated"] is True
        assert station.journal.now(parse_time(saved["stream_start"]) + timedelta(milliseconds=100))["id"] == saved["id"]
        assert station.state == "broadcasting"
    finally:
        station.stop()
    assert station.thread is not None and not station.thread.is_alive()


def test_local_corruption_switches_to_fallback_without_restarting_encoder(tmp_path):
    runtime = tmp_path / "runtime"
    programme, _ = publish(runtime, tmp_path, seconds=1.2)
    ident, _ = publish(runtime, tmp_path, ident=True, seconds=0.4)
    station = Playout(runtime)
    station.start()
    try:
        wait_until(lambda: station.snapshot()["stream_ready"], timeout=12)
        original_session = station.session_id
        package = station.known[programme.release_id][1]
        audio = next(a for a in programme.assets if a.role == "audio")
        previous_sequence = max((e["sequence"] for e in station.journal.events()), default=0)
        (package / audio.path).write_bytes(b"task-created corrupt fixture")
        wait_until(lambda: any(e["event"] == "fallback" and e["sequence"] > previous_sequence
                               for e in station.journal.events()), timeout=6)
        assert station.session_id == original_session
        assert station.state == "fallback"
        assert programme.release_id in station.rejections
        assert any(e["release_id"] == ident.release_id and e["event"] == "fallback"
                   for e in station.journal.events())
    finally:
        station.stop()


def test_second_playout_owner_is_refused_without_changing_journal(tmp_path):
    runtime = tmp_path / "runtime"
    first = Playout(runtime)
    second = Playout(runtime)
    first.start()
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            second.start()
        assert second.journal.events() == []
    finally:
        first.stop()
    second.start()
    second.stop()


def test_contract_maximum_length_release_id_is_accepted(tmp_path):
    runtime = tmp_path / "runtime"
    rid = "r" * 128
    manifest, _ = publish(runtime, tmp_path, release_id=rid)
    station = Playout(runtime)
    station.refresh()
    assert station._select()[0].release_id == manifest.release_id
