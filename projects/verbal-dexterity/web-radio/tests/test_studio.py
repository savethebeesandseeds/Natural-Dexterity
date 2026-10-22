from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import math
import shutil
import struct
import wave
from datetime import date
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from web_radio.budget import Budget, BudgetError, reservation_cost
from web_radio.contract import validate_package
from web_radio.gemini import Script, build_request
from web_radio.studio import (
    BudgetSettings, GenerationRequest, Reconciliation, Studio, create_app,
)

SOURCE = Path(__file__).resolve().parents[1]
HEADERS = {"Origin": "http://testserver", "X-Web-Radio-Request": "studio"}


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    shutil.copytree(SOURCE / "content", root / "content")
    # Short native fixture; authored long scripts have separate chunk-plan tests.
    programme = root / "content/series/first-light/programme.json"
    script = json.loads(programme.read_text())
    script["turns"] = script["turns"][:2]
    programme.write_text(json.dumps(script))
    (root / "web" / "studio").mkdir(parents=True)
    (root / "web" / "studio" / "index.html").write_text("<!doctype html><title>Studio</title>", encoding="utf-8")
    return root


@pytest.fixture
def studio(project):
    service = Studio(project)
    service.save_key("fixture-key-never-an-actual-credential")
    service.configure(BudgetSettings(kind="programme", episode_limit_usd="1.00", series_limit_usd="2.00"))
    return service


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"".join(struct.pack("<h", round(3000 * math.sin(2 * math.pi * 220 * i / 24000))) for i in range(24000)))
    return buffer.getvalue()


def success_transport(requests):
    def response(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "fixture-interaction", "model": "gemini-3.8-flash-tts", "status": "completed",
            "steps": [{"type": "model_output", "content": [{
                "type": "audio", "mime_type": "audio/wav",
                "data": base64.b64encode(wav_bytes()).decode("ascii"),
            }]}],
            "usage": {"total_input_tokens": 100, "total_output_tokens": 25},
        })
    return httpx.MockTransport(response)


def test_script_directions_are_metadata_and_all_exact_inputs_survive(studio):
    script = studio.read_script("programme")
    script.turns[0].text = "Speak exactly this; do not invent an introduction."
    script.turns[0].style = "whispered"
    studio.save_script(script)
    take = studio.enqueue(GenerationRequest(model="gemini-3.8-flash-tts"))
    stored = json.loads((studio.state / "takes" / take["id"] / "request.json").read_text())
    first = stored["input"][0]["content"][0]
    assert first["text"] == script.turns[0].text
    assert first["annotations"][0]["speaker"] == "Moss"
    assert "whispered" in first["annotations"][0]["style"]
    assert "whispered" not in first["text"]
    assert stored["store"] is False
    assert stored["generation_config"]["max_output_tokens"] == 16384
    assert [speaker["voice"] for speaker in stored["generation_config"]["speech_config"]["speakers"]] == ["Charon", "Kore"]
    assert "fixture-key" not in json.dumps(take)
    assert take["script"]["speakers"][0]["design_prompt"] == script.speakers[0].design_prompt


@pytest.mark.parametrize("voice", ["voice_fixture", "voicekey_fixture", "UnknownVoice"])
def test_generation_rejects_unapproved_voice_before_budget_or_take_but_preserves_import(studio, monkeypatch, voice):
    script = studio.read_script("programme")
    script.speakers[0].voice = voice
    revision = studio.save_script(script)
    assert studio.revision("programme", revision["id"])["script"]["speakers"][0]["voice"] == voice

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported voice reached reservation or provider dispatch.")

    monkeypatch.setattr(studio.budget, "reserve", forbidden)
    monkeypatch.setattr("web_radio.gemini.generate", forbidden)
    with pytest.raises(ValueError, match="prebuilt Gemini voices only|registered character"):
        studio.enqueue(GenerationRequest(model="gemini-3.8-flash-tts", script_revision_id=revision["id"]))
    assert studio.takes() == []
    assert not (studio.state / "takes").exists()
    with studio.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0
    assert studio.run_one() is False

    # Editing and imported performances are independent of the TTS adapter's
    # current supported voice catalog; retain their exact recipe provenance.
    take = studio.begin_import("programme", "fixture.wav", script_revision_id=revision["id"])
    (studio.state / "takes" / take["id"] / "original.wav").write_bytes(wav_bytes())
    imported = studio.finish_import(take["id"])
    assert imported["status"] == "ready"
    assert imported["source"] == "imported"
    assert imported["reservation_microdollars"] == 0
    assert imported["script"]["speakers"][0]["voice"] == voice
    with studio.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0


def test_no_generation_without_explicit_budget(project):
    service = Studio(project)
    service.save_key("fixture-key-never-an-actual-credential")
    with pytest.raises(BudgetError, match="explicit episode"):
        service.enqueue(GenerationRequest(model="gemini-3.8-flash-tts"))
    assert service.takes() == []
    assert service.budget.summary("first-light-001", "first-light")["episode"]["reserved_microdollars"] == 0


def test_concurrent_attempts_cannot_oversubscribe_episode_or_series(tmp_path):
    budget = Budget(tmp_path / "budget.sqlite3")
    budget.configure("episode-a", "series", 1_000_000, 200_000)
    budget.configure("episode-b", "series", 1_000_000, 200_000)

    def attempt(index):
        try:
            budget.reserve(str(index), "episode-a" if index % 2 else "episode-b", "series", 151_552)
            return True
        except BudgetError:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as workers:
        accepted = list(workers.map(attempt, range(16)))
    assert sum(accepted) == 1
    total = budget.summary("episode-a", "series")["series"]
    assert total["reserved_microdollars"] == 151_552
    with pytest.raises(BudgetError, match="below spent plus reserved"):
        budget.configure("episode-a", "series", 1_000_000, 100_000)


def test_timeout_retains_reservation_and_inputs_without_automatic_retry(studio):
    take = studio.enqueue(GenerationRequest(model="gemini-3.8-flash-tts"))
    calls = []

    def timeout(request):
        calls.append(request)
        raise httpx.ReadTimeout("fixture timeout", request=request)

    assert studio.run_one(httpx.MockTransport(timeout))
    result = studio.take(take["id"])
    assert result["status"] == "uncertain"
    assert len(calls) == 1
    assert studio.run_one(httpx.MockTransport(timeout)) is False
    assert len(calls) == 1
    assert (studio.state / "takes" / take["id"] / "request.json").is_file()
    assert studio.budget.summary("first-light-001", "first-light")["episode"]["reserved_microdollars"] == 151_552
    studio.reconcile(take["id"], Reconciliation(charged_usd="0", evidence="Confirmed zero charge in fixture billing record."))
    assert studio.budget.summary("first-light-001", "first-light")["episode"]["reserved_microdollars"] == 0


def test_interrupted_attempt_is_not_reissued_after_studio_restart(studio):
    take = studio.enqueue(GenerationRequest(model="gemini-3.8-flash-tts"))
    with studio.connect() as db:
        db.execute("UPDATE takes SET status='running' WHERE id=?", (take["id"],))
        db.execute("UPDATE generation_requests SET state='running' WHERE take_id=?", (take["id"],))
    studio.start()
    studio.stop()
    assert studio.take(take["id"])["status"] == "uncertain"
    assert studio.budget.summary("first-light-001", "first-light")["episode"]["reserved_microdollars"] == 151_552


def test_completed_take_requires_acceptance_and_publishes_valid_immutable_release(studio):
    requests = []
    take = studio.enqueue(GenerationRequest(model="gemini-3.8-flash-tts"))
    assert studio.run_one(success_transport(requests))
    result = studio.take(take["id"])
    assert result["status"] == "ready"
    assert result["duration_ms"] == 1000
    assert len(requests) == 1
    summary = studio.budget.summary("first-light-001", "first-light")["episode"]
    assert summary["reserved_microdollars"] == 0
    assert summary["spent_microdollars"] == 275
    with pytest.raises(ValueError, match="accept"):
        studio.publish(take["id"])
    studio.decide(take["id"], "rejected")
    with pytest.raises(ValueError, match="accept"):
        studio.publish(take["id"])
    studio.decide(take["id"], "accepted")
    published = studio.publish(take["id"])
    catalog = json.loads((studio.root / ".runtime" / "library" / "catalog.json").read_text())
    assert len(catalog["releases"]) == 1
    record = catalog["releases"][0]
    package = studio.root / ".runtime" / "library" / "releases" / record["release_id"]
    manifest = validate_package(package, expected_digest=record["package_sha256"])
    assert manifest.release_id == published["release_id"]
    assert manifest.kind == "programme"
    assert studio.publish(take["id"])["release_id"] == published["release_id"]
    assert len(json.loads((studio.root / ".runtime" / "library" / "catalog.json").read_text())["releases"]) == 1
    with studio.connect() as db:
        history = list(db.execute("SELECT decision FROM decisions WHERE take_id=? ORDER BY id", (take["id"],)))
    assert [row["decision"] for row in history] == ["rejected", "accepted"]


def test_unknown_usage_keeps_the_full_conservative_allocation(studio):
    from web_radio.budget import usage_cost
    assert usage_cost("gemini-3.8-flash-tts", {}, 151552) == (151552, "conservative-ceiling")
    assert usage_cost("gemini-3.8-flash-tts", {"total_input_tokens": 1, "total_output_tokens": -1}, 151552)[0] == 151552
    with pytest.raises(BudgetError, match="expired"):
        reservation_cost("gemini-3.8-flash-tts", date(2027, 1, 1))


def test_studio_rejects_cross_origin_writes_and_never_returns_saved_key(project):
    with TestClient(create_app(project)) as client:
        payload = {"key": "fixture-key-never-an-actual-credential"}
        assert client.post("/api/credentials", json=payload).status_code == 403
        assert client.post("/api/credentials", json=payload, headers={**HEADERS, "Origin": "https://elsewhere.example"}).status_code == 403
        response = client.post("/api/credentials", json=payload, headers=HEADERS)
        assert response.status_code == 200
        assert response.json() == {"key_configured": True}
        status = client.get("/api/status")
        assert status.status_code == 200
        assert status.json()["key_configured"] is True
        assert payload["key"] not in status.text
        assert client.get("/.runtime/secrets/gemini-api-key").status_code == 404
        assert client.get("/assets/../../.runtime/secrets/gemini-api-key").status_code == 404
        assert client.get("/api/takes").json() == {"takes": []}


def test_voice_edits_create_immutable_recipe_revisions(studio):
    script = studio.read_script("programme")
    studio.save_script(script)
    original = studio.enqueue(GenerationRequest(model="gemini-3.8-flash-tts"))
    original_recipe = original["script"]["speakers"][0]
    script.speakers[0].style = "A new, distinctly quieter performance direction."
    studio.save_script(script)
    revised = studio.enqueue(GenerationRequest(model="gemini-3.8-flash-tts"))
    assert revised["script"]["speakers"][0]["recipe_version"] == original_recipe["recipe_version"] + 1
    assert studio.take(original["id"])["script"]["speakers"][0] == original_recipe
    snapshots = list((studio.state / "voice-recipes").glob("*.json"))
    assert len(snapshots) == 3  # two initial voices, plus one revised Moss recipe


def test_noncompleted_provider_response_never_becomes_ready(studio):
    take = studio.enqueue(GenerationRequest(model="gemini-3.8-flash-tts"))
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        "status": "failed",
        "steps": [{"type": "model_output", "content": [{
            "type": "audio", "mime_type": "audio/wav", "data": base64.b64encode(wav_bytes()).decode("ascii")
        }]}],
    }))
    studio.run_one(transport)
    assert studio.take(take["id"])["status"] == "uncertain"
    assert studio.budget.summary("first-light-001", "first-light")["episode"]["reserved_microdollars"] == 151_552


@pytest.mark.parametrize(("mime_type", "payload_kind"), [
    (None, "wav"),
    ("audio/pcm", "wav"),
    ("audio/mpeg", "wav"),
    ("audio/x-wav", "wav"),
    ("audio/wav", "raw-pcm"),
    ("audio/wav", "wrong-container"),
])
def test_unexpected_provider_audio_keeps_reservation_and_never_becomes_ready(studio, mime_type, payload_kind):
    take = studio.enqueue(GenerationRequest(model="gemini-3.8-flash-tts"))
    audio = {
        "wav": wav_bytes(),
        "raw-pcm": b"\x00" * 48000,
        "wrong-container": b"RIFF" + b"\x00" * 4 + b"AVI " + b"\x00" * 48,
    }[payload_kind]
    block = {"type": "audio", "data": base64.b64encode(audio).decode("ascii")}
    if mime_type is not None:
        block["mime_type"] = mime_type
    calls = []

    def response(request):
        calls.append(request)
        return httpx.Response(200, json={
            "status": "completed",
            "steps": [{"type": "model_output", "content": [block]}],
            "usage": {"total_input_tokens": 100, "total_output_tokens": 25},
        })

    transport = httpx.MockTransport(response)
    assert studio.run_one(transport) is True
    result = studio.take(take["id"])
    assert result["status"] == "uncertain"
    assert "Expected the requested" in result["error"]
    assert studio.budget.summary("first-light-001", "first-light")["episode"]["reserved_microdollars"] == 151_552
    assert studio.budget.summary("first-light-001", "first-light")["episode"]["spent_microdollars"] == 0
    directory = studio.state / "takes" / take["id"]
    assert (directory / "request.json").is_file()
    assert (directory / "script.json").is_file()
    assert not (directory / "raw.wav").exists()
    assert not (directory / "mix.wav").exists()
    assert studio.run_one(transport) is False
    assert len(calls) == 1


def test_script_revisions_restore_append_only_and_reject_stale_save(project):
    from web_radio.studio import ScriptConflict
    service = Studio(project)
    first = service.head_revision("programme")
    script = service.read_script("programme")
    script.title = "A revised title"
    second = service.save_script(script, expected_revision=first["id"])
    assert second["revision"] == first["revision"] + 1
    assert service.revision("programme", first["id"])["script"]["title"] != script.title
    with pytest.raises(ScriptConflict):
        service.save_script(Script.model_validate(first["script"]), expected_revision=first["id"])
    restored = service.restore("programme", first["id"], expected_revision=second["id"])
    assert restored["revision"] == second["revision"] + 1
    assert restored["restored_from"] == first["id"]
    assert restored["script"] == first["script"]
    assert service.revision("programme", second["id"])["script"]["title"] == "A revised title"


def test_import_without_keys_preserves_original_and_snapshot_and_can_publish(project):
    import hashlib
    with TestClient(create_app(project)) as client:
        head = client.get("/api/scripts/programme/head").json()
        payload = wav_bytes()
        result = client.post(
            "/api/import?kind=programme&script_revision_id=" + head["id"],
            content=payload, headers={**HEADERS, "Content-Type": "audio/wav", "X-Recording-Filename": "../../outside.wav"},
        )
        assert result.status_code == 201, result.text
        take = result.json()
        assert take["source"] == "imported"
        assert take["model"] == "local-recording"
        assert take["reservation_microdollars"] == 0
        assert take["script_revision_id"] == head["id"]
        assert take["script"] == head["script"]
        assert take["metadata"]["provider"] is None
        assert take["metadata"]["filename"] == "outside.wav"
        assert take["metadata"]["original_sha256"] == hashlib.sha256(payload).hexdigest()
        original = project / ".runtime/studio/takes" / take["id"] / "original.wav"
        assert original.read_bytes() == payload
        assert not (project.parent / "outside.wav").exists()
        assert client.get("/api/status").json()["key_configured"] is False
        assert client.post("/api/takes/" + take["id"] + "/publish", headers=HEADERS).status_code == 422
        assert client.post("/api/takes/" + take["id"] + "/decision", json={"decision":"accepted"}, headers=HEADERS).status_code == 200
        published = client.post("/api/takes/" + take["id"] + "/publish", headers=HEADERS)
        assert published.status_code == 200, published.text
        catalog = json.loads((project / ".runtime/library/catalog.json").read_text())
        manifest = validate_package(project / ".runtime/library/releases" / published.json()["release_id"], expected_digest=catalog["releases"][0]["package_sha256"])
        assert manifest.kind == "programme"
        with client.app.state.studio.connect() as db:
            assert db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0


@pytest.mark.parametrize("filename", [r"C:\\other\\secret.wav", "../../../escape.wav", "%2e%2e%2fescape.wav"])
def test_recording_names_are_never_paths(project, filename):
    service = Studio(project)
    take = service.begin_import("ident", filename)
    directory = service.state / "takes" / take["id"]
    (directory / "original.wav").write_bytes(wav_bytes())
    result = service.finish_import(take["id"])
    assert result["status"] == "ready"
    assert "/" not in result["metadata"]["filename"]
    assert "\\" not in result["metadata"]["filename"]


def test_corrupt_and_truncated_imports_fail_without_displacing_earlier_take(project):
    service = Studio(project)
    valid = service.begin_import("programme", "good.wav")
    (service.state / "takes" / valid["id"] / "original.wav").write_bytes(wav_bytes())
    service.finish_import(valid["id"])
    for payload in (b"this is not a wave recording", wav_bytes()[:-100]):
        bad = service.begin_import("programme", "bad.wav")
        original = service.state / "takes" / bad["id"] / "original.wav"
        original.write_bytes(payload)
        with pytest.raises(ValueError):
            service.finish_import(bad["id"])
        assert service.take(bad["id"])["status"] == "failed"
        assert original.read_bytes() == payload
    assert service.take(valid["id"])["status"] == "ready"
    assert service.take(valid["id"])["script_revision_id"] == valid["script_revision_id"]


def test_raw_import_size_and_declared_duration_are_bounded(project, monkeypatch):
    import web_radio.studio as studio_module
    with TestClient(create_app(project)) as client:
        too_large = client.post("/api/import", content=b"x", headers={**HEADERS, "Content-Type":"audio/wav", "Content-Length": str(studio_module.MAX_IMPORT_BYTES + 1)})
        assert too_large.status_code == 413
        monkeypatch.setattr(studio_module, "MAX_IMPORT_BYTES", 512)
        streamed = client.post("/api/import", content=iter([b"x"*256,b"x"*300]), headers={**HEADERS, "Content-Type":"audio/wav"})
        assert streamed.status_code == 413
    monkeypatch.setattr(studio_module, "MAX_IMPORT_BYTES", 64 * 1024 * 1024)
    service = Studio(project)
    take = service.begin_import("programme", "long.wav")
    forged = bytearray(wav_bytes())
    # Set the data chunk to a 601-second declaration without allocating it.
    data_index = forged.index(b"data")
    forged[data_index+4:data_index+8] = struct.pack("<I", 601*24000*2)
    (service.state / "takes" / take["id"] / "original.wav").write_bytes(forged)
    with pytest.raises(ValueError, match="10 minutes"):
        service.finish_import(take["id"])


def test_explicit_revision_binding_does_not_revert_current_script(project):
    service = Studio(project)
    original = service.head_revision("programme")
    updated = service.read_script("programme")
    updated.title = "Keep this newer editor version"
    latest = service.save_script(updated)
    imported = service.begin_import("programme", "one.wav", original["id"])
    assert imported["script"]["title"] == original["script"]["title"]
    assert service.head_revision("programme")["id"] == latest["id"]
    default_import = service.begin_import("programme", "two.wav")
    assert default_import["script"]["title"] == updated.title
    assert service.head_revision("programme")["id"] == latest["id"]


def test_process_exit_cannot_leave_a_reservation_without_a_take(studio):
    import subprocess
    import sys
    code = """
import os
from pathlib import Path
from web_radio.studio import Studio, GenerationRequest
service=Studio(Path(ROOT))
reserve=service.budget.reserve
def crash(*args,**kwargs):
    reserve(*args,**kwargs)
    os._exit(23)
service.budget.reserve=crash
service.enqueue(GenerationRequest(model="gemini-3.8-flash-tts"))
""".replace("ROOT", repr(str(studio.root)))
    result = subprocess.run([sys.executable,"-c",code], cwd=SOURCE, timeout=20, check=False)
    assert result.returncode == 23
    with studio.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM takes").fetchone()[0] == 0


def test_rejection_committed_before_publication_prevents_stale_acceptance(project, monkeypatch):
    import threading
    service = Studio(project)
    take = service.begin_import("programme", "take.wav")
    (service.state / "takes" / take["id"] / "original.wav").write_bytes(wav_bytes())
    service.finish_import(take["id"])
    service.decide(take["id"], "accepted")
    initial_read, proceed = threading.Event(), threading.Event()
    original_take = service.take

    def delayed_take(identifier):
        result = original_take(identifier)
        if threading.current_thread().name.startswith("publication") and not initial_read.is_set():
            initial_read.set()
            assert proceed.wait(5)
        return result

    monkeypatch.setattr(service, "take", delayed_take)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="publication") as worker:
        future = worker.submit(service.publish, take["id"])
        assert initial_read.wait(5)
        service.decide(take["id"], "rejected")
        proceed.set()
        with pytest.raises(ValueError, match="accept"):
            future.result(timeout=10)
    assert not (project / ".runtime/library/catalog.json").exists()


def test_catalogued_publication_receipt_recovers_without_duplicate_release(project):
    service = Studio(project)
    take = service.begin_import("programme", "take.wav")
    (service.state / "takes" / take["id"] / "original.wav").write_bytes(wav_bytes())
    service.finish_import(take["id"])
    service.decide(take["id"], "accepted")
    released = service.publish(take["id"])
    expected = released["release_id"]
    for recover_by_reject in (False, True):
        with service.connect() as db:
            db.execute("DELETE FROM publications WHERE take_id=?", (take["id"],))
            db.execute("UPDATE takes SET release_id=NULL WHERE id=?", (take["id"],))
        if recover_by_reject:
            with pytest.raises(ValueError, match="already published"):
                service.decide(take["id"], "rejected")
        else:
            assert service.publish(take["id"])["release_id"] == expected
        assert service.take(take["id"])["release_id"] == expected
        assert service.take(take["id"])["decision"] == "accepted"
        catalog = json.loads((project / ".runtime/library/catalog.json").read_text())
        assert [entry["release_id"] for entry in catalog["releases"]] == [expected]


def test_invalid_language_rejected_before_production(project):
    from pydantic import ValidationError
    script = Studio(project).read_script("programme").model_dump()
    script["language"] = "English"
    with pytest.raises(ValidationError):
        Script.model_validate(script)


@pytest.mark.parametrize("invalid_catalog", [
    b'{"schema_version":1,"schema_version":1,"releases":[]}',
    b'{"schema_version":1,"releases":[{"release_id":"same","package_sha256":"' + b"a"*64 + b'"},{"release_id":"same","package_sha256":"' + b"a"*64 + b'"}]}',
    b'{"schema_version":1,"releases":[],"unexpected":true}',
])
def test_publication_preserves_invalid_catalogues(project, invalid_catalog):
    service = Studio(project)
    take = service.begin_import("programme", "recording.wav")
    (service.state / "takes" / take["id"] / "original.wav").write_bytes(wav_bytes())
    service.finish_import(take["id"])
    service.decide(take["id"], "accepted")
    catalog = project / ".runtime/library/catalog.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_bytes(invalid_catalog)
    with pytest.raises(ValueError):
        service.publish(take["id"])
    assert catalog.read_bytes() == invalid_catalog
    assert service.take(take["id"])["release_id"] is None


def test_publication_does_not_exceed_catalogue_capacity(project, monkeypatch):
    import web_radio.studio as studio_module
    service = Studio(project)
    take = service.begin_import("programme", "recording.wav")
    (service.state / "takes" / take["id"] / "original.wav").write_bytes(wav_bytes())
    service.finish_import(take["id"])
    service.decide(take["id"], "accepted")
    catalog = project / ".runtime/library/catalog.json"
    catalog.parent.mkdir(parents=True)
    original = json.dumps({"schema_version":1,"releases":[{"release_id":"existing","package_sha256":"a"*64}]}).encode()
    catalog.write_bytes(original)
    monkeypatch.setattr(studio_module, "CATALOG_ENTRIES", 1)
    with pytest.raises(ValueError, match="full"):
        service.publish(take["id"])
    assert catalog.read_bytes() == original
    monkeypatch.setattr(studio_module, "CATALOG_ENTRIES", 1000)
    monkeypatch.setattr(studio_module, "CATALOG_LIMIT", len(original)+2)
    with pytest.raises(ValueError, match="size limit"):
        service.publish(take["id"])
    assert catalog.read_bytes() == original


def test_import_decoder_timeout_preserves_original(project, monkeypatch):
    import subprocess
    import web_radio.studio as studio_module
    service = Studio(project)
    take = service.begin_import("programme", "recording.wav")
    payload = wav_bytes()
    original = service.state / "takes" / take["id"] / "original.wav"
    original.write_bytes(payload)

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 60
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(studio_module.subprocess, "run", timeout)
    with pytest.raises(ValueError, match="time limit"):
        service.finish_import(take["id"])
    assert service.take(take["id"])["status"] == "failed"
    assert original.read_bytes() == payload
