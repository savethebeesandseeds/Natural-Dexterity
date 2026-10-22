"""Backend cast rendering: order, upfront bounds, partial charges, and no retries."""
from __future__ import annotations

import base64
import io
import json
import shutil
import struct
import wave
from pathlib import Path

import httpx
import pytest

from web_radio.budget import BudgetError, reservation_cost
from web_radio.gemini import Script
from web_radio.studio import Studio, GenerationRequest, BudgetSettings, Reconciliation

SOURCE = Path(__file__).resolve().parents[1]
MODEL = "gemini-3.8-flash-lite-tts"


@pytest.fixture
def service(tmp_path):
    root = tmp_path / "project"
    shutil.copytree(SOURCE / "content", root / "content")
    programme = root / "content/series/first-light/programme.json"
    raw = json.loads(programme.read_text())
    third = dict(raw["speakers"][0], name="Ash", voice="Puck", style="", pronunciation_notes="")
    raw["speakers"].append(third)
    raw["turns"] = [
        {"speaker": speaker["name"], "text": f"Turn {index}.", "style": ""}
        for index, speaker in enumerate(raw["speakers"])
    ]
    programme.write_text(json.dumps(raw))
    backend = Studio(root)
    backend.save_key("fixture-only-never-a-real-credential")
    backend.configure(BudgetSettings(episode_limit_usd="1.00", series_limit_usd="2.00"))
    return backend


def audio(amplitude):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(struct.pack("<h", amplitude) * 24000)
    return buffer.getvalue()


def reply(request, amplitude=1000, usage=True):
    value = {"status": "completed", "model": MODEL, "id": "fixture",
             "steps": [{"type": "model_output", "content": [{
                 "type": "audio", "mime_type": "audio/wav",
                 "data": base64.b64encode(audio(amplitude)).decode()}]}]}
    if usage:
        value["usage"] = {"total_input_tokens": 23, "total_output_tokens": 25}
    return httpx.Response(200, json=value)


def test_three_characters_render_in_order_with_exact_originals_and_one_mix(service):
    take = service.enqueue(GenerationRequest(model=MODEL))
    assert take["plan"]["strategy"] == "turns"
    assert take["reservation_microdollars"] == 3 * reservation_cost(MODEL)
    calls = []
    def perform(request):
        calls.append(json.loads(request.content))
        return reply(request, len(calls) * 500)
    assert service.run_one(httpx.MockTransport(perform), take_id=take["id"])
    result = service.take(take["id"])
    assert result["status"] == "ready" and result["duration_ms"] == 3000
    assert [call["input"][0]["content"][0]["text"] for call in calls] == ["Turn 0.", "Turn 1.", "Turn 2."]
    assert [call["generation_config"]["speech_config"][0]["voice"] for call in calls] == ["Charon", "Kore", "Puck"]
    directory = service.state / "takes" / take["id"]
    for index in range(3):
        assert (directory / f"requests/{index:04d}/raw.wav").read_bytes() == audio((index + 1) * 500)
    with wave.open(str(directory / "raw.wav"), "rb") as stream:
        levels = []
        for position in (24000, 72000, 120000):
            stream.setpos(position)
            levels.append(struct.unpack("<h", stream.readframes(1)[:2])[0])
        assert levels[0] < levels[1] < levels[2]
    summary = service.budget.summary("first-light-001", "first-light")["episode"]
    assert summary["reserved_microdollars"] == 0 and summary["spent_microdollars"] > 0
    assert [r["state"] for r in result["requests"]] == ["completed"] * 3
    assert service.run_one(httpx.MockTransport(perform), take_id=take["id"]) is False
    assert len(calls) == 3


def test_all_requests_must_fit_budget_before_any_take_commits(service):
    service.configure(BudgetSettings(episode_limit_usd="0.25", series_limit_usd="0.25"))
    with pytest.raises(BudgetError, match="maximum allocation"):
        service.enqueue(GenerationRequest(model=MODEL))
    assert service.takes() == []
    with service.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM generation_requests").fetchone()[0] == 0


def test_partial_failure_keeps_completed_cost_and_holds_only_uncertain_call(service):
    take = service.enqueue(GenerationRequest(model=MODEL))
    calls = []
    def perform(request):
        calls.append(request)
        if len(calls) == 2:
            raise httpx.ReadTimeout("fixture timeout", request=request)
        return reply(request)
    assert service.run_one(httpx.MockTransport(perform), take_id=take["id"])
    result = service.take(take["id"])
    assert result["status"] == "uncertain"
    assert [r["state"] for r in result["requests"]] == ["completed", "uncertain", "cancelled"]
    summary = service.budget.summary("first-light-001", "first-light")["episode"]
    assert summary["spent_microdollars"] > 0
    assert summary["reserved_microdollars"] == reservation_cost(MODEL)
    assert not (service.state / "takes" / take["id"] / "mix.wav").exists()
    assert service.run_one(httpx.MockTransport(perform)) is False
    assert len(calls) == 2
    service.reconcile(take["id"], Reconciliation(
        charged_usd="0", request_index=1, evidence="Fixture billing confirms zero charge for the timeout."))
    assert service.budget.summary("first-light-001", "first-light")["episode"]["reserved_microdollars"] == 0


def test_specific_run_does_not_drain_an_older_queued_take(service):
    older = service.enqueue(GenerationRequest(model=MODEL))
    chosen = service.enqueue(GenerationRequest(model=MODEL))
    calls = []
    def perform(request):
        calls.append(request)
        return reply(request)
    assert service.run_one(httpx.MockTransport(perform), take_id=chosen["id"])
    assert service.take(chosen["id"])["status"] == "ready"
    assert service.take(older["id"])["status"] == "queued"
    assert len(calls) == 3


def test_unknown_usage_does_not_break_assembly_or_invent_measured_tokens(service):
    take = service.enqueue(GenerationRequest(model=MODEL))
    assert service.run_one(httpx.MockTransport(lambda request: reply(request, usage=False)), take_id=take["id"])
    result = service.take(take["id"])
    assert result["status"] == "ready"
    assert result["metadata"]["usage"] == {} and result["metadata"]["usage_complete"] is False
    summary = service.budget.summary("first-light-001", "first-light")["episode"]
    assert summary["spent_microdollars"] == 3 * reservation_cost(MODEL)
    assert summary["reserved_microdollars"] == 0


def test_stop_between_turns_preserves_finished_original_and_releases_unsent(service):
    take = service.enqueue(GenerationRequest(model=MODEL))
    calls = []
    def perform(request):
        calls.append(request)
        service.stop_event.set()
        return reply(request)
    assert service.run_one(httpx.MockTransport(perform), take_id=take["id"])
    result = service.take(take["id"])
    assert result["status"] == "failed"
    assert len(calls) == 1
    assert [r["state"] for r in result["requests"]] == ["completed", "cancelled", "cancelled"]
    summary = service.budget.summary("first-light-001", "first-light")["episode"]
    assert summary["spent_microdollars"] > 0 and summary["reserved_microdollars"] == 0
    assert (service.state / "takes" / take["id"] / "requests/0000/raw.wav").exists()


def test_queued_job_cannot_dispatch_after_pricing_expires(service, monkeypatch):
    from web_radio import studio as module
    take = service.enqueue(GenerationRequest(model=MODEL))
    def expired(*args, **kwargs):
        raise BudgetError("The stored price schedule expired.")
    monkeypatch.setattr(module, "reservation_cost", expired)
    def forbidden(request):
        pytest.fail("Expired allocation reached provider.")
    assert service.run_one(httpx.MockTransport(forbidden), take_id=take["id"])
    assert service.take(take["id"])["status"] == "failed"
    summary = service.budget.summary("first-light-001", "first-light")["episode"]
    assert summary["spent_microdollars"] == 0 and summary["reserved_microdollars"] == 0


def test_reconciliation_failure_rolls_back_charge_and_preserves_pending_evidence(service, monkeypatch):
    take = service.enqueue(GenerationRequest(model=MODEL))
    def timeout(request):
        raise httpx.ReadTimeout("fixture", request=request)
    service.run_one(httpx.MockTransport(timeout), take_id=take["id"])
    with service.connect() as db:
        db.execute("""CREATE TRIGGER fail_reconciliation BEFORE INSERT ON reconciliations
                      BEGIN SELECT RAISE(ABORT, 'fixture evidence write failure'); END""")
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        service.reconcile(take["id"], Reconciliation(
            charged_usd="0", request_index=0, evidence="Fixture billing confirms zero charged."))
    summary = service.budget.summary("first-light-001", "first-light")["episode"]
    assert summary["reserved_microdollars"] == reservation_cost(MODEL)
    assert service.take(take["id"])["requests"][0]["state"] == "uncertain"


def test_description_design_to_three_character_publication_is_backend_only(service):
    from datetime import datetime, timedelta, timezone
    from web_radio.characters import CharacterStore
    from web_radio.voice_design import DesignRequest, VoiceDesignService
    from web_radio.contract import validate_package
    store = CharacterStore(service.root)
    for identity, name in (("moss", "Moss"), ("wren", "Wren"), ("ash", "Ash")):
        store.create({"character_id": identity, "name": name, "personality": "Fixture fictional character"})
    store.add_prebuilt("moss", "Charon", MODEL)
    store.add_prebuilt("wren", "Kore", MODEL)
    design = VoiceDesignService(service.root, key_provider=lambda: "fixture-only-never-real")
    design.configure_budget("ash", "0.20", "0.20")
    job = design.enqueue(DesignRequest(
        character_id="ash", description="An original fictional weather keeper with a dry, gentle voice.",
        model=MODEL, quoted_cost_usd="0.10", quote_evidence="Mock fixture quote only; not a real provider price.",
        quote_valid_until=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat()))
    requests = []
    def create_voice(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "voice_original_fixture", "type": "prompted",
            "sample_audio": {"mime_type": "audio/wav", "data": base64.b64encode(audio(1000)).decode()}})
    ready = design.run_one_specific(job["id"], httpx.MockTransport(create_voice))
    assert ready["status"] == "ready"
    assert len(requests) == 1 and requests[0]["voice"]["type"] == "prompted"
    script = service.read_script("programme")
    for speaker, identity in zip(script.speakers, ("moss", "wren", "ash")):
        speaker.character_id = identity
    revision = service.save_script(script)
    take = service.enqueue(GenerationRequest(model=MODEL, script_revision_id=revision["id"]))
    assert len(take["cast_snapshot"]["speakers"]) == 3
    frozen_character_revision = take["script"]["speakers"][2]["character_revision"]
    latest = store.get("ash")
    store.update("ash", {"character_id": "ash", "name": "Ash", "personality": "A changed later draft"},
                 expected_revision=latest["revision"])
    assert store.get("ash")["revision"] > frozen_character_revision
    calls = []
    def perform(request):
        calls.append(json.loads(request.content))
        return reply(request)
    assert service.run_one(httpx.MockTransport(perform), take_id=take["id"])
    assert service.take(take["id"])["status"] == "ready"
    assert calls[2]["generation_config"]["speech_config"][0]["voice"] == "voice_original_fixture"
    assert service.take(take["id"])["script"]["speakers"][2]["character_revision"] == frozen_character_revision
    service.decide(take["id"], "accepted")
    published = service.publish(take["id"])
    package = service.root / ".runtime/library/releases" / published["release_id"]
    assert validate_package(package).kind == "programme"
    assert len(calls) == 3
