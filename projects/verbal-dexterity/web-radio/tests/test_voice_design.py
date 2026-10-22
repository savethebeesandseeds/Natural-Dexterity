from __future__ import annotations

from datetime import datetime, timedelta, timezone
import base64
import fcntl
import io
import json
import hashlib
import os
import threading
import concurrent.futures
import wave

import httpx
import pytest

from web_radio.characters import CharacterStore
from web_radio.voice_design import DesignRequest, VoiceDesignService
from web_radio.budget import BudgetError
from web_radio import gemini

MODEL = "gemini-3.8-flash-lite-tts"


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(b"\x00\x00" * 2400)
    return output.getvalue()


def request(**changes):
    data = {
        "character_id": "moss", "description": "A patient woodland creature with a warm low voice.",
        "model": MODEL, "language_code": "en", "quoted_cost_usd": "0.10",
        "quote_evidence": "Fixture-only verified operator quote, not live provider pricing.",
        "quote_valid_until": (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    }
    data.update(changes)
    return DesignRequest.model_validate(data)


def response(**changes):
    data = {"id": "voice_original_moss", "type": "prompted", "model": MODEL,
            "sample_audio": {"mime_type": "audio/wav", "data": base64.b64encode(wav_bytes()).decode()}}
    data.update(changes)
    return data


@pytest.fixture
def service(tmp_path):
    CharacterStore(tmp_path).create({"character_id": "moss", "name": "Moss", "background": "An original fictional forest character."})
    result = VoiceDesignService(tmp_path, key_provider=lambda: "fixture-key-never-a-real-credential")
    result.configure_budget("moss", "0.25", "0.25")
    return result


def totals(service):
    return service.budget.summary("voice-design:moss", "voice-design")["episode"]


def test_original_voice_design_uses_description_only_and_records_complete_provenance(service):
    job = service.enqueue(request(gender="neutral"))
    assert totals(service)["reserved_microdollars"] == 100000
    sent = []
    def provider(call):
        sent.append(json.loads(call.content))
        assert call.url.path == "/v1beta/voices" and call.method == "POST"
        assert call.headers["x-goog-api-key"] == "fixture-key-never-a-real-credential"
        assert "fixture-key" not in call.content.decode()
        return httpx.Response(200, json=response())
    ready = service.run_one_specific(job["id"], transport=httpx.MockTransport(provider))
    assert ready["status"] == "ready" and ready["dispatch_count"] == 1
    assert ready["recipe"]["origin"] == "original_design"
    assert sent[0]["voice"]["type"] == "prompted"
    assert sent[0]["voice"]["prompted"] == {"input": request().description}
    assert set(sent[0]["voice"]) == {"model", "type", "display_name", "language_code", "gender", "prompted"}
    assert service.characters.approved_designed_voices() == {"voice_original_moss"}
    assert totals(service)["spent_microdollars"] == 100000
    assert totals(service)["reserved_microdollars"] == 0
    assert ready["quote"]["guarantee"].startswith("Local allocation only")
    assert service.run_one_specific(job["id"], transport=httpx.MockTransport(provider))["status"] == "ready"
    assert len(sent) == 1


def test_missing_explicit_budget_or_quote_cannot_dispatch(tmp_path):
    CharacterStore(tmp_path).create({"character_id": "moss", "name": "Moss"})
    service = VoiceDesignService(tmp_path, key_provider=lambda: "fixture")
    with pytest.raises(BudgetError, match="explicit episode"):
        service.enqueue(request())
    with pytest.raises(ValueError):
        DesignRequest.model_validate({"character_id": "moss", "description": "An original character description", "model": MODEL})
    with service.budget.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM voice_design_jobs").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0


@pytest.mark.parametrize("changes", [
    {"quote_valid_until": "2000-01-01T00:00:00+00:00"},
    {"quote_valid_until": "2099-01-01T00:00:00"},
    {"quoted_cost_usd": "NaN"},
    {"quoted_cost_usd": "0"},
    {"reference_audio": "person.wav"},
    {"type": "replicated"},
    {"base_voice": "voice_someone"},
])
def test_invalid_quotes_and_replication_inputs_are_rejected(changes):
    with pytest.raises(ValueError):
        request(**changes)


def test_concurrent_design_jobs_cannot_exceed_character_or_global_budget(service):
    def enqueue(_):
        try:
            return service.enqueue(request())["id"]
        except BudgetError:
            return None
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as workers:
        accepted = [x for x in workers.map(enqueue, range(8)) if x]
    assert len(accepted) == 2
    assert totals(service)["reserved_microdollars"] == 200000


@pytest.mark.parametrize("kind", ["timeout", "replicated", "wrong-mime", "bad-audio", "invalid-id"])
def test_uncertain_or_invalid_provider_result_retains_allocation_without_retry(service, kind):
    job = service.enqueue(request())
    calls = []
    def provider(call):
        calls.append(call)
        if kind == "timeout":
            raise httpx.ReadTimeout("fixture transport ended", request=call)
        value = response()
        if kind == "replicated":
            value["type"] = "replicated"
        elif kind == "wrong-mime":
            value["sample_audio"]["mime_type"] = "audio/l16"
        elif kind == "bad-audio":
            value["sample_audio"]["data"] = base64.b64encode(b"not a wav").decode()
        else:
            value["id"] = "voicekey_replicated"
        return httpx.Response(200, json=value)
    result = service.run_one_specific(job["id"], transport=httpx.MockTransport(provider))
    assert result["status"] == "uncertain"
    assert totals(service)["reserved_microdollars"] == 100000
    assert totals(service)["spent_microdollars"] == 0
    assert service.characters.recipes("moss") == []
    assert service.run_one_specific(job["id"], transport=httpx.MockTransport(provider))["status"] == "uncertain"
    assert len(calls) == 1
    service.reconcile(job["id"], "0", "Fixture billing record confirms no charge.")
    assert totals(service)["reserved_microdollars"] == 0


def test_queued_quote_expiry_releases_only_known_unsent_allocation(service):
    job = service.enqueue(request())
    with service.budget.connect() as db:
        expired = {**job["quote"], "valid_until": "2000-01-01T00:00:00+00:00"}
        db.execute("UPDATE voice_design_jobs SET quote_json=? WHERE id=?", (json.dumps(expired), job["id"]))
    result = service.run_one_specific(job["id"], transport=httpx.MockTransport(lambda call: pytest.fail("Expired quote dispatched")))
    assert result["status"] == "failed" and result["dispatch_count"] == 0
    assert totals(service)["reserved_microdollars"] == 0


def test_missing_key_releases_only_known_unsent_job(service):
    job = service.enqueue(request())
    service.key_provider = lambda: None
    result = service.run_one_specific(job["id"], transport=httpx.MockTransport(lambda call: pytest.fail("Missing credential dispatched")))
    assert result["status"] == "failed" and result["dispatch_count"] == 0
    assert totals(service)["reserved_microdollars"] == 0


def test_interrupted_job_requires_recovery_and_is_never_redispatched(service):
    job = service.enqueue(request())
    with service.budget.connect() as db:
        db.execute("UPDATE voice_design_jobs SET status='running',dispatch_count=1 WHERE id=?", (job["id"],))
    assert service.run_one_specific(job["id"], transport=httpx.MockTransport(lambda call: pytest.fail("Interrupted job retried")))["status"] == "running"
    recovered = service.recover(job["id"])
    assert recovered["status"] == "uncertain"
    assert totals(service)["reserved_microdollars"] == 100000
    service.reconcile(job["id"], "0.02", "Confirmed fixture billing charge after interruption.")
    assert totals(service)["spent_microdollars"] == 20000


def test_recovery_refuses_an_active_dispatch_lease(service):
    job = service.enqueue(request())
    path = service.characters.artifact(f"design-jobs/{job['id']}/run.lock")
    with path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="active"):
            service.recover(job["id"])
        with pytest.raises(ValueError, match="active"):
            service.run_one_specific(job["id"])


def test_retained_success_finishes_locally_without_duplicate_voice_creation(service):
    job = service.enqueue(request())
    with service.budget.connect() as db:
        db.execute("UPDATE voice_design_jobs SET status='running',dispatch_count=1 WHERE id=?", (job["id"],))
    payload = response()
    from web_radio.characters import _encode
    encoded = _encode(payload)
    with service.budget.connect() as db:
        db.execute("UPDATE voice_design_jobs SET response_sha256=? WHERE id=?", (hashlib.sha256(encoded).hexdigest(), job["id"]))
    service.characters.write_artifact(f"design-jobs/{job['id']}/response.json", encoded)
    result = service.recover(job["id"])
    assert result["status"] == "ready"
    assert len(service.characters.recipes("moss")) == 1
    assert totals(service)["spent_microdollars"] == 100000
    assert service.recover(job["id"])["recipe"]["recipe_id"] == result["recipe"]["recipe_id"]


def test_success_accounting_and_job_ready_commit_in_same_transaction(service, monkeypatch):
    job = service.enqueue(request())
    original = service.budget.settle
    def crash_after_settlement(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("fixture crash before job update")
    monkeypatch.setattr(service.budget, "settle", crash_after_settlement)
    result = service.run_one_specific(job["id"], transport=httpx.MockTransport(lambda call: httpx.Response(200, json=response())))
    assert result["status"] == "uncertain"
    assert totals(service)["reserved_microdollars"] == 100000
    assert totals(service)["spent_microdollars"] == 0
    monkeypatch.setattr(service.budget, "settle", original)
    assert service.recover(job["id"])["status"] == "ready"
    assert len(service.characters.recipes("moss")) == 1
    assert totals(service)["spent_microdollars"] == 100000


@pytest.mark.parametrize("digest", ["missing", "mismatched"])
def test_recovery_does_not_promote_unproven_or_swapped_response(service, digest):
    job = service.enqueue(request())
    from web_radio.characters import _encode
    payload = response()
    recorded = hashlib.sha256(_encode(payload)).hexdigest() if digest == "mismatched" else None
    with service.budget.connect() as db:
        db.execute("UPDATE voice_design_jobs SET status='running',response_sha256=? WHERE id=?", (recorded, job["id"]))
    if digest == "mismatched":
        payload["id"] = "voice_swapped_response"
    service.characters.write_artifact(f"design-jobs/{job['id']}/response.json", _encode(payload))
    assert service.recover(job["id"])["status"] == "uncertain"
    assert service.characters.recipes("moss") == []
    assert totals(service)["reserved_microdollars"] == 100000


def test_delayed_recovery_keeps_original_retention_bound(service):
    from web_radio.characters import _encode, parse_timestamp
    job = service.enqueue(request())
    creation = datetime.now(timezone.utc) - timedelta(days=20)
    encoded = _encode(response())
    with service.budget.connect() as db:
        db.execute(
            "UPDATE voice_design_jobs SET status='running',dispatched_at=?,response_sha256=? WHERE id=?",
            (creation.isoformat(), hashlib.sha256(encoded).hexdigest(), job["id"]),
        )
    service.characters.write_artifact(f"design-jobs/{job['id']}/response.json", encoded)
    result = service.recover(job["id"])
    assert result["status"] == "ready"
    assert parse_timestamp(result["recipe"]["expires_at"]) == creation + timedelta(days=365)
    assert parse_timestamp(result["recipe"]["created_at"]) == creation


def test_offline_backup_lease_blocks_design_mutations_without_changing_jobs(service):
    job = service.enqueue(request())
    with service.budget.connect() as db:
        db.execute("UPDATE voice_design_jobs SET status='uncertain' WHERE id=?", (job["id"],))
    lock_path = service.root / ".runtime/studio/design-worker.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for action in (
            lambda: service.enqueue(request()),
            lambda: service.configure_budget("moss", "1", "1"),
            lambda: service.run_one_specific(job["id"]),
            lambda: service.recover(job["id"]),
            lambda: service.reconcile(job["id"], "0", "Confirmed fixture zero charge."),
        ):
            with pytest.raises(ValueError, match="backup"):
                action()
    assert service.status(job["id"])["dispatch_count"] == 0
    assert service.status(job["id"])["status"] == "uncertain"
    assert totals(service)["reserved_microdollars"] == 100000


def test_reconciliation_audit_and_charge_commit_together_even_if_export_fails(service, monkeypatch):
    job = service.enqueue(request())
    with service.budget.connect() as db:
        db.execute("UPDATE voice_design_jobs SET status='uncertain' WHERE id=?", (job["id"],))
    original = service.budget.settle
    def fail_after_settle(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("fixture transaction interruption")
    monkeypatch.setattr(service.budget, "settle", fail_after_settle)
    with pytest.raises(RuntimeError):
        service.reconcile(job["id"], "0.02", "Fixture provider billing confirmed.")
    assert totals(service)["reserved_microdollars"] == 100000
    assert service.status(job["id"])["reconciliations"] == []
    monkeypatch.setattr(service.budget, "settle", original)
    def no_export(*args, **kwargs):
        raise OSError("fixture export interrupted")
    monkeypatch.setattr(service.characters, "write_artifact", no_export)
    with pytest.raises(OSError):
        service.reconcile(job["id"], "0.02", "Fixture provider billing confirmed.")
    persisted = service.status(job["id"])
    assert persisted["allocation"]["state"] == "spent"
    assert persisted["allocation"]["charged_micro"] == 20000
    assert persisted["reconciliations"][0]["charged_microdollars"] == 20000


def test_corrupted_retained_request_fails_before_dispatch_and_releases_allocation(service):
    job = service.enqueue(request())
    with service.budget.connect() as db:
        bad = {**job["request"], "voice": {"model": MODEL, "type": "replicated"}}
        db.execute("UPDATE voice_design_jobs SET request_json=? WHERE id=?", (json.dumps(bad), job["id"]))
    result = service.run_one_specific(job["id"], transport=httpx.MockTransport(lambda call: pytest.fail("Bad retained request dispatched")))
    assert result["status"] == "failed"
    assert result["dispatch_count"] == 0
    assert totals(service)["reserved_microdollars"] == 0


@pytest.mark.parametrize("returned_model", [MODEL, "models/" + MODEL])
def test_provider_model_resource_name_preserves_exact_provenance_and_recovers(service, returned_model):
    job = service.enqueue(request())
    payload = response(model=returned_model)
    from web_radio.characters import _encode
    encoded = _encode(payload)
    with service.budget.connect() as db:
        db.execute("UPDATE voice_design_jobs SET status='running',dispatch_count=1,response_sha256=? WHERE id=?",
                   (hashlib.sha256(encoded).hexdigest(), job["id"]))
    path = service.characters.write_artifact(f"design-jobs/{job['id']}/response.json", encoded)
    recovered = service.recover(job["id"])
    assert recovered["status"] == "ready"
    assert recovered["dispatch_count"] == 1
    assert recovered["recipe"]["model"] == MODEL
    assert path.read_bytes() == encoded
    assert json.loads(path.read_bytes())["model"] == returned_model
    assert service.characters.approved_designed_voices() == {"voice_original_moss"}


@pytest.mark.parametrize("returned_model", ["gemini-3.8-flash-tts", "models/gemini-3.8-flash-tts",
                                           "models/models/" + MODEL])
def test_different_or_malformed_provider_model_cannot_register_a_voice(service, returned_model):
    job = service.enqueue(request())
    result = service.run_one_specific(job["id"], transport=httpx.MockTransport(
        lambda call: httpx.Response(200, json=response(model=returned_model))))
    assert result["status"] == "uncertain"
    assert service.characters.recipes("moss") == []
    assert totals(service)["reserved_microdollars"] == 100000


def test_design_request_from_preset_needs_no_operator_fabricated_quote_fields():
    preset = DesignRequest.from_preset("moss", "A patient original woodland creature.", MODEL)
    assert preset.quoted_cost_usd == "0.250000"
    assert preset.allocation_preset == "standard"
    assert "not a provider quote" in preset.quote_evidence
    changed = preset.model_dump()
    changed["quoted_cost_usd"] = "0.01"
    with pytest.raises(ValueError, match="versioned local policy"):
        DesignRequest.model_validate(changed)


def test_design_preset_queues_against_explicit_budget_without_dispatch(service):
    job = service.enqueue_preset("moss", "A patient original woodland creature.", MODEL)
    assert job["status"] == "queued" and job["dispatch_count"] == 0
    assert job["quote"]["basis"] == "versioned-local-allocation-policy"
    assert job["quote"]["allocation_preset"] == "standard"
    assert job["reservation_microdollars"] == 250000
    assert job["diagnostic"] is None and job["usage_estimate"] is None


def test_design_rejection_retains_private_sanitized_diagnostic_and_authoritative_hash(service):
    job = service.enqueue(request())
    key = "fixture-key-never-a-real-credential"
    body = {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "Provider-specific rejection " + key,
                     "details": [{"@type": "type.googleapis.com/google.rpc.BadRequest",
                                  "fieldViolations": [{"field": "voice.prompted.input",
                                                       "description": "Provider did not accept this description."}]}]}}
    failed = service.run_one_specific(job["id"], transport=httpx.MockTransport(lambda _: httpx.Response(400, json=body)))
    assert failed["status"] == "failed" and failed["dispatch_count"] == 1
    assert failed["diagnostic"]["classification"] == "provider-rejected"
    assert failed["diagnostic"]["field_violations"][0]["field"] == "voice.prompted.input"
    assert failed["allocation"]["state"] == "reserved"
    artifact = service.characters.artifact(f"design-jobs/{job['id']}/failure-diagnostic.json")
    raw = artifact.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == failed["diagnostic_sha256"]
    assert json.loads(raw) == failed["diagnostic"]
    assert key not in raw.decode()


def test_design_usage_estimate_is_separate_from_conservative_allocation_and_invoice(service):
    job = service.enqueue(request())
    ready = service.run_one_specific(job["id"], transport=httpx.MockTransport(lambda _: httpx.Response(
        200, json=response(usage={"total_input_tokens": 20, "total_output_tokens": 25}))))
    assert ready["status"] == "ready"
    assert ready["usage_estimate"]["estimated_microdollars"] == 160
    assert not ready["usage_estimate"]["is_invoice_charge"]
    assert ready["allocation"]["charged_micro"] == 100000


def test_design_constructor_cannot_mutate_database_while_backup_holds_exclusive_lock(tmp_path):
    directory = tmp_path / ".runtime/studio"
    directory.mkdir(parents=True)
    with (directory / "design-worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="backup"):
            VoiceDesignService(tmp_path, key_provider=lambda: "fixture")
    assert not (directory / "studio.sqlite3").exists()
    assert not (tmp_path / ".runtime/characters/characters.sqlite3").exists()
