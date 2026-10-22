"""Isolated timed-embedding pipeline acceptance; mocked providers only."""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import sqlite3
import wave

import httpx
import pytest
from pydantic import ValidationError
from fastapi.testclient import TestClient

from web_radio import station, embedding_service as module
from web_radio.budget import BudgetError
from web_radio.contract import create_release
from web_radio.embedding_contract import read_bundle

TEXT = "The rain is near."


@pytest.fixture
def prepared(tmp_path):
    runtime = tmp_path / ".runtime"
    source = tmp_path / "content/stem.wav"
    source.parent.mkdir()
    with wave.open(str(source), "wb") as stream:
        stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
        stream.writeframes(b"\0\0" * 32000)
    package = runtime / "library/releases/fixture-embedding"
    manifest = create_release(source, "Moss: " + TEXT, package, "fixture", "ISOLATED SILENCE FIXTURE",
                              release_id="fixture-embedding")
    (runtime / "library/catalog.json").write_text(json.dumps({
        "schema_version": 1, "releases": [{"release_id": manifest.release_id,
                                           "package_sha256": manifest.package_sha256}]}))
    app = station.create_app(runtime, start_engine=False)
    app.state.playout.refresh()
    accepted = app.state.playout.known[manifest.release_id][1]
    audio_asset = next(item for item in manifest.assets if item.role == "audio")
    spec = {
        "schema_version": 1, "release_id": manifest.release_id,
        "package_sha256": manifest.package_sha256, "duration_ms": manifest.duration_ms,
        "dimensions": 128, "windows": [{
            "window_id": "passage-one", "passage_id": "passage-one",
            "character_id": None, "start_ms": 0, "end_ms": 1000,
            "timing_evidence": "Explicit editorial alignment for isolated fixture.",
            "text": TEXT, "text_start_byte": 6,
            "audio": {"kind": "release_mix", "source_sha256": audio_asset.sha256,
                      "source_start_frame": 0,
                      "alignment_evidence": "Exact accepted fixture audio and authored interval."},
            "modalities": ["text", "audio", "text_audio"],
        }]}
    service = module.EmbeddingService(tmp_path, key_provider=lambda: "fixture-only-never-real")
    return service, spec, app, accepted, source


def response(request):
    dimensions = json.loads(request.content)["embedContentConfig"]["outputDimensionality"]
    return httpx.Response(200, json={"embedding": {"values": [0.25] * dimensions},
                                    "usageMetadata": {"promptTokenCount": 20}})


def enqueue(service, spec, budget="1"):
    return service.enqueue(spec, episode_limit_usd=budget, series_limit_usd=budget)


def snapshots(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()}


def test_all_modalities_retain_comparable_provenance_and_install_read_only(prepared):
    service, spec, app, accepted, _ = prepared
    before = snapshots(accepted)
    plan = service.plan(spec)
    assert plan["dispatch"] is False and plan["reserved"] is False
    assert plan["embedding_count"] == plan["unique_request_count"] == 3
    assert plan["reservation_ceiling_microdollars"] == 108135
    job = enqueue(service, spec)
    calls = []
    def perform(request):
        calls.append(json.loads(request.content))
        return response(request)
    result = service.run_one(job["id"], httpx.MockTransport(perform))
    assert result["status"] == "ready", result["error"]
    assert [item["state"] for item in result["requests"]] == ["completed"] * 3
    assert len(calls) == 3 and len(calls[2]["content"]["parts"]) == 2
    manifest, shards = read_bundle(service.root / result["bundle_path"],
                                   module.Manifest.model_validate(result["manifest"]))
    assert len(manifest.windows) == 3 and all(item.character_id is None for item in manifest.windows)
    assert {item.passage_id for item in manifest.windows} == {"passage-one"}
    assert manifest.config.model == "gemini-embedding-2" and manifest.config.dimensions == 128
    assert manifest.space_id == plan["space_id"]
    assert manifest.windows[0].text == TEXT
    assert manifest.windows[1].audio.source_end_frame == 16000
    assert manifest.windows[1].audio.window_sha256 != manifest.windows[1].audio.source_sha256
    assert result["budget"]["episode"]["reserved_microdollars"] == 0
    assert result["budget"]["episode"]["spent_microdollars"] == 108135
    service.install(job["id"])
    with TestClient(app) as client:
        catalog = client.get("/api/embeddings/" + spec["release_id"])
        assert catalog.status_code == 200
        base = "/api/embeddings/" + spec["release_id"] + "/" + manifest.bundle_id
        assert client.get(base).status_code == 200
        assert client.get(base + "/" + manifest.shards[0].id).content == shards[manifest.shards[0].id]
        assert client.post(base).status_code == 405
    assert snapshots(accepted) == before
    assert service.run_one(job["id"], httpx.MockTransport(lambda r: pytest.fail("Redispatched ready job")))["status"] == "ready"


def test_cached_results_survive_new_jobs_without_key_or_provider(prepared):
    service, spec, _, _, _ = prepared
    first = enqueue(service, spec)
    assert service.run_one(first["id"], httpx.MockTransport(response))["status"] == "ready"
    service.key_provider = lambda: pytest.fail("Cache hit read a credential")
    plan = service.plan(spec)
    assert plan["cache_hits"] == 3 and plan["reservation_ceiling_microdollars"] == 0
    second = enqueue(service, spec)
    assert all(item["attempt_id"] is None for item in second["requests"])
    result = service.run_one(second["id"], httpx.MockTransport(lambda r: pytest.fail("Cache hit dispatched")))
    assert result["status"] == "ready"
    assert result["bundle_path"] != service.status(first["id"])["bundle_path"]
    assert all(item["state"] == "cached" for item in result["requests"])


def test_cache_is_shared_across_character_ids_in_same_space(prepared):
    service, spec, _, _, _ = prepared
    spec["windows"][0].update(audio=None, modalities=["text"], character_id="moss")
    first = enqueue(service, spec)
    result = service.run_one(first["id"], httpx.MockTransport(response))
    assert result["status"] == "ready"
    spec["windows"][0]["character_id"] = "wren"
    assert service.plan(spec)["cache_hits"] == 1
    second = enqueue(service, spec)
    result = service.run_one(second["id"], httpx.MockTransport(lambda r: pytest.fail("Character prefix changed shared space")))
    assert result["status"] == "ready"
    assert result["rows"][0]["character_id"] == "wren"


def test_all_reservations_are_atomic_when_aggregate_budget_is_too_small(prepared):
    service, spec, *_ = prepared
    with pytest.raises(BudgetError):
        enqueue(service, spec, "0.10")
    with service.budget.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM embedding_jobs").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM embedding_requests").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] == 0


def test_partial_failure_preserves_completed_and_holds_only_dispatched(prepared):
    service, spec, *_ = prepared
    job = enqueue(service, spec)
    calls = []
    def perform(request):
        calls.append(request)
        if len(calls) == 2:
            raise httpx.ReadTimeout("isolated fixture", request=request)
        return response(request)
    result = service.run_one(job["id"], httpx.MockTransport(perform))
    assert result["status"] == "uncertain"
    assert [item["state"] for item in result["requests"]] == ["completed", "uncertain", "cancelled"]
    assert result["budget"]["episode"]["reserved_microdollars"] == 53248
    assert result["budget"]["episode"]["spent_microdollars"] == 1639
    assert service.run_one(job["id"], httpx.MockTransport(perform))["status"] == "uncertain"
    recovered = service.recover(job["id"])
    assert recovered["status"] == "uncertain" and len(calls) == 2
    service.reconcile(job["id"], result["requests"][1]["request_sha256"], "0",
                      "Fixture verified billing confirms no charge for timed out request.")
    assert service.status(job["id"])["budget"]["episode"]["reserved_microdollars"] == 0


def test_recovery_finishes_saved_response_without_network_or_key(prepared, monkeypatch):
    service, spec, *_ = prepared
    spec["windows"][0].update(audio=None, modalities=["text"])
    job = enqueue(service, spec)
    original = service._complete_step
    def interrupted(*args):
        raise OSError("Fixture interruption after durable response")
    monkeypatch.setattr(service, "_complete_step", interrupted)
    result = service.run_one(job["id"], httpx.MockTransport(response))
    assert result["status"] == "uncertain"
    monkeypatch.setattr(service, "_complete_step", original)
    service.key_provider = lambda: pytest.fail("Recovery read key")
    recovered = service.recover(job["id"])
    assert recovered["status"] == "ready"
    assert recovered["budget"]["episode"]["reserved_microdollars"] == 0


def test_response_tamper_is_not_a_cache_hit_or_recoverable_result(prepared, monkeypatch):
    service, spec, *_ = prepared
    spec["windows"][0].update(audio=None, modalities=["text"])
    job = enqueue(service, spec)
    original = service._complete_step
    monkeypatch.setattr(service, "_complete_step", lambda *a: (_ for _ in ()).throw(OSError("fixture")))
    result = service.run_one(job["id"], httpx.MockTransport(response))
    digest = result["requests"][0]["request_sha256"]
    (service.state / "jobs" / job["id"] / "responses" / (digest + ".json")).write_text(
        json.dumps({"embedding": {"values": [0.5] * 128}}))
    monkeypatch.setattr(service, "_complete_step", original)
    assert service.recover(job["id"])["status"] == "uncertain"
    assert service.status(job["id"])["budget"]["episode"]["reserved_microdollars"] == 1639


def test_cache_tamper_prevents_accidental_new_paid_request(prepared):
    service, spec, *_ = prepared
    job = enqueue(service, spec)
    result = service.run_one(job["id"], httpx.MockTransport(response))
    digest = result["requests"][0]["request_sha256"]
    (service.state / "cache" / digest / "response.json").write_text("{}")
    with pytest.raises(ValueError, match="verification"):
        service.plan(spec)


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(schema_version=True),
    lambda s: s["windows"][0].update(timing_evidence=""),
    lambda s: s["windows"][0].update(character_id="moss"),
    lambda s: s["windows"][0].update(text_start_byte=7),
    lambda s: s["windows"][0]["audio"].update(source_sha256="0" * 64),
    lambda s: s["windows"][0]["audio"].update(source_start_frame=1),
    lambda s: s["windows"][0].update(end_ms=30001),
])
def test_untrusted_or_mismatched_alignment_is_rejected_before_queue(prepared, mutation):
    service, spec, *_ = prepared
    mutation(spec)
    with pytest.raises(ValueError):
        service.plan(spec)
    with service.budget.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM embedding_jobs").fetchone()[0] == 0


def test_per_character_audio_requires_authored_known_stem_binding(prepared):
    service, spec, _, _, source = prepared
    spec["windows"][0].update(character_id="moss")
    spec["windows"][0]["audio"].update(
        kind="character_stem", path="content/stem.wav",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        alignment_evidence="Fixture editorial source: this known character stem maps to accepted release [0,1000).")
    job = enqueue(service, spec)
    result = service.run_one(job["id"], httpx.MockTransport(response))
    assert result["status"] == "ready"
    assert result["rows"][1]["character_id"] == "moss"
    assert result["rows"][1]["audio"]["kind"] == "character_stem"


def test_expired_prices_cancel_without_dispatch(prepared, monkeypatch):
    service, spec, *_ = prepared
    job = enqueue(service, spec)
    def expired(*args):
        raise BudgetError("Fixture pricing expired.")
    monkeypatch.setattr(module, "_allocation", expired)
    result = service.run_one(job["id"], httpx.MockTransport(lambda r: pytest.fail("Expired prices dispatched")))
    assert result["status"] == "failed"
    assert all(item["state"] == "cancelled" for item in result["requests"])
    assert result["budget"]["episode"]["reserved_microdollars"] == 0


def test_unresolved_equivalent_request_blocks_duplicate_paid_dispatch(prepared):
    service, spec, *_ = prepared
    spec["windows"][0].update(audio=None, modalities=["text"])
    first = enqueue(service, spec)
    def timeout(request):
        raise httpx.ReadTimeout("fixture", request=request)
    assert service.run_one(first["id"], httpx.MockTransport(timeout))["status"] == "uncertain"
    second = enqueue(service, spec)
    result = service.run_one(second["id"], httpx.MockTransport(lambda r: pytest.fail("Duplicate unresolved request")))
    assert result["status"] == "failed"
    assert result["requests"][0]["state"] == "cancelled"


def test_reconciliation_audit_failure_rolls_back_billing(prepared):
    service, spec, *_ = prepared
    spec["windows"][0].update(audio=None, modalities=["text"])
    job = enqueue(service, spec)
    def timeout(request):
        raise httpx.ReadTimeout("fixture", request=request)
    result = service.run_one(job["id"], httpx.MockTransport(timeout))
    with service.budget.connect() as db:
        db.execute("""CREATE TRIGGER fail_embedding_audit BEFORE INSERT ON embedding_reconciliations
                      BEGIN SELECT RAISE(ABORT,'fixture'); END""")
    with pytest.raises(sqlite3.IntegrityError):
        service.reconcile(job["id"], result["requests"][0]["request_sha256"], "0", "Fixture billing evidence is verified.")
    assert service.status(job["id"])["budget"]["episode"]["reserved_microdollars"] == 1639


def test_completed_conservative_charge_can_be_reconciled_with_evidence(prepared):
    service, spec, *_ = prepared
    spec["windows"][0].update(audio=None, modalities=["text"])
    job = enqueue(service, spec)
    result = service.run_one(job["id"], httpx.MockTransport(response))
    service.reconcile(job["id"], result["requests"][0]["request_sha256"], "0.000004",
                      "Fixture-only verified billing amount, not a provider price claim.")
    result = service.status(job["id"])
    assert result["status"] == "ready" and result["requests"][0]["state"] == "completed"
    assert result["budget"]["episode"]["spent_microdollars"] == 4


def test_backup_lease_blocks_direct_service_operations_and_constructor(prepared):
    service, spec, *_ = prepared
    lock = service.root / ".runtime/studio/generation-worker.lock"
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="backup"):
            service.plan(spec)
        with pytest.raises(ValueError, match="backup"):
            module.EmbeddingService(service.root)
        fcntl.flock(stream, fcntl.LOCK_UN)


def test_actual_wav_frames_override_rounded_manifest_duration(prepared, monkeypatch):
    service, spec, _, release, _ = prepared
    original = service._accepted
    manifest, directory = original(module.PreparationSpec.model_validate(spec))
    padded = manifest.model_copy(update={"duration_ms": manifest.duration_ms + 50})
    spec["duration_ms"] += 50
    spec["windows"][0]["end_ms"] = spec["duration_ms"]
    monkeypatch.setattr(service, "_accepted", lambda value: (padded, directory))
    with pytest.raises(ValueError, match="actual accepted audio"):
        service.plan(spec)


def test_reported_model_revision_is_bound_to_exported_space(prepared):
    service, spec, *_ = prepared
    job = enqueue(service, spec)
    def versioned(request):
        base = json.loads(response(request).content)
        return httpx.Response(200, json=base | {"modelVersion": "fixture-revision-one"})
    result = service.run_one(job["id"], httpx.MockTransport(versioned))
    assert result["status"] == "ready"
    manifest, _ = read_bundle(service.root / result["bundle_path"], module.Manifest.model_validate(result["manifest"]))
    assert manifest.config.provider_revision == "fixture-revision-one"
    assert manifest.space_id != service.plan(spec)["space_id"]


def test_mixed_reported_model_revisions_do_not_publish_one_space(prepared):
    service, spec, *_ = prepared
    job = enqueue(service, spec)
    calls = []
    def versioned(request):
        calls.append(request)
        base = json.loads(response(request).content)
        return httpx.Response(200, json=base | {"modelVersion": "fixture-revision-" + str(len(calls))})
    result = service.run_one(job["id"], httpx.MockTransport(versioned))
    assert result["status"] == "failed" and result["bundle_path"] is None
    assert len(calls) == 3
    assert all(item["state"] == "completed" for item in result["requests"])
    assert service.recover(job["id"])["status"] == "failed"


def test_delayed_recovery_keeps_manually_reconciled_charge(prepared, monkeypatch):
    service, spec, *_ = prepared
    spec["windows"][0].update(audio=None, modalities=["text"])
    job = enqueue(service, spec)
    original = service._complete_step
    monkeypatch.setattr(service, "_complete_step", lambda *a: (_ for _ in ()).throw(OSError("fixture")))
    result = service.run_one(job["id"], httpx.MockTransport(response))
    service.reconcile(job["id"], result["requests"][0]["request_sha256"], "0.000004",
                      "Fixture-only billing evidence verified after interruption.")
    monkeypatch.setattr(service, "_complete_step", original)
    result = service.recover(job["id"])
    assert result["status"] == "ready"
    assert result["budget"]["episode"]["spent_microdollars"] == 4


def test_actual_cli_can_plan_and_queue_without_provider_access(prepared, capsys):
    from web_radio.embedding_cli import main
    service, spec, *_ = prepared
    path = service.root / "preparation.json"
    path.write_text(json.dumps(spec))
    prefix = ["--root", str(service.root)]
    assert main(prefix + ["plan", "--file", str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] and result["result"]["dispatch"] is False
    assert main(prefix + ["enqueue", "--file", str(path), "--episode-budget", "1", "--series-budget", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["ok"] and result["result"]["status"] == "queued"
    assert all(row["state"] == "pending" for row in result["result"]["requests"])


def test_per_character_audio_accepts_known_scene_export_stem_only(prepared):
    service, spec, _, _, source = prepared
    relative = ".runtime/studio/scenes/" + "f" * 32 + "/output/stems/moss.wav"
    stem = service.root / relative
    stem.parent.mkdir(parents=True)
    original = source.read_bytes()
    stem.write_bytes(original)
    spec["windows"][0].update(character_id="moss")
    spec["windows"][0]["audio"].update(
        kind="character_stem", path=relative,
        source_sha256=hashlib.sha256(original).hexdigest(),
        alignment_evidence="Fixture scene assembly: retained Moss turn is mapped to accepted release [0,1000).")
    job = enqueue(service, spec)
    result = service.run_one(job["id"], httpx.MockTransport(response))
    assert result["status"] == "ready", result["error"]
    audio = result["rows"][1]["audio"]
    assert result["rows"][1]["character_id"] == "moss"
    assert audio["kind"] == "character_stem"
    assert audio["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert audio["source_start_frame"] == 0 and audio["source_end_frame"] == 16000
    assert stem.read_bytes() == original
    # The scene master and arbitrary scene files remain outside the stem allowlist.
    spec["windows"][0]["audio"]["path"] = relative.replace("stems/moss.wav", "scene.wav")
    with pytest.raises(ValueError, match="stem"):
        service.plan(spec)
