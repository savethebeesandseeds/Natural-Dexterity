"""Isolated source-embedding fixtures. No provider, network, real voice, or secrets."""
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import struct
import wave

from fastapi.testclient import TestClient
import httpx
import pytest

from web_radio import embedding_contract as contract, listener, station
from web_radio.contract import create_release

TEXT = "Moss: The radio is a fixture.\nWren: Only a fixture.\n"


@pytest.fixture
def accepted(tmp_path):
    runtime = tmp_path / "runtime"
    source = tmp_path / "fixture.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\0\0" * 6000)
    release = create_release(source, TEXT, runtime / "library/releases/fixture-embeddings",
                             "fixture", "ISOLATED EMBEDDING FIXTURE",
                             release_id="fixture-embeddings")
    (runtime / "library/catalog.json").write_text(json.dumps({
        "schema_version": 1,
        "releases": [{"release_id": release.release_id, "package_sha256": release.package_sha256}],
    }))
    app = station.create_app(runtime, start_engine=False)
    app.state.playout.refresh()
    assert release.release_id in app.state.playout.known
    return runtime, release, app


def windows(release, *, modality="text", count=1):
    assets = {item.role: item.sha256 for item in release.assets}
    rows = []
    for index in range(count):
        row = {"id": f"window-{index}-{modality}", "passage_id": f"passage-{index}",
               "source_modality": modality, "character_id": "moss" if modality == "text" else None,
               "start_ms": 0, "end_ms": 250,
               "timing_evidence": "Authored fixture timing; not inferred from Gemini.",
               "request_sha256": "1" * 64, "response_sha256": "2" * 64}
        if modality in ("text", "text_audio"):
            row["text"] = "Moss: The radio is a fixture."
            row["text_source"] = {"transcript_sha256": assets["transcript"], "start_byte": 0,
                                  "end_byte": len(row["text"].encode())}
        if modality in ("audio", "text_audio"):
            row["audio"] = {"source_sha256": assets["audio"], "window_sha256": "3" * 64,
                            "source_start_ms": 0, "kind": "release_mix",
                            "alignment_evidence": "Exact fixture release mix crop.",
                            "source_sample_rate": 24000, "source_start_frame": 0,
                            "source_end_frame": 6000, "extraction": "pcm_s16le_wav_v1"}
        rows.append(row)
    return rows


def build(tmp_path, release, *, modality="text", count=1, dimensions=128, name="source"):
    rows = windows(release, modality=modality, count=count)
    source = tmp_path / name
    target = contract.write_bundle(source, release, {"dimensions": dimensions},
                                   rows, [[0.125] * dimensions for _ in rows])
    return target, rows


def update_document(source, change):
    path = source if source.name == "manifest.json" else source / "manifest.json"
    data = json.loads(path.read_bytes())
    change(data)
    data["bundle_id"] = contract.bundle_digest(data)
    path.write_text(json.dumps(data))
    return data


def test_shared_unprojected_space_and_immutable_deterministic_bundle(accepted, tmp_path):
    _, release, _ = accepted
    text = windows(release)[0]
    audio = windows(release, modality="audio")[0]
    both = windows(release, modality="text_audio")[0]
    audio["passage_id"] = text["passage_id"]
    both["passage_id"] = text["passage_id"]
    rows = [text, audio, both]
    source = tmp_path / "multimodal"
    target = contract.write_bundle(source, release, {"dimensions": 3072}, rows,
                                   [[0.125] * 3072 for _ in rows])
    value, shards = contract.read_bundle(target, release)
    assert value.kind == "source_embeddings" and value.schema_version == 1
    assert {row.source_modality for row in value.windows} == {"text", "audio", "text_audio"}
    assert value.space_id == contract.EmbeddingConfig(dimensions=3072).space_id
    assert all(len(data) == 3 * 3072 * 4 for data in shards.values())
    assert "projection" not in value.model_dump_json()
    assert struct.unpack("<f", next(iter(shards.values()))[:4]) == (0.125,)
    assert target == contract.write_bundle(source, release, {"dimensions": 3072}, rows,
                                           [[0.125] * 3072 for _ in rows])
    with pytest.raises(contract.EmbeddingError, match="conflicts"):
        contract.write_bundle(source, release, {"dimensions": 3072}, rows,
                              [[0.25] * 3072 for _ in rows])


def test_shards_have_at_most_64_vectors_and_the_document_is_bounded(accepted, tmp_path):
    _, release, _ = accepted
    source, _ = build(tmp_path, release, count=65, dimensions=3072)
    value, shards = contract.read_bundle(source, release)
    assert [item.count for item in value.shards] == [64, 1]
    assert [len(data) for data in shards.values()] == [786432, 12288]
    assert source.stat().st_size <= contract.MAX_EMBEDDING_MANIFEST_BYTES
    assert max(row.vector_index for row in value.windows) == 63


@pytest.mark.parametrize("mutation", [
    lambda rows: rows[0].update(source_modality="image"),
    lambda rows: rows[0].update(character_id="../moss"),
    lambda rows: rows[0].update(start_ms=True),
    lambda rows: rows[0].update(end_ms=251),
    lambda rows: rows[0].update(end_ms=0),
    lambda rows: rows[0].update(text=""),
    lambda rows: rows[0].update(text_source=None),
    lambda rows: rows[0].update(timing_evidence=" "),
    lambda rows: rows[0].update(request_sha256="not-a-hash"),
    lambda rows: rows[0].update(extra="private"),
    lambda rows: rows[0]["text_source"].update(transcript_sha256="0" * 64),
    lambda rows: rows[0]["text_source"].update(end_byte=1000),
    lambda rows: rows.append(copy.deepcopy(rows[0])),
])
def test_preflight_rejects_invalid_source_metadata_before_generation(accepted, mutation):
    _, release, _ = accepted
    rows = windows(release)
    mutation(rows)
    with pytest.raises(contract.EmbeddingError):
        contract.preflight_windows(release, {"dimensions": 128}, rows)


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(character_id="moss"),
    lambda row: row.update(audio=None),
    lambda row: row["audio"].update(source_sha256="0" * 64),
    lambda row: row["audio"].update(source_start_ms=5),
    lambda row: row["audio"].update(source_end_frame=5000),
    lambda row: row["audio"].update(source_end_frame=0),
    lambda row: row["audio"].update(alignment_evidence=""),
    lambda row: row["audio"].update(extraction="unknown"),
    lambda row: row["audio"].update(source_sample_rate=True),
    lambda row: row["audio"].update(kind="character_stem"),
])
def test_audio_identity_and_mixed_speaker_attribution_are_strict(accepted, mutation):
    _, release, _ = accepted
    rows = windows(release, modality="audio")
    mutation(rows[0])
    with pytest.raises(contract.EmbeddingError):
        contract.preflight_windows(release, {"dimensions": 128}, rows)


def test_character_stem_retains_exact_crop_identity_with_authored_evidence(accepted):
    _, release, _ = accepted
    rows = windows(release, modality="audio")
    rows[0]["character_id"] = "moss"
    rows[0]["audio"].update(kind="character_stem", source_sha256="8" * 64,
                             alignment_evidence="Editorial stem assembly fixture mapping.")
    assert contract.preflight_windows(release, {"dimensions": 128}, rows)[0].character_id == "moss"


@pytest.mark.parametrize("config", [
    {"dimensions": 127}, {"dimensions": 3073}, {"dimensions": True},
    {"dimensions": 128, "model": "different-model"},
    {"dimensions": 128, "task_prefix": "RETRIEVAL_QUERY"},
    {"dimensions": 128, "projection": 3},
    {"dimensions": 128, "input_format": "task-specific"},
])
def test_all_modalities_use_only_fixed_common_configuration(accepted, config):
    _, release, _ = accepted
    with pytest.raises(contract.EmbeddingError):
        contract.preflight_windows(release, config, windows(release))


@pytest.mark.parametrize("vector", [
    [0.0] * 127, [float("nan")] * 128, [float("inf")] * 128,
    [1e39] * 128, [True] * 128, ["0.0"] * 128,
])
def test_vectors_are_bounded_finite_float32(accepted, tmp_path, vector):
    _, release, _ = accepted
    with pytest.raises(contract.EmbeddingError):
        contract.write_bundle(tmp_path / "invalid", release, {"dimensions": 128},
                              windows(release), [vector])
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(schema_version=True),
    lambda d: d.update(schema_version=2),
    lambda d: d.update(kind="performance"),
    lambda d: d.update(release_id="different-release"),
    lambda d: d.update(package_sha256="0" * 64),
    lambda d: d.update(release_audio_sha256="0" * 64),
    lambda d: d.update(release_transcript_sha256="0" * 64),
    lambda d: d.update(duration_ms=200),
    lambda d: d.update(space_id="0" * 64),
    lambda d: d.update(vector_format="float64"),
    lambda d: d["windows"][0].update(vector_index=1),
    lambda d: d["windows"][0].update(shard_id="vectors-9999.f32"),
    lambda d: d["shards"][0].update(count=65),
    lambda d: d["shards"][0].update(size_bytes=1),
])
def test_manifest_identity_schema_and_vector_references_are_checked(accepted, tmp_path, mutation):
    _, release, _ = accepted
    source, _ = build(tmp_path, release)
    value = update_document(source, mutation)
    with pytest.raises(contract.EmbeddingError):
        contract.validate_document(json.dumps(value).encode(), release)


def test_preflight_byte_bound_and_duplicate_keys(accepted, tmp_path, monkeypatch):
    _, release, _ = accepted
    source, _ = build(tmp_path, release)
    raw = source.read_bytes()
    with pytest.raises(contract.EmbeddingError):
        contract.validate_document(b'{"schema_version":1,' + raw[1:], release)
    with pytest.raises(contract.EmbeddingError):
        contract.validate_document(b'{"a":NaN}', release)
    with monkeypatch.context() as limits:
        limits.setattr(contract, "MAX_EMBEDDING_MANIFEST_BYTES", 64)
        with pytest.raises(contract.EmbeddingError):
            contract.preflight_windows(release, {"dimensions": 128}, windows(release))
    with pytest.raises(contract.EmbeddingError):
        contract.preflight_windows(release, {"dimensions": 128}, windows(release, count=2001))


def test_shards_require_exact_inventory_checksum_shape_and_finite_values(accepted, tmp_path):
    _, release, _ = accepted
    source, _ = build(tmp_path, release)
    directory = source.parent
    shard = directory / "vectors-0000.f32"
    first = shard.read_bytes()
    shard.write_bytes(first[:-4])
    with pytest.raises(contract.EmbeddingError):
        contract.read_bundle(source, release)
    invalid = struct.pack("<f", float("nan")) + first[4:]
    shard.write_bytes(invalid)
    update_document(source, lambda d: d["shards"][0].update(sha256=hashlib.sha256(invalid).hexdigest()))
    with pytest.raises(contract.EmbeddingError):
        contract.read_bundle(source, release)
    shard.write_bytes(first)
    update_document(source, lambda d: d["shards"][0].update(sha256=hashlib.sha256(first).hexdigest()))
    (directory / "extra.bin").write_bytes(b"private")
    with pytest.raises(contract.EmbeddingError):
        contract.read_bundle(source, release)


def test_install_requires_accepted_owned_release_and_preserves_every_bundle(accepted, tmp_path):
    runtime, release, app = accepted
    package = app.state.playout.known[release.release_id][1]
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in package.iterdir()}
    source, rows = build(tmp_path, release)
    with pytest.raises(contract.EmbeddingError):
        contract.install_bundle(source, tmp_path / "unaccepted")
    assert not (tmp_path / "unaccepted").exists()
    first = contract.install_bundle(source, runtime)
    assert first == contract.install_bundle(source.parent, runtime)
    other = contract.write_bundle(tmp_path / "other", release, {"dimensions": 256}, rows, [[0.25] * 256])
    second = contract.install_bundle(other, runtime)
    assert first != second and first.is_file() and second.is_file()
    assert len(contract.bundle_catalog(runtime, release.release_id)["bundles"]) == 2
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in package.iterdir()} == before


def test_install_verifies_exact_text_against_accepted_transcript(accepted, tmp_path):
    runtime, release, _ = accepted
    rows = windows(release)
    rows[0]["text"] = rows[0]["text"].replace("Moss", "Fake")
    source = contract.write_bundle(tmp_path / "forged", release, {"dimensions": 128}, rows, [[0.25] * 128])
    with pytest.raises(contract.EmbeddingError, match="exact text"):
        contract.install_bundle(source, runtime)
    assert not (runtime / "station/embeddings").exists()


def test_symbolic_sources_shards_and_destination_ancestors_are_rejected(accepted, tmp_path):
    runtime, release, _ = accepted
    source, _ = build(tmp_path, release)
    alias = tmp_path / "alias"
    alias.symlink_to(source.parent, target_is_directory=True)
    with pytest.raises(contract.EmbeddingError):
        contract.install_bundle(alias, runtime)
    shard = source.parent / "vectors-0000.f32"
    held = tmp_path / "held.f32"
    shard.rename(held)
    shard.symlink_to(held)
    with pytest.raises(contract.EmbeddingError):
        contract.install_bundle(source, runtime)
    shard.unlink()
    held.rename(shard)
    (runtime / "station/embeddings").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(contract.EmbeddingError):
        contract.install_bundle(source, runtime)
    assert not (tmp_path / "elsewhere").exists()


def test_public_embedding_routes_optional_read_only_bounded_and_verified(accepted, tmp_path):
    runtime, release, app = accepted
    route = "/api/embeddings/" + release.release_id
    source, _ = build(tmp_path, release)
    with TestClient(app) as http:
        assert http.get(route).status_code == 404
        assert http.get("/api/embeddings/Invalid").status_code == 404
        assert http.get("/api/embeddings/unknown").status_code == 404
        assert http.get(route + "?url=private").status_code == 422
        assert http.post(route, json={}).status_code == 405
        installed = contract.install_bundle(source, runtime)
        value = json.loads(installed.read_bytes())
        bundle_route = route + "/" + value["bundle_id"]
        response = http.get(route)
        assert response.status_code == 200 and len(response.json()["bundles"]) == 1
        response = http.get(bundle_route)
        assert response.status_code == 200 and response.json()["space_id"] == value["space_id"]
        assert response.headers["cache-control"] == "no-store"
        assert "script-src 'self'" in response.headers["content-security-policy"]
        shard_route = bundle_route + "/vectors-0000.f32"
        response = http.get(shard_route)
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.content == (installed.parent / "vectors-0000.f32").read_bytes()
        assert http.post(shard_route).status_code == 405
        assert http.get(shard_route + "?source=private").status_code == 422
        assert http.get(bundle_route + "/vectors-9999.f32").status_code == 404
        assert http.get(route + "/" + "0" * 64).status_code == 404
        (installed.parent / "vectors-0000.f32").write_bytes(b"corrupt")
        for path in (route, bundle_route, shard_route):
            result = http.get(path)
            assert result.status_code == 409 and str(runtime) not in result.text


@pytest.mark.parametrize("kind", ["catalog", "manifest", "shard"])
def test_listener_uses_only_fixed_get_upstream_and_strips_credentials(kind):
    route = "/api/embeddings/fixture-embeddings"
    if kind in ("manifest", "shard"):
        route += "/" + "1" * 64
    if kind == "shard":
        route += "/vectors-0000.f32"
    calls = []
    def upstream(request):
        calls.append(request)
        assert str(request.url) == "http://127.0.0.1:43121" + route
        assert request.method == "GET"
        assert "authorization" not in request.headers and "cookie" not in request.headers
        assert request.headers["accept-encoding"] == "identity"
        if kind == "shard":
            return httpx.Response(200, content=b"\0" * 512,
                                  headers={"Content-Type": "application/octet-stream", "Set-Cookie": "secret"})
        return httpx.Response(200, json={"schema_version": 1}, headers={"Set-Cookie": "secret"})
    with TestClient(listener.create_app(transport=httpx.MockTransport(upstream))) as http:
        response = http.get(route, headers={"Authorization": "secret", "Cookie": "secret"})
        assert response.status_code == 200 and "set-cookie" not in response.headers
        assert response.headers["cache-control"] == "no-store"
        assert http.get(route + "?url=https://example.invalid/").status_code == 422
        assert http.post(route).status_code == 405
    assert len(calls) == 1


@pytest.mark.parametrize("status,expected", [(404, 404), (409, 409), (500, 503), (302, 503)])
def test_listener_missing_invalid_and_upstream_failures_are_sanitized(status, expected):
    transport = httpx.MockTransport(lambda request: httpx.Response(status, text="/private/runtime/path"))
    with TestClient(listener.create_app(transport=transport)) as http:
        result = http.get("/api/embeddings/fixture")
        assert result.status_code == expected and "/private/runtime/path" not in result.text


@pytest.mark.parametrize("headers,body", [
    ({"Content-Type": "application/json"}, b"binary"),
    ({"Content-Type": "application/octet-stream", "Content-Length": "99999999"}, b"tiny"),
    ({"Content-Type": "application/octet-stream"}, b"x" * 786433),
    ({"Content-Type": "application/octet-stream", "Content-Encoding": "gzip"}, b"compressed"),
])
def test_listener_rejects_wrong_type_encoding_and_oversized_binary(headers, body):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body, headers=headers))
    with TestClient(listener.create_app(transport=transport)) as http:
        result = http.get("/api/embeddings/fixture/" + "1" * 64 + "/vectors-0000.f32")
        assert result.status_code == 503


def test_listener_rejects_path_like_ids_before_upstream():
    calls = []
    def upstream(request):
        calls.append(request)
        return httpx.Response(500)
    with TestClient(listener.create_app(transport=httpx.MockTransport(upstream))) as http:
        for path in ("/api/embeddings/Invalid", "/api/embeddings/fixture/Invalid",
                     "/api/embeddings/fixture/" + "1" * 64 + "/private.bin"):
            assert http.get(path).status_code == 404
    assert not calls


def test_standalone_install_refuses_exclusive_backup_without_mutation(accepted, tmp_path):
    runtime, release, _ = accepted
    source, _ = build(tmp_path, release)
    studio = runtime / "studio"
    studio.mkdir()
    lock = studio / "generation-worker.lock"
    with lock.open("a+b") as lease:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = {path.relative_to(runtime).as_posix() for path in runtime.rglob("*")}
        with pytest.raises(contract.EmbeddingError, match="backup"):
            contract.install_bundle(source, runtime)
        assert {path.relative_to(runtime).as_posix() for path in runtime.rglob("*")} == before
        assert not (runtime / "station/embeddings").exists()
        fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
    assert contract.install_bundle(source, runtime).is_file()


def test_provider_revision_fingerprints_reported_weights_without_claiming_unknown_stability():
    missing = contract.EmbeddingConfig(dimensions=128)
    first = contract.EmbeddingConfig(dimensions=128, provider_revision="revision-one")
    second = contract.EmbeddingConfig(dimensions=128, provider_revision="revision-two")
    assert len({missing.space_id, first.space_id, second.space_id}) == 3
    for revision in ("", " ", "x" * 257, "\0"):
        with pytest.raises(ValueError):
            contract.EmbeddingConfig(dimensions=128, provider_revision=revision)
