"""Local audition preview tests never publish, schedule, generate or call providers."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import wave

import httpx
import pytest
from fastapi.testclient import TestClient

from web_radio import listener, preview, station


def encode(value):
    return json.dumps(value, separators=(",", ":")).encode()


def descriptor(name, data):
    return {"file": name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def write_asset(root, manifest, role, value):
    data = value if isinstance(value, bytes) else encode(value)
    (root / manifest[role]["file"]).write_bytes(data)
    manifest[role] = descriptor(manifest[role]["file"], data)
    (root / "manifest.json").write_bytes(encode(manifest))


@pytest.fixture(params=["listener", "station"])
def bundle(tmp_path, request):
    runtime = tmp_path / "runtime"
    root = runtime / "visual-previews/fixture-audition"
    root.mkdir(parents=True)
    stream = io.BytesIO()
    with wave.open(stream, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 8000)
    audio_bytes = stream.getvalue()
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    performance = {
        "schema_version": 1, "preview_id": "fixture-audition", "audio_sha256": audio_hash,
        "duration_ms": 500,
        "cast": [{"id": "one", "color": "#508b82"}, {"id": "two", "color": "#937794"}],
        "cues": [{"speaker_id": "one", "start_ms": 0, "end_ms": 250},
                 {"speaker_id": "two", "start_ms": 250, "end_ms": 500}],
        "features": {"hop_ms": 50, "tracks": {
            char: {band: [0.0] * 11 for band in ("energy", "low", "mid", "high")}
            for char in ("one", "two")
        }},
    }
    performance_bytes = encode(performance)
    manifest = {
        "schema_version": 1, "preview_id": "fixture-audition", "title": "Isolated audition fixture",
        "duration_ms": 500, "audio": descriptor("audio.wav", audio_bytes),
        "performance": descriptor("performance.json", performance_bytes), "semantic": None,
    }
    (root / "audio.wav").write_bytes(audio_bytes)
    (root / "performance.json").write_bytes(performance_bytes)
    (root / "manifest.json").write_bytes(encode(manifest))
    calls = []
    def upstream(request):
        calls.append(request)
        if request.url.path == "/api/status":
            return httpx.Response(200, json={"state": "waiting", "stream_ready": False})
        if request.url.path == "/api/now":
            return httpx.Response(200, json={"airing": None})
        pytest.fail("Unexpected preview proxy or private upstream request")
    app = (listener.create_app(runtime=runtime, transport=httpx.MockTransport(upstream))
           if request.param == "listener" else station.create_app(runtime, start_engine=False))
    return runtime, root, manifest, performance, app, calls


def semantic_document(manifest):
    return {
        "schema_version": 1, "preview_id": manifest["preview_id"],
        "audio_sha256": manifest["audio"]["sha256"], "duration_ms": 500,
        "model": "gemini-embedding-2", "space_id": "1" * 64,
        "projection": {"method": "pca", "dimensions": 3, "fitted_count": 2, "basis_sha256": "2" * 64},
        "windows": [
            {"id": "turn-0000", "character_id": "one", "start_ms": 0, "end_ms": 250,
             "position": [-0.4, 0.0, 0.3]},
            {"id": "turn-0001", "character_id": "two", "start_ms": 250, "end_ms": 500,
             "position": [0.4, 0.0, -0.3]},
        ],
    }


def test_metadata_and_fixed_assets_are_local_only_and_audio_supports_ranges(bundle):
    runtime, root, manifest, performance, app, calls = bundle
    with TestClient(app) as http:
        result = http.get("/api/previews/fixture-audition")
        assert result.status_code == 200
        data = result.json()
        assert data["kind"] == "local_preview" and data["local_preview"] is True
        assert data["audio"]["url"] == "/preview/fixture-audition/audio.wav"
        assert data["performance"]["url"] == "/preview/fixture-audition/performance.json"
        assert data["semantic"] is None
        assert str(runtime) not in result.text
        assert result.headers["cache-control"] == "no-store"
        assert result.headers["x-robots-tag"] == "noindex, nofollow"
        audio = http.get(data["audio"]["url"])
        assert audio.status_code == 200
        assert audio.headers["content-type"] == "audio/wav"
        assert hashlib.sha256(audio.content).hexdigest() == manifest["audio"]["sha256"]
        part = http.get(data["audio"]["url"], headers={"Range": "bytes=0-15"})
        assert part.status_code == 206 and len(part.content) == 16
        assert part.headers["content-range"] == "bytes 0-15/" + str(manifest["audio"]["size_bytes"])
        assert part.content == audio.content[:16]
        prepared = http.get(data["performance"]["url"])
        assert prepared.headers["content-type"] == "application/json"
        assert prepared.json() == performance
        assert "release_id" not in prepared.json() and "package_sha256" not in prepared.json()
    assert calls == []


def test_optional_semantic_projection_is_verified_and_served(bundle):
    _, root, manifest, _, app, calls = bundle
    semantic = semantic_document(manifest)
    manifest["semantic"] = descriptor("semantic.json", encode(semantic))
    write_asset(root, manifest, "semantic", semantic)
    with TestClient(app) as http:
        response = http.get("/api/previews/fixture-audition")
        assert response.status_code == 200
        assert response.json()["semantic"]["url"] == "/preview/fixture-audition/semantic.json"
        assert http.get(response.json()["semantic"]["url"]).json() == semantic
    assert calls == []


@pytest.mark.parametrize("path", [
    "/api/previews/missing",
    "/api/previews/Invalid",
    "/preview/fixture-audition/source.wav",
    "/preview/fixture-audition/manifest.json",
    "/preview/fixture-audition/provenance.json",
    "/preview/fixture-audition/semantic.json",
    "/preview/fixture-audition/the-original.mp3",
    "/preview/fixture-audition/%2e%2e%2fprivate.json",
])
def test_absent_unknown_and_private_preview_paths_remain_404(bundle, path):
    _, _, _, _, app, calls = bundle
    with TestClient(app) as http:
        result = http.get(path)
        assert result.status_code == 404
    assert calls == []


@pytest.mark.parametrize("path", [
    "/api/previews/fixture-audition?source=/private/input.wav",
    "/preview/fixture-audition/audio.wav?file=../private",
    "/preview/fixture-audition/performance.json?raw=true",
])
def test_queries_and_write_methods_are_not_supported(bundle, path):
    _, _, _, _, app, calls = bundle
    with TestClient(app) as http:
        assert http.get(path).status_code == 422
        assert http.post(path.split("?")[0], json={}).status_code == 405
    assert calls == []


@pytest.mark.parametrize("mutation", [
    lambda m: m.update(preview_id="another-preview"),
    lambda m: m.update(schema_version=True),
    lambda m: m.update(title=""),
    lambda m: m.update(evidence_path="/private/provenance.json"),
    lambda m: m["audio"].update(file="../input.wav"),
    lambda m: m["audio"].update(size_bytes=True),
    lambda m: m["audio"].update(sha256="0" * 64),
    lambda m: m["performance"].update(size_bytes=1),
])
def test_corrupt_manifest_and_source_binding_return_clean_409(bundle, mutation):
    runtime, root, manifest, _, app, _ = bundle
    mutation(manifest)
    (root / "manifest.json").write_bytes(encode(manifest))
    with TestClient(app) as http:
        response = http.get("/api/previews/fixture-audition")
        assert response.status_code == 409
        assert str(runtime) not in response.text and "private" not in response.text
        assert response.headers["x-robots-tag"] == "noindex, nofollow"


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(audio_sha256="0" * 64),
    lambda p: p.update(duration_ms=499),
    lambda p: p.update(preview_id="another-preview"),
    lambda p: p.update(private_source="/private/recording.wav"),
    lambda p: p["cues"][0].update(speaker_id="unknown"),
    lambda p: p["features"]["tracks"]["one"]["energy"].pop(),
    lambda p: p["features"]["tracks"]["one"]["low"].__setitem__(0, float("nan")),
])
def test_performance_binding_and_timeline_are_verified(bundle, mutation):
    _, root, manifest, prepared, app, _ = bundle
    mutation(prepared)
    write_asset(root, manifest, "performance", prepared)
    with TestClient(app) as http:
        assert http.get("/api/previews/fixture-audition").status_code == 409
        assert http.get("/preview/fixture-audition/performance.json").status_code == 409


@pytest.mark.parametrize("mutation", [
    lambda s: s.update(preview_id="another-preview"),
    lambda s: s.update(audio_sha256="0" * 64),
    lambda s: s.update(duration_ms=499),
    lambda s: s.update(raw_text="private text is not public semantic metadata"),
    lambda s: s["projection"].update(method="invented"),
    lambda s: s["projection"].update(dimensions=True),
    lambda s: s["projection"].update(fitted_count=1),
    lambda s: s["windows"][0].update(character_id="unknown"),
    lambda s: s["windows"][0]["position"].__setitem__(0, 1.01),
    lambda s: s["windows"][0]["position"].__setitem__(0, float("inf")),
    lambda s: s["windows"][0].update(end_ms=501),
    lambda s: s["windows"].reverse(),
])
def test_semantic_binding_known_characters_and_coordinates_are_verified(bundle, mutation):
    _, root, manifest, _, app, _ = bundle
    value = semantic_document(manifest)
    mutation(value)
    manifest["semantic"] = descriptor("semantic.json", encode(value))
    write_asset(root, manifest, "semantic", value)
    with TestClient(app) as http:
        assert http.get("/api/previews/fixture-audition").status_code == 409


def test_missing_or_symlinked_assets_are_not_exposed(bundle, tmp_path):
    _, root, _, _, app, _ = bundle
    asset = root / "audio.wav"
    private = tmp_path / "private.wav"
    asset.rename(private)
    with TestClient(app) as http:
        assert http.get("/preview/fixture-audition/audio.wav").status_code == 409
        asset.symlink_to(private)
        response = http.get("/preview/fixture-audition/audio.wav")
        assert response.status_code == 409
        assert str(private) not in response.text


def test_manifest_limit_and_incomplete_wav_are_rejected(bundle, monkeypatch):
    _, root, manifest, _, app, _ = bundle
    with monkeypatch.context() as bounded:
        bounded.setattr(preview, "MAX_MANIFEST_BYTES", 10)
        with TestClient(app) as http:
            assert http.get("/api/previews/fixture-audition").status_code == 409
    audio = (root / "audio.wav").read_bytes()
    write_asset(root, manifest, "audio", audio[:-100])
    # Rebind performance to isolate complete-audio validation from its checksum.
    prepared = json.loads((root / "performance.json").read_text())
    prepared["audio_sha256"] = manifest["audio"]["sha256"]
    write_asset(root, manifest, "performance", prepared)
    with TestClient(app) as http:
        assert http.get("/api/previews/fixture-audition").status_code == 409


def test_live_player_and_station_endpoints_keep_their_existing_behavior(bundle):
    _, _, _, _, app, _ = bundle
    with TestClient(app) as http:
        normal = http.get("/")
        audition = http.get("/?preview=fixture-audition")
        recording = http.get("/?recording=fixture-audition")
        assert normal.status_code == audition.status_code == recording.status_code == 200
        assert "x-robots-tag" not in normal.headers
        assert audition.headers["x-robots-tag"] == recording.headers["x-robots-tag"] == "noindex, nofollow"
        assert normal.content == audition.content == recording.content
        assert normal.headers["content-security-policy"] == recording.headers["content-security-policy"]
        assert "unsafe-inline" not in recording.headers["content-security-policy"]
        assert http.get("/api/status").status_code == 200
        assert http.get("/api/now").status_code == 200
        assert http.get("/api/generate").status_code == 404


def test_four_dimensional_projection_keeps_position_and_rotation_separate(bundle):
    _, root, manifest, _, app, _ = bundle
    value = semantic_document(manifest)
    value["projection"]["dimensions"] = 4
    value["windows"][0]["rotation"] = -0.4
    value["windows"][1]["rotation"] = 0.6
    manifest["semantic"] = descriptor("semantic.json", encode(value))
    write_asset(root, manifest, "semantic", value)
    with TestClient(app) as http:
        assert http.get("/api/previews/fixture-audition").status_code == 200
        response = http.get("/preview/fixture-audition/semantic.json")
        assert response.status_code == 200
        assert response.json() == value
        assert all(len(window["position"]) == 3 for window in response.json()["windows"])


@pytest.mark.parametrize("mutation", [
    lambda s: s["windows"][0].pop("rotation"),
    lambda s: s["windows"][1].update(rotation=None),
    lambda s: s["windows"][0].update(rotation=1.01),
    lambda s: s["windows"][0].update(rotation=-1.01),
    lambda s: s["windows"][0].update(rotation=float("nan")),
    lambda s: s["windows"][0].update(rotation=float("inf")),
    lambda s: s["windows"][0].update(rotation=True),
    lambda s: s["windows"][0]["position"].append(0.0),
    lambda s: s["projection"].update(dimensions=3),
    lambda s: s["projection"].update(dimensions=4.0),
    lambda s: s["projection"].update(dimensions=5),
])
def test_four_dimensional_rotation_contract_rejects_mixed_or_invalid_data(bundle, mutation):
    _, root, manifest, _, app, _ = bundle
    value = semantic_document(manifest)
    value["projection"]["dimensions"] = 4
    for window in value["windows"]:
        window["rotation"] = 0.0
    mutation(value)
    manifest["semantic"] = descriptor("semantic.json", encode(value))
    write_asset(root, manifest, "semantic", value)
    with TestClient(app) as http:
        assert http.get("/api/previews/fixture-audition").status_code == 409


def test_legacy_three_dimensional_projection_accepts_null_rotation(bundle):
    _, root, manifest, _, app, _ = bundle
    value = semantic_document(manifest)
    for window in value["windows"]:
        window["rotation"] = None
    manifest["semantic"] = descriptor("semantic.json", encode(value))
    write_asset(root, manifest, "semantic", value)
    with TestClient(app) as http:
        assert http.get("/api/previews/fixture-audition").status_code == 200
