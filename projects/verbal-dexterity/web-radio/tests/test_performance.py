"""Prepared visual performance tests use isolated labelled fixture releases only."""
import hashlib
import json
from pathlib import Path
import wave

from fastapi.testclient import TestClient
import httpx
import pytest

from web_radio import listener, performance, station
from web_radio.contract import create_release


@pytest.fixture
def accepted(tmp_path):
    runtime = tmp_path / "runtime"
    source = tmp_path / "fixture.wav"
    with wave.open(str(source), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\0\0" * 6000)
    package = runtime / "library/releases/fixture-performance"
    manifest = create_release(source, "ISOLATED SILENCE FIXTURE, not a real programme.",
                              package, "fixture", "Isolated performance test",
                              release_id="fixture-performance")
    (runtime / "library/catalog.json").write_text(json.dumps({
        "schema_version": 1,
        "releases": [{"release_id": manifest.release_id, "package_sha256": manifest.package_sha256}],
    }))
    app = station.create_app(runtime, start_engine=False)
    app.state.playout.refresh()
    assert manifest.release_id in app.state.playout.known
    return runtime, manifest, app


def document(manifest, *, features=True):
    data = {
        "schema_version": 1, "release_id": manifest.release_id,
        "package_sha256": manifest.package_sha256, "duration_ms": manifest.duration_ms,
        "cast": [{"id": "one", "color": "#668877"}, {"id": "two", "color": "#996677"}],
        "cues": [{"speaker_id": "one", "start_ms": 0, "end_ms": 150},
                 {"speaker_id": "two", "start_ms": 100, "end_ms": 250}],
    }
    if features:
        data["features"] = {"hop_ms": 50, "tracks": {
            character: {band: [0.0, 0.2, 0.4, 0.1, 0.0, 0.0]
                        for band in ("energy", "low", "mid", "high")}
            for character in ("one", "two")
        }}
    return data


def encoded(data):
    return json.dumps(data).encode()


def install_source(tmp_path, runtime, manifest, *, features=True):
    source = tmp_path / "authored.json"
    source.write_bytes(encoded(document(manifest, features=features)))
    target = performance.install_performance(source, runtime)
    return source, target


def test_overlapping_different_speakers_and_optional_tracks_are_valid(accepted):
    _, manifest, _ = accepted
    value = document(manifest)
    prepared = performance.validate_document(encoded(value), manifest)
    assert [character.id for character in prepared.cast] == ["one", "two"]
    assert prepared.cues[0].end_ms > prepared.cues[1].start_ms
    del value["features"]["tracks"]["two"]
    assert set(performance.validate_document(encoded(value), manifest).features.tracks) == {"one"}
    assert performance.validate_document(encoded(document(manifest, features=False)), manifest).features is None


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(schema_version=True),
    lambda d: d.update(schema_version=2),
    lambda d: d.update(duration_ms=True),
    lambda d: d.update(extra="not permitted"),
    lambda d: d.update(cast=[]),
    lambda d: d.update(cast=d["cast"] * 7),
    lambda d: d["cast"][1].update(id="one"),
    lambda d: d["cast"][0].update(id="../one"),
    lambda d: d["cast"][0].update(color="red"),
    lambda d: d["cues"][0].update(speaker_id="unknown"),
    lambda d: d["cues"][0].update(start_ms=-1),
    lambda d: d["cues"][0].update(start_ms=False),
    lambda d: d["cues"][0].update(end_ms=0),
    lambda d: d["cues"][0].update(end_ms=251),
    lambda d: d["cues"].reverse(),
    lambda d: d["cues"].append({"speaker_id": "two", "start_ms": 150, "end_ms": 200}),
    lambda d: d["features"].update(hop_ms=19),
    lambda d: d["features"].update(hop_ms=1001),
    lambda d: d["features"].update(hop_ms=True),
    lambda d: d["features"]["tracks"].update(unknown=d["features"]["tracks"]["one"]),
    lambda d: d["features"]["tracks"]["one"]["energy"].pop(),
    lambda d: d["features"]["tracks"]["one"]["high"].append(0.5),
    lambda d: d["features"]["tracks"]["one"]["low"].__setitem__(0, 1.01),
    lambda d: d["features"]["tracks"]["one"]["mid"].__setitem__(0, -0.01),
    lambda d: d["features"]["tracks"]["one"]["energy"].__setitem__(0, True),
    lambda d: d["features"]["tracks"]["one"]["energy"].__setitem__(0, float("nan")),
    lambda d: d["features"]["tracks"]["one"]["energy"].__setitem__(0, float("inf")),
])
def test_invalid_schema_and_timeline_are_rejected(accepted, mutation):
    _, manifest, _ = accepted
    value = document(manifest)
    mutation(value)
    with pytest.raises(performance.PerformanceError):
        performance.validate_document(encoded(value), manifest)


@pytest.mark.parametrize("field,value", [
    ("release_id", "other-release"),
    ("package_sha256", "0" * 64),
    ("duration_ms", 200),
])
def test_stale_release_binding_is_rejected(accepted, field, value):
    _, manifest, _ = accepted
    data = document(manifest, features=False)
    data[field] = value
    with pytest.raises(performance.PerformanceError):
        performance.validate_document(encoded(data), manifest)


def test_byte_feature_and_duplicate_key_limits(accepted, monkeypatch):
    _, manifest, _ = accepted
    data = encoded(document(manifest))
    with monkeypatch.context() as limits:
        limits.setattr(performance, "MAX_PERFORMANCE_BYTES", 16)
        with pytest.raises(performance.PerformanceError):
            performance.validate_document(data, manifest)
    with monkeypatch.context() as limits:
        limits.setattr(performance, "MAX_FEATURE_VALUES", 3)
        with pytest.raises(performance.PerformanceError):
            performance.validate_document(data, manifest)
    with pytest.raises(performance.PerformanceError):
        performance.validate_document(b'{"schema_version":1,' + data[1:], manifest)
    with pytest.raises(performance.PerformanceError):
        performance.validate_document(b'[]', manifest)


def test_normalized_response_limit_includes_integer_to_float_expansion(accepted, monkeypatch):
    _, manifest, _ = accepted
    value = document(manifest)
    for track in value["features"]["tracks"].values():
        for band in ("energy", "low", "mid", "high"):
            track[band] = [0] * 6
    raw = json.dumps(value, separators=(",", ":")).encode()
    monkeypatch.setattr(performance, "MAX_PERFORMANCE_BYTES", len(raw) + 1)
    with pytest.raises(performance.PerformanceError, match="normalized"):
        performance.validate_document(raw, manifest)


def test_atomic_install_preserves_release_and_requires_explicit_replace(accepted, tmp_path):
    runtime, manifest, app = accepted
    package = app.state.playout.known[manifest.release_id][1]
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in package.iterdir() if path.is_file()}
    source, target = install_source(tmp_path, runtime, manifest)
    assert target == runtime / "station/performance/fixture-performance.json"
    first = target.read_bytes()
    changed = document(manifest)
    changed["cast"][0]["color"] = "#123456"
    source.write_bytes(encoded(changed))
    with pytest.raises(performance.PerformanceError):
        performance.install_performance(source, runtime)
    assert target.read_bytes() == first
    performance.install_performance(source, runtime, replace=True)
    assert performance.read_performance(target, manifest).cast[0].color == "#123456"
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in package.iterdir() if path.is_file()} == before
    assert not list(target.parent.glob("*.tmp"))


def test_install_requires_previously_accepted_release(accepted, tmp_path):
    _, manifest, _ = accepted
    source = tmp_path / "authored.json"
    source.write_bytes(encoded(document(manifest)))
    unaccepted_runtime = tmp_path / "not-accepted"
    with pytest.raises(performance.PerformanceError):
        performance.install_performance(source, unaccepted_runtime)
    assert not unaccepted_runtime.exists()


def test_symbolic_sidecar_is_rejected(accepted, tmp_path):
    runtime, manifest, _ = accepted
    source, target = install_source(tmp_path, runtime, manifest)
    target.unlink()
    target.symlink_to(source)
    with pytest.raises(performance.PerformanceError):
        performance.read_performance(target, manifest)


def test_station_performance_is_optional_bounded_and_read_only(accepted, tmp_path):
    runtime, manifest, app = accepted
    route = "/api/performance/" + manifest.release_id
    with TestClient(app) as http:
        assert http.get(route).status_code == 404
        assert http.get("/api/performance/unknown").status_code == 404
        assert http.get(route + "?source=private").status_code == 422
        assert http.get("/api/performance/Invalid").status_code == 404
        assert http.post(route, json={}).status_code == 405
        _, target = install_source(tmp_path, runtime, manifest)
        response = http.get(route)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json()["cast"][1]["id"] == "two"
        stale = document(manifest)
        stale["package_sha256"] = "0" * 64
        target.write_bytes(encoded(stale))
        response = http.get(route)
        assert response.status_code == 409
        assert str(runtime) not in response.text
        target.write_text("{malformed")
        assert http.get(route).status_code == 409


def test_listener_performance_uses_fixed_bounded_upstream():
    calls = []
    def upstream(request):
        calls.append(request)
        assert str(request.url) == "http://127.0.0.1:43121/api/performance/fixture-performance"
        assert request.method == "GET"
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(200, json={"schema_version": 1, "cast": []},
                              headers={"set-cookie": "private"})
    with TestClient(listener.create_app(transport=httpx.MockTransport(upstream))) as http:
        response = http.get("/api/performance/fixture-performance",
                            headers={"Authorization": "private", "Cookie": "private"})
        assert response.status_code == 200
        assert "set-cookie" not in response.headers
        assert response.headers["cache-control"] == "no-store"
        assert http.get("/api/performance/Invalid").status_code == 404
        assert http.get("/api/performance/fixture-performance?url=https://example.invalid/").status_code == 422
        assert http.post("/api/performance/fixture-performance").status_code == 405
    assert len(calls) == 1


@pytest.mark.parametrize("status,expected", [(404, 404), (409, 409), (500, 503)])
def test_listener_performance_missing_and_stale_statuses_are_safe(status, expected):
    transport = httpx.MockTransport(lambda request: httpx.Response(status, text="/private/runtime/path"))
    with TestClient(listener.create_app(transport=transport)) as http:
        result = http.get("/api/performance/fixture-performance")
        assert result.status_code == expected
        assert "/private/runtime/path" not in result.text


def test_listener_rejects_oversized_performance_before_reading(monkeypatch):
    monkeypatch.setattr(listener, "MAX_PERFORMANCE_BYTES", 16)
    transport = httpx.MockTransport(lambda request: httpx.Response(
        200, content=b'{"payload":"far too long"}', headers={"Content-Type": "application/json"}))
    with TestClient(listener.create_app(transport=transport)) as http:
        assert http.get("/api/performance/fixture-performance").status_code == 503
