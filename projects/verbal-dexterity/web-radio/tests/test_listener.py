"""Listener facade preserves radio access and never exposes backend production."""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from web_radio import listener

SEGMENT = "a" * 32 + "-7.ts"
PRIVATE = [
    "/api/credentials", "/api/settings", "/api/script", "/api/scripts/programme/head",
    "/api/generate", "/api/takes", "/api/import", "/api/programming",
    "/assets/studio.js", "/assets/studio.css", "/assets/station-controls.js",
    "/.runtime/secrets/gemini-api-key", "/docs", "/redoc", "/openapi.json",
]


class Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def client(handler):
    return TestClient(listener.create_app(transport=httpx.MockTransport(handler)))


def test_shared_player_and_security_policy_need_no_backend_worker(monkeypatch):
    from web_radio import studio
    monkeypatch.setattr(studio, "Studio", lambda *_: pytest.fail("Listener started production"))
    with TestClient(studio.app) as http:
        response = http.get("/")
        assert response.status_code == 200
        assert "<html" in response.text.lower()
        assert response.headers["cache-control"] == "no-cache"
        policy = response.headers["content-security-policy"]
        assert "script-src 'self'" in policy and "style-src 'self'" in policy
        assert "media-src 'self' blob:" in policy
        assert "unsafe-inline" not in policy
        assert http.get("/healthz").json()["service"] == "listener"
        assert not hasattr(http.app.state, "studio")
        assert http.get("/assets/player.js").status_code == 200
        assert http.get("/assets/player.css").status_code == 200
        assert http.get("/vendor/hls.min.js").status_code == 200


@pytest.mark.parametrize("asset_name", [
    "botanical-frame-v1.png", "parchment-texture-v1.png",
    "waajacu-macaw.png",
])
def test_player_image_assets_are_pngs_without_reaching_station(asset_name):
    calls = []
    def forbidden(request):
        calls.append(request)
        pytest.fail("Artwork request reached station proxy")
    with client(forbidden) as http:
        response = http.get("/assets/" + asset_name)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG\r\n\x1a\n")
        assert response.headers["cache-control"] == "no-cache"
    assert calls == []


@pytest.mark.parametrize("asset_name", [
    "unknown.png", "botanical-source-v1.png", "parchment-source-v1.png",
    "artwork-provenance-v1.json", "prompts.json", "index.html",
])
def test_artwork_allowlist_does_not_expose_source_or_provenance(asset_name):
    calls = []
    def forbidden(request):
        calls.append(request)
        pytest.fail("Unknown asset reached station proxy")
    with client(forbidden) as http:
        assert http.get("/assets/" + asset_name).status_code == 404
    assert calls == []


@pytest.mark.parametrize("method", ["GET", "POST", "PUT"])
@pytest.mark.parametrize("path", PRIVATE)
def test_production_paths_and_old_assets_are_not_mounted(path, method):
    calls = []
    def forbidden(request):
        calls.append(request)
        pytest.fail("Private request reached station proxy")
    with client(forbidden) as http:
        response = http.request(method, path, json={"key": "fixture-never-a-real-key"})
        assert response.status_code in {404, 405}
    assert calls == []


@pytest.mark.parametrize("path", ["/api/status", "/api/now", "/api/live-window"])
def test_station_json_uses_fixed_upstream_without_browser_credentials(path):
    calls = []
    def response(request):
        calls.append(request)
        assert request.url.host == "127.0.0.1"
        assert request.url.port == 43121
        assert request.method == "GET" and request.url.path == path
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        assert "x-goog-api-key" not in request.headers
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(200, json={"state": "broadcasting"}, headers={"set-cookie": "do-not-forward=1"})
    with client(response) as http:
        result = http.get(path, headers={"Authorization": "fixture-secret", "Cookie": "private=1", "X-Goog-Api-Key": "fixture-key"})
        assert result.json() == {"state": "broadcasting"}
        assert result.headers["cache-control"] == "no-store"
        assert "set-cookie" not in result.headers
    assert len(calls) == 1


def test_playback_timestamp_is_forwarded_as_one_encoded_query():
    stamp = "2026-10-22T17:22:16.000Z"
    calls = []
    def response(request):
        calls.append(request)
        assert list(request.url.params.multi_items()) == [("at", stamp)]
        return httpx.Response(200, json={"airing": None})
    with client(response) as http:
        assert http.get("/api/now", params={"at": stamp}).status_code == 200
    assert len(calls) == 1


@pytest.mark.parametrize("path", [
    "/api/status?url=https://example.invalid",
    "/api/now?at=x&at=y",
    "/api/now?at=" + "x" * 101,
    "/api/live-window?upstream=elsewhere",
    "/hls/" + SEGMENT + "?path=../../secrets",
    "/hls/http:%2f%2fexample.invalid.ts",
    "/hls/%2e%2e%2fapi%2fcredentials",
    "/hls/" + "a" * 32 + "-7.ts%0a",
    "/api/replay", "/api/journal", "/api/releases/secret", "/media/secret/audio.wav",
])
def test_extra_queries_paths_and_nonradio_routes_do_not_forward(path):
    calls = []
    def forbidden(request):
        calls.append(request)
        pytest.fail("Unsupported request reached upstream")
    with client(forbidden) as http:
        assert http.get(path).status_code in {404, 422}
    assert calls == []


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_public_read_routes_reject_mutations_without_forwarding(method):
    calls = []
    def forbidden(request):
        calls.append(request)
        pytest.fail("Mutation reached upstream")
    with client(forbidden) as http:
        assert http.request(method, "/api/status", json={}).status_code == 405
    assert calls == []


def test_playlist_and_uuid_segment_keep_audio_bytes_and_cache_semantics():
    playlist = b"#EXTM3U\n#EXT-X-VERSION:3\n#EXTINF:2.0,\n" + SEGMENT.encode() + b"\n"
    segment = b"\x47" + b"bounded-fixture-packet" * 20
    calls = []
    def response(request):
        calls.append(request)
        if request.url.path == "/hls/live.m3u8":
            return httpx.Response(200, content=playlist, headers={"Content-Type": "application/vnd.apple.mpegurl"})
        assert request.url.path == "/hls/" + SEGMENT
        return httpx.Response(200, content=segment, headers={"Content-Type": "video/mp2t"})
    with client(response) as http:
        result = http.get("/hls/live.m3u8")
        assert result.content == playlist and result.headers["cache-control"] == "no-store"
        result = http.get("/hls/" + SEGMENT)
        assert result.content == segment
        assert result.headers["content-type"] == "video/mp2t"
        assert result.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert len(calls) == 2


@pytest.mark.parametrize("failure", [
    "connection", "timeout", "redirect", "unavailable", "wrong-mime",
    "bad-json", "declared-oversize", "streamed-oversize", "compressed",
])
def test_upstream_failures_are_bounded_and_become_clean_unavailable(failure, monkeypatch):
    monkeypatch.setattr(listener, "JSON_LIMIT", 32)
    calls = []
    def response(request):
        calls.append(request)
        if failure == "connection":
            raise httpx.ConnectError("private-upstream-debug", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("private-upstream-debug", request=request)
        if failure == "redirect":
            return httpx.Response(302, headers={"Location": "https://example.invalid/secrets"})
        if failure == "unavailable":
            return httpx.Response(500, text="private-upstream-debug")
        if failure == "wrong-mime":
            return httpx.Response(200, text="<html>private-upstream-debug</html>")
        if failure == "bad-json":
            return httpx.Response(200, content=b"{oops", headers={"Content-Type": "application/json"})
        if failure == "declared-oversize":
            return httpx.Response(200, stream=Chunks([b"{}"]), headers={"Content-Type": "application/json", "Content-Length": "33"})
        if failure == "streamed-oversize":
            return httpx.Response(200, stream=Chunks([b"x" * 20, b"x" * 20]), headers={"Content-Type": "application/json"})
        return httpx.Response(200, stream=Chunks([b"not-gzip"]), headers={"Content-Type": "application/json", "Content-Encoding": "gzip"})
    with client(response) as http:
        result = http.get("/api/status")
        assert result.status_code == 503
        assert "private-upstream-debug" not in result.text
        assert "example.invalid" not in result.text
        assert "location" not in result.headers
    assert len(calls) == 1


def test_missing_valid_segment_stays_404_without_upstream_debug():
    with client(lambda request: httpx.Response(404, text="/private/internal/path")) as http:
        result = http.get("/hls/" + SEGMENT)
        assert result.status_code == 404
        assert "/private/internal/path" not in result.text


def test_overall_request_deadline_bounds_a_slow_upstream(monkeypatch):
    import asyncio
    monkeypatch.setattr(listener, "UPSTREAM_TIMEOUT_SECONDS", 0.01)
    async def slow(request):
        await asyncio.sleep(30)
        return httpx.Response(200, json={"state": "waiting"})
    with client(slow) as http:
        result = http.get("/api/status")
        assert result.status_code == 503
        assert result.headers["cache-control"] == "no-store"
