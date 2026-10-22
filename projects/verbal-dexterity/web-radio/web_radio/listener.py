"""Listener-only entrance on the original web port; no production worker or API."""
from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import json
import os
from pathlib import Path
import re

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .contract import SLUG_PATTERN
from .performance import MAX_PERFORMANCE_BYTES
from .preview import register_preview_routes
from .embedding_contract import (
    BUNDLE_PATTERN, SHARD_PATTERN, MAX_CATALOG_BYTES,
    MAX_EMBEDDING_MANIFEST_BYTES, MAX_EMBEDDING_SHARD_BYTES,
)

PROJECT = Path(__file__).resolve().parents[1]
PLAYER = PROJECT / "web" / "player"
STATION = "http://127.0.0.1:43121"
SEGMENT = re.compile(r"[0-9a-f]{32}-[0-9]+\.ts")
JSON_LIMIT = 512 * 1024
PLAYLIST_LIMIT = 64 * 1024
SEGMENT_LIMIT = 2 * 1024 * 1024
UPSTREAM_TIMEOUT_SECONDS = 8
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    "style-src-attr 'none'; media-src 'self' blob:; connect-src 'self'; "
    "img-src 'self' data:; worker-src 'self' blob:; object-src 'none'; "
    "base-uri 'none'; frame-ancestors 'none'"
)


def create_app(*, transport: httpx.AsyncBaseTransport | None = None, runtime: Path | None = None) -> FastAPI:
    runtime = Path(runtime or os.environ.get("WEB_RADIO_DATA", PROJECT / ".runtime"))
    @asynccontextmanager
    async def lifespan(application):
        async with httpx.AsyncClient(
            transport=transport, trust_env=False, follow_redirects=False,
            timeout=httpx.Timeout(5, connect=1),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=8),
        ) as client:
            application.state.station_client = client
            yield

    application = FastAPI(
        title="Radio Waajacu", docs_url=None, redoc_url=None, openapi_url=None,
        lifespan=lifespan,
    )
    application.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"],
    )

    @application.middleware("http")
    async def response_headers(request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @application.get("/")
    def index(request: Request):
        headers = {"Cache-Control": "no-cache"}
        if "preview" in request.query_params:
            headers["X-Robots-Tag"] = "noindex, nofollow"
        return FileResponse(PLAYER / "index.html", headers=headers)

    @application.get("/assets/{asset_name}")
    def player_asset(asset_name: str):
        if asset_name not in {
            "player.css", "player.js", "spirits.js", "performance.js", "semantic.js",
            "botanical-frame-v1.png", "parchment-texture-v1.png",
            "waajacu-macaw.png",
        }:
            raise HTTPException(404, "Unknown player asset.")
        path = PLAYER / asset_name
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404, "Player asset is unavailable.")
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    @application.get("/vendor/hls.min.js")
    def hls_library():
        path = Path("/opt/web-radio/vendor/hls.min.js")
        if not path.is_file() or path.is_symlink():
            raise HTTPException(503, "Audio playback is temporarily unavailable.")
        return FileResponse(
            path, media_type="application/javascript",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @application.get("/healthz")
    def health():
        return {"ok": True, "service": "listener"}

    async def station_response(
        request: Request, path: str, *, media_type: str, limit: int,
        cache: str = "no-store", allow_at: bool = False,
    ):
        # The upstream and every permitted path are fixed by the routes below.
        # Do not forward browser headers, arbitrary query strings, or methods.
        allowed = {"at"} if allow_at else set()
        if any(name not in allowed for name in request.query_params):
            raise HTTPException(422, "Unsupported station query.")
        values = request.query_params.getlist("at")
        if len(values) > 1 or (values and len(values[0]) > 100):
            raise HTTPException(422, "Invalid playback timestamp.")
        params = {"at": values[0]} if values else None
        try:
            async with asyncio.timeout(UPSTREAM_TIMEOUT_SECONDS), request.app.state.station_client.stream(
                "GET", STATION + path, params=params,
                headers={"Accept-Encoding": "identity"},
            ) as upstream:
                if upstream.status_code != 200:
                    if upstream.status_code == 404:
                        raise HTTPException(404, "Broadcast data is unavailable.")
                    if path.startswith(("/api/performance/", "/api/embeddings/")) and upstream.status_code == 409:
                        raise HTTPException(409, "Prepared performance is unavailable.")
                    raise HTTPException(503, "The broadcast is temporarily unavailable.")
                if upstream.headers.get("content-encoding", "identity").lower() != "identity":
                    raise HTTPException(503, "The broadcast response is unavailable.")
                actual_type = upstream.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if actual_type != media_type:
                    raise HTTPException(503, "The broadcast response is unavailable.")
                declared = upstream.headers.get("content-length")
                if declared is not None and (not declared.isdecimal() or int(declared) > limit):
                    raise HTTPException(503, "The broadcast response exceeds its limit.")
                body = bytearray()
                async for chunk in upstream.aiter_bytes(chunk_size=64 * 1024):
                    if len(body) + len(chunk) > limit:
                        raise HTTPException(503, "The broadcast response exceeds its limit.")
                    body.extend(chunk)
        except (httpx.HTTPError, TimeoutError):
            raise HTTPException(503, "The broadcast is temporarily unavailable.") from None
        if media_type == "application/json":
            try:
                if not isinstance(json.loads(body), dict):
                    raise ValueError
            except (ValueError, UnicodeError):
                raise HTTPException(503, "The station information is temporarily unavailable.") from None
        return Response(bytes(body), media_type=media_type, headers={"Cache-Control": cache})

    @application.get("/api/status")
    async def status(request: Request):
        return await station_response(
            request, "/api/status", media_type="application/json", limit=JSON_LIMIT,
        )

    @application.get("/api/now")
    async def now(request: Request):
        return await station_response(
            request, "/api/now", media_type="application/json", limit=JSON_LIMIT, allow_at=True,
        )

    @application.get("/api/live-window")
    async def live_window(request: Request):
        return await station_response(
            request, "/api/live-window", media_type="application/json", limit=JSON_LIMIT,
        )

    @application.get("/api/performance/{release_id}")
    async def performance(request: Request, release_id: str):
        if re.fullmatch(SLUG_PATTERN, release_id) is None:
            raise HTTPException(404, "Prepared performance is unavailable.")
        return await station_response(
            request, "/api/performance/" + release_id,
            media_type="application/json", limit=MAX_PERFORMANCE_BYTES,
        )

    def embedding_identity(release_id: str, bundle_id: str | None = None,
                           shard_id: str | None = None):
        if (re.fullmatch(SLUG_PATTERN, release_id) is None
                or (bundle_id is not None and re.fullmatch(BUNDLE_PATTERN, bundle_id) is None)
                or (shard_id is not None and re.fullmatch(SHARD_PATTERN, shard_id) is None)):
            raise HTTPException(404, "Prepared embeddings are unavailable.")

    @application.get("/api/embeddings/{release_id}")
    async def embeddings(request: Request, release_id: str):
        embedding_identity(release_id)
        return await station_response(
            request, "/api/embeddings/" + release_id,
            media_type="application/json", limit=MAX_CATALOG_BYTES,
        )

    @application.get("/api/embeddings/{release_id}/{bundle_id}")
    async def embedding_manifest(request: Request, release_id: str, bundle_id: str):
        embedding_identity(release_id, bundle_id)
        return await station_response(
            request, "/api/embeddings/" + release_id + "/" + bundle_id,
            media_type="application/json", limit=MAX_EMBEDDING_MANIFEST_BYTES,
        )

    @application.get("/api/embeddings/{release_id}/{bundle_id}/{shard_id}")
    async def embedding_shard(request: Request, release_id: str, bundle_id: str, shard_id: str):
        embedding_identity(release_id, bundle_id, shard_id)
        return await station_response(
            request, "/api/embeddings/" + release_id + "/" + bundle_id + "/" + shard_id,
            media_type="application/octet-stream", limit=MAX_EMBEDDING_SHARD_BYTES,
        )

    @application.get("/hls/live.m3u8")
    async def playlist(request: Request):
        return await station_response(
            request, "/hls/live.m3u8", media_type="application/vnd.apple.mpegurl",
            limit=PLAYLIST_LIMIT,
        )

    @application.get("/hls/{segment_name}")
    async def segment(request: Request, segment_name: str):
        if len(segment_name) > 96 or not SEGMENT.fullmatch(segment_name):
            raise HTTPException(404, "Unknown broadcast segment.")
        return await station_response(
            request, "/hls/" + segment_name, media_type="video/mp2t",
            limit=SEGMENT_LIMIT, cache="public, max-age=31536000, immutable",
        )

    register_preview_routes(application, runtime)
    return application
