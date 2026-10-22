"""Public station API: serves only validated station-owned media, never studio data."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .contract import ContractError, SLUG_PATTERN, validate_package
from .performance import PerformanceError, read_performance
from .preview import register_preview_routes
from .embedding_contract import (
    BUNDLE_PATTERN, SHARD_PATTERN, EmbeddingError, bundle_catalog, installed_bundle,
)
from .playout import Playout, iso, parse_time

PROJECT = Path(__file__).resolve().parents[1]
PLAYER = PROJECT / "web" / "player"
SEGMENT = re.compile(r"^[0-9a-f]{32}-[0-9]+\.ts$")


def create_app(runtime: Path | None = None, *, start_engine=True) -> FastAPI:
    runtime = Path(runtime or os.environ.get("WEB_RADIO_DATA", PROJECT / ".runtime"))
    engine = Playout(runtime)

    @asynccontextmanager
    async def lifespan(_app):
        if start_engine:
            engine.start()
        try:
            yield
        finally:
            engine.stop()

    app = FastAPI(title="Radio Waajacu — station", docs_url=None, redoc_url=None,
                  openapi_url=None, lifespan=lifespan)
    app.state.playout = engine

    @app.middleware("http")
    async def response_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "media-src 'self' blob:; connect-src 'self'; img-src 'self' data:; "
            "worker-src 'self' blob:; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'")
        return response

    @app.get("/healthz")
    def health():
        return {"ok": True, "service": "station", "state": engine.state}

    @app.get("/")
    def index(request: Request):
        headers = {"Cache-Control": "no-cache"}
        if any(key in request.query_params for key in ("recording", "preview")):
            headers["X-Robots-Tag"] = "noindex, nofollow"
        return FileResponse(PLAYER / "index.html", headers=headers)

    @app.get("/vendor/hls.min.js")
    def hls_library():
        vendor = Path("/opt/web-radio/vendor/hls.min.js")
        if not vendor.is_file():
            raise HTTPException(503, "HLS playback library has not been installed.")
        return FileResponse(vendor, media_type="application/javascript",
                            headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/status")
    def status():
        return engine.snapshot()

    @app.get("/api/now")
    def now(at: str | None = None):
        try:
            position = parse_time(at) if at else None
        except (ValueError, OverflowError):
            raise HTTPException(422, "at must be an ISO timestamp with timezone.")
        return {"airing": engine.journal.now(position),
                "timing": "stream" if position else "station-clock",
                "requested_at": iso(position) if position else None}

    @app.get("/api/journal")
    def journal(limit: int = Query(default=50, ge=1, le=200)):
        return {"events": engine.journal.events(limit)}

    def verified(release_id: str):
        pair = engine.known.get(release_id)
        if not pair:
            raise HTTPException(404, "Release has not been accepted by this station.")
        manifest, package = pair
        try:
            validate_package(package, expected_digest=manifest.package_sha256)
        except (OSError, ValueError, ContractError):
            raise HTTPException(409, "Station media failed verification.")
        return manifest, package

    def public_release(release_id: str):
        manifest, _ = verified(release_id)
        audio = next(a for a in manifest.assets if a.role == "audio")
        transcript = next(a for a in manifest.assets if a.role == "transcript")
        return {
            "release_id": manifest.release_id, "programme_id": manifest.programme_id,
            "title": manifest.title, "language": manifest.language,
            "duration_ms": manifest.duration_ms, "kind": manifest.kind,
            "audio_url": f"/media/{manifest.release_id}/{audio.path}",
            "transcript_url": f"/media/{manifest.release_id}/{transcript.path}",
        }

    @app.get("/api/replay")
    def replay():
        airing = engine.journal.last_completed()
        if not airing:
            return {"airing": None, "release": None}
        return {"airing": airing, "release": public_release(airing["release_id"])}

    @app.get("/api/releases/{release_id}")
    def release(release_id: str):
        return public_release(release_id)

    @app.get("/api/performance/{release_id}")
    def performance(release_id: str, request: Request):
        if request.query_params:
            raise HTTPException(422, "Unsupported performance query.")
        if re.fullmatch(SLUG_PATTERN, release_id) is None:
            raise HTTPException(404, "Prepared performance is unavailable.")
        manifest, _ = verified(release_id)
        try:
            prepared = read_performance(engine.root / "performance" / (release_id + ".json"), manifest)
        except FileNotFoundError:
            raise HTTPException(404, "Prepared performance is unavailable.")
        except (PerformanceError, OSError, ValueError):
            raise HTTPException(409, "Prepared performance does not match the accepted release.")
        return JSONResponse(prepared.model_dump(mode="json", exclude_none=True),
                            headers={"Cache-Control": "no-store"})

    def embedding_request(request: Request, release_id: str, bundle_id: str | None = None,
                          shard_id: str | None = None):
        if request.query_params:
            raise HTTPException(422, "Unsupported embedding query.")
        if (re.fullmatch(SLUG_PATTERN, release_id) is None
                or (bundle_id is not None and re.fullmatch(BUNDLE_PATTERN, bundle_id) is None)
                or (shard_id is not None and re.fullmatch(SHARD_PATTERN, shard_id) is None)):
            raise HTTPException(404, "Prepared embeddings are unavailable.")
        verified(release_id)

    def embedding_error(exc):
        if isinstance(exc, FileNotFoundError):
            raise HTTPException(404, "Prepared embeddings are unavailable.")
        raise HTTPException(409, "Prepared embeddings do not match the accepted release.")

    @app.get("/api/embeddings/{release_id}")
    def embeddings(release_id: str, request: Request):
        embedding_request(request, release_id)
        try:
            data = bundle_catalog(runtime, release_id)
        except (EmbeddingError, OSError, ValueError) as exc:
            embedding_error(exc)
        return JSONResponse(data, headers={"Cache-Control": "no-store"})

    @app.get("/api/embeddings/{release_id}/{bundle_id}")
    def embedding_manifest(release_id: str, bundle_id: str, request: Request):
        embedding_request(request, release_id, bundle_id)
        try:
            document, _ = installed_bundle(runtime, release_id, bundle_id)
        except (EmbeddingError, OSError, ValueError) as exc:
            embedding_error(exc)
        return JSONResponse(document.model_dump(mode="json"), headers={"Cache-Control": "no-store"})

    @app.get("/api/embeddings/{release_id}/{bundle_id}/{shard_id}")
    def embedding_shard(release_id: str, bundle_id: str, shard_id: str, request: Request):
        embedding_request(request, release_id, bundle_id, shard_id)
        try:
            _, shards = installed_bundle(runtime, release_id, bundle_id)
            data = shards.get(shard_id)
            if data is None:
                raise FileNotFoundError("undeclared embedding shard")
        except (EmbeddingError, OSError, ValueError) as exc:
            embedding_error(exc)
        return Response(data, media_type="application/octet-stream",
                        headers={"Cache-Control": "no-store"})

    @app.get("/media/{release_id}/{asset_path:path}")
    def media(release_id: str, asset_path: str):
        manifest, package = verified(release_id)
        asset = next((a for a in manifest.assets if a.path == asset_path), None)
        if not asset:
            raise HTTPException(404, "Asset is not declared in the release.")
        return FileResponse(package / asset.path, media_type=asset.mime_type,
                            headers={"Cache-Control": "public, max-age=31536000, immutable",
                                     "ETag": f'"{asset.sha256}"'})

    @app.get("/hls/live.m3u8")
    def playlist():
        path = engine.hls / "live.m3u8"
        if not path.is_file() or engine.state not in ("broadcasting", "fallback"):
            raise HTTPException(503, "The station is waiting for a verified programme or ident.")
        return FileResponse(path, media_type="application/vnd.apple.mpegurl",
                            headers={"Cache-Control": "no-store"})

    @app.get("/hls/{segment_name}")
    def segment(segment_name: str):
        if not SEGMENT.fullmatch(segment_name):
            raise HTTPException(404, "Unknown stream segment.")
        path = engine.hls / segment_name
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404, "Stream segment is unavailable.")
        return FileResponse(path, media_type="video/mp2t",
                            headers={"Cache-Control": "public, max-age=31536000, immutable"})

    @app.get("/api/live-window")
    def live_window():
        # A native player without playingDate may show an explicitly labelled estimate.
        try:
            text = (engine.hls / "live.m3u8").read_text()
            date = re.search(r"^#EXT-X-PROGRAM-DATE-TIME:(.+)$", text, re.MULTILINE)
            durations = [float(v) for v in re.findall(r"^#EXTINF:([0-9.]+)", text, re.MULTILINE)]
            if not date or not durations:
                raise ValueError
            start = parse_time(date.group(1).strip())
            return {"start": iso(start),
                    "end": iso(start + timedelta(seconds=sum(durations))),
                    "duration": sum(durations)}
        except (OSError, ValueError):
            raise HTTPException(503, "The live timeline is warming up.")

    register_preview_routes(app, runtime)
    app.mount("/assets", StaticFiles(directory=PLAYER), name="player-assets")
    return app


app = create_app()
