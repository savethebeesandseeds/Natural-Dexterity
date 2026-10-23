"""Read-only local audition previews, separate from public release and playout data."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Annotated, Literal
import wave

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contract import HASH_PATTERN, MAX_AUDIO_BYTES, MAX_DURATION_MS, SLUG_PATTERN
from .performance import MAX_PERFORMANCE_BYTES, Performance

MAX_MANIFEST_BYTES = 64 * 1024
MAX_SEMANTIC_BYTES = 1024 * 1024
PREVIEW_FILES = {"audio.wav", "performance.json", "semantic.json"}
PREVIEW_HEADERS = {"Cache-Control": "no-store", "X-Robots-Tag": "noindex, nofollow"}
Hash = Annotated[str, Field(pattern=HASH_PATTERN)]
Slug = Annotated[str, Field(pattern=SLUG_PATTERN)]


class PreviewError(ValueError):
    """A local preview bundle cannot be verified."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class AudioAsset(StrictModel):
    file: Literal["audio.wav"]
    sha256: Hash
    size_bytes: Annotated[int, Field(gt=0, le=MAX_AUDIO_BYTES)]


class PerformanceAsset(StrictModel):
    file: Literal["performance.json"]
    sha256: Hash
    size_bytes: Annotated[int, Field(gt=0, le=MAX_PERFORMANCE_BYTES)]


class SemanticAsset(StrictModel):
    file: Literal["semantic.json"]
    sha256: Hash
    size_bytes: Annotated[int, Field(gt=0, le=MAX_SEMANTIC_BYTES)]


class PreviewManifest(StrictModel):
    schema_version: Literal[1]
    preview_id: Slug
    title: Annotated[str, Field(min_length=1, max_length=300)]
    duration_ms: Annotated[int, Field(gt=0, le=MAX_DURATION_MS)]
    audio: AudioAsset
    performance: PerformanceAsset
    semantic: SemanticAsset | None

    @model_validator(mode="before")
    @classmethod
    def exact_version(cls, value):
        if isinstance(value, dict) and type(value.get("schema_version")) is not int:
            raise ValueError("preview version must be integer one")
        return value

    @model_validator(mode="after")
    def ordinary_title(self):
        if not self.title.strip() or "\0" in self.title:
            raise ValueError("preview title must be ordinary nonblank text")
        return self


class Projection(StrictModel):
    method: Literal["pca"]
    dimensions: Literal[3, 4, 5]
    fitted_count: Annotated[int, Field(ge=1, le=2000)]
    basis_sha256: Hash

    @model_validator(mode="before")
    @classmethod
    def exact_dimensions(cls, value):
        if isinstance(value, dict) and type(value.get("dimensions")) is not int:
            raise ValueError("projection dimensions must be integer three, four or five")
        return value


Coordinate = Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]


class SemanticWindow(StrictModel):
    id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,159}$")]
    character_id: Slug
    start_ms: Annotated[int, Field(ge=0, le=MAX_DURATION_MS)]
    end_ms: Annotated[int, Field(gt=0, le=MAX_DURATION_MS)]
    position: Annotated[list[Coordinate], Field(min_length=3, max_length=3)]
    rotation: Coordinate | None = None
    depth_rotation: Coordinate | None = None


class PreviewSemantic(StrictModel):
    schema_version: Literal[1]
    preview_id: Slug
    audio_sha256: Hash
    duration_ms: Annotated[int, Field(gt=0, le=MAX_DURATION_MS)]
    model: Literal["gemini-embedding-2"]
    space_id: Hash
    projection: Projection
    windows: Annotated[list[SemanticWindow], Field(min_length=1, max_length=2000)]

    @model_validator(mode="before")
    @classmethod
    def exact_version(cls, value):
        if isinstance(value, dict) and type(value.get("schema_version")) is not int:
            raise ValueError("semantic version must be integer one")
        return value


def _ordinary_parents(path: Path):
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise PreviewError("preview paths must be ordinary")


def _open(path: Path):
    _ordinary_parents(path.parent)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    source = os.fdopen(descriptor, "rb")
    if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
        source.close()
        raise PreviewError("preview files must be ordinary")
    return source


def _read(path: Path, limit: int) -> bytes:
    with _open(path) as source:
        size = os.fstat(source.fileno()).st_size
        if not 0 < size <= limit:
            raise PreviewError("preview data exceeds its limit")
        data = source.read(limit + 1)
    if len(data) > limit:
        raise PreviewError("preview data exceeds its limit")
    return data


def _unique_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise PreviewError("duplicate JSON key")
        value[key] = item
    return value


def _nonfinite(_value):
    raise PreviewError("nonfinite JSON number")


def _json(data: bytes):
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_keys,
                           parse_constant=_nonfinite)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise PreviewError("invalid preview JSON") from exc
    if not isinstance(value, dict):
        raise PreviewError("preview JSON must be an object")
    return value


def _verify_asset(path: Path, asset) -> None:
    digest = hashlib.sha256()
    with _open(path) as source:
        if os.fstat(source.fileno()).st_size != asset.size_bytes:
            raise PreviewError("preview asset size mismatch")
        total = 0
        while chunk := source.read(1024 * 1024):
            total += len(chunk)
            if total > asset.size_bytes:
                raise PreviewError("preview asset exceeds its declared size")
            digest.update(chunk)
        if total != asset.size_bytes or digest.hexdigest() != asset.sha256:
            raise PreviewError("preview asset checksum mismatch")


def _verify_audio(path: Path, duration_ms: int):
    try:
        with _open(path) as source:
            with wave.open(source, "rb") as audio:
                if (audio.getcomptype() != "NONE" or audio.getnchannels() not in (1, 2)
                        or audio.getsampwidth() not in (2, 3, 4)
                        or not 8000 <= audio.getframerate() <= 192000):
                    raise PreviewError("preview audio must be supported PCM WAV")
                measured = audio.getnframes() * 1000 / audio.getframerate()
                if abs(measured - duration_ms) > 1:
                    raise PreviewError("preview duration differs from its audio")
                remaining = audio.getnframes()
                frame_bytes = audio.getnchannels() * audio.getsampwidth()
                while remaining:
                    count = min(65536, remaining)
                    if len(audio.readframes(count)) != count * frame_bytes:
                        raise PreviewError("preview audio frames are incomplete")
                    remaining -= count
    except (wave.Error, EOFError) as exc:
        raise PreviewError("preview audio must be ordinary PCM WAV") from exc


def _performance(raw: dict, manifest: PreviewManifest) -> Performance:
    required = {"schema_version", "preview_id", "audio_sha256", "duration_ms", "cast", "cues"}
    if not required <= set(raw) or set(raw) - required - {"features"}:
        raise PreviewError("preview performance fields are invalid")
    if (raw["preview_id"] != manifest.preview_id or raw["audio_sha256"] != manifest.audio.sha256
            or raw["duration_ms"] != manifest.duration_ms):
        raise PreviewError("preview performance identity does not match audio")
    adapted = dict(raw)
    adapted["release_id"] = adapted.pop("preview_id")
    adapted["package_sha256"] = adapted.pop("audio_sha256")
    try:
        return Performance.model_validate(adapted)
    except ValueError as exc:
        raise PreviewError("preview performance timeline is invalid") from exc


def _semantic(raw: dict, manifest: PreviewManifest, performance: Performance):
    try:
        semantic = PreviewSemantic.model_validate(raw)
    except ValueError as exc:
        raise PreviewError("preview semantic document is invalid") from exc
    if (semantic.preview_id != manifest.preview_id or semantic.audio_sha256 != manifest.audio.sha256
            or semantic.duration_ms != manifest.duration_ms):
        raise PreviewError("preview semantic identity does not match audio")
    if semantic.projection.fitted_count != len(semantic.windows):
        raise PreviewError("preview projection fit count does not match its windows")
    if semantic.projection.dimensions >= 4:
        if any(window.rotation is None for window in semantic.windows):
            raise PreviewError("four- and five-dimensional preview windows require rotation")
    elif any(window.rotation is not None for window in semantic.windows):
        raise PreviewError("three-dimensional preview windows may not declare rotation")
    if semantic.projection.dimensions == 5:
        if any(window.depth_rotation is None for window in semantic.windows):
            raise PreviewError("five-dimensional preview windows require depth rotation")
    elif any(window.depth_rotation is not None for window in semantic.windows):
        raise PreviewError("three- and four-dimensional preview windows may not declare depth rotation")
    known = {character.id for character in performance.cast}
    previous = -1
    ids = set()
    for window in semantic.windows:
        if (window.character_id not in known or window.id in ids
                or not 0 <= window.start_ms < window.end_ms <= manifest.duration_ms
                or window.start_ms < previous):
            raise PreviewError("preview semantic windows are invalid")
        ids.add(window.id)
        previous = window.start_ms
    return semantic


def read_preview(runtime: Path, preview_id: str) -> tuple[PreviewManifest, dict[str, Path]]:
    if not isinstance(preview_id, str) or re.fullmatch(SLUG_PATTERN, preview_id) is None:
        raise FileNotFoundError("unknown local preview")
    root = Path(runtime) / "visual-previews" / preview_id
    raw = _json(_read(root / "manifest.json", MAX_MANIFEST_BYTES))
    try:
        manifest = PreviewManifest.model_validate(raw)
    except ValueError as exc:
        raise PreviewError("preview manifest is invalid") from exc
    if manifest.preview_id != preview_id:
        raise PreviewError("preview directory identity mismatch")
    paths = {}
    try:
        for asset in (manifest.audio, manifest.performance, manifest.semantic):
            if asset is not None:
                path = root / asset.file
                _verify_asset(path, asset)
                paths[asset.file] = path
        _verify_audio(paths["audio.wav"], manifest.duration_ms)
        prepared = _performance(_json(_read(paths["performance.json"], MAX_PERFORMANCE_BYTES)), manifest)
        if manifest.semantic is not None:
            _semantic(_json(_read(paths["semantic.json"], MAX_SEMANTIC_BYTES)), manifest, prepared)
    except FileNotFoundError as exc:
        raise PreviewError("preview bundle is incomplete") from exc
    return manifest, paths


def register_preview_routes(app, runtime: Path):
    """Register fixed GET-only local preview routes on an existing application."""
    def request_is_local_preview(request: Request):
        if request.query_params:
            raise HTTPException(422, "Unsupported preview query.", headers=PREVIEW_HEADERS)

    def verified(preview_id: str):
        try:
            return read_preview(runtime, preview_id)
        except FileNotFoundError:
            raise HTTPException(404, "Local preview is unavailable.", headers=PREVIEW_HEADERS)
        except (PreviewError, OSError, ValueError, RecursionError):
            raise HTTPException(409, "Local preview could not be verified.", headers=PREVIEW_HEADERS)

    @app.get("/api/previews/{preview_id}")
    def preview_catalog(preview_id: str, request: Request):
        request_is_local_preview(request)
        manifest, _ = verified(preview_id)
        result = manifest.model_dump(mode="json")
        result.update(kind="local_preview", local_preview=True)
        for role in ("audio", "performance", "semantic"):
            asset = result[role]
            if asset is not None:
                asset["url"] = "/preview/" + preview_id + "/" + asset["file"]
        return JSONResponse(result, headers=PREVIEW_HEADERS)

    @app.get("/preview/{preview_id}/{asset_name}")
    def preview_asset(preview_id: str, asset_name: str, request: Request):
        request_is_local_preview(request)
        if asset_name not in PREVIEW_FILES:
            raise HTTPException(404, "Local preview asset is unavailable.", headers=PREVIEW_HEADERS)
        _, paths = verified(preview_id)
        path = paths.get(asset_name)
        if path is None:
            raise HTTPException(404, "Local preview asset is unavailable.", headers=PREVIEW_HEADERS)
        return FileResponse(path, media_type="audio/wav" if asset_name == "audio.wav" else "application/json",
                            headers=PREVIEW_HEADERS)
