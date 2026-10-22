"""Timed source embeddings: independent, immutable sidecars for accepted releases.

Hashes identify bytes and authored alignment claims, not publisher signatures.
No embedding generation, projection, alignment inference, or network access occurs here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import struct
import tempfile
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
import rfc8785

from .contract import HASH_PATTERN, MAX_DURATION_MS, SLUG_PATTERN, Manifest, validate_package

MAX_EMBEDDING_MANIFEST_BYTES = 1024 * 1024
MAX_EMBEDDING_SHARD_BYTES = 786432  # 64 * 3072 * sizeof(float32).
MAX_EMBEDDING_WINDOWS = 2000
MAX_VECTORS_PER_SHARD = 64
MAX_CATALOG_BYTES = 1024 * 1024
BUNDLE_PATTERN = HASH_PATTERN
SHARD_PATTERN = r"^vectors-[0-9]{4}\.f32$"
Hash = Annotated[str, Field(pattern=HASH_PATTERN)]
Slug = Annotated[str, Field(pattern=SLUG_PATTERN)]
Evidence = Annotated[str, Field(min_length=1, max_length=2000)]
WindowId = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,159}$")]


class EmbeddingError(ValueError):
    """Embedding sidecar is invalid, stale, or cannot be safely published."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class EmbeddingConfig(StrictModel):
    model: Literal["gemini-embedding-2"] = "gemini-embedding-2"
    dimensions: Annotated[int, Field(ge=128, le=3072)]
    input_format: Literal["common_multimodal"] = "common_multimodal"
    task_prefix: None = None
    provider_revision: Annotated[str, Field(min_length=1, max_length=256)] | None = None

    @field_validator("provider_revision")
    @classmethod
    def revision_identity(cls, value):
        if value is not None and (not value.strip() or "\0" in value):
            raise ValueError("provider revision must be ordinary nonblank text")
        return value

    @property
    def space_id(self) -> str:
        return _digest(self.model_dump(mode="json"))


class AudioSource(StrictModel):
    source_sha256: Hash
    window_sha256: Hash
    source_start_ms: Annotated[int, Field(ge=0, le=MAX_DURATION_MS)]
    kind: Literal["release_mix", "character_stem"]
    alignment_evidence: Evidence
    source_sample_rate: Annotated[int, Field(ge=8000, le=192000)]
    source_start_frame: Annotated[int, Field(ge=0)]
    source_end_frame: Annotated[int, Field(gt=0)]
    extraction: Literal["pcm_s16le_wav_v1"] = "pcm_s16le_wav_v1"

    @field_validator("alignment_evidence")
    @classmethod
    def nonblank_evidence(cls, value):
        if not value.strip() or "\0" in value:
            raise ValueError("alignment evidence must be ordinary nonblank text")
        return value

    @model_validator(mode="after")
    def frame_identity(self):
        if self.source_end_frame <= self.source_start_frame:
            raise ValueError("audio source frame interval must be nonempty")
        if self.source_start_ms != self.source_start_frame * 1000 // self.source_sample_rate:
            raise ValueError("audio source millisecond offset must match exact source frames")
        if self.source_end_frame > MAX_DURATION_MS * self.source_sample_rate // 1000:
            raise ValueError("audio source interval exceeds the duration limit")
        return self


class TextSource(StrictModel):
    transcript_sha256: Hash
    start_byte: Annotated[int, Field(ge=0, le=1024 * 1024)]
    end_byte: Annotated[int, Field(gt=0, le=1024 * 1024)]

    @model_validator(mode="after")
    def interval(self):
        if self.end_byte <= self.start_byte:
            raise ValueError("transcript byte interval must be nonempty")
        return self


class EmbeddingWindow(StrictModel):
    id: WindowId
    passage_id: Slug
    source_modality: Literal["text", "audio", "text_audio"]
    character_id: Slug | None = None
    start_ms: Annotated[int, Field(ge=0, le=MAX_DURATION_MS)]
    end_ms: Annotated[int, Field(gt=0, le=MAX_DURATION_MS)]
    text: Annotated[str, Field(min_length=1, max_length=20000)] | None = None
    audio: AudioSource | None = None
    text_source: TextSource | None = None
    timing_evidence: Evidence
    request_sha256: Hash
    response_sha256: Hash

    @field_validator("timing_evidence")
    @classmethod
    def nonblank_evidence(cls, value):
        if not value.strip() or "\0" in value:
            raise ValueError("timing evidence must be ordinary nonblank text")
        return value

    @model_validator(mode="after")
    def source_shape(self):
        if self.end_ms <= self.start_ms:
            raise ValueError("embedding interval must be nonempty")
        if (self.text is not None) != (self.source_modality in ("text", "text_audio")):
            raise ValueError("exact text must match source modality")
        if self.text is not None and (not self.text.strip() or "\0" in self.text):
            raise ValueError("exact text must contain ordinary nonblank text")
        if (self.text_source is not None) != (self.text is not None):
            raise ValueError("exact text requires transcript byte provenance")
        if self.text_source is not None:
            if self.text_source.end_byte - self.text_source.start_byte != len(self.text.encode("utf-8")):
                raise ValueError("transcript interval must contain exactly the UTF-8 text bytes")
        if (self.audio is not None) != (self.source_modality in ("audio", "text_audio")):
            raise ValueError("audio identity must match source modality")
        if self.audio is not None:
            if self.audio.kind == "release_mix" and self.character_id is not None:
                raise ValueError("release mix cannot be attributed to one character")
            if self.audio.kind == "character_stem" and self.character_id is None:
                raise ValueError("character stems require an explicit character identity")
            measured = (self.audio.source_end_frame - self.audio.source_start_frame) * 1000
            if abs(measured - (self.end_ms - self.start_ms) * self.audio.source_sample_rate) > self.audio.source_sample_rate:
                raise ValueError("source frame duration must match release window duration within one millisecond")
            if self.audio.kind == "release_mix" and abs(self.audio.source_start_ms - self.start_ms) > 1:
                raise ValueError("release mix offset must match release timeline")
        return self


class StoredWindow(EmbeddingWindow):
    shard_id: Annotated[str, Field(pattern=SHARD_PATTERN)]
    vector_index: Annotated[int, Field(ge=0, lt=MAX_VECTORS_PER_SHARD)]


class Shard(StrictModel):
    id: Annotated[str, Field(pattern=SHARD_PATTERN)]
    sha256: Hash
    count: Annotated[int, Field(ge=1, le=MAX_VECTORS_PER_SHARD)]
    size_bytes: Annotated[int, Field(gt=0, le=MAX_EMBEDDING_SHARD_BYTES)]


class EmbeddingManifest(StrictModel):
    schema_version: Literal[1]
    kind: Literal["source_embeddings"]
    bundle_id: Hash
    release_id: Slug
    package_sha256: Hash
    release_audio_sha256: Hash
    release_transcript_sha256: Hash
    duration_ms: Annotated[int, Field(gt=0, le=MAX_DURATION_MS)]
    config: EmbeddingConfig
    space_id: Hash
    vector_format: Literal["float32_le"] = "float32_le"
    windows: Annotated[list[StoredWindow], Field(min_length=1, max_length=MAX_EMBEDDING_WINDOWS)]
    shards: Annotated[list[Shard], Field(min_length=1, max_length=32)]

    @field_validator("schema_version", mode="before")
    @classmethod
    def exact_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema version must be the integer 1")
        return value

    @model_validator(mode="after")
    def identities(self):
        if self.space_id != self.config.space_id:
            raise ValueError("space identity does not match common embedding configuration")
        if len({item.id for item in self.windows}) != len(self.windows):
            raise ValueError("window vector IDs must be unique")
        shards = {shard.id: shard for shard in self.shards}
        if len(shards) != len(self.shards):
            raise ValueError("shard IDs must be unique")
        references = set()
        passages = {}
        passage_sources = {}
        passage_modalities = set()
        for window in self.windows:
            if window.end_ms > self.duration_ms:
                raise ValueError("embedding window extends beyond the accepted release")
            shard = shards.get(window.shard_id)
            if shard is None or window.vector_index >= shard.count:
                raise ValueError("window references an undeclared vector")
            reference = (window.shard_id, window.vector_index)
            if reference in references:
                raise ValueError("each stored vector must be referenced exactly once")
            references.add(reference)
            if window.audio is not None and window.audio.kind == "release_mix":
                if window.audio.source_sha256 != self.release_audio_sha256:
                    raise ValueError("release mix source must be the accepted audio bytes")
            if (window.passage_id, window.source_modality) in passage_modalities:
                raise ValueError("each passage modality must be unique")
            passage_modalities.add((window.passage_id, window.source_modality))
            sources = passage_sources.setdefault(window.passage_id, {})
            for label, source in (("text", (window.text, window.text_source) if window.text is not None else None),
                                  ("audio", window.audio)):
                if source is not None:
                    if label in sources and sources[label] != source:
                        raise ValueError("matched modalities must retain identical source provenance")
                    sources[label] = source
            signature = (window.start_ms, window.end_ms)
            old = passages.setdefault(window.passage_id, signature)
            if old != signature:
                raise ValueError("matched passage modalities must share the same release interval")
        for shard in self.shards:
            if shard.size_bytes != shard.count * self.config.dimensions * 4:
                raise ValueError("shard size must match its vector shape")
            if any((shard.id, index) not in references for index in range(shard.count)):
                raise ValueError("unreferenced vectors are forbidden")
        if self.bundle_id != bundle_digest(self):
            raise ValueError("bundle content identity does not match manifest")
        return self


def _canonical(value) -> bytes:
    try:
        return rfc8785.dumps(value)
    except (TypeError, ValueError) as exc:
        raise EmbeddingError("embedding metadata is not canonical JSON") from exc


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def bundle_digest(manifest: EmbeddingManifest | dict) -> str:
    value = manifest.model_dump(mode="json") if isinstance(manifest, EmbeddingManifest) else dict(manifest)
    value.pop("bundle_id", None)
    return _digest(value)


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise EmbeddingError("duplicate JSON key")
        result[key] = value
    return result


def _finite_json(_value):
    raise EmbeddingError("non-finite JSON number")


def _ordinary_parents(path: Path):
    # Reject symlinks anywhere in the supplied path, including ancestor directories.
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise EmbeddingError("embedding paths must not contain symbolic links")


def _read_bounded(path: Path, limit: int) -> bytes:
    path = Path(path)
    _ordinary_parents(path.parent)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= limit:
                raise EmbeddingError("embedding data must be a bounded regular file")
            data = source.read(limit + 1)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise EmbeddingError("embedding data is unavailable") from exc
    if len(data) > limit:
        raise EmbeddingError("embedding data exceeds its byte limit")
    return data


def validate_document(data: bytes, release: Manifest) -> EmbeddingManifest:
    if not data or len(data) > MAX_EMBEDDING_MANIFEST_BYTES:
        raise EmbeddingError("embedding manifest exceeds its byte limit")
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_keys, parse_constant=_finite_json)
        document = EmbeddingManifest.model_validate(value)
    except (UnicodeError, ValueError, TypeError, RecursionError) as exc:
        raise EmbeddingError("invalid embedding manifest") from exc
    assets = {asset.role: asset.sha256 for asset in release.assets}
    transcript_size = next(asset.size_bytes for asset in release.assets if asset.role == "transcript")
    for window in document.windows:
        if window.text_source is not None and (
                window.text_source.transcript_sha256 != assets["transcript"]
                or window.text_source.end_byte > transcript_size):
            raise EmbeddingError("text source does not match the accepted transcript")
    if (document.release_id != release.release_id
            or document.package_sha256 != release.package_sha256
            or document.duration_ms != release.duration_ms
            or document.release_audio_sha256 != assets["audio"]
            or document.release_transcript_sha256 != assets["transcript"]):
        raise EmbeddingError("embeddings do not match the accepted release")
    if len(_canonical(document.model_dump(mode="json"))) > MAX_EMBEDDING_MANIFEST_BYTES:
        raise EmbeddingError("normalized embedding manifest exceeds its byte limit")
    return document


def read_bundle(source: Path, release: Manifest) -> tuple[EmbeddingManifest, dict[str, bytes]]:
    source = Path(source)
    root = source.parent if source.name == "manifest.json" else source
    _ordinary_parents(root)
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise EmbeddingError("embedding bundle must be an ordinary directory")
    try:
        metadata = _read_bounded(root / "manifest.json", MAX_EMBEDDING_MANIFEST_BYTES)
    except FileNotFoundError as exc:
        raise EmbeddingError("existing embedding bundle is incomplete") from exc
    document = validate_document(metadata, release)
    expected = {"manifest.json"} | {shard.id for shard in document.shards}
    if {path.name for path in root.iterdir()} != expected:
        raise EmbeddingError("embedding bundle inventory differs from its manifest")
    shards = {}
    for shard in document.shards:
        try:
            data = _read_bounded(root / shard.id, MAX_EMBEDDING_SHARD_BYTES)
        except FileNotFoundError as exc:
            raise EmbeddingError("embedding shard is missing") from exc
        if len(data) != shard.size_bytes or hashlib.sha256(data).hexdigest() != shard.sha256:
            raise EmbeddingError("embedding shard checksum or shape mismatch")
        if any(not math.isfinite(value[0]) for value in struct.iter_unpack("<f", data)):
            raise EmbeddingError("embedding vectors must contain finite float32 values")
        shards[shard.id] = data
    return document, shards


@contextmanager
def _publication_lock(parent: Path):
    _ordinary_parents(parent)
    descriptor = os.open(parent / ".publication.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as lock:
        if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
            raise EmbeddingError("embedding publication lock must be ordinary")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _publish(destination: Path, document: EmbeddingManifest, shards: dict[str, bytes]) -> Path:
    destination = Path(destination)
    _ordinary_parents(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".embedding-", dir=destination.parent))
    try:
        encoded = _canonical(document.model_dump(mode="json")) + b"\n"
        if len(encoded) > MAX_EMBEDDING_MANIFEST_BYTES:
            raise EmbeddingError("embedding manifest exceeds its byte limit")
        for name, data in {"manifest.json": encoded, **shards}.items():
            with (staging / name).open("xb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
        descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        with _publication_lock(destination.parent):
            if destination.exists() or destination.is_symlink():
                raise EmbeddingError("embedding bundle destination already exists; bundles are immutable")
            os.rename(staging, destination)
            descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        return destination / "manifest.json"
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def preflight_windows(release: Manifest, config: EmbeddingConfig | dict,
                      windows: list[EmbeddingWindow | dict]) -> list[EmbeddingWindow]:
    """Validate all metadata and the final manifest byte bound before dispatch."""
    try:
        config = EmbeddingConfig.model_validate(config)
        windows = [EmbeddingWindow.model_validate(window) for window in windows]
        if not 0 < len(windows) <= MAX_EMBEDDING_WINDOWS:
            raise EmbeddingError("embedding window count exceeds its limit")
        descriptors = []
        stored = []
        for offset in range(0, len(windows), MAX_VECTORS_PER_SHARD):
            name = f"vectors-{offset // MAX_VECTORS_PER_SHARD:04d}.f32"
            rows = windows[offset:offset + MAX_VECTORS_PER_SHARD]
            descriptors.append({"id": name, "sha256": "0" * 64, "count": len(rows),
                                "size_bytes": len(rows) * config.dimensions * 4})
            stored.extend({**window.model_dump(mode="json"), "shard_id": name, "vector_index": index}
                          for index, window in enumerate(rows))
        assets = {asset.role: asset.sha256 for asset in release.assets}
        raw = {"schema_version": 1, "kind": "source_embeddings", "release_id": release.release_id,
               "package_sha256": release.package_sha256, "release_audio_sha256": assets["audio"],
               "release_transcript_sha256": assets["transcript"], "duration_ms": release.duration_ms,
               "config": config.model_dump(mode="json"), "space_id": config.space_id,
               "vector_format": "float32_le", "windows": stored, "shards": descriptors}
        raw["bundle_id"] = bundle_digest(raw)
        encoded = _canonical(raw) + b"\n"
        # A response may reveal an otherwise unknown immutable provider revision.
        # Reserve its worst-case UTF-8 expansion before any paid dispatch.
        revision_reserve = 1024 if config.provider_revision is None else 0
        if len(encoded) + revision_reserve > MAX_EMBEDDING_MANIFEST_BYTES:
            raise EmbeddingError("embedding manifest exceeds its byte limit")
        validate_document(encoded, release)
        return windows
    except (TypeError, ValueError, UnicodeError, OverflowError) as exc:
        if isinstance(exc, EmbeddingError):
            raise
        raise EmbeddingError("invalid embedding window metadata") from exc


def write_bundle(destination: Path, release: Manifest, config: EmbeddingConfig | dict,
                 windows: list[EmbeddingWindow | dict], vectors: list[list[float]]) -> Path:
    """Build an immutable source bundle without installing or generating anything.

    destination is a NEW final directory. vectors follows windows order. All
    modalities use the same config and one unprojected space. A missing provider
    revision means the provider did not expose immutable weights, never historical
    bitwise reproducibility. Alignment is
    authored/verified by the caller; this function never guesses speaker times.
    """
    try:
        config = EmbeddingConfig.model_validate(config)
        windows = preflight_windows(release, config, windows)
        if len(windows) != len(vectors):
            raise EmbeddingError("one vector is required per bounded source window")
        shards = {}
        descriptors = []
        stored = []
        for offset in range(0, len(windows), MAX_VECTORS_PER_SHARD):
            name = f"vectors-{offset // MAX_VECTORS_PER_SHARD:04d}.f32"
            rows = vectors[offset:offset + MAX_VECTORS_PER_SHARD]
            values = []
            for row in rows:
                if not isinstance(row, (list, tuple)) or len(row) != config.dimensions:
                    raise EmbeddingError("embedding vector dimensions do not match configuration")
                for value in row:
                    if type(value) not in (float, int) or not math.isfinite(value):
                        raise EmbeddingError("embedding vectors must contain finite numeric values")
                    values.append(value)
            try:
                data = struct.pack("<" + "f" * len(values), *values)
            except (OverflowError, struct.error) as exc:
                raise EmbeddingError("embedding value is outside float32 range") from exc
            shards[name] = data
            descriptors.append({"id": name, "sha256": hashlib.sha256(data).hexdigest(),
                                "count": len(rows), "size_bytes": len(data)})
            for index, window in enumerate(windows[offset:offset + MAX_VECTORS_PER_SHARD]):
                stored.append({**window.model_dump(mode="json"), "shard_id": name, "vector_index": index})
        assets = {asset.role: asset.sha256 for asset in release.assets}
        raw = {"schema_version": 1, "kind": "source_embeddings", "release_id": release.release_id,
               "package_sha256": release.package_sha256, "release_audio_sha256": assets["audio"],
               "release_transcript_sha256": assets["transcript"], "duration_ms": release.duration_ms,
               "config": config.model_dump(mode="json"), "space_id": config.space_id,
               "vector_format": "float32_le", "windows": stored, "shards": descriptors}
        raw["bundle_id"] = bundle_digest(raw)
        document = validate_document(_canonical(raw), release)
    except (ValidationError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, EmbeddingError):
            raise
        raise EmbeddingError("invalid embedding bundle inputs") from exc
    destination = Path(destination)
    if destination.exists():
        existing, existing_shards = read_bundle(destination, release)
        if existing == document and existing_shards == shards:
            return destination / "manifest.json"
        raise EmbeddingError("existing embedding bundle conflicts with source")
    try:
        return _publish(destination, document, shards)
    except EmbeddingError:
        if destination.exists():
            existing, existing_shards = read_bundle(destination, release)
            if existing == document and existing_shards == shards:
                return destination / "manifest.json"
        raise


def _verify_text_sources(document: EmbeddingManifest, runtime: Path, release: Manifest):
    asset = next(asset for asset in release.assets if asset.role == "transcript")
    data = _read_bounded(Path(runtime) / "station" / "releases" / release.release_id / asset.path,
                         1024 * 1024)
    if hashlib.sha256(data).hexdigest() != asset.sha256:
        raise EmbeddingError("accepted transcript bytes failed verification")
    for window in document.windows:
        source = window.text_source
        if source is not None and data[source.start_byte:source.end_byte] != window.text.encode("utf-8"):
            raise EmbeddingError("exact text does not match the accepted transcript interval")


def accepted_release(runtime: Path, release_id: str) -> Manifest:
    if not isinstance(release_id, str) or re.fullmatch(SLUG_PATTERN, release_id) is None:
        raise EmbeddingError("invalid release identity")
    station = Path(runtime) / "station"
    journal = station / "journal.sqlite3"
    _ordinary_parents(journal)
    if not journal.is_file():
        raise EmbeddingError("an existing accepted station release is required")
    try:
        with sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True) as db:
            row = db.execute("SELECT digest FROM accepted WHERE release_id=?", (release_id,)).fetchone()
        if row is None:
            raise EmbeddingError("release has not been accepted")
        _ordinary_parents(station / "releases" / release_id)
        return validate_package(station / "releases" / release_id, expected_digest=row[0])
    except (sqlite3.Error, OSError, ValueError) as exc:
        raise EmbeddingError("accepted release could not be verified") from exc


@contextmanager
def _installation_lease(runtime: Path):
    studio = Path(runtime) / "studio"
    _ordinary_parents(studio)
    studio.mkdir(parents=False, exist_ok=True)
    if not studio.is_dir():
        raise EmbeddingError("studio runtime must be an ordinary directory")
    descriptor = os.open(studio / "generation-worker.lock",
                         os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    with os.fdopen(descriptor, "a+b") as lease:
        if not stat.S_ISREG(os.fstat(lease.fileno()).st_mode):
            raise EmbeddingError("generation lease must be an ordinary file")
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise EmbeddingError("runtime backup is active") from exc
        try:
            yield
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)


def install_bundle(source: Path, runtime: Path) -> Path:
    """Install immutable bytes while holding the backup-coordinated shared lease."""
    source = Path(source)
    root = source.parent if source.name == "manifest.json" else source
    data = _read_bounded(root / "manifest.json", MAX_EMBEDDING_MANIFEST_BYTES)
    try:
        raw = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_keys, parse_constant=_finite_json)
        release_id = raw.get("release_id") if isinstance(raw, dict) else None
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise EmbeddingError("invalid embedding manifest") from exc
    # Even SQLite mode=ro can create WAL/SHM coordination files. Do not open
    # the journal until the backup lease is held. A missing runtime stays absent.
    journal = Path(runtime) / "station" / "journal.sqlite3"
    _ordinary_parents(journal)
    if not journal.is_file():
        raise EmbeddingError("an existing accepted station release is required")
    with _installation_lease(runtime):
        return _install_bundle_unlocked(root, runtime)


def _install_bundle_unlocked(source: Path, runtime: Path) -> Path:
    source = Path(source)
    root = source.parent if source.name == "manifest.json" else source
    data = _read_bounded(root / "manifest.json", MAX_EMBEDDING_MANIFEST_BYTES)
    try:
        raw = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_keys, parse_constant=_finite_json)
        release_id = raw.get("release_id") if isinstance(raw, dict) else None
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise EmbeddingError("invalid embedding manifest") from exc
    release = accepted_release(runtime, release_id)
    document, shards = read_bundle(root, release)
    _verify_text_sources(document, runtime, release)
    destination = Path(runtime) / "station" / "embeddings" / release.release_id / document.bundle_id
    _ordinary_parents(destination)
    if destination.exists():
        existing, existing_shards = read_bundle(destination, release)
        if existing != document or existing_shards != shards:
            raise EmbeddingError("existing embedding bundle conflicts with source")
        return destination / "manifest.json"
    try:
        return _publish(destination, document, shards)
    except EmbeddingError:
        # Another installer may have published the exact same immutable bytes.
        if destination.exists():
            existing, existing_shards = read_bundle(destination, release)
            if existing == document and existing_shards == shards:
                return destination / "manifest.json"
        raise


def installed_bundle(runtime: Path, release_id: str, bundle_id: str) -> tuple[EmbeddingManifest, dict[str, bytes]]:
    if not isinstance(bundle_id, str) or re.fullmatch(BUNDLE_PATTERN, bundle_id) is None:
        raise FileNotFoundError("unknown embedding bundle")
    release = accepted_release(runtime, release_id)
    root = Path(runtime) / "station" / "embeddings" / release_id / bundle_id
    document, shards = read_bundle(root, release)
    _verify_text_sources(document, runtime, release)
    if document.bundle_id != bundle_id:
        raise EmbeddingError("installed embedding directory identity does not match its content")
    return document, shards


def bundle_catalog(runtime: Path, release_id: str) -> dict:
    release = accepted_release(runtime, release_id)
    root = Path(runtime) / "station" / "embeddings" / release_id
    _ordinary_parents(root)
    if not root.exists():
        raise FileNotFoundError("no installed embedding bundles")
    if not root.is_dir():
        raise EmbeddingError("embedding catalog must be an ordinary directory")
    bundles = []
    for path in sorted(root.iterdir()):
        if path.name.startswith("."):
            continue
        if re.fullmatch(BUNDLE_PATTERN, path.name) is None:
            raise EmbeddingError("unexpected installed embedding entry")
        document, _ = read_bundle(path, release)
        _verify_text_sources(document, runtime, release)
        if document.bundle_id != path.name:
            raise EmbeddingError("installed embedding directory identity does not match its content")
        bundles.append({"bundle_id": document.bundle_id, "space_id": document.space_id,
                        "config": document.config.model_dump(mode="json"),
                        "window_count": len(document.windows)})
        if len(bundles) > 2048:
            raise EmbeddingError("embedding catalog exceeds its entry limit")
    if not bundles:
        raise FileNotFoundError("no installed embedding bundles")
    value = {"schema_version": 1, "kind": "source_embeddings_catalog", "release_id": release_id,
             "package_sha256": release.package_sha256, "duration_ms": release.duration_ms,
             "bundles": bundles}
    if len(_canonical(value)) > MAX_CATALOG_BYTES:
        raise EmbeddingError("embedding catalog exceeds its byte limit")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("install")
    install.add_argument("source", type=Path)
    install.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    try:
        target = install_bundle(args.source, args.runtime)
    except (EmbeddingError, OSError, ValueError):
        parser.exit(2, "Embedding bundle was not installed.\n")
    print("Installed embedding bundle", target.parent.name)


if __name__ == "__main__":
    main()
