"""Offline preparation of timed source embeddings; no projection or automatic dispatch."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
import subprocess
import tempfile
from typing import Annotated, Callable, Literal
from uuid import uuid4
import wave

from pydantic import BaseModel, ConfigDict, Field, model_validator
from .budget import Budget, BudgetError, dollars_to_microdollars
from .contract import HASH_PATTERN, SLUG_PATTERN, MAX_AUDIO_BYTES, MAX_DURATION_MS, validate_package, Manifest

MAX_PLAN_BYTES = 64 * 1024 * 1024
MAX_SOURCE_BYTES = MAX_AUDIO_BYTES
MAX_RESPONSE_BYTES = 1024 * 1024
PRICE_CHECKED = "2026-10-22"
PRICE_VALID_THROUGH = date(2026, 12, 31)
PRICE_SOURCE = "https://ai.google.dev/gemini-api/docs/pricing#gemini-embedding-2"
ID = r"^[0-9a-f]{32}$"
Slug = Annotated[str, Field(pattern=SLUG_PATTERN)]
Digest = Annotated[str, Field(pattern=HASH_PATTERN)]


class EmbeddingError(ValueError):
    """Rejected local preparation or unresolved provider outcome."""


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class AudioInput(StrictInput):
    kind: Literal["release_mix", "character_stem"]
    path: str | None = Field(default=None, min_length=1, max_length=240)
    source_sha256: Digest
    source_start_frame: Annotated[int, Field(ge=0)]
    alignment_evidence: str = Field(min_length=12, max_length=2000)

    @model_validator(mode="after")
    def source_kind(self):
        if (self.kind == "character_stem") != (self.path is not None):
            raise ValueError("Only a declared character stem has an authored source path.")
        return self


class WindowInput(StrictInput):
    window_id: Slug
    passage_id: Slug
    character_id: Slug | None = None
    start_ms: Annotated[int, Field(ge=0, le=MAX_DURATION_MS)]
    end_ms: Annotated[int, Field(gt=0, le=MAX_DURATION_MS)]
    timing_evidence: str = Field(min_length=12, max_length=2000)
    text: str | None = Field(default=None, min_length=1, max_length=4096)
    text_start_byte: Annotated[int, Field(ge=0)] | None = None
    audio: AudioInput | None = None
    modalities: Annotated[list[Literal["text", "audio", "text_audio"]], Field(min_length=1, max_length=3)]

    @model_validator(mode="after")
    def window_shape(self):
        if self.end_ms <= self.start_ms or self.end_ms - self.start_ms > 30_000:
            raise ValueError("Embedding windows must be nonempty and at most thirty seconds.")
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError("A window may request each modality once.")
        needs_text = any(mode in ("text", "text_audio") for mode in self.modalities)
        needs_audio = any(mode in ("audio", "text_audio") for mode in self.modalities)
        if needs_text != (self.text is not None) or needs_text != (self.text_start_byte is not None):
            raise ValueError("Text modes require exact transcript text and byte offset.")
        if needs_audio != (self.audio is not None):
            raise ValueError("Audio modes require an explicit source.")
        if self.audio:
            if self.audio.kind == "release_mix" and self.character_id is not None:
                raise ValueError("Mixed audio cannot establish character identity.")
            if self.audio.kind == "character_stem" and self.character_id is None:
                raise ValueError("A known character stem needs its editorial character ID.")
        return self


class PreparationSpec(StrictInput):
    schema_version: Literal[1]
    release_id: Slug
    package_sha256: Digest
    duration_ms: Annotated[int, Field(gt=0, le=MAX_DURATION_MS)]
    dimensions: Annotated[int, Field(ge=128, le=3072)] = 3072
    windows: Annotated[list[WindowInput], Field(min_length=1, max_length=512)]

    @model_validator(mode="before")
    @classmethod
    def version_is_integer(cls, value):
        if isinstance(value, dict) and type(value.get("schema_version")) is not int:
            raise ValueError("The preparation schema version must be integer one.")
        return value

    @model_validator(mode="after")
    def identities(self):
        if len({item.window_id for item in self.windows}) != len(self.windows):
            raise ValueError("Window IDs must be unique.")
        if any(item.end_ms > self.duration_ms for item in self.windows):
            raise ValueError("Windows must fit the declared release.")
        # A passage identity names one actual audible interval, across modalities.
        bindings = {}
        for item in self.windows:
            binding = (item.start_ms, item.end_ms, item.character_id)
            if item.passage_id in bindings and bindings[item.passage_id] != binding:
                raise ValueError("A passage ID cannot name different timing or characters.")
            bindings[item.passage_id] = binding
        return self


def _encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _ordinary(path: Path):
    path = Path(path)
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise EmbeddingError("Storage/source paths must not traverse symlinks.")
    return path


def _read(path: Path, limit: int):
    _ordinary(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= limit:
            raise EmbeddingError("Expected a bounded ordinary file.")
        data = source.read(limit + 1)
    if len(data) > limit:
        raise EmbeddingError("File exceeds its bound.")
    return data


def _write(path: Path, data: bytes):
    _ordinary(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read(path, max(len(data), 1)) != data:
            raise EmbeddingError("Immutable preparation artifact conflicts with existing data.")
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".write-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _read(path, max(len(data), 1)) != data:
                raise EmbeddingError("Immutable preparation artifact conflicts with existing data.")
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def _allocation(modality: str, today: date | None = None):
    if (today or date.today()) > PRICE_VALID_THROUGH:
        raise BudgetError("Embedding price schedule expired; verify rates before dispatch.")
    # Shared 8192-token provider context. Charge every token at the highest enabled
    # modality rate; this is not an estimate of the actual audio tokenization.
    return 1639 if modality == "text" else 53248


class EmbeddingService:
    def __init__(self, project_root: Path, *, key_provider: Callable[[], str | None] | None = None):
        self.root = _ordinary(Path(os.path.abspath(project_root)))
        if not self.root.is_dir():
            raise EmbeddingError("An existing project directory is required.")
        self.state = self.root / ".runtime/studio/embeddings"
        self.database = self.root / ".runtime/studio/studio.sqlite3"
        self.key_provider = key_provider or self._stored_key
        with self._operation():
            self.state.mkdir(parents=True, exist_ok=True)
            self.budget = Budget(self.database)
            with self.budget.connect() as db:
                db.executescript("""
                CREATE TABLE IF NOT EXISTS embedding_jobs(
                    id TEXT PRIMARY KEY,status TEXT NOT NULL,created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,spec_json TEXT NOT NULL,manifest_json TEXT NOT NULL,
                    rows_json TEXT NOT NULL,config_json TEXT NOT NULL,bundle_path TEXT,error TEXT);
                CREATE TABLE IF NOT EXISTS embedding_requests(
                    job_id TEXT NOT NULL,request_sha256 TEXT NOT NULL,ordinal INTEGER NOT NULL,
                    modality TEXT NOT NULL,state TEXT NOT NULL,attempt_id TEXT,
                    allocation_micro INTEGER NOT NULL,response_sha256 TEXT,paid_attempt_id TEXT,
                    error TEXT,PRIMARY KEY(job_id,request_sha256));
                CREATE TABLE IF NOT EXISTS embedding_cache(
                    request_sha256 TEXT PRIMARY KEY,response_sha256 TEXT NOT NULL,
                    paid_attempt_id TEXT NOT NULL,created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS embedding_reconciliations(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,job_id TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,charged_micro INTEGER NOT NULL,
                    evidence TEXT NOT NULL,recorded_at TEXT NOT NULL);
                """)

    @contextmanager
    def _operation(self):
        studio = _ordinary(self.root / ".runtime/studio")
        _ordinary(self.state)
        studio.mkdir(parents=True, exist_ok=True)
        path = studio / "generation-worker.lock"
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        with os.fdopen(descriptor, "r+b") as lease:
            if not stat.S_ISREG(os.fstat(lease.fileno()).st_mode):
                raise EmbeddingError("The authoring lease must be ordinary.")
            try:
                fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raise EmbeddingError("An offline backup or exclusive generation operation is active.") from None
            try:
                yield
            finally:
                fcntl.flock(lease, fcntl.LOCK_UN)

    @contextmanager
    def _worker(self):
        with self._operation():
            path = self.state / "worker.lock"
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
            with os.fdopen(descriptor, "r+b") as lease:
                if not stat.S_ISREG(os.fstat(lease.fileno()).st_mode):
                    raise EmbeddingError("The embedding worker lease must be ordinary.")
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise EmbeddingError("An embedding operation is active; state preserved.") from None
                try:
                    yield
                finally:
                    fcntl.flock(lease, fcntl.LOCK_UN)

    def _stored_key(self):
        path = _ordinary(self.root / ".runtime/secrets/gemini-api-key")
        if path.exists():
            return _read(path, 4096).decode("utf-8").strip() or None
        return os.environ.get("GEMINI_API_KEY", "").strip() or None

    def _accepted(self, spec):
        station = _ordinary(self.root / ".runtime/station")
        journal = _ordinary(station / "journal.sqlite3")
        if not journal.is_file():
            raise EmbeddingError("A previously accepted station release is required.")
        with sqlite3.connect(journal.as_uri() + "?mode=ro", uri=True) as db:
            row = db.execute("SELECT digest FROM accepted WHERE release_id=?", (spec.release_id,)).fetchone()
        if row is None or row[0] != spec.package_sha256:
            raise EmbeddingError("The preparation does not match an accepted release.")
        directory = _ordinary(station / "releases" / spec.release_id)
        manifest = validate_package(directory, expected_digest=spec.package_sha256)
        if manifest.duration_ms != spec.duration_ms:
            raise EmbeddingError("The preparation duration does not match the release.")
        return manifest, directory

    @staticmethod
    def _wav(data):
        try:
            with wave.open(io.BytesIO(data), "rb") as stream:
                if (stream.getcomptype() != "NONE" or stream.getsampwidth() not in (2, 3, 4)
                        or stream.getnchannels() not in (1, 2)
                        or not 8000 <= stream.getframerate() <= 192000
                        or stream.getnframes() <= 0):
                    raise EmbeddingError("Only bounded mono/stereo PCM WAV sources are supported.")
                params = stream.getparams()
                frames = stream.readframes(stream.getnframes())
                if len(frames) != params.nframes * params.nchannels * params.sampwidth:
                    raise EmbeddingError("The source WAV is incomplete.")
                return params, frames
        except (wave.Error, EOFError) as exc:
            raise EmbeddingError("A complete ordinary PCM WAV source is required.") from exc

    def _audio(self, source, window, release_audio, audio_asset):
        if source.kind == "release_mix":
            path = release_audio
            if source.source_sha256 != audio_asset.sha256:
                raise EmbeddingError("Mixed source must be the exact accepted audio.")
        else:
            relative = Path(source.path)
            if (relative.is_absolute() or chr(92) in source.path or ".." in relative.parts
                    or source.path.startswith("/") or relative.suffix.lower() != ".wav"
                    or not (source.path.startswith("content/") or source.path.startswith(".runtime/studio/takes/")
                            or re.fullmatch(r"\.runtime/studio/scenes/[0-9a-f]{32}/output/stems/[a-z0-9][a-z0-9-]{0,127}\.wav",
                                            source.path) is not None)):
                raise EmbeddingError("A stem must be an ordinary project WAV under content, retained takes, or a scene output's stems.")
            path = self.root / relative
        data = _read(path, MAX_SOURCE_BYTES)
        if _hash(data) != source.source_sha256:
            raise EmbeddingError("The authored audio source hash does not match.")
        params, frames = self._wav(data)
        start = source.source_start_frame
        count = (window.end_ms - window.start_ms) * params.framerate // 1000
        end = start + count
        if count <= 0 or end > params.nframes:
            raise EmbeddingError("The audio window exceeds actual source frames.")
        if source.kind == "release_mix" and start != window.start_ms * params.framerate // 1000:
            raise EmbeddingError("Mixed audio frame offsets must match the release timeline.")
        cropped = io.BytesIO()
        with wave.open(cropped, "wb") as stream:
            stream.setparams(params._replace(nframes=count))
            frame_bytes = params.sampwidth * params.nchannels
            stream.writeframes(frames[start * frame_bytes:end * frame_bytes])
        converted = subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "wav", "-i", "pipe:0",
             "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav", "pipe:1"],
            input=cropped.getvalue(), capture_output=True, timeout=30)
        if converted.returncode:
            raise EmbeddingError("The audio window could not be prepared.")
        # Pipe-WAV headers can use sentinel lengths: rewrap actual decoded frames.
        with wave.open(io.BytesIO(converted.stdout), "rb") as stream:
            raw = stream.readframes(16_000 * 31)
        expected = (window.end_ms - window.start_ms) * 16
        if not raw or abs(len(raw) // 2 - expected) > 2:
            raise EmbeddingError("Prepared audio duration does not match authored release timing.")
        output = io.BytesIO()
        with wave.open(output, "wb") as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
            stream.writeframes(raw)
        audio = output.getvalue()
        return audio, {
            "source_sha256": source.source_sha256, "window_sha256": _hash(audio),
            "source_start_ms": start * 1000 // params.framerate,
            "source_sample_rate": params.framerate, "source_start_frame": start, "source_end_frame": end,
            "kind": source.kind, "alignment_evidence": source.alignment_evidence,
            "extraction": "pcm_s16le_wav_v1",
        }

    def _prepare(self, value):
        from . import embedding_provider as provider
        from .embedding_contract import EmbeddingConfig, preflight_windows
        spec = PreparationSpec.model_validate(value.model_dump() if isinstance(value, PreparationSpec) else value)
        manifest, release = self._accepted(spec)
        audio_asset = next(asset for asset in manifest.assets if asset.role == "audio")
        transcript_asset = next(asset for asset in manifest.assets if asset.role == "transcript")
        release_audio = release / audio_asset.path
        release_params, _ = self._wav(_read(release_audio, MAX_SOURCE_BYTES))
        transcript = _read(release / transcript_asset.path, 1024 * 1024)
        config = EmbeddingConfig(dimensions=spec.dimensions)
        rows, requests = [], {}
        total = 0
        for window in spec.windows:
            if window.end_ms * release_params.framerate > release_params.nframes * 1000:
                raise EmbeddingError("Window exceeds the actual accepted audio, regardless of rounded manifest duration.")
            text_source = None
            if window.text is not None:
                encoded = window.text.encode("utf-8")
                begin = window.text_start_byte
                if transcript[begin:begin + len(encoded)] != encoded:
                    raise EmbeddingError("Exact text and UTF-8 offset must match the accepted transcript.")
                text_source = {"transcript_sha256": transcript_asset.sha256,
                               "start_byte": begin, "end_byte": begin + len(encoded)}
            audio, audio_source = (None, None)
            if window.audio:
                audio, audio_source = self._audio(window.audio, window, release_audio, audio_asset)
            for modality in window.modalities:
                text = window.text if modality in ("text", "text_audio") else None
                clip = audio if modality in ("audio", "text_audio") else None
                request = provider.build_request(text=text, audio=clip, dimensions=spec.dimensions)
                preflight = provider.validate_request(request)
                encoded = _encode(request)
                digest = _hash(encoded)
                if digest not in requests:
                    total += len(encoded)
                    if total > MAX_PLAN_BYTES:
                        raise EmbeddingError("Prepared requests exceed the 64 MiB job limit; use smaller jobs.")
                    requests[digest] = {"request": request, "bytes": encoded, "modality": modality,
                                        "allocation": _allocation(modality), "preflight": preflight}
                rows.append({
                    "id": window.window_id + "-" + modality.replace("_", "-"),
                    "passage_id": window.passage_id, "source_modality": modality,
                    "character_id": window.character_id, "start_ms": window.start_ms, "end_ms": window.end_ms,
                    "timing_evidence": window.timing_evidence, "text": text,
                    "text_source": text_source if text is not None else None,
                    "audio": audio_source if clip is not None else None,
                    "request_sha256": digest, "response_sha256": "0" * 64,
                })
        preflight_windows(manifest, config, rows)
        return spec, manifest, config, rows, requests

    def _job_dir(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch(ID, identifier):
            raise EmbeddingError("Unknown embedding job.")
        return _ordinary(self.state / "jobs" / identifier)

    def _cache(self, digest, dimensions):
        from .embedding_provider import parse_response
        with self.budget.connect() as db:
            row = db.execute("SELECT * FROM embedding_cache WHERE request_sha256=?", (digest,)).fetchone()
        if row is None:
            return None
        directory = self.state / "cache" / digest
        request = _read(directory / "request.json", MAX_PLAN_BYTES)
        response = _read(directory / "response.json", MAX_RESPONSE_BYTES)
        if _hash(request) != digest or _hash(response) != row["response_sha256"]:
            raise EmbeddingError("Cached provider artifacts failed authoritative verification.")
        vector, metadata = parse_response(json.loads(response), dimensions)
        return {"vector": vector, "metadata": metadata, "response_sha256": row["response_sha256"],
                "paid_attempt_id": row["paid_attempt_id"]}

    def plan(self, spec):
        with self._operation():
            prepared, manifest, config, rows, requests = self._prepare(spec)
            cached = {digest: self._cache(digest, config.dimensions) is not None for digest in requests}
            allocation = sum(value["allocation"] for digest, value in requests.items() if not cached[digest])
            return {"model": config.model, "dimensions": config.dimensions, "space_id": config.space_id,
                    "release_id": manifest.release_id, "embedding_count": len(rows),
                    "unique_request_count": len(requests), "cache_hits": sum(cached.values()),
                    "reservation_ceiling_microdollars": allocation,
                    "reservation_ceiling_usd": f"{allocation / 1_000_000:.6f}",
                    "price_checked": PRICE_CHECKED, "price_valid_through": PRICE_VALID_THROUGH.isoformat(),
                    "dispatch": False, "reserved": False,
                    "preflight": [value["preflight"] for value in requests.values()]}

    def enqueue(self, spec, episode_limit_usd: str, series_limit_usd: str):
        with self._operation():
            prepared, manifest, config, rows, requests = self._prepare(spec)
            cached = {digest: self._cache(digest, config.dimensions) for digest in requests}
            episode = "embedding:" + manifest.release_id
            series = "embeddings"
            self.budget.configure(episode, series, dollars_to_microdollars(episode_limit_usd),
                                  dollars_to_microdollars(series_limit_usd))
            identifier = uuid4().hex
            directory = self._job_dir(identifier)
            stamp = _now()
            with self.budget.transaction() as db:
                for index, (digest, value) in enumerate(requests.items()):
                    attempt = None if cached[digest] else "embedding:" + identifier + ":" + str(index)
                    if attempt:
                        self.budget.reserve(attempt, episode, series, value["allocation"], connection=db)
                    _write(directory / "requests" / (digest + ".json"), value["bytes"])
                    hit = cached[digest]
                    db.execute("""INSERT INTO embedding_requests
                        (job_id,request_sha256,ordinal,modality,state,attempt_id,allocation_micro,
                        response_sha256,paid_attempt_id) VALUES(?,?,?,?,?,?,?,?,?)""",
                        (identifier, digest, index, value["modality"], "cached" if hit else "pending",
                         attempt, value["allocation"], hit["response_sha256"] if hit else None,
                         hit["paid_attempt_id"] if hit else None))
                db.execute("""INSERT INTO embedding_jobs
                    (id,status,created_at,updated_at,spec_json,manifest_json,rows_json,config_json)
                    VALUES(?,'queued',?,?,?,?,?,?)""",
                    (identifier, stamp, stamp, _encode(prepared.model_dump()).decode(),
                     _encode(manifest.model_dump()).decode(), _encode(rows).decode(),
                     _encode(config.model_dump()).decode()))
            _write(directory / "spec.json", _encode(prepared.model_dump()))
            return self.status(identifier)

    def status(self, job_id):
        self._job_dir(job_id)
        with self._operation(), self.budget.connect() as db:
            row = db.execute("SELECT * FROM embedding_jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise EmbeddingError("Unknown embedding job.")
            result = dict(row)
            for key in ("spec", "manifest", "rows", "config"):
                result[key] = json.loads(result.pop(key + "_json"))
            result["requests"] = [dict(item) for item in db.execute(
                "SELECT * FROM embedding_requests WHERE job_id=? ORDER BY ordinal", (job_id,))]
            result["budget"] = self.budget.summary("embedding:" + result["manifest"]["release_id"], "embeddings")
            return result

    def _failure(self, job_id, status, error):
        # Only pending rows have never passed the dispatch boundary.
        with self.budget.transaction() as db:
            pending = db.execute("SELECT * FROM embedding_requests WHERE job_id=? AND state='pending'", (job_id,)).fetchall()
            for request in pending:
                if request["attempt_id"]:
                    db.execute("UPDATE reservations SET state='released',accounting='never-dispatched' "
                               "WHERE attempt_id=? AND state='reserved'", (request["attempt_id"],))
            db.execute("UPDATE embedding_requests SET state='cancelled' WHERE job_id=? AND state='pending'", (job_id,))
            db.execute("UPDATE embedding_requests SET state='uncertain',error=? WHERE job_id=? AND state='running'",
                       ("Dispatched outcome is unresolved; allocation retained.", job_id))
            db.execute("UPDATE embedding_jobs SET status=?,error=?,updated_at=? WHERE id=?",
                       (status, error, _now(), job_id))

    def _complete_step(self, job, request, raw):
        from .embedding_provider import parse_response
        digest = request["request_sha256"]
        if not request["response_sha256"] or _hash(raw) != request["response_sha256"]:
            raise EmbeddingError("Retained provider response failed authoritative verification.")
        vector, metadata = parse_response(json.loads(raw), job["config"]["dimensions"])
        directory = self.state / "cache" / digest
        payload = _read(self._job_dir(job["id"]) / "requests" / (digest + ".json"), MAX_PLAN_BYTES)
        if _hash(payload) != digest:
            raise EmbeddingError("Frozen provider request failed authoritative verification.")
        _write(directory / "request.json", payload)
        _write(directory / "response.json", raw)
        with self.budget.transaction() as db:
            existing = db.execute("SELECT * FROM embedding_cache WHERE request_sha256=?", (digest,)).fetchone()
            if existing and existing["response_sha256"] != request["response_sha256"]:
                raise EmbeddingError("A conflicting cached provider result is preserved.")
            reservation = db.execute("SELECT * FROM reservations WHERE attempt_id=?", (request["attempt_id"],)).fetchone()
            if reservation is None:
                raise EmbeddingError("Dispatched work has no accounting record.")
            if reservation["state"] == "reserved":
                self.budget.settle(request["attempt_id"], request["allocation_micro"],
                                   "embedding-serving-ceiling", connection=db)
            elif reservation["state"] != "spent":
                raise EmbeddingError("Dispatched allocation is unavailable.")
            db.execute("INSERT OR IGNORE INTO embedding_cache VALUES(?,?,?,?)",
                       (digest, request["response_sha256"], request["attempt_id"], _now()))
            db.execute("""UPDATE embedding_requests SET state='completed',paid_attempt_id=?,error=NULL
                       WHERE job_id=? AND request_sha256=?""", (request["attempt_id"], job["id"], digest))
        return vector

    def _finish(self, job_id):
        from .embedding_contract import write_bundle, EmbeddingConfig
        job = self.status(job_id)
        if any(item["state"] not in ("completed", "cached") for item in job["requests"]):
            return False
        spec = PreparationSpec.model_validate(job["spec"])
        manifest, _ = self._accepted(spec)
        by_request = {item["request_sha256"]: item for item in job["requests"]}
        rows, vectors, revisions = [], [], set()
        for row in job["rows"]:
            request = by_request[row["request_sha256"]]
            cached = self._cache(row["request_sha256"], job["config"]["dimensions"])
            if cached is None or cached["response_sha256"] != request["response_sha256"]:
                raise EmbeddingError("Completed request is missing verified cached output.")
            rows.append(row | {"response_sha256": cached["response_sha256"]})
            vectors.append(cached["vector"])
            returned = cached["metadata"].get("returned_model", {})
            named = {returned[key] for key in ("modelVersion", "model_version") if key in returned}
            if len(named) > 1:
                raise EmbeddingError("Provider reported conflicting model revisions.")
            revisions.add(next(iter(named)) if named else None)
        if len(revisions) != 1:
            raise EmbeddingError("A bundle cannot mix reported model revisions or known/unknown revisions.")
        config = EmbeddingConfig.model_validate(job["config"] | {"provider_revision": next(iter(revisions))})
        output = self._job_dir(job_id) / "bundle"
        result = write_bundle(output, manifest, config, rows, vectors)
        with self.budget.transaction() as db:
            db.execute("UPDATE embedding_jobs SET status='ready',bundle_path=?,error=NULL,updated_at=? WHERE id=?",
                       (str(result.relative_to(self.root)), _now(), job_id))
        return True

    def run_one(self, job_id, transport=None):
        from . import embedding_provider as provider
        self._job_dir(job_id)
        with self._worker():
            job = self.status(job_id)
            if job["status"] != "queued":
                return job
            try:
                self._accepted(PreparationSpec.model_validate(job["spec"]))
                with self.budget.transaction() as db:
                    db.execute("UPDATE embedding_jobs SET status='running',updated_at=? WHERE id=?", (_now(), job_id))
                for request in job["requests"]:
                    if request["state"] == "cached":
                        self._cache(request["request_sha256"], job["config"]["dimensions"])
                        continue
                    digest = request["request_sha256"]
                    cached = self._cache(digest, job["config"]["dimensions"])
                    if cached:
                        with self.budget.transaction() as db:
                            db.execute("UPDATE reservations SET state='released',accounting='cached-before-dispatch' "
                                       "WHERE attempt_id=? AND state='reserved'", (request["attempt_id"],))
                            db.execute("""UPDATE embedding_requests SET state='cached',response_sha256=?,paid_attempt_id=?
                                       WHERE job_id=? AND request_sha256=?""",
                                       (cached["response_sha256"], cached["paid_attempt_id"], job_id, digest))
                        continue
                    with self.budget.connect() as db:
                        unresolved = db.execute("""SELECT 1 FROM embedding_requests
                            WHERE request_sha256=? AND job_id<>? AND state IN ('running','uncertain') LIMIT 1""",
                            (digest, job_id)).fetchone()
                    if unresolved:
                        raise EmbeddingError("Equivalent dispatched work is unresolved; reconcile it before a new attempt.")
                    payload = _read(self._job_dir(job_id) / "requests" / (digest + ".json"), MAX_PLAN_BYTES)
                    if _hash(payload) != digest:
                        raise EmbeddingError("Frozen request failed authoritative verification.")
                    body = json.loads(payload)
                    provider.validate_request(body)
                    if _allocation(request["modality"]) > request["allocation_micro"]:
                        raise BudgetError("Frozen allocation no longer covers current rates.")
                    key = self.key_provider()
                    if not key:
                        raise EmbeddingError("No private Gemini credential is configured.")
                    with self.budget.transaction() as db:
                        db.execute("UPDATE embedding_requests SET state='running' WHERE job_id=? AND request_sha256=?",
                                   (job_id, digest))
                    _, metadata = provider.embed(body, key, transport=transport)
                    del key
                    raw = _encode(metadata["response"])
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise EmbeddingError("Provider response exceeds the retained response limit.")
                    response_digest = _hash(raw)
                    with self.budget.transaction() as db:
                        db.execute("UPDATE embedding_requests SET response_sha256=? WHERE job_id=? AND request_sha256=?",
                                   (response_digest, job_id, digest))
                    _write(self._job_dir(job_id) / "responses" / (digest + ".json"), raw)
                    request = dict(request) | {"response_sha256": response_digest}
                    self._complete_step(job, request, raw)
                self._finish(job_id)
            except Exception:
                current = self.status(job_id)
                ambiguous = any(item["state"] in ("running", "uncertain") for item in current["requests"])
                self._failure(job_id, "uncertain" if ambiguous else "failed",
                              "Preparation did not complete. Retained inputs/results and accounting are preserved; "
                              "recover locally or reconcile unresolved dispatches before a new job.")
            return self.status(job_id)

    def recover(self, job_id):
        self._job_dir(job_id)
        with self._worker():
            job = self.status(job_id)
            if job["status"] in ("queued", "ready"):
                return job
            for request in job["requests"]:
                if request["state"] not in ("running", "uncertain", "reconciled"):
                    continue
                path = self._job_dir(job_id) / "responses" / (request["request_sha256"] + ".json")
                try:
                    raw = _read(path, MAX_RESPONSE_BYTES)
                    self._complete_step(job, request, raw)
                except (OSError, ValueError, TypeError, KeyError):
                    continue
            current = self.status(job_id)
            if any(item["state"] not in ("completed", "cached") for item in current["requests"]):
                uncertain = any(item["state"] in ("running", "uncertain") for item in current["requests"])
                self._failure(job_id, "uncertain" if uncertain else "failed",
                              "Recovery makes no provider calls; unresolved dispatched work needs billing evidence.")
            else:
                try:
                    self._finish(job_id)
                except (OSError, ValueError, TypeError, KeyError):
                    self._failure(job_id, "failed", "Verified results are retained; bundle finalization needs inspection.")
            return self.status(job_id)

    def reconcile(self, job_id, request_sha256, charged_usd, evidence):
        self._job_dir(job_id)
        if not isinstance(request_sha256, str) or not re.fullmatch(HASH_PATTERN, request_sha256):
            raise EmbeddingError("Choose one exact request digest.")
        if not isinstance(evidence, str) or not 12 <= len(evidence) <= 2000:
            raise EmbeddingError("Verified billing evidence is required.")
        try:
            amount = Decimal(charged_usd) * 1_000_000
            if not amount.is_finite() or amount < 0 or amount != amount.to_integral_value():
                raise ValueError
            charged = int(amount)
        except (TypeError, InvalidOperation, ValueError):
            raise EmbeddingError("Use a nonnegative USD amount with at most six decimal places.") from None
        with self._worker(), self.budget.transaction() as db:
            request = db.execute("SELECT * FROM embedding_requests WHERE job_id=? AND request_sha256=?",
                                 (job_id, request_sha256)).fetchone()
            if request is None or request["state"] not in ("uncertain", "completed", "reconciled"):
                raise EmbeddingError("Only dispatched, unresolved or completed work can be reconciled.")
            reservation = db.execute("SELECT * FROM reservations WHERE attempt_id=?", (request["attempt_id"],)).fetchone()
            if reservation is None or charged > reservation["reserved_micro"]:
                raise BudgetError("Reconciled billing exceeds this request allocation.")
            if reservation["state"] == "reserved":
                self.budget.settle(request["attempt_id"], charged, "manual-embedding-billing", connection=db)
            elif reservation["state"] == "spent":
                db.execute("UPDATE reservations SET charged_micro=?,accounting='manual-embedding-billing' WHERE attempt_id=?",
                           (charged, request["attempt_id"]))
            else:
                raise BudgetError("No dispatched allocation can be reconciled.")
            db.execute("INSERT INTO embedding_reconciliations(job_id,request_sha256,charged_micro,evidence,recorded_at) "
                       "VALUES(?,?,?,?,?)", (job_id, request_sha256, charged, evidence, _now()))
            if request["state"] != "completed":
                db.execute("UPDATE embedding_requests SET state='reconciled' WHERE job_id=? AND request_sha256=?",
                           (job_id, request_sha256))
        return self.status(job_id)

    def install(self, job_id):
        from .embedding_contract import install_bundle
        with self._worker():
            job = self.status(job_id)
            if job["status"] != "ready" or not job["bundle_path"]:
                raise EmbeddingError("Only a completed prepared bundle can be installed.")
            result = install_bundle(self.root / job["bundle_path"], self.root / ".runtime")
            return {"job_id": job_id, "installed_manifest": str(result.relative_to(self.root)), "provider_calls": 0}
