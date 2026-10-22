"""Backend production services and explicit internal API; default app is listener-only."""
from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import threading
import time
import uuid
import wave
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import unquote
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_in_threadpool

from . import gemini
from .budget import (
    Budget, BudgetError, PRICE_CHECKED, PRICE_SOURCE, PRICE_VALID_THROUGH,
    RATES, dollars_to_microdollars, reservation_cost, usage_cost,
)
from .gemini import Script, StrictModel
from .rundown import CATALOG_LIMIT, CATALOG_ENTRIES, ReleaseRef


MAX_IMPORT_BYTES = 64 * 1024 * 1024
MAX_IMPORT_SECONDS = 600
UPLOAD_TIMEOUT_SECONDS = 20


def read_catalog(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "releases": []}
    if path.is_symlink() or not path.is_file():
        raise ValueError("Publication catalogue is not an ordinary local file; preserved.")
    with path.open("rb") as stream:
        raw = stream.read(CATALOG_LIMIT + 1)
    if len(raw) > CATALOG_LIMIT:
        raise ValueError("Publication catalogue exceeds its size limit; preserved.")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Publication catalogue repeats a JSON key; preserved.")
            result[key] = value
        return result
    catalog = json.loads(raw, object_pairs_hook=pairs)
    if (not isinstance(catalog, dict) or set(catalog) != {"schema_version", "releases"}
            or type(catalog["schema_version"]) is not int or catalog["schema_version"] != 1
            or not isinstance(catalog["releases"], list) or len(catalog["releases"]) > CATALOG_ENTRIES):
        raise ValueError("Publication catalogue is invalid or exceeds its entry limit; preserved.")
    seen = set()
    for entry in catalog["releases"]:
        ref = ReleaseRef.model_validate(entry)
        if ref.release_id in seen:
            raise ValueError("Publication catalogue repeats a release; preserved.")
        seen.add(ref.release_id)
    return catalog


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


class GenerationRequest(StrictModel):
    kind: Literal["programme", "ident"] = "programme"
    model: Literal["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"]
    script_revision_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


class BudgetSettings(StrictModel):
    kind: Literal["programme", "ident"] = "programme"
    episode_limit_usd: str = Field(min_length=1, max_length=32)
    series_limit_usd: str = Field(min_length=1, max_length=32)


class Credential(StrictModel):
    key: str = Field(min_length=20, max_length=256)


class Decision(StrictModel):
    decision: Literal["accepted", "rejected"]


class Reconciliation(StrictModel):
    charged_usd: str = Field(pattern=r"^\d+(\.\d{1,6})?$")
    evidence: str = Field(min_length=12, max_length=2000)
    request_index: int | None = Field(default=None, ge=0, le=199)


class RestoreRevision(StrictModel):
    revision_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    expected_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


class ScriptConflict(ValueError):
    pass


class Studio:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.state = self.root / ".runtime" / "studio"
        self.state.mkdir(parents=True, exist_ok=True)
        self.database = self.state / "studio.sqlite3"
        self.budget = Budget(self.database)
        self.key_path = self.root / ".runtime" / "secrets" / "gemini-api-key"
        self.stop_event = threading.Event()
        self.thread = None
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS takes (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, programme_id TEXT NOT NULL,
                    title TEXT NOT NULL, model TEXT NOT NULL, status TEXT NOT NULL,
                    decision TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    request_json TEXT NOT NULL, script_json TEXT NOT NULL,
                    reservation_microdollars INTEGER NOT NULL,
                    duration_ms INTEGER, error TEXT, metadata_json TEXT,
                    release_id TEXT
                );
                CREATE TABLE IF NOT EXISTS generation_requests (
                    take_id TEXT NOT NULL, request_index INTEGER NOT NULL,
                    attempt_id TEXT NOT NULL UNIQUE, state TEXT NOT NULL,
                    request_json TEXT NOT NULL, metadata_json TEXT,
                    charged_microdollars INTEGER, error TEXT,
                    PRIMARY KEY(take_id, request_index)
                );
                CREATE TABLE IF NOT EXISTS script_revisions (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, revision INTEGER NOT NULL,
                    script_sha256 TEXT NOT NULL, script_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, restored_from TEXT, UNIQUE(kind,revision)
                );
                CREATE TABLE IF NOT EXISTS publications (
                    take_id TEXT PRIMARY KEY, release_id TEXT NOT NULL UNIQUE,
                    package_sha256 TEXT NOT NULL, recorded_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, take_id TEXT NOT NULL,
                    decision TEXT NOT NULL, recorded_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS voice_recipes (
                    recipe_hash TEXT PRIMARY KEY, name TEXT NOT NULL,
                    version INTEGER NOT NULL, recipe_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, UNIQUE(name,version)
                );
                CREATE TABLE IF NOT EXISTS reconciliations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, take_id TEXT NOT NULL,
                    charged_microdollars INTEGER NOT NULL, evidence TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                );
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(takes)")}
            for name, definition in (("source", "TEXT NOT NULL DEFAULT 'gemini'"),
                                     ("script_revision_id", "TEXT"), ("plan_json", "TEXT"),
                                     ("cast_snapshot_json", "TEXT")):
                if name not in columns:
                    db.execute(f"ALTER TABLE takes ADD COLUMN {name} {definition}")
        self.script_lock = threading.RLock()
        for kind in ("programme", "ident"):
            with self.connect() as db:
                exists = db.execute("SELECT 1 FROM script_revisions WHERE kind=? LIMIT 1", (kind,)).fetchone()
            if not exists:
                self.save_script(self.read_script(kind))

    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def read_script(self, kind: str) -> Script:
        if kind not in {"programme", "ident"}:
            raise ValueError("Unknown script kind.")
        with self.connect() as db:
            head = db.execute("SELECT script_json FROM script_revisions WHERE kind=? ORDER BY revision DESC LIMIT 1", (kind,)).fetchone()
        if head:
            return Script.model_validate_json(head["script_json"])
        saved = self.state / "scripts" / f"{kind}.json"
        source = saved if saved.exists() else self.root / "content" / "series" / "first-light" / f"{kind}.json"
        return Script.model_validate_json(source.read_text(encoding="utf-8"))

    def save_script(self, script: Script, restored_from: str | None = None, expected_revision: str | None = None) -> dict:
        # The recipe's content identifies it; a changed direction creates a new,
        # immutable revision even when the editor submits an old version number.
        with self.script_lock, self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT id FROM script_revisions WHERE kind=? ORDER BY revision DESC LIMIT 1", (script.kind,)).fetchone()
            if expected_revision is not None and (current is None or current["id"] != expected_revision):
                raise ScriptConflict("A newer script was saved elsewhere. Your draft has been kept; review history before replacing it.")
            for recipe in script.speakers:
                details = recipe.model_dump()
                details.pop("recipe_version")
                fingerprint = hashlib.sha256(
                    json.dumps(details, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                known = db.execute("SELECT version FROM voice_recipes WHERE recipe_hash=?", (fingerprint,)).fetchone()
                if known:
                    recipe.recipe_version = known["version"]
                else:
                    last = db.execute("SELECT COALESCE(MAX(version),0) last FROM voice_recipes WHERE name=?", (recipe.name,)).fetchone()
                    recipe.recipe_version = last["last"] + 1
                    snapshot = recipe.model_dump()
                    db.execute(
                        "INSERT INTO voice_recipes(recipe_hash,name,version,recipe_json,created_at) VALUES(?,?,?,?,?)",
                        (fingerprint, recipe.name, recipe.recipe_version, json.dumps(snapshot,ensure_ascii=False), timestamp()),
                    )
                    recipe_path = self.state / "voice-recipes" / f"{fingerprint}.json"
                    if not recipe_path.exists():
                        write_json(recipe_path, snapshot)
            encoded = script.model_dump_json()
            fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
            head = db.execute("SELECT * FROM script_revisions WHERE kind=? ORDER BY revision DESC LIMIT 1", (script.kind,)).fetchone()
            if head and head["script_sha256"] == fingerprint and restored_from is None:
                db.commit()
                return self.revision(script.kind, head["id"])
            identifier = uuid.uuid4().hex
            number = head["revision"] + 1 if head else 1
            now = timestamp()
            db.execute(
                "INSERT INTO script_revisions(id,kind,revision,script_sha256,script_json,created_at,restored_from) VALUES(?,?,?,?,?,?,?)",
                (identifier, script.kind, number, fingerprint, encoded, now, restored_from),
            )
            db.commit()
        snapshot = {"id": identifier, "kind": script.kind, "revision": number, "script_sha256": fingerprint,
                    "created_at": now, "restored_from": restored_from, "script": script.model_dump()}
        write_json(self.state / "script-revisions" / script.kind / f"{number:06d}-{identifier}.json", snapshot)
        write_json(self.state / "scripts" / f"{script.kind}.json", script.model_dump())
        return snapshot

    def revisions(self, kind: str) -> list[dict]:
        if kind not in {"programme", "ident"}:
            raise ValueError("Unknown script kind.")
        with self.connect() as db:
            rows = db.execute("SELECT id,kind,revision,script_sha256,created_at,restored_from FROM script_revisions WHERE kind=? ORDER BY revision DESC", (kind,)).fetchall()
        return [dict(row) for row in rows]

    def revision(self, kind: str, identifier: str) -> dict:
        if kind not in {"programme", "ident"} or not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise ValueError("Unknown script revision.")
        with self.connect() as db:
            row = db.execute("SELECT * FROM script_revisions WHERE kind=? AND id=?", (kind, identifier)).fetchone()
        if row is None:
            raise ValueError("Unknown script revision.")
        result = dict(row)
        result["script"] = json.loads(result.pop("script_json"))
        return result

    def head_revision(self, kind: str) -> dict:
        if kind not in {"programme", "ident"}:
            raise ValueError("Unknown script kind.")
        with self.connect() as db:
            row = db.execute("SELECT * FROM script_revisions WHERE kind=? ORDER BY revision DESC LIMIT 1", (kind,)).fetchone()
        if row is None:
            raise ValueError("This script has no saved revision.")
        result = dict(row)
        result["script"] = json.loads(result.pop("script_json"))
        return result

    def restore(self, kind: str, identifier: str, expected_revision: str | None = None) -> dict:
        chosen = self.revision(kind, identifier)
        return self.save_script(Script.model_validate(chosen["script"]), restored_from=identifier, expected_revision=expected_revision)

    def key(self) -> str | None:
        value = self.key_path.read_text(encoding="utf-8").strip() if self.key_path.is_file() else os.environ.get("GEMINI_API_KEY", "").strip()
        return value or None

    def save_key(self, key: str):
        key = key.strip()
        if not 20 <= len(key) <= 256 or any(c.isspace() for c in key):
            raise ValueError("Enter a single Gemini API key without whitespace.")
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.key_path.with_name("gemini-api-key.tmp-" + uuid.uuid4().hex)
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(key)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, self.key_path)

    def configure(self, settings: BudgetSettings):
        script = self.read_script(settings.kind)
        self.budget.configure(
            script.programme_id, script.series_id,
            dollars_to_microdollars(settings.episode_limit_usd),
            dollars_to_microdollars(settings.series_limit_usd),
        )

    def status(self) -> dict:
        budgets = {}
        for kind in ("programme", "ident"):
            script = self.read_script(kind)
            budgets[kind] = self.budget.summary(script.programme_id, script.series_id)
        return {
            "ok": True, "service": "studio", "key_configured": self.key() is not None,
            "editing_ready": True, "import_ready": True,
            "import_limits": {"max_bytes": MAX_IMPORT_BYTES, "max_seconds": MAX_IMPORT_SECONDS, "upload_seconds": UPLOAD_TIMEOUT_SECONDS},
            "budgets": budgets,
            "models": list(RATES),
            "request_allocation_microdollars": {model: reservation_cost(model) for model in RATES}
                if datetime.now(timezone.utc).date() <= PRICE_VALID_THROUGH else {},
            "price_checked": PRICE_CHECKED, "price_valid_through": str(PRICE_VALID_THROUGH),
            "price_source": PRICE_SOURCE, "price_expired": datetime.now(timezone.utc).date() > PRICE_VALID_THROUGH,
            "worker": "running" if self.thread and self.thread.is_alive() else "stopped",
        }

    def take(self, take_id: str) -> dict:
        if not re.fullmatch(r"[a-f0-9]{32}", take_id):
            raise ValueError("Unknown take.")
        with self.connect() as db:
            row = db.execute("SELECT * FROM takes WHERE id=?", (take_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown take.")
        result = dict(row)
        revision_id = result.get("script_revision_id")
        with self.connect() as db:
            revision = db.execute("SELECT revision FROM script_revisions WHERE id=?", (revision_id,)).fetchone() if revision_id else None
        result["script_revision"] = revision["revision"] if revision else None
        for name in ("request_json", "script_json", "metadata_json", "plan_json", "cast_snapshot_json"):
            raw = result.pop(name)
            result[name.removesuffix("_json")] = json.loads(raw) if raw else None
        with self.connect() as db:
            steps = list(db.execute(
                "SELECT request_index,attempt_id,state,charged_microdollars,error FROM generation_requests "
                "WHERE take_id=? ORDER BY request_index", (take_id,)))
        result["requests"] = [dict(step) for step in steps]
        return result

    def takes(self) -> list[dict]:
        with self.connect() as db:
            identifiers = [row["id"] for row in db.execute("SELECT id FROM takes ORDER BY created_at DESC")]
        return [self.take(identifier) for identifier in identifiers]

    def enqueue(self, request: GenerationRequest) -> dict:
        if self.key() is None:
            raise ValueError("Configure the private Gemini credential before generating.")
        if request.script_revision_id:
            revision = self.revision(request.kind, request.script_revision_id)
        else:
            revision = self.head_revision(request.kind)
        script = Script.model_validate(revision["script"])
        from .characters import CharacterStore
        characters = CharacterStore(self.root)
        script, cast = characters.bind_cast(script)
        plan = gemini.build_plan(script, request.model,
                                 approved_designed_voices=characters.approved_designed_voices())
        per_request = reservation_cost(request.model, datetime.now(timezone.utc).date())
        total = per_request * len(plan["requests"])
        identifier = uuid.uuid4().hex
        now = timestamp()
        take_path = self.state / "takes" / identifier
        payload = plan["requests"][0]["request"] if len(plan["requests"]) == 1 else plan
        # Exact input/provenance snapshots contain no request headers or secrets.
        write_json(take_path / "request.json", payload)
        write_json(take_path / "plan.json", plan)
        write_json(take_path / "script.json", script.model_dump())
        write_json(take_path / "cast.json", cast)
        with self.budget.transaction() as db:
            for index, item in enumerate(plan["requests"]):
                attempt_id = identifier if index == 0 else f"{identifier}:{index}"
                self.budget.reserve(attempt_id, script.programme_id, script.series_id, per_request, connection=db)
                db.execute(
                    "INSERT INTO generation_requests(take_id,request_index,attempt_id,state,request_json) "
                    "VALUES(?,?,?,'pending',?)",
                    (identifier, index, attempt_id, json.dumps(item["request"], ensure_ascii=False)),
                )
            db.execute(
                """INSERT INTO takes(id,kind,programme_id,title,model,status,created_at,updated_at,
                request_json,script_json,reservation_microdollars,source,script_revision_id,plan_json,cast_snapshot_json)
                VALUES(?,?,?,?,?,'queued',?,?,?,?,?,'gemini',?,?,?)""",
                (identifier, script.kind, script.programme_id, script.title, request.model, now, now,
                 json.dumps(payload, ensure_ascii=False), script.model_dump_json(), total, revision["id"],
                 json.dumps(plan, ensure_ascii=False), json.dumps(cast, ensure_ascii=False)),
            )
        return self.take(identifier)

    def begin_import(self, kind: str, filename: str, script_revision_id: str | None = None) -> dict:
        if script_revision_id:
            revision = self.revision(kind, script_revision_id)
            script = Script.model_validate(revision["script"])
        else:
            revision = self.head_revision(kind)
            script = Script.model_validate(revision["script"])
        identifier = uuid.uuid4().hex
        now = timestamp()
        basename = PurePosixPath(unquote(filename).replace("\\", "/")).name
        safe_name = re.sub(r"[^A-Za-z0-9 ._-]", "_", basename)[:160].strip(" .") or "recording.wav"
        provenance = {"operation": "local-recording-import", "filename": safe_name,
                      "script_revision_id": revision["id"], "source": "imported"}
        directory = self.state / "takes" / identifier
        write_json(directory / "script.json", script.model_dump())
        write_json(directory / "request.json", provenance)
        with self.connect() as db:
            db.execute(
                """INSERT INTO takes(id,kind,programme_id,title,model,status,created_at,updated_at,
                request_json,script_json,reservation_microdollars,source,script_revision_id)
                VALUES(?,?,?,?,?,'importing',?,?,?,?,0,'imported',?)""",
                (identifier, script.kind, script.programme_id, script.title, "local-recording", now, now,
                 json.dumps(provenance), script.model_dump_json(), revision["id"]),
            )
        return self.take(identifier)

    def fail_import(self, identifier: str, error: str):
        with self.connect() as db:
            db.execute(
                "UPDATE takes SET status='failed',error=?,updated_at=? WHERE id=? AND source='imported' AND status='importing'",
                (error, timestamp(), identifier),
            )

    def finish_import(self, identifier: str) -> dict:
        take = self.take(identifier)
        if take["source"] != "imported" or take["status"] != "importing":
            raise ValueError("This recording import is no longer pending.")
        directory = self.state / "takes" / identifier
        original = directory / "original.wav"
        try:
            size = original.stat().st_size
            if not 44 <= size <= MAX_IMPORT_BYTES:
                raise ValueError("Choose a WAV recording no larger than 64 MiB.")
            digest = hashlib.sha256()
            with original.open("rb") as recording:
                for chunk in iter(lambda: recording.read(1024 * 1024), b""):
                    digest.update(chunk)
            # Reject compressed/container tricks and truncated frame data before
            # invoking a decoder. The original file is never modified.
            with wave.open(str(original), "rb") as recording:
                if recording.getcomptype() != "NONE" or recording.getnchannels() not in {1, 2}:
                    raise ValueError("Import an uncompressed mono or stereo PCM WAV recording.")
                if recording.getsampwidth() not in {1, 2, 3, 4} or not 8000 <= recording.getframerate() <= 192000:
                    raise ValueError("The WAV sample format or sample rate is unsupported.")
                frames = recording.getnframes()
                duration = frames / recording.getframerate()
                if not 0.2 <= duration <= MAX_IMPORT_SECONDS:
                    raise ValueError("Recordings must be between 0.2 seconds and 10 minutes.")
                expected_bytes = frames * recording.getsampwidth() * recording.getnchannels()
                decoded_bytes = 0
                while chunk := recording.readframes(65536):
                    decoded_bytes += len(chunk)
                if decoded_bytes != expected_bytes:
                    raise ValueError("The WAV recording is truncated or has inconsistent frame data.")
            metadata = {**take["request"], "original_sha256": digest.hexdigest(),
                        "original_bytes": size, "input_duration_ms": round(duration * 1000)}
            write_json(directory / "import.json", metadata)
            result = subprocess.run(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                 "-protocol_whitelist", "file,pipe", "-f", "wav", "-i", str(original),
                 "-map", "0:a:0", "-vn", "-sn", "-dn", "-threads", "1",
                 "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(directory / "mix.wav")],
                capture_output=True, timeout=60, check=False,
            )
            if result.returncode:
                raise ValueError("The recording could not be fully decoded. Its original bytes have been retained.")
            with wave.open(str(directory / "mix.wav"), "rb") as recording:
                measured = recording.getnframes() / recording.getframerate()
                if recording.getsampwidth() != 2 or recording.getnchannels() != 2 or recording.getframerate() != 48000:
                    raise ValueError("The imported mix does not match the supported playback format.")
                output_frames = recording.getnframes()
                output_bytes = 0
                while chunk := recording.readframes(65536):
                    output_bytes += len(chunk)
                if output_bytes != output_frames * 4 or abs(measured - duration) > 0.1:
                    raise ValueError("The normalized recording is incomplete.")
            metadata["duration_ms"] = round(measured * 1000)
            metadata["provider"] = None
            with self.connect() as db:
                db.execute(
                    "UPDATE takes SET status='ready',duration_ms=?,metadata_json=?,updated_at=? WHERE id=?",
                    (metadata["duration_ms"], json.dumps(metadata), timestamp(), identifier),
                )
        except (ValueError, wave.Error, EOFError, OSError, subprocess.TimeoutExpired) as exc:
            message = str(exc) if isinstance(exc, ValueError) else "The WAV recording is invalid or decoding exceeded the time limit; original bytes retained."
            self.fail_import(identifier, message)
            raise ValueError(message) from exc
        return self.take(identifier)

    @contextmanager
    def _generation_lease(self):
        # Recovery must never mark another active renderer's work interrupted.
        with (self.state / "generation-worker.lock").open("a") as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Another backend generation process is active.") from None
            try:
                yield
            finally:
                fcntl.flock(lease, fcntl.LOCK_UN)

    def _cancel_pending(self, identifier: str):
        with self.budget.transaction() as db:
            pending = list(db.execute(
                "SELECT attempt_id FROM generation_requests WHERE take_id=? AND state='pending'", (identifier,)))
            for step in pending:
                db.execute("UPDATE reservations SET state='released',accounting='never-dispatched' "
                           "WHERE attempt_id=? AND state='reserved'", (step["attempt_id"],))
            db.execute("UPDATE generation_requests SET state='cancelled' WHERE take_id=? AND state='pending'",
                       (identifier,))

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        with self._generation_lease():
            with self.connect() as db:
                interrupted = [row["id"] for row in db.execute("SELECT id FROM takes WHERE status='running'")]
                db.execute(
                    "UPDATE takes SET status='uncertain',error=?,updated_at=? WHERE status='running'",
                    ("Generation stopped during an attempt; reconcile before retrying.", timestamp()))
                db.execute("UPDATE generation_requests SET state='uncertain',error=? WHERE state='running'",
                           ("Generation stopped during dispatch; reservation retained.",))
                db.execute(
                    "UPDATE takes SET status='failed',error=?,updated_at=? WHERE status='importing' AND source='imported'",
                    ("Production stopped during import; original bytes retained.", timestamp()))
            for identifier in interrupted:
                self._cancel_pending(identifier)
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._loop, name="studio-generation-worker", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                worked = self.run_one()
            except ValueError:
                worked = False
            if not worked:
                self.stop_event.wait(0.5)

    def run_one(self, transport=None, *, take_id: str | None = None) -> bool:
        if take_id is not None and not re.fullmatch(r"[a-f0-9]{32}", take_id):
            raise ValueError("Unknown take.")
        with self._generation_lease():
            return self._run_one(transport, take_id=take_id)

    def _run_one(self, transport=None, *, take_id=None) -> bool:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if take_id is None:
                row = db.execute("SELECT id FROM takes WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            else:
                row = db.execute("SELECT id FROM takes WHERE id=? AND status='queued'", (take_id,)).fetchone()
            if row is None:
                db.commit()
                return False
            identifier = row["id"]
            db.execute("UPDATE takes SET status='running',updated_at=? WHERE id=?", (timestamp(), identifier))
            # Migrate a historical queued native request without re-planning it.
            count = db.execute("SELECT COUNT(*) FROM generation_requests WHERE take_id=?", (identifier,)).fetchone()[0]
            if not count:
                frozen = db.execute("SELECT request_json FROM takes WHERE id=?", (identifier,)).fetchone()[0]
                db.execute("INSERT INTO generation_requests(take_id,request_index,attempt_id,state,request_json) "
                           "VALUES(?,0,?,'pending',?)", (identifier, identifier, frozen))
            db.commit()
        take = self.take(identifier)
        directory = self.state / "takes" / identifier
        current_index = None
        completed = []
        try:
            key = self.key()
            if key is None:
                self._cancel_pending(identifier)
                raise ValueError("The Gemini key is missing. No request was dispatched.")
            from .characters import CharacterStore
            frozen_script = Script.model_validate(take["script"])
            verified_script, _ = CharacterStore(self.root).bind_cast(frozen_script)
            if verified_script.model_dump() != frozen_script.model_dump():
                raise ValueError("Frozen character recipe no longer matches its retained script.")
            with self.connect() as db:
                steps = [dict(row) for row in db.execute(
                    "SELECT * FROM generation_requests WHERE take_id=? ORDER BY request_index", (identifier,))]
            for step in steps:
                if self.stop_event.is_set():
                    raise ValueError("Production stopped before remaining requests; completed originals retained.")
                current_index = step["request_index"]
                request_payload = json.loads(step["request_json"])
                # Recheck price validity and the exact retained request before dispatch.
                # A queued job cannot silently cross a pricing expiry/increase.
                current_ceiling = reservation_cost(take["model"], datetime.now(timezone.utc).date())
                with self.connect() as db:
                    reserved = db.execute("SELECT reserved_micro FROM reservations WHERE attempt_id=? AND state='reserved'",
                                          (step["attempt_id"],)).fetchone()
                if reserved is None or reserved[0] < current_ceiling:
                    raise ValueError("Stored allocation cannot cover the current request ceiling; re-plan with explicit budgets.")
                gemini.validate_synthesis_request(request_payload)
                with self.connect() as db:
                    db.execute("UPDATE generation_requests SET state='running' WHERE take_id=? AND request_index=?",
                               (identifier, current_index))
                audio, metadata = gemini.generate(request_payload, key, transport=transport)
                part = directory if len(steps) == 1 else directory / "requests" / f"{current_index:04d}"
                part.mkdir(parents=True, exist_ok=True)
                (part / "raw.wav").write_bytes(audio)
                write_json(part / "response.json", metadata)
                with self.budget.transaction() as db:
                    allocation = db.execute("SELECT reserved_micro FROM reservations WHERE attempt_id=?",
                                            (step["attempt_id"],)).fetchone()[0]
                    charged, accounting = usage_cost(take["model"], metadata.get("usage") or {}, allocation)
                    self.budget.settle(step["attempt_id"], charged, accounting, connection=db)
                    db.execute("UPDATE generation_requests SET state='completed',metadata_json=?,charged_microdollars=? "
                               "WHERE take_id=? AND request_index=?",
                               (json.dumps(metadata), charged, identifier, current_index))
                completed.append((part / "raw.wav", metadata))
            key = None
            if len(completed) > 1:
                self._assemble_dialogue(directory, completed)
                usage_fields = ("total_input_tokens", "total_output_tokens")
                complete_usage = all(
                    isinstance(info.get("usage"), dict)
                    and all(type(info["usage"].get(name)) is int for name in usage_fields)
                    for _, info in completed)
                metadata = {"strategy": (take.get("plan") or {}).get("strategy"),
                            "requests": [info for _, info in completed],
                            "usage": {name: sum(info["usage"][name] for _, info in completed)
                                      for name in usage_fields} if complete_usage else {},
                            "usage_complete": complete_usage}
                write_json(directory / "response.json", metadata)
            else:
                metadata = completed[0][1]
            result = subprocess.run(
                ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                 "-protocol_whitelist", "file,pipe", "-f", "wav", "-i", str(directory / "raw.wav"), "-vn",
                 "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", "48000", "-ac", "2",
                 "-c:a", "pcm_s16le", str(directory / "mix.wav")],
                capture_output=True, timeout=180, check=False)
            if result.returncode != 0:
                raise ValueError("Audio preparation failed; original recordings and requests retained.")
            with wave.open(str(directory / "mix.wav"), "rb") as prepared:
                duration = round(prepared.getnframes() / prepared.getframerate() * 1000)
            if not 0 < duration <= 1_800_000 or (directory / "mix.wav").stat().st_size > 128 * 1024 * 1024:
                raise ValueError("Prepared programme exceeds release duration or size limits; originals retained.")
            with self.connect() as db:
                db.execute("UPDATE takes SET status='ready',duration_ms=?,metadata_json=?,updated_at=? WHERE id=?",
                           (duration, json.dumps(metadata), timestamp(), identifier))
        except gemini.GeminiFailure as exc:
            if current_index is not None:
                with self.connect() as db:
                    db.execute("UPDATE generation_requests SET state=?,error=? WHERE take_id=? AND request_index=?",
                               ("uncertain" if exc.ambiguous else "failed", str(exc), identifier, current_index))
            self._cancel_pending(identifier)
            with self.connect() as db:
                db.execute("UPDATE takes SET status=?,error=?,updated_at=? WHERE id=?",
                           ("uncertain" if exc.ambiguous else "failed", str(exc), timestamp(), identifier))
        except Exception as exc:
            message = str(exc) if isinstance(exc, ValueError) else f"Local production failed ({type(exc).__name__}); inputs retained."
            self._cancel_pending(identifier)
            # If dispatch occurred but local persistence failed, keep that allocation.
            with self.connect() as db:
                db.execute("UPDATE generation_requests SET state='uncertain',error=? "
                           "WHERE take_id=? AND state='running'", (message, identifier))
                db.execute("UPDATE takes SET status='failed',error=?,updated_at=? WHERE id=?",
                           (message, timestamp(), identifier))
        return True

    def _assemble_dialogue(self, directory: Path, completed):
        total_frames = 0
        with wave.open(str(directory / "raw.wav"), "wb") as programme:
            programme.setnchannels(2)
            programme.setsampwidth(2)
            programme.setframerate(48000)
            for index, (original, _) in enumerate(completed):
                prepared = original.with_name("prepared.wav")
                result = subprocess.run(
                    ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                     "-protocol_whitelist", "file,pipe", "-f", "wav", "-i", str(original),
                     "-vn", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(prepared)],
                    capture_output=True, timeout=180, check=False)
                if result.returncode != 0:
                    raise ValueError("Dialogue turn preparation failed; all returned originals retained.")
                with wave.open(str(prepared), "rb") as source:
                    total_frames += source.getnframes()
                    if total_frames > 1800 * 48000 or total_frames * 4 + 44 > 128 * 1024 * 1024:
                        raise ValueError("Dialogue exceeds the release limits; individual turns retained.")
                    while chunk := source.readframes(65536):
                        programme.writeframesraw(chunk)
        write_json(directory / "assembly.json", {
            "schema_version": 1, "method": "ordered-pcm-concatenation",
            "sample_rate": 48000, "channels": 2,
            "sources": [str(path.relative_to(directory)) for path, _ in completed]})

    def matching_release(self, take: dict, package: Path, expected_digest: str | None = None):
        from .contract import validate_package
        script = Script.model_validate_json(take["script_json"])
        manifest = validate_package(package, expected_digest=expected_digest)
        assets = {asset.role: asset for asset in manifest.assets}
        expected_audio = hashlib.sha256((self.state / "takes" / take["id"] / "mix.wav").read_bytes()).hexdigest()
        expected_transcript = hashlib.sha256(script.transcript().encode("utf-8")).hexdigest()
        if (
            manifest.release_id != f"{script.programme_id}-{take['id'][:12]}"
            or manifest.programme_id != script.programme_id or manifest.title != script.title
            or manifest.language != script.language or manifest.kind != script.kind
            or assets["audio"].sha256 != expected_audio or assets["transcript"].sha256 != expected_transcript
        ):
            raise ValueError("Preserved conflicting release; it does not match this take and script.")
        return manifest

    def decide(self, take_id: str, decision: str) -> dict:
        self.take(take_id)
        with self.budget.transaction() as db:
            take = dict(db.execute("SELECT * FROM takes WHERE id=?", (take_id,)).fetchone())
            # Recover a catalogued publication if a process died just before
            # committing its receipt. An already released take cannot be rejected.
            if take["status"] == "ready" and not take["release_id"]:
                release_id = f"{take['programme_id']}-{take_id[:12]}"
                catalog_path = self.root / ".runtime" / "library" / "catalog.json"
                if catalog_path.is_file():
                    catalog = read_catalog(catalog_path)
                    reference = next((entry for entry in catalog.get("releases", []) if entry["release_id"] == release_id), None)
                    if reference:
                        self.matching_release(take, catalog_path.parent / "releases" / release_id, reference["package_sha256"])
                        db.execute("INSERT OR IGNORE INTO publications(take_id,release_id,package_sha256,recorded_at) VALUES(?,?,?,?)",
                                   (take_id, release_id, reference["package_sha256"], timestamp()))
                        db.execute("UPDATE takes SET release_id=?,updated_at=? WHERE id=?", (release_id, timestamp(), take_id))
                        db.commit()
                        raise ValueError("This take was already published; its publication receipt has been recovered.")
            if take["status"] != "ready" or take["release_id"]:
                raise ValueError("Only an unpublished, completed take can be accepted or rejected.")
            db.execute("UPDATE takes SET decision=?,updated_at=? WHERE id=?", (decision, timestamp(), take_id))
            db.execute("INSERT INTO decisions(take_id,decision,recorded_at) VALUES(?,?,?)", (take_id, decision, timestamp()))
        return self.take(take_id)

    def reconcile(self, take_id: str, request: Reconciliation) -> dict:
        from decimal import Decimal
        take = self.take(take_id)
        if take["status"] not in {"uncertain", "failed"}:
            raise ValueError("Only an unresolved failed attempt can be reconciled.")
        with self.connect() as db:
            rows = [dict(row) for row in db.execute(
                "SELECT r.attempt_id,g.request_index FROM reservations r LEFT JOIN generation_requests g "
                "ON r.attempt_id=g.attempt_id WHERE r.state='reserved' AND "
                "(g.take_id=? OR r.attempt_id=?) ORDER BY g.request_index", (take_id, take_id))]
        if request.request_index is not None:
            rows = [row for row in rows if row["request_index"] == request.request_index]
        if len(rows) != 1:
            raise ValueError("Choose one unresolved request index for billing reconciliation.")
        charged = int(Decimal(request.charged_usd) * 1_000_000)
        with self.budget.transaction() as db:
            self.budget.settle(rows[0]["attempt_id"], charged, "manual-billing-reconciliation", connection=db)
            db.execute("INSERT INTO reconciliations(take_id,charged_microdollars,evidence,recorded_at) VALUES(?,?,?,?)",
                       (take_id, charged, request.evidence, timestamp()))
            db.execute("UPDATE generation_requests SET state='reconciled' WHERE attempt_id=?", (rows[0]["attempt_id"],))
        return self.take(take_id)

    def publish(self, take_id: str) -> dict:
        from .contract import create_release, validate_package
        self.take(take_id)
        library = self.root / ".runtime" / "library"
        library.mkdir(parents=True, exist_ok=True)
        with (library / ".catalog.lock").open("a") as lock, self.budget.transaction() as db:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = dict(db.execute("SELECT * FROM takes WHERE id=?", (take_id,)).fetchone())
            if current["status"] != "ready" or current["decision"] != "accepted":
                raise ValueError("Listen to and accept a completed take before publishing.")
            if current["release_id"]:
                return self.take(take_id)
            script = Script.model_validate_json(current["script_json"])
            release_id = f"{script.programme_id}-{take_id[:12]}"
            catalog_path = library / "catalog.json"
            catalog = read_catalog(catalog_path)
            if len(catalog["releases"]) >= CATALOG_ENTRIES and not any(entry["release_id"] == release_id for entry in catalog["releases"]):
                raise ValueError("Publication catalogue is full; preserve existing releases before expanding its supported capacity.")
            package = library / "releases" / release_id
            if package.exists():
                manifest = self.matching_release(current, package)
            else:
                manifest = create_release(
                    self.state / "takes" / take_id / "mix.wav", script.transcript(), package,
                    programme_id=script.programme_id, title=script.title, language=script.language,
                    kind=script.kind, release_id=release_id,
                )
            record = manifest.model_dump(mode="json")
            existing = next((entry for entry in catalog["releases"] if entry["release_id"] == release_id), None)
            reference = {"release_id": release_id, "package_sha256": record["package_sha256"]}
            if existing and existing != reference:
                raise ValueError("Preserved conflicting catalog reference.")
            if existing is None:
                catalog["releases"].append(reference)
                serialized = (json.dumps(catalog, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
                if len(serialized) > CATALOG_LIMIT:
                    raise ValueError("Publication catalogue would exceed its size limit; existing catalogue preserved.")
                write_json(catalog_path, catalog)
            db.execute("INSERT INTO publications(take_id,release_id,package_sha256,recorded_at) VALUES(?,?,?,?)",
                       (take_id, release_id, record["package_sha256"], timestamp()))
            db.execute("UPDATE takes SET release_id=?,updated_at=? WHERE id=?", (release_id, timestamp(), take_id))
        return self.take(take_id)


def create_app(root: Path | None = None) -> FastAPI:
    root = Path(root or os.environ.get("WEB_RADIO_ROOT", "/workspace"))

    @asynccontextmanager
    async def lifespan(application):
        service = Studio(root)
        application.state.studio = service
        application.state.import_slots = asyncio.Semaphore(2)
        service.start()
        try:
            yield
        finally:
            service.stop()

    application = FastAPI(title="Web Radio Studio", lifespan=lifespan)
    application.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])

    @application.middleware("http")
    async def local_origin(request: Request, call_next):
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
            if request.headers.get("origin") != expected or request.headers.get("x-web-radio-request") != "studio":
                return JSONResponse({"detail": "Studio changes require a request from this studio page."}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; style-src-attr 'none'; img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        return response

    from .station_controls import make_router
    application.include_router(make_router(root))

    def studio(request: Request) -> Studio:
        return request.app.state.studio

    @application.exception_handler(ScriptConflict)
    async def script_conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @application.exception_handler(BudgetError)
    async def budget_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @application.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @application.get("/healthz")
    def health(request: Request):
        return {"ok": True, "service": "studio"}

    @application.get("/api/status")
    def status(request: Request):
        return studio(request).status()

    @application.get("/api/script")
    def script(request: Request, kind: Literal["programme", "ident"] = "programme"):
        return studio(request).read_script(kind).model_dump()

    @application.put("/api/script")
    def save_script(body: Script, request: Request):
        revision = studio(request).save_script(body, expected_revision=request.headers.get("x-script-revision"))
        return {"saved": True, "script": body.model_dump(), "revision_id": revision["id"], "revision": revision["revision"]}

    @application.get("/api/scripts/{kind}/head")
    def head(kind: Literal["programme", "ident"], request: Request):
        service = studio(request)
        return service.head_revision(kind)

    @application.get("/api/scripts/{kind}/revisions")
    def revisions(kind: Literal["programme", "ident"], request: Request):
        return {"revisions": studio(request).revisions(kind)}

    @application.get("/api/scripts/{kind}/revisions/{revision_id}")
    def revision(kind: Literal["programme", "ident"], revision_id: str, request: Request):
        return studio(request).revision(kind, revision_id)

    @application.post("/api/scripts/{kind}/restore")
    def restore(kind: Literal["programme", "ident"], body: RestoreRevision, request: Request):
        return studio(request).restore(kind, body.revision_id, body.expected_revision)

    @application.post("/api/import", status_code=201)
    async def import_recording(request: Request, kind: Literal["programme", "ident"] = "programme", script_revision_id: str | None = None):
        if request.headers.get("content-type", "").split(";")[0].lower() not in {"audio/wav", "audio/x-wav", "application/octet-stream"}:
            raise HTTPException(status_code=415, detail="Upload raw PCM WAV audio.")
        supplied_length = request.headers.get("content-length")
        if supplied_length:
            if not supplied_length.isdecimal():
                raise HTTPException(status_code=400, detail="Invalid recording length.")
            if int(supplied_length) > MAX_IMPORT_BYTES:
                raise HTTPException(status_code=413, detail="Recordings must be no larger than 64 MiB.")
        service = studio(request)
        async with request.app.state.import_slots:
            take = service.begin_import(kind, request.headers.get("x-recording-filename", "recording.wav"), script_revision_id)
            identifier = take["id"]
            original = service.state / "takes" / identifier / "original.wav"
            received = 0
            try:
                async with asyncio.timeout(UPLOAD_TIMEOUT_SECONDS):
                    with original.open("xb") as output:
                        async for chunk in request.stream():
                            received += len(chunk)
                            if received > MAX_IMPORT_BYTES:
                                raise HTTPException(status_code=413, detail="Recordings must be no larger than 64 MiB.")
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                if supplied_length and received != int(supplied_length):
                    raise HTTPException(status_code=400, detail="The recording upload was incomplete.")
                return await run_in_threadpool(service.finish_import, identifier)
            except TimeoutError as exc:
                service.fail_import(identifier, "Recording upload exceeded 20 seconds; received bytes retained.")
                raise HTTPException(status_code=408, detail="Recording upload exceeded 20 seconds.") from exc
            except BaseException as exc:
                service.fail_import(identifier, str(exc.detail) if isinstance(exc, HTTPException) else "Recording import failed; received bytes retained.")
                raise

    @application.post("/api/settings")
    def settings(body: BudgetSettings, request: Request):
        studio(request).configure(body)
        return studio(request).status()

    @application.post("/api/credentials")
    def credentials(body: Credential, request: Request):
        studio(request).save_key(body.key)
        return {"key_configured": True}

    @application.post("/api/generate", status_code=202)
    def enqueue(body: GenerationRequest, request: Request):
        return studio(request).enqueue(body)

    @application.get("/api/takes")
    def takes(request: Request):
        return {"takes": studio(request).takes()}

    @application.post("/api/takes/{take_id}/decision")
    def decision(take_id: str, body: Decision, request: Request):
        return studio(request).decide(take_id, body.decision)

    @application.post("/api/takes/{take_id}/publish")
    def publish(take_id: str, request: Request):
        return studio(request).publish(take_id)

    @application.post("/api/takes/{take_id}/reconcile")
    def reconcile(take_id: str, body: Reconciliation, request: Request):
        return studio(request).reconcile(take_id, body)

    @application.get("/api/takes/{take_id}/audio")
    def audio(take_id: str, request: Request):
        service = studio(request)
        take = service.take(take_id)
        if take["status"] != "ready":
            raise HTTPException(status_code=404, detail="This take has no finished mix.")
        return FileResponse(service.state / "takes" / take_id / "mix.wav", media_type="audio/wav")

    return application


# Keep the historical process target for safe managed PID/lifecycle reuse.
# Explicit internal create_app() owns production; published ports never mount it.
from .listener import create_app as create_listener_app

app = create_listener_app()
