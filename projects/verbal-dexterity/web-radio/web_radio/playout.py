"""Independent, local-file playout with a journal tied to emitted PCM time."""
from __future__ import annotations

import json
import hashlib
import fcntl
from contextlib import contextmanager
import logging
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

from .contract import (ContractError, Manifest, HASH_PATTERN, SLUG_PATTERN,
                       import_release, validate_package)

from .rundown import (Rundown, ControlError, parse_rundown, config_fingerprint,
                       CONTROL_LIMIT, CATALOG_LIMIT, CATALOG_ENTRIES)

log = logging.getLogger("web_radio.station")
RATE = 48000
CHANNELS = 2
FRAME_BYTES = CHANNELS * 2
SLUG = re.compile(SLUG_PATTERN)
DIGEST = re.compile(HASH_PATTERN)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_time(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return result.astimezone(timezone.utc)


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


class Journal:
    """SQLite is private to the station; the studio cannot supply journal events."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS sessions (
              id TEXT PRIMARY KEY, origin TEXT NOT NULL, calibrated INTEGER NOT NULL DEFAULT 0,
              emitted_ms INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS airings (
              id TEXT PRIMARY KEY, session_id TEXT NOT NULL, release_id TEXT NOT NULL,
              kind TEXT NOT NULL, title TEXT NOT NULL, start_ms INTEGER NOT NULL,
              end_ms INTEGER, status TEXT NOT NULL, recorded_start TEXT NOT NULL,
              recorded_end TEXT);
            CREATE TABLE IF NOT EXISTS events (
              sequence INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT NOT NULL,
              recorded_at TEXT NOT NULL, airing_id TEXT, session_id TEXT,
              offset_ms INTEGER, release_id TEXT, detail TEXT);
            CREATE TABLE IF NOT EXISTS accepted (
              release_id TEXT PRIMARY KEY, digest TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def recover_interrupted(self):
        # Only call after acquiring the exclusive playout owner lock.
        with self.db() as db:
            # Recover unfinished airings and journal the interruption exactly once.
            unfinished = db.execute(
                "SELECT a.*,s.emitted_ms FROM airings a JOIN sessions s ON s.id=a.session_id "
                "WHERE a.status='playing'").fetchall()
            recovered_at = iso(utcnow())
            for row in unfinished:
                db.execute("UPDATE airings SET status='interrupted',end_ms=?,recorded_end=? WHERE id=?",
                           (row["emitted_ms"], recovered_at, row["id"]))
                db.execute(
                    "INSERT INTO events(event,recorded_at,airing_id,session_id,offset_ms,release_id,detail)"
                    " VALUES('interrupted',?,?,?,?,?,?)",
                    (recovered_at, row["id"], row["session_id"], row["emitted_ms"],
                     row["release_id"], "Previous station process ended before completion."))

    def session(self, session_id: str):
        with self.db() as db:
            db.execute("INSERT INTO sessions(id,origin) VALUES(?,?)", (session_id, iso(utcnow())))

    def calibrate(self, session_id: str, origin: str):
        parse_time(origin)
        with self.db() as db:
            db.execute("UPDATE sessions SET origin=?,calibrated=1 WHERE id=?", (origin, session_id))

    def advance(self, session_id: str, emitted_ms: int):
        with self.db() as db:
            db.execute("UPDATE sessions SET emitted_ms=? WHERE id=?", (emitted_ms, session_id))

    def event(self, event: str, *, airing_id=None, session_id=None, offset_ms=None,
              release_id=None, detail=None):
        with self.db() as db:
            db.execute(
                "INSERT INTO events(event,recorded_at,airing_id,session_id,offset_ms,release_id,detail)"
                " VALUES(?,?,?,?,?,?,?)",
                (event, iso(utcnow()), airing_id, session_id, offset_ms, release_id, detail))

    def begin(self, session_id: str, manifest: Manifest, offset_ms: int) -> str:
        airing_id = uuid.uuid4().hex
        with self.db() as db:
            db.execute("INSERT INTO airings VALUES(?,?,?,?,?,?,NULL,'playing',?,NULL)",
                       (airing_id, session_id, manifest.release_id, manifest.kind,
                        manifest.title, offset_ms, iso(utcnow())))
        self.event("fallback" if manifest.kind == "ident" else "started",
                   airing_id=airing_id, session_id=session_id, offset_ms=offset_ms,
                   release_id=manifest.release_id)
        return airing_id

    def finish(self, airing_id: str, end_ms: int, *, interrupted=False):
        status = "interrupted" if interrupted else "completed"
        with self.db() as db:
            row = db.execute("SELECT * FROM airings WHERE id=?", (airing_id,)).fetchone()
            db.execute("UPDATE airings SET status=?,end_ms=?,recorded_end=? WHERE id=?",
                       (status, end_ms, iso(utcnow()), airing_id))
        if row:
            self.event(status, airing_id=airing_id, session_id=row["session_id"],
                       offset_ms=end_ms, release_id=row["release_id"])

    def remember(self, release_id: str, digest: str):
        with self.db() as db:
            db.execute("INSERT OR REPLACE INTO accepted VALUES(?,?)", (release_id, digest))

    def accepted(self):
        with self.db() as db:
            return [dict(row) for row in db.execute("SELECT * FROM accepted")]

    def save_schedule(self, schedule):
        with self.db() as db:
            db.execute("INSERT OR REPLACE INTO settings VALUES('schedule',?)",
                       (json.dumps(schedule),))

    def schedule(self):
        with self.db() as db:
            row = db.execute("SELECT value FROM settings WHERE key='schedule'").fetchone()
        return json.loads(row["value"]) if row else []

    def control(self):
        with self.db() as db:
            row = db.execute("SELECT value FROM settings WHERE key='station_control'").fetchone()
        return json.loads(row["value"]) if row else None

    def save_control(self, control: Rundown):
        with self.db() as db:
            db.execute("INSERT OR REPLACE INTO settings VALUES('station_control',?)",
                       (json.dumps(control.model_dump(mode="json"), separators=(",", ":")),))
            db.execute("INSERT OR REPLACE INTO settings VALUES('schedule',?)",
                       (json.dumps([ref.release_id for ref in control.programme_releases]),))

    @staticmethod
    def _airing(row):
        if not row:
            return None
        item = dict(row)
        origin = parse_time(item.pop("origin"))
        item["stream_start"] = iso(origin + timedelta(milliseconds=item["start_ms"]))
        item["stream_end"] = (iso(origin + timedelta(milliseconds=item["end_ms"]))
                              if item["end_ms"] is not None else None)
        item["timing_calibrated"] = bool(item.pop("calibrated"))
        item.pop("emitted_ms", None)
        return item

    def now(self, at: datetime | None = None):
        at = at or utcnow()
        with self.db() as db:
            rows = db.execute(
                "SELECT a.*,s.origin,s.calibrated,s.emitted_ms FROM airings a "
                "JOIN sessions s ON s.id=a.session_id ORDER BY a.rowid DESC LIMIT 2000").fetchall()
        for row in rows:
            offset = (at - parse_time(row["origin"])).total_seconds() * 1000
            end = row["end_ms"] if row["end_ms"] is not None else row["emitted_ms"]
            if row["start_ms"] <= offset < end:
                return self._airing(row)
        return None

    def last_completed(self):
        with self.db() as db:
            row = db.execute(
                "SELECT a.*,s.origin,s.calibrated,s.emitted_ms FROM airings a "
                "JOIN sessions s ON s.id=a.session_id "
                "WHERE a.status='completed' AND a.kind='programme' "
                "ORDER BY a.rowid DESC LIMIT 1").fetchone()
        return self._airing(row)

    def events(self, limit=100):
        with self.db() as db:
            rows = db.execute(
                "SELECT e.*,s.origin FROM events e LEFT JOIN sessions s ON s.id=e.session_id "
                "ORDER BY sequence DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            origin = item.pop("origin")
            item["stream_at"] = (iso(parse_time(origin) + timedelta(milliseconds=item["offset_ms"]))
                                 if origin and item["offset_ms"] is not None else None)
            result.append(item)
        return result


class Playout:
    """Stream verified local WAVs through an independently owned FFmpeg process."""

    def __init__(self, runtime: Path):
        self.runtime = Path(runtime)
        self.root = self.runtime / "station"
        self.releases = self.root / "releases"
        self.hls = self.root / "hls"
        self.releases.mkdir(parents=True, exist_ok=True)
        self.hls.mkdir(parents=True, exist_ok=True)
        self.journal = Journal(self.root / "journal.sqlite3")
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self._updates_thread: threading.Thread | None = None
        self.lock = threading.RLock()
        self.known: dict[str, tuple[Manifest, Path]] = {}
        self.schedule_ids: list[str] = []
        self.rejections: dict[str, str] = {}
        self.catalog_fingerprint = None
        self.encoder: subprocess.Popen | None = None
        self.session_id: str | None = None
        self.frames = 0
        self.session_monotonic = 0.0
        self.calibrated = False
        self.position = 0
        self.state = "waiting"
        self.error: str | None = None
        self._owner = None
        self.control: Rundown | None = None
        self._last_control_observed = None
        self._last_control_error = None
        self.playback_kind = None
        self.waiting_reason = "no_released_programmes"
        self.fallback_reason = None
        self.selection_reason = None
        for record in self.journal.accepted():
            try:
                package = self.releases / record["release_id"]
                manifest = validate_package(package, expected_digest=record["digest"])
                self.known[manifest.release_id] = (manifest, package)
            except (OSError, ValueError, ContractError):
                pass
        self.schedule_ids = [rid for rid in self.journal.schedule() if rid in self.known]
        saved_control = self.journal.control()
        if saved_control is not None:
            try:
                self.control = Rundown.model_validate(saved_control)
                self.schedule_ids = [ref.release_id for ref in self.control.programme_releases]
            except ValueError:
                self.error = "Saved station control is invalid; using the last verified rundown."

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        owner = (self.root / "playout-owner.lock").open("a")
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            owner.close()
            raise RuntimeError("A station already owns this runtime; refusing a second playout.")
        self._owner = owner
        try:
            self.journal.recover_interrupted()
            self.stop_event.clear()
            self._updates_thread = threading.Thread(
                target=self._update_loop, name="radio-release-updates", daemon=True)
            self.thread = threading.Thread(target=self._run, name="radio-playout", daemon=True)
            self._updates_thread.start()
            self.thread.start()
        except BaseException:
            self.stop_event.set()
            if self._updates_thread and self._updates_thread.is_alive():
                self._updates_thread.join(timeout=12)
            fcntl.flock(owner, fcntl.LOCK_UN)
            owner.close()
            self._owner = None
            raise

    def stop(self):
        self.stop_event.set()
        if self._updates_thread:
            self._updates_thread.join(timeout=12)
            if self._updates_thread.is_alive():
                raise RuntimeError("Station update worker did not stop; owner lock retained.")
        if self.thread:
            self.thread.join(timeout=12)
            if self.thread.is_alive():
                raise RuntimeError("Playout did not stop; owner lock retained.")
        self._close_encoder()
        if self._owner:
            fcntl.flock(self._owner, fcntl.LOCK_UN)
            self._owner.close()
            self._owner = None

    def reject(self, key: str, reason: str):
        # No credential values, request inputs, or private full paths enter public events.
        clean = reason[:240]
        with self.lock:
            changed = self.rejections.get(key) != clean
            if changed:
                self.rejections[key] = clean
        if changed:
            self.journal.event("rejected", release_id=key if SLUG.fullmatch(key) else None,
                               detail=clean)

    def _update_loop(self):
        while not self.stop_event.is_set():
            try:
                self.refresh()
            except Exception:
                log.exception("Station release/control update failed")
                self.reject("station-updates", "Station release updates are temporarily unavailable.")
            self.stop_event.wait(0.5)

    def refresh(self):
        # Called by the update worker, never the realtime PCM feed.
        self._refresh_catalog()
        self._refresh_control()

    @staticmethod
    def _read_bounded(path: Path, limit: int):
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError("metadata symlinks are not supported")
        with path.open("rb") as handle:
            raw = handle.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("metadata exceeds its size limit")
        return raw

    def _refresh_catalog(self):
        catalog = self.runtime / "library" / "catalog.json"
        try:
            raw = self._read_bounded(catalog, CATALOG_LIMIT)
        except FileNotFoundError:
            return  # Continue the last accepted local rundown if the studio disappears.
        except (OSError, ValueError):
            self.reject("catalogue", "Catalogue is unavailable, unsafe, or larger than 1 MiB.")
            return
        fingerprint = hashlib.sha256(raw).hexdigest()
        if fingerprint == self.catalog_fingerprint:
            return
        try:
            data = json.loads(raw, object_pairs_hook=_unique_object)
            if (not isinstance(data, dict) or set(data) != {"schema_version", "releases"}
                    or type(data["schema_version"]) is not int or data["schema_version"] != 1
                    or not isinstance(data["releases"], list)
                    or len(data["releases"]) > CATALOG_ENTRIES):
                raise ValueError("unsupported catalogue structure")
            entries = []
            seen = set()
            for entry in data["releases"]:
                if (not isinstance(entry, dict) or set(entry) != {"release_id", "package_sha256"}
                        or not isinstance(entry["release_id"], str)
                        or not SLUG.fullmatch(entry["release_id"])
                        or not isinstance(entry["package_sha256"], str)
                        or not DIGEST.fullmatch(entry["package_sha256"])
                        or entry["release_id"] in seen):
                    raise ValueError("invalid or duplicate catalogue entry")
                seen.add(entry["release_id"])
                entries.append(entry)
        except (TypeError, ValueError, KeyError):
            self.reject("catalogue", "Catalogue structure, version, or release count is invalid.")
            return
        approved = []
        for entry in entries:
            rid, digest = entry["release_id"], entry["package_sha256"]
            try:
                with self.lock:
                    pair = self.known.get(rid)
                if pair:
                    manifest, package = pair
                    if manifest.package_sha256 != digest:
                        raise ValueError("a released identity cannot change its digest")
                    validate_package(package, expected_digest=digest)
                else:
                    manifest, package = import_release(
                        self.runtime / "library" / "releases" / rid, self.releases, digest)
                    if manifest.release_id != rid:
                        raise ValueError("release identity disagrees with publication record")
                    with self.lock:
                        self.known[rid] = (manifest, package)
                    self.journal.remember(rid, digest)
                    self.journal.event("accepted", release_id=rid)
                approved.append(rid)
                with self.lock:
                    self.rejections.pop(rid, None)
            except (OSError, ValueError, ContractError):
                self.reject(rid, "Release failed station package validation.")
        with self.lock:
            # Once an operator has chosen a rundown, catalogue changes only add media.
            if self.control is None:
                self.schedule_ids = approved
                self.journal.save_schedule(approved)
            self.catalog_fingerprint = fingerprint

    def _write_control_ack(self, revision, status, reason):
        ack = {"schema_version": 1, "requested_revision": revision,
               "accepted_revision": self.control.revision if self.control else 0,
               "status": status, "reason": reason, "updated_at": iso(utcnow())}
        parent = self.runtime / "library"
        if parent.is_symlink() or not parent.is_dir():
            return False
        temporary = parent / (".station-control-status-" + uuid.uuid4().hex + ".tmp")
        target = parent / "station-control-status.json"
        if target.is_symlink():
            return False
        try:
            with temporary.open("x") as handle:
                json.dump(ack, handle, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            return True
        except OSError:
            log.warning("Cannot write private station-control acknowledgement")
            return False
        finally:
            temporary.unlink(missing_ok=True)

    def _refresh_control(self):
        path = self.runtime / "library" / "station-control.json"
        try:
            raw = self._read_bounded(path, CONTROL_LIMIT)
        except FileNotFoundError:
            return  # Persisted station-owned control remains authoritative offline.
        except (OSError, ValueError):
            # Bounded rejection deduplication also handles files too large to hash.
            observed = ("unreadable", None)
            if self._last_control_observed == observed:
                return
            reason = "Station control is unavailable, unsafe, or larger than 64 KiB."
            if self._write_control_ack(None, "rejected", reason):
                self._last_control_observed = observed
            self.journal.event("control_rejected", detail=reason)
            return
        observed = (hashlib.sha256(raw).hexdigest(), self.catalog_fingerprint)
        if self._last_control_observed == observed:
            return
        revision = None
        try:
            requested = parse_rundown(raw)
            revision = requested.revision
            with self.lock:
                accepted = self.control
            if accepted and revision < accepted.revision:
                raise ControlError("Station-control revision is older than the accepted revision.")
            if accepted and revision == accepted.revision:
                if config_fingerprint(requested) != config_fingerprint(accepted):
                    raise ControlError("Station-control revision conflicts with accepted content.")
                # Identical accepted revision is an idempotent acknowledgement.
            else:
                references = [(ref, "programme") for ref in requested.programme_releases]
                if requested.fallback_release:
                    references.append((requested.fallback_release, "ident"))
                for ref, kind in references:
                    with self.lock:
                        pair = self.known.get(ref.release_id)
                    if pair is None:
                        raise ControlError("A requested release has not been accepted by the station.")
                    manifest, package = pair
                    if manifest.package_sha256 != ref.package_sha256 or manifest.kind != kind:
                        raise ControlError("A requested release digest or media role does not match.")
                    validate_package(package, expected_digest=ref.package_sha256)
                # Commit before exposing the new schedule to the playout thread.
                self.journal.save_control(requested)
                with self.lock:
                    self.control = requested
                    self.schedule_ids = [ref.release_id for ref in requested.programme_releases]
                    self.position = 0
                self.journal.event("control_accepted",
                                   detail=f"Rundown revision {revision} accepted for the next airing.")
            self._last_control_error = None
            acknowledged = self._write_control_ack(revision, "accepted", None)
        except (ControlError, ContractError, ValueError, OSError) as exc:
            reason = (str(exc) if isinstance(exc, ControlError) else
                      "A requested release failed station-owned media verification.")
            error = (observed[0], reason)
            if self._last_control_error != error:
                self.journal.event("control_rejected", detail=reason)
                self._last_control_error = error
            acknowledged = self._write_control_ack(revision, "rejected", reason)
        if acknowledged:
            self._last_control_observed = observed

    def _select(self):
        with self.lock:
            mode = self.control.mode if self.control else "programmes"
            programmes = ([self.known[rid] for rid in self.schedule_ids
                           if rid in self.known and self.known[rid][0].kind == "programme"
                           and rid not in self.rejections] if mode == "programmes" else [])
            if self.control:
                fallback_id = (self.control.fallback_release.release_id
                               if self.control.fallback_release else None)
                idents = ([self.known[fallback_id]]
                          if fallback_id in self.known and fallback_id not in self.rejections else [])
            else:
                idents = [pair for rid, pair in self.known.items()
                          if pair[0].kind == "ident" and rid not in self.rejections]
            if programmes:
                pivot = self.position % len(programmes)
                candidates = programmes[pivot:] + programmes[:pivot] + idents
                self.position += 1
            else:
                candidates = idents
            has_rundown = bool(self.schedule_ids)
        for manifest, package in candidates:
            try:
                validate_package(package, expected_digest=manifest.package_sha256)
                self.selection_reason = (None if manifest.kind == "programme" else
                                         "operator_selected_fallback" if mode == "fallback" else
                                         "no_valid_programmes" if has_rundown else
                                         "no_released_programmes")
                return manifest, package
            except (OSError, ValueError, ContractError):
                self.reject(manifest.release_id, "Local release failed validation before playback.")
        self.selection_reason = ("fallback_unavailable" if mode == "fallback" else
                                 "no_valid_programmes_or_fallback" if has_rundown else
                                 "no_released_programmes")
        return None

    def _open_encoder(self):
        self.session_id = uuid.uuid4().hex
        self.frames = 0
        self.calibrated = False
        self.session_monotonic = time.monotonic()
        self.journal.session(self.session_id)
        log_path = self.root / "encoder.log"
        log_handle = log_path.open("ab")
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
            "-f", "s16le", "-ar", str(RATE), "-ac", str(CHANNELS), "-i", "pipe:0",
            "-map", "0:a:0", "-c:a", "aac", "-b:a", "96k", "-ar", str(RATE),
            "-f", "hls", "-hls_time", "2", "-hls_list_size", "12",
            "-hls_delete_threshold", "15", "-hls_start_number_source", "epoch_us",
            "-hls_flags", "delete_segments+program_date_time+temp_file+discont_start",
            "-hls_segment_filename", str(self.hls / (self.session_id + "-%d.ts")),
            str(self.hls / "live.m3u8")]
        try:
            self.encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                            stderr=log_handle)
        finally:
            log_handle.close()

    def _calibrate(self):
        if self.calibrated or not self.session_id:
            return
        try:
            playlist = (self.hls / "live.m3u8").read_text()
            if self.session_id not in playlist:
                return
            stamp = re.search(r"^#EXT-X-PROGRAM-DATE-TIME:(.+)$", playlist, re.MULTILINE)
            if stamp:
                self.journal.calibrate(self.session_id, iso(parse_time(stamp.group(1).strip())))
                self.calibrated = True
        except (OSError, ValueError):
            pass

    def _close_encoder(self):
        encoder, self.encoder = self.encoder, None
        if encoder:
            try:
                if encoder.stdin:
                    encoder.stdin.close()
                encoder.wait(timeout=5)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                encoder.terminate()
                try:
                    encoder.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    encoder.kill()
                    encoder.wait(timeout=2)
        self.session_id = None

    def _play(self, manifest: Manifest, package: Path):
        audio = next(asset for asset in manifest.assets if asset.role == "audio")
        decoder = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i",
             str(package / audio.path), "-map", "0:a:0", "-f", "s16le", "-acodec",
             "pcm_s16le", "-ar", str(RATE), "-ac", str(CHANNELS), "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        airing_id = None
        played_frames = 0
        complete = False
        try:
            while not self.stop_event.is_set():
                chunk = decoder.stdout.read(RATE // 10 * FRAME_BYTES)
                if not chunk:
                    complete = decoder.wait(timeout=3) == 0
                    break
                if len(chunk) % FRAME_BYTES:
                    raise RuntimeError("decoder emitted incomplete PCM frames")
                if not self.encoder or self.encoder.poll() is not None:
                    raise RuntimeError("HLS encoder stopped")
                offset_ms = self.frames * 1000 // RATE
                self.encoder.stdin.write(chunk)
                self.encoder.stdin.flush()
                if airing_id is None:
                    airing_id = self.journal.begin(self.session_id, manifest, offset_ms)
                frames = len(chunk) // FRAME_BYTES
                self.frames += frames
                played_frames += frames
                self.journal.advance(self.session_id, self.frames * 1000 // RATE)
                self._calibrate()
                delay = self.session_monotonic + self.frames / RATE - time.monotonic()
                if delay > 0:
                    self.stop_event.wait(delay)
            duration_ms = played_frames * 1000 // RATE
            complete = complete and abs(duration_ms - manifest.duration_ms) <= 150
        finally:
            if decoder.poll() is None:
                decoder.terminate()
            try:
                decoder.wait(timeout=3)
            except subprocess.TimeoutExpired:
                decoder.kill()
                decoder.wait(timeout=2)
            if decoder.stdout:
                decoder.stdout.close()
            if airing_id:
                self.journal.finish(airing_id, self.frames * 1000 // RATE,
                                    interrupted=not complete)
            self._calibrate()
        if not complete and not self.stop_event.is_set():
            raise RuntimeError("programme did not decode to its declared duration")

    def _run(self):
        while not self.stop_event.is_set():
            try:
                selected = self._select()
                if not selected:
                    self.state = "waiting"
                    self.playback_kind = None
                    self.waiting_reason = self.selection_reason
                    self.fallback_reason = None
                    self._close_encoder()
                    self.stop_event.wait(0.75)
                    continue
                if self.encoder is None:
                    self._open_encoder()
                self.playback_kind = selected[0].kind
                self.state = "fallback" if self.playback_kind == "ident" else "broadcasting"
                self.waiting_reason = None
                self.fallback_reason = self.selection_reason if self.playback_kind == "ident" else None
                self.error = None
                self._play(*selected)
            except Exception as exc:
                self.error = "Playout interrupted; retrying verified material."
                self.state = "recovering"
                self.playback_kind = None
                self.journal.event("engine_error", detail=type(exc).__name__)
                log.exception("Playout interrupted")
                self._close_encoder()
                self.stop_event.wait(1)
        self.state = "stopped"
        self.playback_kind = None
        self._close_encoder()

    def snapshot(self):
        with self.lock:
            schedule = [self.known[rid][0].model_dump(mode="json")
                        for rid in self.schedule_ids if rid in self.known]
            rejected = [{"release_id": key, "reason": value}
                        for key, value in self.rejections.items()]
        return {
            "state": self.state,
            "mode": self.control.mode if self.control else "programmes",
            "control_revision": self.control.revision if self.control else 0,
            "playback_kind": self.playback_kind,
            "waiting_reason": self.waiting_reason,
            "fallback_reason": self.fallback_reason,
            "error": self.error,
            "stream_ready": self.state in ("broadcasting", "fallback") and self.calibrated and (self.hls / "live.m3u8").exists(),
            "live_url": "/hls/live.m3u8",
            "schedule": schedule,
            "rejected": rejected,
            "last_completed": self.journal.last_completed(),
        }
