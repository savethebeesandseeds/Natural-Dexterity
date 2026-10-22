"""Description-only original voice creation with explicit quoted allocations.

No replication, reference-upload, automatic queue drain, or automatic retry.
A caller-supplied quote is a local allocation, never a provider billing hard cap.
"""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import fcntl
import hashlib
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import re
import stat
from typing import Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .budget import Budget, BudgetError, dollars_to_microdollars
from .characters import CharacterStore, CharacterError, IDENTIFIER, SLUG, _encode, now, parse_timestamp
from . import gemini, production_pricing


class DesignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    character_id: str = Field(pattern=SLUG)
    description: str = Field(min_length=10, max_length=6000)
    model: Literal["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"]
    language_code: str = Field(default="en", min_length=2, max_length=35,
                               pattern=r"^[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})*$")
    gender: Literal["male", "female", "neutral"] | None = None
    quoted_cost_usd: str = Field(min_length=1, max_length=32)
    quote_evidence: str = Field(min_length=12, max_length=2000)
    quote_valid_until: str = Field(min_length=10, max_length=64)
    allocation_preset: Literal["standard"] | None = None

    @classmethod
    def from_preset(cls, character_id: str, description: str, model: str, *,
                    preset: str = "standard", language_code: str = "en", gender: str | None = None):
        policy = production_pricing.voice_design_preset(model, preset)
        return cls(
            character_id=character_id, description=description, model=model,
            language_code=language_code, gender=gender, allocation_preset=preset,
            quoted_cost_usd=policy["quoted_cost_usd"], quote_evidence=policy["quote_evidence"],
            quote_valid_until=policy["quote_valid_until"],
        )

    @model_validator(mode="after")
    def quote_is_usable(self):
        if not self.description.strip():
            raise ValueError("Describe an original voice in words.")
        dollars_to_microdollars(self.quoted_cost_usd)
        if self.allocation_preset is not None:
            policy = production_pricing.voice_design_preset(self.model, self.allocation_preset)
            if any(getattr(self, name) != policy[name] for name in (
                "quoted_cost_usd", "quote_evidence", "quote_valid_until"
            )):
                raise ValueError("Preset allocations must match their versioned local policy.")
        if parse_timestamp(self.quote_valid_until) <= datetime.now(timezone.utc):
            raise ValueError("Voice-design quote expired; verify a new allocation before generation.")
        return self


class VoiceDesignService:
    def __init__(self, project_root: Path, *, key_provider: Callable[[], str | None] | None = None):
        self.root = Path(project_root)
        with self._global_lease():
            self.characters = CharacterStore(self.root)
            self.database = self.root / ".runtime/studio/studio.sqlite3"
            self.budget = Budget(self.database)
            self.key_provider = key_provider or self._stored_key
            with self.budget.connect() as db:
                db.execute("""
                    CREATE TABLE IF NOT EXISTS voice_design_jobs (
                        id TEXT PRIMARY KEY, character_id TEXT NOT NULL, model TEXT NOT NULL,
                        status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                        request_json TEXT NOT NULL, quote_json TEXT NOT NULL,
                        character_json TEXT NOT NULL, reservation_microdollars INTEGER NOT NULL,
                        recipe_json TEXT, error TEXT, dispatch_count INTEGER NOT NULL DEFAULT 0,
                        runner_pid INTEGER, runner_start_ticks INTEGER,
                        response_sha256 TEXT, dispatched_at TEXT
                    )
                """)
                columns = {row["name"] for row in db.execute("PRAGMA table_info(voice_design_jobs)")}
                for name, definition in (("runner_pid", "INTEGER"), ("runner_start_ticks", "INTEGER"),
                                         ("response_sha256", "TEXT"), ("dispatched_at", "TEXT"),
                                         ("diagnostic_json", "TEXT"), ("diagnostic_sha256", "TEXT"),
                                         ("usage_estimate_json", "TEXT")):
                    if name not in columns:
                        db.execute("ALTER TABLE voice_design_jobs ADD COLUMN " + name + " " + definition)
                db.execute("""
                    CREATE TABLE IF NOT EXISTS voice_design_reconciliations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                        charged_microdollars INTEGER NOT NULL, evidence TEXT NOT NULL,
                        recorded_at TEXT NOT NULL
                    )
                """)

    @staticmethod
    def _start_ticks(pid):
        try:
            return int((Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19])
        except (OSError, ValueError, IndexError):
            return None

    @contextmanager
    def _global_lease(self):
        directory = self.root / ".runtime/studio"
        if any(parent.is_symlink() for parent in (directory, directory.parent)):
            raise ValueError("Voice-design storage is not an ordinary local directory.")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "design-worker.lock"
        try:
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        except OSError:
            raise ValueError("Voice-design operation lock is not an ordinary local file.") from None
        with os.fdopen(descriptor, "a") as lock:
            if not stat.S_ISREG(os.fstat(lock.fileno()).st_mode):
                raise ValueError("Voice-design operation lock must be an ordinary file.")
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Offline backup is active; voice-design operation preserved.") from None
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    @contextmanager
    def _lease(self, job_id: str):
        with self._global_lease():
            path = self.characters.artifact(f"design-jobs/{job_id}/run.lock")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise ValueError("Voice-design job is active in another process; preserved.") from None
                try:
                    yield
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def _stored_key(self):
        path = self.root / ".runtime/secrets/gemini-api-key"
        if path.is_symlink():
            raise CharacterError("Credential path is not an ordinary local file.")
        return (path.read_text(encoding="utf-8").strip() if path.is_file()
                else os.environ.get("GEMINI_API_KEY", "").strip()) or None

    def configure_budget(self, character_id: str, episode_limit_usd: str, series_limit_usd: str):
        with self._global_lease():
            self.characters.get(character_id)
            self.budget.configure("voice-design:" + character_id, "voice-design",
                                  dollars_to_microdollars(episode_limit_usd),
                                  dollars_to_microdollars(series_limit_usd))

    def enqueue(self, request: DesignRequest) -> dict:
        with self._global_lease():
            return self._enqueue_owned(request)

    def enqueue_preset(self, character_id: str, description: str, model: str, *,
                       preset: str = "standard", language_code: str = "en", gender: str | None = None) -> dict:
        return self.enqueue(DesignRequest.from_preset(
            character_id, description, model, preset=preset, language_code=language_code, gender=gender))

    def _enqueue_owned(self, request: DesignRequest) -> dict:
        # Revalidate expiry even if a caller held a previously constructed model.
        request = DesignRequest.model_validate(request.model_dump())
        character = self.characters.get(request.character_id)
        voice = {
            "model": request.model, "type": "prompted", "display_name": character["name"],
            "language_code": request.language_code, "prompted": {"input": request.description},
        }
        if request.gender is not None:
            voice["gender"] = request.gender
        payload = {"store": True, "voice": voice}
        gemini.validate_design_request(payload)
        quote = {
            "quoted_cost_usd": request.quoted_cost_usd,
            "evidence": request.quote_evidence, "valid_until": request.quote_valid_until,
            "basis": "versioned-local-allocation-policy" if request.allocation_preset
                     else "operator-supplied-local-allocation",
            "guarantee": "Local allocation only; not a provider quote, verified charge, or billing hard cap.",
        }
        if request.allocation_preset:
            policy = production_pricing.voice_design_preset(request.model, request.allocation_preset)
            quote["allocation_preset"] = request.allocation_preset
            quote["pricing_version"] = policy["pricing_version"]
            quote["pricing_evidence"] = policy["evidence"]
        identifier = uuid4().hex
        allocation = dollars_to_microdollars(request.quoted_cost_usd)
        stamp = now()
        # A queued job and its reservation commit together, so a crash cannot
        # leave money reserved without an identifiable set of exact inputs.
        with self.budget.transaction() as db:
            self.budget.reserve(identifier, "voice-design:" + request.character_id,
                                "voice-design", allocation, connection=db)
            db.execute(
                """INSERT INTO voice_design_jobs
                (id,character_id,model,status,created_at,updated_at,request_json,
                 quote_json,character_json,reservation_microdollars)
                VALUES(?,?,?,'queued',?,?,?,?,?,?)""",
                (identifier, request.character_id, request.model, stamp, stamp,
                 json.dumps(payload, ensure_ascii=False), json.dumps(quote, ensure_ascii=False),
                 json.dumps(character, ensure_ascii=False), allocation),
            )
        self.characters.write_artifact(f"design-jobs/{identifier}/request.json", _encode(payload))
        self.characters.write_artifact(f"design-jobs/{identifier}/quote.json", _encode(quote))
        self.characters.write_artifact(f"design-jobs/{identifier}/character.json", _encode(character))
        return self.status(identifier)

    def status(self, job_id: str) -> dict:
        if not isinstance(job_id, str) or not re.fullmatch(IDENTIFIER, job_id):
            raise ValueError("Unknown voice-design job.")
        with self.budget.connect() as db:
            row = db.execute("SELECT * FROM voice_design_jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise ValueError("Unknown voice-design job.")
        result = dict(row)
        for key in ("request", "quote", "character", "recipe", "diagnostic", "usage_estimate"):
            value = result.pop(key + "_json")
            result[key] = json.loads(value) if value else None
        with self.budget.connect() as db:
            allocation = db.execute("SELECT state,charged_micro,accounting FROM reservations WHERE attempt_id=?", (job_id,)).fetchone()
            result["allocation"] = dict(allocation) if allocation else None
            result["reconciliations"] = [dict(row) for row in db.execute(
                "SELECT charged_microdollars,evidence,recorded_at FROM voice_design_reconciliations WHERE job_id=? ORDER BY id",
                (job_id,),
            )]
        result["accounting_notice"] = (
            "Reservations and consumed allocations are local accounting. "
            "Token-priced usage is an estimate, not a verified provider invoice.")
        return result

    def _failure(self, job_id: str, state: str, message: str, diagnostic: dict | None = None):
        encoded = _encode(diagnostic) if diagnostic is not None else None
        digest = hashlib.sha256(encoded).hexdigest() if encoded is not None else None
        with self.budget.connect() as db:
            db.execute(
                "UPDATE voice_design_jobs SET status=?,error=?,updated_at=?,"
                "diagnostic_json=COALESCE(?,diagnostic_json),diagnostic_sha256=COALESCE(?,diagnostic_sha256) WHERE id=?",
                (state, message, now(), encoded.decode("utf-8") if encoded else None, digest, job_id))
        if encoded is not None:
            try:
                self.characters.write_artifact(f"design-jobs/{job_id}/failure-diagnostic.json", encoded)
            except (OSError, CharacterError):
                # The authoritative diagnostic and hash remain in SQLite even
                # if its readable private snapshot could not be written.
                pass

    def run_one_specific(self, job_id: str, transport=None) -> dict:
        self.status(job_id)
        with self._lease(job_id):
            return self._run_one_owned(job_id, transport=transport)

    def _run_one_owned(self, job_id: str, transport=None) -> dict:
        with self.budget.transaction() as db:
            row = db.execute("SELECT * FROM voice_design_jobs WHERE id=?", (job_id,)).fetchone()
            if row["status"] != "queued":
                return self.status(job_id)
            db.execute(
                "UPDATE voice_design_jobs SET status='running',runner_pid=?,runner_start_ticks=?,updated_at=? WHERE id=?",
                (os.getpid(), self._start_ticks(os.getpid()), now(), job_id),
            )
        job = self.status(job_id)
        dispatched = False
        try:
            if parse_timestamp(job["quote"]["valid_until"]) <= datetime.now(timezone.utc):
                self.budget.release_unsent(job_id)
                self._failure(job_id, "failed", "Quote expired before dispatch; allocation released.")
                return self.status(job_id)
            key = self.key_provider()
            if not key:
                self.budget.release_unsent(job_id)
                self._failure(job_id, "failed", "Credential missing; no request dispatched and allocation released.")
                return self.status(job_id)
            # Inputs are authoritative in the transaction even after a crash
            # while creating their readable snapshots.
            try:
                gemini.validate_design_request(job["request"])
            except ValueError:
                self.budget.release_unsent(job_id)
                self._failure(job_id, "failed", "Retained request failed local validation; no request dispatched.")
                return self.status(job_id)
            self.characters.write_artifact(f"design-jobs/{job_id}/request.json", _encode(job["request"]))
            with self.budget.connect() as db:
                db.execute("UPDATE voice_design_jobs SET dispatch_count=1,dispatched_at=?,updated_at=? WHERE id=?",
                           (now(), now(), job_id))
            dispatched = True
            response = gemini.post_json("/v1beta/voices", job["request"], key, transport=transport)
            response_bytes = _encode(response)
            if len(response_bytes) > gemini.MAX_RESPONSE_BYTES:
                raise gemini.GeminiFailure("Retained voice-design response exceeds its size limit; allocation retained.", True)
            response_digest = hashlib.sha256(response_bytes).hexdigest()
            # Authenticate recovery artifacts against the authoritative database,
            # not a digest stored next to a replaceable response file.
            with self.budget.connect() as db:
                db.execute("UPDATE voice_design_jobs SET response_sha256=?,updated_at=? WHERE id=?",
                           (response_digest, now(), job_id))
            self.characters.write_artifact(f"design-jobs/{job_id}/response.json", response_bytes)
            self._complete_from_response(self.status(job_id), response)
        except gemini.GeminiFailure as exc:
            self._failure(job_id, "uncertain" if exc.ambiguous else "failed", str(exc), exc.diagnostic)
        except Exception as exc:
            if not dispatched:
                self.budget.release_unsent(job_id)
            # Never log raw exception messages: local transport/provider objects
            # may contain request headers or user prompts.
            state = "uncertain" if dispatched else "failed"
            self._failure(job_id, state,
                          "Voice-design outcome requires inspection; exact inputs and any provider result retained."
                          if dispatched else "Local voice-design setup failed; no request dispatched.")
        return self.status(job_id)

    def _complete_from_response(self, job: dict, response: dict):
        job_id = job["id"]
        expected_digest = job.get("response_sha256")
        if (not expected_digest or hashlib.sha256(_encode(response)).hexdigest() != expected_digest):
            raise CharacterError("Retained provider response failed authoritative verification.")
        if response.get("type") != "prompted":
            raise gemini.GeminiFailure("Provider did not confirm description-only voice design; allocation retained.", True)
        audio = gemini.parse_audio_block(response.get("sample_audio"))
        matches = [recipe for recipe in self.characters.recipes(job["character_id"])
                   if recipe["design_job_id"] == job_id]
        if len(matches) > 1:
            raise CharacterError("Multiple recipes claim one design job; preserved for inspection.")
        if matches:
            recipe = matches[0]
            self.characters._verify_recipe(recipe)
        else:
            recipe = self.characters._register_designed(
                job["character_id"], request=job["request"], response=response,
                audio=audio, job_id=job_id,
                created_at=job.get("dispatched_at") or job["created_at"],
            )
        # Prompted CreateVoice reports token usage. Keep its rate-based estimate
        # separate from the conservatively consumed allocation and any invoice.
        try:
            usage_estimate = production_pricing.provider_usage_estimate(job["model"], response.get("usage"))
        except BudgetError:
            usage_estimate = None
        with self.budget.transaction() as db:
            reservation = db.execute("SELECT state FROM reservations WHERE attempt_id=?", (job_id,)).fetchone()
            if reservation["state"] == "reserved":
                self.budget.settle(job_id, job["reservation_microdollars"],
                                   "voice-design-quoted-allocation", connection=db)
            elif reservation["state"] != "spent":
                raise BudgetError("Voice-design allocation is no longer available.")
            db.execute(
                "UPDATE voice_design_jobs SET status='ready',recipe_json=?,usage_estimate_json=?,error=NULL,updated_at=? WHERE id=?",
                (json.dumps(recipe, ensure_ascii=False), json.dumps(usage_estimate, ensure_ascii=False)
                 if usage_estimate is not None else None, now(), job_id),
            )

    def recover(self, job_id: str) -> dict:
        self.status(job_id)
        with self._lease(job_id):
            job = self.status(job_id)
            if job["status"] not in ("running", "uncertain"):
                return job
            response_path = self.characters.artifact(f"design-jobs/{job_id}/response.json")
            if response_path.is_file():
                try:
                    with response_path.open("rb") as stream:
                        raw = stream.read(gemini.MAX_RESPONSE_BYTES + 1)
                    if len(raw) > gemini.MAX_RESPONSE_BYTES:
                        raise ValueError
                    if not job.get("response_sha256") or hashlib.sha256(raw).hexdigest() != job["response_sha256"]:
                        raise CharacterError("Retained provider response failed authoritative verification.")
                    response = json.loads(raw)
                    self._complete_from_response(job, response)
                    return self.status(job_id)
                except Exception:
                    # Recovery never submits anything to the provider.
                    pass
            self._failure(job_id, "uncertain",
                          "Voice-design attempt was interrupted; allocation retained. Reconcile before a new attempt.")
            return self.status(job_id)

    def reconcile(self, job_id: str, charged_usd: str, evidence: str) -> dict:
        self.status(job_id)
        with self._lease(job_id):
            return self._reconcile_owned(job_id, charged_usd, evidence)

    def _reconcile_owned(self, job_id: str, charged_usd: str, evidence: str) -> dict:
        job = self.status(job_id)
        if job["status"] not in ("failed", "uncertain"):
            raise ValueError("Only unresolved voice-design outcomes can be reconciled.")
        if not isinstance(evidence, str) or not 12 <= len(evidence) <= 2000:
            raise ValueError("Record billing evidence before reconciling a voice-design attempt.")
        try:
            number = Decimal(charged_usd)
            units = number * 1_000_000
            if not number.is_finite() or number < 0 or units != units.to_integral_value():
                raise ValueError
            charged = int(units)
        except (InvalidOperation, ValueError, TypeError):
            raise ValueError("Use a nonnegative USD amount with up to six decimal places.") from None
        stamp = now()
        with self.budget.transaction() as db:
            self.budget.settle(job_id, charged, "manual-voice-design-billing-reconciliation", connection=db)
            db.execute(
                "INSERT INTO voice_design_reconciliations(job_id,charged_microdollars,evidence,recorded_at) VALUES(?,?,?,?)",
                (job_id, charged, evidence, stamp),
            )
        self.characters.write_artifact(
            f"design-jobs/{job_id}/reconciliation-{uuid4().hex}.json",
            _encode({"charged_usd": charged_usd, "evidence": evidence, "recorded_at": stamp}),
        )
        return self.status(job_id)
