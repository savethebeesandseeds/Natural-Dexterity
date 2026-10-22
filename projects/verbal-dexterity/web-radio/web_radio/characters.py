"""Versioned backend characters and provenance-bound synthetic voice recipes."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .budget import RATES
from .gemini import PREBUILT_VOICES, MAX_RESPONSE_BYTES

SLUG = r"^[a-z0-9][a-z0-9-]{0,79}$"
IDENTIFIER = r"^[a-f0-9]{32}$"
RECORD_LIMIT = 1024 * 1024
PROVENANCE_LIMIT = MAX_RESPONSE_BYTES


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("A timezone is required.")
    return result.astimezone(timezone.utc)


class CharacterError(ValueError):
    pass


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class CharacterInput(FrozenModel):
    character_id: str = Field(pattern=SLUG)
    name: str = Field(min_length=1, max_length=80)
    personality: str = Field(default="", max_length=6000)
    background: str = Field(default="", max_length=12000)
    pronunciation_notes: str = Field(default="", max_length=2000)


class CharacterRecord(CharacterInput):
    schema_version: Literal[1] = 1
    revision: int = Field(ge=1)
    created_at: str
    preferred_voice_recipe_id: str | None = Field(default=None, pattern=IDENTIFIER)


class VoiceRecipe(FrozenModel):
    schema_version: Literal[1] = 1
    recipe_id: str = Field(pattern=IDENTIFIER)
    character_id: str = Field(pattern=SLUG)
    revision: int = Field(ge=1)
    origin: Literal["prebuilt", "original_design"]
    provider: Literal["gemini"] = "gemini"
    voice: str = Field(min_length=1, max_length=160)
    model: str
    language: str = Field(min_length=2, max_length=35)
    created_at: str
    design_prompt: str | None = Field(default=None, max_length=6000)
    design_job_id: str | None = Field(default=None, pattern=IDENTIFIER)
    request_path: str | None = None
    request_sha256: str | None = None
    response_path: str | None = None
    response_sha256: str | None = None
    audition_path: str | None = None
    audition_sha256: str | None = None
    expires_at: str | None = None
    expiry_basis: Literal["not_applicable", "provider", "documented_ttl_conservative"] = "not_applicable"

    @model_validator(mode="after")
    def truthful_origin(self):
        if self.model not in RATES:
            raise ValueError("Unsupported voice model.")
        if self.origin == "prebuilt":
            if self.voice not in PREBUILT_VOICES or any((
                self.design_job_id, self.request_path, self.response_path, self.audition_path,
                self.request_sha256, self.response_sha256, self.audition_sha256, self.expires_at,
            )):
                raise ValueError("A prebuilt recipe cannot claim custom design provenance.")
        else:
            if not re.fullmatch(r"voice_[A-Za-z0-9_-]{1,150}", self.voice):
                raise ValueError("An original design needs a stateful provider voice ID.")
            if not all((self.design_prompt, self.design_job_id, self.request_path,
                        self.response_path, self.audition_path, self.request_sha256,
                        self.response_sha256, self.audition_sha256, self.expires_at)):
                raise ValueError("An original design needs complete creation and audition provenance.")
            parse_timestamp(self.expires_at)
        return self


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _encode(data) -> bytes:
    return (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise CharacterError("Duplicate record keys.")
        result[key] = value
    return result


class CharacterStore:
    """SQLite is authoritative; readable immutable snapshots accompany each revision."""

    def __init__(self, project_root: Path):
        self.root = Path(project_root)
        self.state = self.root / ".runtime/studio/characters"
        for path in (self.root / ".runtime", self.root / ".runtime/studio", self.state):
            if path.is_symlink():
                raise CharacterError("Character storage cannot use symlink roots.")
        self.state.mkdir(parents=True, exist_ok=True)
        self.database = self.state / "registry.sqlite3"
        if self.database.is_symlink():
            raise CharacterError("Character registry is not an ordinary local file.")
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS characters (
                    character_id TEXT NOT NULL, revision INTEGER NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY(character_id,revision)
                );
                CREATE TABLE IF NOT EXISTS recipes (
                    recipe_id TEXT PRIMARY KEY, character_id TEXT NOT NULL,
                    revision INTEGER NOT NULL, payload TEXT NOT NULL,
                    UNIQUE(character_id,revision)
                );
            """)

    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA busy_timeout=30000")
        return db

    def artifact(self, relative: str) -> Path:
        if not relative or "\\" in relative:
            raise CharacterError("Unsafe character artifact path.")
        path = Path(relative)
        if path.is_absolute() or any(p in ("", ".", "..") for p in path.parts):
            raise CharacterError("Unsafe character artifact path.")
        target = self.state / path
        for part in [target, *target.parents]:
            if part == self.state.parent:
                break
            if part.is_symlink():
                raise CharacterError("Character artifacts cannot be symlinks.")
        if not target.resolve().is_relative_to(self.state.resolve()):
            raise CharacterError("Character artifact escaped storage.")
        return target

    def write_artifact(self, relative: str, data: bytes) -> Path:
        target = self.artifact(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not target.is_file() or target.read_bytes() != data:
                raise CharacterError("Conflicting immutable character artifact preserved.")
            return target
        temporary = target.with_name(target.name + ".tmp-" + uuid4().hex)
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # Writers use unique job/recipe IDs or a SQLite-serialized revision.
        if target.exists():
            temporary.unlink()
            raise CharacterError("Conflicting immutable artifact preserved.")
        os.replace(temporary, target)
        return target

    def _read_artifact(self, relative: str, checksum: str, *, limit=PROVENANCE_LIMIT) -> bytes:
        path = self.artifact(relative)
        if not path.is_file():
            raise CharacterError("Character provenance artifact is unavailable.")
        with path.open("rb") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit or _sha(data) != checksum:
            raise CharacterError("Character provenance failed verification.")
        return data

    @staticmethod
    def _slug(character_id):
        if not isinstance(character_id, str) or not re.fullmatch(SLUG, character_id):
            raise CharacterError("Use a stable lowercase character ID.")

    def create(self, data: dict | CharacterInput) -> dict:
        item = data if isinstance(data, CharacterInput) else CharacterInput.model_validate(data)
        record = CharacterRecord(**item.model_dump(), revision=1, created_at=now())
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM characters WHERE character_id=?", (item.character_id,)).fetchone():
                raise CharacterError("Character already exists; update it as a new revision.")
            db.execute("INSERT INTO characters VALUES(?,?,?)",
                       (item.character_id, 1, record.model_dump_json()))
        result = record.model_dump()
        self.write_artifact(f"{item.character_id}/revisions/000001.json", _encode(result))
        return result

    def get(self, character_id: str, revision: int | None = None) -> dict:
        self._slug(character_id)
        with self.connect() as db:
            if revision is None:
                row = db.execute("SELECT payload FROM characters WHERE character_id=? ORDER BY revision DESC LIMIT 1",
                                 (character_id,)).fetchone()
            elif type(revision) is int and revision > 0:
                row = db.execute("SELECT payload FROM characters WHERE character_id=? AND revision=?",
                                 (character_id, revision)).fetchone()
            else:
                raise CharacterError("Invalid character revision.")
        if not row:
            raise CharacterError("Character revision does not exist.")
        return CharacterRecord.model_validate_json(row["payload"]).model_dump()

    def list(self) -> list[dict]:
        with self.connect() as db:
            ids = [row[0] for row in db.execute("SELECT DISTINCT character_id FROM characters ORDER BY character_id")]
        return [self.get(identity) for identity in ids]

    def update(self, character_id: str, data: dict | CharacterInput, *, expected_revision: int) -> dict:
        self._slug(character_id)
        values = data.model_dump() if isinstance(data, CharacterInput) else dict(data)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM characters WHERE character_id=? ORDER BY revision DESC LIMIT 1",
                             (character_id,)).fetchone()
            if not row:
                raise CharacterError("Character does not exist.")
            current = CharacterRecord.model_validate_json(row["payload"])
            if type(expected_revision) is not int or current.revision != expected_revision:
                raise CharacterError("Character changed; use its current revision.")
            base = {key: value for key, value in current.model_dump().items() if key in CharacterInput.model_fields}
            base.update(values)
            item = CharacterInput.model_validate(base)
            if item.character_id != character_id:
                raise CharacterError("Character identity cannot be changed.")
            record = CharacterRecord(**item.model_dump(), revision=current.revision + 1, created_at=now(),
                                     preferred_voice_recipe_id=current.preferred_voice_recipe_id)
            db.execute("INSERT INTO characters VALUES(?,?,?)",
                       (character_id, record.revision, record.model_dump_json()))
        result = record.model_dump()
        self.write_artifact(f"{character_id}/revisions/{record.revision:06d}.json", _encode(result))
        return result

    def _save_recipe(self, values: dict) -> dict:
        character_id = values["character_id"]
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            head = db.execute("SELECT payload FROM characters WHERE character_id=? ORDER BY revision DESC LIMIT 1",
                              (character_id,)).fetchone()
            if not head:
                raise CharacterError("Character does not exist.")
            character = CharacterRecord.model_validate_json(head["payload"])
            number = db.execute("SELECT COALESCE(MAX(revision),0)+1 FROM recipes WHERE character_id=?",
                                (character_id,)).fetchone()[0]
            recipe = VoiceRecipe(**values, revision=number)
            db.execute("INSERT INTO recipes VALUES(?,?,?,?)",
                       (recipe.recipe_id, character_id, number, recipe.model_dump_json()))
            preferred = CharacterRecord(**{
                **character.model_dump(), "revision": character.revision + 1,
                "created_at": now(), "preferred_voice_recipe_id": recipe.recipe_id,
            })
            db.execute("INSERT INTO characters VALUES(?,?,?)",
                       (character_id, preferred.revision, preferred.model_dump_json()))
        result = recipe.model_dump()
        self.write_artifact(f"{character_id}/voice-recipes/{recipe.recipe_id}.json", _encode(result))
        self.write_artifact(f"{character_id}/revisions/{preferred.revision:06d}.json", _encode(preferred.model_dump()))
        return result

    def add_prebuilt(self, character_id: str, voice: str, model: str, *, language="en") -> dict:
        self.get(character_id)
        return self._save_recipe({
            "recipe_id": uuid4().hex, "character_id": character_id, "origin": "prebuilt",
            "voice": voice, "model": model, "language": language, "created_at": now(),
        })

    def _register_designed(self, character_id: str, *, request: dict, response: dict,
                           audio: bytes, job_id: str, created_at: str | None = None) -> dict:
        """Internal promotion used only after a description-only provider job succeeds."""
        self.get(character_id)
        if not re.fullmatch(IDENTIFIER, job_id):
            raise CharacterError("Invalid design job ID.")
        voice_request = request.get("voice", {})
        if (request.get("store") is not True or voice_request.get("type") != "prompted"
                or set(voice_request.get("prompted", {})) != {"input"}
                or response.get("type") != "prompted"
                or response.get("model") not in (None, voice_request.get("model"),
                                                f"models/{voice_request.get('model')}")):
            raise CharacterError("Only verified description-designed voices can be registered.")
        voice_id = response.get("id", "")
        if not isinstance(voice_id, str) or not re.fullmatch(r"voice_[A-Za-z0-9_-]{1,150}", voice_id):
            raise CharacterError("Provider did not return an original designed voice ID.")
        if any(key in voice_request for key in ("replicated", "audio", "reference_audio", "base_voice")):
            raise CharacterError("Voice replication is excluded.")
        creation_time = created_at or now()
        creation_date = parse_timestamp(creation_time)
        recipe_id = uuid4().hex
        request_path = f"design-jobs/{job_id}/request.json"
        response_path = f"design-jobs/{job_id}/response.json"
        audition_path = f"{character_id}/auditions/{recipe_id}.wav"
        request_bytes, response_bytes = _encode(request), _encode(response)
        if len(request_bytes) > RECORD_LIMIT or len(response_bytes) > PROVENANCE_LIMIT:
            raise CharacterError("Voice-design provenance exceeds its retained size limit.")
        from .gemini import parse_audio_block
        if parse_audio_block(response.get("sample_audio")) != audio:
            raise CharacterError("Audition does not match the provider result.")
        self.write_artifact(request_path, request_bytes)
        self.write_artifact(response_path, response_bytes)
        self.write_artifact(audition_path, audio)
        expires = response.get("expire_time") or response.get("expires_at")
        if expires is not None:
            if not isinstance(expires, str) or parse_timestamp(expires) <= datetime.now(timezone.utc):
                raise CharacterError("Provider voice is already expired or has an invalid expiry.")
            expiry_basis = "provider"
        else:
            # Documented one-year TTL is a retention bound, not a version guarantee.
            expires = (creation_date + timedelta(days=365)).isoformat()
            expiry_basis = "documented_ttl_conservative"
            if parse_timestamp(expires) <= datetime.now(timezone.utc):
                raise CharacterError("Designed voice retention bound expired; preserved for inspection.")
        return self._save_recipe({
            "recipe_id": recipe_id, "character_id": character_id, "origin": "original_design",
            "voice": voice_id, "model": voice_request["model"],
            "language": voice_request["language_code"], "created_at": creation_time,
            "design_prompt": voice_request["prompted"]["input"], "design_job_id": job_id,
            "request_path": request_path, "request_sha256": _sha(request_bytes),
            "response_path": response_path, "response_sha256": _sha(response_bytes),
            "audition_path": audition_path, "audition_sha256": _sha(audio),
            "expires_at": expires, "expiry_basis": expiry_basis,
        })

    def get_recipe(self, character_id: str, recipe_id: str) -> dict:
        self._slug(character_id)
        if not isinstance(recipe_id, str) or not re.fullmatch(IDENTIFIER, recipe_id):
            raise CharacterError("Invalid voice recipe ID.")
        with self.connect() as db:
            row = db.execute("SELECT payload FROM recipes WHERE character_id=? AND recipe_id=?",
                             (character_id, recipe_id)).fetchone()
        if not row:
            raise CharacterError("Voice recipe does not exist for this character.")
        return VoiceRecipe.model_validate_json(row["payload"]).model_dump()

    def recipes(self, character_id: str) -> list[dict]:
        self.get(character_id)
        with self.connect() as db:
            ids = [row[0] for row in db.execute("SELECT recipe_id FROM recipes WHERE character_id=? ORDER BY revision",
                                              (character_id,))]
        return [self.get_recipe(character_id, identity) for identity in ids]

    def _verify_recipe(self, recipe: dict):
        item = VoiceRecipe.model_validate(recipe)
        if item.origin == "prebuilt":
            return
        if parse_timestamp(item.expires_at) <= datetime.now(timezone.utc):
            raise CharacterError("Designed voice expired; choose or design a new recipe.")
        request = json.loads(self._read_artifact(item.request_path, item.request_sha256), object_pairs_hook=_pairs)
        response = json.loads(self._read_artifact(item.response_path, item.response_sha256), object_pairs_hook=_pairs)
        audio = self._read_artifact(item.audition_path, item.audition_sha256)
        voice = request.get("voice", {})
        if (request.get("store") is not True or voice.get("type") != "prompted"
                or voice.get("model") != item.model or voice.get("prompted") != {"input": item.design_prompt}
                or voice.get("language_code") != item.language
                or response.get("id") != item.voice or response.get("type") != "prompted"):
            raise CharacterError("Designed voice provenance does not match its recipe.")
        from .gemini import parse_audio_block
        if parse_audio_block(response.get("sample_audio")) != audio:
            raise CharacterError("Designed audition does not match the original response.")

    def approved_designed_voices(self) -> set[str]:
        approved = set()
        for character in self.list():
            for recipe in self.recipes(character["character_id"]):
                if recipe["origin"] != "original_design":
                    continue
                try:
                    self._verify_recipe(recipe)
                except (CharacterError, ValueError, RuntimeError):
                    continue
                approved.add(recipe["voice"])
        return approved

    def bind_cast(self, script):
        from .gemini import Script
        bound = script.model_dump()
        snapshots = []
        for speaker in bound["speakers"]:
            character_id = speaker.get("character_id")
            if not character_id:
                if speaker.get("voice_recipe_id") or speaker.get("character_revision"):
                    raise CharacterError("A pinned voice recipe needs its character identity.")
                if speaker["voice"].startswith(("voice_", "voicekey_")):
                    raise CharacterError("Designed voices need a registered character and pinned recipe.")
                continue
            character = self.get(character_id, speaker.get("character_revision"))
            recipe_id = speaker.get("voice_recipe_id") or character["preferred_voice_recipe_id"]
            if not recipe_id:
                raise CharacterError("Character has no selected voice recipe.")
            recipe = self.get_recipe(character_id, recipe_id)
            self._verify_recipe(recipe)
            speaker.update(
                voice=recipe["voice"], recipe_version=recipe["revision"],
                design_prompt=recipe["design_prompt"] or "",
                pronunciation_notes=character["pronunciation_notes"],
                reference_recording=recipe["audition_path"],
                character_revision=character["revision"], voice_recipe_id=recipe["recipe_id"],
            )
            snapshots.append({"speaker": speaker["name"], "character": character, "voice_recipe": recipe})
        return Script.model_validate(bound), {"schema_version": 1, "speakers": snapshots}
