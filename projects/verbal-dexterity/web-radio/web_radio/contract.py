"""The only accepted handover from the private studio to station-owned media.

Package identity binds canonical manifest metadata and every asset checksum.
The expected digest must arrive through the trusted publication channel.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import hmac
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Annotated, Literal
from uuid import uuid4

import rfc8785
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

HASH_PATTERN = r"^[0-9a-f]{64}$"
SLUG_PATTERN = r"^[a-z0-9][a-z0-9-]{0,127}$"
SAFE_INTEGER = 9_007_199_254_740_991
MANIFEST_LIMIT = 131_072
MAX_AUDIO_BYTES = 128 * 1024 * 1024
MAX_TRANSCRIPT_BYTES = 1024 * 1024
MAX_DURATION_MS = 30 * 60 * 1000
MANIFEST_NAME = "manifest.json"
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


class ContractError(ValueError):
    """A package cannot be accepted or safely published."""


def _safe_relative_path(value: str) -> str:
    if not value or "\\" in value or value.startswith("/") or len(value) > 240:
        raise ValueError("asset path must be a short relative POSIX path")
    components = value.split("/")
    for component in components:
        if (
            component in ("", ".", "..")
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", component)
            or component.endswith(".")
            or component.split(".")[0].upper() in _RESERVED_NAMES
        ):
            raise ValueError("asset path contains an unsafe component")
    if str(PurePosixPath(value)) != value or value == MANIFEST_NAME:
        raise ValueError("asset path is not canonical or shadows the manifest")
    return value


def _asset_schema_extra(schema: dict) -> None:
    path = schema["properties"]["path"]
    path["pattern"] = r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*$"
    path["allOf"] = [
        {"not": {"pattern": r"(^|/)[^/]*\.(?:/|$)"}},
        {"not": {"pattern": r"(^|/)([Cc][Oo][Nn]|[Pp][Rr][Nn]|[Aa][Uu][Xx]|[Nn][Uu][Ll]|[Cc][Oo][Mm][1-9]|[Ll][Pp][Tt][1-9])(?:\.[^/]*)?(/|$)"}},
        {"not": {"const": MANIFEST_NAME}},
    ]
    schema["allOf"] = [
        {
            "if": {"properties": {"role": {"const": role}}},
            "then": {"properties": {
                "mime_type": {"const": mime},
                "size_bytes": {"maximum": MAX_AUDIO_BYTES if role == "audio" else MAX_TRANSCRIPT_BYTES},
            }},
        }
        for role, mime in (("audio", "audio/wav"), ("transcript", "text/plain"))
    ]


def _manifest_schema_extra(schema: dict) -> None:
    schema["properties"]["title"]["pattern"] = r"\S"
    schema["properties"]["published_at"].update(
        pattern=r"^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d{1,6})?Z$",
        format="date-time",
    )
    schema["properties"]["assets"]["allOf"] = [
        {
            "contains": {"type": "object", "properties": {"role": {"const": role}}, "required": ["role"]},
            "minContains": 1,
            "maxContains": 1,
        }
        for role in ("audio", "transcript")
    ]


class Asset(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, json_schema_extra=_asset_schema_extra)

    role: Literal["audio", "transcript"]
    path: Annotated[str, Field(min_length=1, max_length=240)]
    mime_type: Literal["audio/wav", "text/plain"]
    size_bytes: Annotated[int, Field(gt=0, le=MAX_AUDIO_BYTES)]
    sha256: Annotated[str, Field(pattern=HASH_PATTERN)]

    @field_validator("path")
    @classmethod
    def check_path(cls, value: str) -> str:
        return _safe_relative_path(value)

    @model_validator(mode="after")
    def check_role_format(self) -> "Asset":
        expected = "audio/wav" if self.role == "audio" else "text/plain"
        if self.mime_type != expected:
            raise ValueError("asset MIME type does not match its role")
        if self.role == "transcript" and self.size_bytes > MAX_TRANSCRIPT_BYTES:
            raise ValueError("transcript exceeds the v1 1 MiB size limit")
        return self


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, json_schema_extra=_manifest_schema_extra)

    schema_version: Literal[1]
    programme_id: Annotated[str, Field(pattern=SLUG_PATTERN)]
    release_id: Annotated[str, Field(pattern=SLUG_PATTERN)]
    kind: Literal["programme", "ident"]
    title: Annotated[str, Field(min_length=1, max_length=300)]
    language: Annotated[str, Field(pattern=r"^[a-zA-Z]{2,3}(?:-[a-zA-Z0-9]{2,8})*$")]
    published_at: str
    duration_ms: Annotated[int, Field(gt=0, le=MAX_DURATION_MS)]
    assets: Annotated[list[Asset], Field(min_length=2, max_length=2)]
    package_sha256: Annotated[str, Field(pattern=HASH_PATTERN)]

    @field_validator("schema_version", mode="before")
    @classmethod
    def check_exact_version_type(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema version must be the integer 1")
        return value

    @field_validator("published_at")
    @classmethod
    def check_timestamp(cls, value: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z", value):
            raise ValueError("published_at must be a UTC timestamp ending in Z")
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("published_at is not a valid timestamp") from exc
        return value

    @field_validator("title")
    @classmethod
    def check_title(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title may not be blank")
        return value

    @model_validator(mode="after")
    def check_inventory(self) -> "Manifest":
        if {asset.role for asset in self.assets} != {"audio", "transcript"}:
            raise ValueError("exactly one audio asset and one transcript are required")
        if len({asset.path.casefold() for asset in self.assets}) != len(self.assets):
            raise ValueError("asset paths must be unique, including case")
        return self


def package_digest(manifest: Manifest | dict) -> str:
    """SHA-256 over RFC 8785 JSON with only package_sha256 omitted.

    Asset checksums are included, binding the exact bytes of all package assets.
    This is a content identity, not a publisher signature.
    """
    data = manifest.model_dump(mode="json") if isinstance(manifest, Manifest) else dict(manifest)
    data.pop("package_sha256", None)
    try:
        return hashlib.sha256(rfc8785.dumps(data)).hexdigest()
    except (ValueError, TypeError) as exc:
        raise ContractError("manifest cannot be represented as canonical JSON") from exc


def _sha256_file(path: Path, max_bytes: int = MAX_AUDIO_BYTES) -> str:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            total += len(chunk)
            if total > max_bytes:
                raise ContractError("asset exceeds its declared or v1 byte limit")
            digest.update(chunk)
    return digest.hexdigest()


def _copy_bounded(source: Path, destination: Path, max_bytes: int) -> None:
    total = 0
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
            total += len(chunk)
            if total > max_bytes:
                raise ContractError("asset exceeds its declared or v1 byte limit")
            outgoing.write(chunk)


def _read_transcript(path: Path) -> str:
    try:
        with path.open("rb") as source:
            data = source.read(MAX_TRANSCRIPT_BYTES + 1)
        if len(data) > MAX_TRANSCRIPT_BYTES:
            raise ContractError("transcript exceeds the v1 1 MiB size limit")
        text = data.decode("utf-8")
    except (UnicodeError, OSError) as exc:
        raise ContractError("transcript is not valid UTF-8 text") from exc
    if not text.strip() or "\x00" in text:
        raise ContractError("transcript must contain ordinary nonempty text")
    return text


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ContractError(f"non-finite JSON number is forbidden: {value}")


def _read_manifest(path: Path) -> tuple[Manifest, dict]:
    if path.is_symlink() or not path.is_file():
        raise ContractError("manifest must be a regular file")
    if path.stat().st_size > MANIFEST_LIMIT:
        raise ContractError("manifest exceeds the v1 size limit")
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
        )
        if not isinstance(raw, dict):
            raise ContractError("manifest must be a JSON object")
        manifest = Manifest.model_validate(raw)
    except (UnicodeError, json.JSONDecodeError, ValidationError, OSError) as exc:
        raise ContractError(f"invalid manifest: {exc.__class__.__name__}") from exc
    return manifest, raw


def _audio_duration_ms(path: Path) -> int:
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-f", "wav", "-show_entries",
                "stream=codec_type,codec_name,channels,sample_rate:format=format_name,duration",
                "-of", "json", str(path),
            ],
            check=True, capture_output=True, text=True, timeout=30,
        )
        data = json.loads(probe.stdout)
        streams = data.get("streams", [])
        if (
            len(streams) != 1
            or streams[0].get("codec_type") != "audio"
            or streams[0].get("codec_name") != "pcm_s16le"
            or streams[0].get("channels") not in (1, 2)
            or data.get("format", {}).get("format_name") != "wav"
        ):
            raise ContractError("v1 audio must be a single signed-16-bit PCM WAV stream with 1 or 2 channels")
        sample_rate = int(streams[0].get("sample_rate", 0))
        if not 8000 <= sample_rate <= 192000:
            raise ContractError("unsupported audio sample rate")
        duration = Decimal(data["format"]["duration"])
        if not duration.is_finite() or duration <= 0:
            raise ContractError("audio must have a positive finite duration")
        duration_ms = int((duration * 1000).to_integral_value())
        if duration_ms <= 0:
            raise ContractError("audio is too short")
        if duration_ms > MAX_DURATION_MS:
            raise ContractError("audio exceeds the v1 30 minute duration limit")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-xerror", "-protocol_whitelist", "file,pipe", "-f", "wav",
             "-i", str(path), "-map", "0:a:0", "-f", "null", "-"],
            check=True, capture_output=True, timeout=120,
        )
        return duration_ms
    except (subprocess.SubprocessError, OSError, ValueError, KeyError, InvalidOperation) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"audio could not be fully decoded: {exc.__class__.__name__}") from exc


def _inventory(root: Path) -> set[str]:
    result: set[str] = set()
    try:
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ContractError("package may not contain symbolic links")
            if path.is_file():
                result.add(path.relative_to(root).as_posix())
            elif not path.is_dir():
                raise ContractError("package may contain only ordinary files and directories")
    except OSError as exc:
        raise ContractError("package inventory could not be inspected") from exc
    return result


def validate_package(path: str | Path, expected_digest: str | None = None) -> Manifest:
    """Validate manifest structure, exact inventory, file bytes, and decoded audio.

    A self-consistent package is not automatically an approved publication.
    Station ingestion must also receive an expected digest from its trusted
    publication record.
    """
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ContractError("package must be an ordinary directory")
    manifest, raw = _read_manifest(root / MANIFEST_NAME)
    computed_digest = package_digest(raw)
    if not hmac.compare_digest(computed_digest, manifest.package_sha256):
        raise ContractError("package digest does not match manifest metadata")
    if expected_digest is not None:
        if not isinstance(expected_digest, str) or not re.fullmatch(HASH_PATTERN, expected_digest):
            raise ContractError("expected digest must be a lowercase SHA-256")
        if not hmac.compare_digest(manifest.package_sha256, expected_digest):
            raise ContractError("package does not match the trusted publication digest")
    declared = {asset.path for asset in manifest.assets} | {MANIFEST_NAME}
    if _inventory(root) != declared:
        raise ContractError("package inventory differs from the declared assets")
    audio_path: Path | None = None
    for asset in manifest.assets:
        file_path = root / asset.path
        if file_path.stat().st_size != asset.size_bytes:
            raise ContractError(f"asset byte size mismatch: {asset.path}")
        if not hmac.compare_digest(_sha256_file(file_path, max_bytes=asset.size_bytes), asset.sha256):
            raise ContractError(f"asset checksum mismatch: {asset.path}")
        if asset.role == "audio":
            audio_path = file_path
        else:
            _read_transcript(file_path)
    assert audio_path is not None
    measured = _audio_duration_ms(audio_path.resolve())
    if abs(measured - manifest.duration_ms) > 100:
        raise ContractError("manifest duration differs from decoded audio by more than 100 ms")
    return manifest


@contextlib.contextmanager
def _publication_lock(parent: Path):
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / ".publication.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _publish_directory(staging: Path, final: Path) -> None:
    # All publishers serialize the existence check and same-filesystem rename.
    with _publication_lock(final.parent):
        if final.exists() or final.is_symlink():
            raise ContractError("release destination already exists; releases are immutable")
        os.rename(staging, final)


def create_release(
    source_audio: str | Path,
    transcript: str,
    output_dir: str | Path,
    programme_id: str,
    title: str,
    language: str = "en",
    kind: Literal["programme", "ident"] = "programme",
    release_id: str | None = None,
) -> Manifest:
    """Build and atomically publish a new immutable release directory.

    output_dir is the final package directory, not a library root.
    source_audio must already be a supported PCM WAV mix.
    """
    source = Path(source_audio)
    final = Path(output_dir)
    if source.is_symlink() or not source.is_file():
        raise ContractError("source audio must be a regular file")
    if not 0 < source.stat().st_size <= MAX_AUDIO_BYTES:
        raise ContractError("source audio exceeds the v1 128 MiB size limit or is empty")
    if not isinstance(transcript, str) or not transcript.strip() or "\x00" in transcript:
        raise ContractError("transcript must be nonempty text")
    try:
        transcript_bytes = transcript.encode("utf-8")
    except UnicodeError as exc:
        raise ContractError("transcript is not valid UTF-8 text") from exc
    if len(transcript_bytes) > MAX_TRANSCRIPT_BYTES:
        raise ContractError("transcript exceeds the v1 1 MiB size limit")
    if final.exists() or final.is_symlink():
        raise ContractError("release destination already exists; releases are immutable")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=final.parent))
    try:
        _copy_bounded(source, staging / "audio.wav", MAX_AUDIO_BYTES)
        (staging / "transcript.txt").write_bytes(transcript_bytes)
        duration_ms = _audio_duration_ms((staging / "audio.wav").resolve())
        assets = [
            Asset(
                role=role, path=asset_path, mime_type=mime_type,
                size_bytes=(staging / asset_path).stat().st_size,
                sha256=_sha256_file(staging / asset_path),
            )
            for role, asset_path, mime_type in (
                ("audio", "audio.wav", "audio/wav"),
                ("transcript", "transcript.txt", "text/plain"),
            )
        ]
        manifest = Manifest(
            schema_version=1, programme_id=programme_id,
            release_id=release_id or f"{programme_id[:100]}-{uuid4().hex[:16]}",
            kind=kind, title=title, language=language,
            published_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            duration_ms=duration_ms, assets=assets, package_sha256="0" * 64,
        )
        manifest = manifest.model_copy(update={"package_sha256": package_digest(manifest)})
        (staging / MANIFEST_NAME).write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8",
        )
        validate_package(staging, manifest.package_sha256)
        _publish_directory(staging, final)
        return manifest
    except ValidationError as exc:
        raise ContractError("release metadata fails the v1 contract") from exc
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def import_release(
    source: str | Path,
    destination_root: str | Path,
    expected_digest: str,
) -> tuple[Manifest, Path]:
    """Copy a trusted publication into station-owned storage, then verify it.

    The copied bytes are verified before the directory becomes schedulable.
    Reimporting the same release and digest is idempotent; identity conflicts fail.
    """
    if not expected_digest:
        raise ContractError("station import requires a trusted publication digest")
    source_path = Path(source)
    manifest = validate_package(source_path, expected_digest)
    destination = Path(destination_root)
    destination.mkdir(parents=True, exist_ok=True)
    final = destination / manifest.release_id
    if final.exists() or final.is_symlink():
        existing = validate_package(final, expected_digest)
        return existing, final
    staging = Path(tempfile.mkdtemp(prefix=f".{manifest.release_id}.import-", dir=destination))
    try:
        for asset in manifest.assets:
            target = staging / asset.path
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_bounded(source_path / asset.path, target, asset.size_bytes)
        _copy_bounded(source_path / MANIFEST_NAME, staging / MANIFEST_NAME, MANIFEST_LIMIT)
        imported = validate_package(staging, expected_digest)
        with _publication_lock(destination):
            if final.exists() or final.is_symlink():
                return validate_package(final, expected_digest), final
            os.rename(staging, final)
        return imported, final
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def write_schema(path: str | Path) -> None:
    schema = Manifest.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["title"] = "Web Radio release manifest v1"
    schema["description"] = (
        "Structure only. Package acceptance additionally verifies inventory, hashes, "
        "safe paths, supported PCM WAV audio, decoded duration, and publication trust."
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    schema = commands.add_parser("schema")
    schema.add_argument("output")
    validate = commands.add_parser("validate")
    validate.add_argument("package")
    validate.add_argument("--expected-digest")
    args = parser.parse_args()
    if args.command == "schema":
        write_schema(args.output)
    else:
        print(validate_package(args.package, args.expected_digest).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
