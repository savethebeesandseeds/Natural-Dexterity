"""Optional prepared character cues/features, separate from immutable audio releases.

This module does not infer speakers, generate audio, or change playout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import tempfile
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .contract import HASH_PATTERN, MAX_DURATION_MS, SLUG_PATTERN, Manifest, validate_package

MAX_PERFORMANCE_BYTES = 8 * 1024 * 1024
MAX_FEATURE_VALUES = 500_000
MAX_CUES = 100_000
Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Slug = Annotated[str, Field(pattern=SLUG_PATTERN)]


class PerformanceError(ValueError):
    """Prepared performance data is invalid or does not match the release."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class Character(StrictModel):
    id: Slug
    color: Annotated[str, Field(pattern=r"^#[0-9A-Fa-f]{6}$")]


def _fallback_character_color(character_id: str) -> str:
    """Match RadioSpirits' stable identity palette for older character records."""
    identity_hash = 2166136261
    for character in character_id:
        identity_hash = ((identity_hash ^ ord(character)) * 16777619) & 0xffffffff
    hue = (identity_hash % 360) / 60
    chroma = .30
    middle = chroma * (1 - abs(hue % 2 - 1))
    segments = (
        (chroma, middle, 0), (middle, chroma, 0),
        (0, chroma, middle), (0, middle, chroma),
        (middle, 0, chroma), (chroma, 0, middle),
    )
    return "#" + "".join(
        f"{math.floor((component + .28) * 255 + .5):02x}"
        for component in segments[math.floor(hue)]
    )


def performance_cast_from_characters(characters) -> list[dict[str, str]]:
    """Extract public identity/colors from selected character definitions.

    The caller selects authored definitions or frozen revisions before preparation.
    No registry is opened and no existing performance document is changed. Missing
    or null optional colors use the same stable-ID palette as the renderer.
    Explicit invalid colors fail rather than silently falling back.
    """
    if not isinstance(characters, (list, tuple)) or not 1 <= len(characters) <= 12:
        raise PerformanceError("Supply between one and twelve character definitions.")
    result, seen = [], set()
    for character in characters:
        if not isinstance(character, dict):
            raise PerformanceError("Each character definition must be an object.")
        character_id = character.get("character_id")
        if not isinstance(character_id, str) or re.fullmatch(SLUG_PATTERN, character_id) is None:
            raise PerformanceError("A stable character ID is required for visual performance.")
        if character_id in seen:
            raise PerformanceError("Visual performance character IDs must be unique.")
        seen.add(character_id)
        color = character.get("spiral_color")
        if color is None:
            color = _fallback_character_color(character_id)
        try:
            item = Character.model_validate({"id": character_id, "color": color})
        except (ValidationError, TypeError, ValueError) as exc:
            raise PerformanceError("Character spiral_color must be a #RRGGBB color.") from exc
        result.append(item.model_dump())
    return result


def cast_from_snapshot(snapshot) -> list[dict[str, str]]:
    """Resolve only public colors from one already-frozen bound cast snapshot."""
    if (not isinstance(snapshot, dict) or type(snapshot.get("schema_version")) is not int
            or snapshot["schema_version"] != 1 or not isinstance(snapshot.get("speakers"), list)
            or not 1 <= len(snapshot["speakers"]) <= 12):
        raise PerformanceError("Supply a version-one bound cast with one to twelve speakers.")
    characters = []
    for speaker in snapshot["speakers"]:
        if not isinstance(speaker, dict) or not isinstance(speaker.get("character"), dict):
            raise PerformanceError("Bound speakers must retain their character snapshots.")
        characters.append(speaker["character"])
    return performance_cast_from_characters(characters)


class Cue(StrictModel):
    speaker_id: Slug
    start_ms: Annotated[int, Field(ge=0, le=MAX_DURATION_MS)]
    end_ms: Annotated[int, Field(gt=0, le=MAX_DURATION_MS)]


class Track(StrictModel):
    energy: list[Unit]
    low: list[Unit]
    mid: list[Unit]
    high: list[Unit]


class Features(StrictModel):
    hop_ms: Annotated[int, Field(ge=20, le=1000)]
    tracks: Annotated[dict[Slug, Track], Field(min_length=1, max_length=12)]


class Performance(StrictModel):
    schema_version: Literal[1]
    release_id: Slug
    package_sha256: Annotated[str, Field(pattern=HASH_PATTERN)]
    duration_ms: Annotated[int, Field(gt=0, le=MAX_DURATION_MS)]
    cast: Annotated[list[Character], Field(min_length=1, max_length=12)]
    cues: Annotated[list[Cue], Field(max_length=MAX_CUES)]
    features: Features | None = None

    @model_validator(mode="before")
    @classmethod
    def bound_raw_features(cls, value):
        if not isinstance(value, dict):
            return value
        if type(value.get("schema_version")) is not int:
            raise ValueError("schema_version must be the integer 1")
        features = value.get("features")
        tracks = features.get("tracks") if isinstance(features, dict) else None
        if isinstance(tracks, dict):
            total = 0
            for track in tracks.values():
                if isinstance(track, dict):
                    for band in ("energy", "low", "mid", "high"):
                        samples = track.get(band)
                        if isinstance(samples, list):
                            total += len(samples)
                            if total > MAX_FEATURE_VALUES:
                                raise ValueError("too many feature values")
        return value

    @model_validator(mode="after")
    def check_timeline(self):
        ids = [character.id for character in self.cast]
        if len(set(ids)) != len(ids):
            raise ValueError("character IDs must be unique")
        known = set(ids)
        previous_start = -1
        last_end: dict[str, int] = {}
        for cue in self.cues:
            if cue.speaker_id not in known:
                raise ValueError("cue references an unknown character")
            if not 0 <= cue.start_ms < cue.end_ms <= self.duration_ms:
                raise ValueError("cue must be a nonempty half-open interval within the release")
            if cue.start_ms < previous_start:
                raise ValueError("cues must be sorted by start_ms")
            if cue.start_ms < last_end.get(cue.speaker_id, 0):
                raise ValueError("a character's own cues may not overlap")
            previous_start = cue.start_ms
            last_end[cue.speaker_id] = cue.end_ms
        if self.features is not None:
            expected = math.ceil(self.duration_ms / self.features.hop_ms) + 1
            for character_id, track in self.features.tracks.items():
                if character_id not in known:
                    raise ValueError("feature track references an unknown character")
                if any(len(getattr(track, band)) != expected for band in ("energy", "low", "mid", "high")):
                    raise ValueError("feature arrays must cover the release at the declared hop")
        return self


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PerformanceError("duplicate JSON key")
        result[key] = value
    return result


def _finite_json(_value):
    raise PerformanceError("non-finite JSON number")


def validate_document(data: bytes, manifest: Manifest) -> Performance:
    if not data or len(data) > MAX_PERFORMANCE_BYTES:
        raise PerformanceError("performance document exceeds its byte limit")
    try:
        raw = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_keys, parse_constant=_finite_json)
        performance = Performance.model_validate(raw)
    except (UnicodeError, json.JSONDecodeError, ValidationError, TypeError, ValueError, RecursionError) as exc:
        raise PerformanceError("invalid performance document") from exc
    if (performance.release_id != manifest.release_id
            or performance.package_sha256 != manifest.package_sha256
            or performance.duration_ms != manifest.duration_ms):
        raise PerformanceError("performance does not match the accepted release")
    normalized = json.dumps(performance.model_dump(mode="json", exclude_none=True),
                            separators=(",", ":"), allow_nan=False).encode()
    if len(normalized) > MAX_PERFORMANCE_BYTES:
        raise PerformanceError("normalized performance exceeds its byte limit")
    return performance


def _read_bounded(path: Path) -> bytes:
    path = Path(path)
    if path.parent.is_symlink():
        raise PerformanceError("performance directory must be ordinary")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_PERFORMANCE_BYTES:
                raise PerformanceError("performance must be a bounded regular file")
            data = source.read(MAX_PERFORMANCE_BYTES + 1)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise PerformanceError("performance is unavailable") from exc
    if len(data) > MAX_PERFORMANCE_BYTES:
        raise PerformanceError("performance document exceeds its byte limit")
    return data


def read_performance(path: Path, manifest: Manifest) -> Performance:
    return validate_document(_read_bounded(path), manifest)


def install_performance(source: Path, runtime: Path, *, replace: bool = False) -> Path:
    """Validate against a previously accepted release and atomically install.

    This function never adds to the release package or imports a new release.
    """
    source = Path(source)
    runtime = Path(runtime)
    data = _read_bounded(source)
    # Read the bounded document's identity before opening the accepted journal.
    try:
        raw = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_keys, parse_constant=_finite_json)
        release_id = raw.get("release_id") if isinstance(raw, dict) else None
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise PerformanceError("invalid performance document") from exc
    if not isinstance(release_id, str) or re.fullmatch(SLUG_PATTERN, release_id) is None:
        raise PerformanceError("invalid release identity")
    station = runtime / "station"
    journal = station / "journal.sqlite3"
    if journal.is_symlink() or not journal.is_file():
        raise PerformanceError("an existing accepted station release is required")
    try:
        with sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True) as db:
            row = db.execute("SELECT digest FROM accepted WHERE release_id=?", (release_id,)).fetchone()
        if row is None:
            raise PerformanceError("release has not been accepted")
        manifest = validate_package(station / "releases" / release_id, expected_digest=row[0])
    except (sqlite3.Error, OSError, ValueError) as exc:
        raise PerformanceError("accepted release could not be verified") from exc
    performance = validate_document(data, manifest)
    encoded = (json.dumps(performance.model_dump(mode="json", exclude_none=True),
                          separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(encoded) > MAX_PERFORMANCE_BYTES:
        raise PerformanceError("normalized performance exceeds its byte limit")
    destination = station / "performance"
    if destination.is_symlink():
        raise PerformanceError("performance directory must be ordinary")
    destination.mkdir(parents=False, exist_ok=True)
    target = destination / (release_id + ".json")
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise PerformanceError("existing performance is not an ordinary file")
    if target.exists() and not replace:
        raise PerformanceError("performance already exists; explicit --replace is required")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix="." + release_id + "-", suffix=".tmp",
                                         dir=destination, delete=False) as output:
            temporary = Path(output.name)
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        if replace:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise PerformanceError("performance already exists; explicit --replace is required") from exc
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("install", help="Install authored cues/features for an accepted release")
    install.add_argument("source", type=Path)
    install.add_argument("--runtime", required=True, type=Path)
    install.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    try:
        target = install_performance(args.source, args.runtime, replace=args.replace)
    except (PerformanceError, OSError) as exc:
        parser.exit(2, f"Performance was not installed: {exc.__class__.__name__}.\n")
    print("Installed prepared performance for", target.stem)


if __name__ == "__main__":
    main()
