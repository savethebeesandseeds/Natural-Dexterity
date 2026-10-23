"""Read-only authored character visuals for prepared listening performances.

Only canonical character.json templates are read. Registry revisions, provider
credentials, voice recipes and existing performance bundles are never changed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat

from pydantic import ValidationError

from .characters import CharacterInput, SLUG
from .performance import PerformanceError, performance_cast_from_characters

MAX_CHARACTER_BYTES = 64 * 1024
MAX_CHARACTERS = 512
MAX_DIRECTORIES = 4096


class CharacterVisualError(ValueError):
    """Authored character visuals cannot be selected safely."""


def _ordinary(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise CharacterVisualError("Character paths must not traverse symbolic links.")
    return path


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CharacterVisualError("Character JSON contains duplicate keys.")
        result[key] = value
    return result


def _finite_json(_value):
    raise CharacterVisualError("Character JSON contains a non-finite number.")


def _read_template(path: Path) -> dict:
    path = _ordinary(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_CHARACTER_BYTES:
                raise CharacterVisualError("Character JSON must be a bounded ordinary file.")
            data = source.read(MAX_CHARACTER_BYTES + 1)
    except OSError as exc:
        raise CharacterVisualError("Character JSON is unavailable.") from exc
    if len(data) > MAX_CHARACTER_BYTES:
        raise CharacterVisualError("Character JSON exceeds its byte limit.")
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_keys,
                           parse_constant=_finite_json)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CharacterVisualError("Character JSON is invalid.") from exc
    if not isinstance(value, dict):
        raise CharacterVisualError("Character JSON must contain an object.")
    return value


def _index_templates(root: Path) -> dict[str, dict]:
    root = _ordinary(root)
    if not root.is_dir():
        raise CharacterVisualError("An ordinary character template directory is required.")
    records = {}
    directories = 0

    def unavailable(error):
        raise CharacterVisualError("Character template directory is unavailable.") from error

    for directory, children, names in os.walk(root, followlinks=False, onerror=unavailable):
        directories += 1
        if directories > MAX_DIRECTORIES:
            raise CharacterVisualError("Character template directory exceeds its bound.")
        children.sort()
        for child in children:
            _ordinary(Path(directory) / child)
        if "character.json" not in names:
            continue
        if len(records) >= MAX_CHARACTERS:
            raise CharacterVisualError("Character template count exceeds its bound.")
        value = _read_template(Path(directory) / "character.json")
        character_id = value.get("character_id")
        if not isinstance(character_id, str) or re.fullmatch(SLUG, character_id) is None:
            raise CharacterVisualError("A canonical character template needs a stable identity.")
        if character_id in records:
            raise CharacterVisualError("Canonical character identities must be unique.")
        records[character_id] = value
    return records


def load_character_cast(characters_root: Path, character_ids) -> list[dict[str, str]]:
    """Select canonical templates in requested order and expose only id/color.

    Missing spiral_color uses the shared stable-ID fallback. Present colors and
    all selected character fields are validated. Retained templates in nested
    folders (including others/) participate in the same unique identity index.
    """
    if not isinstance(character_ids, (list, tuple)) or not 1 <= len(character_ids) <= 12:
        raise CharacterVisualError("Select between one and twelve character identities.")
    seen = set()
    for character_id in character_ids:
        if (not isinstance(character_id, str) or re.fullmatch(SLUG, character_id) is None
                or character_id in seen):
            raise CharacterVisualError("Requested character identities must be valid and unique.")
        seen.add(character_id)
    records = _index_templates(characters_root)
    selected = []
    for character_id in character_ids:
        if character_id not in records:
            raise CharacterVisualError("Requested character identity has no canonical template.")
        try:
            selected.append(CharacterInput.model_validate(records[character_id]).model_dump(mode="json"))
        except (ValidationError, ValueError, TypeError) as exc:
            raise CharacterVisualError("Selected character template is invalid.") from exc
    try:
        return performance_cast_from_characters(selected)
    except PerformanceError as exc:
        raise CharacterVisualError("Selected character visuals are invalid.") from exc
