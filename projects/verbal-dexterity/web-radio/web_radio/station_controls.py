"""Private programming API; public station accepts commands only through its local handover."""
from __future__ import annotations

import fcntl
import hmac
import json
import os
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .contract import ContractError, Manifest, MANIFEST_LIMIT, SAFE_INTEGER, package_digest, validate_package
from .rundown import CATALOG_LIMIT, CATALOG_ENTRIES, CONTROL_LIMIT, ControlError, ReleaseRef, Rundown, parse_rundown

def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result

def _read(path: Path, limit: int):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected an ordinary local record")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Local record exceeds its supported size")
    return json.loads(raw, object_pairs_hook=_pairs)

def _write(path: Path, data):
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

class ProgrammingChange(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_revision: int = Field(ge=0, le=SAFE_INTEGER)
    mode: Literal["programmes", "fallback"]
    programme_releases: list[ReleaseRef] = Field(max_length=128)
    fallback_release: ReleaseRef | None

def _published(library: Path):
    catalog_path = library / "catalog.json"
    if not catalog_path.exists():
        return {}, []
    catalog = _read(catalog_path, CATALOG_LIMIT)
    if (not isinstance(catalog, dict) or set(catalog) != {"schema_version", "releases"}
            or type(catalog["schema_version"]) is not int or catalog["schema_version"] != 1
            or not isinstance(catalog["releases"], list) or len(catalog["releases"]) > CATALOG_ENTRIES):
        raise ValueError("Publication catalogue is invalid")
    releases = {}
    seen = set()
    problems = []
    for raw in catalog["releases"]:
        ref = ReleaseRef.model_validate(raw)
        if ref.release_id in seen:
            raise ValueError("Publication catalogue repeats a release")
        seen.add(ref.release_id)
        try:
            # Listing reads bounded metadata; selected packages undergo full byte
            # verification before command publication and again at the station.
            package = library / "releases" / ref.release_id
            if package.is_symlink():
                raise ValueError("Package is a symlink")
            manifest = Manifest.model_validate(_read(package / "manifest.json", MANIFEST_LIMIT))
            if (manifest.release_id != ref.release_id
                    or not hmac.compare_digest(manifest.package_sha256, ref.package_sha256)
                    or not hmac.compare_digest(package_digest(manifest), ref.package_sha256)):
                raise ValueError("Publication identity mismatch")
            releases[ref.release_id] = (ref, manifest, package)
        except (ValueError, OSError, ContractError):
            problems.append({"release_id": ref.release_id, "reason": "Release metadata is unavailable or invalid."})
    return releases, problems

def _revision_floor(current, acknowledgement: Path) -> int:
    revision = current.revision if current else 0
    if acknowledgement.exists():
        try:
            ack = _read(acknowledgement, CONTROL_LIMIT)
            if (isinstance(ack, dict) and type(ack.get("schema_version")) is int
                    and ack["schema_version"] == 1
                    and type(ack.get("accepted_revision")) is int
                    and 0 <= ack["accepted_revision"] <= SAFE_INTEGER):
                revision = max(revision, ack["accepted_revision"])
        except (OSError, ValueError):
            pass
    return revision

def make_router(project_root: Path) -> APIRouter:
    router = APIRouter(prefix="/api/programming", tags=["private-programming"])
    library = Path(project_root) / ".runtime" / "library"
    command = library / "station-control.json"
    acknowledgement = library / "station-control-status.json"

    @router.get("")
    def programming():
        problems = []
        try:
            releases, problems = _published(library)
        except (OSError, ValueError):
            releases = {}
            problems.append({"reason": "Publication catalogue is unavailable or invalid."})
        config = None
        if command.exists():
            try:
                with command.open("rb") as stream:
                    config = parse_rundown(stream.read(CONTROL_LIMIT + 1)).model_dump(mode="json")
            except (OSError, ValueError, ControlError):
                problems.append({"reason": "Existing programming record needs inspection."})
        ack = None
        if acknowledgement.exists():
            try:
                ack = _read(acknowledgement, CONTROL_LIMIT)
            except (OSError, ValueError):
                problems.append({"reason": "Station acknowledgement is unavailable."})
        return {
            "configuration": config, "revision": _revision_floor(Rundown.model_validate(config) if config else None, acknowledgement),
            "acknowledgement": ack, "problems": problems,
            "releases": [{"release_id": ref.release_id, "package_sha256": ref.package_sha256,
                          "title": manifest.title, "kind": manifest.kind,
                          "duration_ms": manifest.duration_ms, "language": manifest.language}
                         for ref, manifest, _ in releases.values()],
        }

    @router.put("")
    def apply(body: ProgrammingChange):
        library.mkdir(parents=True, exist_ok=True)
        with (library / ".station-control.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = None
            if command.exists():
                try:
                    with command.open("rb") as stream:
                        current = parse_rundown(stream.read(CONTROL_LIMIT + 1))
                except (OSError, ValueError, ControlError):
                    raise HTTPException(409, "Existing programming record is invalid and has been preserved.")
            revision = _revision_floor(current, acknowledgement)
            if body.expected_revision != revision:
                raise HTTPException(409, "Programming changed in another window. Refresh before applying.")
            if revision >= SAFE_INTEGER:
                raise HTTPException(409, "Programming revision limit reached; preserve and inspect local records.")
            try:
                candidate = Rundown(schema_version=1, revision=revision + 1, mode=body.mode,
                                    programme_releases=body.programme_releases,
                                    fallback_release=body.fallback_release)
                releases, _ = _published(library)
                selected = [(ref, "programme") for ref in candidate.programme_releases]
                if candidate.fallback_release:
                    selected.append((candidate.fallback_release, "ident"))
                for ref, kind in selected:
                    pair = releases.get(ref.release_id)
                    if not pair or pair[0] != ref:
                        raise ValueError("Choose a currently published release")
                    manifest = validate_package(pair[2], ref.package_sha256)
                    if manifest.kind != kind:
                        raise ValueError("Programme and fallback release roles must match")
            except (OSError, ValueError, ControlError, ContractError):
                raise HTTPException(422, "Programming contains an unavailable, invalid, or incorrectly assigned release.")
            _write(command, candidate.model_dump(mode="json"))
            return {"submitted": True, "revision": candidate.revision,
                    "applies": "next-airing-boundary"}
    return router

