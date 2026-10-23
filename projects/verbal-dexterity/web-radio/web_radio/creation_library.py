"""Read-only listening access to retained local creations, without publication."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import threading
import wave

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from .character_visuals import load_character_cast
from .contract import HASH_PATTERN, MAX_AUDIO_BYTES, MAX_DURATION_MS, SLUG_PATTERN
from .performance import MAX_CUES, MAX_PERFORMANCE_BYTES, Performance, cast_from_snapshot, performance_cast_from_characters
from .preview import (
    MAX_SEMANTIC_BYTES, PREVIEW_HEADERS, PreviewManifest, _json, _open, _performance,
    _read, _semantic, read_preview,
)

MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_DIRECTORIES = 4096
MAX_DIRECTORY_ENTRIES = 16384
MAX_CREATIONS = 512
MAX_PREVIEWS = 512
ID_PATTERN = r"^creation-[0-9a-f]{24}$"
PROJECT_ID_PATTERN = r"^project-[0-9a-f]{24}$"
PODCAST_KINDS = {"podcast_production", "complete_podcast_production"}


class CreationLibraryError(ValueError):
    """A retained listening source cannot be safely verified."""


class VisualPreparationRequired(CreationLibraryError):
    """A programme is retained, but its verified five-dimensional visuals are absent."""


def _ordinary(path):
    path = Path(os.path.abspath(path))
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise CreationLibraryError("Creation paths must be ordinary.")
    return path


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _identity(path, project):
    return "creation-" + hashlib.sha256(path.relative_to(project).as_posix().encode()).hexdigest()[:24]


def _project_identity(root, project):
    parts = Path(root).relative_to(Path(project) / "creations").parts
    if len(parts) not in (3, 4):
        raise CreationLibraryError("Project selection requires a canonical creation.")
    return "project-" + hashlib.sha256(("project:" + parts[-3]).encode()).hexdigest()[:24]


def _version_number(root_or_name):
    matched = re.fullmatch(r"v([0-9]+)", Path(root_or_name).name)
    return int(matched.group(1)) if matched is not None else None


def latest_project_creation(root, project):
    """Resolve a curated project's highest folder version without preparing it."""
    project = Path(project)
    return _Library(project, project / ".runtime").latest_project(_project_identity(root, project))


def _fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _relative(relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative or "\0" in relative
            or PurePosixPath(relative).is_absolute()
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise CreationLibraryError("Creation reference must stay within its bundle.")
    return relative


def _safe(root, relative):
    # _read/_open verify every parent and open the final file with O_NOFOLLOW.
    return root / _relative(relative)


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(HASH_PATTERN, value) is None:
        raise CreationLibraryError("Creation assets require an exact checksum.")
    return value


class _Library:
    def __init__(self, project_root, runtime):
        self.project = _ordinary(project_root)
        self.runtime = _ordinary(runtime)
        self.creations = _ordinary(self.project / "creations")
        self.audio_cache = OrderedDict()
        self.preview_cache = OrderedDict()
        self.cache_lock = threading.Lock()

    def roots(self):
        if not self.creations.is_dir():
            return []
        pending, roots, seen, entries = [(self.creations, 0)], [], 0, 0
        while pending:
            directory, depth = pending.pop()
            seen += 1
            if seen > MAX_DIRECTORIES:
                raise CreationLibraryError("Creation directory count exceeds its bound.")
            with os.scandir(_ordinary(directory)) as listing:
                children = []
                for item in listing:
                    entries += 1
                    if entries > MAX_DIRECTORY_ENTRIES:
                        raise CreationLibraryError("Creation directory exceeds its entry bound.")
                    if item.is_symlink():
                        continue
                    if item.is_dir(follow_symlinks=False) and depth < 4:
                        children.append(Path(item.path))
            if (depth in (3, 4) and not (directory / "production.json").is_symlink()
                    and (directory / "production.json").is_file()):
                _ordinary(directory / "production.json")
                roots.append(directory)
                if len(roots) > MAX_CREATIONS:
                    raise CreationLibraryError("Creation count exceeds its bound.")
                continue
            if depth < 4:
                pending.extend((child, depth + 1) for child in sorted(children, reverse=True))
        return sorted(roots)

    def _project_candidates(self):
        groups, auditions = {}, []
        for root in self.roots():
            try:
                kind = self._json(root / "production.json").get("kind")
            except (OSError, ValueError, TypeError, RecursionError):
                # A malformed newest vN must prevent an older version from
                # silently winning. Canonical audition folders remain separate.
                kind = None
            if kind == "voice_audition":
                auditions.append(root)
                continue
            version = _version_number(root)
            if version is None or root.relative_to(self.creations).parts[-3] == "voice-auditions":
                continue
            groups.setdefault(_project_identity(root, self.project), []).append((version, root))
        return groups, auditions

    def _selected_projects(self):
        groups, auditions = self._project_candidates()
        selected = {}
        for identifier, candidates in groups.items():
            highest = max(version for version, _ in candidates)
            selected[identifier] = [root for version, root in candidates if version == highest]
        return selected, auditions

    def latest_project(self, identifier):
        """Return the unique highest numeric vN; ties fail closed for this project."""
        if not isinstance(identifier, str) or re.fullmatch(PROJECT_ID_PATTERN, identifier) is None:
            raise FileNotFoundError()
        projects, _ = self._selected_projects()
        roots = projects.get(identifier)
        if roots is None:
            raise FileNotFoundError()
        if len(roots) != 1:
            raise CreationLibraryError("The project's highest version is ambiguous.")
        return roots[0]

    def catalogue_roots(self):
        projects, auditions = self._selected_projects()
        return sorted(auditions + [roots[0] for roots in projects.values() if len(roots) == 1])

    def find(self, identifier, *, allow_project=True):
        if isinstance(identifier, str) and re.fullmatch(PROJECT_ID_PATTERN, identifier) is not None:
            if not allow_project:
                raise FileNotFoundError()
            return self.latest_project(identifier)
        if not isinstance(identifier, str) or re.fullmatch(ID_PATTERN, identifier) is None:
            raise FileNotFoundError()
        for root in self.roots():
            if _identity(root, self.project) == identifier:
                return root
        raise FileNotFoundError()

    def _json(self, path):
        return _json(_read(path, MAX_JSON_BYTES))

    def _inventory(self, root):
        raw = _read(root / "manifest.json", MAX_JSON_BYTES)
        manifest = _json(raw)
        rows = manifest.get("files")
        if not isinstance(rows, list) or not 1 <= len(rows) <= 16384:
            raise CreationLibraryError("Creation inventory is unavailable.")
        entries = {}
        for row in rows:
            if not isinstance(row, dict):
                raise CreationLibraryError("Invalid creation inventory.")
            relative = row.get("path")
            _relative(relative)
            if relative in entries or type(row.get("bytes")) is not int or not 0 <= row["bytes"] <= MAX_AUDIO_BYTES:
                raise CreationLibraryError("Invalid creation inventory binding.")
            _digest(row.get("sha256"))
            entries[relative] = row
        return entries, hashlib.sha256(raw).hexdigest()

    def _bound(self, root, entries, relative, reference=None, *, json_document=True):
        path = _safe(root, relative)
        entry = entries.get(relative)
        if entry is None:
            raise CreationLibraryError("A listening source is absent from the creation inventory.")
        if reference is not None and (not isinstance(reference, dict)
                or reference.get("path") != relative
                or reference.get("sha256", entry["sha256"]) != entry["sha256"]):
            raise CreationLibraryError("A listening reference differs from the creation inventory.")
        if not json_document:
            return path, entry["sha256"], entry["bytes"]
        data = _read(path, MAX_JSON_BYTES)
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise CreationLibraryError("Creation metadata checksum differs.")
        return _json(data)

    def _audio(self, path, expected, expected_size=None, *, verify=False):
        with _open(path) as source:
            initial = os.fstat(source.fileno())
            if not 0 < initial.st_size <= MAX_AUDIO_BYTES or (expected_size is not None and initial.st_size != expected_size):
                raise CreationLibraryError("Creation audio size differs.")
            cache_key = (str(path), _fingerprint(initial), expected)
            with self.cache_lock:
                cached = self.audio_cache.get(cache_key)
            if cached is not None:
                return cached
            try:
                with wave.open(source, "rb") as audio:
                    params = audio.getparams()
                    if (params.comptype != "NONE" or params.nchannels not in (1, 2)
                            or params.sampwidth not in (2, 3, 4) or not 8000 <= params.framerate <= 192000
                            or params.nframes <= 0):
                        raise CreationLibraryError("Creation audio must be ordinary PCM WAV.")
                    if source.tell() + params.nframes * params.nchannels * params.sampwidth > initial.st_size:
                        raise CreationLibraryError("Creation audio frames are incomplete.")
                    if verify:
                        remaining = params.nframes
                        while remaining:
                            count = min(65536, remaining)
                            if len(audio.readframes(count)) != count * params.nchannels * params.sampwidth:
                                raise CreationLibraryError("Creation audio frames are incomplete.")
                            remaining -= count
                duration = round(params.nframes * 1000 / params.framerate)
                if not 0 < duration <= MAX_DURATION_MS:
                    raise CreationLibraryError("Creation audio duration exceeds its bound.")
                result = {"frames": params.nframes, "rate": params.framerate,
                          "duration_ms": duration, "final_ms": params.nframes * 1000 // params.framerate,
                          "size_bytes": initial.st_size}
                if verify:
                    source.seek(0)
                    digest = hashlib.sha256()
                    while chunk := source.read(1024 * 1024):
                        digest.update(chunk)
                    if digest.hexdigest() != expected or _fingerprint(os.fstat(source.fileno())) != _fingerprint(initial):
                        raise CreationLibraryError("Creation audio checksum differs.")
                    with self.cache_lock:
                        self.audio_cache[cache_key] = result
                        self.audio_cache.move_to_end(cache_key)
                        while len(self.audio_cache) > MAX_CREATIONS:
                            self.audio_cache.popitem(last=False)
                return result
            except (wave.Error, EOFError) as exc:
                raise CreationLibraryError("Creation audio is not complete PCM WAV.") from exc

    def entry(self, root, *, verify=False):
        production = self._json(root / "production.json")
        kind = production.get("kind")
        depth = len(root.relative_to(self.creations).parts)
        title = production.get("title")
        if (type(production.get("schema_version")) is not int or not isinstance(title, str)
                or not title.strip() or len(title) > 300 or "\0" in title):
            raise CreationLibraryError("Creation listening metadata is invalid.")
        entries, source_sha = None, None
        if kind == "voice_audition" and depth == 3:
            reference = production.get("native_sample")
            if not isinstance(reference, dict) or reference.get("path") != "audio/provider-sample.wav":
                raise CreationLibraryError("Audition has no bound native sample.")
            audio = _safe(root, "audio/provider-sample.wav")
            expected, size = _digest(reference.get("sha256")), None
            public_kind = "voice_audition"
        elif kind in PODCAST_KINDS | {"complete_creative_production"} and depth in (3, 4):
            entries, source_sha = self._inventory(root)
            self._bound(root, entries, "production.json")
            role = "master" if kind in PODCAST_KINDS else "lossless_master"
            reference = production.get("deliverables", {}).get(role)
            if not isinstance(reference, dict):
                raise CreationLibraryError("A declared retained master is required.")
            audio, expected, size = self._bound(root, entries, "audio/master.wav", reference, json_document=False)
            public_kind = "episode"
        else:
            raise CreationLibraryError("Creation has no complete listening source.")
        measured = self._audio(audio, expected, size, verify=verify)
        declared_duration = production.get("duration_ms")
        if declared_duration is not None and declared_duration != measured["duration_ms"]:
            raise CreationLibraryError("Creation duration differs from its audio.")
        if public_kind == "episode":
            # Both curated project/work/version and legacy series/project/work/
            # version bundles display the tool name without rewriting evidence.
            title = root.relative_to(self.creations).parts[-3]
        identifier = _identity(root, self.project)
        return dict(root=root, production=production, entries=entries, source_sha=source_sha,
                    audio=audio, audio_sha=expected, measured=measured, id=identifier,
                    title=title, version=root.name, kind=public_kind)

    def performance(self, entry):
        root, production, measured = entry["root"], entry["production"], entry["measured"]
        if entry["kind"] == "voice_audition":
            identity = production.get("character_id")
            # Canonical visuals are permitted for a one-character audition; no
            # diarization or programme cast is inferred from the waveform.
            try:
                cast = load_character_cast(self.project / "characters", [identity])
            except (ValueError, OSError):
                cast = performance_cast_from_characters([{"character_id": identity}])
            cues = [{"speaker_id": identity, "start_ms": 0, "end_ms": measured["final_ms"]}]
        else:
            entries = entry["entries"]
            kind = production["kind"]
            voices_ref = (production.get("voices") if kind in PODCAST_KINDS
                          else production.get("casting", {}).get("voices"))
            if not isinstance(voices_ref, dict):
                raise CreationLibraryError("A frozen voice binding is required.")
            voices = self._bound(root, entries, "voices.json", voices_ref)
            if voices.get("schema_version") == 1:
                cast = cast_from_snapshot(voices)
            elif voices.get("schema_version") == 2 and isinstance(voices.get("voices"), list):
                characters = []
                for row in voices["voices"]:
                    reference = row.get("character_snapshot")
                    if not isinstance(reference, dict):
                        raise CreationLibraryError("Frozen character snapshot is missing.")
                    character = self._bound(root, entries, reference.get("path"), reference)
                    if character.get("character_id") != row.get("character_id"):
                        raise CreationLibraryError("Frozen character attribution differs.")
                    characters.append(character)
                cast = performance_cast_from_characters(characters)
            else:
                raise CreationLibraryError("Frozen voices are unavailable.")
            role = "performance" if kind in PODCAST_KINDS else "timing"
            timing_ref = production.get("deliverables", {}).get(role)
            if not isinstance(timing_ref, dict):
                raise CreationLibraryError("A declared character timing binding is required.")
            timing = self._bound(root, entries, "performance.json", timing_ref)
            if (timing.get("sample_rate") != measured["rate"]
                    or timing.get("duration_ms") != measured["duration_ms"]
                    or timing.get("audio_sha256", entry["audio_sha"]) != entry["audio_sha"]
                    or timing.get("total_frames", measured["frames"]) != measured["frames"]):
                raise CreationLibraryError("Character timing differs from the retained audio.")
            raw_cues = timing.get("cues")
            if not isinstance(raw_cues, list) or not 1 <= len(raw_cues) <= MAX_CUES:
                raise CreationLibraryError("Character timing is unavailable.")
            cues = []
            for raw in raw_cues:
                start, end = raw.get("start_frame"), raw.get("end_frame")
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= measured["frames"]
                        or raw.get("start_ms") != round(start * 1000 / measured["rate"])
                        or raw.get("end_ms") != round(end * 1000 / measured["rate"])):
                    raise CreationLibraryError("Character cue frame binding differs.")
                cues.append({"speaker_id": raw.get("character_id"),
                             "start_ms": min(raw["start_ms"], measured["final_ms"]),
                             "end_ms": min(raw["end_ms"], measured["final_ms"])})
        public = {"schema_version": 1, "preview_id": entry["id"], "audio_sha256": entry["audio_sha"],
                  "duration_ms": measured["duration_ms"], "cast": cast, "cues": cues, "features": None}
        adapted = dict(public)
        adapted["release_id"] = adapted.pop("preview_id")
        adapted["package_sha256"] = adapted.pop("audio_sha256")
        Performance.model_validate(adapted)
        data = _encoded(public)
        if len(data) > MAX_PERFORMANCE_BYTES:
            raise CreationLibraryError("Public performance exceeds its bound.")
        return data

    def public_assets(self, entry):
        prepared = self.prepared(entry)
        if prepared is None:
            if entry["kind"] == "episode":
                raise VisualPreparationRequired("Visual preparation is required before listening to this programme.")
            return self.performance(entry), None
        manifest, paths = prepared
        performance = _json(_read(paths["performance.json"], MAX_PERFORMANCE_BYTES))
        performance["preview_id"] = entry["id"]
        semantic = _json(_read(paths["semantic.json"], MAX_SEMANTIC_BYTES))
        semantic["preview_id"] = entry["id"]
        performance_bytes, semantic_bytes = _encoded(performance), _encoded(semantic)
        adapted_manifest = manifest.model_copy(update={"preview_id": entry["id"]})
        validated = _performance(performance, adapted_manifest)
        _semantic(semantic, adapted_manifest, validated)
        return performance_bytes, semantic_bytes

    def metadata(self, entry, performance, semantic=None):
        manifest = PreviewManifest.model_validate({
            "schema_version": 1, "preview_id": entry["id"], "title": entry["title"],
            "duration_ms": entry["measured"]["duration_ms"],
            "audio": {"file": "audio.wav", "sha256": entry["audio_sha"],
                      "size_bytes": entry["measured"]["size_bytes"]},
            "performance": {"file": "performance.json", "sha256": hashlib.sha256(performance).hexdigest(),
                            "size_bytes": len(performance)},
            "semantic": ({"file": "semantic.json", "sha256": hashlib.sha256(semantic).hexdigest(),
                          "size_bytes": len(semantic)} if semantic is not None else None),
        }).model_dump(mode="json")
        manifest.update(kind="local_preview", local_preview=True)
        for role in ("audio", "performance", "semantic"):
            if manifest[role] is not None:
                manifest[role]["url"] = "/creation-library/" + entry["id"] + "/" + manifest[role]["file"]
        return manifest

    def _receipt_matches(self, entry, receipt):
        if (receipt.get("source_manifest_sha256") != entry["source_sha"]
                or receipt.get("audio_sha256") != entry["audio_sha"]):
            return False
        current = entry["root"].relative_to(self.project).as_posix()
        previous = receipt.get("creation_root")
        if previous == current:
            return True
        # A curator may move an unchanged creation while immutable preparation
        # evidence retains its old path. Reuse content, never revive that path.
        if not isinstance(previous, str) or not previous.startswith("creations/"):
            return False
        old = _ordinary(_safe(self.project, previous))
        if old.exists():
            return False
        entries = entry["entries"]
        if (receipt.get("script_sha256") != entries.get("script.json", {}).get("sha256")
                or receipt.get("timing_sha256") != entries.get("performance.json", {}).get("sha256")
                or receipt.get("script_sha256") is None
                or receipt.get("timing_sha256") is None):
            return False
        matches = []
        for candidate in self.roots():
            try:
                digest = hashlib.sha256(_read(candidate / "manifest.json", MAX_JSON_BYTES)).hexdigest()
            except (OSError, ValueError):
                continue
            if digest == entry["source_sha"]:
                matches.append(candidate)
                if len(matches) > 1:
                    return False
        return matches == [entry["root"]]

    def prepared(self, entry):
        directory = _ordinary(self.runtime / "visual-previews")
        if entry["source_sha"] is None or not directory.is_dir():
            return None
        with os.scandir(directory) as listing:
            candidates = []
            for item in listing:
                if len(candidates) >= MAX_PREVIEWS:
                    raise CreationLibraryError("Prepared preview count exceeds its bound.")
                if not item.is_symlink() and item.is_dir(follow_symlinks=False):
                    candidates.append(Path(item.path))
        for preview in sorted(candidates):
            if re.fullmatch(SLUG_PATTERN, preview.name) is None:
                continue
            try:
                receipt = self._json(preview / "evidence/creation-source-v1.json")
                if not self._receipt_matches(entry, receipt):
                    continue
                # A repeated menu open need not hash the same prepared WAV
                # again. Any fixed asset or manifest change invalidates reuse.
                signature = []
                for name in ("manifest.json", "audio.wav", "performance.json", "semantic.json"):
                    info = (preview / name).stat(follow_symlinks=False)
                    if not stat.S_ISREG(info.st_mode):
                        raise CreationLibraryError("Prepared assets must remain ordinary.")
                    signature.append(_fingerprint(info))
                key = (str(preview), tuple(signature))
                with self.cache_lock:
                    verified = self.preview_cache.get(key)
                if verified is None:
                    verified = read_preview(self.runtime, preview.name)
                    with self.cache_lock:
                        self.preview_cache[key] = verified
                        self.preview_cache.move_to_end(key)
                        while len(self.preview_cache) > MAX_PREVIEWS:
                            self.preview_cache.popitem(last=False)
                manifest, paths = verified
                if (manifest.audio.sha256 == entry["audio_sha"]
                        and manifest.duration_ms == entry["measured"]["duration_ms"]
                        and manifest.semantic is not None):
                    semantic = _json(_read(paths["semantic.json"], MAX_SEMANTIC_BYTES))
                    if semantic.get("projection", {}).get("dimensions") == 5:
                        return manifest, paths
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return None


def register_creation_library(app, project_root, runtime):
    """Register fixed, read-only local catalogue, metadata and listening routes."""
    library = _Library(project_root, runtime)

    def checked_request(request):
        if request.query_params:
            raise HTTPException(422, "Unsupported creation library query.", headers=PREVIEW_HEADERS)

    def verified(identifier, *, allow_project=False):
        try:
            entry = library.entry(library.find(identifier, allow_project=allow_project), verify=True)
            performance, semantic = library.public_assets(entry)
            return entry, performance, semantic
        except VisualPreparationRequired:
            raise HTTPException(409, "Visual preparation is required before listening to this programme.",
                                headers=PREVIEW_HEADERS) from None
        except FileNotFoundError:
            raise HTTPException(404, "Local creation is unavailable.", headers=PREVIEW_HEADERS) from None
        except (OSError, ValueError, KeyError, TypeError, RecursionError):
            raise HTTPException(409, "Local creation could not be verified.", headers=PREVIEW_HEADERS) from None

    @app.get("/api/creation-library")
    def creation_catalog(request: Request):
        checked_request(request)
        items = []
        try:
            roots = library.catalogue_roots()
        except (OSError, ValueError):
            raise HTTPException(409, "Local creation library could not be read.", headers=PREVIEW_HEADERS) from None
        for root in roots:
            try:
                entry = library.entry(root)
                library.performance(entry)
                semantic_ready = entry["kind"] == "episode" and library.prepared(entry) is not None
                items.append({"id": entry["id"], "title": entry["title"], "version": entry["version"],
                              "duration_ms": entry["measured"]["duration_ms"], "kind": entry["kind"],
                              "playback_url": "/?creation=" + (_project_identity(root, library.project)
                                  if entry["kind"] == "episode" else entry["id"]),
                              "semantic_ready": semantic_ready,
                              "playback_ready": entry["kind"] == "voice_audition" or semantic_ready})
            except (OSError, ValueError, KeyError, TypeError, RecursionError):
                continue
        items.sort(key=lambda item: (item["kind"] != "episode", item["title"].casefold(), item["version"]))
        return JSONResponse({"schema_version": 1, "items": items}, headers=PREVIEW_HEADERS)

    @app.get("/api/creation-library/{identifier}")
    def creation_metadata(identifier: str, request: Request):
        checked_request(request)
        entry, performance, semantic = verified(identifier, allow_project=True)
        return JSONResponse(library.metadata(entry, performance, semantic), headers=PREVIEW_HEADERS)

    @app.get("/creation-library/{identifier}/{asset_name}")
    def creation_asset(identifier: str, asset_name: str, request: Request):
        checked_request(request)
        if asset_name not in {"audio.wav", "performance.json", "semantic.json"}:
            raise HTTPException(404, "Local creation asset is unavailable.", headers=PREVIEW_HEADERS)
        entry, performance, semantic = verified(identifier)
        if asset_name == "performance.json":
            return Response(performance, media_type="application/json", headers=PREVIEW_HEADERS)
        if asset_name == "semantic.json":
            if semantic is None:
                raise HTTPException(404, "Local semantic performance is unavailable.", headers=PREVIEW_HEADERS)
            return Response(semantic, media_type="application/json", headers=PREVIEW_HEADERS)
        return FileResponse(entry["audio"], media_type="audio/wav", headers=PREVIEW_HEADERS)
