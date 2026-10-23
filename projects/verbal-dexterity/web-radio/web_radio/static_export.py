"""Export only validated, curated listening data; never run production jobs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen
import re

from .creation_library import (
    MAX_JSON_BYTES, _Library, _encoded, _ordinary, _project_identity, _relative,
)
from .preview import PreviewManifest, _open, _performance, _read, _semantic

PROJECT = Path(__file__).resolve().parents[1]
MAX_STATIC_FILE_BYTES = 25 * 1024 * 1024
PLAYER_ASSETS = (
    "player.css", "player.js", "spirits.js", "performance.js", "semantic.js", "library.js",
    "botanical-frame-v1.png", "parchment-texture-v1.png", "waajacu-macaw.png",
)
ROOT_PLAYER_ASSETS = ("favicon.ico",)
PUBLIC_NOTICES = ("NOTICE", "LICENSE", "CONTENT-PERMISSIONS.md")
STATIC_CONFIG = ("_headers", "_redirects", "404.html")
MANIFEST = "site-manifest.json"
CUSTOM_DOMAIN = "radio.waajacu.com"
PAGES_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; style-src-attr 'none'; "
    "media-src 'self' blob:; connect-src 'self'; img-src 'self' data:; "
    "worker-src 'self' blob:; object-src 'none'; base-uri 'none'"
)
PAGES_CSP_META = '<meta http-equiv="Content-Security-Policy" content="' + PAGES_CSP + '">'
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_FILES = 4096


class StaticExportError(ValueError):
    """The static site is incomplete or differs from its validated inventory."""


@dataclass(frozen=True)
class _File:
    path: str
    size: int
    sha256: str
    data: bytes | None = None
    source: Path | None = None


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _bytes(path, data):
    return _File(path, len(data), _digest(data), data=data)


def _json(path):
    document = json.loads(_read(path, MAX_JSON_BYTES))
    if not isinstance(document, dict):
        raise StaticExportError("Expected a JSON object.")
    return document


def _curated_roots(project, library):
    document = _json(project / "creations/catalogue.json")
    rows = document.get("creations")
    if not isinstance(rows, list) or not rows:
        raise StaticExportError("A nonempty curated catalogue is required.")
    roots, selected = [], set(library.catalogue_roots())
    for row in rows:
        if (not isinstance(row, dict) or row.get("curated") is not True
                or row.get("complete") is not True or row.get("listener_approval") != "approved"):
            raise StaticExportError("Every exported programme must be curated, complete and listener-approved.")
        relative = _relative(row.get("bundle"))
        root = _ordinary(project / "creations" / relative)
        if root in roots or root not in selected:
            raise StaticExportError("Curated catalogue must name unique current library selections.")
        if root.name != row.get("version"):
            raise StaticExportError("Curated version does not match its production folder.")
        roots.append(root)
    if set(roots) != selected:
        raise StaticExportError("Curated catalogue and current listening library differ.")
    return sorted(roots)


def _frontend_files(project, identities):
    files = []
    player = project / "web/player"
    html = _read(player / "index.html", MAX_JSON_BYTES).decode("utf-8")
    main = list(re.finditer(r"<main\b[^>]*>", html))
    if len(main) != 1 or "data-static-catalogue" in main[0].group():
        raise StaticExportError("Expected one local player main element.")
    selected = re.search(r'\bdata-default-recording="([^"]+)"', main[0].group())
    if selected is None or selected.group(1) not in identities:
        raise StaticExportError("Static playback requires an exported default recording.")
    position = main[0].start() + len("<main")
    html = html[:position] + ' data-static-catalogue="/data/library.json"' + html[position:]
    files.append(_bytes("index.html", html.encode("utf-8")))
    for name in PLAYER_ASSETS:
        files.append(_bytes("assets/" + name, _read(player / name, MAX_STATIC_FILE_BYTES)))
    for name in ROOT_PLAYER_ASSETS:
        source = player / name
        if source.exists() or source.is_symlink():
            files.append(_bytes(name, _read(source, MAX_STATIC_FILE_BYTES)))
        elif "/" + name in html:
            raise StaticExportError("Player HTML references a missing root asset: " + name)
    for name in PUBLIC_NOTICES:
        files.append(_bytes(name, _read(project / name, MAX_JSON_BYTES)))
    for name in STATIC_CONFIG:
        source = project / "web/static" / name
        if source.exists() or source.is_symlink():
            files.append(_bytes(name, _read(source, MAX_JSON_BYTES)))
    return files, selected.group(1)

def _collect(project, runtime, vendor):
    library = _Library(project, runtime)
    files, items, identities = [], [], set()
    for root in _curated_roots(project, library):
        entry = library.entry(root, verify=True)
        if entry["kind"] != "episode":
            raise StaticExportError("Only curated finished episodes may be exported.")
        performance, semantic = library.public_assets(entry)
        if semantic is None:
            raise StaticExportError("Every exported episode requires prepared semantic visuals.")
        metadata = library.metadata(entry, performance, semantic)
        identifier = entry["id"]
        alias = _project_identity(root, project)
        if identities.intersection((identifier, alias)):
            raise StaticExportError("Recording identities must be unique.")
        identities.update((identifier, alias))
        data = _encoded(metadata)
        files.extend((_bytes("data/recordings/" + identifier + ".json", data),
                      _bytes("data/recordings/" + alias + ".json", data)))
        base = "creation-library/" + identifier + "/"
        files.append(_File(base + "audio.wav", metadata["audio"]["size_bytes"],
                           metadata["audio"]["sha256"], source=entry["audio"]))
        files.extend((_bytes(base + "performance.json", performance),
                      _bytes(base + "semantic.json", semantic)))
        items.append({"id": identifier, "title": entry["title"], "version": entry["version"],
                      "duration_ms": entry["measured"]["duration_ms"], "kind": "episode",
                      "playback_url": "/?creation=" + alias,
                      "semantic_ready": True, "playback_ready": True})
    items.sort(key=lambda item: (item["title"].casefold(), item["version"]))
    files.append(_bytes("data/library.json", _encoded({"schema_version": 1, "items": items})))
    frontend, selected = _frontend_files(project, identities)
    files.extend(frontend)
    for name in ("hls.min.js", "HLS-LICENSE.txt"):
        files.append(_bytes("vendor/" + name, _read(vendor / name, MAX_STATIC_FILE_BYTES)))
    expected_vendor_hash = _read(vendor / "hls.sha256", 256).decode("ascii").strip()
    if _digest(next(item.data for item in files if item.path == "vendor/hls.min.js")) != expected_vendor_hash:
        raise StaticExportError("Installed playback vendor checksum differs.")
    for item in files:
        _relative(item.path)
        if item.size > MAX_STATIC_FILE_BYTES:
            raise StaticExportError("Static file exceeds the 25 MiB limit: " + item.path)
    if len({item.path for item in files}) != len(files):
        raise StaticExportError("Duplicate static output path.")
    return sorted(files, key=lambda item: item.path), items, selected


def _write(output, item):
    destination = output / item.path
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest, length = hashlib.sha256(), 0
    with destination.open("xb") as target:
        if item.data is not None:
            target.write(item.data)
            digest.update(item.data)
            length = len(item.data)
        else:
            with _open(item.source) as source:
                while chunk := source.read(1024 * 1024):
                    length += len(chunk)
                    if length > item.size:
                        raise StaticExportError("Source changed during export: " + item.path)
                    digest.update(chunk)
                    target.write(chunk)
    if length != item.size or digest.hexdigest() != item.sha256:
        raise StaticExportError("Source changed during export: " + item.path)


def _platform_files(files, platform, custom_domain):
    if platform not in {"cloudflare", "github-pages"}:
        raise StaticExportError("Unsupported static hosting platform.")
    if platform == "cloudflare":
        return files
    if not isinstance(custom_domain, str) or re.fullmatch(r"[a-z0-9]+(?:[.-][a-z0-9]+)*\.[a-z]{2,}", custom_domain) is None:
        raise StaticExportError("Pages custom domain is invalid.")
    result = []
    for item in files:
        if item.path in {"_headers", "_redirects", "CNAME"}:
            continue
        if item.path in {"index.html", "404.html"}:
            html = item.data.decode("utf-8")
            if re.search(r"http-equiv\s*=\s*['\"]?Content-Security-Policy", html, re.IGNORECASE):
                raise StaticExportError("Source HTML must not contain a second CSP policy.")
            heads = list(re.finditer(r"<head\b[^>]*>", html, re.IGNORECASE))
            if len(heads) != 1:
                raise StaticExportError("Pages HTML requires one head element for its CSP.")
            position = heads[0].end()
            html = html[:position] + "\n  " + PAGES_CSP_META + html[position:]
            item = _bytes(item.path, html.encode("utf-8"))
        result.append(item)
    # Actions publishes the artifact directly without Jekyll. Its upload action
    # excludes hidden files, so no .nojekyll file is added to this inventory.
    result.append(_bytes("CNAME", (custom_domain + "\n").encode("ascii")))
    return result


def _new_output(output):
    output = _ordinary(output)
    if output.exists():
        raise StaticExportError("Export output already exists; choose a new directory.")
    if not output.parent.is_dir():
        raise StaticExportError("Export parent directory must already exist.")
    return output


def _write_site(output, files, recordings, selected, platform, custom_domain):
    files = sorted(_platform_files(files, platform, custom_domain), key=lambda item: item.path)
    if len({item.path for item in files}) != len(files):
        raise StaticExportError("Duplicate static output path.")
    for item in files:
        if not _public_path(item.path) or item.size > MAX_STATIC_FILE_BYTES:
            raise StaticExportError("Static file exceeds its public path or size limit: " + item.path)
    output.mkdir()
    for item in files:
        _write(output, item)
    manifest = {"schema_version": 1, "default_recording": selected,
                "recordings": recordings,
                "files": [{"path": item.path, "bytes": item.size, "sha256": item.sha256} for item in files]}
    if platform == "github-pages":
        manifest.update(platform=platform, custom_domain=custom_domain)
    _write(output, _bytes(MANIFEST, _encoded(manifest)))
    return verify_export(output)

def export_static(output, *, project=PROJECT, runtime=None, vendor=Path("/opt/web-radio/vendor"), platform="cloudflare", custom_domain=CUSTOM_DOMAIN):
    """Write a new, deterministic deployment directory after complete preflight validation.

    Existing outputs are never changed. A write failure preserves the partial output
    for inspection; absence of the final inventory means it is not deployable.
    """
    output, project = _new_output(output), _ordinary(project)

    runtime = _ordinary(runtime or project / ".runtime")
    try:
        files, items, selected = _collect(project, runtime, _ordinary(vendor))
        return _write_site(output, files, [item["id"] for item in items], selected, platform, custom_domain)
    except StaticExportError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StaticExportError("Export stopped; sources and any partial output are preserved.") from exc


def _checked_hash(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{64}", value) is None:
        raise StaticExportError(label + " requires a SHA-256 digest.")
    return value


def _archive_hash(path):
    digest, size = hashlib.sha256(), 0
    with _open(path) as source:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_ARCHIVE_BYTES:
                raise StaticExportError("Public archive exceeds its size limit.")
            digest.update(chunk)
    return digest.hexdigest()


def _extract_public_archive(path, output):
    """Extract regular public files only, without tar's filesystem extraction API."""
    names, total = set(), 0
    with _open(path) as source, tarfile.open(fileobj=source, mode="r:*") as archive:
        for member in archive:
            if not member.isfile() or member.issparse():
                raise StaticExportError("Public archives may contain only regular files.")
            relative = _relative(member.name)
            if relative in names or (relative != MANIFEST and not _public_path(relative)):
                raise StaticExportError("Public archive contains duplicate or unlisted paths.")
            if not 0 <= member.size <= MAX_STATIC_FILE_BYTES:
                raise StaticExportError("Public archive member exceeds its size limit.")
            names.add(relative)
            total += member.size
            if len(names) > MAX_ARCHIVE_FILES or total > MAX_ARCHIVE_BYTES:
                raise StaticExportError("Public archive exceeds its file or expanded size limit.")
            stream = archive.extractfile(member)
            if stream is None:
                raise StaticExportError("Public archive member is unavailable.")
            with stream:
                data = stream.read(member.size + 1)
            if len(data) != member.size:
                raise StaticExportError("Public archive member is incomplete.")
            _write(output, _bytes(relative, data))
    if MANIFEST not in names:
        raise StaticExportError("Public archive is missing its verification inventory.")


def assemble_public_archive(archive, output, *, sha256, manifest_sha256, project=PROJECT,
                            platform="github-pages", custom_domain=CUSTOM_DOMAIN):
    """Combine pinned public media with the current frontend, without private sources."""
    output, project = _new_output(output), _ordinary(project)
    expected = _checked_hash(sha256, "Public archive")
    expected_manifest = _checked_hash(manifest_sha256, "Public inventory")
    try:
        if _archive_hash(archive) != expected:
            raise StaticExportError("Public archive checksum differs from the pinned release.")
        with tempfile.TemporaryDirectory(prefix="web-radio-public-archive-") as scratch:
            extracted = Path(scratch)
            _extract_public_archive(archive, extracted)
            if _digest(_read(extracted / MANIFEST, MAX_JSON_BYTES)) != expected_manifest:
                raise StaticExportError("Public inventory checksum differs from the pinned release.")
            verify_export(extracted)
            source_manifest = _json(extracted / MANIFEST)
            files = [_File(row["path"], row["bytes"], row["sha256"], source=extracted / row["path"])
                     for row in source_manifest["files"]
                     if row["path"].startswith(("creation-library/", "data/", "vendor/"))]
            identities = {Path(item.path).stem for item in files if item.path.startswith("data/recordings/")}
            frontend, selected = _frontend_files(project, identities)
            files.extend(frontend)
            return _write_site(output, files, source_manifest["recordings"], selected, platform, custom_domain)
    except StaticExportError:
        raise
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as exc:
        raise StaticExportError("Public assembly stopped; release sources and any partial output are preserved.") from exc


def _download_public_archive(url, output):
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc != "github.com" or parsed.query or parsed.fragment
            or re.fullmatch(r"/[^/]+/[^/]+/releases/download/[^/]+/[^/]+", parsed.path) is None):
        raise StaticExportError("Public media must use a pinned HTTPS GitHub Release asset URL.")
    size = 0
    with urlopen(url, timeout=60) as source, output.open("xb") as target:
        final = urlsplit(source.geturl())
        if final.scheme != "https" or not (final.hostname == "github.com" or (final.hostname or "").endswith(".githubusercontent.com")):
            raise StaticExportError("Public media redirected outside GitHub HTTPS storage.")
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_ARCHIVE_BYTES:
                raise StaticExportError("Public archive download exceeds its size limit.")
            target.write(chunk)


def assemble_publication(publication, output, *, project=PROJECT, platform="github-pages"):
    """Download only the hash-pinned public release named by a tracked publication file."""
    output = _new_output(output)
    try:
        document = _json(publication)
        if (type(document.get("schema_version")) is not int or document["schema_version"] != 1
                or document.get("platform") != "github-pages" or platform != "github-pages"):
            raise StaticExportError("Publication must select GitHub Pages schema version one.")
        media = document.get("media")
        if not isinstance(media, dict) or not isinstance(media.get("url"), str):
            raise StaticExportError("Publication requires a pinned public media release.")
        _checked_hash(media.get("sha256"), "Public archive")
        _checked_hash(media.get("manifest_sha256"), "Public inventory")
        with tempfile.TemporaryDirectory(prefix="web-radio-publication-") as scratch:
            archive = Path(scratch) / "public-media.tar.gz"
            _download_public_archive(media["url"], archive)
            return assemble_public_archive(archive, output, sha256=media["sha256"],
                manifest_sha256=media["manifest_sha256"], project=project, platform=platform,
                custom_domain=document.get("custom_domain"))
    except StaticExportError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StaticExportError("Publication assembly stopped; no automatic retry or source changes.") from exc

def _public_path(relative):
    fixed = {"index.html", "data/library.json", "CNAME", *ROOT_PLAYER_ASSETS, *PUBLIC_NOTICES, *STATIC_CONFIG,
             *("assets/" + name for name in PLAYER_ASSETS),
             "vendor/hls.min.js", "vendor/HLS-LICENSE.txt"}
    return (relative in fixed
            or re.fullmatch(r"data/recordings/(?:creation|project)-[a-f0-9]{24}\.json", relative) is not None
            or re.fullmatch(r"creation-library/creation-[a-f0-9]{24}/(?:audio\.wav|performance\.json|semantic\.json)", relative) is not None)


def _verify_contract(output, manifest, inventory):
    recordings = manifest.get("recordings")
    if (not isinstance(recordings, list) or not 1 <= len(recordings) <= 512
            or any(not isinstance(identifier, str) or re.fullmatch(r"creation-[a-f0-9]{24}", identifier) is None
                   for identifier in recordings)
            or len(set(recordings)) != len(recordings)):
        raise StaticExportError("Static recording list is invalid.")
    required = {"index.html", "data/library.json", *PUBLIC_NOTICES,
                *("assets/" + name for name in PLAYER_ASSETS),
                "vendor/hls.min.js", "vendor/HLS-LICENSE.txt"}
    for identifier in recordings:
        required.add("data/recordings/" + identifier + ".json")
        required.update("creation-library/" + identifier + "/" + name
                        for name in ("audio.wav", "performance.json", "semantic.json"))
    if not required <= inventory.keys():
        raise StaticExportError("Static export is missing required playback files.")
    metadata = {}
    keys = {"schema_version", "preview_id", "title", "duration_ms", "audio", "performance", "semantic"}
    for identifier in recordings:
        document = _json(output / ("data/recordings/" + identifier + ".json"))
        if (set(document) != keys | {"kind", "local_preview"}
                or document.get("preview_id") != identifier
                or document.get("kind") != "local_preview" or document.get("local_preview") is not True):
            raise StaticExportError("Static recording metadata is invalid.")
        contract = {key: document[key] for key in keys - {"audio", "performance", "semantic"}}
        for role, name in (("audio", "audio.wav"), ("performance", "performance.json"), ("semantic", "semantic.json")):
            asset = document.get(role)
            path = "creation-library/" + identifier + "/" + name
            if (not isinstance(asset, dict) or set(asset) != {"file", "sha256", "size_bytes", "url"}
                    or asset.get("file") != name or asset.get("url") != "/" + path
                    or asset.get("sha256") != inventory[path]["sha256"]
                    or asset.get("size_bytes") != inventory[path]["bytes"]):
                raise StaticExportError("Static recording references differ from the file inventory.")
            contract[role] = {key: value for key, value in asset.items() if key != "url"}
        validated = PreviewManifest.model_validate(contract)
        performance = _performance(_json(output / ("creation-library/" + identifier + "/performance.json")), validated)
        semantic = _semantic(_json(output / ("creation-library/" + identifier + "/semantic.json")), validated, performance)
        if semantic.projection.dimensions != 5:
            raise StaticExportError("Static episodes require their prepared five-dimensional visuals.")
        metadata[identifier] = document
    catalogue = _json(output / "data/library.json")
    items = catalogue.get("items")
    if (type(catalogue.get("schema_version")) is not int or catalogue["schema_version"] != 1
            or not isinstance(items, list) or len(items) != len(recordings)):
        raise StaticExportError("Static catalogue is invalid.")
    selected, resolved = set(), {identifier: identifier for identifier in recordings}
    for item in items:
        if not isinstance(item, dict):
            raise StaticExportError("Static catalogue entry is invalid.")
        identifier = item.get("id")
        if (not isinstance(identifier, str) or identifier not in metadata or identifier in selected
                or item.get("kind") != "episode" or item.get("playback_ready") is not True
                or item.get("semantic_ready") is not True or item.get("title") != metadata[identifier]["title"]
                or item.get("duration_ms") != metadata[identifier]["duration_ms"]
                or not isinstance(item.get("version"), str) or re.fullmatch(r"v[0-9]+", item["version"]) is None):
            raise StaticExportError("Static catalogue entry differs from its recording.")
        match = re.fullmatch(r"/\?creation=(project-[a-f0-9]{24})", item.get("playback_url", ""))
        if match is None or match[1] in resolved:
            raise StaticExportError("Static catalogue project alias is invalid.")
        alias = match[1]
        path = "data/recordings/" + alias + ".json"
        if path not in inventory or _json(output / path) != metadata[identifier]:
            raise StaticExportError("Static project alias does not resolve to its canonical recording.")
        required.add(path)
        resolved[alias] = identifier
        selected.add(identifier)
    if selected != set(recordings):
        raise StaticExportError("Static catalogue must include every declared recording.")
    platform = manifest.get("platform", "cloudflare")
    if platform == "github-pages":
        if "_headers" in inventory or "_redirects" in inventory or "CNAME" not in inventory:
            raise StaticExportError("Pages output contains incompatible hosting configuration.")
        domain = manifest.get("custom_domain")
        if not isinstance(domain, str) or _read(output / "CNAME", 256) != (domain + "\n").encode("ascii"):
            raise StaticExportError("Pages custom domain differs from its inventory.")
        required.add("CNAME")
    elif platform != "cloudflare":
        raise StaticExportError("Static inventory hosting platform is invalid.")
    required.update(set(STATIC_CONFIG).intersection(inventory))
    required.update(set(ROOT_PLAYER_ASSETS).intersection(inventory))
    if inventory.keys() != required:
        raise StaticExportError("Static output contains assets outside its playback graph.")
    default = manifest.get("default_recording")
    if not isinstance(default, str) or default not in resolved:
        raise StaticExportError("Static default recording is unavailable.")
    html = _read(output / "index.html", MAX_JSON_BYTES).decode("utf-8")
    for name in ROOT_PLAYER_ASSETS:
        if "/" + name in html and name not in inventory:
            raise StaticExportError("Static player is missing a referenced root asset: " + name)
    if platform == "github-pages":
        for name in ("index.html", "404.html"):
            if name not in inventory:
                continue
            page = _read(output / name, MAX_JSON_BYTES).decode("utf-8")
            if page.count(PAGES_CSP_META) != 1:
                raise StaticExportError("Pages output requires its strict HTML CSP.")
            before = page[:page.index(PAGES_CSP_META)]
            if re.search(r"<(?:script|link)\b", before, re.IGNORECASE):
                raise StaticExportError("Pages CSP must precede resource loading.")
    main = list(re.finditer(r"<main\b[^>]*>", html))
    if len(main) != 1:
        raise StaticExportError("Static player main element is invalid.")
    static = re.search(r'\bdata-static-catalogue="([^"]+)"', main[0].group())
    selection = re.search(r'\bdata-default-recording="([^"]+)"', main[0].group())
    if static is None or static[1] != "/data/library.json" or selection is None or selection[1] != default:
        raise StaticExportError("Static player selection differs from its inventory.")

def verify_export(output):
    """Check every deployment byte and reject unlisted files and symlinks."""
    output = _ordinary(output)
    try:
        manifest = _json(output / MANIFEST)
        rows = manifest.get("files")
        if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1 or not isinstance(rows, list) or not rows:
            raise StaticExportError("Static inventory is invalid.")
        expected = {MANIFEST}
        total = 0
        for row in rows:
            if not isinstance(row, dict):
                raise StaticExportError("Static inventory entry is invalid.")
            relative = _relative(row.get("path"))
            size, expected_hash = row.get("bytes"), row.get("sha256")
            if (not _public_path(relative) or relative in expected or type(size) is not int or not 0 <= size <= MAX_STATIC_FILE_BYTES
                    or not isinstance(expected_hash, str) or re.fullmatch(r"[a-f0-9]{64}", expected_hash) is None):
                raise StaticExportError("Static inventory entry is invalid.")
            expected.add(relative)
            digest, count = hashlib.sha256(), 0
            with _open(output / relative) as source:
                while chunk := source.read(1024 * 1024):
                    count += len(chunk)
                    if count > size:
                        raise StaticExportError("Exported file size differs: " + relative)
                    digest.update(chunk)
            if count != size or digest.hexdigest() != expected_hash:
                raise StaticExportError("Exported file checksum differs: " + relative)
            total += count
        actual = set()
        for path in output.rglob("*"):
            if path.is_symlink():
                raise StaticExportError("Static output may not contain symlinks.")
            if path.is_file():
                actual.add(path.relative_to(output).as_posix())
        if actual != expected:
            raise StaticExportError("Static output contains missing or unlisted files.")
        _verify_contract(output, manifest, {row["path"]: row for row in rows})
        return {"ok": True, "output": str(output), "manifest": str(output / MANIFEST),
                "files": len(actual), "bytes": total + (output / MANIFEST).stat().st_size,
                "recordings": len(manifest.get("recordings", [])),
                "default_recording": manifest.get("default_recording"),
                "manifest_sha256": _digest(_read(output / MANIFEST, MAX_JSON_BYTES))}
    except StaticExportError:
        raise
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise StaticExportError("Static export could not be verified.") from exc
