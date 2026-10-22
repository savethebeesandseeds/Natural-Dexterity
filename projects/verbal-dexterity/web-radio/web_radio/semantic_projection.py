"""One deterministic shared PCA basis for verified source-embedding previews.

No provider access, audio generation, scheduling, per-character displacement, or
repulsion occurs here. Public coordinates are a lossy view of a common basis.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import struct
import tempfile

from .contract import validate_package
from .embedding_contract import read_bundle

MAX_SAMPLES = 512
MAX_DIMENSIONS = 3072
FIT_EXTENT = 0.68


def _dot(left, right):
    return math.fsum(a * b for a, b in zip(left, right))


def _jacobi(matrix):
    """Cyclic symmetric Jacobi rotations; eigenvectors are returned by column."""
    size = len(matrix)
    values = [list(row) for row in matrix]
    vectors = [[float(i == j) for j in range(size)] for i in range(size)]
    tolerance = 1e-13 * max(1.0, max(abs(values[i][i]) for i in range(size)))
    for sweep in range(80):
        largest = max((abs(values[i][j]) for i in range(size)
                       for j in range(i + 1, size)), default=0.0)
        if largest <= tolerance:
            return [values[i][i] for i in range(size)], vectors, sweep, largest
        for p in range(size - 1):
            for q in range(p + 1, size):
                cross = values[p][q]
                if abs(cross) <= tolerance:
                    continue
                tau = (values[q][q] - values[p][p]) / (2 * cross)
                tangent = math.copysign(1.0, tau) / (abs(tau) + math.hypot(1.0, tau))
                cosine = 1 / math.sqrt(1 + tangent * tangent)
                sine = tangent * cosine
                left, right = values[p][p], values[q][q]
                values[p][p] = left - tangent * cross
                values[q][q] = right + tangent * cross
                values[p][q] = values[q][p] = 0.0
                for k in range(size):
                    if k != p and k != q:
                        old_p, old_q = values[k][p], values[k][q]
                        values[k][p] = values[p][k] = cosine * old_p - sine * old_q
                        values[k][q] = values[q][k] = sine * old_p + cosine * old_q
                    old_p, old_q = vectors[k][p], vectors[k][q]
                    vectors[k][p] = cosine * old_p - sine * old_q
                    vectors[k][q] = sine * old_p + cosine * old_q
    raise ValueError("Symmetric PCA decomposition did not converge; no projection produced.")


def pca(vectors, dimensions=3):
    """Return 3D or 4D scores from one normalized/centered PCA fit and diagnostics.

    A fourth component uses the unchanged XYZ scale and must fit [-1,1].
    Zero inputs normalize to zero. Missing rank produces zero axes. The input order
    breaks exact eigenvalue ties deterministically; signs use the earliest feature
    with largest absolute loading. A single scalar fits all axes to +/-0.68.
    """
    if type(dimensions) is not int or dimensions not in (3, 4):
        raise ValueError("Choose exactly three or four PCA components.")
    rows = [list(row) for row in vectors]
    if not 1 <= len(rows) <= MAX_SAMPLES or not rows or not rows[0]:
        raise ValueError("PCA requires between one and 512 nonempty vectors.")
    width = len(rows[0])
    if width > MAX_DIMENSIONS or any(len(row) != width for row in rows):
        raise ValueError("PCA inputs must share a dimension no greater than 3072.")
    for row in rows:
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in row):
            raise ValueError("PCA inputs must contain finite ordinary numbers.")
    normalized = []
    for row in rows:
        norm = math.hypot(*row)
        normalized.append([value / norm for value in row] if norm else [0.0] * width)
    count = len(normalized)
    mean = [math.fsum(row[j] for row in normalized) / count for j in range(width)]
    centered = [[value - mean[j] for j, value in enumerate(row)] for row in normalized]
    gram = [[0.0] * count for _ in range(count)]
    for i in range(count):
        for j in range(i, count):
            gram[i][j] = gram[j][i] = _dot(centered[i], centered[j])
    eigenvalues, eigenvectors, sweeps, residual = _jacobi(gram)
    order = sorted(range(count), key=lambda index: (-eigenvalues[index], index))
    ordered = [max(0.0, eigenvalues[index]) for index in order]
    threshold = max(1.0, ordered[0]) * 1e-10
    rank = sum(value > threshold for value in ordered)
    basis, retained = [], []
    for index in order:
        value = eigenvalues[index]
        if value <= threshold or len(basis) == dimensions:
            continue
        axis = [math.fsum(eigenvectors[i][index] * centered[i][j] for i in range(count))
                / math.sqrt(value) for j in range(width)]
        for previous in basis:
            coefficient = _dot(axis, previous)
            axis = [a - coefficient * b for a, b in zip(axis, previous)]
        norm = math.hypot(*axis)
        if norm <= 1e-10:
            continue
        axis = [value / norm for value in axis]
        pivot = max(range(width), key=lambda feature: abs(axis[feature]))
        if axis[pivot] < 0:
            axis = [-value for value in axis]
        basis.append(axis)
        retained.append(max(0.0, value))
    while len(basis) < dimensions:
        basis.append([0.0] * width)
        retained.append(0.0)
    scores = [[_dot(row, axis) for axis in basis] for row in centered]
    maximum = max(abs(value) for row in scores for value in row[:3])
    scale = FIT_EXTENT / maximum if maximum else 0.0
    positions = [[value * scale for value in row] for row in scores]
    if dimensions == 4 and any(abs(row[3]) > 1.0 for row in positions):
        raise ValueError("Fourth PCA component exceeds [-1,1] under the preserved XYZ scale.")
    total = math.fsum(ordered)
    return {
        "positions": positions, "mean": mean, "basis": basis, "scale": scale,
        "diagnostics": {
            "fitted_count": count, "input_dimensions": width, "rank": rank,
            "gram_eigenvalues": ordered, "retained_eigenvalues": retained,
            "explained_variance_ratio": [value / total if total else 0.0 for value in retained],
            "rank_threshold": threshold, "jacobi_sweeps": sweeps,
            "remaining_off_diagonal_max": residual, "maximum_unscaled_score": maximum,
        },
    }


def _ordinary(path):
    path = Path(os.path.abspath(path))
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Projection paths must not traverse symbolic links.")
    return path


def _read(path, limit):
    path = _ordinary(path)
    if not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise ValueError("Projection input must be a bounded ordinary file.")
    data = path.read_bytes()
    if len(data) > limit:
        raise ValueError("Projection input exceeds its bound.")
    return data


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _write_once(path, data):
    path = _ordinary(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read(path, max(1, len(data))) != data:
            raise ValueError("Conflicting immutable projection artifact is preserved.")
        return
    with path.open("xb") as output:
        output.write(data)


def project_preview(preparation_root, bundle, preview_root, *, dimensions=3):
    """Project one already-paid, validated text bundle; never dispatch provider work."""
    preparation = _ordinary(preparation_root)
    preview = _ordinary(preview_root)
    bundle = _ordinary(bundle)
    bundle_root = bundle.parent if bundle.name == "manifest.json" else bundle
    bundle_bytes = _read(bundle_root / "manifest.json", 1024 * 1024)
    raw_bundle = json.loads(bundle_bytes)
    release_id, package_sha256 = raw_bundle["release_id"], raw_bundle["package_sha256"]
    if not isinstance(release_id, str) or "/" in release_id or chr(92) in release_id or release_id in {".", ".."}:
        raise ValueError("Invalid isolated release identity.")
    journal = _ordinary(preparation / ".runtime/station/journal.sqlite3")
    with sqlite3.connect(journal.as_uri() + "?mode=ro", uri=True) as database:
        accepted = database.execute("SELECT digest FROM accepted WHERE release_id=?", (release_id,)).fetchone()
    if accepted is None or accepted[0] != package_sha256:
        raise ValueError("Embedding bundle must name an accepted isolated release.")
    release = validate_package(preparation / ".runtime/station/releases" / release_id, package_sha256)
    document, shards = read_bundle(bundle_root, release)
    if document.config.model != "gemini-embedding-2" or any(
            window.source_modality != "text" or window.character_id is None for window in document.windows):
        raise ValueError("This preview requires only explicitly attributed text windows.")
    manifest_path = preview / "manifest.json"
    manifest_bytes = _read(manifest_path, 1024 * 1024)
    manifest = json.loads(manifest_bytes)
    if manifest["duration_ms"] != document.duration_ms:
        raise ValueError("Preview and embedding duration differ.")
    if (manifest["audio"]["file"] != "audio.wav" or
            manifest["performance"]["file"] != "performance.json"):
        raise ValueError("Preview asset filenames must remain fixed.")
    audio = _read(preview / "audio.wav", 128 * 1024 * 1024)
    performance_bytes = _read(preview / "performance.json", 8 * 1024 * 1024)
    for name, data in (("audio", audio), ("performance", performance_bytes)):
        if _sha(data) != manifest[name]["sha256"] or len(data) != manifest[name]["size_bytes"]:
            raise ValueError("Preview asset failed its manifest identity.")
    audio_sha256 = _sha(audio)
    if audio_sha256 != document.release_audio_sha256:
        raise ValueError("Raw embeddings must name the preview's exact audio bytes.")
    performance = json.loads(performance_bytes)
    if (performance["preview_id"] != manifest["preview_id"] or
            performance["audio_sha256"] != audio_sha256 or
            performance["duration_ms"] != manifest["duration_ms"]):
        raise ValueError("Prepared performance is bound to another preview.")
    cast = {character["id"] for character in performance["cast"]}
    rows = sorted(document.windows, key=lambda row: (row.start_ms, row.end_ms, row.id))
    expected_cues = [(cue["speaker_id"], cue["start_ms"], cue["end_ms"]) for cue in performance["cues"]]
    actual_cues = [(row.character_id, row.start_ms, row.end_ms) for row in rows]
    if actual_cues != expected_cues or any(row.character_id not in cast for row in rows):
        raise ValueError("Text embeddings must exactly cover the known source-turn cues.")
    vectors = [list(struct.unpack_from("<" + "f" * document.config.dimensions,
                    shards[row.shard_id], row.vector_index * document.config.dimensions * 4)) for row in rows]
    result = pca(vectors, dimensions=dimensions)
    basis_document = {
        "schema_version": 1, "method": "pca", "dimensions": dimensions,
        "model": document.config.model, "space_id": document.space_id,
        "fitted_count": len(rows), "input_normalization": "l2", "centering": "common_mean",
        "mean": result["mean"], "basis": result["basis"], "uniform_scale": result["scale"],
        "fit_extent": FIT_EXTENT, "component_sign": "earliest largest-absolute feature loading positive",
    }
    if dimensions == 4:
        basis_document["rotation_mapping"] = {
            "component": 4, "scale": "same_uniform_scale_as_xyz",
            "normalized_range": [-1, 1], "render_degrees": [-50, 50], "axis": "view",
        }
    basis_bytes = _encoded(basis_document)
    basis_sha256 = _sha(basis_bytes)
    semantic = {
        "schema_version": 1, "preview_id": manifest["preview_id"], "audio_sha256": audio_sha256,
        "duration_ms": manifest["duration_ms"], "model": "gemini-embedding-2",
        "space_id": document.space_id,
        "projection": {"method": "pca", "dimensions": dimensions, "fitted_count": len(rows),
                       "basis_sha256": basis_sha256},
        "windows": [{"id": row.id, "character_id": row.character_id,
                     "start_ms": row.start_ms, "end_ms": row.end_ms,
                     "position": [round(value, 12) for value in position[:3]],
                     **({"rotation": round(position[3], 12)} if dimensions == 4 else {})}
                    for row, position in zip(rows, result["positions"])],
    }
    semantic_bytes = _encoded(semantic)
    descriptor = {"file": "semantic.json", "sha256": _sha(semantic_bytes),
                  "size_bytes": len(semantic_bytes)}
    if manifest.get("semantic") not in (None, descriptor):
        raise ValueError("An existing different semantic projection is preserved.")
    receipt = {
        "schema_version": 1, "preview_id": manifest["preview_id"],
        "release_id": release.release_id, "package_sha256": release.package_sha256,
        "audio_sha256": audio_sha256, "raw_bundle_id": document.bundle_id,
        "raw_bundle_manifest_sha256": _sha(bundle_bytes),
        "raw_bundle_path": str(bundle_root),
        "raw_shards": [shard.model_dump(mode="json") for shard in document.shards],
        "source_window_ids": [row.id for row in rows],
        "semantic_sha256": descriptor["sha256"], "basis_sha256": basis_sha256,
        "diagnostics": result["diagnostics"],
        "character_offsets": False, "repulsion": False, "provider_calls": 0,
    }
    if dimensions == 4:
        receipt["coordinate_mapping"] = {
            "position_components": [1, 2, 3], "rotation_component": 4,
            "uniform_scale": "fitted_only_to_first_three_components_and_shared_by_fourth",
            "rotation_render_degrees": [-50, 50], "rotation_axis": "view",
            "fourth_component_variance_ratio": result["diagnostics"]["explained_variance_ratio"][3],
        }
    evidence_version = "v2" if dimensions == 4 else "v1"
    _write_once(preview / ("evidence/pca-basis-" + evidence_version + ".json"), basis_bytes)
    _write_once(preview / ("evidence/pca-projection-" + evidence_version + ".json"), _encoded(receipt))
    _write_once(preview / "semantic.json", semantic_bytes)
    manifest["semantic"] = descriptor
    encoded_manifest = _encoded(manifest)
    if _read(manifest_path, 1024 * 1024) != manifest_bytes:
        raise ValueError("Preview manifest changed concurrently; projection files retained for review.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=preview, prefix=".semantic-manifest-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded_manifest)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"semantic": str(preview / "semantic.json"), "semantic_sha256": descriptor["sha256"],
            "basis_sha256": basis_sha256, "fitted_count": len(rows),
            "space_id": document.space_id, "rank": result["diagnostics"]["rank"],
            "dimensions": dimensions,
            "rotation_max_absolute": max(abs(row[3]) for row in result["positions"]) if dimensions == 4 else None,
            "explained_variance_ratio": result["diagnostics"]["explained_variance_ratio"],
            "provider_calls": 0}


def clone_preview_variant(source_root, preview_root, preview_id):
    """Create a new preview identity while preserving every source-preview byte."""
    source = _ordinary(source_root)
    destination = _ordinary(preview_root)
    if source.parent != destination.parent or source == destination or destination.exists():
        raise ValueError("Create a new sibling preview directory; existing previews are preserved.")
    if not isinstance(preview_id, str) or not preview_id or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in preview_id):
        raise ValueError("Use an ordinary lowercase preview identity.")
    source_manifest_bytes = _read(source / "manifest.json", 1024 * 1024)
    manifest = json.loads(source_manifest_bytes)
    if manifest["audio"]["file"] != "audio.wav" or manifest["performance"]["file"] != "performance.json":
        raise ValueError("Source preview assets must use fixed filenames.")
    preserved_paths = [source / name for name in ("manifest.json", "audio.wav", "performance.json", "semantic.json")]
    preserved_paths += sorted((source / "evidence").glob("*.json"))
    preserved = {str(path.relative_to(source)): _sha(_read(path, 128 * 1024 * 1024))
                 for path in preserved_paths}
    audio = _read(source / "audio.wav", 128 * 1024 * 1024)
    performance_bytes = _read(source / "performance.json", 8 * 1024 * 1024)
    for role, data in (("audio", audio), ("performance", performance_bytes)):
        if _sha(data) != manifest[role]["sha256"] or len(data) != manifest[role]["size_bytes"]:
            raise ValueError("Source preview failed its fixed-asset integrity check.")
    performance = json.loads(performance_bytes)
    if performance["preview_id"] != manifest["preview_id"]:
        raise ValueError("Source performance has another preview identity.")
    source_id = manifest["preview_id"]
    performance["preview_id"] = preview_id
    adapted = _encoded(performance)
    manifest["preview_id"] = preview_id
    manifest["performance"] = {"file": "performance.json", "sha256": _sha(adapted), "size_bytes": len(adapted)}
    manifest["semantic"] = None
    destination.mkdir()
    _write_once(destination / "audio.wav", audio)
    _write_once(destination / "performance.json", adapted)
    _write_once(destination / "manifest.json", _encoded(manifest))
    for name, digest in preserved.items():
        if _sha(_read(source / name, 128 * 1024 * 1024)) != digest:
            raise ValueError("Source preview changed during creation; retained output requires review.")
    _write_once(destination / "evidence/preview-source-v2.json", _encoded({
        "schema_version": 1, "source_preview_id": source_id, "preview_id": preview_id,
        "source_preview_root": str(source), "source_files_preserved_sha256": preserved,
        "source_preservation_verified": True, "audio_changed": False,
        "performance_change": "preview_id only", "provider_calls": 0,
    }))
    return destination
