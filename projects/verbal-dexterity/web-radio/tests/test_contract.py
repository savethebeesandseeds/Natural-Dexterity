from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import wave

from jsonschema import Draft202012Validator, FormatChecker
import pytest

from web_radio.contract import (
    ContractError, Manifest, create_release, import_release, package_digest,
    validate_package,
)


@pytest.fixture
def source_wav(tmp_path: Path) -> Path:
    source = tmp_path / "source.wav"
    with wave.open(str(source), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        frames = b"".join(struct.pack("<h", (1000 if (index // 24) % 2 else -1000)) for index in range(4800))
        wav.writeframes(frames)
    return source


@pytest.fixture
def release(tmp_path: Path, source_wav: Path) -> tuple[Path, Manifest]:
    target = tmp_path / "studio" / "programme-release"
    manifest = create_release(
        source_wav, "Mira: Welcome.\nRowan: We are on air.\n", target,
        programme_id="first-programme", title="First programme",
        release_id="programme-release",
    )
    return target, manifest


def change_manifest(root: Path, change, *, rehash: bool = True) -> dict:
    path = root / "manifest.json"
    data = json.loads(path.read_text())
    change(data)
    if rehash:
        data["package_sha256"] = package_digest(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def test_release_roundtrip_and_schema_agree(release):
    root, manifest = release
    loaded = validate_package(root, manifest.package_sha256)
    assert loaded == manifest
    schema_path = Path(__file__).resolve().parents[1] / "packages/media-contract/manifest.schema.json"
    schema = json.loads(schema_path.read_text())
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(json.loads((root / "manifest.json").read_text()))
    generated = Manifest.model_json_schema()
    assert schema["properties"] == generated["properties"]
    assert manifest.duration_ms == 200


def test_package_digest_is_canonical_not_json_whitespace(release):
    root, manifest = release
    path = root / "manifest.json"
    raw = json.loads(path.read_text())
    path.write_text(json.dumps(dict(reversed(list(raw.items()))), indent=4), encoding="utf-8")
    assert validate_package(root, manifest.package_sha256) == manifest


def test_tampered_metadata_is_rejected_even_when_assets_are_unchanged(release):
    root, _ = release
    change_manifest(root, lambda data: data.update(title="Unapproved replacement"), rehash=False)
    with pytest.raises(ContractError, match="package digest"):
        validate_package(root)


def test_self_consistent_rewrite_does_not_match_trusted_publication(release):
    root, original = release
    change_manifest(root, lambda data: data.update(title="Self-consistent replacement"))
    with pytest.raises(ContractError, match="trusted publication"):
        validate_package(root, original.package_sha256)


@pytest.mark.parametrize("mutation", [
    lambda data: data.update(schema_version=2),
    lambda data: data.update(schema_version=True),
    lambda data: data.update(duration_ms=True),
    lambda data: data.update(duration_ms="200"),
    lambda data: data.update(extra_private_prompt="not allowed"),
    lambda data: data.update(release_id="../outside"),
    lambda data: data.update(published_at="2026-10-22T12:00:00+00:00"),
    lambda data: data.update(published_at="2026-99-22T12:00:00Z"),
    lambda data: data["assets"][0].update(size_bytes=True),
    lambda data: data["assets"][0].update(mime_type="text/plain"),
    lambda data: data["assets"][1].update(role="audio"),
])
def test_strict_manifest_rejects_unknown_or_coerced_values(release, mutation):
    root, _ = release
    change_manifest(root, mutation)
    with pytest.raises(ContractError, match="invalid manifest"):
        validate_package(root)


@pytest.mark.parametrize("unsafe", [
    "../outside.wav", "/outside.wav", "folder/../audio.wav",
    "folder/./audio.wav", "folder//audio.wav", "folder\\audio.wav",
    "C:/audio.wav", "NUL.wav", "folder/COM1.wav", "audio.wav.",
    "manifest.json",
])
def test_unsafe_asset_paths_are_rejected_before_access(release, unsafe):
    root, _ = release
    change_manifest(root, lambda data: data["assets"][0].update(path=unsafe))
    with pytest.raises(ContractError, match="invalid manifest"):
        validate_package(root)


def test_modified_audio_with_same_size_is_rejected(release):
    root, _ = release
    path = root / "audio.wav"
    raw = bytearray(path.read_bytes())
    raw[-20] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ContractError, match="checksum mismatch"):
        validate_package(root)


def test_missing_audio_is_rejected(release):
    root, _ = release
    (root / "audio.wav").unlink()
    with pytest.raises(ContractError, match="inventory"):
        validate_package(root)


def test_extra_private_file_is_rejected(release):
    root, _ = release
    (root / "private-prompt.txt").write_text("This must not cross the boundary")
    with pytest.raises(ContractError, match="inventory"):
        validate_package(root)


def test_symlink_asset_is_rejected(release):
    root, _ = release
    transcript = root / "transcript.txt"
    outside = root.parent / "outside.txt"
    outside.write_bytes(transcript.read_bytes())
    transcript.unlink()
    try:
        transcript.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"filesystem does not support test symlinks: {exc}")
    with pytest.raises(ContractError, match="symbolic links"):
        validate_package(root)


def test_duplicate_json_keys_are_rejected(release):
    root, _ = release
    path = root / "manifest.json"
    data = path.read_text()
    path.write_text('{"schema_version": 1, ' + data[1:], encoding="utf-8")
    with pytest.raises(ContractError, match="duplicate JSON key"):
        validate_package(root)


def test_fake_duration_is_rejected_after_rehash(release):
    root, _ = release
    change_manifest(root, lambda data: data.update(duration_ms=5000))
    with pytest.raises(ContractError, match="duration differs"):
        validate_package(root)


def test_invalid_utf8_transcript_is_rejected_even_with_updated_hash(release):
    root, _ = release
    transcript = root / "transcript.txt"
    transcript.write_bytes(b"\xff\xfeinvalid")
    change_manifest(root, lambda data: data["assets"][1].update(
        size_bytes=transcript.stat().st_size,
        sha256=hashlib.sha256(transcript.read_bytes()).hexdigest(),
    ))
    with pytest.raises(ContractError, match="UTF-8"):
        validate_package(root)


def test_garbage_audio_is_rejected_even_with_updated_hash(release):
    root, _ = release
    audio = root / "audio.wav"
    audio.write_bytes(b"not a wave file")
    change_manifest(root, lambda data: data["assets"][0].update(
        size_bytes=audio.stat().st_size,
        sha256=hashlib.sha256(audio.read_bytes()).hexdigest(),
    ))
    with pytest.raises(ContractError, match="audio"):
        validate_package(root)


def test_create_never_overwrites_existing_release(release, source_wav):
    root, original = release
    original_bytes = (root / "manifest.json").read_bytes()
    with pytest.raises(ContractError, match="immutable"):
        create_release(source_wav, "Different transcript", root, "other", "Other")
    assert (root / "manifest.json").read_bytes() == original_bytes
    assert validate_package(root) == original


def test_import_uses_an_independent_verified_copy(release, tmp_path):
    source, manifest = release
    imported, path = import_release(source, tmp_path / "station", manifest.package_sha256)
    assert imported == manifest
    assert path == tmp_path / "station" / manifest.release_id
    assert validate_package(path, manifest.package_sha256) == manifest
    (source / "audio.wav").write_bytes(b"studio changed after import")
    assert validate_package(path, manifest.package_sha256) == manifest


def test_import_requires_expected_publication_digest(release, tmp_path):
    source, _ = release
    with pytest.raises(ContractError, match="trusted publication digest"):
        import_release(source, tmp_path / "station", "")


def test_concurrent_imports_are_idempotent_and_do_not_expose_staging(release, tmp_path):
    source, manifest = release
    destination = tmp_path / "station"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda _: import_release(source, destination, manifest.package_sha256), range(2),
        ))
    assert results[0] == results[1]
    assert [path.name for path in destination.iterdir() if path.is_dir()] == [manifest.release_id]
    assert validate_package(results[0][1], manifest.package_sha256) == manifest


def test_import_cannot_replace_existing_release_identity(release, tmp_path, source_wav):
    source, manifest = release
    _, path = import_release(source, tmp_path / "station", manifest.package_sha256)
    replacement_dir = tmp_path / "replacement"
    replacement = create_release(
        source_wav, "Different text", replacement_dir,
        programme_id="first-programme", title="Replacement",
        release_id=manifest.release_id,
    )
    with pytest.raises(ContractError):
        import_release(replacement_dir, tmp_path / "station", replacement.package_sha256)
    assert validate_package(path, manifest.package_sha256) == manifest

@pytest.mark.parametrize("mutation", [
    lambda data: data.update(schema_version=2),
    lambda data: data.update(schema_version=True),
    lambda data: data.update(duration_ms=True),
    lambda data: data.update(duration_ms="200"),
    lambda data: data.update(extra_private_prompt="not allowed"),
    lambda data: data.update(release_id="../outside"),
    lambda data: data.update(published_at="2026-10-22T12:00:00+00:00"),
    lambda data: data.update(published_at="2026-99-22T12:00:00Z"),
    lambda data: data["assets"][0].update(path="../outside.wav"),
    lambda data: data["assets"][0].update(path="folder/../audio.wav"),
    lambda data: data["assets"][0].update(path="NUL.wav"),
    lambda data: data["assets"][0].update(path="audio.wav."),
    lambda data: data["assets"][0].update(path="manifest.json"),
    lambda data: data["assets"][0].update(mime_type="text/plain"),
    lambda data: data["assets"][1].update(role="audio"),
])
def test_json_schema_rejects_expressible_type_and_semantic_errors(release, mutation):
    root, _ = release
    data = json.loads((root / "manifest.json").read_text())
    mutation(data)
    schema_path = Path(__file__).resolve().parents[1] / "packages/media-contract/manifest.schema.json"
    validator = Draft202012Validator(json.loads(schema_path.read_text()), format_checker=FormatChecker())
    assert list(validator.iter_errors(data)), "Schema accepted a value rejected by Python types"
