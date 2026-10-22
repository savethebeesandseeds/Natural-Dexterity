"""Offline fixture recovery verifies retained media, SQLite state, and archive boundaries."""
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import tarfile

import pytest

from web_radio.backup import BackupError, MANIFEST, backup_runtime, restore_runtime
from web_radio.contract import validate_package
from web_radio.playout import Journal, Playout
from test_station import publish


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_default_backup_restores_media_sqlite_wal_without_copying_credentials(tmp_path):
    runtime = tmp_path / "original"
    manifest, library_package = publish(runtime, tmp_path)
    station = Playout(runtime)
    station.refresh()
    station.journal.session("fixture")
    airing = station.journal.begin("fixture", manifest, 0)
    station.journal.finish(airing, manifest.duration_ms)
    secret = runtime / "secrets" / "gemini-api-key"
    secret.parent.mkdir()
    secret.write_text("TEST KEY FIXTURE — NEVER A REAL CREDENTIAL")
    studio = runtime / "studio"
    (studio / "takes" / "fixture").mkdir(parents=True)
    request = studio / "takes" / "fixture" / "request.json"
    request.write_text('{"fixture":true}')
    database = studio / "studio.sqlite3"
    writer = sqlite3.connect(database)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE takes(id TEXT PRIMARY KEY, status TEXT)")
    writer.execute("INSERT INTO takes VALUES('fixture','accepted')")
    writer.commit()
    # Keep WAL open: read-only backup must include committed WAL pages.
    before = {p.relative_to(runtime).as_posix(): sha(p)
              for p in (database, request, secret, library_package / "manifest.json")}
    (runtime / "station" / "hls").mkdir(exist_ok=True)
    (runtime / "station" / "hls" / "derived.ts").write_bytes(b"DERIVED")
    (runtime / "station" / "encoder.log").write_text("DERIVED")
    (runtime / "backups").mkdir()
    (runtime / "backups" / "prior.tar.gz").write_bytes(b"DO NOT RECURSE")
    recovered_secret = runtime / "restores" / "prior" / "secrets" / "gemini-api-key"
    recovered_secret.parent.mkdir(parents=True)
    recovered_secret.write_text("OLD RESTORED KEY FIXTURE")
    archive = backup_runtime(runtime, tmp_path / "backup.tar.gz")
    with tarfile.open(archive, "r:gz") as snapshot:
        names = snapshot.getnames()
        assert not any(name.startswith(("data/secrets/", "data/restores/", "data/backups/")) for name in names)
    restored = restore_runtime(archive, tmp_path / "restored")
    assert before == {name: sha(runtime / name) for name in before}
    writer.close()
    assert not (restored / "secrets").exists()
    assert recovered_secret.read_text() == "OLD RESTORED KEY FIXTURE"
    assert (restored / "studio" / "takes" / "fixture" / "request.json").read_bytes() == request.read_bytes()
    with sqlite3.connect(restored / "studio" / "studio.sqlite3") as db:
        assert db.execute("SELECT status FROM takes WHERE id='fixture'").fetchone()[0] == "accepted"
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert validate_package(restored / "library" / "releases" / manifest.release_id,
                            expected_digest=manifest.package_sha256).release_id == manifest.release_id
    assert validate_package(restored / "station" / "releases" / manifest.release_id,
                            expected_digest=manifest.package_sha256).release_id == manifest.release_id
    recovered_journal = Journal(restored / "station" / "journal.sqlite3")
    assert recovered_journal.last_completed()["id"] == airing
    assert not (restored / "station" / "hls").exists()
    assert not (restored / "station" / "encoder.log").exists()
    assert not (restored / "backups").exists()
    assert not (restored / "restores").exists()
    assert not (restored / "control").exists()


def test_explicit_secret_backup_roundtrip_excludes_old_restores(tmp_path):
    runtime = tmp_path / "runtime"
    secret = runtime / "secrets" / "gemini-api-key"
    secret.parent.mkdir(parents=True)
    secret.write_text("TEST KEY FIXTURE — NEVER A REAL CREDENTIAL")
    recovered_secret = runtime / "restores" / "prior" / "secrets" / "old-key"
    recovered_secret.parent.mkdir(parents=True)
    recovered_secret.write_text("OLD RESTORED KEY FIXTURE")
    before = secret.read_bytes()
    archive = backup_runtime(runtime, tmp_path / "with-secrets.tar.gz", include_secrets=True)
    assert archive.stat().st_mode & 0o777 == 0o600
    with tarfile.open(archive, "r:gz") as snapshot:
        assert "data/secrets/gemini-api-key" in snapshot.getnames()
        assert not any(name.startswith("data/restores/") for name in snapshot.getnames())
    restored = restore_runtime(archive, tmp_path / "restored")
    assert (restored / "secrets" / "gemini-api-key").read_bytes() == before
    assert (restored / "secrets" / "gemini-api-key").stat().st_mode & 0o777 == 0o600
    assert secret.read_bytes() == before
    assert not (restored / "restores").exists()


def test_backup_and_restore_never_overwrite_existing_data(tmp_path):
    runtime = tmp_path / "runtime"
    (runtime / "secrets").mkdir(parents=True)
    (runtime / "secrets" / "key").write_text("fixture")
    archive = backup_runtime(runtime, tmp_path / "backup.tar.gz")
    before = archive.read_bytes()
    with pytest.raises(BackupError, match="already exists"):
        backup_runtime(runtime, archive)
    assert archive.read_bytes() == before
    destination = tmp_path / "restore"
    destination.mkdir()
    sentinel = destination / "keep"
    sentinel.write_text("preserve")
    with pytest.raises(BackupError, match="already exists"):
        restore_runtime(archive, destination)
    assert sentinel.read_text() == "preserve"


def test_backup_refuses_symlink_in_persistent_data(tmp_path):
    runtime = tmp_path / "runtime"
    (runtime / "studio").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_text("private outside the project")
    (runtime / "studio" / "record").symlink_to(outside)
    with pytest.raises(BackupError, match="symlink"):
        backup_runtime(runtime, tmp_path / "backup.tar.gz")
    assert not (tmp_path / "backup.tar.gz").exists()


def test_default_backup_skips_secret_root_before_inspecting_it(tmp_path):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "key").write_text("PRIVATE FIXTURE")
    (runtime / "secrets").symlink_to(outside, target_is_directory=True)
    archive = backup_runtime(runtime, tmp_path / "backup.tar.gz")
    with tarfile.open(archive, "r:gz") as snapshot:
        assert snapshot.getnames() == [MANIFEST]
    with pytest.raises(BackupError, match="regular directories"):
        backup_runtime(runtime, tmp_path / "with-secrets.tar.gz", include_secrets=True)
    assert not (tmp_path / "with-secrets.tar.gz").exists()


def crafted_archive(path, files, *, declared=None, special=None):
    records = [{"path": name, "size_bytes": len(value),
                "sha256": hashlib.sha256(value).hexdigest()} for name, value in files.items()]
    manifest = {"schema_version": 1, "created_at": "fixture", "files": declared or records}
    with tarfile.open(path, "w:gz") as tar:
        payload = json.dumps(manifest).encode()
        entry = tarfile.TarInfo(MANIFEST)
        entry.size = len(payload)
        tar.addfile(entry, io.BytesIO(payload))
        for name, value in files.items():
            entry = tarfile.TarInfo("data/" + name)
            entry.size = len(value)
            tar.addfile(entry, io.BytesIO(value))
        if special:
            tar.addfile(special)
    return path


def test_restore_remains_compatible_with_existing_secret_bearing_archives(tmp_path):
    archive = crafted_archive(tmp_path / "old-schema-v1.tar.gz", {"secrets/key": b"LEGACY FIXTURE"})
    restored = restore_runtime(archive, tmp_path / "restored")
    assert (restored / "secrets" / "key").read_bytes() == b"LEGACY FIXTURE"


@pytest.mark.parametrize("path", [
    "../escape", "/absolute", "studio/../../escape", "studio\\escape",
    "station/hls/derived.ts", "control/station.json", "backups/recurse.tar.gz",
    "secrets/key:stream",
])
def test_restore_rejects_traversal_and_unexpected_paths_before_creating_destination(tmp_path, path):
    archive = crafted_archive(tmp_path / "unsafe.tar.gz", {path: b"fixture"})
    destination = tmp_path / "restored"
    with pytest.raises(BackupError):
        restore_runtime(archive, destination)
    assert not destination.exists()
    assert not (tmp_path / "escape").exists()


def test_restore_rejects_content_tampering_and_undeclared_members(tmp_path):
    declared = [{"path": "secrets/key", "size_bytes": 7, "sha256": "0" * 64}]
    archive = crafted_archive(tmp_path / "corrupt.tar.gz", {"secrets/key": b"fixture"}, declared=declared)
    with pytest.raises(BackupError, match="checksum"):
        restore_runtime(archive, tmp_path / "restored")
    extra = tarfile.TarInfo("data/studio/extra.json")
    extra.size = 0
    archive = crafted_archive(tmp_path / "extra.tar.gz", {"secrets/key": b"fixture"}, special=extra)
    with pytest.raises(BackupError, match="undeclared"):
        restore_runtime(archive, tmp_path / "restored")


def test_restore_rejects_symlink_archive_members(tmp_path):
    special = tarfile.TarInfo("data/secrets/link")
    special.type = tarfile.SYMTYPE
    special.linkname = "/etc/passwd"
    archive = crafted_archive(tmp_path / "symlink.tar.gz", {}, special=special)
    with pytest.raises(BackupError, match="linked"):
        restore_runtime(archive, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_backup_refuses_live_managed_control_record_without_stopping_process(tmp_path):
    runtime = tmp_path / "runtime"
    control = runtime / "control"
    control.mkdir(parents=True)
    # Temporary Python process uses the exact managed command markers.
    process = subprocess.Popen(["/opt/web-radio/venv/bin/python", "-c",
                                "import time;time.sleep(30)", "uvicorn", "web_radio.station:app"])
    try:
        (control / "station.json").write_text(json.dumps({"pid": process.pid}))
        with pytest.raises(BackupError, match="Stop the managed"):
            backup_runtime(runtime, tmp_path / "backup.tar.gz")
        assert process.poll() is None
        assert not (tmp_path / "backup.tar.gz").exists()
    finally:
        process.terminate()
        process.wait()


def test_backup_refuses_unmanaged_active_playout_owner(tmp_path):
    runtime = tmp_path / "runtime"
    station = Playout(runtime)
    station.start()
    try:
        with pytest.raises(BackupError, match="station is running"):
            backup_runtime(runtime, tmp_path / "backup.tar.gz")
        assert station.thread.is_alive()
    finally:
        station.stop()


def test_backup_cli_and_restore_destination_boundary(tmp_path):
    import sys
    runtime = tmp_path / "runtime"
    (runtime / "secrets").mkdir(parents=True)
    (runtime / "secrets" / "key").write_text("private fixture")
    env = dict(os.environ, WEB_RADIO_DATA=str(runtime))
    backed = subprocess.run([sys.executable, "-m", "web_radio.backup", "backup"],
                            env=env, check=True, capture_output=True, text=True)
    archive = Path(json.loads(backed.stdout)["path"])
    assert archive.parent == runtime / "backups"
    outside = tmp_path / "outside"
    refused = subprocess.run([sys.executable, "-m", "web_radio.backup", "restore",
                              str(archive), "--destination", str(outside)],
                             env=env, capture_output=True, text=True)
    assert refused.returncode == 1
    assert not outside.exists()
    destination = runtime / "restores" / "recovery-check"
    restored = subprocess.run([sys.executable, "-m", "web_radio.backup", "restore",
                               str(archive), "--destination", str(destination)],
                              env=env, check=True, capture_output=True, text=True)
    assert json.loads(restored.stdout)["ok"] is True
    assert not (destination / "secrets").exists()
    with tarfile.open(archive, "r:gz") as snapshot:
        assert snapshot.getnames() == [MANIFEST]

    included = subprocess.run([sys.executable, "-m", "web_radio.backup", "backup", "--include-secrets"],
                              env=env, check=True, capture_output=True, text=True)
    private_archive = Path(json.loads(included.stdout)["path"])
    private_destination = runtime / "restores" / "private-recovery-check"
    restored_private = subprocess.run([sys.executable, "-m", "web_radio.backup", "restore",
                                      str(private_archive), "--destination", str(private_destination)],
                                     env=env, check=True, capture_output=True, text=True)
    assert json.loads(restored_private.stdout)["ok"] is True
    assert (private_destination / "secrets" / "key").read_text() == "private fixture"
    assert not (private_destination / "restores").exists()

    rejected = subprocess.run([sys.executable, "-m", "web_radio.backup", "restore",
                               str(private_archive), "--destination", str(runtime / "restores" / "rejected"),
                               "--include-secrets"], env=env, capture_output=True, text=True)
    assert rejected.returncode == 2
    assert "supported only for backup" in rejected.stderr
    assert not (runtime / "restores" / "rejected").exists()


@pytest.mark.parametrize("lease_name,shared", [
    ("generation-worker.lock", False), ("design-worker.lock", True),
])
def test_backup_refuses_active_backend_jobs_and_releases_leases_when_idle(tmp_path, lease_name, shared):
    import fcntl
    runtime = tmp_path / "runtime"
    studio = runtime / "studio"
    studio.mkdir(parents=True)
    (studio / "retained.json").write_text('{"fixture":true}')
    archive = tmp_path / "backup.tar.gz"
    with (studio / lease_name).open("a") as lease:
        fcntl.flock(lease, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        with pytest.raises(BackupError, match="Backend production is active"):
            backup_runtime(runtime, archive)
        assert not archive.exists()
        fcntl.flock(lease, fcntl.LOCK_UN)
    backup_runtime(runtime, archive)
    with tarfile.open(archive, "r:gz") as snapshot:
        assert "data/studio/retained.json" in snapshot.getnames()
        assert not any(name.endswith(".lock") for name in snapshot.getnames())


def test_backup_holds_both_backend_leases_during_the_actual_snapshot(tmp_path, monkeypatch):
    import fcntl
    from web_radio import backup
    runtime = tmp_path / "runtime"
    studio = runtime / "studio"
    studio.mkdir(parents=True)
    (studio / "retained.json").write_text('{"fixture":true}')
    original_copy = backup._copy_snapshot
    inspected = []
    def checked_copy(source, target):
        for name in ("generation-worker.lock", "design-worker.lock"):
            with (studio / name).open("a") as contender:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(contender, fcntl.LOCK_SH | fcntl.LOCK_NB)
            inspected.append(name)
        original_copy(source, target)
    monkeypatch.setattr(backup, "_copy_snapshot", checked_copy)
    archive = backup_runtime(runtime, tmp_path / "backup.tar.gz")
    assert archive.is_file()
    assert inspected == ["generation-worker.lock", "design-worker.lock"]
    for name in inspected:
        with (studio / name).open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lease, fcntl.LOCK_UN)


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo"])
def test_backup_production_lease_must_be_an_ordinary_file(tmp_path, kind):
    runtime = tmp_path / "runtime"
    studio = runtime / "studio"
    studio.mkdir(parents=True)
    lease = studio / "generation-worker.lock"
    outside = tmp_path / "outside"
    outside.write_text("retain outside")
    if kind == "symlink":
        lease.symlink_to(outside)
    elif kind == "directory":
        lease.mkdir()
    else:
        os.mkfifo(lease)
    archive = tmp_path / "backup.tar.gz"
    with pytest.raises(BackupError):
        backup_runtime(runtime, archive)
    assert not archive.exists()
    assert outside.read_text() == "retain outside"


def test_restore_keeps_compatibility_with_legacy_unlocked_lock_members(tmp_path):
    archive = crafted_archive(tmp_path / "legacy.tar.gz", {"studio/generation-worker.lock": b"",
                                                          "studio/retained.json": b'{"fixture":true}'})
    restored = restore_runtime(archive, tmp_path / "restored")
    assert (restored / "studio/generation-worker.lock").read_bytes() == b""
    assert (restored / "studio/retained.json").read_bytes() == b'{"fixture":true}'


def test_default_backup_roundtrips_authored_station_sidecars_and_excludes_derived_files(tmp_path):
    runtime = tmp_path / "runtime"
    bundle = "a" * 64
    retained = {
        "station/performance/first-light.json": b'{"fixture":"authored timing"}',
        f"station/embeddings/first-light/{bundle}/manifest.json": b'{"fixture":"source embedding"}',
        f"station/embeddings/first-light/{bundle}/vectors-0000.f32": b"\0\0\x80?\0\0\0@",
        f"station/embeddings/first-light/{bundle}/vectors-0012.f32": b"\0\0@@",
    }
    excluded = [
        "station/hls/stream.m3u8", "station/hls/segment-0001.ts",
        "station/performance/First-Light.json",
        "station/performance/first-light/extra.json",
        "station/performance/first-light.txt",
        f"station/embeddings/first-light/{bundle}/request.json",
        f"station/embeddings/first-light/{bundle}/vectors-00000.f32",
        f"station/embeddings/first-light/{bundle}/.publication.lock",
        "station/embeddings/first-light/not-a-hash/manifest.json",
        "station/encoder.log",
    ]
    for name, data in retained.items():
        path = runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for name in excluded:
        path = runtime / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"DERIVED OR UNEXPECTED FIXTURE")
    archive = backup_runtime(runtime, tmp_path / "sidecars.tar.gz")
    with tarfile.open(archive, "r:gz") as snapshot:
        assert set(snapshot.getnames()) == {MANIFEST, *("data/" + name for name in retained)}
    restored = restore_runtime(archive, tmp_path / "restored")
    for name, data in retained.items():
        assert (restored / name).read_bytes() == data
        assert (runtime / name).read_bytes() == data
    assert all(not (restored / name).exists() for name in excluded)


@pytest.mark.parametrize("path", [
    "station/performance/First-Light.json",
    "station/performance/.json",
    "station/performance/first-light.txt",
    "station/performance/first-light/nested.json",
    "station/performance/" + "a" * 129 + ".json",
    "station/embeddings/First-Light/" + "a" * 64 + "/manifest.json",
    "station/embeddings/first-light/" + "A" * 64 + "/manifest.json",
    "station/embeddings/first-light/" + "a" * 63 + "/manifest.json",
    "station/embeddings/first-light/" + "a" * 64 + "/request.json",
    "station/embeddings/first-light/" + "a" * 64 + "/vectors-1.f32",
    "station/embeddings/first-light/" + "a" * 64 + "/vectors-00000.f32",
    "station/embeddings/first-light/" + "a" * 64 + "/nested/manifest.json",
])
def test_restore_rejects_malformed_station_sidecar_paths_before_claiming_destination(tmp_path, path):
    archive = crafted_archive(tmp_path / "malformed-sidecar.tar.gz", {path: b"fixture"})
    destination = tmp_path / "restored"
    with pytest.raises(BackupError, match="station data"):
        restore_runtime(archive, destination)
    assert not destination.exists()
