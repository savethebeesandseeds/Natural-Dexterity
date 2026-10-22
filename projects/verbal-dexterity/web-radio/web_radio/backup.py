"""Offline, checksum-verified snapshots; credentials are excluded unless explicitly requested."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import tarfile
import tempfile
import uuid

MANIFEST = "BACKUP-MANIFEST.json"
ALLOWED_ROOTS = {"studio", "library", "station", "secrets"}
MAX_MANIFEST = 8 * 1024 * 1024


class BackupError(ValueError):
    pass


def _absolute(path: Path) -> Path:
    result = Path(os.path.abspath(path))
    for parent in reversed([result, *result.parents]):
        if parent.is_symlink():
            raise BackupError("Symlink paths are not supported for backup or restore.")
    return result


def _hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _station_data_path(name: str) -> bool:
    """Keep durable release media and narrowly named authored station sidecars."""
    return (name == "station/journal.sqlite3" or name.startswith("station/releases/")
            or re.fullmatch(
                r"station/(?:performance/[a-z0-9][a-z0-9-]{0,127}\.json"
                r"|embeddings/[a-z0-9][a-z0-9-]{0,127}/[0-9a-f]{64}/"
                r"(?:manifest\.json|vectors-[0-9]{4}\.f32))", name) is not None)


def _safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    components = name.split("/")
    if (not name or "\\" in name or path.is_absolute() or path.as_posix() != name
            or any(p in ("", ".", "..") or ":" in p or "\x00" in p for p in components)
            or components[0] not in ALLOWED_ROOTS):
        raise BackupError("Archive contains an unsafe or unexpected data path.")
    if components[0] == "station" and not _station_data_path(name):
        raise BackupError("Archive contains derived or unexpected station data.")
    if name.endswith(("-wal", "-shm", ".log")):
        raise BackupError("Archive contains derived data excluded from this backup format.")
    return path


def _assert_offline(source: Path):
    for name, target in (("studio", "web_radio.studio:app"), ("station", "web_radio.station:app")):
        record_path = source / "control" / (name + ".json")
        if not record_path.exists():
            continue
        try:
            record = json.loads(record_path.read_text())
            pid = record["pid"]
            if type(pid) is not int or pid <= 0:
                raise ValueError
            command_path = Path("/proc") / str(pid) / "cmdline"
            if not command_path.exists():
                continue
            command = command_path.read_bytes().split(b"\0")
            if not any(command):
                continue
            if target.encode() not in command or b"uvicorn" not in command:
                raise BackupError("A control record points to another live process; inspect it first.")
            raise BackupError("Stop the managed studio and station before taking a consistent backup.")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, BackupError):
                raise
            raise BackupError("Cannot verify that managed services are offline.") from exc


def _source_files(source: Path, *, include_secrets: bool = False):
    for root_name in sorted(ALLOWED_ROOTS):
        # Skip before inspecting the root: normal backups never read credentials.
        # Backups and restored trees are excluded in both modes by ALLOWED_ROOTS.
        if root_name == "secrets" and not include_secrets:
            continue
        root = source / root_name
        if not root.exists() and not root.is_symlink():
            continue
        if root.is_symlink() or not root.is_dir():
            raise BackupError("Persistent data roots must be regular directories.")
        for path in sorted(root.rglob("*")):
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise BackupError("Persistent data contains a symlink.")
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                raise BackupError("Persistent data contains a special file.")
            relative = path.relative_to(source).as_posix()
            if relative.endswith(("-wal", "-shm", ".log", ".lock")):
                continue
            if root_name == "station" and not _station_data_path(relative):
                continue
            _safe_path(relative)
            yield relative, path


def _copy_snapshot(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix in (".sqlite3", ".sqlite", ".db"):
        # Read-only SQLite backup includes committed WAL pages without mutating the source DB.
        with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as original:
            with sqlite3.connect(target) as snapshot:
                original.backup(snapshot)
    else:
        with source.open("rb") as original, target.open("xb") as snapshot:
            shutil.copyfileobj(original, snapshot, length=1024 * 1024)
    target.chmod(0o600)



@contextmanager
def _station_offline_lock(source: Path):
    station = source / "station"
    if not station.exists():
        yield
        return
    path = _absolute(station / "playout-owner.lock")
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BackupError("The station is running; stop it before taking an offline backup.") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def _production_offline_locks(source: Path):
    """Exclude generation/design dispatch for the entire persistent snapshot."""
    studio = source / "studio"
    if not studio.exists() and not studio.is_symlink():
        yield
        return
    studio = _absolute(studio)
    if not studio.is_dir():
        raise BackupError("Backend production data is not an ordinary directory.")
    with ExitStack() as held:
        for name in ("generation-worker.lock", "design-worker.lock"):
            path = _absolute(studio / name)
            try:
                descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
                handle = held.enter_context(os.fdopen(descriptor, "r+b"))
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise BackupError("Backend production lease is not an ordinary file.")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise BackupError("Backend production is active; stop it before taking an offline backup.") from None
                held.callback(fcntl.flock, handle, fcntl.LOCK_UN)
            except OSError:
                raise BackupError("Cannot safely acquire the backend production lease.") from None
        yield


def backup_runtime(source_runtime: Path, destination_archive: Path, *, include_secrets: bool = False) -> Path:
    """Snapshot offline data; opt in to credentials only for a private, unencrypted archive."""
    source = _absolute(Path(source_runtime))
    _assert_offline(source)
    with ExitStack() as held:
        held.enter_context(_station_offline_lock(source))
        held.enter_context(_production_offline_locks(source))
        return _backup_unlocked(source, destination_archive, include_secrets=include_secrets)


def _backup_unlocked(source_runtime: Path, destination_archive: Path, *, include_secrets: bool = False) -> Path:

    source = _absolute(Path(source_runtime))
    destination = _absolute(Path(destination_archive))
    if not source.is_dir():
        raise BackupError("Source runtime does not exist.")
    if destination.exists():
        raise BackupError("Backup destination already exists; refusing to overwrite.")
    if destination.is_relative_to(source) and not destination.is_relative_to(source / "backups"):
        raise BackupError("Archives inside a runtime must stay in its excluded backups directory.")
    _assert_offline(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = list(_source_files(source, include_secrets=include_secrets))
    with tempfile.TemporaryDirectory(prefix=".radio-backup-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        records = []
        for relative, original in files:
            target = stage / "data" / relative
            _copy_snapshot(original, target)
            records.append({"path": relative, "size_bytes": target.stat().st_size,
                            "sha256": _hash(target)})
        manifest = {"schema_version": 1,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "files": records}
        archive_path = stage / "snapshot.tar.gz"
        with tarfile.open(archive_path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
            payload = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
            info = tarfile.TarInfo(MANIFEST)
            info.size, info.mode = len(payload), 0o600
            archive.addfile(info, io.BytesIO(payload))
            for record in records:
                archive.add(stage / "data" / record["path"],
                            arcname="data/" + record["path"], recursive=False)
        # Exclusive creation preserves every preexisting destination, including race winners.
        try:
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise BackupError("Backup destination already exists; refusing to overwrite.") from exc
        try:
            with os.fdopen(descriptor, "wb") as output, archive_path.open("rb") as input_file:
                shutil.copyfileobj(input_file, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            # Only the exclusively created, task-owned incomplete file is removed.
            destination.unlink(missing_ok=True)
            raise
    return destination


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise BackupError("Duplicate backup manifest key.")
        result[key] = value
    return result


def restore_runtime(source_archive: Path, destination_runtime: Path) -> Path:
    """Validate before extraction; restore exclusively into a fresh path, never merge."""
    source = _absolute(Path(source_archive))
    destination = _absolute(Path(destination_runtime))
    if not source.is_file():
        raise BackupError("Backup archive does not exist.")
    if destination.exists():
        raise BackupError("Restore destination already exists; refusing to merge or overwrite.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".radio-restore-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        try:
            with tarfile.open(source, "r:gz") as archive:
                members = archive.getmembers()
                by_name = {}
                for member in members:
                    if member.name in by_name or not member.isfile():
                        raise BackupError("Archive contains duplicate, linked, or special members.")
                    by_name[member.name] = member
                info = by_name.get(MANIFEST)
                if info is None or info.size > MAX_MANIFEST:
                    raise BackupError("Backup manifest is missing or too large.")
                manifest = json.load(archive.extractfile(info), object_pairs_hook=_unique_object)
                if (not isinstance(manifest, dict)
                        or set(manifest) != {"schema_version", "created_at", "files"}
                        or type(manifest["schema_version"]) is not int
                        or manifest["schema_version"] != 1
                        or not isinstance(manifest["created_at"], str)
                        or not isinstance(manifest["files"], list)):
                    raise BackupError("Unsupported backup manifest.")
                expected = {MANIFEST}
                for record in manifest["files"]:
                    if (not isinstance(record, dict) or set(record) != {"path", "size_bytes", "sha256"}
                            or not isinstance(record["path"], str)
                            or type(record["size_bytes"]) is not int or record["size_bytes"] < 0
                            or not isinstance(record["sha256"], str)
                            or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
                        raise BackupError("Invalid backup asset record.")
                    relative = _safe_path(record["path"])
                    member_name = "data/" + relative.as_posix()
                    if member_name in expected:
                        raise BackupError("Duplicate backup asset.")
                    expected.add(member_name)
                    member = by_name.get(member_name)
                    if not member or member.size != record["size_bytes"]:
                        raise BackupError("Backup inventory or size does not match.")
                    target = stage / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as original, target.open("xb") as output:
                        shutil.copyfileobj(original, output, length=1024 * 1024)
                    if _hash(target) != record["sha256"]:
                        raise BackupError("Backup content checksum does not match.")
                if set(by_name) != expected:
                    raise BackupError("Archive contains undeclared members.")
        except (tarfile.TarError, json.JSONDecodeError, UnicodeDecodeError, EOFError) as exc:
            raise BackupError("Backup archive cannot be read safely.") from exc
        try:
            destination.mkdir(mode=0o700, exist_ok=False)
        except FileExistsError as exc:
            raise BackupError("Restore destination already exists; refusing to overwrite.") from exc
        # All content was verified before claiming the fresh destination.
        # Failure preserves the new partial restore for inspection; no source is deleted.
        for record in manifest["files"]:
            relative = PurePosixPath(record["path"])
            output_path = destination / relative
            output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with (stage / relative).open("rb") as original, output_path.open("xb") as output:
                shutil.copyfileobj(original, output, length=1024 * 1024)
            output_path.chmod(0o600)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("backup", "restore"))
    parser.add_argument("archive", nargs="?", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--include-secrets", action="store_true",
                        help="backup only: include credentials in the unencrypted archive; keep it private")
    args = parser.parse_args()
    if args.action != "backup" and args.include_secrets:
        parser.error("--include-secrets is supported only for backup; restore preserves the archive contents")
    runtime = _absolute(Path(os.environ.get("WEB_RADIO_DATA", "/workspace/.runtime")))
    control = runtime / "control"
    control.mkdir(parents=True, exist_ok=True)
    try:
        with (control / "operation.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.action == "backup":
                if args.archive or args.destination:
                    parser.error("backup chooses a new archive under .runtime/backups; no path arguments")
                name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8] + ".tar.gz"
                result = backup_runtime(runtime, runtime / "backups" / name, include_secrets=args.include_secrets)
            else:
                if args.archive is None or args.destination is None:
                    parser.error("restore requires ARCHIVE --destination PATH inside .runtime/restores")
                destination = _absolute(args.destination)
                restore_root = _absolute(runtime / "restores")
                if destination == restore_root or not destination.is_relative_to(restore_root):
                    raise BackupError("Restore destination must be a fresh child of .runtime/restores.")
                result = restore_runtime(args.archive, destination)
        print(json.dumps({"ok": True, "action": args.action, "path": str(result)}))
    except (BackupError, OSError, sqlite3.Error) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
