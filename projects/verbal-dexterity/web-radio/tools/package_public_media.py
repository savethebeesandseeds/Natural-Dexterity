#!/usr/bin/env python3
"""Package a verified public export for an immutable GitHub release asset."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web_radio.static_export import verify_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = verify_export(args.export)
    if args.output.exists() or args.output.is_symlink():
        parser.error("Archive already exists; choose a new name.")
    with args.output.open("xb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for path in sorted(args.export.rglob("*")):
                    if not path.is_file():
                        continue
                    info = archive.gettarinfo(str(path), arcname=path.relative_to(args.export).as_posix())
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    with path.open("rb") as source:
                        archive.addfile(info, source)
    digest = hashlib.file_digest(args.output.open("rb"), "sha256").hexdigest()
    print(json.dumps({"archive": str(args.output), "bytes": args.output.stat().st_size,
                      "sha256": digest, "manifest_sha256": receipt["manifest_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
