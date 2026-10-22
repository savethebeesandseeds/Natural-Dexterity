"""Install one pinned HLS browser asset inside the container; verify npm integrity."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen

VERSION = "1.6.13"
root = Path("/opt/web-radio/vendor")
root.mkdir(parents=True, exist_ok=True)
lock_path = Path("/workspace/infra/vendor.lock.json")
if lock_path.exists():
    lock = json.loads(lock_path.read_text())
    if lock["version"] != VERSION:
        raise SystemExit("Vendor lock version mismatch")
else:
    with urlopen("https://registry.npmjs.org/hls.js/" + VERSION, timeout=30) as response:
        package = json.load(response)
    lock = {"package": "hls.js", "version": VERSION, "tarball": package["dist"]["tarball"],
            "integrity": package["dist"]["integrity"]}
    lock_path.write_text(json.dumps(lock, indent=2) + "\n")
marker = root / "hls.sha256"
if (root / "hls.min.js").is_file() and marker.is_file():
    if hashlib.sha256((root / "hls.min.js").read_bytes()).hexdigest() == marker.read_text().strip():
        raise SystemExit(0)
with urlopen(lock["tarball"], timeout=60) as response:
    data = response.read(20_000_000)
algorithm, expected = lock["integrity"].split("-", 1)
if algorithm != "sha512" or base64.b64encode(hashlib.sha512(data).digest()).decode() != expected:
    raise SystemExit("HLS vendor integrity check failed")
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
    for source, target in [("package/dist/hls.min.js", "hls.min.js"), ("package/LICENSE", "HLS-LICENSE.txt")]:
        item = archive.getmember(source)
        if not item.isfile():
            raise SystemExit("Unexpected vendor archive member")
        with archive.extractfile(item) as stream:
            (root / target).write_bytes(stream.read())
marker.write_text(hashlib.sha256((root / "hls.min.js").read_bytes()).hexdigest() + "\n")

