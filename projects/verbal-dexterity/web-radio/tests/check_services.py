"""Read-only checks of the two listener-facing development ports."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError


def read(url):
    with urlopen(url, timeout=10) as response:
        return response.status, response.headers, response.read()


def main():
    report = {"checked_at_unix": time.time(), "checks": []}
    private_paths = (
        "/api/credentials", "/api/settings", "/api/script", "/api/scripts/programme/head",
        "/api/generate", "/api/takes", "/api/programming", "/api/import",
        "/.runtime/secrets/gemini-api-key", "/assets/studio.js",
        "/assets/station-controls.js", "/docs", "/redoc", "/openapi.json",
    )
    for name, port in (("listener", 43120), ("station", 43121)):
        base = "http://127.0.0.1:" + str(port)
        status, _, body = read(base + "/api/status")
        state = json.loads(body)
        assert status == 200 and isinstance(state, dict)
        assert "key_configured" not in state and "budgets" not in state
        status, headers, html = read(base + "/")
        assert status == 200 and b"<html" in html.lower()
        assert "script-src 'self'" in headers["Content-Security-Policy"]
        assert b"/assets/player.js" in html and b"/assets/studio.js" not in html
        for suffix in private_paths:
            for method in ("GET", "POST", "PUT"):
                request = Request(base + suffix, method=method)
                try:
                    with urlopen(request, timeout=10):
                        raise AssertionError("Listener exposed a backend production path")
                except HTTPError as error:
                    assert error.code in {404, 405}, (port, suffix, method, error.code)
        report["checks"].append({
            "service": name, "port": port, "reachable": True,
            "listener_only": True, "production_routes_absent": True,
        })
    report_root = Path("/workspace/.runtime/verification")
    report_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = Path(tempfile.mkdtemp(prefix="services-" + run_stamp + "-", dir=report_root))
    (folder / "services.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("Retained verification reports:", folder)


if __name__ == "__main__":
    main()
