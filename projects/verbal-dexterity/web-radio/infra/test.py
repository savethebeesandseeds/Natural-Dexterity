"""Run checks in one disposable container scratch folder; retain results."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

project = Path("/workspace")
report_root = project / ".runtime" / "verification"
report_root.mkdir(parents=True, exist_ok=True)
run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
reports = Path(tempfile.mkdtemp(prefix="tests-" + run_stamp + "-", dir=report_root))
print("Retained verification reports:", reports, flush=True)
scratch = Path(tempfile.mkdtemp(prefix="web-radio-tests-", dir="/tmp"))
try:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "--basetemp=" + str(scratch), "--junitxml=" + str(reports / "junit.xml"),
         str(project / "tests")], cwd=project, capture_output=True, text=True)
    output = result.stdout + result.stderr
    (reports / "pytest.log").write_text(output)
    print(output, end="")
    raise SystemExit(result.returncode)
finally:
    resolved = scratch.resolve()
    if resolved.parent != Path("/tmp") or not resolved.name.startswith("web-radio-tests-") or scratch.is_symlink():
        raise RuntimeError("Unexpected test scratch path; preserved for inspection")
    shutil.rmtree(resolved)
