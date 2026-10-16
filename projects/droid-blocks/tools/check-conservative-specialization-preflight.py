#!/usr/bin/env python3
"""Run new synthetic contracts without constructing native worlds."""
from datetime import datetime, timezone
import importlib.util
import io
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "training"))
from generalization_pretraining import require_container
from conservative_specialization_manifest import ref, write_new


def main():
    require_container()
    suite = unittest.TestSuite()
    files = ("tests/check_conservative_specialization_manifest.py",
             "tests/check_conservative_specialization_driver.py")
    for index, name in enumerate(files):
        spec = importlib.util.spec_from_file_location(f"conservative_preflight_{index}", PROJECT / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
    console = io.StringIO()
    result = unittest.TextTestRunner(stream=console, verbosity=2).run(suite)
    value = {"schema": "conservative_specialization_synthetic_preflight_v1",
             "created_at_utc": datetime.now(timezone.utc).isoformat(),
             "native_worlds_constructed": 0, "passed": result.wasSuccessful(),
             "tests_run": result.testsRun, "failures": len(result.failures),
             "errors": len(result.errors), "test_sources": [ref(PROJECT / name) for name in files],
             "console": console.getvalue()}
    folder = PROJECT / "artifacts/reports/conservative-specialization-preservation-v1-20261009"
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / "preflight.json"
    write_new(output, value)
    print(console.getvalue(), end="")
    print({"passed": value["passed"], "tests": value["tests_run"], "receipt": str(output)})
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
