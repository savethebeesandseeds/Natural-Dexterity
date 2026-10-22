from __future__ import annotations

import json
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict
import pytest

from web_radio import embedding_cli


SECRET = "fixture-private-value-never-print"


class FakeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    title: str


@pytest.fixture(autouse=True)
def forbid_network_and_environment_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Embedding CLI fixture attempted a network request")

    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)


@pytest.fixture
def operator(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    calls = []

    class Service:
        def __init__(self, project_root):
            calls.append(("construct", project_root))

        def plan(self, spec):
            calls.append(("plan", spec.model_dump()))
            return {"dispatch": False, "request_count": 2}

        def enqueue(self, spec, *, episode_limit_usd, series_limit_usd):
            calls.append(("enqueue", spec.model_dump(), episode_limit_usd, series_limit_usd))
            return {"id": "queued-job", "status": "queued"}

        def run_one(self, job_id):
            calls.append(("run", job_id))
            return {"id": job_id, "status": "ready"}

        def status(self, job_id):
            calls.append(("show", job_id))
            return {"id": job_id, "status": "queued"}

        def recover(self, job_id):
            calls.append(("recover", job_id))
            return {"id": job_id, "status": "uncertain"}

        def install(self, job_id):
            calls.append(("install", job_id))
            return {"id": job_id, "installed": True}

        def reconcile(self, job_id, *, request_sha256, charged_usd, evidence):
            calls.append(("reconcile", job_id, request_sha256, charged_usd, evidence))
            return {"id": job_id, "reconciled": True}

    monkeypatch.setattr(embedding_cli, "_components", lambda: (FakeSpec, Service))
    return root, calls, Service


def input_file(root, content=None):
    path = root / "input.json"
    path.write_text(json.dumps(content or {"title": "Embedding fixture"}), encoding="utf-8")
    return path


def invoke(root, capsys, *args):
    code = embedding_cli.main(["--root", str(root), *map(str, args)])
    captured = capsys.readouterr()
    assert not (captured.out and captured.err)
    return code, json.loads(captured.out if code == 0 else captured.err)


def test_plan_invokes_only_planning_without_queuing_dispatch_or_install(operator, capsys):
    root, calls, _ = operator
    code, result = invoke(root, capsys, "plan", "--file", input_file(root))
    assert code == 0 and result["result"]["dispatch"] is False
    assert calls == [("construct", root), ("plan", {"title": "Embedding fixture"})]
    assert not (root / ".runtime").exists()


def test_enqueue_keeps_explicit_budgets_and_never_runs_or_installs(operator, capsys):
    root, calls, _ = operator
    code, result = invoke(
        root, capsys, "enqueue", "--file", input_file(root),
        "--episode-budget", "0.25", "--series-budget", "1.000000",
    )
    assert code == 0 and result["result"]["status"] == "queued"
    assert calls == [
        ("construct", root),
        ("enqueue", {"title": "Embedding fixture"}, "0.25", "1.000000"),
    ]


@pytest.mark.parametrize("command", ["run", "show", "recover", "install"])
def test_job_commands_select_only_the_exact_named_job(operator, capsys, command):
    root, calls, _ = operator
    code, result = invoke(root, capsys, command, "--job", "job-two")
    assert code == 0 and result["result"]["id"] == "job-two"
    assert calls == [("construct", root), (command, "job-two")]


def test_reconcile_forwards_request_and_zero_charge_without_dispatch(operator, capsys):
    root, calls, _ = operator
    request = "b" * 64
    evidence = "Verified fixture billing evidence."
    code, result = invoke(
        root, capsys, "reconcile", "--job", "job-two", "--request", request,
        "--charged", "0", "--evidence", evidence,
    )
    assert code == 0 and result["result"]["reconciled"] is True
    assert calls == [("construct", root), ("reconcile", "job-two", request, "0", evidence)]


@pytest.mark.parametrize("raw", [
    b'{"title":"one","title":"two"}',
    b'{"title":',
    b'["not an object"]',
    b"\xff\xfe",
])
def test_invalid_json_is_rejected_before_service_initialization(operator, capsys, raw):
    root, calls, _ = operator
    path = root / "invalid.json"
    path.write_bytes(raw + SECRET.encode())
    code, result = invoke(root, capsys, "plan", "--file", path)
    assert code == 2 and result["ok"] is False
    assert SECRET not in json.dumps(result) and not calls


def test_oversize_input_is_bounded_before_service_initialization(operator, capsys):
    root, calls, _ = operator
    path = root / "large.json"
    path.write_bytes(b" " * (1024 * 1024 + 1))
    code, result = invoke(root, capsys, "plan", "--file", path)
    assert code == 2 and result["error"]["code"] == "input_too_large"
    assert not calls


def test_pydantic_inputs_and_arbitrary_field_names_are_not_echoed(operator, capsys):
    root, calls, _ = operator
    code, result = invoke(root, capsys, "plan", "--file", input_file(root, {SECRET: SECRET}))
    assert code == 2 and result["error"]["code"] == "validation_failed"
    assert SECRET not in json.dumps(result) and not calls


@pytest.mark.parametrize("budget", ["0", "-1", "NaN", SECRET])
def test_invalid_budgets_reject_without_initializing_service(operator, capsys, budget):
    root, calls, _ = operator
    code, result = invoke(
        root, capsys, "enqueue", "--file", input_file(root),
        "--episode-budget", budget, "--series-budget", "1",
    )
    assert code == 2 and SECRET not in json.dumps(result)
    assert not calls


@pytest.mark.parametrize("exception", [
    ValueError(SECRET), OSError(SECRET), RuntimeError(SECRET),
])
def test_service_errors_are_sanitized_without_traceback(operator, capsys, monkeypatch, exception):
    root, calls, Service = operator

    def fail(self, job_id):
        raise exception

    monkeypatch.setattr(Service, "run_one", fail)
    code, result = invoke(root, capsys, "run", "--job", "job-two")
    assert code == 2 and SECRET not in json.dumps(result)
    assert "Traceback" not in json.dumps(result)
    assert calls == [("construct", root)]


def test_unknown_argument_values_are_not_echoed(operator, capsys):
    root, calls, _ = operator
    code, result = invoke(root, capsys, SECRET)
    assert code == 2 and result["error"]["code"] == "invalid_arguments"
    assert SECRET not in json.dumps(result) and not calls


def test_missing_required_job_is_rejected_without_service(operator, capsys):
    root, calls, _ = operator
    code, result = invoke(root, capsys, "run")
    assert code == 2 and result["error"]["code"] == "invalid_arguments"
    assert not calls


def test_root_symlink_is_rejected_before_service_initialization(operator, tmp_path, capsys):
    root, calls, _ = operator
    link = tmp_path / "project-link"
    link.symlink_to(root, target_is_directory=True)
    code, result = invoke(link, capsys, "show", "--job", "job-two")
    assert code == 2 and result["error"]["code"] == "invalid_root"
    assert not calls


def test_default_root_is_workspace():
    args = embedding_cli.parser().parse_args(["show", "--job", "a-job"])
    assert args.root == Path("/workspace")


def test_help_does_not_initialize_service_or_load_spec(monkeypatch, capsys):
    def forbidden():
        pytest.fail("--help initialized the embedding service")

    monkeypatch.setattr(embedding_cli, "_components", forbidden)
    with pytest.raises(SystemExit) as exc:
        embedding_cli.main(["--help"])
    assert exc.value.code == 0
    assert "enqueue" in capsys.readouterr().out
