from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from web_radio import scene_cli


SECRET = "fixture-private-value-never-print"


@pytest.fixture(autouse=True)
def no_network_or_environment_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Scene CLI fixture attempted a network request")

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

        def plan(self, script, *, policy=None, budget_usd=None):
            calls.append(("plan", script.model_dump(), policy, budget_usd))
            return {"dispatch": False, "fits_budget": True}

        def create(self, script, *, budget_usd, policy=None):
            calls.append(("create", script.model_dump(), budget_usd, policy))
            return {"id": "new-job", "status": "queued"}

        def run(self, job_id, *, resume=False):
            calls.append(("run", job_id, resume))
            return {"id": job_id, "status": "ready"}

        def status(self, job_id):
            calls.append(("show", job_id))
            return {"id": job_id, "status": "queued"}

        def recover(self, job_id):
            calls.append(("recover", job_id))
            return {"id": job_id, "status": "uncertain"}

        def reconcile(self, job_id, *, attempt_id, charged_usd, evidence):
            calls.append(("reconcile", job_id, attempt_id, charged_usd, evidence))
            return {"id": job_id, "reconciled": True}

        def revise(self, job_id, *, line_index, text=None, style=None, recipe_id=None, budget_usd, policy=None):
            calls.append(("revise", job_id, line_index, text, style, recipe_id, budget_usd, policy))
            return {"id": "revision-job", "parent_id": job_id, "status": "queued"}

        def audition(self, character_id, text, *, style="", recipe_id=None, budget_usd):
            calls.append(("audition", character_id, text, style, recipe_id, budget_usd))
            return {"id": "audition-job", "status": "queued"}

        def import_sources(self, script, mapping, *, budget_usd, policy=None):
            calls.append(("import", script.model_dump(), mapping, budget_usd, policy))
            return {"id": "import-job", "status": "queued"}

    monkeypatch.setattr(scene_cli, "_service_class", lambda: Service)
    return root, calls, Service


def script_file(root):
    raw = {
        "programme_id": "creative-fixture", "series_id": "fixtures",
        "kind": "programme", "title": "Creative fixture", "language": "en",
        "speakers": [{"name": "Moss", "voice": "Charon"}],
        "turns": [
            {"speaker": "Moss", "text": "The door is open."},
            {"speaker": "Moss", "text": "Shall we go inside?"},
        ],
    }
    path = root / "scene.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def json_file(root, name, value):
    path = root / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def invoke(root, capsys, *args):
    code = scene_cli.main(["--root", str(root), *map(str, args)])
    captured = capsys.readouterr()
    assert not (captured.out and captured.err)
    return code, json.loads(captured.out if code == 0 else captured.err)


def test_plan_checks_script_and_optional_budget_without_creating_or_rendering(operator, capsys):
    root, calls, _ = operator
    code, output = invoke(root, capsys, "plan", "--file", script_file(root), "--budget", "0.25")
    assert code == 0 and output["result"]["dispatch"] is False
    assert [event[0] for event in calls] == ["construct", "plan"]
    assert calls[-1][2:] == (None, "0.25")
    assert len(calls[-1][1]["turns"]) == 2
    assert not (root / ".runtime").exists()


def test_plan_allows_preview_without_a_budget(operator, capsys):
    root, calls, _ = operator
    code, _ = invoke(root, capsys, "plan", "--file", script_file(root))
    assert code == 0 and calls[-1][3] is None


def test_create_freezes_only_and_preserves_exact_policy_and_budget(operator, capsys):
    root, calls, _ = operator
    policy = {"fixture_policy": "explicit-value"}
    code, output = invoke(
        root, capsys, "create", "--file", script_file(root), "--budget", "1.000000",
        "--policy", json_file(root, "policy.json", policy),
    )
    assert code == 0 and output["result"]["status"] == "queued"
    assert [event[0] for event in calls] == ["construct", "create"]
    assert calls[-1][2:] == ("1.000000", policy)


@pytest.mark.parametrize("command,resume", [("render", False), ("resume", True)])
def test_selected_job_rendering_is_explicit_and_resume_is_distinct(operator, capsys, command, resume):
    root, calls, _ = operator
    code, output = invoke(root, capsys, command, "--job", "job-two")
    assert code == 0 and output["result"]["id"] == "job-two"
    assert calls == [("construct", root), ("run", "job-two", resume)]


def test_show_calls_only_status(operator, capsys):
    root, calls, _ = operator
    code, _ = invoke(root, capsys, "show", "--job", "job-two")
    assert code == 0 and calls == [("construct", root), ("show", "job-two")]


def test_revision_maps_human_line_number_and_combines_changes_without_rendering(operator, capsys):
    root, calls, _ = operator
    recipe = "a" * 32
    code, output = invoke(
        root, capsys, "revise", "--job", "job-two", "--line", "2",
        "--text", "A different ending.", "--style", "Quietly, with relief.",
        "--recipe", recipe, "--budget", "0.25",
    )
    assert code == 0 and output["result"]["id"] == "revision-job"
    assert calls == [
        ("construct", root),
        ("revise", "job-two", 1, "A different ending.", "Quietly, with relief.", recipe, "0.25", None),
    ]


def test_revision_can_explicitly_clear_line_direction(operator, capsys):
    root, calls, _ = operator
    code, _ = invoke(root, capsys, "revise", "--job", "job-two", "--line", "1", "--style", "", "--budget", "1")
    assert code == 0 and calls[-1] == ("revise", "job-two", 0, None, "", None, "1", None)


@pytest.mark.parametrize("extra", [
    ["--line", "0", "--text", "change"],
    ["--line", "-1", "--text", "change"],
    ["--line", "1"],
])
def test_invalid_revision_is_rejected_before_service_creation(operator, capsys, extra):
    root, calls, _ = operator
    code, output = invoke(root, capsys, "revise", "--job", "job-two", "--budget", "1", *extra)
    assert code == 2 and output["ok"] is False and not calls


def test_audition_queues_one_selected_character_and_recipe_without_rendering(operator, capsys):
    root, calls, _ = operator
    recipe = "b" * 32
    code, output = invoke(
        root, capsys, "audition", "--character", "moss", "--recipe", recipe,
        "--text", "A short audition.", "--style", "Warm but cautious.", "--budget", "0.25",
    )
    assert code == 0 and output["result"]["status"] == "queued"
    assert calls == [
        ("construct", root),
        ("audition", "moss", "A short audition.", "Warm but cautious.", recipe, "0.25"),
    ]


def test_import_preserves_authored_mapping_without_rendering(operator, capsys):
    root, calls, _ = operator
    mapping = [{"turn_index": 0, "take_id": "c" * 32, "request_index": 0}]
    code, output = invoke(
        root, capsys, "import-sources", "--file", script_file(root),
        "--sources", json_file(root, "sources.json", {"sources": mapping}), "--budget", "0.25",
    )
    assert code == 0 and output["result"]["id"] == "import-job"
    assert [event[0] for event in calls] == ["construct", "import"]
    assert calls[-1][2:] == (mapping, "0.25", None)


@pytest.mark.parametrize("raw", [
    b'{"title":"one","title":"two"}', b'{"title":', b'["wrong shape"]', b"\xff\xfe",
])
def test_invalid_json_is_bounded_and_never_echoed(operator, capsys, raw):
    root, calls, _ = operator
    path = root / "invalid.json"
    path.write_bytes(raw + SECRET.encode())
    code, output = invoke(root, capsys, "plan", "--file", path)
    assert code == 2 and SECRET not in json.dumps(output) and not calls


def test_script_validation_does_not_echo_unknown_fields_or_values(operator, capsys):
    root, calls, _ = operator
    path = script_file(root)
    value = json.loads(path.read_text())
    value[SECRET] = SECRET
    path.write_text(json.dumps(value))
    code, output = invoke(root, capsys, "plan", "--file", path)
    assert code == 2 and output["error"]["code"] == "validation_failed"
    assert SECRET not in json.dumps(output) and not calls


def test_oversize_policy_is_rejected_before_service(operator, capsys):
    root, calls, _ = operator
    policy = root / "policy.json"
    policy.write_bytes(b" " * (1024 * 1024 + 1))
    code, output = invoke(root, capsys, "create", "--file", script_file(root), "--budget", "1", "--policy", policy)
    assert code == 2 and output["error"]["code"] == "input_too_large"
    assert not calls


@pytest.mark.parametrize("budget", ["0", "-1", "NaN", SECRET])
def test_invalid_budget_rejects_before_service_without_echoing_input(operator, capsys, budget):
    root, calls, _ = operator
    code, output = invoke(root, capsys, "create", "--file", script_file(root), "--budget", budget)
    assert code == 2 and SECRET not in json.dumps(output) and not calls


@pytest.mark.parametrize("exception", [ValueError(SECRET), OSError(SECRET), RuntimeError(SECRET)])
def test_service_errors_do_not_echo_exception_details(operator, capsys, monkeypatch, exception):
    root, calls, Service = operator

    def fail(self, job_id):
        raise exception

    monkeypatch.setattr(Service, "status", fail)
    code, output = invoke(root, capsys, "show", "--job", "job-two")
    assert code == 2 and SECRET not in json.dumps(output)
    assert "Traceback" not in json.dumps(output)


def test_unknown_argument_value_is_not_echoed(operator, capsys):
    root, calls, _ = operator
    code, output = invoke(root, capsys, SECRET)
    assert code == 2 and output["error"]["code"] == "invalid_arguments"
    assert SECRET not in json.dumps(output) and not calls


def test_default_root_is_workspace():
    assert scene_cli.parser().parse_args(["show", "--job", "a-job"]).root == Path("/workspace")


def test_help_never_constructs_service(monkeypatch, capsys):
    def forbidden():
        pytest.fail("Help constructed the scene service")

    monkeypatch.setattr(scene_cli, "_service_class", forbidden)
    with pytest.raises(SystemExit) as exc:
        scene_cli.main(["--help"])
    assert exc.value.code == 0
    assert "audition" in capsys.readouterr().out


def test_invalid_source_receipt_is_rejected_before_service(operator, capsys):
    root, calls, _ = operator
    code, output = invoke(
        root, capsys, "import-sources", "--file", script_file(root), "--budget", "1",
        "--sources", json_file(root, "sources.json", {"sources": SECRET}),
    )
    assert code == 2 and output["error"]["code"] == "invalid_sources"
    assert SECRET not in json.dumps(output) and not calls


def test_existing_receipt_metadata_is_preserved_for_service_validation(operator, capsys):
    root, calls, _ = operator
    mapping = [{
        "turn_index": 1, "take_id": "c" * 32, "request_index": 2,
        "raw_sha256": "d" * 64, "fixture_known_metadata": "preserve for service validation",
    }]
    code, _ = invoke(
        root, capsys, "import-sources", "--file", script_file(root), "--budget", "1",
        "--sources", json_file(root, "receipt.json", {"sources": mapping, "other_receipt_field": "not source data"}),
    )
    assert code == 0 and calls[-1][2] == mapping


def test_character_setup_reuses_registered_store_without_constructing_scene_service(operator, capsys, monkeypatch):
    root, calls, _ = operator
    events = []

    class Store:
        def __init__(self, project_root):
            events.append(("construct", project_root))

        def create(self, data):
            events.append(("create", data))
            return data

        def add_prebuilt(self, character_id, voice, model, *, language="en"):
            events.append(("prebuilt", character_id, voice, model, language))
            return {"recipe_id": "e" * 32}

        def list(self):
            events.append(("list",))
            return [{"character_id": "moss"}]

        def get(self, character_id):
            events.append(("get", character_id))
            return {"character_id": character_id}

        def recipes(self, character_id):
            events.append(("recipes", character_id))
            return [{"recipe_id": "e" * 32}]

    monkeypatch.setattr(scene_cli, "_character_class", lambda: Store)
    identity = {"character_id": "moss", "name": "Moss"}
    code, _ = invoke(root, capsys, "character", "load", "--file", json_file(root, "character.json", identity))
    assert code == 0 and events[-1] == ("create", identity)
    code, _ = invoke(
        root, capsys, "character", "prebuilt", "--id", "moss", "--voice", "Charon",
        "--model", "gemini-3.8-flash-tts", "--language", "en",
    )
    assert code == 0 and events[-1] == ("prebuilt", "moss", "Charon", "gemini-3.8-flash-tts", "en")
    for command, event in [("list", ("list",)), ("show", ("get", "moss")), ("recipes", ("recipes", "moss"))]:
        args = [] if command == "list" else ["--id", "moss"]
        code, _ = invoke(root, capsys, "character", command, *args)
        assert code == 0 and events[-1] == event
    assert not calls


def test_character_mutation_refuses_backup_lease_before_registry_construction(operator, capsys, monkeypatch):
    import fcntl
    root, calls, _ = operator
    studio = root / ".runtime/studio"
    studio.mkdir(parents=True)
    constructed = []

    def store(project_root):
        constructed.append(project_root)
        pytest.fail("Backup-held lease allowed registry construction")

    monkeypatch.setattr(scene_cli, "_character_class", lambda: store)
    with (studio / "generation-worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code, output = invoke(
            root, capsys, "character", "load",
            "--file", json_file(root, "character.json", {"character_id": "moss", "name": "Moss"}),
        )
        assert code == 2 and output["error"]["code"] == "backend_busy"
    assert not calls and not constructed
    assert {path.name for path in studio.iterdir()} == {"generation-worker.lock"}


def test_recover_is_explicit_and_never_dispatches_selected_job(operator, capsys):
    root, calls, _ = operator
    code, output = invoke(root, capsys, "recover", "--job", "job-two")
    assert code == 0 and output["result"]["status"] == "uncertain"
    assert calls == [("construct", root), ("recover", "job-two")]


def test_reconcile_passes_verified_zero_charge_and_attempt_without_running(operator, capsys):
    root, calls, _ = operator
    attempt = "f" * 32
    evidence = "Verified invoice fixture evidence."
    code, output = invoke(
        root, capsys, "reconcile", "--job", "job-two", "--attempt", attempt,
        "--charged", "0", "--evidence", evidence,
    )
    assert code == 0 and output["result"]["reconciled"] is True
    assert calls == [("construct", root), ("reconcile", "job-two", attempt, "0", evidence)]


@pytest.fixture
def design_operator(operator, monkeypatch):
    root, scene_calls, _ = operator
    events = []
    request = object()

    class Request:
        @classmethod
        def from_preset(cls, character_id, description, model, *, preset="standard"):
            events.append(("preset", character_id, description, model, preset))
            return request

    class Service:
        def __init__(self, project_root):
            events.append(("construct", project_root))

        def configure_budget(self, character_id, episode_budget, series_budget):
            events.append(("configure", character_id, episode_budget, series_budget))

        def enqueue(self, submitted):
            assert submitted is request
            events.append(("enqueue",))
            return {"id": "design-job", "status": "queued"}

        def run_one_specific(self, job_id):
            events.append(("run", job_id))
            return {"id": job_id, "status": "ready"}

        def status(self, job_id):
            events.append(("show", job_id))
            return {"id": job_id, "status": "queued"}

        def recover(self, job_id):
            events.append(("recover", job_id))
            return {"id": job_id, "status": "uncertain"}

    monkeypatch.setattr(scene_cli, "_design_types", lambda: (Request, Service))
    return root, scene_calls, events


@pytest.mark.parametrize("series_budget", [None, "1.00"])
def test_character_design_queues_exact_preset_under_explicit_caps_without_dispatch(design_operator, capsys, series_budget):
    root, scene_calls, events = design_operator
    extra = [] if series_budget is None else ["--series-budget", series_budget]
    code, output = invoke(
        root, capsys, "character", "design", "--id", "moss",
        "--description", "An original fictional forest guide with a gentle rasp.",
        "--model", "gemini-3.8-flash-tts", "--budget", "0.25", *extra,
    )
    assert code == 0 and output["result"]["status"] == "queued"
    assert events == [
        ("preset", "moss", "An original fictional forest guide with a gentle rasp.", "gemini-3.8-flash-tts", "standard"),
        ("construct", root),
        ("configure", "moss", "0.25", series_budget or "0.25"),
        ("enqueue",),
    ]
    assert not scene_calls


@pytest.mark.parametrize("command,event,status", [
    ("design-run", "run", "ready"), ("design-show", "show", "queued"),
    ("design-recover", "recover", "uncertain"),
])
def test_character_design_operations_select_only_named_job(design_operator, capsys, command, event, status):
    root, scene_calls, events = design_operator
    code, output = invoke(root, capsys, "character", command, "--job", "design-two")
    assert code == 0 and output["result"]["status"] == status
    assert events == [("construct", root), (event, "design-two")]
    assert not scene_calls


def test_character_design_invalid_budget_rejects_before_job_setup(design_operator, capsys):
    root, scene_calls, events = design_operator
    code, output = invoke(
        root, capsys, "character", "design", "--id", "moss", "--description", SECRET,
        "--model", "gemini-3.8-flash-tts", "--budget", SECRET,
    )
    assert code == 2 and SECRET not in json.dumps(output)
    assert not events and not scene_calls


@pytest.mark.parametrize("command", ["plan", "create", "import-sources"])
def test_direct_model_and_fallback_flags_build_explicit_ordered_policy(operator, capsys, command):
    root, calls, _ = operator
    extra = [] if command == "plan" else ["--budget", "0.25"]
    if command == "import-sources":
        extra += ["--sources", json_file(root, "sources.json", {"sources": []})]
    code, _ = invoke(
        root, capsys, command, "--file", script_file(root),
        "--model", "gemini-3.8-flash-tts", "--fallback-model", "gemini-3.8-flash-lite-tts", *extra,
    )
    assert code == 0
    event = calls[-1]
    policy = event[2] if command == "plan" else event[-1]
    assert policy == {
        "models": ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"],
        "fallback_on_daily": True,
    }
    assert [event[0] for event in calls] == ["construct", "import" if command == "import-sources" else command]


def test_output_flag_and_matching_policy_merge_preserve_other_explicit_choices(operator, capsys):
    root, calls, _ = operator
    policy = {
        "models": ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"],
        "fallback_on_daily": True, "max_output_tokens": 2048,
        "min_interval_seconds": 12,
    }
    code, _ = invoke(
        root, capsys, "create", "--file", script_file(root), "--budget", "0.25",
        "--policy", json_file(root, "policy.json", policy),
        "--model", "gemini-3.8-flash-tts", "--fallback-model", "gemini-3.8-flash-lite-tts",
        "--max-output-tokens", "2048",
    )
    assert code == 0 and calls[-1][-1] == policy


@pytest.mark.parametrize("policy,flags,error", [
    ({"models": ["gemini-3.8-flash-tts"]}, ["--model", "gemini-3.8-flash-lite-tts"], "policy_conflict"),
    ({"fallback_on_daily": False}, ["--fallback-model", "gemini-3.8-flash-lite-tts"], "policy_conflict"),
    ({"max_output_tokens": 1024}, ["--max-output-tokens", "2048"], "policy_conflict"),
    ({}, ["--model", "gemini-3.8-flash-tts", "--fallback-model", "gemini-3.8-flash-tts"], "invalid_fallback"),
])
def test_conflicting_creative_flags_fail_clearly_before_service(operator, capsys, policy, flags, error):
    root, calls, _ = operator
    code, output = invoke(
        root, capsys, "plan", "--file", script_file(root),
        "--policy", json_file(root, "policy.json", policy), *flags,
    )
    assert code == 2 and output["error"]["code"] == error and not calls


@pytest.mark.parametrize("limit", ["0", "16385"])
def test_output_ceiling_must_fit_provider_bounds_before_service(operator, capsys, limit):
    root, calls, _ = operator
    code, output = invoke(root, capsys, "plan", "--file", script_file(root), "--max-output-tokens", limit)
    assert code == 2 and output["error"]["code"] == "invalid_output_limit" and not calls


def test_primary_model_flag_never_enables_implicit_fallback(operator, capsys):
    root, calls, _ = operator
    code, _ = invoke(
        root, capsys, "plan", "--file", script_file(root),
        "--model", "gemini-3.8-flash-lite-tts", "--max-output-tokens", "2048",
    )
    assert code == 0
    assert calls[-1][2] == {"models": ["gemini-3.8-flash-lite-tts"], "max_output_tokens": 2048}


def test_revision_can_change_output_policy_without_changing_text_or_voice(operator, capsys):
    root, calls, _ = operator
    policy = {"max_output_tokens": 2048}
    code, _ = invoke(
        root, capsys, "revise", "--job", "job-two", "--line", "2", "--budget", "0.25",
        "--policy", json_file(root, "larger-cap.json", policy),
    )
    assert code == 0
    assert calls == [("construct", root), ("revise", "job-two", 1, None, None, None, "0.25", policy)]


def test_perform_creates_then_runs_only_its_new_scene_with_explicit_policy(operator, capsys):
    root, calls, _ = operator
    code, output = invoke(
        root, capsys, "perform", "--file", script_file(root), "--budget", "0.25",
        "--model", "gemini-3.8-flash-tts", "--fallback-model", "gemini-3.8-flash-lite-tts",
        "--max-output-tokens", "2048",
    )
    assert code == 0 and output["result"]["id"] == "new-job"
    assert [event[0] for event in calls] == ["construct", "create", "run"]
    assert calls[1][2:] == (
        "0.25",
        {"models": ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"],
         "fallback_on_daily": True, "max_output_tokens": 2048},
    )
    assert calls[2] == ("run", "new-job", False)
