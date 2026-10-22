from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import wave

import httpx
import pytest

from web_radio import production
from web_radio.budget import reservation_cost
from web_radio.studio import Studio

SOURCE = Path(__file__).resolve().parents[1]
MODEL = "gemini-3.8-flash-lite-tts"


@pytest.fixture(autouse=True)
def no_live_network_or_environment_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail("CLI fixture attempted a network request")
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    shutil.copytree(SOURCE / "content", root / "content")
    return root


def input_script(project, **updates):
    raw = json.loads((project / "content/series/first-light/programme.json").read_text())
    # Use a short native fixture; the authored programme also exercises chunking below.
    raw["turns"] = raw["turns"][:2]
    raw.update(updates)
    path = project / "input.json"
    path.write_text(json.dumps(raw))
    return path


def invoke(project, capsys, *args):
    code = production.main(["--root", str(project), *map(str, args)])
    captured = capsys.readouterr()
    return code, json.loads(captured.out if code == 0 else captured.err)


def test_plan_previews_cost_without_database_reservation_or_dispatch(project, capsys):
    code, output = invoke(project, capsys, "plan", "--script", input_script(project), "--model", MODEL)
    assert code == 0
    plan = output["result"]
    assert plan["strategy"] == "native" and plan["request_count"] == 1
    assert plan["reservation_ceiling_microdollars"] == reservation_cost(MODEL)
    assert not plan["dispatch"] and not plan["reserved"]
    assert not list((project / ".runtime").rglob("*.sqlite3"))
    assert not (project / ".runtime/library").exists()


def test_plan_larger_cast_counts_every_planned_request(project, capsys):
    script = input_script(project)
    raw = json.loads(script.read_text())
    third = dict(raw["speakers"][0], name="Finch", voice="Puck")
    raw["speakers"].append(third)
    raw["turns"] = [{"speaker": name, "text": "A short fixture line.", "style": ""} for name in ("Moss", "Wren", "Finch")]
    script.write_text(json.dumps(raw))
    code, output = invoke(project, capsys, "plan", "--script", script, "--model", MODEL)
    assert code == 0
    plan = output["result"]
    assert plan["strategy"] == "turns" and plan["request_count"] == 3
    assert plan["reservation_ceiling_microdollars"] == 3 * reservation_cost(MODEL)


def test_render_queues_with_explicit_caps_and_never_runs_or_publishes(project, capsys):
    Studio(project).save_key("cli-fixture-key-never-real")
    code, output = invoke(project, capsys, "render", "--script", input_script(project), "--model", MODEL,
                          "--episode-budget", "1.00", "--series-budget", "2.00")
    assert code == 0
    take = output["result"]
    assert take["status"] == "queued" and take["decision"] == "pending"
    assert take["reservation_microdollars"] == reservation_cost(MODEL)
    assert take["script_revision_id"]
    assert "cli-fixture-key" not in json.dumps(output)
    assert not (project / ".runtime/library").exists()
    assert not (project / ".runtime/studio/takes" / take["id"] / "raw.wav").exists()


def test_run_selects_only_the_named_take_and_does_not_start_worker(project, capsys, monkeypatch):
    service = Studio(project)
    service.save_key("cli-fixture-key-never-real")
    script = input_script(project)
    ids = []
    for _ in range(2):
        code, result = invoke(project, capsys, "render", "--script", script, "--model", MODEL,
                              "--episode-budget", "1", "--series-budget", "2")
        assert code == 0
        ids.append(result["result"]["id"])
    calls = []
    def run(self, transport=None, *, take_id=None):
        calls.append(take_id)
        return False
    def start(*args):
        pytest.fail("CLI started an unbounded worker")
    monkeypatch.setattr(Studio, "run_one", run)
    monkeypatch.setattr(Studio, "start", start)
    code, result = invoke(project, capsys, "run", "--take", ids[1])
    assert code == 0 and calls == [ids[1]]
    assert result["result"]["take"]["id"] == ids[1]
    assert all(take["status"] == "queued" for take in Studio(project).takes())


def wav_bytes():
    stream = io.BytesIO()
    with wave.open(stream, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(24000)
        output.writeframes(b"\0\0" * 24000)
    return stream.getvalue()


def test_import_preserves_original_and_exact_script_without_spend_or_publish(project, capsys):
    script = input_script(project, title="CLI imported fixture")
    recording = project / "fixture.wav"
    original = wav_bytes()
    recording.write_bytes(original)
    code, output = invoke(project, capsys, "import", "--script", script, "--file", recording)
    assert code == 0
    take = output["result"]
    assert take["status"] == "ready" and take["source"] == "imported"
    assert take["script"]["title"] == "CLI imported fixture"
    assert take["reservation_microdollars"] == 0
    assert (project / ".runtime/studio/takes" / take["id"] / "original.wav").read_bytes() == original
    assert not (project / ".runtime/library").exists()
    code, _ = invoke(project, capsys, "publish", "--take", take["id"])
    assert code == 2  # explicit acceptance is still required
    code, _ = invoke(project, capsys, "decision", "--take", take["id"], "--decision", "accepted")
    assert code == 0
    code, released = invoke(project, capsys, "publish", "--take", take["id"])
    assert code == 0 and released["result"]["release_id"]


@pytest.mark.parametrize("payload", [
    '{"x":1,"x":2}',
    '{"unexpected":"cli-input-secret-never-echo"}',
    '{"programme_id":',
    '[]',
])
def test_invalid_inputs_are_sanitized_and_never_dispatch(project, capsys, payload):
    path = project / "invalid.json"
    path.write_text(payload)
    code, output = invoke(project, capsys, "plan", "--script", path, "--model", MODEL)
    assert code == 2 and output["ok"] is False
    assert "cli-input-secret-never-echo" not in json.dumps(output)
    assert "Traceback" not in json.dumps(output)
    assert not (project / ".runtime/library").exists()


def test_unknown_arguments_do_not_repeat_arbitrary_values(project, capsys):
    code, output = invoke(project, capsys, "plan", "--key", "cli-input-secret-never-echo")
    assert code == 2
    assert output["error"]["code"] == "invalid_arguments"
    assert "cli-input-secret-never-echo" not in json.dumps(output)


def test_input_size_is_bounded_before_schema_processing(project, capsys):
    path = project / "oversize.json"
    path.write_bytes(b" " * (production.JSON_LIMIT + 1))
    code, output = invoke(project, capsys, "plan", "--script", path, "--model", MODEL)
    assert code == 2 and output["error"]["code"] == "input_too_large"


def test_root_symlinks_are_rejected_before_backend_creation(project, tmp_path, capsys):
    link = tmp_path / "alias"
    link.symlink_to(project, target_is_directory=True)
    code, output = invoke(link, capsys, "status")
    assert code == 2 and output["error"]["code"] == "invalid_root"
    assert not (project / ".runtime/studio/studio.sqlite3").exists()


def test_status_returns_only_key_presence_not_credential(project, capsys):
    Studio(project).save_key("cli-fixture-key-never-real")
    code, output = invoke(project, capsys, "status")
    assert code == 0
    assert output["result"]["key_configured"] is True
    assert output["result"]["worker"] == "stopped"
    assert "cli-fixture-key-never-real" not in json.dumps(output)


def test_character_create_revision_and_prebuilt_recipe_are_backend_commands(project, capsys):
    path = project / "character.json"
    payload = {"character_id": "cli-finch", "name": "Finch", "personality": "Curious", "background": "", "pronunciation_notes": ""}
    path.write_text(json.dumps(payload))
    code, output = invoke(project, capsys, "character", "create", "--file", path)
    assert code == 0 and output["result"]["character_id"] == "cli-finch"
    code, recipe = invoke(project, capsys, "character", "prebuilt", "--id", "cli-finch", "--voice", "Puck", "--model", MODEL)
    assert code == 0
    assert recipe["result"]["origin"] == "prebuilt"
    code, show = invoke(project, capsys, "character", "show", "--id", "cli-finch")
    assert code == 0 and show["result"]["name"] == "Finch"


def test_voice_design_enqueue_is_quote_required_and_does_not_execute(project, capsys, monkeypatch):
    from web_radio.voice_design import VoiceDesignService
    from web_radio.characters import CharacterStore
    CharacterStore(project).create({"character_id": "cli-finch", "name": "Finch"})
    Studio(project).save_key("cli-fixture-key-never-real")
    request = {
        "character_id": "cli-finch", "description": "An original small creature with a clear curious voice.",
        "model": MODEL, "language_code": "en", "quoted_cost_usd": "0.10",
        "quote_evidence": "Fixture verified ceiling, not a provider billing guarantee.",
        "quote_valid_until": "2099-01-01T00:00:00Z",
    }
    path = project / "design.json"
    path.write_text(json.dumps(request))
    def forbidden(*args, **kwargs):
        pytest.fail("Enqueue dispatched voice design")
    monkeypatch.setattr(VoiceDesignService, "run_one_specific", forbidden)
    code, output = invoke(project, capsys, "voice-design", "enqueue", "--file", path,
                          "--episode-budget", "1", "--series-budget", "1")
    assert code == 0
    assert output["result"]["status"] == "queued"
    assert "cli-fixture-key" not in json.dumps(output)


def test_json_input_requires_utf8(project, capsys):
    path = project / "encoded.json"
    path.write_bytes('{"character_id":"finch"}'.encode("utf-16"))
    code, output = invoke(project, capsys, "character", "create", "--file", path)
    assert code == 2 and output["error"]["code"] == "invalid_json"


def test_long_authored_script_plan_prices_every_chunk(project, capsys):
    script = project / "content/series/first-light/programme.json"
    code, output = invoke(project, capsys, "plan", "--script", script, "--model", MODEL)
    assert code == 0
    plan = output["result"]
    assert plan["strategy"] == "chunks" and plan["request_count"] > 1
    assert plan["reservation_ceiling_microdollars"] == plan["request_count"] * reservation_cost(MODEL)
    assert plan["dispatch"] is False and plan["reserved"] is False


def test_voice_design_recovery_is_explicit_and_never_dispatches(project, capsys, monkeypatch):
    from web_radio.voice_design import VoiceDesignService
    calls = []
    def recover(self, identifier):
        calls.append(identifier)
        return {"id": identifier, "status": "uncertain"}
    def forbidden(*args, **kwargs):
        pytest.fail("Recovery dispatched voice generation")
    monkeypatch.setattr(VoiceDesignService, "recover", recover, raising=False)
    monkeypatch.setattr(VoiceDesignService, "run_one_specific", forbidden)
    identifier = "a" * 32
    code, output = invoke(project, capsys, "voice-design", "recover", "--job", identifier)
    assert code == 0 and calls == [identifier]
    assert output["result"]["status"] == "uncertain"


@pytest.mark.parametrize("operation", ["render", "character"])
def test_backup_lease_blocks_authoring_before_any_backend_state_is_created(project, capsys, operation):
    import fcntl
    studio = project / ".runtime/studio"
    studio.mkdir(parents=True)
    if operation == "render":
        args = ["render", "--script", str(input_script(project)), "--model", MODEL,
                "--episode-budget", "1", "--series-budget", "2"]
    else:
        path = project / "character-input.json"
        path.write_text('{"character_id":"finch","name":"Finch"}')
        args = ["character", "create", "--file", str(path)]
    with (studio / "generation-worker.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code, output = invoke(project, capsys, *args)
        assert code == 2 and output["error"]["code"] == "backend_busy"
        assert {path.name for path in studio.iterdir()} == {"generation-worker.lock"}
        fcntl.flock(lease, fcntl.LOCK_UN)


def test_pure_legacy_plan_stays_available_during_backup_without_database_writes(project, capsys):
    import fcntl
    studio = project / ".runtime/studio"
    studio.mkdir(parents=True)
    script = input_script(project)
    with (studio / "generation-worker.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code, output = invoke(project, capsys, "plan", "--script", script, "--model", MODEL)
        assert code == 0 and output["result"]["dispatch"] is False
        assert {path.name for path in studio.iterdir()} == {"generation-worker.lock"}
        fcntl.flock(lease, fcntl.LOCK_UN)
