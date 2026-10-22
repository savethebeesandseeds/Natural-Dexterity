"""Backend operator commands. This module never starts a web API or queue worker."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import stat
import sys
from typing import Sequence

from pydantic import ValidationError

from . import gemini
from .budget import dollars_to_microdollars, reservation_cost
from .studio import BudgetSettings, GenerationRequest, MAX_IMPORT_BYTES, Studio

JSON_LIMIT = 1024 * 1024
MODELS = ("gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts")


class CommandError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse's default error can repeat arbitrary submitted values.
        raise CommandError("invalid_arguments", "Invalid arguments. Use --help for command usage.")


def _unique(items):
    result = {}
    for key, value in items:
        if key in result:
            raise CommandError("invalid_json", "The JSON input repeats a field.")
        result[key] = value
    return result


def read_json(path: Path):
    try:
        with path.open("rb") as stream:
            raw = stream.read(JSON_LIMIT + 1)
        if len(raw) > JSON_LIMIT:
            raise CommandError("input_too_large", "JSON input exceeds one MiB.")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise CommandError("invalid_json", "Cannot read a valid UTF-8 JSON input file.") from None
    if not isinstance(value, dict):
        raise CommandError("invalid_json", "JSON input must be an object.")
    return value


def _root(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    if any(parent.is_symlink() for parent in (absolute, *absolute.parents)):
        raise CommandError("invalid_root", "The project root must not pass through symlinks.")
    if not absolute.is_dir():
        raise CommandError("invalid_root", "The project root must be an existing directory.")
    return absolute


@contextmanager
def _authoring_lease(root: Path):
    # Offline backup holds this same file exclusively for the entire snapshot.
    studio = root / ".runtime" / "studio"
    if any(parent.is_symlink() for parent in (studio, studio.parent)):
        raise CommandError("invalid_root", "Backend storage must not pass through symlinks.")
    studio.mkdir(parents=True, exist_ok=True)
    path = studio / "generation-worker.lock"
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    except OSError:
        raise CommandError("invalid_lease", "Cannot safely acquire the backend authoring lease.") from None
    with os.fdopen(descriptor, "r+b") as lease:
        if not stat.S_ISREG(os.fstat(lease.fileno()).st_mode):
            raise CommandError("invalid_lease", "Backend authoring lease must be an ordinary file.")
        try:
            fcntl.flock(lease, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CommandError("backend_busy", "Backup or exclusive generation is active; retry after it completes.") from None
        try:
            yield
        finally:
            fcntl.flock(lease, fcntl.LOCK_UN)


def _characters(root):
    from .characters import CharacterStore
    return CharacterStore(root)


def _script(path):
    return gemini.Script.model_validate(read_json(path))


def _design(root):
    from .voice_design import VoiceDesignService
    return VoiceDesignService(root)


def parser():
    result = SafeParser(description=__doc__)
    result.add_argument("--root", type=Path, default=Path("/workspace"))
    commands = result.add_subparsers(dest="command", required=True, parser_class=SafeParser)
    commands.add_parser("status", help="Show backend readiness, counts, and budgets; no secret values.")
    plan = commands.add_parser("plan", help="Inspect a cast/render plan without reserving or dispatching.")
    plan.add_argument("--script", type=Path, required=True)
    plan.add_argument("--model", choices=MODELS, required=True)
    render = commands.add_parser("render", help="Save the script and queue one take; does not execute.")
    render.add_argument("--script", type=Path, required=True)
    render.add_argument("--model", choices=MODELS, required=True)
    render.add_argument("--episode-budget", required=True)
    render.add_argument("--series-budget", required=True)
    run = commands.add_parser("run", help="Execute only the named queued take; never drain the queue.")
    run.add_argument("--take", required=True)
    takes = commands.add_parser("takes", help="List takes, or inspect one.")
    takes.add_argument("--take")
    decision = commands.add_parser("decision", help="Record an explicit review decision.")
    decision.add_argument("--take", required=True)
    decision.add_argument("--decision", choices=("accepted", "rejected"), required=True)
    publish = commands.add_parser("publish", help="Publish one already accepted, completed take.")
    publish.add_argument("--take", required=True)
    recording = commands.add_parser("import", help="Import a bounded WAV against an exact script.")
    recording.add_argument("--file", type=Path, required=True)
    recording.add_argument("--script", type=Path, required=True)

    character = commands.add_parser("character", help="Manage reusable identities and voice recipes.")
    char_commands = character.add_subparsers(dest="character_command", required=True, parser_class=SafeParser)
    char_commands.add_parser("list")
    create = char_commands.add_parser("create")
    create.add_argument("--file", type=Path, required=True)
    update = char_commands.add_parser("update")
    update.add_argument("--id", required=True)
    update.add_argument("--file", type=Path, required=True)
    update.add_argument("--expected-revision", type=int, required=True)
    show = char_commands.add_parser("show")
    show.add_argument("--id", required=True)
    show.add_argument("--revision", type=int)
    prebuilt = char_commands.add_parser("prebuilt")
    prebuilt.add_argument("--id", required=True)
    prebuilt.add_argument("--voice", required=True)
    prebuilt.add_argument("--model", choices=MODELS, required=True)
    prebuilt.add_argument("--language", default="en")
    recipes = char_commands.add_parser("recipes")
    recipes.add_argument("--id", required=True)
    recipe = char_commands.add_parser("recipe")
    recipe.add_argument("--id", required=True)
    recipe.add_argument("--recipe", required=True)

    voice = commands.add_parser("voice-design", help="Description-only voice design; no replication.")
    voice_commands = voice.add_subparsers(dest="voice_command", required=True, parser_class=SafeParser)
    enqueue = voice_commands.add_parser("enqueue", help="Reserve a quoted allocation and queue; no dispatch.")
    enqueue.add_argument("--file", type=Path, required=True)
    enqueue.add_argument("--episode-budget", required=True)
    enqueue.add_argument("--series-budget", required=True)
    for name in ("run", "show", "recover"):
        operation = voice_commands.add_parser(name)
        operation.add_argument("--job", required=True)
    reconcile = voice_commands.add_parser("reconcile")
    reconcile.add_argument("--job", required=True)
    reconcile.add_argument("--charged", required=True)
    reconcile.add_argument("--evidence", required=True)
    return result


def _import_recording(service: Studio, args):
    path = Path(os.path.abspath(args.file))
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise CommandError("invalid_recording", "The recording must be an ordinary local file.")
    with path.open("rb") as original:
        metadata = os.fstat(original.fileno())
        if not stat.S_ISREG(metadata.st_mode) or not 44 <= metadata.st_size <= MAX_IMPORT_BYTES:
            raise CommandError("invalid_recording", "Choose an ordinary WAV file of at most 64 MiB.")
        script = _script(args.script)
        revision = service.save_script(script)
        take = service.begin_import(script.kind, path.name, script_revision_id=revision["id"])
        destination = service.state / "takes" / take["id"] / "original.wav"
        copied = 0
        try:
            with destination.open("xb") as output:
                while block := original.read(64 * 1024):
                    copied += len(block)
                    if copied > MAX_IMPORT_BYTES:
                        raise CommandError("recording_too_large", "The recording exceeds 64 MiB.")
                    output.write(block)
            return service.finish_import(take["id"])
        except BaseException:
            service.fail_import(take["id"], "Import did not complete; received original bytes are retained.")
            raise


def execute(args):
    root = _root(args.root)
    if args.command == "plan":
        return _execute(args, root)
    if args.command == "run":
        # Construction can migrate/create state, so protect it, then release
        # the shared lease before run_one takes the exclusive dispatch lease.
        with _authoring_lease(root):
            service = Studio(root)
        processed = service.run_one(take_id=args.take)
        return {"processed": processed, "take": service.take(args.take)}
    with _authoring_lease(root):
        return _execute(args, root)


def _execute(args, root):
    if args.command == "plan":
        script = _script(args.script)
        if any(speaker.character_id or speaker.voice_recipe_id or speaker.character_revision for speaker in script.speakers):
            # Registry initialization can write even for a lookup; coordinate it.
            with _authoring_lease(root):
                store = _characters(root)
                bound, cast = store.bind_cast(script)
                approved = store.approved_designed_voices()
        else:
            # Legacy/prebuilt scripts need no persistent registry or database.
            bound, cast, approved = script, {"schema_version": 1, "speakers": []}, ()
        plan = gemini.build_plan(bound, args.model, approved_designed_voices=approved)
        ceiling = reservation_cost(args.model) * plan["request_count"]
        return {
            "strategy": plan["strategy"], "request_count": plan["request_count"],
            "input_token_upper_bound": plan["input_token_upper_bound"],
            "reservation_ceiling_microdollars": ceiling,
            "reservation_ceiling_usd": f"{ceiling / 1_000_000:.6f}",
            "cast": cast, "dispatch": False, "reserved": False,
        }
    if args.command == "character":
        store = _characters(root)
        action = args.character_command
        if action == "list":
            return {"characters": store.list()}
        if action == "create":
            return store.create(read_json(args.file))
        if action == "update":
            return store.update(args.id, read_json(args.file), expected_revision=args.expected_revision)
        if action == "show":
            return store.get(args.id, revision=args.revision)
        if action == "prebuilt":
            return store.add_prebuilt(args.id, args.voice, args.model, language=args.language)
        if action == "recipes":
            return {"recipes": store.recipes(args.id)}
        return store.get_recipe(args.id, args.recipe)
    if args.command == "voice-design":
        from .voice_design import DesignRequest
        service = _design(root)
        if args.voice_command == "enqueue":
            request = DesignRequest.model_validate(read_json(args.file))
            dollars_to_microdollars(args.episode_budget)
            dollars_to_microdollars(args.series_budget)
            service.configure_budget(request.character_id, args.episode_budget, args.series_budget)
            return service.enqueue(request)
        if args.voice_command == "run":
            return service.run_one_specific(args.job)
        if args.voice_command == "show":
            return service.status(args.job)
        if args.voice_command == "recover":
            return service.recover(args.job)
        return service.reconcile(args.job, args.charged, args.evidence)

    service = Studio(root)
    if args.command == "status":
        status = service.status()
        status["take_count"] = len(service.takes())
        return status
    if args.command == "render":
        script = _script(args.script)
        dollars_to_microdollars(args.episode_budget)
        dollars_to_microdollars(args.series_budget)
        revision = service.save_script(script)
        service.configure(BudgetSettings(
            kind=script.kind, episode_limit_usd=args.episode_budget,
            series_limit_usd=args.series_budget,
        ))
        return service.enqueue(GenerationRequest(
            kind=script.kind, model=args.model, script_revision_id=revision["id"],
        ))
    if args.command == "takes":
        return service.take(args.take) if args.take else {"takes": service.takes()}
    if args.command == "decision":
        return service.decide(args.take, args.decision)
    if args.command == "publish":
        return service.publish(args.take)
    return _import_recording(service, args)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = execute(parser().parse_args(argv))
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, default=str))
        return 0
    except CommandError as exc:
        error = {"code": exc.code, "message": str(exc)}
    except ValidationError as exc:
        # Pydantic input/context/locations can contain submitted secret values.
        error = {"code": "validation_failed", "message": "Input does not match the command schema.",
                 "issues": [{"type": item["type"]} for item in exc.errors(include_input=False, include_context=False)]}
    except (ValueError, OSError) as exc:
        error = {"code": "operation_rejected", "message": "The operation was rejected; check input and current state."}
    except Exception:
        # No traceback or provider/debug body is emitted to a terminal/log.
        error = {"code": "operation_failed", "message": "The backend operation could not complete; retained records are unchanged where possible."}
    print(json.dumps({"ok": False, "error": error}), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
