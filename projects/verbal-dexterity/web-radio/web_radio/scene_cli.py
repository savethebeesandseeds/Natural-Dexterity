"""One explicit operator workflow for creative scene production."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Sequence

from pydantic import ValidationError

from .budget import dollars_to_microdollars
from .gemini import Script
from .production import CommandError, MODELS, SafeParser, _authoring_lease, _root, read_json


def _service_class():
    from .scene_service import SceneService
    return SceneService


def _character_class():
    from .characters import CharacterStore
    return CharacterStore


def _design_types():
    from .voice_design import DesignRequest, VoiceDesignService
    return DesignRequest, VoiceDesignService


def parser():
    result = SafeParser(description=__doc__)
    result.add_argument("--root", type=Path, default=Path("/workspace"))
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("plan", "create", "perform", "import-sources"):
        command = commands.add_parser(name)
        command.add_argument("--file", type=Path, required=True, help="Authored Script JSON.")
        command.add_argument("--budget", required=name != "plan", help="Total USD ceiling for this scene.")
        command.add_argument("--policy", type=Path, help="Optional rendering-policy JSON.")
        command.add_argument("--model", choices=MODELS, help="Primary TTS model; defaults to Flash.")
        command.add_argument("--fallback-model", choices=MODELS, help="Explicit second model for daily-quota fallback.")
        if name in ("plan", "create", "perform"):
            command.add_argument("--max-output-tokens", type=int, help="Explicit per-line output ceiling.")
        if name == "import-sources":
            command.add_argument("--sources", type=Path, required=True, help="Authored source-mapping JSON.")
    for name in ("render", "resume", "show", "recover"):
        command = commands.add_parser(name)
        command.add_argument("--job", required=True)
    command = commands.add_parser("revise")
    command.add_argument("--job", required=True)
    command.add_argument("--line", type=int, required=True, help="One-based spoken line number.")
    command.add_argument("--text")
    command.add_argument("--style")
    command.add_argument("--recipe")
    command.add_argument("--budget", required=True)
    command.add_argument("--policy", type=Path, help="Explicit policy changes for the revised scene.")
    command = commands.add_parser("reconcile")
    command.add_argument("--job", required=True)
    command.add_argument("--attempt", required=True)
    command.add_argument("--charged", required=True)
    command.add_argument("--evidence", required=True)
    command = commands.add_parser("audition")
    command.add_argument("--character", required=True)
    command.add_argument("--recipe")
    command.add_argument("--text", required=True)
    command.add_argument("--style", default="")
    command.add_argument("--budget", required=True)
    character = commands.add_parser("character")
    actions = character.add_subparsers(dest="character_command", required=True)
    action = actions.add_parser("load")
    action.add_argument("--file", type=Path, required=True)
    action = actions.add_parser("prebuilt")
    action.add_argument("--id", required=True)
    action.add_argument("--voice", required=True)
    action.add_argument("--model", choices=MODELS, required=True)
    action.add_argument("--language", default="en")
    action = actions.add_parser("design")
    action.add_argument("--id", required=True)
    action.add_argument("--description", required=True)
    action.add_argument("--model", choices=MODELS, required=True)
    action.add_argument("--budget", required=True)
    action.add_argument("--series-budget")
    action.add_argument("--preset", choices=("standard",), default="standard")
    for name in ("design-run", "design-show", "design-recover"):
        action = actions.add_parser(name)
        action.add_argument("--job", required=True)
    actions.add_parser("list")
    for name in ("show", "recipes"):
        action = actions.add_parser(name)
        action.add_argument("--id", required=True)
    return result


def _render_policy(args):
    policy = read_json(args.policy) if getattr(args, "policy", None) else None
    model = getattr(args, "model", None)
    fallback = getattr(args, "fallback_model", None)
    output_tokens = getattr(args, "max_output_tokens", None)
    if model is None and fallback is None and output_tokens is None:
        return policy
    merged = dict(policy or {})
    configured = merged.get("models")
    if (model is not None or fallback is not None) and configured is not None:
        if (not isinstance(configured, list) or not 1 <= len(configured) <= 2
                or any(value not in MODELS for value in configured)
                or len(set(configured)) != len(configured)):
            raise CommandError("invalid_policy", "Policy models must be a unique ordered list of supported models.")
    primary = model or (configured[0] if configured else MODELS[0])
    if model is not None and configured and configured[0] != model:
        raise CommandError("policy_conflict", "The --model choice conflicts with models in the policy file.")
    if fallback is not None:
        if primary == fallback:
            raise CommandError("invalid_fallback", "Primary and fallback models must differ.")
        selected = [primary, fallback]
        if configured is not None and configured != selected:
            raise CommandError("policy_conflict", "The fallback model order conflicts with models in the policy file.")
        if "fallback_on_daily" in merged and merged["fallback_on_daily"] is not True:
            raise CommandError("policy_conflict", "The fallback choice conflicts with fallback_on_daily in the policy file.")
        merged["models"] = selected
        merged["fallback_on_daily"] = True
    elif model is not None:
        merged["models"] = list(configured) if configured is not None else [model]
    if output_tokens is not None:
        if not 128 <= output_tokens <= 16384:
            raise CommandError("invalid_output_limit", "Choose an output ceiling from 128 to 16384 tokens.")
        if "max_output_tokens" in merged and merged["max_output_tokens"] != output_tokens:
            raise CommandError("policy_conflict", "The output ceiling conflicts with max_output_tokens in the policy file.")
        merged["max_output_tokens"] = output_tokens
    return merged


def execute(args):
    root = _root(args.root)
    if args.command == "character":
        data = read_json(args.file) if args.character_command == "load" else None
        if args.character_command == "design":
            dollars_to_microdollars(args.budget)
            dollars_to_microdollars(args.series_budget or args.budget)
        with _authoring_lease(root):
            if args.character_command.startswith("design"):
                DesignRequest, VoiceDesignService = _design_types()
                if args.character_command == "design":
                    request = DesignRequest.from_preset(
                        args.id, args.description, args.model, preset=args.preset,
                    )
                    service = VoiceDesignService(root)
                    service.configure_budget(args.id, args.budget, args.series_budget or args.budget)
                    return service.enqueue(request)
                service = VoiceDesignService(root)
                if args.character_command == "design-run":
                    return service.run_one_specific(args.job)
                if args.character_command == "design-show":
                    return service.status(args.job)
                return service.recover(args.job)
            store = _character_class()(root)
            if args.character_command == "load":
                return store.create(data)
            if args.character_command == "prebuilt":
                return store.add_prebuilt(args.id, args.voice, args.model, language=args.language)
            if args.character_command == "list":
                return {"characters": store.list()}
            if args.character_command == "show":
                return store.get(args.id)
            return {"recipes": store.recipes(args.id)}
    script = Script.model_validate(read_json(args.file)) if args.command in ("plan", "create", "perform", "import-sources") else None
    policy = _render_policy(args)
    mapping = None
    if args.command == "import-sources":
        receipt = read_json(args.sources)
        mapping = receipt.get("sources")
        if not isinstance(mapping, list):
            raise CommandError("invalid_sources", "The source receipt must contain a sources list.")
    if getattr(args, "budget", None) is not None:
        dollars_to_microdollars(args.budget)
    if args.command == "revise":
        if args.line < 1:
            raise CommandError("invalid_line", "Choose a spoken line number starting at one.")
        if all(value is None for value in (args.text, args.style, args.recipe, policy)):
            raise CommandError("missing_revision", "Provide text, style, or a voice recipe to change.")
    service = _service_class()(root)
    if args.command == "plan":
        return service.plan(script, policy=policy, budget_usd=args.budget)
    if args.command == "create":
        return service.create(script, budget_usd=args.budget, policy=policy)
    if args.command == "perform":
        created = service.create(script, budget_usd=args.budget, policy=policy)
        return service.run(created["id"])
    if args.command == "import-sources":
        return service.import_sources(script, mapping, budget_usd=args.budget, policy=policy)
    if args.command in ("render", "resume"):
        return service.run(args.job, resume=args.command == "resume")
    if args.command == "show":
        return service.status(args.job)
    if args.command == "recover":
        return service.recover(args.job)
    if args.command == "reconcile":
        return service.reconcile(
            args.job, attempt_id=args.attempt, charged_usd=args.charged,
            evidence=args.evidence,
        )
    if args.command == "revise":
        return service.revise(
            args.job, line_index=args.line - 1, text=args.text, style=args.style,
            recipe_id=args.recipe, budget_usd=args.budget, policy=policy,
        )
    return service.audition(
        args.character, args.text, style=args.style, recipe_id=args.recipe,
        budget_usd=args.budget,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = execute(parser().parse_args(argv))
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, default=str))
        return 0
    except CommandError as exc:
        error = {"code": exc.code, "message": str(exc)}
    except ValidationError as exc:
        error = {
            "code": "validation_failed", "message": "Input does not match the command schema.",
            "issues": [{"type": item["type"]} for item in exc.errors(include_input=False, include_context=False)],
        }
    except (ValueError, OSError):
        error = {
            "code": "operation_rejected",
            "message": "The operation was rejected. Check inputs and use show --job for saved job diagnostics.",
        }
    except Exception:
        error = {
            "code": "operation_failed",
            "message": "The operation could not complete. Use show --job to inspect retained job diagnostics.",
        }
    print(json.dumps({"ok": False, "error": error}), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
