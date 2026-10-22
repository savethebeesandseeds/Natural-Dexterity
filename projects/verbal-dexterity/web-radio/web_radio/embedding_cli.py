"""Explicit backend operator commands for embedding preparation jobs."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Sequence

from pydantic import ValidationError

from .budget import dollars_to_microdollars
from .production import CommandError, SafeParser, _root, read_json


def _components():
    # Keep --help and input parsing independent from service initialization.
    from .embedding_service import EmbeddingService, PreparationSpec
    return PreparationSpec, EmbeddingService


def parser():
    result = SafeParser(description=__doc__)
    result.add_argument("--root", type=Path, default=Path("/workspace"))
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("plan", "enqueue"):
        command = commands.add_parser(name)
        command.add_argument("--file", type=Path, required=True)
        if name == "enqueue":
            command.add_argument("--episode-budget", required=True)
            command.add_argument("--series-budget", required=True)
    for name in ("run", "show", "recover", "install"):
        command = commands.add_parser(name)
        command.add_argument("--job", required=True)
    command = commands.add_parser("reconcile")
    command.add_argument("--job", required=True)
    command.add_argument("--request", required=True)
    command.add_argument("--charged", required=True)
    command.add_argument("--evidence", required=True)
    return result


def execute(args):
    root = _root(args.root)
    spec_input = read_json(args.file) if args.command in ("plan", "enqueue") else None
    if args.command == "enqueue":
        dollars_to_microdollars(args.episode_budget)
        dollars_to_microdollars(args.series_budget)
    PreparationSpec, EmbeddingService = _components()
    spec = PreparationSpec.model_validate(spec_input) if spec_input is not None else None
    # The service coordinates its construction and every public operation with
    # the offline-backup lease; the CLI must not acquire a competing worker lock.
    service = EmbeddingService(root)
    if args.command == "plan":
        return service.plan(spec)
    if args.command == "enqueue":
        return service.enqueue(
            spec, episode_limit_usd=args.episode_budget,
            series_limit_usd=args.series_budget,
        )
    if args.command == "run":
        return service.run_one(args.job)
    if args.command == "show":
        return service.status(args.job)
    if args.command == "recover":
        return service.recover(args.job)
    if args.command == "install":
        return service.install(args.job)
    return service.reconcile(
        args.job, request_sha256=args.request, charged_usd=args.charged,
        evidence=args.evidence,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = execute(parser().parse_args(argv))
        print(json.dumps({"ok": True, "result": result}, ensure_ascii=False, default=str))
        return 0
    except CommandError as exc:
        error = {"code": exc.code, "message": str(exc)}
    except ValidationError as exc:
        # Inputs, validation context, and even field names may contain secrets.
        error = {
            "code": "validation_failed",
            "message": "Input does not match the command schema.",
            "issues": [{"type": item["type"]} for item in exc.errors(include_input=False, include_context=False)],
        }
    except (ValueError, OSError):
        error = {"code": "operation_rejected", "message": "The operation was rejected; check input and current state."}
    except Exception:
        error = {"code": "operation_failed", "message": "The backend operation could not complete; inspect retained job records."}
    print(json.dumps({"ok": False, "error": error}), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
