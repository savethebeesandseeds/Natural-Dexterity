#!/usr/bin/env python3
"""Export, assemble or verify a static Radio Waajacu site without generation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web_radio.static_export import (
    CUSTOM_DOMAIN, StaticExportError, assemble_publication, export_static, verify_export,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path, help="New output directory; existing directories are preserved.")
    action.add_argument("--verify", type=Path, help="Verify an existing export without changing it.")
    parser.add_argument("--publication", type=Path, help="Assemble current frontend with hash-pinned public release media.")
    parser.add_argument("--platform", choices=("cloudflare", "github-pages"), default="cloudflare")
    parser.add_argument("--custom-domain", default=CUSTOM_DOMAIN)
    args = parser.parse_args(argv)
    if args.publication and (not args.output or args.platform != "github-pages"):
        parser.error("--publication requires --output and --platform github-pages")
    try:
        if args.verify:
            result = verify_export(args.verify)
        elif args.publication:
            result = assemble_publication(args.publication, args.output, platform=args.platform)
        else:
            result = export_static(args.output, platform=args.platform, custom_domain=args.custom_domain)
    except (StaticExportError, ValueError, OSError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
