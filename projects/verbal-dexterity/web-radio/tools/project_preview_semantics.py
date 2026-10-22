#!/usr/bin/env python3
"""Project an existing verified text-embedding bundle into one audition preview.

Run inside the managed container. No provider calls or credentials are used.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from web_radio.semantic_projection import clone_preview_variant, project_preview


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--preview-root", type=Path, required=True)
    parser.add_argument("--dimensions", type=int, choices=(3, 4), default=3)
    parser.add_argument("--source-preview-root", type=Path,
                        help="Create a new sibling preview by copying verified audio and adapting its identity")
    args = parser.parse_args()
    if args.source_preview_root:
        clone_preview_variant(args.source_preview_root, args.preview_root, args.preview_root.name)
    print(json.dumps(project_preview(args.preparation_root, args.bundle, args.preview_root,
                                     dimensions=args.dimensions), indent=2))


if __name__ == "__main__":
    main()
