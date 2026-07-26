"""Minimal offline CLI for atlas validation, snapshots and the reproducible demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from .api import create, load


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="transition-evidence")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate an atlas")
    validate.add_argument("atlas", type=Path)
    initialise = commands.add_parser("init-snapshot", help="create an empty shelf snapshot")
    initialise.add_argument("atlas", type=Path)
    initialise.add_argument("snapshot", type=Path)
    inspect = commands.add_parser("inspect", help="print canonical snapshot JSON")
    inspect.add_argument("atlas", type=Path)
    inspect.add_argument("snapshot", type=Path)
    args = parser.parse_args(argv)

    if args.command == "validate":
        engine = create(args.atlas.read_bytes())
        print(engine.snapshot().to_json_bytes().decode("utf-8"))
    elif args.command == "init-snapshot":
        create(args.atlas.read_bytes()).save(args.snapshot)
    else:
        print(load(args.snapshot, args.atlas.read_bytes()).snapshot().to_json_bytes().decode("utf-8"))
    return 0
