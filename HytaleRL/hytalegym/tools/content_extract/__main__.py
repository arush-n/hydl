"""CLI: ``python -m content_extract {extract,diff,verify}`` (run from ``tools/``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import FAMILY_MODULES, extract_tables, write_tables
from .archive import GameArchive
from .common import dump_table
from .diff import diff_directories, markdown
from .manifest import build_manifest, write_manifest


def _archive(args: argparse.Namespace) -> GameArchive:
    return GameArchive(
        args.assets,
        game_version=args.game_version,
        server_jar=args.server_jar,
        assets_sha256=args.assets_sha256,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="content_extract", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("extract", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--assets", type=Path, required=True, help="the version's Assets.zip")
        command.add_argument("--server-jar", type=Path, help="default: <assets dir>/Server/HytaleServer.jar")
        command.add_argument("--game-version", help="default: the jar's Implementation-Version")
        command.add_argument("--assets-sha256", help="skip hashing the archive (trusted value)")
        command.add_argument("--families", default=",".join(FAMILY_MODULES))
        command.add_argument("--out" if name == "extract" else "--data", type=Path, required=True)
    diff = sub.add_parser("diff")
    diff.add_argument("old", type=Path)
    diff.add_argument("new", type=Path)
    diff.add_argument("--json", type=Path)
    diff.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)

    if args.command == "diff":
        report = diff_directories(args.old, args.new)
        if args.json:
            args.json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        text = markdown(report)
        if args.markdown:
            args.markdown.write_text(text, encoding="utf-8")
        print(text if not args.markdown else f"wrote {args.markdown}")
        return 0

    families = tuple(name for name in args.families.split(",") if name)
    with _archive(args) as archive:
        tables = extract_tables(archive, families)
        if args.command == "extract":
            written = write_tables(tables, args.out)
            manifest = build_manifest(tables, archive)
            write_manifest(manifest, args.out)
            for path in written:
                print(f"wrote {path} ({path.stat().st_size} bytes)")
            print(f"content_sha256 {manifest['content_sha256']}")
            return 0
        stale = []
        for family, table in tables.items():
            path = args.data / f"{family}.json"
            if not path.is_file() or path.read_text(encoding="utf-8") != dump_table(table):
                stale.append(family)
        print(json.dumps({"stale": stale, "checked": sorted(tables)}, sort_keys=True))
        return 1 if stale else 0


if __name__ == "__main__":
    sys.exit(main())
