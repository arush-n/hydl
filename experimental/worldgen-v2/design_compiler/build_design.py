"""Compile or validate a WorldGen Studio design asset pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


if __package__ in (None, ""):
    module_root = Path(__file__).resolve().parents[1]
    workspace = Path(__file__).resolve().parents[3]
    for entry in (workspace, module_root):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    from design_compiler.compiler import compile_design, default_assets_path, validate_pack
else:
    from .compiler import compile_design, default_assets_path, validate_pack


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, help="saved Studio record or config JSON")
    parser.add_argument("--assets", type=Path, default=default_assets_path())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--entry-pins-only",
        action="store_true",
        help="skip the whole-archive hash (faster but weaker; receipt records this)",
    )
    args = parser.parse_args()
    if args.validate_only:
        result = validate_pack(args.output)
    else:
        if args.design is None:
            parser.error("--design is required unless --validate-only is used")
        try:
            value = json.loads(args.design.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            parser.error(f"could not read design JSON: {exc}")
        if not isinstance(value, dict):
            parser.error("design JSON must be an object")
        config = value.get("config", value)
        if not isinstance(config, dict):
            parser.error("design config must be an object")
        result = compile_design(
            config,
            assets=args.assets,
            output=args.output,
            verify_archive=not args.entry_pins_only,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
