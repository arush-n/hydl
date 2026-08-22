"""Build or validate one reproducible custom WorldGen V2 asset pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from custom_environments.asset_pack import (
        build_asset_pack,
        default_hytale_assets_path,
        load_environment_spec,
        validate_asset_pack,
    )
else:
    from .asset_pack import (
        build_asset_pack,
        default_hytale_assets_path,
        load_environment_spec,
        validate_asset_pack,
    )


DEFAULT_SPEC = (
    Path(__file__).resolve().parent
    / "specs"
    / "mountain-forest-lakes-v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--assets", type=Path, default=default_hytale_assets_path())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate an existing output without reading or rebuilding Assets.zip",
    )
    args = parser.parse_args()

    spec = load_environment_spec(args.spec)
    if args.validate_only:
        report = validate_asset_pack(args.output, spec)
    else:
        report = build_asset_pack(spec, args.assets, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
