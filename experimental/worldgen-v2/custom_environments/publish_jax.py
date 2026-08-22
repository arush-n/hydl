"""Publish or validate a complete authored WorldGen V2-to-JAX environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ""):
    experiment_root = Path(__file__).resolve().parent.parent
    hytalegym_root = experiment_root.parents[1] / "HytaleRL" / "hytalegym"
    for source in (experiment_root, hytalegym_root):
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
    from custom_environments.publication import (
        publish_custom_environment,
        validate_custom_environment_publication,
    )
else:
    from .publication import (
        publish_custom_environment,
        validate_custom_environment_publication,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--blueprint", type=Path)
    parser.add_argument("--asset-pack", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--traversal-pack", type=Path)
    parser.add_argument("--traversal-receipt", type=Path)
    parser.add_argument("--structure-pack", type=Path)
    parser.add_argument("--structure-receipt", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        report = validate_custom_environment_publication(
            args.publication,
            source_spec_path=args.spec,
            blueprint_path=args.blueprint,
        )
    else:
        required = {
            "spec": args.spec,
            "blueprint": args.blueprint,
            "asset_pack": args.asset_pack,
            "bundle": args.bundle,
            "capture_root": args.capture_root,
            "traversal_pack": args.traversal_pack,
            "traversal_receipt": args.traversal_receipt,
            "structure_pack": args.structure_pack,
            "structure_receipt": args.structure_receipt,
            "recipe": args.recipe,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            parser.error("publication requires: " + ", ".join(missing))
        report = publish_custom_environment(
            args.publication,
            source_spec_path=args.spec,
            blueprint_path=args.blueprint,
            asset_pack_root=args.asset_pack,
            bundle_path=args.bundle,
            capture_root=args.capture_root,
            traversal_pack_path=args.traversal_pack,
            traversal_receipt_path=args.traversal_receipt,
            structure_pack_path=args.structure_pack,
            structure_receipt_path=args.structure_receipt,
            recipe_path=args.recipe,
        )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        _atomic_text(args.output.resolve(), encoded)
    print(encoded, end="")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
