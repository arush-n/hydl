"""Regenerate compiled-weapon durability from the pinned Hytale 0.5.7 assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from hytalegym.combat.assets.catalog import COMPILED_SOURCE_IDS
from hytalegym.rulesets import require_hytale_0_5_7_assets_archive


REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = (
    REPO_ROOT
    / "hytalegym"
    / "hytalegym"
    / "jax"
    / "combat"
    / "arsenal"
    / "assets"
    / "hytale_0_5_7_item_programs_v1.json"
)


def compiled_weapon_durability() -> list[dict[str, object]]:
    """Resolve positive MaxDurability values for every compiled source item."""

    archive = require_hytale_0_5_7_assets_archive()
    with ZipFile(archive) as source:
        paths = {
            Path(name).stem: name
            for name in source.namelist()
            if name.startswith("Server/Item/Items/") and name.endswith(".json")
        }
        cache: dict[str, dict[str, object]] = {}

        def document(asset_id: str) -> dict[str, object]:
            if asset_id not in paths:
                raise ValueError(f"compiled item asset is absent: {asset_id}")
            if asset_id not in cache:
                value = json.loads(source.read(paths[asset_id]))
                if not isinstance(value, dict):
                    raise TypeError(f"{paths[asset_id]} is not an object")
                cache[asset_id] = value
            return cache[asset_id]

        def inherited_number(
            asset_id: str,
            key: str,
        ) -> tuple[float, str] | None:
            seen: set[str] = set()
            current = asset_id
            while current:
                if current in seen:
                    raise ValueError(f"item inheritance cycle at {current}")
                seen.add(current)
                value = document(current)
                number = value.get(key)
                if isinstance(number, (int, float)) and not isinstance(
                    number,
                    bool,
                ):
                    return float(number), paths[current]
                parent = value.get("Parent")
                current = parent if isinstance(parent, str) else ""
            return None

        rows: list[dict[str, object]] = []
        for asset_id in sorted(COMPILED_SOURCE_IDS):
            resolved = inherited_number(asset_id, "MaxDurability")
            if resolved is None or resolved[0] <= 0.0:
                continue
            maximum, source_path = resolved
            loss = inherited_number(asset_id, "DurabilityLossOnHit")
            rows.append(
                {
                    "item_asset_id": asset_id,
                    "max_durability": maximum,
                    # Item's native field default is zero. Preserve that
                    # default when no inherited asset authors the field.
                    "durability_loss_on_hit": (
                        0.0 if loss is None else loss[0]
                    ),
                    "source_path": source_path,
                }
            )
    return rows


def rendered_catalog() -> str:
    """Return the catalog with refreshed rows and semantic identity."""

    document = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    document["weapon_items"] = compiled_weapon_durability()
    canonical = dict(document)
    canonical.pop("semantic_sha256", None)
    document["semantic_sha256"] = hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest().upper()
    return json.dumps(document, indent=2, ensure_ascii=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="replace the packaged catalog; otherwise only check it",
    )
    args = parser.parse_args()
    expected = rendered_catalog()
    current = CATALOG_PATH.read_text(encoding="utf-8")
    if args.write:
        CATALOG_PATH.write_text(expected, encoding="utf-8", newline="\n")
        print(
            f"wrote {len(json.loads(expected)['weapon_items'])} durability rows "
            f"to {CATALOG_PATH}"
        )
        return 0
    if current != expected:
        print(f"stale durability catalog: run {Path(__file__).name} --write")
        return 1
    print(f"durability catalog is current: {CATALOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
