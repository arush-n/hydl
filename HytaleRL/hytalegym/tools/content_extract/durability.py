"""Item durability: ``MaxDurability`` and ``DurabilityLossOnHit`` per item.

The resolution rule is the existing generator's
(``HytaleRL/scripts/combat/assets/generate_item_durability.py``,
``compiled_weapon_durability``): walk the item's ``Parent`` chain child-first
and take the first numeric (non-boolean) value; keep items whose maximum is
positive; a missing loss is the native default 0.0. The generator only covers
the simulator's compiled source items and needs the pinned 0.5.7 archive; this
extractor applies the same rule to every item of any version.
"""

from __future__ import annotations

from typing import Any

from .archive import GameArchive
from .common import is_number

FAMILY = "durability"
VERSION = 1
ITEM_PREFIX = "Server/Item/Items/"


def inherited_number(
    archive: GameArchive, paths: dict[str, str], asset_id: str, key: str
) -> tuple[float, str] | None:
    seen: set[str] = set()
    current = asset_id
    while current:
        if current in seen:
            raise ValueError(f"item inheritance cycle at {current}")
        seen.add(current)
        if current not in paths:
            raise ValueError(f"item asset is absent: {current}")
        value = archive.read_json(paths[current])
        number = value.get(key)
        if is_number(number):
            return float(number), paths[current]
        parent = value.get("Parent")
        current = parent if isinstance(parent, str) else ""
    return None


def extract(archive: GameArchive) -> dict[str, Any]:
    paths = archive.index(ITEM_PREFIX)
    rows: dict[str, Any] = {}
    for asset_id in sorted(paths):
        resolved = inherited_number(archive, paths, asset_id, "MaxDurability")
        if resolved is None or resolved[0] <= 0.0:
            continue
        maximum, source = resolved
        loss = inherited_number(archive, paths, asset_id, "DurabilityLossOnHit")
        rows[f"item:{asset_id}"] = {
            "item_asset_id": asset_id,
            "item_source": paths[asset_id],
            "max_durability": maximum,
            "durability_loss_on_hit": 0.0 if loss is None else loss[0],
            "source": source,
            "loss_source": None if loss is None else loss[1],
        }
    return rows
