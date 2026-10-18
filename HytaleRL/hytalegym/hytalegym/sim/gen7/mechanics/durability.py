"""Weapon durability maximum and loss per hit (0.7.2). Switch ``weapon_durability`` (a ``Constructs`` field); row 8d.

The engine reads the 0.5.7 item-program catalog ``jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json`` through
``hytale_0_5_7_item_program_catalog()`` (``jax/combat/arsenal/item_programs.py``). Its ``weapon_items`` hold, for 143
weapons, ``max_durability`` and ``durability_loss_on_hit``: 120 and 0.21 for ``Weapon_Sword_Iron``. The 0.7.2 item rows
(``sim/gen7/data/durability.json``, ``durability/item:*``) hold 250 and 0.13 for the same item; 57 maximum and 82 loss
values differ. This construct wraps that loader so the gen-7 render serves the table values. The rest of the catalog,
including ``semantic_sha256``, is unchanged. Gen 5 returns the original function.

Status (2026-10-18): registered in ``mechanics/__init__.py`` and hooked in ``overlay.py`` as
``weapon_durability_catalog`` (row 8d, active). The switch is the ``Constructs`` field ``weapon_durability``
(``sim/common/profile.py``; gen 7 sets ``gen7_table``). Unit-tested in ``tests/sim/test_durability.py``.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Any

SWITCH = "weapon_durability"
IMPLEMENTED = True
SHIPPED_JSON = "shipped_json"
GEN7_TABLE = "gen7_table"


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` for gen 5; for ``gen7_table`` a loader that serves the 0.7.2 maximum and loss per hit."""

    value = profile.constructs.weapon_durability
    if value == SHIPPED_JSON:
        return original
    if value != GEN7_TABLE:
        raise ValueError(f"unknown weapon_durability switch {value!r}")

    from hytalegym.sim.common.content import load

    tables = load(profile)

    @lru_cache(maxsize=1)
    def hytale_0_5_7_item_program_catalog() -> Any:
        catalog = original()
        items = []
        for item in catalog.document["weapon_items"]:
            entry = tables.durability(item["item_asset_id"])
            if entry is None:
                raise ValueError(f"no 0.7.2 durability row for {item['item_asset_id']}")
            items.append({
                **item,
                "max_durability": float(entry.max_durability),
                "durability_loss_on_hit": float(entry.durability_loss_on_hit),
            })
        return replace(catalog, document={**catalog.document, "weapon_items": items})

    return hytale_0_5_7_item_program_catalog
