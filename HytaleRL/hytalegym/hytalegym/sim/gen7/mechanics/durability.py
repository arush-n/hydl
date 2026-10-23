"""Weapon durability maximum and loss per hit (0.7.2). Switch ``weapon_durability`` (a ``Constructs`` field); row 8d.

The engine reads the 0.5.7 item-program catalog ``jax/combat/arsenal/assets/hytale_0_5_7_item_programs_v1.json`` through
``hytale_0_5_7_item_program_catalog()`` (``jax/combat/arsenal/item_programs.py``). Its ``weapon_items`` hold, for 143
weapons, ``max_durability`` and ``durability_loss_on_hit``: 120 and 0.21 for ``Weapon_Sword_Iron``. The 0.7.2 item rows
(``sim/gen7/data/durability.json``, ``durability/item:*``) hold 250 and 0.13 for the same item; 57 maximum and 82 loss
values differ. This construct wraps that loader so the gen-7 render serves the table values. The rest of the catalog,
including ``semantic_sha256``, is unchanged. Gen 5 returns the original function.

Switch values:

* ``gen7_table``: the 0.7.2 maximum and loss per hit, as a Player wielder would see them.
* ``gen7_npc_wielder`` (gen 7 sets this): the same maxima, with ``durability_loss_on_hit = 0.0`` for every weapon row.
  Live 0.7.2 never wears a weapon held by one of our bodies, because they are NPC entities:
  ``ItemUtils.canDecreaseItemStackDurability(ref, accessor)`` returns false when the entity has no ``Player`` component
  (and for a Player in Creative); ``ItemUtils.decreaseItemStackDurability`` returns before applying
  ``getDurabilityLossOnHit`` unless that check is true; ``DamageSystems$DamageAttackerTool.handle`` calls
  ``decreaseItemStackDurability`` for every ``DurabilityLoss`` damage from an ``EntitySource``. Checked in the installed
  ``HytaleServer.jar`` bytecode (javap, 2026-10-23). Live receipt, learner body ``Kweebec_Razorleaf``
  (``HytaleRL/artifacts/fidelity_evidence/gen7_probes/durability_loss.json``, main tree): ``Weapon_Sword_Iron`` took 4
  hits per seed and ``Weapon_Battleaxe_Iron`` 3, and durability read 250.0 on every tick (``NativeInventoryCapture.java:72``).

Status (2026-10-23): registered in ``mechanics/__init__.py`` and hooked in ``overlay.py`` as
``weapon_durability_catalog`` (row 8d, active). The switch is the ``Constructs`` field ``weapon_durability``
(``sim/common/profile.py``; gen 7 sets ``gen7_npc_wielder``). Unit-tested in ``tests/sim/test_durability.py``.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Any

SWITCH = "weapon_durability"
IMPLEMENTED = True
SHIPPED_JSON = "shipped_json"
GEN7_TABLE = "gen7_table"
GEN7_NPC_WIELDER = "gen7_npc_wielder"


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` for gen 5; for ``gen7_table`` and ``gen7_npc_wielder`` a loader that serves the 0.7.2 maximum
    and, for ``gen7_table`` only, the 0.7.2 loss per hit (``gen7_npc_wielder`` serves a loss of 0.0)."""

    value = profile.constructs.weapon_durability
    if value == SHIPPED_JSON:
        return original
    if value not in (GEN7_TABLE, GEN7_NPC_WIELDER):
        raise ValueError(f"unknown weapon_durability switch {value!r}")
    npc_wielder = value == GEN7_NPC_WIELDER

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
                "durability_loss_on_hit": 0.0 if npc_wielder else float(entry.durability_loss_on_hit),
            })
        return replace(catalog, document={**catalog.document, "weapon_items": items})

    return hytale_0_5_7_item_program_catalog
