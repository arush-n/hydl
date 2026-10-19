"""Crossbow scalar-variant damage (0.7.2). Switch ``Constructs.crossbow_scalar_variant``; phase 4, item 5d.

The scalar Crossbow variants (``hytalegym/combat/assets/weapon_variants/hytale_0_5_7.py``, a shared module the render
does not copy) carry the 0.5.7 damage triple ``(standard, combo, Big Arrow)``; Ancient Steel is (16, 42, 123). The 0.7.2
item rows (``weapons/item:Weapon_Crossbow_*`` in ``sim/gen7/data/weapons.json``) state the same three contacts as
``Standard_Projectile_Damage``, ``Combo_Projectile_Damage`` and ``Signature_BigArrow_Damage``; Ancient Steel is
(10.5, 14, 82). This construct takes the triple from the gen-7 table. Durability is not touched here: the
``weapon_durability`` construct (row 8d) already serves the 0.7.2 maximum and loss for every weapon item, this one
included. Gen 5 returns the original function, so its rows are unchanged.

The table triple is the authored contact value. The server's applied integer is not modelled here: it applied a 10.5
standard hit as 11.0 on both seeds (phase4o), and the rounding rule is not yet separated.
"""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any

SWITCH = "crossbow_scalar_variant"
IMPLEMENTED = True
SHARED_ROWS = "shared_rows"
GEN7_TABLE = "gen7_table"
TEMPLATE = "crossbow"
#: Standard, combo and Big Arrow contacts, in the order of ``ScalarVariant.damage``.
DAMAGE_KEYS = (
    "Standard_Projectile_Damage",
    "Combo_Projectile_Damage",
    "Signature_BigArrow_Damage",
)


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` for gen 5; for ``gen7_table`` the wrapper that sets the Crossbow triple from the table."""

    value = profile.constructs.crossbow_scalar_variant
    if value == SHARED_ROWS:
        return original
    if value != GEN7_TABLE:
        raise ValueError(f"unknown crossbow_scalar_variant switch {value!r}")

    from hytalegym.sim.common.content import load

    tables = load(profile)
    triples: dict[str, tuple[float, float, float]] = {}

    def damage_triple(asset_id: str) -> tuple[float, float, float]:
        if asset_id not in triples:
            item = tables.row("weapons", "item:" + asset_id)
            triples[asset_id] = tuple(
                float(item["damage"][key]["base_first"]) for key in DAMAGE_KEYS
            )
        return triples[asset_id]

    @wraps(original)
    def _scalar_variant_profile(spec: Any) -> Any:
        if spec.template != TEMPLATE:
            return original(spec)
        return original(replace(spec, damage=damage_triple(spec.asset_id)))

    return _scalar_variant_profile
