"""Dagger swing damage per material (0.7.2). Switch ``Constructs.dagger_swing_damage``; phase 4, item 5c.

The scalar dagger variants (``hytalegym/combat/assets/weapon_variants/hytale_0_5_7.py``, a shared module the
render does not copy) carry the 0.5.7 damage tuples; their first two entries are Swing_Left and Swing_Right.
The 0.7.2 item rows (``weapons/item:Weapon_Daggers_*`` in ``sim/gen7/data/weapons.json``) raise those two
swings for every material (Iron 4 -> 6, Bronze 3 -> 5, Mithril 8 -> 12, ...); the other five direct values
and the angled values are unchanged. This construct takes the swing values from the gen-7 table, so the
numbers are content, and gen 5 keeps the shared rows (the original function, returned unchanged).

The hook sits on the rendered profile module's ``_scalar_variant_profile(spec)``; the profile registry
binds it with ``functools.partial(_scalar_variant_profile, spec)`` after this definition, so it sees the wrapper.
"""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any

SWITCH = "dagger_swing_damage"
IMPLEMENTED = True
SHARED_ROWS = "shared_rows"
GEN7_TABLE = "gen7_table"


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` for gen 5; the wrapper that sets the table swings for daggers when the switch selects it."""

    value = profile.constructs.dagger_swing_damage
    if value == SHARED_ROWS:
        return original
    if value != GEN7_TABLE:
        raise ValueError(f"unknown dagger_swing_damage switch {value!r}")

    from hytalegym.sim.common.content import load

    tables = load(profile)
    swings: dict[str, tuple[float, float]] = {}

    def swing_values(asset_id: str) -> tuple[float, float]:
        if asset_id not in swings:
            item = tables.row("weapons", "item:" + asset_id)
            swings[asset_id] = (
                float(item["damage"]["Swing_Left_Damage"]["base_first"]),
                float(item["damage"]["Swing_Right_Damage"]["base_first"]),
            )
        return swings[asset_id]

    @wraps(original)
    def _scalar_variant_profile(spec: Any) -> Any:
        if spec.template != "daggers":
            return original(spec)
        left, right = swing_values(spec.asset_id)
        return original(replace(spec, damage=(left, right, *spec.damage[2:])))

    return _scalar_variant_profile
