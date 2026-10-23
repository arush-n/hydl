"""Shortbow scalar-variant damage (0.7.2). Switch ``Constructs.shortbow_scalar_variant``; fidelity row 6a.

The twelve non-Iron shortbow variants (``hytalegym/combat/assets/weapon_variants/hytale_0_5_7.py``, a shared module the
render does not copy) carry the 0.5.7 damage tuples; the Iron shortbow's 0.7.2 values reach only the Iron row, through
the ``_iron_shortbow`` Redirect, because the variant rows pass their own damage. This construct takes each variant's
six-entry tuple (Strength_0..4, Signature_Volley) from the 0.7.2 item rows through the content adapter
``scalar_variant_values``. Strength_0..3 change (for example Adamantite 9, 14, 18, 25 -> 1, 3, 6, 10); Strength_4 and the
volley are equal on both tables. Gen 5 returns the original function, so its rows are unchanged.

Speeds are shared configs and are not touched here. The Iron row is not a scalar variant and passes through.
"""

from __future__ import annotations

from dataclasses import replace
from functools import wraps
from typing import Any

SWITCH = "shortbow_scalar_variant"
IMPLEMENTED = True
SHARED_ROWS = "shared_rows"
GEN7_TABLE = "gen7_table"
TEMPLATE = "shortbow"


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` for gen 5; for ``gen7_table`` the wrapper that sets the shortbow tuples from the table."""

    value = profile.constructs.shortbow_scalar_variant
    if value == SHARED_ROWS:
        return original
    if value != GEN7_TABLE:
        raise ValueError(f"unknown shortbow_scalar_variant switch {value!r}")

    from hytalegym.sim.common.content import load
    from hytalegym.sim.common.content.adapters import scalar_variant_values

    tables = load(profile)
    damages: dict[str, tuple[float, ...]] = {}

    def damage_tuple(asset_id: str) -> tuple[float, ...]:
        if asset_id not in damages:
            damages[asset_id] = scalar_variant_values(tables, asset_id, TEMPLATE)["damage"]
        return damages[asset_id]

    @wraps(original)
    def _scalar_variant_profile(spec: Any) -> Any:
        if spec.template != TEMPLATE:
            return original(spec)
        return original(replace(spec, damage=damage_tuple(spec.asset_id)))

    return _scalar_variant_profile
