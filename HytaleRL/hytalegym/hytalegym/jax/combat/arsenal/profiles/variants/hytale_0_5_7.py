"""JAX-facing re-export of the host-pinned Hytale 0.5.7 variant rows."""

from hytalegym.combat.assets.weapon_variants.hytale_0_5_7 import (
    SCALAR_VARIANTS,
    SCALAR_VARIANT_BY_PROFILE,
    SCALAR_VARIANT_PROFILE_NAMES,
    ScalarVariant,
)

__all__ = [
    "SCALAR_VARIANTS",
    "SCALAR_VARIANT_BY_PROFILE",
    "SCALAR_VARIANT_PROFILE_NAMES",
    "ScalarVariant",
]
