"""Projectile-contact handoff into the ordered Crossbow interaction chain."""

from __future__ import annotations

from typing import NamedTuple

import jax


Array = jax.Array


class CrossbowProjectileImpactCommands(NamedTuple):
    """Contact facts plus profile-resolved damage for each projectile slot."""

    requested: Array
    kind: Array
    source_slot: Array
    target_slot: Array
    knockback_yaw_degrees: Array
    standard_damage: Array
    combo_damage: Array
    big_arrow_damage: Array
