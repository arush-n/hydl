"""Fixed World-to-Combat hazard request surfaces."""

from __future__ import annotations

from typing import NamedTuple

import jax

Array = jax.Array


class EnvironmentDamageContact(NamedTuple):
    """Maximum native block/fluid damage requested by current contact."""

    requested: Array
    amount: Array
    block_damage: Array
    fluid_damage: Array
    geometry_exhausted: Array


__all__ = ["EnvironmentDamageContact"]
