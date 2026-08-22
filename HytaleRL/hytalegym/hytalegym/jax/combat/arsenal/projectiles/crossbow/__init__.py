"""Arsenal projectile adapters for the shared Crossbow impact program."""

from hytalegym.jax.combat.arsenal.projectiles.crossbow.adapter import (
    apply_crossbow_projectile_impacts,
)
from hytalegym.jax.combat.arsenal.projectiles.crossbow.factory import (
    empty_crossbow_projectile_impact_commands,
)
from hytalegym.jax.combat.arsenal.projectiles.crossbow.types import (
    CrossbowProjectileImpactCommands,
)

__all__ = [
    "CrossbowProjectileImpactCommands",
    "apply_crossbow_projectile_impacts",
    "empty_crossbow_projectile_impact_commands",
]
