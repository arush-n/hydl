"""JAX adapters for typed projectile-terminal deployable graphs."""

from hytalegym.combat.assets.weapon_archetypes.deployables import *

from .adapter import (
    DeployableTerminalEvent,
    adapt_projectile_terminal_deployable,
)

__all__ = [
    "DeployableTerminalEvent",
    "adapt_projectile_terminal_deployable",
]
