"""Materialized worlds and what runs inside them.

``composite_world``     exact same-reset multi-Region worlds as one JAX world
``voxel_volume``        browser-sized voxel views over composite artifacts
``environment_recipe``  pinned minigame recipes over exact Region artifacts
``minigame_runtime``    pure-JAX objective progress, reward and termination
"""

from __future__ import annotations

from . import composite_world as composite_world
from . import environment_recipe as environment_recipe
from . import minigame_runtime as minigame_runtime
from . import voxel_volume as voxel_volume


__all__ = [
    "composite_world",
    "environment_recipe",
    "minigame_runtime",
    "voxel_volume",
]
