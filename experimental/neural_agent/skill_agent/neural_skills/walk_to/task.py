"""walk_to's episodes: goals 3 to 15 blocks from the spawn at a gait and a radius, and the success test.

The flat world and the episode runner are shared (``tasks/simple_tasks.py``). Units: blocks, degrees, decision ticks.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.tasks.simple_tasks import (
    DEFAULT_SPAWN_XZ,
    NPC_WALK_BASE_SPEED,
    TICKS_PER_SECOND,
)
from hytalegym.jax.combat.types import (
    AGENT_ENTITY,
    GAIT_RUN,
    GAIT_SNEAK,
    GAIT_WALK,
    PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS,
    CombatState,
)

WALK_DISTANCE_RANGE = (3.0, 15.0)
WALK_RADIUS_RANGE = (0.5, 1.5)
#: Walk, run and sneak. Sprint is left out for now.
WALK_GAITS = (GAIT_WALK, GAIT_RUN, GAIT_SNEAK)
WALK_HORIZON_SLACK = 1.5
WALK_HORIZON_PAD_TICKS = 30


class WalkToGoals(NamedTuple):
    target_xz: jax.Array  # f32[B, 2]
    radius: jax.Array  # f32[B]
    gait: jax.Array  # int32[B]
    horizon_ticks: jax.Array  # int32[B]


class WalkToTasks:
    """walk_to: walk to a world point at a pace and stop inside a radius."""

    @staticmethod
    def sample(
        key: jax.Array, batch: int, *, spawn_xz: tuple[float, float] = DEFAULT_SPAWN_XZ
    ) -> WalkToGoals:
        """Goals 3 to 15 blocks from ``spawn_xz`` in a random direction."""

        key_distance, key_angle, key_radius, key_gait = jax.random.split(key, 4)
        distance = jax.random.uniform(
            key_distance, (batch,), dtype=jnp.float32,
            minval=WALK_DISTANCE_RANGE[0], maxval=WALK_DISTANCE_RANGE[1],
        )
        angle = jax.random.uniform(
            key_angle, (batch,), dtype=jnp.float32, minval=0.0, maxval=2.0 * math.pi
        )
        direction = jnp.stack((jnp.sin(angle), jnp.cos(angle)), axis=1)
        spawn = jnp.asarray(spawn_xz, dtype=jnp.float32)
        target_xz = spawn[None, :] + distance[:, None] * direction
        radius = jax.random.uniform(
            key_radius, (batch,), dtype=jnp.float32,
            minval=WALK_RADIUS_RANGE[0], maxval=WALK_RADIUS_RANGE[1],
        )
        gait_table = jnp.asarray(WALK_GAITS, dtype=jnp.int32)
        gait = gait_table[jax.random.randint(key_gait, (batch,), 0, len(WALK_GAITS))]
        multiplier = jnp.asarray(PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS, dtype=jnp.float32)[gait]
        horizon_ticks = (
            jnp.ceil(
                distance / (NPC_WALK_BASE_SPEED * multiplier)
                * WALK_HORIZON_SLACK * TICKS_PER_SECOND
            ).astype(jnp.int32)
            + WALK_HORIZON_PAD_TICKS
        )
        return WalkToGoals(
            target_xz=target_xz, radius=radius, gait=gait, horizon_ticks=horizon_ticks
        )

    @staticmethod
    def success(combat: CombatState, goals: WalkToGoals) -> jax.Array:
        """True where the learner's planar distance to the target is within the radius."""

        planar = combat.position[:, AGENT_ENTITY][:, [0, 2]]
        offset = planar - goals.target_xz
        return jnp.hypot(offset[:, 0], offset[:, 1]) <= goals.radius


__all__ = [
    "WALK_DISTANCE_RANGE",
    "WALK_GAITS",
    "WALK_HORIZON_PAD_TICKS",
    "WALK_HORIZON_SLACK",
    "WALK_RADIUS_RANGE",
    "WalkToGoals",
    "WalkToTasks",
]
