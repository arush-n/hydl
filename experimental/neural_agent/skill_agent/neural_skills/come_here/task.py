"""come_here's episodes: entity 1 placed 3 to 15 blocks from the spawn at any bearing (walk_to's range), an arrival radius
and a gait as walk_to's, and the success test: the learner's planar distance to entity 1's current position is within
the radius.

Entity 1 is the flat world's second entity (``hytalegym``'s ``TARGET_ENTITY``). It is inert and stands where it was
placed at reset, so its goals carry that place (``other_xyz``): the trainer and the certificate spawn it there per lane.
Units: blocks, degrees, decision ticks.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.neural_skills.walk_to.task import WalkToTasks
from experimental.neural_agent.skill_agent.tasks.simple_tasks import DEFAULT_OTHER_XYZ
from hytalegym.jax.combat.types import AGENT_ENTITY, TARGET_ENTITY, CombatState


class ComeHereGoals(NamedTuple):
    target_xz: jax.Array  # f32[B, 2]: entity 1's place, planar (walk_to's field, for its pace and radius)
    other_xyz: jax.Array  # f32[B, 3]: entity 1's place at reset, on the floor (the trainer's and certificate's)
    radius: jax.Array  # f32[B]
    gait: jax.Array  # int32[B]
    horizon_ticks: jax.Array  # int32[B]


class ComeHereTasks:
    """come_here: walk to entity 1 and stop inside a radius of it."""

    @staticmethod
    def sample(key: jax.Array, batch: int) -> ComeHereGoals:
        """Entity 1 where walk_to's goals put a target (3 to 15 blocks from the spawn), on the floor's height, with
        walk_to's radius, gait and horizon."""

        walk = WalkToTasks.sample(key, batch)
        floor = jnp.full((batch, 1), DEFAULT_OTHER_XYZ[1], dtype=jnp.float32)
        other_xyz = jnp.concatenate([walk.target_xz[:, :1], floor, walk.target_xz[:, 1:]], axis=1)
        return ComeHereGoals(
            target_xz=walk.target_xz, other_xyz=other_xyz, radius=walk.radius, gait=walk.gait,
            horizon_ticks=walk.horizon_ticks,
        )

    @staticmethod
    def success(combat: CombatState, goals: ComeHereGoals) -> jax.Array:
        """True where the learner's planar distance to entity 1's current position is within the radius."""

        planar = combat.position[:, AGENT_ENTITY][:, [0, 2]]
        offset = planar - combat.position[:, TARGET_ENTITY][:, [0, 2]]
        return jnp.hypot(offset[:, 0], offset[:, 1]) <= goals.radius


__all__ = ["ComeHereGoals", "ComeHereTasks"]
