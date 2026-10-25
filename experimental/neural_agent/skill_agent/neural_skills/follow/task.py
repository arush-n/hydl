"""follow's episodes: a person walking a looped random path, the learner 3.5 blocks behind the person's start, and the
success test: the learner's planar distance to entity 1 is inside the band (2 to 5 blocks).

The person is ``tasks/simple_tasks.sample_random_walk_mover``'s path (8 waypoints within 8 blocks, looped), as
``bench/mover_checks`` runs it. Each lane's path is translated so that the learner's spawn (the flat world's default
spawn) is ``BEHIND_BLOCKS`` back along the path's first segment from the person's start, and the learner's spawn yaw faces
the person. Training lanes walk (1.5) or sneak (2.75) blocks a second; held-out lanes take a speed between them, 2.0 to 2.3.
The horizon is 300 ticks (10 s at 1/30 s). Units: blocks, blocks a second, degrees, decision ticks.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.tasks import simple_tasks as st
from hytalegym.jax.combat.types import AGENT_ENTITY, GAIT_SNEAK, GAIT_WALK, TARGET_ENTITY, CombatState

BAND_BLOCKS = (2.0, 5.0)
"""The band the learner keeps from the person, planar (near, far), in blocks."""
BEHIND_BLOCKS = (BAND_BLOCKS[0] + BAND_BLOCKS[1]) / 2.0
"""Where the learner starts and where the teacher aims: the middle of the band, 3.5 blocks behind the person."""
TRAIN_SPEEDS = (st.gait_speed(GAIT_WALK), st.gait_speed(GAIT_SNEAK))
"""Walk and sneak, the speeds trained (blocks a second)."""
HELD_OUT_SPEED_BAND = (2.0, 2.3)
"""Speeds between walk and sneak, held out (blocks a second)."""
WAYPOINTS = 8
PATH_RADIUS_BLOCKS = 8.0
HORIZON_TICKS = 300
"""10 s at 1/30 s: the episode's length, the same for every speed."""


class FollowGoals(NamedTuple):
    near: jax.Array  # f32[B]: the band's near edge, blocks
    far: jax.Array  # f32[B]: the band's far edge, blocks
    horizon_ticks: jax.Array  # int32[B]
    mover: st.Mover  # the person's path, one per lane, translated to the learner's spawn (see ``sample``)
    spawn_yaw: jax.Array  # f32[B]: the learner's spawn yaw in degrees, facing the person


def _start_behind(mover: st.Mover) -> tuple[st.Mover, jax.Array]:
    """The path translated so that the learner's spawn sits ``BEHIND_BLOCKS`` back along the first segment from the
    person's start, and the yaw that faces the person from there (``simple_tasks.mover_path``'s yaw convention)."""

    first, second = mover.waypoints[:, 0], mover.waypoints[:, 1]
    segment = second - first
    length = jnp.linalg.norm(segment, axis=1, keepdims=True)
    heading = segment / jnp.maximum(length, 1.0e-6)
    spawn = jnp.asarray(st.DEFAULT_SPAWN_XZ, dtype=jnp.float32)
    shift = spawn[None, :] + BEHIND_BLOCKS * heading - first
    yaw = jnp.rad2deg(jnp.arctan2(-heading[:, 0], -heading[:, 1]))
    return mover._replace(waypoints=mover.waypoints + shift[:, None, :]), yaw


class FollowTasks:
    """follow: stay 2 to 5 blocks behind a person who walks."""

    @staticmethod
    def sample(key: jax.Array, batch: int, *, held_out: bool) -> FollowGoals:
        """A person's path per lane, started where the learner stands 3.5 blocks behind it. Training lanes walk or sneak;
        held-out lanes take a speed in ``HELD_OUT_SPEED_BAND``."""

        key_path, key_speed = jax.random.split(key)
        if held_out:
            mover = st.sample_random_walk_mover(key_path, batch, speed_band=HELD_OUT_SPEED_BAND,
                                                waypoints=WAYPOINTS, radius=PATH_RADIUS_BLOCKS)
        else:
            mover = st.sample_random_walk_mover(key_path, batch, speed_band=(TRAIN_SPEEDS[0], TRAIN_SPEEDS[0]),
                                                waypoints=WAYPOINTS, radius=PATH_RADIUS_BLOCKS)
            sneak = jax.random.bernoulli(key_speed, 0.5, (batch,))
            mover = mover._replace(speed=jnp.where(sneak, TRAIN_SPEEDS[1], TRAIN_SPEEDS[0]).astype(jnp.float32))
        mover, spawn_yaw = _start_behind(mover)
        return FollowGoals(
            near=jnp.full((batch,), BAND_BLOCKS[0], dtype=jnp.float32),
            far=jnp.full((batch,), BAND_BLOCKS[1], dtype=jnp.float32),
            horizon_ticks=jnp.full((batch,), HORIZON_TICKS, dtype=jnp.int32),
            mover=mover,
            spawn_yaw=spawn_yaw.astype(jnp.float32),
        )

    @staticmethod
    def success(combat: CombatState, goals: FollowGoals) -> jax.Array:
        """True where the learner's planar distance to entity 1's current position is inside the band."""

        gap = combat.position[:, AGENT_ENTITY][:, [0, 2]] - combat.position[:, TARGET_ENTITY][:, [0, 2]]
        distance = jnp.hypot(gap[:, 0], gap[:, 1])
        return (distance >= goals.near) & (distance <= goals.far)


def at_speed(goals: FollowGoals, speed: float) -> FollowGoals:
    """The same paths walked at one speed (the certificate's run-speed check); the horizon is the same."""

    batch = goals.near.shape[0]
    return goals._replace(mover=goals.mover._replace(speed=jnp.full((batch,), speed, dtype=jnp.float32)))


__all__ = [
    "BAND_BLOCKS",
    "BEHIND_BLOCKS",
    "FollowGoals",
    "FollowTasks",
    "HELD_OUT_SPEED_BAND",
    "HORIZON_TICKS",
    "TRAIN_SPEEDS",
    "at_speed",
]
