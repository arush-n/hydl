"""track's episodes: a person crossing in front of the agent on a straight path, and the success test: the head within
look_at's tolerance of entity 1.

The path is a two-point line perpendicular to the body's forward axis: it crosses the forward axis at 3 to 10 blocks
from the spawn, starts on one side of the body and ends on the other, and the person walks it at a speed and stops at
its end (``tasks/simple_tasks.py``, ``Mover``). Training draws walk (1.5) or sneak (2.75) blocks a second; held-out
lanes draw a speed between them, 2.0 to 2.3. The horizon is the path's duration at the speed, capped at 300 ticks.
Entity 1 is placed where the path starts (the mover does it at reset). Units: blocks, degrees, decision ticks.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.neural_skills.look_at.task import (
    LOOK_TOLERANCE_DEGREES,
    LookAtGoals,
    LookAtTasks,
)
from experimental.neural_agent.skill_agent.tasks import simple_tasks as st
from hytalegym.jax.combat.types import GAIT_SNEAK, GAIT_WALK, TARGET_ENTITY, CombatState

PATH_HALF_WIDTH_BLOCKS = 6.0
"""The path runs this far to each side of the crossing point (12 blocks in all)."""
CROSSING_RANGE_BLOCKS = (3.0, 10.0)
"""The crossing point's distance from the spawn, in front of the body."""
TRAIN_SPEEDS = (st.gait_speed(GAIT_WALK), st.gait_speed(GAIT_SNEAK))
"""Walk and sneak, the speeds trained (blocks a second)."""
HELD_OUT_SPEED_BAND = (2.0, 2.3)
"""Speeds between walk and sneak, held out (blocks a second; the plan's "speeds not trained")."""
HORIZON_CAP_TICKS = 300


class TrackGoals(NamedTuple):
    tolerance_degrees: jax.Array  # f32[B]: look_at's tolerance
    horizon_ticks: jax.Array  # int32[B]: the path's duration at its speed, capped
    mover: st.Mover | None  # entity 1's path, one per lane: the trainer and the certificate run the environment with it


def horizon_ticks(speed: jax.Array) -> jax.Array:
    """int32[B]: the ticks the person takes to walk the path at ``speed``, capped at ``HORIZON_CAP_TICKS``."""

    length = 2.0 * PATH_HALF_WIDTH_BLOCKS
    ticks = jnp.ceil(length / speed * st.TICKS_PER_SECOND)
    return jnp.minimum(ticks, HORIZON_CAP_TICKS).astype(jnp.int32)


class TrackTasks:
    """track: keep the head on a moving entity."""

    @staticmethod
    def sample(key: jax.Array, batch: int, spawn_yaw: jax.Array, *, held_out: bool) -> TrackGoals:
        """A path per lane, crossing in front of the spawn body (yaw ``spawn_yaw``). Training lanes walk or sneak; held-out
        lanes take a speed in ``HELD_OUT_SPEED_BAND``."""

        key_side, key_distance, key_speed = jax.random.split(key, 3)
        side = jnp.where(jax.random.bernoulli(key_side, 0.5, (batch,)), 1.0, -1.0).astype(jnp.float32)
        distance = jax.random.uniform(key_distance, (batch,), dtype=jnp.float32,
                                      minval=CROSSING_RANGE_BLOCKS[0], maxval=CROSSING_RANGE_BLOCKS[1])
        if held_out:
            speed = jax.random.uniform(key_speed, (batch,), dtype=jnp.float32,
                                       minval=HELD_OUT_SPEED_BAND[0], maxval=HELD_OUT_SPEED_BAND[1])
        else:
            sneak = jax.random.bernoulli(key_speed, 0.5, (batch,))
            speed = jnp.where(sneak, TRAIN_SPEEDS[1], TRAIN_SPEEDS[0]).astype(jnp.float32)
        yaw = jnp.deg2rad(jnp.asarray(spawn_yaw, dtype=jnp.float32))
        forward = jnp.stack((-jnp.sin(yaw), -jnp.cos(yaw)), axis=1)  # the engine's forward, in x-z
        right = jnp.stack((jnp.cos(yaw), -jnp.sin(yaw)), axis=1)
        spawn = jnp.asarray(st.DEFAULT_SPAWN_XZ, dtype=jnp.float32)
        crossing = spawn[None, :] + distance[:, None] * forward
        start = crossing - side[:, None] * PATH_HALF_WIDTH_BLOCKS * right
        end = crossing + side[:, None] * PATH_HALF_WIDTH_BLOCKS * right
        mover = st.Mover(
            waypoints=jnp.stack((start, end), axis=1),
            speed=speed,
            loop=jnp.zeros((batch,), dtype=jnp.bool_),
        )
        return TrackGoals(
            tolerance_degrees=jnp.full((batch,), LOOK_TOLERANCE_DEGREES, dtype=jnp.float32),
            horizon_ticks=horizon_ticks(speed),
            mover=mover,
        )

    @staticmethod
    def success(combat: CombatState, goals: TrackGoals) -> jax.Array:
        """True where the head is within tolerance of entity 1 (look_at's own test, aimed at entity 1's position)."""

        look = LookAtGoals(target_xyz=combat.position[:, TARGET_ENTITY], tolerance_degrees=goals.tolerance_degrees,
                           horizon_ticks=goals.horizon_ticks)
        return LookAtTasks.success(combat, look)


def at_speed(goals: TrackGoals, speed: float) -> TrackGoals:
    """The same paths walked at one speed (the certificate's run-speed check), with the horizon that speed implies."""

    batch = goals.horizon_ticks.shape[0]
    pace = jnp.full((batch,), speed, dtype=jnp.float32)
    return goals._replace(horizon_ticks=horizon_ticks(pace), mover=goals.mover._replace(speed=pace))


__all__ = [
    "CROSSING_RANGE_BLOCKS",
    "HELD_OUT_SPEED_BAND",
    "HORIZON_CAP_TICKS",
    "PATH_HALF_WIDTH_BLOCKS",
    "TRAIN_SPEEDS",
    "TrackGoals",
    "TrackTasks",
    "at_speed",
    "horizon_ticks",
]
