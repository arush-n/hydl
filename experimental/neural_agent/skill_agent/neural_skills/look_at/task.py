"""look_at's episodes: points 2 to 10 blocks from the eye within 40 degrees of the body yaw and 30 of pitch, and the
success test (the head within tolerance of facing the point).

The flat world and the episode runner are shared (``tasks/simple_tasks.py``). Units: blocks, degrees, decision ticks.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.tasks.simple_tasks import AGENT_EYE_HEIGHT, DEFAULT_EYE_XYZ
from hytalegym.jax.combat.types import AGENT_ENTITY, CombatState

LOOK_DISTANCE_RANGE = (2.0, 10.0)
LOOK_YAW_RANGE_DEGREES = 40.0
LOOK_PITCH_RANGE_DEGREES = 30.0
LOOK_TOLERANCE_DEGREES = 3.0
LOOK_HORIZON_TICKS = 90


class LookAtGoals(NamedTuple):
    target_xyz: jax.Array  # f32[B, 3]
    tolerance_degrees: jax.Array  # f32[B]
    horizon_ticks: jax.Array  # int32[B]


def _wrap_degrees(value: jax.Array) -> jax.Array:
    return jnp.mod(value + 180.0, 360.0) - 180.0


class LookAtTasks:
    """look_at: turn the head to face a point."""

    @staticmethod
    def sample(
        key: jax.Array,
        batch: int,
        *,
        eye_xyz: tuple[float, float, float] = DEFAULT_EYE_XYZ,
        body_yaw_degrees: float = 0.0,
    ) -> LookAtGoals:
        """Points 2 to 10 blocks from the eye, within +-40 degrees of the body yaw and +-30 of pitch."""

        key_distance, key_yaw, key_pitch = jax.random.split(key, 3)
        distance = jax.random.uniform(
            key_distance, (batch,), dtype=jnp.float32,
            minval=LOOK_DISTANCE_RANGE[0], maxval=LOOK_DISTANCE_RANGE[1],
        )
        body_yaw = jnp.broadcast_to(jnp.asarray(body_yaw_degrees, dtype=jnp.float32), (batch,))
        yaw_offset = jax.random.uniform(
            key_yaw, (batch,), dtype=jnp.float32,
            minval=-LOOK_YAW_RANGE_DEGREES, maxval=LOOK_YAW_RANGE_DEGREES,
        )
        pitch = jax.random.uniform(
            key_pitch, (batch,), dtype=jnp.float32,
            minval=-LOOK_PITCH_RANGE_DEGREES, maxval=LOOK_PITCH_RANGE_DEGREES,
        )
        yaw = jnp.deg2rad(body_yaw + yaw_offset)
        pitch = jnp.deg2rad(pitch)
        horizontal = distance * jnp.cos(pitch)
        # The inverse of the bearing and pitch in ``success``, and of
        # ``projectile_look_direction`` (arsenal/projectiles/runtime.py:2267-2275).
        direction = jnp.stack(
            (-jnp.sin(yaw) * horizontal, distance * jnp.sin(pitch), -jnp.cos(yaw) * horizontal),
            axis=1,
        )
        target_xyz = jnp.asarray(eye_xyz, dtype=jnp.float32)[None, :] + direction
        return LookAtGoals(
            target_xyz=target_xyz,
            tolerance_degrees=jnp.full((batch,), LOOK_TOLERANCE_DEGREES, dtype=jnp.float32),
            horizon_ticks=jnp.full((batch,), LOOK_HORIZON_TICKS, dtype=jnp.int32),
        )

    @staticmethod
    def success(combat: CombatState, goals: LookAtGoals) -> jax.Array:
        """True where the head's yaw and pitch are within tolerance of facing the target."""

        eye = combat.position[:, AGENT_ENTITY] + jnp.asarray(
            [0.0, AGENT_EYE_HEIGHT, 0.0], dtype=jnp.float32
        )
        offset = goals.target_xyz - eye
        dx, dy, dz = offset[:, 0], offset[:, 1], offset[:, 2]
        yaw_facing = jnp.rad2deg(jnp.arctan2(-dx, -dz))
        pitch_facing = jnp.rad2deg(jnp.arctan2(dy, jnp.hypot(dx, dz)))
        yaw_error = _wrap_degrees(combat.agent_head_yaw - yaw_facing)
        pitch_error = combat.agent_head_pitch - pitch_facing
        return (jnp.abs(yaw_error) <= goals.tolerance_degrees) & (
            jnp.abs(pitch_error) <= goals.tolerance_degrees
        )


__all__ = [
    "LOOK_DISTANCE_RANGE",
    "LOOK_HORIZON_TICKS",
    "LOOK_PITCH_RANGE_DEGREES",
    "LOOK_TOLERANCE_DEGREES",
    "LOOK_YAW_RANGE_DEGREES",
    "LookAtGoals",
    "LookAtTasks",
]
