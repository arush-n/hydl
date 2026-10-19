"""look_at as the training blocks and the runtime see it: its request, its held-out region, its goals, the placement of
a look beside a walk, and its spec (:func:`spec`, which binds the request's direction and its frame).

Moved from ``training/distill.py`` on 2026-10-19: the left-band held-out rule, the look placed beside a walk, and the
look request's arguments with the unit direction to the target in the body's or the head's frame.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.model.generalist import ARG_GROUPS
from experimental.neural_agent.skill_agent.neural_skills.look_at.task import (
    LOOK_DISTANCE_RANGE,
    LookAtGoals,
    LookAtTasks,
)
from experimental.neural_agent.skill_agent.neural_skills.look_at.teacher import look_at_choices
from experimental.neural_agent.skill_agent.neural_skills.spec import Pose, RequestRow, SkillSpec
from experimental.neural_agent.skill_agent.tasks import simple_tasks as st
from experimental.neural_agent.skill_agent.teaching.teachers import state_for
from experimental.neural_agent.skill_agent.training.blocks import (
    EYE_XYZ,
    POSITION_SCALE,
    argument_row,
    body_vector,
    facing_degrees,
    head_vector,
    resample,
    wrap_degrees,
)

TOLERANCE_SCALE = 45.0
LOOK_HELD_OUT_DEGREES = (25.0, 40.0)
BOTH_LOOK_OFFSET_DEGREES = 30.0
BOTH_LOOK_PITCH_DEGREES = 20.0
FRAMES = ("body", "head")
PRESENT = np.asarray([True, False, True, True, True, True])
"""The argument groups look_at sets: position, scalars (tolerance), mode, urgency and age; the direction group is added
when the request carries the direction (:func:`request`)."""
DIRECTION_GROUP = [name for name, _ in ARG_GROUPS].index("direction")


def look_offset(target_xyz: jax.Array, spawn_yaw: jax.Array) -> jax.Array:
    """The look point's yaw offset from the spawn body, in the engine's yaw convention (positive is left)."""

    eye = jnp.asarray(EYE_XYZ, dtype=jnp.float32)
    facing = facing_degrees(target_xyz[:, 0] - eye[0], target_xyz[:, 2] - eye[2])
    return wrap_degrees(facing - spawn_yaw)


def in_look_held_out(target_xyz: jax.Array, spawn_yaw: jax.Array) -> jax.Array:
    """Held-out (b): the look point's yaw offset from the body is in [25, 40]."""

    offset = look_offset(target_xyz, spawn_yaw)
    return (offset >= LOOK_HELD_OUT_DEGREES[0]) & (offset <= LOOK_HELD_OUT_DEGREES[1])


def place_beside_walk(key: jax.Array, walk_bearing: jax.Array) -> jax.Array:
    """Look points at the walk bearing +/- 30 degrees, 2 to 10 blocks from the eye, within +/- 20 degrees of pitch."""

    batch = walk_bearing.shape[0]
    key_offset, key_pitch, key_distance = jax.random.split(key, 3)
    yaw = jnp.deg2rad(
        walk_bearing
        + jax.random.uniform(key_offset, (batch,), dtype=jnp.float32,
                             minval=-BOTH_LOOK_OFFSET_DEGREES, maxval=BOTH_LOOK_OFFSET_DEGREES)
    )
    pitch = jnp.deg2rad(
        jax.random.uniform(key_pitch, (batch,), dtype=jnp.float32,
                           minval=-BOTH_LOOK_PITCH_DEGREES, maxval=BOTH_LOOK_PITCH_DEGREES)
    )
    distance = jax.random.uniform(key_distance, (batch,), dtype=jnp.float32,
                                  minval=LOOK_DISTANCE_RANGE[0], maxval=LOOK_DISTANCE_RANGE[1])
    horizontal = distance * jnp.cos(pitch)
    direction = jnp.stack(
        [-jnp.sin(yaw) * horizontal, distance * jnp.sin(pitch), -jnp.cos(yaw) * horizontal], axis=1
    )
    return jnp.asarray(EYE_XYZ, dtype=jnp.float32)[None, :] + direction


def beside_walk(key: jax.Array, walk_target_xz: jax.Array, spawn_yaw: jax.Array) -> LookAtGoals:
    """Looks to hold while walking to ``walk_target_xz``: points placed beside the walk bearing
    (:func:`place_beside_walk`), with look_at's tolerance and horizon (``LookAtTasks.sample``)."""

    batch = walk_target_xz.shape[0]
    key_sampled, key_placed = jax.random.split(key)
    sampled = LookAtTasks.sample(key_sampled, batch, body_yaw_degrees=spawn_yaw)
    eye = jnp.asarray(EYE_XYZ, dtype=jnp.float32)
    bearing = facing_degrees(walk_target_xz[:, 0] - eye[0], walk_target_xz[:, 1] - eye[2])
    return LookAtGoals(place_beside_walk(key_placed, bearing), sampled.tolerance_degrees, sampled.horizon_ticks)


def request_args(goals: LookAtGoals, position: jax.Array, yaw: jax.Array, ticks: jax.Array,
                 head_yaw: jax.Array | None = None, head_pitch: jax.Array | None = None, *,
                 direction: bool = False, frame: str = "body") -> jax.Array:
    """look_at's 15 argument features: the target as a body-frame vector from the eye over 16 blocks, the tolerance over
    45 degrees, and with ``direction`` the unit vector from the eye to the target: in the body frame, or with
    ``frame="head"`` in the head's frame, which needs ``head_yaw`` and ``head_pitch`` (absolute, degrees)."""

    batch = position.shape[0]
    to_look = goals.target_xyz - (position + jnp.asarray([0.0, st.AGENT_EYE_HEIGHT, 0.0], dtype=jnp.float32))
    zeros = jnp.zeros((batch,), dtype=jnp.float32)
    scalars = jnp.stack([goals.tolerance_degrees / TOLERANCE_SCALE, zeros, zeros, zeros], axis=1)
    look_body = body_vector(to_look, yaw)
    unit = None
    if direction:
        look_from = look_body
        if frame == "head":
            if head_yaw is None or head_pitch is None:
                raise ValueError("look_frame 'head' needs head_yaw and head_pitch")
            look_from = head_vector(to_look, head_yaw, head_pitch)
        unit = look_from / jnp.maximum(jnp.linalg.norm(look_from, axis=1, keepdims=True), 1e-6)
    return argument_row(look_body / POSITION_SCALE, scalars, ticks, unit)


def request(goals: LookAtGoals, pose: Pose, ticks: jax.Array, *, direction: bool = True,
            frame: str = "head") -> RequestRow:
    """look_at's request at one decision, from the agent's own pose and head. It names no entity and no cell."""

    batch = pose.position.shape[0]
    present = np.asarray(PRESENT).copy()
    present[DIRECTION_GROUP] = direction
    none = jnp.full((batch,), -1, dtype=jnp.int32)
    args = request_args(goals, pose.position, pose.yaw, ticks, pose.head_yaw, pose.head_pitch, direction=direction,
                        frame=frame)
    return RequestRow(args=args, present=jnp.broadcast_to(jnp.asarray(present), (batch, len(present))),
                      entity_pointer=none, cell_pointer=none)


def sample(key: jax.Array, batch: int, spawn_yaw: jax.Array, *, held_out: bool) -> LookAtGoals:
    """Look-only goals (``LookAtTasks.sample`` around the spawn body), redrawn until every point is in the held-out left
    band (``held_out=True``) or outside it (``held_out=False``)."""

    def accept(goals: LookAtGoals) -> jax.Array:
        held = in_look_held_out(goals.target_xyz, spawn_yaw)
        return held if held_out else ~held

    return resample(lambda step_key: LookAtTasks.sample(step_key, batch, body_yaw_degrees=spawn_yaw), accept, key)


@functools.lru_cache(maxsize=None)
def spec(*, direction: bool = True, frame: str = "head", maximum_turn_degrees: float = 45.0) -> SkillSpec:
    """look_at's spec: the request carries the direction to its target in the head's frame by default (run C', README
    2.16); the teacher is the registered v2 (the rule on the observed state). Cached, so equal settings give the same
    spec object (the training kernels are cached on it)."""

    if frame not in FRAMES:
        raise ValueError(f"frame must be one of {FRAMES}, got {frame!r}")
    if frame == "head" and not direction:
        raise ValueError("frame 'head' needs direction: it measures the request's direction")

    def look_request(goals: LookAtGoals, pose: Pose, ticks: jax.Array) -> RequestRow:
        return request(goals, pose, ticks, direction=direction, frame=frame)

    def teacher(combat: Any, goals: LookAtGoals):
        return look_at_choices(state_for("look_at", combat), goals.target_xyz, goals.tolerance_degrees,
                               maximum_turn_degrees=maximum_turn_degrees)

    return SkillSpec(name="look_at", scope="gaze", sample=sample, request=look_request, teacher=teacher,
                     success=LookAtTasks.success)


__all__ = [
    "BOTH_LOOK_OFFSET_DEGREES",
    "BOTH_LOOK_PITCH_DEGREES",
    "DIRECTION_GROUP",
    "LOOK_HELD_OUT_DEGREES",
    "PRESENT",
    "TOLERANCE_SCALE",
    "beside_walk",
    "in_look_held_out",
    "look_offset",
    "place_beside_walk",
    "request",
    "request_args",
    "sample",
    "spec",
]
