"""walk_to as the training blocks and the runtime see it: its request, its held-out region, its goals, its metrics and
its spec (:func:`spec`, which binds the teacher's version).

Moved from ``training/distill.py`` on 2026-10-19: the rear-left held-out rule, the walk request's arguments, and the
pace and radius checks (E4), which now read one skill's goals and a rollout trace, so every trainer computes them the
same way.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.neural_skills.spec import Pose, RequestRow, SkillSpec
from experimental.neural_agent.skill_agent.neural_skills.walk_to.task import WalkToGoals, WalkToTasks
from experimental.neural_agent.skill_agent.neural_skills.walk_to.teacher import (
    WALK_TO_V3_STOP_MARGIN_BLOCKS,
    walk_to_choices,
)
from experimental.neural_agent.skill_agent.tasks import simple_tasks as st
from experimental.neural_agent.skill_agent.teaching.teachers import state_for
from experimental.neural_agent.skill_agent.training.blocks import (
    POSITION_SCALE,
    SPAWN_XZ,
    argument_row,
    body_vector,
    planar,
    resample,
    wrap_degrees_np,
)
from hytalegym.jax.combat.types import PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS

GAIT_SCALE = 4.0
REAR_LEFT_DEGREES = (180.0, 270.0)
STEADY_FIRST_DECISION = 10
STEADY_CLEARANCE_BLOCKS = 2.0
STEADY_HEADING_DEGREES = 10.0
STEADY_MIN_TICKS = 10
PACE_TOLERANCE = 0.05
PRESENT = np.asarray([True, False, True, True, True, True])
"""The argument groups walk_to sets: position, scalars (gait, radius), mode, urgency and age; no direction."""


def in_rear_left(target_xz: jax.Array, spawn_yaw: jax.Array) -> jax.Array:
    """Held-out (a): the walk target's bearing from the spawn body, clockwise from forward, is in [180, 270)."""

    offset = planar(target_xz - jnp.asarray(SPAWN_XZ, dtype=jnp.float32))
    forward_right = body_vector(offset, spawn_yaw)
    bearing = jnp.mod(jnp.rad2deg(jnp.arctan2(forward_right[:, 1], forward_right[:, 0])), 360.0)
    return (bearing >= REAR_LEFT_DEGREES[0]) & (bearing < REAR_LEFT_DEGREES[1])


def request_args(goals: WalkToGoals, position: jax.Array, yaw: jax.Array, ticks: jax.Array) -> jax.Array:
    """walk_to's 15 argument features: the target as a body-frame vector over 16 blocks, then the gait over 4 and the
    radius as scalars (``blocks.argument_row``)."""

    batch = position.shape[0]
    to_walk = planar(goals.target_xz - jnp.stack([position[:, 0], position[:, 2]], axis=1))
    zeros = jnp.zeros((batch,), dtype=jnp.float32)
    scalars = jnp.stack([goals.gait.astype(jnp.float32) / GAIT_SCALE, goals.radius, zeros, zeros], axis=1)
    return argument_row(body_vector(to_walk, yaw) / POSITION_SCALE, scalars, ticks)


def request(goals: WalkToGoals, pose: Pose, ticks: jax.Array) -> RequestRow:
    """walk_to's request at one decision, from the agent's own pose. It names no entity and no cell."""

    batch = pose.position.shape[0]
    none = jnp.full((batch,), -1, dtype=jnp.int32)
    return RequestRow(args=request_args(goals, pose.position, pose.yaw, ticks),
                      present=jnp.broadcast_to(jnp.asarray(PRESENT), (batch, len(PRESENT))),
                      entity_pointer=none, cell_pointer=none)


def sample(key: jax.Array, batch: int, spawn_yaw: jax.Array, *, held_out: bool) -> WalkToGoals:
    """Goals 3 to 15 blocks from the spawn (``WalkToTasks.sample``), redrawn until every target is in the rear-left
    quadrant of the spawn body (``held_out=True``) or outside it (``held_out=False``)."""

    def accept(goals: WalkToGoals) -> jax.Array:
        rear_left = in_rear_left(goals.target_xz, spawn_yaw)
        return rear_left if held_out else ~rear_left

    return resample(lambda step_key: WalkToTasks.sample(step_key, batch), accept, key)


def pace(goals: WalkToGoals, active: np.ndarray, positions: np.ndarray, yaws: np.ndarray, seconds: np.ndarray) -> dict:
    """Walk pace over the steady walk lanes: a lane is steady after ``STEADY_FIRST_DECISION`` decisions, while it is
    still well short of its target and its body points at the target to within ``STEADY_HEADING_DEGREES``. A steady
    lane's mean planar speed (over its steady decisions) must sit within 5% of the requested gait's forward speed.
    ``positions`` is float32[T+1, B, 2] and ``yaws`` float32[T+1, B] (before each decision, and after the last);
    ``seconds`` is float32[T, B]."""

    target = np.asarray(goals.target_xz, dtype=np.float64)
    radius = np.asarray(goals.radius, dtype=np.float64)
    gait = np.asarray(goals.gait)
    walk_active = np.asarray(active)
    positions = positions.astype(np.float64)
    yaws = yaws.astype(np.float64)
    seconds = np.maximum(seconds.astype(np.float64), 1e-9)
    before = positions[:-1]
    offset = target[None] - before
    distance = np.hypot(offset[..., 0], offset[..., 1])
    heading = wrap_degrees_np(np.degrees(np.arctan2(-offset[..., 0], -offset[..., 1])) - yaws[:-1])
    decision = np.arange(seconds.shape[0])[:, None]
    steady = (
        (decision >= STEADY_FIRST_DECISION)
        & (distance > radius[None] + STEADY_CLEARANCE_BLOCKS)
        & (np.abs(heading) <= STEADY_HEADING_DEGREES)
        & walk_active[None]
    )
    delta = positions[1:] - before
    speed = np.hypot(delta[..., 0], delta[..., 1]) / seconds
    counts = steady.sum(axis=0)
    lanes = counts >= STEADY_MIN_TICKS
    mean_speed = (speed * steady).sum(axis=0) / np.maximum(counts, 1)
    requested = st.NPC_WALK_BASE_SPEED * np.asarray(PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS)[gait]
    within = np.abs(mean_speed - requested) <= PACE_TOLERANCE * requested
    count = int(lanes.sum())
    return {
        "steady_lanes": count,
        "within_5_percent": float(within[lanes].mean()) if count else None,
        "mean_speed_steady_lanes": float(mean_speed[lanes].mean()) if count else None,
    }


def radius_kept(goals: WalkToGoals, active: np.ndarray, positions: np.ndarray, hit: np.ndarray) -> dict:
    """Radius respected at the first success: recomputed from the recorded position, not the success predicate. ``hit``
    is int32[B], the decision count of each lane's first success, or -1."""

    lanes = np.flatnonzero((hit >= 1) & np.asarray(active))
    if lanes.size == 0:
        return {"successes": 0, "within_radius": None}
    at = positions[hit[lanes], lanes]
    target = np.asarray(goals.target_xz)[lanes]
    distance = np.hypot(*(at - target).T)
    within = distance <= np.asarray(goals.radius)[lanes] + 1e-4
    return {"successes": int(lanes.size), "within_radius": float(within.mean())}


def metrics(goals: WalkToGoals, active: np.ndarray, trace: Any, hit: np.ndarray) -> dict:
    """walk_to's own evaluation numbers (E4): the pace and the radius, from a rollout trace with ``positions``, ``yaws``
    and ``seconds``."""

    return {"pace": pace(goals, active, trace.positions, trace.yaws, trace.seconds),
            "radius": radius_kept(goals, active, trace.positions, hit)}


@functools.lru_cache(maxsize=None)
def spec(*, stop_margin_blocks: float = WALK_TO_V3_STOP_MARGIN_BLOCKS, maximum_turn_degrees: float = 45.0) -> SkillSpec:
    """walk_to's spec with its teacher bound: v3 with the default margin (0.25 blocks), v2 at margin 0. Cached, so equal
    settings give the same spec object (the training kernels are cached on it)."""

    def teacher(combat: Any, goals: WalkToGoals):
        return walk_to_choices(state_for("walk_to", combat), goals.target_xz, goals.radius, goals.gait,
                               maximum_turn_degrees=maximum_turn_degrees, stop_margin_blocks=stop_margin_blocks)

    return SkillSpec(name="walk_to", scope="locomotion", sample=sample, request=request, teacher=teacher,
                     success=WalkToTasks.success, metrics=metrics)


__all__ = [
    "GAIT_SCALE",
    "PRESENT",
    "REAR_LEFT_DEGREES",
    "in_rear_left",
    "metrics",
    "pace",
    "radius_kept",
    "request",
    "request_args",
    "sample",
    "spec",
]
