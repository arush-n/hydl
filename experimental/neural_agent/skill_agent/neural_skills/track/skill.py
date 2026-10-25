"""track as the training blocks and the runtime see it: its request, its goals, its success and reach tests, and its spec
(:func:`spec`, which binds the teacher's version).

The request names no position and no direction: the target is entity 1's token, which the request points at by its handle
(entity 1's id), as come_here's. The gaze is look_at's: its tolerance is look_at's scalar, and the time limit is the
horizon. track is a holding skill: its share is the ticks in view among the ticks entity 1 is within the head's reach
(:func:`in_reach`), not a first hit, and it has no done (``TRAINING.md`` section 8, the owner's ruling of 2026-10-24).
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.neural_skills.look_at.skill import TOLERANCE_SCALE
from experimental.neural_agent.skill_agent.neural_skills.look_at.teacher import HEAD_WINDOW_DEGREES
from experimental.neural_agent.skill_agent.neural_skills.spec import Pose, RequestRow, SkillSpec
from experimental.neural_agent.skill_agent.neural_skills.track.task import TrackGoals, TrackTasks
from experimental.neural_agent.skill_agent.neural_skills.track.teacher import track_choices
from experimental.neural_agent.skill_agent.tasks import simple_tasks as st
from experimental.neural_agent.skill_agent.training.blocks import argument_row, argument_row_np
from hytalegym.jax.combat.types import AGENT_ENTITY, TARGET_ENTITY, CombatState

PRESENT = np.asarray([False, False, True, True, True, True])
"""The argument groups track sets: scalars (tolerance), mode, urgency and age. No position, no direction."""


def request_args(goals: TrackGoals, ticks: jax.Array) -> jax.Array:
    """track's 15 argument features: the position and direction groups are zero (the entity's token holds the place),
    then look_at's tolerance over 45 degrees as the scalar, the mode, urgency and age (``blocks.argument_row``)."""

    batch = goals.tolerance_degrees.shape[0]
    zeros = jnp.zeros((batch,), dtype=jnp.float32)
    scalars = jnp.stack([goals.tolerance_degrees / TOLERANCE_SCALE, zeros, zeros, zeros], axis=1)
    return argument_row(jnp.zeros((batch, 3), dtype=jnp.float32), scalars, ticks)


def request(goals: TrackGoals, pose: Pose, ticks: jax.Array) -> RequestRow:
    """track's request at one decision. Its entity pointer holds entity 1's handle (its id), not a row: the trainer and the
    runtime resolve it to the row holding that entity. It names no cell."""

    batch = pose.position.shape[0]
    handle = jnp.full((batch,), TARGET_ENTITY, dtype=jnp.int32)
    none = jnp.full((batch,), -1, dtype=jnp.int32)
    return RequestRow(args=request_args(goals, ticks),
                      present=jnp.broadcast_to(jnp.asarray(PRESENT), (batch, len(PRESENT))),
                      entity_pointer=handle, cell_pointer=none)


def lowering(incoming: Any, pose: Pose, tick: int) -> tuple[np.ndarray, np.ndarray]:
    """track's request arguments for one middle-layer request (``runtime/state_tick.Lowering``): the tolerance is the
    request's first scalar, in degrees, as look_at's lowering reads it; the pose and tick are the agent's own."""

    goals = TrackGoals(tolerance_degrees=jnp.asarray([incoming.args.scalars[0]], dtype=jnp.float32),
                       horizon_ticks=jnp.zeros((1,), dtype=jnp.int32), mover=None)
    row = request(goals, pose, jnp.float32(tick))
    return np.asarray(row.args[0]), np.asarray(row.present[0])


def lowering_batch(incomings: list[Any], poses: Pose, ticks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """:func:`lowering` for N requests at once: the numpy twin of :func:`request_args` (the position and direction groups
    are zero), in the same float32 steps. ``poses`` is unused, as in :func:`lowering`; ``ticks`` is int[N]. Returns args
    float32[N, 15] and present bool[N, 6]."""

    count = len(incomings)
    tolerance = np.asarray([item.args.scalars[0] for item in incomings], np.float32)
    zeros = np.zeros(count, np.float32)
    scalars = np.stack([tolerance / TOLERANCE_SCALE, zeros, zeros, zeros], axis=1)
    return argument_row_np(np.zeros((count, 3), np.float32), scalars, ticks), np.tile(PRESENT, (count, 1))


def sample(key: jax.Array, batch: int, spawn_yaw: jax.Array, *, held_out: bool) -> TrackGoals:
    """The paths of ``TrackTasks.sample``: crossing in front of the spawn body, walk or sneak (trained) or a speed between
    them (held out)."""

    return TrackTasks.sample(key, batch, spawn_yaw, held_out=held_out)


def in_reach(combat: CombatState, goals: TrackGoals) -> jax.Array:
    """bool[B]: entity 1 is within the head's reach from a still body: its bearing from the body within the head's window
    (45 degrees each side) plus the tolerance, and its pitch within the same reach. This is ``bench/mover_checks``' ceiling
    rule (``_in_head_window``), so the numbers match."""

    reach = HEAD_WINDOW_DEGREES + goals.tolerance_degrees
    eye = combat.position[:, AGENT_ENTITY] + jnp.asarray([0.0, st.AGENT_EYE_HEIGHT, 0.0], dtype=jnp.float32)
    offset = combat.position[:, TARGET_ENTITY] - eye
    dx, dy, dz = offset[:, 0], offset[:, 1], offset[:, 2]
    yaw = jnp.rad2deg(jnp.arctan2(-dx, -dz))
    pitch = jnp.rad2deg(jnp.arctan2(dy, jnp.hypot(dx, dz)))
    yaw_offset = jnp.mod(yaw - combat.yaw[:, AGENT_ENTITY] + 180.0, 360.0) - 180.0
    return (jnp.abs(yaw_offset) <= reach) & (jnp.abs(pitch) <= reach)


TEACHER_STEP_DEGREES = 18.0
"""The yaw cap track's teacher uses (owner, 2026-10-25: "Cap track only"): the head's per-tick rate on the gen-7 ruleset
(``teacher.py``; ``TRAINING.md`` section 9). ``spec(step_degrees=None)`` binds the uncapped rule of earlier runs."""


def spec(*, maximum_turn_degrees: float = 45.0, step_degrees: float | None = TEACHER_STEP_DEGREES) -> SkillSpec:
    """track's spec with its teacher bound: look_at v2 at entity 1, no lead, its yaw command capped at ``step_degrees``
    (``TEACHER_STEP_DEGREES`` since 2026-10-25; None is the uncapped rule of earlier runs). A holding skill: ``counts`` is
    :func:`in_reach`. Cached, so equal settings give the same spec object (the training kernels are cached on it)."""

    return _spec(float(maximum_turn_degrees), None if step_degrees is None else float(step_degrees))


@functools.lru_cache(maxsize=None)
def _spec(maximum_turn_degrees: float, step_degrees: float | None) -> SkillSpec:
    def teacher(combat: Any, goals: TrackGoals):
        return track_choices(combat, goals.tolerance_degrees, maximum_turn_degrees=maximum_turn_degrees,
                             step_degrees=step_degrees)

    return SkillSpec(name="track", scope="gaze", sample=sample, request=request, teacher=teacher,
                     success=TrackTasks.success, metrics=None, holding=True, counts=in_reach)


__all__ = [
    "PRESENT",
    "TEACHER_STEP_DEGREES",
    "in_reach",
    "lowering",
    "lowering_batch",
    "request",
    "request_args",
    "sample",
    "spec",
]
