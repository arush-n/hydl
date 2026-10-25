"""come_here as the training blocks and the runtime see it: its request, its held-out region, its goals, its metrics and
its spec (:func:`spec`, which binds the teacher's version).

The request names no position: the target is entity 1's token, which the request points at by its handle (entity 1's
id). The trainer and the runtime resolve the handle to the token row (``runtime/state_tick.point``). Its metrics are
walk_to's pace and radius (E4), since come_here is walk_to with the target taken from the entity.
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.neural_skills.come_here.task import ComeHereGoals, ComeHereTasks
from experimental.neural_agent.skill_agent.neural_skills.come_here.teacher import come_here_choices
from experimental.neural_agent.skill_agent.neural_skills.spec import Pose, RequestRow, SkillSpec
from experimental.neural_agent.skill_agent.neural_skills.walk_to import skill as walk_skill
from experimental.neural_agent.skill_agent.neural_skills.walk_to.teacher import WALK_TO_V3_STOP_MARGIN_BLOCKS
from experimental.neural_agent.skill_agent.training.blocks import argument_row, resample
from hytalegym.jax.combat.types import TARGET_ENTITY

GAIT_SCALE = 4.0
PRESENT = np.asarray([False, False, True, True, True, True])
"""The argument groups come_here sets: scalars (gait, radius), mode, urgency and age. No position, no direction."""


def request_args(goals: ComeHereGoals, ticks: jax.Array) -> jax.Array:
    """come_here's 15 argument features: the position group is zero (the entity's token holds the place), then the gait
    over 4 and the radius as scalars (walk_to's), the mode, urgency and age (``blocks.argument_row``)."""

    batch = goals.radius.shape[0]
    zeros = jnp.zeros((batch,), dtype=jnp.float32)
    scalars = jnp.stack([goals.gait.astype(jnp.float32) / GAIT_SCALE, goals.radius, zeros, zeros], axis=1)
    return argument_row(jnp.zeros((batch, 3), dtype=jnp.float32), scalars, ticks)


def request(goals: ComeHereGoals, pose: Pose, ticks: jax.Array) -> RequestRow:
    """come_here's request at one decision. Its entity pointer holds entity 1's handle (its id), not a row: the trainer
    and the runtime resolve it to the row holding that entity. It names no cell."""

    batch = pose.position.shape[0]
    handle = jnp.full((batch,), TARGET_ENTITY, dtype=jnp.int32)
    none = jnp.full((batch,), -1, dtype=jnp.int32)
    return RequestRow(args=request_args(goals, ticks),
                      present=jnp.broadcast_to(jnp.asarray(PRESENT), (batch, len(PRESENT))),
                      entity_pointer=handle, cell_pointer=none)


def lowering(incoming: Any, pose: Pose, tick: int) -> tuple[np.ndarray, np.ndarray]:
    """come_here's request arguments for one middle-layer request (``runtime/state_tick.Lowering``): the radius is its
    first scalar and the pace its second (the gait). The entity handle is the request's own, which ``ModuleBody`` points;
    the pose and tick are the agent's own."""

    radius, pace = incoming.args.scalars[0], incoming.args.scalars[1]
    goals = ComeHereGoals(target_xz=jnp.zeros((1, 2), dtype=jnp.float32),
                          other_xyz=jnp.zeros((1, 3), dtype=jnp.float32),
                          radius=jnp.asarray([radius], dtype=jnp.float32),
                          gait=jnp.asarray([int(round(pace))], dtype=jnp.int32),
                          horizon_ticks=jnp.zeros((1,), dtype=jnp.int32))
    row = request(goals, pose, jnp.float32(tick))
    return np.asarray(row.args[0]), np.asarray(row.present[0])


def sample(key: jax.Array, batch: int, spawn_yaw: jax.Array, *, held_out: bool) -> ComeHereGoals:
    """Goals with entity 1 placed (``ComeHereTasks.sample``), redrawn until its bearing from the spawn body is in the
    rear-left quadrant (``held_out=True``) or outside it (``held_out=False``), as walk_to's targets are."""

    def accept(goals: ComeHereGoals) -> jax.Array:
        rear_left = walk_skill.in_rear_left(goals.target_xz, spawn_yaw)
        return rear_left if held_out else ~rear_left

    return resample(lambda step_key: ComeHereTasks.sample(step_key, batch), accept, key)


@functools.lru_cache(maxsize=None)
def spec(*, stop_margin_blocks: float = WALK_TO_V3_STOP_MARGIN_BLOCKS, maximum_turn_degrees: float = 45.0) -> SkillSpec:
    """come_here's spec with its teacher bound: walk_to v3's rule toward entity 1 (stop margin 0.25 blocks by default).
    Cached, so equal settings give the same spec object (the training kernels are cached on it)."""

    def teacher(combat: Any, goals: ComeHereGoals):
        return come_here_choices(combat, goals.radius, goals.gait, maximum_turn_degrees=maximum_turn_degrees,
                                 stop_margin_blocks=stop_margin_blocks)

    return SkillSpec(name="come_here", scope="locomotion", sample=sample, request=request, teacher=teacher,
                     success=ComeHereTasks.success, metrics=walk_skill.metrics)


__all__ = [
    "GAIT_SCALE",
    "PRESENT",
    "lowering",
    "request",
    "request_args",
    "sample",
    "spec",
]
