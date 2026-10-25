"""follow as the training blocks and the runtime see it: its request (entity 1 by its handle, the band as two scalars, no
position), its goals, its success test, its teacher binding (:func:`spec`) and its lowering.

The request names no position and no direction: the person is entity 1's token, which the request points at by its handle
(entity 1's id), as come_here's. The band's two edges are the request's first two scalars, in blocks (come_here's radius
is raw blocks too). follow is a holding skill: its share is the ticks inside the band over its horizon, and it has no done
(``TRAINING.md`` section 8).
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.skill_agent.neural_skills.follow.task import FollowGoals, FollowTasks
from experimental.neural_agent.skill_agent.neural_skills.follow.teacher import follow_choices
from experimental.neural_agent.skill_agent.neural_skills.spec import Pose, RequestRow, SkillSpec
from experimental.neural_agent.skill_agent.training.blocks import argument_row
from hytalegym.jax.combat.types import TARGET_ENTITY

PRESENT = np.asarray([False, False, True, True, True, True])
"""The argument groups follow sets: scalars (the band's two edges), mode, urgency and age. No position, no direction."""


def request_args(goals: FollowGoals, ticks: jax.Array) -> jax.Array:
    """follow's 15 argument features: the position and direction groups are zero (the entity's token holds the place), then
    the band's near and far edges as the scalars, the mode, urgency and age (``blocks.argument_row``)."""

    batch = goals.near.shape[0]
    zeros = jnp.zeros((batch,), dtype=jnp.float32)
    scalars = jnp.stack([goals.near, goals.far, zeros, zeros], axis=1)
    return argument_row(jnp.zeros((batch, 3), dtype=jnp.float32), scalars, ticks)


def request(goals: FollowGoals, pose: Pose, ticks: jax.Array) -> RequestRow:
    """follow's request at one decision. Its entity pointer holds entity 1's handle (its id), not a row: the trainer and the
    runtime resolve it to the row holding that entity. It names no cell."""

    batch = pose.position.shape[0]
    handle = jnp.full((batch,), TARGET_ENTITY, dtype=jnp.int32)
    none = jnp.full((batch,), -1, dtype=jnp.int32)
    return RequestRow(args=request_args(goals, ticks),
                      present=jnp.broadcast_to(jnp.asarray(PRESENT), (batch, len(PRESENT))),
                      entity_pointer=handle, cell_pointer=none)


def lowering(incoming: Any, pose: Pose, tick: int) -> tuple[np.ndarray, np.ndarray]:
    """follow's request arguments for one middle-layer request (``runtime/state_tick.Lowering``): the band's near and far
    edges are the request's first two scalars; the pose and tick are the agent's own."""

    goals = FollowGoals(near=jnp.asarray([incoming.args.scalars[0]], dtype=jnp.float32),
                        far=jnp.asarray([incoming.args.scalars[1]], dtype=jnp.float32),
                        horizon_ticks=jnp.zeros((1,), dtype=jnp.int32), mover=None,
                        spawn_yaw=jnp.zeros((1,), dtype=jnp.float32))
    row = request(goals, pose, jnp.float32(tick))
    return np.asarray(row.args[0]), np.asarray(row.present[0])


def sample(key: jax.Array, batch: int, spawn_yaw: jax.Array, *, held_out: bool) -> FollowGoals:
    """The goals of ``FollowTasks.sample`` (the SkillSpec signature). ``spawn_yaw`` is not used: the learner faces the person,
    so its spawn yaw is the goals' own (``FollowGoals.spawn_yaw``), and the trainer and the certificate read it from there."""

    del spawn_yaw
    return FollowTasks.sample(key, batch, held_out=held_out)


@functools.lru_cache(maxsize=None)
def spec(*, maximum_turn_degrees: float = 45.0) -> SkillSpec:
    """follow's spec with its teacher bound: walk_to v3's rule toward the band's middle behind entity 1, sprinting. A holding
    skill: its share is every tick of its horizon (``counts`` None). Cached, so equal settings give the same spec object."""

    def teacher(combat: Any, goals: FollowGoals):
        return follow_choices(combat, goals.near, goals.far, maximum_turn_degrees=maximum_turn_degrees)

    return SkillSpec(name="follow", scope="locomotion", sample=sample, request=request, teacher=teacher,
                     success=FollowTasks.success, metrics=None, holding=True, counts=None)


__all__ = [
    "PRESENT",
    "lowering",
    "request",
    "request_args",
    "sample",
    "spec",
]
