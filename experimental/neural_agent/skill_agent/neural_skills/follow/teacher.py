"""follow's scripted teacher: walk_to v3's rule toward a point behind the person, at a sprint, idle inside the stop radius.

The point is the band's middle (3.5 blocks) behind the person, along their heading: their velocity while they move, their
yaw when they stand (the same direction, for a moving person, as ``bench/mover_checks``' yaw rule). The rule is walk_to's
(``../walk_to/teacher.py``, :func:`walk_to_choices`) with the stop margin of v3, a radius of 1 block, and the sprint gait:
the teacher sprints when behind (``TRAINING.md`` section 8). It reads ``combat`` (privileged: it only labels the student's
decisions), with the learner's own state as the registered walk_to teacher reads it (``teaching/teachers.state_for``). Its
done is always False: follow holds for its horizon.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.neural_skills.walk_to.teacher import (
    WALK_TO_V3_STOP_MARGIN_BLOCKS,
    walk_to_choices,
)
from experimental.neural_agent.skill_agent.teaching.teachers import TeacherOutput, state_for
from hytalegym.jax.combat.types import GAIT_SPRINT, TARGET_ENTITY, CombatState

WALK_RADIUS_BLOCKS = 1.0
"""walk_to's radius for the stand-off point (``mover_checks``' ``WALK_RADIUS_BLOCKS``)."""


def follow_choices(combat: CombatState, near: jax.Array, far: jax.Array, *, maximum_turn_degrees: float,
                   stop_margin_blocks: float = WALK_TO_V3_STOP_MARGIN_BLOCKS) -> TeacherOutput:
    """Walk the learner to the band's middle behind entity 1, sprinting, and idle inside the stop radius; never done."""

    person = combat.position[:, TARGET_ENTITY][:, [0, 2]]
    velocity = combat.velocity[:, TARGET_ENTITY][:, [0, 2]]
    speed = jnp.hypot(velocity[:, 0], velocity[:, 1])
    moving = speed > 0.0
    radians = jnp.deg2rad(combat.yaw[:, TARGET_ENTITY])
    yaw_heading = jnp.stack((-jnp.sin(radians), -jnp.cos(radians)), axis=1)
    velocity_heading = velocity / jnp.where(moving, speed, 1.0)[:, None]
    heading = jnp.where(moving[:, None], velocity_heading, yaw_heading)
    behind = 0.5 * (near + far)
    target = person - behind[:, None] * heading
    radius = jnp.full_like(near, WALK_RADIUS_BLOCKS)
    gait = jnp.full(near.shape, GAIT_SPRINT, dtype=jnp.int32)
    out = walk_to_choices(state_for("walk_to", combat), target, radius, gait,
                          maximum_turn_degrees=maximum_turn_degrees, stop_margin_blocks=stop_margin_blocks)
    return TeacherOutput(factors=out.factors, done=jnp.zeros_like(out.done))


__all__ = ["WALK_RADIUS_BLOCKS", "follow_choices"]
