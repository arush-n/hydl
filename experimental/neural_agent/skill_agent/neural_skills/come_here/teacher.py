"""come_here's scripted teacher: walk_to v3's rule, aimed at entity 1's current position each tick.

The rule is walk_to's (``../walk_to/teacher.py``, :func:`walk_to_choices`), with the target read from the state every
tick, so a target that moves is followed. The teacher reads ``combat`` (privileged: it only labels the student's
decisions). Its stop margin is walk_to v3's (0.25 blocks by default).
"""

from __future__ import annotations

import jax

from experimental.neural_agent.skill_agent.neural_skills.walk_to.teacher import (
    WALK_TO_V3_STOP_MARGIN_BLOCKS,
    walk_to_choices,
)
from experimental.neural_agent.skill_agent.teaching.teachers import TeacherOutput, state_for
from hytalegym.jax.combat.types import TARGET_ENTITY, CombatState


def come_here_choices(
    combat: CombatState,
    radius: jax.Array,
    gait: jax.Array,
    *,
    maximum_turn_degrees: float,
    stop_margin_blocks: float = WALK_TO_V3_STOP_MARGIN_BLOCKS,
) -> TeacherOutput:
    """Walk the learner to entity 1's position now, at ``gait``, and stop inside ``radius`` (walk_to's rule)."""

    target = combat.position[:, TARGET_ENTITY][:, [0, 2]]
    return walk_to_choices(
        state_for("walk_to", combat), target, radius, gait,
        maximum_turn_degrees=maximum_turn_degrees, stop_margin_blocks=stop_margin_blocks,
    )


__all__ = ["come_here_choices"]
