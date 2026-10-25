"""track's scripted teacher: look_at v2 aimed at entity 1's current position each tick, with no lead, and never done.

The rule is look_at's (``../look_at/teacher.py``, :func:`look_at_choices`), read on the observed state as the registered
v2 reads it (``teaching/teachers.py``). A lead hurts at every speed (``TRAINING.md`` section 8), so there is none. The
teacher reads ``combat`` (privileged: it only labels the student's decisions). Its done is False: track holds for its
whole horizon, and the request's time limit ends it, not success (``TRAINING.md`` section 1).
"""

from __future__ import annotations

import jax.numpy as jnp

from experimental.neural_agent.skill_agent.neural_skills.look_at.teacher import look_at_choices
from experimental.neural_agent.skill_agent.teaching.teachers import TeacherOutput, state_for
from hytalegym.jax.combat.types import TARGET_ENTITY, CombatState


def track_choices(combat: CombatState, tolerance_degrees, *, maximum_turn_degrees: float) -> TeacherOutput:
    """Turn the head to entity 1's current position, with look_at's rule; done is never set."""

    out = look_at_choices(state_for("look_at", combat), combat.position[:, TARGET_ENTITY], tolerance_degrees,
                          maximum_turn_degrees=maximum_turn_degrees)
    return TeacherOutput(factors=out.factors, done=jnp.zeros_like(out.done))


__all__ = ["track_choices"]
