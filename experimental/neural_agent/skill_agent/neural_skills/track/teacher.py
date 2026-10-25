
"""track's scripted teacher: look_at v2 aimed at entity 1's current position each tick, with no lead, and never done.

The rule is look_at's (``../look_at/teacher.py``, :func:`look_at_choices`), read on the observed state as the registered
v2 reads it (``teaching/teachers.py``). A lead hurts at every speed (``TRAINING.md`` section 8), so there is none. The
teacher reads ``combat`` (privileged: it only labels the student's decisions). Its done is False: track holds for its
whole horizon, and the request's time limit ends it, not success (``TRAINING.md`` section 1).

``step_degrees`` (opt-in, 2026-10-25, off by default, so the default is the registered rule): the yaw command is capped
at the largest codebook step within the head's per-tick rate (``agent_max_head_rotation_degrees`` x
``agent_steering_relative_turn_speed`` / ticks per second: 18 degrees on the 0.5.7 and gen-7 rulesets). The engine adds
each command to the commanded heading (``runtime/step.py`` 167), while v2 measures it from the observed head, so a
45-degree bin that the head cannot follow in one tick leaves a lead that builds up and drives the head past the person
(``TRAINING.md`` section 9).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.neural_skills.look_at.teacher import look_at_choices
from experimental.neural_agent.skill_agent.teaching.teachers import YAW_BINS, YAW_HEAD, TeacherOutput, state_for
from hytalegym.jax.combat.observation.v3.policy.look_deltas import look_delta_values
from hytalegym.jax.combat.types import TARGET_ENTITY, CombatState


def _within_step(index: jax.Array, maximum_turn_degrees: float, step_degrees: float) -> jax.Array:
    """The yaw codebook choice whose step is the decoded ``index`` clipped to +-``step_degrees``, nearest bin within
    that cap: a step over the cap becomes the largest bin inside it, and a step inside it is unchanged."""

    table = jnp.asarray(look_delta_values(YAW_BINS, maximum_degrees=maximum_turn_degrees), dtype=jnp.float32)
    capped = jnp.clip(table[index], -step_degrees, step_degrees)
    distance = jnp.where(jnp.abs(table)[None, :] <= step_degrees, jnp.abs(capped[:, None] - table[None, :]), jnp.inf)
    return jnp.argmin(distance, axis=-1).astype(jnp.int32)


def track_choices(combat: CombatState, tolerance_degrees, *, maximum_turn_degrees: float,
                  step_degrees: float | None = None) -> TeacherOutput:
    """Turn the head to entity 1's current position, with look_at's rule; done is never set. ``step_degrees`` caps the
    yaw command (the module docstring's option); None is the registered rule."""

    out = look_at_choices(state_for("look_at", combat), combat.position[:, TARGET_ENTITY], tolerance_degrees,
                          maximum_turn_degrees=maximum_turn_degrees)
    factors = out.factors
    if step_degrees is not None:
        factors = factors.at[:, YAW_HEAD].set(_within_step(factors[:, YAW_HEAD], maximum_turn_degrees, step_degrees))
    return TeacherOutput(factors=factors, done=jnp.zeros_like(out.done))


__all__ = ["track_choices"]
