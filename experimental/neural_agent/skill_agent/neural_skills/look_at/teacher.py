"""look_at's scripted teacher: turn the head to a point within the window around the body, done once the actual head
is within tolerance.

The conventions and the registry entry (``TEACHERS["look_at"]``, v2: this rule on ``observed_state``) are the shared
teachers' (``teaching/teachers.py``). Pure JAX over the batch axis; ``maximum_turn_degrees`` must be a Python float.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.teaching.teachers import (
    PITCH_BINS,
    PITCH_HEAD,
    YAW_BINS,
    YAW_HEAD,
    TeacherOutput,
)
from hytalegym.jax.combat.observation.v3.policy.layout import neutral_arsenal_policy_action_factors
from hytalegym.jax.combat.observation.v3.policy.look_deltas import encode_look_delta
from hytalegym.jax.combat.runtime.math import _canonical_bearing, _normalize_degrees
from hytalegym.jax.combat.types import AGENT_ENTITY, CombatState

#: The agent's effective eye height (ruleset ``agent.effective_eye_height`` = 1.0).
EYE_OFFSET = (0.0, 1.0, 0.0)
#: The head window around the body, ruleset ``agent.head_yaw_min/max`` and ``head_pitch_min/max`` (+-45).
HEAD_WINDOW_DEGREES = 45.0


def look_at_choices(
    combat: CombatState,
    target_point: jax.Array,
    tolerance_degrees: float,
    *,
    maximum_turn_degrees: float,
) -> TeacherOutput:
    """Turn the head to ``target_point`` and report done once the actual head is within tolerance.

    Reads ``position``, ``yaw`` (body), ``desired_yaw``, ``desired_pitch``, ``agent_head_yaw`` and ``agent_head_pitch``.
    Writes the yaw and pitch heads; the other heads stay neutral.
    """

    position = combat.position[:, AGENT_ENTITY]
    eye = position + jnp.asarray(EYE_OFFSET, dtype=jnp.float32)
    target = jnp.asarray(target_point, dtype=jnp.float32)
    dx = target[:, 0] - eye[:, 0]
    dy = target[:, 1] - eye[:, 1]
    dz = target[:, 2] - eye[:, 2]

    # The head aims at the bearing, clamped to the window around the body. The offset is wrapped before the clamp,
    # so the window holds across the 180-degree seam.
    body_yaw = combat.yaw[:, AGENT_ENTITY]
    offset = jnp.clip(
        _normalize_degrees(_canonical_bearing(dx, dz) - body_yaw),
        -HEAD_WINDOW_DEGREES,
        HEAD_WINDOW_DEGREES,
    )
    head_yaw = _normalize_degrees(body_yaw + offset)
    head_pitch = jnp.clip(
        jnp.degrees(jnp.arctan2(dy, jnp.hypot(dx, dz))),
        -HEAD_WINDOW_DEGREES,
        HEAD_WINDOW_DEGREES,
    )

    yaw_index = encode_look_delta(
        _normalize_degrees(head_yaw - combat.desired_yaw), YAW_BINS, maximum_degrees=maximum_turn_degrees
    )
    pitch_index = encode_look_delta(
        head_pitch - combat.desired_pitch, PITCH_BINS, maximum_degrees=maximum_turn_degrees
    )

    tolerance = jnp.asarray(tolerance_degrees, dtype=jnp.float32)
    yaw_gap = _normalize_degrees(combat.agent_head_yaw - head_yaw)
    pitch_gap = combat.agent_head_pitch - head_pitch
    done = (jnp.abs(yaw_gap) <= tolerance) & (jnp.abs(pitch_gap) <= tolerance)

    neutral = neutral_arsenal_policy_action_factors(position.shape[0])
    factors = neutral.at[:, YAW_HEAD].set(yaw_index).at[:, PITCH_HEAD].set(pitch_index)
    return TeacherOutput(factors=factors, done=done)


__all__ = ["EYE_OFFSET", "HEAD_WINDOW_DEGREES", "look_at_choices"]
