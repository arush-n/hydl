"""walk_to's scripted teacher: walk the learner to a point at a gait and stop inside a radius (v2, v3 with a margin).

The conventions (degrees, axes, the decoder's codebooks, the locomotion choice layout) and the registry entry
(``TEACHERS["walk_to"]``) are the shared teachers' (``teaching/teachers.py``). Pure JAX over the batch axis;
``maximum_turn_degrees`` must be a Python float, static under ``jit``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.teaching.teachers import (
    BODY_YAW_HEAD,
    COMPASS_DIRECTIONS,
    LOCOMOTION_HEAD,
    TEACHERS,
    YAW_BINS,
    TeacherOutput,
)
from hytalegym.jax.combat.observation.v3.policy.layout import neutral_arsenal_policy_action_factors
from hytalegym.jax.combat.observation.v3.policy.look_deltas import encode_look_delta, look_delta_values
from hytalegym.jax.combat.runtime.math import _canonical_bearing, _normalize_degrees
from hytalegym.jax.combat.types import AGENT_ENTITY, GAIT_RUN, GAIT_SPRINT, CombatState

#: walk_to v3 stops this far inside the radius, or half the radius if that is less. v2 stops on the edge, and its A2
#: student parks a median 0.02 blocks outside it (README 2.16).
WALK_TO_V3_STOP_MARGIN_BLOCKS = 0.25


def _codebook(bins: int, maximum_turn_degrees: float) -> jax.Array:
    """Decoded value of each bin, exactly as ``decode_look_delta`` builds it for this turn limit."""

    return jnp.asarray(look_delta_values(bins, maximum_degrees=maximum_turn_degrees), dtype=jnp.float32)


def _compass_direction(dx: jax.Array, dz: jax.Array, body_yaw_degrees: jax.Array) -> jax.Array:
    """Body-relative direction choice 1..8 of the planar heading (dx, dz) for a body at ``body_yaw_degrees``."""

    radians = jnp.deg2rad(body_yaw_degrees)
    along = -jnp.sin(radians) * dx - jnp.cos(radians) * dz  # component on forward
    across = jnp.cos(radians) * dx - jnp.sin(radians) * dz  # component on right
    angle = jnp.degrees(jnp.arctan2(across, along))
    sector = jnp.mod(jnp.round(angle / 45.0).astype(jnp.int32), COMPASS_DIRECTIONS)
    return sector + 1


def walk_to_choices(
    combat: CombatState,
    target_xz: jax.Array,
    radius: jax.Array,
    gait: jax.Array,
    *,
    maximum_turn_degrees: float,
    stop_margin_blocks: float = 0.0,
) -> TeacherOutput:
    """Walk the learner to ``target_xz`` at gait ``gait`` (1..4) and stop once inside ``radius``.

    Reads ``position[:, 0]`` and ``desired_body_yaw``. Writes the locomotion and body-yaw heads; the other heads stay
    neutral. Once inside the radius the teacher is done; once inside its stop radius it idles with a neutral body steer.
    The stop radius is ``radius`` less ``stop_margin_blocks``, but never under half the radius: v3 walks on past the
    edge, so a student whose stop is slightly off still stops inside. At 0 (v2, ``walk_to_version``) the two radii are
    the same.
    """

    if stop_margin_blocks < 0:
        raise ValueError(f"stop_margin_blocks must be at least 0 (a stop inside the radius), got {stop_margin_blocks}")
    position = combat.position[:, AGENT_ENTITY]
    target = jnp.asarray(target_xz, dtype=jnp.float32)
    dx = target[:, 0] - position[:, 0]
    dz = target[:, 1] - position[:, 2]
    distance = jnp.hypot(dx, dz)
    done = distance <= radius
    stopped = distance <= jnp.maximum(radius - stop_margin_blocks, 0.5 * radius)

    # The body steers toward the target by one codebook step per tick. Travel uses the commanded body yaw, which
    # already includes that step, so the direction is taken in that frame.
    body_error = _normalize_degrees(_canonical_bearing(dx, dz) - combat.desired_body_yaw)
    body_index = encode_look_delta(body_error, YAW_BINS, maximum_degrees=maximum_turn_degrees)
    commanded_yaw = combat.desired_body_yaw + _codebook(YAW_BINS, maximum_turn_degrees)[body_index]
    direction = _compass_direction(dx, dz, commanded_yaw)

    gait_choice = jnp.asarray(gait, dtype=jnp.int32)
    # Sprint only applies to forward travel; any other direction falls back to run.
    gait_choice = jnp.where((gait_choice == GAIT_SPRINT) & (direction != 1), GAIT_RUN, gait_choice)
    locomotion = jnp.where(stopped, 0, (gait_choice - 1) * COMPASS_DIRECTIONS + direction)

    neutral = neutral_arsenal_policy_action_factors(position.shape[0])
    body_index = jnp.where(stopped, neutral[:, BODY_YAW_HEAD], body_index)
    factors = neutral.at[:, LOCOMOTION_HEAD].set(locomotion).at[:, BODY_YAW_HEAD].set(body_index)
    return TeacherOutput(factors=factors, done=done)


def walk_to_version(stop_margin_blocks: float) -> str:
    """The walk_to version a stop margin gives: v3 when it is positive, else the registered version."""

    return "v3" if stop_margin_blocks > 0 else TEACHERS["walk_to"].version


__all__ = ["WALK_TO_V3_STOP_MARGIN_BLOCKS", "walk_to_choices", "walk_to_version"]
