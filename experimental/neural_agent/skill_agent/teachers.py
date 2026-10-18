"""Scripted teachers: privileged-state controllers that label data for the skill policies.

A teacher reads the simulator's full ``CombatState`` (teachers may read privileged state; SKILL.md, privileged-input
rule) and returns one action row per lane on the gym's 12-head factored surface, ``int32[B, 12]`` (layout in
``hytalegym.jax.combat.observation.v3.policy.layout``). Every function is pure JAX over the batch axis, with no
Python loop over lanes. ``maximum_turn_degrees`` must be a Python float: the look-delta codebook is built from it at
trace time, so it is static under ``jit``.

Conventions, each checked against the engine:

* Degrees throughout. Axes are (x, y up, z). At body yaw ``y`` the body's forward is (-sin y, -cos y) and its right is
  (cos y, -sin y), in (x, z) (``runtime/step.py``, desired velocity).
* Yaw, body-yaw and pitch deltas are quantised against the decoder's own codebook,
  ``look_delta_values(size, maximum_degrees=M)`` (``observation/v3/policy/look_deltas.py``), so a factor decodes back
  to exactly the value that was chosen.
* Locomotion choice 0 is idle. Choice ``(gait - 1) * 8 + d`` walks gait 1..4 (walk, run, sprint, sneak) in body
  direction ``d``: 1 forward, 2 forward-right, 3 right, and so on in 45-degree steps
  (``observation/v3/policy/actions.py``, ``apply_world_move_override``). The engine moves along the COMMANDED body
  yaw, the body yaw after this tick's body-yaw delta, so a walking teacher takes its direction in that frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.observation.v3.policy.layout import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    neutral_arsenal_policy_action_factors,
)
from hytalegym.jax.combat.observation.v3.policy.look_deltas import (
    encode_look_delta,
    look_delta_values,
)
from hytalegym.jax.combat.runtime.math import _canonical_bearing, _normalize_degrees
from hytalegym.jax.combat.types import AGENT_ENTITY, GAIT_RUN, GAIT_SPRINT, CombatState

_HEAD = {name: index for index, name in enumerate(ARSENAL_POLICY_ACTION_HEAD_NAMES)}
LOCOMOTION_HEAD = _HEAD["locomotion_gait_compass"]
YAW_HEAD = _HEAD["yaw_delta_bins"]
BODY_YAW_HEAD = _HEAD["body_yaw_delta_bins"]
PITCH_HEAD = _HEAD["pitch_delta_bins"]

YAW_BINS = 9
PITCH_BINS = 5
COMPASS_DIRECTIONS = 8
#: The agent's effective eye height (ruleset ``agent.effective_eye_height`` = 1.0).
EYE_OFFSET = (0.0, 1.0, 0.0)
#: The head window around the body, ruleset ``agent.head_yaw_min/max`` and ``head_pitch_min/max`` (+-45).
HEAD_WINDOW_DEGREES = 45.0


class TeacherOutput(NamedTuple):
    """One teacher decision per lane: the factor row to execute and whether the goal already holds."""

    factors: jax.Array  # int32[B, 12]
    done: jax.Array  # bool[B]


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
) -> TeacherOutput:
    """Walk the learner to ``target_xz`` at gait ``gait`` (1..4) and stop once inside ``radius``.

    Reads ``position[:, 0]`` and ``desired_body_yaw``. Writes the locomotion and body-yaw heads; the other heads stay
    neutral. Once inside the radius the teacher is done: it idles with a neutral body steer.
    """

    position = combat.position[:, AGENT_ENTITY]
    target = jnp.asarray(target_xz, dtype=jnp.float32)
    dx = target[:, 0] - position[:, 0]
    dz = target[:, 1] - position[:, 2]
    done = jnp.hypot(dx, dz) <= radius

    # The body steers toward the target by one codebook step per tick. Travel uses the commanded body yaw, which
    # already includes that step, so the direction is taken in that frame.
    body_error = _normalize_degrees(_canonical_bearing(dx, dz) - combat.desired_body_yaw)
    body_index = encode_look_delta(body_error, YAW_BINS, maximum_degrees=maximum_turn_degrees)
    commanded_yaw = combat.desired_body_yaw + _codebook(YAW_BINS, maximum_turn_degrees)[body_index]
    direction = _compass_direction(dx, dz, commanded_yaw)

    gait_choice = jnp.asarray(gait, dtype=jnp.int32)
    # Sprint only applies to forward travel; any other direction falls back to run.
    gait_choice = jnp.where((gait_choice == GAIT_SPRINT) & (direction != 1), GAIT_RUN, gait_choice)
    locomotion = jnp.where(done, 0, (gait_choice - 1) * COMPASS_DIRECTIONS + direction)

    neutral = neutral_arsenal_policy_action_factors(position.shape[0])
    body_index = jnp.where(done, neutral[:, BODY_YAW_HEAD], body_index)
    factors = neutral.at[:, LOCOMOTION_HEAD].set(locomotion).at[:, BODY_YAW_HEAD].set(body_index)
    return TeacherOutput(factors=factors, done=done)


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


@dataclass(frozen=True)
class TeacherSpec:
    """Registry record for one scripted teacher: the verbs it teaches, what it reads and its parameters."""

    verbs: tuple[str, ...]
    reads: tuple[str, ...]
    parameters: tuple[str, ...]
    kind: str = "scripted"
    version: str = "v1"
    certificate: object | None = None


TEACHERS: dict[str, TeacherSpec] = {
    "walk_to": TeacherSpec(
        verbs=("walk_to",),
        reads=("position", "desired_body_yaw"),
        parameters=("target_xz", "radius", "gait", "maximum_turn_degrees"),
    ),
    "look_at": TeacherSpec(
        verbs=("look_at",),
        reads=(
            "position",
            "yaw",
            "desired_yaw",
            "desired_pitch",
            "agent_head_yaw",
            "agent_head_pitch",
        ),
        parameters=("target_point", "tolerance_degrees", "maximum_turn_degrees"),
    ),
}


__all__ = [
    "TEACHERS",
    "TeacherOutput",
    "TeacherSpec",
    "look_at_choices",
    "walk_to_choices",
]
