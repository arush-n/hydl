"""JAX-native translation of the nine transfer-safe melee skills."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.types import (
    ACTION_ATTACK,
    ACTION_BACK,
    ACTION_BODY_YAW_DELTA,
    ACTION_FORWARD,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_SIZE,
    ACTION_YAW_DELTA,
)


SKILL_IDLE = 0
SKILL_FACE_TARGET = 1
SKILL_APPROACH = 2
SKILL_RETREAT = 3
SKILL_STRAFE_LEFT = 4
SKILL_STRAFE_RIGHT = 5
SKILL_ATTACK = 6
SKILL_APPROACH_ATTACK = 7
SKILL_RETREAT_ATTACK = 8
SKILL_COUNT = 9

TARGET_FORWARD_OBSERVATION = 6
TARGET_RIGHT_OBSERVATION = 7
TARGET_VISIBLE_OBSERVATION = 10
AGENT_ATTACK_EXECUTING_OBSERVATION = 11


def skills_to_actions(
    observation: jax.Array,
    skill_ids: jax.Array,
    *,
    maximum_turn_degrees: float = 45.0,
) -> jax.Array:
    """Translate discrete skills to the fixed eight-value combat action.

    The turn calculation uses only policy-visible compact observation values.
    It is therefore identical at JAX training time and when the same weights
    consume `ActiveCombatObsWrapper` output from native Hytale.
    """

    skill_ids = jnp.asarray(skill_ids, dtype=jnp.int32)
    target_forward = observation[:, TARGET_FORWARD_OBSERVATION]
    target_right = observation[:, TARGET_RIGHT_OBSERVATION]
    target_visible = (
        observation[:, TARGET_VISIBLE_OBSERVATION] > jnp.float32(0.5)
    )
    turn_delta = jnp.rad2deg(jnp.arctan2(
        -target_right,
        target_forward,
    ))
    turn_delta = jnp.clip(
        turn_delta,
        -jnp.float32(maximum_turn_degrees),
        jnp.float32(maximum_turn_degrees),
    )
    should_face = (skill_ids != SKILL_IDLE) & target_visible
    turn_delta = jnp.where(should_face, turn_delta, jnp.float32(0.0))

    actions = jnp.zeros(
        (observation.shape[0], ACTION_SIZE),
        dtype=jnp.float32,
    )
    actions = actions.at[:, ACTION_FORWARD].set(
        (
            (skill_ids == SKILL_APPROACH)
            | (skill_ids == SKILL_APPROACH_ATTACK)
        ).astype(jnp.float32)
    )
    actions = actions.at[:, ACTION_BACK].set(
        (
            (skill_ids == SKILL_RETREAT)
            | (skill_ids == SKILL_RETREAT_ATTACK)
        ).astype(jnp.float32)
    )
    actions = actions.at[:, ACTION_LEFT].set(
        (skill_ids == SKILL_STRAFE_LEFT).astype(jnp.float32)
    )
    actions = actions.at[:, ACTION_RIGHT].set(
        (skill_ids == SKILL_STRAFE_RIGHT).astype(jnp.float32)
    )
    actions = actions.at[:, ACTION_ATTACK].set(
        (
            (skill_ids == SKILL_ATTACK)
            | (skill_ids == SKILL_APPROACH_ATTACK)
            | (skill_ids == SKILL_RETREAT_ATTACK)
        ).astype(jnp.float32)
    )
    # A scripted "face the target" turns the whole actor, not just its neck, so
    # the same delta drives both seams. Body and head are separate channels now;
    # writing only the head here would leave a skill-driven actor looking at its
    # target while its chest -- and therefore its travel -- stayed put.
    actions = actions.at[:, ACTION_YAW_DELTA].set(turn_delta)
    return actions.at[:, ACTION_BODY_YAW_DELTA].set(turn_delta)


def legal_skill_mask(observation: jax.Array) -> jax.Array:
    """Return the observation-derived legal mask for all nine skills.

    Locomotion remains legal without a visible target. Attack-bearing skills
    are masked while the authored attack interaction/cooldown is executing or
    when no live target is visible. IDLE is always legal, so no row is empty.
    """

    target_visible = (
        observation[:, TARGET_VISIBLE_OBSERVATION] > jnp.float32(0.5)
    )
    attack_ready = (
        observation[:, AGENT_ATTACK_EXECUTING_OBSERVATION]
        < jnp.float32(0.5)
    )
    attack_legal = target_visible & attack_ready
    mask = jnp.ones(
        (observation.shape[0], SKILL_COUNT),
        dtype=jnp.bool_,
    )
    mask = mask.at[:, SKILL_ATTACK].set(attack_legal)
    mask = mask.at[:, SKILL_APPROACH_ATTACK].set(attack_legal)
    return mask.at[:, SKILL_RETREAT_ATTACK].set(attack_legal)
