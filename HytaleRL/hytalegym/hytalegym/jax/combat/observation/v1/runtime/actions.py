"""Learner action decoding, including intent-only door requests."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from hytalegym.jax.combat.observation.v1.schema.contract import (
    ACTION_DOOR_OPEN,
    DOOR_INTENT_COUNT,
    DOOR_INTENT_NONE,
    LEARNER_ACTION_COUNT,
)
from hytalegym.jax.combat.observation.v1.schema.types import (
    DoorIntentRequest,
    LearnerActionDecode,
    LearnerCombatObservation,
)
from hytalegym.jax.combat.skills import (
    SKILL_COUNT,
    SKILL_IDLE,
    skills_to_actions,
)


class LearnerActionContext(NamedTuple):
    """Minimal observation fields required to decode one learner action."""

    combat_f32: jnp.ndarray
    interaction_object_id: jnp.ndarray
    interaction_is_door: jnp.ndarray
    interaction_door_intent_mask: jnp.ndarray
    interaction_mask: jnp.ndarray
    action_mask: jnp.ndarray


def learner_action_context(
    observation: LearnerCombatObservation,
) -> LearnerActionContext:
    """Project the exact action-decoding subset of an observation."""

    return LearnerActionContext(
        combat_f32=observation.combat_f32,
        interaction_object_id=observation.interaction_object_id,
        interaction_is_door=observation.interaction_is_door,
        interaction_door_intent_mask=(
            observation.interaction_door_intent_mask
        ),
        interaction_mask=observation.interaction_mask,
        action_mask=observation.action_mask,
    )


def decode_learner_actions(
    observation: LearnerCombatObservation,
    action_ids: jnp.ndarray,
    *,
    maximum_turn_degrees: float = 45.0,
) -> LearnerActionDecode:
    """Decode combat skills and door intents without changing door state."""

    return decode_learner_action_context(
        learner_action_context(observation),
        action_ids,
        maximum_turn_degrees=maximum_turn_degrees,
    )


def decode_learner_action_context(
    context: LearnerActionContext,
    action_ids: jnp.ndarray,
    *,
    maximum_turn_degrees: float = 45.0,
) -> LearnerActionDecode:
    """Decode from the compact context carried between environment steps."""

    action_ids = jnp.asarray(action_ids, dtype=jnp.int32)
    in_range = (action_ids >= 0) & (action_ids < LEARNER_ACTION_COUNT)
    safe_action_ids = jnp.clip(action_ids, 0, LEARNER_ACTION_COUNT - 1)
    action_legal = (
        in_range
        & jnp.take_along_axis(
            context.action_mask,
            safe_action_ids[:, None],
            axis=1,
        )[:, 0]
    )

    is_skill = action_ids < SKILL_COUNT
    skill_ids = jnp.where(
        is_skill & action_legal,
        action_ids,
        SKILL_IDLE,
    )
    low_level_action = skills_to_actions(
        context.combat_f32,
        skill_ids,
        maximum_turn_degrees=maximum_turn_degrees,
    )

    is_door = (action_ids >= ACTION_DOOR_OPEN) & (
        action_ids < ACTION_DOOR_OPEN + DOOR_INTENT_COUNT
    )
    intent_index = jnp.clip(
        action_ids - ACTION_DOOR_OPEN,
        0,
        DOOR_INTENT_COUNT - 1,
    )
    selected_intent_mask = jnp.take_along_axis(
        context.interaction_door_intent_mask,
        intent_index[:, None, None],
        axis=2,
    )[:, :, 0]
    candidates = (
        context.interaction_mask
        & context.interaction_is_door
        & selected_intent_mask
    )
    has_candidate = jnp.any(candidates, axis=1)
    candidate_slot = jnp.argmax(candidates, axis=1).astype(jnp.int32)
    gathered_object_id = jnp.take_along_axis(
        context.interaction_object_id,
        candidate_slot[:, None],
        axis=1,
    )[:, 0]
    accepted = is_door & action_legal & has_candidate
    requested_intent = jnp.where(
        is_door,
        intent_index + 1,
        DOOR_INTENT_NONE,
    ).astype(jnp.int32)
    door = DoorIntentRequest(
        requested=is_door,
        accepted=accepted,
        intent=requested_intent,
        interaction_slot=jnp.where(
            accepted,
            candidate_slot,
            -1,
        ).astype(jnp.int32),
        object_id=jnp.where(
            accepted,
            gathered_object_id,
            -1,
        ).astype(jnp.int32),
    )
    return LearnerActionDecode(
        low_level_action=low_level_action,
        action_legal=action_legal,
        door=door,
    )
