"""Compiled semantic-action adapter for the flat combat environment."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.env import reset_batch, step_batch
from hytalegym.jax.combat.observation.v1.runtime.actions import decode_learner_actions
from hytalegym.jax.combat.observation.v1.runtime.encoder import (
    combat_scene_from_state,
    encode_learner_observation,
)
from hytalegym.jax.combat.observation.v1.schema.types import (
    InjectedWorldFeatures,
    LearnerActionDecode,
    LearnerCombatObservation,
)
from hytalegym.jax.combat.types import (
    CombatInfo,
    CombatParams,
    CombatState,
)


class LearnerCombatTransition(NamedTuple):
    """One semantic combat decision and its intent-only side effect."""

    state: CombatState
    observation: LearnerCombatObservation
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    info: CombatInfo
    action: LearnerActionDecode


def reset_learner_combat_batch(
    keys: jax.Array,
    params: CombatParams,
    world: InjectedWorldFeatures,
) -> tuple[CombatState, LearnerCombatObservation]:
    """Reset flat combat and encode the injected learner-facing features."""

    state, combat_observation = reset_batch(keys, params)
    scene = combat_scene_from_state(state, params, combat_observation)
    observation = encode_learner_observation(
        state,
        params,
        combat_observation,
        scene,
        world,
    )
    return state, observation


def step_learner_combat_batch(
    state: CombatState,
    observation: LearnerCombatObservation,
    action_ids: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    next_world: InjectedWorldFeatures,
    *,
    maximum_turn_degrees: float = 45.0,
) -> LearnerCombatTransition:
    """Decode, step, and re-encode one semantic combat decision.

    ``next_world`` is already selected for the post-step actor state by its
    owning lane. This function never selects tiles or mutates world/door state.
    An invalid current observation freezes its batch row and truncates it.
    Invalid post-step injected features truncate with zero reward.
    """

    decoded = decode_learner_actions(
        observation,
        action_ids,
        maximum_turn_degrees=maximum_turn_degrees,
    )
    (
        proposed_state,
        combat_observation,
        reward,
        done,
        info,
    ) = step_batch(
        state,
        decoded.low_level_action,
        keys,
        params,
    )
    proposed_scene = combat_scene_from_state(
        proposed_state,
        params,
        combat_observation,
    )
    proposed_observation = encode_learner_observation(
        proposed_state,
        params,
        combat_observation,
        proposed_scene,
        next_world,
    )

    current_valid = observation.valid
    state_result = _select_batch_tree(
        current_valid,
        proposed_state,
        state,
    )
    observation_result = _select_batch_tree(
        current_valid,
        proposed_observation,
        observation,
    )
    info_result = _mask_batch_tree(current_valid, info)
    truncated = ~current_valid | ~proposed_observation.valid
    terminated = done & ~truncated
    reward_result = jnp.where(
        truncated,
        jnp.float32(0.0),
        reward,
    )
    return LearnerCombatTransition(
        state=state_result,
        observation=observation_result,
        reward=reward_result,
        terminated=terminated,
        truncated=truncated,
        info=info_result,
        action=decoded,
    )


def _select_batch_tree(
    mask: jax.Array,
    when_true,
    when_false,
):
    def select(true_value, false_value):
        batch_mask = _broadcast_batch_mask(mask, true_value.ndim)
        return jnp.where(batch_mask, true_value, false_value)

    return jax.tree_util.tree_map(select, when_true, when_false)


def _mask_batch_tree(mask: jax.Array, tree):
    def apply(value):
        batch_mask = _broadcast_batch_mask(mask, value.ndim)
        return jnp.where(batch_mask, value, jnp.zeros_like(value))

    return jax.tree_util.tree_map(apply, tree)


def _broadcast_batch_mask(mask: jax.Array, rank: int) -> jax.Array:
    return jnp.reshape(mask, mask.shape + (1,) * (rank - 1))
