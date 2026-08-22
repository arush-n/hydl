"""Compiled semantic combat runtime for learner observation v2."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.env import step_batch_melee_loadout
from hytalegym.jax.combat.loadout import MeleeLoadoutBatch
from hytalegym.jax.combat.observation.v1.runtime.encoder import (
    combat_scene_from_state,
    encode_learner_observation,
)
from hytalegym.jax.combat.observation.v1.runtime.transition import (
    _mask_batch_tree,
    _select_batch_tree,
    reset_learner_combat_batch,
)
from hytalegym.jax.combat.observation.v1.schema.types import (
    InjectedWorldFeatures,
    LearnerActionDecode,
)
from hytalegym.jax.combat.observation.v2 import (
    LearnerCombatObservationV2,
    decode_learner_actions_v2,
    extend_learner_observation_v2,
)
from hytalegym.jax.combat.types import (
    CombatInfo,
    CombatParams,
    CombatState,
)


class LearnerCombatTransitionV2(NamedTuple):
    """One loadout-aware semantic transition."""

    state: CombatState
    observation: LearnerCombatObservationV2
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    info: CombatInfo
    action: LearnerActionDecode


def reset_learner_combat_batch_v2(
    keys: jax.Array,
    params: CombatParams,
    world: InjectedWorldFeatures,
    loadout: MeleeLoadoutBatch,
) -> tuple[CombatState, LearnerCombatObservationV2]:
    """Reset flat combat with explicit episode-pinned melee equipment."""

    state, base = reset_learner_combat_batch(keys, params, world)
    return state, extend_learner_observation_v2(base, params, loadout)


def step_learner_combat_batch_v2(
    state: CombatState,
    observation: LearnerCombatObservationV2,
    action_ids: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    next_world: InjectedWorldFeatures,
    loadout: MeleeLoadoutBatch,
    *,
    maximum_turn_degrees: float = 45.0,
) -> LearnerCombatTransitionV2:
    """Step one loadout-aware decision without world or equipment mutation."""

    decoded = decode_learner_actions_v2(
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
    ) = step_batch_melee_loadout(
        state,
        decoded.low_level_action,
        keys,
        params,
        loadout,
    )
    scene = combat_scene_from_state(
        proposed_state,
        params,
        combat_observation,
    )
    proposed_base = encode_learner_observation(
        proposed_state,
        params,
        combat_observation,
        scene,
        next_world,
    )
    proposed_observation = extend_learner_observation_v2(
        proposed_base,
        params,
        loadout,
    )

    profile_matches = _batch_tree_equal(
        observation.loadout,
        proposed_observation.loadout,
    )
    current_valid = observation.valid & profile_matches
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
    return LearnerCombatTransitionV2(
        state=state_result,
        observation=observation_result,
        reward=reward_result,
        terminated=terminated,
        truncated=truncated,
        info=info_result,
        action=decoded,
    )


def _batch_tree_equal(left, right) -> jax.Array:
    """Compare every semantic leaf independently for each batch row."""

    left_leaves = jax.tree_util.tree_leaves(left)
    right_leaves = jax.tree_util.tree_leaves(right)
    if len(left_leaves) != len(right_leaves):
        raise ValueError("loadout observation tree structures do not match")
    batch = left_leaves[0].shape[0]
    equal = jnp.ones((batch,), dtype=jnp.bool_)
    for left_leaf, right_leaf in zip(
        left_leaves,
        right_leaves,
        strict=True,
    ):
        leaf_equal = jnp.all(
            left_leaf.reshape((batch, -1)) == right_leaf.reshape((batch, -1)),
            axis=1,
        )
        equal &= leaf_equal
    return equal
