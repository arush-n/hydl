"""Learner-facing v2 adapter for the isolated effects runtime."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.effects.factory import empty_effects_state
from hytalegym.jax.combat.effects.runtime.engine import (
    step_batch_melee_loadout_effects,
)
from hytalegym.jax.combat.effects.schema.types import (
    CombatEffectCommands,
    CombatEffectsEnvironmentState,
    CombatEffectsInfo,
)
from hytalegym.jax.combat.loadout.schema.types import MeleeLoadoutBatch
from hytalegym.jax.combat.observation.v1.runtime.encoder import (
    encode_learner_observation,
)
from hytalegym.jax.combat.observation.v1.runtime.transition import (
    _mask_batch_tree,
    _select_batch_tree,
)
from hytalegym.jax.combat.observation.v2.runtime import (
    _batch_tree_equal,
    reset_learner_combat_batch_v2,
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
from hytalegym.jax.combat.types import CombatInfo, CombatParams


class LearnerCombatEffectsTransitionV2(NamedTuple):
    """One semantic learner transition with external effect commands."""

    state: CombatEffectsEnvironmentState
    observation: LearnerCombatObservationV2
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    combat_info: CombatInfo
    effects_info: CombatEffectsInfo
    action: LearnerActionDecode


def reset_learner_combat_effects_batch_v2(
    keys: jax.Array,
    params: CombatParams,
    world: InjectedWorldFeatures,
    loadout: MeleeLoadoutBatch,
) -> tuple[CombatEffectsEnvironmentState, LearnerCombatObservationV2]:
    """Reset v2 learner combat with empty projectile/hazard state."""

    combat, observation = reset_learner_combat_batch_v2(
        keys,
        params,
        world,
        loadout,
    )
    return (
        CombatEffectsEnvironmentState(
            combat,
            empty_effects_state(combat.position.shape[0]),
        ),
        observation,
    )


def step_learner_combat_effects_batch_v2(
    state: CombatEffectsEnvironmentState,
    observation: LearnerCombatObservationV2,
    action_ids: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    next_world: InjectedWorldFeatures,
    loadout: MeleeLoadoutBatch,
    commands: CombatEffectCommands,
    *,
    maximum_turn_degrees: float = 45.0,
) -> LearnerCombatEffectsTransitionV2:
    """Step semantic combat; world tensors and effect events stay injected."""

    decoded = decode_learner_actions_v2(
        observation,
        action_ids,
        maximum_turn_degrees=maximum_turn_degrees,
    )
    transition = step_batch_melee_loadout_effects(
        state,
        decoded.low_level_action,
        keys,
        params,
        commands,
        loadout,
    )
    proposed_base = encode_learner_observation(
        transition.state.combat,
        params,
        transition.observation,
        transition.scene,
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
    current_valid = (
        observation.valid
        & profile_matches
        & (state.effects.failure_bits == jnp.uint32(0))
    )
    state_result = _select_batch_tree(
        current_valid,
        transition.state,
        state,
    )
    observation_result = _select_batch_tree(
        current_valid,
        proposed_observation,
        observation,
    )
    next_valid = proposed_observation.valid & transition.effects_info.valid
    truncated = ~current_valid | ~next_valid
    terminated = transition.done & ~truncated
    return LearnerCombatEffectsTransitionV2(
        state=state_result,
        observation=observation_result,
        reward=jnp.where(truncated, jnp.float32(0.0), transition.reward),
        terminated=terminated,
        truncated=truncated,
        combat_info=_mask_batch_tree(
            current_valid & transition.effects_info.valid,
            transition.combat_info,
        ),
        effects_info=transition.effects_info,
        action=decoded,
    )
