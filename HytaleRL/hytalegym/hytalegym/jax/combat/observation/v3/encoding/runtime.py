"""Learner-facing arsenal reset and step with injected world tensors."""

from __future__ import annotations

import jax

from hytalegym.jax.combat.arsenal.runtime import (
    reset_arsenal_batch,
    step_arsenal_batch,
)
from hytalegym.jax.combat.arsenal.schema.types import (
    ArsenalRuntimeConfig,
    ArsenalWorldCapabilities,
)
from hytalegym.jax.combat.observation.v1.schema.types import InjectedWorldFeatures
from hytalegym.jax.combat.observation.v3.policy.actions import (
    decode_learner_arsenal_action,
)
from hytalegym.jax.combat.observation.v3.encoding.encoder import (
    encode_learner_observation_v3,
)
from hytalegym.jax.combat.observation.v3.schema.types import (
    LearnerArsenalAction,
    LearnerArsenalTransition,
    LearnerCombatObservationV3,
)
from hytalegym.jax.combat.types import CombatParams


def reset_learner_arsenal_batch(
    keys: jax.Array,
    params: CombatParams,
    world_features: InjectedWorldFeatures,
    world_capabilities: ArsenalWorldCapabilities,
    config: ArsenalRuntimeConfig,
):
    """Reset and encode v3 with already-selected, masked world features."""

    state, _ = reset_arsenal_batch(keys, params, config)
    observation = encode_learner_observation_v3(
        state,
        params,
        world_features,
        world_capabilities,
        config,
    )
    return state, observation


def step_learner_arsenal_batch(
    state,
    observation: LearnerCombatObservationV3,
    action: LearnerArsenalAction,
    keys: jax.Array,
    params: CombatParams,
    current_world_capabilities: ArsenalWorldCapabilities,
    next_world_features: InjectedWorldFeatures,
    next_world_capabilities: ArsenalWorldCapabilities,
    config: ArsenalRuntimeConfig,
) -> LearnerArsenalTransition:
    """Step combat; return door intent without changing any world object."""

    decoded = decode_learner_arsenal_action(
        observation,
        action,
        current_world_capabilities,
    )
    transition = step_arsenal_batch(
        state,
        decoded.low_level_action,
        keys,
        params,
        decoded.commands,
        config,
    )
    next_observation = encode_learner_observation_v3(
        transition.state,
        params,
        next_world_features,
        next_world_capabilities,
        config,
    )
    return LearnerArsenalTransition(
        state=transition.state,
        observation=next_observation,
        reward=transition.reward,
        terminated=transition.terminated,
        truncated=transition.truncated,
        action_valid=decoded.valid,
        door=decoded.door,
        combat_info=transition.combat_info,
        arsenal_info=transition.arsenal_info,
    )


__all__ = [
    "reset_learner_arsenal_batch",
    "step_learner_arsenal_batch",
]
