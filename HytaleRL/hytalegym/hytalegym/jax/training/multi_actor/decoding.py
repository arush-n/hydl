"""Policy-factor decoding for actor-major flat shared arenas."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.types import (
    ArsenalEnvironmentState,
    ArsenalRuntimeConfig,
    ArsenalWorldCapabilities,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    decode_arsenal_policy_actions,
)
from hytalegym.jax.combat.observation.v3.policy.actions import (
    decode_learner_arsenal_action,
)
from hytalegym.jax.combat.observation.v3.schema.types import (
    LearnerCombatObservationV3,
)
from hytalegym.jax.combat.types import AGENT_ENTITY

from .assignment import PolicyActorAssignment
from .controls import PolicyActorControls, policy_actor_controls
from .locomotion import EntityLocomotionState
from .views import actor_first_arsenal_view


def decode_policy_actor_factors_flat(
    state: ArsenalEnvironmentState,
    observation: LearnerCombatObservationV3,
    action_factors: jax.Array,
    capabilities: ArsenalWorldCapabilities,
    config: ArsenalRuntimeConfig,
    locomotion: EntityLocomotionState,
    assignment: PolicyActorAssignment,
    *,
    maximum_turn_degrees: float = 45.0,
) -> PolicyActorControls:
    """Decode every active actor through the unchanged learner action path."""

    actor_shape = assignment.actor_index.shape
    factors = jnp.asarray(action_factors)
    expected = actor_shape + (len(ARSENAL_POLICY_ACTION_HEAD_SIZES),)
    if factors.dtype != jnp.int32 or factors.shape != expected:
        raise TypeError(f"action_factors must be int32 with shape {expected}")
    leaves = jax.tree_util.tree_leaves(observation)
    if not leaves or any(
        value.ndim < 2 or value.shape[:2] != actor_shape for value in leaves
    ):
        raise ValueError("observation leaves must begin with [B,P]")

    rows = []
    for slot in range(actor_shape[1]):
        actor = jnp.where(
            assignment.active[:, slot],
            assignment.actor_index[:, slot],
            jnp.int32(AGENT_ENTITY),
        )
        _, _, actor_capabilities = actor_first_arsenal_view(
            state,
            config,
            capabilities,
            locomotion,
            actor,
        )
        actor_observation = jax.tree_util.tree_map(
            lambda value: value[:, slot], observation
        )
        semantic = decode_arsenal_policy_actions(
            factors[:, slot], maximum_turn_degrees=maximum_turn_degrees
        )
        # The world-move override resolves its compass into body-relative
        # channels, so it needs the BODY's commanded heading, not the camera's.
        desired_body_yaw = jnp.take_along_axis(
            locomotion.desired_body_yaw,
            actor[:, None],
            axis=1,
        )[:, 0]
        decoded = decode_learner_arsenal_action(
            actor_observation,
            semantic,
            actor_capabilities,
            maximum_turn_degrees=maximum_turn_degrees,
            desired_body_yaw_degrees=desired_body_yaw,
        )
        rows.append(
            (
                decoded.low_level_action,
                decoded.commands.ability_slot[:, AGENT_ENTITY],
                decoded.commands.defense.guard_held[:, AGENT_ENTITY],
                decoded.commands.defense.dodge_direction[:, AGENT_ENTITY],
                decoded.commands.defense.dodge_corridor_clear[:, AGENT_ENTITY],
                decoded.valid,
            )
        )

    return policy_actor_controls(
        *(jnp.stack([row[index] for row in rows], axis=1) for index in range(6)),
        assignment,
    )


__all__ = ["decode_policy_actor_factors_flat"]
