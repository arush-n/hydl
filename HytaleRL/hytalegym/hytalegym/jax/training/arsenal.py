"""PPO compatibility adapter for the neutral structured Arsenal environment.

New agents should import environment/runtime primitives from
hytalegym.jax.combat.arsenal.environment. This module retains the historical
import surface and adds only the dense recurrent-PPO adapter.
"""

from __future__ import annotations

# ruff: noqa: F401,F403,F405

from typing import Any

import jax

from hytalegym.jax.combat.arsenal import environment as _environment
from hytalegym.jax.combat.arsenal.environment import *  # noqa: F403
from hytalegym.jax.combat.arsenal.programs.event_storage import ability_event_bank
from hytalegym.jax.combat.arsenal.schema.types import ArsenalRuntimeConfig
from hytalegym.jax.combat.arsenal.training_inventory import (
    training_inventory_from_config,
)
from hytalegym.jax.combat.environment import EnvironmentSpec
from hytalegym.jax.combat.observation.v3.policy.candidates.encoder import (
    RecipeCandidateEncoderParams,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_SIZE,
    ARSENAL_POLICY_SCHEMA,
    ARSENAL_STANDARD_ROOT_DISTRIBUTION,
    arsenal_policy_observation,
    arsenal_policy_observation_size,
)
from hytalegym.jax.combat.observation.v3.policy.actions import (
    learner_arsenal_action_context,
)
from hytalegym.jax.combat.observation.v3.tokens.world import (
    WorldGeometryPolicyConfig,
    normalize_world_geometry_policy_config,
)
from hytalegym.jax.combat.opponents.runtime.policy import (
    first_legal_opponent_ability_slots,
)
from hytalegym.jax.combat.types import CombatParams
from hytalegym.jax.training.ppo import (
    PPOEnvironment,
    combat_episode_outcome,
)
from hytalegym.jax.training.types import PPOConfig
from hytalegym.jax.world import GeometryProvider


# Historical experimental helpers imported these private names directly.
_ability_event_bank = ability_event_bank
_dodge_corridor_displacements = _environment._dodge_corridor_displacements
_force_corridor_clearance = _environment._force_corridor_clearance
_motion_delta = _environment._motion_delta
_swept_clearance = _environment._swept_clearance
_compiled_policy_observation = jax.jit(
    arsenal_policy_observation,
    inline=False,
)


def _arsenal_ppo_environment_spec(
    world_geometry_config: WorldGeometryPolicyConfig,
) -> EnvironmentSpec:
    return EnvironmentSpec(
        observation_schema=ARSENAL_POLICY_SCHEMA,
        action_schema="hytalerl_reference_factored_arsenal_action_v2",
        action_components=ARSENAL_ENVIRONMENT_SPEC.action_components,
        observation_size=arsenal_policy_observation_size(world_geometry_config),
        action_size=ARSENAL_POLICY_ACTION_SIZE,
    )


def arsenal_ppo_config(
    *,
    world_geometry_config: WorldGeometryPolicyConfig | None = None,
    **overrides: Any,
) -> PPOConfig:
    """Build PPO with the complete current Arsenal policy surface."""

    protected = {
        "observation_size",
        "action_size",
        "action_head_sizes",
        "action_transport",
        "action_distribution",
    }.intersection(overrides)
    if protected:
        names = ", ".join(sorted(protected))
        raise ValueError(
            f"{names} are derived from ARSENAL_ENVIRONMENT_SPEC "
            "and cannot be overridden"
        )
    token_config = normalize_world_geometry_policy_config(world_geometry_config)
    return PPOConfig.from_environment_spec(
        _arsenal_ppo_environment_spec(token_config),
        action_distribution=ARSENAL_STANDARD_ROOT_DISTRIBUTION,
        **overrides,
    )


def _ppo_state(carry, observation):
    return ArsenalPPOEnvironmentState(
        runtime=carry.runtime,
        learner_observation=observation,
        action_surface=carry.action_surface,
        action_surface_runtime=carry.action_surface_runtime,
        light_tokens=carry.light_tokens,
        inventory_tokens=carry.inventory_tokens,
        world_capabilities=carry.world_capabilities,
        action_mask=carry.action_mask,
    )


def _environment_carry(state):
    return ArsenalEnvironmentCarry(
        runtime=state.runtime,
        action_context=learner_arsenal_action_context(state.learner_observation),
        action_surface=state.action_surface,
        action_surface_runtime=state.action_surface_runtime,
        light_tokens=state.light_tokens,
        inventory_tokens=state.inventory_tokens,
        world_capabilities=state.world_capabilities,
        action_mask=state.action_mask,
    )


def make_arsenal_ppo_environment(
    params: CombatParams,
    config: ArsenalRuntimeConfig,
    *,
    maximum_turn_degrees: float = REFERENCE_SKILL_MAXIMUM_TURN_DEGREES,
    geometry_provider: GeometryProvider | None = None,
    target_navigation_provider: TargetNavigationProvider | None = None,
    world_capability_provider: ArsenalWorldCapabilityProvider | None = None,
    world_feature_provider: ArsenalWorldFeatureProvider | None = None,
    world_token_provider: ArsenalWorldTokenProvider | None = None,
    world_light_token_provider: ArsenalWorldLightTokenProvider | None = None,
    world_geometry_config: WorldGeometryPolicyConfig | None = None,
    reset_provider: ArsenalResetProvider | None = None,
    inventory_reset_provider: ArsenalInventoryResetProvider | None = None,
    action_surface_provider: ArsenalActionSurfaceProvider | None = None,
    action_surface_executor: ArsenalActionSurfaceExecutor | None = None,
    action_surface_runtime_initializer: (
        ArsenalActionSurfaceRuntimeInitializer | None
    ) = None,
    world_runtime_provider: ArsenalWorldRuntimeProvider | None = None,
    explosion_candidate_provider: (
        ArsenalRuntimeExplosionCandidateProvider | None
    ) = None,
    recipe_candidate_encoder_params: RecipeCandidateEncoderParams | None = None,
    opponent_ability_provider: ArsenalOpponentAbilityProvider = (
        first_legal_opponent_ability_slots
    ),
) -> PPOEnvironment:
    """Adapt Arsenal to PPO with authored positive consumables by default.

    The structured environment keeps its explicit fail-closed provider seams.
    This training entry point supplies the loadout's authored positive initial
    consumables and the deterministic first-legal opponent controller when a
    caller does not replace those scenario inputs. Diagnostics that require an
    inert opponent must pass ``inert_opponent_ability_slots`` explicitly.
    """

    if opponent_ability_provider is None:
        raise ValueError(
            "production PPO requires an explicit opponent ability provider; "
            "pass inert_opponent_ability_slots for an inert diagnostic"
        )

    resolved_inventory_provider = inventory_reset_provider
    if resolved_inventory_provider is None:
        initial_inventory = training_inventory_from_config(config)

        def resolved_inventory_provider(keys, runtime_config):
            del keys, runtime_config
            return initial_inventory

    environment = make_arsenal_environment(
        params,
        config,
        maximum_turn_degrees=maximum_turn_degrees,
        geometry_provider=geometry_provider,
        target_navigation_provider=target_navigation_provider,
        world_capability_provider=world_capability_provider,
        world_feature_provider=world_feature_provider,
        world_token_provider=world_token_provider,
        world_light_token_provider=world_light_token_provider,
        world_geometry_config=world_geometry_config,
        reset_provider=reset_provider,
        inventory_reset_provider=resolved_inventory_provider,
        action_surface_provider=action_surface_provider,
        action_surface_executor=action_surface_executor,
        action_surface_runtime_initializer=(action_surface_runtime_initializer),
        world_runtime_provider=world_runtime_provider,
        explosion_candidate_provider=explosion_candidate_provider,
        recipe_candidate_encoder_params=recipe_candidate_encoder_params,
        opponent_ability_provider=opponent_ability_provider,
    )
    token_config = normalize_world_geometry_policy_config(world_geometry_config)

    def reset(keys):
        carry, observation = environment.reset(keys)
        state = _ppo_state(carry, observation)
        return (
            state,
            _compiled_policy_observation(
                observation,
                state.action_surface.block_candidates,
                state.action_surface.recipe_encoding,
                inventory_tokens=state.inventory_tokens,
                light_tokens=state.light_tokens,
            ),
            state.action_mask,
        )

    def step_detailed(state, _policy_observation, action_ids, keys):
        (
            next_state,
            next_observation,
            reward,
            done,
            info,
        ) = environment.step_factors(
            _environment_carry(state),
            action_ids,
            keys,
        )
        next_state = _ppo_state(next_state, next_observation)
        return (
            next_state,
            _compiled_policy_observation(
                next_observation,
                next_state.action_surface.block_candidates,
                next_state.action_surface.recipe_encoding,
                inventory_tokens=next_state.inventory_tokens,
                light_tokens=next_state.light_tokens,
            ),
            reward,
            done,
            next_state.action_mask,
            info,
        )

    def step(state, policy_observation, action_ids, keys):
        return step_detailed(
            state,
            policy_observation,
            action_ids,
            keys,
        )[:5]

    def episode_outcome(state, done):
        runtime = state.runtime
        valid = (
            runtime.arsenal.failure_bits
            | runtime.mechanics.failure_bits
            | runtime.inventory.failure_bits
        ) == 0
        return combat_episode_outcome(runtime.combat, done, valid=valid)

    return PPOEnvironment(
        reset=reset,
        step=step,
        spec=_arsenal_ppo_environment_spec(token_config),
        episode_outcome=episode_outcome,
        step_detailed=step_detailed,
    )


def __getattr__(name: str):
    """Delegate historical private attributes to the neutral implementation."""

    return getattr(_environment, name)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(dir(_environment)))


__all__ = [
    *_environment.__all__,
    "arsenal_ppo_config",
    "make_arsenal_ppo_environment",
]
