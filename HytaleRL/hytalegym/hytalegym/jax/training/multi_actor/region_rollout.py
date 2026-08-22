"""Recurrent actor-major rollouts over one persistent exact Region runtime."""

from __future__ import annotations

from hytalegym.jax.combat import TargetNavigationProvider
from hytalegym.jax.combat.arsenal.environment import (
    ArsenalActionSurfaceExecutor,
    ArsenalActionSurfaceProvider,
    ArsenalActionSurfaceRuntime,
    ArsenalActionSurfaceRuntimeInitializer,
    ArsenalRuntimeExplosionCandidateProvider,
    ArsenalWorldRuntimeProvider,
    ArsenalWorldRuntimeViews,
)
from hytalegym.jax.combat.arsenal.schema.types import ArsenalRuntimeConfig
from hytalegym.jax.combat.block_interactions import empty_block_interaction_state
from hytalegym.jax.combat.types import CombatParams

from .assignment import PolicyActorAssignment
from .region import (
    ActorMajorWorldRuntimeProvider,
    decode_policy_actor_factors_region,
    observe_policy_actors_region,
    step_multi_actor_region_arena,
)
from .rollout import (
    ActionMaskTransform,
    ActorObservationTransform,
    ActorTransitionTransform,
    ArenaResetProvider,
    ObservationTelemetryTransform,
    _make_actor_major_rollout_collector,
)
from .runtime import initialize_multi_actor_arena_state


def make_multi_actor_region_rollout_collector(
    params: CombatParams,
    config: ArsenalRuntimeConfig,
    assignment: PolicyActorAssignment,
    *,
    reset_provider: ArenaResetProvider,
    world_runtime_provider: ArsenalWorldRuntimeProvider,
    actor_world_runtime_provider: ActorMajorWorldRuntimeProvider,
    action_surface_provider: ArsenalActionSurfaceProvider,
    action_surface_executor: ArsenalActionSurfaceExecutor | None,
    action_surface_runtime_initializer: ArsenalActionSurfaceRuntimeInitializer,
    rollout_steps: int,
    action_mask_transform: ActionMaskTransform | None = None,
    observation_transform: ActorObservationTransform | None = None,
    transition_transform: ActorTransitionTransform | None = None,
    carry_recurrent_state: bool = True,
    record_policy_inputs: bool = True,
    observation_telemetry_transform: ObservationTelemetryTransform | None = None,
    record_actions: bool = True,
    maximum_turn_degrees: float = 45.0,
    scripted_actor_slots: tuple[int, ...] = (),
    record_policy_slots: tuple[int, ...] | None = None,
    target_navigation_provider: TargetNavigationProvider | None = None,
    explosion_candidate_provider: (
        ArsenalRuntimeExplosionCandidateProvider | None
    ) = None,
):
    """Build an end-to-end Region collector without relaxing any World gate."""

    for name, value in (
        ("reset_provider", reset_provider),
        ("world_runtime_provider", world_runtime_provider),
        ("actor_world_runtime_provider", actor_world_runtime_provider),
        ("action_surface_provider", action_surface_provider),
        (
            "action_surface_runtime_initializer",
            action_surface_runtime_initializer,
        ),
    ):
        if not callable(value):
            raise TypeError(f"{name} must be callable")
    if action_surface_executor is not None and not callable(action_surface_executor):
        raise TypeError("action_surface_executor must be callable or None")
    batch = assignment.actor_index.shape[0]
    entity_count = config.loadout.weapon_id.shape[1]

    def project(arena):
        action_runtime = arena.action_surface_runtime
        if action_runtime is None:
            raise ValueError("Region arena lost its action-surface runtime")
        global_views = world_runtime_provider(
            arena.arsenal,
            action_runtime.world,
            params,
        )
        if not isinstance(global_views, ArsenalWorldRuntimeViews):
            raise TypeError(
                "world_runtime_provider must return ArsenalWorldRuntimeViews"
            )
        observation = observe_policy_actors_region(
            arena.arsenal,
            params,
            config,
            arena.locomotion,
            assignment,
            action_runtime,
            actor_world_runtime_provider=actor_world_runtime_provider,
            action_surface_provider=action_surface_provider,
        )
        return observation, global_views

    def decode(arena, observation, _global_views, factors):
        action_runtime = arena.action_surface_runtime
        if action_runtime is None:
            raise ValueError("Region arena lost its action-surface runtime")
        return decode_policy_actor_factors_region(
            arena.arsenal,
            observation,
            factors,
            arena.locomotion,
            assignment,
            action_runtime,
            maximum_turn_degrees=maximum_turn_degrees,
        )

    def advance(arena, actions, observation, global_views, environment_keys):
        runtime_explosion_provider = None
        if explosion_candidate_provider is not None:

            def runtime_explosion_provider(
                combat_state,
                impact,
                block_damage_radius,
                entity_damage_radius,
                damage_blocks,
                query_mask,
                candidate_mask,
            ):
                return explosion_candidate_provider(
                    arena.action_surface_runtime.world,
                    combat_state,
                    params,
                    impact,
                    block_damage_radius,
                    entity_damage_radius,
                    damage_blocks,
                    query_mask,
                    candidate_mask,
                )

        return step_multi_actor_region_arena(
            arena,
            actions,
            observation,
            assignment,
            environment_keys,
            params,
            config,
            global_views.capabilities,
            action_surface_executor,
            target_navigation_provider=target_navigation_provider,
            explosion_candidate_provider=runtime_explosion_provider,
        )

    def reset(reset_keys):
        state = reset_provider(reset_keys)
        runtime = ArsenalActionSurfaceRuntime(
            block_interactions=empty_block_interaction_state(
                batch,
                entity_count=entity_count,
            ),
            world=action_surface_runtime_initializer(
                reset_keys,
                state,
                params,
                config,
            ),
        )
        return initialize_multi_actor_arena_state(
            state,
            params,
            runtime,
        )

    return _make_actor_major_rollout_collector(
        assignment,
        rollout_steps=rollout_steps,
        project=project,
        decode=decode,
        advance=advance,
        reset=reset,
        action_mask_transform=action_mask_transform,
        observation_transform=observation_transform,
        transition_transform=transition_transform,
        carry_recurrent_state=carry_recurrent_state,
        record_policy_inputs=record_policy_inputs,
        observation_telemetry_transform=observation_telemetry_transform,
        record_actions=record_actions,
        scripted_actor_slots=scripted_actor_slots,
        record_policy_slots=record_policy_slots,
    )


__all__ = ["make_multi_actor_region_rollout_collector"]
