"""Actor-first, ID-remapped simulator views for legal policy projection."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.types import (
    ArsenalEnvironmentState,
    ArsenalRuntimeConfig,
    ArsenalWorldCapabilities,
)
from hytalegym.jax.combat.block_interactions import BlockInteractionState
from hytalegym.jax.combat.types import AGENT_ENTITY, CombatState

from .locomotion import EntityLocomotionState


def actor_first_arsenal_view(
    state: ArsenalEnvironmentState,
    config: ArsenalRuntimeConfig,
    capabilities: ArsenalWorldCapabilities,
    locomotion: EntityLocomotionState,
    actor_index: jax.Array,
) -> tuple[ArsenalEnvironmentState, ArsenalRuntimeConfig, ArsenalWorldCapabilities]:
    """Move each batch row's actor to entity zero and remap entity IDs.

    The permutation is a view for the unchanged learner-v3 projector. It does
    not copy entity-zero evidence into another actor row. Actor-private Walk
    memory comes from ``locomotion``; entity-shaped mechanics and loadouts are
    permuted from their actual rows.
    """

    safe_actor, permutation = actor_first_entity_permutation(
        state,
        actor_index,
    )
    entity_count = state.combat.health.shape[1]
    inverse = _inverse_permutation(permutation)

    combat = _combat_view(state.combat, locomotion, safe_actor, permutation, inverse)
    mechanics = jax.tree_util.tree_map(
        lambda value: _permute_if_entity_axis(value, permutation, entity_count),
        state.mechanics,
    )
    mechanics = mechanics._replace(
        statuses=mechanics.statuses._replace(
            source_entity_id=_remap_entity_ids(
                mechanics.statuses.source_entity_id,
                inverse,
            )
        )
    )
    arsenal = state.arsenal._replace(
        active_ability_slot=_permute_entity_axis(
            state.arsenal.active_ability_slot, permutation
        ),
        ability_elapsed_seconds=_permute_entity_axis(
            state.arsenal.ability_elapsed_seconds, permutation
        ),
        ability_scheduler_tick=_permute_entity_axis(
            state.arsenal.ability_scheduler_tick, permutation
        ),
        ability_scheduler_clock_seconds=_permute_entity_axis(
            state.arsenal.ability_scheduler_clock_seconds, permutation
        ),
        ability_selector_hit_bits=_permute_entity_axis(
            state.arsenal.ability_selector_hit_bits, permutation
        ),
        ability_cooldown_seconds=_permute_entity_axis(
            state.arsenal.ability_cooldown_seconds, permutation
        ),
        ability_charge_count=_permute_entity_axis(
            state.arsenal.ability_charge_count, permutation
        ),
        ability_charge_timer_seconds=_permute_entity_axis(
            state.arsenal.ability_charge_timer_seconds, permutation
        ),
        projectiles=state.arsenal.projectiles._replace(
            owner_entity_id=_remap_entity_ids(
                state.arsenal.projectiles.owner_entity_id,
                inverse,
            )
        ),
        areas=state.arsenal.areas._replace(
            owner_entity_id=_remap_entity_ids(
                state.arsenal.areas.owner_entity_id,
                inverse,
            )
        ),
    )
    inventory = state.inventory._replace(
        item_id=_permute_entity_axis(state.inventory.item_id, permutation),
        quantity=_permute_entity_axis(state.inventory.quantity, permutation),
        durability=_permute_entity_axis(state.inventory.durability, permutation),
        max_durability=_permute_entity_axis(
            state.inventory.max_durability, permutation
        ),
        metadata_hash=_permute_entity_axis(
            state.inventory.metadata_hash, permutation
        ),
        active_hotbar_slot=_permute_entity_axis(
            state.inventory.active_hotbar_slot, permutation
        ),
        active_utility_slot=_permute_entity_axis(
            state.inventory.active_utility_slot, permutation
        ),
        active_tools_slot=_permute_entity_axis(
            state.inventory.active_tools_slot, permutation
        ),
    )
    opponent_memory = jax.tree_util.tree_map(
        lambda value: _permute_if_entity_axis(value, permutation, entity_count),
        state.opponent_memory,
    )
    view_state = ArsenalEnvironmentState(
        combat=combat,
        mechanics=mechanics,
        arsenal=arsenal,
        inventory=inventory,
        opponent_memory=opponent_memory,
    )
    view_config = config._replace(
        mechanics_rules=jax.tree_util.tree_map(
            lambda value: _permute_if_entity_axis(
                value, permutation, entity_count
            ),
            config.mechanics_rules,
        ),
        loadout=jax.tree_util.tree_map(
            lambda value: _permute_if_entity_axis(
                value, permutation, entity_count
            ),
            config.loadout,
        ),
        targeting_rules=jax.tree_util.tree_map(
            lambda value: _permute_if_entity_axis(
                value, permutation, entity_count
            ),
            config.targeting_rules,
        ),
    )
    view_capabilities = _capability_view(
        capabilities,
        safe_actor,
        permutation,
    )
    return view_state, view_config, view_capabilities


def actor_first_entity_permutation(
    state: ArsenalEnvironmentState,
    actor_index: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return validated actor IDs and the exact actor/target-first ordering."""

    batch, entity_count = state.combat.health.shape
    actor = jnp.asarray(actor_index, dtype=jnp.int32)
    if actor.shape != (batch,):
        raise ValueError(f"actor_index must have shape {(batch,)}")
    if entity_count < 2:
        raise ValueError("an actor view requires at least two entities")
    valid_actor = (actor >= 0) & (actor < entity_count)
    safe_actor = jnp.where(valid_actor, actor, jnp.int32(AGENT_ENTITY))
    engagement_target = _gather_entity(
        state.combat.engagement_target_id,
        safe_actor,
    )
    fallback_target = jnp.where(
        safe_actor == jnp.int32(0),
        jnp.int32(1),
        jnp.int32(0),
    )
    target_valid = (
        (engagement_target >= 0)
        & (engagement_target < entity_count)
        & (engagement_target != safe_actor)
    )
    safe_target = jnp.where(
        target_valid,
        engagement_target,
        fallback_target,
    )
    return safe_actor, _actor_first_permutation(
        safe_actor,
        safe_target,
        entity_count,
    )


def actor_first_block_interaction_view(
    state: ArsenalEnvironmentState,
    interactions: BlockInteractionState,
    actor_index: jax.Array,
) -> BlockInteractionState:
    """Permute the entity-indexed interaction bank with the actor view."""

    if not isinstance(interactions, BlockInteractionState):
        raise TypeError("interactions must be BlockInteractionState")
    _, permutation = actor_first_entity_permutation(state, actor_index)
    entity_count = state.combat.health.shape[1]
    leaves = jax.tree_util.tree_leaves(interactions)
    if not leaves or any(
        value.ndim < 1 or value.shape[0] != permutation.shape[0]
        for value in leaves
    ):
        raise ValueError("interaction leaves must begin with [B]")
    return jax.tree_util.tree_map(
        lambda value: (
            _permute_entity_axis(value, permutation)
            if value.ndim >= 2 and value.shape[1] == entity_count
            else value
        ),
        interactions,
    )


def _combat_view(
    combat: CombatState,
    locomotion: EntityLocomotionState,
    actor: jax.Array,
    permutation: jax.Array,
    inverse: jax.Array,
) -> CombatState:
    batch = combat.health.shape[0]
    actor_is_legacy = actor == AGENT_ENTITY
    position = _permute_entity_axis(combat.position, permutation)
    velocity = _permute_entity_axis(combat.velocity, permutation)
    health = _permute_entity_axis(combat.health, permutation)
    yaw = _permute_entity_axis(combat.yaw, permutation)
    target_old_index = permutation[:, 1]
    # The realized head, not the commanded one. This read `desired_yaw` because
    # a realized head yaw did not exist per entity until the body/head split;
    # the command skips the authored offset window, so a non-legacy target would
    # have reported a head the engine would never let it hold.
    target_yaw = jnp.where(
        actor_is_legacy,
        combat.target_head_yaw,
        _gather_entity(locomotion.head_yaw, target_old_index),
    )
    target_pitch = jnp.where(
        actor_is_legacy,
        combat.target_head_pitch,
        _gather_entity(locomotion.pitch, target_old_index),
    )
    walk = jax.tree_util.tree_map(
        lambda value: _gather_entity(value, actor),
        locomotion.walk,
    )
    zeros_i = jnp.zeros((batch,), dtype=jnp.int32)
    zeros_f = jnp.zeros((batch,), dtype=jnp.float32)
    zeros_b = jnp.zeros((batch,), dtype=jnp.bool_)

    def legacy_or_zero(value):
        return jnp.where(actor_is_legacy, value, jnp.zeros_like(value))

    engagement = _remap_entity_ids(
        _permute_entity_axis(combat.engagement_target_id, permutation),
        inverse,
    )
    return combat._replace(
        position=position,
        velocity=velocity,
        health=health,
        yaw=yaw,
        target_head_yaw=target_yaw,
        target_head_pitch=target_pitch,
        desired_yaw=_gather_entity(locomotion.desired_yaw, actor),
        desired_body_yaw=_gather_entity(locomotion.desired_body_yaw, actor),
        pitch=_gather_entity(locomotion.pitch, actor),
        desired_pitch=_gather_entity(locomotion.desired_pitch, actor),
        # This view puts one actor in the agent slot, so its head has to travel
        # with it -- otherwise every actor would wear whichever head the
        # underlying state happened to be carrying. It used to take the body
        # bank, which pinned the head/body offset to zero in every view however
        # far the two seams had actually separated.
        agent_head_yaw=_gather_entity(locomotion.head_yaw, actor),
        agent_head_pitch=_gather_entity(locomotion.pitch, actor),
        desired_velocity=_gather_entity(locomotion.desired_velocity, actor),
        agent_move_speed=_gather_entity(locomotion.move_speed, actor),
        agent_fall_speed=_gather_entity(locomotion.fall_speed, actor),
        agent_fall_start_y=_gather_entity(locomotion.fall_start_y, actor),
        target_attack_sequence_index=legacy_or_zero(
            combat.target_attack_sequence_index
        ),
        target_ai_activation_tick=legacy_or_zero(combat.target_ai_activation_tick),
        target_attack_index=jnp.where(
            actor_is_legacy, combat.target_attack_index, jnp.int32(-1)
        ),
        last_target_attack_index=jnp.where(
            actor_is_legacy, combat.last_target_attack_index, jnp.int32(-1)
        ),
        target_attack_elapsed_ticks=legacy_or_zero(
            combat.target_attack_elapsed_ticks
        ),
        target_attack_cooldown_seconds=legacy_or_zero(
            combat.target_attack_cooldown_seconds
        ),
        target_next_attack_queue_tick=legacy_or_zero(
            combat.target_next_attack_queue_tick
        ),
        target_attack_queued=legacy_or_zero(combat.target_attack_queued),
        target_attack_hit_applied=legacy_or_zero(
            combat.target_attack_hit_applied
        ),
        target_damage_pending=legacy_or_zero(combat.target_damage_pending),
        pending_knockback_velocity=jnp.where(
            actor_is_legacy[:, None],
            combat.pending_knockback_velocity,
            jnp.zeros_like(combat.pending_knockback_velocity),
        ),
        target_damage_applied_this_tick=_gather_entity(
            locomotion.force_pushed, actor
        ),
        target_out_of_range_ticks=legacy_or_zero(combat.target_out_of_range_ticks),
        target_maintain_approaching=legacy_or_zero(
            combat.target_maintain_approaching
        ),
        target_maintain_moving_away=legacy_or_zero(
            combat.target_maintain_moving_away
        ),
        target_strafe_delay_seconds=legacy_or_zero(
            combat.target_strafe_delay_seconds
        ),
        target_strafe_paused=legacy_or_zero(combat.target_strafe_paused),
        target_strafe_direction=legacy_or_zero(combat.target_strafe_direction),
        target_last_seen_position=jnp.where(
            actor_is_legacy[:, None],
            combat.target_last_seen_position,
            jnp.zeros_like(combat.target_last_seen_position),
        ),
        target_last_seen_valid=legacy_or_zero(combat.target_last_seen_valid),
        engagement_target_id=engagement,
        ticks_since_agent_damage=_gather_entity(
            locomotion.ticks_since_damage, actor
        ),
        vertical_impulse_applied=_gather_entity(
            locomotion.vertical_impulse_applied, actor
        ),
        agent_applied_vertical_velocity=_gather_entity(
            locomotion.applied_vertical_velocity, actor
        ),
        grounded_with_residual_velocity=_gather_entity(
            locomotion.grounded_with_residual_velocity, actor
        ),
        knockback_control_lock=_gather_entity(
            locomotion.knockback_control_lock, actor
        ),
        agent_grounded=_gather_entity(locomotion.grounded, actor),
        agent_force_velocity=_gather_entity(locomotion.force_velocity, actor),
        agent_walk_movement_state=walk,
        geometry_exhausted=_gather_entity(
            locomotion.geometry_exhausted, actor
        ),
        # Keep dtype/shape-static zero values explicit for the nonlegacy view.
        agent_hit_delay=jnp.where(actor_is_legacy, combat.agent_hit_delay, zeros_i),
        agent_attack_cooldown_seconds=jnp.where(
            actor_is_legacy,
            combat.agent_attack_cooldown_seconds,
            zeros_f,
        ),
        pending_agent_attack_index=jnp.where(
            actor_is_legacy,
            combat.pending_agent_attack_index,
            jnp.int32(-1),
        ),
        completion_awarded=jnp.where(
            actor_is_legacy, combat.completion_awarded, zeros_b
        ),
        death_penalty_awarded=jnp.where(
            actor_is_legacy, combat.death_penalty_awarded, zeros_b
        ),
    )


def _capability_view(
    capabilities: ArsenalWorldCapabilities,
    actor: jax.Array,
    permutation: jax.Array,
) -> ArsenalWorldCapabilities:
    legacy = actor == AGENT_ENTITY

    def legacy_scalar(value):
        mask = legacy.reshape((legacy.shape[0],) + (1,) * (value.ndim - 1))
        return jnp.where(mask, value, jnp.zeros_like(value))

    return capabilities._replace(
        actor_world_state_available=legacy_scalar(
            capabilities.actor_world_state_available
        ),
        actor_controller_medium_available=legacy_scalar(
            capabilities.actor_controller_medium_available
        ),
        actor_submersion_available=legacy_scalar(
            capabilities.actor_submersion_available
        ),
        actor_drop_available=legacy_scalar(capabilities.actor_drop_available),
        actor_controller_in_fluid=legacy_scalar(
            capabilities.actor_controller_in_fluid
        ),
        actor_feet_submerged=legacy_scalar(capabilities.actor_feet_submerged),
        actor_eyes_submerged=legacy_scalar(capabilities.actor_eyes_submerged),
        actor_drop_support_found=legacy_scalar(
            capabilities.actor_drop_support_found
        ),
        actor_drop_height=legacy_scalar(capabilities.actor_drop_height),
        target_candidate_perceptible=_permute_pairwise_entity_axes(
            capabilities.target_candidate_perceptible,
            permutation,
        ),
        target_candidate_perception_valid=_permute_pairwise_entity_axes(
            capabilities.target_candidate_perception_valid,
            permutation,
        ),
        line_of_sight=_permute_entity_axis(capabilities.line_of_sight, permutation),
        line_of_sight_valid=_permute_entity_axis(
            capabilities.line_of_sight_valid, permutation
        ),
        selector_line_of_sight=_permute_entity_axis(
            capabilities.selector_line_of_sight, permutation
        ),
        selector_line_of_sight_valid=_permute_entity_axis(
            capabilities.selector_line_of_sight_valid, permutation
        ),
        direct_target_selected=_permute_entity_axis(
            capabilities.direct_target_selected, permutation
        ),
        direct_target_selection_valid=_permute_entity_axis(
            capabilities.direct_target_selection_valid, permutation
        ),
        muzzle_position=_permute_entity_axis(
            capabilities.muzzle_position, permutation
        ),
        muzzle_yaw_degrees=_permute_entity_axis(
            capabilities.muzzle_yaw_degrees, permutation
        ),
        muzzle_pitch_degrees=_permute_entity_axis(
            capabilities.muzzle_pitch_degrees, permutation
        ),
        muzzle_valid=_permute_entity_axis(capabilities.muzzle_valid, permutation),
        clear_projectile_flight=_permute_entity_axis(
            capabilities.clear_projectile_flight, permutation
        ),
        projectile_world_collision_available=_permute_entity_axis(
            capabilities.projectile_world_collision_available, permutation
        ),
        clear_force_path=_permute_entity_axis(
            capabilities.clear_force_path, permutation
        ),
        applied_force_collision_available=_permute_entity_axis(
            capabilities.applied_force_collision_available, permutation
        ),
        dodge_corridor_clear=_permute_entity_axis(
            capabilities.dodge_corridor_clear, permutation
        ),
        entity_only_area=_permute_entity_axis(
            capabilities.entity_only_area, permutation
        ),
        static_area_placement=_permute_entity_axis(
            capabilities.static_area_placement, permutation
        ),
        area_center=_permute_entity_axis(capabilities.area_center, permutation),
    )


def _actor_first_permutation(
    actor: jax.Array,
    target: jax.Array,
    entity_count: int,
) -> jax.Array:
    entity = jnp.arange(entity_count, dtype=jnp.int32)[None, :]
    key = jnp.where(
        entity == actor[:, None],
        jnp.int32(-2),
        jnp.where(entity == target[:, None], jnp.int32(-1), entity),
    )
    return jnp.argsort(key, axis=1).astype(jnp.int32)


def _inverse_permutation(permutation: jax.Array) -> jax.Array:
    batch, entity_count = permutation.shape
    inverse = jnp.zeros_like(permutation)
    return inverse.at[
        jnp.arange(batch, dtype=jnp.int32)[:, None],
        permutation,
    ].set(jnp.arange(entity_count, dtype=jnp.int32)[None, :])


def _permute_entity_axis(value: jax.Array, permutation: jax.Array) -> jax.Array:
    index = permutation.reshape(
        permutation.shape + (1,) * (value.ndim - 2)
    )
    index = jnp.broadcast_to(index, value.shape)
    return jnp.take_along_axis(value, index, axis=1)


def _permute_if_entity_axis(
    value: jax.Array,
    permutation: jax.Array,
    entity_count: int,
) -> jax.Array:
    if value.ndim >= 2 and value.shape[1] == entity_count:
        return _permute_entity_axis(value, permutation)
    return value


def _permute_pairwise_entity_axes(
    value: jax.Array,
    permutation: jax.Array,
) -> jax.Array:
    first = _permute_entity_axis(value, permutation)
    index = permutation.reshape(
        (permutation.shape[0], 1, permutation.shape[1])
        + (1,) * (value.ndim - 3)
    )
    index = jnp.broadcast_to(index, first.shape)
    return jnp.take_along_axis(first, index, axis=2)


def _gather_entity(value: jax.Array, entity_index: jax.Array) -> jax.Array:
    return value[jnp.arange(value.shape[0], dtype=jnp.int32), entity_index]


def _remap_entity_ids(value: jax.Array, inverse: jax.Array) -> jax.Array:
    valid = (value >= 0) & (value < inverse.shape[1])
    safe = jnp.clip(value, 0, inverse.shape[1] - 1)
    batch_index = jnp.arange(value.shape[0], dtype=jnp.int32).reshape(
        (value.shape[0],) + (1,) * (value.ndim - 1)
    )
    remapped = inverse[batch_index, safe]
    return jnp.where(valid, remapped, value)


__all__ = [
    "actor_first_arsenal_view",
    "actor_first_block_interaction_view",
    "actor_first_entity_permutation",
]
