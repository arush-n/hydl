"""Host factories for empty fixed-capacity arsenal trees."""

from __future__ import annotations

import operator

import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.arsenal.schema.contract import (
    ABILITY_CAPACITY,
    ABILITY_CHARGE_CAPACITY,
    ABILITY_HOLD_CAPACITY,
    AREA_CAPACITY,
    EVENT_CAPACITY,
    EVENT_FLOAT_FEATURES,
    EVENT_INTEGER_FEATURES,
    EVENT_SCHEDULER_CLOCK_COUNT,
    EI_ON_HIT_RESOURCE_ID,
    EI_STATUS_RESOURCE_ID,
    PROJECTILE_CAPACITY,
    INTERACTION_TYPE_UNRESOLVED,
)
from hytalegym.jax.combat.arsenal.schema.types import (
    AbilityLoadout,
    AreaState,
    ArsenalCommands,
    ArsenalRuntimeCapacity,
    ArsenalState,
    ArsenalWorldCapabilities,
    ProjectileState,
)
from hytalegym.jax.combat.mechanics import (
    GUARD_ENTRY_STAMINA_COST,
    GUARD_EXIT_REGEN_DELAY_SECONDS,
    GUARD_HALF_ANGLE_DEGREES,
    RESOURCE_COUNT,
    RESOURCE_OXYGEN,
    RESOURCE_STAMINA,
    OXYGEN_MAXIMUM,
    STAMINA_MAXIMUM,
    default_mechanics_rules,
    empty_defense_commands,
)
from hytalegym.jax.combat.types import ENTITY_COUNT


def empty_ability_loadout(
    batch_size: int,
    *,
    entity_count: int = ENTITY_COUNT,
) -> AbilityLoadout:
    batch = _batch_size(batch_size)
    entity = (
        batch,
        _positive_capacity(entity_count, "entity_count"),
    )
    ability = entity + (ABILITY_CAPACITY,)
    event = ability + (EVENT_CAPACITY,)
    resource_maximum = jnp.zeros(
        entity + (RESOURCE_COUNT,),
        dtype=jnp.float32,
    )
    resource_maximum = resource_maximum.at[..., RESOURCE_STAMINA].set(STAMINA_MAXIMUM)
    resource_maximum = resource_maximum.at[..., RESOURCE_OXYGEN].set(OXYGEN_MAXIMUM)
    resource_initial = jnp.zeros_like(resource_maximum)
    resource_initial = resource_initial.at[..., RESOURCE_STAMINA].set(STAMINA_MAXIMUM)
    resource_initial = resource_initial.at[..., RESOURCE_OXYGEN].set(OXYGEN_MAXIMUM)
    event_i32 = jnp.zeros(
        event + (EVENT_INTEGER_FEATURES,),
        dtype=jnp.int32,
    )
    event_i32 = event_i32.at[..., EI_STATUS_RESOURCE_ID].set(-1)
    event_i32 = event_i32.at[..., EI_ON_HIT_RESOURCE_ID].set(-1)
    return AbilityLoadout(
        weapon_id=jnp.full(entity, -1, dtype=jnp.int32),
        weapon_family=jnp.zeros(entity, dtype=jnp.int32),
        equipped=jnp.zeros(entity, dtype=jnp.bool_),
        guard_entry_cost=jnp.full(
            entity,
            GUARD_ENTRY_STAMINA_COST,
            dtype=jnp.float32,
        ),
        guard_stamina_value=jnp.zeros(entity, dtype=jnp.float32),
        guard_half_angle_degrees=jnp.full(
            entity,
            GUARD_HALF_ANGLE_DEGREES,
            dtype=jnp.float32,
        ),
        guard_entry_delay_seconds=jnp.zeros(entity, dtype=jnp.float32),
        guard_exit_regen_delay_seconds=jnp.full(
            entity,
            GUARD_EXIT_REGEN_DELAY_SECONDS,
            dtype=jnp.float32,
        ),
        guard_required_resource_id=jnp.full(entity, -1, dtype=jnp.int32),
        guard_required_resource_minimum=jnp.zeros(
            entity,
            dtype=jnp.float32,
        ),
        guard_interrupting_type_mask=jnp.zeros(entity, dtype=jnp.uint32),
        resource_maximum=resource_maximum,
        resource_initial=resource_initial,
        ability_id=jnp.zeros(ability, dtype=jnp.int32),
        ability_interaction_type=jnp.full(
            ability,
            INTERACTION_TYPE_UNRESOLVED,
            dtype=jnp.int32,
        ),
        ability_guard_fork_type=jnp.full(
            ability,
            INTERACTION_TYPE_UNRESOLVED,
            dtype=jnp.int32,
        ),
        ability_mask=jnp.zeros(ability, dtype=jnp.bool_),
        ability_evidence=jnp.zeros(ability, dtype=jnp.int32),
        ability_duration_seconds=jnp.zeros(ability, dtype=jnp.float32),
        ability_cooldown_seconds=jnp.zeros(ability, dtype=jnp.float32),
        ability_requested_charge_time_seconds=jnp.full(
            ability,
            -1.0,
            dtype=jnp.float32,
        ),
        ability_charge_times_seconds=jnp.zeros(
            ability + (ABILITY_CHARGE_CAPACITY,),
            dtype=jnp.float32,
        ),
        ability_charge_capacity=jnp.ones(ability, dtype=jnp.int32),
        ability_hold_threshold_seconds=jnp.zeros(
            ability + (ABILITY_HOLD_CAPACITY,),
            dtype=jnp.float32,
        ),
        ability_hold_child_slot=jnp.full(
            ability + (ABILITY_HOLD_CAPACITY,),
            -1,
            dtype=jnp.int32,
        ),
        ability_hold_count=jnp.zeros(ability, dtype=jnp.int32),
        ability_hold_allow_indefinite=jnp.zeros(ability, dtype=jnp.bool_),
        ability_hold_speed_multiplier=jnp.ones(ability, dtype=jnp.float32),
        ability_hold_speed_multiplier_after_seconds=jnp.zeros(
            ability, dtype=jnp.float32
        ),
        ability_continuation_mode=jnp.zeros(ability, dtype=jnp.int32),
        ability_continuation_ground_slot=jnp.full(ability, -1, dtype=jnp.int32),
        ability_continuation_collision_slot=jnp.full(ability, -1, dtype=jnp.int32),
        ability_continuation_ground_check_delay_seconds=jnp.zeros(
            ability, dtype=jnp.float32
        ),
        ability_continuation_run_time_seconds=jnp.full(
            ability, -1.0, dtype=jnp.float32
        ),
        ability_interrupt_recharge=jnp.zeros(ability, dtype=jnp.bool_),
        ability_interrupted_by_type_mask=jnp.zeros(
            ability,
            dtype=jnp.uint32,
        ),
        ability_interruptible_after_seconds=jnp.zeros(
            ability,
            dtype=jnp.float32,
        ),
        ability_stamina_regen_delay_seconds=jnp.zeros(
            ability,
            dtype=jnp.float32,
        ),
        ability_stamina_regen_delay_start_tick=jnp.zeros(
            ability,
            dtype=jnp.int32,
        ),
        ability_stamina_regen_delay_end_tick=jnp.full(
            ability,
            -1,
            dtype=jnp.int32,
        ),
        ability_scheduler_prelude_ticks=jnp.zeros(ability, dtype=jnp.int32),
        ability_outer_root_selector=jnp.zeros(ability, dtype=jnp.bool_),
        ability_outer_root_item_dispatch_tick=jnp.full(
            ability,
            -1,
            dtype=jnp.int32,
        ),
        ability_resource_phase_mask=jnp.zeros(
            ability + (RESOURCE_COUNT,),
            dtype=jnp.bool_,
        ),
        ability_resource_phase_value=jnp.zeros(
            ability + (RESOURCE_COUNT,),
            dtype=jnp.float32,
        ),
        ability_resource_phase_inactive_value=jnp.zeros(
            ability + (RESOURCE_COUNT,),
            dtype=jnp.float32,
        ),
        ability_resource_phase_start_tick=jnp.zeros(
            ability + (RESOURCE_COUNT,),
            dtype=jnp.int32,
        ),
        ability_resource_phase_end_tick=jnp.zeros(
            ability + (RESOURCE_COUNT,),
            dtype=jnp.int32,
        ),
        ability_resource_cost=jnp.zeros(
            ability + (RESOURCE_COUNT,),
            dtype=jnp.float32,
        ),
        ability_resource_cost_kind=jnp.zeros(
            ability + (RESOURCE_COUNT,),
            dtype=jnp.int32,
        ),
        ability_resource_commit_time_seconds=jnp.zeros(
            ability + (RESOURCE_COUNT,),
            dtype=jnp.float32,
        ),
        ability_resource_commit_flags=jnp.zeros(
            ability + (RESOURCE_COUNT,),
            dtype=jnp.uint32,
        ),
        ability_resource_minimum=jnp.zeros(
            ability + (RESOURCE_COUNT,),
            dtype=jnp.float32,
        ),
        ability_requirements=jnp.zeros(ability, dtype=jnp.uint32),
        ability_static_placement_maximum_distance=jnp.zeros(
            ability,
            dtype=jnp.float32,
        ),
        ability_static_placement_allow_walls=jnp.zeros(
            ability,
            dtype=jnp.bool_,
        ),
        ability_event_start=jnp.zeros(ability, dtype=jnp.int32),
        ability_event_count=jnp.zeros(ability, dtype=jnp.int32),
        ability_event_mask=jnp.zeros(event, dtype=jnp.bool_),
        event_mask=jnp.zeros(event, dtype=jnp.bool_),
        event_time_seconds=jnp.zeros(event, dtype=jnp.float32),
        event_kind=jnp.zeros(event, dtype=jnp.int32),
        event_f32=jnp.zeros(
            event + (EVENT_FLOAT_FEATURES,),
            dtype=jnp.float32,
        ),
        event_i32=event_i32,
        event_flags=jnp.zeros(event, dtype=jnp.uint32),
        overflow=jnp.zeros(entity, dtype=jnp.bool_),
    )


def mechanics_rules_for_loadout(loadout: AbilityLoadout):
    """Project equipment maxima and guard drain into common mechanics rules."""

    batch = loadout.weapon_id.shape[0]
    rules = default_mechanics_rules(
        batch,
        entity_count=loadout.weapon_id.shape[1],
    )
    regen_amount = rules.resource_regen_amount
    regen_interval = rules.resource_regen_interval_seconds
    return rules._replace(
        resource_maximum=loadout.resource_maximum,
        resource_regen_amount=regen_amount,
        resource_regen_interval_seconds=regen_interval,
        guard_entry_cost=loadout.guard_entry_cost,
        guard_stamina_value=loadout.guard_stamina_value,
        guard_half_angle_degrees=loadout.guard_half_angle_degrees,
        guard_entry_delay_seconds=loadout.guard_entry_delay_seconds,
        guard_exit_regen_delay_seconds=(loadout.guard_exit_regen_delay_seconds),
        guard_required_resource_id=loadout.guard_required_resource_id,
        guard_required_resource_minimum=(loadout.guard_required_resource_minimum),
    )


def empty_arsenal_state(
    batch_size: int,
    *,
    ability_capacity: int = ABILITY_CAPACITY,
    projectile_capacity: int = PROJECTILE_CAPACITY,
    area_capacity: int = AREA_CAPACITY,
    entity_count: int = ENTITY_COUNT,
) -> ArsenalState:
    batch = _batch_size(batch_size)
    capacity = _positive_capacity(ability_capacity, "ability_capacity")
    if capacity > ABILITY_CAPACITY:
        raise ValueError(
            f"ability_capacity must be <= {ABILITY_CAPACITY}, got {capacity}"
        )
    entity = (
        batch,
        _positive_capacity(entity_count, "entity_count"),
    )
    return ArsenalState(
        active_ability_slot=jnp.full(entity, -1, dtype=jnp.int32),
        active_ability_root_slot=jnp.full(entity, -1, dtype=jnp.int32),
        ability_elapsed_seconds=jnp.zeros(entity, dtype=jnp.float32),
        ability_scheduler_tick=jnp.zeros(entity, dtype=jnp.int32),
        ability_scheduler_clock_seconds=jnp.zeros(
            entity + (EVENT_SCHEDULER_CLOCK_COUNT,),
            dtype=jnp.float32,
        ),
        ability_selector_hit_bits=jnp.zeros(entity, dtype=jnp.uint32),
        ability_cooldown_seconds=jnp.zeros(
            entity + (capacity,),
            dtype=jnp.float32,
        ),
        ability_charge_count=jnp.ones(
            entity + (capacity,),
            dtype=jnp.int32,
        ),
        ability_charge_timer_seconds=jnp.zeros(
            entity + (capacity,),
            dtype=jnp.float32,
        ),
        ability_hold_slot=jnp.full(entity, -1, dtype=jnp.int32),
        ability_hold_seconds=jnp.zeros(entity, dtype=jnp.float32),
        ability_continuation_slot=jnp.full(entity, -1, dtype=jnp.int32),
        ability_continuation_seconds=jnp.zeros(entity, dtype=jnp.float32),
        projectiles=_empty_projectiles(batch, projectile_capacity),
        areas=_empty_areas(batch, area_capacity),
        failure_bits=jnp.zeros((batch,), dtype=jnp.uint32),
    )


def arsenal_runtime_capacity(loadout: AbilityLoadout) -> ArsenalRuntimeCapacity:
    """Derive the smallest lossless static shape for one selected loadout."""

    ability_mask = np.asarray(loadout.ability_mask, dtype=np.bool_)
    event_mask = np.asarray(loadout.event_mask, dtype=np.bool_)
    if ability_mask.ndim != 3 or event_mask.ndim not in (3, 4):
        raise ValueError("loadout ability/event masks have invalid ranks")
    if ability_mask.shape[2] > ABILITY_CAPACITY:
        raise ValueError("loadout exceeds the authored catalog capacity")

    if event_mask.ndim == 4:
        if event_mask.shape[:3] != ability_mask.shape:
            raise ValueError("loadout ability/event mask shapes do not align")
        if event_mask.shape[3] > EVENT_CAPACITY:
            raise ValueError("loadout exceeds the authored catalog capacity")
        occupied_ability = np.any(ability_mask, axis=(0, 1)) | np.any(
            event_mask,
            axis=(0, 1, 3),
        )
        event_counts = np.sum(event_mask, axis=3)
        events_per_entity = max(
            1,
            int(np.max(np.sum(event_counts, axis=2))),
        )
        events_per_ability = max(1, int(np.max(event_counts)))
    else:
        if event_mask.shape[:2] != ability_mask.shape[:2]:
            raise ValueError("packed event batch/entity axes do not align")
        counts = np.asarray(loadout.ability_event_count, dtype=np.int32)
        starts = np.asarray(loadout.ability_event_start, dtype=np.int32)
        if counts.shape != ability_mask.shape or starts.shape != ability_mask.shape:
            raise ValueError("packed ability event offsets do not align")
        if np.any(counts < 0) or np.any(starts < 0):
            raise ValueError("packed ability event offsets must be non-negative")
        if np.any(starts + counts > event_mask.shape[2]):
            raise ValueError("packed ability event offsets exceed the event bank")
        occupied_ability = np.any(
            ability_mask | (counts > 0),
            axis=(0, 1),
        )
        events_per_entity = event_mask.shape[2]
        events_per_ability = max(1, int(np.max(counts)))

    return ArsenalRuntimeCapacity(
        abilities_per_entity=_last_occupied(occupied_ability),
        events_per_entity=events_per_entity,
        events_per_ability=events_per_ability,
    )


def specialize_ability_loadout(loadout: AbilityLoadout) -> AbilityLoadout:
    """Pack selected programs into one lossless flat event bank per entity."""

    if loadout.event_mask.ndim == 3:
        arsenal_runtime_capacity(loadout)
        return loadout
    capacity = arsenal_runtime_capacity(loadout)
    ability_fields = {
        "ability_id",
        "ability_interaction_type",
        "ability_guard_fork_type",
        "ability_mask",
        "ability_evidence",
        "ability_duration_seconds",
        "ability_cooldown_seconds",
        "ability_requested_charge_time_seconds",
        "ability_charge_times_seconds",
        "ability_charge_capacity",
        "ability_hold_threshold_seconds",
        "ability_hold_child_slot",
        "ability_hold_count",
        "ability_hold_allow_indefinite",
        "ability_hold_speed_multiplier",
        "ability_hold_speed_multiplier_after_seconds",
        "ability_continuation_mode",
        "ability_continuation_ground_slot",
        "ability_continuation_collision_slot",
        "ability_continuation_ground_check_delay_seconds",
        "ability_continuation_run_time_seconds",
        "ability_interrupt_recharge",
        "ability_interrupted_by_type_mask",
        "ability_interruptible_after_seconds",
        "ability_stamina_regen_delay_seconds",
        "ability_stamina_regen_delay_start_tick",
        "ability_stamina_regen_delay_end_tick",
        "ability_scheduler_prelude_ticks",
        "ability_outer_root_selector",
        "ability_outer_root_item_dispatch_tick",
        "ability_resource_phase_mask",
        "ability_resource_phase_value",
        "ability_resource_phase_inactive_value",
        "ability_resource_phase_start_tick",
        "ability_resource_phase_end_tick",
        "ability_resource_cost",
        "ability_resource_cost_kind",
        "ability_resource_commit_time_seconds",
        "ability_resource_commit_flags",
        "ability_resource_minimum",
        "ability_requirements",
        "ability_static_placement_maximum_distance",
        "ability_static_placement_allow_walls",
        "ability_event_start",
        "ability_event_count",
    }
    event_window_fields = {"ability_event_mask"}
    event_fields = {
        "event_mask",
        "event_time_seconds",
        "event_kind",
        "event_f32",
        "event_i32",
        "event_flags",
    }
    batch, entities = loadout.event_mask.shape[:2]
    ability_capacity = capacity.abilities_per_entity
    event_capacity = capacity.events_per_entity
    source_mask = np.asarray(
        loadout.event_mask[:, :, :ability_capacity],
        dtype=np.bool_,
    )
    starts = np.zeros(
        (batch, entities, ability_capacity),
        dtype=np.int32,
    )
    counts = np.zeros_like(starts)
    sources = {}
    packed = {}
    for field in event_fields:
        source = np.asarray(getattr(loadout, field)[:, :, :ability_capacity])
        sources[field] = source
        packed[field] = np.zeros(
            (batch, entities, event_capacity) + source.shape[4:],
            dtype=source.dtype,
        )
    for row in range(batch):
        for entity in range(entities):
            cursor = 0
            for ability in range(ability_capacity):
                source_indices = np.flatnonzero(source_mask[row, entity, ability])
                count = int(source_indices.size)
                starts[row, entity, ability] = cursor
                counts[row, entity, ability] = count
                if count:
                    target = slice(cursor, cursor + count)
                    for field in event_fields:
                        packed[field][row, entity, target] = sources[field][
                            row,
                            entity,
                            ability,
                            source_indices,
                        ]
                cursor += count

    values = {}
    for field in loadout._fields:
        value = getattr(loadout, field)
        if field in event_fields:
            value = jnp.asarray(packed[field])
        elif field in ability_fields:
            value = value[:, :, :ability_capacity, ...]
        elif field in event_window_fields:
            value = value[
                :,
                :,
                :ability_capacity,
                : capacity.events_per_ability,
            ]
        values[field] = value
    values["ability_event_start"] = jnp.asarray(starts)
    values["ability_event_count"] = jnp.asarray(counts)
    return type(loadout)(**values)


def empty_arsenal_commands(
    batch_size: int,
    *,
    entity_count: int = ENTITY_COUNT,
) -> ArsenalCommands:
    batch = _batch_size(batch_size)
    entities = _positive_capacity(entity_count, "entity_count")
    entity = (batch, entities)
    return ArsenalCommands(
        ability_slot=jnp.full(entity, -1, dtype=jnp.int32),
        # -1 everywhere: an empty command switches nobody's weapon.
        hotbar_slot=jnp.full(entity, -1, dtype=jnp.int32),
        defense=empty_defense_commands(batch, entity_count=entities),
        world=ArsenalWorldCapabilities(
            actor_world_state_available=jnp.zeros((batch,), dtype=jnp.bool_),
            actor_controller_medium_available=jnp.zeros(
                (batch,),
                dtype=jnp.bool_,
            ),
            actor_submersion_available=jnp.zeros((batch,), dtype=jnp.bool_),
            actor_drop_available=jnp.zeros((batch,), dtype=jnp.bool_),
            actor_controller_in_fluid=jnp.zeros((batch,), dtype=jnp.bool_),
            actor_feet_submerged=jnp.zeros((batch,), dtype=jnp.bool_),
            actor_eyes_submerged=jnp.zeros((batch,), dtype=jnp.bool_),
            actor_drop_support_found=jnp.zeros((batch,), dtype=jnp.bool_),
            actor_drop_height=jnp.zeros((batch,), dtype=jnp.float32),
            target_candidate_perceptible=jnp.zeros(
                (batch, entities, entities),
                dtype=jnp.bool_,
            ),
            target_candidate_perception_valid=jnp.zeros(
                (batch, entities, entities),
                dtype=jnp.bool_,
            ),
            line_of_sight=jnp.zeros(entity, dtype=jnp.bool_),
            line_of_sight_valid=jnp.zeros(entity, dtype=jnp.bool_),
            selector_line_of_sight=jnp.zeros(entity, dtype=jnp.bool_),
            selector_line_of_sight_valid=jnp.zeros(
                entity,
                dtype=jnp.bool_,
            ),
            direct_target_selected=jnp.zeros(entity, dtype=jnp.bool_),
            direct_target_selection_valid=jnp.zeros(
                entity,
                dtype=jnp.bool_,
            ),
            muzzle_position=jnp.zeros(entity + (3,), dtype=jnp.float32),
            muzzle_yaw_degrees=jnp.zeros(entity, dtype=jnp.float32),
            muzzle_pitch_degrees=jnp.zeros(entity, dtype=jnp.float32),
            muzzle_valid=jnp.zeros(entity, dtype=jnp.bool_),
            clear_projectile_flight=jnp.zeros(entity, dtype=jnp.bool_),
            projectile_world_collision_available=jnp.zeros(
                entity,
                dtype=jnp.bool_,
            ),
            deployable_intended_graph_available=jnp.zeros(
                entity,
                dtype=jnp.bool_,
            ),
            clear_force_path=jnp.zeros(
                entity + (ABILITY_CAPACITY,),
                dtype=jnp.bool_,
            ),
            applied_force_collision_available=jnp.zeros(
                entity + (ABILITY_CAPACITY,),
                dtype=jnp.bool_,
            ),
            dodge_corridor_clear=jnp.zeros(
                entity + (4,),
                dtype=jnp.bool_,
            ),
            entity_only_area=jnp.zeros(entity, dtype=jnp.bool_),
            static_area_placement=jnp.zeros(entity, dtype=jnp.bool_),
            area_center=jnp.zeros(entity + (3,), dtype=jnp.float32),
        ),
    )


def _empty_projectiles(batch: int, capacity: int) -> ProjectileState:
    shape = (batch, _bounded_capacity(capacity, PROJECTILE_CAPACITY, "projectile"))
    f32 = jnp.zeros(shape, dtype=jnp.float32)
    i32 = jnp.zeros(shape, dtype=jnp.int32)
    return ProjectileState(
        position=jnp.zeros(shape + (3,), dtype=jnp.float32),
        velocity=jnp.zeros(shape + (3,), dtype=jnp.float32),
        launch_yaw_degrees=f32,
        half_extent=jnp.zeros(shape + (3,), dtype=jnp.float32),
        collision_center_offset=jnp.zeros(
            shape + (3,),
            dtype=jnp.float32,
        ),
        age_seconds=f32,
        lifetime_seconds=f32,
        damage=f32,
        direct_damage=f32,
        random_percentage=f32,
        damage_cause=i32,
        direct_damage_cause=i32,
        damage_class=i32,
        gravity=f32,
        terminal_velocity=f32,
        standard_physics=jnp.zeros(shape, dtype=jnp.bool_),
        bounciness=f32,
        bounce_limit=jnp.full(shape, 0.4, dtype=jnp.float32),
        bounce_count_limit=i32,
        bounce_count=i32,
        allow_rolling=jnp.zeros(shape, dtype=jnp.bool_),
        rolling_friction_factor=jnp.ones(shape, dtype=jnp.float32),
        sticks_vertically=jnp.zeros(shape, dtype=jnp.bool_),
        on_ground=jnp.zeros(shape, dtype=jnp.bool_),
        fuse_seconds=f32,
        dead_time_seconds=jnp.full(shape, -1.0, dtype=jnp.float32),
        dead_time_remaining=f32,
        explosion_radius=f32,
        explosion_falloff=f32,
        block_damage_radius=i32,
        damage_blocks=jnp.zeros(shape, dtype=jnp.bool_),
        force_direction=jnp.zeros(shape + (3,), dtype=jnp.float32),
        force_direction_mode=i32,
        force_velocity_y=f32,
        force_magnitude=f32,
        force_mode=i32,
        air_resistance=jnp.ones(shape, dtype=jnp.float32),
        air_resistance_max=jnp.ones(shape, dtype=jnp.float32),
        ground_resistance=jnp.ones(shape, dtype=jnp.float32),
        ground_resistance_max=jnp.ones(shape, dtype=jnp.float32),
        resistance_threshold=jnp.ones(shape, dtype=jnp.float32),
        resistance_style=i32,
        on_hit_resource_id=jnp.full(shape, -1, dtype=jnp.int32),
        on_hit_resource_delta=f32,
        on_hit_healing=f32,
        status_id=i32,
        status_duration_seconds=f32,
        status_cooldown_seconds=f32,
        status_damage=f32,
        status_damage_cause=i32,
        status_resource_id=jnp.full(shape, -1, dtype=jnp.int32),
        status_resource_delta=f32,
        status_speed_multiplier=jnp.ones(shape, dtype=jnp.float32),
        status_flags=jnp.zeros(shape, dtype=jnp.uint32),
        status_overlap_mode=i32,
        terminal_deployable_area=jnp.zeros(shape, dtype=jnp.bool_),
        terminal_intended_graph_available=jnp.zeros(
            shape,
            dtype=jnp.bool_,
        ),
        terminal_entity_contact_inactive=jnp.zeros(shape, dtype=jnp.bool_),
        terminal_deployable_collision_half_extent=jnp.zeros(
            shape + (3,),
            dtype=jnp.float32,
        ),
        terminal_deployable_collision_center_offset=jnp.zeros(
            shape + (3,),
            dtype=jnp.float32,
        ),
        terminal_deployable_id=i32,
        terminal_deployable_count_towards_global_limit=jnp.zeros(
            shape,
            dtype=jnp.bool_,
        ),
        terminal_deployable_max_live_count=i32,
        terminal_area_shape=i32,
        terminal_area_duration_seconds=f32,
        terminal_area_interval_seconds=f32,
        terminal_area_end_radius=f32,
        terminal_area_height=f32,
        terminal_area_radius_change_seconds=f32,
        terminal_area_attack_flags=i32,
        kind=i32,
        crossbow_program=jnp.zeros(shape, dtype=jnp.bool_),
        crossbow_combo_damage=f32,
        owner_entity_id=jnp.full(shape, -1, dtype=jnp.int32),
        active=jnp.zeros(shape, dtype=jnp.bool_),
        impacted=jnp.zeros(shape, dtype=jnp.bool_),
        physics_initialized=jnp.zeros(shape, dtype=jnp.bool_),
        entity_collision_only=jnp.zeros(shape, dtype=jnp.bool_),
        world_hit=jnp.zeros(shape, dtype=jnp.bool_),
        world_hit_fraction=jnp.ones(shape, dtype=jnp.float32),
        world_contact_point=jnp.zeros(shape + (3,), dtype=jnp.float32),
        world_contact_normal=jnp.zeros(shape + (3,), dtype=jnp.float32),
        world_geometry_exhausted=jnp.zeros(shape, dtype=jnp.bool_),
        world_capacity_exceeded=jnp.zeros(shape, dtype=jnp.bool_),
        world_contact_invalid=jnp.zeros(shape, dtype=jnp.bool_),
        world_segment_count=jnp.zeros(shape, dtype=jnp.int32),
    )


def _empty_areas(batch: int, capacity: int) -> AreaState:
    shape = (batch, _bounded_capacity(capacity, AREA_CAPACITY, "area"))
    f32 = jnp.zeros(shape, dtype=jnp.float32)
    i32 = jnp.zeros(shape, dtype=jnp.int32)
    return AreaState(
        center=jnp.zeros(shape + (3,), dtype=jnp.float32),
        collision_half_extent=jnp.zeros(shape + (3,), dtype=jnp.float32),
        collision_center_offset=jnp.zeros(shape + (3,), dtype=jnp.float32),
        age_seconds=f32,
        duration_seconds=f32,
        interval_seconds=f32,
        interval_clock_seconds=f32,
        radius_change_seconds=f32,
        start_radius=f32,
        end_radius=f32,
        height=f32,
        damage=f32,
        random_percentage=f32,
        damage_cause=i32,
        damage_class=i32,
        force_direction=jnp.zeros(shape + (3,), dtype=jnp.float32),
        force_magnitude=f32,
        force_mode=i32,
        air_resistance=jnp.ones(shape, dtype=jnp.float32),
        air_resistance_max=jnp.ones(shape, dtype=jnp.float32),
        ground_resistance=jnp.ones(shape, dtype=jnp.float32),
        ground_resistance_max=jnp.ones(shape, dtype=jnp.float32),
        resistance_threshold=jnp.ones(shape, dtype=jnp.float32),
        resistance_style=i32,
        status_id=i32,
        status_duration_seconds=f32,
        status_cooldown_seconds=f32,
        status_damage=f32,
        status_healing=f32,
        status_damage_cause=i32,
        status_resource_id=jnp.full(shape, -1, dtype=jnp.int32),
        status_resource_delta=f32,
        status_speed_multiplier=jnp.ones(shape, dtype=jnp.float32),
        status_flags=jnp.zeros(shape, dtype=jnp.uint32),
        status_overlap_mode=i32,
        kind=i32,
        shape=i32,
        owner_entity_id=jnp.full(shape, -1, dtype=jnp.int32),
        active=jnp.zeros(shape, dtype=jnp.bool_),
        entity_overlap_only=jnp.zeros(shape, dtype=jnp.bool_),
        friendly_fire=jnp.zeros(shape, dtype=jnp.bool_),
        target_mask=i32,
        deployable_attack_flags=i32,
        deployable_id=i32,
        deployable_count_towards_global_limit=jnp.zeros(
            shape,
            dtype=jnp.bool_,
        ),
        deployable_max_live_count=i32,
    )


def _batch_size(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("batch_size must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError("batch_size must be a positive integer") from error
    if result <= 0:
        raise ValueError("batch_size must be a positive integer")
    return result


def _bounded_capacity(value: int, maximum: int, name: str) -> int:
    capacity = _positive_capacity(value, f"{name}_capacity")
    if capacity > maximum:
        raise ValueError(f"{name}_capacity must be <= {maximum}, got {capacity}")
    return capacity


def _positive_capacity(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a positive integer") from error
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _last_occupied(mask: np.ndarray) -> int:
    occupied = np.flatnonzero(mask)
    return 1 if occupied.size == 0 else int(occupied[-1]) + 1
