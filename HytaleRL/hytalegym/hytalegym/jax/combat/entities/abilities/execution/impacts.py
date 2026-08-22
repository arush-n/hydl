"""Atomic handoff from scheduled abilities to existing impact allocators."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal import (
    ABILITY_CAPACITY,
    EVENT_CAPACITY,
    EVENT_AREA,
    EVENT_PROJECTILE,
)
from hytalegym.jax.combat.entities import (
    ENTITY_CAPACITY,
    validate_roster_layout,
)
from hytalegym.jax.combat.entities.abilities.schema.contract import (
    ABILITY_FAILURE_INVALID_PROGRAM,
    ABILITY_FAILURE_INVALID_STATE,
    ABILITY_FAILURE_LAUNCH,
    ABILITY_FAILURE_STALE_SOURCE,
    ABILITY_FAILURE_UPSTREAM,
)
from hytalegym.jax.combat.entities.abilities.schema.types import (
    EntityAbilityLaunchInfo,
)
from hytalegym.jax.combat.entities.abilities.execution.programs import (
    events_match_program,
)
from hytalegym.jax.combat.entities.abilities.schema.validation import (
    validate_entity_ability_event_layout,
)
from hytalegym.jax.combat.entities.impacts import (
    AREAS_PER_LAUNCH,
    RANGED_PROJECTILES_PER_LAUNCH,
    EntityProjectileLaunchCommands,
    launch_entity_areas,
    launch_entity_projectiles,
)


def launch_entity_ability_impacts(
    state,
    arsenal,
    bindings,
    roster,
    events,
    programs,
    projectile_programs,
    area_programs,
    projectile_world,
    area_world,
    friendly_fire,
):
    """Allocate scheduled projectile/area groups, or commit nothing."""

    validate_roster_layout(roster)
    batch = validate_entity_ability_event_layout(events)
    entity = (batch, ENTITY_CAPACITY)
    _field(friendly_fire, entity, jnp.bool_, "friendly_fire")
    if state.weapon_family.shape != entity:
        raise ValueError(f"state.weapon_family must have shape {entity}")

    source = events.source_slot
    source_valid = (source >= 0) & (source < ENTITY_CAPACITY)
    safe_source = jnp.clip(source, 0, ENTITY_CAPACITY - 1)
    batch_index = jnp.arange(batch, dtype=jnp.int32)[:, None]
    event_source_valid = (
        source_valid
        & roster.active[batch_index, safe_source]
        & ~roster.dead[batch_index, safe_source]
        & (events.source_generation == roster.generation[batch_index, safe_source])
        & (
            events.source_generation
            == state.source_generation[batch_index, safe_source]
        )
        & (events.weapon_family == state.weapon_family[batch_index, safe_source])
        & (events.ability_slot >= 0)
        & (events.ability_slot < ABILITY_CAPACITY)
    )
    requested_invalid = events.requested & ~event_source_valid
    bits = state.failure_bits
    world_bits = state.world_failure_bits | events.world_failure_bits
    bits = _set_failure(
        bits,
        ~events.valid | (events.failure_bits != jnp.uint32(0)),
        ABILITY_FAILURE_UPSTREAM,
    )
    bits = _set_failure(
        bits,
        jnp.any(
            events.requested
            & (
                ~source_valid
                | (events.ability_slot < 0)
                | (events.ability_slot >= ABILITY_CAPACITY)
            ),
            axis=1,
        ),
        ABILITY_FAILURE_INVALID_STATE,
    )
    bits = _set_failure(
        bits,
        jnp.any(requested_invalid, axis=1),
        ABILITY_FAILURE_STALE_SOURCE,
    )
    event_matches = events_match_program(events, programs)

    projectile = _dense_launch(
        events,
        programs,
        projectile_programs,
        programs.projectile_launch_compatible,
        EVENT_PROJECTILE,
        RANGED_PROJECTILES_PER_LAUNCH,
    )
    area = _dense_launch(
        events,
        programs,
        area_programs,
        programs.area_launch_compatible,
        EVENT_AREA,
        AREAS_PER_LAUNCH,
    )
    invalid_program = projectile[3] | area[3]
    bits = _set_failure(
        bits,
        invalid_program | jnp.any(events.requested & ~event_matches, axis=1),
        ABILITY_FAILURE_INVALID_PROGRAM,
    )
    pre_valid = (bits == jnp.uint32(0)) & (world_bits == jnp.uint32(0))
    projectile_commands = EntityProjectileLaunchCommands(
        weapon_family=projectile[0],
        ability_slot=projectile[1],
        requested=projectile[2] & pre_valid[:, None],
        damage_multiplier=jnp.ones(entity, dtype=jnp.float32),
        friendly_fire=friendly_fire,
        failure_bits=jnp.zeros((batch,), dtype=jnp.uint32),
        valid=jnp.ones((batch,), dtype=jnp.bool_),
    )
    area_commands = EntityProjectileLaunchCommands(
        weapon_family=area[0],
        ability_slot=area[1],
        requested=area[2] & pre_valid[:, None],
        damage_multiplier=jnp.ones(entity, dtype=jnp.float32),
        friendly_fire=friendly_fire,
        failure_bits=jnp.zeros((batch,), dtype=jnp.uint32),
        valid=jnp.ones((batch,), dtype=jnp.bool_),
    )
    projectile_arsenal, projectile_bindings, projectile_info = (
        launch_entity_projectiles(
            arsenal,
            bindings,
            roster,
            projectile_commands,
            projectile_programs,
            projectile_world,
        )
    )
    area_arsenal, area_bindings, area_info = launch_entity_areas(
        projectile_arsenal,
        projectile_bindings,
        roster,
        area_commands,
        area_programs,
        area_world,
    )
    bits = _set_failure(
        bits,
        ~projectile_info.valid | ~area_info.valid,
        ABILITY_FAILURE_LAUNCH,
    )
    world_bits |= projectile_info.world_failure_bits | area_info.world_failure_bits
    valid = (bits == jnp.uint32(0)) & (world_bits == jnp.uint32(0))
    result_state = state._replace(
        failure_bits=bits,
        world_failure_bits=world_bits,
    )
    result_arsenal = _select_tree(valid, area_arsenal, arsenal)._replace(
        failure_bits=area_arsenal.failure_bits
    )
    result_bindings = _select_tree(valid, area_bindings, bindings)
    projectile_info = projectile_info._replace(
        projectile_spawned=jnp.where(
            valid,
            projectile_info.projectile_spawned,
            jnp.int32(0),
        ),
        projectile_slots=(projectile_info.projectile_slots & valid[:, None]),
        valid=projectile_info.valid & valid,
        binding=projectile_info.binding._replace(
            valid=projectile_info.binding.valid & valid
        ),
    )
    area_info = area_info._replace(
        area_spawned=jnp.where(valid, area_info.area_spawned, jnp.int32(0)),
        area_slots=area_info.area_slots & valid[:, None],
        valid=area_info.valid & valid,
        binding=area_info.binding._replace(valid=area_info.binding.valid & valid),
    )
    return (
        result_state,
        result_arsenal,
        result_bindings,
        EntityAbilityLaunchInfo(
            projectile=projectile_info,
            area=area_info,
            failure_bits=bits,
            world_failure_bits=world_bits,
            valid=valid,
        ),
    )


def _dense_launch(
    events,
    programs,
    launch_programs,
    compatible,
    event_kind,
    event_capacity,
):
    safe_source = jnp.clip(events.source_slot, 0, ENTITY_CAPACITY - 1)
    membership = jax.nn.one_hot(safe_source, ENTITY_CAPACITY, dtype=jnp.bool_)
    selected = (
        events.requested
        & (events.kind == event_kind)
        & (events.source_slot >= 0)
        & (events.source_slot < ENTITY_CAPACITY)
    )
    selected_membership = selected[..., None] & membership
    requested = jnp.any(selected_membership, axis=1)
    count = jnp.sum(selected_membership.astype(jnp.int32), axis=1)
    family = jnp.max(
        jnp.where(
            selected_membership,
            events.weapon_family[..., None],
            jnp.int32(0),
        ),
        axis=1,
    )
    ability = jnp.max(
        jnp.where(
            selected_membership,
            events.ability_slot[..., None],
            jnp.int32(-1),
        ),
        axis=1,
    )
    safe_family = jnp.clip(family, 0, launch_programs.event_mask.shape[0] - 1)
    safe_ability = jnp.clip(ability, 0, ABILITY_CAPACITY - 1)
    expected = jnp.sum(
        launch_programs.event_mask[safe_family, safe_ability].astype(jnp.int32),
        axis=2,
    )
    compatible_group = compatible[safe_family, safe_ability]
    full_mask = programs.event_mask[
        jnp.clip(
            events.weapon_family,
            0,
            programs.event_mask.shape[0] - 1,
        ),
        jnp.clip(events.ability_slot, 0, ABILITY_CAPACITY - 1),
    ]
    full_kind = programs.event_kind[
        jnp.clip(
            events.weapon_family,
            0,
            programs.event_kind.shape[0] - 1,
        ),
        jnp.clip(events.ability_slot, 0, ABILITY_CAPACITY - 1),
    ]
    rank = jnp.sum(
        (
            full_mask
            & (full_kind == event_kind)
            & (
                jnp.arange(EVENT_CAPACITY, dtype=jnp.int32)[None, None, :]
                < events.event_slot[..., None]
            )
        ).astype(jnp.int32),
        axis=2,
    )
    safe_rank = jnp.clip(rank, 0, event_capacity - 1)
    expected_mask = launch_programs.event_mask[
        jnp.clip(
            events.weapon_family,
            0,
            launch_programs.event_mask.shape[0] - 1,
        ),
        jnp.clip(events.ability_slot, 0, ABILITY_CAPACITY - 1),
        safe_rank,
    ]
    expected_f32 = launch_programs.event_f32[
        jnp.clip(
            events.weapon_family,
            0,
            launch_programs.event_f32.shape[0] - 1,
        ),
        jnp.clip(events.ability_slot, 0, ABILITY_CAPACITY - 1),
        safe_rank,
    ]
    expected_i32 = launch_programs.event_i32[
        jnp.clip(
            events.weapon_family,
            0,
            launch_programs.event_i32.shape[0] - 1,
        ),
        jnp.clip(events.ability_slot, 0, ABILITY_CAPACITY - 1),
        safe_rank,
    ]
    expected_flags = launch_programs.event_flags[
        jnp.clip(
            events.weapon_family,
            0,
            launch_programs.event_flags.shape[0] - 1,
        ),
        jnp.clip(events.ability_slot, 0, ABILITY_CAPACITY - 1),
        safe_rank,
    ]
    payload_matches = (
        expected_mask
        & jnp.all(events.f32 == expected_f32, axis=2)
        & jnp.all(events.i32 == expected_i32, axis=2)
        & (events.flags == expected_flags)
    )
    invalid_event = selected & (
        (rank < 0) | (rank >= event_capacity) | ~payload_matches
    )
    invalid = jnp.any(
        requested
        & (
            (family <= 0)
            | (ability < 0)
            | ~compatible_group
            | launch_programs.overflow[safe_family]
            | (count != expected)
        ),
        axis=1,
    ) | jnp.any(invalid_event, axis=1)
    return family, ability, requested, invalid


def _field(value, shape, dtype, name):
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}")


def _set_failure(bits, mask, code):
    return jnp.where(mask, bits | jnp.uint32(code), bits)


def _select_tree(mask, selected, fallback):
    return jax.tree_util.tree_map(
        lambda yes, no: jnp.where(
            mask.reshape((mask.shape[0],) + (1,) * (yes.ndim - 1)),
            yes,
            no,
        ),
        selected,
        fallback,
    )
