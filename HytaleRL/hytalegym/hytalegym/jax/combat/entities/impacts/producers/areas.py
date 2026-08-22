"""Vectorized 32-source persistent-area allocation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal import (
    ABILITY_CAPACITY,
    AREA_CAPACITY,
    AREA_GENERIC,
    AREA_NONE,
    ARSENAL_FAILURE_AREA_OVERFLOW,
    ARSENAL_FAILURE_INVALID_COMMAND,
    ARSENAL_FAILURE_INVALID_LOADOUT,
    ARSENAL_FAILURE_INVALID_STATE,
    ARSENAL_FAILURE_UNSUPPORTED_WORLD,
    EF_AIR_RESISTANCE,
    EF_AIR_RESISTANCE_MAX,
    EF_AREA_DURATION_SECONDS,
    EF_AREA_END_RADIUS,
    EF_AREA_HEIGHT,
    EF_AREA_INTERVAL_SECONDS,
    EF_AREA_RADIUS_CHANGE_SECONDS,
    EF_DAMAGE,
    EF_FORCE_MAGNITUDE,
    EF_FORCE_X,
    EF_FORCE_Y,
    EF_FORCE_Z,
    EF_GROUND_RESISTANCE,
    EF_GROUND_RESISTANCE_MAX,
    EF_RANDOM_PERCENTAGE,
    EF_RADIUS,
    EF_RESISTANCE_THRESHOLD,
    EF_STATUS_COOLDOWN_SECONDS,
    EF_STATUS_DAMAGE,
    EF_STATUS_DURATION_SECONDS,
    EF_STATUS_RESOURCE_DELTA,
    EF_STATUS_SPEED_MULTIPLIER,
    EI_DAMAGE_CAUSE,
    EI_DAMAGE_CLASS,
    EI_FORCE_MODE,
    EI_PROJECTILE_KIND,
    EI_RESISTANCE_STYLE,
    EI_STATUS_DAMAGE_CAUSE,
    EI_STATUS_ID,
    EI_STATUS_OVERLAP_MODE,
    EI_STATUS_RESOURCE_ID,
    EVENT_FLOAT_FEATURES,
    EVENT_INTEGER_FEATURES,
    EVENT_STATUS_FLAG_MASK,
    FORCE_ADD,
    FORCE_SET,
    ArsenalState,
)
from hytalegym.jax.combat.entities import (
    ENTITY_CAPACITY,
    EntityRoster,
    validate_roster_layout,
)
from hytalegym.jax.combat.entities.impacts.schema.contract import (
    AREAS_PER_LAUNCH,
    AREA_PROGRAM_FAMILY_CAPACITY,
)
from hytalegym.jax.combat.entities.impacts.schema.types import (
    EntityAreaLaunchCommands,
    EntityAreaLaunchInfo,
    EntityAreaLaunchWorld,
    EntityAreaProgramBank,
    EntityImpactBindingInfo,
    EntityImpactBindings,
)
from hytalegym.jax.combat.mechanics import (
    DAMAGE_CLASS_COUNT,
    DAMAGE_COUNT,
    RESOURCE_COUNT,
    STATUS_FLAG_DEBUFF,
    STATUS_FLAG_CONTROL_IMMUNITY_GATED,
    STATUS_FLAG_DISABLE_ABILITIES,
    STATUS_FLAG_DISABLE_MOVEMENT,
    STATUS_FLAG_DISABLE_SPRINT,
    STATUS_FLAG_IGNORE_KNOCKBACK,
    STATUS_FLAG_INVULNERABLE,
    STATUS_OVERLAP_IGNORE,
    STATUS_OVERLAP_OVERWRITE,
)


_STATUS_FLAG_MASK = (
    STATUS_FLAG_INVULNERABLE
    | STATUS_FLAG_DISABLE_MOVEMENT
    | STATUS_FLAG_DISABLE_ABILITIES
    | STATUS_FLAG_IGNORE_KNOCKBACK
    | STATUS_FLAG_DEBUFF
    | STATUS_FLAG_DISABLE_SPRINT
    | STATUS_FLAG_CONTROL_IMMUNITY_GATED
)


def launch_entity_areas(
    arsenal: ArsenalState,
    bindings: EntityImpactBindings,
    roster: EntityRoster,
    commands: EntityAreaLaunchCommands,
    programs: EntityAreaProgramBank,
    world: EntityAreaLaunchWorld,
) -> tuple[ArsenalState, EntityImpactBindings, EntityAreaLaunchInfo]:
    """Allocate every event-ready persistent area atomically per row."""

    validate_roster_layout(roster)
    batch = roster.active.shape[0]
    _validate_layout(arsenal, bindings, commands, programs, world, batch)
    family = jnp.clip(
        commands.weapon_family,
        0,
        AREA_PROGRAM_FAMILY_CAPACITY - 1,
    )
    ability = jnp.clip(commands.ability_slot, 0, ABILITY_CAPACITY - 1)
    event_mask = programs.event_mask[family, ability]
    event_f32 = programs.event_f32[family, ability]
    event_i32 = programs.event_i32[family, ability]
    event_flags = programs.event_flags[family, ability]
    count = jnp.sum(event_mask.astype(jnp.int32), axis=2)
    requested = commands.requested[..., None] & event_mask
    supported = (
        (commands.weapon_family > 0)
        & (commands.weapon_family < AREA_PROGRAM_FAMILY_CAPACITY)
        & (count > 0)
    )
    invalid_command = (
        ~commands.valid
        | (commands.failure_bits != jnp.uint32(0))
        | jnp.any(
            commands.requested
            & (
                ~jnp.isfinite(commands.damage_multiplier)
                | (commands.damage_multiplier <= 0.0)
            ),
            axis=1,
        )
    )
    invalid_program = jnp.any(
        commands.requested
        & (
            (commands.ability_slot < 0)
            | (commands.ability_slot >= ABILITY_CAPACITY)
            | ~supported
            | programs.overflow[family]
        ),
        axis=1,
    ) | jnp.any(
        requested & ~_valid_program(event_f32, event_i32, event_flags),
        axis=(1, 2),
    )
    source_valid = roster.active & ~roster.dead & (roster.generation != jnp.uint32(0))
    invalid_source = jnp.any(
        commands.requested
        & (~source_valid | (world.source_generation != roster.generation)),
        axis=1,
    )
    placement_valid = (
        world.area_valid
        & (world.failure_bits == jnp.uint32(0))
        & jnp.all(jnp.isfinite(world.area_center), axis=2)
        & jnp.isfinite(world.source_yaw_degrees)
    )
    invalid_world = jnp.any(commands.requested & ~placement_valid, axis=1)
    requested_count = jnp.sum(
        jnp.where(commands.requested, count, jnp.int32(0)),
        axis=1,
    )
    available = jnp.sum((~arsenal.areas.active).astype(jnp.int32), axis=1)
    bits = arsenal.failure_bits
    bits = _set_failure(
        bits,
        invalid_command,
        ARSENAL_FAILURE_INVALID_COMMAND,
    )
    bits = _set_failure(
        bits,
        invalid_program,
        ARSENAL_FAILURE_INVALID_LOADOUT,
    )
    bits = _set_failure(
        bits,
        invalid_source | bindings.overflow,
        ARSENAL_FAILURE_INVALID_STATE,
    )
    bits = _set_failure(
        bits,
        invalid_world,
        ARSENAL_FAILURE_UNSUPPORTED_WORLD,
    )
    bits = _set_failure(
        bits,
        requested_count > available,
        ARSENAL_FAILURE_AREA_OVERFLOW,
    )
    row_valid = bits == jnp.uint32(0)
    after, new_slots = _allocate(
        arsenal._replace(failure_bits=bits),
        requested,
        event_f32,
        event_i32,
        event_flags,
        commands.damage_multiplier,
        world,
        row_valid,
    )
    bound, binding_info = _bind_areas(
        bindings,
        after,
        roster,
        commands.friendly_fire,
        new_slots,
        row_valid,
    )
    world_bits = jnp.bitwise_or.reduce(
        jnp.where(
            commands.requested,
            world.failure_bits,
            jnp.uint32(0),
        ),
        axis=1,
    )
    valid = row_valid & binding_info.valid
    return (
        after,
        bound,
        EntityAreaLaunchInfo(
            area_requested=requested_count,
            area_spawned=jnp.sum(new_slots.astype(jnp.int32), axis=1),
            area_slots=new_slots,
            world_failure_bits=world_bits,
            failure_bits=bits,
            valid=valid,
            binding=binding_info,
        ),
    )


def _allocate(
    arsenal,
    requested,
    event_f32,
    event_i32,
    event_flags,
    damage_multiplier,
    world,
    row_valid,
):
    batch = requested.shape[0]
    source = jnp.broadcast_to(
        jnp.arange(ENTITY_CAPACITY, dtype=jnp.int32)[None, :, None],
        requested.shape,
    )
    center = jnp.broadcast_to(
        world.area_center[:, :, None, :],
        requested.shape + (3,),
    )
    yaw = jnp.broadcast_to(world.source_yaw_degrees[:, :, None], requested.shape)
    multiplier = jnp.broadcast_to(damage_multiplier[:, :, None], requested.shape)
    force = _rotate_local_direction(
        event_f32[..., [EF_FORCE_X, EF_FORCE_Y, EF_FORCE_Z]],
        yaw,
    )
    events = ENTITY_CAPACITY * AREAS_PER_LAUNCH
    flat_requested = requested.reshape((batch, events))
    flat_source = source.reshape((batch, events))
    flat_center = center.reshape((batch, events, 3))
    flat_force = force.reshape((batch, events, 3))
    flat_multiplier = multiplier.reshape((batch, events))
    flat_f32 = event_f32.reshape((batch, events, EVENT_FLOAT_FEATURES))
    flat_i32 = event_i32.reshape((batch, events, EVENT_INTEGER_FEATURES))
    flat_flags = event_flags.reshape((batch, events))
    free = ~arsenal.areas.active
    event_rank = jnp.cumsum(flat_requested.astype(jnp.int32), axis=1) - 1
    free_rank = jnp.cumsum(free.astype(jnp.int32), axis=1) - 1
    match = (
        row_valid[:, None, None]
        & free[:, :, None]
        & flat_requested[:, None, :]
        & (free_rank[:, :, None] == event_rank[:, None, :])
    )
    event_index = jnp.argmax(match, axis=2)
    selected = jnp.any(match, axis=2)
    source_value = _take_event(flat_source, event_index)
    center_value = _take_event(flat_center, event_index)
    force_value = _take_event(flat_force, event_index)
    multiplier_value = _take_event(flat_multiplier, event_index)
    f32 = _take_event(flat_f32, event_index)
    i32 = _take_event(flat_i32, event_index)
    flags = _take_event(flat_flags, event_index)
    area = arsenal.areas
    zero = jnp.zeros_like(f32[..., 0])
    candidate = area._replace(
        center=_write(area.center, center_value, selected),
        age_seconds=_write(area.age_seconds, zero, selected),
        duration_seconds=_write(
            area.duration_seconds,
            f32[..., EF_AREA_DURATION_SECONDS],
            selected,
        ),
        interval_seconds=_write(
            area.interval_seconds,
            f32[..., EF_AREA_INTERVAL_SECONDS],
            selected,
        ),
        interval_clock_seconds=_write(area.interval_clock_seconds, zero, selected),
        radius_change_seconds=_write(
            area.radius_change_seconds,
            f32[..., EF_AREA_RADIUS_CHANGE_SECONDS],
            selected,
        ),
        start_radius=_write(area.start_radius, f32[..., EF_RADIUS], selected),
        end_radius=_write(
            area.end_radius,
            f32[..., EF_AREA_END_RADIUS],
            selected,
        ),
        height=_write(area.height, f32[..., EF_AREA_HEIGHT], selected),
        damage=_write(
            area.damage,
            f32[..., EF_DAMAGE] * multiplier_value,
            selected,
        ),
        random_percentage=_write(
            area.random_percentage,
            f32[..., EF_RANDOM_PERCENTAGE],
            selected,
        ),
        damage_cause=_write(
            area.damage_cause,
            i32[..., EI_DAMAGE_CAUSE],
            selected,
        ),
        damage_class=_write(
            area.damage_class,
            i32[..., EI_DAMAGE_CLASS],
            selected,
        ),
        force_direction=_write(area.force_direction, force_value, selected),
        force_magnitude=_write(
            area.force_magnitude,
            f32[..., EF_FORCE_MAGNITUDE],
            selected,
        ),
        force_mode=_write(area.force_mode, i32[..., EI_FORCE_MODE], selected),
        air_resistance=_write(
            area.air_resistance,
            f32[..., EF_AIR_RESISTANCE],
            selected,
        ),
        air_resistance_max=_write(
            area.air_resistance_max,
            f32[..., EF_AIR_RESISTANCE_MAX],
            selected,
        ),
        ground_resistance=_write(
            area.ground_resistance,
            f32[..., EF_GROUND_RESISTANCE],
            selected,
        ),
        ground_resistance_max=_write(
            area.ground_resistance_max,
            f32[..., EF_GROUND_RESISTANCE_MAX],
            selected,
        ),
        resistance_threshold=_write(
            area.resistance_threshold,
            f32[..., EF_RESISTANCE_THRESHOLD],
            selected,
        ),
        resistance_style=_write(
            area.resistance_style,
            i32[..., EI_RESISTANCE_STYLE],
            selected,
        ),
        status_id=_write(area.status_id, i32[..., EI_STATUS_ID], selected),
        status_duration_seconds=_write(
            area.status_duration_seconds,
            f32[..., EF_STATUS_DURATION_SECONDS],
            selected,
        ),
        status_cooldown_seconds=_write(
            area.status_cooldown_seconds,
            f32[..., EF_STATUS_COOLDOWN_SECONDS],
            selected,
        ),
        status_damage=_write(
            area.status_damage,
            f32[..., EF_STATUS_DAMAGE],
            selected,
        ),
        status_damage_cause=_write(
            area.status_damage_cause,
            i32[..., EI_STATUS_DAMAGE_CAUSE],
            selected,
        ),
        status_resource_id=_write(
            area.status_resource_id,
            i32[..., EI_STATUS_RESOURCE_ID],
            selected,
        ),
        status_resource_delta=_write(
            area.status_resource_delta,
            f32[..., EF_STATUS_RESOURCE_DELTA],
            selected,
        ),
        status_speed_multiplier=_write(
            area.status_speed_multiplier,
            f32[..., EF_STATUS_SPEED_MULTIPLIER],
            selected,
        ),
        status_flags=_write(
            area.status_flags,
            flags & jnp.uint32(EVENT_STATUS_FLAG_MASK),
            selected,
        ),
        status_overlap_mode=_write(
            area.status_overlap_mode,
            i32[..., EI_STATUS_OVERLAP_MODE],
            selected,
        ),
        kind=_write(area.kind, i32[..., EI_PROJECTILE_KIND], selected),
        owner_entity_id=_write(area.owner_entity_id, source_value, selected),
        active=area.active | selected,
        entity_overlap_only=_write(
            area.entity_overlap_only,
            jnp.ones_like(selected),
            selected,
        ),
    )
    return arsenal._replace(areas=candidate), selected


def _bind_areas(
    bindings,
    arsenal,
    roster,
    friendly_fire,
    new_slots,
    row_valid,
):
    owner = jnp.clip(arsenal.areas.owner_entity_id, 0, ENTITY_CAPACITY - 1)
    generation = _gather(roster.generation, owner)
    area_friendly_fire = _gather(friendly_fire, owner)
    candidate = bindings._replace(
        area_source_generation=jnp.where(
            new_slots,
            generation,
            jnp.where(
                arsenal.areas.active,
                bindings.area_source_generation,
                jnp.uint32(0),
            ),
        ),
        area_friendly_fire=jnp.where(
            new_slots,
            area_friendly_fire,
            jnp.where(
                arsenal.areas.active,
                bindings.area_friendly_fire,
                True,
            ),
        ),
    )
    result = _select_tree(row_valid, candidate, bindings)
    projectile = arsenal.projectiles.active.shape
    entity = roster.active.shape
    return result, EntityImpactBindingInfo(
        projectile_bound=jnp.zeros(projectile, dtype=jnp.bool_),
        area_bound=new_slots & row_valid[:, None],
        crossbow_yaw_required=jnp.zeros(projectile, dtype=jnp.bool_),
        expected_ranged_projectiles=jnp.zeros(entity, dtype=jnp.int32),
        observed_ranged_projectiles=jnp.zeros(entity, dtype=jnp.int32),
        overflow=bindings.overflow,
        valid=row_valid & ~bindings.overflow,
    )


def _valid_program(f32, i32, event_flags):
    force = f32[..., [EF_FORCE_X, EF_FORCE_Y, EF_FORCE_Z]]
    force_length = jnp.linalg.norm(force, axis=3)
    status = i32[..., EI_STATUS_ID] > 0
    return (
        (i32[..., EI_PROJECTILE_KIND] > AREA_NONE)
        & (i32[..., EI_PROJECTILE_KIND] <= AREA_GENERIC)
        & jnp.isfinite(f32[..., EF_AREA_DURATION_SECONDS])
        & (f32[..., EF_AREA_DURATION_SECONDS] > 0.0)
        & jnp.isfinite(f32[..., EF_AREA_INTERVAL_SECONDS])
        & (f32[..., EF_AREA_INTERVAL_SECONDS] > 0.0)
        & jnp.isfinite(f32[..., EF_RADIUS])
        & (f32[..., EF_RADIUS] >= 0.0)
        & jnp.isfinite(f32[..., EF_AREA_END_RADIUS])
        & (f32[..., EF_AREA_END_RADIUS] >= 0.0)
        & jnp.isfinite(f32[..., EF_AREA_RADIUS_CHANGE_SECONDS])
        & (f32[..., EF_AREA_RADIUS_CHANGE_SECONDS] > 0.0)
        & jnp.isfinite(f32[..., EF_AREA_HEIGHT])
        & (f32[..., EF_AREA_HEIGHT] > 0.0)
        & jnp.isfinite(f32[..., EF_DAMAGE])
        & (f32[..., EF_DAMAGE] >= 0.0)
        & (i32[..., EI_DAMAGE_CAUSE] >= 0)
        & (i32[..., EI_DAMAGE_CAUSE] < DAMAGE_COUNT)
        & (i32[..., EI_DAMAGE_CLASS] >= 0)
        & (i32[..., EI_DAMAGE_CLASS] < DAMAGE_CLASS_COUNT)
        & jnp.isfinite(f32[..., EF_RANDOM_PERCENTAGE])
        & (f32[..., EF_RANDOM_PERCENTAGE] >= 0.0)
        & jnp.isfinite(f32[..., EF_FORCE_MAGNITUDE])
        & (f32[..., EF_FORCE_MAGNITUDE] >= 0.0)
        & ((f32[..., EF_FORCE_MAGNITUDE] == 0.0) | (force_length > 0.0))
        & (
            (i32[..., EI_FORCE_MODE] == FORCE_SET)
            | (i32[..., EI_FORCE_MODE] == FORCE_ADD)
        )
        & _unit_interval(f32[..., EF_AIR_RESISTANCE])
        & _unit_interval(f32[..., EF_AIR_RESISTANCE_MAX])
        & _unit_interval(f32[..., EF_GROUND_RESISTANCE])
        & _unit_interval(f32[..., EF_GROUND_RESISTANCE_MAX])
        & jnp.isfinite(f32[..., EF_RESISTANCE_THRESHOLD])
        & (f32[..., EF_RESISTANCE_THRESHOLD] > 0.0)
        & ((i32[..., EI_RESISTANCE_STYLE] == 0) | (i32[..., EI_RESISTANCE_STYLE] == 1))
        & (i32[..., EI_STATUS_ID] >= 0)
        & jnp.isfinite(f32[..., EF_STATUS_DURATION_SECONDS])
        & (f32[..., EF_STATUS_DURATION_SECONDS] >= 0.0)
        & (~status | (f32[..., EF_STATUS_DURATION_SECONDS] > 0.0))
        & jnp.isfinite(f32[..., EF_STATUS_COOLDOWN_SECONDS])
        & (f32[..., EF_STATUS_COOLDOWN_SECONDS] >= 0.0)
        & jnp.isfinite(f32[..., EF_STATUS_DAMAGE])
        & (f32[..., EF_STATUS_DAMAGE] >= 0.0)
        & (i32[..., EI_STATUS_DAMAGE_CAUSE] >= 0)
        & (i32[..., EI_STATUS_DAMAGE_CAUSE] < DAMAGE_COUNT)
        & (i32[..., EI_STATUS_RESOURCE_ID] >= -1)
        & (i32[..., EI_STATUS_RESOURCE_ID] < RESOURCE_COUNT)
        & jnp.isfinite(f32[..., EF_STATUS_RESOURCE_DELTA])
        & jnp.isfinite(f32[..., EF_STATUS_SPEED_MULTIPLIER])
        & (f32[..., EF_STATUS_SPEED_MULTIPLIER] > 0.0)
        & (
            (
                (event_flags & jnp.uint32(EVENT_STATUS_FLAG_MASK))
                & jnp.uint32(0xFFFFFFFF ^ _STATUS_FLAG_MASK)
            )
            == 0
        )
        & (i32[..., EI_STATUS_OVERLAP_MODE] >= STATUS_OVERLAP_IGNORE)
        & (i32[..., EI_STATUS_OVERLAP_MODE] <= STATUS_OVERLAP_OVERWRITE)
    )


def _validate_layout(
    arsenal,
    bindings,
    commands,
    programs,
    world,
    batch,
):
    entity = (batch, ENTITY_CAPACITY)
    area = (batch, AREA_CAPACITY)
    _field(arsenal.areas.active, area, jnp.bool_, "areas.active")
    _field(arsenal.failure_bits, (batch,), jnp.uint32, "failure_bits")
    _field(bindings.overflow, (batch,), jnp.bool_, "bindings.overflow")
    _field(
        bindings.area_source_generation,
        area,
        jnp.uint32,
        "bindings.area_source_generation",
    )
    _field(
        bindings.area_friendly_fire,
        area,
        jnp.bool_,
        "bindings.area_friendly_fire",
    )
    for name, dtype in (
        ("weapon_family", jnp.int32),
        ("ability_slot", jnp.int32),
        ("requested", jnp.bool_),
        ("damage_multiplier", jnp.float32),
        ("friendly_fire", jnp.bool_),
    ):
        _field(
            getattr(commands, name),
            entity,
            dtype,
            f"commands.{name}",
        )
    _field(
        commands.failure_bits,
        (batch,),
        jnp.uint32,
        "commands.failure_bits",
    )
    _field(commands.valid, (batch,), jnp.bool_, "commands.valid")
    program = (
        AREA_PROGRAM_FAMILY_CAPACITY,
        ABILITY_CAPACITY,
        AREAS_PER_LAUNCH,
    )
    _field(programs.event_mask, program, jnp.bool_, "programs.event_mask")
    _field(
        programs.event_f32,
        program + (EVENT_FLOAT_FEATURES,),
        jnp.float32,
        "programs.event_f32",
    )
    _field(
        programs.event_i32,
        program + (EVENT_INTEGER_FEATURES,),
        jnp.int32,
        "programs.event_i32",
    )
    _field(
        programs.event_flags,
        program,
        jnp.uint32,
        "programs.event_flags",
    )
    _field(
        programs.overflow,
        (AREA_PROGRAM_FAMILY_CAPACITY,),
        jnp.bool_,
        "programs.overflow",
    )
    _field(
        world.area_center,
        entity + (3,),
        jnp.float32,
        "world.area_center",
    )
    _field(
        world.source_yaw_degrees,
        entity,
        jnp.float32,
        "world.source_yaw_degrees",
    )
    _field(
        world.source_generation,
        entity,
        jnp.uint32,
        "world.source_generation",
    )
    _field(world.area_valid, entity, jnp.bool_, "world.area_valid")
    _field(
        world.failure_bits,
        entity,
        jnp.uint32,
        "world.failure_bits",
    )


def _rotate_local_direction(direction, yaw_degrees):
    length = jnp.linalg.norm(direction, axis=-1)
    local = direction / jnp.maximum(length[..., None], jnp.finfo(jnp.float32).tiny)
    yaw = jnp.deg2rad(yaw_degrees)
    return jnp.stack(
        (
            local[..., 0] * jnp.cos(yaw) + local[..., 2] * jnp.sin(yaw),
            local[..., 1],
            -local[..., 0] * jnp.sin(yaw) + local[..., 2] * jnp.cos(yaw),
        ),
        axis=-1,
    )


def _gather(array, slot):
    batch = jnp.arange(array.shape[0]).reshape(
        (array.shape[0],) + (1,) * (slot.ndim - 1)
    )
    return array[batch, slot]


def _take_event(array, index):
    shape = index.shape + (1,) * (array.ndim - 2)
    gather = jnp.broadcast_to(index.reshape(shape), index.shape + array.shape[2:])
    return jnp.take_along_axis(array, gather, axis=1)


def _write(current, value, selected):
    shape = selected.shape + (1,) * (current.ndim - 2)
    return jnp.where(selected.reshape(shape), value, current)


def _unit_interval(value):
    return jnp.isfinite(value) & (value >= 0.0) & (value <= 1.0)


def _set_failure(bits, mask, code):
    return jnp.where(mask, bits | jnp.uint32(code), bits)


def _field(value, shape, dtype, name):
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}")


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
