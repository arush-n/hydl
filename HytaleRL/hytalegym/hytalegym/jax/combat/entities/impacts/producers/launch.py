"""Vectorized 32-source ranged-controller projectile allocation."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal import (
    ABILITY_CAPACITY,
    ARSENAL_FAILURE_INVALID_COMMAND,
    ARSENAL_FAILURE_INVALID_LOADOUT,
    ARSENAL_FAILURE_INVALID_STATE,
    ARSENAL_FAILURE_PROJECTILE_OVERFLOW,
    ARSENAL_FAILURE_UNSUPPORTED_WORLD,
    EF_AIR_RESISTANCE,
    EF_AIR_RESISTANCE_MAX,
    EF_DAMAGE,
    EF_FALLOFF,
    EF_FORCE_MAGNITUDE,
    EF_FORCE_X,
    EF_FORCE_Y,
    EF_FORCE_Z,
    EF_GROUND_RESISTANCE,
    EF_GROUND_RESISTANCE_MAX,
    EF_ON_HIT_RESOURCE_DELTA,
    EF_PITCH_OFFSET_DEGREES,
    EF_PROJECTILE_FUSE_SECONDS,
    EF_PROJECTILE_DEAD_TIME_SECONDS,
    EF_PROJECTILE_DIRECT_DAMAGE,
    EF_PROJECTILE_GRAVITY,
    EF_PROJECTILE_HALF_EXTENT,
    EF_PROJECTILE_LIFETIME_SECONDS,
    EF_PROJECTILE_SPAWN_OFFSET_X,
    EF_PROJECTILE_SPAWN_OFFSET_Y,
    EF_PROJECTILE_SPAWN_OFFSET_Z,
    EF_PROJECTILE_SPEED,
    EF_PROJECTILE_TERMINAL_VELOCITY,
    EF_RANDOM_PERCENTAGE,
    EF_RADIUS,
    EF_RESISTANCE_THRESHOLD,
    EF_STATUS_COOLDOWN_SECONDS,
    EF_STATUS_DAMAGE,
    EF_STATUS_DURATION_SECONDS,
    EF_STATUS_RESOURCE_DELTA,
    EF_STATUS_SPEED_MULTIPLIER,
    EF_YAW_OFFSET_DEGREES,
    EI_DAMAGE_CAUSE,
    EI_DAMAGE_CLASS,
    EI_FORCE_DIRECTION_MODE,
    EI_FORCE_MODE,
    EI_ON_HIT_RESOURCE_ID,
    EI_PROJECTILE_KIND,
    EI_PROJECTILE_DIRECT_DAMAGE_CAUSE,
    EI_RESISTANCE_STYLE,
    EI_STATUS_DAMAGE_CAUSE,
    EI_STATUS_ID,
    EI_STATUS_OVERLAP_MODE,
    EI_STATUS_RESOURCE_ID,
    EVENT_FLOAT_FEATURES,
    EVENT_INTEGER_FEATURES,
    EVENT_STATUS_FLAG_MASK,
    FORCE_ADD,
    FORCE_DIRECTION_LOCAL,
    FORCE_DIRECTION_POINT,
    FORCE_SET,
    PROJECTILE_BLUNDERBUSS_BULLET,
    PROJECTILE_CAPACITY,
    ArsenalState,
    projectile_spawn_offset,
)
from hytalegym.jax.combat.controllers import (
    RangedControllerInfo,
    RangedControllerRules,
)
from hytalegym.jax.combat.entities import (
    ENTITY_CAPACITY,
    EntityRoster,
    validate_roster_layout,
)
from hytalegym.jax.combat.entities.impacts.producers.binding import (
    bind_arsenal_launches,
)
from hytalegym.jax.combat.entities.impacts.schema.contract import (
    RANGED_PROGRAM_FAMILY_CAPACITY,
    RANGED_PROJECTILES_PER_LAUNCH,
)
from hytalegym.jax.combat.entities.impacts.schema.types import (
    EntityImpactBindings,
    EntityProjectileLaunchInfo,
    EntityProjectileLaunchCommands,
    EntityProjectileLaunchWorld,
    RangedProjectileProgramBank,
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


def launch_entity_projectiles(
    arsenal: ArsenalState,
    bindings: EntityImpactBindings,
    roster: EntityRoster,
    commands: EntityProjectileLaunchCommands,
    programs: RangedProjectileProgramBank,
    world: EntityProjectileLaunchWorld,
) -> tuple[ArsenalState, EntityImpactBindings, EntityProjectileLaunchInfo]:
    """Allocate an event-ready projectile ability for any pinned family."""

    batch = roster.active.shape[0]
    entity = (batch, ENTITY_CAPACITY)
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
    family = jnp.clip(
        commands.weapon_family,
        0,
        RANGED_PROGRAM_FAMILY_CAPACITY - 1,
    )
    ability = jnp.clip(commands.ability_slot, 0, ABILITY_CAPACITY - 1)
    count = jnp.sum(
        programs.event_mask[family, ability].astype(jnp.int32),
        axis=2,
    )
    supported = (
        (commands.weapon_family > 0)
        & (commands.weapon_family < RANGED_PROGRAM_FAMILY_CAPACITY)
        & (count > 0)
    )
    zeros_i32 = jnp.zeros(entity, dtype=jnp.int32)
    zeros_f32 = jnp.zeros(entity, dtype=jnp.float32)
    rules = RangedControllerRules(
        weapon_family=commands.weapon_family,
        active=supported,
        max_ammo=zeros_i32,
        max_durability=zeros_f32,
        durability_loss_per_projectile=zeros_f32,
        signature_energy_cost=zeros_f32,
    )
    info = RangedControllerInfo(
        ability_slot=commands.ability_slot,
        ability_requested=commands.requested,
        charge_level=jnp.full(entity, -1, dtype=jnp.int32),
        projectile_count=jnp.where(commands.requested, count, jnp.int32(0)),
        projectile_damage_multiplier=commands.damage_multiplier,
        signature_activated=jnp.zeros(entity, dtype=jnp.bool_),
        ammo_loaded=zeros_i32,
        arrows_removed=zeros_i32,
        arrows_returned=zeros_i32,
        arrows_dropped=zeros_i32,
        swap_accepted=jnp.zeros(entity, dtype=jnp.bool_),
        swap_blocked=jnp.zeros(entity, dtype=jnp.bool_),
        failure_bits=commands.failure_bits,
        valid=commands.valid,
    )
    return launch_ranged_controller_projectiles(
        arsenal,
        bindings,
        roster,
        rules,
        info,
        programs,
        world,
        commands.friendly_fire,
    )


def launch_ranged_controller_projectiles(
    arsenal: ArsenalState,
    bindings: EntityImpactBindings,
    roster: EntityRoster,
    controller_rules: RangedControllerRules,
    controller_info: RangedControllerInfo,
    programs: RangedProjectileProgramBank,
    world: EntityProjectileLaunchWorld,
    source_friendly_fire: jax.Array,
) -> tuple[ArsenalState, EntityImpactBindings, EntityProjectileLaunchInfo]:
    """Allocate every controller-requested projectile atomically per row."""

    validate_roster_layout(roster)
    batch = roster.active.shape[0]
    _validate_layout(
        arsenal,
        controller_rules,
        controller_info,
        programs,
        world,
        source_friendly_fire,
        batch,
    )
    family = controller_rules.weapon_family
    family_index = jnp.clip(family, 0, RANGED_PROGRAM_FAMILY_CAPACITY - 1)
    ability = controller_info.ability_slot
    ability_index = jnp.clip(ability, 0, ABILITY_CAPACITY - 1)
    event_mask = programs.event_mask[family_index, ability_index]
    event_f32 = programs.event_f32[family_index, ability_index]
    event_i32 = programs.event_i32[family_index, ability_index]
    event_flags = programs.event_flags[family_index, ability_index]
    source_requested = controller_info.ability_requested
    launch_requested = source_requested & (controller_info.projectile_count > 0)
    requested = source_requested[..., None] & event_mask
    program_count = jnp.sum(event_mask.astype(jnp.int32), axis=2)
    supported_family = (family > 0) & (family < RANGED_PROGRAM_FAMILY_CAPACITY)
    invalid_controller = (
        ~controller_info.valid
        | (controller_info.failure_bits != jnp.uint32(0))
        | jnp.any(
            (~source_requested) & (controller_info.projectile_count != 0),
            axis=1,
        )
        | jnp.any(
            source_requested
            & (
                ~jnp.isfinite(controller_info.projectile_damage_multiplier)
                | (controller_info.projectile_damage_multiplier <= 0.0)
            ),
            axis=1,
        )
    )
    invalid_program = jnp.any(
        source_requested
        & (
            (ability < 0)
            | (ability >= ABILITY_CAPACITY)
            | ~supported_family
            | ~controller_rules.active
            | programs.overflow[family_index]
            | (program_count != controller_info.projectile_count)
        ),
        axis=1,
    ) | jnp.any(
        requested & ~_valid_program(event_f32, event_i32, event_flags),
        axis=(1, 2),
    )
    source_valid = roster.active & ~roster.dead & (roster.generation != jnp.uint32(0))
    invalid_source = jnp.any(
        launch_requested
        & (~source_valid | (world.source_generation != roster.generation)),
        axis=1,
    )
    transform_valid = (
        world.muzzle_valid
        & (world.failure_bits == jnp.uint32(0))
        & jnp.all(jnp.isfinite(world.muzzle_position), axis=2)
        & jnp.isfinite(world.muzzle_yaw_degrees)
        & jnp.isfinite(world.muzzle_pitch_degrees)
    )
    invalid_world = jnp.any(launch_requested & ~transform_valid, axis=1)
    requested_count = jnp.sum(
        jnp.where(
            source_requested,
            controller_info.projectile_count,
            jnp.int32(0),
        ),
        axis=1,
    )
    available = jnp.sum((~arsenal.projectiles.active).astype(jnp.int32), axis=1)
    bits = arsenal.failure_bits
    bits = _set_failure(
        bits,
        invalid_controller,
        ARSENAL_FAILURE_INVALID_COMMAND,
    )
    bits = _set_failure(
        bits,
        invalid_program,
        ARSENAL_FAILURE_INVALID_LOADOUT,
    )
    bits = _set_failure(
        bits,
        invalid_source,
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
        ARSENAL_FAILURE_PROJECTILE_OVERFLOW,
    )
    row_valid = bits == jnp.uint32(0)
    after, new_slots = _allocate(
        arsenal._replace(failure_bits=bits),
        requested,
        event_f32,
        event_i32,
        event_flags,
        world,
        row_valid,
    )
    bound, binding_info = bind_arsenal_launches(
        bindings,
        arsenal,
        after,
        roster,
        controller_rules,
        controller_info,
        source_friendly_fire,
    )
    world_bits = jnp.bitwise_or.reduce(
        jnp.where(
            launch_requested,
            world.failure_bits,
            jnp.uint32(0),
        ),
        axis=1,
    )
    valid = row_valid & binding_info.valid
    return (
        after,
        bound,
        EntityProjectileLaunchInfo(
            projectile_requested=requested_count,
            projectile_spawned=jnp.sum(new_slots.astype(jnp.int32), axis=1),
            projectile_slots=new_slots,
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
    world,
    row_valid,
):
    batch = requested.shape[0]
    source = jnp.broadcast_to(
        jnp.arange(ENTITY_CAPACITY, dtype=jnp.int32)[None, :, None],
        requested.shape,
    )
    yaw = jnp.broadcast_to(world.muzzle_yaw_degrees[:, :, None], requested.shape)
    pitch = jnp.broadcast_to(world.muzzle_pitch_degrees[:, :, None], requested.shape)
    launch_yaw = yaw + event_f32[..., EF_YAW_OFFSET_DEGREES]
    launch_pitch = pitch + event_f32[..., EF_PITCH_OFFSET_DEGREES]
    heading = _look_direction(launch_yaw, launch_pitch)
    spawn_offset = projectile_spawn_offset(
        event_f32[
            ...,
            [
                EF_PROJECTILE_SPAWN_OFFSET_X,
                EF_PROJECTILE_SPAWN_OFFSET_Y,
                EF_PROJECTILE_SPAWN_OFFSET_Z,
            ],
        ],
        launch_yaw,
        launch_pitch,
        event_flags,
    )
    position = (
        jnp.broadcast_to(
            world.muzzle_position[:, :, None, :],
            requested.shape + (3,),
        )
        + spawn_offset
    )
    force = _rotate_local_direction(
        event_f32[..., [EF_FORCE_X, EF_FORCE_Y, EF_FORCE_Z]],
        yaw,
    )
    event_count = ENTITY_CAPACITY * RANGED_PROJECTILES_PER_LAUNCH
    flat_requested = requested.reshape((batch, event_count))
    flat_source = source.reshape((batch, event_count))
    flat_position = position.reshape((batch, event_count, 3))
    flat_heading = heading.reshape((batch, event_count, 3))
    flat_launch_yaw = launch_yaw.reshape((batch, event_count))
    flat_force = force.reshape((batch, event_count, 3))
    flat_f32 = event_f32.reshape((batch, event_count, EVENT_FLOAT_FEATURES))
    flat_i32 = event_i32.reshape((batch, event_count, EVENT_INTEGER_FEATURES))
    flat_flags = event_flags.reshape((batch, event_count))
    free = ~arsenal.projectiles.active
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
    position_value = _take_event(flat_position, event_index)
    heading_value = _take_event(flat_heading, event_index)
    launch_yaw_value = _take_event(flat_launch_yaw, event_index)
    force_value = _take_event(flat_force, event_index)
    f32 = _take_event(flat_f32, event_index)
    i32 = _take_event(flat_i32, event_index)
    flags = _take_event(flat_flags, event_index)
    projectile = arsenal.projectiles
    candidate = projectile._replace(
        position=_write(projectile.position, position_value, selected),
        velocity=_write(
            projectile.velocity,
            heading_value * f32[..., EF_PROJECTILE_SPEED, None],
            selected,
        ),
        launch_yaw_degrees=_write(
            projectile.launch_yaw_degrees,
            launch_yaw_value,
            selected,
        ),
        half_extent=_write(
            projectile.half_extent,
            jnp.broadcast_to(
                f32[..., EF_PROJECTILE_HALF_EXTENT, None],
                selected.shape + (3,),
            ),
            selected,
        ),
        age_seconds=_write(
            projectile.age_seconds,
            jnp.zeros_like(f32[..., 0]),
            selected,
        ),
        lifetime_seconds=_write(
            projectile.lifetime_seconds,
            f32[..., EF_PROJECTILE_LIFETIME_SECONDS],
            selected,
        ),
        damage=_write(projectile.damage, f32[..., EF_DAMAGE], selected),
        direct_damage=_write(
            projectile.direct_damage,
            f32[..., EF_PROJECTILE_DIRECT_DAMAGE],
            selected,
        ),
        random_percentage=_write(
            projectile.random_percentage,
            f32[..., EF_RANDOM_PERCENTAGE],
            selected,
        ),
        damage_cause=_write(
            projectile.damage_cause,
            i32[..., EI_DAMAGE_CAUSE],
            selected,
        ),
        direct_damage_cause=_write(
            projectile.direct_damage_cause,
            i32[..., EI_PROJECTILE_DIRECT_DAMAGE_CAUSE],
            selected,
        ),
        damage_class=_write(
            projectile.damage_class,
            i32[..., EI_DAMAGE_CLASS],
            selected,
        ),
        gravity=_write(
            projectile.gravity,
            f32[..., EF_PROJECTILE_GRAVITY],
            selected,
        ),
        terminal_velocity=_write(
            projectile.terminal_velocity,
            f32[..., EF_PROJECTILE_TERMINAL_VELOCITY],
            selected,
        ),
        fuse_seconds=_write(
            projectile.fuse_seconds,
            f32[..., EF_PROJECTILE_FUSE_SECONDS],
            selected,
        ),
        dead_time_seconds=_write(
            projectile.dead_time_seconds,
            f32[..., EF_PROJECTILE_DEAD_TIME_SECONDS],
            selected,
        ),
        dead_time_remaining=_write(
            projectile.dead_time_remaining,
            jnp.zeros_like(f32[..., 0]),
            selected,
        ),
        explosion_radius=_write(
            projectile.explosion_radius,
            f32[..., EF_RADIUS],
            selected,
        ),
        explosion_falloff=_write(
            projectile.explosion_falloff,
            f32[..., EF_FALLOFF],
            selected,
        ),
        force_direction=_write(projectile.force_direction, force_value, selected),
        force_direction_mode=_write(
            projectile.force_direction_mode,
            i32[..., EI_FORCE_DIRECTION_MODE],
            selected,
        ),
        force_velocity_y=_write(
            projectile.force_velocity_y,
            f32[..., EF_FORCE_Y],
            selected,
        ),
        force_magnitude=_write(
            projectile.force_magnitude,
            f32[..., EF_FORCE_MAGNITUDE],
            selected,
        ),
        force_mode=_write(
            projectile.force_mode,
            i32[..., EI_FORCE_MODE],
            selected,
        ),
        air_resistance=_write(
            projectile.air_resistance,
            f32[..., EF_AIR_RESISTANCE],
            selected,
        ),
        air_resistance_max=_write(
            projectile.air_resistance_max,
            f32[..., EF_AIR_RESISTANCE_MAX],
            selected,
        ),
        ground_resistance=_write(
            projectile.ground_resistance,
            f32[..., EF_GROUND_RESISTANCE],
            selected,
        ),
        ground_resistance_max=_write(
            projectile.ground_resistance_max,
            f32[..., EF_GROUND_RESISTANCE_MAX],
            selected,
        ),
        resistance_threshold=_write(
            projectile.resistance_threshold,
            f32[..., EF_RESISTANCE_THRESHOLD],
            selected,
        ),
        resistance_style=_write(
            projectile.resistance_style,
            i32[..., EI_RESISTANCE_STYLE],
            selected,
        ),
        on_hit_resource_id=_write(
            projectile.on_hit_resource_id,
            i32[..., EI_ON_HIT_RESOURCE_ID],
            selected,
        ),
        on_hit_resource_delta=_write(
            projectile.on_hit_resource_delta,
            f32[..., EF_ON_HIT_RESOURCE_DELTA],
            selected,
        ),
        status_id=_write(
            projectile.status_id,
            i32[..., EI_STATUS_ID],
            selected,
        ),
        status_duration_seconds=_write(
            projectile.status_duration_seconds,
            f32[..., EF_STATUS_DURATION_SECONDS],
            selected,
        ),
        status_cooldown_seconds=_write(
            projectile.status_cooldown_seconds,
            f32[..., EF_STATUS_COOLDOWN_SECONDS],
            selected,
        ),
        status_damage=_write(
            projectile.status_damage,
            f32[..., EF_STATUS_DAMAGE],
            selected,
        ),
        status_damage_cause=_write(
            projectile.status_damage_cause,
            i32[..., EI_STATUS_DAMAGE_CAUSE],
            selected,
        ),
        status_resource_id=_write(
            projectile.status_resource_id,
            i32[..., EI_STATUS_RESOURCE_ID],
            selected,
        ),
        status_resource_delta=_write(
            projectile.status_resource_delta,
            f32[..., EF_STATUS_RESOURCE_DELTA],
            selected,
        ),
        status_speed_multiplier=_write(
            projectile.status_speed_multiplier,
            f32[..., EF_STATUS_SPEED_MULTIPLIER],
            selected,
        ),
        status_flags=_write(
            projectile.status_flags,
            flags & jnp.uint32(EVENT_STATUS_FLAG_MASK),
            selected,
        ),
        status_overlap_mode=_write(
            projectile.status_overlap_mode,
            i32[..., EI_STATUS_OVERLAP_MODE],
            selected,
        ),
        kind=_write(projectile.kind, i32[..., EI_PROJECTILE_KIND], selected),
        owner_entity_id=_write(projectile.owner_entity_id, source_value, selected),
        active=projectile.active | selected,
        impacted=_write(
            projectile.impacted,
            jnp.zeros_like(selected),
            selected,
        ),
        physics_initialized=_write(
            projectile.physics_initialized,
            jnp.zeros_like(selected),
            selected,
        ),
        entity_collision_only=_write(
            projectile.entity_collision_only,
            jnp.ones_like(selected),
            selected,
        ),
    )
    return arsenal._replace(projectiles=candidate), selected


def _valid_program(f32, i32, event_flags):
    force = f32[..., [EF_FORCE_X, EF_FORCE_Y, EF_FORCE_Z]]
    force_length = jnp.linalg.norm(force, axis=3)
    flags = i32[..., EI_STATUS_ID] > 0
    return (
        (i32[..., EI_PROJECTILE_KIND] > 0)
        & (i32[..., EI_PROJECTILE_KIND] <= PROJECTILE_BLUNDERBUSS_BULLET)
        & jnp.isfinite(f32[..., EF_DAMAGE])
        & (f32[..., EF_DAMAGE] >= 0.0)
        & jnp.isfinite(f32[..., EF_PROJECTILE_DIRECT_DAMAGE])
        & (f32[..., EF_PROJECTILE_DIRECT_DAMAGE] >= 0.0)
        & (
            (f32[..., EF_PROJECTILE_DIRECT_DAMAGE] == 0.0)
            | (
                (i32[..., EI_PROJECTILE_DIRECT_DAMAGE_CAUSE] >= 0)
                & (i32[..., EI_PROJECTILE_DIRECT_DAMAGE_CAUSE] < DAMAGE_COUNT)
            )
        )
        & (i32[..., EI_DAMAGE_CAUSE] >= 0)
        & (i32[..., EI_DAMAGE_CAUSE] < DAMAGE_COUNT)
        & (i32[..., EI_DAMAGE_CLASS] >= 0)
        & (i32[..., EI_DAMAGE_CLASS] < DAMAGE_CLASS_COUNT)
        & jnp.isfinite(f32[..., EF_RANDOM_PERCENTAGE])
        & (f32[..., EF_RANDOM_PERCENTAGE] >= 0.0)
        & jnp.isfinite(f32[..., EF_PROJECTILE_SPEED])
        & (f32[..., EF_PROJECTILE_SPEED] > 0.0)
        & jnp.isfinite(f32[..., EF_PROJECTILE_GRAVITY])
        & (f32[..., EF_PROJECTILE_GRAVITY] >= 0.0)
        & jnp.isfinite(f32[..., EF_PROJECTILE_TERMINAL_VELOCITY])
        & (f32[..., EF_PROJECTILE_TERMINAL_VELOCITY] > 0.0)
        & jnp.isfinite(f32[..., EF_PROJECTILE_HALF_EXTENT])
        & (f32[..., EF_PROJECTILE_HALF_EXTENT] > 0.0)
        & jnp.isfinite(f32[..., EF_PROJECTILE_LIFETIME_SECONDS])
        & (f32[..., EF_PROJECTILE_LIFETIME_SECONDS] > 0.0)
        & jnp.isfinite(f32[..., EF_PROJECTILE_FUSE_SECONDS])
        & (f32[..., EF_PROJECTILE_FUSE_SECONDS] >= 0.0)
        & jnp.isfinite(f32[..., EF_PROJECTILE_DEAD_TIME_SECONDS])
        & (f32[..., EF_PROJECTILE_DEAD_TIME_SECONDS] >= -1.0)
        & jnp.all(
            jnp.isfinite(
                f32[
                    ...,
                    [
                        EF_PROJECTILE_SPAWN_OFFSET_X,
                        EF_PROJECTILE_SPAWN_OFFSET_Y,
                        EF_PROJECTILE_SPAWN_OFFSET_Z,
                    ],
                ]
            ),
            axis=3,
        )
        & jnp.isfinite(f32[..., EF_RADIUS])
        & (f32[..., EF_RADIUS] >= 0.0)
        & jnp.isfinite(f32[..., EF_FALLOFF])
        & (f32[..., EF_FALLOFF] >= 0.0)
        & jnp.isfinite(f32[..., EF_FORCE_MAGNITUDE])
        & (f32[..., EF_FORCE_MAGNITUDE] >= 0.0)
        & (
            (f32[..., EF_FORCE_MAGNITUDE] == 0.0)
            | (i32[..., EI_FORCE_DIRECTION_MODE] == FORCE_DIRECTION_POINT)
            | (force_length > 0.0)
        )
        & (
            (i32[..., EI_FORCE_DIRECTION_MODE] == FORCE_DIRECTION_LOCAL)
            | (i32[..., EI_FORCE_DIRECTION_MODE] == FORCE_DIRECTION_POINT)
        )
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
        & (i32[..., EI_ON_HIT_RESOURCE_ID] >= -1)
        & (i32[..., EI_ON_HIT_RESOURCE_ID] < RESOURCE_COUNT)
        & jnp.isfinite(f32[..., EF_ON_HIT_RESOURCE_DELTA])
        & (i32[..., EI_STATUS_ID] >= 0)
        & jnp.isfinite(f32[..., EF_STATUS_DURATION_SECONDS])
        & (f32[..., EF_STATUS_DURATION_SECONDS] >= 0.0)
        & (~flags | (f32[..., EF_STATUS_DURATION_SECONDS] > 0.0))
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


def _unit_interval(value):
    return jnp.isfinite(value) & (value >= 0.0) & (value <= 1.0)


def _look_direction(yaw_degrees, pitch_degrees):
    yaw = jnp.deg2rad(yaw_degrees)
    pitch = jnp.deg2rad(pitch_degrees)
    horizontal = jnp.cos(pitch)
    return jnp.stack(
        (
            -jnp.sin(yaw) * horizontal,
            jnp.sin(pitch),
            -jnp.cos(yaw) * horizontal,
        ),
        axis=-1,
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


def _take_event(array, index):
    shape = index.shape + (1,) * (array.ndim - 2)
    gather = jnp.broadcast_to(index.reshape(shape), index.shape + array.shape[2:])
    return jnp.take_along_axis(array, gather, axis=1)


def _write(current, value, selected):
    shape = selected.shape + (1,) * (current.ndim - 2)
    return jnp.where(selected.reshape(shape), value, current)


def _set_failure(bits, mask, code):
    return jnp.where(mask, bits | jnp.uint32(code), bits)


def _validate_layout(
    arsenal,
    rules,
    info,
    programs,
    world,
    friendly_fire,
    batch,
):
    entity = (batch, ENTITY_CAPACITY)
    projectile = (batch, PROJECTILE_CAPACITY)
    _field(
        arsenal.projectiles.active,
        projectile,
        jnp.bool_,
        "arsenal.projectiles.active",
    )
    _field(arsenal.failure_bits, (batch,), jnp.uint32, "failure_bits")
    for name, dtype in (
        ("weapon_family", jnp.int32),
        ("active", jnp.bool_),
    ):
        _field(getattr(rules, name), entity, dtype, f"rules.{name}")
    for name, dtype in (
        ("ability_slot", jnp.int32),
        ("ability_requested", jnp.bool_),
        ("projectile_count", jnp.int32),
        ("projectile_damage_multiplier", jnp.float32),
    ):
        _field(getattr(info, name), entity, dtype, f"info.{name}")
    _field(info.failure_bits, (batch,), jnp.uint32, "info.failure_bits")
    _field(info.valid, (batch,), jnp.bool_, "info.valid")
    program = (
        RANGED_PROGRAM_FAMILY_CAPACITY,
        ABILITY_CAPACITY,
        RANGED_PROJECTILES_PER_LAUNCH,
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
        (RANGED_PROGRAM_FAMILY_CAPACITY,),
        jnp.bool_,
        "programs.overflow",
    )
    _field(
        world.muzzle_position,
        entity + (3,),
        jnp.float32,
        "world.muzzle_position",
    )
    _field(
        world.muzzle_yaw_degrees,
        entity,
        jnp.float32,
        "world.muzzle_yaw_degrees",
    )
    _field(
        world.muzzle_pitch_degrees,
        entity,
        jnp.float32,
        "world.muzzle_pitch_degrees",
    )
    _field(
        world.source_generation,
        entity,
        jnp.uint32,
        "world.source_generation",
    )
    _field(
        world.muzzle_valid,
        entity,
        jnp.bool_,
        "world.muzzle_valid",
    )
    _field(
        world.failure_bits,
        entity,
        jnp.uint32,
        "world.failure_bits",
    )
    _field(
        friendly_fire,
        entity,
        jnp.bool_,
        "source_friendly_fire",
    )


def _field(value, shape, dtype, name):
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != jnp.dtype(dtype):
        raise TypeError(f"{name} must have dtype {jnp.dtype(dtype)}")
