"""Fixed-capacity entity-only area and deployable-effect kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.contract import (
    ARSENAL_FAILURE_AREA_OVERFLOW,
    ARSENAL_FAILURE_EVENT_OVERFLOW,
    ARSENAL_FAILURE_INVALID_LOADOUT,
    ARSENAL_FAILURE_INVALID_STATE,
    ARSENAL_FAILURE_UNSUPPORTED_WORLD,
    AREA_DEPLOYABLE_AOE,
    AREA_SHAPE_CYLINDER,
    AREA_SHAPE_NONE,
    AREA_SHAPE_SPHERE,
    DEPLOYABLE_ATTACK_FLAG_MASK,
    AREA_TARGET_ENEMIES,
    AREA_TARGET_MASK,
    AREA_TARGET_TEAM,
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
    EF_STATUS_HEALING,
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
    EVENT_AREA,
    EVENT_FLAG_AREA_FRIENDLY_FIRE,
    EVENT_STATUS_FLAG_MASK,
    TERMINAL_UNKNOWN_STATUS_FLAG_MASK,
)
from hytalegym.jax.combat.arsenal.effects.events import (
    _flatten_events,
    pack_status_applications,
)
from hytalegym.jax.combat.arsenal.effects.impact_events import (
    empty_dense_damage_events,
    native_deployable_attack_mask,
    pack_dense_damage_events,
    typed_relationship_target_mask,
)
from hytalegym.jax.combat.arsenal.schema.types import (
    AreaState,
    ArsenalState,
    ArsenalWorldCapabilities,
    FiredEvents,
    ProjectileState,
)
from hytalegym.jax.combat.mechanics import (
    CombatMechanicsRules,
    DAMAGE_CLASS_COUNT,
    DAMAGE_COUNT,
    RESOURCE_COUNT,
    STATUS_OVERLAP_IGNORE,
    STATUS_OVERLAP_OVERWRITE,
    empty_damage_events,
    empty_status_applications,
)
from hytalegym.jax.combat.types import CombatParams, CombatState


_UNKNOWN_EVENT_STATUS_FLAG_MASK = 0xFFFFFFFF ^ EVENT_STATUS_FLAG_MASK


def spawn_areas(
    state: ArsenalState,
    events: FiredEvents,
    world: ArsenalWorldCapabilities,
) -> tuple[ArsenalState, jax.Array]:
    """Spawn injected, static area placements atomically per row."""

    flat = _flatten_events(events)
    requested = flat.requested & (flat.kind == EVENT_AREA)
    entity_count = world.area_center.shape[1]
    source = jnp.clip(flat.source_entity_id, 0, entity_count - 1)
    center = _gather_entity(world.area_center, source)
    placement = _gather_entity(world.static_area_placement, source)
    entity_only = _gather_entity(world.entity_only_area, source)
    yaw = _gather_entity(world.muzzle_yaw_degrees, source)
    f32, i32 = flat.f32, flat.i32
    valid_program = (
        (i32[..., EI_PROJECTILE_KIND] > 0)
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
        & jnp.isfinite(f32[..., EF_RANDOM_PERCENTAGE])
        & (f32[..., EF_RANDOM_PERCENTAGE] >= 0.0)
        & (i32[..., EI_DAMAGE_CLASS] >= 0)
        & (i32[..., EI_DAMAGE_CLASS] < DAMAGE_CLASS_COUNT)
    )
    available = jnp.sum((~state.areas.active).astype(jnp.int32), axis=1)
    count = jnp.sum(requested.astype(jnp.int32), axis=1)
    bits = state.failure_bits
    bits = _set_failure(
        bits,
        jnp.any(requested & ~valid_program, axis=1),
        ARSENAL_FAILURE_INVALID_LOADOUT,
    )
    bits = _set_failure(
        bits,
        jnp.any(
            requested
            & ~(placement & entity_only & jnp.all(jnp.isfinite(center), axis=2)),
            axis=1,
        ),
        ARSENAL_FAILURE_UNSUPPORTED_WORLD,
    )
    bits = _set_failure(
        bits,
        count > available,
        ARSENAL_FAILURE_AREA_OVERFLOW,
    )
    row_valid = bits == jnp.uint32(0)
    force = _rotate_local_direction(
        f32[..., [EF_FORCE_X, EF_FORCE_Y, EF_FORCE_Z]],
        yaw,
    )
    inputs = (
        jnp.swapaxes(requested & row_valid[:, None], 0, 1),
        jnp.swapaxes(flat.source_entity_id, 0, 1),
        jnp.swapaxes(center, 0, 1),
        jnp.swapaxes(force, 0, 1),
        jnp.swapaxes(f32, 0, 1),
        jnp.swapaxes(i32, 0, 1),
        jnp.swapaxes(flat.flags, 0, 1),
    )

    def spawn_one(areas: AreaState, values):
        active, owner, origin, force_direction, event_f32, event_i32, flags = values
        slot = jnp.argmax(~areas.active, axis=1)
        selected = (
            jax.nn.one_hot(
                slot,
                areas.active.shape[1],
                dtype=jnp.bool_,
            )
            & active[:, None]
        )
        zero = jnp.zeros_like(event_f32[:, 0])
        areas = areas._replace(
            center=_write_slot(areas.center, origin, selected),
            age_seconds=_write_slot(areas.age_seconds, zero, selected),
            duration_seconds=_write_slot(
                areas.duration_seconds,
                event_f32[:, EF_AREA_DURATION_SECONDS],
                selected,
            ),
            interval_seconds=_write_slot(
                areas.interval_seconds,
                event_f32[:, EF_AREA_INTERVAL_SECONDS],
                selected,
            ),
            interval_clock_seconds=_write_slot(
                areas.interval_clock_seconds,
                zero,
                selected,
            ),
            radius_change_seconds=_write_slot(
                areas.radius_change_seconds,
                event_f32[:, EF_AREA_RADIUS_CHANGE_SECONDS],
                selected,
            ),
            start_radius=_write_slot(
                areas.start_radius,
                event_f32[:, EF_RADIUS],
                selected,
            ),
            end_radius=_write_slot(
                areas.end_radius,
                event_f32[:, EF_AREA_END_RADIUS],
                selected,
            ),
            height=_write_slot(
                areas.height,
                event_f32[:, EF_AREA_HEIGHT],
                selected,
            ),
            damage=_write_slot(
                areas.damage,
                event_f32[:, EF_DAMAGE],
                selected,
            ),
            random_percentage=_write_slot(
                areas.random_percentage,
                event_f32[:, EF_RANDOM_PERCENTAGE],
                selected,
            ),
            damage_cause=_write_slot(
                areas.damage_cause,
                event_i32[:, EI_DAMAGE_CAUSE],
                selected,
            ),
            damage_class=_write_slot(
                areas.damage_class,
                event_i32[:, EI_DAMAGE_CLASS],
                selected,
            ),
            force_direction=_write_slot(
                areas.force_direction,
                force_direction,
                selected,
            ),
            force_magnitude=_write_slot(
                areas.force_magnitude,
                event_f32[:, EF_FORCE_MAGNITUDE],
                selected,
            ),
            force_mode=_write_slot(
                areas.force_mode,
                event_i32[:, EI_FORCE_MODE],
                selected,
            ),
            air_resistance=_write_slot(
                areas.air_resistance,
                event_f32[:, EF_AIR_RESISTANCE],
                selected,
            ),
            air_resistance_max=_write_slot(
                areas.air_resistance_max,
                event_f32[:, EF_AIR_RESISTANCE_MAX],
                selected,
            ),
            ground_resistance=_write_slot(
                areas.ground_resistance,
                event_f32[:, EF_GROUND_RESISTANCE],
                selected,
            ),
            ground_resistance_max=_write_slot(
                areas.ground_resistance_max,
                event_f32[:, EF_GROUND_RESISTANCE_MAX],
                selected,
            ),
            resistance_threshold=_write_slot(
                areas.resistance_threshold,
                event_f32[:, EF_RESISTANCE_THRESHOLD],
                selected,
            ),
            resistance_style=_write_slot(
                areas.resistance_style,
                event_i32[:, EI_RESISTANCE_STYLE],
                selected,
            ),
            status_id=_write_slot(
                areas.status_id,
                event_i32[:, EI_STATUS_ID],
                selected,
            ),
            status_duration_seconds=_write_slot(
                areas.status_duration_seconds,
                event_f32[:, EF_STATUS_DURATION_SECONDS],
                selected,
            ),
            status_cooldown_seconds=_write_slot(
                areas.status_cooldown_seconds,
                event_f32[:, EF_STATUS_COOLDOWN_SECONDS],
                selected,
            ),
            status_damage=_write_slot(
                areas.status_damage,
                event_f32[:, EF_STATUS_DAMAGE],
                selected,
            ),
            status_healing=_write_slot(
                areas.status_healing,
                event_f32[:, EF_STATUS_HEALING],
                selected,
            ),
            status_damage_cause=_write_slot(
                areas.status_damage_cause,
                event_i32[:, EI_STATUS_DAMAGE_CAUSE],
                selected,
            ),
            status_resource_id=_write_slot(
                areas.status_resource_id,
                event_i32[:, EI_STATUS_RESOURCE_ID],
                selected,
            ),
            status_resource_delta=_write_slot(
                areas.status_resource_delta,
                event_f32[:, EF_STATUS_RESOURCE_DELTA],
                selected,
            ),
            status_speed_multiplier=_write_slot(
                areas.status_speed_multiplier,
                event_f32[:, EF_STATUS_SPEED_MULTIPLIER],
                selected,
            ),
            status_flags=_write_slot(
                areas.status_flags,
                flags & jnp.uint32(EVENT_STATUS_FLAG_MASK),
                selected,
            ),
            status_overlap_mode=_write_slot(
                areas.status_overlap_mode,
                event_i32[:, EI_STATUS_OVERLAP_MODE],
                selected,
            ),
            kind=_write_slot(
                areas.kind,
                event_i32[:, EI_PROJECTILE_KIND],
                selected,
            ),
            shape=_write_slot(
                areas.shape,
                jnp.full_like(owner, AREA_SHAPE_NONE),
                selected,
            ),
            owner_entity_id=_write_slot(
                areas.owner_entity_id,
                owner,
                selected,
            ),
            active=areas.active | selected,
            entity_overlap_only=_write_slot(
                areas.entity_overlap_only,
                jnp.ones_like(active),
                selected,
            ),
            friendly_fire=_write_slot(
                areas.friendly_fire,
                (flags & jnp.uint32(EVENT_FLAG_AREA_FRIENDLY_FIRE)) != 0,
                selected,
            ),
            target_mask=_write_slot(
                areas.target_mask,
                jnp.where(
                    (flags & jnp.uint32(EVENT_FLAG_AREA_FRIENDLY_FIRE)) != 0,
                    jnp.int32(AREA_TARGET_TEAM | AREA_TARGET_ENEMIES),
                    jnp.int32(AREA_TARGET_ENEMIES),
                ),
                selected,
            ),
            deployable_attack_flags=_write_slot(
                areas.deployable_attack_flags,
                jnp.zeros_like(owner),
                selected,
            ),
        )
        return areas, None

    candidate, _ = jax.lax.scan(spawn_one, state.areas, inputs)
    return (
        state._replace(
            areas=_select_tree(row_valid, candidate, state.areas),
            failure_bits=bits,
        ),
        jnp.where(row_valid, count, jnp.int32(0)),
    )


def spawn_terminal_areas(
    state: ArsenalState,
    projectile: ProjectileState,
    requested: jax.Array,
    contact_point: jax.Array,
    contact_normal: jax.Array,
) -> tuple[ArsenalState, jax.Array]:
    """Atomically materialize typed world-contact deployable AOEs."""

    shape = projectile.active.shape
    if requested.shape != shape or requested.dtype != jnp.dtype(jnp.bool_):
        raise ValueError("terminal area requested must be bool[B, projectile]")
    if contact_point.shape != shape + (3,) or contact_normal.shape != shape + (3,):
        raise ValueError("terminal contact point/normal must be [B, projectile, 3]")
    attack_flags = projectile.terminal_area_attack_flags
    valid_payload = (
        projectile.terminal_deployable_area
        & projectile.terminal_intended_graph_available
        & jnp.isfinite(projectile.terminal_area_duration_seconds)
        & (projectile.terminal_area_duration_seconds > 0.0)
        & jnp.isfinite(projectile.terminal_area_interval_seconds)
        & (projectile.terminal_area_interval_seconds > 0.0)
        & jnp.isfinite(projectile.explosion_radius)
        & (projectile.explosion_radius >= 0.0)
        & jnp.isfinite(projectile.terminal_area_end_radius)
        & (projectile.terminal_area_end_radius >= 0.0)
        & jnp.isfinite(projectile.terminal_area_height)
        & (projectile.terminal_area_height > 0.0)
        & jnp.isfinite(projectile.terminal_area_radius_change_seconds)
        & (projectile.terminal_area_radius_change_seconds > 0.0)
        & jnp.isfinite(projectile.damage)
        & (projectile.damage == 0.0)
        & jnp.all(
            jnp.isfinite(projectile.terminal_deployable_collision_half_extent),
            axis=2,
        )
        & jnp.all(
            projectile.terminal_deployable_collision_half_extent > 0.0,
            axis=2,
        )
        & jnp.all(
            jnp.isfinite(projectile.terminal_deployable_collision_center_offset),
            axis=2,
        )
        & (projectile.terminal_deployable_id > 0)
        & projectile.terminal_deployable_count_towards_global_limit
        & (projectile.terminal_deployable_max_live_count == 2_147_483_647)
        & (projectile.damage_cause >= 0)
        & (projectile.damage_cause < DAMAGE_COUNT)
        & (projectile.damage_class >= 0)
        & (projectile.damage_class < DAMAGE_CLASS_COUNT)
        & (projectile.status_id > 0)
        & jnp.isfinite(projectile.status_duration_seconds)
        & (projectile.status_duration_seconds > 0.0)
        & jnp.isfinite(projectile.status_cooldown_seconds)
        & (projectile.status_cooldown_seconds >= 0.0)
        & jnp.isfinite(projectile.status_damage)
        & (projectile.status_damage == 0.0)
        & jnp.isfinite(projectile.on_hit_healing)
        & (projectile.on_hit_healing >= 0.0)
        & (projectile.status_damage_cause >= 0)
        & (projectile.status_damage_cause < DAMAGE_COUNT)
        & (projectile.status_resource_id == -1)
        & jnp.isfinite(projectile.status_resource_delta)
        & (projectile.status_resource_delta == 0.0)
        & jnp.isfinite(projectile.status_speed_multiplier)
        & (projectile.status_speed_multiplier > 0.0)
        & (
            (projectile.status_flags & jnp.uint32(TERMINAL_UNKNOWN_STATUS_FLAG_MASK))
            == 0
        )
        & (projectile.status_overlap_mode >= STATUS_OVERLAP_IGNORE)
        & (projectile.status_overlap_mode <= STATUS_OVERLAP_OVERWRITE)
        & (
            (projectile.terminal_area_shape == AREA_SHAPE_SPHERE)
            | (projectile.terminal_area_shape == AREA_SHAPE_CYLINDER)
        )
        & ((attack_flags & jnp.int32(~DEPLOYABLE_ATTACK_FLAG_MASK)) == 0)
    )
    normal_squared = jnp.sum(contact_normal * contact_normal, axis=2)
    valid_contact = (
        jnp.all(jnp.isfinite(contact_point), axis=2)
        & jnp.all(jnp.isfinite(contact_normal), axis=2)
        & jnp.isfinite(normal_squared)
        & (normal_squared > jnp.float32(1.0e-12))
    )
    available = jnp.sum((~state.areas.active).astype(jnp.int32), axis=1)
    count = jnp.sum(requested.astype(jnp.int32), axis=1)
    existing_counting = (
        state.areas.active & state.areas.deployable_count_towards_global_limit
    )
    conflicts_existing = jnp.any(
        requested[:, :, None]
        & projectile.terminal_deployable_count_towards_global_limit[:, :, None]
        & existing_counting[:, None, :]
        & (
            projectile.owner_entity_id[:, :, None]
            == state.areas.owner_entity_id[:, None, :]
        ),
        axis=(1, 2),
    )
    requested_counting = (
        requested & projectile.terminal_deployable_count_towards_global_limit
    )
    projectile_index = jnp.arange(requested.shape[1], dtype=jnp.int32)
    conflicts_same_step = jnp.any(
        requested_counting[:, :, None]
        & requested_counting[:, None, :]
        & (
            projectile.owner_entity_id[:, :, None]
            == projectile.owner_entity_id[:, None, :]
        )
        & (projectile_index[:, None] < projectile_index[None, :])[None, :, :],
        axis=(1, 2),
    )
    bits = state.failure_bits
    bits = _set_failure(
        bits,
        jnp.any(requested & ~valid_payload, axis=1),
        ARSENAL_FAILURE_INVALID_STATE,
    )
    bits = _set_failure(
        bits,
        jnp.any(requested & ~valid_contact, axis=1),
        ARSENAL_FAILURE_UNSUPPORTED_WORLD,
    )
    # The native owner component permits one counting deployable and retires
    # the oldest only later through OwnerTicker/DeathComponent. This bounded
    # asset-intended slice makes no timing approximation: a second launch is
    # outside its full-life noninterference contract and fails before mutation.
    bits = _set_failure(
        bits,
        conflicts_existing | conflicts_same_step,
        ARSENAL_FAILURE_UNSUPPORTED_WORLD,
    )
    bits = _set_failure(bits, count > available, ARSENAL_FAILURE_AREA_OVERFLOW)
    row_valid = bits == jnp.uint32(0)
    inputs = jax.tree_util.tree_map(
        lambda value: jnp.swapaxes(value, 0, 1),
        (
            requested & row_valid[:, None],
            projectile.owner_entity_id,
            contact_point,
            projectile.terminal_deployable_collision_half_extent,
            projectile.terminal_deployable_collision_center_offset,
            projectile.terminal_deployable_id,
            projectile.terminal_deployable_count_towards_global_limit,
            projectile.terminal_deployable_max_live_count,
            projectile.terminal_area_duration_seconds,
            projectile.terminal_area_interval_seconds,
            projectile.explosion_radius,
            projectile.terminal_area_end_radius,
            projectile.terminal_area_height,
            projectile.terminal_area_radius_change_seconds,
            projectile.terminal_area_shape,
            projectile.damage,
            projectile.random_percentage,
            projectile.damage_cause,
            projectile.damage_class,
            projectile.status_id,
            projectile.status_duration_seconds,
            projectile.status_cooldown_seconds,
            projectile.status_damage,
            projectile.on_hit_healing,
            projectile.status_damage_cause,
            projectile.status_resource_id,
            projectile.status_resource_delta,
            projectile.status_speed_multiplier,
            projectile.status_flags,
            projectile.status_overlap_mode,
            attack_flags,
        ),
    )

    def spawn_one(areas: AreaState, values):
        (
            active,
            owner,
            center,
            collision_half_extent,
            collision_center_offset,
            deployable_id,
            count_towards_global_limit,
            max_live_count,
            duration,
            interval,
            start_radius,
            end_radius,
            height,
            radius_change,
            area_shape,
            damage,
            random_percentage,
            damage_cause,
            damage_class,
            status_id,
            status_duration,
            status_cooldown,
            status_damage,
            status_healing,
            status_damage_cause,
            status_resource_id,
            status_resource_delta,
            status_speed,
            status_flags,
            status_overlap,
            raw_attack_flags,
        ) = values
        slot = jnp.argmax(~areas.active, axis=1)
        selected = (
            jax.nn.one_hot(slot, areas.active.shape[1], dtype=jnp.bool_)
            & active[:, None]
        )
        zero = jnp.zeros_like(duration)
        zero_i32 = jnp.zeros_like(owner)
        zero_vec = jnp.zeros_like(center)
        candidate = areas._replace(
            center=_write_slot(areas.center, center, selected),
            collision_half_extent=_write_slot(
                areas.collision_half_extent,
                collision_half_extent,
                selected,
            ),
            collision_center_offset=_write_slot(
                areas.collision_center_offset,
                collision_center_offset,
                selected,
            ),
            age_seconds=_write_slot(areas.age_seconds, zero, selected),
            duration_seconds=_write_slot(areas.duration_seconds, duration, selected),
            interval_seconds=_write_slot(areas.interval_seconds, interval, selected),
            interval_clock_seconds=_write_slot(
                areas.interval_clock_seconds, zero, selected
            ),
            radius_change_seconds=_write_slot(
                areas.radius_change_seconds, radius_change, selected
            ),
            start_radius=_write_slot(areas.start_radius, start_radius, selected),
            end_radius=_write_slot(areas.end_radius, end_radius, selected),
            height=_write_slot(areas.height, height, selected),
            damage=_write_slot(areas.damage, damage, selected),
            random_percentage=_write_slot(
                areas.random_percentage, random_percentage, selected
            ),
            damage_cause=_write_slot(areas.damage_cause, damage_cause, selected),
            damage_class=_write_slot(areas.damage_class, damage_class, selected),
            force_direction=_write_slot(areas.force_direction, zero_vec, selected),
            force_magnitude=_write_slot(areas.force_magnitude, zero, selected),
            force_mode=_write_slot(areas.force_mode, zero_i32, selected),
            air_resistance=_write_slot(
                areas.air_resistance, jnp.ones_like(duration), selected
            ),
            air_resistance_max=_write_slot(
                areas.air_resistance_max, jnp.ones_like(duration), selected
            ),
            ground_resistance=_write_slot(
                areas.ground_resistance, jnp.ones_like(duration), selected
            ),
            ground_resistance_max=_write_slot(
                areas.ground_resistance_max, jnp.ones_like(duration), selected
            ),
            resistance_threshold=_write_slot(
                areas.resistance_threshold, jnp.ones_like(duration), selected
            ),
            resistance_style=_write_slot(areas.resistance_style, zero_i32, selected),
            status_id=_write_slot(areas.status_id, status_id, selected),
            status_duration_seconds=_write_slot(
                areas.status_duration_seconds, status_duration, selected
            ),
            status_cooldown_seconds=_write_slot(
                areas.status_cooldown_seconds, status_cooldown, selected
            ),
            status_damage=_write_slot(areas.status_damage, status_damage, selected),
            status_healing=_write_slot(areas.status_healing, status_healing, selected),
            status_damage_cause=_write_slot(
                areas.status_damage_cause, status_damage_cause, selected
            ),
            status_resource_id=_write_slot(
                areas.status_resource_id, status_resource_id, selected
            ),
            status_resource_delta=_write_slot(
                areas.status_resource_delta, status_resource_delta, selected
            ),
            status_speed_multiplier=_write_slot(
                areas.status_speed_multiplier, status_speed, selected
            ),
            status_flags=_write_slot(areas.status_flags, status_flags, selected),
            status_overlap_mode=_write_slot(
                areas.status_overlap_mode, status_overlap, selected
            ),
            kind=_write_slot(
                areas.kind,
                jnp.full_like(owner, AREA_DEPLOYABLE_AOE),
                selected,
            ),
            shape=_write_slot(areas.shape, area_shape, selected),
            owner_entity_id=_write_slot(areas.owner_entity_id, owner, selected),
            active=areas.active | selected,
            entity_overlap_only=_write_slot(
                areas.entity_overlap_only, jnp.ones_like(active), selected
            ),
            friendly_fire=_write_slot(
                areas.friendly_fire,
                jnp.zeros_like(active),
                selected,
            ),
            target_mask=_write_slot(
                areas.target_mask,
                jnp.zeros_like(owner),
                selected,
            ),
            deployable_attack_flags=_write_slot(
                areas.deployable_attack_flags,
                raw_attack_flags,
                selected,
            ),
            deployable_id=_write_slot(
                areas.deployable_id,
                deployable_id,
                selected,
            ),
            deployable_count_towards_global_limit=_write_slot(
                areas.deployable_count_towards_global_limit,
                count_towards_global_limit,
                selected,
            ),
            deployable_max_live_count=_write_slot(
                areas.deployable_max_live_count,
                max_live_count,
                selected,
            ),
        )
        return candidate, None

    candidate, _ = jax.lax.scan(spawn_one, state.areas, inputs)
    return (
        state._replace(
            areas=_select_tree(row_valid, candidate, state.areas),
            failure_bits=bits,
        ),
        jnp.where(row_valid, count, jnp.int32(0)),
    )


def tick_areas(
    combat: CombatState,
    state: ArsenalState,
    dt_seconds: jax.Array,
    params: CombatParams,
    rules: CombatMechanicsRules,
    *,
    entity_team_id: jax.Array | None = None,
    deployable_spatial_roster_and_group_equivalence_attested: (jax.Array | None) = None,
    deployable_area_effect_eligible_mask: jax.Array | None = None,
    deployable_area_effect_candidates_available: jax.Array | None = None,
):
    """Advance interval clocks and emit every eligible typed-area overlap."""

    area = state.areas
    batch, entity_count = combat.health.shape
    dt = _batch_dt(dt_seconds, batch)
    invalid = (
        _invalid_area_state(
            area,
            entity_count,
        )
        | ~jnp.isfinite(dt)[:, None]
    )
    bits = _set_failure(
        state.failure_bits,
        jnp.any(invalid, axis=1) | (dt < 0.0),
        ARSENAL_FAILURE_INVALID_STATE,
    )
    bits = _set_failure(
        bits,
        jnp.any(area.active & ~area.entity_overlap_only, axis=1),
        ARSENAL_FAILURE_UNSUPPORTED_WORLD,
    )
    deployable_active = area.active & (area.kind == AREA_DEPLOYABLE_AOE)
    if deployable_spatial_roster_and_group_equivalence_attested is None:
        group_attested = jnp.zeros((batch,), dtype=jnp.bool_)
    else:
        group_attested = jnp.asarray(
            deployable_spatial_roster_and_group_equivalence_attested
        )
        if group_attested.dtype != jnp.bool_:
            raise TypeError(
                "deployable_spatial_roster_and_group_equivalence_attested "
                f"must have dtype bool, got {group_attested.dtype}"
            )
        if group_attested.shape != (batch,):
            raise ValueError(
                "deployable_spatial_roster_and_group_equivalence_attested "
                "must be bool[B]"
            )
    if deployable_area_effect_eligible_mask is None:
        area_effect_eligible = jnp.zeros(
            (batch, entity_count),
            dtype=jnp.bool_,
        )
    else:
        area_effect_eligible = jnp.asarray(deployable_area_effect_eligible_mask)
        if area_effect_eligible.dtype != jnp.bool_:
            raise TypeError(
                "deployable_area_effect_eligible_mask must have dtype bool, "
                f"got {area_effect_eligible.dtype}"
            )
        if area_effect_eligible.shape != (batch, entity_count):
            raise ValueError("deployable_area_effect_eligible_mask must be bool[B,E]")
    if deployable_area_effect_candidates_available is None:
        candidates_available = jnp.zeros((batch,), dtype=jnp.bool_)
    else:
        candidates_available = jnp.asarray(deployable_area_effect_candidates_available)
        if candidates_available.dtype != jnp.bool_:
            raise TypeError(
                "deployable_area_effect_candidates_available must have dtype "
                f"bool, got {candidates_available.dtype}"
            )
        if candidates_available.shape != (batch,):
            raise ValueError(
                "deployable_area_effect_candidates_available must be bool[B]"
            )
    group_evidence_available = (
        group_attested & (entity_team_id is not None) & candidates_available
    )
    if entity_team_id is not None and jnp.asarray(entity_team_id).dtype != jnp.int32:
        raise TypeError(
            "entity_team_id must have dtype int32, "
            f"got {jnp.asarray(entity_team_id).dtype}"
        )
    bits = _set_failure(
        bits,
        jnp.any(deployable_active, axis=1) & ~group_evidence_available,
        ARSENAL_FAILURE_UNSUPPORTED_WORLD,
    )
    row_valid = bits == jnp.uint32(0)
    active = area.active & row_valid[:, None]
    delta = dt[:, None]
    # Native getRadius reads Duration(spawnInstant, now) before this system's
    # independent interval clock consumes dt. Evaluate radius/lifetime from
    # the prior wall age, then persist age+dt after the phase.
    evaluation_age = area.age_seconds
    age = area.age_seconds + jnp.where(active, delta, 0.0)
    radius_fraction = jnp.clip(
        evaluation_age
        / jnp.maximum(
            area.radius_change_seconds,
            jnp.finfo(jnp.float32).tiny,
        ),
        0.0,
        1.0,
    )
    radius = area.start_radius + (area.end_radius - area.start_radius) * radius_fraction
    clock = area.interval_clock_seconds + jnp.where(active, delta, 0.0)
    deployable = area.kind == AREA_DEPLOYABLE_AOE
    # DeployablesUtils installs a DespawnComponent at spawn+LiveDuration and
    # DespawnSystem removes it only when now.isAfter(expiry). Preserve the
    # equality tick for typed deployables, while leaving legacy areas on their
    # historical half-open lifetime.
    within_live = jnp.where(
        deployable,
        evaluation_age <= area.duration_seconds,
        jnp.ones_like(active),
    )
    # Native DeployableAoeConfig uses a strict comparison and resets to zero.
    fires = active & within_live & (clock > area.interval_seconds)
    clock = jnp.where(fires, jnp.float32(0.0), clock)
    offset = combat.position[:, None, :, :] - area.center[:, :, None, :]
    overlaps = typed_area_transform_overlap(
        offset,
        radius[..., None],
        area.height[..., None],
        area.shape[..., None],
    )
    legacy_target = typed_relationship_target_mask(
        combat.health,
        area.owner_entity_id,
        area.target_mask,
        entity_team_id=entity_team_id,
    )
    if entity_team_id is None:
        deployable_target = jnp.zeros_like(legacy_target)
    else:
        deployable_target = native_deployable_attack_mask(
            combat.health,
            area.owner_entity_id,
            area.deployable_attack_flags,
            entity_team_id=entity_team_id,
            area_effect_eligible=area_effect_eligible,
        )
    target_mask = overlaps & jnp.where(
        (area.kind == AREA_DEPLOYABLE_AOE)[..., None],
        deployable_target,
        legacy_target,
    )
    hit = fires[..., None] & target_mask
    query_count = area.active.shape[1]
    dense_damage = empty_dense_damage_events(
        batch,
        query_count,
        entity_count,
    )
    target_entity_id = jnp.broadcast_to(
        jnp.arange(entity_count, dtype=jnp.int32)[None, None, :],
        hit.shape,
    )

    def query_scalar(value):
        return jnp.broadcast_to(value[..., None], hit.shape)

    def query_vector(value):
        return jnp.broadcast_to(
            value[:, :, None, :],
            hit.shape + (value.shape[-1],),
        )

    dense_damage = dense_damage._replace(
        requested=hit & (area.damage[..., None] > 0.0),
        source_entity_id=query_scalar(area.owner_entity_id),
        target_entity_id=target_entity_id,
        amount=jnp.where(
            hit,
            area.damage[..., None],
            jnp.float32(0.0),
        ),
        random_percentage=query_scalar(area.random_percentage),
        damage_class=query_scalar(area.damage_class),
        cause=query_scalar(area.damage_cause),
        knockback_velocity=jnp.where(
            hit[..., None],
            query_vector(area.force_direction) * area.force_magnitude[..., None, None],
            jnp.float32(0.0),
        ),
        force_mode=query_scalar(area.force_mode),
        air_resistance=query_scalar(area.air_resistance),
        air_resistance_max=query_scalar(area.air_resistance_max),
        ground_resistance=query_scalar(area.ground_resistance),
        ground_resistance_max=query_scalar(area.ground_resistance_max),
        resistance_threshold=query_scalar(area.resistance_threshold),
        resistance_style=query_scalar(area.resistance_style),
    )
    damage, damage_overflow = pack_dense_damage_events(dense_damage)
    max_health = (
        jnp.full(
            combat.health.shape,
            params.target_max_health,
            dtype=jnp.float32,
        )
        .at[:, 0]
        .set(params.agent_max_health)
    )
    applications, overflow = pack_status_applications(
        fires & (area.status_id > 0),
        jnp.where(
            area.kind == AREA_DEPLOYABLE_AOE,
            jnp.int32(-1),
            area.owner_entity_id,
        ),
        target_mask,
        area.status_id,
        area.status_duration_seconds,
        area.status_cooldown_seconds,
        area.status_damage,
        area.status_damage_cause,
        area.status_healing,
        area.status_resource_id,
        area.status_resource_delta,
        area.status_speed_multiplier,
        area.status_flags,
        area.status_overlap_mode,
        max_health,
        rules,
    )
    bits = _set_failure(
        bits,
        damage_overflow | overflow,
        ARSENAL_FAILURE_EVENT_OVERFLOW,
    )
    final_valid = bits == jnp.uint32(0)
    candidate = area._replace(
        age_seconds=age,
        interval_clock_seconds=clock,
        active=area.active
        & jnp.where(
            deployable,
            evaluation_age <= area.duration_seconds,
            age < area.duration_seconds,
        ),
    )
    empty_damage = empty_damage_events(
        batch,
        damage.requested.shape[1],
    )
    return (
        state._replace(
            areas=_select_tree(final_valid, candidate, area),
            failure_bits=bits,
        ),
        _select_tree(final_valid, damage, empty_damage),
        _select_tree(
            final_valid,
            applications,
            empty_status_applications(
                combat.position.shape[0],
                entity_count=entity_count,
            ),
        ),
        jnp.where(
            final_valid,
            jnp.sum(hit.astype(jnp.int32), axis=(1, 2)),
            jnp.int32(0),
        ),
    )


def _invalid_area_state(
    area: AreaState,
    entity_count: int,
):
    status_present = area.status_id > 0
    status_payload = (
        jnp.isfinite(area.status_healing)
        & (area.status_healing >= 0.0)
        & jnp.isfinite(area.status_duration_seconds)
        & (area.status_duration_seconds > 0.0)
        & jnp.isfinite(area.status_cooldown_seconds)
        & (area.status_cooldown_seconds >= 0.0)
        & jnp.isfinite(area.status_damage)
        & (area.status_damage >= 0.0)
        & jnp.isfinite(area.status_resource_delta)
        & jnp.isfinite(area.status_speed_multiplier)
        & (area.status_speed_multiplier > 0.0)
        & (area.status_damage_cause >= 0)
        & (area.status_damage_cause < DAMAGE_COUNT)
        & (area.status_resource_id >= -1)
        & (area.status_resource_id < RESOURCE_COUNT)
        & (area.status_overlap_mode >= STATUS_OVERLAP_IGNORE)
        & (area.status_overlap_mode <= STATUS_OVERLAP_OVERWRITE)
    )
    terminal_status_payload = (
        status_present
        & status_payload
        & ((area.status_flags & jnp.uint32(TERMINAL_UNKNOWN_STATUS_FLAG_MASK)) == 0)
        & (area.status_damage == 0.0)
        & (area.status_resource_id == -1)
        & (area.status_resource_delta == 0.0)
    )
    legacy_status_payload = (area.status_id == 0) | (
        status_present
        & status_payload
        & ((area.status_flags & jnp.uint32(_UNKNOWN_EVENT_STATUS_FLAG_MASK)) == 0)
    )
    return area.active & ~(
        (area.owner_entity_id >= 0)
        & (area.owner_entity_id < entity_count)
        & (area.kind > 0)
        & (area.kind <= AREA_DEPLOYABLE_AOE)
        & jnp.where(
            area.kind == AREA_DEPLOYABLE_AOE,
            (area.shape == AREA_SHAPE_SPHERE) | (area.shape == AREA_SHAPE_CYLINDER),
            area.shape == AREA_SHAPE_NONE,
        )
        & jnp.where(
            area.kind == AREA_DEPLOYABLE_AOE,
            jnp.all(
                jnp.isfinite(area.collision_half_extent)
                & (area.collision_half_extent > 0.0),
                axis=2,
            )
            & jnp.all(
                jnp.isfinite(area.collision_center_offset),
                axis=2,
            )
            & (area.deployable_id > 0)
            & area.deployable_count_towards_global_limit
            & (area.deployable_max_live_count == 2_147_483_647),
            jnp.all(area.collision_half_extent == 0.0, axis=2)
            & jnp.all(area.collision_center_offset == 0.0, axis=2)
            & (area.deployable_id == 0)
            & ~area.deployable_count_towards_global_limit
            & (area.deployable_max_live_count == 0),
        )
        & jnp.all(jnp.isfinite(area.center), axis=2)
        & jnp.isfinite(area.age_seconds)
        & (area.age_seconds >= 0.0)
        & jnp.isfinite(area.duration_seconds)
        & (area.duration_seconds > 0.0)
        & jnp.isfinite(area.interval_seconds)
        & (area.interval_seconds > 0.0)
        & jnp.isfinite(area.interval_clock_seconds)
        & (area.interval_clock_seconds >= 0.0)
        & jnp.isfinite(area.radius_change_seconds)
        & (area.radius_change_seconds > 0.0)
        & jnp.isfinite(area.start_radius)
        & (area.start_radius >= 0.0)
        & jnp.isfinite(area.end_radius)
        & (area.end_radius >= 0.0)
        & jnp.isfinite(area.height)
        & (area.height > 0.0)
        & jnp.isfinite(area.damage)
        & jnp.where(
            area.kind == AREA_DEPLOYABLE_AOE,
            area.damage == 0.0,
            area.damage >= 0.0,
        )
        & (area.damage_class >= 0)
        & (area.damage_class < DAMAGE_CLASS_COUNT)
        & jnp.where(
            area.kind == AREA_DEPLOYABLE_AOE,
            terminal_status_payload,
            legacy_status_payload,
        )
        & jnp.where(
            area.kind == AREA_DEPLOYABLE_AOE,
            (area.target_mask == 0)
            & (
                (area.deployable_attack_flags & jnp.int32(~DEPLOYABLE_ATTACK_FLAG_MASK))
                == 0
            ),
            (area.target_mask > 0)
            & ((area.target_mask & jnp.int32(~AREA_TARGET_MASK)) == 0)
            & (area.deployable_attack_flags == 0),
        )
    )


def typed_area_transform_overlap(
    offset: jax.Array,
    radius: jax.Array,
    height: jax.Array,
    shape: jax.Array,
) -> jax.Array:
    """Match TargetUtil Transform-point Sphere/Cylinder boundaries exactly."""

    squared_radius = radius**2
    sphere = jnp.sum(offset**2, axis=-1) < squared_radius
    cylinder = (jnp.sum(offset[..., [0, 2]] ** 2, axis=-1) <= squared_radius) & (
        jnp.abs(offset[..., 1]) <= height * jnp.float32(0.5)
    )
    return jnp.where(
        shape == jnp.int32(AREA_SHAPE_SPHERE),
        sphere,
        jnp.where(
            (shape == jnp.int32(AREA_SHAPE_CYLINDER))
            | (shape == jnp.int32(AREA_SHAPE_NONE)),
            cylinder,
            jnp.bool_(False),
        ),
    )


def _rotate_local_direction(direction, yaw_degrees):
    length = jnp.linalg.norm(direction, axis=2)
    local = direction / jnp.maximum(
        length[..., None],
        jnp.finfo(jnp.float32).tiny,
    )
    yaw = jnp.deg2rad(yaw_degrees)
    return jnp.stack(
        (
            local[..., 0] * jnp.cos(yaw) + local[..., 2] * jnp.sin(yaw),
            local[..., 1],
            -local[..., 0] * jnp.sin(yaw) + local[..., 2] * jnp.cos(yaw),
        ),
        axis=2,
    )


def _gather_entity(array, entity_id):
    batch_index = jnp.arange(array.shape[0]).reshape(
        (array.shape[0],) + (1,) * (entity_id.ndim - 1)
    )
    return array[
        batch_index,
        jnp.clip(entity_id, 0, array.shape[1] - 1),
    ]


def _write_slot(array, value, selected):
    shape = selected.shape + (1,) * (array.ndim - 2)
    return jnp.where(selected.reshape(shape), value[:, None], array)


def _batch_dt(value, batch):
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.ndim == 0:
        result = jnp.broadcast_to(result, (batch,))
    if result.shape != (batch,):
        raise ValueError(f"dt_seconds must be scalar or [{batch}]")
    return result


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
