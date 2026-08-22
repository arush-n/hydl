"""Vectorized direct ability-event decoding and payload packing."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.contract import *
from hytalegym.jax.combat.arsenal.schema.types import (
    ArsenalState,
    ArsenalWorldCapabilities,
    FiredEvents,
)
from hytalegym.jax.combat.arsenal.programs.selectors import (
    horizontal_selector_intersects_aabb,
    stab_selector_intersects_aabb,
)
from hytalegym.jax.combat.arsenal.programs.scheduling import (
    event_uses_progressive_selector,
)
from hytalegym.jax.combat.mechanics import (
    APPLIED_RESISTANCE_STYLE_FORCE_VELOCITY,
    CombatMechanicsRules,
    CombatMechanicsState,
    RESOURCE_COUNT,
    RESOURCE_STAMINA,
    STATUS_APPLICATION_CAPACITY,
    apply_statuses,
    clear_statuses,
    empty_damage_events,
    empty_status_applications,
    native_npc_null_config_force_velocity,
)
from hytalegym.jax.combat.types import (
    AGENT_ENTITY,
    TARGET_ENTITY,
    CombatParams,
    CombatState,
)


def process_direct_events(
    combat: CombatState,
    mechanics: CombatMechanicsState,
    arsenal: ArsenalState,
    events: FiredEvents,
    world: ArsenalWorldCapabilities,
    params: CombatParams,
    rules: CombatMechanicsRules,
    *,
    engagement_target_id: jax.Array | None = None,
) -> tuple[
    CombatMechanicsState,
    ArsenalState,
    jax.Array,
    object,
    jax.Array,
]:
    """Decode direct selectors, heals, resources, statuses, and self-forces."""

    flat = _flatten_events(events)
    batch, entity_count = combat.health.shape
    source = jnp.clip(flat.source_entity_id, 0, entity_count - 1)
    if engagement_target_id is None:
        engagement_target_id = _default_engagement_targets(
            batch,
            entity_count,
        )
    if engagement_target_id.shape != (batch, entity_count):
        raise ValueError("engagement_target_id must match combat entity axis")
    other = _gather_entity(engagement_target_id, source)
    target_valid = (other >= 0) & (other < entity_count)
    source_position = _gather_entity(combat.position, source)
    target_position = _gather_entity(combat.position, other)
    source_yaw = _gather_entity(combat.yaw, source)
    target_yaw = _gather_entity(combat.yaw, other)
    offset = target_position - source_position
    distance = jnp.hypot(offset[..., 0], offset[..., 2])
    bearing = jnp.rad2deg(jnp.arctan2(-offset[..., 0], -offset[..., 2]))
    facing_error = jnp.abs(_normalize_degrees(bearing - source_yaw))
    selector_line_of_sight = _gather_entity(
        world.selector_line_of_sight & world.selector_line_of_sight_valid,
        source,
    )
    requires_line_of_sight = (
        flat.requirements & jnp.uint32(REQUIRE_LINE_OF_SIGHT)
    ) != 0
    other_target_available = target_valid & (
        ~requires_line_of_sight | selector_line_of_sight
    )
    alive = target_valid & (_gather_entity(combat.health, other) > 0.0)
    kind = flat.kind
    f32, i32 = flat.f32, flat.i32
    progressive_selector = event_uses_progressive_selector(flat.flags)
    selector_tests_line_of_sight = (
        flat.flags & jnp.uint32(EVENT_FLAG_SELECTOR_IGNORES_LINE_OF_SIGHT)
    ) == 0
    selector_geometry_hit, selector_geometry_valid = _progressive_selector_result(
        combat,
        source,
        other,
        source_position,
        target_position,
        flat,
        params,
    )
    selector_los_valid = _gather_entity(
        world.selector_line_of_sight_valid,
        source,
    )
    selector_evidence_valid = selector_geometry_valid & (
        ~selector_tests_line_of_sight | selector_los_valid
    )
    selector_selected = (
        selector_geometry_hit
        & selector_evidence_valid
        & (~selector_tests_line_of_sight | selector_line_of_sight)
    )
    cone_geometry_hit = (
        (distance <= f32[..., EF_RANGE])
        & (facing_error <= f32[..., EF_HALF_ANGLE_DEGREES])
        & (~selector_tests_line_of_sight | selector_line_of_sight)
    )
    cone_hit = (
        flat.requested
        & (kind == EVENT_MELEE_CONE)
        & jnp.where(
            progressive_selector,
            selector_selected,
            cone_geometry_hit,
        )
        & alive
    )
    radial_hit = (
        flat.requested
        & (kind == EVENT_RADIAL_DAMAGE)
        & (distance <= f32[..., EF_RADIUS])
        & other_target_available
        & alive
    )
    damage_requested = cone_hit | radial_hit
    angled = _angled_damage_mask(
        flat,
        source_position,
        target_position,
        target_yaw,
    )
    force_velocity = damage_force_velocity(
        flat.f32,
        flat.i32,
        source_position,
        target_position,
        source_yaw,
        angled,
    )
    # Both roles in the certified 0.5.7 matchup author KnockbackScale=0.5.
    # Keep the scale data-driven through CombatParams; a heterogeneous role
    # surface must widen that parameter before it is admitted.
    force_velocity = force_velocity * params.agent_knockback_scale
    damage = empty_damage_events(combat.position.shape[0], source.shape[1])
    damage = damage._replace(
        requested=damage_requested,
        source_entity_id=source,
        target_entity_id=other,
        amount=jnp.where(
            damage_requested,
            jnp.where(
                angled,
                f32[..., EF_ANGLED_DAMAGE],
                f32[..., EF_DAMAGE],
            ),
            jnp.float32(0.0),
        ),
        random_percentage=f32[..., EF_RANDOM_PERCENTAGE],
        damage_class=i32[..., EI_DAMAGE_CLASS],
        cause=i32[..., EI_DAMAGE_CAUSE],
        knockback_velocity=jnp.where(
            damage_requested[..., None],
            force_velocity,
            jnp.float32(0.0),
        ),
        force_mode=i32[..., EI_FORCE_MODE],
        air_resistance=f32[..., EF_AIR_RESISTANCE],
        air_resistance_max=f32[..., EF_AIR_RESISTANCE_MAX],
        # MotionControllerWalk.setVelocity clears onGround before movement,
        # but MotionControllerBase chooses ground/air resistance *after*
        # moveEntity has resolved support (MotionControllerBase.java:510-545).
        # Preserve both authored branches here and let the motion consumer's
        # post-collision grounded result select between them.
        ground_resistance=f32[..., EF_GROUND_RESISTANCE],
        ground_resistance_max=f32[..., EF_GROUND_RESISTANCE_MAX],
        resistance_threshold=f32[..., EF_RESISTANCE_THRESHOLD],
        resistance_style=i32[..., EI_RESISTANCE_STYLE],
        on_hit_resource_id=i32[..., EI_ON_HIT_RESOURCE_ID],
        on_hit_resource_delta=f32[..., EF_ON_HIT_RESOURCE_DELTA],
    )
    max_health = _max_health(params, batch, entity_count)
    health = _apply_heals(
        combat.health,
        max_health,
        flat,
        source,
        other,
        other_target_available,
    )
    mechanics = _apply_resource_events(
        mechanics,
        rules,
        flat,
        source,
        other,
        other_target_available,
    )
    mechanics = _apply_force_events(
        mechanics,
        flat,
        source,
        other,
        source_yaw,
        other_target_available,
        params,
    )
    clear_target = _event_target_mask(
        flat.requested & (kind == EVENT_CLEAR_STATUS),
        source,
        other,
        i32[..., EI_TARGET_MODE],
        other_target_available,
        entity_count,
    )
    clear_ids = jnp.swapaxes(
        jnp.where(
            clear_target,
            i32[..., EI_STATUS_ID, None],
            jnp.int32(0),
        ),
        1,
        2,
    )
    mechanics, _ = clear_statuses(mechanics, clear_ids)
    status_target = _event_target_mask(
        flat.requested & (kind == EVENT_STATUS),
        source,
        other,
        i32[..., EI_TARGET_MODE],
        other_target_available,
        entity_count,
    )
    applications, overflow = pack_status_applications(
        flat.requested & (kind == EVENT_STATUS),
        source,
        status_target,
        i32[..., EI_STATUS_ID],
        f32[..., EF_STATUS_DURATION_SECONDS],
        f32[..., EF_STATUS_COOLDOWN_SECONDS],
        f32[..., EF_STATUS_DAMAGE],
        i32[..., EI_STATUS_DAMAGE_CAUSE],
        f32[..., EF_STATUS_HEALING],
        i32[..., EI_STATUS_RESOURCE_ID],
        f32[..., EF_STATUS_RESOURCE_DELTA],
        f32[..., EF_STATUS_SPEED_MULTIPLIER],
        flat.flags,
        i32[..., EI_STATUS_OVERLAP_MODE],
        max_health,
        rules,
    )
    bits = _set_failure(
        arsenal.failure_bits,
        jnp.any(
            flat.requested
            & (kind == EVENT_MELEE_CONE)
            & progressive_selector
            & ~selector_evidence_valid,
            axis=1,
        ),
        ARSENAL_FAILURE_UNSUPPORTED_WORLD,
    )
    bits = _set_failure(
        bits,
        overflow,
        ARSENAL_FAILURE_EVENT_OVERFLOW,
    )
    row_valid = bits == jnp.uint32(0)
    mechanics_candidate = apply_statuses(mechanics, applications)
    mechanics = _select_tree(row_valid, mechanics_candidate, mechanics)
    selector_hits = (
        flat.requested
        & (kind == EVENT_MELEE_CONE)
        & progressive_selector
        & selector_selected
        & target_valid
    )
    selector_hit_bits = _record_selector_hits(
        arsenal.ability_selector_hit_bits,
        selector_hits,
        source,
        flat.ability_event_index,
    )
    arsenal_candidate = arsenal._replace(
        ability_selector_hit_bits=selector_hit_bits,
    )
    arsenal = _select_tree(
        row_valid,
        arsenal_candidate,
        arsenal,
    )._replace(failure_bits=bits)
    event_count = jnp.sum(
        flat.requested.astype(jnp.int32),
        axis=1,
        dtype=jnp.int32,
    )
    return mechanics, arsenal, health, damage, event_count


def pack_status_applications(
    requested: jax.Array,
    source_entity_id: jax.Array,
    target_mask: jax.Array,
    effect_id: jax.Array,
    duration: jax.Array,
    cooldown: jax.Array,
    damage: jax.Array,
    damage_cause: jax.Array,
    healing: jax.Array,
    resource_id: jax.Array,
    resource_delta: jax.Array,
    speed_multiplier: jax.Array,
    flags: jax.Array,
    overlap_mode: jax.Array,
    max_health: jax.Array,
    rules: CombatMechanicsRules,
):
    """Pack arbitrary payloads into four deterministic applications/entity."""

    eligible = target_mask & requested[:, :, None] & (effect_id > 0)[:, :, None]
    entity_major = jnp.swapaxes(eligible, 1, 2)
    rank = jnp.cumsum(entity_major.astype(jnp.int32), axis=2) - 1
    selected = entity_major[..., None] & (
        rank[..., None] == jnp.arange(STATUS_APPLICATION_CAPACITY)
    )
    overflow = jnp.any(
        jnp.sum(entity_major.astype(jnp.int32), axis=2) > STATUS_APPLICATION_CAPACITY,
        axis=1,
    )
    apps = empty_status_applications(
        requested.shape[0],
        entity_count=target_mask.shape[2],
    )
    control_flags = flags & jnp.uint32(EVENT_STATUS_FLAG_MASK)
    percent = (flags & jnp.uint32(EVENT_FLAG_STATUS_VALUE_PERCENT)) != 0
    healing_value = jnp.where(
        percent[:, :, None],
        healing[:, :, None] * max_health[:, None, :] / 100.0,
        healing[:, :, None],
    )
    # Out-of-range authored ids are rejected by validate_ability_loadout().
    # Keep this low-level path non-aliasing as well: JAX one_hot maps invalid
    # indices to an all-zero row, whereas clipping would silently redirect a
    # malformed future asset to resource 0 or RESOURCE_COUNT - 1.
    resource_one_hot = jax.nn.one_hot(
        resource_id,
        RESOURCE_COUNT,
        dtype=jnp.float32,
    )
    target_resource_max = jnp.einsum(
        "ber,bmr->bme",
        rules.resource_maximum,
        resource_one_hot,
    )
    resource_value = jnp.where(
        percent[:, :, None],
        resource_delta[:, :, None] * target_resource_max / 100.0,
        resource_delta[:, :, None],
    )

    def pack(value: jax.Array):
        expanded = jnp.swapaxes(value, 1, 2)[..., None]
        return jnp.sum(
            jnp.where(selected, expanded, jnp.zeros_like(expanded)),
            axis=2,
        )

    return (
        apps._replace(
            requested=jnp.any(selected, axis=2),
            effect_id=pack(jnp.broadcast_to(effect_id[:, :, None], eligible.shape)),
            source_entity_id=pack(
                jnp.broadcast_to(source_entity_id[:, :, None], eligible.shape)
            ),
            duration_seconds=pack(
                jnp.broadcast_to(duration[:, :, None], eligible.shape)
            ),
            cycle_cooldown_seconds=pack(
                jnp.broadcast_to(cooldown[:, :, None], eligible.shape)
            ),
            damage_per_cycle=pack(jnp.broadcast_to(damage[:, :, None], eligible.shape)),
            damage_cause=pack(
                jnp.broadcast_to(damage_cause[:, :, None], eligible.shape)
            ),
            healing_per_cycle=pack(healing_value),
            resource_id=pack(jnp.broadcast_to(resource_id[:, :, None], eligible.shape)),
            resource_delta_per_cycle=pack(resource_value),
            speed_multiplier=pack(
                jnp.broadcast_to(speed_multiplier[:, :, None], eligible.shape)
            ),
            flags=pack(jnp.broadcast_to(control_flags[:, :, None], eligible.shape)),
            overlap_mode=pack(
                jnp.broadcast_to(overlap_mode[:, :, None], eligible.shape)
            ),
        ),
        overflow,
    )


def _apply_heals(
    health,
    maximum,
    flat,
    source,
    other,
    other_target_available,
):
    requested = flat.requested & (flat.kind == EVENT_HEAL)
    target_mask = _event_target_mask(
        requested,
        source,
        other,
        flat.i32[..., EI_TARGET_MODE],
        other_target_available,
        health.shape[1],
    )
    percent = (flat.flags & jnp.uint32(EVENT_FLAG_VALUE_PERCENT)) != 0
    amount = flat.f32[..., EF_STATUS_HEALING]
    values = jnp.where(
        percent[:, :, None],
        amount[:, :, None] * maximum[:, None, :] / 100.0,
        amount[:, :, None],
    )
    total = jnp.sum(
        jnp.where(target_mask, values, jnp.float32(0.0)),
        axis=1,
    )
    return jnp.where(
        total != jnp.float32(0.0),
        jnp.minimum(maximum, health + total),
        health,
    )


def _apply_resource_events(
    mechanics,
    rules,
    flat,
    source,
    other,
    other_target_available,
):
    requested = flat.requested & (flat.kind == EVENT_RESOURCE)
    target_mask = _event_target_mask(
        requested,
        source,
        other,
        flat.i32[..., EI_TARGET_MODE],
        other_target_available,
        mechanics.resources.shape[1],
    )
    resource_id = flat.i32[..., EI_STATUS_RESOURCE_ID]
    one_hot = jax.nn.one_hot(
        resource_id,
        RESOURCE_COUNT,
        dtype=jnp.float32,
    )
    percent = (flat.flags & jnp.uint32(EVENT_FLAG_VALUE_PERCENT)) != 0
    absolute = flat.f32[..., EF_STATUS_RESOURCE_DELTA]
    maximum = rules.resource_maximum[:, None, :, :]
    value = jnp.where(
        percent[:, :, None, None],
        absolute[:, :, None, None] * maximum / 100.0,
        absolute[:, :, None, None],
    )
    delta = jnp.sum(
        target_mask[..., None] * one_hot[:, :, None, :] * value,
        axis=1,
    )
    resources = jnp.clip(
        mechanics.resources + delta,
        rules.resource_minimum,
        rules.resource_maximum,
    )
    stamina_full = (
        resources[..., RESOURCE_STAMINA]
        >= rules.resource_maximum[..., RESOURCE_STAMINA]
    )
    return mechanics._replace(
        resources=resources,
        stamina_broken=mechanics.stamina_broken & ~stamina_full,
    )


def _apply_force_events(
    mechanics,
    flat,
    source,
    other,
    source_yaw,
    other_target_available,
    params,
):
    entity_count = mechanics.applied_velocity.shape[1]
    requested = flat.requested & (flat.kind == EVENT_FORCE)
    target_mode = flat.i32[..., EI_TARGET_MODE]
    primary_targets_other = target_mode == TARGET_OTHER
    other_or_self = target_mode == TARGET_OTHER_OR_SELF
    target = jnp.where(
        primary_targets_other | (other_or_self & other_target_available),
        other,
        source,
    )
    primary_requested = requested & (~primary_targets_other | other_target_available)
    velocity = local_force_velocity(flat.f32, source_yaw)
    velocity = adjust_vertical_force(velocity, flat.f32, flat.i32)
    velocity = native_npc_null_config_force_velocity(velocity, params)
    inputs = (
        jnp.swapaxes(primary_requested, 0, 1),
        jnp.swapaxes(target, 0, 1),
        jnp.swapaxes(other, 0, 1),
        jnp.swapaxes(velocity, 0, 1),
        jnp.swapaxes(flat.i32, 0, 1),
        jnp.swapaxes(other_target_available, 0, 1),
    )

    def apply_one(current, values):
        (
            active,
            victim,
            other_entity,
            event_velocity,
            i32,
            other_available,
        ) = values
        victim = jnp.clip(victim, 0, entity_count - 1)
        other_entity = jnp.clip(other_entity, 0, entity_count - 1)

        def apply_target(state, target, enabled):
            previous = _gather_entity(state.applied_velocity, target)
            next_velocity = jnp.where(
                i32[:, EI_FORCE_MODE, None] == FORCE_SET,
                event_velocity,
                previous + event_velocity,
            )
            return state._replace(
                applied_velocity=_set_entity_where(
                    state.applied_velocity, target, next_velocity, enabled
                ),
                applied_velocity_can_clear=_set_entity_where(
                    state.applied_velocity_can_clear,
                    target,
                    jnp.zeros(enabled.shape, dtype=jnp.bool_),
                    enabled,
                ),
                applied_air_resistance=_set_entity_where(
                    state.applied_air_resistance,
                    target,
                    jnp.broadcast_to(
                        params.agent_force_air_drag_min,
                        enabled.shape,
                    ),
                    enabled,
                ),
                applied_air_resistance_max=_set_entity_where(
                    state.applied_air_resistance_max,
                    target,
                    jnp.broadcast_to(
                        params.agent_force_air_drag_max,
                        enabled.shape,
                    ),
                    enabled,
                ),
                applied_ground_resistance=_set_entity_where(
                    state.applied_ground_resistance,
                    target,
                    jnp.broadcast_to(
                        params.agent_force_ground_drag_base,
                        enabled.shape,
                    ),
                    enabled,
                ),
                applied_ground_resistance_max=_set_entity_where(
                    state.applied_ground_resistance_max,
                    target,
                    jnp.broadcast_to(
                        params.agent_force_ground_drag_base,
                        enabled.shape,
                    ),
                    enabled,
                ),
                applied_resistance_threshold=_set_entity_where(
                    state.applied_resistance_threshold,
                    target,
                    jnp.broadcast_to(
                        params.agent_force_air_drag_max_speed,
                        enabled.shape,
                    ),
                    enabled,
                ),
                applied_resistance_style=_set_entity_where(
                    state.applied_resistance_style,
                    target,
                    jnp.full(
                        enabled.shape,
                        APPLIED_RESISTANCE_STYLE_FORCE_VELOCITY,
                        dtype=jnp.int32,
                    ),
                    enabled,
                ),
                applied_dampen_y=_set_entity_where(
                    state.applied_dampen_y,
                    target,
                    jnp.zeros(enabled.shape, dtype=jnp.bool_),
                    enabled,
                ),
            )

        current = apply_target(current, victim, active)
        current = apply_target(
            current,
            other_entity,
            active & (i32[:, EI_TARGET_MODE] == TARGET_BOTH) & other_available,
        )
        return current, None

    result, _ = jax.lax.scan(apply_one, mechanics, inputs)
    return result


def _event_target_mask(
    requested,
    source,
    other,
    target_mode,
    other_target_available,
    entity_count,
):
    entity_ids = jnp.arange(entity_count)[None, None, :]
    self_mask = entity_ids == source[..., None]
    raw_other_mask = entity_ids == other[..., None]
    other_mask = raw_other_mask & other_target_available[..., None]
    other_or_self_mask = jnp.where(
        other_target_available[..., None],
        raw_other_mask,
        self_mask,
    )
    return requested[..., None] & jnp.where(
        (target_mode == TARGET_BOTH)[..., None],
        self_mask | other_mask,
        jnp.where(
            (target_mode == TARGET_OTHER)[..., None],
            other_mask,
            jnp.where(
                (target_mode == TARGET_OTHER_OR_SELF)[..., None],
                other_or_self_mask,
                self_mask,
            ),
        ),
    )


def adjust_vertical_force(velocity, f32, i32):
    """Clamp the vertical component of an authored force when C3 asks for it.

    `AdjustVertical` is authored alongside a two-element `VerticalClamp` on the
    ApplyForce node (Battleaxe Downstrike `[10, 30]`, Daggers Pounce
    `[-20, 10]`). The clamp bounds are read straight from the asset; what the
    engine does *in addition* -- how it re-aims the vertical component from look
    pitch before clamping -- is not recoverable from either the assets or the
    decompiled server, which never mentions `AdjustVertical`. So this applies
    only the authored, unambiguous half: with the flag set, the vertical
    component is bounded into the authored window.

    Shared by every force path on purpose. The arsenal and entity ability
    executors each carry their own copy of the direction maths, and a law that
    lived in one of them would silently not apply to the other.
    """

    enabled = i32[..., EI_ADJUST_VERTICAL] != 0
    bounded = jnp.clip(
        velocity[..., 1],
        f32[..., EF_VERTICAL_CLAMP_MIN],
        f32[..., EF_VERTICAL_CLAMP_MAX],
    )
    return velocity.at[..., 1].set(
        jnp.where(enabled, bounded, velocity[..., 1]),
    )


def local_force_velocity(f32, yaw_degrees):
    direction = f32[..., [EF_FORCE_X, EF_FORCE_Y, EF_FORCE_Z]]
    length = jnp.linalg.norm(direction, axis=-1)
    normalized = direction / jnp.maximum(
        length[..., None],
        jnp.finfo(jnp.float32).tiny,
    )
    radians = jnp.deg2rad(yaw_degrees)
    world_x = normalized[..., 0] * jnp.cos(radians) + normalized[..., 2] * jnp.sin(
        radians
    )
    world_z = -normalized[..., 0] * jnp.sin(radians) + normalized[..., 2] * jnp.cos(
        radians
    )
    # Preserve the input lane shape explicitly. GPU tail tiles for odd program
    # counts cannot safely lower a stacked scalar-component fusion here.
    world_direction = normalized.at[..., 0].set(world_x)
    world_direction = world_direction.at[..., 2].set(world_z)
    return world_direction * f32[..., EF_FORCE_MAGNITUDE, None]


def _angled_damage_mask(
    events,
    source_position,
    target_position,
    target_yaw_degrees,
):
    offset = source_position - target_position
    bearing = jnp.rad2deg(jnp.arctan2(offset[..., 0], offset[..., 2]))
    relative = _normalize_degrees(bearing + 180.0 - target_yaw_degrees)
    delta = jnp.abs(
        _normalize_degrees(relative - events.f32[..., EF_ANGLED_ANGLE_DEGREES])
    )
    return ((events.flags & jnp.uint32(EVENT_FLAG_ANGLED_DAMAGE)) != 0) & (
        delta < events.f32[..., EF_ANGLED_DISTANCE_DEGREES]
    )


def damage_force_velocity(
    f32,
    i32,
    source_position,
    target_position,
    source_yaw_degrees,
    angled,
):
    direction = jnp.where(
        angled[..., None],
        f32[
            ...,
            [
                EF_ANGLED_FORCE_X,
                EF_ANGLED_FORCE_Y,
                EF_ANGLED_FORCE_Z,
            ],
        ],
        f32[..., [EF_FORCE_X, EF_FORCE_Y, EF_FORCE_Z]],
    )
    magnitude = jnp.where(
        angled,
        f32[..., EF_ANGLED_FORCE_MAGNITUDE],
        f32[..., EF_FORCE_MAGNITUDE],
    )
    length = jnp.linalg.norm(direction, axis=-1)
    normalized = direction / jnp.maximum(
        length[..., None],
        jnp.finfo(jnp.float32).tiny,
    )
    radians = jnp.deg2rad(source_yaw_degrees)
    local_x = normalized[..., 0] * jnp.cos(radians) + normalized[..., 2] * jnp.sin(
        radians
    )
    local_z = -normalized[..., 0] * jnp.sin(radians) + normalized[..., 2] * jnp.cos(
        radians
    )
    local = normalized.at[..., 0].set(local_x)
    local = local.at[..., 2].set(local_z)
    local = local * magnitude[..., None]
    point = target_position - source_position
    point = point / jnp.maximum(
        jnp.linalg.norm(point, axis=-1, keepdims=True),
        jnp.finfo(jnp.float32).tiny,
    )
    point = point * magnitude[..., None]
    point = point.at[..., 1].set(direction[..., 1])
    target_to_source = source_position - target_position
    target_to_source = target_to_source / jnp.maximum(
        jnp.linalg.norm(target_to_source, axis=-1, keepdims=True),
        jnp.finfo(jnp.float32).tiny,
    )
    relative_x = direction[..., 0] * jnp.cos(radians) + direction[..., 2] * jnp.sin(
        radians
    )
    relative_z = -direction[..., 0] * jnp.sin(radians) + direction[..., 2] * jnp.cos(
        radians
    )
    relative = direction.at[..., 0].set(relative_x)
    relative = relative.at[..., 1].set(jnp.zeros_like(direction[..., 1]))
    relative = relative.at[..., 2].set(relative_z)
    combined = target_to_source + relative
    directional = combined * magnitude[..., None]
    directional = directional.at[..., 1].set(direction[..., 1])
    result = jnp.where(
        (i32[..., EI_FORCE_DIRECTION_MODE] == FORCE_DIRECTION_POINT)[..., None],
        point,
        local,
    )
    return jnp.where(
        (i32[..., EI_FORCE_DIRECTION_MODE] == FORCE_DIRECTION_DIRECTIONAL)[..., None],
        directional,
        result,
    )


def _progressive_selector_result(
    combat: CombatState,
    source: jax.Array,
    target: jax.Array,
    source_position: jax.Array,
    target_position: jax.Array,
    flat: FiredEvents,
    params: CombatParams,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate one authored progressive selector against the locked target."""

    batch, event_count = source.shape
    entity_count = combat.position.shape[1]
    head_yaw = combat.yaw
    head_pitch = jnp.zeros_like(head_yaw)
    head_pitch = head_pitch.at[:, AGENT_ENTITY].set(combat.pitch)
    if entity_count > TARGET_ENTITY:
        # ``target_head_yaw`` is already the absolute world yaw published by
        # the target motion controller. Adding body yaw a second time turns a
        # correctly facing 180-degree target back to zero and makes every
        # progressive opponent selector miss.
        head_yaw = head_yaw.at[:, TARGET_ENTITY].set(
            combat.target_head_yaw,
        )
        head_pitch = head_pitch.at[:, TARGET_ENTITY].set(
            combat.target_head_pitch,
        )
    source_head_yaw = _gather_entity(head_yaw, source)
    source_head_pitch = _gather_entity(head_pitch, source)

    eye_offsets = jnp.broadcast_to(
        params.target_eye_offset,
        (batch, entity_count, 3),
    )
    eye_offsets = eye_offsets.at[:, AGENT_ENTITY].set(
        params.agent_eye_offset,
    )
    selector_origin = source_position + _gather_entity(
        eye_offsets,
        source,
    )

    entity_bounds = jnp.broadcast_to(
        params.target_bounds,
        (batch, entity_count, 6),
    )
    entity_bounds = entity_bounds.at[:, AGENT_ENTITY].set(
        params.agent_bounds,
    )
    target_bounds = _gather_entity(entity_bounds, target)
    f32 = flat.f32
    previous = flat.selector_previous_progress
    current = flat.selector_current_progress

    def flatten(value: jax.Array, trailing: int = 0) -> jax.Array:
        return value.reshape(
            (batch * event_count,) + value.shape[value.ndim - trailing :]
        )

    stab_selected = stab_selector_intersects_aabb(
        flatten(selector_origin, 1),
        flatten(source_head_yaw),
        flatten(source_head_pitch),
        flatten(target_position, 1),
        flatten(target_bounds, 1),
        flatten(previous),
        flatten(current),
        flatten(f32[..., EF_SELECTOR_START_DISTANCE]),
        flatten(f32[..., EF_SELECTOR_END_DISTANCE]),
        flatten(f32[..., EF_SELECTOR_EXTEND_LEFT]),
        flatten(f32[..., EF_SELECTOR_EXTEND_RIGHT]),
        flatten(f32[..., EF_SELECTOR_EXTEND_BOTTOM]),
        flatten(f32[..., EF_SELECTOR_EXTEND_TOP]),
        flatten(f32[..., EF_SELECTOR_YAW_OFFSET_DEGREES]),
        flatten(f32[..., EF_SELECTOR_PITCH_OFFSET_DEGREES]),
        flatten(f32[..., EF_SELECTOR_ROLL_OFFSET_DEGREES]),
    ).reshape((batch, event_count))
    horizontal_selected = horizontal_selector_intersects_aabb(
        flatten(selector_origin, 1),
        flatten(source_head_yaw),
        flatten(source_head_pitch),
        flatten(target_position, 1),
        flatten(target_bounds, 1),
        flatten(previous),
        flatten(current),
        flatten(f32[..., EF_SELECTOR_START_DISTANCE]),
        flatten(f32[..., EF_SELECTOR_END_DISTANCE]),
        flatten(f32[..., EF_SELECTOR_EXTEND_BOTTOM]),
        flatten(f32[..., EF_SELECTOR_EXTEND_TOP]),
        flatten(f32[..., EF_SELECTOR_YAW_LENGTH_DEGREES]),
        flatten(f32[..., EF_SELECTOR_YAW_OFFSET_DEGREES]),
        flatten(f32[..., EF_SELECTOR_PITCH_OFFSET_DEGREES]),
        flatten(f32[..., EF_SELECTOR_ROLL_OFFSET_DEGREES]),
    ).reshape((batch, event_count))
    horizontal = (
        flat.flags & jnp.uint32(EVENT_FLAG_PROGRESSIVE_HORIZONTAL_SELECTOR)
    ) != 0
    selected = jnp.where(
        horizontal,
        horizontal_selected,
        stab_selected,
    )
    parameters = f32[
        ...,
        [
            EF_SELECTOR_RUNTIME_SECONDS,
            EF_SELECTOR_START_DISTANCE,
            EF_SELECTOR_END_DISTANCE,
            EF_SELECTOR_EXTEND_LEFT,
            EF_SELECTOR_EXTEND_RIGHT,
            EF_SELECTOR_EXTEND_BOTTOM,
            EF_SELECTOR_EXTEND_TOP,
            EF_SELECTOR_YAW_OFFSET_DEGREES,
            EF_SELECTOR_PITCH_OFFSET_DEGREES,
            EF_SELECTOR_ROLL_OFFSET_DEGREES,
            EF_SELECTOR_YAW_LENGTH_DEGREES,
        ],
    ]
    common_valid = (
        jnp.all(jnp.isfinite(parameters), axis=2)
        & jnp.all(jnp.isfinite(selector_origin), axis=2)
        & jnp.all(jnp.isfinite(target_position), axis=2)
        & jnp.all(jnp.isfinite(target_bounds), axis=2)
        & (flat.ability_event_index >= 0)
        & (flat.ability_event_index < EVENT_CAPACITY)
        & (f32[..., EF_SELECTOR_RUNTIME_SECONDS] > 0.0)
        & (f32[..., EF_SELECTOR_START_DISTANCE] >= 0.0)
        & (f32[..., EF_SELECTOR_END_DISTANCE] > f32[..., EF_SELECTOR_START_DISTANCE])
        & (f32[..., EF_SELECTOR_EXTEND_BOTTOM] >= 0.0)
        & (f32[..., EF_SELECTOR_EXTEND_TOP] >= 0.0)
        & (previous >= 0.0)
        & (current <= 1.0)
        & (current > previous)
    )
    stab_valid = (
        (f32[..., EF_SELECTOR_EXTEND_LEFT] >= 0.0)
        & (f32[..., EF_SELECTOR_EXTEND_RIGHT] >= 0.0)
        & (f32[..., EF_SELECTOR_YAW_LENGTH_DEGREES] == 0.0)
    )
    horizontal_valid = (
        (f32[..., EF_SELECTOR_START_DISTANCE] > 0.0)
        & (f32[..., EF_SELECTOR_EXTEND_LEFT] == 0.0)
        & (f32[..., EF_SELECTOR_EXTEND_RIGHT] == 0.0)
        & (jnp.abs(f32[..., EF_SELECTOR_YAW_LENGTH_DEGREES]) > 0.0)
    )
    valid = common_valid & jnp.where(
        horizontal,
        horizontal_valid,
        stab_valid,
    )
    return selected, valid


def _record_selector_hits(
    current: jax.Array,
    selected: jax.Array,
    source: jax.Array,
    event_index: jax.Array,
) -> jax.Array:
    """Retain native ``HIT_ENTITIES``-style once-per-selector behavior."""

    batch = current.shape[0]
    batch_index = jnp.arange(batch, dtype=jnp.int32)

    def record(index, bits):
        owner = jnp.clip(source[:, index], 0, bits.shape[1] - 1)
        local_event = jnp.clip(
            event_index[:, index],
            jnp.int32(0),
            jnp.int32(EVENT_CAPACITY - 1),
        )
        event_bit = jnp.left_shift(
            jnp.uint32(1),
            local_event.astype(jnp.uint32),
        )
        previous = bits[batch_index, owner]
        replacement = jnp.where(
            selected[:, index],
            previous | event_bit,
            previous,
        )
        return bits.at[batch_index, owner].set(replacement)

    return jax.lax.fori_loop(0, selected.shape[1], record, current)


def _flatten_events(events: FiredEvents) -> FiredEvents:
    batch = events.requested.shape[0]
    return jax.tree_util.tree_map(
        lambda value: value.reshape((batch, -1) + value.shape[3:]),
        events,
    )


def _max_health(params: CombatParams, batch: int, entity_count: int):
    maximum = jnp.full(
        (batch, entity_count),
        params.target_max_health,
        dtype=jnp.float32,
    )
    return maximum.at[:, 0].set(params.agent_max_health)


def _gather_entity(array, entity_id):
    batch_index = jnp.arange(array.shape[0]).reshape(
        (array.shape[0],) + (1,) * (entity_id.ndim - 1)
    )
    return array[
        batch_index,
        jnp.clip(entity_id, 0, array.shape[1] - 1),
    ]


def _default_engagement_targets(batch: int, entity_count: int) -> jax.Array:
    target = jnp.zeros((entity_count,), dtype=jnp.int32)
    target = target.at[0].set(jnp.int32(1))
    return jnp.broadcast_to(target[None, :], (batch, entity_count))


def _set_entity_where(array, entity_id, value, mask):
    batch = array.shape[0]
    old = array[jnp.arange(batch), entity_id]
    shaped = mask.reshape(mask.shape + (1,) * (value.ndim - 1))
    return array.at[jnp.arange(batch), entity_id].set(jnp.where(shaped, value, old))


def _normalize_degrees(value):
    return jnp.mod(value + 180.0, 360.0) - 180.0


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
