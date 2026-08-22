"""Episode-pinned validation for fixed-shape arsenal programs."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.contract import *
from hytalegym.jax.combat.arsenal.schema.types import AbilityLoadout
from hytalegym.jax.combat.arsenal.projectiles.terminal import (
    decode_terminal_deployable_payload,
)
from hytalegym.jax.combat.mechanics import (
    DAMAGE_CLASS_COUNT,
    DAMAGE_COUNT,
    RESOURCE_COUNT,
    STATUS_OVERLAP_IGNORE,
    STATUS_OVERLAP_OVERWRITE,
)

_REQUIREMENT_MASK = (
    REQUIRE_CLEAR_PROJECTILE_FLIGHT
    | REQUIRE_CLEAR_FORCE_PATH
    | REQUIRE_ENTITY_ONLY_AREA
    | REQUIRE_STATIC_AREA_PLACEMENT
    | REQUIRE_WORLD_PROJECTILE_COLLISION
    | REQUIRE_LINE_OF_SIGHT
    | REQUIRE_INJECTED_SELECTOR
    | REQUIRE_DEPLOYABLE_INTENDED_GRAPH_EVIDENCE
)
_EVENT_FLAG_MASK = (
    EVENT_STATUS_FLAG_MASK
    | EVENT_FLAG_VALUE_PERCENT
    | EVENT_FLAG_STATUS_VALUE_PERCENT
    | EVENT_FLAG_ANGLED_DAMAGE
    | EVENT_FLAG_INJECTED_SELECTOR
    | EVENT_FLAG_SERVER_SELECTOR
    | EVENT_FLAG_SELECTOR_IGNORES_LINE_OF_SIGHT
    | EVENT_FLAG_PROJECTILE_LEGACY_OFFSET
    | EVENT_FLAG_PROJECTILE_PITCH_ADJUST_OFFSET
    | EVENT_FLAG_ENTITY_ONLY_AREA
    | EVENT_FLAG_PARALLEL_FORK
    | EVENT_FLAG_AREA_FRIENDLY_FIRE
    | EVENT_FLAG_PROGRESSIVE_HORIZONTAL_SELECTOR
    | EVENT_FLAG_PROGRESSIVE_STAB_SELECTOR
    | EVENT_FLAG_PROJECTILE_STANDARD_PHYSICS
    | EVENT_FLAG_PROJECTILE_ALLOW_ROLLING
    | EVENT_FLAG_PROJECTILE_TERMINAL_DEPLOYABLE_AREA
)
_UNKNOWN_REQUIREMENT_MASK = 0xFFFFFFFF ^ _REQUIREMENT_MASK
_UNKNOWN_EVENT_FLAG_MASK = 0xFFFFFFFF ^ _EVENT_FLAG_MASK
_RESOURCE_COMMIT_FLAG_MASK = EVENT_FLAG_SERVER_SELECTOR | EVENT_FLAG_PARALLEL_FORK
_UNKNOWN_RESOURCE_COMMIT_FLAG_MASK = 0xFFFFFFFF ^ _RESOURCE_COMMIT_FLAG_MASK


def validate_ability_loadout(loadout: AbilityLoadout) -> jax.Array:
    """Return sticky per-row failure bits; structural errors raise eagerly."""

    batch = _validate_shapes(loadout)
    ability = loadout.ability_mask
    event = loadout.event_mask
    duration = loadout.ability_duration_seconds
    requirements = loadout.ability_requirements
    static_placement_required = ability & (
        (requirements & jnp.uint32(REQUIRE_STATIC_AREA_PLACEMENT)) != 0
    )
    event_kind = loadout.event_kind
    f32, i32 = loadout.event_f32, loadout.event_i32
    terminal = decode_terminal_deployable_payload(
        f32,
        i32,
        loadout.event_flags,
    )
    charge_slot = (
        jnp.arange(ABILITY_CHARGE_CAPACITY, dtype=jnp.int32)
        < loadout.ability_charge_capacity[..., None]
    )

    invalid_entity = (
        ~jnp.all(jnp.isfinite(loadout.guard_entry_cost), axis=1)
        | jnp.any(loadout.guard_entry_cost < 0.0, axis=1)
        | ~jnp.all(jnp.isfinite(loadout.guard_stamina_value), axis=1)
        | jnp.any(loadout.guard_stamina_value < 0.0, axis=1)
        | ~jnp.all(
            jnp.isfinite(loadout.guard_half_angle_degrees),
            axis=1,
        )
        | jnp.any(
            (loadout.guard_stamina_value > 0.0)
            & (
                (loadout.guard_half_angle_degrees <= 0.0)
                | (loadout.guard_half_angle_degrees > 180.0)
            ),
            axis=1,
        )
        | ~jnp.all(
            jnp.isfinite(loadout.guard_entry_delay_seconds),
            axis=1,
        )
        | jnp.any(loadout.guard_entry_delay_seconds < 0.0, axis=1)
        | ~jnp.all(
            jnp.isfinite(loadout.guard_exit_regen_delay_seconds),
            axis=1,
        )
        | jnp.any(
            (loadout.guard_required_resource_id < -1)
            | (loadout.guard_required_resource_id >= RESOURCE_COUNT),
            axis=1,
        )
        | ~jnp.all(
            jnp.isfinite(loadout.guard_required_resource_minimum),
            axis=1,
        )
        | jnp.any(loadout.guard_required_resource_minimum < 0.0, axis=1)
        # A requirement above the current scenario maximum is unavailable,
        # not malformed. guard_resource_available owns that runtime gate.
        | jnp.any(
            loadout.guard_interrupting_type_mask
            >= jnp.uint32(1 << INTERACTION_TYPE_COUNT),
            axis=1,
        )
        | ~jnp.all(jnp.isfinite(loadout.resource_maximum), axis=(1, 2))
        | ~jnp.all(jnp.isfinite(loadout.resource_initial), axis=(1, 2))
        | jnp.any(loadout.resource_maximum < 0.0, axis=(1, 2))
        | jnp.any(loadout.resource_initial < 0.0, axis=(1, 2))
        | jnp.any(
            loadout.resource_initial > loadout.resource_maximum,
            axis=(1, 2),
        )
        | jnp.any(
            loadout.equipped
            & ((loadout.weapon_id <= 0) | (loadout.weapon_family <= 0)),
            axis=1,
        )
        | jnp.any(
            jnp.sum(static_placement_required, axis=2) > 1,
            axis=1,
        )
        | jnp.any(
            (loadout.guard_stamina_value[..., None] > 0.0)
            & ability
            & (loadout.ability_interaction_type == INTERACTION_TYPE_UNRESOLVED),
            axis=(1, 2),
        )
    )
    invalid_ability = ability & (
        (loadout.ability_id <= 0)
        | (loadout.ability_interaction_type < INTERACTION_TYPE_UNRESOLVED)
        | (loadout.ability_interaction_type >= INTERACTION_TYPE_COUNT)
        | (loadout.ability_guard_fork_type < INTERACTION_TYPE_UNRESOLVED)
        | (loadout.ability_guard_fork_type >= INTERACTION_TYPE_COUNT)
        | (
            (loadout.ability_guard_fork_type >= 0)
            & (loadout.ability_interaction_type != loadout.ability_guard_fork_type)
        )
        | (
            (loadout.ability_guard_fork_type >= 0)
            & (loadout.guard_stamina_value[..., None] <= 0.0)
        )
        | (loadout.ability_evidence < EVIDENCE_UNRESOLVED)
        | (loadout.ability_evidence > EVIDENCE_NATIVE_DIFFERENTIAL)
        | ~jnp.isfinite(duration)
        | (duration <= 0.0)
        | ~jnp.isfinite(loadout.ability_cooldown_seconds)
        | (loadout.ability_cooldown_seconds < 0.0)
        | ~jnp.isfinite(loadout.ability_requested_charge_time_seconds)
        | (loadout.ability_requested_charge_time_seconds < -1.0)
        | (loadout.ability_charge_capacity < 1)
        | (loadout.ability_charge_capacity > ABILITY_CHARGE_CAPACITY)
        | jnp.any(
            ~jnp.isfinite(loadout.ability_charge_times_seconds)
            | (charge_slot & (loadout.ability_charge_times_seconds < 0.0))
            | (~charge_slot & (loadout.ability_charge_times_seconds != 0.0)),
            axis=3,
        )
        | (
            (loadout.ability_charge_capacity > 1)
            & (loadout.ability_cooldown_seconds <= 0.0)
        )
        | ~jnp.isfinite(loadout.ability_stamina_regen_delay_seconds)
        | (loadout.ability_stamina_regen_delay_start_tick < 0)
        | (loadout.ability_stamina_regen_delay_end_tick < -1)
        | (
            (loadout.ability_stamina_regen_delay_end_tick >= 0)
            & (
                loadout.ability_stamina_regen_delay_end_tick
                <= loadout.ability_stamina_regen_delay_start_tick
            )
        )
        | (
            (loadout.ability_stamina_regen_delay_seconds >= 0.0)
            & (
                (loadout.ability_stamina_regen_delay_start_tick != 0)
                | (loadout.ability_stamina_regen_delay_end_tick != -1)
            )
        )
        | (
            loadout.ability_interrupted_by_type_mask
            >= jnp.uint32(1 << INTERACTION_TYPE_COUNT)
        )
        | ~jnp.isfinite(loadout.ability_interruptible_after_seconds)
        | (loadout.ability_interruptible_after_seconds < 0.0)
        | (
            (loadout.ability_interrupted_by_type_mask == 0)
            & (loadout.ability_interruptible_after_seconds != 0.0)
        )
        | (
            (loadout.ability_interrupted_by_type_mask != 0)
            & (loadout.ability_interruptible_after_seconds > duration)
        )
        | (loadout.ability_scheduler_prelude_ticks < 0)
        | (
            loadout.ability_outer_root_selector
            != (loadout.ability_outer_root_item_dispatch_tick >= 0)
        )
        | jnp.any(
            loadout.ability_resource_phase_mask
            & (
                ~jnp.isfinite(loadout.ability_resource_phase_value)
                | ~jnp.isfinite(loadout.ability_resource_phase_inactive_value)
                | (loadout.ability_resource_phase_start_tick < 0)
                | (
                    loadout.ability_resource_phase_end_tick
                    <= loadout.ability_resource_phase_start_tick
                )
            ),
            axis=3,
        )
        | jnp.any(
            loadout.ability_resource_phase_mask
            & ~loadout.ability_outer_root_selector[..., None],
            axis=3,
        )
        | jnp.any(
            ~jnp.isfinite(loadout.ability_resource_cost)
            | (loadout.ability_resource_cost < 0.0),
            axis=3,
        )
        | jnp.any(
            (loadout.ability_resource_cost_kind < RESOURCE_COST_NONE)
            | (loadout.ability_resource_cost_kind >= RESOURCE_COST_KIND_COUNT)
            | (
                (loadout.ability_resource_cost > 0.0)
                != (loadout.ability_resource_cost_kind != RESOURCE_COST_NONE)
            ),
            axis=3,
        )
        | jnp.any(
            ~jnp.isfinite(loadout.ability_resource_commit_time_seconds)
            | (loadout.ability_resource_commit_time_seconds < 0.0)
            | (
                loadout.ability_resource_commit_time_seconds
                > duration[..., None] + jnp.float32(1.0e-6)
            )
            | (
                (loadout.ability_resource_cost_kind == RESOURCE_COST_NONE)
                & (
                    (loadout.ability_resource_commit_time_seconds != 0.0)
                    | (loadout.ability_resource_commit_flags != 0)
                )
            )
            | (
                (
                    loadout.ability_resource_commit_flags
                    & jnp.uint32(_UNKNOWN_RESOURCE_COMMIT_FLAG_MASK)
                )
                != 0
            ),
            axis=3,
        )
        | jnp.any(
            ~jnp.isfinite(loadout.ability_resource_minimum)
            | (loadout.ability_resource_minimum < 0.0),
            axis=3,
        )
        # Resource maxima are actor/scenario inputs. An authored requirement
        # above the current maximum remains a valid, runtime-illegal program.
        | ((requirements & jnp.uint32(_UNKNOWN_REQUIREMENT_MASK)) != 0)
        | (
            static_placement_required
            & (
                ~jnp.isfinite(loadout.ability_static_placement_maximum_distance)
                | (loadout.ability_static_placement_maximum_distance <= 0.0)
            )
        )
        | (
            ability
            & ~static_placement_required
            & (
                (loadout.ability_static_placement_maximum_distance != 0.0)
                | loadout.ability_static_placement_allow_walls
            )
        )
    )
    invalid_event = event & (
        ~ability[..., None]
        | (event_kind <= EVENT_NONE)
        | (event_kind > EVENT_CLEAR_STATUS)
        | ~jnp.isfinite(loadout.event_time_seconds)
        | (loadout.event_time_seconds < 0.0)
        | jnp.where(
            (
                loadout.event_flags
                & jnp.uint32(EVENT_FLAG_PROJECTILE_TERMINAL_DEPLOYABLE_AREA)
            )
            != 0,
            loadout.event_time_seconds > duration[..., None],
            loadout.event_time_seconds >= duration[..., None],
        )
        | ~jnp.all(jnp.isfinite(f32), axis=4)
        | ((loadout.event_flags & jnp.uint32(_UNKNOWN_EVENT_FLAG_MASK)) != 0)
        | (i32[..., EI_ON_HIT_RESOURCE_ID] < -1)
        | (i32[..., EI_ON_HIT_RESOURCE_ID] >= RESOURCE_COUNT)
    )
    damage_kind = (
        (event_kind == EVENT_MELEE_CONE)
        | (event_kind == EVENT_RADIAL_DAMAGE)
        | (event_kind == EVENT_PROJECTILE)
        | (event_kind == EVENT_AREA)
    )
    injected_selector = (
        loadout.event_flags & jnp.uint32(EVENT_FLAG_INJECTED_SELECTOR)
    ) != 0
    progressive_stab_selector = (
        loadout.event_flags & jnp.uint32(EVENT_FLAG_PROGRESSIVE_STAB_SELECTOR)
    ) != 0
    progressive_horizontal_selector = (
        loadout.event_flags & jnp.uint32(EVENT_FLAG_PROGRESSIVE_HORIZONTAL_SELECTOR)
    ) != 0
    progressive_selector = (
        injected_selector | progressive_stab_selector | progressive_horizontal_selector
    )
    selector_parameters = f32[
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
    ignores_selector_line_of_sight = (
        loadout.event_flags & jnp.uint32(EVENT_FLAG_SELECTOR_IGNORES_LINE_OF_SIGHT)
    ) != 0
    legacy_projectile_offset = (
        loadout.event_flags & jnp.uint32(EVENT_FLAG_PROJECTILE_LEGACY_OFFSET)
    ) != 0
    pitch_adjust_projectile_offset = (
        loadout.event_flags & jnp.uint32(EVENT_FLAG_PROJECTILE_PITCH_ADJUST_OFFSET)
    ) != 0
    standard_projectile_physics = (
        loadout.event_flags & jnp.uint32(EVENT_FLAG_PROJECTILE_STANDARD_PHYSICS)
    ) != 0
    projectile_allow_rolling = (
        loadout.event_flags & jnp.uint32(EVENT_FLAG_PROJECTILE_ALLOW_ROLLING)
    ) != 0
    entity_only_area = (
        loadout.event_flags & jnp.uint32(EVENT_FLAG_ENTITY_ONLY_AREA)
    ) != 0
    area_effect = (event_kind == EVENT_AREA) | (
        (event_kind == EVENT_PROJECTILE) & (f32[..., EF_RADIUS] > 0.0)
    )
    invalid_event |= (
        event
        & (event_kind == EVENT_PROJECTILE)
        & (
            ~terminal.requested
            & (
                ((f32[..., EF_RADIUS] > 0.0) & (i32[..., EI_BLOCK_DAMAGE_RADIUS] <= 0))
                | (
                    (f32[..., EF_RADIUS] == 0.0)
                    & (i32[..., EI_BLOCK_DAMAGE_RADIUS] != 0)
                )
            )
        )
    )
    requires_entity_only_area = (
        requirements & jnp.uint32(REQUIRE_ENTITY_ONLY_AREA)
    ) != 0
    requires_world_projectile_collision = (
        requirements & jnp.uint32(REQUIRE_WORLD_PROJECTILE_COLLISION)
    ) != 0
    requires_player_proxy_contact_chain = (
        requirements & jnp.uint32(REQUIRE_DEPLOYABLE_INTENDED_GRAPH_EVIDENCE)
    ) != 0
    invalid_event |= (
        event
        & progressive_selector
        & (
            (event_kind != EVENT_MELEE_CONE)
            | (f32[..., EF_SELECTOR_RUNTIME_SECONDS] <= 0.0)
            | (f32[..., EF_SELECTOR_START_DISTANCE] < 0.0)
            | (
                f32[..., EF_SELECTOR_END_DISTANCE]
                <= f32[..., EF_SELECTOR_START_DISTANCE]
            )
            | (f32[..., EF_SELECTOR_EXTEND_LEFT] < 0.0)
            | (f32[..., EF_SELECTOR_EXTEND_RIGHT] < 0.0)
            | (f32[..., EF_SELECTOR_EXTEND_BOTTOM] < 0.0)
            | (f32[..., EF_SELECTOR_EXTEND_TOP] < 0.0)
            | (
                loadout.event_time_seconds + f32[..., EF_SELECTOR_RUNTIME_SECONDS]
                > duration[..., None] + jnp.float32(1.0e-6)
            )
        )
    )
    invalid_event |= (
        event
        & injected_selector
        & ((requirements[..., None] & jnp.uint32(REQUIRE_INJECTED_SELECTOR)) == 0)
    )
    invalid_event |= event & (
        injected_selector.astype(jnp.int32)
        + progressive_stab_selector.astype(jnp.int32)
        + progressive_horizontal_selector.astype(jnp.int32)
        > 1
    )
    invalid_event |= (
        event
        & (injected_selector | progressive_stab_selector)
        & (f32[..., EF_SELECTOR_YAW_LENGTH_DEGREES] != 0.0)
    )
    invalid_event |= (
        event
        & progressive_horizontal_selector
        & (
            (f32[..., EF_SELECTOR_START_DISTANCE] <= 0.0)
            | (f32[..., EF_SELECTOR_EXTEND_LEFT] != 0.0)
            | (f32[..., EF_SELECTOR_EXTEND_RIGHT] != 0.0)
            | (jnp.abs(f32[..., EF_SELECTOR_YAW_LENGTH_DEGREES]) <= 0.0)
        )
    )
    invalid_event |= (
        event
        & ~progressive_selector
        & jnp.any(selector_parameters != jnp.float32(0.0), axis=4)
    )
    invalid_event |= (
        event & ignores_selector_line_of_sight & (event_kind != EVENT_MELEE_CONE)
    )
    invalid_event |= (
        event
        & (legacy_projectile_offset | pitch_adjust_projectile_offset)
        & (event_kind != EVENT_PROJECTILE)
    )
    invalid_event |= event & pitch_adjust_projectile_offset & ~legacy_projectile_offset
    invalid_event |= (
        event
        & (standard_projectile_physics | projectile_allow_rolling)
        & (event_kind != EVENT_PROJECTILE)
    )
    invalid_event |= event & projectile_allow_rolling & ~standard_projectile_physics
    invalid_event |= (
        event
        & entity_only_area
        & (~area_effect | ~requires_entity_only_area[..., None])
    )
    invalid_ability |= (
        ability
        & requires_entity_only_area
        & (
            ~jnp.any(event & area_effect, axis=3)
            | jnp.any(event & area_effect & ~entity_only_area, axis=3)
        )
    )
    direct_target_kind = (
        (event_kind == EVENT_STATUS)
        | (event_kind == EVENT_RESOURCE)
        | (event_kind == EVENT_HEAL)
        | (event_kind == EVENT_FORCE)
        | (event_kind == EVENT_CLEAR_STATUS)
    )
    invalid_event |= (
        event
        & direct_target_kind
        & (
            (i32[..., EI_TARGET_MODE] < TARGET_SELF)
            | (i32[..., EI_TARGET_MODE] > TARGET_OTHER_OR_SELF)
        )
    )
    invalid_event |= (
        event
        & damage_kind
        & ~terminal.requested
        & (i32[..., EI_TARGET_MODE] != TARGET_OTHER)
    )
    invalid_event |= (
        event
        & damage_kind
        & (
            (i32[..., EI_DAMAGE_CAUSE] < 0)
            | (i32[..., EI_DAMAGE_CAUSE] >= DAMAGE_COUNT)
            | (i32[..., EI_DAMAGE_CLASS] < 0)
            | (i32[..., EI_DAMAGE_CLASS] >= DAMAGE_CLASS_COUNT)
            | (f32[..., EF_DAMAGE] < 0.0)
            | (f32[..., EF_RANDOM_PERCENTAGE] < 0.0)
            | (
                ~terminal.requested
                & (
                    (i32[..., EI_FORCE_DIRECTION_MODE] < FORCE_DIRECTION_LOCAL)
                    | (i32[..., EI_FORCE_DIRECTION_MODE] > FORCE_DIRECTION_DIRECTIONAL)
                )
            )
        )
    )
    invalid_event |= event & ~damage_kind & (f32[..., EF_RANDOM_PERCENTAGE] != 0.0)
    angled = (loadout.event_flags & jnp.uint32(EVENT_FLAG_ANGLED_DAMAGE)) != 0
    angled_force_squared = jnp.sum(
        f32[
            ...,
            [
                EF_ANGLED_FORCE_X,
                EF_ANGLED_FORCE_Y,
                EF_ANGLED_FORCE_Z,
            ],
        ]
        ** 2,
        axis=4,
    )
    invalid_event |= (
        event
        & angled
        & (
            (event_kind != EVENT_MELEE_CONE)
            | (f32[..., EF_ANGLED_DAMAGE] < 0.0)
            | (f32[..., EF_ANGLED_ANGLE_DEGREES] < -180.0)
            | (f32[..., EF_ANGLED_ANGLE_DEGREES] > 180.0)
            | (f32[..., EF_ANGLED_DISTANCE_DEGREES] <= 0.0)
            | (f32[..., EF_ANGLED_DISTANCE_DEGREES] > 180.0)
            | (f32[..., EF_ANGLED_FORCE_MAGNITUDE] < 0.0)
            | (
                (f32[..., EF_ANGLED_FORCE_MAGNITUDE] > 0.0)
                & (angled_force_squared <= 0.0)
            )
        )
    )
    force_kind = damage_kind | (event_kind == EVENT_FORCE)
    has_force = force_kind & (f32[..., EF_FORCE_MAGNITUDE] != 0.0)
    resistance_valid = (
        (f32[..., EF_AIR_RESISTANCE] >= 0.0)
        & (f32[..., EF_AIR_RESISTANCE] <= 1.0)
        & (f32[..., EF_AIR_RESISTANCE_MAX] >= 0.0)
        & (f32[..., EF_AIR_RESISTANCE_MAX] <= 1.0)
        & (f32[..., EF_GROUND_RESISTANCE] >= 0.0)
        & (f32[..., EF_GROUND_RESISTANCE] <= 1.0)
        & (f32[..., EF_GROUND_RESISTANCE_MAX] >= 0.0)
        & (f32[..., EF_GROUND_RESISTANCE_MAX] <= 1.0)
        & (f32[..., EF_RESISTANCE_THRESHOLD] > 0.0)
        & (i32[..., EI_RESISTANCE_STYLE] >= 0)
        & (i32[..., EI_RESISTANCE_STYLE] <= 1)
    )
    local_force = has_force & (
        (event_kind == EVENT_FORCE)
        | (i32[..., EI_FORCE_DIRECTION_MODE] == FORCE_DIRECTION_LOCAL)
    )
    force_direction_squared = jnp.sum(
        f32[..., [EF_FORCE_X, EF_FORCE_Y, EF_FORCE_Z]] ** 2,
        axis=4,
    )
    invalid_event |= (
        event
        & force_kind
        & (
            (f32[..., EF_FORCE_MAGNITUDE] < 0.0)
            & (i32[..., EI_FORCE_DIRECTION_MODE] != FORCE_DIRECTION_POINT)
        )
    )
    invalid_event |= (
        event
        & ((event_kind == EVENT_FORCE) | has_force)
        & (
            (i32[..., EI_FORCE_MODE] < FORCE_SET)
            | (i32[..., EI_FORCE_MODE] > FORCE_ADD)
        )
    )
    invalid_event |= event & has_force & ~resistance_valid
    invalid_event |= event & local_force & (force_direction_squared <= 0.0)
    invalid_event |= (
        event
        & (event_kind == EVENT_PROJECTILE)
        & (
            (i32[..., EI_PROJECTILE_KIND] <= PROJECTILE_NONE)
            | (i32[..., EI_PROJECTILE_KIND] > PROJECTILE_DEPLOYABLE)
            | (f32[..., EF_PROJECTILE_SPEED] <= 0.0)
            | (f32[..., EF_PROJECTILE_GRAVITY] < 0.0)
            | (f32[..., EF_PROJECTILE_TERMINAL_VELOCITY] <= 0.0)
            | jnp.where(
                terminal.requested,
                jnp.any(
                    ~jnp.isfinite(terminal.collision_half_extent)
                    | (terminal.collision_half_extent <= 0.0),
                    axis=4,
                )
                | jnp.any(
                    ~jnp.isfinite(terminal.collision_center_offset),
                    axis=4,
                ),
                f32[..., EF_PROJECTILE_HALF_EXTENT] <= 0.0,
            )
            | (f32[..., EF_PROJECTILE_LIFETIME_SECONDS] <= 0.0)
            | (f32[..., EF_PROJECTILE_FUSE_SECONDS] < 0.0)
            | ~jnp.isfinite(f32[..., EF_PROJECTILE_DEAD_TIME_SECONDS])
            | (f32[..., EF_PROJECTILE_DEAD_TIME_SECONDS] < -1.0)
            | (~terminal.requested & (f32[..., EF_PROJECTILE_DIRECT_DAMAGE] < 0.0))
            | (
                ~terminal.requested
                & (f32[..., EF_PROJECTILE_DIRECT_DAMAGE] > 0.0)
                & (
                    (i32[..., EI_PROJECTILE_DIRECT_DAMAGE_CAUSE] < 0)
                    | (i32[..., EI_PROJECTILE_DIRECT_DAMAGE_CAUSE] >= DAMAGE_COUNT)
                )
            )
            | (
                standard_projectile_physics
                & (
                    (f32[..., EF_PROJECTILE_BOUNCINESS] < 0.0)
                    | (f32[..., EF_PROJECTILE_BOUNCINESS] > 1.0)
                    | (f32[..., EF_PROJECTILE_BOUNCE_LIMIT] < 0.0)
                    | (i32[..., EI_PROJECTILE_BOUNCE_COUNT] < -1)
                    | (f32[..., EF_PROJECTILE_ROLLING_FRICTION_FACTOR] < 0.0)
                )
            )
            | (f32[..., EF_RADIUS] < 0.0)
            | (~terminal.requested & (f32[..., EF_FALLOFF] < 0.0))
        )
    )
    terminal_entity_only = (
        loadout.event_flags & jnp.uint32(EVENT_FLAG_ENTITY_ONLY_AREA)
    ) != 0
    invalid_event |= (
        event
        & terminal.requested
        & (
            (event_kind != EVENT_PROJECTILE)
            | (i32[..., EI_PROJECTILE_KIND] != PROJECTILE_DEPLOYABLE)
            | ~standard_projectile_physics
            | ~terminal_entity_only
            | ~requires_entity_only_area[..., None]
            | ~requires_world_projectile_collision[..., None]
            | ~requires_player_proxy_contact_chain[..., None]
            | (
                (i32[..., EI_BLOCK_DAMAGE_RADIUS] != 0)
                & (i32[..., EI_BLOCK_DAMAGE_RADIUS] != 1)
            )
            | (f32[..., EF_PROJECTILE_FUSE_SECONDS] != 0.0)
            | (f32[..., EF_PROJECTILE_DEAD_TIME_SECONDS] != -1.0)
            | (f32[..., EF_FORCE_MAGNITUDE] != 0.0)
            | (f32[..., EF_PROJECTILE_LIFETIME_SECONDS] != 300.0)
            | (terminal.area_damage != 0.0)
            | (f32[..., EF_STATUS_DAMAGE] != 0.0)
            | angled
            | jnp.any(
                ~jnp.isfinite(terminal.deployable_collision_half_extent)
                | (terminal.deployable_collision_half_extent <= 0.0),
                axis=4,
            )
            | jnp.any(
                ~jnp.isfinite(terminal.deployable_collision_center_offset),
                axis=4,
            )
            | (terminal.deployable_id <= 0)
            | (terminal.deployable_max_live_count != 2_147_483_647)
            | ~terminal.deployable_count_towards_global_limit
            | (i32[..., EI_ON_HIT_RESOURCE_ID] != -1)
            | (f32[..., EF_ON_HIT_RESOURCE_DELTA] != 0.0)
            | (terminal.area_duration_seconds <= 0.0)
            | (terminal.area_interval_seconds <= 0.0)
            | (terminal.area_start_radius < 0.0)
            | (terminal.area_end_radius < 0.0)
            | (terminal.area_height <= 0.0)
            | (terminal.area_radius_change_seconds <= 0.0)
            | (terminal.area_damage < 0.0)
            | (
                (terminal.area_shape != AREA_SHAPE_SPHERE)
                & (terminal.area_shape != AREA_SHAPE_CYLINDER)
            )
            | (
                (terminal.area_attack_flags & jnp.int32(~DEPLOYABLE_ATTACK_FLAG_MASK))
                != 0
            )
            | (i32[..., EI_STATUS_ID] <= 0)
            | (
                (
                    loadout.event_flags
                    & jnp.uint32(EVENT_STATUS_FLAG_MASK ^ TERMINAL_STATUS_FLAG_MASK)
                )
                != 0
            )
            | ~jnp.isfinite(f32[..., EF_STATUS_DURATION_SECONDS])
            | (f32[..., EF_STATUS_DURATION_SECONDS] <= 0.0)
            | ~jnp.isfinite(f32[..., EF_STATUS_COOLDOWN_SECONDS])
            | (f32[..., EF_STATUS_COOLDOWN_SECONDS] < 0.0)
            | ~jnp.isfinite(f32[..., EF_STATUS_DAMAGE])
            | (f32[..., EF_STATUS_DAMAGE] != 0.0)
            | ~jnp.isfinite(f32[..., EF_STATUS_HEALING])
            | (f32[..., EF_STATUS_HEALING] < 0.0)
            | ~jnp.isfinite(f32[..., EF_STATUS_RESOURCE_DELTA])
            | (
                (i32[..., EI_STATUS_RESOURCE_ID] == -1)
                & (f32[..., EF_STATUS_RESOURCE_DELTA] != 0.0)
            )
            | ~jnp.isfinite(f32[..., EF_STATUS_SPEED_MULTIPLIER])
            | (f32[..., EF_STATUS_SPEED_MULTIPLIER] <= 0.0)
            | (i32[..., EI_STATUS_DAMAGE_CAUSE] < 0)
            | (i32[..., EI_STATUS_DAMAGE_CAUSE] >= DAMAGE_COUNT)
            | (i32[..., EI_STATUS_RESOURCE_ID] != -1)
            | (f32[..., EF_STATUS_RESOURCE_DELTA] != 0.0)
            | (i32[..., EI_STATUS_OVERLAP_MODE] < STATUS_OVERLAP_IGNORE)
            | (i32[..., EI_STATUS_OVERLAP_MODE] > STATUS_OVERLAP_OVERWRITE)
        )
    )
    invalid_event |= (
        event
        & ~terminal.requested
        & (i32[..., EI_PROJECTILE_KIND] == PROJECTILE_DEPLOYABLE)
    )
    invalid_event |= (
        event
        & (event_kind == EVENT_AREA)
        & (
            (i32[..., EI_PROJECTILE_KIND] <= AREA_NONE)
            | (i32[..., EI_PROJECTILE_KIND] > AREA_GENERIC)
            | (f32[..., EF_AREA_DURATION_SECONDS] <= 0.0)
            | (f32[..., EF_AREA_INTERVAL_SECONDS] <= 0.0)
            | (f32[..., EF_RADIUS] < 0.0)
            | (f32[..., EF_AREA_END_RADIUS] < 0.0)
            | (f32[..., EF_AREA_HEIGHT] <= 0.0)
            | (f32[..., EF_AREA_RADIUS_CHANGE_SECONDS] <= 0.0)
        )
    )
    status_payload = (event_kind == EVENT_STATUS) | (
        ((event_kind == EVENT_PROJECTILE) | (event_kind == EVENT_AREA))
        & (i32[..., EI_STATUS_ID] > 0)
    )
    invalid_event |= (
        event
        & status_payload
        & (
            (i32[..., EI_STATUS_ID] <= 0)
            | (f32[..., EF_STATUS_DURATION_SECONDS] <= 0.0)
            | (f32[..., EF_STATUS_COOLDOWN_SECONDS] < 0.0)
            | (f32[..., EF_STATUS_DAMAGE] < 0.0)
            | (f32[..., EF_STATUS_HEALING] < 0.0)
            | (f32[..., EF_STATUS_SPEED_MULTIPLIER] <= 0.0)
            | (i32[..., EI_STATUS_DAMAGE_CAUSE] < 0)
            | (i32[..., EI_STATUS_DAMAGE_CAUSE] >= DAMAGE_COUNT)
            | (i32[..., EI_STATUS_RESOURCE_ID] < -1)
            | (i32[..., EI_STATUS_RESOURCE_ID] >= RESOURCE_COUNT)
            | (i32[..., EI_STATUS_OVERLAP_MODE] < STATUS_OVERLAP_IGNORE)
            | (i32[..., EI_STATUS_OVERLAP_MODE] > STATUS_OVERLAP_OVERWRITE)
        )
    )
    invalid_event |= (
        event
        & (event_kind == EVENT_RESOURCE)
        & (
            (i32[..., EI_STATUS_RESOURCE_ID] < 0)
            | (i32[..., EI_STATUS_RESOURCE_ID] >= RESOURCE_COUNT)
        )
    )
    invalid_event |= (
        event & (event_kind == EVENT_HEAL) & (f32[..., EF_STATUS_HEALING] < 0.0)
    )
    invalid_event |= (
        event & (event_kind == EVENT_CLEAR_STATUS) & (i32[..., EI_STATUS_ID] <= 0)
    )
    event_without_ability = jnp.any(event & ~ability[..., None], axis=(1, 2, 3))
    invalid = (
        invalid_entity
        | jnp.any(invalid_ability, axis=(1, 2))
        | jnp.any(invalid_event, axis=(1, 2, 3))
        | event_without_ability
        | jnp.any(loadout.overflow, axis=1)
    )
    return jnp.where(
        invalid,
        jnp.uint32(ARSENAL_FAILURE_INVALID_LOADOUT),
        jnp.zeros((batch,), dtype=jnp.uint32),
    )


def _validate_shapes(loadout: AbilityLoadout) -> int:
    if loadout.weapon_id.ndim != 2:
        raise ValueError("loadout.weapon_id must have rank 2")
    batch = loadout.weapon_id.shape[0]
    entity_count = loadout.weapon_id.shape[1]
    if entity_count <= 0:
        raise ValueError("loadout entity count must be positive")
    entity = (batch, entity_count)
    if loadout.ability_mask.ndim != 3:
        raise ValueError("loadout.ability_mask must have rank 3")
    if loadout.event_mask.ndim != 4:
        raise ValueError("loadout.event_mask must have rank 4")
    ability_capacity = loadout.ability_mask.shape[2]
    event_capacity = loadout.event_mask.shape[3]
    if not 1 <= ability_capacity <= ABILITY_CAPACITY:
        raise ValueError(
            "loadout ability capacity must be in "
            f"[1, {ABILITY_CAPACITY}], got {ability_capacity}"
        )
    if not 1 <= event_capacity <= EVENT_CAPACITY:
        raise ValueError(
            "loadout event capacity must be in "
            f"[1, {EVENT_CAPACITY}], got {event_capacity}"
        )
    ability = entity + (ability_capacity,)
    event = ability + (event_capacity,)
    expected = {
        "weapon_id": entity,
        "weapon_family": entity,
        "equipped": entity,
        "guard_entry_cost": entity,
        "guard_stamina_value": entity,
        "guard_half_angle_degrees": entity,
        "guard_entry_delay_seconds": entity,
        "guard_exit_regen_delay_seconds": entity,
        "guard_required_resource_id": entity,
        "guard_required_resource_minimum": entity,
        "guard_interrupting_type_mask": entity,
        "resource_maximum": entity + (RESOURCE_COUNT,),
        "resource_initial": entity + (RESOURCE_COUNT,),
        "ability_id": ability,
        "ability_interaction_type": ability,
        "ability_guard_fork_type": ability,
        "ability_mask": ability,
        "ability_evidence": ability,
        "ability_duration_seconds": ability,
        "ability_cooldown_seconds": ability,
        "ability_requested_charge_time_seconds": ability,
        "ability_charge_times_seconds": ability + (ABILITY_CHARGE_CAPACITY,),
        "ability_charge_capacity": ability,
        "ability_interrupt_recharge": ability,
        "ability_interrupted_by_type_mask": ability,
        "ability_interruptible_after_seconds": ability,
        "ability_stamina_regen_delay_seconds": ability,
        "ability_stamina_regen_delay_start_tick": ability,
        "ability_stamina_regen_delay_end_tick": ability,
        "ability_scheduler_prelude_ticks": ability,
        "ability_outer_root_selector": ability,
        "ability_outer_root_item_dispatch_tick": ability,
        "ability_resource_phase_mask": ability + (RESOURCE_COUNT,),
        "ability_resource_phase_value": ability + (RESOURCE_COUNT,),
        "ability_resource_phase_inactive_value": ability + (RESOURCE_COUNT,),
        "ability_resource_phase_start_tick": ability + (RESOURCE_COUNT,),
        "ability_resource_phase_end_tick": ability + (RESOURCE_COUNT,),
        "ability_resource_cost": ability + (RESOURCE_COUNT,),
        "ability_resource_cost_kind": ability + (RESOURCE_COUNT,),
        "ability_resource_commit_time_seconds": ability + (RESOURCE_COUNT,),
        "ability_resource_commit_flags": ability + (RESOURCE_COUNT,),
        "ability_resource_minimum": ability + (RESOURCE_COUNT,),
        "ability_requirements": ability,
        "ability_static_placement_maximum_distance": ability,
        "ability_static_placement_allow_walls": ability,
        "ability_event_start": ability,
        "ability_event_count": ability,
        "ability_event_mask": event,
        "event_mask": event,
        "event_time_seconds": event,
        "event_kind": event,
        "event_f32": event + (EVENT_FLOAT_FEATURES,),
        "event_i32": event + (EVENT_INTEGER_FEATURES,),
        "event_flags": event,
        "overflow": entity,
    }
    for field, shape in expected.items():
        actual = getattr(loadout, field).shape
        if actual != shape:
            raise ValueError(f"loadout.{field} must have shape {shape}, got {actual}")
    return batch


__all__ = ["validate_ability_loadout"]
