"""Fixed-capacity entity-first projectile, world-contact, and explosion kernels."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.contract import (
    ARSENAL_FAILURE_EVENT_OVERFLOW,
    ARSENAL_FAILURE_INVALID_LOADOUT,
    ARSENAL_FAILURE_INVALID_STATE,
    ARSENAL_FAILURE_PROJECTILE_OVERFLOW,
    ARSENAL_FAILURE_UNSUPPORTED_WORLD,
    AREA_DEPLOYABLE_AOE,
    AREA_SHAPE_CYLINDER,
    AREA_SHAPE_SPHERE,
    DEPLOYABLE_ATTACK_FLAG_MASK,
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
    EF_PROJECTILE_BOUNCINESS,
    EF_PROJECTILE_BOUNCE_LIMIT,
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
    EF_PROJECTILE_ROLLING_FRICTION_FACTOR,
    EF_RANDOM_PERCENTAGE,
    EF_RADIUS,
    EF_RESISTANCE_THRESHOLD,
    EF_STATUS_COOLDOWN_SECONDS,
    EF_STATUS_DAMAGE,
    EF_STATUS_DURATION_SECONDS,
    EF_STATUS_HEALING,
    EF_STATUS_RESOURCE_DELTA,
    EF_STATUS_SPEED_MULTIPLIER,
    EF_YAW_OFFSET_DEGREES,
    EI_BLOCK_DAMAGE_RADIUS,
    EI_DAMAGE_CAUSE,
    EI_DAMAGE_CLASS,
    EI_FORCE_DIRECTION_MODE,
    EI_FORCE_MODE,
    EI_ON_HIT_RESOURCE_ID,
    EI_PROJECTILE_KIND,
    EI_PROJECTILE_BOUNCE_COUNT,
    EI_PROJECTILE_DIRECT_DAMAGE_CAUSE,
    EI_RESISTANCE_STYLE,
    EI_STATUS_DAMAGE_CAUSE,
    EI_STATUS_ID,
    EI_STATUS_OVERLAP_MODE,
    EI_STATUS_RESOURCE_ID,
    EVENT_PROJECTILE,
    EVENT_FLAG_ANGLED_DAMAGE,
    EVENT_FLAG_ENTITY_ONLY_AREA,
    EVENT_FLAG_PROJECTILE_LEGACY_OFFSET,
    EVENT_FLAG_PROJECTILE_ALLOW_ROLLING,
    EVENT_FLAG_PROJECTILE_PITCH_ADJUST_OFFSET,
    EVENT_FLAG_PROJECTILE_STANDARD_PHYSICS,
    EVENT_STATUS_FLAG_MASK,
    FORCE_DIRECTION_LOCAL,
    FORCE_DIRECTION_POINT,
    FORCE_SET,
    PROJECTILE_ARROW,
    PROJECTILE_BIG_ARROW,
    PROJECTILE_DEPLOYABLE,
    TERMINAL_STATUS_FLAG_MASK,
    TERMINAL_UNKNOWN_STATUS_FLAG_MASK,
)
from hytalegym.jax.combat.arsenal.effects.areas import spawn_terminal_areas
from hytalegym.jax.combat.arsenal.effects.events import (
    _flatten_events,
    pack_status_applications,
)
from hytalegym.jax.combat.arsenal.effects.impact_events import (
    empty_dense_damage_events,
    pack_dense_damage_events,
    relationship_target_mask,
)
from hytalegym.jax.combat.arsenal.projectiles.response import (
    standard_projectile_contact_response,
)
from hytalegym.jax.combat.arsenal.projectiles.terminal import (
    decode_terminal_deployable_payload,
)
from hytalegym.jax.combat.arsenal.projectiles.crossbow.types import (
    CrossbowProjectileImpactCommands,
)
from hytalegym.jax.combat.arsenal.profiles.hytale_0_5_7 import (
    CROSSBOW_FAMILY_IDS,
)
from hytalegym.jax.combat.arsenal.schema.types import (
    AbilityLoadout,
    ArsenalExplosionCandidates,
    ArsenalState,
    ArsenalWorldCapabilities,
    FiredEvents,
    ProjectileState,
)
from hytalegym.jax.combat.entities.interactions.schema.contract import (
    CROSSBOW_IMPACT_BIG_ARROW,
    CROSSBOW_IMPACT_STANDARD,
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
from hytalegym.jax.combat.types import AGENT_ENTITY, CombatParams, CombatState
from hytalegym.jax.world import (
    GeometryProvider,
    GeometryState,
    geometry_projectile_first_contact_result,
)


ArsenalExplosionCandidateProvider = Callable[
    [
        CombatState,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ],
    ArsenalExplosionCandidates,
]


def spawn_projectiles(
    state: ArsenalState,
    events: FiredEvents,
    world: ArsenalWorldCapabilities,
    loadout: AbilityLoadout | None = None,
) -> tuple[ArsenalState, jax.Array]:
    """Spawn every fired projectile atomically for each environment row."""

    candidate, count, _ = spawn_projectiles_with_sources(
        state,
        events,
        world,
        loadout,
    )
    return candidate, count


def spawn_projectiles_with_sources(
    state: ArsenalState,
    events: FiredEvents,
    world: ArsenalWorldCapabilities,
    loadout: AbilityLoadout | None = None,
) -> tuple[ArsenalState, jax.Array, jax.Array]:
    """Spawn projectiles and return successful counts per source entity."""

    flat = _flatten_events(events)
    requested = flat.requested & (flat.kind == EVENT_PROJECTILE)
    entity_count = world.muzzle_valid.shape[1]
    source = jnp.clip(flat.source_entity_id, 0, entity_count - 1)
    if loadout is None:
        source_crossbow = jnp.zeros(
            (world.muzzle_valid.shape[0], entity_count),
            dtype=jnp.bool_,
        )
        source_combo_damage = jnp.zeros(
            (world.muzzle_valid.shape[0], entity_count),
            dtype=jnp.float32,
        )
    else:
        if loadout.weapon_family.shape != world.muzzle_valid.shape:
            raise ValueError("loadout weapon family must match the World entity axis")
        source_crossbow = _family_matches(
            loadout.weapon_family,
            CROSSBOW_FAMILY_IDS,
        )
        if loadout.event_mask.ndim == 4:
            combo_events = loadout.event_mask[:, :, 1] & (
                loadout.event_kind[:, :, 1] == EVENT_PROJECTILE
            )
            combo_damage_bank = loadout.event_f32[:, :, 1, :, EF_DAMAGE]
        else:
            event_index = jnp.arange(
                loadout.event_mask.shape[2],
                dtype=jnp.int32,
            )[None, None, :]
            combo_start = loadout.ability_event_start[:, :, 1, None]
            combo_count = loadout.ability_event_count[:, :, 1, None]
            combo_window = (event_index >= combo_start) & (
                event_index < combo_start + combo_count
            )
            combo_events = (
                loadout.event_mask
                & combo_window
                & (loadout.event_kind == EVENT_PROJECTILE)
            )
            combo_damage_bank = loadout.event_f32[..., EF_DAMAGE]
        combo_index = jnp.argmax(combo_events, axis=2)
        source_combo_damage = jnp.take_along_axis(
            combo_damage_bank,
            combo_index[..., None],
            axis=2,
        )[..., 0]
        source_combo_damage = jnp.where(
            source_crossbow & jnp.any(combo_events, axis=2),
            source_combo_damage,
            jnp.float32(0.0),
        )
    crossbow = _gather_entity(source_crossbow, source)
    combo_damage = _gather_entity(source_combo_damage, source)
    muzzle_valid = _gather_entity(world.muzzle_valid, source)
    clear = _gather_entity(world.clear_projectile_flight, source)
    world_collision_available = _gather_entity(
        world.projectile_world_collision_available,
        source,
    )
    contact_chain_available = _gather_entity(
        world.deployable_intended_graph_available,
        source,
    )
    position = _gather_entity(world.muzzle_position, source)
    yaw = _gather_entity(world.muzzle_yaw_degrees, source)
    pitch = _gather_entity(world.muzzle_pitch_degrees, source)
    f32, i32 = flat.f32, flat.i32
    terminal = decode_terminal_deployable_payload(f32, i32, flat.flags)
    ordinary_half_extent = jnp.broadcast_to(
        f32[..., EF_PROJECTILE_HALF_EXTENT, None],
        f32.shape[:-1] + (3,),
    )
    collision_half_extent = jnp.where(
        terminal.requested[..., None],
        terminal.collision_half_extent,
        ordinary_half_extent,
    )
    collision_center_offset = jnp.where(
        terminal.requested[..., None],
        terminal.collision_center_offset,
        jnp.float32(0.0),
    )
    launch_yaw = yaw + f32[..., EF_YAW_OFFSET_DEGREES]
    launch_pitch = pitch + f32[..., EF_PITCH_OFFSET_DEGREES]
    spawn_offset = projectile_spawn_offset(
        f32[
            ...,
            [
                EF_PROJECTILE_SPAWN_OFFSET_X,
                EF_PROJECTILE_SPAWN_OFFSET_Y,
                EF_PROJECTILE_SPAWN_OFFSET_Z,
            ],
        ],
        launch_yaw,
        launch_pitch,
        flat.flags,
    )
    position = position + spawn_offset
    finite_transform = (
        jnp.all(jnp.isfinite(position), axis=2)
        & jnp.isfinite(yaw)
        & jnp.isfinite(pitch)
        & jnp.all(jnp.isfinite(spawn_offset), axis=2)
    )
    standard_physics = (
        flat.flags & jnp.uint32(EVENT_FLAG_PROJECTILE_STANDARD_PHYSICS)
    ) != 0
    allow_rolling = (flat.flags & jnp.uint32(EVENT_FLAG_PROJECTILE_ALLOW_ROLLING)) != 0
    valid_standard_physics = jnp.where(
        standard_physics,
        jnp.isfinite(f32[..., EF_PROJECTILE_BOUNCINESS])
        & (f32[..., EF_PROJECTILE_BOUNCINESS] >= 0.0)
        & (f32[..., EF_PROJECTILE_BOUNCINESS] <= 1.0)
        & jnp.isfinite(f32[..., EF_PROJECTILE_BOUNCE_LIMIT])
        & (f32[..., EF_PROJECTILE_BOUNCE_LIMIT] >= 0.0)
        & (i32[..., EI_PROJECTILE_BOUNCE_COUNT] >= -1)
        & jnp.isfinite(f32[..., EF_PROJECTILE_ROLLING_FRICTION_FACTOR])
        & (f32[..., EF_PROJECTILE_ROLLING_FRICTION_FACTOR] >= 0.0),
        ~allow_rolling,
    )
    terminal_payload_valid = jnp.where(
        terminal.requested,
        (i32[..., EI_PROJECTILE_KIND] == PROJECTILE_DEPLOYABLE)
        & standard_physics
        & ((flat.flags & jnp.uint32(EVENT_FLAG_ENTITY_ONLY_AREA)) != 0)
        & world_collision_available
        & jnp.all(jnp.isfinite(collision_half_extent), axis=2)
        & jnp.all(collision_half_extent > 0.0, axis=2)
        & jnp.all(jnp.isfinite(collision_center_offset), axis=2)
        & jnp.all(
            jnp.isfinite(terminal.deployable_collision_half_extent),
            axis=2,
        )
        & jnp.all(
            terminal.deployable_collision_half_extent > 0.0,
            axis=2,
        )
        & jnp.all(
            jnp.isfinite(terminal.deployable_collision_center_offset),
            axis=2,
        )
        & (terminal.deployable_id > 0)
        & terminal.deployable_count_towards_global_limit
        & (terminal.deployable_max_live_count == 2_147_483_647)
        & jnp.isfinite(terminal.area_duration_seconds)
        & (terminal.area_duration_seconds > 0.0)
        & jnp.isfinite(terminal.area_interval_seconds)
        & (terminal.area_interval_seconds > 0.0)
        & jnp.isfinite(terminal.area_start_radius)
        & (terminal.area_start_radius >= 0.0)
        & jnp.isfinite(terminal.area_end_radius)
        & (terminal.area_end_radius >= 0.0)
        & jnp.isfinite(terminal.area_height)
        & (terminal.area_height > 0.0)
        & jnp.isfinite(terminal.area_radius_change_seconds)
        & (terminal.area_radius_change_seconds > 0.0)
        & (
            (terminal.area_shape == AREA_SHAPE_SPHERE)
            | (terminal.area_shape == AREA_SHAPE_CYLINDER)
        )
        & ((terminal.area_attack_flags & jnp.int32(~DEPLOYABLE_ATTACK_FLAG_MASK)) == 0)
        & (
            (i32[..., EI_BLOCK_DAMAGE_RADIUS] == 0)
            | (i32[..., EI_BLOCK_DAMAGE_RADIUS] == 1)
        )
        & (f32[..., EF_PROJECTILE_FUSE_SECONDS] == 0.0)
        & (f32[..., EF_PROJECTILE_DEAD_TIME_SECONDS] == -1.0)
        & (f32[..., EF_PROJECTILE_LIFETIME_SECONDS] == 300.0)
        & (f32[..., EF_FORCE_MAGNITUDE] == 0.0)
        & (terminal.area_damage == 0.0)
        & ((flat.flags & jnp.uint32(EVENT_FLAG_ANGLED_DAMAGE)) == 0)
        & (i32[..., EI_ON_HIT_RESOURCE_ID] == -1)
        & (f32[..., EF_ON_HIT_RESOURCE_DELTA] == 0.0)
        & (i32[..., EI_STATUS_ID] > 0)
        & jnp.isfinite(f32[..., EF_STATUS_DURATION_SECONDS])
        & (f32[..., EF_STATUS_DURATION_SECONDS] > 0.0)
        & jnp.isfinite(f32[..., EF_STATUS_COOLDOWN_SECONDS])
        & (f32[..., EF_STATUS_COOLDOWN_SECONDS] >= 0.0)
        & jnp.isfinite(f32[..., EF_STATUS_DAMAGE])
        & (f32[..., EF_STATUS_DAMAGE] == 0.0)
        & jnp.isfinite(f32[..., EF_STATUS_HEALING])
        & (f32[..., EF_STATUS_HEALING] >= 0.0)
        & (i32[..., EI_STATUS_DAMAGE_CAUSE] >= 0)
        & (i32[..., EI_STATUS_DAMAGE_CAUSE] < DAMAGE_COUNT)
        & (i32[..., EI_STATUS_RESOURCE_ID] == -1)
        & jnp.isfinite(f32[..., EF_STATUS_RESOURCE_DELTA])
        & (f32[..., EF_STATUS_RESOURCE_DELTA] == 0.0)
        & jnp.isfinite(f32[..., EF_STATUS_SPEED_MULTIPLIER])
        & (f32[..., EF_STATUS_SPEED_MULTIPLIER] > 0.0)
        & (
            (
                flat.flags
                & jnp.uint32(EVENT_STATUS_FLAG_MASK ^ TERMINAL_STATUS_FLAG_MASK)
            )
            == 0
        )
        & (i32[..., EI_STATUS_OVERLAP_MODE] >= STATUS_OVERLAP_IGNORE)
        & (i32[..., EI_STATUS_OVERLAP_MODE] <= STATUS_OVERLAP_OVERWRITE)
        & (i32[..., EI_DAMAGE_CAUSE] >= 0)
        & (i32[..., EI_DAMAGE_CAUSE] < DAMAGE_COUNT),
        i32[..., EI_PROJECTILE_KIND] != PROJECTILE_DEPLOYABLE,
    )
    valid_program = (
        (i32[..., EI_PROJECTILE_KIND] > 0)
        & jnp.isfinite(f32[..., EF_DAMAGE])
        & (f32[..., EF_DAMAGE] >= 0.0)
        & jnp.isfinite(f32[..., EF_RANDOM_PERCENTAGE])
        & (f32[..., EF_RANDOM_PERCENTAGE] >= 0.0)
        & (i32[..., EI_DAMAGE_CLASS] >= 0)
        & (i32[..., EI_DAMAGE_CLASS] < DAMAGE_CLASS_COUNT)
        & jnp.isfinite(f32[..., EF_PROJECTILE_SPEED])
        & (f32[..., EF_PROJECTILE_SPEED] > 0.0)
        & jnp.isfinite(f32[..., EF_PROJECTILE_GRAVITY])
        & (f32[..., EF_PROJECTILE_GRAVITY] >= 0.0)
        & jnp.isfinite(f32[..., EF_PROJECTILE_TERMINAL_VELOCITY])
        & (f32[..., EF_PROJECTILE_TERMINAL_VELOCITY] > 0.0)
        & jnp.all(jnp.isfinite(collision_half_extent), axis=2)
        & jnp.all(collision_half_extent > 0.0, axis=2)
        & jnp.all(jnp.isfinite(collision_center_offset), axis=2)
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
            axis=2,
        )
        & jnp.where(
            terminal.requested,
            terminal.deployable_max_live_count > 0,
            (i32[..., EI_FORCE_DIRECTION_MODE] >= FORCE_DIRECTION_LOCAL)
            & (i32[..., EI_FORCE_DIRECTION_MODE] <= FORCE_DIRECTION_POINT),
        )
        & jnp.isfinite(f32[..., EF_RADIUS])
        & (f32[..., EF_RADIUS] >= 0.0)
        & jnp.where(
            terminal.requested,
            (i32[..., EI_BLOCK_DAMAGE_RADIUS] == 0)
            | (i32[..., EI_BLOCK_DAMAGE_RADIUS] == 1),
            jnp.where(
                f32[..., EF_RADIUS] > 0.0,
                i32[..., EI_BLOCK_DAMAGE_RADIUS] > 0,
                i32[..., EI_BLOCK_DAMAGE_RADIUS] == 0,
            ),
        )
        & (i32[..., EI_ON_HIT_RESOURCE_ID] >= -1)
        & (i32[..., EI_ON_HIT_RESOURCE_ID] < RESOURCE_COUNT)
        & jnp.isfinite(f32[..., EF_ON_HIT_RESOURCE_DELTA])
        & valid_standard_physics
        & terminal_payload_valid
    )
    count = jnp.sum(requested.astype(jnp.int32), axis=1)
    available = jnp.sum((~state.projectiles.active).astype(jnp.int32), axis=1)
    bits = state.failure_bits
    bits = _set_failure(
        bits,
        jnp.any(requested & ~valid_program, axis=1),
        ARSENAL_FAILURE_INVALID_LOADOUT,
    )
    # ``clear`` is an outcome, while ``world_collision_available`` says that
    # exact contact can be resolved after launch.  A delayed projectile may
    # acquire an obstruction after its ability was accepted; that is ordinary
    # terrain contact, not missing World evidence.
    bits = _set_failure(
        bits,
        jnp.any(
            requested
            & ~(
                muzzle_valid
                & finite_transform
                & (clear | world_collision_available)
                & (~terminal.requested | world_collision_available)
                & (~terminal.requested | contact_chain_available)
            ),
            axis=1,
        ),
        ARSENAL_FAILURE_UNSUPPORTED_WORLD,
    )
    bits = _set_failure(
        bits,
        count > available,
        ARSENAL_FAILURE_PROJECTILE_OVERFLOW,
    )
    row_valid = bits == jnp.uint32(0)
    direction = projectile_look_direction(launch_yaw, launch_pitch)
    force = _rotate_local_direction(
        f32[..., [EF_FORCE_X, EF_FORCE_Y, EF_FORCE_Z]],
        yaw,
    )
    force = jnp.where(terminal.requested[..., None], jnp.float32(0.0), force)
    inputs = (
        jnp.swapaxes(requested & row_valid[:, None], 0, 1),
        jnp.swapaxes(world_collision_available, 0, 1),
        jnp.swapaxes(flat.source_entity_id, 0, 1),
        jnp.swapaxes(crossbow, 0, 1),
        jnp.swapaxes(combo_damage, 0, 1),
        jnp.swapaxes(launch_yaw, 0, 1),
        jnp.swapaxes(position, 0, 1),
        jnp.swapaxes(direction, 0, 1),
        jnp.swapaxes(force, 0, 1),
        jnp.swapaxes(terminal.requested, 0, 1),
        jnp.swapaxes(contact_chain_available, 0, 1),
        jnp.swapaxes(collision_half_extent, 0, 1),
        jnp.swapaxes(collision_center_offset, 0, 1),
        jnp.swapaxes(terminal.deployable_collision_half_extent, 0, 1),
        jnp.swapaxes(terminal.deployable_collision_center_offset, 0, 1),
        jnp.swapaxes(terminal.deployable_id, 0, 1),
        jnp.swapaxes(
            terminal.deployable_count_towards_global_limit,
            0,
            1,
        ),
        jnp.swapaxes(terminal.deployable_max_live_count, 0, 1),
        jnp.swapaxes(terminal.area_duration_seconds, 0, 1),
        jnp.swapaxes(terminal.area_interval_seconds, 0, 1),
        jnp.swapaxes(terminal.area_end_radius, 0, 1),
        jnp.swapaxes(terminal.area_height, 0, 1),
        jnp.swapaxes(terminal.area_radius_change_seconds, 0, 1),
        jnp.swapaxes(terminal.area_shape, 0, 1),
        jnp.swapaxes(terminal.area_attack_flags, 0, 1),
        jnp.swapaxes(terminal.sticks_vertically, 0, 1),
        jnp.swapaxes(f32, 0, 1),
        jnp.swapaxes(i32, 0, 1),
        jnp.swapaxes(flat.flags, 0, 1),
    )

    def spawn_one(projectiles: ProjectileState, values):
        (
            active,
            world_collision,
            owner,
            crossbow_program,
            crossbow_combo_damage,
            yaw_degrees,
            origin,
            heading,
            force_direction,
            terminal_deployable,
            terminal_intended_graph_available,
            collision_extent,
            collision_offset,
            deployable_collision_extent,
            deployable_collision_offset,
            deployable_id,
            deployable_count_towards_global_limit,
            deployable_max_live_count,
            terminal_area_duration,
            terminal_area_interval,
            terminal_area_end_radius,
            terminal_area_height,
            terminal_area_radius_change,
            terminal_area_shape,
            terminal_area_attack_flags,
            sticks_vertically,
            event_f32,
            event_i32,
            flags,
        ) = values
        slot = jnp.argmax(~projectiles.active, axis=1)
        selected = (
            jax.nn.one_hot(
                slot,
                projectiles.active.shape[1],
                dtype=jnp.bool_,
            )
            & active[:, None]
        )
        projectiles = projectiles._replace(
            position=_write_slot(projectiles.position, origin, selected),
            velocity=_write_slot(
                projectiles.velocity,
                heading * event_f32[:, EF_PROJECTILE_SPEED, None],
                selected,
            ),
            launch_yaw_degrees=_write_slot(
                projectiles.launch_yaw_degrees,
                yaw_degrees,
                selected,
            ),
            half_extent=_write_slot(
                projectiles.half_extent,
                collision_extent,
                selected,
            ),
            collision_center_offset=_write_slot(
                projectiles.collision_center_offset,
                collision_offset,
                selected,
            ),
            age_seconds=_write_slot(
                projectiles.age_seconds,
                jnp.zeros_like(event_f32[:, 0]),
                selected,
            ),
            lifetime_seconds=_write_slot(
                projectiles.lifetime_seconds,
                event_f32[:, EF_PROJECTILE_LIFETIME_SECONDS],
                selected,
            ),
            damage=_write_slot(
                projectiles.damage,
                event_f32[:, EF_DAMAGE],
                selected,
            ),
            direct_damage=_write_slot(
                projectiles.direct_damage,
                jnp.where(
                    terminal_deployable,
                    jnp.float32(0.0),
                    event_f32[:, EF_PROJECTILE_DIRECT_DAMAGE],
                ),
                selected,
            ),
            random_percentage=_write_slot(
                projectiles.random_percentage,
                event_f32[:, EF_RANDOM_PERCENTAGE],
                selected,
            ),
            damage_cause=_write_slot(
                projectiles.damage_cause,
                event_i32[:, EI_DAMAGE_CAUSE],
                selected,
            ),
            direct_damage_cause=_write_slot(
                projectiles.direct_damage_cause,
                jnp.where(
                    terminal_deployable,
                    event_i32[:, EI_DAMAGE_CAUSE],
                    event_i32[:, EI_PROJECTILE_DIRECT_DAMAGE_CAUSE],
                ),
                selected,
            ),
            damage_class=_write_slot(
                projectiles.damage_class,
                event_i32[:, EI_DAMAGE_CLASS],
                selected,
            ),
            gravity=_write_slot(
                projectiles.gravity,
                event_f32[:, EF_PROJECTILE_GRAVITY],
                selected,
            ),
            terminal_velocity=_write_slot(
                projectiles.terminal_velocity,
                event_f32[:, EF_PROJECTILE_TERMINAL_VELOCITY],
                selected,
            ),
            standard_physics=_write_slot(
                projectiles.standard_physics,
                (flags & jnp.uint32(EVENT_FLAG_PROJECTILE_STANDARD_PHYSICS)) != 0,
                selected,
            ),
            bounciness=_write_slot(
                projectiles.bounciness,
                event_f32[:, EF_PROJECTILE_BOUNCINESS],
                selected,
            ),
            bounce_limit=_write_slot(
                projectiles.bounce_limit,
                event_f32[:, EF_PROJECTILE_BOUNCE_LIMIT],
                selected,
            ),
            bounce_count_limit=_write_slot(
                projectiles.bounce_count_limit,
                event_i32[:, EI_PROJECTILE_BOUNCE_COUNT],
                selected,
            ),
            bounce_count=_write_slot(
                projectiles.bounce_count,
                jnp.zeros_like(event_i32[:, 0]),
                selected,
            ),
            allow_rolling=_write_slot(
                projectiles.allow_rolling,
                (flags & jnp.uint32(EVENT_FLAG_PROJECTILE_ALLOW_ROLLING)) != 0,
                selected,
            ),
            rolling_friction_factor=_write_slot(
                projectiles.rolling_friction_factor,
                event_f32[:, EF_PROJECTILE_ROLLING_FRICTION_FACTOR],
                selected,
            ),
            sticks_vertically=_write_slot(
                projectiles.sticks_vertically,
                jnp.where(
                    terminal_deployable,
                    sticks_vertically,
                    jnp.zeros_like(sticks_vertically),
                ),
                selected,
            ),
            on_ground=_write_slot(
                projectiles.on_ground,
                jnp.zeros_like(active),
                selected,
            ),
            fuse_seconds=_write_slot(
                projectiles.fuse_seconds,
                event_f32[:, EF_PROJECTILE_FUSE_SECONDS],
                selected,
            ),
            dead_time_seconds=_write_slot(
                projectiles.dead_time_seconds,
                event_f32[:, EF_PROJECTILE_DEAD_TIME_SECONDS],
                selected,
            ),
            dead_time_remaining=_write_slot(
                projectiles.dead_time_remaining,
                jnp.zeros_like(event_f32[:, 0]),
                selected,
            ),
            explosion_radius=_write_slot(
                projectiles.explosion_radius,
                event_f32[:, EF_RADIUS],
                selected,
            ),
            explosion_falloff=_write_slot(
                projectiles.explosion_falloff,
                jnp.where(
                    terminal_deployable,
                    jnp.float32(0.0),
                    event_f32[:, EF_FALLOFF],
                ),
                selected,
            ),
            block_damage_radius=_write_slot(
                projectiles.block_damage_radius,
                jnp.where(
                    terminal_deployable,
                    jnp.int32(0),
                    event_i32[:, EI_BLOCK_DAMAGE_RADIUS],
                ),
                selected,
            ),
            damage_blocks=_write_slot(
                projectiles.damage_blocks,
                ~terminal_deployable
                & (event_f32[:, EF_RADIUS] > 0.0)
                & ((flags & jnp.uint32(EVENT_FLAG_ENTITY_ONLY_AREA)) == 0),
                selected,
            ),
            force_direction=_write_slot(
                projectiles.force_direction,
                force_direction,
                selected,
            ),
            force_direction_mode=_write_slot(
                projectiles.force_direction_mode,
                jnp.where(
                    terminal_deployable,
                    jnp.int32(FORCE_DIRECTION_LOCAL),
                    event_i32[:, EI_FORCE_DIRECTION_MODE],
                ),
                selected,
            ),
            force_velocity_y=_write_slot(
                projectiles.force_velocity_y,
                jnp.where(
                    terminal_deployable,
                    jnp.float32(0.0),
                    event_f32[:, EF_FORCE_Y],
                ),
                selected,
            ),
            force_magnitude=_write_slot(
                projectiles.force_magnitude,
                event_f32[:, EF_FORCE_MAGNITUDE],
                selected,
            ),
            force_mode=_write_slot(
                projectiles.force_mode,
                jnp.where(
                    terminal_deployable,
                    jnp.int32(FORCE_SET),
                    event_i32[:, EI_FORCE_MODE],
                ),
                selected,
            ),
            air_resistance=_write_slot(
                projectiles.air_resistance,
                event_f32[:, EF_AIR_RESISTANCE],
                selected,
            ),
            air_resistance_max=_write_slot(
                projectiles.air_resistance_max,
                event_f32[:, EF_AIR_RESISTANCE_MAX],
                selected,
            ),
            ground_resistance=_write_slot(
                projectiles.ground_resistance,
                event_f32[:, EF_GROUND_RESISTANCE],
                selected,
            ),
            ground_resistance_max=_write_slot(
                projectiles.ground_resistance_max,
                event_f32[:, EF_GROUND_RESISTANCE_MAX],
                selected,
            ),
            resistance_threshold=_write_slot(
                projectiles.resistance_threshold,
                event_f32[:, EF_RESISTANCE_THRESHOLD],
                selected,
            ),
            resistance_style=_write_slot(
                projectiles.resistance_style,
                jnp.where(
                    terminal_deployable,
                    jnp.int32(1),
                    event_i32[:, EI_RESISTANCE_STYLE],
                ),
                selected,
            ),
            on_hit_resource_id=_write_slot(
                projectiles.on_hit_resource_id,
                event_i32[:, EI_ON_HIT_RESOURCE_ID],
                selected,
            ),
            on_hit_resource_delta=_write_slot(
                projectiles.on_hit_resource_delta,
                event_f32[:, EF_ON_HIT_RESOURCE_DELTA],
                selected,
            ),
            on_hit_healing=_write_slot(
                projectiles.on_hit_healing,
                event_f32[:, EF_STATUS_HEALING],
                selected,
            ),
            status_id=_write_slot(
                projectiles.status_id,
                event_i32[:, EI_STATUS_ID],
                selected,
            ),
            status_duration_seconds=_write_slot(
                projectiles.status_duration_seconds,
                event_f32[:, EF_STATUS_DURATION_SECONDS],
                selected,
            ),
            status_cooldown_seconds=_write_slot(
                projectiles.status_cooldown_seconds,
                event_f32[:, EF_STATUS_COOLDOWN_SECONDS],
                selected,
            ),
            status_damage=_write_slot(
                projectiles.status_damage,
                event_f32[:, EF_STATUS_DAMAGE],
                selected,
            ),
            status_damage_cause=_write_slot(
                projectiles.status_damage_cause,
                event_i32[:, EI_STATUS_DAMAGE_CAUSE],
                selected,
            ),
            status_resource_id=_write_slot(
                projectiles.status_resource_id,
                event_i32[:, EI_STATUS_RESOURCE_ID],
                selected,
            ),
            status_resource_delta=_write_slot(
                projectiles.status_resource_delta,
                event_f32[:, EF_STATUS_RESOURCE_DELTA],
                selected,
            ),
            status_speed_multiplier=_write_slot(
                projectiles.status_speed_multiplier,
                event_f32[:, EF_STATUS_SPEED_MULTIPLIER],
                selected,
            ),
            status_flags=_write_slot(
                projectiles.status_flags,
                flags & jnp.uint32(EVENT_STATUS_FLAG_MASK),
                selected,
            ),
            status_overlap_mode=_write_slot(
                projectiles.status_overlap_mode,
                event_i32[:, EI_STATUS_OVERLAP_MODE],
                selected,
            ),
            terminal_deployable_area=_write_slot(
                projectiles.terminal_deployable_area,
                terminal_deployable,
                selected,
            ),
            terminal_intended_graph_available=_write_slot(
                projectiles.terminal_intended_graph_available,
                terminal_intended_graph_available & terminal_deployable,
                selected,
            ),
            terminal_entity_contact_inactive=_write_slot(
                projectiles.terminal_entity_contact_inactive,
                jnp.zeros_like(active),
                selected,
            ),
            terminal_deployable_collision_half_extent=_write_slot(
                projectiles.terminal_deployable_collision_half_extent,
                deployable_collision_extent,
                selected,
            ),
            terminal_deployable_collision_center_offset=_write_slot(
                projectiles.terminal_deployable_collision_center_offset,
                deployable_collision_offset,
                selected,
            ),
            terminal_deployable_id=_write_slot(
                projectiles.terminal_deployable_id,
                jnp.where(terminal_deployable, deployable_id, jnp.int32(0)),
                selected,
            ),
            terminal_deployable_count_towards_global_limit=_write_slot(
                projectiles.terminal_deployable_count_towards_global_limit,
                terminal_deployable & deployable_count_towards_global_limit,
                selected,
            ),
            terminal_deployable_max_live_count=_write_slot(
                projectiles.terminal_deployable_max_live_count,
                jnp.where(
                    terminal_deployable,
                    deployable_max_live_count,
                    jnp.int32(0),
                ),
                selected,
            ),
            terminal_area_shape=_write_slot(
                projectiles.terminal_area_shape,
                jnp.where(
                    terminal_deployable,
                    terminal_area_shape,
                    jnp.int32(0),
                ),
                selected,
            ),
            terminal_area_duration_seconds=_write_slot(
                projectiles.terminal_area_duration_seconds,
                terminal_area_duration,
                selected,
            ),
            terminal_area_interval_seconds=_write_slot(
                projectiles.terminal_area_interval_seconds,
                terminal_area_interval,
                selected,
            ),
            terminal_area_end_radius=_write_slot(
                projectiles.terminal_area_end_radius,
                terminal_area_end_radius,
                selected,
            ),
            terminal_area_height=_write_slot(
                projectiles.terminal_area_height,
                terminal_area_height,
                selected,
            ),
            terminal_area_radius_change_seconds=_write_slot(
                projectiles.terminal_area_radius_change_seconds,
                terminal_area_radius_change,
                selected,
            ),
            terminal_area_attack_flags=_write_slot(
                projectiles.terminal_area_attack_flags,
                jnp.where(
                    terminal_deployable,
                    terminal_area_attack_flags,
                    jnp.int32(0),
                ),
                selected,
            ),
            kind=_write_slot(
                projectiles.kind,
                event_i32[:, EI_PROJECTILE_KIND],
                selected,
            ),
            crossbow_program=_write_slot(
                projectiles.crossbow_program,
                crossbow_program,
                selected,
            ),
            crossbow_combo_damage=_write_slot(
                projectiles.crossbow_combo_damage,
                crossbow_combo_damage,
                selected,
            ),
            owner_entity_id=_write_slot(
                projectiles.owner_entity_id,
                owner,
                selected,
            ),
            active=projectiles.active | selected,
            impacted=_write_slot(
                projectiles.impacted,
                jnp.zeros_like(active),
                selected,
            ),
            physics_initialized=_write_slot(
                projectiles.physics_initialized,
                jnp.zeros_like(active),
                selected,
            ),
            entity_collision_only=_write_slot(
                projectiles.entity_collision_only,
                ~world_collision,
                selected,
            ),
            world_hit=_write_slot(
                projectiles.world_hit,
                jnp.zeros_like(active),
                selected,
            ),
            world_hit_fraction=_write_slot(
                projectiles.world_hit_fraction,
                jnp.ones_like(event_f32[:, 0]),
                selected,
            ),
            world_contact_point=_write_slot(
                projectiles.world_contact_point,
                jnp.zeros_like(origin),
                selected,
            ),
            world_contact_normal=_write_slot(
                projectiles.world_contact_normal,
                jnp.zeros_like(origin),
                selected,
            ),
            world_geometry_exhausted=_write_slot(
                projectiles.world_geometry_exhausted,
                jnp.zeros_like(active),
                selected,
            ),
            world_capacity_exceeded=_write_slot(
                projectiles.world_capacity_exceeded,
                jnp.zeros_like(active),
                selected,
            ),
            world_contact_invalid=_write_slot(
                projectiles.world_contact_invalid,
                jnp.zeros_like(active),
                selected,
            ),
            world_segment_count=_write_slot(
                projectiles.world_segment_count,
                jnp.zeros_like(event_i32[:, 0]),
                selected,
            ),
        )
        return projectiles, None

    candidate, _ = jax.lax.scan(spawn_one, state.projectiles, inputs)
    result = _select_tree(row_valid, candidate, state.projectiles)
    spawned = requested & row_valid[:, None]
    spawned_by_entity = jnp.sum(
        jax.nn.one_hot(
            source,
            entity_count,
            dtype=jnp.int32,
        )
        * spawned[..., None].astype(jnp.int32),
        axis=1,
        dtype=jnp.int32,
    )
    return (
        state._replace(projectiles=result, failure_bits=bits),
        jnp.where(row_valid, count, jnp.int32(0)),
        spawned_by_entity,
    )


def projectile_entity_first_contact(
    position: jax.Array,
    displacement: jax.Array,
    half_extent: jax.Array,
    target_position: jax.Array,
    target_bounds: jax.Array,
    target_mask: jax.Array,
    collision_center_offset: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return nearest target-box contact for arbitrary leading axes."""

    center_offset = (
        jnp.zeros_like(half_extent)
        if collision_center_offset is None
        else jnp.asarray(collision_center_offset, dtype=jnp.float32)
    )
    collision_min = center_offset - half_extent
    collision_max = center_offset + half_extent
    box_min = target_position + target_bounds[..., :3] - collision_max
    box_max = target_position + target_bounds[..., 3:] - collision_min
    hit, fraction = _segment_aabb(
        position,
        position + displacement,
        box_min,
        box_max,
    )
    return hit & target_mask, fraction


def _projectile_nearest_entity_first_contact(
    position: jax.Array,
    displacement: jax.Array,
    half_extent: jax.Array,
    entity_position: jax.Array,
    entity_bounds: jax.Array,
    candidate_mask: jax.Array,
    owner_entity_id: jax.Array,
    collision_center_offset: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return the nearest admitted non-owner contact for each projectile.

    Hytale's ``EntityCollisionProvider.computeNearest`` sweeps every collidable
    entity and ignores the projectile plus its creator. The caller supplies
    the complete live/team-policy candidate mask. Exact-fraction ties use the
    lowest entity index; the server's spatial-iteration tie order is
    intentionally not claimed.
    """

    entity_count = entity_position.shape[1]
    candidate_id = jnp.arange(entity_count, dtype=jnp.int32)[None, None, :]
    if candidate_mask.ndim == 2:
        candidate_mask = candidate_mask[:, None, :]
    if candidate_mask.shape != (
        entity_position.shape[0],
        position.shape[1],
        entity_count,
    ):
        raise ValueError("candidate_mask must have shape (batch, projectile, entity)")
    candidate_mask &= candidate_id != owner_entity_id[:, :, None]
    candidate_hit, candidate_fraction = projectile_entity_first_contact(
        position[:, :, None, :],
        displacement[:, :, None, :],
        half_extent[:, :, None, :],
        entity_position[:, None, :, :],
        entity_bounds[:, None, :, :],
        candidate_mask,
        (
            None
            if collision_center_offset is None
            else collision_center_offset[:, :, None, :]
        ),
    )
    ordered_fraction = jnp.where(
        candidate_hit,
        candidate_fraction,
        jnp.float32(jnp.inf),
    )
    nearest_entity_id = jnp.argmin(ordered_fraction, axis=2).astype(jnp.int32)
    nearest_hit = jnp.any(candidate_hit, axis=2)
    nearest_fraction = jnp.min(ordered_fraction, axis=2)
    return (
        nearest_hit,
        jnp.where(nearest_hit, nearest_fraction, jnp.float32(1.0)),
        nearest_entity_id,
    )


def tick_projectiles(
    combat: CombatState,
    state: ArsenalState,
    dt_seconds: jax.Array,
    params: CombatParams,
    rules: CombatMechanicsRules,
    *,
    entity_team_id: jax.Array | None = None,
    entity_projectile_collidable_mask: jax.Array | None = None,
    friendly_fire: bool | jax.Array = True,
    geometry: GeometryProvider | None = None,
    explosion_candidate_provider: (ArsenalExplosionCandidateProvider | None) = None,
    include_crossbow_program: bool = False,
):
    """Advance projectiles with entity-first, exact-world contact ordering."""

    projectile = state.projectiles
    batch, entity_count = combat.health.shape
    dt = _batch_dt(dt_seconds, batch)
    invalid = (
        _invalid_projectile_state(
            projectile,
            entity_count,
        )
        | ~jnp.isfinite(dt)[:, None]
    )
    bits = _set_failure(
        state.failure_bits,
        jnp.any(invalid, axis=1) | (dt < 0.0),
        ARSENAL_FAILURE_INVALID_STATE,
    )
    preliminary_valid = bits == jnp.uint32(0)
    delta = dt[:, None]
    ticking_dead = projectile.active & projectile.impacted & preliminary_valid[:, None]
    dead_remaining = jnp.where(
        ticking_dead,
        projectile.dead_time_remaining - delta,
        projectile.dead_time_remaining,
    )
    removed_dead = ticking_dead & (dead_remaining <= jnp.float32(0.0))
    active_before = projectile.active & ~removed_dead
    free = (
        active_before
        & ~projectile.impacted
        & ~projectile.terminal_entity_contact_inactive
        & preliminary_valid[:, None]
    )
    velocity = _standard_velocity(projectile, delta)
    end = projectile.position + velocity * delta[..., None]
    displacement = end - projectile.position
    relationship_mask = relationship_target_mask(
        combat.health,
        projectile.owner_entity_id,
        entity_team_id=entity_team_id,
        friendly_fire=friendly_fire,
    )
    if entity_projectile_collidable_mask is None:
        terminal_collision_candidates = jnp.zeros(
            (batch, entity_count),
            dtype=jnp.bool_,
        )
        terminal_collision_available = jnp.zeros((batch,), dtype=jnp.bool_)
    else:
        terminal_collision_candidates = jnp.asarray(entity_projectile_collidable_mask)
        if terminal_collision_candidates.dtype != jnp.bool_:
            raise TypeError(
                "entity_projectile_collidable_mask must have dtype bool, "
                f"got {terminal_collision_candidates.dtype}"
            )
        if terminal_collision_candidates.shape != (batch, entity_count):
            raise ValueError("entity_projectile_collidable_mask must be bool[B,E]")
        terminal_collision_available = jnp.ones((batch,), dtype=jnp.bool_)
    bits = _set_failure(
        bits,
        jnp.any(
            projectile.active & projectile.terminal_deployable_area,
            axis=1,
        )
        & ~terminal_collision_available,
        ARSENAL_FAILURE_UNSUPPORTED_WORLD,
    )
    relationship_mask = jnp.where(
        projectile.terminal_deployable_area[..., None],
        terminal_collision_candidates[:, None, :],
        relationship_mask,
    )
    # Typed deployables are tangible ECS entities with their own model AABB.
    # AreaState cannot model arbitrary damage/death against that entity, but
    # it can conservatively detect projectile contact and reject the complete
    # row before any projectile/area mutation. Full-life noninterference is
    # still required at admission for non-projectile attacks.
    deployable_bounds = jnp.concatenate(
        (
            state.areas.collision_center_offset - state.areas.collision_half_extent,
            state.areas.collision_center_offset + state.areas.collision_half_extent,
        ),
        axis=2,
    )
    deployable_hit, deployable_hit_fraction = projectile_entity_first_contact(
        projectile.position[:, :, None, :],
        displacement[:, :, None, :],
        projectile.half_extent[:, :, None, :],
        state.areas.center[:, None, :, :],
        deployable_bounds[:, None, :, :],
        state.areas.active[:, None, :]
        & (state.areas.kind[:, None, :] == AREA_DEPLOYABLE_AOE),
        projectile.collision_center_offset[:, :, None, :],
    )
    entity_bounds = (
        jnp.broadcast_to(
            params.target_bounds[None, None, :],
            (batch, entity_count, 6),
        )
        .at[:, AGENT_ENTITY]
        .set(params.agent_bounds)
    )
    entity_hit, entity_hit_fraction, contact_victim = (
        _projectile_nearest_entity_first_contact(
            projectile.position,
            displacement,
            projectile.half_extent,
            combat.position,
            entity_bounds,
            relationship_mask,
            projectile.owner_entity_id,
            projectile.collision_center_offset,
        )
    )
    entity_hit &= free
    moving = jnp.any(
        jnp.abs(displacement) > jnp.float32(1.0e-8),
        axis=2,
    )
    needs_world_contact = free & ~projectile.entity_collision_only & moving
    if geometry is None:
        world_available = jnp.zeros_like(entity_hit)
        world_hit = jnp.zeros_like(entity_hit)
        world_hit_fraction = jnp.ones_like(entity_hit_fraction)
        world_contact_point = jnp.zeros_like(projectile.position)
        world_contact_normal = jnp.zeros_like(projectile.position)
        world_geometry_exhausted = jnp.zeros_like(entity_hit)
        world_capacity_exceeded = jnp.zeros_like(entity_hit)
        world_contact_invalid = jnp.zeros_like(entity_hit)
        world_segment_count = jnp.zeros_like(
            entity_hit_fraction,
            dtype=jnp.int32,
        )
        queried_world = jnp.zeros_like(entity_hit)
    else:
        if isinstance(geometry, GeometryState) and geometry.origin.shape[0] == 1:
            projectile_count = projectile.active.shape[1]
            world_contact = geometry_projectile_first_contact_result(
                geometry,
                projectile.position.reshape((batch * projectile_count, 3)),
                displacement.reshape((batch * projectile_count, 3)),
                jnp.concatenate(
                    (
                        projectile.collision_center_offset - projectile.half_extent,
                        projectile.collision_center_offset + projectile.half_extent,
                    ),
                    axis=2,
                ).reshape((batch * projectile_count, 6)),
                entity_hit_fraction.reshape((batch * projectile_count,)),
                entity_hit.reshape((batch * projectile_count,)),
            )
            world_contact = jax.tree.map(
                lambda value: value.reshape(
                    (batch, projectile_count) + value.shape[1:]
                ),
                world_contact,
            )
        else:
            world_contact = jax.vmap(
                lambda position, motion, extent, offset, fraction, mask: (
                    geometry_projectile_first_contact_result(
                        geometry,
                        position,
                        motion,
                        jnp.concatenate(
                            (offset - extent, offset + extent),
                            axis=1,
                        ),
                        fraction,
                        mask,
                    )
                ),
                in_axes=(1, 1, 1, 1, 1, 1),
                out_axes=1,
            )(
                projectile.position,
                displacement,
                projectile.half_extent,
                projectile.collision_center_offset,
                entity_hit_fraction,
                entity_hit,
            )
        world_available = world_contact.available
        world_hit = world_contact.world_hit
        world_hit_fraction = world_contact.hit_fraction
        world_contact_point = world_contact.contact_point
        world_contact_normal = world_contact.contact_normal
        world_geometry_exhausted = world_contact.geometry_exhausted
        world_capacity_exceeded = world_contact.capacity_exceeded
        world_contact_invalid = world_contact.invalid
        world_segment_count = world_contact.segment_count
        queried_world = preliminary_valid[:, None] & needs_world_contact
    # Reject only when the effect-only AreaState deployable is the first
    # reachable contact. A nearer ordinary entity, static-world hit, or fuse
    # owns the step and must not be turned into an avoidable false negative.
    # Exact-fraction ties fail closed because native spatial iteration order is
    # not represented by this bounded state.
    nearest_deployable_fraction = jnp.min(
        jnp.where(
            deployable_hit,
            deployable_hit_fraction,
            jnp.float32(jnp.inf),
        ),
        axis=2,
    )
    pre_after_age = projectile.age_seconds + jnp.where(
        active_before,
        delta,
        0.0,
    )
    pre_fuse = (
        free
        & (projectile.fuse_seconds > 0.0)
        & (projectile.age_seconds < projectile.fuse_seconds)
        & (pre_after_age >= projectile.fuse_seconds)
    )
    pre_fuse_fraction = jnp.clip(
        (projectile.fuse_seconds - projectile.age_seconds)
        / jnp.maximum(delta, jnp.finfo(jnp.float32).tiny),
        0.0,
        1.0,
    )
    earlier_supported_fraction = jnp.minimum(
        jnp.where(entity_hit, entity_hit_fraction, jnp.float32(jnp.inf)),
        jnp.where(world_hit, world_hit_fraction, jnp.float32(jnp.inf)),
    )
    earlier_supported_fraction = jnp.minimum(
        earlier_supported_fraction,
        jnp.where(pre_fuse, pre_fuse_fraction, jnp.float32(jnp.inf)),
    )
    deployable_first = (
        free
        & jnp.any(deployable_hit, axis=2)
        & (nearest_deployable_fraction <= earlier_supported_fraction)
    )
    bits = _set_failure(
        bits,
        jnp.any(deployable_first, axis=1),
        ARSENAL_FAILURE_UNSUPPORTED_WORLD,
    )
    bits = _set_failure(
        bits,
        jnp.any(needs_world_contact & ~world_available, axis=1),
        ARSENAL_FAILURE_UNSUPPORTED_WORLD,
    )
    row_valid = bits == jnp.uint32(0)
    free &= row_valid[:, None]
    entity_hit &= row_valid[:, None]
    world_hit &= needs_world_contact & row_valid[:, None]
    after_age = projectile.age_seconds + jnp.where(
        active_before & row_valid[:, None],
        delta,
        0.0,
    )
    fuse = (
        free
        & (projectile.fuse_seconds > 0.0)
        & (projectile.age_seconds < projectile.fuse_seconds)
        & (after_age >= projectile.fuse_seconds)
    )
    safe_dt = jnp.maximum(delta, jnp.finfo(jnp.float32).tiny)
    fuse_fraction = jnp.clip(
        (projectile.fuse_seconds - projectile.age_seconds) / safe_dt,
        0.0,
        1.0,
    )
    world_before_fuse = world_hit & (~fuse | (world_hit_fraction <= fuse_fraction))
    standard_contact = (
        world_hit
        & projectile.standard_physics
        # Timer expiry owns an exact-fraction tie. A strictly earlier contact
        # receives the native block response before the timer can fire.
        & (~fuse | (world_hit_fraction < fuse_fraction))
    )
    contact_response = standard_projectile_contact_response(
        contact_mask=standard_contact,
        end=end,
        contact_point=world_contact_point,
        contact_normal=world_contact_normal,
        velocity=velocity,
        dt_seconds=delta,
        bounciness=projectile.bounciness,
        bounce_limit=projectile.bounce_limit,
        bounce_count_limit=projectile.bounce_count_limit,
        bounce_count=projectile.bounce_count,
        allow_rolling=projectile.allow_rolling,
        rolling_friction_factor=projectile.rolling_friction_factor,
        sticks_vertically=projectile.sticks_vertically,
    )
    continued_contact = (
        contact_response.bounced | contact_response.rolling | contact_response.stopped
    )
    world_terminal = world_before_fuse & (
        ~projectile.standard_physics | contact_response.impacted
    )
    hit = entity_hit & ~world_before_fuse
    terminal_entity_contact = hit & projectile.terminal_deployable_area
    effect_hit = hit & ~projectile.terminal_deployable_area
    trigger_fraction = jnp.minimum(
        jnp.where(effect_hit, entity_hit_fraction, jnp.float32(1.0)),
        jnp.where(
            world_terminal,
            world_hit_fraction,
            jnp.float32(1.0),
        ),
    )
    trigger_fraction = jnp.minimum(
        trigger_fraction,
        jnp.where(fuse, fuse_fraction, jnp.float32(1.0)),
    )
    triggered = effect_hit | world_terminal | fuse
    computed_impact = projectile.position + displacement * trigger_fraction[..., None]
    world_trigger = world_terminal & (world_hit_fraction == trigger_fraction)
    impact = jnp.where(
        world_trigger[..., None],
        world_contact_point,
        computed_impact,
    )
    terminal_requested = (
        world_trigger
        & projectile.terminal_deployable_area
        & projectile.terminal_intended_graph_available
    )
    terminal_state, _ = spawn_terminal_areas(
        state._replace(failure_bits=bits),
        projectile,
        terminal_requested,
        world_contact_point,
        world_contact_normal,
    )
    bits = terminal_state.failure_bits
    terminal_areas = terminal_state.areas
    has_explosion = (
        projectile.explosion_radius > 0.0
    ) & ~projectile.terminal_deployable_area
    # ExplosionUtils.processTargetEntity measures to the entity transform,
    # not to the closest point on its collision bounds.
    target_offset = combat.position[:, None, :, :] - impact[:, :, None, :]
    transform_distance = jnp.linalg.norm(target_offset, axis=3)
    explosion_requested = triggered & has_explosion
    if explosion_candidate_provider is None:
        explosion_distance = transform_distance
        explosion_target_mask = (
            explosion_requested[..., None]
            & relationship_mask
            & (explosion_distance < projectile.explosion_radius[..., None])
        )
    else:
        explosion_candidates = explosion_candidate_provider(
            combat,
            impact,
            projectile.block_damage_radius,
            projectile.explosion_radius,
            projectile.damage_blocks,
            explosion_requested,
            relationship_mask,
        )
        _validate_explosion_candidate_shapes(
            explosion_candidates,
            batch,
            projectile.active.shape[1],
            entity_count,
        )
        admitted = explosion_candidates.candidate_mask
        admitted_distance = explosion_candidates.distance
        admitted_invalid = jnp.any(
            admitted
            & (
                ~relationship_mask
                | ~explosion_requested[..., None]
                | ~jnp.isfinite(admitted_distance)
                | (admitted_distance < jnp.float32(0.0))
                | (admitted_distance >= projectile.explosion_radius[..., None])
            ),
            axis=2,
        )
        query_invalid = explosion_requested & (
            explosion_candidates.invalid | admitted_invalid
        )
        query_unavailable = explosion_requested & (
            ~explosion_candidates.available
            | explosion_candidates.geometry_exhausted
            | explosion_candidates.capacity_exceeded
            | explosion_candidates.damage_blocks_unsupported
            | projectile.damage_blocks
        )
        bits = _set_failure(
            bits,
            jnp.any(query_invalid, axis=1),
            ARSENAL_FAILURE_INVALID_STATE,
        )
        bits = _set_failure(
            bits,
            jnp.any(query_unavailable, axis=1),
            ARSENAL_FAILURE_UNSUPPORTED_WORLD,
        )
        provider_valid = (bits == jnp.uint32(0))[:, None, None]
        explosion_distance = admitted_distance
        explosion_target_mask = admitted & provider_valid
    separate_direct = projectile.direct_damage > 0.0
    direct_hit = (
        hit & ~projectile.terminal_deployable_area & (~has_explosion | separate_direct)
    )
    direct_target_mask = (
        jax.nn.one_hot(
            contact_victim,
            entity_count,
            dtype=jnp.bool_,
        )
        & direct_hit[..., None]
    )
    crossbow_direct = direct_hit & projectile.crossbow_program
    crossbow_kind = jnp.where(
        projectile.kind == PROJECTILE_BIG_ARROW,
        jnp.int32(CROSSBOW_IMPACT_BIG_ARROW),
        jnp.int32(CROSSBOW_IMPACT_STANDARD),
    )
    crossbow_commands = CrossbowProjectileImpactCommands(
        requested=crossbow_direct,
        kind=crossbow_kind,
        source_slot=projectile.owner_entity_id,
        target_slot=contact_victim,
        knockback_yaw_degrees=projectile.launch_yaw_degrees,
        standard_damage=jnp.where(
            projectile.kind == PROJECTILE_ARROW,
            projectile.damage,
            jnp.float32(0.0),
        ),
        combo_damage=projectile.crossbow_combo_damage,
        big_arrow_damage=jnp.where(
            projectile.kind == PROJECTILE_BIG_ARROW,
            projectile.damage,
            jnp.float32(0.0),
        ),
    )
    handled_crossbow = (
        crossbow_direct if include_crossbow_program else jnp.zeros_like(crossbow_direct)
    )
    impact_target_mask = direct_target_mask | explosion_target_mask
    primary_target_mask = (
        jnp.where(
            separate_direct[..., None],
            direct_target_mask,
            impact_target_mask,
        )
        & ~handled_crossbow[..., None]
    )
    point_direction = target_offset / jnp.maximum(
        jnp.linalg.norm(target_offset, axis=3, keepdims=True),
        jnp.finfo(jnp.float32).tiny,
    )
    point_velocity = point_direction * projectile.force_magnitude[..., None, None]
    point_velocity = point_velocity.at[..., 1].set(
        jnp.broadcast_to(
            projectile.force_velocity_y[..., None],
            point_velocity.shape[:-1],
        )
    )
    knockback_velocity = jnp.where(
        (projectile.force_direction_mode == FORCE_DIRECTION_POINT)[..., None, None],
        point_velocity,
        (
            projectile.force_direction[:, :, None, :]
            * projectile.force_magnitude[..., None, None]
        ),
    )
    # DamageEntityInteraction hands authored knockback to the victim's motion
    # controller, which applies the role-authored KnockbackScale. Both roles in
    # the certified 0.5.7 matchup use 0.5; heterogeneous-role admission must
    # widen this scalar to a per-entity input rather than adding weapon cases.
    knockback_velocity = knockback_velocity * params.agent_knockback_scale
    attenuation = jnp.where(
        has_explosion[..., None],
        jnp.power(
            jnp.clip(
                1.0
                - explosion_distance
                / jnp.maximum(
                    projectile.explosion_radius[..., None],
                    jnp.finfo(jnp.float32).tiny,
                ),
                0.0,
                1.0,
            ),
            projectile.explosion_falloff[..., None],
        ),
        1.0,
    )
    query_count = projectile.active.shape[1]
    dense_damage = empty_dense_damage_events(
        batch,
        query_count,
        entity_count,
    )
    target_entity_id = jnp.broadcast_to(
        jnp.arange(entity_count, dtype=jnp.int32)[None, None, :],
        impact_target_mask.shape,
    )

    def query_scalar(value):
        return jnp.broadcast_to(
            value[..., None],
            impact_target_mask.shape,
        )

    explosion_amount = projectile.damage[..., None] * attenuation
    primary_amount = jnp.where(
        separate_direct[..., None],
        projectile.direct_damage[..., None],
        explosion_amount,
    )
    primary_cause = jnp.broadcast_to(
        jnp.where(
            separate_direct,
            projectile.direct_damage_cause,
            projectile.damage_cause,
        )[..., None],
        impact_target_mask.shape,
    )
    primary_knockback = jnp.where(
        separate_direct[..., None, None],
        jnp.float32(0.0),
        knockback_velocity,
    )
    dense_damage = dense_damage._replace(
        requested=primary_target_mask,
        source_entity_id=query_scalar(projectile.owner_entity_id),
        target_entity_id=target_entity_id,
        amount=jnp.where(
            primary_target_mask,
            primary_amount,
            jnp.float32(0.0),
        ),
        random_percentage=query_scalar(projectile.random_percentage),
        damage_class=query_scalar(projectile.damage_class),
        cause=primary_cause,
        knockback_velocity=jnp.where(
            primary_target_mask[..., None],
            primary_knockback,
            jnp.float32(0.0),
        ),
        force_mode=query_scalar(projectile.force_mode),
        air_resistance=query_scalar(projectile.air_resistance),
        air_resistance_max=query_scalar(projectile.air_resistance_max),
        ground_resistance=query_scalar(projectile.ground_resistance),
        ground_resistance_max=query_scalar(projectile.ground_resistance_max),
        resistance_threshold=query_scalar(projectile.resistance_threshold),
        resistance_style=query_scalar(projectile.resistance_style),
        on_hit_resource_id=query_scalar(projectile.on_hit_resource_id),
        on_hit_resource_delta=query_scalar(projectile.on_hit_resource_delta),
        on_hit_healing=query_scalar(projectile.on_hit_healing),
    )
    pass_requested = jnp.stack(
        (primary_target_mask, explosion_target_mask),
        axis=2,
    )

    def repeat_pass(value):
        repeated = jnp.broadcast_to(
            value[:, :, None, ...],
            value.shape[:2] + (2,) + value.shape[2:],
        )
        return repeated.reshape((batch, query_count * 2) + value.shape[2:])

    damage_with_passes = jax.tree_util.tree_map(
        repeat_pass,
        dense_damage,
    )

    def flatten_passes(primary, secondary):
        return jnp.stack((primary, secondary), axis=2).reshape(
            (batch, query_count * 2) + primary.shape[2:]
        )

    secondary_resource_id = jnp.where(
        separate_direct[..., None],
        jnp.int32(-1),
        query_scalar(projectile.on_hit_resource_id),
    )
    secondary_resource_delta = jnp.where(
        separate_direct[..., None],
        jnp.float32(0.0),
        query_scalar(projectile.on_hit_resource_delta),
    )
    secondary_healing = jnp.where(
        separate_direct[..., None],
        jnp.float32(0.0),
        query_scalar(projectile.on_hit_healing),
    )
    damage_with_passes = damage_with_passes._replace(
        requested=pass_requested.reshape((batch, query_count * 2, entity_count)),
        amount=flatten_passes(
            dense_damage.amount,
            jnp.where(
                explosion_target_mask,
                explosion_amount,
                jnp.float32(0.0),
            ),
        ),
        cause=flatten_passes(
            dense_damage.cause,
            query_scalar(projectile.damage_cause),
        ),
        knockback_velocity=flatten_passes(
            dense_damage.knockback_velocity,
            jnp.where(
                explosion_target_mask[..., None],
                knockback_velocity,
                jnp.float32(0.0),
            ),
        ),
        on_hit_resource_id=flatten_passes(
            dense_damage.on_hit_resource_id,
            secondary_resource_id,
        ),
        on_hit_resource_delta=flatten_passes(
            dense_damage.on_hit_resource_delta,
            secondary_resource_delta,
        ),
        on_hit_healing=flatten_passes(
            dense_damage.on_hit_healing,
            secondary_healing,
        ),
    )
    damage, damage_overflow = pack_dense_damage_events(damage_with_passes)
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
        jnp.any(impact_target_mask, axis=2) & (projectile.status_id > 0),
        projectile.owner_entity_id,
        impact_target_mask,
        projectile.status_id,
        projectile.status_duration_seconds,
        projectile.status_cooldown_seconds,
        projectile.status_damage,
        projectile.status_damage_cause,
        jnp.zeros_like(projectile.status_damage),
        projectile.status_resource_id,
        projectile.status_resource_delta,
        projectile.status_speed_multiplier,
        projectile.status_flags,
        projectile.status_overlap_mode,
        max_health,
        rules,
    )
    bits = _set_failure(
        bits,
        damage_overflow | overflow,
        ARSENAL_FAILURE_EVENT_OVERFLOW,
    )
    final_valid = bits == jnp.uint32(0)
    expired = active_before & jnp.where(
        projectile.terminal_deployable_area,
        after_age > projectile.lifetime_seconds,
        after_age >= projectile.lifetime_seconds,
    )
    retained_impact = triggered & (projectile.dead_time_seconds >= jnp.float32(0.0))
    active = active_before & ~expired & (~triggered | retained_impact)
    impacted = (projectile.impacted & active) | retained_impact
    dead_remaining = jnp.where(
        retained_impact,
        projectile.dead_time_seconds,
        jnp.where(removed_dead, jnp.float32(0.0), dead_remaining),
    )
    candidate = projectile._replace(
        position=jnp.where(
            free[..., None],
            jnp.where(
                terminal_entity_contact[..., None],
                projectile.position + displacement * entity_hit_fraction[..., None],
                jnp.where(
                    continued_contact[..., None],
                    contact_response.position,
                    jnp.where(triggered[..., None], impact, end),
                ),
            ),
            projectile.position,
        ),
        velocity=jnp.where(
            projectile.terminal_entity_contact_inactive[..., None],
            jnp.float32(0.0),
            jnp.where(
                (free & (continued_contact | terminal_entity_contact))[..., None],
                jnp.where(
                    terminal_entity_contact[..., None],
                    velocity,
                    contact_response.velocity,
                ),
                jnp.where(
                    triggered[..., None],
                    jnp.float32(0.0),
                    jnp.where(
                        free[..., None],
                        velocity,
                        projectile.velocity,
                    ),
                ),
            ),
        ),
        age_seconds=jnp.where(
            active_before & row_valid[:, None],
            after_age,
            projectile.age_seconds,
        ),
        dead_time_remaining=dead_remaining,
        active=active,
        impacted=impacted,
        terminal_entity_contact_inactive=(
            projectile.terminal_entity_contact_inactive | terminal_entity_contact
        )
        & active,
        physics_initialized=projectile.physics_initialized | free,
        bounce_count=jnp.where(
            free,
            contact_response.bounce_count,
            projectile.bounce_count,
        ),
        on_ground=jnp.where(
            free,
            contact_response.on_ground,
            projectile.on_ground,
        ),
        world_hit=projectile.world_hit | world_before_fuse,
        world_hit_fraction=jnp.where(
            world_before_fuse,
            world_hit_fraction,
            projectile.world_hit_fraction,
        ),
        world_contact_point=jnp.where(
            world_before_fuse[..., None],
            world_contact_point,
            projectile.world_contact_point,
        ),
        world_contact_normal=jnp.where(
            world_before_fuse[..., None],
            world_contact_normal,
            projectile.world_contact_normal,
        ),
        world_geometry_exhausted=jnp.where(
            queried_world,
            world_geometry_exhausted,
            projectile.world_geometry_exhausted,
        ),
        world_capacity_exceeded=jnp.where(
            queried_world,
            world_capacity_exceeded,
            projectile.world_capacity_exceeded,
        ),
        world_contact_invalid=jnp.where(
            queried_world,
            world_contact_invalid,
            projectile.world_contact_invalid,
        ),
        world_segment_count=jnp.where(
            queried_world,
            world_segment_count,
            projectile.world_segment_count,
        ),
    )
    result = _select_tree(final_valid, candidate, projectile)
    # Contact diagnostics are evidence, not mechanics. Preserve a failed
    # world's reason while the row's gameplay state remains frozen.
    result = result._replace(
        world_geometry_exhausted=jnp.where(
            queried_world,
            candidate.world_geometry_exhausted,
            result.world_geometry_exhausted,
        ),
        world_capacity_exceeded=jnp.where(
            queried_world,
            candidate.world_capacity_exceeded,
            result.world_capacity_exceeded,
        ),
        world_contact_invalid=jnp.where(
            queried_world,
            candidate.world_contact_invalid,
            result.world_contact_invalid,
        ),
        world_segment_count=jnp.where(
            queried_world,
            candidate.world_segment_count,
            result.world_segment_count,
        ),
    )
    damage = _mask_tree(
        final_valid,
        damage,
        empty_damage_events(
            combat.position.shape[0],
            damage.requested.shape[1],
        ),
    )
    applications = _mask_tree(
        final_valid,
        applications,
        empty_status_applications(
            combat.position.shape[0],
            entity_count=entity_count,
        ),
    )
    output = (
        state._replace(
            projectiles=result,
            areas=_select_tree(final_valid, terminal_areas, state.areas),
            failure_bits=bits,
        ),
        damage,
        applications,
        jnp.where(
            final_valid,
            jnp.sum(
                impact_target_mask.astype(jnp.int32),
                axis=(1, 2),
            ),
            jnp.int32(0),
        ),
    )
    if include_crossbow_program:
        return (*output, crossbow_commands)
    return output


def _standard_velocity(projectile: ProjectileState, dt: jax.Array):
    """Match StandardPhysicsTickSystem's first force-free tick and drag."""

    velocity = projectile.velocity
    size = projectile.half_extent * jnp.float32(2.0)
    width, height, depth = size[..., 0], size[..., 1], size[..., 2]
    weighted_area_speed = (
        jnp.abs(velocity[..., 0]) * depth * height
        + jnp.abs(velocity[..., 1]) * depth * width
        + jnp.abs(velocity[..., 2]) * width * height
    )
    horizontal_area = jnp.maximum(width * depth, jnp.float32(1.0e-12))
    terminal_squared = jnp.maximum(
        projectile.terminal_velocity**2,
        jnp.float32(1.0e-12),
    )
    drag_rate = (
        projectile.gravity * weighted_area_speed / (horizontal_area * terminal_squared)
    )
    after_drag = velocity - velocity * drag_rate[..., None] * dt[..., None]
    reversed_component = ((velocity > 0.0) & (after_drag < 0.0)) | (
        (velocity < 0.0) & (after_drag > 0.0)
    )
    after_drag = jnp.where(reversed_component, 0.0, after_drag)
    vertical_force = jnp.where(
        projectile.on_ground,
        jnp.float32(0.0),
        -projectile.gravity * dt,
    )
    forced = after_drag.at[..., 1].add(vertical_force)
    return jnp.where(
        projectile.physics_initialized[..., None],
        forced,
        velocity,
    )


def _segment_aabb(start, end, box_min, box_max):
    delta = end - start
    parallel = jnp.abs(delta) <= jnp.float32(1.0e-8)
    safe = jnp.where(parallel, 1.0, delta)
    t0, t1 = (box_min - start) / safe, (box_max - start) / safe
    entry_axis = jnp.where(parallel, -jnp.inf, jnp.minimum(t0, t1))
    exit_axis = jnp.where(parallel, jnp.inf, jnp.maximum(t0, t1))
    entry, exit_ = jnp.max(entry_axis, axis=-1), jnp.min(exit_axis, axis=-1)
    inside = jnp.all(
        ~parallel | ((start >= box_min) & (start <= box_max)),
        axis=-1,
    )
    hit = inside & (entry <= exit_) & (exit_ >= 0.0) & (entry <= 1.0)
    return hit, jnp.clip(entry, 0.0, 1.0)


def _validate_explosion_candidate_shapes(
    evidence: ArsenalExplosionCandidates,
    batch: int,
    projectiles: int,
    entities: int,
) -> None:
    if not isinstance(evidence, ArsenalExplosionCandidates):
        raise TypeError(
            "explosion candidate provider must return ArsenalExplosionCandidates"
        )
    query_shape = (batch, projectiles)
    candidate_shape = query_shape + (entities,)
    for field in (
        "available",
        "geometry_exhausted",
        "capacity_exceeded",
        "damage_blocks_unsupported",
        "invalid",
    ):
        value = getattr(evidence, field)
        if value.shape != query_shape or value.dtype != jnp.dtype(jnp.bool_):
            raise ValueError(f"explosion candidate {field} must be bool[B, projectile]")
    if (
        evidence.candidate_mask.shape != candidate_shape
        or evidence.candidate_mask.dtype != jnp.dtype(jnp.bool_)
    ):
        raise ValueError("explosion candidate_mask must be bool[B, projectile, entity]")
    if (
        evidence.distance.shape != candidate_shape
        or evidence.distance.dtype != jnp.dtype(jnp.float32)
    ):
        raise ValueError("explosion distance must be float32[B, projectile, entity]")


def _invalid_projectile_state(
    state: ProjectileState,
    entity_count: int,
):
    terminal_valid = jnp.where(
        state.terminal_deployable_area,
        (state.kind == PROJECTILE_DEPLOYABLE)
        & state.standard_physics
        & state.terminal_intended_graph_available
        & ~state.entity_collision_only
        & ~state.damage_blocks
        & (state.direct_damage == 0.0)
        & (state.damage == 0.0)
        & (state.explosion_falloff == 0.0)
        & (state.block_damage_radius == 0)
        & (state.force_magnitude == 0.0)
        & jnp.all(state.force_direction == 0.0, axis=2)
        & (state.force_velocity_y == 0.0)
        & (state.fuse_seconds == 0.0)
        & (state.dead_time_seconds == -1.0)
        & (state.on_hit_resource_id == -1)
        & (state.on_hit_resource_delta == 0.0)
        & (state.status_id > 0)
        & jnp.all(
            jnp.isfinite(state.terminal_deployable_collision_half_extent)
            & (state.terminal_deployable_collision_half_extent > 0.0),
            axis=2,
        )
        & jnp.all(
            jnp.isfinite(state.terminal_deployable_collision_center_offset),
            axis=2,
        )
        & (state.terminal_deployable_id > 0)
        & state.terminal_deployable_count_towards_global_limit
        & (state.terminal_deployable_max_live_count == 2_147_483_647)
        & (state.lifetime_seconds == 300.0)
        & jnp.isfinite(state.status_duration_seconds)
        & (state.status_duration_seconds > 0.0)
        & jnp.isfinite(state.status_cooldown_seconds)
        & (state.status_cooldown_seconds >= 0.0)
        & jnp.isfinite(state.status_damage)
        & (state.status_damage == 0.0)
        & jnp.isfinite(state.on_hit_healing)
        & (state.on_hit_healing >= 0.0)
        & (state.status_damage_cause >= 0)
        & (state.status_damage_cause < DAMAGE_COUNT)
        & (state.status_resource_id == -1)
        & jnp.isfinite(state.status_resource_delta)
        & (state.status_resource_delta == 0.0)
        & jnp.isfinite(state.status_speed_multiplier)
        & (state.status_speed_multiplier > 0.0)
        & ((state.status_flags & jnp.uint32(TERMINAL_UNKNOWN_STATUS_FLAG_MASK)) == 0)
        & (state.status_overlap_mode >= STATUS_OVERLAP_IGNORE)
        & (state.status_overlap_mode <= STATUS_OVERLAP_OVERWRITE)
        & jnp.isfinite(state.terminal_area_duration_seconds)
        & (state.terminal_area_duration_seconds > 0.0)
        & jnp.isfinite(state.terminal_area_interval_seconds)
        & (state.terminal_area_interval_seconds > 0.0)
        & jnp.isfinite(state.terminal_area_end_radius)
        & (state.terminal_area_end_radius >= 0.0)
        & jnp.isfinite(state.terminal_area_height)
        & (state.terminal_area_height > 0.0)
        & jnp.isfinite(state.terminal_area_radius_change_seconds)
        & (state.terminal_area_radius_change_seconds > 0.0)
        & (
            (state.terminal_area_shape == AREA_SHAPE_SPHERE)
            | (state.terminal_area_shape == AREA_SHAPE_CYLINDER)
        )
        & (
            (state.terminal_area_attack_flags & jnp.int32(~DEPLOYABLE_ATTACK_FLAG_MASK))
            == 0
        ),
        (state.kind != PROJECTILE_DEPLOYABLE)
        & ~state.terminal_intended_graph_available
        & jnp.all(
            state.terminal_deployable_collision_half_extent == 0.0,
            axis=2,
        )
        & jnp.all(
            state.terminal_deployable_collision_center_offset == 0.0,
            axis=2,
        )
        & (state.terminal_deployable_id == 0)
        & ~state.terminal_deployable_count_towards_global_limit
        & (state.terminal_deployable_max_live_count == 0)
        & jnp.all(state.collision_center_offset == 0.0, axis=2)
        & (state.terminal_area_shape == 0)
        & (state.terminal_area_duration_seconds == 0.0)
        & (state.terminal_area_interval_seconds == 0.0)
        & (state.terminal_area_end_radius == 0.0)
        & (state.terminal_area_height == 0.0)
        & (state.terminal_area_radius_change_seconds == 0.0)
        & (state.terminal_area_attack_flags == 0),
    )
    return state.active & ~(
        (state.owner_entity_id >= 0)
        & (state.owner_entity_id < entity_count)
        & (state.kind > 0)
        & jnp.isfinite(state.launch_yaw_degrees)
        & jnp.isfinite(state.crossbow_combo_damage)
        & (state.crossbow_combo_damage >= 0.0)
        & jnp.where(
            state.crossbow_program,
            (
                (
                    (state.kind == PROJECTILE_ARROW)
                    | (state.kind == PROJECTILE_BIG_ARROW)
                )
                & (state.crossbow_combo_damage > 0.0)
            ),
            state.crossbow_combo_damage == 0.0,
        )
        & jnp.all(jnp.isfinite(state.position), axis=2)
        & jnp.all(jnp.isfinite(state.velocity), axis=2)
        & jnp.all(jnp.isfinite(state.half_extent), axis=2)
        & jnp.all(state.half_extent > 0.0, axis=2)
        & jnp.all(jnp.isfinite(state.collision_center_offset), axis=2)
        & jnp.isfinite(state.age_seconds)
        & (state.age_seconds >= 0.0)
        & jnp.isfinite(state.lifetime_seconds)
        & (state.lifetime_seconds > 0.0)
        & jnp.isfinite(state.damage)
        & (state.damage >= 0.0)
        & jnp.isfinite(state.direct_damage)
        & (state.direct_damage >= 0.0)
        & (state.damage_cause >= 0)
        & (state.damage_cause < DAMAGE_COUNT)
        & (
            (state.direct_damage == 0.0)
            | (
                (state.direct_damage_cause >= 0)
                & (state.direct_damage_cause < DAMAGE_COUNT)
            )
        )
        & (state.damage_class >= 0)
        & (state.damage_class < DAMAGE_CLASS_COUNT)
        & jnp.isfinite(state.gravity)
        & (state.gravity >= 0.0)
        & jnp.isfinite(state.terminal_velocity)
        & (state.terminal_velocity > 0.0)
        & jnp.isfinite(state.bounciness)
        & (state.bounciness >= 0.0)
        & (state.bounciness <= 1.0)
        & jnp.isfinite(state.bounce_limit)
        & (state.bounce_limit >= 0.0)
        & (state.bounce_count_limit >= -1)
        & (state.bounce_count >= 0)
        & jnp.isfinite(state.rolling_friction_factor)
        & (state.rolling_friction_factor >= 0.0)
        & (~state.allow_rolling | state.standard_physics)
        & (~state.on_ground | (state.standard_physics & state.allow_rolling))
        & jnp.isfinite(state.dead_time_seconds)
        & (state.dead_time_seconds >= -1.0)
        & jnp.isfinite(state.dead_time_remaining)
        & (state.dead_time_remaining >= 0.0)
        & (~state.impacted | (state.dead_time_seconds >= 0.0))
        & jnp.isfinite(state.world_hit_fraction)
        & (state.world_hit_fraction >= 0.0)
        & (state.world_hit_fraction <= 1.0)
        & jnp.all(jnp.isfinite(state.world_contact_point), axis=2)
        & jnp.all(jnp.isfinite(state.world_contact_normal), axis=2)
        & (state.world_segment_count >= 0)
        & jnp.isfinite(state.explosion_radius)
        & (state.explosion_radius >= 0.0)
        & jnp.isfinite(state.explosion_falloff)
        & (state.explosion_falloff >= 0.0)
        & jnp.where(
            state.terminal_deployable_area,
            state.block_damage_radius == 0,
            jnp.where(
                state.explosion_radius > 0.0,
                state.block_damage_radius > 0,
                state.block_damage_radius == 0,
            ),
        )
        & jnp.all(jnp.isfinite(state.force_direction), axis=2)
        & jnp.isfinite(state.force_velocity_y)
        & jnp.isfinite(state.force_magnitude)
        & (state.force_magnitude >= 0.0)
        & (state.force_direction_mode >= FORCE_DIRECTION_LOCAL)
        & (state.force_direction_mode <= FORCE_DIRECTION_POINT)
        & (state.on_hit_resource_id >= -1)
        & (state.on_hit_resource_id < RESOURCE_COUNT)
        & jnp.isfinite(state.on_hit_resource_delta)
        & jnp.isfinite(state.on_hit_healing)
        & (state.on_hit_healing >= 0.0)
        & terminal_valid
    )


def _family_matches(family: jax.Array, candidates: tuple[int, ...]) -> jax.Array:
    result = jnp.zeros_like(family, dtype=jnp.bool_)
    for candidate in candidates:
        result |= family == jnp.int32(candidate)
    return result


def projectile_look_direction(yaw_degrees, pitch_degrees):
    """Return native yaw/pitch forward vectors for arbitrary leading axes."""

    yaw, pitch = jnp.deg2rad(yaw_degrees), jnp.deg2rad(pitch_degrees)
    horizontal = jnp.cos(pitch)
    return jnp.stack(
        (-jnp.sin(yaw) * horizontal, jnp.sin(pitch), -jnp.cos(yaw) * horizontal),
        axis=-1,
    )


def projectile_spawn_offset(
    local_offset,
    yaw_degrees,
    pitch_degrees,
    flags,
):
    """Apply native modern or legacy projectile launch-offset rotation."""

    yaw = jnp.deg2rad(yaw_degrees)
    pitch = jnp.deg2rad(pitch_degrees)
    cos_pitch, sin_pitch = jnp.cos(pitch), jnp.sin(pitch)
    pitch_x = local_offset[..., 0]
    pitch_y = local_offset[..., 1] * cos_pitch - local_offset[..., 2] * sin_pitch
    pitch_z = local_offset[..., 1] * sin_pitch + local_offset[..., 2] * cos_pitch
    cos_yaw, sin_yaw = jnp.cos(yaw), jnp.sin(yaw)
    modern = jnp.stack(
        (
            pitch_x * cos_yaw + pitch_z * sin_yaw,
            pitch_y,
            -pitch_x * sin_yaw + pitch_z * cos_yaw,
        ),
        axis=-1,
    )

    pitch_adjust = (flags & jnp.uint32(EVENT_FLAG_PROJECTILE_PITCH_ADJUST_OFFSET)) != 0
    depth_direction = projectile_look_direction(
        yaw_degrees,
        jnp.where(pitch_adjust, pitch_degrees, jnp.float32(0.0)),
    )
    horizontal = local_offset[..., 0]
    legacy = jnp.stack(
        (
            horizontal * cos_yaw,
            local_offset[..., 1],
            -horizontal * sin_yaw,
        ),
        axis=-1,
    )
    legacy += depth_direction * (-local_offset[..., 2, None])
    legacy_mode = (flags & jnp.uint32(EVENT_FLAG_PROJECTILE_LEGACY_OFFSET)) != 0
    return jnp.where(legacy_mode[..., None], legacy, modern)


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


_mask_tree = _select_tree
