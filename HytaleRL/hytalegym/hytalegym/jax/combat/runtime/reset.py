"""Leaf helpers extracted verbatim from env.py."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
from hytalegym.jax.world.geometry.atlas import select_geometry_tile
from hytalegym.jax.world.region.types import RegionGeometryState
from hytalegym.jax.world.types import GeometryAtlas, GeometryState
from hytalegym.jax.combat.motion.movement_state import empty_actor_walk_movement_state
from hytalegym.jax.combat.types import AGENT_ENTITY, ENTITY_COUNT, PLAYER_STAMINA_MAXIMUM, TARGET_ENTITY, CombatParams, CombatState
from .math import (  # noqa: F401  (re-exported: callers unchanged)
    _approach,
    _approach_angle,
    _approach_target_angle,
    _canonical_bearing,
    _motion_delta,
    _normalize_degrees,
    _randint_batch,
    _rotation_x,
    _rotation_y,
    _rotation_z,
    _round_to_scale,
    _select_state,
    _target_distance,
    _target_facing_error,
    _target_head_facing_error,
    _terminated,
    _unit_fraction,
    _uniform_batch,
)
from hytalegym.jax.combat.opponents.legacy.controller import (  # noqa: F401
    COOLDOWN,
    GeometryProvider,
    IDLE,
    RECOVERY,
    SWEEP,
    WINDUP,
    _combat_perception_line_of_sight,
    _combat_positions_perception_line_of_sight,
    _combat_window_exhausted,
    _movement_medium_result,
    _target_detects_actor,
    _target_phase,
)
from .agent import (  # noqa: F401  (re-exported: callers unchanged)
    _ACTOR_WALK_MOVEMENT_CONFIG,
    _aabb_grounded_result,
    _aabb_support_sweep_result,
    _damp_force_velocity,
    _force_pushed_agent_motion,
    _gather_loadout_profile,
    _loadout_attack_count,
    _loadout_profile_index,
    _loadout_row_valid,
    _refresh_agent_walk_movement_state,
    _tick_agent_attack,
    _tick_agent_geometry_motion,
    _tick_agent_motion,
    _tick_agent_vertical,
)


class TargetDetection(NamedTuple):
    """Named internal stages of the target-perception filter."""

    target_alive: jax.Array
    target_in_line_of_sight: jax.Array
    target_perceptible: jax.Array


class TargetEvidence(NamedTuple):
    """Fixed-shape legal target evidence available to the actor encoder."""

    perceptible: jax.Array
    offset: jax.Array
    planar_distance: jax.Array
    health_fraction: jax.Array
    attack_phase_one_hot: jax.Array
    attack_progress: jax.Array
    facing_error_degrees: jax.Array
    velocity: jax.Array
    normalized_attack_index: jax.Array
    head_facing_error_degrees: jax.Array
    head_pitch_degrees: jax.Array


class ActorEvidence(NamedTuple):
    """The complete legal input to the 24-value actor encoder."""

    velocity: jax.Array
    #: HEAD yaw. Perception is head-frame, so every relative bearing below is
    #: measured from here. The BODY yaw is NOT here: this row is the frozen
    #: 24-value v1 prefix (`contract.py` 280 fails closed on any other width),
    #: so the body heading is published as a v3-only feature instead.
    yaw_radians: jax.Array
    health_fraction: jax.Array
    attack_executing: jax.Array
    target: TargetEvidence


def reset_batch(
    keys: jax.Array,
    params: CombatParams,
    *,
    entity_count: int = ENTITY_COUNT,
) -> tuple[CombatState, jax.Array]:
    """Reset one fixed-shape combat environment per PRNG key.

    ``keys`` must contain one JAX key per batch element. Domain parameters are
    sampled at reset, while all gameplay state remains in array leaves.
    """

    if (
        isinstance(entity_count, bool)
        or not isinstance(entity_count, int)
        or entity_count < 2
    ):
        raise ValueError("entity_count must be an integer of at least 2")
    batch_size = keys.shape[0]
    split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(keys)
    timing_keys = split_keys[:, 0]
    activation_keys = split_keys[:, 1]
    timing_profile = _randint_batch(
        timing_keys,
        params.motion_timing_profile_min,
        params.motion_timing_profile_max + 1,
    )
    activation_tick = _randint_batch(
        activation_keys,
        params.target_activation_min_tick,
        params.target_activation_max_tick + 1,
    )

    f32 = jnp.float32
    i32 = jnp.int32
    position = jnp.zeros((batch_size, entity_count, 3), dtype=f32)
    position = position.at[:, AGENT_ENTITY].set(params.agent_spawn)
    position = position.at[:, TARGET_ENTITY].set(
        params.agent_spawn + params.target_offset
    )

    # Slots beyond the default pair are inactive padding until an Arsenal
    # reset supplies explicit entity state.
    health = jnp.zeros((batch_size, entity_count), dtype=f32)
    health = health.at[:, AGENT_ENTITY].set(params.agent_max_health)
    health = health.at[:, TARGET_ENTITY].set(params.target_max_health)

    zeros_f = jnp.zeros((batch_size,), dtype=f32)
    zeros_i = jnp.zeros((batch_size,), dtype=i32)
    zeros_b = jnp.zeros((batch_size,), dtype=jnp.bool_)
    state = CombatState(
        position=position,
        velocity=jnp.zeros_like(position),
        health=health,
        yaw=jnp.zeros((batch_size, entity_count), dtype=f32),
        target_head_yaw=zeros_f,
        target_head_pitch=zeros_f,
        # `calculateYaw` starts an unsteered head on the body's own heading,
        # so a fresh actor is looking where it faces, not off to one side.
        agent_head_yaw=zeros_f,
        agent_head_pitch=zeros_f,
        desired_yaw=zeros_f,
        desired_body_yaw=zeros_f,
        pitch=zeros_f,
        desired_pitch=zeros_f,
        desired_velocity=jnp.zeros((batch_size, 2), dtype=f32),
        agent_move_speed=zeros_f,
        agent_fall_speed=zeros_f,
        agent_fall_start_y=jnp.full(
            (batch_size,),
            params.agent_spawn[1],
            dtype=f32,
        ),
        tick_count=zeros_i,
        motion_timing_profile=timing_profile,
        last_motion_delta_seconds=jnp.full(
            (batch_size,),
            params.nominal_dt,
            dtype=f32,
        ),
        attack_sequence_index=zeros_i,
        agent_attack_cooldown_seconds=zeros_f,
        agent_hit_delay=zeros_i,
        pending_agent_attack_index=jnp.full(
            (batch_size,),
            -1,
            dtype=i32,
        ),
        target_attack_sequence_index=zeros_i,
        target_ai_activation_tick=activation_tick,
        target_attack_index=jnp.full((batch_size,), -1, dtype=i32),
        last_target_attack_index=jnp.full((batch_size,), -1, dtype=i32),
        target_attack_elapsed_ticks=zeros_i,
        target_attack_cooldown_seconds=zeros_f,
        target_next_attack_queue_tick=activation_tick + 1,
        target_attack_queued=zeros_b,
        target_attack_hit_applied=zeros_b,
        target_damage_pending=zeros_b,
        pending_knockback_velocity=jnp.zeros(
            (batch_size, 3),
            dtype=f32,
        ),
        agent_force_velocity=jnp.zeros(
            (batch_size, 3),
            dtype=f32,
        ),
        target_damage_applied_this_tick=zeros_b,
        target_out_of_range_ticks=zeros_i,
        target_maintain_approaching=zeros_b,
        target_maintain_moving_away=zeros_b,
        target_strafe_delay_seconds=zeros_f,
        target_strafe_paused=zeros_b,
        target_strafe_direction=jnp.ones((batch_size,), dtype=i32),
        target_last_seen_position=jnp.zeros(
            (batch_size, 3),
            dtype=f32,
        ),
        target_last_seen_valid=zeros_b,
        engagement_target_id=jnp.full(
            (batch_size, entity_count),
            -1,
            dtype=i32,
        ),
        ticks_since_agent_damage=zeros_i,
        completion_awarded=zeros_b,
        death_penalty_awarded=zeros_b,
        vertical_impulse_applied=zeros_b,
        agent_applied_vertical_velocity=zeros_f,
        grounded_with_residual_velocity=zeros_b,
        knockback_control_lock=zeros_b,
        agent_grounded=jnp.ones((batch_size,), dtype=jnp.bool_),
        agent_walk_movement_state=empty_actor_walk_movement_state(batch_size),
        geometry_exhausted=zeros_b,
        target_navigation_unsupported=zeros_b,
        # A full bar and no regen block, matching how the entity bank
        # initialises every other actor.
        agent_locomotion_stamina=jnp.full(
            (batch_size,), PLAYER_STAMINA_MAXIMUM, dtype=f32
        ),
        agent_locomotion_stamina_regen_delay=zeros_f,
    )
    target_detected = (
        jnp.broadcast_to(params.target_active, (batch_size,))
        & (health[:, AGENT_ENTITY] > jnp.float32(0.0))
        & (health[:, TARGET_ENTITY] > jnp.float32(0.0))
        & _target_detects_actor(
            state,
            params,
            jnp.ones((batch_size,), dtype=jnp.bool_),
        )
    )
    state = state._replace(
        target_last_seen_position=jnp.where(
            target_detected[:, None],
            position[:, AGENT_ENTITY],
            jnp.float32(0.0),
        ),
        target_last_seen_valid=target_detected,
    )
    state = _refresh_agent_walk_movement_state(
        state,
        params,
        controller_in_fluid=zeros_b,
    )
    return state, observe_batch(state, params)


def reset_batch_geometry(
    keys: jax.Array,
    params: CombatParams,
    geometry: GeometryState,
) -> tuple[CombatState, jax.Array]:
    """Reset combat and derive initial grounding/visibility from geometry."""

    state, _ = reset_batch(keys, params)
    return _finish_geometry_reset(state, params, geometry)


def reset_batch_region(
    keys: jax.Array,
    params: CombatParams,
    geometry: RegionGeometryState,
) -> tuple[CombatState, jax.Array]:
    """Reset combat directly against an immutable Region v1 atlas."""

    state, _ = reset_batch(keys, params)
    return _finish_geometry_reset(state, params, geometry)


def reset_batch_geometry_at(
    keys: jax.Array,
    params: CombatParams,
    geometry: GeometryState,
    agent_position: jax.Array,
    target_position: jax.Array,
) -> tuple[CombatState, jax.Array]:
    """Reset exact-terrain combat at native-derived world coordinates.

    This is the generated-world entry point. It avoids reusing the flat combat
    fixture's spawn coordinates when replaying a native worldgen frame.
    """

    state, _ = reset_batch(keys, params)
    batch = state.position.shape[0]
    if agent_position.shape != (batch, 3):
        raise ValueError("agent_position must have shape (batch, 3)")
    if target_position.shape != (batch, 3):
        raise ValueError("target_position must have shape (batch, 3)")
    state = state._replace(
        position=state.position.at[:, AGENT_ENTITY]
        .set(jnp.asarray(agent_position, dtype=jnp.float32))
        .at[:, TARGET_ENTITY]
        .set(jnp.asarray(target_position, dtype=jnp.float32)),
    )
    return _finish_geometry_reset(state, params, geometry)


def reset_batch_region_at(
    keys: jax.Array,
    params: CombatParams,
    geometry: RegionGeometryState,
    agent_position: jax.Array,
    target_position: jax.Array,
) -> tuple[CombatState, jax.Array]:
    """Reset Region-backed combat at native-derived world coordinates."""

    state, _ = reset_batch(keys, params)
    batch = state.position.shape[0]
    if agent_position.shape != (batch, 3):
        raise ValueError("agent_position must have shape (batch, 3)")
    if target_position.shape != (batch, 3):
        raise ValueError("target_position must have shape (batch, 3)")
    state = state._replace(
        position=state.position.at[:, AGENT_ENTITY]
        .set(jnp.asarray(agent_position, dtype=jnp.float32))
        .at[:, TARGET_ENTITY]
        .set(jnp.asarray(target_position, dtype=jnp.float32)),
    )
    return _finish_geometry_reset(state, params, geometry)


def _finish_geometry_reset(
    state: CombatState,
    params: CombatParams,
    geometry: GeometryProvider,
    *,
    provider_available: jax.Array | None = None,
) -> tuple[CombatState, jax.Array]:
    ground = _aabb_grounded_result(
        geometry,
        state.position[:, AGENT_ENTITY],
    )
    medium = _movement_medium_result(
        geometry,
        state.position[:, AGENT_ENTITY],
    )
    provider_exhausted = (
        jnp.zeros_like(state.geometry_exhausted)
        if provider_available is None
        else ~jnp.asarray(provider_available, dtype=jnp.bool_)
    )
    state = state._replace(
        agent_grounded=ground.grounded,
        agent_fall_speed=jnp.zeros_like(state.agent_fall_speed),
        agent_fall_start_y=state.position[:, AGENT_ENTITY, 1],
        geometry_exhausted=(
            ground.geometry_exhausted
            | medium.geometry_exhausted
            | provider_exhausted
            | _combat_window_exhausted(state, geometry)
        ),
    )
    target_detected = (
        jnp.broadcast_to(params.target_active, state.tick_count.shape)
        & (state.health[:, AGENT_ENTITY] > jnp.float32(0.0))
        & (state.health[:, TARGET_ENTITY] > jnp.float32(0.0))
        & ~state.geometry_exhausted
        & _target_detects_actor(
            state,
            params,
            _combat_perception_line_of_sight(state, geometry),
        )
    )
    state = state._replace(
        target_last_seen_position=jnp.where(
            target_detected[:, None],
            state.position[:, AGENT_ENTITY],
            jnp.float32(0.0),
        ),
        target_last_seen_valid=target_detected,
    )
    state = _refresh_agent_walk_movement_state(
        state,
        params,
        controller_in_fluid=medium.in_fluid,
    )
    return state, observe_batch(state, params, geometry)


def reset_batch_geometry_atlas(
    keys: jax.Array,
    params: CombatParams,
    atlas: GeometryAtlas,
    environment_world_id: jax.Array,
) -> tuple[CombatState, jax.Array]:
    """Reset the flat scenario while selecting an exact compiled atlas tile."""

    state, _ = reset_batch(keys, params)
    return _finish_atlas_reset(
        state,
        params,
        atlas,
        environment_world_id,
    )


def reset_batch_geometry_atlas_at(
    keys: jax.Array,
    params: CombatParams,
    atlas: GeometryAtlas,
    environment_world_id: jax.Array,
    agent_position: jax.Array,
    target_position: jax.Array,
) -> tuple[CombatState, jax.Array]:
    """Reset native-derived coordinates inside an overlapping tile atlas."""

    state, _ = reset_batch(keys, params)
    batch = state.position.shape[0]
    if agent_position.shape != (batch, 3):
        raise ValueError("agent_position must have shape (batch, 3)")
    if target_position.shape != (batch, 3):
        raise ValueError("target_position must have shape (batch, 3)")
    state = state._replace(
        position=state.position.at[:, AGENT_ENTITY]
        .set(jnp.asarray(agent_position, dtype=jnp.float32))
        .at[:, TARGET_ENTITY]
        .set(jnp.asarray(target_position, dtype=jnp.float32)),
    )
    return _finish_atlas_reset(
        state,
        params,
        atlas,
        environment_world_id,
    )


def _finish_atlas_reset(
    state: CombatState,
    params: CombatParams,
    atlas: GeometryAtlas,
    environment_world_id: jax.Array,
) -> tuple[CombatState, jax.Array]:
    selected = select_geometry_tile(
        atlas,
        state.position,
        environment_world_id,
    )
    state, _ = _finish_geometry_reset(
        state,
        params,
        selected.geometry,
        provider_available=selected.available,
    )
    return state, observe_batch(state, params, selected.geometry)


def observe_batch(
    state: CombatState,
    params: CombatParams,
    geometry: GeometryProvider | None = None,
    *,
    perception_line_of_sight: jax.Array | None = None,
    target_entity_id: jax.Array | None = None,
) -> jax.Array:
    """Return the backend-compatible 24-value active combat observation."""

    evidence = ActorEvidence(
        velocity=state.velocity[:, AGENT_ENTITY],
        # Perception is head-frame: an actor sees from where it is looking, not
        # from where its chest points. The target has always been scored this
        # way -- `_target_head_facing_error` reads `target_head_yaw` while
        # `_target_facing_error` reads the body -- and the agent now has the
        # same two seams. Unslowed the head tracks the body exactly, so this
        # only diverges when a slow scales the head's rotation ceiling.
        yaw_radians=jnp.deg2rad(state.agent_head_yaw),
        health_fraction=_unit_fraction(
            state.health[:, AGENT_ENTITY],
            params.agent_max_health,
        ),
        attack_executing=(
            (state.agent_hit_delay > 0) | (state.agent_attack_cooldown_seconds > 0.0)
        ),
        target=target_evidence(
            state,
            params,
            geometry,
            perception_line_of_sight=perception_line_of_sight,
            target_entity_id=target_entity_id,
        ),
    )
    return _encode_actor_observation(evidence, params)


def _encode_actor_observation(
    evidence: ActorEvidence,
    params: CombatParams,
) -> jax.Array:
    """Encode actor evidence without privileged ``CombatState`` in scope."""

    agent_velocity = evidence.velocity
    agent_yaw_radians = evidence.yaw_radians
    target = evidence.target
    forward_x = -jnp.sin(agent_yaw_radians)
    forward_z = -jnp.cos(agent_yaw_radians)
    right_x = jnp.cos(agent_yaw_radians)
    right_z = -jnp.sin(agent_yaw_radians)

    forward_velocity = (
        agent_velocity[:, 0] * forward_x + agent_velocity[:, 2] * forward_z
    ) / params.agent_max_speed
    right_velocity = (
        agent_velocity[:, 0] * right_x + agent_velocity[:, 2] * right_z
    ) / params.agent_max_speed

    target_forward = (
        target.offset[:, 0] * forward_x + target.offset[:, 2] * forward_z
    ) / params.sensor_range
    target_right = (
        target.offset[:, 0] * right_x + target.offset[:, 2] * right_z
    ) / params.sensor_range

    base = jnp.stack(
        (
            forward_velocity,
            right_velocity,
            agent_velocity[:, 1] / params.vertical_speed_scale,
            evidence.health_fraction,
            jnp.sin(agent_yaw_radians),
            jnp.cos(agent_yaw_radians),
            target_forward,
            target_right,
            target.planar_distance / params.sensor_range,
            target.health_fraction,
            target.perceptible.astype(jnp.float32),
        ),
        axis=1,
    )

    target_forward_velocity = (
        target.velocity[:, 0] * forward_x + target.velocity[:, 2] * forward_z
    ) / params.target_chase_speed
    target_right_velocity = (
        target.velocity[:, 0] * right_x + target.velocity[:, 2] * right_z
    ) / params.target_chase_speed
    extra = jnp.concatenate(
        (
            evidence.attack_executing.astype(jnp.float32)[:, None],
            target.attack_phase_one_hot,
            target.attack_progress[:, None],
            target.facing_error_degrees[:, None] / params.facing_error_degrees_scale,
            target_forward_velocity[:, None],
            target_right_velocity[:, None],
            target.normalized_attack_index[:, None],
            target.head_facing_error_degrees[:, None]
            / params.facing_error_degrees_scale,
            target.head_pitch_degrees[:, None] / params.head_pitch_degrees_scale,
        ),
        axis=1,
    )
    return jnp.clip(
        jnp.concatenate((base, extra), axis=1),
        -1.0,
        1.0,
    ).astype(jnp.float32)


def target_evidence(
    state: CombatState,
    params: CombatParams,
    geometry: GeometryProvider | None = None,
    *,
    perception_line_of_sight: jax.Array | None = None,
    target_entity_id: jax.Array | None = None,
) -> TargetEvidence:
    """Filter privileged target state into legal fixed-shape actor evidence."""

    batch, entity_count = state.health.shape
    if target_entity_id is None:
        target_entity_id = jnp.full(
            (batch,),
            TARGET_ENTITY,
            dtype=jnp.int32,
        )
    if target_entity_id.shape != (batch,):
        raise ValueError("target_entity_id must have shape (batch,)")
    target_valid = (
        (target_entity_id >= 0)
        & (target_entity_id < entity_count)
    )
    agent_position = state.position[:, AGENT_ENTITY]
    target_position = _gather_entity_axis(
        state.position,
        target_entity_id,
    )
    target_health = _gather_entity_axis(
        state.health,
        target_entity_id,
    )
    target_yaw = _gather_entity_axis(
        state.yaw,
        target_entity_id,
    )
    target_velocity = _gather_entity_axis(
        state.velocity,
        target_entity_id,
    )
    privileged_offset = target_position - agent_position
    privileged_distance = jnp.linalg.norm(
        privileged_offset[:, (0, 2)],
        axis=1,
    )
    detection = _target_detection_from_positions(
        target_valid & (target_health > 0.0),
        agent_position,
        target_position,
        privileged_distance,
        params,
        geometry,
        perception_line_of_sight=perception_line_of_sight,
    )
    target_perceptible = detection.target_perceptible
    legal_scalar = target_perceptible.astype(jnp.float32)
    legal_vector = target_perceptible[:, None]

    # Match the Java wire format and Python decoder before exposing evidence:
    # values are rounded to integer tenths by Java Math.round, then decoded.
    wire_offset = _round_to_scale(
        privileged_offset,
        params.wire_fixed_point_scale,
    )
    offset = jnp.where(legal_vector, wire_offset, 0.0)
    planar_distance = jnp.where(
        target_perceptible,
        jnp.hypot(wire_offset[:, 0], wire_offset[:, 2]),
        0.0,
    )
    encoded_health = _round_to_scale(
        target_health,
        params.wire_fixed_point_scale,
    )
    health_fraction = jnp.where(
        target_perceptible,
        _unit_fraction(encoded_health, params.target_max_health),
        0.0,
    )

    phase, progress, reported_index, _ = _target_phase(state, params)
    primary_target = target_entity_id == TARGET_ENTITY
    phase = jnp.where(primary_target, phase, jnp.int32(IDLE))
    progress = jnp.where(primary_target, progress, jnp.float32(0.0))
    reported_index = jnp.where(
        primary_target,
        reported_index,
        jnp.int32(-1),
    )
    attack_phase_one_hot = (
        jax.nn.one_hot(
            jnp.clip(phase, IDLE, COOLDOWN),
            COOLDOWN + 1,
            dtype=jnp.float32,
        )
        * legal_scalar[:, None]
    )
    attack_progress = jnp.where(target_perceptible, progress, 0.0)
    normalized_attack_index = jnp.where(
        target_perceptible,
        jnp.where(
            reported_index < 0,
            -1.0,
            reported_index.astype(jnp.float32)
            / jnp.asarray(
                params.target_damage.shape[0] - 1,
                dtype=jnp.float32,
            ),
        ),
        0.0,
    )

    target_to_agent = -privileged_offset
    facing_distance = jnp.hypot(
        target_to_agent[:, 0],
        target_to_agent[:, 2],
    )
    target_bearing = _canonical_bearing(
        target_to_agent[:, 0],
        target_to_agent[:, 2],
    )
    facing_error = jnp.where(
        facing_distance <= 1.0e-9,
        0.0,
        _normalize_degrees(target_bearing - target_yaw),
    ).astype(jnp.float32)
    target_head_yaw = jnp.where(
        primary_target,
        state.target_head_yaw,
        target_yaw,
    )
    head_facing_error = jnp.where(
        facing_distance <= 1.0e-9,
        0.0,
        _normalize_degrees(target_bearing - target_head_yaw),
    ).astype(jnp.float32)

    return TargetEvidence(
        perceptible=target_perceptible,
        offset=offset.astype(jnp.float32),
        planar_distance=planar_distance.astype(jnp.float32),
        health_fraction=health_fraction.astype(jnp.float32),
        attack_phase_one_hot=attack_phase_one_hot,
        attack_progress=attack_progress.astype(jnp.float32),
        facing_error_degrees=jnp.where(
            target_perceptible,
            facing_error,
            0.0,
        ),
        velocity=jnp.where(
            legal_vector,
            target_velocity,
            0.0,
        ).astype(jnp.float32),
        normalized_attack_index=normalized_attack_index.astype(jnp.float32),
        head_facing_error_degrees=jnp.where(
            target_perceptible,
            head_facing_error,
            0.0,
        ),
        head_pitch_degrees=jnp.where(
            target_perceptible,
            jnp.where(
                primary_target,
                state.target_head_pitch,
                jnp.float32(0.0),
            ),
            0.0,
        ).astype(jnp.float32),
    )


def _gather_entity_axis(
    array: jax.Array,
    entity_id: jax.Array,
) -> jax.Array:
    return array[
        jnp.arange(array.shape[0]),
        jnp.clip(entity_id, 0, array.shape[1] - 1),
    ]


def _target_detection_from_positions(
    target_alive: jax.Array,
    agent_position: jax.Array,
    target_position: jax.Array,
    target_distance: jax.Array,
    params: CombatParams,
    geometry: GeometryProvider | None,
    *,
    perception_line_of_sight: jax.Array | None = None,
) -> TargetDetection:
    """Apply the named alive → LOS → range filter stages."""

    target_in_line_of_sight = target_alive
    if perception_line_of_sight is not None:
        supplied = jnp.asarray(perception_line_of_sight, dtype=jnp.bool_)
        if supplied.shape != target_alive.shape:
            raise ValueError("perception_line_of_sight must match the combat batch")
        target_in_line_of_sight &= supplied
    elif geometry is not None:
        target_in_line_of_sight = (
            target_in_line_of_sight
            & _combat_positions_perception_line_of_sight(
                agent_position,
                target_position,
                geometry,
            )
        )
    return TargetDetection(
        target_alive=target_alive,
        target_in_line_of_sight=target_in_line_of_sight,
        target_perceptible=(
            target_in_line_of_sight & (target_distance <= params.sensor_range)
        ),
    )
