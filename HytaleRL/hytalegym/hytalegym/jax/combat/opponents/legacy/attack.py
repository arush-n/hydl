"""Leaf helpers extracted verbatim from _target.py."""

import jax
import jax.numpy as jnp
import hytalegym.jax.world as world_api
from hytalegym.jax.combat.types import AGENT_ENTITY, TARGET_ENTITY, CombatParams, CombatState
from hytalegym.jax.combat.runtime.math import (  # noqa: F401
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
    _uniform_batch,
)
from .motion import (  # noqa: F401  (re-exported: callers unchanged)
    GeometryProvider,
    TargetNavigationProvider,
    _ACTOR_CROUCHING_INDEX,
    _combat_perception_line_of_sight,
    _combat_positions_perception_line_of_sight,
    _movement_medium_result,
    _pursue_speed_scale,
    _resolve_aabb_motion_result,
    _target_detects_actor,
    _target_navigation_rejected,
    _tick_target_motion,
)


NO_QUEUE_TICK = 2_147_483_647


DIRECTIONAL_KNOCKBACK_FALLBACK_SQUARED_DISTANCE = 1.0e-8


def _combat_hitbox_line_of_sight(
    state: CombatState,
    geometry: GeometryProvider,
) -> jax.Array:
    batch = state.position.shape[0]
    agent_offset = (
        geometry.agent_los_offset
        if geometry.agent_los_offset.shape[0] == batch
        else jnp.broadcast_to(geometry.agent_los_offset, (batch, 3))
    )
    target_offset = (
        geometry.target_los_offset
        if geometry.target_los_offset.shape[0] == batch
        else jnp.broadcast_to(geometry.target_los_offset, (batch, 3))
    )
    return world_api.geometry_hitbox_line_of_sight_result(
        geometry,
        state.position[:, AGENT_ENTITY] + agent_offset,
        state.position[:, TARGET_ENTITY] + target_offset,
    ).clear


def _pending_target_knockback(
    selector_state: CombatState,
    params: CombatParams,
) -> jax.Array:
    """Build the exact pending 0.5.7 directional external velocity."""

    attacker = selector_state.position[:, TARGET_ENTITY]
    victim = selector_state.position[:, AGENT_ENTITY]
    base_x = attacker[:, 0] - victim[:, 0]
    base_y = attacker[:, 1] - victim[:, 1]
    base_z = attacker[:, 2] - victim[:, 2]
    base_squared_length = base_x * base_x + base_y * base_y + base_z * base_z
    base_length = jnp.sqrt(base_squared_length)
    has_separation_direction = base_squared_length > jnp.float32(
        DIRECTIONAL_KNOCKBACK_FALLBACK_SQUARED_DISTANCE
    )
    yaw = jnp.deg2rad(selector_state.target_head_yaw)
    fallback_x = -jnp.sin(yaw)
    fallback_z = -jnp.cos(yaw)
    safe_length = jnp.maximum(
        base_length,
        jnp.finfo(jnp.float32).tiny,
    )
    base_x = jnp.where(
        has_separation_direction,
        base_x / safe_length,
        fallback_x,
    )
    base_z = jnp.where(
        has_separation_direction,
        base_z / safe_length,
        fallback_z,
    )

    # JOML Vector3d.rotateY.
    relative_x = params.target_knockback_relative_x * jnp.cos(
        yaw
    ) + params.target_knockback_relative_z * jnp.sin(yaw)
    relative_z = -params.target_knockback_relative_x * jnp.sin(
        yaw
    ) + params.target_knockback_relative_z * jnp.cos(yaw)
    authored_x = (base_x + relative_x) * params.target_knockback_force
    authored_z = (base_z + relative_z) * params.target_knockback_force
    horizontal_scale = (
        params.legacy_horizontal_knockback_scale
        * params.agent_knockback_scale
        * params.legacy_motion_controller_horizontal_factor
        * params.agent_movement_velocity_resistance
    )
    return jnp.stack(
        (
            authored_x * horizontal_scale,
            jnp.full_like(
                authored_x,
                params.target_knockback_velocity_y * params.agent_knockback_scale,
            ),
            authored_z * horizontal_scale,
        ),
        axis=1,
    ).astype(jnp.float32)


def _tick_target_attack(
    state: CombatState,
    pause_draw: jax.Array,
    motion_delta: jax.Array,
    queue_distance: jax.Array,
    queue_visible: jax.Array,
    params: CombatParams,
    geometry: GeometryProvider | None = None,
    selector_reference: CombatState | None = None,
) -> CombatState:
    abort = (state.health[:, TARGET_ENTITY] <= 0.0) | (
        (state.health[:, AGENT_ENTITY] <= 0.0) & ~state.target_damage_applied_this_tick
    )
    cooldown = jnp.maximum(
        jnp.float32(0.0),
        state.target_attack_cooldown_seconds - motion_delta,
    )
    attack_index = state.target_attack_index
    elapsed = state.target_attack_elapsed_ticks
    clipped_index = jnp.clip(
        attack_index,
        0,
        params.target_windup_ticks.shape[0] - 1,
    )
    chain_ticks = (
        params.target_windup_ticks[clipped_index]
        + params.target_sweep_ticks[clipped_index]
        + params.target_recovery_ticks[clipped_index]
    )
    completed = (attack_index >= 0) & (elapsed >= chain_ticks)
    attack_index = jnp.where(completed, -1, attack_index)
    elapsed = jnp.where(completed, 0, elapsed)

    was_queued = state.target_attack_queued
    started_index = state.target_attack_sequence_index
    attack_index = jnp.where(was_queued, started_index, attack_index)
    last_index = jnp.where(
        was_queued,
        started_index,
        state.last_target_attack_index,
    )
    sequence_index = jnp.where(
        was_queued,
        (state.target_attack_sequence_index + 1) % params.target_windup_ticks.shape[0],
        state.target_attack_sequence_index,
    )
    elapsed = jnp.where(was_queued, 0, elapsed)
    hit_applied = jnp.where(
        was_queued,
        False,
        state.target_attack_hit_applied,
    )

    ready = (
        ~was_queued
        & (attack_index < 0)
        & (cooldown <= 0.0)
        & (state.tick_count >= state.target_ai_activation_tick)
    )
    in_attack_range = (
        queue_distance <= jnp.max(params.target_end_distances)
    ) & queue_visible
    no_queue_tick = jnp.asarray(NO_QUEUE_TICK, dtype=jnp.int32)
    schedule_ready = ready & (state.target_next_attack_queue_tick == no_queue_tick)
    next_queue_tick = jnp.where(
        schedule_ready,
        state.tick_count + params.target_decision_delay_ticks,
        state.target_next_attack_queue_tick,
    )
    next_queue_tick = jnp.where(
        ready & ~in_attack_range,
        state.tick_count + params.target_decision_delay_ticks,
        next_queue_tick,
    )
    can_queue = (
        ready
        & ~schedule_ready
        & (state.tick_count >= next_queue_tick)
        & in_attack_range
    )
    queued = can_queue
    cooldown = jnp.where(can_queue, pause_draw, cooldown)
    next_queue_tick = jnp.where(
        can_queue,
        no_queue_tick,
        next_queue_tick,
    )

    has_attack = (attack_index >= 0) & ~can_queue
    elapsed = jnp.where(has_attack, elapsed + 1, elapsed)
    candidate_state = state._replace(
        target_attack_sequence_index=sequence_index,
        target_attack_index=attack_index,
        last_target_attack_index=last_index,
        target_attack_elapsed_ticks=elapsed,
        target_attack_cooldown_seconds=cooldown,
        target_next_attack_queue_tick=next_queue_tick,
        target_attack_queued=queued,
        target_attack_hit_applied=hit_applied,
    )
    selector_state = candidate_state
    if selector_reference is not None:
        # Native interaction selectors sample the transform snapshot that
        # precedes this tick's body-motion integration.
        selector_state = candidate_state._replace(
            position=selector_reference.position,
            yaw=selector_reference.yaw,
            target_head_yaw=selector_reference.target_head_yaw,
            target_head_pitch=selector_reference.target_head_pitch,
        )
    selector_hit = (
        has_attack
        & ~hit_applied
        & _target_selector_intersects(
            selector_state,
            params,
            geometry,
        )
    )
    if geometry is not None:
        selector_hit = selector_hit & _combat_hitbox_line_of_sight(
            selector_state,
            geometry,
        )
    pending_knockback = _pending_target_knockback(
        selector_state,
        params,
    )
    candidate_state = candidate_state._replace(
        target_attack_hit_applied=hit_applied | selector_hit,
        target_damage_pending=(candidate_state.target_damage_pending | selector_hit),
        pending_knockback_velocity=jnp.where(
            selector_hit[:, None],
            pending_knockback,
            candidate_state.pending_knockback_velocity,
        ),
    )

    return candidate_state._replace(
        target_attack_index=jnp.where(
            abort,
            -1,
            candidate_state.target_attack_index,
        ),
        target_attack_elapsed_ticks=jnp.where(
            abort,
            0,
            candidate_state.target_attack_elapsed_ticks,
        ),
        target_attack_queued=jnp.where(
            abort,
            False,
            candidate_state.target_attack_queued,
        ),
        target_damage_pending=jnp.where(
            abort,
            False,
            candidate_state.target_damage_pending,
        ),
        pending_knockback_velocity=jnp.where(
            abort[:, None],
            jnp.zeros_like(candidate_state.pending_knockback_velocity),
            candidate_state.pending_knockback_velocity,
        ),
        target_attack_cooldown_seconds=jnp.where(
            abort,
            state.target_attack_cooldown_seconds,
            candidate_state.target_attack_cooldown_seconds,
        ),
        target_next_attack_queue_tick=jnp.where(
            abort,
            state.target_next_attack_queue_tick,
            candidate_state.target_next_attack_queue_tick,
        ),
        target_attack_sequence_index=jnp.where(
            abort,
            state.target_attack_sequence_index,
            candidate_state.target_attack_sequence_index,
        ),
        last_target_attack_index=jnp.where(
            abort,
            state.last_target_attack_index,
            candidate_state.last_target_attack_index,
        ),
        target_attack_hit_applied=jnp.where(
            abort,
            state.target_attack_hit_applied,
            candidate_state.target_attack_hit_applied,
        ),
    )


def _selector_sweep_window(
    elapsed: jax.Array,
    windup: jax.Array,
    selector_runtime: jax.Array,
    nominal_dt: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    # SelectInteraction.tick0 returns before ticking a non-player selector on
    # its first run at time zero.  The first sweep-phase sample therefore has
    # no volume; HorizontalSelector starts advancing on the following tick.
    sample = jnp.maximum(elapsed - windup - 1, 0).astype(jnp.float32)
    current = jnp.minimum(sample * nominal_dt / selector_runtime, 1.0)
    previous = jnp.minimum(
        jnp.maximum(sample - 1.0, 0.0) * nominal_dt / selector_runtime,
        1.0,
    )
    return previous, current - previous


def _target_selector_intersects(
    state: CombatState,
    params: CombatParams,
    geometry: GeometryProvider | None = None,
) -> jax.Array:
    """Test Hytale 0.5.7's head-aimed horizontal selector frustum.

    The native selector starts at time zero, then constructs one truncated 3D
    frustum per logical 30 Hz sample and ends exactly at the authored runtime.
    Entity selection tests the exact world-space AABB against that frustum.
    The separating-axis test below is the fixed-shape equivalent of native
    ``HitDetectionExecutor`` cube-face clipping for this matchup.
    """

    attack_index = jnp.clip(
        state.target_attack_index,
        0,
        params.target_windup_ticks.shape[0] - 1,
    )
    elapsed = state.target_attack_elapsed_ticks
    windup = params.target_windup_ticks[attack_index]
    sweep_ticks = params.target_sweep_ticks[attack_index]
    is_sweep = (elapsed > windup) & (elapsed <= windup + sweep_ticks)

    selector_runtime = params.target_selector_runtime_seconds[attack_index]
    prior_percentage, delta_percentage = _selector_sweep_window(
        elapsed,
        windup,
        selector_runtime,
        params.nominal_dt,
    )
    arc_radians = jnp.deg2rad(params.target_arc_degrees[attack_index])
    yaw_delta = arc_radians * delta_percentage
    prior_yaw = arc_radians * prior_percentage
    direction_modifier = jnp.where(
        params.target_sweep_directions[attack_index] == 0,
        jnp.float32(1.0),
        jnp.float32(-1.0),
    )
    selector_yaw = (
        prior_yaw
        + yaw_delta
        + jnp.deg2rad(params.target_yaw_start_offsets[attack_index])
    ) * direction_modifier

    near = params.target_start_distances[attack_index]
    far = params.target_end_distances[attack_index]
    near_far_ratio = near / far
    horizontal_far = jnp.float32(2.0) * far * yaw_delta / params.horizontal_selector_pi
    horizontal_near = horizontal_far * near_far_ratio
    bottom_near = params.target_extend_bottom[attack_index] * near_far_ratio
    top_near = params.target_extend_top[attack_index] * near_far_ratio
    bottom_far = params.target_extend_bottom[attack_index]
    top_far = params.target_extend_top[attack_index]

    pitch_offset = jnp.deg2rad(params.target_pitch_offsets[attack_index])
    roll_offset = jnp.deg2rad(params.target_roll_offsets[attack_index])
    selector_rotation = jnp.matmul(
        _rotation_x(-pitch_offset),
        jnp.matmul(
            _rotation_y(-selector_yaw),
            _rotation_z(-roll_offset),
        ),
    )

    head_yaw = jnp.deg2rad(state.target_head_yaw)
    head_pitch = jnp.deg2rad(state.target_head_pitch)
    cos_pitch = jnp.cos(head_pitch)
    forward = jnp.stack(
        (
            -jnp.sin(head_yaw) * cos_pitch,
            jnp.sin(head_pitch),
            -jnp.cos(head_yaw) * cos_pitch,
        ),
        axis=1,
    )
    world_up = jnp.broadcast_to(
        jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32),
        forward.shape,
    )
    right = jnp.cross(forward, world_up)
    right = right / jnp.maximum(
        jnp.linalg.norm(right, axis=1, keepdims=True),
        jnp.float32(1.0e-9),
    )
    camera_up = jnp.cross(right, forward)
    camera_rotation = jnp.stack(
        (right, camera_up, -forward),
        axis=1,
    )
    world_to_selector = jnp.matmul(
        selector_rotation,
        camera_rotation,
    )

    batch = state.position.shape[0]
    if geometry is None:
        agent_bounds = jnp.broadcast_to(params.agent_bounds, (batch, 6))
        target_eye_offset = jnp.broadcast_to(
            params.target_eye_offset,
            (batch, 3),
        )
    else:
        agent_bounds = (
            geometry.agent_bounds
            if geometry.agent_bounds.shape[0] == batch
            else jnp.broadcast_to(geometry.agent_bounds, (batch, 6))
        )
        target_eye_offset = (
            geometry.target_los_offset
            if geometry.target_los_offset.shape[0] == batch
            else jnp.broadcast_to(
                geometry.target_los_offset,
                (batch, 3),
            )
        )
    selector_origin = state.position[:, TARGET_ENTITY] + target_eye_offset
    agent_min = state.position[:, AGENT_ENTITY] + agent_bounds[:, :3]
    agent_max = state.position[:, AGENT_ENTITY] + agent_bounds[:, 3:]
    cube_bits = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ),
        dtype=jnp.float32,
    )
    agent_corners_world = (
        agent_min[:, None, :]
        + (agent_max - agent_min)[:, None, :] * cube_bits[None, :, :]
    )
    agent_corners = jnp.einsum(
        "bij,bvj->bvi",
        world_to_selector,
        agent_corners_world - selector_origin[:, None, :],
    )

    frustum_corners = jnp.stack(
        (
            jnp.stack(
                (-horizontal_near, -bottom_near, -near),
                axis=1,
            ),
            jnp.stack(
                (horizontal_near, -bottom_near, -near),
                axis=1,
            ),
            jnp.stack(
                (-horizontal_near, top_near, -near),
                axis=1,
            ),
            jnp.stack(
                (horizontal_near, top_near, -near),
                axis=1,
            ),
            jnp.stack(
                (-horizontal_far, -bottom_far, -far),
                axis=1,
            ),
            jnp.stack(
                (horizontal_far, -bottom_far, -far),
                axis=1,
            ),
            jnp.stack(
                (-horizontal_far, top_far, -far),
                axis=1,
            ),
            jnp.stack(
                (horizontal_far, top_far, -far),
                axis=1,
            ),
        ),
        axis=1,
    )
    face_indices = jnp.asarray(
        (
            (0, 1, 3),
            (4, 6, 7),
            (0, 2, 6),
            (1, 5, 7),
            (0, 4, 5),
            (2, 3, 7),
        ),
        dtype=jnp.int32,
    )
    face_points = frustum_corners[:, face_indices, :]
    frustum_normals = jnp.cross(
        face_points[:, :, 1] - face_points[:, :, 0],
        face_points[:, :, 2] - face_points[:, :, 0],
    )
    box_axes = jnp.swapaxes(world_to_selector, 1, 2)
    frustum_edges = jnp.stack(
        (
            frustum_corners[:, 1] - frustum_corners[:, 0],
            frustum_corners[:, 2] - frustum_corners[:, 0],
            frustum_corners[:, 4] - frustum_corners[:, 0],
            frustum_corners[:, 5] - frustum_corners[:, 1],
            frustum_corners[:, 6] - frustum_corners[:, 2],
            frustum_corners[:, 7] - frustum_corners[:, 3],
        ),
        axis=1,
    )
    edge_cross_axes = jnp.cross(
        box_axes[:, :, None, :],
        frustum_edges[:, None, :, :],
    ).reshape((batch, 18, 3))
    axes = jnp.concatenate(
        (frustum_normals, box_axes, edge_cross_axes),
        axis=1,
    )
    axis_norm = jnp.linalg.norm(axes, axis=2)
    valid_axis = axis_norm > jnp.float32(1.0e-8)
    normalized_axes = axes / jnp.maximum(
        axis_norm[:, :, None],
        jnp.float32(1.0e-8),
    )
    agent_projection = jnp.einsum(
        "bvi,bai->bav",
        agent_corners,
        normalized_axes,
    )
    frustum_projection = jnp.einsum(
        "bvi,bai->bav",
        frustum_corners,
        normalized_axes,
    )
    separated = (
        (jnp.max(agent_projection, axis=2) < jnp.min(frustum_projection, axis=2))
        | (jnp.max(frustum_projection, axis=2) < jnp.min(agent_projection, axis=2))
    ) & valid_axis
    volumes_intersect = ~jnp.any(separated, axis=1)

    # Native HitDetectionExecutor clips the six cube surfaces. Match its sole
    # volume-intersection exception: a frustum wholly contained in the entity
    # box has no cube surface inside the selector and therefore is not a hit.
    frustum_world = selector_origin[:, None, :] + jnp.einsum(
        "bvi,bij->bvj",
        frustum_corners,
        world_to_selector,
    )
    frustum_fully_inside_agent = jnp.all(
        (frustum_world >= agent_min[:, None, :])
        & (frustum_world <= agent_max[:, None, :]),
        axis=(1, 2),
    )
    return (
        is_sweep
        & (delta_percentage > 0.0)
        & volumes_intersect
        & ~frustum_fully_inside_agent
    )
