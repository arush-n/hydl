"""Leaf helpers extracted verbatim from _target.py."""

from collections.abc import Callable
import jax
import jax.numpy as jnp
import hytalegym.jax.world as world_api
from hytalegym.jax.world.geometry import movement_medium_result as local_movement_medium_result, resolve_aabb_motion as local_resolve_aabb_motion
from hytalegym.jax.world.region.geometry import region_movement_medium_result, region_resolve_aabb_motion
from hytalegym.jax.world.region.types import RegionGeometryState
from hytalegym.jax.world.entities.movement_states import MOVEMENT_STATE_ORDER
from hytalegym.jax.world.types import GeometryState
from hytalegym.jax.combat.motion.navigation import (
    TargetNavigationStep,
    validate_target_navigation_step,
)
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


GeometryProvider = GeometryState | RegionGeometryState


TargetNavigationProvider = Callable[
    [CombatState, CombatParams, GeometryProvider, jax.Array, jax.Array],
    TargetNavigationStep,
]


_ACTOR_CROUCHING_INDEX = MOVEMENT_STATE_ORDER.index("crouching")


def _movement_medium_result(
    geometry: GeometryProvider,
    position: jax.Array,
):
    if isinstance(geometry, RegionGeometryState):
        return region_movement_medium_result(geometry, position)
    return local_movement_medium_result(geometry, position)


def _resolve_aabb_motion_result(
    geometry: GeometryProvider,
    position: jax.Array,
    displacement: jax.Array,
    bounds: jax.Array | None = None,
    stop_on_vertical_collision: jax.Array | None = None,
    skin_distance: float | None = None,
):
    """Dispatch local/Region motion while preserving caller-owned skin."""

    extra = () if skin_distance is None else (skin_distance,)
    if isinstance(geometry, RegionGeometryState):
        return region_resolve_aabb_motion(
            geometry,
            position,
            displacement,
            bounds,
            stop_on_vertical_collision,
            *extra,
        )
    return local_resolve_aabb_motion(
        geometry,
        position,
        displacement,
        bounds,
        stop_on_vertical_collision,
        *extra,
    )


def _combat_positions_perception_line_of_sight(
    agent_position: jax.Array,
    target_position: jax.Array,
    geometry: GeometryProvider,
) -> jax.Array:
    batch = agent_position.shape[0]
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
    return world_api.geometry_perception_line_of_sight_result(
        geometry,
        agent_position + agent_offset,
        target_position + target_offset,
        role_opaque_mask=None,
    ).visible


def _combat_perception_line_of_sight(
    state: CombatState,
    geometry: GeometryProvider,
) -> jax.Array:
    return _combat_positions_perception_line_of_sight(
        state.position[:, AGENT_ENTITY],
        state.position[:, TARGET_ENTITY],
        geometry,
    )


def _target_detects_actor(
    state: CombatState,
    params: CombatParams,
    target_in_line_of_sight: jax.Array,
) -> jax.Array:
    """Apply the Brawler's authored sight-or-hearing LastSeen gate."""

    offset = (
        state.position[:, AGENT_ENTITY]
        - state.position[:, TARGET_ENTITY]
    )
    squared_distance = jnp.sum(offset * offset, axis=1)
    crouching = state.agent_walk_movement_state.values[
        :,
        _ACTOR_CROUCHING_INDEX,
    ]
    seen = target_in_line_of_sight & (
        squared_distance <= params.target_chase_view_range**2
    )
    hearing_allowed = (
        ~params.target_chase_hearing_suppressed_by_crouching
        | ~crouching
    )
    heard = hearing_allowed & (
        squared_distance <= params.target_chase_hearing_range**2
    )
    return seen | heard


def _tick_target_motion(
    state: CombatState,
    motion_delta: jax.Array,
    strafe_frequency_draw: jax.Array,
    strafe_duration_draw: jax.Array,
    strafe_direction_draw: jax.Array,
    params: CombatParams,
    geometry: GeometryProvider | None = None,
    target_navigation_provider: TargetNavigationProvider | None = None,
) -> CombatState:
    target_dead = (state.health[:, TARGET_ENTITY] <= 0.0) | (
        state.health[:, AGENT_ENTITY] <= 0.0
    )
    activated = state.tick_count >= state.target_ai_activation_tick
    can_turn = ~target_dead & activated

    if geometry is None:
        visible = jnp.ones_like(can_turn)
        target_speed_multiplier = jnp.ones_like(motion_delta)
        medium_exhausted = jnp.zeros_like(can_turn)
    else:
        visible = _combat_perception_line_of_sight(state, geometry)
        medium = _movement_medium_result(
            geometry,
            state.position[:, TARGET_ENTITY],
        )
        target_speed_multiplier = medium.horizontal_speed_multiplier
        medium_exhausted = medium.geometry_exhausted

    # Component_Instruction_Intelligent_Chase writes LastSeen only from legal
    # sight or hearing evidence. Pursuit without either must consume that
    # stored coordinate, never the actor's current privileged position.
    actor_position = state.position[:, AGENT_ENTITY]
    target_detected = _target_detects_actor(state, params, visible)
    refresh_last_seen = can_turn & target_detected
    last_seen_position = jnp.where(
        refresh_last_seen[:, None],
        actor_position,
        state.target_last_seen_position,
    )
    last_seen_valid = state.target_last_seen_valid | refresh_last_seen
    tracked_position = jnp.where(
        target_detected[:, None],
        actor_position,
        last_seen_position,
    )
    memory_offset = tracked_position - state.position[:, TARGET_ENTITY]
    memory_distance = jnp.linalg.norm(memory_offset, axis=1)
    memory_in_range = (
        target_detected
        | (memory_distance <= params.target_chase_view_range)
    )
    can_track = can_turn & last_seen_valid & memory_in_range
    # Native BodyMotionMaintainDistance continues to correct range and advance
    # its authored strafe schedule while the melee interaction is rooted.
    can_move = can_track

    offset = memory_offset
    dx = offset[:, 0]
    dz = offset[:, 2]
    distance = jnp.hypot(dx, dz)
    bearing = _canonical_bearing(dx, dz)

    maintain = (
        can_move & visible & (distance <= params.target_maintain_activation_range)
    )
    desired_min = params.target_maintain_desired_distance_min
    desired_max = params.target_maintain_desired_distance_max
    lower_threshold = jnp.maximum(
        jnp.float32(0.0),
        desired_min - params.target_maintain_move_threshold,
    )
    upper_threshold = desired_max + params.target_maintain_move_threshold
    approach_target = desired_min + (desired_max - desired_min) * (
        jnp.float32(1.0) - params.target_maintain_target_distance_factor
    )
    away_target = (
        desired_min
        + (desired_max - desired_min) * params.target_maintain_target_distance_factor
    )
    approaching = maintain & (
        (distance > upper_threshold)
        | (state.target_maintain_approaching & (distance > approach_target))
    )
    moving_away = (
        maintain
        & ~approaching
        & (
            (distance < lower_threshold)
            | (state.target_maintain_moving_away & (distance < away_target))
        )
    )
    next_approaching = jnp.where(
        maintain,
        approaching,
        state.target_maintain_approaching,
    )
    next_moving_away = jnp.where(
        maintain,
        moving_away,
        state.target_maintain_moving_away,
    )

    had_strafe_delay = state.target_strafe_delay_seconds > 0.0
    decremented_strafe_delay = state.target_strafe_delay_seconds - motion_delta
    toggle_strafe = maintain & ~had_strafe_delay
    begin_strafe = toggle_strafe & state.target_strafe_paused
    begin_strafe_pause = toggle_strafe & ~state.target_strafe_paused
    strafe_delay = jnp.where(
        begin_strafe,
        strafe_duration_draw,
        jnp.where(
            begin_strafe_pause,
            strafe_frequency_draw,
            decremented_strafe_delay,
        ),
    )
    strafe_delay = jnp.where(
        maintain,
        strafe_delay,
        state.target_strafe_delay_seconds,
    )
    strafe_paused = jnp.where(
        begin_strafe,
        False,
        jnp.where(
            begin_strafe_pause,
            True,
            state.target_strafe_paused,
        ),
    )
    strafe_direction = jnp.where(
        begin_strafe,
        strafe_direction_draw,
        state.target_strafe_direction,
    )
    strafing = maintain & ~strafe_paused

    approach_relative_speed = (
        params.target_maintain_forward_relative_speed
        * _pursue_speed_scale(
            distance,
            approach_target,
            approach_target + params.target_maintain_slowdown_distance,
            params.target_steering_slowdown_falloff,
        )
    )
    maintain_relative_speed = jnp.where(
        approaching,
        approach_relative_speed,
        jnp.where(
            moving_away,
            params.target_maintain_backward_relative_speed,
            jnp.where(
                strafing,
                params.target_maintain_forward_relative_speed,
                jnp.float32(0.0),
            ),
        ),
    )
    base_maintain_yaw = jnp.where(
        moving_away,
        _normalize_degrees(bearing + jnp.float32(180.0)),
        bearing,
    )
    moving_strafe_offset = (
        strafe_direction.astype(jnp.float32)
        * params.target_strafe_yaw_offset_degrees
        * jnp.where(moving_away, -1.0, 1.0)
    )
    idle_strafe_offset = (
        strafe_direction.astype(jnp.float32)
        * params.target_strafe_translation_offset_degrees
    )
    maintain_translation_yaw = _normalize_degrees(
        base_maintain_yaw
        + jnp.where(
            strafing,
            jnp.where(
                approaching | moving_away,
                moving_strafe_offset,
                idle_strafe_offset,
            ),
            jnp.float32(0.0),
        )
    )
    maintain_body_yaw = _normalize_degrees(
        bearing
        + jnp.where(
            strafing,
            strafe_direction.astype(jnp.float32)
            * params.target_strafe_yaw_offset_degrees,
            jnp.float32(0.0),
        )
    )
    chase_relative_speed = (
        params.target_chase_speed
        / params.target_max_speed
        * _pursue_speed_scale(
            distance,
            params.target_chase_stop_distance,
            params.target_chase_slowdown_distance,
            params.target_steering_slowdown_falloff,
        )
    )
    requested_relative_speed = jnp.where(
        maintain,
        maintain_relative_speed,
        chase_relative_speed,
    )
    motion_requested = can_move & (requested_relative_speed > 0.0)
    range_ticks = jnp.where(
        motion_requested,
        state.target_out_of_range_ticks + 1,
        0,
    )
    range_ticks = jnp.where(
        can_move,
        range_ticks,
        state.target_out_of_range_ticks,
    )
    reaction_ready = range_ticks >= params.target_chase_reaction_ticks

    desired_body_yaw = jnp.where(
        maintain,
        maintain_body_yaw,
        bearing,
    )
    turned_yaw = _approach_target_angle(
        state.yaw[:, TARGET_ENTITY],
        desired_body_yaw,
        params.target_turn_speed_degrees * motion_delta,
    )
    target_yaw = jnp.where(
        can_track & (distance > 1.0e-9),
        turned_yaw,
        state.yaw[:, TARGET_ENTITY],
    )
    yaw = state.yaw.at[:, TARGET_ENTITY].set(target_yaw)

    batch = distance.shape[0]
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
    target_eye = state.position[:, TARGET_ENTITY] + target_eye_offset
    agent_min_y = tracked_position[:, 1] + agent_bounds[:, 1]
    agent_max_y = tracked_position[:, 1] + agent_bounds[:, 4]
    aim_y = jnp.clip(target_eye[:, 1], agent_min_y, agent_max_y)
    aim_dx = tracked_position[:, 0] - target_eye[:, 0]
    aim_dz = tracked_position[:, 2] - target_eye[:, 2]
    aim_distance = jnp.hypot(aim_dx, aim_dz)
    desired_head_yaw = _canonical_bearing(aim_dx, aim_dz)
    desired_head_pitch = jnp.rad2deg(
        jnp.arctan2(
            aim_y - target_eye[:, 1],
            jnp.maximum(aim_distance, jnp.float32(1.0e-9)),
        )
    )
    head_aiming = (
        can_track & visible & (distance <= jnp.max(params.target_end_distances))
    )
    requested_head_yaw = jnp.where(
        head_aiming,
        desired_head_yaw,
        target_yaw,
    )
    requested_head_pitch = jnp.where(
        head_aiming,
        desired_head_pitch,
        jnp.float32(0.0),
    )
    head_relative_speed = jnp.where(
        head_aiming,
        params.target_head_aim_relative_turn_speed,
        params.target_head_default_relative_turn_speed,
    )
    head_turn_step = (
        params.target_max_head_rotation_degrees * head_relative_speed * motion_delta
    )
    turned_head_yaw = _approach_target_angle(
        state.target_head_yaw,
        requested_head_yaw,
        head_turn_step,
    )
    turned_head_pitch = _approach(
        state.target_head_pitch,
        requested_head_pitch,
        head_turn_step,
    )
    relative_head_yaw = jnp.clip(
        _normalize_degrees(turned_head_yaw - target_yaw),
        params.target_head_yaw_min_degrees,
        params.target_head_yaw_max_degrees,
    )
    next_head_yaw = _normalize_degrees(target_yaw + relative_head_yaw)
    next_head_pitch = jnp.clip(
        turned_head_pitch,
        params.target_head_pitch_min_degrees,
        params.target_head_pitch_max_degrees,
    )
    next_head_yaw = jnp.where(
        can_track,
        next_head_yaw,
        state.target_head_yaw,
    )
    next_head_pitch = jnp.where(
        can_track,
        next_head_pitch,
        state.target_head_pitch,
    )

    target_velocity = state.velocity[:, TARGET_ENTITY]
    target_speed = jnp.hypot(
        target_velocity[:, 0],
        target_velocity[:, 2],
    )
    desired_speed = (
        params.target_max_speed * target_speed_multiplier * requested_relative_speed
    )
    next_speed = jnp.minimum(
        desired_speed,
        target_speed + params.target_acceleration * motion_delta,
    )
    next_speed = jnp.where(
        motion_requested & reaction_ready,
        next_speed,
        jnp.float32(0.0),
    )
    translation_yaw = jnp.where(
        maintain,
        maintain_translation_yaw,
        target_yaw,
    )
    translation_yaw_radians = jnp.deg2rad(translation_yaw)
    next_vx = -jnp.sin(translation_yaw_radians) * next_speed
    next_vz = -jnp.cos(translation_yaw_radians) * next_speed
    target_velocity = target_velocity.at[:, 0].set(next_vx)
    target_velocity = target_velocity.at[:, 2].set(next_vz)
    velocity = state.velocity.at[:, TARGET_ENTITY].set(target_velocity)

    target_position = state.position[:, TARGET_ENTITY]
    target_displacement = jnp.stack(
        (
            next_vx * motion_delta,
            jnp.zeros_like(next_vx),
            next_vz * motion_delta,
        ),
        axis=1,
    )
    target_navigation_unsupported = state.target_navigation_unsupported
    # Search/Wander/ReturnHome and TargetLost are not modelled yet. Reaching
    # LastSeen, or losing the authored ReadPosition range, is the exact
    # boundary of this certified slice.
    target_navigation_unsupported = (
        target_navigation_unsupported
        | (
            can_turn
            & last_seen_valid
            & ~target_detected
            & (
                (distance <= params.target_chase_stop_distance)
                | ~memory_in_range
            )
        )
    )
    target_motion_exhausted = jnp.zeros_like(motion_requested)
    if geometry is None:
        target_position = target_position + target_displacement
    else:
        batch = target_position.shape[0]
        target_bounds = (
            geometry.target_bounds
            if geometry.target_bounds.shape[0] == batch
            else jnp.broadcast_to(geometry.target_bounds, (batch, 6))
        )
        bounds_extent = target_bounds[:, 3:] - target_bounds[:, :3]
        bounds_available = jnp.all(bounds_extent > 0.0, axis=1)
        target_motion = _resolve_aabb_motion_result(
            geometry,
            target_position,
            target_displacement,
            target_bounds,
        )
        # Hytale routes obstructed pursuit through native steering/pathfinding.
        # This accelerated slice currently certifies only an unobstructed
        # straight-line translation. Reject the whole transition instead of
        # training on a fabricated slide, tunnel, or stopped target.
        navigation_required = motion_requested & (
            ~visible | target_motion.collided
        )
        if target_navigation_provider is None:
            navigation_step = TargetNavigationStep(
                certified_available=jnp.zeros_like(motion_requested),
                position=jnp.zeros_like(target_position),
                velocity=jnp.zeros_like(target_velocity),
                geometry_exhausted=jnp.zeros_like(motion_requested),
            )
        else:
            provider_state = state._replace(
                position=state.position.at[:, AGENT_ENTITY].set(
                    tracked_position
                ),
                target_last_seen_position=last_seen_position,
                target_last_seen_valid=last_seen_valid,
            )
            navigation_displacement = jnp.where(
                navigation_required[:, None],
                target_displacement,
                jnp.float32(0.0),
            )

            def query_navigation(_):
                return validate_target_navigation_step(
                    target_navigation_provider(
                        provider_state,
                        params,
                        geometry,
                        navigation_displacement,
                        motion_delta,
                    ),
                    batch,
                )

            navigation_step = jax.lax.cond(
                jnp.any(navigation_required),
                query_navigation,
                lambda _: TargetNavigationStep(
                    certified_available=jnp.zeros_like(motion_requested),
                    position=jnp.zeros_like(target_position),
                    velocity=jnp.zeros_like(target_velocity),
                    geometry_exhausted=jnp.zeros_like(motion_requested),
                ),
                operand=None,
            )
        use_navigation = (
            navigation_required
            & bounds_available
            & navigation_step.certified_available
        )
        navigation_rejected = _target_navigation_rejected(
            motion_requested=motion_requested,
            target_in_line_of_sight=visible,
            target_bounds_available=bounds_available,
            straight_motion_collided=target_motion.collided,
            certified_navigation_step_available=(
                navigation_step.certified_available
            ),
        )
        target_navigation_unsupported = (
            target_navigation_unsupported | navigation_rejected
        )
        target_motion_exhausted = jnp.where(
            use_navigation,
            navigation_step.geometry_exhausted,
            target_motion.geometry_exhausted,
        )
        reject_motion = navigation_rejected | target_motion_exhausted
        resolved_position = jnp.where(
            use_navigation[:, None],
            navigation_step.position,
            target_motion.position,
        )
        resolved_velocity = jnp.where(
            use_navigation[:, None],
            navigation_step.velocity,
            target_velocity,
        )
        target_position = jnp.where(
            reject_motion[:, None],
            target_position,
            resolved_position,
        )
        target_velocity = jnp.where(
            reject_motion[:, None],
            jnp.float32(0.0),
            resolved_velocity,
        )
        velocity = state.velocity.at[:, TARGET_ENTITY].set(target_velocity)
    position = state.position.at[:, TARGET_ENTITY].set(target_position)
    return state._replace(
        position=position,
        velocity=velocity,
        yaw=yaw,
        target_head_yaw=next_head_yaw,
        target_head_pitch=next_head_pitch,
        target_out_of_range_ticks=range_ticks,
        target_maintain_approaching=next_approaching,
        target_maintain_moving_away=next_moving_away,
        target_strafe_delay_seconds=strafe_delay,
        target_strafe_paused=strafe_paused,
        target_strafe_direction=strafe_direction,
        target_last_seen_position=last_seen_position,
        target_last_seen_valid=last_seen_valid,
        geometry_exhausted=(
            state.geometry_exhausted | medium_exhausted | target_motion_exhausted
        ),
        target_navigation_unsupported=target_navigation_unsupported,
    )


def _tick_target_vertical_motion(
    state: CombatState,
    motion_delta: jax.Array,
    params: CombatParams,
    geometry: GeometryProvider | None = None,
) -> CombatState:
    """Advance passive Walk gravity for the target through first landing.

    ``MotionControllerWalk.computeMove`` applies its nonlinear fall recurrence
    even without horizontal role steering. The current Kweebec/Trork matchup
    authors the same Walk gravity, acceleration multiplier, drag exponent,
    and terminal speed already carried by the shared ruleset. Target-AI
    activation therefore must not freeze vertical physics.

    Horizontal pursuit remains in ``_tick_target_motion``. Keeping the
    passive vertical sweep separate prevents a floor contact from being
    mistaken for a request for native pathfinding.
    """

    position = state.position[:, TARGET_ENTITY]
    velocity = state.velocity[:, TARGET_ENTITY]
    downward_speed = jnp.maximum(jnp.float32(0.0), -velocity[:, 1])
    maximum_fall_speed = jnp.maximum(
        params.agent_walk_max_fall_speed,
        jnp.finfo(jnp.float32).tiny,
    )
    fall_ratio = downward_speed / maximum_fall_speed
    fall_drag = jnp.power(
        fall_ratio,
        params.agent_walk_gravity_drag_exponent,
    )
    next_fall_speed = jnp.clip(
        downward_speed
        + motion_delta
        * params.agent_walk_fall_acceleration_multiplier
        * params.agent_walk_gravity
        * (jnp.float32(1.0) - fall_drag),
        jnp.float32(0.0),
        maximum_fall_speed,
    )

    if geometry is None:
        grounded = (position[:, 1] <= params.floor_y) & (
            velocity[:, 1] <= jnp.float32(0.0)
        )
        proposed_vy = jnp.where(
            grounded,
            jnp.float32(0.0),
            -next_fall_speed,
        )
        proposed_y = position[:, 1] + proposed_vy * motion_delta
        landed = proposed_y < params.floor_y
        resolved_y = jnp.maximum(proposed_y, params.floor_y)
        safe_delta = jnp.maximum(
            motion_delta,
            jnp.finfo(jnp.float32).tiny,
        )
        resolved_vy = jnp.where(
            landed,
            (resolved_y - position[:, 1]) / safe_delta,
            proposed_vy,
        )
        next_position = position.at[:, 1].set(resolved_y)
        next_velocity = velocity.at[:, 1].set(resolved_vy)
        exhausted = jnp.zeros_like(state.geometry_exhausted)
    else:
        batch = position.shape[0]
        target_bounds = (
            geometry.target_bounds
            if geometry.target_bounds.shape[0] == batch
            else jnp.broadcast_to(geometry.target_bounds, (batch, 6))
        )
        probe = jnp.zeros_like(position).at[:, 1].set(jnp.float32(-0.002))
        support = _resolve_aabb_motion_result(
            geometry,
            position,
            probe,
            target_bounds,
            stop_on_vertical_collision=jnp.ones(
                (batch,),
                dtype=jnp.bool_,
            ),
        )
        grounded = support.grounded & (velocity[:, 1] <= jnp.float32(0.0))
        proposed_vy = jnp.where(
            grounded,
            jnp.float32(0.0),
            -next_fall_speed,
        )
        displacement = jnp.zeros_like(position).at[:, 1].set(
            proposed_vy * motion_delta
        )
        motion = _resolve_aabb_motion_result(
            geometry,
            position,
            displacement,
            target_bounds,
            stop_on_vertical_collision=jnp.ones(
                (batch,),
                dtype=jnp.bool_,
            ),
        )
        exhausted = support.geometry_exhausted | motion.geometry_exhausted
        next_position = jnp.where(
            exhausted[:, None],
            position,
            motion.position,
        )
        safe_delta = jnp.maximum(
            motion_delta,
            jnp.finfo(jnp.float32).tiny,
        )
        resolved_vy = jnp.where(
            motion.collided,
            (next_position[:, 1] - position[:, 1]) / safe_delta,
            proposed_vy,
        )
        next_velocity = velocity.at[:, 1].set(
            jnp.where(exhausted, jnp.float32(0.0), resolved_vy)
        )

    return state._replace(
        position=state.position.at[:, TARGET_ENTITY].set(next_position),
        velocity=state.velocity.at[:, TARGET_ENTITY].set(next_velocity),
        geometry_exhausted=state.geometry_exhausted | exhausted,
    )


def _target_navigation_rejected(
    *,
    motion_requested: jax.Array,
    target_in_line_of_sight: jax.Array,
    target_bounds_available: jax.Array,
    straight_motion_collided: jax.Array,
    certified_navigation_step_available: jax.Array,
) -> jax.Array:
    """Reject target motion unless its exact execution path is certified.

    Unobstructed straight-line motion is already certified locally. A motion
    that loses perception LOS or collides may proceed only when the World
    capability supplies a certified navigation step for that same transition.
    Missing target bounds can certify neither path and therefore always reject.
    """

    navigation_required = motion_requested & (
        ~target_in_line_of_sight | straight_motion_collided
    )
    execution_available = target_bounds_available & (
        ~navigation_required | certified_navigation_step_available
    )
    return motion_requested & ~execution_available


def _pursue_speed_scale(
    distance: jax.Array,
    stop_distance: jax.Array,
    slowdown_distance: jax.Array,
    falloff: jax.Array,
) -> jax.Array:
    """Hytale 0.5.7 ``SteeringForcePursue`` translation magnitude."""

    ratio = jnp.clip(
        (distance - stop_distance) / (slowdown_distance - stop_distance),
        jnp.float32(0.0),
        jnp.float32(1.0),
    )
    slowed = jnp.power(ratio, jnp.float32(1.0) / falloff)
    return jnp.where(
        distance <= stop_distance,
        jnp.float32(0.0),
        jnp.where(
            distance >= slowdown_distance,
            jnp.float32(1.0),
            slowed,
        ),
    )
