"""Leaf helpers extracted verbatim from env.py."""

import jax
import jax.numpy as jnp
from hytalegym.jax.world.geometry import aabb_grounded as local_aabb_grounded, aabb_support_sweep as local_aabb_support_sweep
from hytalegym.jax.world.region.geometry import region_aabb_grounded, region_aabb_support_sweep
from hytalegym.jax.world.region.types import RegionGeometryState
from hytalegym.jax.combat.loadout.schema.types import MeleeLoadoutBatch
from hytalegym.jax.combat.motion.movement_state import actor_walk_movement_state_result, hytale_0_5_7_actor_walk_movement_config
from hytalegym.jax.combat.types import AGENT_ENTITY, TARGET_ENTITY, CombatParams, CombatState
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
    _uniform_batch,
)
from hytalegym.jax.combat.opponents.legacy.controller import (  # noqa: F401
    COOLDOWN,
    GeometryProvider,
    IDLE,
    RECOVERY,
    SWEEP,
    WINDUP,
    _combat_hitbox_line_of_sight,
    _combat_window_exhausted,
    _movement_medium_result,
    _resolve_aabb_motion_result,
    _target_phase,
)


_ACTOR_WALK_MOVEMENT_CONFIG = hytale_0_5_7_actor_walk_movement_config()


def _loadout_attack_count(loadout: MeleeLoadoutBatch) -> jax.Array:
    return jnp.sum(
        loadout.attack_mask.astype(jnp.int32),
        axis=1,
        dtype=jnp.int32,
    )


def _loadout_profile_index(
    loadout: MeleeLoadoutBatch,
    requested_index: jax.Array,
) -> jax.Array:
    attack_count = jnp.maximum(
        _loadout_attack_count(loadout),
        jnp.int32(1),
    )
    return jnp.mod(
        jnp.maximum(requested_index, jnp.int32(0)),
        attack_count,
    ).astype(jnp.int32)


def _loadout_row_valid(loadout: MeleeLoadoutBatch) -> jax.Array:
    return (
        loadout.equipped_mask & ~loadout.overflow & (_loadout_attack_count(loadout) > 0)
    )


def _convert_to_new_range(
    value: jax.Array,
    old_min: jax.Array,
    old_max: jax.Array,
    new_min: jax.Array,
    new_max: jax.Array,
) -> jax.Array:
    """Port of ``MotionControllerBase.convertToNewRange``.

    A clamped linear remap, and the source's own helper for every speed-indexed
    air parameter. The clamp is taken over ``min(new_min, new_max)`` and
    ``max(new_min, new_max)`` rather than over the arguments in order, because
    the friction call site deliberately passes the pair reversed.
    """

    span = old_max - old_min
    safe_span = jnp.where(
        jnp.abs(span) <= jnp.float32(1.0e-9),
        jnp.float32(1.0),
        span,
    )
    remapped = (value - old_min) * (new_max - new_min) / safe_span + new_min
    remapped = jnp.where(
        jnp.abs(span) <= jnp.float32(1.0e-9),
        new_min,
        remapped,
    )
    return jnp.clip(
        remapped,
        jnp.minimum(new_min, new_max),
        jnp.maximum(new_min, new_max),
    )


def _gather_loadout_profile(
    values: jax.Array,
    profile_index: jax.Array,
) -> jax.Array:
    return jnp.take_along_axis(
        values,
        profile_index[:, None],
        axis=1,
    )[:, 0]


def _airborne_walk_motion(
    prior_velocity_x: jax.Array,
    prior_velocity_z: jax.Array,
    yaw_degrees: jax.Array,
    carried_move_speed: jax.Array,
    speed_cap: jax.Array,
    wish_x: jax.Array,
    wish_z: jax.Array,
    motion_delta: jax.Array,
    params: CombatParams,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Airborne horizontal motion for a *player*, returning ``(vx, vz, speed)``.

    The learner jumps with ``PLAYER_JUMP_FORCE`` and runs the player gait table,
    so it steers and bleeds speed in the air like a player. It used to glide:
    this function replaced a faithful port of ``MotionControllerWalk``'s
    airborne branch, which takes its heading from the current velocity and never
    touches ``moveSpeed`` -- correct for an NPC, wrong for the actor it was
    applied to. See ``PLAYER_AIR_*`` in ``types.py`` for the sourcing and for
    which half of this is authored and which half is INFERRED.

    ``wish_x``/``wish_z`` are a unit steering direction, or zero for no input.
    """

    prior_horizontal_speed = jnp.hypot(prior_velocity_x, prior_velocity_z)
    safe_prior_horizontal_speed = jnp.maximum(
        prior_horizontal_speed,
        jnp.finfo(jnp.float32).tiny,
    )
    # A jump straight up has no horizontal direction to carry, so fall back to
    # where the actor is looking rather than dividing by zero.
    yaw_radians = jnp.deg2rad(yaw_degrees)
    moving = prior_horizontal_speed > jnp.float32(1.0e-9)
    direction_x = jnp.where(
        moving,
        prior_velocity_x / safe_prior_horizontal_speed,
        -jnp.sin(yaw_radians),
    )
    direction_z = jnp.where(
        moving,
        prior_velocity_z / safe_prior_horizontal_speed,
        -jnp.cos(yaw_radians),
    )
    carried_speed = jnp.minimum(carried_move_speed, speed_cap)
    air_control = _convert_to_new_range(
        prior_horizontal_speed,
        params.agent_air_control_min_speed,
        params.agent_air_control_max_speed,
        params.agent_air_control_min_multiplier,
        params.agent_air_control_max_multiplier,
    )
    # Normalised against the authored ceiling, so this is the *fraction* of
    # ground control retained in the air and it saturates at parity rather than
    # exceeding it. The alternative reading -- that 3.13 scales the ground
    # acceleration outright, giving 3.13x ground responsiveness while airborne
    # -- spends the constant more literally, but it mixes unit systems: the
    # multiplier is authored against the player's ``Acceleration`` of 0.1 while
    # ``agent_acceleration`` here is the NPC ruleset's 10.0 per second. It would
    # also make a jump strictly more manoeuvrable than a step, which is the
    # direction that made the reward exploitable in the first place. Because
    # ``convertToNewRange`` saturates at ``AirControlMaxSpeed`` = 3 and every
    # pursuit gait clears that, the two readings differ only by this scale.
    # THE KNOB: raise toward ``agent_air_control_max_multiplier`` for the
    # literal reading.
    air_control_fraction = air_control / params.agent_air_control_max_multiplier
    air_drag = _convert_to_new_range(
        prior_horizontal_speed,
        params.agent_air_drag_min_speed,
        params.agent_air_drag_max_speed,
        params.agent_air_drag_min,
        params.agent_air_drag_max,
    )
    # ``drag = pow(drag, 60.0 / serverTps)`` in the source. 60 / tps is 60 * dt,
    # so one exponent holds at both the nominal 30 Hz tick and the 0.045 s
    # loaded tick. ``_damp_force_velocity`` uses the literal ``60 / tps`` form
    # because it damps once per tick; this runs on ``motion_delta``, which is
    # the tick that actually elapsed.
    air_drag = air_drag ** (
        params.agent_force_reference_ticks_per_second * motion_delta
    )
    step = params.agent_acceleration * air_control_fraction * motion_delta
    steered_x = (direction_x * carried_speed + wish_x * step) * air_drag
    steered_z = (direction_z * carried_speed + wish_z * step) * air_drag
    steered_speed = jnp.hypot(steered_x, steered_z)
    clamp = jnp.minimum(
        jnp.float32(1.0),
        speed_cap / jnp.maximum(steered_speed, jnp.finfo(jnp.float32).tiny),
    )
    return steered_x * clamp, steered_z * clamp, steered_speed * clamp


def _force_pushed_agent_motion(
    external_velocity: jax.Array,
    prior_move_speed: jax.Array,
    agent_yaw_degrees: jax.Array,
    params: CombatParams,
) -> tuple[jax.Array, jax.Array]:
    """Apply 0.5.7 ``MotionControllerWalk`` forced-push carry-through."""

    external_length = jnp.hypot(
        external_velocity[:, 0],
        external_velocity[:, 2],
    )
    yaw = jnp.deg2rad(agent_yaw_degrees)
    heading_x = -jnp.sin(yaw)
    heading_z = -jnp.cos(yaw)
    alignment_numerator = (
        heading_x * external_velocity[:, 0] + heading_z * external_velocity[:, 2]
    )
    alignment = jnp.where(
        external_length > 0.0,
        alignment_numerator / jnp.maximum(external_length, jnp.finfo(jnp.float32).tiny),
        jnp.float32(0.0),
    )
    max_walk_speed_after_hit_multiplier = (
        jnp.float32(1.0) - params.agent_min_hit_slowdown
    )
    walk_multiplier = jnp.minimum(
        (alignment + jnp.float32(1.0)) / jnp.float32(2.0),
        max_walk_speed_after_hit_multiplier,
    )
    reduced_move_speed = prior_move_speed * walk_multiplier
    can_reduce_walk = (external_length > 0.0) & (prior_move_speed > 0.0)
    add_reduced_walk = can_reduce_walk & (
        reduced_move_speed > params.agent_min_walk_speed
    )
    resulting_move_speed = jnp.where(
        can_reduce_walk,
        jnp.where(
            add_reduced_walk,
            reduced_move_speed,
            jnp.float32(0.0),
        ),
        prior_move_speed,
    )
    carried_velocity = jnp.where(
        add_reduced_walk,
        reduced_move_speed,
        jnp.float32(0.0),
    )
    velocity = (
        external_velocity.at[:, 0]
        .add(heading_x * carried_velocity)
        .at[:, 2]
        .add(heading_z * carried_velocity)
    )
    return velocity, resulting_move_speed


def _damp_force_velocity(
    force_velocity: jax.Array,
    grounded: jax.Array,
    in_fluid: jax.Array,
    params: CombatParams,
) -> jax.Array:
    """Apply the 0.5.7 post-move ``forceVelocity`` drag and deadzone."""

    horizontal_speed = jnp.hypot(
        force_velocity[:, 0],
        force_velocity[:, 2],
    )
    air_speed_span = jnp.maximum(
        params.agent_force_air_drag_max_speed - params.agent_force_air_drag_min_speed,
        jnp.finfo(jnp.float32).tiny,
    )
    air_drag_fraction = jnp.clip(
        (horizontal_speed - params.agent_force_air_drag_min_speed) / air_speed_span,
        jnp.float32(0.0),
        jnp.float32(1.0),
    )
    air_drag = params.agent_force_air_drag_min + air_drag_fraction * (
        params.agent_force_air_drag_max - params.agent_force_air_drag_min
    )
    drag_base = jnp.where(
        grounded | in_fluid,
        params.agent_force_ground_drag_base,
        air_drag,
    )
    tick_rate = params.agent_force_reference_ticks_per_second / params.ticks_per_second
    drag = jnp.power(drag_base, tick_rate)
    damped = force_velocity.at[:, 0].multiply(drag).at[:, 2].multiply(drag)
    return jnp.where(
        jnp.abs(damped) <= params.agent_force_per_axis_deadzone,
        jnp.float32(0.0),
        damped,
    )


def _head_steering(
    prior_head_yaw: jax.Array,
    prior_head_pitch: jax.Array,
    body_yaw: jax.Array,
    body_pitch: jax.Array,
    desired_yaw: jax.Array,
    desired_pitch: jax.Array,
    horizontal_speed_multiplier: jax.Array,
    motion_delta: jax.Array,
    params: CombatParams,
) -> tuple[jax.Array, jax.Array]:
    """Turn the head, then bound it to the authored window around the body.

    A second steering seam beside the body turn. `MotionControllerBase` 506-507
    builds the two ceilings differently::

        maxBodyRotation = interval * bodySpeed * bodySteering.relativeTurnSpeed
        maxHeadRotation = interval * headSpeed * headSteering.relativeTurnSpeed
                                   * effectHorizontalSpeedMultiplier

    Only the head carries that last factor, and for this actor it is the *only*
    thing that ever separates the two seams: the bridge commands body and head
    to the same yaw (`NativeEnvironmentSession` 3957 and 3964) and both base
    ceilings are 360 deg/s, so an unslowed head and body turn together and the
    window never binds. Slow the actor -- fluid, or a charge's
    ``HorizontalSpeedMultiplier`` -- and the head falls behind its own chest,
    which is when the window starts to matter.

    `calculateYaw` 628-656 bounds the head into that window and pins it at the
    edge. `calculatePitch` 700-730 does the same for pitch but never drags the
    body: it has no ``blendBodyYaw`` counterpart.
    """

    head_step = (
        params.agent_max_head_rotation_degrees
        * params.agent_steering_relative_turn_speed
        * horizontal_speed_multiplier
        * motion_delta
    )
    head_yaw = _approach_angle(prior_head_yaw, desired_yaw, head_step)
    head_pitch = _approach(prior_head_pitch, desired_pitch, head_step)
    head_yaw = _normalize_degrees(
        body_yaw
        + jnp.clip(
            _normalize_degrees(head_yaw - body_yaw),
            params.agent_head_yaw_min_degrees,
            params.agent_head_yaw_max_degrees,
        )
    )
    head_pitch = jnp.clip(
        head_pitch,
        body_pitch + params.agent_head_pitch_min_degrees,
        body_pitch + params.agent_head_pitch_max_degrees,
    )
    return head_yaw, head_pitch


def _tick_agent_motion(
    state: CombatState,
    motion_delta: jax.Array,
    params: CombatParams,
    geometry: GeometryProvider | None = None,
) -> CombatState:
    # The body chases its OWN commanded heading at its own ceiling. It used to
    # chase `desired_yaw`, which is the camera, so the chest was dragged around
    # by every look and an actor could not strafe. `calculateYaw` 604-608 keeps
    # these separate; the head is bounded around the result in `_head_steering`.
    agent_yaw = _approach_angle(
        state.yaw[:, AGENT_ENTITY],
        state.desired_body_yaw,
        params.agent_turn_speed_degrees * motion_delta,
    )
    # The body does not pitch from the camera. `calculatePitch` 674-685 takes
    # body pitch from an explicit body steer or, failing that, from the
    # translation direction -- and the bridge only ever hands it a planar
    # translation (`NativeEnvironmentSession` 3940-3942 pins dy to 0 outside the
    # dive fixture), so a walking actor's chest stays level however far it looks
    # up or down. Only the head pitches, and it is bounded against this level
    # body, which is what makes the authored +/-45 window an absolute aim limit
    # rather than a window around an already-tilted chest.
    pitch = jnp.zeros_like(state.pitch)
    yaw = state.yaw.at[:, AGENT_ENTITY].set(agent_yaw)

    force_velocity = state.agent_force_velocity
    force_active = jnp.any(
        jnp.abs(force_velocity) > jnp.float32(1.0e-9),
        axis=1,
    )
    force_pushed = state.target_damage_applied_this_tick
    pushed_velocity, pushed_move_speed = _force_pushed_agent_motion(
        force_velocity,
        state.agent_move_speed,
        state.yaw[:, AGENT_ENTITY],
        params,
    )
    pushed_velocity = pushed_velocity.at[:, 1].add(
        state.agent_applied_vertical_velocity
    )
    agent_velocity = jnp.where(
        force_pushed[:, None],
        pushed_velocity,
        state.velocity[:, AGENT_ENTITY],
    )
    vertical_impulse_at_start = jnp.where(
        force_pushed,
        True,
        state.vertical_impulse_applied,
    )
    residual_velocity_at_start = jnp.where(
        force_pushed,
        False,
        state.grounded_with_residual_velocity,
    )
    control_lock_at_start = jnp.where(
        force_pushed,
        True,
        state.knockback_control_lock,
    )
    grounded_at_start = jnp.where(
        force_pushed,
        False,
        state.agent_grounded,
    )
    desired_velocity = state.desired_velocity
    medium_exhausted = jnp.zeros(
        (agent_velocity.shape[0],),
        dtype=jnp.bool_,
    )
    medium_in_fluid = jnp.zeros(
        (agent_velocity.shape[0],),
        dtype=jnp.bool_,
    )
    horizontal_speed_multiplier = jnp.ones(
        (agent_velocity.shape[0],),
        dtype=jnp.float32,
    )
    if geometry is not None:
        medium = _movement_medium_result(
            geometry,
            state.position[:, AGENT_ENTITY],
        )
        horizontal_speed_multiplier = medium.horizontal_speed_multiplier
        desired_velocity = desired_velocity * horizontal_speed_multiplier[:, None]
        medium_in_fluid = medium.in_fluid
        medium_exhausted = medium.geometry_exhausted

    head_yaw, head_pitch = _head_steering(
        state.agent_head_yaw,
        state.agent_head_pitch,
        agent_yaw,
        pitch,
        state.desired_yaw,
        state.desired_pitch,
        horizontal_speed_multiplier,
        motion_delta,
        params,
    )
    desired_zero = jnp.all(
        jnp.abs(desired_velocity) <= 1.0e-9,
        axis=1,
    )
    stop = control_lock_at_start | desired_zero
    desired_speed = jnp.hypot(
        desired_velocity[:, 0],
        desired_velocity[:, 1],
    )
    next_move_speed = jnp.minimum(
        desired_speed,
        state.agent_move_speed + params.agent_acceleration * motion_delta,
    )
    safe_desired_speed = jnp.maximum(
        desired_speed,
        jnp.finfo(jnp.float32).tiny,
    )
    accelerated_x = desired_velocity[:, 0] / safe_desired_speed * next_move_speed
    accelerated_z = desired_velocity[:, 1] / safe_desired_speed * next_move_speed
    move_speed = jnp.where(
        force_active | control_lock_at_start,
        state.agent_move_speed,
        jnp.where(desired_zero, jnp.float32(0.0), next_move_speed),
    )
    move_speed = jnp.where(
        force_pushed,
        pushed_move_speed,
        move_speed,
    )
    # A JUMP IS AIRBORNE. This used to also require
    # `agent_applied_vertical_velocity` to be ~0, which is exactly the window a
    # jump is not in -- so for the whole of a jump the actor fell through to the
    # grounded branch below and got full gait velocity and full steering
    # authority in mid-air, every tick.
    #
    # That is not how the game moves: horizontal velocity is set at takeoff on
    # the ground and then carried, with limited air steering and no way to
    # accelerate or jump again. `_airborne_walk_motion` already implements that
    # model; it was simply never reached during a jump.
    #
    # This is deliberately NOT gated on `geometry`. Air steering is kinematics,
    # not collision: `_airborne_walk_motion` takes no geometry, and the
    # no-geometry branch below maintains `agent_grounded` from the floor plane
    # and integrates horizontal position straight from `agent_velocity`, which
    # is what this selects. Gating it meant flat and design worlds -- which
    # carry no geometry provider -- kept full mid-air acceleration.
    #
    # Knockback does not need the vertical term to stay out: `~force_active`
    # already excludes it.
    walk_airborne = (
        ~grounded_at_start
        & ~control_lock_at_start
        & ~force_active
    )
    airborne_velocity_x, airborne_velocity_z, airborne_move_speed = (
        _airborne_walk_motion(
            agent_velocity[:, 0],
            agent_velocity[:, 2],
            agent_yaw,
            state.agent_move_speed,
            params.agent_max_speed * horizontal_speed_multiplier,
            jnp.where(
                desired_zero,
                jnp.float32(0.0),
                desired_velocity[:, 0] / safe_desired_speed,
            ),
            jnp.where(
                desired_zero,
                jnp.float32(0.0),
                desired_velocity[:, 1] / safe_desired_speed,
            ),
            motion_delta,
            params,
        )
    )
    move_speed = jnp.where(
        walk_airborne,
        airborne_move_speed,
        move_speed,
    )
    agent_velocity = agent_velocity.at[:, 0].set(
        jnp.where(
            walk_airborne,
            airborne_velocity_x,
            jnp.where(
                force_active,
                jnp.where(
                    force_pushed,
                    agent_velocity[:, 0],
                    force_velocity[:, 0],
                ),
                jnp.where(stop, 0.0, accelerated_x),
            ),
        )
    )
    agent_velocity = agent_velocity.at[:, 2].set(
        jnp.where(
            walk_airborne,
            airborne_velocity_z,
            jnp.where(
                force_active,
                jnp.where(
                    force_pushed,
                    agent_velocity[:, 2],
                    force_velocity[:, 2],
                ),
                jnp.where(stop, 0.0, accelerated_z),
            ),
        )
    )

    agent_position = state.position[:, AGENT_ENTITY]
    if geometry is None:
        agent_position = agent_position.at[:, 0].add(
            agent_velocity[:, 0] * motion_delta
        )
        agent_position = agent_position.at[:, 2].add(
            agent_velocity[:, 2] * motion_delta
        )
        (
            agent_position,
            agent_velocity,
            vertical_impulse,
            residual_velocity,
            control_lock,
        ) = _tick_agent_vertical(
            agent_position,
            agent_velocity,
            vertical_impulse_at_start,
            residual_velocity_at_start,
            control_lock_at_start,
            motion_delta,
            params,
        )
        agent_grounded = (
            agent_position[:, 1] <= params.floor_y + 1.0e-9
        ) & ~vertical_impulse
        geometry_exhausted = state.geometry_exhausted
        agent_fall_speed = state.agent_fall_speed
        agent_fall_start_y = state.agent_fall_start_y
    else:
        (
            agent_position,
            agent_velocity,
            vertical_impulse,
            residual_velocity,
            control_lock,
            agent_grounded,
            geometry_exhausted,
            agent_fall_speed,
            agent_fall_start_y,
            move_speed,
        ) = _tick_agent_geometry_motion(
            geometry,
            agent_position,
            agent_velocity,
            vertical_impulse_at_start,
            residual_velocity_at_start,
            control_lock_at_start,
            grounded_at_start,
            state.agent_applied_vertical_velocity,
            state.agent_fall_speed,
            state.agent_fall_start_y,
            move_speed,
            medium_in_fluid,
            motion_delta,
            params,
        )
        geometry_exhausted = geometry_exhausted | medium_exhausted
    force_velocity = force_velocity.at[:, 1].set(
        jnp.where(
            force_active,
            agent_velocity[:, 1],
            force_velocity[:, 1],
        )
    )
    force_velocity = _damp_force_velocity(
        force_velocity,
        agent_grounded,
        medium_in_fluid,
        params,
    )
    position = state.position.at[:, AGENT_ENTITY].set(agent_position)
    velocity = state.velocity.at[:, AGENT_ENTITY].set(agent_velocity)
    next_state = state._replace(
        position=position,
        velocity=velocity,
        agent_move_speed=move_speed,
        agent_fall_speed=agent_fall_speed,
        agent_fall_start_y=agent_fall_start_y,
        yaw=yaw,
        pitch=pitch,
        agent_head_yaw=head_yaw,
        agent_head_pitch=head_pitch,
        vertical_impulse_applied=vertical_impulse,
        agent_applied_vertical_velocity=jnp.where(
            agent_grounded,
            jnp.float32(0.0),
            state.agent_applied_vertical_velocity,
        ),
        grounded_with_residual_velocity=residual_velocity,
        knockback_control_lock=control_lock,
        agent_grounded=agent_grounded,
        agent_force_velocity=force_velocity,
        geometry_exhausted=(state.geometry_exhausted | geometry_exhausted),
    )
    return _refresh_agent_walk_movement_state(
        next_state,
        params,
        controller_in_fluid=medium_in_fluid,
    )


def _refresh_agent_walk_movement_state(
    state: CombatState,
    params: CombatParams,
    *,
    controller_in_fluid: jax.Array,
) -> CombatState:
    """Advance source-backed actor telemetry at the same microtick boundary."""

    controller_available = (
        (state.health[:, AGENT_ENTITY] > jnp.float32(0.0))
        & ~state.geometry_exhausted
    )
    result = actor_walk_movement_state_result(
        state.agent_walk_movement_state,
        state,
        params,
        controller_in_fluid=controller_in_fluid,
        controller_available=controller_available,
        config=_ACTOR_WALK_MOVEMENT_CONFIG,
    )
    return state._replace(agent_walk_movement_state=result.state)


def _tick_agent_vertical(
    position: jax.Array,
    velocity: jax.Array,
    impulse: jax.Array,
    residual: jax.Array,
    control_lock: jax.Array,
    motion_delta: jax.Array,
    params: CombatParams,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    y = position[:, 1]
    vy = velocity[:, 1]
    grounded = (y <= params.floor_y) & (vy <= 0.0)

    gravity_vy = vy - params.world_gravity * motion_delta
    gravity_y = y + gravity_vy * motion_delta
    landed = gravity_y <= params.floor_y
    gravity_y = jnp.where(landed, params.floor_y, gravity_y)
    gravity_vy = jnp.where(
        landed,
        gravity_vy * params.landing_velocity_scale,
        gravity_vy,
    )

    next_y = jnp.where(
        impulse,
        y + vy * motion_delta,
        jnp.where(
            residual,
            y,
            jnp.where(grounded, params.floor_y, gravity_y),
        ),
    )
    next_vy = jnp.where(
        impulse,
        vy,
        jnp.where(
            residual,
            0.0,
            jnp.where(grounded, 0.0, gravity_vy),
        ),
    )
    next_residual = jnp.where(
        impulse | residual | grounded,
        False,
        landed,
    )
    next_lock = jnp.where(
        residual,
        False,
        control_lock,
    )
    position = position.at[:, 1].set(next_y)
    velocity = velocity.at[:, 1].set(next_vy)
    return (
        position,
        velocity,
        jnp.zeros_like(impulse),
        next_residual,
        next_lock,
    )


def _tick_agent_geometry_motion(
    geometry: GeometryProvider,
    position: jax.Array,
    velocity: jax.Array,
    impulse: jax.Array,
    residual: jax.Array,
    control_lock: jax.Array,
    grounded_at_start: jax.Array,
    applied_vertical_velocity: jax.Array,
    fall_speed: jax.Array,
    fall_start_y: jax.Array,
    move_speed: jax.Array,
    in_fluid: jax.Array,
    motion_delta: jax.Array,
    params: CombatParams,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    grounded_at_start = grounded_at_start & (velocity[:, 1] <= 0.0)
    walk_controller = (
        (jnp.abs(applied_vertical_velocity) <= jnp.float32(1.0e-9))
        & ~control_lock
        & ~impulse
        & ~residual
    )
    walk_airborne = walk_controller & ~grounded_at_start
    walk_grounded = walk_controller & grounded_at_start
    maximum_fall_speed = jnp.where(
        in_fluid,
        params.agent_walk_max_sink_speed_fluid,
        params.agent_walk_max_fall_speed,
    )
    safe_maximum_fall_speed = jnp.maximum(
        maximum_fall_speed,
        jnp.finfo(jnp.float32).tiny,
    )
    fall_ratio = jnp.abs(fall_speed / safe_maximum_fall_speed)
    fall_drag = jnp.power(
        fall_ratio,
        params.agent_walk_gravity_drag_exponent,
    )
    walk_fall_speed = fall_speed + (
        motion_delta
        * params.agent_walk_fall_acceleration_multiplier
        * params.agent_walk_gravity
        * (jnp.float32(1.0) - fall_drag)
    )
    walk_fall_speed = jnp.clip(
        walk_fall_speed,
        0.0,
        maximum_fall_speed,
    )
    gravity_vy = velocity[:, 1] - params.world_gravity * motion_delta
    legacy_proposed_vy = jnp.where(
        impulse,
        velocity[:, 1],
        jnp.where(
            residual | grounded_at_start,
            0.0,
            gravity_vy,
        ),
    )
    proposed_vy = jnp.where(
        walk_controller,
        jnp.where(walk_grounded, 0.0, -walk_fall_speed),
        legacy_proposed_vy,
    )
    legacy_vertical_displacement = jnp.where(
        impulse,
        velocity[:, 1] * motion_delta,
        jnp.where(
            residual | grounded_at_start,
            0.0,
            gravity_vy * motion_delta,
        ),
    )
    vertical_displacement = jnp.where(
        walk_controller,
        jnp.where(
            walk_grounded,
            0.0,
            -walk_fall_speed * motion_delta,
        ),
        legacy_vertical_displacement,
    )
    displacement = jnp.stack(
        (
            velocity[:, 0] * motion_delta,
            vertical_displacement,
            velocity[:, 2] * motion_delta,
        ),
        axis=1,
    )
    support = _aabb_support_sweep_result(
        geometry,
        position,
        displacement.at[:, 1].set(0.0),
    )
    leaves_support = walk_grounded & support.leaves_support
    clipped_displacement = displacement * support.travel_fraction[:, None]
    requested_displacement = jnp.where(
        leaves_support[:, None],
        clipped_displacement,
        displacement,
    )
    motion = _resolve_aabb_motion_result(
        geometry,
        position,
        requested_displacement,
        stop_on_vertical_collision=walk_airborne,
    )
    # A fixed window cannot certify even a collision-free path once the swept
    # AABB leaves it. Reject that entire motion and terminate through the
    # sticky exhaustion bit instead of exposing a state reached through
    # invented all-air terrain.
    safe_position = jnp.where(
        motion.geometry_exhausted[:, None],
        position,
        motion.position,
    )

    legacy_velocity = velocity.at[:, 1].set(proposed_vy)
    normal_speed = jnp.sum(
        legacy_velocity * motion.first_normal,
        axis=1,
        keepdims=True,
    )
    legacy_velocity = legacy_velocity - motion.first_normal * jnp.minimum(
        normal_speed,
        0.0,
    )
    legacy_landed = motion.grounded & ~grounded_at_start & (proposed_vy < 0.0)
    legacy_velocity = legacy_velocity.at[:, 1].set(
        jnp.where(
            legacy_landed,
            proposed_vy * params.landing_velocity_scale,
            jnp.where(
                grounded_at_start | motion.ceiling_contact,
                0.0,
                legacy_velocity[:, 1],
            ),
        )
    )
    actual_velocity = (safe_position - position) / jnp.maximum(
        motion_delta[:, None], jnp.finfo(jnp.float32).tiny
    )
    next_velocity = jnp.where(
        walk_controller[:, None],
        actual_velocity,
        legacy_velocity,
    )
    next_residual = jnp.where(
        walk_controller,
        False,
        legacy_landed,
    )
    next_lock = jnp.where(residual, False, control_lock)
    ground = _aabb_grounded_result(geometry, safe_position)
    agent_grounded = ground.grounded & ~impulse
    walk_landed = walk_airborne & agent_grounded & (proposed_vy < 0.0)
    walk_left_ground = walk_grounded & ~agent_grounded
    next_fall_speed = jnp.where(
        walk_controller,
        jnp.where(
            walk_landed | walk_left_ground | agent_grounded,
            jnp.float32(0.0),
            walk_fall_speed,
        ),
        fall_speed,
    )
    next_fall_start_y = jnp.where(
        walk_left_ground,
        safe_position[:, 1],
        fall_start_y,
    )
    fall_height = jnp.maximum(
        jnp.float32(0.0),
        fall_start_y - safe_position[:, 1],
    )
    drop_range = jnp.maximum(
        params.agent_walk_max_drop_height - params.agent_walk_max_climb_height,
        jnp.finfo(jnp.float32).tiny,
    )
    landing_move_scale = jnp.where(
        fall_height >= params.agent_walk_max_drop_height,
        jnp.float32(0.0),
        jnp.where(
            fall_height > params.agent_walk_max_climb_height,
            (fall_height - params.agent_walk_max_climb_height) / drop_range,
            jnp.float32(1.0),
        ),
    )
    next_move_speed = jnp.where(
        walk_landed,
        move_speed * landing_move_scale,
        move_speed,
    )
    return (
        safe_position,
        next_velocity,
        jnp.zeros_like(impulse),
        next_residual,
        next_lock,
        agent_grounded,
        (
            motion.geometry_exhausted
            | ground.geometry_exhausted
            | (walk_grounded & support.geometry_exhausted)
        ),
        next_fall_speed,
        next_fall_start_y,
        next_move_speed,
    )


def _aabb_grounded_result(
    geometry: GeometryProvider,
    position: jax.Array,
):
    if isinstance(geometry, RegionGeometryState):
        return region_aabb_grounded(geometry, position)
    return local_aabb_grounded(geometry, position)


def _aabb_support_sweep_result(
    geometry: GeometryProvider,
    position: jax.Array,
    displacement: jax.Array,
):
    if isinstance(geometry, RegionGeometryState):
        return region_aabb_support_sweep(
            geometry,
            position,
            displacement,
        )
    return local_aabb_support_sweep(geometry, position, displacement)


def _tick_agent_attack(
    state: CombatState,
    params: CombatParams,
    geometry: GeometryProvider | None = None,
    agent_loadout: MeleeLoadoutBatch | None = None,
) -> tuple[CombatState, jax.Array]:
    active_delay = state.agent_hit_delay > 0
    hit_delay = jnp.where(
        active_delay,
        state.agent_hit_delay - 1,
        state.agent_hit_delay,
    )
    if agent_loadout is None:
        attack_index = jnp.clip(
            state.pending_agent_attack_index,
            0,
            params.agent_hit_delays.shape[0] - 1,
        )
        profile_available = jnp.ones_like(active_delay)
        attack_range = params.agent_attack_ranges[attack_index]
        half_angle = params.agent_half_angles[attack_index]
        attack_damage = jnp.broadcast_to(
            params.agent_damage,
            active_delay.shape,
        )
        requires_line_of_sight = jnp.ones_like(active_delay)
    else:
        attack_index = _loadout_profile_index(
            agent_loadout,
            state.pending_agent_attack_index,
        )
        profile_available = _loadout_row_valid(agent_loadout) & (
            _gather_loadout_profile(
                agent_loadout.attack_mask,
                attack_index,
            )
        )
        attack_range = _gather_loadout_profile(
            agent_loadout.range_blocks,
            attack_index,
        )
        half_angle = _gather_loadout_profile(
            agent_loadout.half_angle_degrees,
            attack_index,
        )
        attack_damage = _gather_loadout_profile(
            agent_loadout.damage,
            attack_index,
        )
        requires_line_of_sight = _gather_loadout_profile(
            agent_loadout.requires_line_of_sight,
            attack_index,
        )
    resolves = (
        active_delay
        & (hit_delay <= 0)
        & (state.pending_agent_attack_index >= 0)
        & (state.health[:, TARGET_ENTITY] > 0.0)
    )

    offset = state.position[:, TARGET_ENTITY] - state.position[:, AGENT_ENTITY]
    # The shipped NPC role surface never authors UseProjectedDistance, so
    # BuilderSensorEntityBase's stable false default selects Euclidean XYZ.
    # The asset-surface audit fails if future content activates projection.
    distance = jnp.linalg.norm(offset, axis=1)
    target_bearing = jnp.rad2deg(jnp.arctan2(-offset[:, 0], -offset[:, 2]))
    facing_error = jnp.abs(
        _normalize_degrees(target_bearing - state.yaw[:, AGENT_ENTITY])
    )
    hit = (
        resolves
        & profile_available
        & (distance <= attack_range)
        & (facing_error <= half_angle)
    )
    if geometry is not None:
        hit = hit & (
            ~requires_line_of_sight | _combat_hitbox_line_of_sight(state, geometry)
        )
    before = state.health[:, TARGET_ENTITY]
    after = jnp.where(
        hit,
        jnp.maximum(0.0, before - attack_damage),
        before,
    )
    health = state.health.at[:, TARGET_ENTITY].set(after)
    updated = state._replace(
        health=health,
        agent_hit_delay=hit_delay,
        pending_agent_attack_index=jnp.where(
            resolves,
            -1,
            state.pending_agent_attack_index,
        ),
    )
    return updated, before - after
