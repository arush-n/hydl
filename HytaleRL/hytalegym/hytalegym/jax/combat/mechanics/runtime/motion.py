"""Applied motion translation/damping and defense commands."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.mechanics.schema.contract import (
    APPLIED_RESISTANCE_STYLE_FORCE_VELOCITY,
    DODGE_DIRECTION_COUNT,
    DODGE_EXECUTION_AUTHORED_CLIENT_CONFIGURED,
    DODGE_EXECUTION_NATIVE_NPC_NULL_CONFIG,
    DODGE_LEFT,
    DODGE_NONE,
    DODGE_RIGHT,
    DODGE_VELOCITY_REMOVAL_SQUARED,
    MECHANICS_FAILURE_INVALID_COMMAND,
    MECHANICS_FAILURE_INVALID_STATE,
    MECHANICS_FAILURE_UNSUPPORTED_DODGE_COLLISION,
    RESOURCE_STAMINA,
    STAMINA_BROKEN_REGEN_DELAY_SECONDS,
    STATUS_FLAG_DISABLE_MOVEMENT,
)
from hytalegym.jax.combat.mechanics.schema.types import (
    CombatMechanicsRules,
    CombatMechanicsState,
    DefenseCommandInfo,
    DefenseCommands,
)
from hytalegym.jax.combat.types import CombatParams
from hytalegym.jax.combat.mechanics.runtime.access import (
    _dodge_local_direction,
    _entity_dt,
    _gather_entity,
    _select_tree,
    _set_entity,
    _set_failure,
    guard_resource_available,
    status_modifiers,
)


def _set_entity_where(
    array: jax.Array,
    entity_id: jax.Array,
    value: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    old = _gather_entity(array, entity_id)
    shaped = mask.reshape(mask.shape + (1,) * (value.ndim - 1))
    return _set_entity(array, entity_id, jnp.where(shaped, value, old))


def native_npc_null_config_force_velocity(
    authored_velocity: jax.Array,
    params: CombatParams,
) -> jax.Array:
    """Map an authored force vector onto the controlled native NPC path.

    Native ``simulateTick0`` executes ``ApplyForceInteraction`` without the
    authored client ``VelocityConfig``.  ``MotionControllerBase`` therefore
    applies the entity knockback scale followed by its legacy horizontal
    motion-controller factor and movement resistance.  Keeping this mapping
    as one compiled primitive prevents generic force events and dodge launch
    from silently drifting apart.
    """

    knockback_scale = jnp.asarray(params.agent_knockback_scale)
    velocity = authored_velocity * knockback_scale[..., None]
    horizontal_scale = (
        params.legacy_motion_controller_horizontal_factor
        * params.agent_movement_velocity_resistance
    )
    axis_scale = jnp.stack(
        (
            horizontal_scale,
            jnp.ones_like(horizontal_scale),
            horizontal_scale,
        ),
        axis=-1,
    )
    return velocity * axis_scale


def resolve_dodge_launch_velocity(
    authored_velocity: jax.Array,
    execution_profile: jax.Array,
    params: CombatParams,
) -> jax.Array:
    """Resolve a dodge launch without discarding its authored source vector."""

    configured_velocity = authored_velocity * jnp.asarray(
        params.agent_knockback_scale
    )[..., None]
    native_velocity = native_npc_null_config_force_velocity(
        authored_velocity,
        params,
    )
    native_profile = (
        execution_profile == DODGE_EXECUTION_NATIVE_NPC_NULL_CONFIG
    )
    return jnp.where(
        native_profile[..., None],
        native_velocity,
        configured_velocity,
    )


def damp_applied_velocity(
    velocity: jax.Array,
    grounded: jax.Array,
    air_resistance: jax.Array,
    air_resistance_max: jax.Array,
    ground_resistance: jax.Array,
    ground_resistance_max: jax.Array,
    resistance_threshold: jax.Array,
    resistance_style: jax.Array,
    dampen_y: jax.Array,
    server_ticks_per_second: jax.Array,
    params: CombatParams,
) -> jax.Array:
    """Apply one native-compatible post-translation velocity damping step.

    Leading dimensions are broadcastable, so the same primitive serves live
    entity state and vectorized prospective collision corridors.
    """

    force_velocity = resistance_style == APPLIED_RESISTANCE_STYLE_FORCE_VELOCITY
    speed_squared = jnp.sum(velocity * velocity, axis=-1)
    threshold_squared = jnp.maximum(
        resistance_threshold**2,
        jnp.finfo(jnp.float32).tiny,
    )
    linear_blend = jnp.clip(
        jnp.sqrt(speed_squared) / jnp.sqrt(threshold_squared),
        0.0,
        1.0,
    )
    exponential_blend = jnp.clip(
        speed_squared / threshold_squared,
        0.0,
        1.0,
    )
    blend = jnp.where(resistance_style == 1, exponential_blend, linear_blend)
    configured_minimum = jnp.where(
        grounded,
        ground_resistance,
        air_resistance,
    )
    configured_maximum = jnp.where(
        grounded,
        ground_resistance_max,
        air_resistance_max,
    )
    configured_resistance = (
        configured_minimum * blend + configured_maximum * (1.0 - blend)
    )
    horizontal_speed = jnp.linalg.norm(velocity[..., (0, 2)], axis=-1)
    air_blend = jnp.clip(
        (horizontal_speed - params.agent_force_air_drag_min_speed)
        / jnp.maximum(
            params.agent_force_air_drag_max_speed
            - params.agent_force_air_drag_min_speed,
            jnp.finfo(jnp.float32).tiny,
        ),
        0.0,
        1.0,
    )
    force_air_drag = (
        params.agent_force_air_drag_min * (1.0 - air_blend)
        + params.agent_force_air_drag_max * air_blend
    )
    force_resistance = jnp.where(
        grounded,
        params.agent_force_ground_drag_base,
        force_air_drag,
    )
    resistance = jnp.where(
        force_velocity,
        force_resistance,
        configured_resistance,
    )
    reference_ticks = jnp.where(
        force_velocity,
        params.agent_force_reference_ticks_per_second,
        jnp.float32(60.0),
    )
    resistance_scale = resistance ** (reference_ticks / server_ticks_per_second)
    damped = velocity.at[..., 0].set(velocity[..., 0] * resistance_scale)
    damped = damped.at[..., 2].set(velocity[..., 2] * resistance_scale)
    damped = damped.at[..., 1].set(
        jnp.where(
            dampen_y,
            velocity[..., 1] * resistance_scale,
            velocity[..., 1],
        )
    )
    damped = jnp.where(
        force_velocity[..., None]
        & (jnp.abs(damped) <= params.agent_force_per_axis_deadzone),
        jnp.float32(0.0),
        damped,
    )
    retained = jnp.where(
        force_velocity,
        jnp.any(damped != jnp.float32(0.0), axis=-1),
        jnp.sum(damped * damped, axis=-1)
        >= jnp.float32(DODGE_VELOCITY_REMOVAL_SQUARED),
    )
    return jnp.where(retained[..., None], damped, jnp.float32(0.0))


def apply_defense_commands(
    state: CombatMechanicsState,
    commands: DefenseCommands,
    yaw_degrees: jax.Array,
    alive: jax.Array,
    rules: CombatMechanicsRules,
    params: CombatParams,
) -> tuple[CombatMechanicsState, DefenseCommandInfo]:
    """Admit defense inputs without executing their queued native chains."""

    del params
    requested = commands.dodge_direction != DODGE_NONE
    authored = (commands.dodge_direction == DODGE_LEFT) | (
        commands.dodge_direction == DODGE_RIGHT
    )
    profile_valid = (
        rules.dodge_execution_profile
        == DODGE_EXECUTION_NATIVE_NPC_NULL_CONFIG
    ) | (
        rules.dodge_execution_profile
        == DODGE_EXECUTION_AUTHORED_CLIENT_CONFIGURED
    )
    invalid = (
        (commands.dodge_direction < 0)
        | (commands.dodge_direction >= DODGE_DIRECTION_COUNT)
        | ~jnp.isfinite(yaw_degrees)
        | ~profile_valid
    )
    unsupported = requested & authored & ~commands.dodge_corridor_clear
    bits = _set_failure(
        state.failure_bits,
        jnp.any(invalid, axis=1),
        MECHANICS_FAILURE_INVALID_COMMAND,
    )
    bits = _set_failure(
        bits,
        jnp.any(unsupported, axis=1),
        MECHANICS_FAILURE_UNSUPPORTED_DODGE_COLLISION,
    )
    row_valid = bits == jnp.uint32(0)
    flags, _ = status_modifiers(state.statuses)
    movement_enabled = (flags & jnp.uint32(STATUS_FLAG_DISABLE_MOVEMENT)) == 0
    stamina = state.resources[..., RESOURCE_STAMINA]
    guard_supported = (rules.guard_stamina_value > 0.0) & guard_resource_available(
        state.resources,
        rules,
    )

    dodge_pending = state.dodge_operation_tick >= jnp.int32(0)
    can_dodge = (
        requested
        & authored
        & commands.dodge_corridor_clear
        & alive
        & movement_enabled
        & (stamina >= rules.dodge_admission_cost)
        & (state.dodge_cooldown_remaining_seconds <= jnp.float32(0.0))
        & ~dodge_pending
        & row_valid[:, None]
    )

    guard_pending = state.guard_operation_tick >= jnp.int32(0)
    guard_rising = commands.guard_held & ~state.guard_held
    guard_falling = ~commands.guard_held & state.guard_held
    can_prepare_guard = (
        guard_rising
        & ~state.guard_active
        & ~guard_pending
        & ~state.stamina_broken
        & alive
        & movement_enabled
        & guard_supported
        & (stamina >= rules.guard_entry_cost)
        & row_valid[:, None]
    )
    can_hold_guard = (
        commands.guard_held
        & (state.guard_active | guard_pending)
        & ~state.stamina_broken
        & alive
        & movement_enabled
        & guard_supported
        & row_valid[:, None]
    )
    guard_held = can_prepare_guard | can_hold_guard
    release_queued = guard_falling & (state.guard_active | guard_pending)
    guard_operation_tick = jnp.where(
        can_prepare_guard | release_queued,
        jnp.int32(0),
        state.guard_operation_tick,
    )
    guard_operation_tick = jnp.where(
        ~guard_held & ~state.guard_active & ~release_queued,
        jnp.int32(-1),
        guard_operation_tick,
    )
    candidate = state._replace(
        guard_held=guard_held,
        guard_operation_tick=guard_operation_tick,
        guard_windup_elapsed_seconds=jnp.where(
            can_prepare_guard,
            jnp.float32(0.0),
            state.guard_windup_elapsed_seconds,
        ),
        dodge_operation_tick=jnp.where(
            can_dodge,
            jnp.int32(0),
            state.dodge_operation_tick,
        ),
        dodge_pending_direction=jnp.where(
            can_dodge,
            commands.dodge_direction,
            state.dodge_pending_direction,
        ),
        failure_bits=bits,
    )
    result = _select_tree(row_valid, candidate, state)._replace(failure_bits=bits)
    return result, DefenseCommandInfo(
        guard_started=can_prepare_guard & row_valid[:, None],
        guard_ended=jnp.zeros_like(can_prepare_guard),
        dodge_requested=requested & row_valid[:, None],
        dodge_accepted=can_dodge & row_valid[:, None],
    )


def advance_defense_interactions(
    state: CombatMechanicsState,
    yaw_degrees: jax.Array,
    alive: jax.Array,
    dt_seconds: jax.Array,
    rules: CombatMechanicsRules,
    params: CombatParams,
) -> CombatMechanicsState:
    """Advance queued defense roots after the native stat-regeneration stage."""

    invalid_rules = (
        ~jnp.isfinite(rules.guard_entry_cost)
        | (rules.guard_entry_cost < jnp.float32(0.0))
        | ~jnp.isfinite(rules.guard_entry_delay_seconds)
        | (rules.guard_entry_delay_seconds < jnp.float32(0.0))
        | (rules.guard_entry_cost_tick < jnp.int32(1))
        | (rules.guard_activation_tick < rules.guard_entry_cost_tick)
        | (rules.guard_release_tick < jnp.int32(1))
        | ~jnp.isfinite(rules.dodge_admission_cost)
        | (rules.dodge_admission_cost < jnp.float32(0.0))
        | ~jnp.isfinite(rules.dodge_spend_cost)
        | (rules.dodge_spend_cost < jnp.float32(0.0))
        | (rules.dodge_launch_tick < jnp.int32(1))
        | (rules.dodge_cost_tick <= rules.dodge_launch_tick)
        | ~jnp.isfinite(rules.server_ticks_per_second)
        | (rules.server_ticks_per_second <= jnp.float32(0.0))
    )
    bits = _set_failure(
        state.failure_bits,
        jnp.any(invalid_rules, axis=1),
        MECHANICS_FAILURE_INVALID_STATE,
    )
    row_valid = bits == jnp.uint32(0)
    dt = _entity_dt(
        dt_seconds,
        state.resources.shape[0],
        state.resources.shape[1],
    )
    guard_pending = state.guard_operation_tick >= jnp.int32(0)
    next_guard_tick = jnp.where(
        guard_pending,
        state.guard_operation_tick + jnp.int32(1),
        state.guard_operation_tick,
    )
    guard_entry = guard_pending & state.guard_held & ~state.guard_active & alive
    guard_release = guard_pending & ~state.guard_held & state.guard_active
    guard_cancel = guard_pending & ~state.guard_held & ~state.guard_active
    guard_cost_due = guard_entry & (
        next_guard_tick == rules.guard_entry_cost_tick
    )
    safe_ticks_per_second = jnp.where(
        jnp.isfinite(rules.server_ticks_per_second)
        & (rules.server_ticks_per_second > jnp.float32(0.0)),
        rules.server_ticks_per_second,
        jnp.float32(1.0),
    )
    safe_guard_delay = jnp.where(
        jnp.isfinite(rules.guard_entry_delay_seconds)
        & (rules.guard_entry_delay_seconds >= jnp.float32(0.0)),
        rules.guard_entry_delay_seconds,
        jnp.float32(0.0),
    )
    authored_delay_ticks = jnp.ceil(
        safe_guard_delay * safe_ticks_per_second
    ).astype(jnp.int32)
    guard_activate_due = guard_entry & (
        next_guard_tick >= rules.guard_activation_tick + authored_delay_ticks
    )
    guard_release_due = guard_release & (
        next_guard_tick >= rules.guard_release_tick
    )
    guard_cancel_due = guard_cancel & (
        next_guard_tick >= rules.guard_release_tick
    )

    dodge_pending = state.dodge_operation_tick >= jnp.int32(0)
    next_dodge_tick = jnp.where(
        dodge_pending,
        state.dodge_operation_tick + jnp.int32(1),
        state.dodge_operation_tick,
    )
    dodge_launch = dodge_pending & alive & (
        next_dodge_tick == rules.dodge_launch_tick
    )
    dodge_cost_due = dodge_pending & (
        next_dodge_tick == rules.dodge_cost_tick
    )

    total_cost = (
        guard_cost_due.astype(jnp.float32) * rules.guard_entry_cost
        + dodge_cost_due.astype(jnp.float32) * rules.dodge_spend_cost
    )
    stamina = jnp.maximum(
        rules.resource_minimum[..., RESOURCE_STAMINA],
        state.resources[..., RESOURCE_STAMINA] - total_cost,
    )
    resources = state.resources.at[..., RESOURCE_STAMINA].set(stamina)
    newly_broken = (total_cost > jnp.float32(0.0)) & (
        stamina <= jnp.float32(0.0)
    )
    broken = state.stamina_broken | newly_broken
    guard_active = (
        (state.guard_active | guard_activate_due)
        & ~guard_release_due
        & ~broken
    )
    guard_held = state.guard_held & ~newly_broken
    guard_finished = guard_activate_due | guard_release_due | guard_cancel_due
    guard_operation_tick = jnp.where(
        guard_finished,
        jnp.int32(-1),
        next_guard_tick,
    )
    guard_windup = jnp.where(
        guard_entry & ~guard_activate_due,
        state.guard_windup_elapsed_seconds + dt,
        jnp.float32(0.0),
    )
    delay = state.stamina_regen_delay_seconds
    delay = jnp.where(
        guard_release_due,
        jnp.minimum(delay, rules.guard_exit_regen_delay_seconds),
        delay,
    )
    delay = jnp.where(
        dodge_cost_due,
        jnp.minimum(delay, rules.dodge_regen_delay_seconds),
        delay,
    )
    delay = jnp.where(
        newly_broken,
        jnp.minimum(
            delay,
            jnp.float32(STAMINA_BROKEN_REGEN_DELAY_SECONDS),
        ),
        delay,
    )

    local = _dodge_local_direction(state.dodge_pending_direction)
    radians = jnp.deg2rad(yaw_degrees)
    world_x = local[..., 0] * jnp.cos(radians) + local[..., 2] * jnp.sin(radians)
    world_z = -local[..., 0] * jnp.sin(radians) + local[..., 2] * jnp.cos(radians)
    authored_velocity = (
        jnp.stack((world_x, jnp.zeros_like(world_x), world_z), axis=2)
        * rules.dodge_force[..., None]
    )
    launched_velocity = resolve_dodge_launch_velocity(
        authored_velocity,
        rules.dodge_execution_profile,
        params,
    )
    native_profile = (
        rules.dodge_execution_profile
        == DODGE_EXECUTION_NATIVE_NPC_NULL_CONFIG
    )
    launch_air_resistance = jnp.where(
        native_profile,
        jnp.broadcast_to(params.agent_force_air_drag_min, native_profile.shape),
        rules.dodge_air_resistance,
    )
    launch_air_resistance_max = jnp.where(
        native_profile,
        jnp.broadcast_to(params.agent_force_air_drag_max, native_profile.shape),
        rules.dodge_air_resistance_max,
    )
    launch_ground_resistance = jnp.where(
        native_profile,
        jnp.broadcast_to(params.agent_force_ground_drag_base, native_profile.shape),
        rules.dodge_ground_resistance,
    )
    launch_ground_resistance_max = jnp.where(
        native_profile,
        jnp.broadcast_to(params.agent_force_ground_drag_base, native_profile.shape),
        rules.dodge_ground_resistance_max,
    )
    launch_resistance_threshold = jnp.where(
        native_profile,
        jnp.broadcast_to(params.agent_force_air_drag_max_speed, native_profile.shape),
        rules.dodge_resistance_threshold,
    )
    launch_resistance_style = jnp.where(
        native_profile,
        jnp.int32(APPLIED_RESISTANCE_STYLE_FORCE_VELOCITY),
        jnp.int32(1),
    )
    dodge_finished = dodge_cost_due | ~alive
    candidate = state._replace(
        resources=resources,
        stamina_regen_delay_seconds=delay,
        stamina_broken=broken,
        guard_held=guard_held,
        guard_active=guard_active,
        guard_windup_elapsed_seconds=guard_windup,
        guard_operation_tick=guard_operation_tick,
        applied_velocity=jnp.where(
            dodge_launch[..., None],
            launched_velocity,
            state.applied_velocity,
        ),
        external_velocity_y=jnp.where(
            dodge_launch & native_profile,
            jnp.float32(0.0),
            state.external_velocity_y,
        ),
        applied_velocity_can_clear=jnp.where(
            dodge_launch,
            False,
            state.applied_velocity_can_clear,
        ),
        applied_air_resistance=jnp.where(
            dodge_launch,
            launch_air_resistance,
            state.applied_air_resistance,
        ),
        applied_air_resistance_max=jnp.where(
            dodge_launch,
            launch_air_resistance_max,
            state.applied_air_resistance_max,
        ),
        applied_ground_resistance=jnp.where(
            dodge_launch,
            launch_ground_resistance,
            state.applied_ground_resistance,
        ),
        applied_ground_resistance_max=jnp.where(
            dodge_launch,
            launch_ground_resistance_max,
            state.applied_ground_resistance_max,
        ),
        applied_resistance_threshold=jnp.where(
            dodge_launch,
            launch_resistance_threshold,
            state.applied_resistance_threshold,
        ),
        applied_resistance_style=jnp.where(
            dodge_launch,
            launch_resistance_style,
            state.applied_resistance_style,
        ),
        applied_dampen_y=jnp.where(
            dodge_launch,
            False,
            state.applied_dampen_y,
        ),
        dodge_invulnerability_remaining_seconds=jnp.where(
            dodge_launch,
            rules.dodge_invulnerability_seconds + dt,
            state.dodge_invulnerability_remaining_seconds,
        ),
        dodge_cooldown_remaining_seconds=jnp.where(
            dodge_launch,
            rules.dodge_cooldown_seconds + dt,
            state.dodge_cooldown_remaining_seconds,
        ),
        dodge_operation_tick=jnp.where(
            dodge_finished,
            jnp.int32(-1),
            next_dodge_tick,
        ),
        dodge_pending_direction=jnp.where(
            dodge_finished,
            jnp.int32(DODGE_NONE),
            state.dodge_pending_direction,
        ),
        failure_bits=bits,
    )
    return _select_tree(row_valid, candidate, state)._replace(failure_bits=bits)


def damp_applied_motion(
    state: CombatMechanicsState,
    grounded: jax.Array,
    dt_seconds: jax.Array,
    rules: CombatMechanicsRules,
    params: CombatParams,
) -> CombatMechanicsState:
    """Apply resistance after collision has established the new ground state."""

    dt = _entity_dt(
        dt_seconds,
        state.applied_velocity.shape[0],
        state.applied_velocity.shape[1],
    )
    damped = damp_applied_velocity(
        state.applied_velocity,
        grounded,
        state.applied_air_resistance,
        state.applied_air_resistance_max,
        state.applied_ground_resistance,
        state.applied_ground_resistance_max,
        state.applied_resistance_threshold,
        state.applied_resistance_style,
        state.applied_dampen_y,
        rules.server_ticks_per_second,
        params,
    )
    remaining = jnp.maximum(
        jnp.float32(0.0),
        state.dodge_invulnerability_remaining_seconds - dt,
    )
    cooldown_remaining = jnp.maximum(
        jnp.float32(0.0),
        state.dodge_cooldown_remaining_seconds - dt,
    )
    row_valid = state.failure_bits == jnp.uint32(0)
    candidate = state._replace(
        applied_velocity=damped,
        dodge_invulnerability_remaining_seconds=remaining,
        dodge_cooldown_remaining_seconds=cooldown_remaining,
    )
    return _select_tree(row_valid, candidate, state)


def translate_applied_motion(
    state: CombatMechanicsState,
    position: jax.Array,
    grounded: jax.Array,
    dt_seconds: jax.Array,
    params: CombatParams,
    *,
    controlled_entity_index: int = 0,
) -> tuple[CombatMechanicsState, jax.Array]:
    """Translate applied force before post-move resistance is selected.

    ``MotionControllerWalk`` keeps configured applied Y velocity separate from
    the gravity-driven legacy external Y velocity, while a null-config force
    lives wholly in the legacy channel.  All simulated actors use the same NPC
    controller boundary, including the policy-controlled entity.
    """

    dt = _entity_dt(dt_seconds, position.shape[0], position.shape[1])
    velocity = state.applied_velocity
    force_velocity = (
        state.applied_resistance_style
        == APPLIED_RESISTANCE_STYLE_FORCE_VELOCITY
    )
    # Every actor in the native comparison is an NPC motion-controller actor,
    # including the policy-controlled entity.  Keep the argument for API
    # compatibility, but do not give entity zero a different force model.
    # MotionControllerWalk owns the external vertical channel for every NPC.
    del controlled_entity_index
    configured_velocity = ~force_velocity
    external_y = state.external_velocity_y
    can_clear = state.applied_velocity_can_clear | (
        configured_velocity
        & (
            (velocity[..., 1] + external_y <= jnp.float32(0.0))
            | (velocity[..., 1] < jnp.float32(0.0))
        )
    )
    clear_configured_y = (
        configured_velocity
        & grounded
        & can_clear
    )
    prepared_velocity = velocity.at[..., 1].set(
        jnp.where(clear_configured_y, jnp.float32(0.0), velocity[..., 1])
    )
    configured_translation_y = prepared_velocity[..., 1] + external_y
    clear_external_fall = (
        configured_velocity
        & grounded
        & (external_y < jnp.float32(0.0))
        & (configured_translation_y <= jnp.float32(0.0))
    )
    clear_force_fall = (
        force_velocity
        & grounded
        & (prepared_velocity[..., 1] < jnp.float32(0.0))
    )
    translation_y = jnp.where(
        configured_velocity,
        jnp.where(
            clear_external_fall,
            jnp.float32(0.0),
            configured_translation_y,
        ),
        jnp.where(
            clear_force_fall,
            jnp.float32(0.0),
            prepared_velocity[..., 1],
        ),
    )
    translation_velocity = prepared_velocity.at[..., 1].set(translation_y)
    force_active = (
        jnp.any(prepared_velocity != jnp.float32(0.0), axis=2)
        | (configured_velocity & (external_y != jnp.float32(0.0)))
    )
    moving = jnp.any(
        translation_velocity != jnp.float32(0.0),
        axis=2,
    )
    candidate_position = position + jnp.where(
        moving[..., None],
        translation_velocity * dt[..., None],
        jnp.float32(0.0),
    )

    gravity_step = params.world_gravity * dt
    next_external_y = jnp.where(
        configured_velocity & force_active,
        jnp.where(
            clear_external_fall,
            jnp.float32(0.0),
            external_y - gravity_step,
        ),
        external_y,
    )
    next_force_y = jnp.where(
        clear_force_fall,
        jnp.float32(0.0),
        prepared_velocity[..., 1] - gravity_step,
    )
    prepared_velocity = prepared_velocity.at[..., 1].set(
        jnp.where(
            force_velocity & force_active,
            next_force_y,
            prepared_velocity[..., 1],
        )
    )
    row_valid = state.failure_bits == jnp.uint32(0)
    candidate = state._replace(
        applied_velocity=prepared_velocity,
        external_velocity_y=next_external_y,
        applied_velocity_can_clear=can_clear,
    )
    return (
        _select_tree(row_valid, candidate, state),
        jnp.where(row_valid[:, None, None], candidate_position, position),
    )


def project_applied_motion_velocity(
    state: CombatMechanicsState,
    *,
    controlled_entity_index: int = 0,
) -> jax.Array:
    """Return the engine-visible velocity used for this force translation.

    A configured split velocity retains its authored Y component while Walk's
    legacy external Y channel accumulates gravity. Native ``Velocity`` exposes
    their sum for the current translation. ``MotionControllerWalk`` builds
    that translation before advancing gravity for the next tick
    (``MotionControllerWalk.java:1026-1046``). A null-config
    ``forceVelocity`` already lives wholly in the legacy vector, so adding the
    separate configured-path external value there would double-count it.

    Call this immediately before :func:`translate_applied_motion`.
    """

    force_velocity = (
        state.applied_resistance_style
        == APPLIED_RESISTANCE_STYLE_FORCE_VELOCITY
    )
    # Kept for callers compiled against the prior helper signature.  Native
    # MotionControllerWalk does not exempt the policy actor from its legacy
    # external-velocity channel.
    del controlled_entity_index
    configured_external_y = ~force_velocity
    return state.applied_velocity.at[..., 1].set(
        state.applied_velocity[..., 1]
        + jnp.where(
            configured_external_y,
            state.external_velocity_y,
            jnp.float32(0.0),
        )
    )


def resolve_applied_motion_velocity(
    requested_velocity: jax.Array,
    position: jax.Array,
    resolved_position: jax.Array,
    dt_seconds: jax.Array,
) -> jax.Array:
    """Project the engine ``Velocity`` after movement constraints resolve.

    Native publishes the displacement that physics actually accepted. This
    differs from the requested applied-force vector when Walk clears a
    downward force on a grounded actor or when collision clips an axis. Keep
    untouched axes on the direct force projection so float32 subtraction at
    large world coordinates does not manufacture velocity error.
    """

    dt = _entity_dt(
        dt_seconds,
        position.shape[0],
        position.shape[1],
    )
    requested_position = position + requested_velocity * dt[..., None]
    constrained_axis = resolved_position != requested_position
    safe_dt = jnp.maximum(dt, jnp.finfo(jnp.float32).tiny)
    resolved_velocity = (resolved_position - position) / safe_dt[..., None]
    return jnp.where(
        constrained_axis,
        resolved_velocity,
        requested_velocity,
    )
