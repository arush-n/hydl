"""Leaf helpers extracted verbatim from env.py."""

import jax
import jax.numpy as jnp
from hytalegym.jax.combat.loadout.schema.types import MeleeLoadoutBatch
from hytalegym.jax.combat.types import ACTION_ATTACK, ACTION_BACK, ACTION_BODY_YAW_DELTA, ACTION_FORWARD, ACTION_GAIT, ACTION_JUMP, ACTION_LEFT, ACTION_PITCH_DELTA, ACTION_RIGHT, ACTION_YAW_DELTA, AGENT_ENTITY, GAIT_COUNT, GAIT_RUN, GAIT_SPRINT, PLAYER_BROKEN_STAMINA_REGEN_DELAY_SECONDS, PLAYER_GAIT_BACKWARD_SPEED_MULTIPLIERS, PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS, PLAYER_GAIT_STRAFE_SPEED_MULTIPLIERS, PLAYER_SPRINT_STAMINA_DRAIN_PER_SECOND, PLAYER_SPRINT_STAMINA_REGEN_DELAY_SECONDS, PLAYER_STAMINA_MAXIMUM, PLAYER_STAMINA_MINIMUM, PLAYER_STAMINA_REGEN_DELAY_RECOVERY_PER_SECOND, PLAYER_STAMINA_REGEN_PER_SECOND, TARGET_ENTITY, CombatInfo, CombatParams, CombatState, RewardComponents
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
    TargetNavigationProvider,
    WINDUP,
    _combat_perception_line_of_sight,
    _combat_window_exhausted,
    _target_phase,
    _tick_target_attack,
    _tick_target_motion,
    _tick_target_vertical_motion,
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
from .reset import (  # noqa: F401  (re-exported: callers unchanged)
    ActorEvidence,
    TargetDetection,
    TargetEvidence,
    _encode_actor_observation,
    _finish_atlas_reset,
    _finish_geometry_reset,
    _gather_entity_axis,
    _target_detection_from_positions,
    observe_batch,
    reset_batch,
    reset_batch_geometry,
    reset_batch_geometry_at,
    reset_batch_geometry_atlas,
    reset_batch_geometry_atlas_at,
    reset_batch_region,
    reset_batch_region_at,
    target_evidence,
)


def active_combat_state(
    state: CombatState,
    params: CombatParams,
    geometry: GeometryProvider | None = None,
) -> jax.Array:
    """Return the shared raw twelve-scalar simulator/native combat state."""

    (
        phase,
        progress,
        reported_index,
        elapsed,
    ) = _target_phase(state, params)
    agent_position = state.position[:, AGENT_ENTITY]
    target_position = state.position[:, TARGET_ENTITY]
    target_detection = _target_detection_from_positions(
        state.health[:, TARGET_ENTITY] > 0.0,
        agent_position,
        target_position,
        jnp.linalg.norm(
            (target_position - agent_position)[:, (0, 2)],
            axis=1,
        ),
        params,
        geometry,
    )
    agent_executing = (state.agent_hit_delay > 0) | (
        state.agent_attack_cooldown_seconds > 0.0
    )
    facing_error = _target_facing_error(state)
    return jnp.stack(
        (
            agent_executing.astype(jnp.float32),
            target_detection.target_in_line_of_sight.astype(jnp.float32),
            phase.astype(jnp.float32),
            progress,
            reported_index.astype(jnp.float32),
            elapsed.astype(jnp.float32),
            facing_error,
            state.yaw[:, TARGET_ENTITY],
            state.velocity[:, TARGET_ENTITY, 0],
            state.velocity[:, TARGET_ENTITY, 2],
            state.target_head_yaw,
            state.target_head_pitch,
        ),
        axis=1,
    ).astype(jnp.float32)


def _apply_action(
    state: CombatState,
    actions: jax.Array,
    pause_draw: jax.Array,
    params: CombatParams,
    geometry: GeometryProvider | None = None,
    agent_loadout: MeleeLoadoutBatch | None = None,
) -> tuple[CombatState, jax.Array, jax.Array]:
    forward = jnp.clip(
        actions[:, ACTION_FORWARD].astype(jnp.float32),
        0.0,
        1.0,
    )
    back = jnp.clip(
        actions[:, ACTION_BACK].astype(jnp.float32),
        0.0,
        1.0,
    )
    left = jnp.clip(
        actions[:, ACTION_LEFT].astype(jnp.float32),
        0.0,
        1.0,
    )
    right = jnp.clip(
        actions[:, ACTION_RIGHT].astype(jnp.float32),
        0.0,
        1.0,
    )
    jump = actions[:, ACTION_JUMP] > 0.5
    attack = actions[:, ACTION_ATTACK] > 0.5

    desired_yaw = _normalize_degrees(state.desired_yaw + actions[:, ACTION_YAW_DELTA])
    # The body carries its own commanded heading. `calculateYaw` 604-608 steers
    # the body from an explicit body steer and only falls back to the travel
    # heading when there is none, so these two channels are independent and the
    # head is bounded around the body afterwards rather than dragging it.
    desired_body_yaw = _normalize_degrees(
        state.desired_body_yaw + actions[:, ACTION_BODY_YAW_DELTA]
    )
    desired_pitch = jnp.clip(
        state.desired_pitch + actions[:, ACTION_PITCH_DELTA],
        -90.0,
        90.0,
    )
    longitudinal = forward - back
    lateral = right - left
    length = jnp.hypot(longitudinal, lateral)
    safe_length = jnp.maximum(length, jnp.float32(1.0e-9))
    longitudinal = jnp.where(length > 1.0e-9, longitudinal / safe_length, 0.0)
    lateral = jnp.where(length > 1.0e-9, lateral / safe_length, 0.0)
    # Actor zero's gait. Until this landed, the agent's desired speed was a flat
    # `agent_max_speed` -- which is only the *run* gait, as the field's own
    # comment says -- so its published `locomotion_gait_compass` head changed
    # nothing about how fast it moved. Measured on archived lane
    # 20260818T035105Z-29d65816-u0200-l000: walk, run, sprint and sneak all
    # produced the same 0.2475 blocks/tick, walk exceeding its own cap by 3.3x
    # and sprint never reaching its own. `tick_entity_policy_locomotion` already
    # did this correctly, but `runtime.py` excludes AGENT_ENTITY from it because
    # actor zero is stepped here instead, so the gait table reached every entity
    # except the learner.
    #
    # Hytale authors a different multiplier per travel direction, so the axes are
    # scaled separately after normalisation, exactly as the entity path does.
    # That is what makes facing matter: travelling where you are looking is the
    # fastest way to travel.
    requested_gait = jnp.clip(
        jnp.round(actions[:, ACTION_GAIT]).astype(jnp.int32), 0, GAIT_COUNT - 1
    )
    # Sprint is forward-only *and* stamina-priced. The shipped config authors no
    # Backward or Strafe sprint multiplier at all, and the engine drops an
    # exhausted actor out of its sprint rather than refusing to move, so both
    # failures fall back to a run instead of stopping.
    forward_dominant = longitudinal > jnp.abs(lateral)
    wants_sprint = requested_gait == jnp.int32(GAIT_SPRINT)
    stamina = state.agent_locomotion_stamina
    regen_delay = state.agent_locomotion_stamina_regen_delay
    sprinting = wants_sprint & forward_dominant & (stamina > jnp.float32(0.0))
    gait = jnp.where(wants_sprint & ~sprinting, jnp.int32(GAIT_RUN), requested_gait)
    longitudinal_multiplier = jnp.where(
        longitudinal >= 0.0,
        jnp.asarray(PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS, dtype=jnp.float32)[gait],
        jnp.asarray(PLAYER_GAIT_BACKWARD_SPEED_MULTIPLIERS, dtype=jnp.float32)[gait],
    )
    lateral_multiplier = jnp.asarray(
        PLAYER_GAIT_STRAFE_SPEED_MULTIPLIERS, dtype=jnp.float32
    )[gait]
    longitudinal = longitudinal * longitudinal_multiplier
    lateral = lateral * lateral_multiplier
    # Travel rides the BODY, not the camera. The compass channels are resolved
    # against the commanded body heading so looking away from the direction of
    # travel costs nothing and changes nothing about where the actor goes.
    radians = jnp.deg2rad(desired_body_yaw)
    desired_vx = params.agent_max_speed * (
        longitudinal * -jnp.sin(radians) + lateral * jnp.cos(radians)
    )
    desired_vz = params.agent_max_speed * (
        longitudinal * -jnp.cos(radians) - lateral * jnp.sin(radians)
    )
    desired_velocity = jnp.stack((desired_vx, desired_vz), axis=1)

    # Sprinting spends stamina and blocks regeneration, and pins the regen delay
    # so a brief pause does not immediately refill; bottoming out re-arms a
    # shorter delay, which is the engine's Stamina_Broken behaviour. This
    # mirrors `tick_entity_policy_locomotion` term for term so the learner and
    # every other actor price a sprint identically.
    #
    # The delta is the previous tick's, because this runs before the motion
    # integrator picks the current one. That is the same one-tick lag the C4
    # continuation reads its ground and collision conditions with, and at a
    # steady 0.045 s tick the difference is not observable.
    motion_delta = state.last_motion_delta_seconds
    drained = stamina - jnp.float32(
        PLAYER_SPRINT_STAMINA_DRAIN_PER_SECOND
    ) * motion_delta
    regenerating = (~sprinting) & (regen_delay >= jnp.float32(0.0))
    recovered = stamina + jnp.float32(
        PLAYER_STAMINA_REGEN_PER_SECOND
    ) * motion_delta * params.locomotion_stamina_regen_scale
    next_stamina = jnp.clip(
        jnp.where(sprinting, drained, jnp.where(regenerating, recovered, stamina)),
        PLAYER_STAMINA_MINIMUM,
        PLAYER_STAMINA_MAXIMUM,
    )
    emptied = sprinting & (next_stamina <= jnp.float32(0.0))
    next_regen_delay = jnp.where(
        sprinting,
        jnp.float32(-PLAYER_SPRINT_STAMINA_REGEN_DELAY_SECONDS),
        jnp.minimum(
            regen_delay
            + jnp.float32(PLAYER_STAMINA_REGEN_DELAY_RECOVERY_PER_SECOND)
            * motion_delta,
            jnp.float32(0.0),
        ),
    )
    next_regen_delay = jnp.where(
        emptied,
        jnp.float32(-PLAYER_BROKEN_STAMINA_REGEN_DELAY_SECONDS),
        next_regen_delay,
    )

    grounded_jump = jump & state.agent_grounded
    agent_velocity = state.velocity[:, AGENT_ENTITY]
    agent_velocity = agent_velocity.at[:, 1].set(
        jnp.where(
            grounded_jump,
            params.agent_jump_velocity,
            agent_velocity[:, 1],
        )
    )
    velocity = state.velocity.at[:, AGENT_ENTITY].set(agent_velocity)

    attack_accepted = (
        attack
        & (state.health[:, TARGET_ENTITY] > 0.0)
        & (state.agent_attack_cooldown_seconds <= 0.0)
    )
    if agent_loadout is None:
        accepted_index = jnp.clip(
            state.attack_sequence_index,
            0,
            params.agent_hit_delays.shape[0] - 1,
        )
        accepted_available = jnp.ones_like(attack_accepted)
        accepted_hit_delay = params.agent_hit_delays[accepted_index]
        next_sequence_index = (
            state.attack_sequence_index + 1
        ) % params.agent_hit_delays.shape[0]
    else:
        accepted_index = _loadout_profile_index(
            agent_loadout,
            state.attack_sequence_index,
        )
        accepted_available = _loadout_row_valid(agent_loadout) & (
            _gather_loadout_profile(
                agent_loadout.attack_mask,
                accepted_index,
            )
        )
        accepted_hit_delay = _gather_loadout_profile(
            agent_loadout.hit_delay_ticks,
            accepted_index,
        )
        attack_count = jnp.maximum(
            _loadout_attack_count(agent_loadout),
            jnp.int32(1),
        )
        next_sequence_index = (accepted_index + 1) % attack_count
    attack_accepted = attack_accepted & accepted_available
    pending_index = jnp.where(
        attack_accepted,
        accepted_index,
        state.pending_agent_attack_index,
    )
    hit_delay = jnp.where(
        attack_accepted,
        accepted_hit_delay,
        state.agent_hit_delay,
    )
    sequence_index = jnp.where(
        attack_accepted,
        next_sequence_index,
        state.attack_sequence_index,
    )
    cooldown = jnp.where(
        attack_accepted,
        pause_draw,
        state.agent_attack_cooldown_seconds,
    )

    updated = state._replace(
        desired_yaw=desired_yaw,
        desired_body_yaw=desired_body_yaw,
        desired_pitch=desired_pitch,
        desired_velocity=desired_velocity,
        agent_locomotion_stamina=next_stamina,
        agent_locomotion_stamina_regen_delay=next_regen_delay,
        velocity=velocity,
        vertical_impulse_applied=jnp.where(
            grounded_jump,
            True,
            state.vertical_impulse_applied,
        ),
        agent_applied_vertical_velocity=jnp.where(
            grounded_jump,
            params.agent_jump_velocity,
            state.agent_applied_vertical_velocity,
        ),
        grounded_with_residual_velocity=jnp.where(
            grounded_jump,
            False,
            state.grounded_with_residual_velocity,
        ),
        knockback_control_lock=jnp.where(
            grounded_jump,
            False,
            state.knockback_control_lock,
        ),
        agent_grounded=jnp.where(
            grounded_jump,
            False,
            state.agent_grounded,
        ),
        pending_agent_attack_index=pending_index,
        agent_hit_delay=hit_delay,
        attack_sequence_index=sequence_index,
        agent_attack_cooldown_seconds=cooldown,
    )
    return updated, attack, attack_accepted


def _microtick(
    state: CombatState,
    target_pause_draw: jax.Array,
    target_strafe_frequency_draw: jax.Array,
    target_strafe_duration_draw: jax.Array,
    target_strafe_direction_draw: jax.Array,
    params: CombatParams,
    geometry: GeometryProvider | None = None,
    agent_loadout: MeleeLoadoutBatch | None = None,
    target_navigation_provider: TargetNavigationProvider | None = None,
    motion_delta_override_seconds: jax.Array | None = None,
) -> tuple[CombatState, RewardComponents]:
    tick_count = state.tick_count + 1
    if motion_delta_override_seconds is None:
        motion_delta = _motion_delta(
            tick_count,
            state.motion_timing_profile,
            params,
        )
    else:
        motion_delta = jnp.asarray(
            motion_delta_override_seconds,
            dtype=jnp.float32,
        )
        if motion_delta.shape != tick_count.shape:
            raise ValueError(
                "motion delta override must have one value per environment"
            )
    state = state._replace(
        tick_count=tick_count,
        last_motion_delta_seconds=motion_delta,
    )
    state = _tick_agent_motion(state, motion_delta, params, geometry)
    state = state._replace(
        target_damage_applied_this_tick=jnp.zeros_like(
            state.target_damage_applied_this_tick
        ),
    )
    state, agent_damage = _apply_pending_target_damage(state, params)
    geometry_valid = ~state.geometry_exhausted
    attacked_agent, target_damage = _tick_agent_attack(
        state,
        params,
        geometry,
        agent_loadout,
    )
    state = _select_state(geometry_valid, attacked_agent, state)
    target_damage = jnp.where(
        geometry_valid,
        target_damage,
        jnp.float32(0.0),
    )
    target_active = (
        jnp.broadcast_to(
            params.target_active,
            state.tick_count.shape,
        )
        & ~state.geometry_exhausted
    )
    target_attack_queue_distance = _target_distance(state)
    target_attack_queue_visible = (
        jnp.ones_like(target_active)
        if geometry is None
        else _combat_perception_line_of_sight(state, geometry)
    )
    target_selector_reference = state
    moved_target = _tick_target_motion(
        state,
        motion_delta,
        target_strafe_frequency_draw,
        target_strafe_duration_draw,
        target_strafe_direction_draw,
        params,
        geometry,
        target_navigation_provider,
    )
    moved_target = _select_state(target_active, moved_target, state)
    if geometry is not None:
        target_window_exhausted = _combat_window_exhausted(
            moved_target,
            geometry,
        )
        state = _select_state(
            ~target_window_exhausted,
            moved_target,
            state,
        )._replace(
            geometry_exhausted=(state.geometry_exhausted | target_window_exhausted)
        )
    else:
        state = moved_target
    target_vertical_state = _tick_target_vertical_motion(
        state,
        motion_delta,
        params,
        geometry,
    )
    # Native capture fixtures may intentionally omit target geometry when the
    # target is disabled. Do not let that absent actor fall through the world
    # and turn an otherwise valid learner transition into geometry exhaustion.
    state = _select_state(target_active, target_vertical_state, state)
    attacked_target = _tick_target_attack(
        state,
        target_pause_draw,
        motion_delta,
        target_attack_queue_distance,
        target_attack_queue_visible,
        params,
        geometry,
        target_selector_reference,
    )
    state = _select_state(
        target_active & ~state.geometry_exhausted,
        attacked_target,
        state,
    )

    state = state._replace(
        agent_attack_cooldown_seconds=jnp.maximum(
            jnp.float32(0.0),
            state.agent_attack_cooldown_seconds - motion_delta,
        )
    )
    state = _tick_regeneration(state, params)

    target_dead = state.health[:, TARGET_ENTITY] <= 0.0
    agent_dead = state.health[:, AGENT_ENTITY] <= 0.0
    completion = target_dead & ~state.completion_awarded
    death = agent_dead & ~state.death_penalty_awarded
    components = RewardComponents(
        target_damage=target_damage,
        agent_damage=agent_damage,
        completion=completion,
        death=death,
    )
    state = state._replace(
        completion_awarded=state.completion_awarded | completion,
        death_penalty_awarded=state.death_penalty_awarded | death,
    )
    return state, components


def _apply_pending_target_damage(
    state: CombatState,
    params: CombatParams,
) -> tuple[CombatState, jax.Array]:
    """Apply legacy 24×9 target damage and stage next-tick knockback.

    ``DamageEntityInteraction`` writes the component through its command
    buffer after this tick's motion consumer has run.  The following
    microtick therefore applies the staged force before processing new damage.

    This legacy core has no guard/dodge mechanics state and is not a defensive
    fidelity surface. The policy-reachable Arsenal path routes the same
    pending hit through ``_defend_pending_target_damage`` and the shared
    mechanics resolver. Do not add a second mitigation formula here.
    """

    pending = state.target_damage_pending
    profile_index = jnp.clip(
        state.last_target_attack_index,
        0,
        params.target_damage.shape[0] - 1,
    )
    before = state.health[:, AGENT_ENTITY]
    after = jnp.where(
        pending,
        jnp.maximum(0.0, before - params.target_damage[profile_index]),
        before,
    )
    health = state.health.at[:, AGENT_ENTITY].set(after)
    external_force_velocity = state.pending_knockback_velocity
    updated = state._replace(
        health=health,
        target_damage_pending=jnp.where(pending, False, pending),
        pending_knockback_velocity=jnp.where(
            pending[:, None],
            jnp.zeros_like(state.pending_knockback_velocity),
            state.pending_knockback_velocity,
        ),
        # Native clears ``appliedForce`` after its one-tick retained-walk
        # translation. Only the raw external force survives for later damping.
        agent_force_velocity=jnp.where(
            pending[:, None],
            external_force_velocity,
            state.agent_force_velocity,
        ),
        target_damage_applied_this_tick=pending,
        ticks_since_agent_damage=jnp.where(
            pending,
            0,
            state.ticks_since_agent_damage,
        ),
    )
    return updated, before - after


def _geometry_grounded(
    geometry: GeometryProvider,
    position: jax.Array,
) -> jax.Array:
    return _aabb_grounded_result(geometry, position).grounded


def _tick_regeneration(
    state: CombatState,
    params: CombatParams,
) -> CombatState:
    health = state.health[:, AGENT_ENTITY]
    eligible = (
        (health > 0.0)
        & (health < params.agent_max_health)
        & ~state.target_damage_applied_this_tick
    )
    ticks = jnp.where(
        eligible,
        state.ticks_since_agent_damage + 1,
        state.ticks_since_agent_damage,
    )
    regen_tick = (
        eligible
        & (ticks >= params.regen_delay_ticks)
        & ((ticks - params.regen_delay_ticks) % params.regen_interval_ticks == 0)
    )
    next_health = jnp.where(
        regen_tick,
        jnp.minimum(
            params.agent_max_health,
            health + params.agent_max_health * params.regen_fraction,
        ),
        health,
    )
    all_health = state.health.at[:, AGENT_ENTITY].set(next_health)
    return state._replace(
        health=all_health,
        ticks_since_agent_damage=ticks,
    )


def _combat_info(
    state: CombatState,
    attack_requested: jax.Array,
    attack_accepted: jax.Array,
    params: CombatParams,
    geometry: GeometryProvider | None,
    reward_components: RewardComponents,
) -> CombatInfo:
    raw = active_combat_state(state, params, geometry)
    return CombatInfo(
        attack_requested=attack_requested,
        attack_accepted=attack_accepted,
        agent_attack_executing=raw[:, 0] > 0.5,
        target_visible=raw[:, 1] > 0.5,
        target_attack_phase=raw[:, 2].astype(jnp.int32),
        target_attack_progress=raw[:, 3],
        target_attack_index=raw[:, 4].astype(jnp.int32),
        target_attack_elapsed_ticks=raw[:, 5].astype(jnp.int32),
        target_facing_error_degrees=raw[:, 6],
        target_yaw_degrees=raw[:, 7],
        target_head_yaw_degrees=raw[:, 10],
        target_head_pitch_degrees=raw[:, 11],
        target_velocity=raw[:, 8:10],
        target_distance=_target_distance(state),
        target_health=state.health[:, TARGET_ENTITY],
        tick_count=state.tick_count,
        motion_timing_profile=state.motion_timing_profile,
        motion_delta_seconds=state.last_motion_delta_seconds,
        geometry_exhausted=state.geometry_exhausted,
        target_navigation_unsupported=(state.target_navigation_unsupported),
        reward_components=reward_components,
    )
