"""Fixed-shape state and 0.5.7 rules for the compiled combat environment."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.world.entities.movement_states import NpcWalkMovementState
from hytalegym.rulesets import load_combat_ruleset


Array = jax.Array

AGENT_ENTITY = 0
TARGET_ENTITY = 1
DEFAULT_ENTITY_COUNT = 2
# Compatibility name for the certified default contract. Runtime code derives
# the active entity axis from arrays instead of treating this as a maximum.
ENTITY_COUNT = DEFAULT_ENTITY_COUNT

ACTION_FORWARD = 0
ACTION_BACK = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_JUMP = 4
ACTION_ATTACK = 5
ACTION_YAW_DELTA = 6
ACTION_PITCH_DELTA = 7
#: Which gait the translation channels are asking for. The compass channels say
#: *where*, this says *how fast*: idle/walk/run/sprint/sneak, matching
#: ``ARSENAL_GAIT_*``. Carried separately because the direction channels are
#: normalised to a unit vector, which discards magnitude.
ACTION_GAIT = 8
#: Body yaw, steered independently of the head. `MotionControllerBase`
#: `calculateYaw` 604-608 takes an explicit body steer and only falls back to
#: the travel heading when none is given, then advances the head on its own
#: ceiling and bounds it into an authored window around this. One shared yaw
#: cannot express "walk north while looking east", which is what a strafing
#: actor does constantly. Zero means "no body steer this tick", so an actor
#: that never writes this channel keeps its previous heading.
ACTION_BODY_YAW_DELTA = 9
#: Requested live hotbar slot carried as ``slot + 1``, so ZERO means "no switch
#: this tick" and a neutral all-zeros action never moves the live slot. The bridge
#: already applies this -- `NativeEnvironmentSession` 3831 calls
#: `inventory.setActiveHotbarSlot` -- so only the head and the transport were
#: ever missing. The sim reaches it through `ArsenalCommands.hotbar_slot`.
ACTION_HOTBAR_SLOT = 10
ACTION_SIZE = 11

#: Disjoint locomotion gaits. Canonical here because both the published policy
#: layout and the movement kernel need them, and the kernel must not import the
#: observation layer to find out how fast a sprint is.
GAIT_IDLE = 0
GAIT_WALK = 1
GAIT_RUN = 2
GAIT_SPRINT = 3
GAIT_SNEAK = 4
GAIT_COUNT = 5

ACTIVE_OBSERVATION_SIZE = 24
RAW_COMBAT_STATE_SIZE = 12
MAX_MICROTICKS = 4

# Player locomotion, read from the shipped 0.5.7 assets rather than the NPC
# ruleset. ``Server/GameplayConfigs/Default.json`` binds the player to
# ``Player.MovementConfig = "Default"``, which is
# ``Server/Entity/MovementConfig/Default.json``; the stamina numbers come from
# ``Server/Entity/Stats/Stamina.json`` and ``StaminaRegenDelay.json``.
#
# The NPC ruleset derives its jump from ``sqrt(2 * jump_velocity_gravity_floor *
# jump_height_parameter)`` = 5.099, which apexes at only 0.41 blocks under the
# engine's own gravity of 32 -- far short of a one-block step. A player jumps
# with a flat force instead, so 11.8 apexes at ``11.8**2 / (2 * 32)`` = 2.18
# blocks. Everything here is per *second*; callers scale by ``motion_delta`` so
# the same constants hold at the 30 Hz nominal tick and the 0.045 s loaded tick.
PLAYER_JUMP_FORCE = 11.8
#: Gravity applied while ascending after the jump button is released, which is
#: what makes a Hytale jump variable-height: hold for the full 2.18 blocks, tap
#: for less. Only ever exceeds ``engine.gravity`` (32.0), never falls below it.
PLAYER_VARIABLE_JUMP_FALL_FORCE = 35.0

#: ``BaseSpeed`` and the forward gait multipliers, indexed by ``ARSENAL_GAIT_*``
#: (idle, walk, run, sprint, sneak). Idle is zero so an idle choice cannot creep.
#: Run is the unmultiplied base, which is why the NPC ruleset's flat
#: ``max_speed`` of 5.0 looked like a whole gait system on its own.
PLAYER_BASE_SPEED = 5.5
#: Indexed by ``GAIT_*``. Idle carries the *run* multiplier rather than zero so
#: that a caller which predates the gait channel -- the legacy single-actor
#: step, scripted controllers, imitation replay -- keeps moving at the base
#: speed instead of silently freezing. Idle still produces no motion, because an
#: idle locomotion choice also zeroes the compass channels, and the direction
#: vector is what gets scaled.
#:
#: This is the *forward* column only. Hytale authors a separate multiplier per
#: travel direction, so it is kept as the forward table rather than being the
#: whole story; see ``PLAYER_GAIT_*_SPEED_MULTIPLIERS`` below.
PLAYER_GAIT_SPEED_MULTIPLIERS = (1.0, 0.3, 1.0, 1.273, 0.55)

# Direction-dependent locomotion speed, from the same
# ``Server/Entity/MovementConfig/Default.json`` the forward table above is read
# from. Hytale authors three multipliers per gait -- Forward, Backward and
# Strafe -- and moving off-axis is genuinely slower: a run backwards is 0.65 of
# a run forwards and a strafe is 0.80. Applying the forward number in every
# direction, as this module did until 2026-08-17, made an actor's *facing* free:
# it could travel at full speed in any direction regardless of where it looked,
# which is not how the shipped config behaves.
#
# Idle stays isotropic at the run multiplier for the legacy-caller reason given
# above, so a pre-gait caller is unaffected by direction.
PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS = (1.0, 0.3, 1.0, 1.273, 0.55)
PLAYER_GAIT_BACKWARD_SPEED_MULTIPLIERS = (1.0, 0.3, 0.65, 1.273, 0.4)
PLAYER_GAIT_STRAFE_SPEED_MULTIPLIERS = (1.0, 0.3, 0.8, 1.273, 0.45)

#: Sprint is the one gait with no Backward or Strafe key anywhere in the shipped
#: config -- ``ForwardSprintSpeedMultiplier`` is the only ``*Sprint*`` speed
#: field in ``MovementConfig.java`` and in the installed Default.json. So sprint
#: is modelled the way the asset states it: not a per-axis scale but a *state*
#: that multiplies the whole wish vector, enterable only while travel is
#: forward-dominant. That is why the sprint entries above are 1.273 in all three
#: tables -- they are only ever reached once this gate has passed, and a sprint
#: that fails the gate degrades to a run rather than being refused.
#:
#: INFERRED, and deliberately so: the multipliers are transported to the client
#: (``protocol/MovementSettings``) and applied there. The server jar does not
#: contain the transition physics -- the locomotion source audit already records
#: this boundary as ``transition_physics_location: not_present_in_server_jar``
#: and ``native_npc_oracle: cannot_certify_player_sprint_transition``. The
#: *values* are authored; this decomposition rule is not certifiable from the
#: server and is the nearest reading that spends every authored number and
#: invents none.
PLAYER_SPRINT_REQUIRES_FORWARD_DOMINANT_TRAVEL = True

# Airborne motion, from the same
# ``Server/Entity/MovementConfig/Default.json`` that supplies the jump force and
# the gait tables above. Until 2026-08-18 this module read only the ground half
# of that file: the learner jumped with the *player's* 11.8 force and then flew
# with the *NPC* walk controller's airborne rule, which takes its heading from
# the current velocity and never touches ``moveSpeed``
# (``MotionControllerWalk`` line 1169-1209, reached because ``canAct`` requires
# ``onGround``). That combination -- a player's 2.18-block jump with an NPC's
# total absence of air control -- is not any entity the server ships, and it
# made a jump strictly better than a step: measured across 175 archived airborne
# arcs of eight or more ticks, horizontal speed did not decay at all.
#
# ``AirSpeedMultiplier`` is 1.0 and ``ComboAirSpeedMultiplier`` (1.05) applies
# only to combo motion, so neither is carried here.
PLAYER_AIR_DRAG_MIN = 0.96
PLAYER_AIR_DRAG_MAX = 0.995
PLAYER_AIR_DRAG_MIN_SPEED = 6.0
PLAYER_AIR_DRAG_MAX_SPEED = 10.0
#: Air control scales *up* with speed: ``convertToNewRange`` maps the min speed
#: to the min multiplier, so 0 -> 0 and 3 -> 3.13. It is weakest at rest, not
#: strongest, which is the opposite of what the key names suggest on their own.
PLAYER_AIR_CONTROL_MIN_SPEED = 0.0
PLAYER_AIR_CONTROL_MAX_SPEED = 3.0
PLAYER_AIR_CONTROL_MIN_MULTIPLIER = 0.0
PLAYER_AIR_CONTROL_MAX_MULTIPLIER = 3.13
#: INFERRED, on the same footing and for the same reason as
#: ``PLAYER_SPRINT_REQUIRES_FORWARD_DOMINANT_TRAVEL`` above. The *values* are
#: authored, and the drag *form* is sourced exactly -- ``MotionControllerBase``
#: and ``KnockbackPredictionSystems`` both compute
#: ``convertToNewRange(horizontal_speed, min_speed, max_speed, min, max)``, raise
#: it by ``pow(drag, 60 / server_tps)`` and multiply the horizontal components.
#: What is *not* in the server jar is where a player's air control enters the
#: tick: ``airControl*`` has no consumer anywhere in the decompiled tree, only
#: the wire struct (``protocol/MovementSettings``, byte offsets 69/73/77/81),
#: the codec and the defaults, because player air physics runs client-side.
#: Applying the multiplier as a scale on the ground acceleration -- so a wish
#: direction steers the airborne velocity and drag then bleeds it -- is the
#: nearest reading that spends every authored number and invents no new shape.
#: ``AirFrictionMin/Max`` and ``FallMomentumLoss`` are deliberately NOT applied:
#: the friction term is only ever read back through ``friction / (1 - drag)`` in
#: knockback *prediction*, and neither has a legible application point for walk
#: motion. Their absence is a known, bounded gap rather than an oversight.
PLAYER_AIR_CONTROL_APPLICATION_IS_INFERRED = True

# The native bridge controls NPCEntity roles, whose MotionControllerWalk
# movement model differs from the player-only client path above. Keep both
# models explicit so a native NPC fidelity run cannot accidentally certify the
# inferred player recurrence or use the player's base speed in FluidFX.
AGENT_MOTION_PLAYER = 0
AGENT_MOTION_NPC_WALK = 1

# Stamina, from ``Server/Entity/Stats/Stamina.json`` and
# ``StaminaRegenDelay.json``. The assets express these as an amount per 0.1 s
# interval; they are stated per second here and scaled by ``motion_delta``.
# Sprinting spends 0.1 every 0.1 s and suppresses regen; recovery is 0.3 every
# 0.1 s but only once the regen delay has climbed back to zero, and the delay is
# pushed to -0.75 by sprinting (``Plugin.Stamina.SprintRegenDelay``) or to -0.5
# by bottoming out, then recovers at 0.1 every 0.1 s.
PLAYER_STAMINA_MAXIMUM = 10.0
PLAYER_STAMINA_MINIMUM = -4.0
PLAYER_SPRINT_STAMINA_DRAIN_PER_SECOND = 1.0
PLAYER_STAMINA_REGEN_PER_SECOND = 3.0
PLAYER_STAMINA_REGEN_DELAY_RECOVERY_PER_SECOND = 1.0
PLAYER_SPRINT_STAMINA_REGEN_DELAY_SECONDS = 0.75
PLAYER_BROKEN_STAMINA_REGEN_DELAY_SECONDS = 0.5


class CombatParams(NamedTuple):
    """Array-only combat ruleset suitable for use as a JAX PyTree."""

    microticks: Array
    target_active: Array

    nominal_dt: Array
    loaded_dt: Array
    ticks_per_second: Array
    motion_timing_profile_count: Array
    motion_timing_profile_min: Array
    motion_timing_profile_max: Array
    agent_spawn: Array
    target_offset: Array
    agent_max_health: Array
    agent_max_speed: Array
    agent_acceleration: Array
    agent_motion_model: Array
    agent_jump_velocity: Array
    agent_variable_jump_fall_gravity: Array
    #: Fastest speed any gait can reach. Observation normalisation divides by
    #: this rather than by ``agent_max_speed``, which is only the run gait -- a
    #: sprint is 1.273x that and would leave the unit range.
    agent_maximum_gait_speed: Array
    agent_knockback_scale: Array
    agent_movement_velocity_resistance: Array
    agent_min_walk_speed: Array
    agent_min_hit_slowdown: Array
    agent_force_ground_drag_base: Array
    agent_force_air_drag_min: Array
    agent_force_air_drag_max: Array
    agent_force_air_drag_min_speed: Array
    agent_force_air_drag_max_speed: Array
    agent_force_reference_ticks_per_second: Array
    agent_force_per_axis_deadzone: Array
    #: Airborne *walk* motion, distinct from the ``agent_force_air_*`` fields
    #: above: those damp a knockback push, these damp and steer an ordinary
    #: jump. Both read the same authored numbers; they are separate fields
    #: because the server damps only the force path and the player path is
    #: client-side. See ``PLAYER_AIR_*``.
    agent_air_drag_min: Array
    agent_air_drag_max: Array
    agent_air_drag_min_speed: Array
    agent_air_drag_max_speed: Array
    agent_air_control_min_speed: Array
    agent_air_control_max_speed: Array
    agent_air_control_min_multiplier: Array
    agent_air_control_max_multiplier: Array
    world_gravity: Array
    floor_y: Array
    landing_velocity_scale: Array
    agent_turn_speed_degrees: Array
    #: Head steering, separate from the body above. `MotionControllerBase`
    #: builds `maxBodyRotation` and `maxHeadRotation` from two different
    #: ceilings and clamps the head into an authored window around the body,
    #: so the agent needs its own head rotation budget and yaw/pitch limits.
    #: Mirrors the `target_head_*` block further down.
    agent_max_head_rotation_degrees: Array
    #: The bridge applies this to BOTH steerings
    #: (`NativeEnvironmentSession` 3958 and 3966), so the head earns the same
    #: 1.5x the body already has folded into `agent_turn_speed_degrees`.
    agent_steering_relative_turn_speed: Array
    agent_head_yaw_min_degrees: Array
    agent_head_yaw_max_degrees: Array
    agent_head_pitch_min_degrees: Array
    agent_head_pitch_max_degrees: Array
    agent_walk_gravity: Array
    agent_walk_fall_acceleration_multiplier: Array
    agent_walk_gravity_drag_exponent: Array
    agent_walk_max_fall_speed: Array
    agent_walk_max_sink_speed_fluid: Array
    agent_walk_max_climb_height: Array
    agent_walk_max_drop_height: Array
    agent_damage: Array
    agent_attack_pause_min_seconds: Array
    agent_attack_pause_max_seconds: Array

    target_max_health: Array
    target_max_speed: Array
    target_chase_speed: Array
    target_acceleration: Array
    target_initial_distance: Array
    target_steering_slowdown_falloff: Array
    horizontal_selector_pi: Array
    legacy_horizontal_knockback_scale: Array
    legacy_motion_controller_horizontal_factor: Array
    target_chase_stop_distance: Array
    target_chase_slowdown_distance: Array
    target_chase_view_range: Array
    target_chase_hearing_range: Array
    target_chase_hearing_suppressed_by_crouching: Array
    target_maintain_activation_range: Array
    target_maintain_desired_distance_min: Array
    target_maintain_desired_distance_max: Array
    target_maintain_move_threshold: Array
    target_maintain_target_distance_factor: Array
    target_maintain_slowdown_distance: Array
    target_maintain_forward_relative_speed: Array
    target_maintain_backward_relative_speed: Array
    target_strafe_duration_min_seconds: Array
    target_strafe_duration_max_seconds: Array
    target_strafe_frequency_min_seconds: Array
    target_strafe_frequency_max_seconds: Array
    target_strafe_yaw_offset_degrees: Array
    target_strafe_translation_offset_degrees: Array
    target_chase_reaction_ticks: Array
    target_turn_speed_degrees: Array
    agent_bounds: Array
    target_bounds: Array
    agent_eye_offset: Array
    target_eye_offset: Array
    target_max_head_rotation_degrees: Array
    target_head_aim_relative_turn_speed: Array
    target_head_default_relative_turn_speed: Array
    target_head_yaw_min_degrees: Array
    target_head_yaw_max_degrees: Array
    target_head_pitch_min_degrees: Array
    target_head_pitch_max_degrees: Array
    target_activation_min_tick: Array
    target_activation_max_tick: Array
    target_decision_delay_ticks: Array
    target_attack_pause_min_seconds: Array
    target_attack_pause_max_seconds: Array
    regen_delay_ticks: Array
    regen_interval_ticks: Array
    regen_fraction: Array
    sensor_range: Array

    agent_hit_delays: Array
    agent_attack_ranges: Array
    agent_half_angles: Array

    target_windup_ticks: Array
    target_sweep_ticks: Array
    target_recovery_ticks: Array
    target_selector_runtime_seconds: Array
    target_start_distances: Array
    target_end_distances: Array
    target_arc_degrees: Array
    target_sweep_directions: Array
    target_yaw_start_offsets: Array
    target_pitch_offsets: Array
    target_roll_offsets: Array
    target_extend_top: Array
    target_extend_bottom: Array
    target_damage: Array
    target_knockback_force: Array
    target_knockback_relative_x: Array
    target_knockback_relative_z: Array
    target_knockback_velocity_y: Array

    target_damage_reward_scale: Array
    agent_damage_reward_scale: Array
    completion_reward: Array
    death_reward: Array
    wire_fixed_point_scale: Array
    vertical_speed_scale: Array
    facing_error_degrees_scale: Array
    head_pitch_degrees_scale: Array
    #: Multiplier on the authored locomotion-stamina regeneration rate. 1.0 is
    #: the shipped Hytale value (`PLAYER_STAMINA_REGEN_PER_SECOND`), so the
    #: default reproduces the game exactly and only a caller that opts in
    #: changes it. Below 1.0 a spent bar takes proportionally longer to refill,
    #: which is how an actor is handicapped without touching authored data.
    #: Declared last and defaulted so no existing construction site changes.
    locomotion_stamina_regen_scale: Array = 1.0


class RewardComponents(NamedTuple):
    """Unweighted per-decision reward facts exposed to consumers."""

    target_damage: Array
    agent_damage: Array
    completion: Array
    death: Array


def compose_reward(
    components: RewardComponents,
    params: CombatParams,
) -> Array:
    """Apply the ruleset objective to raw combat outcomes."""

    return (
        components.target_damage * params.target_damage_reward_scale
        + components.agent_damage * params.agent_damage_reward_scale
        + jnp.where(components.completion, params.completion_reward, 0.0)
        + jnp.where(components.death, params.death_reward, 0.0)
    ).astype(jnp.float32)


def sum_reward_components(
    components: RewardComponents,
    *,
    axis: int = 0,
) -> RewardComponents:
    """Reduce microtick reward facts without weighting them."""

    return RewardComponents(
        target_damage=jnp.sum(
            components.target_damage,
            axis=axis,
            dtype=jnp.float32,
        ),
        agent_damage=jnp.sum(
            components.agent_damage,
            axis=axis,
            dtype=jnp.float32,
        ),
        completion=jnp.any(components.completion, axis=axis),
        death=jnp.any(components.death, axis=axis),
    )


class CombatState(NamedTuple):
    """Struct-of-arrays state; every leaf has batch as its first dimension."""

    position: Array
    velocity: Array
    health: Array
    yaw: Array
    target_head_yaw: Array
    target_head_pitch: Array
    #: The agent's head, tracked apart from its body the way the target's
    #: already is. `yaw[:, AGENT_ENTITY]` and `pitch` are the BODY; these two
    #: are where the agent is looking. `desired_yaw`/`desired_pitch` stay the
    #: commanded camera target that both seams chase at their own rates.
    agent_head_yaw: Array
    agent_head_pitch: Array

    desired_yaw: Array
    #: The commanded BODY heading, chased by `yaw[:, AGENT_ENTITY]` at
    #: `agent_turn_speed_degrees`. `desired_yaw` above is the commanded HEAD
    #: heading; the two are steered by separate action channels so travel and
    #: view can point different ways, and the head is bounded into
    #: `agent_head_yaw_min/max_degrees` around the body afterwards.
    desired_body_yaw: Array
    pitch: Array
    desired_pitch: Array
    desired_velocity: Array
    agent_move_speed: Array
    agent_fall_speed: Array
    agent_fall_start_y: Array

    tick_count: Array
    motion_timing_profile: Array
    last_motion_delta_seconds: Array

    attack_sequence_index: Array
    agent_attack_cooldown_seconds: Array
    agent_hit_delay: Array
    pending_agent_attack_index: Array

    target_attack_sequence_index: Array
    target_ai_activation_tick: Array
    target_attack_index: Array
    last_target_attack_index: Array
    target_attack_elapsed_ticks: Array
    target_attack_cooldown_seconds: Array
    target_next_attack_queue_tick: Array
    target_attack_queued: Array
    target_attack_hit_applied: Array
    target_damage_pending: Array
    pending_knockback_velocity: Array
    agent_force_velocity: Array
    target_damage_applied_this_tick: Array

    target_out_of_range_ticks: Array
    target_maintain_approaching: Array
    target_maintain_moving_away: Array
    target_strafe_delay_seconds: Array
    target_strafe_paused: Array
    target_strafe_direction: Array
    target_last_seen_position: Array
    target_last_seen_valid: Array
    engagement_target_id: Array
    ticks_since_agent_damage: Array
    completion_awarded: Array
    death_penalty_awarded: Array

    vertical_impulse_applied: Array
    agent_applied_vertical_velocity: Array
    grounded_with_residual_velocity: Array
    knockback_control_lock: Array
    agent_grounded: Array
    agent_walk_movement_state: NpcWalkMovementState
    geometry_exhausted: Array
    # Sticky episode telemetry; it is not current attack legality.
    target_navigation_unsupported: Array
    #: Actor zero's locomotion stamina, and the delay before it regenerates.
    #:
    #: Distinct from the guard/ability stamina resource: this is the sprint
    #: budget, the same quantity ``EntityLocomotionState`` carries for entities
    #: 1..N-1. Actor zero needs its own copy because ``multi_actor/runtime.py``
    #: excludes ``AGENT_ENTITY`` from ``tick_entity_policy_locomotion`` -- the
    #: combat engine steps it instead, and until 2026-08-18 that engine had no
    #: stamina at all, so the learner's sprint was free while every other actor
    #: paid for it. Measured on lane 20260818T035105Z-29d65816-u0200-l000: the
    #: stamina channel sat pinned at 10.0 for all 211 steps.
    #:
    #: The delay is negative while regeneration is blocked and climbs back to
    #: zero, matching the entity bank's convention rather than inventing a
    #: second one.
    agent_locomotion_stamina: Array
    agent_locomotion_stamina_regen_delay: Array


class CombatInfo(NamedTuple):
    """Fixed numeric diagnostics returned without Python dictionaries."""

    attack_requested: Array
    attack_accepted: Array
    agent_attack_executing: Array
    target_visible: Array
    target_attack_phase: Array
    target_attack_progress: Array
    target_attack_index: Array
    target_attack_elapsed_ticks: Array
    target_facing_error_degrees: Array
    target_yaw_degrees: Array
    target_head_yaw_degrees: Array
    target_head_pitch_degrees: Array
    target_velocity: Array
    target_distance: Array
    target_health: Array
    tick_count: Array
    motion_timing_profile: Array
    motion_delta_seconds: Array
    geometry_exhausted: Array
    target_navigation_unsupported: Array
    reward_components: RewardComponents


class CombatTrajectory(NamedTuple):
    """Time-major output from one compiled environment rollout."""

    observation: Array
    reward: Array
    done: Array
    info: CombatInfo


def default_combat_params(
    *,
    microticks: int = 4,
    target_active: bool = True,
    motion_model: str = "player",
) -> CombatParams:
    """Return the packaged Kweebec-versus-Brawler ruleset.

    Microticks are bounded to four so one compiled ``lax.scan`` can serve the
    intended two-to-four-tick policy-decision range without recompilation.
    """

    if not 1 <= microticks <= MAX_MICROTICKS:
        raise ValueError(
            f"microticks must be in [1, {MAX_MICROTICKS}], got {microticks}"
        )
    if motion_model not in {"player", "npc_walk"}:
        raise ValueError(
            "motion_model must be 'player' or 'npc_walk'"
        )

    f32 = jnp.float32
    i32 = jnp.int32
    rules = load_combat_ruleset()
    engine = rules["engine"]
    fixture = rules["fixture"]
    agent = rules["agent"]
    target = rules["target"]
    target_chase = target["chase"]
    target_chase_senses = target["chase_senses"]
    target_maintain = target["maintain_distance"]
    regeneration = rules["regeneration"]
    reward = rules["reward"]
    normalization = rules["observation_normalization"]
    agent_attacks = agent["attacks"]
    target_attacks = target["attacks"]
    knockback = rules["damage_interaction"]["knockback"]
    walk = agent["walk_controller"]
    force_damping = walk["force_velocity_damping"]
    target_offset = fixture["target_offset"]
    initial_distance = math.hypot(target_offset[0], target_offset[2])
    sweep_direction = {"LEFT": 0, "RIGHT": 1}
    agent_base_speed = (
        agent["max_speed"] if motion_model == "npc_walk" else PLAYER_BASE_SPEED
    )
    return CombatParams(
        microticks=jnp.asarray(microticks, dtype=i32),
        target_active=jnp.asarray(target_active, dtype=jnp.bool_),
        nominal_dt=jnp.asarray(engine["nominal_delta_seconds"], dtype=f32),
        loaded_dt=jnp.asarray(engine["loaded_delta_seconds"], dtype=f32),
        ticks_per_second=jnp.asarray(engine["ticks_per_second"], dtype=f32),
        motion_timing_profile_count=jnp.asarray(
            engine["motion_timing_profile_count"], dtype=i32
        ),
        # Inclusive reset band.  Keeping it in CombatParams lets fidelity
        # callers pin min == max without changing the calibrated profile
        # catalogue or introducing a second reset implementation.
        motion_timing_profile_min=jnp.asarray(0, dtype=i32),
        motion_timing_profile_max=jnp.asarray(
            engine["motion_timing_profile_count"] - 1,
            dtype=i32,
        ),
        agent_spawn=jnp.asarray(fixture["agent_spawn"], dtype=f32),
        target_offset=jnp.asarray(fixture["target_offset"], dtype=f32),
        agent_max_health=jnp.asarray(agent["max_health"], dtype=f32),
        # The selected role's run speed; gait multipliers scale it. Both the
        # legacy single-actor step and the entity locomotion path read it, so
        # the two stay in step.
        agent_max_speed=jnp.asarray(agent_base_speed, dtype=f32),
        agent_acceleration=jnp.asarray(agent["acceleration"], dtype=f32),
        agent_motion_model=jnp.asarray(
            AGENT_MOTION_NPC_WALK
            if motion_model == "npc_walk"
            else AGENT_MOTION_PLAYER,
            dtype=i32,
        ),
        # The NPC formula this replaces --
        # ``sqrt(2 * jump_velocity_gravity_floor * jump_height_parameter)`` --
        # assumes the ascent runs at the walk controller's gravity floor (10.0),
        # but the ascent is integrated at ``engine.gravity`` (32.0), so it
        # apexed at 0.41 blocks and could not clear a one-block step. The agent
        # is given the player's jump force instead; see PLAYER_JUMP_FORCE.
        agent_jump_velocity=jnp.asarray(PLAYER_JUMP_FORCE, dtype=f32),
        agent_variable_jump_fall_gravity=jnp.asarray(
            PLAYER_VARIABLE_JUMP_FALL_FORCE, dtype=f32
        ),
        agent_maximum_gait_speed=jnp.asarray(
            agent_base_speed * max(PLAYER_GAIT_SPEED_MULTIPLIERS), dtype=f32
        ),
        agent_knockback_scale=jnp.asarray(agent["knockback_scale"], dtype=f32),
        agent_movement_velocity_resistance=jnp.asarray(
            agent["movement_velocity_resistance"], dtype=f32
        ),
        agent_min_walk_speed=jnp.asarray(agent["min_walk_speed"], dtype=f32),
        agent_min_hit_slowdown=jnp.asarray(agent["min_hit_slowdown"], dtype=f32),
        agent_force_ground_drag_base=jnp.asarray(
            force_damping["ground_drag_base"], dtype=f32
        ),
        agent_force_air_drag_min=jnp.asarray(force_damping["air_drag_min"], dtype=f32),
        agent_force_air_drag_max=jnp.asarray(force_damping["air_drag_max"], dtype=f32),
        agent_force_air_drag_min_speed=jnp.asarray(
            force_damping["air_drag_min_speed"], dtype=f32
        ),
        agent_force_air_drag_max_speed=jnp.asarray(
            force_damping["air_drag_max_speed"], dtype=f32
        ),
        agent_force_reference_ticks_per_second=jnp.asarray(
            force_damping["reference_ticks_per_second"], dtype=f32
        ),
        agent_force_per_axis_deadzone=jnp.asarray(
            force_damping["per_axis_deadzone"], dtype=f32
        ),
        # Taken from the player MovementConfig directly, not from the NPC
        # ruleset, for the same reason ``agent_jump_velocity`` is: the learner
        # is a player, and the NPC ruleset has no airborne steering at all.
        agent_air_drag_min=jnp.asarray(PLAYER_AIR_DRAG_MIN, dtype=f32),
        agent_air_drag_max=jnp.asarray(PLAYER_AIR_DRAG_MAX, dtype=f32),
        agent_air_drag_min_speed=jnp.asarray(PLAYER_AIR_DRAG_MIN_SPEED, dtype=f32),
        agent_air_drag_max_speed=jnp.asarray(PLAYER_AIR_DRAG_MAX_SPEED, dtype=f32),
        agent_air_control_min_speed=jnp.asarray(
            PLAYER_AIR_CONTROL_MIN_SPEED, dtype=f32
        ),
        agent_air_control_max_speed=jnp.asarray(
            PLAYER_AIR_CONTROL_MAX_SPEED, dtype=f32
        ),
        agent_air_control_min_multiplier=jnp.asarray(
            PLAYER_AIR_CONTROL_MIN_MULTIPLIER, dtype=f32
        ),
        agent_air_control_max_multiplier=jnp.asarray(
            PLAYER_AIR_CONTROL_MAX_MULTIPLIER, dtype=f32
        ),
        world_gravity=jnp.asarray(engine["gravity"], dtype=f32),
        floor_y=jnp.asarray(fixture["floor_y"], dtype=f32),
        landing_velocity_scale=jnp.asarray(agent["landing_velocity_scale"], dtype=f32),
        agent_turn_speed_degrees=jnp.asarray(
            agent["turn_degrees_per_second"], dtype=f32
        ),
        # Raw engine ceiling; the steering multiplier is applied at the point of
        # use, exactly as the target's head does with its own relative speed.
        # `agent.turn_degrees_per_second` (540) has that multiplier already
        # folded in -- it is 360 x 1.5 -- so do NOT pre-multiply this one too.
        agent_max_head_rotation_degrees=jnp.asarray(
            agent["max_head_rotation_degrees_per_second"], dtype=f32
        ),
        agent_steering_relative_turn_speed=jnp.asarray(
            agent["steering_relative_turn_speed"], dtype=f32
        ),
        agent_head_yaw_min_degrees=jnp.asarray(
            agent["head_yaw_min_degrees"], dtype=f32
        ),
        agent_head_yaw_max_degrees=jnp.asarray(
            agent["head_yaw_max_degrees"], dtype=f32
        ),
        agent_head_pitch_min_degrees=jnp.asarray(
            agent["head_pitch_min_degrees"], dtype=f32
        ),
        agent_head_pitch_max_degrees=jnp.asarray(
            agent["head_pitch_max_degrees"], dtype=f32
        ),
        agent_walk_gravity=jnp.asarray(walk["gravity"], dtype=f32),
        agent_walk_fall_acceleration_multiplier=jnp.asarray(
            walk["fall_acceleration_multiplier"], dtype=f32
        ),
        agent_walk_gravity_drag_exponent=jnp.asarray(
            walk["gravity_drag_exponent"], dtype=f32
        ),
        agent_walk_max_fall_speed=jnp.asarray(walk["max_fall_speed"], dtype=f32),
        agent_walk_max_sink_speed_fluid=jnp.asarray(
            walk["max_sink_speed_fluid"], dtype=f32
        ),
        agent_walk_max_climb_height=jnp.asarray(walk["max_climb_height"], dtype=f32),
        agent_walk_max_drop_height=jnp.asarray(walk["max_drop_height"], dtype=f32),
        agent_damage=jnp.asarray(agent["damage"], dtype=f32),
        agent_attack_pause_min_seconds=jnp.asarray(
            agent["attack_pause_min_seconds"], dtype=f32
        ),
        agent_attack_pause_max_seconds=jnp.asarray(
            agent["attack_pause_max_seconds"], dtype=f32
        ),
        target_max_health=jnp.asarray(target["max_health"], dtype=f32),
        target_max_speed=jnp.asarray(target["asset_max_walk_speed"], dtype=f32),
        target_chase_speed=jnp.asarray(target["chase_speed"], dtype=f32),
        target_acceleration=jnp.asarray(target["acceleration"], dtype=f32),
        target_initial_distance=jnp.asarray(initial_distance, dtype=f32),
        target_steering_slowdown_falloff=jnp.asarray(
            engine["steering_slowdown_falloff"], dtype=f32
        ),
        horizontal_selector_pi=jnp.asarray(engine["horizontal_selector_pi"], dtype=f32),
        legacy_horizontal_knockback_scale=jnp.asarray(
            engine["legacy_horizontal_knockback_scale"], dtype=f32
        ),
        legacy_motion_controller_horizontal_factor=jnp.asarray(
            engine["legacy_motion_controller_horizontal_factor"],
            dtype=f32,
        ),
        target_chase_stop_distance=jnp.asarray(
            target_chase["stop_distance"], dtype=f32
        ),
        target_chase_slowdown_distance=jnp.asarray(
            target_chase["slowdown_distance"], dtype=f32
        ),
        target_chase_view_range=jnp.asarray(
            target_chase_senses["view_range"],
            dtype=f32,
        ),
        target_chase_hearing_range=jnp.asarray(
            target_chase_senses["hearing_range"],
            dtype=f32,
        ),
        target_chase_hearing_suppressed_by_crouching=jnp.asarray(
            target_chase_senses["hearing_suppressed_by_crouching"],
            dtype=jnp.bool_,
        ),
        target_maintain_activation_range=jnp.asarray(
            target_maintain["activation_range"], dtype=f32
        ),
        target_maintain_desired_distance_min=jnp.asarray(
            target_maintain["desired_distance_min"], dtype=f32
        ),
        target_maintain_desired_distance_max=jnp.asarray(
            target_maintain["desired_distance_max"], dtype=f32
        ),
        target_maintain_move_threshold=jnp.asarray(
            target_maintain["move_threshold"], dtype=f32
        ),
        target_maintain_target_distance_factor=jnp.asarray(
            target_maintain["target_distance_factor"], dtype=f32
        ),
        target_maintain_slowdown_distance=jnp.asarray(
            target_maintain["move_towards_slowdown_distance"], dtype=f32
        ),
        target_maintain_forward_relative_speed=jnp.asarray(
            target_maintain["relative_forward_speed"], dtype=f32
        ),
        target_maintain_backward_relative_speed=jnp.asarray(
            target_maintain["relative_backward_speed"], dtype=f32
        ),
        target_strafe_duration_min_seconds=jnp.asarray(
            target_maintain["strafing_duration_min_seconds"], dtype=f32
        ),
        target_strafe_duration_max_seconds=jnp.asarray(
            target_maintain["strafing_duration_max_seconds"], dtype=f32
        ),
        target_strafe_frequency_min_seconds=jnp.asarray(
            target_maintain["strafing_frequency_min_seconds"], dtype=f32
        ),
        target_strafe_frequency_max_seconds=jnp.asarray(
            target_maintain["strafing_frequency_max_seconds"], dtype=f32
        ),
        target_strafe_yaw_offset_degrees=jnp.asarray(
            target_maintain["strafing_yaw_offset_degrees"], dtype=f32
        ),
        target_strafe_translation_offset_degrees=jnp.asarray(
            target_maintain["strafing_translation_offset_degrees"],
            dtype=f32,
        ),
        target_chase_reaction_ticks=jnp.asarray(
            target["chase_reaction_ticks"], dtype=i32
        ),
        target_turn_speed_degrees=jnp.asarray(
            target["turn_degrees_per_second"], dtype=f32
        ),
        agent_bounds=jnp.asarray(agent["bounding_box"], dtype=f32),
        target_bounds=jnp.asarray(target["bounding_box"], dtype=f32),
        agent_eye_offset=jnp.asarray(
            [0.0, agent["effective_eye_height"], 0.0],
            dtype=f32,
        ),
        target_eye_offset=jnp.asarray(
            [0.0, target["effective_eye_height"], 0.0],
            dtype=f32,
        ),
        target_max_head_rotation_degrees=jnp.asarray(
            target["max_head_rotation_degrees_per_second"],
            dtype=f32,
        ),
        target_head_aim_relative_turn_speed=jnp.asarray(
            target["head_aim_relative_turn_speed"],
            dtype=f32,
        ),
        target_head_default_relative_turn_speed=jnp.asarray(
            target["head_default_relative_turn_speed"],
            dtype=f32,
        ),
        target_head_yaw_min_degrees=jnp.asarray(
            target["head_yaw_min_degrees"],
            dtype=f32,
        ),
        target_head_yaw_max_degrees=jnp.asarray(
            target["head_yaw_max_degrees"],
            dtype=f32,
        ),
        target_head_pitch_min_degrees=jnp.asarray(
            target["head_pitch_min_degrees"],
            dtype=f32,
        ),
        target_head_pitch_max_degrees=jnp.asarray(
            target["head_pitch_max_degrees"],
            dtype=f32,
        ),
        target_activation_min_tick=jnp.asarray(
            target["activation_min_tick"], dtype=i32
        ),
        target_activation_max_tick=jnp.asarray(
            target["activation_max_tick"], dtype=i32
        ),
        target_decision_delay_ticks=jnp.asarray(
            target["decision_delay_ticks"], dtype=i32
        ),
        target_attack_pause_min_seconds=jnp.asarray(
            target["attack_pause_min_seconds"], dtype=f32
        ),
        target_attack_pause_max_seconds=jnp.asarray(
            target["attack_pause_max_seconds"], dtype=f32
        ),
        regen_delay_ticks=jnp.asarray(regeneration["delay_ticks"], dtype=i32),
        regen_interval_ticks=jnp.asarray(regeneration["interval_ticks"], dtype=i32),
        regen_fraction=jnp.asarray(regeneration["fraction"], dtype=f32),
        sensor_range=jnp.asarray(target["sensor_range"], dtype=f32),
        agent_hit_delays=jnp.asarray(
            [attack["hit_delay_ticks"] for attack in agent_attacks],
            dtype=i32,
        ),
        agent_attack_ranges=jnp.asarray(
            [attack["range"] for attack in agent_attacks],
            dtype=f32,
        ),
        agent_half_angles=jnp.asarray(
            [attack["half_angle_degrees"] for attack in agent_attacks],
            dtype=f32,
        ),
        target_windup_ticks=jnp.asarray(
            [attack["windup_ticks"] for attack in target_attacks],
            dtype=i32,
        ),
        target_sweep_ticks=jnp.asarray(
            [attack["sweep_ticks"] for attack in target_attacks],
            dtype=i32,
        ),
        target_recovery_ticks=jnp.asarray(
            [attack["recovery_ticks"] for attack in target_attacks],
            dtype=i32,
        ),
        target_selector_runtime_seconds=jnp.asarray(
            [attack["selector_runtime_seconds"] for attack in target_attacks],
            dtype=f32,
        ),
        target_start_distances=jnp.asarray(
            [attack["start_distance"] for attack in target_attacks],
            dtype=f32,
        ),
        target_end_distances=jnp.asarray(
            [attack["end_distance"] for attack in target_attacks],
            dtype=f32,
        ),
        target_arc_degrees=jnp.asarray(
            [attack["arc_degrees"] for attack in target_attacks],
            dtype=f32,
        ),
        target_sweep_directions=jnp.asarray(
            [sweep_direction[attack["sweep_direction"]] for attack in target_attacks],
            dtype=i32,
        ),
        target_yaw_start_offsets=jnp.asarray(
            [attack["yaw_start_offset_degrees"] for attack in target_attacks],
            dtype=f32,
        ),
        target_pitch_offsets=jnp.asarray(
            [attack["pitch_offset_degrees"] for attack in target_attacks],
            dtype=f32,
        ),
        target_roll_offsets=jnp.asarray(
            [attack["roll_offset_degrees"] for attack in target_attacks],
            dtype=f32,
        ),
        target_extend_top=jnp.asarray(
            [attack["extend_top"] for attack in target_attacks],
            dtype=f32,
        ),
        target_extend_bottom=jnp.asarray(
            [attack["extend_bottom"] for attack in target_attacks],
            dtype=f32,
        ),
        target_damage=jnp.asarray(
            [attack["damage"] for attack in target_attacks],
            dtype=f32,
        ),
        target_knockback_force=jnp.asarray(knockback["force"], dtype=f32),
        target_knockback_relative_x=jnp.asarray(knockback["relative_x"], dtype=f32),
        target_knockback_relative_z=jnp.asarray(knockback["relative_z"], dtype=f32),
        target_knockback_velocity_y=jnp.asarray(knockback["velocity_y"], dtype=f32),
        target_damage_reward_scale=jnp.asarray(
            reward["target_damage_scale"], dtype=f32
        ),
        agent_damage_reward_scale=jnp.asarray(reward["agent_damage_scale"], dtype=f32),
        completion_reward=jnp.asarray(reward["completion"], dtype=f32),
        death_reward=jnp.asarray(reward["death"], dtype=f32),
        wire_fixed_point_scale=jnp.asarray(
            normalization["wire_fixed_point_scale"], dtype=f32
        ),
        vertical_speed_scale=jnp.asarray(
            normalization["vertical_speed_scale"], dtype=f32
        ),
        facing_error_degrees_scale=jnp.asarray(
            normalization["facing_error_degrees_scale"], dtype=f32
        ),
        head_pitch_degrees_scale=jnp.asarray(
            normalization["head_pitch_degrees_scale"], dtype=f32
        ),
    )
