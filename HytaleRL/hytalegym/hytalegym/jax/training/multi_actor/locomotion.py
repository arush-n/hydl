"""Entity-indexed policy locomotion through Combat's shipped Walk kernel."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.runtime.agent import _tick_agent_motion
from hytalegym.jax.combat.runtime.math import _normalize_degrees
from hytalegym.jax.combat.arsenal.environment import GeometryProvider
from hytalegym.jax.combat.types import (
    ACTION_BACK,
    ACTION_BODY_YAW_DELTA,
    ACTION_FORWARD,
    ACTION_GAIT,
    ACTION_JUMP,
    ACTION_LEFT,
    ACTION_PITCH_DELTA,
    ACTION_RIGHT,
    ACTION_SIZE,
    ACTION_YAW_DELTA,
    AGENT_ENTITY,
    GAIT_COUNT,
    GAIT_RUN,
    GAIT_SPRINT,
    PLAYER_BROKEN_STAMINA_REGEN_DELAY_SECONDS,
    PLAYER_GAIT_BACKWARD_SPEED_MULTIPLIERS,
    PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS,
    PLAYER_GAIT_STRAFE_SPEED_MULTIPLIERS,
    PLAYER_SPRINT_STAMINA_DRAIN_PER_SECOND,
    PLAYER_SPRINT_STAMINA_REGEN_DELAY_SECONDS,
    PLAYER_STAMINA_MAXIMUM,
    PLAYER_STAMINA_MINIMUM,
    PLAYER_STAMINA_REGEN_DELAY_RECOVERY_PER_SECOND,
    PLAYER_STAMINA_REGEN_PER_SECOND,
    CombatParams,
    CombatState,
)
from hytalegym.jax.world import NpcWalkMovementState, empty_npc_walk_movement_state

from .controls import EntityPolicyControls


class EntityLocomotionState(NamedTuple):
    """Walk-controller memory for every possible policy actor ``[B,N]``."""

    #: The commanded HEAD (camera) heading. `pitch` below is its pitch
    #: counterpart, which is why it is documented as a LOOK pitch throughout.
    desired_yaw: jax.Array
    #: The commanded BODY heading, steered on its own channel. Travel rides
    #: this, not the camera: `calculateYaw` 604-608 takes an explicit body steer
    #: and only falls back to the translation heading when none is given.
    desired_body_yaw: jax.Array
    #: The realized head yaw, bounded into the authored window around the body.
    #: Until this existed the head was re-seeded from the body every tick, so
    #: the two seams could never separate in the training path however the flat
    #: combat kernel behaved.
    head_yaw: jax.Array
    pitch: jax.Array
    desired_pitch: jax.Array
    desired_velocity: jax.Array
    move_speed: jax.Array
    fall_speed: jax.Array
    fall_start_y: jax.Array
    vertical_impulse_applied: jax.Array
    applied_vertical_velocity: jax.Array
    grounded_with_residual_velocity: jax.Array
    knockback_control_lock: jax.Array
    grounded: jax.Array
    force_velocity: jax.Array
    force_pushed: jax.Array
    ticks_since_damage: jax.Array
    geometry_exhausted: jax.Array
    #: Locomotion stamina, distinct from the guard/ability stamina in
    #: ``CombatMechanics``. Sprinting spends it; it only refills once
    #: ``stamina_regen_delay`` has climbed back to zero.
    stamina: jax.Array
    stamina_regen_delay: jax.Array
    walk: NpcWalkMovementState


def initialize_entity_locomotion_state(
    combat: CombatState,
    params: CombatParams,
) -> EntityLocomotionState:
    """Initialize all entity rows while preserving entity-zero state exactly.

    Nonzero rows start from observable reset facts only. Historical controller
    values that do not exist yet are zero/unavailable until that row is stepped.
    """

    batch, entity_count = combat.health.shape
    shape = (batch, entity_count)
    zeros = jnp.zeros(shape, dtype=jnp.float32)
    grounded = (
        (combat.position[..., 1] <= params.floor_y + jnp.float32(1.0e-9))
        & (combat.velocity[..., 1] <= jnp.float32(0.0))
        & (combat.health > jnp.float32(0.0))
    )
    # Both rows carry a LOOK pitch, so entity zero reads its head like every
    # other entity does. It used to read `combat.pitch`, which was the same
    # number only while the body chased the camera; the body is level now, so
    # the old form would report a permanent zero for the learner's aim.
    pitch = zeros.at[:, AGENT_ENTITY].set(combat.agent_head_pitch)
    if entity_count > 1:
        pitch = pitch.at[:, 1].set(combat.target_head_pitch)
    # Every head starts ON its body. That is `calculateYaw` 609-611 -- with no
    # head steering the engine sets the head yaw to the body yaw -- and it is
    # the only seeding that survives a nonzero reset heading. Taking
    # `target_head_yaw` for row 1 instead looks symmetric with the pitch above,
    # but `reset.py` 152 leaves that field at zero while the target's BODY faces
    # 180, so the row would report a head pointing the wrong way.
    head_yaw = combat.yaw.at[:, AGENT_ENTITY].set(combat.agent_head_yaw)
    walk = _empty_entity_walk_state(batch, entity_count)
    walk = _set_walk_row(walk, AGENT_ENTITY, combat.agent_walk_movement_state)
    return EntityLocomotionState(
        desired_yaw=combat.yaw.at[:, AGENT_ENTITY].set(combat.desired_yaw),
        desired_body_yaw=combat.yaw.at[:, AGENT_ENTITY].set(
            combat.desired_body_yaw
        ),
        head_yaw=head_yaw,
        pitch=pitch,
        desired_pitch=pitch.at[:, AGENT_ENTITY].set(combat.desired_pitch),
        desired_velocity=jnp.zeros((batch, entity_count, 2), dtype=jnp.float32).at[
            :, AGENT_ENTITY
        ].set(combat.desired_velocity),
        move_speed=zeros.at[:, AGENT_ENTITY].set(combat.agent_move_speed),
        stamina=jnp.full(shape, PLAYER_STAMINA_MAXIMUM, dtype=jnp.float32),
        stamina_regen_delay=jnp.zeros(shape, dtype=jnp.float32),
        fall_speed=zeros.at[:, AGENT_ENTITY].set(combat.agent_fall_speed),
        fall_start_y=combat.position[..., 1].at[:, AGENT_ENTITY].set(
            combat.agent_fall_start_y
        ),
        vertical_impulse_applied=jnp.zeros(shape, dtype=jnp.bool_).at[
            :, AGENT_ENTITY
        ].set(combat.vertical_impulse_applied),
        applied_vertical_velocity=zeros.at[:, AGENT_ENTITY].set(
            combat.agent_applied_vertical_velocity
        ),
        grounded_with_residual_velocity=jnp.zeros(shape, dtype=jnp.bool_).at[
            :, AGENT_ENTITY
        ].set(combat.grounded_with_residual_velocity),
        knockback_control_lock=jnp.zeros(shape, dtype=jnp.bool_).at[
            :, AGENT_ENTITY
        ].set(combat.knockback_control_lock),
        grounded=grounded.at[:, AGENT_ENTITY].set(combat.agent_grounded),
        force_velocity=jnp.zeros(
            (batch, entity_count, 3), dtype=jnp.float32
        ).at[:, AGENT_ENTITY].set(combat.agent_force_velocity),
        force_pushed=jnp.zeros(shape, dtype=jnp.bool_).at[:, AGENT_ENTITY].set(
            combat.target_damage_applied_this_tick
        ),
        ticks_since_damage=jnp.zeros(shape, dtype=jnp.int32).at[
            :, AGENT_ENTITY
        ].set(combat.ticks_since_agent_damage),
        geometry_exhausted=jnp.zeros(shape, dtype=jnp.bool_).at[
            :, AGENT_ENTITY
        ].set(combat.geometry_exhausted),
        walk=walk,
    )


def tick_entity_policy_locomotion(
    combat: CombatState,
    locomotion: EntityLocomotionState,
    controls: EntityPolicyControls,
    motion_delta: jax.Array,
    params: CombatParams,
    geometry: GeometryProvider | None = None,
    hold_speed_multiplier: jax.Array | None = None,
) -> tuple[CombatState, EntityLocomotionState]:
    """Advance uniquely owned entity rows through the existing Walk kernel.

    Each static entity row is materialized as the legacy actor row, stepped by
    ``_tick_agent_motion``, and scattered back only where ``controlled`` is
    true. This preserves one implementation of acceleration, gravity, fluid,
    collision, force damping, and source-backed movement-state telemetry.
    """

    batch, entity_count = combat.health.shape
    expected = (batch, entity_count)
    if controls.controlled.shape != expected:
        raise ValueError(f"controlled must have shape {expected}")
    if controls.low_level_action.shape != expected + (ACTION_SIZE,):
        raise ValueError(
            "low_level_action must have shape " f"{expected + (ACTION_SIZE,)}"
        )
    if motion_delta.shape != (batch,):
        raise ValueError(f"motion_delta must have shape {(batch,)}")
    _validate_locomotion_shapes(locomotion, batch, entity_count)
    # C2 HorizontalSpeedMultiplier, one scalar per entity for this tick. The
    # caller resolves it from the ability currently being held; 1.0 means no
    # authored slowdown, which is also what every non-charging tick carries.
    if hold_speed_multiplier is None:
        hold_speed_multiplier = jnp.ones(expected, dtype=jnp.float32)
    elif hold_speed_multiplier.shape != expected:
        raise ValueError(f"hold_speed_multiplier must have shape {expected}")

    next_combat = combat
    next_locomotion = locomotion
    for entity_index in range(entity_count):
        active = controls.controlled[:, entity_index]
        prepared_combat, prepared_locomotion = _apply_entity_action(
            next_combat,
            next_locomotion,
            controls.low_level_action[:, entity_index],
            active,
            entity_index,
            params,
            motion_delta,
            hold_speed_multiplier[:, entity_index],
        )
        virtual = _materialize_entity_zero(
            prepared_combat,
            prepared_locomotion,
            entity_index,
        )
        stepped = _tick_agent_motion(virtual, motion_delta, params, geometry)
        next_combat, next_locomotion = _scatter_stepped_entity(
            next_combat,
            prepared_locomotion,
            stepped,
            active,
            entity_index,
        )

    controlled_exhausted = next_locomotion.geometry_exhausted & controls.controlled
    next_combat = next_combat._replace(
        geometry_exhausted=(
            combat.geometry_exhausted | jnp.any(controlled_exhausted, axis=1)
        )
    )
    return next_combat, next_locomotion


def synchronize_entity_zero_locomotion(
    combat: CombatState,
    locomotion: EntityLocomotionState,
) -> EntityLocomotionState:
    """Copy the shipped actor-zero controller state into the entity bank."""

    active = jnp.ones((combat.health.shape[0],), dtype=jnp.bool_)
    _, synchronized = _scatter_stepped_entity(
        combat,
        locomotion,
        combat,
        active,
        AGENT_ENTITY,
    )
    # Actor zero's locomotion stamina lives on `CombatState` because the combat
    # engine steps it, not `tick_entity_policy_locomotion`. Mirror it into the
    # entity bank so entity zero's row is the same quantity every other row
    # holds: the observation, `rollout.entity_stamina` and every replay channel
    # read the bank, and without this the learner's stamina reads as a constant
    # full bar no matter how much it sprints. That is exactly what run
    # 20260818T035105Z-29d65816 recorded -- 10.0 for all 211 steps of a lane.
    return synchronized._replace(
        stamina=_set_masked_row(
            synchronized.stamina,
            AGENT_ENTITY,
            combat.agent_locomotion_stamina,
            active,
        ),
        stamina_regen_delay=_set_masked_row(
            synchronized.stamina_regen_delay,
            AGENT_ENTITY,
            combat.agent_locomotion_stamina_regen_delay,
            active,
        ),
    )


def _apply_entity_action(
    combat: CombatState,
    locomotion: EntityLocomotionState,
    action: jax.Array,
    active: jax.Array,
    entity_index: int,
    params: CombatParams,
    motion_delta: jax.Array,
    hold_speed_multiplier: jax.Array,
) -> tuple[CombatState, EntityLocomotionState]:
    desired_yaw = _normalize_degrees(
        locomotion.desired_yaw[:, entity_index] + action[:, ACTION_YAW_DELTA]
    )
    # Two levers, not one. `ACTION_YAW_DELTA` above steers the camera;
    # `ACTION_BODY_YAW_DELTA` steers the chest, and it is the chest that decides
    # where the actor travels. A zero delta means "no body steer this tick",
    # which leaves the commanded heading exactly where the last steer left it.
    desired_body_yaw = _normalize_degrees(
        locomotion.desired_body_yaw[:, entity_index]
        + action[:, ACTION_BODY_YAW_DELTA]
    )
    desired_pitch = jnp.clip(
        locomotion.desired_pitch[:, entity_index] + action[:, ACTION_PITCH_DELTA],
        -90.0,
        90.0,
    )
    forward = jnp.clip(action[:, ACTION_FORWARD], 0.0, 1.0)
    back = jnp.clip(action[:, ACTION_BACK], 0.0, 1.0)
    left = jnp.clip(action[:, ACTION_LEFT], 0.0, 1.0)
    right = jnp.clip(action[:, ACTION_RIGHT], 0.0, 1.0)
    longitudinal = forward - back
    lateral = right - left
    length = jnp.hypot(longitudinal, lateral)
    safe_length = jnp.maximum(length, jnp.float32(1.0e-9))
    longitudinal = jnp.where(length > 1.0e-9, longitudinal / safe_length, 0.0)
    lateral = jnp.where(length > 1.0e-9, lateral / safe_length, 0.0)
    # Gait decides speed, but Hytale authors a *different* speed per travel
    # direction, so the compass channels decide more than direction: a run
    # backwards is 0.65 of a run forwards and a strafe is 0.80. The longitudinal
    # and lateral components are scaled separately below, which is what makes an
    # actor's facing matter -- travelling where you are looking is the fastest
    # way to travel.
    #
    # A sprint is granted only while stamina remains: the engine drops an
    # exhausted actor out of its sprint rather than refusing to move, so an empty
    # bar falls back to a run. Sprint additionally requires forward-dominant
    # travel, because the shipped config authors no Backward or Strafe sprint
    # multiplier at all -- see PLAYER_SPRINT_REQUIRES_FORWARD_DOMINANT_TRAVEL for
    # why that absence is read as a gate rather than as a missing number.
    requested_gait = jnp.clip(
        jnp.round(action[:, ACTION_GAIT]).astype(jnp.int32), 0, GAIT_COUNT - 1
    )
    stamina = locomotion.stamina[:, entity_index]
    regen_delay = locomotion.stamina_regen_delay[:, entity_index]
    wants_sprint = requested_gait == jnp.int32(GAIT_SPRINT)
    forward_dominant = longitudinal > jnp.abs(lateral)
    sprinting = (
        wants_sprint
        & (stamina > jnp.float32(0.0))
        & active
        & forward_dominant
    )
    gait = jnp.where(wants_sprint & ~sprinting, jnp.int32(GAIT_RUN), requested_gait)
    # C2: a held Charging root scales planar movement. Applied after the gait
    # multiplier because the authored value scales the resulting horizontal
    # speed, not the gait choice.
    base_speed = params.agent_max_speed * hold_speed_multiplier
    longitudinal_multiplier = jnp.where(
        longitudinal >= jnp.float32(0.0),
        jnp.asarray(PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS, dtype=jnp.float32)[gait],
        jnp.asarray(PLAYER_GAIT_BACKWARD_SPEED_MULTIPLIERS, dtype=jnp.float32)[gait],
    )
    lateral_multiplier = jnp.asarray(
        PLAYER_GAIT_STRAFE_SPEED_MULTIPLIERS, dtype=jnp.float32
    )[gait]
    scaled_longitudinal = longitudinal * longitudinal_multiplier * base_speed
    scaled_lateral = lateral * lateral_multiplier * base_speed

    # Travel rides the BODY. This used to read `desired_yaw`, so an actor walked
    # wherever its camera pointed and could not strafe while tracking a target.
    radians = jnp.deg2rad(desired_body_yaw)
    desired_velocity = jnp.stack(
        (
            scaled_longitudinal * -jnp.sin(radians)
            + scaled_lateral * jnp.cos(radians),
            scaled_longitudinal * -jnp.cos(radians)
            - scaled_lateral * jnp.sin(radians),
        ),
        axis=1,
    )
    # Stamina: sprinting spends and blocks regen, and pins the regen delay so a
    # brief pause does not immediately refill. Regen resumes only once the delay
    # has climbed back to zero. Bottoming out re-arms a shorter delay, which is
    # the engine's Stamina_Broken behaviour.
    drained = stamina - jnp.float32(
        PLAYER_SPRINT_STAMINA_DRAIN_PER_SECOND
    ) * motion_delta
    regenerating = (~sprinting) & (regen_delay >= jnp.float32(0.0))
    recovered = stamina + jnp.float32(
        PLAYER_STAMINA_REGEN_PER_SECOND
    ) * motion_delta
    next_stamina = jnp.clip(
        jnp.where(sprinting, drained, jnp.where(regenerating, recovered, stamina)),
        PLAYER_STAMINA_MINIMUM,
        PLAYER_STAMINA_MAXIMUM,
    )
    emptied = sprinting & (next_stamina <= jnp.float32(0.0))
    next_delay = jnp.where(
        sprinting,
        jnp.float32(-PLAYER_SPRINT_STAMINA_REGEN_DELAY_SECONDS),
        jnp.minimum(
            regen_delay
            + jnp.float32(PLAYER_STAMINA_REGEN_DELAY_RECOVERY_PER_SECOND)
            * motion_delta,
            jnp.float32(0.0),
        ),
    )
    next_delay = jnp.where(
        emptied, jnp.float32(-PLAYER_BROKEN_STAMINA_REGEN_DELAY_SECONDS), next_delay
    )
    jump_held = action[:, ACTION_JUMP] > 0.5
    grounded_jump = jump_held & locomotion.grounded[:, entity_index] & active
    # Variable-height jump. Hytale swaps gravity for ``VariableJumpFallForce``
    # the moment the button is released mid-ascent, so a tap rises far less than
    # a held press. The motion kernel already charges ``world_gravity`` every
    # tick, so only the difference is applied here -- and only while rising,
    # because the release penalty must not deepen an ordinary fall.
    ascending = (combat.velocity[:, entity_index, 1] > 0.0) & ~locomotion.grounded[
        :, entity_index
    ]
    released_mid_ascent = ascending & ~jump_held & active
    release_penalty = (
        params.agent_variable_jump_fall_gravity - params.world_gravity
    ) * motion_delta
    velocity = combat.velocity[:, entity_index].at[:, 1].set(
        jnp.where(
            grounded_jump,
            params.agent_jump_velocity,
            combat.velocity[:, entity_index, 1]
            - jnp.where(released_mid_ascent, release_penalty, jnp.float32(0.0)),
        )
    )
    updated_combat = combat._replace(
        velocity=combat.velocity.at[:, entity_index].set(
            jnp.where(active[:, None], velocity, combat.velocity[:, entity_index])
        )
    )
    return updated_combat, locomotion._replace(
        stamina=_set_masked_row(
            locomotion.stamina, entity_index, next_stamina, active
        ),
        stamina_regen_delay=_set_masked_row(
            locomotion.stamina_regen_delay, entity_index, next_delay, active
        ),
        desired_yaw=_set_masked_row(
            locomotion.desired_yaw, entity_index, desired_yaw, active
        ),
        desired_body_yaw=_set_masked_row(
            locomotion.desired_body_yaw, entity_index, desired_body_yaw, active
        ),
        desired_pitch=_set_masked_row(
            locomotion.desired_pitch, entity_index, desired_pitch, active
        ),
        desired_velocity=_set_masked_row(
            locomotion.desired_velocity, entity_index, desired_velocity, active
        ),
        vertical_impulse_applied=_set_masked_row(
            locomotion.vertical_impulse_applied,
            entity_index,
            jnp.where(
                grounded_jump,
                True,
                locomotion.vertical_impulse_applied[:, entity_index],
            ),
            active,
        ),
        applied_vertical_velocity=_set_masked_row(
            locomotion.applied_vertical_velocity,
            entity_index,
            jnp.where(
                grounded_jump,
                params.agent_jump_velocity,
                locomotion.applied_vertical_velocity[:, entity_index],
            ),
            active,
        ),
        grounded_with_residual_velocity=_set_masked_row(
            locomotion.grounded_with_residual_velocity,
            entity_index,
            jnp.where(
                grounded_jump,
                False,
                locomotion.grounded_with_residual_velocity[:, entity_index],
            ),
            active,
        ),
        knockback_control_lock=_set_masked_row(
            locomotion.knockback_control_lock,
            entity_index,
            jnp.where(
                grounded_jump,
                False,
                locomotion.knockback_control_lock[:, entity_index],
            ),
            active,
        ),
        grounded=_set_masked_row(
            locomotion.grounded,
            entity_index,
            jnp.where(
                grounded_jump,
                False,
                locomotion.grounded[:, entity_index],
            ),
            active,
        ),
    )


def _materialize_entity_zero(
    combat: CombatState,
    locomotion: EntityLocomotionState,
    entity_index: int,
) -> CombatState:
    walk = _walk_row(locomotion.walk, entity_index)
    return combat._replace(
        position=combat.position.at[:, AGENT_ENTITY].set(
            combat.position[:, entity_index]
        ),
        velocity=combat.velocity.at[:, AGENT_ENTITY].set(
            combat.velocity[:, entity_index]
        ),
        health=combat.health.at[:, AGENT_ENTITY].set(combat.health[:, entity_index]),
        yaw=combat.yaw.at[:, AGENT_ENTITY].set(combat.yaw[:, entity_index]),
        desired_yaw=locomotion.desired_yaw[:, entity_index],
        desired_body_yaw=locomotion.desired_body_yaw[:, entity_index],
        pitch=locomotion.pitch[:, entity_index],
        desired_pitch=locomotion.desired_pitch[:, entity_index],
        # `locomotion.pitch` is a LOOK pitch, so it seeds the head. Without
        # this the virtual actor would inherit entity zero's head while wearing
        # entity N's body, and the head window would bind against the wrong
        # chest. The head YAW is carried the same way: it used to be re-seeded
        # from the body every tick, which silently pinned the offset to zero and
        # made the two levers one however the kernel behaved.
        agent_head_yaw=locomotion.head_yaw[:, entity_index],
        agent_head_pitch=locomotion.pitch[:, entity_index],
        desired_velocity=locomotion.desired_velocity[:, entity_index],
        agent_move_speed=locomotion.move_speed[:, entity_index],
        agent_fall_speed=locomotion.fall_speed[:, entity_index],
        agent_fall_start_y=locomotion.fall_start_y[:, entity_index],
        vertical_impulse_applied=(
            locomotion.vertical_impulse_applied[:, entity_index]
        ),
        agent_applied_vertical_velocity=(
            locomotion.applied_vertical_velocity[:, entity_index]
        ),
        grounded_with_residual_velocity=(
            locomotion.grounded_with_residual_velocity[:, entity_index]
        ),
        knockback_control_lock=locomotion.knockback_control_lock[:, entity_index],
        agent_grounded=locomotion.grounded[:, entity_index],
        agent_force_velocity=locomotion.force_velocity[:, entity_index],
        target_damage_applied_this_tick=locomotion.force_pushed[:, entity_index],
        ticks_since_agent_damage=locomotion.ticks_since_damage[:, entity_index],
        geometry_exhausted=locomotion.geometry_exhausted[:, entity_index],
        agent_walk_movement_state=walk,
    )


def _scatter_stepped_entity(
    combat: CombatState,
    locomotion: EntityLocomotionState,
    stepped: CombatState,
    active: jax.Array,
    entity_index: int,
) -> tuple[CombatState, EntityLocomotionState]:
    position = _set_masked_row(
        combat.position, entity_index, stepped.position[:, AGENT_ENTITY], active
    )
    velocity = _set_masked_row(
        combat.velocity, entity_index, stepped.velocity[:, AGENT_ENTITY], active
    )
    yaw = _set_masked_row(
        combat.yaw, entity_index, stepped.yaw[:, AGENT_ENTITY], active
    )
    next_combat = combat._replace(position=position, velocity=velocity, yaw=yaw)
    if entity_index == AGENT_ENTITY:
        next_combat = next_combat._replace(
            desired_yaw=jnp.where(active, stepped.desired_yaw, combat.desired_yaw),
            desired_body_yaw=jnp.where(
                active, stepped.desired_body_yaw, combat.desired_body_yaw
            ),
            pitch=jnp.where(active, stepped.pitch, combat.pitch),
            # The head is stepped alongside the body and has to come back with
            # it; dropping it here would leave the agent's aim frozen at
            # whatever the previous tick left behind.
            agent_head_yaw=jnp.where(
                active, stepped.agent_head_yaw, combat.agent_head_yaw
            ),
            agent_head_pitch=jnp.where(
                active, stepped.agent_head_pitch, combat.agent_head_pitch
            ),
            desired_pitch=jnp.where(
                active, stepped.desired_pitch, combat.desired_pitch
            ),
            desired_velocity=jnp.where(
                active[:, None], stepped.desired_velocity, combat.desired_velocity
            ),
            agent_move_speed=jnp.where(
                active, stepped.agent_move_speed, combat.agent_move_speed
            ),
            agent_fall_speed=jnp.where(
                active, stepped.agent_fall_speed, combat.agent_fall_speed
            ),
            agent_fall_start_y=jnp.where(
                active, stepped.agent_fall_start_y, combat.agent_fall_start_y
            ),
            vertical_impulse_applied=jnp.where(
                active,
                stepped.vertical_impulse_applied,
                combat.vertical_impulse_applied,
            ),
            agent_applied_vertical_velocity=jnp.where(
                active,
                stepped.agent_applied_vertical_velocity,
                combat.agent_applied_vertical_velocity,
            ),
            grounded_with_residual_velocity=jnp.where(
                active,
                stepped.grounded_with_residual_velocity,
                combat.grounded_with_residual_velocity,
            ),
            knockback_control_lock=jnp.where(
                active,
                stepped.knockback_control_lock,
                combat.knockback_control_lock,
            ),
            agent_grounded=jnp.where(
                active, stepped.agent_grounded, combat.agent_grounded
            ),
            agent_force_velocity=jnp.where(
                active[:, None],
                stepped.agent_force_velocity,
                combat.agent_force_velocity,
            ),
            target_damage_applied_this_tick=jnp.where(
                active,
                stepped.target_damage_applied_this_tick,
                combat.target_damage_applied_this_tick,
            ),
            ticks_since_agent_damage=jnp.where(
                active,
                stepped.ticks_since_agent_damage,
                combat.ticks_since_agent_damage,
            ),
            agent_walk_movement_state=_select_walk(
                combat.agent_walk_movement_state,
                stepped.agent_walk_movement_state,
                active,
            ),
        )
    next_locomotion = locomotion._replace(
        desired_yaw=_set_masked_row(
            locomotion.desired_yaw, entity_index, stepped.desired_yaw, active
        ),
        desired_body_yaw=_set_masked_row(
            locomotion.desired_body_yaw,
            entity_index,
            stepped.desired_body_yaw,
            active,
        ),
        # LOOK yaw and LOOK pitch, so both take the stepped head rather than the
        # body. The kernel has already bounded the head into its window.
        head_yaw=_set_masked_row(
            locomotion.head_yaw, entity_index, stepped.agent_head_yaw, active
        ),
        pitch=_set_masked_row(
            locomotion.pitch, entity_index, stepped.agent_head_pitch, active
        ),
        desired_pitch=_set_masked_row(
            locomotion.desired_pitch, entity_index, stepped.desired_pitch, active
        ),
        desired_velocity=_set_masked_row(
            locomotion.desired_velocity,
            entity_index,
            stepped.desired_velocity,
            active,
        ),
        move_speed=_set_masked_row(
            locomotion.move_speed, entity_index, stepped.agent_move_speed, active
        ),
        fall_speed=_set_masked_row(
            locomotion.fall_speed, entity_index, stepped.agent_fall_speed, active
        ),
        fall_start_y=_set_masked_row(
            locomotion.fall_start_y,
            entity_index,
            stepped.agent_fall_start_y,
            active,
        ),
        vertical_impulse_applied=_set_masked_row(
            locomotion.vertical_impulse_applied,
            entity_index,
            stepped.vertical_impulse_applied,
            active,
        ),
        applied_vertical_velocity=_set_masked_row(
            locomotion.applied_vertical_velocity,
            entity_index,
            stepped.agent_applied_vertical_velocity,
            active,
        ),
        grounded_with_residual_velocity=_set_masked_row(
            locomotion.grounded_with_residual_velocity,
            entity_index,
            stepped.grounded_with_residual_velocity,
            active,
        ),
        knockback_control_lock=_set_masked_row(
            locomotion.knockback_control_lock,
            entity_index,
            stepped.knockback_control_lock,
            active,
        ),
        grounded=_set_masked_row(
            locomotion.grounded, entity_index, stepped.agent_grounded, active
        ),
        force_velocity=_set_masked_row(
            locomotion.force_velocity,
            entity_index,
            stepped.agent_force_velocity,
            active,
        ),
        force_pushed=_set_masked_row(
            locomotion.force_pushed,
            entity_index,
            stepped.target_damage_applied_this_tick,
            active,
        ),
        ticks_since_damage=_set_masked_row(
            locomotion.ticks_since_damage,
            entity_index,
            stepped.ticks_since_agent_damage,
            active,
        ),
        geometry_exhausted=_set_masked_row(
            locomotion.geometry_exhausted,
            entity_index,
            stepped.geometry_exhausted,
            active,
        ),
        walk=_set_walk_row_masked(
            locomotion.walk,
            entity_index,
            stepped.agent_walk_movement_state,
            active,
        ),
    )
    return next_combat, next_locomotion


def _empty_entity_walk_state(batch: int, entity_count: int) -> NpcWalkMovementState:
    flat = empty_npc_walk_movement_state(batch * entity_count)
    return jax.tree_util.tree_map(
        lambda value: value.reshape((batch, entity_count) + value.shape[1:]),
        flat,
    )


def _walk_row(walk: NpcWalkMovementState, entity_index: int) -> NpcWalkMovementState:
    return jax.tree_util.tree_map(lambda value: value[:, entity_index], walk)


def _set_walk_row(
    walk: NpcWalkMovementState,
    entity_index: int,
    row: NpcWalkMovementState,
) -> NpcWalkMovementState:
    return jax.tree_util.tree_map(
        lambda values, update: values.at[:, entity_index].set(update),
        walk,
        row,
    )


def _set_walk_row_masked(
    walk: NpcWalkMovementState,
    entity_index: int,
    row: NpcWalkMovementState,
    active: jax.Array,
) -> NpcWalkMovementState:
    current = _walk_row(walk, entity_index)
    selected = jax.tree_util.tree_map(
        lambda old, new: jnp.where(
            active.reshape((active.shape[0],) + (1,) * (old.ndim - 1)),
            new,
            old,
        ),
        current,
        row,
    )
    return _set_walk_row(walk, entity_index, selected)


def _select_walk(
    current: NpcWalkMovementState,
    candidate: NpcWalkMovementState,
    active: jax.Array,
) -> NpcWalkMovementState:
    return jax.tree_util.tree_map(
        lambda old, new: jnp.where(
            active.reshape((active.shape[0],) + (1,) * (old.ndim - 1)),
            new,
            old,
        ),
        current,
        candidate,
    )


def _set_masked_row(
    values: jax.Array,
    entity_index: int,
    update: jax.Array,
    active: jax.Array,
) -> jax.Array:
    current = values[:, entity_index]
    mask = active.reshape((active.shape[0],) + (1,) * (current.ndim - 1))
    return values.at[:, entity_index].set(jnp.where(mask, update, current))


def _validate_locomotion_shapes(
    locomotion: EntityLocomotionState,
    batch: int,
    entity_count: int,
) -> None:
    prefix = (batch, entity_count)
    for name, value in zip(
        EntityLocomotionState._fields,
        locomotion,
        strict=True,
    ):
        leaves = jax.tree_util.tree_leaves(value)
        if not leaves or any(leaf.shape[:2] != prefix for leaf in leaves):
            raise ValueError(f"locomotion.{name} must begin with {prefix}")


__all__ = [
    "EntityLocomotionState",
    "initialize_entity_locomotion_state",
    "synchronize_entity_zero_locomotion",
    "tick_entity_policy_locomotion",
]
