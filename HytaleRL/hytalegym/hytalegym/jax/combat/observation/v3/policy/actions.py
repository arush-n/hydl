"""Semantic learner-action decoding for arsenal combat."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.contract import OBSERVATION_CAPACITY
from hytalegym.jax.combat.arsenal.factory import empty_arsenal_commands
from hytalegym.jax.combat.arsenal.schema.types import ArsenalWorldCapabilities
from hytalegym.jax.combat.inventory import HOTBAR_CAPACITY
from hytalegym.jax.combat.mechanics import DODGE_DIRECTION_COUNT
from hytalegym.jax.combat.observation.v1.runtime.actions import (
    LearnerActionContext,
    decode_learner_action_context,
    learner_action_context,
)
from hytalegym.jax.combat.observation.v1.schema.contract import (
    ACTION_DOOR_OPEN,
    DOOR_INTENT_COUNT,
)
from hytalegym.jax.combat.observation.v3.schema.types import (
    LearnerArsenalAction,
    LearnerArsenalDecode,
    LearnerCombatObservationV3,
)
from hytalegym.jax.combat.skills import SKILL_COUNT
from hytalegym.jax.combat.types import (
    ACTION_BACK,
    ACTION_BODY_YAW_DELTA,
    ACTION_FORWARD,
    ACTION_GAIT,
    ACTION_HOTBAR_SLOT,
    ACTION_JUMP,
    ACTION_LEFT,
    ACTION_PITCH_DELTA,
    ACTION_RIGHT,
    ACTION_YAW_DELTA,
    AGENT_ENTITY,
)

WORLD_MOVE_DIRECTION_COUNT = 8
WORLD_MOVE_CHOICE_COUNT = WORLD_MOVE_DIRECTION_COUNT + 1


class LearnerArsenalActionContext(NamedTuple):
    """Minimal learner fields required by the Arsenal transition."""

    base: LearnerActionContext
    ability_action_mask: jnp.ndarray
    jump_action_mask: jnp.ndarray
    guard_action_mask: jnp.ndarray
    dodge_action_mask: jnp.ndarray
    valid: jnp.ndarray


def learner_arsenal_action_context(
    observation: LearnerCombatObservationV3,
) -> LearnerArsenalActionContext:
    """Project an observation to its exact action-decoding subset."""

    base_mask = observation.base.action_mask.at[:, :SKILL_COUNT].set(
        observation.skill_action_mask
    )
    door_legal = jnp.any(observation.door_action_mask, axis=1)
    base_mask = base_mask.at[
        :, ACTION_DOOR_OPEN : ACTION_DOOR_OPEN + DOOR_INTENT_COUNT
    ].set(door_legal)
    return LearnerArsenalActionContext(
        base=learner_action_context(
            observation.base._replace(action_mask=base_mask)
        ),
        ability_action_mask=observation.ability_action_mask,
        jump_action_mask=observation.jump_action_mask,
        guard_action_mask=observation.guard_action_mask,
        dodge_action_mask=observation.dodge_action_mask,
        valid=observation.valid,
    )


def apply_world_move_override(
    low_level_action,
    world_move_direction,
    desired_body_yaw_degrees,
    *,
    legal,
):
    """Override translation with one BODY-RELATIVE direction.

    Choice zero is an exact no-op. Choices one through eight are the body's own
    forward, forward-right, right, ... in 45-degree increments -- the player's
    WASD, which is what Hytale actually accepts and what the native transport
    sends as forward/back/left/right booleans.

    The name says "world" for history: the semantic field, the native transport
    key and the Java `AgentAction` all still spell it ``world_move_direction``.
    Renaming that chain is a separate change; the frame is body-relative.

    This used to be a World-XZ compass (North through North-West) that was then
    rotated into the body frame here. That double transform is what made travel
    uncorrelated with the target: choosing a world bearing requires knowing your
    own heading, and the observation publishes the HEAD yaw while travel rides
    the BODY. ``desired_body_yaw_degrees`` is still taken because the combat
    core rotates by the body heading downstream, so a non-finite value still
    has to fail the move closed -- it just no longer steers the direction.
    """

    actions = jnp.asarray(low_level_action)
    if actions.ndim != 2:
        raise ValueError("low_level_action must have shape (B, A)")
    if actions.dtype != jnp.float32:
        raise ValueError("low_level_action must have float32 dtype")
    batch = actions.shape[0]
    direction = jnp.asarray(world_move_direction, dtype=jnp.int32)
    yaw = jnp.asarray(desired_body_yaw_degrees, dtype=jnp.float32)
    legal_mask = jnp.asarray(legal, dtype=jnp.bool_)
    for name, value in (
        ("world_move_direction", direction),
        ("desired_body_yaw_degrees", yaw),
        ("legal", legal_mask),
    ):
        if value.shape != (batch,):
            raise ValueError(f"{name} must have shape (B,)")

    requested = direction != 0
    in_range = (direction > 0) & (direction < WORLD_MOVE_CHOICE_COUNT)
    orientation_valid = jnp.isfinite(yaw) & jnp.isfinite(
        actions[:, ACTION_BODY_YAW_DELTA]
    )
    accepted = requested & in_range & legal_mask & orientation_valid
    safe_direction = jnp.clip(direction, 1, WORLD_MOVE_DIRECTION_COUNT)
    # BODY-RELATIVE, not world-absolute. Choice one is "forward along the
    # body", three is "strafe right", five "back", seven "left" -- which is
    # what a player expresses with WASD and exactly what the native transport
    # already carries as forward/back/left/right booleans
    # (`native/codec/transport.py` 440-450). The combat core rotates these
    # local channels by the body heading, so no rotation belongs here: doing it
    # here as well was the defect. Previously this read an ABSOLUTE compass
    # bearing and rotated it into the body frame, which meant the agent had to
    # compute its own heading before it could aim its travel -- and since the
    # observation carries the HEAD yaw, not the body's, it could not.
    #
    # Verified against the old formula: at body yaw 0 all eight bins agree
    # exactly (a body frame at yaw 0 IS the world frame), and they diverge by
    # precisely the body heading elsewhere.
    body_angle = (
        safe_direction.astype(jnp.float32) - jnp.float32(1.0)
    ) * jnp.float32(jnp.pi / 4.0)
    longitudinal = jnp.cos(body_angle)
    lateral = jnp.sin(body_angle)
    candidate = actions
    candidate = candidate.at[:, ACTION_FORWARD].set(
        jnp.maximum(longitudinal, 0.0)
    )
    candidate = candidate.at[:, ACTION_BACK].set(
        jnp.maximum(-longitudinal, 0.0)
    )
    candidate = candidate.at[:, ACTION_LEFT].set(
        jnp.maximum(-lateral, 0.0)
    )
    candidate = candidate.at[:, ACTION_RIGHT].set(
        jnp.maximum(lateral, 0.0)
    )
    return jnp.where(accepted[:, None], candidate, actions), (
        (direction == 0) | (in_range & legal_mask & orientation_valid)
    )


def decode_learner_arsenal_action_context(
    context: LearnerArsenalActionContext,
    action: LearnerArsenalAction,
    world: ArsenalWorldCapabilities,
    *,
    maximum_turn_degrees: float = 45.0,
    desired_body_yaw_degrees=None,
) -> LearnerArsenalDecode:
    """Decode an action from the compact environment carry."""

    batch = context.valid.shape[0]
    decoded = decode_learner_action_context(
        context.base,
        action.skill_id,
        maximum_turn_degrees=maximum_turn_degrees,
    )

    ability_requested = action.ability_slot >= 0
    ability_in_range = (action.ability_slot >= 0) & (
        action.ability_slot < OBSERVATION_CAPACITY
    )
    safe_ability = jnp.clip(
        action.ability_slot,
        0,
        OBSERVATION_CAPACITY - 1,
    )
    selected_ability_legal = jnp.take_along_axis(
        context.ability_action_mask,
        safe_ability[:, None],
        axis=1,
    )[:, 0]
    ability_legal = ~ability_requested | (ability_in_range & selected_ability_legal)

    jump_requested = jnp.asarray(action.jump_held, dtype=jnp.bool_)
    jump_legal = ~jump_requested | context.jump_action_mask
    guard_requested = jnp.asarray(action.guard_held, dtype=jnp.bool_)
    guard_legal = ~guard_requested | context.guard_action_mask
    dodge_requested = action.dodge_direction > 0
    dodge_in_range = (action.dodge_direction > 0) & (
        action.dodge_direction < DODGE_DIRECTION_COUNT
    )
    safe_dodge = jnp.clip(action.dodge_direction - 1, 0, 3)
    selected_dodge_clear = jnp.take_along_axis(
        context.dodge_action_mask,
        safe_dodge[:, None],
        axis=1,
    )[:, 0]
    dodge_legal = ~dodge_requested | (dodge_in_range & selected_dodge_clear)
    commands = empty_arsenal_commands(
        batch,
        entity_count=world.line_of_sight.shape[1],
    )
    ability_slot = commands.ability_slot.at[:, AGENT_ENTITY].set(
        jnp.where(
            ability_requested & ability_legal,
            action.ability_slot,
            jnp.int32(-1),
        )
    )
    defense = commands.defense._replace(
        guard_held=commands.defense.guard_held.at[:, AGENT_ENTITY].set(
            guard_requested & guard_legal
        ),
        dodge_direction=commands.defense.dodge_direction.at[:, AGENT_ENTITY].set(
            jnp.where(
                dodge_requested & dodge_legal,
                action.dodge_direction,
                jnp.int32(0),
            )
        ),
        dodge_corridor_clear=commands.defense.dodge_corridor_clear.at[
            :, AGENT_ENTITY
        ].set(dodge_requested & dodge_legal & selected_dodge_clear),
    )
    # Weapon switching. Only the actor row can be commanded; -1 elsewhere leaves
    # every other entity's live slot alone. Out-of-range requests are dropped to
    # -1 here rather than clipped, because clipping would silently switch to a
    # slot the policy did not ask for.
    requested_hotbar = jnp.asarray(action.hotbar_slot, dtype=jnp.int32)
    if requested_hotbar.ndim == 0:
        requested_hotbar = jnp.broadcast_to(requested_hotbar, (batch,))
    if requested_hotbar.shape != (batch,):
        raise ValueError("hotbar_slot must have shape (B,)")
    hotbar_in_range = (requested_hotbar >= jnp.int32(0)) & (
        requested_hotbar < jnp.int32(HOTBAR_CAPACITY)
    )
    hotbar_slot = commands.hotbar_slot
    if hotbar_slot is None:
        hotbar_slot = jnp.full_like(commands.ability_slot, -1)
    hotbar_slot = hotbar_slot.at[:, AGENT_ENTITY].set(
        jnp.where(hotbar_in_range, requested_hotbar, jnp.int32(-1))
    )
    commands = commands._replace(
        ability_slot=ability_slot,
        defense=defense,
        world=world,
        hotbar_slot=hotbar_slot,
    )
    valid = (
        context.valid
        & decoded.action_legal
        & ability_legal
        & jump_legal
        & guard_legal
        & dodge_legal
    )
    low_level_action = decoded.low_level_action.at[:, ACTION_JUMP].set(
        (jump_requested & jump_legal).astype(decoded.low_level_action.dtype)
    )
    yaw_delta = jnp.asarray(action.yaw_delta_degrees, dtype=jnp.float32)
    if yaw_delta.ndim == 0:
        yaw_delta = jnp.broadcast_to(yaw_delta, (batch,))
    if yaw_delta.shape != (batch,):
        raise ValueError("yaw_delta_degrees must have shape (B,)")
    yaw_legal = jnp.isfinite(yaw_delta) & (
        jnp.abs(yaw_delta) <= jnp.float32(maximum_turn_degrees)
    )
    low_level_action = low_level_action.at[:, ACTION_YAW_DELTA].set(
        jnp.where(yaw_legal, yaw_delta, jnp.float32(0.0))
    )
    body_yaw_delta = jnp.asarray(
        action.body_yaw_delta_degrees, dtype=jnp.float32
    )
    if body_yaw_delta.ndim == 0:
        body_yaw_delta = jnp.broadcast_to(body_yaw_delta, (batch,))
    if body_yaw_delta.shape != (batch,):
        raise ValueError("body_yaw_delta_degrees must have shape (B,)")
    body_yaw_legal = jnp.isfinite(body_yaw_delta) & (
        jnp.abs(body_yaw_delta) <= jnp.float32(maximum_turn_degrees)
    )
    low_level_action = low_level_action.at[:, ACTION_BODY_YAW_DELTA].set(
        jnp.where(body_yaw_legal, body_yaw_delta, jnp.float32(0.0))
    )
    # Carried as slot + 1 so that ZERO means "no switch". Every caller that
    # builds a neutral action with `jnp.zeros((B, ACTION_SIZE))` -- the scripted
    # skills, the multi-actor fixtures, the native no-op -- would otherwise be
    # requesting slot 0 on every tick.
    low_level_action = low_level_action.at[:, ACTION_HOTBAR_SLOT].set(
        jnp.where(
            hotbar_in_range,
            (requested_hotbar + jnp.int32(1)).astype(low_level_action.dtype),
            jnp.float32(0.0),
        )
    )
    pitch_delta = jnp.asarray(action.pitch_delta_degrees, dtype=jnp.float32)
    if pitch_delta.ndim == 0:
        pitch_delta = jnp.broadcast_to(pitch_delta, (batch,))
    if pitch_delta.shape != (batch,):
        raise ValueError("pitch_delta_degrees must have shape (B,)")
    pitch_legal = jnp.isfinite(pitch_delta)
    low_level_action = low_level_action.at[:, ACTION_PITCH_DELTA].set(
        jnp.where(pitch_legal, pitch_delta, jnp.float32(0.0))
    )
    # The gait rides alongside the compass channels. Whether a sprint is
    # actually granted is decided by the walk step, which owns stamina.
    low_level_action = low_level_action.at[:, ACTION_GAIT].set(
        jnp.asarray(action.gait, dtype=low_level_action.dtype)
    )
    move_direction = jnp.asarray(
        action.world_move_direction,
        dtype=jnp.int32,
    )
    if move_direction.shape != (batch,):
        raise ValueError("world_move_direction must have shape (B,)")
    move_requested = move_direction != 0
    if desired_body_yaw_degrees is None:
        desired_yaw = jnp.zeros((batch,), dtype=jnp.float32)
        move_context_available = ~move_requested
    else:
        desired_yaw = jnp.asarray(
            desired_body_yaw_degrees,
            dtype=jnp.float32,
        )
        if desired_yaw.shape != (batch,):
            raise ValueError("desired_body_yaw_degrees must have shape (B,)")
        move_context_available = jnp.isfinite(desired_yaw)
    low_level_action, move_legal = apply_world_move_override(
        low_level_action,
        move_direction,
        desired_yaw,
        legal=context.valid & move_context_available,
    )
    valid &= pitch_legal & yaw_legal & body_yaw_legal & move_legal
    return LearnerArsenalDecode(
        low_level_action=low_level_action,
        commands=commands,
        door=decoded.door,
        skill_legal=decoded.action_legal,
        ability_legal=ability_legal,
        jump_legal=jump_legal,
        guard_legal=guard_legal,
        dodge_legal=dodge_legal,
        valid=valid,
    )


def decode_learner_arsenal_action(
    observation: LearnerCombatObservationV3,
    action: LearnerArsenalAction,
    world: ArsenalWorldCapabilities,
    *,
    maximum_turn_degrees: float = 45.0,
    desired_body_yaw_degrees=None,
) -> LearnerArsenalDecode:
    """Decode masks to movement, ability, defense, and intent-only doors."""

    return decode_learner_arsenal_action_context(
        learner_arsenal_action_context(observation),
        action,
        world,
        maximum_turn_degrees=maximum_turn_degrees,
        desired_body_yaw_degrees=desired_body_yaw_degrees,
    )


__all__ = [
    "WORLD_MOVE_CHOICE_COUNT",
    "WORLD_MOVE_DIRECTION_COUNT",
    "LearnerArsenalActionContext",
    "apply_world_move_override",
    "decode_learner_arsenal_action",
    "decode_learner_arsenal_action_context",
    "learner_arsenal_action_context",
]
