"""Pure JAX encoder for the frozen learner combat observation."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from hytalegym.jax.combat.runtime.math import _round_to_scale, _unit_fraction
from hytalegym.jax.combat.observation.v1.schema.contract import (
    COMBAT_FLOAT_FEATURES,
    COMBAT_FLOAT_SIZE,
    COMBAT_INTEGER_SIZE,
    ENTITY_FLOAT_SIZE,
    ENTITY_INTEGER_SIZE,
    ENTITY_KIND_NPC,
    HAZARD_FLOAT_SIZE,
    HAZARD_INTEGER_SIZE,
    LEARNER_ACTION_COUNT,
    LEARNER_OBSERVATION_VERSION,
    NEARBY_ENTITY_RADIUS_BLOCKS,
    OVERFLOW_ENTITY,
    OVERFLOW_HAZARD,
    OVERFLOW_INTERACTION,
    OVERFLOW_PROJECTILE,
    OVERFLOW_TERRAIN,
    OVERFLOW_TRAVERSAL,
    PROJECTILE_FLOAT_SIZE,
    PROJECTILE_INTEGER_SIZE,
    SELF_FLOAT_SIZE,
    SELF_INTEGER_SIZE,
    TARGET_FLOAT_SIZE,
    TARGET_INTEGER_SIZE,
)
from hytalegym.jax.combat.observation.v1.runtime.factory import empty_combat_scene
from hytalegym.jax.combat.observation.v1.schema.types import (
    CombatSceneFeatures,
    InjectedWorldFeatures,
    LearnerCombatObservation,
)
from hytalegym.jax.combat.skills import (
    SKILL_IDLE,
    legal_skill_mask,
)
from hytalegym.jax.combat.types import (
    AGENT_ENTITY,
    TARGET_ENTITY,
    CombatParams,
    CombatState,
)


_TARGET_VISIBLE = COMBAT_FLOAT_FEATURES.index("target_visible")
_AGENT_ATTACK_EXECUTING = COMBAT_FLOAT_FEATURES.index("agent_attack_executing")
_TARGET_PHASE_START = COMBAT_FLOAT_FEATURES.index("target_phase_idle")
_TARGET_PHASE_END = COMBAT_FLOAT_FEATURES.index("target_phase_cooldown") + 1
_TARGET_ATTACK_PROGRESS = COMBAT_FLOAT_FEATURES.index("target_attack_progress")
_TARGET_FACING_ERROR = COMBAT_FLOAT_FEATURES.index("target_facing_error")
_TARGET_ATTACK_INDEX = COMBAT_FLOAT_FEATURES.index("target_attack_index")
_TARGET_HEAD_FACING_ERROR = COMBAT_FLOAT_FEATURES.index("target_head_facing_error")
_TARGET_HEAD_PITCH = COMBAT_FLOAT_FEATURES.index("target_head_pitch")
_TARGET_DERIVED_COMBAT_FIELDS = tuple(
    name.startswith(("target_", "visible_target_"))
    for name in COMBAT_FLOAT_FEATURES
)


class _TargetObservationEvidence(NamedTuple):
    """Legal target evidence; privileged ``CombatState`` stays upstream."""

    perceptible: jnp.ndarray
    offset: jnp.ndarray
    relative_velocity: jnp.ndarray
    health_fraction: jnp.ndarray
    yaw_sin: jnp.ndarray
    yaw_cos: jnp.ndarray
    attack_elapsed_ticks: jnp.ndarray
    attack_queued: jnp.ndarray
    attack_sequence_index: jnp.ndarray


def combat_scene_from_state(
    state: CombatState,
    params: CombatParams,
    combat_observation: jnp.ndarray,
    *,
    target_entity_id: jnp.ndarray | None = None,
) -> CombatSceneFeatures:
    """Represent the currently simulated target in nearby-entity slot zero.

    The calibrated environment is melee-only, so projectile and hazard masks
    remain empty. Extra combat entities must arrive through another
    combat-owned ``CombatSceneFeatures`` producer.
    """

    combat_f32 = _sanitize_combat_observation(combat_observation)
    batch = state.position.shape[0]
    target_entity_id = _target_ids(state, target_entity_id)
    scene = empty_combat_scene(batch)
    evidence = _target_observation_evidence(
        state,
        params,
        combat_f32,
        target_entity_id,
    )
    target_f32, target_i32, target_mask = _target_features(
        evidence,
        state.yaw[:, AGENT_ENTITY],
        params,
        combat_f32,
    )
    target_distance = jnp.linalg.norm(evidence.offset, axis=1)
    entity_mask = target_mask & (
        target_distance <= jnp.float32(NEARBY_ENTITY_RADIUS_BLOCKS)
    )
    entity_f32 = jnp.stack(
        (
            target_f32[:, 0],
            target_f32[:, 1],
            target_f32[:, 2],
            target_f32[:, 3],
            target_f32[:, 4],
            target_f32[:, 5],
            target_f32[:, 7],
            target_f32[:, 8],
            evidence.yaw_sin,
            evidence.yaw_cos,
            target_f32[:, 15],
            target_f32[:, 14],
            target_f32[:, 11],
            target_f32[:, 12],
            jnp.ones((batch,), dtype=jnp.float32),
            entity_mask.astype(jnp.float32),
        ),
        axis=1,
    )
    entity_i32 = jnp.stack(
        (
            target_entity_id,
            jnp.full((batch,), ENTITY_KIND_NPC, dtype=jnp.int32),
            target_i32[:, 0],
            target_i32[:, 1],
        ),
        axis=1,
    )
    entity_f32 = jnp.where(entity_mask[:, None], entity_f32, 0.0)
    entity_i32 = jnp.where(entity_mask[:, None], entity_i32, 0)
    return scene._replace(
        entity_f32=scene.entity_f32.at[:, 0, :].set(entity_f32),
        entity_i32=scene.entity_i32.at[:, 0, :].set(entity_i32),
        entity_mask=scene.entity_mask.at[:, 0].set(entity_mask),
    )


def encode_learner_observation(
    state: CombatState,
    params: CombatParams,
    combat_observation: jnp.ndarray,
    scene: CombatSceneFeatures,
    world: InjectedWorldFeatures,
    *,
    target_entity_id: jnp.ndarray | None = None,
) -> LearnerCombatObservation:
    """Encode one batch without selecting tiles or mutating world state."""

    combat_f32 = _sanitize_combat_observation(combat_observation)
    batch = combat_f32.shape[0]
    target_entity_id = _target_ids(state, target_entity_id)
    self_f32, self_i32 = _self_features(state, params, combat_f32)
    evidence = _target_observation_evidence(
        state,
        params,
        combat_f32,
        target_entity_id,
    )
    target_f32, target_i32, target_mask = _target_features(
        evidence,
        state.yaw[:, AGENT_ENTITY],
        params,
        combat_f32,
    )
    combat_i32 = _combat_integer_features(
        state,
        target_i32,
        evidence.attack_sequence_index,
    )

    entity_f32, entity_i32, entity_mask = _sanitize_scene_category(
        scene.entity_f32,
        scene.entity_i32,
        scene.entity_mask,
        scene.entity_overflow,
        ENTITY_FLOAT_SIZE,
        ENTITY_INTEGER_SIZE,
    )
    projectile_f32, projectile_i32, projectile_mask = _sanitize_scene_category(
        scene.projectile_f32,
        scene.projectile_i32,
        scene.projectile_mask,
        scene.projectile_overflow,
        PROJECTILE_FLOAT_SIZE,
        PROJECTILE_INTEGER_SIZE,
    )
    hazard_f32, hazard_i32, hazard_mask = _sanitize_scene_category(
        scene.hazard_f32,
        scene.hazard_i32,
        scene.hazard_mask,
        scene.hazard_overflow,
        HAZARD_FLOAT_SIZE,
        HAZARD_INTEGER_SIZE,
    )

    terrain_mask = (
        jnp.asarray(world.terrain_mask, dtype=jnp.bool_)
        & ~jnp.asarray(world.terrain_overflow, dtype=jnp.bool_)[:, None]
    )
    terrain_f32 = _masked_f32(world.terrain_f32, terrain_mask)
    terrain_semantic_id = _masked_value(
        world.terrain_semantic_id,
        terrain_mask,
        jnp.int32,
    )
    terrain_flags = _masked_value(
        world.terrain_flags,
        terrain_mask,
        jnp.uint32,
    )

    traversal_mask = (
        jnp.asarray(world.traversal_mask, dtype=jnp.bool_)
        & ~jnp.asarray(world.traversal_overflow, dtype=jnp.bool_)[:, None]
    )
    traversal_f32 = _masked_f32(world.traversal_f32, traversal_mask)
    traversal_id = _masked_value(
        world.traversal_id,
        traversal_mask,
        jnp.int32,
    )
    traversal_flags = _masked_value(
        world.traversal_flags,
        traversal_mask,
        jnp.uint32,
    )

    interaction_mask = (
        jnp.asarray(world.interaction_mask, dtype=jnp.bool_)
        & ~jnp.asarray(world.interaction_overflow, dtype=jnp.bool_)[:, None]
    )
    interaction_f32 = _masked_f32(
        world.interaction_f32,
        interaction_mask,
    )
    interaction_object_id = _masked_value(
        world.interaction_object_id,
        interaction_mask,
        jnp.int32,
    )
    interaction_is_door = (
        jnp.asarray(world.interaction_is_door, dtype=jnp.bool_) & interaction_mask
    )
    interaction_door_intent_mask = (
        jnp.asarray(world.interaction_door_intent_mask, dtype=jnp.bool_)
        & interaction_mask[:, :, None]
        & interaction_is_door[:, :, None]
    )

    overflow_bits = _overflow_bits(scene, world)
    valid = overflow_bits == jnp.uint32(0)
    action_mask = _learner_action_mask(
        combat_f32,
        interaction_door_intent_mask,
        valid,
    )
    return LearnerCombatObservation(
        schema_version=jnp.full(
            (batch,),
            LEARNER_OBSERVATION_VERSION,
            dtype=jnp.int32,
        ),
        self_f32=self_f32,
        self_i32=self_i32,
        target_f32=target_f32,
        target_i32=target_i32,
        target_mask=target_mask,
        combat_f32=combat_f32,
        combat_i32=combat_i32,
        entity_f32=entity_f32,
        entity_i32=entity_i32,
        entity_mask=entity_mask,
        projectile_f32=projectile_f32,
        projectile_i32=projectile_i32,
        projectile_mask=projectile_mask,
        hazard_f32=hazard_f32,
        hazard_i32=hazard_i32,
        hazard_mask=hazard_mask,
        terrain_f32=terrain_f32,
        terrain_semantic_id=terrain_semantic_id,
        terrain_flags=terrain_flags,
        terrain_mask=terrain_mask,
        traversal_f32=traversal_f32,
        traversal_id=traversal_id,
        traversal_flags=traversal_flags,
        traversal_mask=traversal_mask,
        interaction_f32=interaction_f32,
        interaction_object_id=interaction_object_id,
        interaction_is_door=interaction_is_door,
        interaction_door_intent_mask=interaction_door_intent_mask,
        interaction_mask=interaction_mask,
        action_mask=action_mask,
        valid=valid,
        overflow_bits=overflow_bits,
    )


def _self_features(
    state: CombatState,
    params: CombatParams,
    combat_observation: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    velocity = state.velocity[:, AGENT_ENTITY]
    # Head-frame: what the actor perceives follows where it looks, not where
    # its chest points. Identical values until a slow separates the two seams.
    yaw = jnp.deg2rad(state.agent_head_yaw)
    forward_x = -jnp.sin(yaw)
    forward_z = -jnp.cos(yaw)
    right_x = jnp.cos(yaw)
    right_z = -jnp.sin(yaw)
    # The fastest gait, not the run gait: a sprint exceeds ``agent_max_speed``
    # by the sprint multiplier and would push this column past 1.0.
    maximum_speed = _positive(params.agent_maximum_gait_speed)
    forward_velocity = (
        velocity[:, 0] * forward_x + velocity[:, 2] * forward_z
    ) / maximum_speed
    right_velocity = (
        velocity[:, 0] * right_x + velocity[:, 2] * right_z
    ) / maximum_speed
    health = state.health[:, AGENT_ENTITY]
    self_f32 = jnp.stack(
        (
            forward_velocity,
            right_velocity,
            velocity[:, 1] / _positive(params.vertical_speed_scale),
            health / _positive(params.agent_max_health),
            jnp.sin(yaw),
            jnp.cos(yaw),
            state.agent_head_pitch / jnp.float32(90.0),
            state.agent_grounded.astype(jnp.float32),
            combat_observation[:, _AGENT_ATTACK_EXECUTING],
            state.agent_attack_cooldown_seconds
            / _positive(params.agent_attack_pause_max_seconds),
            state.knockback_control_lock.astype(jnp.float32),
            state.ticks_since_agent_damage.astype(jnp.float32)
            / _positive(params.regen_delay_ticks.astype(jnp.float32)),
            state.agent_applied_vertical_velocity
            / _positive(params.vertical_speed_scale),
            state.agent_fall_speed
            / _positive(jnp.abs(params.agent_walk_max_fall_speed)),
            state.last_motion_delta_seconds / _positive(params.loaded_dt),
            (health > 0.0).astype(jnp.float32),
        ),
        axis=1,
    )
    self_i32 = jnp.stack(
        (
            state.tick_count,
            state.attack_sequence_index,
            state.pending_agent_attack_index,
            state.motion_timing_profile,
        ),
        axis=1,
    ).astype(jnp.int32)
    if self_f32.shape[-1] != SELF_FLOAT_SIZE:
        raise AssertionError("self feature contract drift")
    if self_i32.shape[-1] != SELF_INTEGER_SIZE:
        raise AssertionError("self integer contract drift")
    return jnp.clip(self_f32, -1.0, 1.0).astype(jnp.float32), self_i32


def _target_ids(
    state: CombatState,
    target_entity_id: jnp.ndarray | None,
) -> jnp.ndarray:
    batch = state.health.shape[0]
    if target_entity_id is None:
        return jnp.full((batch,), TARGET_ENTITY, dtype=jnp.int32)
    if target_entity_id.shape != (batch,):
        raise ValueError("target_entity_id must have shape (batch,)")
    return jnp.asarray(target_entity_id, dtype=jnp.int32)


def _gather_entity(
    array: jnp.ndarray,
    entity_id: jnp.ndarray,
) -> jnp.ndarray:
    return array[
        jnp.arange(array.shape[0]),
        jnp.clip(entity_id, 0, array.shape[1] - 1),
    ]


def _target_observation_evidence(
    state: CombatState,
    params: CombatParams,
    combat_observation: jnp.ndarray,
    target_entity_id: jnp.ndarray,
) -> _TargetObservationEvidence:
    """Read privileged target state once and immediately discard hidden data."""

    perceptible = combat_observation[:, _TARGET_VISIBLE] > jnp.float32(0.5)
    legal_vector = perceptible[:, None]
    target_position = _gather_entity(state.position, target_entity_id)
    target_velocity = _gather_entity(state.velocity, target_entity_id)
    target_health = _gather_entity(state.health, target_entity_id)
    target_yaw_degrees = _gather_entity(state.yaw, target_entity_id)
    primary_target = target_entity_id == TARGET_ENTITY
    privileged_offset = (
        target_position - state.position[:, AGENT_ENTITY]
    )
    # The 24-value combat prefix already consumes the fixed-point offset that
    # Java publishes on the wire.  Reusing the privileged float position here
    # gave learner-v3 a second, slightly different distance to the same target.
    # Project the legal evidence through the same Java Math.round-compatible
    # boundary before deriving target_f32.
    wire_offset = _round_to_scale(
        privileged_offset,
        params.wire_fixed_point_scale,
    )
    privileged_relative_velocity = (
        target_velocity - state.velocity[:, AGENT_ENTITY]
    )
    target_yaw = jnp.deg2rad(target_yaw_degrees)
    return _TargetObservationEvidence(
        perceptible=perceptible,
        offset=jnp.where(legal_vector, wire_offset, 0.0),
        relative_velocity=jnp.where(
            legal_vector,
            privileged_relative_velocity,
            0.0,
        ),
        health_fraction=jnp.where(
            perceptible,
            _unit_fraction(target_health, params.target_max_health),
            0.0,
        ),
        yaw_sin=jnp.where(perceptible, jnp.sin(target_yaw), 0.0),
        yaw_cos=jnp.where(perceptible, jnp.cos(target_yaw), 0.0),
        attack_elapsed_ticks=jnp.where(
            perceptible,
            jnp.where(
                primary_target,
                state.target_attack_elapsed_ticks,
                jnp.int32(0),
            ),
            0,
        ).astype(jnp.int32),
        attack_queued=(
            state.target_attack_queued & perceptible & primary_target
        ).astype(jnp.int32),
        attack_sequence_index=jnp.where(
            perceptible,
            jnp.where(
                primary_target,
                state.target_attack_sequence_index,
                jnp.int32(0),
            ),
            0,
        ).astype(jnp.int32),
    )


def _target_features(
    evidence: _TargetObservationEvidence,
    agent_yaw_degrees: jnp.ndarray,
    params: CombatParams,
    combat_observation: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    offset = evidence.offset
    relative_velocity = evidence.relative_velocity
    yaw = jnp.deg2rad(agent_yaw_degrees)
    forward_x = -jnp.sin(yaw)
    forward_z = -jnp.cos(yaw)
    right_x = jnp.cos(yaw)
    right_z = -jnp.sin(yaw)
    forward = offset[:, 0] * forward_x + offset[:, 2] * forward_z
    right = offset[:, 0] * right_x + offset[:, 2] * right_z
    velocity_forward = (
        relative_velocity[:, 0] * forward_x + relative_velocity[:, 2] * forward_z
    )
    velocity_right = (
        relative_velocity[:, 0] * right_x + relative_velocity[:, 2] * right_z
    )
    planar_distance = jnp.hypot(forward, right)
    distance = jnp.linalg.norm(offset, axis=1)
    safe_planar = jnp.maximum(planar_distance, jnp.float32(1.0e-6))
    radius = jnp.float32(NEARBY_ENTITY_RADIUS_BLOCKS)
    target_speed = _positive(params.target_chase_speed)
    target_f32 = jnp.stack(
        (
            forward / radius,
            right / radius,
            offset[:, 1] / radius,
            velocity_forward / target_speed,
            velocity_right / target_speed,
            relative_velocity[:, 1] / _positive(params.vertical_speed_scale),
            planar_distance / radius,
            distance / radius,
            evidence.health_fraction,
            right / safe_planar,
            forward / safe_planar,
            combat_observation[:, _TARGET_FACING_ERROR],
            combat_observation[:, _TARGET_HEAD_FACING_ERROR],
            combat_observation[:, _TARGET_HEAD_PITCH],
            combat_observation[:, _TARGET_ATTACK_PROGRESS],
            evidence.perceptible.astype(jnp.float32),
        ),
        axis=1,
    )
    phase = jnp.argmax(
        combat_observation[:, _TARGET_PHASE_START:_TARGET_PHASE_END],
        axis=1,
    ).astype(jnp.int32)
    normalized_attack_index = combat_observation[:, _TARGET_ATTACK_INDEX]
    attack_count_minus_one = jnp.asarray(
        params.target_damage.shape[0] - 1,
        dtype=jnp.float32,
    )
    attack_index = jnp.where(
        normalized_attack_index < 0.0,
        -1,
        jnp.rint(normalized_attack_index * attack_count_minus_one).astype(jnp.int32),
    )
    target_i32 = jnp.stack(
        (
            phase,
            attack_index,
            evidence.attack_elapsed_ticks,
            evidence.attack_queued,
        ),
        axis=1,
    ).astype(jnp.int32)
    target_mask = evidence.perceptible
    if target_f32.shape[-1] != TARGET_FLOAT_SIZE:
        raise AssertionError("target feature contract drift")
    if target_i32.shape[-1] != TARGET_INTEGER_SIZE:
        raise AssertionError("target integer contract drift")
    return (
        jnp.clip(target_f32, -1.0, 1.0).astype(jnp.float32),
        target_i32,
        target_mask,
    )


def _combat_integer_features(
    state: CombatState,
    target_i32: jnp.ndarray,
    target_attack_sequence_index: jnp.ndarray,
) -> jnp.ndarray:
    result = jnp.stack(
        (
            target_i32[:, 0],
            target_i32[:, 1],
            target_i32[:, 2],
            state.attack_sequence_index,
            target_attack_sequence_index,
            state.pending_agent_attack_index,
            state.tick_count,
            state.motion_timing_profile,
        ),
        axis=1,
    ).astype(jnp.int32)
    if result.shape[-1] != COMBAT_INTEGER_SIZE:
        raise AssertionError("combat integer contract drift")
    return result


def _sanitize_combat_observation(
    combat_observation: jnp.ndarray,
) -> jnp.ndarray:
    """Keep target-derived compact fields absent when no target is perceptible."""

    combat_f32 = jnp.asarray(combat_observation, dtype=jnp.float32)
    if combat_f32.shape[-1] != COMBAT_FLOAT_SIZE:
        raise ValueError(
            f"combat_observation must have trailing size {COMBAT_FLOAT_SIZE}"
        )
    perceptible = combat_f32[:, _TARGET_VISIBLE] > jnp.float32(0.5)
    target_fields = jnp.asarray(
        _TARGET_DERIVED_COMBAT_FIELDS,
        dtype=jnp.bool_,
    )
    legal = ~target_fields[None, :] | perceptible[:, None]
    return jnp.where(legal, combat_f32, 0.0)


def _sanitize_scene_category(
    float_values: jnp.ndarray,
    integer_values: jnp.ndarray,
    mask: jnp.ndarray,
    overflow: jnp.ndarray,
    float_size: int,
    integer_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    effective_mask = (
        jnp.asarray(mask, dtype=jnp.bool_)
        & ~jnp.asarray(overflow, dtype=jnp.bool_)[:, None]
    )
    float_result = _masked_f32(float_values, effective_mask)
    integer_result = _masked_value(
        integer_values,
        effective_mask[:, :, None],
        jnp.int32,
    )
    if float_result.shape[-1] != float_size:
        raise ValueError("scene float feature width does not match contract")
    if integer_result.shape[-1] != integer_size:
        raise ValueError("scene integer feature width does not match contract")
    return float_result, integer_result, effective_mask


def _masked_f32(values: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    array = jnp.asarray(values, dtype=jnp.float32)
    return jnp.where(
        mask[:, :, None],
        jnp.clip(array, -1.0, 1.0),
        jnp.float32(0.0),
    )


def _masked_value(
    values: jnp.ndarray,
    mask: jnp.ndarray,
    dtype: jnp.dtype,
) -> jnp.ndarray:
    array = jnp.asarray(values, dtype=dtype)
    return jnp.where(mask, array, jnp.asarray(0, dtype=dtype))


def _overflow_bits(
    scene: CombatSceneFeatures,
    world: InjectedWorldFeatures,
) -> jnp.ndarray:
    batch = scene.entity_overflow.shape[0]
    result = jnp.zeros((batch,), dtype=jnp.uint32)
    pairs = (
        (scene.entity_overflow, OVERFLOW_ENTITY),
        (scene.projectile_overflow, OVERFLOW_PROJECTILE),
        (scene.hazard_overflow, OVERFLOW_HAZARD),
        (world.terrain_overflow, OVERFLOW_TERRAIN),
        (world.traversal_overflow, OVERFLOW_TRAVERSAL),
        (world.interaction_overflow, OVERFLOW_INTERACTION),
    )
    for overflow, bit in pairs:
        result = jnp.bitwise_or(
            result,
            jnp.where(
                jnp.asarray(overflow, dtype=jnp.bool_),
                jnp.uint32(bit),
                jnp.uint32(0),
            ),
        )
    return result


def _learner_action_mask(
    combat_observation: jnp.ndarray,
    interaction_door_intent_mask: jnp.ndarray,
    valid: jnp.ndarray,
) -> jnp.ndarray:
    skill_mask = legal_skill_mask(combat_observation)
    door_mask = jnp.any(interaction_door_intent_mask, axis=1)
    full = jnp.concatenate((skill_mask, door_mask), axis=1)
    safe = jnp.zeros_like(full)
    safe = safe.at[:, SKILL_IDLE].set(True)
    result = jnp.where(valid[:, None], full, safe)
    if result.shape[-1] != LEARNER_ACTION_COUNT:
        raise AssertionError("learner action contract drift")
    return result


def _positive(value: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(jnp.abs(value), jnp.float32(1.0e-6))
