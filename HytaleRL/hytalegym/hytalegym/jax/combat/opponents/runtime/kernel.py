"""Evidence-gated opponent memory and controller-mode transitions."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.opponents.schema.contract import (
    OPPONENT_MODE_CHASE,
    OPPONENT_MODE_INACTIVE,
    OPPONENT_MODE_RETURN_HOME_REQUIRED,
    OPPONENT_MODE_SEARCH_REQUIRED,
)
from hytalegym.jax.combat.opponents.schema.types import OpponentMemoryState
from hytalegym.jax.combat.targeting import CombatTargetSelection


def step_opponent_memory(
    state: OpponentMemoryState,
    position: jax.Array,
    health: jax.Array,
    selection: CombatTargetSelection,
    candidate_perceptible: jax.Array,
    candidate_perception_valid: jax.Array,
    candidate_heard: jax.Array,
    controlled: jax.Array,
    *,
    distance_component_selector: jax.Array,
    chase_stop_distance: jax.Array,
    chase_view_range: jax.Array,
    delta_seconds: jax.Array,
) -> OpponentMemoryState:
    """Advance independent controller memory without fabricating navigation.

    A valid and perceptible selected target refreshes LastSeen. A hidden target
    is pursued only through that stored coordinate. Reaching it enters Search,
    and expiry enters ReturnHome; both modes report navigation unavailable
    until a certified World producer supplies the requested displacement.
    """

    _validate_inputs(
        state,
        position,
        health,
        selection,
        candidate_perceptible,
        candidate_perception_valid,
        candidate_heard,
        controlled,
        distance_component_selector,
    )
    batch, entities = health.shape
    safe_target = jnp.clip(
        selection.engagement_target_id,
        0,
        entities - 1,
    )
    selected_position = jnp.take_along_axis(
        position,
        safe_target[..., None],
        axis=1,
    )
    selected_health = jnp.take_along_axis(
        health,
        safe_target,
        axis=1,
    )
    selected_perceptible = jnp.take_along_axis(
        candidate_perceptible,
        safe_target[..., None],
        axis=2,
    )[..., 0]
    selected_evidence_valid = jnp.take_along_axis(
        candidate_perception_valid,
        safe_target[..., None],
        axis=2,
    )[..., 0]
    selected_heard = jnp.take_along_axis(
        candidate_heard,
        safe_target[..., None],
        axis=2,
    )[..., 0]
    view_range = jnp.broadcast_to(
        jnp.asarray(chase_view_range, dtype=jnp.float32),
        (batch, entities),
    )
    selected_delta = selected_position - position
    selected_distance_squared = jnp.sum(
        selected_delta * selected_delta,
        axis=2,
    )
    sight_detected = (
        selected_evidence_valid
        & selected_perceptible
        & (selected_distance_squared <= view_range * view_range)
    )
    active = (
        controlled
        & (health > jnp.float32(0.0))
        & selection.engagement_target_mask
        & (selected_health > jnp.float32(0.0))
    )
    detected = active & (sight_detected | selected_heard)

    # The privileged selected coordinate is consumed exactly once and becomes
    # a legal coordinate before any downstream controller calculation.
    legal_selected_position = jnp.where(
        detected[..., None],
        selected_position,
        jnp.float32(0.0),
    )
    last_seen_position = jnp.where(
        detected[..., None],
        legal_selected_position,
        state.last_seen_position,
    )
    last_seen_valid = state.last_seen_valid | detected

    selector = jnp.asarray(distance_component_selector, dtype=jnp.float32)
    memory_delta = (last_seen_position - position) * selector
    memory_distance_squared = jnp.sum(memory_delta * memory_delta, axis=2)
    stop_distance = jnp.broadcast_to(
        jnp.asarray(chase_stop_distance, dtype=jnp.float32),
        (batch, entities),
    )
    memory_in_range = (
        memory_distance_squared <= view_range * view_range
    )
    at_last_seen = (
        memory_distance_squared <= stop_distance * stop_distance
    )
    delta = jnp.asarray(delta_seconds, dtype=jnp.float32)
    if delta.ndim == 0:
        delta = jnp.broadcast_to(delta, (batch, entities))
    elif delta.shape == (batch,):
        delta = jnp.broadcast_to(delta[:, None], (batch, entities))
    elif delta.shape != (batch, entities):
        raise ValueError(
            "delta_seconds must be scalar, per batch, or per entity"
        )
    can_track = active & last_seen_valid & memory_in_range
    search_required = can_track & ~detected & at_last_seen
    search_elapsed = jnp.where(
        detected,
        jnp.float32(0.0),
        jnp.where(
            search_required
            | (state.mode == jnp.int32(OPPONENT_MODE_SEARCH_REQUIRED)),
            state.search_elapsed_seconds
            + delta,
            state.search_elapsed_seconds,
        ),
    )
    return_home_required = (
        active
        & ~detected
        & last_seen_valid
        & (
            ~memory_in_range
            | (search_elapsed >= state.search_timeout_seconds)
        )
    )
    chase = can_track & ~search_required & ~return_home_required
    mode = jnp.where(
        return_home_required,
        jnp.int32(OPPONENT_MODE_RETURN_HOME_REQUIRED),
        jnp.where(
            search_required
            | (
                (state.mode == jnp.int32(OPPONENT_MODE_SEARCH_REQUIRED))
                & ~detected
            ),
            jnp.int32(OPPONENT_MODE_SEARCH_REQUIRED),
            jnp.where(
                chase,
                jnp.int32(OPPONENT_MODE_CHASE),
                jnp.int32(OPPONENT_MODE_INACTIVE),
            ),
        ),
    )
    navigation_unavailable = (
        (mode == jnp.int32(OPPONENT_MODE_SEARCH_REQUIRED))
        | (mode == jnp.int32(OPPONENT_MODE_RETURN_HOME_REQUIRED))
    )
    pursuit_elapsed_ticks = jnp.where(
        chase,
        state.pursuit_elapsed_ticks + jnp.int32(1),
        state.pursuit_elapsed_ticks,
    )
    return state._replace(
        mode=mode,
        last_seen_position=last_seen_position,
        last_seen_valid=last_seen_valid,
        pursuit_elapsed_ticks=pursuit_elapsed_ticks,
        search_elapsed_seconds=search_elapsed,
        navigation_unavailable=navigation_unavailable,
    )


def _validate_inputs(
    state: OpponentMemoryState,
    position: jax.Array,
    health: jax.Array,
    selection: CombatTargetSelection,
    candidate_perceptible: jax.Array,
    candidate_perception_valid: jax.Array,
    candidate_heard: jax.Array,
    controlled: jax.Array,
    distance_component_selector: jax.Array,
) -> None:
    if position.ndim != 3 or position.shape[2] != 3:
        raise ValueError("position must have shape (batch, entity, 3)")
    batch, entities, _ = position.shape
    entity_shape = (batch, entities)
    pair_shape = (batch, entities, entities)
    if health.shape != entity_shape:
        raise ValueError("health must match position's entity axes")
    if controlled.shape != entity_shape:
        raise ValueError("controlled must match position's entity axes")
    if candidate_perceptible.shape != pair_shape:
        raise ValueError("candidate_perceptible must be pairwise")
    if candidate_perception_valid.shape != pair_shape:
        raise ValueError("candidate_perception_valid must be pairwise")
    if candidate_heard.shape != pair_shape:
        raise ValueError("candidate_heard must be pairwise")
    if distance_component_selector.shape != (batch, entities, 3):
        raise ValueError("distance_component_selector must be per entity")
    if selection.engagement_target_id.shape != entity_shape:
        raise ValueError("selection must match position's entity axes")
    if state.mode.shape != entity_shape:
        raise ValueError("opponent memory must match position's entity axes")
    if candidate_perceptible.dtype != jnp.dtype(jnp.bool_):
        raise ValueError("candidate_perceptible must have boolean dtype")
    if candidate_perception_valid.dtype != jnp.dtype(jnp.bool_):
        raise ValueError("candidate_perception_valid must have boolean dtype")
    if candidate_heard.dtype != jnp.dtype(jnp.bool_):
        raise ValueError("candidate_heard must have boolean dtype")


def pairwise_hearing_evidence(
    position: jax.Array,
    source_enabled: jax.Array,
    target_audible: jax.Array,
    *,
    hearing_range: jax.Array,
) -> jax.Array:
    """Return legal 3-D hearing evidence for each source-target pair.

    The caller owns role-specific audibility (for example, the shipped
    Brawler cannot hear a crouching actor). This helper only applies source
    enablement, the authored hearing radius, self exclusion, and finite
    geometry-free distance. Hearing intentionally does not depend on LOS
    result validity.
    """

    if position.ndim != 3 or position.shape[2] != 3:
        raise ValueError("position must have shape (batch, entity, 3)")
    batch, entities, _ = position.shape
    entity_shape = (batch, entities)
    if source_enabled.shape != entity_shape:
        raise ValueError("source_enabled must match position's entity axes")
    if target_audible.shape != entity_shape:
        raise ValueError("target_audible must match position's entity axes")
    if source_enabled.dtype != jnp.dtype(jnp.bool_):
        raise ValueError("source_enabled must have boolean dtype")
    if target_audible.dtype != jnp.dtype(jnp.bool_):
        raise ValueError("target_audible must have boolean dtype")
    radius = jnp.asarray(hearing_range, dtype=jnp.float32)
    if radius.ndim == 0:
        radius = jnp.broadcast_to(radius, entity_shape)
    elif radius.shape == (entities,):
        radius = jnp.broadcast_to(radius[None, :], entity_shape)
    if radius.shape != entity_shape:
        raise ValueError(
            "hearing_range must be scalar, per entity, or per source entity"
        )

    delta = position[:, None, :, :] - position[:, :, None, :]
    distance_squared = jnp.sum(delta * delta, axis=3)
    entity_id = jnp.arange(entities, dtype=jnp.int32)
    not_self = entity_id[None, None, :] != entity_id[None, :, None]
    return (
        source_enabled[:, :, None]
        & target_audible[:, None, :]
        & not_self
        & (radius[:, :, None] >= jnp.float32(0.0))
        & (distance_squared <= radius[:, :, None] ** jnp.float32(2.0))
        & jnp.isfinite(distance_squared)
    )


__all__ = ["pairwise_hearing_evidence", "step_opponent_memory"]
