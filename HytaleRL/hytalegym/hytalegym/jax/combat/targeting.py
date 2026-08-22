"""Fixed-shape entity candidate ranking and single-target engagement."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array

TEAM_NONE = -1
TARGET_CANDIDATE_CAPACITY = 8
WALK_DISTANCE_COMPONENT_SELECTOR = (1.0, 0.0, 1.0)
THREE_DIMENSIONAL_DISTANCE_COMPONENT_SELECTOR = (1.0, 1.0, 1.0)


class CombatTargetingRules(NamedTuple):
    """Episode-pinned relationship and controller-distance inputs."""

    team_id: Array
    distance_component_selector: Array
    sensor_range: Array


class CombatTargetSelection(NamedTuple):
    """Distance-sorted candidates plus the one engaged target."""

    candidate_entity_id: Array
    candidate_mask: Array
    candidate_distance_squared: Array
    engagement_target_id: Array
    engagement_target_mask: Array


def default_combat_targeting_rules(
    batch_size: int,
    entity_count: int,
    *,
    sensor_range: Array | float,
    team_id: Array | None = None,
    distance_component_selector: Array | None = None,
) -> CombatTargetingRules:
    """Build the current player-versus-NPC relationship without fixing ``N``.

    Entity zero is the policy actor. By default every other slot shares one
    opposing team and every controller uses Hytale's Walk distance selector.
    Callers may supply per-entity teams and runtime-mutated component selectors
    for allies or Fly/Dive controllers.
    """

    batch = _positive_size(batch_size, "batch_size")
    entities = _positive_size(entity_count, "entity_count")
    if entities < 2:
        raise ValueError("entity_count must be at least 2")

    if team_id is None:
        teams = jnp.ones((batch, entities), dtype=jnp.int32)
        teams = teams.at[:, 0].set(jnp.int32(0))
    else:
        teams = jnp.asarray(team_id, dtype=jnp.int32)
        if teams.shape == (entities,):
            teams = jnp.broadcast_to(teams[None, :], (batch, entities))
        if teams.shape != (batch, entities):
            raise ValueError(
                "team_id must have shape "
                f"({batch}, {entities}) or ({entities},)"
            )

    if distance_component_selector is None:
        selector = jnp.asarray(
            WALK_DISTANCE_COMPONENT_SELECTOR,
            dtype=jnp.float32,
        )
    else:
        selector = jnp.asarray(
            distance_component_selector,
            dtype=jnp.float32,
        )
    if selector.shape == (3,):
        selector = jnp.broadcast_to(
            selector[None, None, :],
            (batch, entities, 3),
        )
    elif selector.shape == (entities, 3):
        selector = jnp.broadcast_to(
            selector[None, :, :],
            (batch, entities, 3),
        )
    if selector.shape != (batch, entities, 3):
        raise ValueError(
            "distance_component_selector must have shape "
            f"(3,), ({entities}, 3), or ({batch}, {entities}, 3)"
        )

    ranges = jnp.asarray(sensor_range, dtype=jnp.float32)
    if ranges.ndim == 0:
        ranges = jnp.broadcast_to(ranges, (batch, entities))
    elif ranges.shape == (entities,):
        ranges = jnp.broadcast_to(ranges[None, :], (batch, entities))
    if ranges.shape != (batch, entities):
        raise ValueError(
            "sensor_range must be scalar or have shape "
            f"({entities},) or ({batch}, {entities})"
        )
    return CombatTargetingRules(
        team_id=teams,
        distance_component_selector=selector,
        sensor_range=ranges,
    )


def select_combat_targets(
    position: Array,
    health: Array,
    rules: CombatTargetingRules,
    *,
    candidate_capacity: int,
    candidate_evidence: Array | None = None,
    locked_target_id: Array | None = None,
) -> CombatTargetSelection:
    """Filter legal evidence, rank fixed-K candidates, and retain one lock.

    Hytale applies the controller's runtime-mutable component selector before
    computing squared distance. Walk therefore ranks in XZ while Fly and Dive
    rank in XYZ. The repeated argmin is O(N*K) per source and uses entity ID as
    the deterministic tie-break, without replacing native distance semantics
    with Euclidean K-nearest. A caller-supplied ``candidate_evidence`` matrix
    carries already-legal FOV/LOS/filter evidence and is applied before the
    rank, so a hidden nearer entity cannot suppress a visible farther one.
    Omitting it retains the relationship/range-only compatibility path.

    A valid ``locked_target_id`` is retained across decisions even when that
    entity is no longer acquisition-visible or inside the acquisition sensor
    range. This mirrors ``MarkedEntitySupport`` identity persistence: legal
    perception gates acquisition, while ability/selector capabilities still
    gate what the controller may do with the retained target. Dead, self,
    same-team, non-finite, or out-of-range IDs release atomically and fall
    back to the first ranked legal candidate.
    """

    capacity = _positive_size(candidate_capacity, "candidate_capacity")
    if position.ndim != 3 or position.shape[2] != 3:
        raise ValueError("position must have shape (batch, entity, 3)")
    batch, entities, _ = position.shape
    if health.shape != (batch, entities):
        raise ValueError("health must match position's batch and entity axes")
    if rules.team_id.shape != (batch, entities):
        raise ValueError("targeting team_id does not match combat state")
    if rules.distance_component_selector.shape != (batch, entities, 3):
        raise ValueError(
            "targeting distance_component_selector does not match combat state"
        )
    if rules.sensor_range.shape != (batch, entities):
        raise ValueError("targeting sensor_range does not match combat state")
    if candidate_evidence is None:
        legal_evidence = jnp.ones(
            (batch, entities, entities),
            dtype=jnp.bool_,
        )
    else:
        legal_evidence = jnp.asarray(candidate_evidence)
        if legal_evidence.dtype != jnp.dtype(jnp.bool_):
            raise ValueError("candidate_evidence must have boolean dtype")
        if legal_evidence.shape != (batch, entities, entities):
            raise ValueError(
                "candidate_evidence must have shape "
                f"({batch}, {entities}, {entities})"
            )

    delta = position[:, None, :, :] - position[:, :, None, :]
    scaled_delta = (
        delta * rules.distance_component_selector[:, :, None, :]
    )
    distance_squared = jnp.sum(scaled_delta * scaled_delta, axis=3)
    entity_id = jnp.arange(entities, dtype=jnp.int32)
    not_self = (
        entity_id[None, None, :]
        != entity_id[None, :, None]
    )
    source_team = rules.team_id[:, :, None]
    same_team = (
        (source_team != jnp.int32(TEAM_NONE))
        & (rules.team_id[:, None, :] == source_team)
    )
    alive = health > jnp.float32(0.0)
    within_range = (
        (rules.sensor_range[:, :, None] >= jnp.float32(0.0))
        & (
            distance_squared
            <= rules.sensor_range[:, :, None] ** jnp.float32(2.0)
        )
    )
    eligible = (
        alive[:, :, None]
        & alive[:, None, :]
        & not_self
        & ~same_team
        & within_range
        & jnp.isfinite(distance_squared)
        & legal_evidence
    )
    remaining = jnp.where(
        eligible,
        distance_squared,
        jnp.float32(jnp.inf),
    )

    def choose_one(current: Array, _):
        selected_id = jnp.argmin(current, axis=2).astype(jnp.int32)
        selected_distance = jnp.take_along_axis(
            current,
            selected_id[..., None],
            axis=2,
        )[..., 0]
        selected_mask = jnp.isfinite(selected_distance)
        next_remaining = current.at[
            jnp.arange(batch)[:, None],
            jnp.arange(entities)[None, :],
            selected_id,
        ].set(jnp.float32(jnp.inf))
        return next_remaining, (
            jnp.where(selected_mask, selected_id, jnp.int32(-1)),
            selected_mask,
            jnp.where(
                selected_mask,
                selected_distance,
                jnp.float32(0.0),
            ),
        )

    _, (selected_id, selected_mask, selected_distance) = jax.lax.scan(
        choose_one,
        remaining,
        xs=None,
        length=capacity,
    )
    candidate_entity_id = jnp.moveaxis(selected_id, 0, 2)
    candidate_mask = jnp.moveaxis(selected_mask, 0, 2)
    candidate_distance_squared = jnp.moveaxis(selected_distance, 0, 2)
    ranked_target_id = candidate_entity_id[..., 0]
    ranked_target_mask = candidate_mask[..., 0]
    if locked_target_id is None:
        locked_id = jnp.full(
            (batch, entities),
            -1,
            dtype=jnp.int32,
        )
    else:
        locked_id = jnp.asarray(locked_target_id)
        if locked_id.dtype != jnp.dtype(jnp.int32):
            raise ValueError("locked_target_id must have int32 dtype")
        if locked_id.shape != (batch, entities):
            raise ValueError(
                "locked_target_id must have shape "
                f"({batch}, {entities})"
            )
    locked_id_in_range = (locked_id >= 0) & (locked_id < entities)
    safe_locked_id = jnp.clip(locked_id, 0, entities - 1)
    retained_eligible = (
        alive[:, :, None]
        & alive[:, None, :]
        & not_self
        & ~same_team
        & jnp.isfinite(distance_squared)
    )
    retained_lock = locked_id_in_range & jnp.take_along_axis(
        retained_eligible,
        safe_locked_id[..., None],
        axis=2,
    )[..., 0]
    engagement_target_id = jnp.where(
        retained_lock,
        locked_id,
        ranked_target_id,
    )
    engagement_target_mask = retained_lock | ranked_target_mask
    return CombatTargetSelection(
        candidate_entity_id=candidate_entity_id,
        candidate_mask=candidate_mask,
        candidate_distance_squared=candidate_distance_squared,
        engagement_target_id=engagement_target_id,
        engagement_target_mask=engagement_target_mask,
    )


def _positive_size(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


__all__ = [
    "CombatTargetSelection",
    "CombatTargetingRules",
    "TARGET_CANDIDATE_CAPACITY",
    "TEAM_NONE",
    "THREE_DIMENSIONAL_DISTANCE_COMPONENT_SELECTOR",
    "WALK_DISTANCE_COMPONENT_SELECTOR",
    "default_combat_targeting_rules",
    "select_combat_targets",
]
