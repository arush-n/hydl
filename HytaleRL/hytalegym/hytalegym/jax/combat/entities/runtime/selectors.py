"""Hytale-shaped exhaustive entity selection over injected candidates."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.entities.schema.contract import (
    ENTITY_CAPACITY,
    ENTITY_FAILURE_INVALID_COMMAND,
    ENTITY_FAILURE_INVALID_STATE,
    ENTITY_FAILURE_NATIVE_RANDOM_SUBSET,
    TEAM_NONE,
)
from hytalegym.jax.combat.entities.schema.types import (
    EntityRoster,
    EntitySelectorQueries,
    EntityTargetSelection,
)
from hytalegym.jax.combat.entities.schema.validation import (
    invalid_roster_rows,
    validate_selector_layout,
)


def select_entity_targets(
    roster: EntityRoster,
    queries: EntitySelectorQueries,
) -> EntityTargetSelection:
    """Filter already-selected geometry without inventing world semantics.

    Hytale's uncapped selectors return every eligible target. If ``MaxTargets``
    would discard candidates, native 0.5.7 uses unseeded reservoir sampling;
    that case is rejected instead of substituting a deterministic winner.
    """

    validate_selector_layout(roster, queries)
    requested = queries.requested
    source = queries.source_slot
    clipped = jnp.clip(source, 0, ENTITY_CAPACITY - 1)
    source_is_entity = (source >= 0) & (source < ENTITY_CAPACITY)
    source_is_environment = (
        (source == -1)
        & (queries.source_generation == jnp.uint32(0))
    )
    source_valid = source_is_environment | (
        source_is_entity
        & _gather(roster.active, clipped)
        & ~_gather(roster.dead, clipped)
        & (
            _gather(roster.generation, clipped)
            == queries.source_generation
        )
    )
    invalid_query = requested & (
        ~source_valid | (queries.max_targets < 0)
    )
    failure = jnp.where(
        invalid_query,
        jnp.uint32(ENTITY_FAILURE_INVALID_COMMAND),
        jnp.uint32(0),
    )
    entity_ids = jnp.arange(ENTITY_CAPACITY, dtype=jnp.int32)[
        None,
        None,
        :,
    ]
    targetable = (
        roster.active
        & roster.damageable
        & ~roster.intangible
        & ~roster.invulnerable
        & ~roster.dead
    )
    candidate = queries.candidate_mask & targetable[:, None, :]
    candidate &= (
        ~queries.ignore_owner[..., None]
        | ~source_is_entity[..., None]
        | (entity_ids != source[..., None])
    )
    source_team = _gather(roster.team_id, clipped)
    same_team = (
        source_is_entity[..., None]
        & (source_team[..., None] != TEAM_NONE)
        & (roster.team_id[:, None, :] == source_team[..., None])
    )
    candidate &= queries.friendly_fire[..., None] | ~same_team
    candidate &= requested[..., None] & source_valid[..., None]
    available_count = jnp.sum(
        candidate.astype(jnp.int32),
        axis=2,
    )
    random_subset = (
        requested
        & (queries.max_targets > 0)
        & (available_count > queries.max_targets)
    )
    failure |= jnp.where(
        random_subset,
        jnp.uint32(ENTITY_FAILURE_NATIVE_RANDOM_SUBSET),
        jnp.uint32(0),
    )
    state_invalid = invalid_roster_rows(roster)
    failure |= jnp.where(
        state_invalid[:, None],
        jnp.uint32(ENTITY_FAILURE_INVALID_STATE),
        jnp.uint32(0),
    )
    failure |= roster.failure_bits[:, None]
    valid = requested & (failure == jnp.uint32(0))
    targets = candidate & valid[..., None]
    row_failure = jnp.bitwise_or.reduce(failure, axis=1)
    return EntityTargetSelection(
        target_mask=targets,
        target_count=jnp.sum(targets.astype(jnp.int32), axis=2),
        available_count=available_count,
        source_slot=jnp.where(valid, source, jnp.int32(-1)),
        failure_bits=failure,
        row_failure_bits=row_failure,
        valid=valid,
    )


def _gather(array: jax.Array, slot: jax.Array) -> jax.Array:
    batch = jnp.arange(array.shape[0]).reshape(
        (array.shape[0],) + (1,) * (slot.ndim - 1)
    )
    return array[batch, slot]

