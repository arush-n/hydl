"""Data-derived admission for entity-only authored area effects."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.contract import (
    EF_RADIUS,
    EVENT_AREA,
    EVENT_FLAG_ENTITY_ONLY_AREA,
    EVENT_PROJECTILE,
    REQUIRE_ENTITY_ONLY_AREA,
    REQUIRE_STATIC_AREA_PLACEMENT,
)
from hytalegym.jax.combat.arsenal.schema.types import AbilityLoadout
from hytalegym.jax.world.geometry.point_raycast import (
    DEFAULT_POINT_RAY_SEGMENT_CAPACITY,
)


class StaticAreaPlacementPlan(NamedTuple):
    """One unambiguous asset-derived placement root per source entity."""

    required: jax.Array
    unique: jax.Array
    maximum_distance: jax.Array
    allow_walls: jax.Array


def static_area_placement_plan(
    loadout: AbilityLoadout,
) -> StaticAreaPlacementPlan:
    """Select a generic static-placement root without weapon-name branches.

    `ArsenalWorldCapabilities` exposes one placement result per source entity,
    so an equipped loadout is admitted only when exactly one active ability
    requests that capability. Loadout validation rejects ambiguous profiles;
    this producer still fails them closed so direct calls remain safe.
    """

    candidates = loadout.ability_mask & (
        (
            loadout.ability_requirements
            & jnp.uint32(REQUIRE_STATIC_AREA_PLACEMENT)
        )
        != 0
    )
    count = jnp.sum(candidates.astype(jnp.int32), axis=2)
    selected = jnp.argmax(candidates.astype(jnp.int32), axis=2)
    maximum_distance = jnp.take_along_axis(
        loadout.ability_static_placement_maximum_distance,
        selected[..., None],
        axis=2,
    )[..., 0]
    allow_walls = jnp.take_along_axis(
        loadout.ability_static_placement_allow_walls,
        selected[..., None],
        axis=2,
    )[..., 0]
    unique = count == 1
    return StaticAreaPlacementPlan(
        required=count > 0,
        unique=unique,
        maximum_distance=jnp.where(unique, maximum_distance, 0.0),
        allow_walls=jnp.where(unique, allow_walls, False),
    )


def static_area_placement_support_mask(loadout: AbilityLoadout):
    """Return asset-complete abilities supported by the point-ray producer."""

    required = loadout.ability_mask & (
        (
            loadout.ability_requirements
            & jnp.uint32(REQUIRE_STATIC_AREA_PLACEMENT)
        )
        != 0
    )
    finite_range = jnp.isfinite(
        loadout.ability_static_placement_maximum_distance
    ) & (
        loadout.ability_static_placement_maximum_distance > 0.0
    ) & (
        loadout.ability_static_placement_maximum_distance
        <= jnp.float32(DEFAULT_POINT_RAY_SEGMENT_CAPACITY)
    )
    unambiguous = jnp.sum(required.astype(jnp.int32), axis=2) == 1
    return required & finite_range & unambiguous[..., None]


def entity_only_area_support_mask(loadout: AbilityLoadout):
    """Return independently flagged entity-only abilities.

    The requirement bit states what an ability needs. The event flag records
    the separate asset fact that every explosion/area node in that ability has
    no block-mutation branch. Keeping those two claims separate prevents a
    requirement from certifying itself.
    """

    event_mask, event_kind, event_f32, event_flags = _ability_event_bank(loadout)
    required = loadout.ability_mask & (
        (loadout.ability_requirements & jnp.uint32(REQUIRE_ENTITY_ONLY_AREA)) != 0
    )
    area_effect = event_mask & (
        (event_kind == EVENT_AREA)
        | ((event_kind == EVENT_PROJECTILE) & (event_f32[..., EF_RADIUS] > 0.0))
    )
    flagged = (
        event_flags & jnp.uint32(EVENT_FLAG_ENTITY_ONLY_AREA)
    ) != 0
    return (
        required
        & jnp.any(area_effect, axis=3)
        & jnp.all(~area_effect | flagged, axis=3)
    )


def entity_only_area_source_support(loadout: AbilityLoadout):
    """Return sources whose complete authored area surface is entity-only."""

    required = loadout.ability_mask & (
        (loadout.ability_requirements & jnp.uint32(REQUIRE_ENTITY_ONLY_AREA)) != 0
    )
    supported = entity_only_area_support_mask(loadout)
    return jnp.all(~required | supported, axis=2)


def _ability_event_bank(loadout: AbilityLoadout):
    """Return the needed event fields in ability-major form."""

    if loadout.event_mask.ndim == 4:
        return (
            loadout.event_mask,
            loadout.event_kind,
            loadout.event_f32,
            loadout.event_flags,
        )
    if loadout.event_mask.ndim != 3:
        raise ValueError("loadout event storage must be packed or ability-major")

    ability_capacity = loadout.ability_mask.shape[2]
    event_capacity = loadout.event_mask.shape[2]
    event_index = jnp.arange(event_capacity)[None, None, None, :]
    start = loadout.ability_event_start[..., None]
    end = start + loadout.ability_event_count[..., None]
    mask = (
        loadout.event_mask[:, :, None, :]
        & (event_index >= start)
        & (event_index < end)
    )

    def expand(value):
        return jnp.broadcast_to(
            value[:, :, None, ...],
            value.shape[:2] + (ability_capacity,) + value.shape[2:],
        )

    return (
        mask,
        expand(loadout.event_kind),
        expand(loadout.event_f32),
        expand(loadout.event_flags),
    )


__all__ = [
    "entity_only_area_source_support",
    "entity_only_area_support_mask",
    "StaticAreaPlacementPlan",
    "static_area_placement_plan",
    "static_area_placement_support_mask",
]
