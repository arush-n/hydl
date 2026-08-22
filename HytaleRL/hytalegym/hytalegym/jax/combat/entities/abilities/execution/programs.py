"""Compiled provenance checks for packed authored ability events."""

from __future__ import annotations

import jax.numpy as jnp

from hytalegym.jax.combat.arsenal import (
    ABILITY_CAPACITY,
    EVENT_CAPACITY,
)


def events_match_program(events, programs):
    """Match every packed event to its exact family/ability/event slot."""

    family = jnp.clip(events.weapon_family, 0, programs.event_mask.shape[0] - 1)
    ability = jnp.clip(events.ability_slot, 0, ABILITY_CAPACITY - 1)
    event = jnp.clip(events.event_slot, 0, EVENT_CAPACITY - 1)
    valid_index = (
        (events.weapon_family >= 0)
        & (events.weapon_family < programs.event_mask.shape[0])
        & (events.ability_slot >= 0)
        & (events.ability_slot < ABILITY_CAPACITY)
        & (events.event_slot >= 0)
        & (events.event_slot < EVENT_CAPACITY)
    )
    return (
        valid_index
        & programs.event_mask[family, ability, event]
        & (events.kind == programs.event_kind[family, ability, event])
        & jnp.all(
            events.f32 == programs.event_f32[family, ability, event],
            axis=2,
        )
        & jnp.all(
            events.i32 == programs.event_i32[family, ability, event],
            axis=2,
        )
        & (events.flags == programs.event_flags[family, ability, event])
    )
