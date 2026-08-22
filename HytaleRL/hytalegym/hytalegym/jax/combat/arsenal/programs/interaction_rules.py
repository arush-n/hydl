"""Bounded Hytale interaction-rule arbitration used by combat commands.

The server evaluates a new root against admitted ``InteractionManager``
chains. Roots still waiting in ``chainStartQueue`` are deliberately absent
from that comparison. This module models the part exercised by the current
Arsenal surface without pretending that its single ability handle is a
general multi-chain interaction graph.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.contract import (
    INTERACTION_TYPE_COUNT,
    INTERACTION_TYPE_SECONDARY,
)
from hytalegym.jax.combat.arsenal.schema.types import AbilityLoadout, ArsenalState


class GuardRuleResolution(NamedTuple):
    """Admission and interruption result for a rising guard edge."""

    can_start: jax.Array
    interrupts_active: jax.Array


def active_ability_interaction_type(
    arsenal: ArsenalState,
    loadout: AbilityLoadout,
) -> jax.Array:
    """Return the protocol type of each active ability, or ``-1``."""

    ability_capacity = loadout.ability_interaction_type.shape[2]
    root_slot = jnp.where(
        arsenal.active_ability_root_slot >= 0,
        arsenal.active_ability_root_slot,
        arsenal.active_ability_slot,
    )
    slot = jnp.clip(root_slot, 0, ability_capacity - 1)
    selected = jnp.take_along_axis(
        loadout.ability_interaction_type,
        slot[..., None],
        axis=2,
    )[..., 0]
    return jnp.where(
        arsenal.active_ability_slot >= 0,
        selected,
        jnp.int32(-1),
    )


def active_operation_interrupted_by(
    admitted_arsenal: ArsenalState,
    loadout: AbilityLoadout,
    incoming_type: jax.Array,
) -> jax.Array:
    """Whether an incoming root cancels the current authored operation.

    Hytale evaluates both the active root rules and the active operation's
    rules.  The latter is observable on Crossbow reload: its 0.8-second
    startup is not interruptible, while the following ``Repeat`` operation is
    ``InterruptedBy`` Primary and Secondary.  ``incoming_type`` may be one
    type per actor or a complete per-ability candidate matrix.
    """

    ability_capacity = loadout.ability_interrupted_by_type_mask.shape[2]
    active = admitted_arsenal.active_ability_slot >= 0
    slot = jnp.clip(
        admitted_arsenal.active_ability_slot,
        0,
        ability_capacity - 1,
    )
    interrupted_by = jnp.take_along_axis(
        loadout.ability_interrupted_by_type_mask,
        slot[..., None],
        axis=2,
    )[..., 0]
    boundary = jnp.take_along_axis(
        loadout.ability_interruptible_after_seconds,
        slot[..., None],
        axis=2,
    )[..., 0]
    while active.ndim < incoming_type.ndim:
        active = active[..., None]
        interrupted_by = interrupted_by[..., None]
        boundary = boundary[..., None]
    valid_type = (
        (incoming_type >= 0)
        & (incoming_type < INTERACTION_TYPE_COUNT)
    )
    safe_type = jnp.clip(incoming_type, 0, INTERACTION_TYPE_COUNT - 1)
    type_bit = jnp.left_shift(
        jnp.uint32(1),
        safe_type.astype(jnp.uint32),
    )
    elapsed = admitted_arsenal.ability_elapsed_seconds
    while elapsed.ndim < incoming_type.ndim:
        elapsed = elapsed[..., None]
    return (
        active
        & valid_type
        & (elapsed + jnp.float32(1.0e-6) >= boundary)
        & ((interrupted_by & type_bit) != 0)
    )


def guard_rule_resolution(
    admitted_arsenal: ArsenalState,
    loadout: AbilityLoadout,
) -> GuardRuleResolution:
    """Resolve a new guard root against currently admitted ability roots.

    ``admitted_arsenal`` must already hide the private queued-root handle.
    Installed 0.5.7 guard roots retain Secondary's default ``BlockedBy`` set.
    Most common guard roots additionally declare ``Interrupting: [Primary]``;
    Spear_Block does not. The asset-derived mask stays in the loadout so that
    distinction and future variants do not require weapon-name branches here.
    """

    active = admitted_arsenal.active_ability_slot >= 0
    interaction_type = active_ability_interaction_type(
        admitted_arsenal,
        loadout,
    )
    valid_type = (
        (interaction_type >= 0)
        & (interaction_type < INTERACTION_TYPE_COUNT)
    )
    safe_type = jnp.clip(interaction_type, 0, INTERACTION_TYPE_COUNT - 1)
    type_bit = jnp.left_shift(
        jnp.uint32(1),
        safe_type.astype(jnp.uint32),
    )
    root_interrupts = (
        active
        & valid_type
        & ((loadout.guard_interrupting_type_mask & type_bit) != 0)
    )
    operation_interrupts = active_operation_interrupted_by(
        admitted_arsenal,
        loadout,
        jnp.full_like(
            interaction_type,
            INTERACTION_TYPE_SECONDARY,
        ),
    )
    interrupts = root_interrupts | operation_interrupts
    return GuardRuleResolution(
        can_start=~active | interrupts,
        interrupts_active=interrupts,
    )


__all__ = [
    "GuardRuleResolution",
    "active_operation_interrupted_by",
    "active_ability_interaction_type",
    "guard_rule_resolution",
]
