"""Actor-slot controls mapped onto the shared combat entity axis."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.types import ArsenalCommands
from hytalegym.jax.combat.types import ACTION_SIZE

from .assignment import (
    PolicyActorAssignment,
    policy_controlled_entity_mask,
    scatter_policy_actor_rows,
)


class PolicyActorControls(NamedTuple):
    """One decoded combat control row per policy slot ``[B,P]``."""

    low_level_action: jax.Array
    ability_slot: jax.Array
    guard_held: jax.Array
    dodge_direction: jax.Array
    dodge_corridor_clear: jax.Array
    valid: jax.Array


class EntityPolicyControls(NamedTuple):
    """Policy controls scattered onto entity rows ``[B,N]``."""

    low_level_action: jax.Array
    ability_slot: jax.Array
    guard_held: jax.Array
    dodge_direction: jax.Array
    dodge_corridor_clear: jax.Array
    controlled: jax.Array


def policy_actor_controls(
    low_level_action,
    ability_slot,
    guard_held,
    dodge_direction,
    dodge_corridor_clear,
    valid,
    assignment: PolicyActorAssignment,
) -> PolicyActorControls:
    """Validate and fail-close decoded actor-slot controls.

    Invalid or inactive policy slots become exact no-ops before any value is
    scattered onto the entity axis.  This keeps malformed actions from
    replacing a scripted controller merely because the slot owns an entity.
    """

    shape = assignment.actor_index.shape
    low_level = jnp.asarray(low_level_action, dtype=jnp.float32)
    ability = jnp.asarray(ability_slot, dtype=jnp.int32)
    guard = jnp.asarray(guard_held, dtype=jnp.bool_)
    dodge = jnp.asarray(dodge_direction, dtype=jnp.int32)
    corridor = jnp.asarray(dodge_corridor_clear, dtype=jnp.bool_)
    valid_rows = jnp.asarray(valid, dtype=jnp.bool_)
    if low_level.shape != shape + (ACTION_SIZE,):
        raise ValueError(f"low_level_action must have shape {shape + (ACTION_SIZE,)}")
    for name, value in (
        ("ability_slot", ability),
        ("guard_held", guard),
        ("dodge_direction", dodge),
        ("dodge_corridor_clear", corridor),
        ("valid", valid_rows),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")

    accepted = assignment.active & valid_rows
    return PolicyActorControls(
        low_level_action=jnp.where(
            accepted[..., None],
            low_level,
            jnp.float32(0.0),
        ),
        ability_slot=jnp.where(accepted, ability, jnp.int32(-1)),
        guard_held=guard & accepted,
        dodge_direction=jnp.where(accepted, dodge, jnp.int32(0)),
        dodge_corridor_clear=corridor & accepted,
        valid=accepted,
    )


def scatter_policy_actor_controls(
    controls: PolicyActorControls,
    assignment: PolicyActorAssignment,
    *,
    entity_count: int,
) -> EntityPolicyControls:
    """Scatter uniquely owned policy controls without assuming ``N == 2``."""

    controlled = policy_controlled_entity_mask(
        assignment._replace(active=controls.valid),
        entity_count=entity_count,
    )
    return EntityPolicyControls(
        low_level_action=scatter_policy_actor_rows(
            controls.low_level_action,
            assignment._replace(active=controls.valid),
            entity_count=entity_count,
            fill_value=0.0,
        ),
        ability_slot=scatter_policy_actor_rows(
            controls.ability_slot,
            assignment._replace(active=controls.valid),
            entity_count=entity_count,
            fill_value=-1,
        ),
        guard_held=scatter_policy_actor_rows(
            controls.guard_held,
            assignment._replace(active=controls.valid),
            entity_count=entity_count,
            fill_value=False,
        ),
        dodge_direction=scatter_policy_actor_rows(
            controls.dodge_direction,
            assignment._replace(active=controls.valid),
            entity_count=entity_count,
            fill_value=0,
        ),
        dodge_corridor_clear=scatter_policy_actor_rows(
            controls.dodge_corridor_clear,
            assignment._replace(active=controls.valid),
            entity_count=entity_count,
            fill_value=False,
        ),
        controlled=controlled,
    )


def merge_policy_actor_commands(
    commands: ArsenalCommands,
    entity_controls: EntityPolicyControls,
) -> ArsenalCommands:
    """Overlay policy ability/defense commands on their owned entity rows.

    Unowned rows retain the supplied scripted commands.  Low-level movement
    intentionally remains in ``EntityPolicyControls`` because the current
    runtime stores actor-zero locomotion state separately; silently dropping
    nonzero actors' movement here would create false self-play support.
    """

    entity_shape = commands.ability_slot.shape
    if len(entity_shape) != 2:
        raise ValueError("commands must have entity shape [B,N]")
    expected = (
        ("ability_slot", entity_controls.ability_slot),
        ("guard_held", entity_controls.guard_held),
        ("dodge_direction", entity_controls.dodge_direction),
        ("dodge_corridor_clear", entity_controls.dodge_corridor_clear),
        ("controlled", entity_controls.controlled),
    )
    for name, value in expected:
        if value.shape != entity_shape:
            raise ValueError(f"{name} must have shape {entity_shape}")
    if entity_controls.low_level_action.shape != entity_shape + (ACTION_SIZE,):
        raise ValueError(
            "low_level_action must have shape "
            f"{entity_shape + (ACTION_SIZE,)}"
        )
    controlled = entity_controls.controlled
    defense = commands.defense._replace(
        guard_held=jnp.where(
            controlled,
            entity_controls.guard_held,
            commands.defense.guard_held,
        ),
        dodge_direction=jnp.where(
            controlled,
            entity_controls.dodge_direction,
            commands.defense.dodge_direction,
        ),
        dodge_corridor_clear=jnp.where(
            controlled,
            entity_controls.dodge_corridor_clear,
            commands.defense.dodge_corridor_clear,
        ),
    )
    return commands._replace(
        ability_slot=jnp.where(
            controlled,
            entity_controls.ability_slot,
            commands.ability_slot,
        ),
        defense=defense,
    )


__all__ = [
    "EntityPolicyControls",
    "PolicyActorControls",
    "merge_policy_actor_commands",
    "policy_actor_controls",
    "scatter_policy_actor_controls",
]
