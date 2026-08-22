"""Source-attributed per-actor rewards for shared combat arenas."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.mechanics import DamageResolution
from hytalegym.jax.combat.types import CombatParams, RewardComponents, compose_reward


class MultiActorRewardMemory(NamedTuple):
    """Per-entity terminal awards; no row borrows another actor's objective."""

    completion_awarded: jax.Array
    death_awarded: jax.Array


def empty_multi_actor_reward_memory(
    batch_size: int,
    entity_count: int,
) -> MultiActorRewardMemory:
    """Return reset memory for all possible policy actors."""

    shape = (batch_size, entity_count)
    return MultiActorRewardMemory(
        completion_awarded=jnp.zeros(shape, dtype=jnp.bool_),
        death_awarded=jnp.zeros(shape, dtype=jnp.bool_),
    )


def per_actor_reward_components(
    resolution: DamageResolution,
    team_id: jax.Array,
    health_after: jax.Array,
    memory: MultiActorRewardMemory,
) -> tuple[RewardComponents, MultiActorRewardMemory]:
    """Attribute damage and terminal facts to every entity row.

    Outgoing damage is credited only to the event's source and only against a
    different team. Incoming damage is charged to the victim even when the
    source is environmental (``-1``). Completion is team-relative; death is
    actor-relative. This prevents a second policy from training on entity
    zero's reward while preserving environmental penalties.
    """

    teams = jnp.asarray(team_id, dtype=jnp.int32)
    health = jnp.asarray(health_after, dtype=jnp.float32)
    batch, entity_count = health.shape
    if teams.shape != health.shape:
        raise ValueError("team_id and health_after must have shape [B,N]")
    if (
        memory.completion_awarded.shape != health.shape
        or memory.death_awarded.shape != health.shape
    ):
        raise ValueError("reward memory must match the health entity axis")
    event_shape = resolution.applied_damage.shape
    if len(event_shape) != 2 or event_shape[0] != batch:
        raise ValueError("damage resolution must have shape [B,E]")
    for field in (
        resolution.requested,
        resolution.source_entity_id,
        resolution.target_entity_id,
    ):
        if field.shape != event_shape:
            raise ValueError("damage resolution fields must share shape [B,E]")

    source_valid = (
        resolution.requested
        & (resolution.source_entity_id >= 0)
        & (resolution.source_entity_id < entity_count)
    )
    target_valid = (
        resolution.requested
        & (resolution.target_entity_id >= 0)
        & (resolution.target_entity_id < entity_count)
    )
    safe_source = jnp.clip(resolution.source_entity_id, 0, entity_count - 1)
    safe_target = jnp.clip(resolution.target_entity_id, 0, entity_count - 1)
    batch_index = jnp.arange(batch, dtype=jnp.int32)[:, None]
    source_team = teams[batch_index, safe_source]
    target_team = teams[batch_index, safe_target]
    enemy_hit = source_valid & target_valid & (source_team != target_team)
    applied = jnp.maximum(
        jnp.asarray(resolution.applied_damage, dtype=jnp.float32),
        jnp.float32(0.0),
    )

    outgoing = jnp.zeros_like(health).at[batch_index, safe_source].add(
        jnp.where(enemy_hit, applied, jnp.float32(0.0))
    )
    incoming = jnp.zeros_like(health).at[batch_index, safe_target].add(
        jnp.where(target_valid, applied, jnp.float32(0.0))
    )

    return per_actor_reward_components_from_totals(
        outgoing,
        incoming,
        teams,
        health,
        memory,
    )


def per_actor_reward_components_from_totals(
    entity_damage_dealt: jax.Array,
    entity_damage_received: jax.Array,
    team_id: jax.Array,
    health_after: jax.Array,
    memory: MultiActorRewardMemory,
) -> tuple[RewardComponents, MultiActorRewardMemory]:
    """Build actor rewards from runtime-preserved entity damage totals.

    ``entity_damage_dealt`` must already exclude friendly damage. Incoming
    damage remains source-agnostic so environmental damage is still charged
    to its victim. This is the production seam used after the Arsenal
    microtick scan has reduced individual damage packets.
    """

    dealt = jnp.asarray(entity_damage_dealt, dtype=jnp.float32)
    received = jnp.asarray(entity_damage_received, dtype=jnp.float32)
    teams = jnp.asarray(team_id, dtype=jnp.int32)
    health = jnp.asarray(health_after, dtype=jnp.float32)
    if (
        dealt.shape != health.shape
        or received.shape != health.shape
        or teams.shape != health.shape
    ):
        raise ValueError(
            "entity damage totals, team_id, and health_after must have shape [B,N]"
        )
    if (
        memory.completion_awarded.shape != health.shape
        or memory.death_awarded.shape != health.shape
    ):
        raise ValueError("reward memory must match the health entity axis")

    enemy = teams[:, :, None] != teams[:, None, :]
    living_enemy = enemy & (health[:, None, :] > jnp.float32(0.0))
    completion = ~jnp.any(living_enemy, axis=2) & ~memory.completion_awarded
    death = (health <= jnp.float32(0.0)) & ~memory.death_awarded
    next_memory = MultiActorRewardMemory(
        completion_awarded=memory.completion_awarded | completion,
        death_awarded=memory.death_awarded | death,
    )
    return (
        RewardComponents(
            target_damage=jnp.maximum(dealt, jnp.float32(0.0)),
            agent_damage=jnp.maximum(received, jnp.float32(0.0)),
            completion=completion,
            death=death,
        ),
        next_memory,
    )


def compose_per_actor_reward(
    components: RewardComponents,
    params: CombatParams,
) -> jax.Array:
    """Apply the existing reward weights independently to every actor row."""

    return compose_reward(components, params)


__all__ = [
    "MultiActorRewardMemory",
    "compose_per_actor_reward",
    "empty_multi_actor_reward_memory",
    "per_actor_reward_components",
    "per_actor_reward_components_from_totals",
]
