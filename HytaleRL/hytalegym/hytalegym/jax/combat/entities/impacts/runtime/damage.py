"""Bounded query-major/entity-major damage resolution for crowd impacts."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.entities import ENTITY_CAPACITY, EntityCombatState
from hytalegym.jax.combat.entities.impacts.schema.contract import (
    IMPACT_DAMAGE_CAPACITY,
)
from hytalegym.jax.combat.mechanics import (
    CombatMechanicsRules,
    DamageResolution,
    apply_damage_forces,
    empty_damage_events,
    resolve_damage_events,
)


class DenseImpactDamage(NamedTuple):
    requested: jax.Array
    source_slot: jax.Array
    amount: jax.Array
    random_percentage: jax.Array
    damage_class: jax.Array
    cause: jax.Array
    knockback_velocity: jax.Array
    force_mode: jax.Array
    air_resistance: jax.Array
    air_resistance_max: jax.Array
    ground_resistance: jax.Array
    ground_resistance_max: jax.Array
    resistance_threshold: jax.Array
    resistance_style: jax.Array
    on_hit_resource_id: jax.Array
    on_hit_resource_delta: jax.Array


def apply_dense_impact_damage(
    combat: EntityCombatState,
    dense: DenseImpactDamage,
    rules: CombatMechanicsRules,
    row_enabled: jax.Array,
    *,
    random_keys: jax.Array | None = None,
) -> tuple[
    EntityCombatState,
    DamageResolution,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Pack at most 64 ordered hits, resolve, and atomically commit death."""

    batch, query_count, entity_count = dense.requested.shape
    if entity_count != ENTITY_CAPACITY:
        raise ValueError(f"dense damage entity axis must be {ENTITY_CAPACITY}")
    mask = dense.requested & row_enabled[:, None, None]
    count = jnp.sum(mask.astype(jnp.int32), axis=(1, 2))
    overflow = count > IMPACT_DAMAGE_CAPACITY
    valid = row_enabled & ~overflow
    flat_mask = mask.reshape((batch, -1))
    flat_size = query_count * entity_count
    score = jnp.where(
        flat_mask,
        flat_size - jnp.arange(flat_size, dtype=jnp.int32)[None, :],
        jnp.int32(-1),
    )
    score, index = jax.lax.top_k(score, IMPACT_DAMAGE_CAPACITY)
    requested = (score >= 0) & valid[:, None]
    target = jnp.broadcast_to(
        jnp.arange(entity_count, dtype=jnp.int32)[None, None, :],
        mask.shape,
    )
    source = jnp.broadcast_to(
        dense.source_slot[..., None],
        mask.shape,
    )

    def query_scalar(value):
        return _gather(
            jnp.broadcast_to(value[..., None], mask.shape),
            index,
        )

    events = empty_damage_events(batch, IMPACT_DAMAGE_CAPACITY)._replace(
        requested=requested,
        source_entity_id=_gather(source, index),
        target_entity_id=_gather(target, index),
        amount=_gather(dense.amount, index),
        random_percentage=query_scalar(dense.random_percentage),
        damage_class=query_scalar(dense.damage_class),
        cause=query_scalar(dense.cause),
        knockback_velocity=_gather(dense.knockback_velocity, index),
        force_mode=query_scalar(dense.force_mode),
        air_resistance=query_scalar(dense.air_resistance),
        air_resistance_max=query_scalar(dense.air_resistance_max),
        ground_resistance=query_scalar(dense.ground_resistance),
        ground_resistance_max=query_scalar(dense.ground_resistance_max),
        resistance_threshold=query_scalar(dense.resistance_threshold),
        resistance_style=query_scalar(dense.resistance_style),
        on_hit_resource_id=query_scalar(dense.on_hit_resource_id),
        on_hit_resource_delta=query_scalar(dense.on_hit_resource_delta),
    )
    mechanics, health, resolution = resolve_damage_events(
        combat.mechanics,
        combat.roster.health,
        combat.roster.position,
        combat.roster.yaw_degrees,
        events,
        rules,
        random_keys=random_keys,
    )
    mechanics = apply_damage_forces(mechanics, resolution)
    mechanics_failed = mechanics.failure_bits != jnp.uint32(0)
    commit = valid & ~mechanics_failed
    newly_dead = (
        commit[:, None]
        & combat.roster.active
        & combat.roster.damageable
        & ~combat.roster.dead
        & (health <= 0.0)
    )
    candidate = EntityCombatState(
        roster=combat.roster._replace(
            health=health,
            dead=combat.roster.dead | newly_dead,
        ),
        mechanics=mechanics,
    )
    result = _select_tree(commit, candidate, combat)
    return (
        result,
        resolution,
        jnp.where(commit, count, jnp.int32(0)),
        overflow,
        mechanics_failed,
    )


def _gather(value: jax.Array, index: jax.Array) -> jax.Array:
    flat = value.reshape((value.shape[0], -1) + value.shape[3:])
    trailing = (1,) * (flat.ndim - 2)
    gather_index = index.reshape(index.shape + trailing)
    gather_index = jnp.broadcast_to(
        gather_index,
        index.shape + flat.shape[2:],
    )
    return jnp.take_along_axis(flat, gather_index, axis=1)


def _select_tree(mask, selected, fallback):
    return jax.tree_util.tree_map(
        lambda yes, no: jnp.where(
            mask.reshape((mask.shape[0],) + (1,) * (yes.ndim - 1)),
            yes,
            no,
        ),
        selected,
        fallback,
    )
