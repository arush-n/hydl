"""Shared all-entity target filtering and bounded Arsenal damage packing."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.contract import (
    AREA_TARGET_ENEMIES,
    AREA_TARGET_OWNER,
    AREA_TARGET_TEAM,
    DEPLOYABLE_ATTACK_ENEMIES,
    DEPLOYABLE_ATTACK_OWNER,
    DEPLOYABLE_ATTACK_TEAM,
    IMPACT_DAMAGE_CAPACITY,
)
from hytalegym.jax.combat.mechanics import DamageEvents, empty_damage_events
from hytalegym.jax.combat.targeting import TEAM_NONE


def relationship_target_mask(
    health: jax.Array,
    source_entity_id: jax.Array,
    *,
    entity_team_id: jax.Array | None,
    friendly_fire: bool | jax.Array,
) -> jax.Array:
    """Return every live non-owner target admitted by scenario team policy."""

    if health.ndim != 2:
        raise ValueError("health must have shape (batch, entity)")
    batch, entity_count = health.shape
    if source_entity_id.ndim != 2 or source_entity_id.shape[0] != batch:
        raise ValueError("source_entity_id must have shape (batch, query)")
    if entity_team_id is None:
        team_id = jnp.full(
            (batch, entity_count),
            TEAM_NONE,
            dtype=jnp.int32,
        )
    else:
        team_id = jnp.asarray(entity_team_id)
        if team_id.dtype != jnp.int32:
            raise TypeError(
                f"entity_team_id must have dtype int32, got {team_id.dtype}"
            )
        if team_id.shape != (batch, entity_count):
            raise ValueError("entity_team_id must match the combat entity axis")
    friendly = jnp.asarray(friendly_fire, dtype=jnp.bool_)
    try:
        friendly = jnp.broadcast_to(friendly, source_entity_id.shape)
    except ValueError as exc:
        raise ValueError("friendly_fire must broadcast to (batch, query)") from exc

    safe_source = jnp.clip(source_entity_id, 0, entity_count - 1)
    source_team = jnp.take_along_axis(team_id, safe_source, axis=1)
    same_team = (source_team[..., None] != jnp.int32(TEAM_NONE)) & (
        team_id[:, None, :] == source_team[..., None]
    )
    entity_id = jnp.arange(entity_count, dtype=jnp.int32)[None, None, :]
    return (
        (health[:, None, :] > jnp.float32(0.0))
        & (entity_id != source_entity_id[..., None])
        & (friendly[..., None] | ~same_team)
    )


def typed_relationship_target_mask(
    health: jax.Array,
    source_entity_id: jax.Array,
    target_mask: jax.Array,
    *,
    entity_team_id: jax.Array | None,
) -> jax.Array:
    """Apply explicit owner/team/enemy relationship-category bits."""

    if health.ndim != 2:
        raise ValueError("health must have shape (batch, entity)")
    batch, entity_count = health.shape
    if source_entity_id.ndim != 2 or source_entity_id.shape[0] != batch:
        raise ValueError("source_entity_id must have shape (batch, query)")
    mask = jnp.asarray(target_mask, dtype=jnp.int32)
    if mask.shape != source_entity_id.shape:
        raise ValueError("target_mask must match source_entity_id")
    if entity_team_id is None:
        team_id = jnp.full(
            (batch, entity_count),
            TEAM_NONE,
            dtype=jnp.int32,
        )
    else:
        team_id = jnp.asarray(entity_team_id)
        if team_id.dtype != jnp.int32:
            raise TypeError(
                f"entity_team_id must have dtype int32, got {team_id.dtype}"
            )
        if team_id.shape != (batch, entity_count):
            raise ValueError("entity_team_id must match the combat entity axis")
    safe_source = jnp.clip(source_entity_id, 0, entity_count - 1)
    source_team = jnp.take_along_axis(team_id, safe_source, axis=1)
    same_team = (source_team[..., None] != jnp.int32(TEAM_NONE)) & (
        team_id[:, None, :] == source_team[..., None]
    )
    entity_id = jnp.arange(entity_count, dtype=jnp.int32)[None, None, :]
    owner = entity_id == source_entity_id[..., None]
    team = ~owner & same_team
    enemies = ~owner & ~same_team
    admitted = (
        (owner & ((mask[..., None] & AREA_TARGET_OWNER) != 0))
        | (team & ((mask[..., None] & AREA_TARGET_TEAM) != 0))
        | (enemies & ((mask[..., None] & AREA_TARGET_ENEMIES) != 0))
    )
    return (health[:, None, :] > jnp.float32(0.0)) & admitted


def native_deployable_attack_mask(
    health: jax.Array,
    source_entity_id: jax.Array,
    attack_flags: jax.Array,
    *,
    entity_team_id: jax.Array,
    area_effect_eligible: jax.Array,
) -> jax.Array:
    """Apply DeployableAoeConfig.canAttackEntity's ordered predicate.

    ``entity_team_id`` is admitted only when the scenario explicitly attests
    that its equivalence classes exactly represent owner EntityGroup
    availability and membership. ``TEAM_NONE`` then means the owner has no
    group; it is never synthesized here from missing evidence.
    """

    if health.ndim != 2:
        raise ValueError("health must have shape (batch, entity)")
    batch, entity_count = health.shape
    if source_entity_id.ndim != 2 or source_entity_id.shape[0] != batch:
        raise ValueError("source_entity_id must have shape (batch, query)")
    flags = jnp.asarray(attack_flags, dtype=jnp.int32)
    if flags.shape != source_entity_id.shape:
        raise ValueError("attack_flags must match source_entity_id")
    team_id = jnp.asarray(entity_team_id)
    if team_id.dtype != jnp.int32:
        raise TypeError(f"entity_team_id must have dtype int32, got {team_id.dtype}")
    if team_id.shape != (batch, entity_count):
        raise ValueError("entity_team_id must match the combat entity axis")
    eligible = jnp.asarray(area_effect_eligible)
    if eligible.dtype != jnp.bool_:
        raise TypeError(
            f"area_effect_eligible must have dtype bool, got {eligible.dtype}"
        )
    if eligible.shape != (batch, entity_count):
        raise ValueError("area_effect_eligible must match the combat entity axis")

    safe_source = jnp.clip(source_entity_id, 0, entity_count - 1)
    source_team = jnp.take_along_axis(team_id, safe_source, axis=1)
    source_has_group = source_team != jnp.int32(TEAM_NONE)
    entity_id = jnp.arange(entity_count, dtype=jnp.int32)[None, None, :]
    owner = entity_id == source_entity_id[..., None]
    same_group = source_has_group[..., None] & (
        team_id[:, None, :] == source_team[..., None]
    )
    attack_owner = (flags & DEPLOYABLE_ATTACK_OWNER) != 0
    attack_team = (flags & DEPLOYABLE_ATTACK_TEAM) != 0
    attack_enemies = (flags & DEPLOYABLE_ATTACK_ENEMIES) != 0

    owner_rejected = owner & ~attack_owner[..., None]
    # Native returns immediately from the !AttackTeam branch: a missing owner
    # group admits the non-owner target, while an existing group rejects its
    # members. AttackEnemies is not consulted on either path.
    after_owner = jnp.where(
        (~attack_team)[..., None],
        (~source_has_group)[..., None] | ~same_group,
        attack_enemies[..., None],
    )
    return eligible[:, None, :] & ~owner_rejected & after_owner


def empty_dense_damage_events(
    batch_size: int,
    query_count: int,
    entity_count: int,
) -> DamageEvents:
    """Return a DamageEvents tree with dense ``(B,Q,N[,3])`` fields."""

    flat = empty_damage_events(
        batch_size,
        query_count * entity_count,
    )
    return jax.tree_util.tree_map(
        lambda value: value.reshape(
            (batch_size, query_count, entity_count) + value.shape[2:]
        ),
        flat,
    )


def pack_dense_damage_events(
    dense: DamageEvents,
) -> tuple[DamageEvents, jax.Array]:
    """Pack query-major/entity-slot hits into 64 slots, atomically on overflow."""

    requested = jnp.asarray(dense.requested, dtype=jnp.bool_)
    if requested.ndim != 3:
        raise ValueError("dense requested mask must have shape (batch, query, entity)")
    batch, query_count, entity_count = requested.shape
    flat_size = query_count * entity_count
    padded_size = max(flat_size, IMPACT_DAMAGE_CAPACITY)
    flat_requested = requested.reshape((batch, flat_size))
    flat_requested = jnp.pad(
        flat_requested,
        ((0, 0), (0, padded_size - flat_size)),
        constant_values=False,
    )
    position = jnp.arange(padded_size, dtype=jnp.int32)
    score = jnp.where(
        flat_requested,
        jnp.int32(padded_size) - position[None, :],
        jnp.int32(-1),
    )
    score, index = jax.lax.top_k(score, IMPACT_DAMAGE_CAPACITY)
    overflow = (
        jnp.sum(requested.astype(jnp.int32), axis=(1, 2)) > IMPACT_DAMAGE_CAPACITY
    )
    packed_requested = (score >= 0) & ~overflow[:, None]
    empty = empty_damage_events(batch, IMPACT_DAMAGE_CAPACITY)

    def pack_field(value: jax.Array, empty_value: jax.Array) -> jax.Array:
        if value.shape[:3] != requested.shape:
            raise ValueError(
                "every dense damage field must share (batch, query, entity)"
            )
        trailing = value.shape[3:]
        flat = value.reshape((batch, flat_size) + trailing)
        gather_index = jnp.minimum(index, flat_size - 1)
        gather_index = gather_index.reshape(
            (batch, IMPACT_DAMAGE_CAPACITY) + (1,) * len(trailing)
        )
        gather_index = jnp.broadcast_to(
            gather_index,
            (batch, IMPACT_DAMAGE_CAPACITY) + trailing,
        )
        packed = jnp.take_along_axis(flat, gather_index, axis=1)
        mask = packed_requested.reshape(packed_requested.shape + (1,) * len(trailing))
        return jnp.where(mask, packed, empty_value)

    values = []
    for field in dense._fields:
        if field == "requested":
            values.append(packed_requested)
        else:
            values.append(
                pack_field(
                    getattr(dense, field),
                    getattr(empty, field),
                )
            )
    return DamageEvents(*values), overflow
