"""Fixed-shape JAX projection of authoritative native transition events."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.worldgen.native_npc_traces import NATIVE_NPC_DAMAGE_EVENT_CAPACITY


_SOURCE_TYPE = {"other": 0, "entity": 1, "projectile": 2, "environment": 3}


class NativeDamageEventBatch(NamedTuple):
    """Actor-major event tensors; ``present`` masks every padded cell."""

    present: jax.Array
    source_type: jax.Array
    damage_cause_index: jax.Array
    initial_amount: jax.Array
    final_amount: jax.Array
    cancelled: jax.Array
    blocked: jax.Array
    actor_source: jax.Array
    actor_target: jax.Array
    source_entity_index: jax.Array
    target_entity_index: jax.Array
    projectile_entity_index: jax.Array
    hit_location_available: jax.Array
    hit_location: jax.Array
    health_available: jax.Array
    target_health_after: jax.Array
    target_max_health: jax.Array
    lethal: jax.Array
    count: jax.Array
    overflow: jax.Array


class NativeDamageTotals(NamedTuple):
    """Exact trace-relative totals from non-cancelled server events."""

    dealt: jax.Array
    received: jax.Array
    dealt_events: jax.Array
    received_events: jax.Array


def pack_native_damage_events(
    rows: Sequence[Sequence[object]],
    *,
    count: Sequence[int] | np.ndarray | None = None,
    overflow: Sequence[bool] | np.ndarray | None = None,
    capacity: int = NATIVE_NPC_DAMAGE_EVENT_CAPACITY,
) -> NativeDamageEventBatch:
    """Validate once on the host and transfer one fixed-shape PyTree to JAX."""

    if isinstance(capacity, bool) or not 1 <= capacity <= NATIVE_NPC_DAMAGE_EVENT_CAPACITY:
        raise ValueError("damage event capacity is out of range")
    events = tuple(tuple(row) for row in rows)
    row_count = len(events)
    lengths = np.asarray([len(row) for row in events], dtype=np.int32)
    if np.any(lengths > capacity):
        raise ValueError("damage event rows exceed the selected JAX capacity")
    totals = lengths if count is None else np.asarray(count, dtype=np.int32)
    dropped = np.zeros(row_count, dtype=np.bool_) if overflow is None else np.asarray(
        overflow, dtype=np.bool_
    )
    if totals.shape != (row_count,) or dropped.shape != (row_count,):
        raise ValueError("damage event count and overflow must be row-aligned")
    if np.any(totals < lengths) or np.any(dropped != (totals > lengths)):
        raise ValueError("damage event count/overflow is inconsistent")

    shape = (row_count, capacity)
    arrays: dict[str, np.ndarray] = {
        "present": np.zeros(shape, dtype=np.bool_),
        "source_type": np.zeros(shape, dtype=np.uint8),
        "damage_cause_index": np.zeros(shape, dtype=np.int32),
        "initial_amount": np.zeros(shape, dtype=np.float32),
        "final_amount": np.zeros(shape, dtype=np.float32),
        "cancelled": np.zeros(shape, dtype=np.bool_),
        "blocked": np.zeros(shape, dtype=np.bool_),
        "actor_source": np.zeros(shape, dtype=np.bool_),
        "actor_target": np.zeros(shape, dtype=np.bool_),
        "source_entity_index": np.full(shape, -1, dtype=np.int32),
        "target_entity_index": np.full(shape, -1, dtype=np.int32),
        "projectile_entity_index": np.full(shape, -1, dtype=np.int32),
        "hit_location_available": np.zeros(shape, dtype=np.bool_),
        "health_available": np.zeros(shape, dtype=np.bool_),
        "target_health_after": np.zeros(shape, dtype=np.float32),
        "target_max_health": np.zeros(shape, dtype=np.float32),
        "lethal": np.zeros(shape, dtype=np.bool_),
    }
    hit_location = np.zeros((*shape, 4), dtype=np.float32)
    for row_index, row in enumerate(events):
        for event_index, event in enumerate(row):
            arrays["present"][row_index, event_index] = True
            arrays["source_type"][row_index, event_index] = _SOURCE_TYPE[
                event.source_type
            ]
            for name in (
                "damage_cause_index",
                "initial_amount",
                "final_amount",
                "cancelled",
                "blocked",
                "actor_source",
                "actor_target",
                "source_entity_index",
                "target_entity_index",
                "projectile_entity_index",
                "hit_location_available",
                "health_available",
                "target_health_after",
                "target_max_health",
                "lethal",
            ):
                arrays[name][row_index, event_index] = getattr(event, name)
            hit_location[row_index, event_index] = event.hit_location
    return NativeDamageEventBatch(
        *(jnp.asarray(arrays[name]) for name in NativeDamageEventBatch._fields[:13]),
        jnp.asarray(hit_location),
        *(jnp.asarray(arrays[name]) for name in NativeDamageEventBatch._fields[14:18]),
        jnp.asarray(totals),
        jnp.asarray(dropped),
    )


@jax.jit
def native_damage_totals(events: NativeDamageEventBatch) -> NativeDamageTotals:
    admitted = events.present & ~events.cancelled
    return NativeDamageTotals(
        dealt=jnp.sum(
            jnp.where(admitted & events.actor_source, events.final_amount, 0.0),
            axis=1,
        ),
        received=jnp.sum(
            jnp.where(admitted & events.actor_target, events.final_amount, 0.0),
            axis=1,
        ),
        dealt_events=jnp.sum(admitted & events.actor_source, axis=1),
        received_events=jnp.sum(admitted & events.actor_target, axis=1),
    )


__all__ = [
    "NativeDamageEventBatch",
    "NativeDamageTotals",
    "native_damage_totals",
    "pack_native_damage_events",
]
