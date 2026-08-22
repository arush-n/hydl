"""Fixed-shape JAX transport for model-neutral native lifecycle facts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from types import MappingProxyType
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.worldgen.native_npc_traces import (
    NATIVE_NPC_LIFECYCLE_ALL_SOURCE_BITS,
    NATIVE_NPC_LIFECYCLE_EVENT_CAPACITY,
)


NATIVE_LIFECYCLE_KIND_ID = MappingProxyType(
    {
        "stat_changed": 1,
        "status_added": 2,
        "status_removed": 3,
        "status_refreshed": 4,
        "inventory_transaction": 5,
        "inventory_slot": 6,
        "projectile_spawned": 7,
        "projectile_despawned": 8,
        "projectile_impacted": 9,
        "actor_death": 10,
        "actor_revived": 11,
    }
)
NATIVE_LIFECYCLE_ORIGIN_ID = MappingProxyType(
    {
        "server_boundary_state_delta": 1,
        "server_inventory_event": 2,
        "server_death_component": 3,
        "server_inter_row_state_delta": 4,
    }
)


class NativeLifecycleEventBatch(NamedTuple):
    """Actor-major tensors; strings use stable IDs and padded cells are masked."""

    present: jax.Array
    kind: jax.Array
    origin: jax.Array
    subject_id: jax.Array
    key_id: jax.Array
    index: jax.Array
    value_before: jax.Array
    value_after: jax.Array
    auxiliary_0: jax.Array
    auxiliary_1: jax.Array
    text_before_id: jax.Array
    text_after_id: jax.Array
    successful: jax.Array
    complete: jax.Array
    entity_index: jax.Array
    entity_uuid_present: jax.Array
    entity_uuid_words: jax.Array
    owner_uuid_present: jax.Array
    owner_uuid_words: jax.Array
    position_available: jax.Array
    position: jax.Array
    flags: jax.Array
    count: jax.Array
    overflow: jax.Array
    source_available: jax.Array
    source_partial: jax.Array


def native_lifecycle_symbol_id(value: str) -> int:
    """Stable uint32 ID; zero remains reserved for the empty/padded symbol."""

    if not isinstance(value, str):
        raise TypeError("native lifecycle symbols must be strings")
    if not value:
        return 0
    result = int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "big")
    return result or 1


def native_lifecycle_symbol_table(
    rows: Sequence[Sequence[object]],
) -> Mapping[int, str]:
    """Return the collision-checked host lookup for one set of event rows."""

    symbols: dict[int, str] = {0: ""}
    for row in rows:
        for event in row:
            for value in (
                event.subject,
                event.key,
                event.text_before,
                event.text_after,
            ):
                symbol = native_lifecycle_symbol_id(value)
                previous = symbols.setdefault(symbol, value)
                if previous != value:
                    raise ValueError("native lifecycle symbol hash collision")
    return MappingProxyType(symbols)


def pack_native_lifecycle_events(
    rows: Sequence[Sequence[object]],
    *,
    count: Sequence[int] | np.ndarray,
    overflow: Sequence[bool] | np.ndarray,
    source_available: Sequence[int] | np.ndarray,
    source_partial: Sequence[int] | np.ndarray,
    capacity: int = NATIVE_NPC_LIFECYCLE_EVENT_CAPACITY,
) -> NativeLifecycleEventBatch:
    """Validate on the host, then transfer one bounded numeric PyTree to JAX."""

    if (
        isinstance(capacity, bool)
        or not 1 <= capacity <= NATIVE_NPC_LIFECYCLE_EVENT_CAPACITY
    ):
        raise ValueError("lifecycle event capacity is out of range")
    events = tuple(tuple(row) for row in rows)
    row_count = len(events)
    lengths = np.asarray([len(row) for row in events], dtype=np.int32)
    totals = np.asarray(count, dtype=np.int32)
    dropped = np.asarray(overflow, dtype=np.bool_)
    available = np.asarray(source_available, dtype=np.int32)
    partial = np.asarray(source_partial, dtype=np.int32)
    if any(
        value.shape != (row_count,) for value in (totals, dropped, available, partial)
    ):
        raise ValueError("lifecycle metadata must be row-aligned")
    if (
        np.any(lengths > capacity)
        or np.any(totals < lengths)
        or np.any(dropped != (totals > lengths))
        or np.any((available < 0) | (available > NATIVE_NPC_LIFECYCLE_ALL_SOURCE_BITS))
        or np.any(partial & ~available)
    ):
        raise ValueError("lifecycle count, overflow, or source masks are invalid")
    native_lifecycle_symbol_table(events)

    shape = (row_count, capacity)
    arrays = {
        "present": np.zeros(shape, dtype=np.bool_),
        "kind": np.zeros(shape, dtype=np.uint8),
        "origin": np.zeros(shape, dtype=np.uint8),
        "subject_id": np.zeros(shape, dtype=np.uint32),
        "key_id": np.zeros(shape, dtype=np.uint32),
        "index": np.full(shape, -1, dtype=np.int32),
        "value_before": np.zeros(shape, dtype=np.float32),
        "value_after": np.zeros(shape, dtype=np.float32),
        "auxiliary_0": np.zeros(shape, dtype=np.float32),
        "auxiliary_1": np.zeros(shape, dtype=np.float32),
        "text_before_id": np.zeros(shape, dtype=np.uint32),
        "text_after_id": np.zeros(shape, dtype=np.uint32),
        "successful": np.zeros(shape, dtype=np.bool_),
        "complete": np.zeros(shape, dtype=np.bool_),
        "entity_index": np.full(shape, -1, dtype=np.int32),
        "entity_uuid_present": np.zeros(shape, dtype=np.bool_),
        "owner_uuid_present": np.zeros(shape, dtype=np.bool_),
        "position_available": np.zeros(shape, dtype=np.bool_),
        "flags": np.zeros(shape, dtype=np.int32),
    }
    entity_uuid = np.zeros((*shape, 4), dtype=np.uint32)
    owner_uuid = np.zeros((*shape, 4), dtype=np.uint32)
    position = np.zeros((*shape, 3), dtype=np.float32)
    for row_index, row in enumerate(events):
        for event_index, event in enumerate(row):
            cell = row_index, event_index
            arrays["present"][cell] = True
            arrays["kind"][cell] = NATIVE_LIFECYCLE_KIND_ID[event.kind]
            arrays["origin"][cell] = NATIVE_LIFECYCLE_ORIGIN_ID[event.origin]
            for name in ("subject", "key", "text_before", "text_after"):
                arrays[f"{name}_id"][cell] = native_lifecycle_symbol_id(
                    getattr(event, name)
                )
            for name in (
                "index",
                "value_before",
                "value_after",
                "auxiliary_0",
                "auxiliary_1",
                "successful",
                "complete",
                "entity_index",
                "position_available",
                "flags",
            ):
                arrays[name][cell] = getattr(event, name)
            if event.entity_uuid is not None:
                arrays["entity_uuid_present"][cell] = True
                entity_uuid[cell] = _uuid_words(event.entity_uuid)
            if event.owner_uuid is not None:
                arrays["owner_uuid_present"][cell] = True
                owner_uuid[cell] = _uuid_words(event.owner_uuid)
            position[cell] = event.position

    return NativeLifecycleEventBatch(
        *(jnp.asarray(arrays[name]) for name in NativeLifecycleEventBatch._fields[:16]),
        jnp.asarray(entity_uuid),
        jnp.asarray(arrays["owner_uuid_present"]),
        jnp.asarray(owner_uuid),
        jnp.asarray(arrays["position_available"]),
        jnp.asarray(position),
        jnp.asarray(arrays["flags"]),
        jnp.asarray(totals),
        jnp.asarray(dropped),
        jnp.asarray(available),
        jnp.asarray(partial),
    )


@jax.jit
def native_lifecycle_kind_counts(events: NativeLifecycleEventBatch) -> jax.Array:
    """Count each stable kind ID per row, including zero only as padding."""

    one_hot = jax.nn.one_hot(
        events.kind, len(NATIVE_LIFECYCLE_KIND_ID) + 1, dtype=jnp.int32
    )
    return jnp.sum(one_hot * events.present[..., None], axis=1)


def _uuid_words(value: object) -> np.ndarray:
    return np.frombuffer(value.bytes, dtype=">u4").astype(np.uint32)


__all__ = [
    "NATIVE_LIFECYCLE_KIND_ID",
    "NATIVE_LIFECYCLE_ORIGIN_ID",
    "NativeLifecycleEventBatch",
    "native_lifecycle_kind_counts",
    "native_lifecycle_symbol_id",
    "native_lifecycle_symbol_table",
    "pack_native_lifecycle_events",
]
