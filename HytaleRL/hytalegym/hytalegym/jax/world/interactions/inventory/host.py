"""Host projection from native sparse inventory frames to JAX tokens."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.interactions.inventory.tokens import (
    INVENTORY_TOKEN_PROVENANCE_NATIVE,
    ActorInventoryTokens,
    produce_actor_inventory_tokens,
)


NATIVE_INVENTORY_SCHEMA = "hytalerl_native_inventory_v2"
NATIVE_INVENTORY_VERSION = 2
NATIVE_INVENTORY_CONTAINER_ORDER = (
    "storage",
    "armor",
    "hotbar",
    "utility",
    "tools",
    "backpack",
)
NATIVE_INVENTORY_SECTION_IDS = (-2, -3, -1, -5, -8, -9)
NATIVE_INVENTORY_ACTIVE_CONTAINERS = {
    "hotbar": "hotbar",
    "utility": "utility",
    "tools": "tools",
}


def native_inventory_frames_to_actor_tokens(
    frames: Sequence[Sequence[Mapping[str, Any]]],
    *,
    actor_legal,
    container_slot_capacities: Sequence[int],
    token_capacity: int,
    native_contract_sha256: str,
    required_container_mask=None,
    metadata_words: int = 8,
) -> ActorInventoryTokens:
    """Project validated native-v2 frames into a configured JAX layout.

    This function is a host boundary and is intentionally not JIT compiled.
    Native frames expose whether metadata exists but not its digest. Such
    stacks retain ``metadata_present=True`` with
    ``metadata_hash_valid=False`` and zero digest words.
    """

    rows, batch, actors = _frame_grid(frames)
    legal = np.asarray(actor_legal)
    if legal.dtype != np.bool_ or legal.shape != (batch, actors):
        raise TypeError("actor_legal must be boolean [B, A]")
    maxima = _configured_capacities(container_slot_capacities)
    words = _positive_int(metadata_words, "metadata_words")
    expected_contract = _sha256(native_contract_sha256)

    offsets = np.cumsum((0, *maxima[:-1]), dtype=np.int32)
    source_slots = int(sum(maxima))
    slot_container_id = np.empty((source_slots,), dtype=np.int32)
    slot_container_index = np.empty((source_slots,), dtype=np.int32)
    for container_id, (offset, capacity) in enumerate(
        zip(offsets, maxima, strict=True)
    ):
        end = int(offset) + capacity
        slot_container_id[int(offset) : end] = container_id
        slot_container_index[int(offset) : end] = np.arange(
            capacity,
            dtype=np.int32,
        )

    shape = (batch, actors, source_slots)
    item_id = np.full(shape, -1, dtype=np.int32)
    quantity = np.zeros(shape, dtype=np.int32)
    durability = np.zeros(shape, dtype=np.float32)
    maximum = np.zeros(shape, dtype=np.float32)
    metadata_hash = np.zeros(shape + (words,), dtype=np.uint32)
    metadata_present = np.zeros(shape, dtype=np.bool_)
    metadata_hash_valid = np.zeros(shape, dtype=np.bool_)
    source_available = np.zeros((batch, actors), dtype=np.bool_)
    container_available = np.zeros(
        (batch, actors, len(maxima)),
        dtype=np.bool_,
    )
    container_capacity = np.zeros_like(
        container_available,
        dtype=np.int32,
    )
    active_slot = np.full(
        container_available.shape,
        -1,
        dtype=np.int32,
    )

    for batch_index, row in enumerate(rows):
        for actor_index, frame in enumerate(row):
            if not legal[batch_index, actor_index]:
                continue
            _populate_frame(
                frame,
                expected_contract=expected_contract,
                maxima=maxima,
                offsets=offsets,
                item_id=item_id[batch_index, actor_index],
                quantity=quantity[batch_index, actor_index],
                durability=durability[batch_index, actor_index],
                maximum=maximum[batch_index, actor_index],
                metadata_present=metadata_present[batch_index, actor_index],
                metadata_hash_valid=metadata_hash_valid[
                    batch_index,
                    actor_index,
                ],
                source_available=source_available[
                    batch_index : batch_index + 1,
                    actor_index : actor_index + 1,
                ],
                container_available=container_available[
                    batch_index,
                    actor_index,
                ],
                container_capacity=container_capacity[
                    batch_index,
                    actor_index,
                ],
                active_slot=active_slot[batch_index, actor_index],
            )

    return produce_actor_inventory_tokens(
        jnp.asarray(item_id),
        jnp.asarray(quantity),
        jnp.asarray(durability),
        jnp.asarray(maximum),
        jnp.asarray(metadata_hash),
        jnp.asarray(metadata_present),
        jnp.asarray(metadata_hash_valid),
        slot_container_id=jnp.asarray(slot_container_id),
        slot_container_index=jnp.asarray(slot_container_index),
        source_available=jnp.asarray(source_available),
        container_available=jnp.asarray(container_available),
        container_capacity=jnp.asarray(container_capacity),
        active_container_slot=jnp.asarray(active_slot),
        actor_legal=jnp.asarray(legal),
        token_capacity=token_capacity,
        provenance=INVENTORY_TOKEN_PROVENANCE_NATIVE,
        required_container_mask=required_container_mask,
    )


def _populate_frame(
    frame: Mapping[str, Any],
    *,
    expected_contract: str,
    maxima: tuple[int, ...],
    offsets: np.ndarray,
    item_id: np.ndarray,
    quantity: np.ndarray,
    durability: np.ndarray,
    maximum: np.ndarray,
    metadata_present: np.ndarray,
    metadata_hash_valid: np.ndarray,
    source_available: np.ndarray,
    container_available: np.ndarray,
    container_capacity: np.ndarray,
    active_slot: np.ndarray,
) -> None:
    if not isinstance(frame, Mapping):
        raise TypeError("native inventory frame must be an object")
    if (
        frame.get("schema") != NATIVE_INVENTORY_SCHEMA
        or frame.get("version") != NATIVE_INVENTORY_VERSION
    ):
        raise ValueError("native inventory frame schema/version mismatch")
    if _sha256(frame.get("contract_sha256")) != expected_contract:
        raise ValueError("native inventory frame contract mismatch")
    available = _boolean(frame.get("available"), "available")
    reason = frame.get("unavailable_reason")
    if not isinstance(reason, str):
        raise TypeError("unavailable_reason must be a string")
    containers = frame.get("containers")
    if not _sequence(containers):
        raise TypeError("containers must be an array")
    if not available:
        if not reason or containers:
            raise ValueError("unavailable inventory frame is inconsistent")
        return
    if reason or len(containers) != len(maxima):
        raise ValueError("available inventory frame is inconsistent")
    active = frame.get("active_slots")
    if not isinstance(active, Mapping):
        raise TypeError("active_slots must be an object")
    source_available[...] = True

    for container_id, (
        row,
        name,
        section_id,
        configured_capacity,
        offset,
    ) in enumerate(
        zip(
            containers,
            NATIVE_INVENTORY_CONTAINER_ORDER,
            NATIVE_INVENTORY_SECTION_IDS,
            maxima,
            offsets,
            strict=True,
        )
    ):
        if not isinstance(row, Mapping):
            raise TypeError(f"{name} container must be an object")
        if row.get("name") != name or row.get("section_id") != section_id:
            raise ValueError("native inventory container order drift")
        row_available = _boolean(row.get("available"), f"{name}.available")
        row_reason = row.get("unavailable_reason")
        if not isinstance(row_reason, str):
            raise TypeError(f"{name}.unavailable_reason must be a string")
        capacity = _nonnegative_int(row.get("capacity"), f"{name}.capacity")
        occupied = row.get("occupied_slots")
        if not _sequence(occupied):
            raise TypeError(f"{name}.occupied_slots must be an array")
        native_active = (
            _integer(active.get(NATIVE_INVENTORY_ACTIVE_CONTAINERS[name]), name)
            if name in NATIVE_INVENTORY_ACTIVE_CONTAINERS
            else -1
        )
        if not row_available:
            if not row_reason or capacity != 0 or occupied or native_active != -1:
                raise ValueError(f"unavailable {name} container is inconsistent")
            continue
        if row_reason or capacity > configured_capacity:
            raise ValueError(f"{name} exceeds configured slot capacity")
        if native_active < -1 or native_active >= capacity:
            raise ValueError(f"{name} active slot is outside native capacity")
        container_available[container_id] = True
        container_capacity[container_id] = capacity
        active_slot[container_id] = native_active

        previous = -1
        for slot_row in occupied:
            if not isinstance(slot_row, Mapping):
                raise TypeError(f"{name} occupied slot must be an object")
            slot = _nonnegative_int(slot_row.get("slot"), f"{name}.slot")
            if slot <= previous or slot >= capacity:
                raise ValueError(f"{name} occupied slot order/capacity drift")
            previous = slot
            runtime_index = _integer(
                slot_row.get("item_runtime_index"),
                f"{name}.item_runtime_index",
            )
            if runtime_index <= 0:
                raise ValueError(
                    f"{name}.item_runtime_index is unmapped or invalid"
                )
            item_name = slot_row.get("item_id")
            if not isinstance(item_name, str) or not item_name:
                raise ValueError(f"{name}.item_id must be nonempty")
            stack_quantity = _positive_int(
                slot_row.get("quantity"),
                f"{name}.quantity",
            )
            stack_durability = _finite_nonnegative(
                slot_row.get("durability"),
                f"{name}.durability",
            )
            stack_maximum = _finite_nonnegative(
                slot_row.get("max_durability"),
                f"{name}.max_durability",
            )
            if stack_durability > stack_maximum:
                raise ValueError(f"{name} durability exceeds maximum")
            has_metadata = _boolean(
                slot_row.get("metadata_present"),
                f"{name}.metadata_present",
            )
            index = int(offset) + slot
            item_id[index] = runtime_index
            quantity[index] = stack_quantity
            durability[index] = stack_durability
            maximum[index] = stack_maximum
            metadata_present[index] = has_metadata
            metadata_hash_valid[index] = not has_metadata


def _frame_grid(
    frames: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[list[list[Mapping[str, Any]]], int, int]:
    if not _sequence(frames) or not frames:
        raise ValueError("frames must be a non-empty [B][A] array")
    rows = []
    actor_count = None
    for row in frames:
        if not _sequence(row) or not row:
            raise ValueError("each frame batch row must contain actors")
        current = list(row)
        if actor_count is None:
            actor_count = len(current)
        elif len(current) != actor_count:
            raise ValueError("frame actor axis must be rectangular")
        rows.append(current)
    assert actor_count is not None
    return rows, len(rows), actor_count


def _configured_capacities(value: Sequence[int]) -> tuple[int, ...]:
    if not _sequence(value) or len(value) != len(
        NATIVE_INVENTORY_CONTAINER_ORDER
    ):
        raise ValueError("container_slot_capacities must contain six values")
    result = tuple(
        _nonnegative_int(item, "container_slot_capacities") for item in value
    )
    if sum(result) <= 0:
        raise ValueError("configured inventory must contain at least one slot")
    return result


def _sha256(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("native inventory contract SHA-256 is invalid")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise ValueError("native inventory contract SHA-256 is invalid") from error
    return value.upper()


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if not -(2**31) <= value <= (2**31 - 1):
        raise ValueError(f"{label} exceeds int32")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    result = _integer(value, label)
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _positive_int(value: Any, label: str) -> int:
    result = _integer(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean")
    return value


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    )


__all__ = [
    "NATIVE_INVENTORY_ACTIVE_CONTAINERS",
    "NATIVE_INVENTORY_CONTAINER_ORDER",
    "NATIVE_INVENTORY_SCHEMA",
    "NATIVE_INVENTORY_SECTION_IDS",
    "NATIVE_INVENTORY_VERSION",
    "native_inventory_frames_to_actor_tokens",
]
