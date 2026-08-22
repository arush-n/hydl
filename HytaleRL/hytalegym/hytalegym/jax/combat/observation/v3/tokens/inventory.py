"""Actor-only semantic inventory rows for the dense Arsenal policy."""

from __future__ import annotations

import copy
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.inventory import (
    CONTAINER_COUNT,
    DEFAULT_CONTAINER_CAPACITIES,
    NATIVE_INVENTORY_SCHEMA,
    NATIVE_INVENTORY_VERSION,
    InventoryLayout,
    InventoryState,
    native_inventory_contract_sha256,
    parse_native_inventory_frame,
)
from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.world import (
    INVENTORY_TOKEN_PROVENANCE_SIMULATED,
    ActorInventoryTokens,
    actor_inventory_token_contract,
    native_inventory_frames_to_actor_tokens,
    produce_actor_inventory_tokens,
)


Array = jax.Array

INVENTORY_POLICY_TOKEN_CAPACITY = sum(DEFAULT_CONTAINER_CAPACITIES)
INVENTORY_POLICY_FLOAT_FEATURES = (
    "container_id",
    "container_slot_fraction",
    "item_id_low_16",
    "item_id_high_15",
    "quantity_log2_fraction",
    "durability_fraction",
    "active",
)
INVENTORY_POLICY_FLOAT_SIZE = len(INVENTORY_POLICY_FLOAT_FEATURES)
INVENTORY_POLICY_SOURCE_FIELDS = (
    "container_available",
    "container_capacity",
    "token_mask",
    "container_id",
    "container_slot",
    "item_id",
    "quantity",
    "durability_fraction",
    "active",
)

_DEFAULT_CAPACITIES = jnp.asarray(
    DEFAULT_CONTAINER_CAPACITIES,
    dtype=jnp.int32,
)
_NO_REQUIRED_CONTAINERS = jnp.zeros(
    (CONTAINER_COUNT,),
    dtype=jnp.bool_,
)

if not set(INVENTORY_POLICY_SOURCE_FIELDS).issubset(
    actor_inventory_token_contract()["output"]["policy_fields"]
):
    raise RuntimeError("actor inventory policy field contract drift")


class InventoryPolicyTokens(NamedTuple):
    """Backend-neutral actor holdings with no source or target diagnostics."""

    available: Array
    container_f32: Array
    container_mask: Array
    token_f32: Array
    token_mask: Array


def empty_inventory_policy_tokens(batch: int) -> InventoryPolicyTokens:
    """Return the fixed-shape unavailable inventory policy row."""

    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("batch must be a positive integer")
    return InventoryPolicyTokens(
        available=jnp.zeros((batch,), dtype=jnp.bool_),
        container_f32=jnp.zeros(
            (batch, CONTAINER_COUNT),
            dtype=jnp.float32,
        ),
        container_mask=jnp.zeros(
            (batch, CONTAINER_COUNT),
            dtype=jnp.bool_,
        ),
        token_f32=jnp.zeros(
            (
                batch,
                INVENTORY_POLICY_TOKEN_CAPACITY,
                INVENTORY_POLICY_FLOAT_SIZE,
            ),
            dtype=jnp.float32,
        ),
        token_mask=jnp.zeros(
            (batch, INVENTORY_POLICY_TOKEN_CAPACITY),
            dtype=jnp.bool_,
        ),
    )


def inventory_policy_flat_size() -> int:
    """Return the fixed dense width contributed by actor inventory."""

    return (
        1
        + 2 * CONTAINER_COUNT
        + INVENTORY_POLICY_TOKEN_CAPACITY
        + INVENTORY_POLICY_TOKEN_CAPACITY * INVENTORY_POLICY_FLOAT_SIZE
    )


def inventory_policy_tokens_from_state(
    state: InventoryState,
    layout: InventoryLayout,
    *,
    actor_valid: Array,
) -> InventoryPolicyTokens:
    """Project only entity zero from JAX inventory state into policy rows."""

    _validate_source_layout(layout)
    valid = jnp.asarray(actor_valid)
    batch = state.item_id.shape[0]
    if valid.dtype != jnp.bool_ or valid.shape != (batch,):
        raise TypeError("actor_valid must be boolean [B]")
    if state.item_id.ndim != 3 or state.item_id.shape[1] < 1:
        raise ValueError("inventory item_id must have shape [B, entity, slot]")
    if state.item_id.shape[2] < INVENTORY_POLICY_TOKEN_CAPACITY:
        raise ValueError(
            "inventory source slots are smaller than the policy token capacity"
        )

    # Slice before the public producer is called. Target inventories never
    # enter the actor-token call graph and therefore cannot be masked leaks.
    item_id = state.item_id[:, :1]
    quantity = state.quantity[:, :1]
    durability = state.durability[:, :1]
    maximum = state.max_durability[:, :1]
    metadata = state.metadata_hash[:, :1]
    present = (item_id >= 0) & (quantity > 0)
    metadata_present = jnp.any(metadata != 0, axis=3) & present
    container_available = jnp.ones(
        (batch, 1, CONTAINER_COUNT),
        dtype=jnp.bool_,
    )
    container_capacity = jnp.broadcast_to(
        layout.capacities[None, None, :],
        (batch, 1, CONTAINER_COUNT),
    )
    active_slots = jnp.full(
        (batch, 1, CONTAINER_COUNT),
        -1,
        dtype=jnp.int32,
    )
    active_slots = active_slots.at[:, 0, 2].set(
        state.active_hotbar_slot[:, 0]
    )
    active_slots = active_slots.at[:, 0, 3].set(
        state.active_utility_slot[:, 0]
    )
    active_slots = active_slots.at[:, 0, 4].set(
        state.active_tools_slot[:, 0]
    )
    source = produce_actor_inventory_tokens(
        item_id,
        quantity,
        durability,
        maximum,
        metadata,
        metadata_present,
        present,
        slot_container_id=layout.container_id,
        slot_container_index=layout.container_slot,
        source_available=(state.failure_bits == 0)[:, None],
        container_available=container_available,
        container_capacity=container_capacity,
        active_container_slot=active_slots,
        actor_legal=valid[:, None],
        token_capacity=INVENTORY_POLICY_TOKEN_CAPACITY,
        provenance=INVENTORY_TOKEN_PROVENANCE_SIMULATED,
        required_container_mask=_NO_REQUIRED_CONTAINERS,
    )
    return encode_inventory_policy_tokens(source)


def inventory_policy_tokens_from_native_frame(
    frame: Mapping[str, object] | None,
) -> InventoryPolicyTokens:
    """Project one validated native-v2 actor frame through the same encoder."""

    if frame is None:
        return empty_inventory_policy_tokens(1)
    parsed = parse_native_inventory_frame(frame)
    if (
        parsed["schema"] != NATIVE_INVENTORY_SCHEMA
        or parsed["version"] != NATIVE_INVENTORY_VERSION
    ):
        raise ValueError("inventory policy requires the native-v2 frame")
    semantic_frame = _semantic_native_frame(parsed)
    source = native_inventory_frames_to_actor_tokens(
        [[semantic_frame]],
        actor_legal=jnp.asarray([[True]], dtype=jnp.bool_),
        container_slot_capacities=DEFAULT_CONTAINER_CAPACITIES,
        token_capacity=INVENTORY_POLICY_TOKEN_CAPACITY,
        native_contract_sha256=native_inventory_contract_sha256(),
        required_container_mask=_NO_REQUIRED_CONTAINERS,
    )
    return encode_inventory_policy_tokens(source)


def encode_inventory_policy_tokens(
    source: ActorInventoryTokens,
    *,
    actor_index: int = 0,
) -> InventoryPolicyTokens:
    """Encode declared actor fields while excluding backend provenance."""

    if not isinstance(source, ActorInventoryTokens):
        raise TypeError("source must be ActorInventoryTokens")
    if source.available.ndim != 2:
        raise ValueError("source.available must have shape [B, actor]")
    batch, actors = source.available.shape
    if (
        isinstance(actor_index, bool)
        or not isinstance(actor_index, int)
        or not 0 <= actor_index < actors
    ):
        raise ValueError("actor_index is outside the source actor axis")
    expected = (batch, actors, INVENTORY_POLICY_TOKEN_CAPACITY)
    for name in (
        "token_mask",
        "container_id",
        "container_slot",
        "item_id",
        "quantity",
        "durability_fraction",
        "active",
    ):
        if getattr(source, name).shape != expected:
            raise ValueError(f"source.{name} must have shape {expected}")
    if source.container_available.shape != (
        batch,
        actors,
        CONTAINER_COUNT,
    ):
        raise ValueError("source.container_available shape drift")
    if source.container_capacity.shape != source.container_available.shape:
        raise ValueError("source.container_capacity shape drift")

    available = source.available[:, actor_index]
    container_mask = (
        source.container_available[:, actor_index] & available[:, None]
    )
    container_capacity = source.container_capacity[:, actor_index]
    configured = _DEFAULT_CAPACITIES[None, :]
    fixed_container = configured > 0
    capacity_valid = jnp.all(
        ~container_mask
        | ~fixed_container
        | (container_capacity <= configured),
        axis=1,
    )
    available &= capacity_valid
    container_mask &= available[:, None]
    fixed_fraction = container_capacity.astype(jnp.float32) / jnp.maximum(
        configured.astype(jnp.float32),
        1.0,
    )
    variable_fraction = container_capacity.astype(jnp.float32) / jnp.float32(
        jnp.iinfo(jnp.int32).max
    )
    container_f32 = jnp.where(
        container_mask,
        jnp.where(fixed_container, fixed_fraction, variable_fraction),
        jnp.float32(0.0),
    )

    token_mask = source.token_mask[:, actor_index] & available[:, None]
    container_id = source.container_id[:, actor_index]
    container_slot = source.container_slot[:, actor_index]
    item_id = source.item_id[:, actor_index]
    quantity = source.quantity[:, actor_index]
    safe_container = jnp.clip(container_id, 0, CONTAINER_COUNT - 1)
    token_capacity = jnp.take_along_axis(
        container_capacity,
        safe_container,
        axis=1,
    )
    features = jnp.stack(
        (
            container_id.astype(jnp.float32)
            / jnp.float32(CONTAINER_COUNT - 1),
            (container_slot.astype(jnp.float32) + 1.0)
            / (token_capacity.astype(jnp.float32) + 1.0),
            (item_id & jnp.int32(0xFFFF)).astype(jnp.float32)
            / jnp.float32(0xFFFF),
            ((item_id >> jnp.int32(16)) & jnp.int32(0x7FFF)).astype(
                jnp.float32
            )
            / jnp.float32(0x7FFF),
            jnp.log2(quantity.astype(jnp.float32) + 1.0)
            / jnp.float32(31.0),
            source.durability_fraction[:, actor_index],
            source.active[:, actor_index].astype(jnp.float32),
        ),
        axis=2,
    )
    token_f32 = jnp.where(
        token_mask[..., None],
        jnp.clip(features, 0.0, 1.0),
        jnp.float32(0.0),
    )
    return InventoryPolicyTokens(
        available=available,
        container_f32=container_f32,
        container_mask=container_mask,
        token_f32=token_f32,
        token_mask=token_mask,
    )


def mask_inventory_policy_tokens(
    tokens: InventoryPolicyTokens,
    valid: Array,
) -> InventoryPolicyTokens:
    """Apply an outer environment-validity gate."""

    gate = jnp.asarray(valid)
    if gate.dtype != jnp.bool_ or gate.shape != tokens.available.shape:
        raise TypeError("valid must be boolean [B]")
    available = tokens.available & gate
    container_mask = tokens.container_mask & available[:, None]
    token_mask = tokens.token_mask & available[:, None]
    return InventoryPolicyTokens(
        available=available,
        container_f32=jnp.where(
            container_mask,
            tokens.container_f32,
            jnp.float32(0.0),
        ),
        container_mask=container_mask,
        token_f32=jnp.where(
            token_mask[..., None],
            tokens.token_f32,
            jnp.float32(0.0),
        ),
        token_mask=token_mask,
    )


def _validate_source_layout(layout: InventoryLayout) -> None:
    if not isinstance(layout, InventoryLayout):
        raise TypeError("layout must be InventoryLayout")
    if layout.capacities.shape != (CONTAINER_COUNT,):
        raise ValueError("inventory layout container shape drift")


def _semantic_native_frame(frame: Mapping[str, object]) -> dict[str, object]:
    result = copy.deepcopy(dict(frame))
    seen: dict[int, str] = {}
    for container in result["containers"]:
        for slot in container["occupied_slots"]:
            asset_id = slot["item_id"]
            value = semantic_id(asset_id)
            previous = seen.get(value)
            if previous is not None and previous != asset_id:
                raise ValueError("native inventory semantic item ID collision")
            seen[value] = asset_id
            slot["item_runtime_index"] = value
    return result


__all__ = [
    "INVENTORY_POLICY_FLOAT_FEATURES",
    "INVENTORY_POLICY_FLOAT_SIZE",
    "INVENTORY_POLICY_SOURCE_FIELDS",
    "INVENTORY_POLICY_TOKEN_CAPACITY",
    "InventoryPolicyTokens",
    "empty_inventory_policy_tokens",
    "encode_inventory_policy_tokens",
    "inventory_policy_flat_size",
    "inventory_policy_tokens_from_native_frame",
    "inventory_policy_tokens_from_state",
    "mask_inventory_policy_tokens",
]
