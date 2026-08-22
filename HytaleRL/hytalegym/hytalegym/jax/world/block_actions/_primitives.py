"""Leaf tuples, scalars, and validators for the block-actions package."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.mutable_blocks import (
    MutableBlockGeometry,
    MutableBlockQueryResult,
    MutableBlockUpdate,
    empty_mutable_block_geometry,
)
from hytalegym.worldgen.block_actions import (
    BLOCK_DROP_METADATA_HASH_WORDS,
)
from hytalegym.worldgen.block_affordances import (
    block_affordance_tag_mask,
)


Array = jax.Array
BLOCK_ACTION_TARGET_SCHEMA = "hytalerl_block_action_target_v4"
BLOCK_ACTION_TARGET_VERSION = 4

BLOCK_ACTION_DIAGNOSTIC_INVALID = 1 << 0
BLOCK_ACTION_DIAGNOSTIC_TARGET_UNAVAILABLE = 1 << 1
BLOCK_ACTION_DIAGNOSTIC_CATALOG_MISSING = 1 << 2
BLOCK_ACTION_DIAGNOSTIC_CATALOG_DUPLICATE = 1 << 3
BLOCK_ACTION_DIAGNOSTIC_TOOL_UNAVAILABLE = 1 << 4
BLOCK_ACTION_DIAGNOSTIC_DROP_UNAVAILABLE = 1 << 5
BLOCK_ACTION_DIAGNOSTIC_PERMISSION_UNAVAILABLE = 1 << 6
BLOCK_ACTION_DIAGNOSTIC_PLACE_UNAVAILABLE = 1 << 7
BLOCK_ACTION_DIAGNOSTIC_INTERACTION_TOOL_MISMATCH = 1 << 8

BLOCK_ACTION_MUTATION_SCHEMA = "hytalerl_block_action_mutation_v1"
BLOCK_ACTION_MUTATION_VERSION = 1
BLOCK_ACTION_MUTATION_DIAGNOSTIC_SELECTION = 1 << 0
BLOCK_ACTION_MUTATION_DIAGNOSTIC_PREDICATE = 1 << 1
BLOCK_ACTION_MUTATION_DIAGNOSTIC_GEOMETRY = 1 << 2
BLOCK_ACTION_MUTATION_DIAGNOSTIC_PROVENANCE = 1 << 3

_TAG_BREAKABLE = np.uint16(block_affordance_tag_mask("breakable"))
_TAG_HARVESTABLE = np.uint16(block_affordance_tag_mask("harvestable"))
_TAG_SOFT = np.uint16(block_affordance_tag_mask("soft"))


class BlockActionMutationPlan(NamedTuple):
    """Revision-bound exact Break/Place update ready for atomic commit."""

    available: Array
    update: MutableBlockUpdate
    diagnostics: Array


class BlockDropProgramCatalog(NamedTuple):
    """Deduplicated exact outcome distributions for authored drop lists."""

    program_mask: Array
    semantic_sha256: Array
    randomized: Array
    outcome_mask: Array
    cumulative_probability: Array
    drop_mask: Array
    item_id: Array
    quantity: Array
    item_max_stack: Array
    durability: Array
    max_durability: Array
    metadata_hash: Array


class BlockDropTable(NamedTuple):
    """Fixed outputs for one route per block-catalog entry."""

    available: Array
    program_index: Array
    mask: Array
    item_id: Array
    quantity: Array
    item_max_stack: Array
    durability: Array
    max_durability: Array
    metadata_hash: Array


class BlockGatherDefaults(NamedTuple):
    """The installed ``ItemToolSpec`` asset map indexed by gather type."""

    available: Array
    quality: Array
    power: Array


class BlockMutationAcknowledgement(NamedTuple):
    """Exact before/after geometry for accepted mutable-block writes."""

    available: Array
    accepted: Array
    applied: Array
    before: MutableBlockQueryResult
    after: MutableBlockQueryResult
    revision_before: Array
    revision_after: Array
    stale_base: Array
    duplicate_position: Array
    capacity_exceeded: Array
    invalid: Array
    resync_required: Array


class BlockResolvedDrops(NamedTuple):
    available: Array
    mask: Array
    item_id: Array
    quantity: Array
    item_max_stack: Array
    durability: Array
    max_durability: Array
    metadata_hash: Array


class BlockToolState(NamedTuple):
    """Held-item tool evidence; specs preserve the native gather axis."""

    available: Array
    weapon: Array
    builder_tool: Array
    tool_present: Array
    spec_mask: Array
    gather_type_index: Array
    quality: Array
    power: Array


class NativeBlockInteractionOutcome(NamedTuple):
    """Fixed-shape native place/break outcome for JAX parity checks."""

    available: Array
    placement: Array
    accepted: Array
    mutation_applied: Array
    runtime_block_id_before: Array
    runtime_block_id_after: Array
    block_health_before: Array
    block_health_after: Array


def _array(
    value: Array,
    shape: tuple[int, ...],
    dtype,
    label: str,
) -> Array:
    result = jnp.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    return result


def _capacity(value: int, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} capacity must be an integer")
    if not 1 <= value <= maximum:
        raise ValueError(f"{label} capacity must be in [1, {maximum}]")
    return value


def _catalog_value(values: Array, slot: Array) -> Array:
    return values[slot]


def _exact_present_geometry(geometry: MutableBlockGeometry) -> Array:
    finite_vectors = (
        jnp.all(jnp.isfinite(geometry.movement), axis=-1)
        & jnp.all(jnp.isfinite(geometry.fluid_movement), axis=-1)
        & jnp.all(
            ~geometry.collision_box_mask
            | jnp.all(jnp.isfinite(geometry.collision_boxes), axis=-1),
            axis=-1,
        )
        & jnp.isfinite(geometry.fluid_fill_height)
    )
    return (
        geometry.exact
        & geometry.block_present
        & geometry.semantic_key_valid
        & jnp.any(geometry.semantic_key != 0, axis=-1)
        & geometry.affordance_valid
        & finite_vectors
    )


def _host_drop_table(capacity: int, drops: int) -> dict[str, np.ndarray]:
    return {
        "available": np.zeros(capacity, dtype=np.bool_),
        "program_index": np.full(capacity, -1, dtype=np.int16),
        "mask": np.zeros((capacity, drops), dtype=np.bool_),
        "item_id": np.zeros((capacity, drops), dtype=np.int32),
        "quantity": np.zeros((capacity, drops), dtype=np.int32),
        "item_max_stack": np.zeros((capacity, drops), dtype=np.int32),
        "durability": np.zeros((capacity, drops), dtype=np.float32),
        "max_durability": np.zeros((capacity, drops), dtype=np.float32),
        "metadata_hash": np.zeros(
            (capacity, drops, BLOCK_DROP_METADATA_HASH_WORDS),
            dtype=np.uint32,
        ),
    }


def _install_drop_route(
    table: dict[str, np.ndarray],
    index: int,
    route,
    program_indices: dict[str, int],
) -> None:
    table["available"][index] = route.available
    if not route.available:
        return
    if route.program_sha256 is not None:
        try:
            table["program_index"][index] = program_indices[
                route.program_sha256
            ]
        except KeyError as error:
            raise ValueError(
                "available drop route references an unpublished program"
            ) from error
        return
    if route.quantity == 0:
        return
    table["mask"][index, 0] = True
    table["item_id"][index, 0] = route.item_id
    table["quantity"][index, 0] = route.quantity
    table["item_max_stack"][index, 0] = route.item_max_stack
    table["durability"][index, 0] = route.durability
    table["max_durability"][index, 0] = route.max_durability
    table["metadata_hash"][index, 0] = np.asarray(
        route.metadata_hash,
        dtype=np.uint32,
    )


def _prefix(value: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError("prefix_shape must be a nonempty tuple")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0
           for item in value):
        raise ValueError("prefix_shape values must be positive integers")
    return value


def _select_geometry(
    mask: Array,
    selected: MutableBlockGeometry,
    fallback: MutableBlockGeometry,
) -> MutableBlockGeometry:
    return jax.tree_util.tree_map(
        lambda left, right: jnp.where(
            mask.reshape(
                mask.shape + (1,) * (left.ndim - mask.ndim)
            ),
            left,
            right,
        ),
        selected,
        fallback,
    )


def _validate_mutation_geometry(
    geometry: MutableBlockGeometry,
    shape: tuple[int, ...],
    label: str,
) -> None:
    template = empty_mutable_block_geometry(shape)
    for name, value, expected in zip(
        geometry._fields,
        geometry,
        template,
        strict=True,
    ):
        if value.shape != expected.shape:
            raise ValueError(f"{label}.{name} has the wrong shape")
        if value.dtype != expected.dtype:
            raise TypeError(f"{label}.{name} has the wrong dtype")


def _validate_target(
    target: MutableBlockQueryResult,
    shape: tuple[int, ...],
) -> None:
    if not isinstance(target, MutableBlockQueryResult):
        raise TypeError("target must be MutableBlockQueryResult")
    if len(shape) != 2:
        raise ValueError("target query must have shape [batch, query]")
