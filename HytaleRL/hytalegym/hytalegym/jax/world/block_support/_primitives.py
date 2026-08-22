"""Leaf scalars, result tuples, and the support-cascade contract."""
from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.worldgen.block_support import (
    LocalBlockSupportSemantics,
)


Array = jax.Array
BLOCK_SUPPORT_CASCADE_SCHEMA = "hytalerl_jax_block_support_cascade_v1"
BLOCK_SUPPORT_CASCADE_VERSION = 1

BLOCK_SUPPORT_DIAGNOSTIC_CATALOG = 1 << 0
BLOCK_SUPPORT_DIAGNOSTIC_NEIGHBOUR = 1 << 1
BLOCK_SUPPORT_DIAGNOSTIC_FLUID = 1 << 2
BLOCK_SUPPORT_DIAGNOSTIC_SINGLE_CELL = 1 << 3
BLOCK_SUPPORT_DIAGNOSTIC_CAPACITY = 1 << 4
BLOCK_SUPPORT_DIAGNOSTIC_INPUT = 1 << 5


class BlockSupportCascadeResult(NamedTuple):
    """Fixed-capacity cascade output; all effect masks fail closed together."""

    available: Array
    removed: Array
    broken: Array
    destroyed: Array
    falling: Array
    support_after: Array
    iterations: Array
    capacity_exceeded: Array
    diagnostics: Array


class BlockSupportCatalog(NamedTuple):
    """Fixed-shape compiled support rules keyed by portable semantic hashes."""

    entry_mask: Array
    semantic_key: Array
    asset_key: Array
    rotation_index: Array
    material_empty: Array
    player_placement_marks_deco: Array
    support_dependent: Array
    support_drop_type: Array
    max_support_distance: Array
    tag_bits: Array
    block_set_bits: Array
    supporting_face_types: Array
    group_mask: Array
    group_neighbour_slot: Array
    group_block_face: Array
    group_neighbour_face: Array
    clause_mask: Array
    clause_face_type_mask: Array
    clause_self_face_type_mask: Array
    clause_block_type_valid: Array
    clause_block_type_key: Array
    clause_fluid_id: Array
    clause_tag_mask: Array
    clause_block_set_mask: Array
    clause_match_self: Array
    clause_support: Array
    clause_allow_support_propagation: Array


class BlockSupportResult(NamedTuple):
    """One native-equivalent support decision for single-cell evidence."""

    available: Array
    stable: Array
    ignored: Array
    direct: Array
    propagated: Array
    unsupported: Array
    support_after: Array
    support_drop_type: Array
    diagnostics: Array


def _array(
    value: Array,
    shape: tuple[int, ...],
    dtype: jnp.dtype,
    label: str,
) -> Array:
    result = jnp.asarray(value)
    if result.shape != shape or result.dtype != dtype:
        raise ValueError(f"{label} must have shape {shape} and dtype {dtype}")
    return result


def _bits(values: Sequence[str], indices: dict[str, int]) -> int:
    result = 0
    for value in values:
        index = indices.get(value)
        if index is not None:
            result |= 1 << index
    return result


def _capacity(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} capacity must be a positive integer")
    return value


def _face_types(entry: LocalBlockSupportSemantics) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                text
                for groups in (
                    (
                        clause.face_type
                        for group in entry.groups
                        for clause in group.clauses
                    ),
                    (
                        clause.self_face_type
                        for group in entry.groups
                        for clause in group.clauses
                    ),
                    (
                        offer.face_type
                        for rows in entry.supporting_by_rotation
                        for offer in rows
                    ),
                )
                for text in groups
                if text is not None
            }
        )
    )


def _vocabulary(
    values: Sequence[str] | object,
    capacity: int,
    label: str,
) -> tuple[str, ...]:
    result = tuple(sorted(set(values)))
    if len(result) > capacity:
        raise ValueError(f"{label} vocabulary exceeds installed capacity")
    return result


def support_cascade_seed_mask(
    cell_position: Array,
    cell_mask: Array,
    mutation_position: Array,
    mutation_mask: Array,
) -> Array:
    """Mark active cells in the native 26-neighbour mutation frontier."""

    positions = jnp.asarray(cell_position, dtype=jnp.int32)
    cells = jnp.asarray(cell_mask)
    mutations = jnp.asarray(mutation_position, dtype=jnp.int32)
    active = jnp.asarray(mutation_mask)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("cell_position must have shape [batch, cell, 3]")
    if mutations.ndim != 3 or mutations.shape[-1] != 3:
        raise ValueError(
            "mutation_position must have shape [batch, mutation, 3]"
        )
    if cells.shape != positions.shape[:2] or cells.dtype != jnp.bool_:
        raise ValueError("cell_mask must be boolean [batch, cell]")
    if active.shape != mutations.shape[:2] or active.dtype != jnp.bool_:
        raise ValueError(
            "mutation_mask must be boolean [batch, mutation]"
        )
    if positions.shape[0] != mutations.shape[0]:
        raise ValueError("cell and mutation batches differ")
    delta = jnp.abs(
        positions[:, :, None, :] - mutations[:, None, :, :]
    )
    return cells & jnp.any(
        active[:, None, :] & (jnp.max(delta, axis=-1) <= 1),
        axis=-1,
    )
