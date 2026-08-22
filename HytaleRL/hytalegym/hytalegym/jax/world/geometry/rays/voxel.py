"""Fixed-capacity translation of Hytale 0.5.7's ``BlockIterator``."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp


Array = jax.Array
NATIVE_BLOCK_ITERATOR_SCHEMA = "hytalerl_native_block_iterator_cells_v1"
NATIVE_BLOCK_ITERATOR_VERSION = 1
_DIRECTION_EPSILON = jnp.float32(1.0e-7)
_DISTANCE_EPSILON = jnp.float32(1.0e-6)


class VoxelRayCells(NamedTuple):
    """Visited voxel cells and explicit fixed-capacity diagnostics."""

    cells: Array
    mask: Array
    entry_distance: Array
    invalid: Array
    capacity_exceeded: Array
    visited_count: Array


def native_block_iterator_cells(
    origin: Array,
    direction: Array,
    maximum_distance: Array | float,
    *,
    cell_capacity: int,
) -> VoxelRayCells:
    """Visit the same voxel sequence as native ``BlockIterator.iterate``.

    The input is ray-major: origins and directions have shape ``[R, 3]`` and
    distance is scalar or ``[R]``. The starting cell is always visited. A ray
    starting on an integer boundary and travelling in the negative direction
    therefore visits the boundary cell followed by its neighbour at distance
    zero, matching the Java implementation.
    """

    if (
        isinstance(cell_capacity, bool)
        or not isinstance(cell_capacity, int)
        or cell_capacity < 1
    ):
        raise ValueError("cell_capacity must be a positive integer")
    starts = jnp.asarray(origin, dtype=jnp.float32)
    vectors = jnp.asarray(direction, dtype=jnp.float32)
    if starts.ndim != 2 or starts.shape[-1] != 3:
        raise ValueError("origin must have shape [ray, 3]")
    if vectors.shape != starts.shape:
        raise ValueError("direction must match origin")
    rays = starts.shape[0]
    maximum = jnp.asarray(maximum_distance, dtype=jnp.float32)
    if maximum.ndim == 0:
        maximum = jnp.broadcast_to(maximum, (rays,))
    elif maximum.shape != (rays,):
        raise ValueError("maximum_distance must be scalar or have shape [ray]")

    norm = jnp.linalg.norm(vectors, axis=1)
    invalid = (
        ~jnp.all(jnp.isfinite(starts), axis=1)
        | ~jnp.all(jnp.isfinite(vectors), axis=1)
        | ~jnp.isfinite(maximum)
        | (maximum < 0.0)
        | (norm <= _DIRECTION_EPSILON)
    )
    safe_starts = jnp.where(invalid[:, None], 0.0, starts)
    unit = jnp.where(
        invalid[:, None],
        jnp.asarray((1.0, 0.0, 0.0), dtype=jnp.float32),
        vectors / jnp.maximum(norm[:, None], _DIRECTION_EPSILON),
    )
    cell = jnp.floor(safe_starts).astype(jnp.int32)
    step = jnp.sign(unit).astype(jnp.int32)
    moving_axis = step != 0
    boundary = jnp.where(step > 0, cell + 1, cell).astype(jnp.float32)
    safe_component = jnp.where(moving_axis, unit, 1.0)
    next_distance = jnp.where(
        moving_axis,
        (boundary - safe_starts) / safe_component,
        jnp.inf,
    )
    distance_delta = jnp.where(
        moving_axis,
        jnp.abs(1.0 / safe_component),
        jnp.inf,
    )
    entry = jnp.where(invalid, jnp.inf, jnp.float32(0.0))

    def visit(carry, _):
        current_cell, current_next, current_entry = carry
        active = (
            ~invalid
            & (current_entry <= maximum + _DISTANCE_EPSILON)
        )
        emitted_cell = jnp.where(active[:, None], current_cell, 0)
        emitted_entry = jnp.where(active, current_entry, jnp.inf)

        following_entry = jnp.min(current_next, axis=1)
        # Java's tie epsilon is 1e-15 in double. Distinct intersections
        # derived from float32 actor inputs cannot fall inside that band, so
        # exact equality preserves real edge/corner ties without merging two
        # merely nearby crossings.
        tied = moving_axis & (current_next == following_entry[:, None])
        following_cell = current_cell + jnp.where(tied, step, 0)
        following_next = current_next + jnp.where(
            tied,
            distance_delta,
            0.0,
        )
        return (
            following_cell,
            following_next,
            following_entry,
        ), (emitted_cell, active, emitted_entry)

    (next_cell, next_faces, next_entry), emitted = jax.lax.scan(
        visit,
        (cell, next_distance, entry),
        xs=None,
        length=cell_capacity,
    )
    del next_cell, next_faces
    cells, mask, entry_distance = emitted
    cells = jnp.swapaxes(cells, 0, 1)
    mask = jnp.swapaxes(mask, 0, 1)
    entry_distance = jnp.swapaxes(entry_distance, 0, 1)
    capacity_exceeded = (
        ~invalid & (next_entry <= maximum + _DISTANCE_EPSILON)
    )
    return VoxelRayCells(
        cells=cells,
        mask=mask,
        entry_distance=entry_distance,
        invalid=invalid,
        capacity_exceeded=capacity_exceeded,
        visited_count=jnp.sum(mask, axis=1, dtype=jnp.int32),
    )


def native_block_iterator_contract() -> dict[str, object]:
    """Return the source and boundary semantics pinned by this translation."""

    return {
        "schema": NATIVE_BLOCK_ITERATOR_SCHEMA,
        "version": NATIVE_BLOCK_ITERATOR_VERSION,
        "native_version": "0.5.7",
        "native_source": (
            "com/hypixel/hytale/math/iterator/BlockIterator.java"
        ),
        "input": {
            "origin": "float32_ray_xyz",
            "direction": "float32_ray_xyz_normalized_internally",
            "maximum_distance": "nonnegative_euclidean_distance",
        },
        "sequence": {
            "starting_cell": "floor_origin_included",
            "negative_integer_boundary": (
                "starting_cell_then_negative_neighbour_at_distance_zero"
            ),
            "corner_or_edge_tie": "advance_all_reached_axes",
            "maximum_distance": "entry_distance_inclusive",
        },
        "fixed_capacity": "explicit_overflow_fail_closed",
        "invalid": [
            "nonfinite_origin",
            "nonfinite_direction",
            "zero_direction",
            "nonfinite_or_negative_distance",
        ],
    }


def native_block_iterator_contract_sha256() -> str:
    payload = json.dumps(
        native_block_iterator_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "NATIVE_BLOCK_ITERATOR_SCHEMA",
    "NATIVE_BLOCK_ITERATOR_VERSION",
    "VoxelRayCells",
    "native_block_iterator_cells",
    "native_block_iterator_contract",
    "native_block_iterator_contract_sha256",
]
