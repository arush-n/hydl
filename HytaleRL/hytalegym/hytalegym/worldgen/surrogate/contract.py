"""Fixed capacities for explicitly non-authoritative surrogate world tiles."""

from __future__ import annotations

from dataclasses import dataclass
import operator

import numpy as np

from hytalegym.geometry.contract import (
    FLUID_MOVEMENT_FEATURES,
    MOVEMENT_FEATURES,
)
from hytalegym.worldgen.region import (
    CAPTURE_BLOCKS_PER_AXIS,
    CORE_BLOCKS_PER_AXIS,
    MAX_CELL_PALETTE,
    MAX_SHAPE_BOXES,
    MAX_SHAPE_PALETTE,
    WORLD_HEIGHT,
)

SURROGATE_WORLD_SCHEMA = "hytalerl_surrogate_world_v5"
SURROGATE_WORLD_VERSION = 5
BIOME_CAPACITY = 16
CAVE_SPAN_CAPACITY = 2
DEFAULT_SPAWN_MARKER_CATALOG_CAPACITY = 256
DEFAULT_SPAWN_MARKER_CHOICE_CAPACITY = 8
TERRAIN_PARAMETER_COUNT = 32
TRAVERSAL_STATE_MASK_BITS = 8


@dataclass(frozen=True)
class SurrogateWorldCapacity:
    """Static per-tile capacities used by host generation and compiled JAX."""

    structure_instance_capacity: int = 128
    structure_cell_capacity: int = 65_536
    stateful_block_capacity: int = 1_024
    states_per_block: int = 8
    entity_capacity: int = 256
    spawn_marker_capacity: int = DEFAULT_SPAWN_MARKER_CATALOG_CAPACITY
    spawn_marker_choice_capacity: int = DEFAULT_SPAWN_MARKER_CHOICE_CAPACITY
    graph_node_capacity: int = 32_768
    graph_edges_per_node: int = 12
    cell_palette_capacity: int = 4_096
    shape_palette_capacity: int = 4_096

    def __post_init__(self) -> None:
        values = {
            name: _positive_int(value, name) for name, value in vars(self).items()
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        if self.structure_cell_capacity > (CAPTURE_BLOCKS_PER_AXIS**2 * WORLD_HEIGHT):
            raise ValueError("structure cell capacity exceeds one tile volume")
        if self.stateful_block_capacity > self.structure_cell_capacity:
            raise ValueError("stateful block capacity exceeds structure cell capacity")
        if self.stateful_block_capacity > np.iinfo(np.int16).max:
            raise ValueError("stateful block capacity exceeds graph gate schema")
        if self.states_per_block > TRAVERSAL_STATE_MASK_BITS:
            raise ValueError("states per block exceeds graph state-mask capacity")
        if self.entity_capacity > np.iinfo(np.uint16).max:
            raise ValueError("entity capacity exceeds uint16 type schema")
        if self.cell_palette_capacity > MAX_CELL_PALETTE:
            raise ValueError("cell palette capacity exceeds uint16 schema")
        if self.shape_palette_capacity > MAX_SHAPE_PALETTE:
            raise ValueError("shape palette capacity exceeds schema maximum")
        if self.graph_node_capacity > np.iinfo(np.int32).max:
            raise ValueError("graph node capacity exceeds int32 indices")


@dataclass(frozen=True)
class SurrogateMemoryEstimate:
    """Logical array payload; allocator, executable, and temporary memory excluded."""

    tile_capacity: int
    environment_capacity: int
    shared_palette_bytes: int
    spawn_marker_catalog_bytes: int
    immutable_tile_bytes: int
    runtime_environment_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.shared_palette_bytes
            + self.spawn_marker_catalog_bytes
            + self.tile_capacity * self.immutable_tile_bytes
            + self.environment_capacity * self.runtime_environment_bytes
        )


def estimate_surrogate_atlas_bytes(
    capacity: SurrogateWorldCapacity | None = None,
    *,
    tile_capacity: int = 1,
    environment_capacity: int = 1,
) -> SurrogateMemoryEstimate:
    """Return exact logical bytes for the current fixed-array layout."""

    layout = SurrogateWorldCapacity() if capacity is None else capacity
    if not isinstance(layout, SurrogateWorldCapacity):
        raise TypeError("capacity must be a SurrogateWorldCapacity")
    tiles = _positive_int(tile_capacity, "tile capacity")
    environments = _positive_int(
        environment_capacity,
        "environment capacity",
    )
    columns = CAPTURE_BLOCKS_PER_AXIS**2
    core_columns = CORE_BLOCKS_PER_AXIS**2

    shared_palette = (
        _bytes(np.int32, 1)
        + _bytes(np.int32, 1)
        + _bytes(
            np.uint16,
            2 * layout.cell_palette_capacity,
        )
        + _bytes(np.uint8, layout.cell_palette_capacity)
        + _bytes(np.int32, 3 * layout.cell_palette_capacity)
        + _bytes(
            np.float32,
            (MOVEMENT_FEATURES + FLUID_MOVEMENT_FEATURES)
            * layout.cell_palette_capacity,
        )
        + _bytes(np.int32, 1)
        + _bytes(
            np.float32,
            (layout.shape_palette_capacity * MAX_SHAPE_BOXES * 6),
        )
        + _bytes(
            np.bool_,
            layout.shape_palette_capacity * MAX_SHAPE_BOXES,
        )
    )
    spawn_markers = layout.spawn_marker_capacity * (
        _bytes(np.bool_, 3)
        + _bytes(np.uint32, 10)
        + _bytes(np.float32, 4)
        + layout.spawn_marker_choice_capacity
        * (
            _bytes(np.bool_, 3)
            + _bytes(np.uint32, 2)
            + _bytes(np.float32, 2)
        )
    )

    tile_metadata = (
        _bytes(np.bool_, 1)
        + _bytes(np.int32, 3)
        + _bytes(np.uint32, 2)
        + _bytes(np.uint32, 1)
        + _bytes(np.uint8, 1)
        + _bytes(np.float32, TERRAIN_PARAMETER_COUNT)
        + _bytes(
            np.float32,
            BIOME_CAPACITY * TERRAIN_PARAMETER_COUNT,
        )
    )
    terrain = (
        _bytes(np.int16, columns)
        + _bytes(np.uint16, 2 * columns)
        + _bytes(np.uint8, columns)
        + _bytes(np.bool_, columns)
        + CAVE_SPAN_CAPACITY * columns * (_bytes(np.bool_, 1) + _bytes(np.int16, 2))
    )
    structure_cells = layout.structure_cell_capacity * (
        _bytes(np.bool_, 1) + _bytes(np.int32, 2) + _bytes(np.uint16, 1)
    )
    traversal_metadata = (
        _bytes(np.bool_, 3)
        + _bytes(np.uint32, 1)
        + _bytes(np.int32, 3)
        + _bytes(np.float32, 8)
        + core_columns * (_bytes(np.int32, 1) + _bytes(np.uint16, 1))
    )
    entity_definitions = layout.entity_capacity * (
        _bytes(np.bool_, 6)
        + _bytes(np.uint32, 4)
        + _bytes(np.int32, 1)
        + _bytes(np.float32, 18)
        + _bytes(np.uint8, 1)
        + _bytes(np.uint16, 1)
    )
    stateful_definitions = layout.stateful_block_capacity * (
        _bytes(np.bool_, 1)
        + _bytes(np.int32, 4)
        + _bytes(np.uint8, 3)
        + _bytes(np.uint16, layout.states_per_block)
        + _bytes(
            np.uint8,
            4 * layout.states_per_block,
        )
        + _bytes(
            np.bool_,
            2 * layout.states_per_block,
        )
    )
    graph_nodes = layout.graph_node_capacity * (
        _bytes(np.bool_, 1)
        + _bytes(np.float32, 4)
        + _bytes(np.int32, 1)
        + _bytes(np.uint8, 2)
        + _bytes(np.int16, 1)
    )
    graph_edges = (
        layout.graph_node_capacity
        * layout.graph_edges_per_node
        * (
            _bytes(np.int32, 1)
            + _bytes(np.float32, 1)
            + _bytes(np.uint8, 3)
            + _bytes(np.bool_, 1)
            + _bytes(np.int16, 1)
        )
    )
    immutable_tile = (
        tile_metadata
        + traversal_metadata
        + terrain
        + structure_cells
        + stateful_definitions
        + entity_definitions
        + graph_nodes
        + graph_edges
    )
    runtime_environment = (
        _bytes(np.int32, 1)
        + _bytes(np.bool_, 1)
        + _bytes(np.uint32, 2)
        + layout.stateful_block_capacity * (_bytes(np.uint8, 1) + _bytes(np.bool_, 1))
        + layout.entity_capacity
        * (
            _bytes(np.bool_, 7)
            + _bytes(np.float32, 18)
            + _bytes(np.uint32, 4)
            + _bytes(np.uint8, 1)
            + _bytes(np.uint16, 1)
        )
    )
    return SurrogateMemoryEstimate(
        tile_capacity=tiles,
        environment_capacity=environments,
        shared_palette_bytes=shared_palette,
        spawn_marker_catalog_bytes=spawn_markers,
        immutable_tile_bytes=immutable_tile,
        runtime_environment_bytes=runtime_environment,
    )


def _bytes(dtype: type[np.generic], count: int) -> int:
    return np.dtype(dtype).itemsize * count


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


__all__ = [
    "CORE_BLOCKS_PER_AXIS",
    "BIOME_CAPACITY",
    "CAVE_SPAN_CAPACITY",
    "DEFAULT_SPAWN_MARKER_CATALOG_CAPACITY",
    "DEFAULT_SPAWN_MARKER_CHOICE_CAPACITY",
    "SURROGATE_WORLD_SCHEMA",
    "SURROGATE_WORLD_VERSION",
    "TRAVERSAL_STATE_MASK_BITS",
    "SurrogateMemoryEstimate",
    "SurrogateWorldCapacity",
    "estimate_surrogate_atlas_bytes",
]
