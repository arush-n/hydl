"""Dense column decoding for region traversal captures."""
from __future__ import annotations


import numpy as np

from hytalegym.geometry.contract import FLAG_SOLID
from hytalegym.worldgen.region.contract import (
    CAPTURE_CHUNKS_PER_AXIS,
    CHUNK_SIZE,
    CORE_CHUNKS_PER_AXIS,
    HEIGHT_SECTIONS,
    MIN_Y,
    WORLD_HEIGHT,
)
from hytalegym.worldgen.region.snapshot import NativeRegionSnapshot

REGION_TRAVERSAL_GRAPH_SCHEMA = "hytalerl_native_region_traversal_graph_v3"
REGION_TRAVERSAL_GRAPH_VERSION = 3
REGION_TRAVERSAL_EDGE_WALK = 1
REGION_TRAVERSAL_EDGE_CLIMB = 2
REGION_TRAVERSAL_EDGE_DROP = 3
REGION_TRAVERSAL_EDGE_SAFE = 1
REGION_TRAVERSAL_EDGES_PER_NODE = 12
REGION_TRAVERSAL_DEFAULT_QUERY_DISTANCE_CAPACITY = 12.0
REGION_TRAVERSAL_HORIZONTAL_TOLERANCE = 2.0e-4
REGION_TRAVERSAL_VERTICAL_TOLERANCE = 2.0e-4

_EPSILON = 1.0e-6
_SUPPORT_EPSILON = 2.0e-4
_FLUID_POLICY = "exclude_swept_actor_aabb_with_one_cell_horizontal_halo"
_GRAPH_SCHEMAS = {
    1: "hytalerl_native_region_traversal_graph_v1",
    2: "hytalerl_native_region_traversal_graph_v2",
    REGION_TRAVERSAL_GRAPH_VERSION: REGION_TRAVERSAL_GRAPH_SCHEMA,
}
_GRAPH_METADATA_KEY = "__metadata_json__"
_GRAPH_FIELDS = frozenset(
    {
        _GRAPH_METADATA_KEY,
        "core_min_chunk_xz",
        "actor_bounds",
        "node_position",
        "node_clearance",
        "edge_mask",
        "edge_destination",
        "edge_cost",
        "edge_kind",
        "edge_flags",
    }
)
_DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _dense_capture(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    expected = (
        CAPTURE_CHUNKS_PER_AXIS * CAPTURE_CHUNKS_PER_AXIS,
        HEIGHT_SECTIONS,
        CHUNK_SIZE**3,
    )
    if source.shape != expected:
        raise ValueError("Region section payload has an invalid shape")
    return source.reshape(
        CAPTURE_CHUNKS_PER_AXIS,
        CAPTURE_CHUNKS_PER_AXIS,
        HEIGHT_SECTIONS,
        CHUNK_SIZE,
        CHUNK_SIZE,
        CHUNK_SIZE,
    ).transpose(0, 5, 2, 3, 1, 4).reshape(
        CAPTURE_CHUNKS_PER_AXIS * CHUNK_SIZE,
        WORLD_HEIGHT,
        CAPTURE_CHUNKS_PER_AXIS * CHUNK_SIZE,
    )


def _unpack_filler(value: int) -> tuple[int, int, int]:
    def signed(axis: int) -> int:
        return axis - 32 if axis & 16 else axis

    return (
        signed(value & 31),
        signed((value >> 10) & 31),
        signed((value >> 5) & 31),
    )


class _RegionColumns:
    def __init__(
        self,
        snapshot: NativeRegionSnapshot,
        bounds: np.ndarray,
        margin: int,
    ):
        self.snapshot = snapshot
        self.bounds = bounds
        self.core_origin = (
            snapshot.core_min_chunk_xz.astype(np.int64) * CHUNK_SIZE
        )
        self.codes = _dense_capture(snapshot.cell_code)
        self.filler = _dense_capture(
            snapshot.filler_root_offset_packed
        )
        self.flags = snapshot.cell_palette.flags
        self.shape_index = snapshot.cell_palette.shape_index
        self.boxes = snapshot.shape_palette.boxes
        self.box_mask = snapshot.shape_palette.box_mask
        self.margin = margin

    def standable_nodes(self) -> tuple[np.ndarray, np.ndarray]:
        positions: list[tuple[float, float, float]] = []
        clearances: list[float] = []
        start = CHUNK_SIZE - self.margin
        stop = (
            CHUNK_SIZE
            + CORE_CHUNKS_PER_AXIS * CHUNK_SIZE
            + self.margin
        )
        for local_x in range(start, stop):
            world_x = int(self.core_origin[0]) + local_x - CHUNK_SIZE
            for local_z in range(start, stop):
                world_z = (
                    int(self.core_origin[1]) + local_z - CHUNK_SIZE
                )
                result = self._column_nodes(local_x, local_z)
                for transform_y, clearance in result:
                    positions.append(
                        (world_x + 0.5, transform_y, world_z + 0.5)
                    )
                    clearances.append(clearance)
        if not positions:
            raise ValueError("Region traversal produced no standable nodes")
        return (
            np.asarray(positions, dtype=np.float64),
            np.asarray(clearances, dtype=np.float64),
        )

    def _column_nodes(
        self,
        local_x: int,
        local_z: int,
    ) -> list[tuple[float, float]]:
        codes = self.codes[local_x, :, local_z]
        filler = self.filler[local_x, :, local_z]
        solid_y = np.flatnonzero(
            (self.flags[codes] & np.uint16(FLAG_SOLID)) != 0
        )
        if not solid_y.size:
            return []
        box_minimum: list[float] = []
        box_maximum: list[float] = []
        source_y: list[float] = []
        horizontal: list[bool] = []
        candidate_feet: set[float] = set()
        for local_y in solid_y:
            code = int(codes[local_y])
            shape = int(self.shape_index[code])
            offset = _unpack_filler(int(filler[local_y]))
            for box in self.boxes[shape, self.box_mask[shape]]:
                minimum_x = -offset[0] + float(box[0])
                maximum_x = -offset[0] + float(box[3])
                minimum_z = -offset[2] + float(box[2])
                maximum_z = -offset[2] + float(box[5])
                overlaps = (
                    0.5 + self.bounds[3]
                    > minimum_x + _SUPPORT_EPSILON
                    and 0.5 + self.bounds[0]
                    < maximum_x - _SUPPORT_EPSILON
                    and 0.5 + self.bounds[5]
                    > minimum_z + _SUPPORT_EPSILON
                    and 0.5 + self.bounds[2]
                    < maximum_z - _SUPPORT_EPSILON
                )
                minimum_y = (
                    MIN_Y
                    + int(local_y)
                    - offset[1]
                    + float(box[1])
                )
                maximum_y = (
                    MIN_Y
                    + int(local_y)
                    - offset[1]
                    + float(box[4])
                )
                box_minimum.append(minimum_y)
                box_maximum.append(maximum_y)
                source_y.append(MIN_Y + int(local_y))
                horizontal.append(overlaps)
                if (
                    minimum_x - _SUPPORT_EPSILON
                    <= 0.5
                    <= maximum_x + _SUPPORT_EPSILON
                    and minimum_z - _SUPPORT_EPSILON
                    <= 0.5
                    <= maximum_z + _SUPPORT_EPSILON
                ):
                    candidate_feet.add(float(np.float32(maximum_y)))
        if not candidate_feet:
            return []
        minimum = np.asarray(box_minimum, dtype=np.float64)
        maximum = np.asarray(box_maximum, dtype=np.float64)
        source = np.asarray(source_y, dtype=np.float64)
        horizontal_mask = np.asarray(horizontal, dtype=np.bool_)
        result: list[tuple[float, float]] = []
        for foot in sorted(candidate_feet):
            actor_minimum = foot + float(self.bounds[1])
            actor_maximum = foot + float(self.bounds[4])
            source_vertical = (
                actor_maximum >= source - _SUPPORT_EPSILON
            ) & (
                actor_minimum <= source + 1.0 + _SUPPORT_EPSILON
            )
            initial = horizontal_mask & source_vertical
            overlap = initial & (
                (actor_maximum > minimum + _SUPPORT_EPSILON)
                & (actor_minimum < maximum - _SUPPORT_EPSILON)
            )
            if np.any(overlap):
                continue
            support = initial & (
                np.abs(maximum - foot) <= _SUPPORT_EPSILON
            )
            if not np.any(support):
                continue
            overhead = horizontal_mask & (
                minimum >= foot - _SUPPORT_EPSILON
            )
            clearance = float(MIN_Y + WORLD_HEIGHT) - foot
            if np.any(overhead):
                clearance = min(
                    clearance,
                    float(np.min(minimum[overhead] - foot)),
                )
            result.append(
                (
                    foot - float(self.bounds[1]),
                    max(0.0, clearance),
                )
            )
        return result
