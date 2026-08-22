"""Candidate and fluid-clear edge construction for region traversal."""
from __future__ import annotations

import math

import numpy as np

from hytalegym.geometry.contract import FLAG_FLUID
from hytalegym.worldgen.region.contract import (
    CHUNK_SIZE,
    MIN_Y,
)
from hytalegym.worldgen.region.snapshot import NativeRegionSnapshot
from hytalegym.worldgen.region._traversal_columns import (
    _dense_capture,
)

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
def _candidate_edges(
    positions: np.ndarray,
    bounds: np.ndarray,
    maximum_climb_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes_by_column: dict[tuple[int, int], list[int]] = {}
    for node, position in enumerate(positions):
        key = (
            math.floor(float(position[0])),
            math.floor(float(position[2])),
        )
        nodes_by_column.setdefault(key, []).append(node)
    for nodes in nodes_by_column.values():
        nodes.sort(key=lambda value: float(positions[value, 1]))
    sources: list[int] = []
    destinations: list[int] = []
    kinds: list[int] = []
    for source, position in enumerate(positions):
        x = math.floor(float(position[0]))
        z = math.floor(float(position[2]))
        source_foot = float(position[1] + bounds[1])
        for dx, dz in _DIRECTIONS:
            targets = nodes_by_column.get((x + dx, z + dz), ())
            selected: list[tuple[int, int]] = []
            drops: list[int] = []
            for target in targets:
                target_foot = float(
                    positions[target, 1] + bounds[1]
                )
                delta = target_foot - source_foot
                if abs(delta) <= _SUPPORT_EPSILON:
                    selected.append((target, REGION_TRAVERSAL_EDGE_WALK))
                elif (
                    _SUPPORT_EPSILON
                    < delta
                    <= maximum_climb_height + _SUPPORT_EPSILON
                ):
                    selected.append((target, REGION_TRAVERSAL_EDGE_CLIMB))
                elif delta < -_SUPPORT_EPSILON:
                    drops.append(target)
            if drops:
                selected.append(
                    (
                        max(
                            drops,
                            key=lambda value: float(positions[value, 1]),
                        ),
                        REGION_TRAVERSAL_EDGE_DROP,
                    )
                )
            for destination, kind in selected:
                sources.append(source)
                destinations.append(destination)
                kinds.append(kind)
    if not sources:
        raise ValueError("Region traversal produced no edge candidates")
    return (
        np.asarray(sources, dtype=np.int32),
        np.asarray(destinations, dtype=np.int32),
        np.asarray(kinds, dtype=np.uint8),
    )


def _fluid_clear_edges(
    snapshot: NativeRegionSnapshot,
    positions: np.ndarray,
    sources: np.ndarray,
    destinations: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    codes = _dense_capture(snapshot.cell_code)
    fluid = (
        snapshot.cell_palette.flags[codes] & np.uint16(FLAG_FLUID)
    ) != 0
    prefix = np.pad(
        fluid.astype(np.int32),
        ((1, 0), (1, 0), (1, 0)),
    )
    for axis in range(3):
        prefix = np.cumsum(prefix, axis=axis, dtype=np.int32)

    starts = positions[sources]
    targets = positions[destinations]
    capture_origin = np.asarray(
        [
            snapshot.capture_min_chunk_xz[0] * CHUNK_SIZE,
            MIN_Y,
            snapshot.capture_min_chunk_xz[1] * CHUNK_SIZE,
        ],
        dtype=np.float64,
    )
    lower = np.floor(
        np.minimum(starts, targets)
        + bounds[:3]
        - capture_origin
        + _EPSILON
    ).astype(np.int32)
    upper = np.ceil(
        np.maximum(starts, targets)
        + bounds[3:]
        - capture_origin
        - _EPSILON
    ).astype(np.int32)
    lower[:, (0, 2)] -= 1
    upper[:, (0, 2)] += 1
    shape = np.asarray(fluid.shape, dtype=np.int32)
    if (
        np.any(lower < 0)
        or np.any(upper > shape)
        or np.any(lower >= upper)
    ):
        raise ValueError("Region traversal fluid query exceeds capture")

    x0, y0, z0 = lower.T
    x1, y1, z1 = upper.T
    counts = (
        prefix[x1, y1, z1]
        - prefix[x0, y1, z1]
        - prefix[x1, y0, z1]
        - prefix[x1, y1, z0]
        + prefix[x0, y0, z1]
        + prefix[x0, y1, z0]
        + prefix[x1, y0, z0]
        - prefix[x0, y0, z0]
    )
    return counts == 0
