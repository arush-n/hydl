"""Scalar validation and hashing helpers for region traversal."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


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


def _array(value: object, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    return result


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _exact_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _supported_graph_version(value: object) -> int:
    version = _exact_nonnegative_int(value, "Region traversal graph version")
    if version not in _GRAPH_SCHEMAS:
        raise ValueError("unsupported Region traversal graph version")
    return version


def _positive_finite(value: object, label: str) -> float:
    result = _finite_nonnegative(value, label)
    if result == 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _sha256_text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{label} must be a SHA-256")
    return value.lower()


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value
