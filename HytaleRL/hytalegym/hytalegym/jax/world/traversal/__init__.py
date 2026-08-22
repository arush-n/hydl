"""Neutral fixed-shape traversal observation contract."""

from __future__ import annotations

from typing import NamedTuple

import jax


Array = jax.Array
GRAPH_DIAGNOSTIC_VERTICAL_TRANSITIONS = 1
GRAPH_DIAGNOSTIC_STATEFUL_TRANSITIONS = 1 << 1
GRAPH_DIAGNOSTIC_OFF_GRID_SURFACE = 1 << 2
GRAPH_DIAGNOSTIC_QUERY_DISTANCE_EXCEEDED = 1 << 3
TRAVERSAL_PROVENANCE_UNKNOWN = 0
TRAVERSAL_PROVENANCE_NATIVE_EXACT_GEOMETRY = 1
TRAVERSAL_PROVENANCE_SURROGATE = 3


class TraversalTokenObservation(NamedTuple):
    """Fixed-shape graph neighborhood selected independently per actor."""

    available: Array
    capacity_exceeded: Array
    diagnostics: Array
    provenance: Array
    tile_index: Array
    token_mask: Array
    node_index: Array
    relative_position: Array
    clearance: Array
    node_flags: Array
    token_dynamic_blocked: Array
    edge_mask: Array
    edge_destination: Array
    edge_cost: Array
    edge_kind: Array
    edge_flags: Array


__all__ = [
    "GRAPH_DIAGNOSTIC_OFF_GRID_SURFACE",
    "GRAPH_DIAGNOSTIC_QUERY_DISTANCE_EXCEEDED",
    "GRAPH_DIAGNOSTIC_STATEFUL_TRANSITIONS",
    "GRAPH_DIAGNOSTIC_VERTICAL_TRANSITIONS",
    "TRAVERSAL_PROVENANCE_NATIVE_EXACT_GEOMETRY",
    "TRAVERSAL_PROVENANCE_SURROGATE",
    "TRAVERSAL_PROVENANCE_UNKNOWN",
    "TraversalTokenObservation",
]
