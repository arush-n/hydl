"""Fixed capacities and diagnostics for surrogate world-feature candidates."""

from __future__ import annotations

import math

TERRAIN_SURFACE_STACK_CAPACITY = 8

FEATURE_DIAGNOSTIC_INVALID_ACTOR = 1
FEATURE_DIAGNOSTIC_UNSUPPORTED_RUNTIME = 1 << 1
FEATURE_DIAGNOSTIC_TERRAIN_COVERAGE = 1 << 2
FEATURE_DIAGNOSTIC_GRAPH_UNAVAILABLE = 1 << 3
FEATURE_DIAGNOSTIC_SURFACE_STACK_CAPACITY = 1 << 4
FEATURE_DIAGNOSTIC_INTERACTION_COVERAGE = 1 << 5


def _terrain_offsets_xz() -> tuple[tuple[int, int], ...]:
    offsets = [(0, 0)]
    for radius in (4, 8, 12, 18, 24):
        diagonal = int(radius / math.sqrt(2.0))
        offsets.extend(
            (
                (radius, 0),
                (diagonal, diagonal),
                (0, radius),
                (-diagonal, diagonal),
                (-radius, 0),
                (-diagonal, -diagonal),
                (0, -radius),
                (diagonal, -diagonal),
            )
        )
    return tuple(offsets)


TERRAIN_OFFSETS_XZ = _terrain_offsets_xz()

__all__ = [
    "FEATURE_DIAGNOSTIC_GRAPH_UNAVAILABLE",
    "FEATURE_DIAGNOSTIC_INTERACTION_COVERAGE",
    "FEATURE_DIAGNOSTIC_INVALID_ACTOR",
    "FEATURE_DIAGNOSTIC_SURFACE_STACK_CAPACITY",
    "FEATURE_DIAGNOSTIC_TERRAIN_COVERAGE",
    "FEATURE_DIAGNOSTIC_UNSUPPORTED_RUNTIME",
    "TERRAIN_SURFACE_STACK_CAPACITY",
]
