"""World-owned surrogate feature candidates for downstream legal filtering."""

from hytalegym.jax.world.surrogate.features.contract import (
    FEATURE_DIAGNOSTIC_GRAPH_UNAVAILABLE,
    FEATURE_DIAGNOSTIC_INTERACTION_COVERAGE,
    FEATURE_DIAGNOSTIC_INVALID_ACTOR,
    FEATURE_DIAGNOSTIC_SURFACE_STACK_CAPACITY,
    FEATURE_DIAGNOSTIC_TERRAIN_COVERAGE,
    FEATURE_DIAGNOSTIC_UNSUPPORTED_RUNTIME,
    TERRAIN_SURFACE_STACK_CAPACITY,
)
from hytalegym.jax.world.surrogate.features.interactions import (
    apply_surrogate_door_intent,
)
from hytalegym.jax.world.surrogate.features.producer import (
    produce_surrogate_world_features,
)

__all__ = [
    "FEATURE_DIAGNOSTIC_GRAPH_UNAVAILABLE",
    "FEATURE_DIAGNOSTIC_INTERACTION_COVERAGE",
    "FEATURE_DIAGNOSTIC_INVALID_ACTOR",
    "FEATURE_DIAGNOSTIC_SURFACE_STACK_CAPACITY",
    "FEATURE_DIAGNOSTIC_TERRAIN_COVERAGE",
    "FEATURE_DIAGNOSTIC_UNSUPPORTED_RUNTIME",
    "TERRAIN_SURFACE_STACK_CAPACITY",
    "apply_surrogate_door_intent",
    "produce_surrogate_world_features",
]
