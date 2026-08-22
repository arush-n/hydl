"""World-owned fixed feature candidates from exact surrogate geometry."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.observation.v1.runtime.factory import (
    empty_injected_world_features,
)
from hytalegym.jax.world.surrogate.features.common import (
    validate_feature_shapes,
)
from hytalegym.jax.world.surrogate.features.contract import (
    FEATURE_DIAGNOSTIC_GRAPH_UNAVAILABLE,
    FEATURE_DIAGNOSTIC_INTERACTION_COVERAGE,
    FEATURE_DIAGNOSTIC_INVALID_ACTOR,
    FEATURE_DIAGNOSTIC_SURFACE_STACK_CAPACITY,
    FEATURE_DIAGNOSTIC_TERRAIN_COVERAGE,
    FEATURE_DIAGNOSTIC_UNSUPPORTED_RUNTIME,
)
from hytalegym.jax.world.surrogate.features.interactions import (
    interaction_features,
)
from hytalegym.jax.world.surrogate.features.terrain import terrain_features
from hytalegym.jax.world.surrogate.types import (
    SurrogateAtlas,
    SurrogateRuntimeState,
    SurrogateTraversalAtlas,
    SurrogateWorldFeatureResult,
)
from hytalegym.worldgen.region import MIN_Y, WORLD_HEIGHT


def produce_surrogate_world_features(
    atlas: SurrogateAtlas,
    traversal: SurrogateTraversalAtlas,
    runtime: SurrogateRuntimeState,
    actor_positions: jax.Array,
    actor_yaw_degrees: jax.Array,
) -> SurrogateWorldFeatureResult:
    """Produce privileged terrain and door candidates for each actor."""

    if not isinstance(atlas, SurrogateAtlas):
        raise TypeError("atlas must be a SurrogateAtlas")
    if not isinstance(traversal, SurrogateTraversalAtlas):
        raise TypeError("traversal must be a SurrogateTraversalAtlas")
    if not isinstance(runtime, SurrogateRuntimeState):
        raise TypeError("runtime must be a SurrogateRuntimeState")
    positions = jnp.asarray(actor_positions, dtype=jnp.float32)
    yaw_degrees = jnp.asarray(actor_yaw_degrees, dtype=jnp.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("actor_positions must have shape [batch, 3]")
    batch = positions.shape[0]
    if yaw_degrees.shape != (batch,):
        raise ValueError("actor_yaw_degrees must have shape [batch]")
    validate_feature_shapes(atlas, traversal, runtime, batch)

    finite = jnp.all(jnp.isfinite(positions), axis=1) & jnp.isfinite(yaw_degrees)
    precise = jnp.all(jnp.abs(positions) < jnp.float32(1 << 24), axis=1)
    inside_y = (positions[:, 1] >= MIN_Y) & (positions[:, 1] < MIN_Y + WORLD_HEIGHT)
    actor_valid = finite & precise & inside_y
    safe_positions = jnp.where(actor_valid[:, None], positions, 0.0)
    terrain = terrain_features(
        atlas,
        traversal,
        runtime,
        safe_positions,
        actor_valid,
    )
    interactions = interaction_features(
        atlas,
        runtime,
        safe_positions,
        yaw_degrees,
        actor_valid,
    )
    unsupported = runtime.unsupported_mechanics
    diagnostics = (
        jnp.where(
            ~actor_valid,
            jnp.uint32(FEATURE_DIAGNOSTIC_INVALID_ACTOR),
            jnp.uint32(0),
        )
        | jnp.where(
            unsupported,
            jnp.uint32(FEATURE_DIAGNOSTIC_UNSUPPORTED_RUNTIME),
            jnp.uint32(0),
        )
        | jnp.where(
            terrain.missing_tile,
            jnp.uint32(FEATURE_DIAGNOSTIC_TERRAIN_COVERAGE),
            jnp.uint32(0),
        )
        | jnp.where(
            terrain.graph_unavailable,
            jnp.uint32(FEATURE_DIAGNOSTIC_GRAPH_UNAVAILABLE),
            jnp.uint32(0),
        )
        | jnp.where(
            terrain.overflow,
            jnp.uint32(FEATURE_DIAGNOSTIC_SURFACE_STACK_CAPACITY),
            jnp.uint32(0),
        )
        | jnp.where(
            interactions.missing_tile,
            jnp.uint32(FEATURE_DIAGNOSTIC_INTERACTION_COVERAGE),
            jnp.uint32(0),
        )
    )
    valid = (
        actor_valid
        & ~unsupported
        & ~terrain.missing_tile
        & ~terrain.graph_unavailable
        & ~terrain.overflow
        & ~interactions.missing_tile
        & ~interactions.overflow
    )
    empty = empty_injected_world_features(batch)
    features = empty._replace(
        terrain_f32=terrain.f32,
        terrain_semantic_id=terrain.semantic_id,
        terrain_flags=terrain.flags,
        terrain_mask=terrain.mask,
        terrain_overflow=terrain.overflow,
        interaction_f32=interactions.f32,
        interaction_object_id=interactions.object_id,
        interaction_is_door=interactions.mask,
        interaction_door_intent_mask=interactions.intent_mask,
        interaction_mask=interactions.mask,
        interaction_overflow=interactions.overflow,
    )
    return SurrogateWorldFeatureResult(
        features=features,
        valid=valid,
        diagnostics=diagnostics,
        graph_diagnostics=terrain.graph_diagnostics,
        terrain_source_tile_index=terrain.tile_index,
        interaction_source_tile_index=interactions.tile_index,
        traversal_provider_available=jnp.zeros(batch, dtype=jnp.bool_),
    )


__all__ = ["produce_surrogate_world_features"]
