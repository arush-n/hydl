"""Public World-to-Combat explosion-query composition."""

from __future__ import annotations

import jax.numpy as jnp

from hytalegym.jax.combat.arsenal import ArsenalExplosionCandidates
from hytalegym.jax.combat.entities.impacts import (
    EntityImpactQueries,
    bind_entity_only_explosion_candidates,
)
from hytalegym.jax.combat.types import AGENT_ENTITY, CombatParams, CombatState
from hytalegym.jax.world import (
    PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NONE,
    PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_SURROGATE,
    PRIVILEGED_ENTITY_PROVENANCE_SURROGATE_RUNTIME,
    PrivilegedEntityScene,
    RegionActionRuntimeState,
    region_entity_only_explosion_candidates,
)


def bind_region_entity_only_explosion_queries(
    queries: EntityImpactQueries,
    runtime: RegionActionRuntimeState,
    scene: PrivilegedEntityScene,
    *,
    origin,
    block_damage_radius,
    entity_damage_radius,
    damage_blocks,
    query_mask,
    candidate_mask,
    candidate_generation,
) -> EntityImpactQueries:
    """Run World's certified Region fan and bind its complete result.

    World remains the only owner of block traversal and physical entity
    admission. Combat supplies logical-slot generations and receives its
    existing ``EntityImpactQueries`` representation.
    """

    result = region_entity_only_explosion_candidates(
        runtime,
        scene,
        origin,
        block_damage_radius,
        entity_damage_radius,
        damage_blocks,
        query_mask,
        candidate_mask,
    )
    return bind_entity_only_explosion_candidates(
        queries,
        query_mask=query_mask,
        available=result.available,
        candidate_mask=result.candidate_mask,
        distance=result.distance,
        candidate_generation=candidate_generation,
        geometry_exhausted=result.geometry_exhausted,
        capacity_exceeded=result.capacity_exceeded,
        damage_blocks_unsupported=result.damage_blocks_unsupported,
        invalid=result.invalid,
    )


def region_arsenal_entity_only_explosion_candidates(
    runtime: RegionActionRuntimeState,
    combat: CombatState,
    params: CombatParams,
    origin,
    block_damage_radius,
    entity_damage_radius,
    damage_blocks,
    query_mask,
    candidate_mask,
) -> ArsenalExplosionCandidates:
    """Project Arsenal entities and run World's certified Region fan.

    The projection is privileged simulator state used only by the physical
    kernel. It never enters the actor observation. World remains the sole
    owner of block traversal and entity-AABB admission.
    """

    result = region_entity_only_explosion_candidates(
        runtime,
        _arsenal_privileged_entity_scene(combat, params),
        origin,
        block_damage_radius,
        entity_damage_radius,
        damage_blocks,
        query_mask,
        candidate_mask,
    )
    return ArsenalExplosionCandidates(
        available=result.available,
        candidate_mask=result.candidate_mask,
        distance=result.distance,
        geometry_exhausted=result.geometry_exhausted,
        capacity_exceeded=result.capacity_exceeded,
        damage_blocks_unsupported=result.damage_blocks_unsupported,
        invalid=result.invalid,
    )


def _arsenal_privileged_entity_scene(
    combat: CombatState,
    params: CombatParams,
) -> PrivilegedEntityScene:
    if combat.position.ndim != 3 or combat.position.shape[2] != 3:
        raise ValueError("combat positions must have shape [batch, entity, 3]")
    batch, entities, _ = combat.position.shape
    entity_shape = (batch, entities)
    entity_id = jnp.arange(entities, dtype=jnp.uint32)
    batch_id = jnp.arange(batch, dtype=jnp.uint32)
    identity = jnp.stack(
        (
            jnp.broadcast_to(entity_id[None, :] + jnp.uint32(1), entity_shape),
            jnp.broadcast_to(batch_id[:, None] + jnp.uint32(1), entity_shape),
        ),
        axis=2,
    )
    bounds = (
        jnp.broadcast_to(
            jnp.asarray(params.target_bounds, dtype=jnp.float32),
            entity_shape + (6,),
        )
        .at[:, AGENT_ENTITY]
        .set(params.agent_bounds)
    )
    finite = (
        jnp.all(jnp.isfinite(combat.position), axis=2)
        & jnp.all(jnp.isfinite(combat.velocity), axis=2)
        & jnp.isfinite(combat.yaw)
        & jnp.all(jnp.isfinite(bounds), axis=2)
        & jnp.all(bounds[..., :3] < bounds[..., 3:], axis=2)
    )
    active = combat.health > jnp.float32(0.0)
    row_available = jnp.all(~active | finite, axis=1)
    physical = active & finite & row_available[:, None]
    component_provenance = jnp.where(
        physical,
        jnp.uint8(PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_SURROGATE),
        jnp.uint8(PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NONE),
    )
    zero_vec = jnp.zeros_like(combat.position, dtype=jnp.float32)
    return PrivilegedEntityScene(
        available=row_available,
        physical_available=row_available,
        dynamics_supported=jnp.zeros((batch,), dtype=jnp.bool_),
        capacity_exceeded=jnp.zeros((batch,), dtype=jnp.bool_),
        diagnostics=jnp.zeros((batch,), dtype=jnp.uint32),
        source_provenance=jnp.full(
            (batch,),
            PRIVILEGED_ENTITY_PROVENANCE_SURROGATE_RUNTIME,
            dtype=jnp.uint8,
        ),
        entity_count=jnp.sum(active, axis=1, dtype=jnp.int32),
        entity_mask=active,
        physical_mask=physical,
        source_slot=jnp.broadcast_to(
            jnp.arange(entities, dtype=jnp.int32)[None, :],
            entity_shape,
        ),
        active=active,
        identity_words=identity,
        type_identity_words=identity,
        kind=jnp.zeros(entity_shape, dtype=jnp.uint8),
        position=combat.position,
        rotation=jnp.stack(
            (
                jnp.zeros_like(combat.yaw),
                combat.yaw,
                jnp.zeros_like(combat.yaw),
            ),
            axis=2,
        ),
        velocity=combat.velocity,
        behavior_supported=jnp.zeros(entity_shape, dtype=jnp.bool_),
        collidable=physical,
        blocks_los=physical,
        local_bounds=bounds,
        los_offset_supported=jnp.zeros(entity_shape, dtype=jnp.bool_),
        los_offset=zero_vec,
        behavior_provenance=jnp.zeros(entity_shape, dtype=jnp.uint8),
        geometry_provenance=component_provenance,
    )


__all__ = [
    "bind_region_entity_only_explosion_queries",
    "region_arsenal_entity_only_explosion_candidates",
]
