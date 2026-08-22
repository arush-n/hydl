"""Atomic dry-block mutation planning for ordered explosion contacts.

This module deliberately does not reproduce ``ExplosionUtils``' sphere fan.
The caller supplies a complete, native-ordered fixed-capacity sequence of
ray-hit cells and effective ``BlockHarvestUtils`` damage.  This kernel only
turns that evidence into one revision-bound :class:`MutableBlockUpdate`.
``commit_region_block_mutation`` remains the sole Region-state writer.

The current slice excludes fluids and filler cells.  Those paths can change
the ray sequence or more than one cell, so treating either as an ordinary dry
block would be a silent fidelity error.  A rejected row produces no mutation,
drop, or entity receipt.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.mutable_blocks import (
    MutableBlockState,
    MutableBlockUpdate,
    empty_mutable_block_geometry,
    empty_mutable_block_update,
    query_mutable_blocks,
)
from hytalegym.jax.world.region.action_runtime import RegionActionRuntimeState
from hytalegym.jax.world.region.block_semantics import (
    query_region_block_action_target,
)
from hytalegym.worldgen.region.contract import HEIGHT_SECTIONS


Array = jax.Array
_WORLD_HEIGHT = HEIGHT_SECTIONS * 32
_FLOAT_EPSILON = jnp.finfo(jnp.float32).eps
_MAX_FIXED_ROWS = 64
# The production privileged-entity surface is capped at 256 rows.  Keep this
# consumer independently bounded so an untrusted source shape cannot expand a
# compiled mutation receipt without an explicit contract change.
MAX_DRY_BLOCK_EXPLOSION_ENTITY_CAPACITY = 256


class DryBlockExplosionMutationReceipt(NamedTuple):
    """Complete-or-empty mutation, drop-chance, and entity evidence."""

    available: Array
    update: MutableBlockUpdate
    changed_cell_mask: Array
    damaged_cell_mask: Array
    destroyed_cell_mask: Array
    position: Array
    health_before: Array
    health_after: Array
    drop_random_sample: Array
    drop_chance_passed: Array
    entity_candidate_mask: Array
    entity_distance: Array
    source_incomplete: Array
    entity_source_incomplete: Array
    coverage_unavailable: Array
    y_out_of_bounds: Array
    filler_unsupported: Array
    fluid_unsupported: Array
    support_cascade_unsupported: Array
    mutation_unsupported: Array
    duplicate_position: Array
    capacity_exceeded: Array
    invalid: Array
    resync_required: Array


def plan_dry_block_explosion_mutation(
    runtime: RegionActionRuntimeState,
    *,
    requested: Array,
    ray_hit_mask: Array,
    ray_hit_position: Array,
    effective_damage: Array,
    damage_evidence_available: Array,
    source_complete: Array,
    source_capacity_exceeded: Array,
    block_drop_chance: Array,
    drop_key: Array | None = None,
    drop_random_samples: Array | None = None,
    entity_candidate_mask: Array,
    entity_distance: Array,
    entity_source_complete: Array,
    filler_footprint_complete: Array,
    fluid_adjacency_complete: Array,
    support_cascade_complete: Array,
) -> DryBlockExplosionMutationReceipt:
    """Plan one atomic dry explosion update from an ordered hit sequence.

    ``effective_damage`` is the already-resolved value passed to native
    ``BlockHealthChunk.damageBlock`` after tool/gathering and radial falloff.
    ``damage_evidence_available`` must therefore be false for an unsupported
    tool-state replacement, fragile-block rule, cancellation, or other
    unresolved ``BlockHarvestUtils`` branch.

    Supply exactly one of ``drop_key`` or ``drop_random_samples``.  Keys are
    per-environment JAX keys, so changing batch composition cannot change a
    surviving environment's draw.  One draw is made per fixed row; the
    returned ``drop_chance_passed`` mask only records native's
    ``draw <= BlockDropChance`` predicate.  Item resolution stays with the
    existing drop catalogue.

    The three ``*_complete`` inputs are positive source certificates, not
    inferred local predicates.  They must be false unless the source proved
    that the complete filler footprint, adjacent-fluid response, and support
    cascade are all representable by this direct-cell transaction.  Entity
    evidence has a fixed width in ``[1,
    MAX_DRY_BLOCK_EXPLOSION_ENTITY_CAPACITY]``; wider or empty sources are
    rejected before planning.
    """

    if not isinstance(runtime, RegionActionRuntimeState):
        raise TypeError("runtime must be a RegionActionRuntimeState")
    state = runtime.geometry.mutable_blocks
    if not isinstance(state, MutableBlockState):
        raise TypeError("runtime must carry a MutableBlockState")

    positions = _int32_positions(ray_hit_position)
    batch, rays, _ = positions.shape
    if state.world_id.shape != (batch,):
        raise ValueError("runtime batch must match ray_hit_position")
    if rays < 1 or rays > _MAX_FIXED_ROWS:
        raise ValueError(
            f"ray rows must be in [1,{_MAX_FIXED_ROWS}]"
        )
    shape = (batch, rays)

    request = _array(requested, (batch,), jnp.bool_, "requested")
    hit = _array(ray_hit_mask, shape, jnp.bool_, "ray_hit_mask")
    damage = _array(
        effective_damage,
        shape,
        jnp.float32,
        "effective_damage",
    )
    damage_known = _array(
        damage_evidence_available,
        shape,
        jnp.bool_,
        "damage_evidence_available",
    )
    complete = _array(
        source_complete,
        (batch,),
        jnp.bool_,
        "source_complete",
    )
    source_overflow = _array(
        source_capacity_exceeded,
        (batch,),
        jnp.bool_,
        "source_capacity_exceeded",
    )
    drop_chance = _array(
        block_drop_chance,
        shape,
        jnp.float32,
        "block_drop_chance",
    )
    entity_mask = jnp.asarray(entity_candidate_mask, dtype=jnp.bool_)
    if entity_mask.ndim != 2 or entity_mask.shape[0] != batch:
        raise ValueError("entity_candidate_mask must have shape [batch, entity]")
    entities = entity_mask.shape[1]
    if not 1 <= entities <= MAX_DRY_BLOCK_EXPLOSION_ENTITY_CAPACITY:
        raise ValueError(
            "entity capacity must be in "
            f"[1,{MAX_DRY_BLOCK_EXPLOSION_ENTITY_CAPACITY}]"
        )
    entity_range = jnp.asarray(entity_distance, dtype=jnp.float32)
    if entity_range.shape != entity_mask.shape:
        raise ValueError("entity_distance must match entity_candidate_mask")
    entity_complete = _array(
        entity_source_complete,
        (batch,),
        jnp.bool_,
        "entity_source_complete",
    )
    filler_complete = _array(
        filler_footprint_complete,
        (batch,),
        jnp.bool_,
        "filler_footprint_complete",
    )
    fluid_complete = _array(
        fluid_adjacency_complete,
        (batch,),
        jnp.bool_,
        "fluid_adjacency_complete",
    )
    support_complete = _array(
        support_cascade_complete,
        (batch,),
        jnp.bool_,
        "support_cascade_complete",
    )
    drop_sample = _drop_samples(
        drop_key,
        drop_random_samples,
        batch=batch,
        rows=rays,
    )

    active = request[:, None] & hit
    immutable = query_region_block_action_target(
        runtime.semantic_atlas,
        runtime.geometry.atlas,
        positions,
        runtime.geometry.environment_world_id,
        require_core=True,
    )
    current = query_mutable_blocks(
        state,
        world_id=runtime.geometry.environment_world_id,
        base_semantic_sha256=state.base_semantic_sha256,
        resync_epoch=state.resync_epoch,
        position=immutable.action_position,
        base_available=immutable.query.available,
        base_geometry=immutable.query.geometry,
        base_block_health=immutable.query.block_health,
    )

    y_invalid_cell = (positions[..., 1] < 0) | (positions[..., 1] >= _WORLD_HEIGHT)
    y_out_of_bounds = request & jnp.any(active & y_invalid_cell, axis=1)
    coverage_unavailable = request & jnp.any(
        active & (~immutable.available | ~current.available),
        axis=1,
    )
    filler_cell = immutable.canonicalized_to_root | immutable.direct_filler
    filler_unsupported = request & (
        ~filler_complete | jnp.any(active & filler_cell, axis=1)
    )
    fluid_cell = (current.geometry.fluid_level != 0) | (
        current.geometry.fluid_fill_height != 0.0
    )
    fluid_unsupported = request & (
        ~fluid_complete | jnp.any(active & fluid_cell, axis=1)
    )
    support_cascade_unsupported = request & ~support_complete
    mutation_unsupported = request & jnp.any(
        active & (~damage_known | ~current.geometry.block_present),
        axis=1,
    )

    numeric_invalid_cell = active & (
        ~jnp.isfinite(damage)
        | (damage < 0.0)
        | ~jnp.isfinite(drop_chance)
        | (drop_chance < 0.0)
        | (drop_chance > 1.0)
        | ~jnp.isfinite(drop_sample)
        | (drop_sample < 0.0)
        | (drop_sample >= 1.0)
    )
    entity_invalid = jnp.any(
        entity_mask & (~jnp.isfinite(entity_range) | (entity_range < 0.0)),
        axis=1,
    )
    invalid = request & (jnp.any(numeric_invalid_cell, axis=1) | entity_invalid)

    canonical = immutable.action_position
    duplicate_position = request & _duplicate_positions(canonical, active)
    state_duplicate = _duplicate_positions(state.cell_position, state.cell_mask)
    will_damage = active & damage_known & (damage > 0.0)
    existing_match = state.cell_mask[:, None, :] & jnp.all(
        state.cell_position[:, None, :, :] == canonical[:, :, None, :],
        axis=3,
    )
    new_position = will_damage & ~jnp.any(existing_match, axis=2)
    storage_overflow = jnp.sum(new_position, axis=1) > jnp.sum(
        ~state.cell_mask,
        axis=1,
    )
    revision_wrap = (
        request
        & will_damage.any(axis=1)
        & (state.mutation_revision == jnp.uint32(0xFFFFFFFF))
    )
    capacity_exceeded = request & (source_overflow | storage_overflow)
    source_incomplete = request & (~complete | source_overflow)
    entity_source_incomplete = request & ~entity_complete
    state_stale = request & (
        ~state.synchronized
        | state_duplicate
        | jnp.any(active & current.resync_required, axis=1)
    )
    resync_required = (
        state_stale
        | source_incomplete
        | entity_source_incomplete
        | (request & storage_overflow)
        | revision_wrap
    )

    available = (
        request
        & ~source_incomplete
        & ~entity_source_incomplete
        & ~coverage_unavailable
        & ~y_out_of_bounds
        & ~filler_unsupported
        & ~fluid_unsupported
        & ~support_cascade_unsupported
        & ~mutation_unsupported
        & ~duplicate_position
        & ~capacity_exceeded
        & ~invalid
        & ~resync_required
    )

    damage_mask = available[:, None] & will_damage
    raw_health_after = current.block_health - damage
    destroyed = damage_mask & (
        (raw_health_after < 0.0) | (jnp.abs(raw_health_after) <= _FLOAT_EPSILON)
    )
    health_after = jnp.where(
        destroyed,
        0.0,
        jnp.maximum(raw_health_after, 0.0),
    )
    air = empty_mutable_block_geometry(shape, exact=True)
    empty = empty_mutable_block_geometry(shape)
    geometry = jax.tree_util.tree_map(
        lambda air_value, empty_value: jnp.where(
            destroyed.reshape(
                destroyed.shape + (1,) * (air_value.ndim - destroyed.ndim)
            ),
            air_value,
            empty_value,
        ),
        air,
        empty,
    )
    update = empty_mutable_block_update(
        state,
        query_capacity=rays,
    )._replace(
        mask=damage_mask,
        position=jnp.where(damage_mask[..., None], canonical, 0),
        geometry_changed=destroyed,
        health_changed=damage_mask,
        damage_applied=damage_mask,
        expected_health=jnp.where(damage_mask, current.block_health, 0.0),
        block_health_after=jnp.where(damage_mask, health_after, 0.0),
        provenance=jnp.where(damage_mask, current.provenance, jnp.uint8(0)),
        geometry=geometry,
    )
    drop_chance_passed = destroyed & (drop_sample <= drop_chance)
    published_entity_mask = available[:, None] & entity_mask

    return DryBlockExplosionMutationReceipt(
        available=available,
        update=update,
        changed_cell_mask=damage_mask,
        damaged_cell_mask=damage_mask,
        destroyed_cell_mask=destroyed,
        position=jnp.where(damage_mask[..., None], canonical, 0),
        health_before=jnp.where(damage_mask, current.block_health, 0.0),
        health_after=jnp.where(damage_mask, health_after, 0.0),
        drop_random_sample=jnp.where(
            available[:, None] & hit,
            drop_sample,
            0.0,
        ),
        drop_chance_passed=drop_chance_passed,
        entity_candidate_mask=published_entity_mask,
        entity_distance=jnp.where(
            published_entity_mask,
            entity_range,
            0.0,
        ),
        source_incomplete=source_incomplete,
        entity_source_incomplete=entity_source_incomplete,
        coverage_unavailable=coverage_unavailable,
        y_out_of_bounds=y_out_of_bounds,
        filler_unsupported=filler_unsupported,
        fluid_unsupported=fluid_unsupported,
        support_cascade_unsupported=support_cascade_unsupported,
        mutation_unsupported=mutation_unsupported,
        duplicate_position=duplicate_position,
        capacity_exceeded=capacity_exceeded,
        invalid=invalid,
        resync_required=resync_required,
    )


def _array(value: Array, shape: tuple[int, ...], dtype, label: str) -> Array:
    result = jnp.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    return result


def _int32_positions(value: Array) -> Array:
    """Validate concrete wide integers before lossless ``int32`` conversion."""

    concrete: np.ndarray | None = None
    if hasattr(value, "dtype") and hasattr(value, "shape"):
        dtype = np.dtype(value.dtype)
        shape = tuple(value.shape)
    else:
        concrete = np.asarray(value)
        dtype = concrete.dtype
        shape = concrete.shape
    if not np.issubdtype(dtype, np.integer):
        raise TypeError("ray_hit_position must use an integer dtype")
    if len(shape) != 3 or shape[-1] != 3:
        raise ValueError("ray_hit_position must have shape [batch, ray, 3]")

    if not np.can_cast(dtype, np.int32, casting="safe"):
        if concrete is None:
            try:
                concrete = np.asarray(value)
            except jax.errors.TracerArrayConversionError as error:
                raise TypeError(
                    "non-int32 ray_hit_position must be concrete for range "
                    "validation"
                ) from error
        if concrete.size:
            minimum = int(np.min(concrete))
            maximum = int(np.max(concrete))
            bounds = np.iinfo(np.int32)
            if minimum < bounds.min or maximum > bounds.max:
                raise ValueError(
                    "ray_hit_position contains coordinates outside int32 range"
                )
    return jnp.asarray(value if concrete is None else concrete, dtype=jnp.int32)


def _drop_samples(
    drop_key: Array | None,
    drop_random_samples: Array | None,
    *,
    batch: int,
    rows: int,
) -> Array:
    if (drop_key is None) == (drop_random_samples is None):
        raise ValueError(
            "supply exactly one of drop_key or drop_random_samples"
        )
    if drop_random_samples is not None:
        return _array(
            drop_random_samples,
            (batch, rows),
            jnp.float32,
            "drop_random_samples",
        )

    keys = jnp.asarray(drop_key)
    if keys.shape == () and batch == 1:
        keys = keys[None]
    if keys.shape != (batch,):
        raise ValueError("drop_key must contain one typed JAX key per batch row")
    return jax.vmap(
        lambda key: jax.random.uniform(
            key,
            shape=(rows,),
            minval=0.0,
            maxval=1.0,
            dtype=jnp.float32,
        )
    )(keys)


def _duplicate_positions(position: Array, mask: Array) -> Array:
    slots = mask.shape[1]
    same = (
        mask[:, :, None]
        & mask[:, None, :]
        & jnp.all(position[:, :, None, :] == position[:, None, :, :], axis=3)
    )
    upper = jnp.triu(jnp.ones((slots, slots), dtype=jnp.bool_), k=1)
    return jnp.any(same & upper[None, :, :], axis=(1, 2))


__all__ = [
    "DryBlockExplosionMutationReceipt",
    "MAX_DRY_BLOCK_EXPLOSION_ENTITY_CAPACITY",
    "plan_dry_block_explosion_mutation",
]
