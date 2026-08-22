"""Entity admission for Hytale's entity-only explosion fan."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.mutable_blocks import (
    MutableBlockState,
    mutable_block_contract_sha256,
)
from hytalegym.jax.world.entities.privileged import (
    PrivilegedEntityScene,
    privileged_entity_contract_sha256,
)
from hytalegym.jax.world.region.action_runtime import RegionActionRuntimeState
from hytalegym.jax.world.region.atlas import lookup_region_blocks
from hytalegym.jax.world.region.block_semantics import (
    lookup_region_block_semantics,
    region_block_semantic_atlas_contract_sha256,
)
from hytalegym.worldgen.block_affordances import block_affordance_tag_mask


Array = jax.Array
ENTITY_ONLY_EXPLOSION_SCHEMA = "hytalerl_entity_only_explosion_candidates_v1"
ENTITY_ONLY_EXPLOSION_VERSION = 1

# The largest installed entity-only root is Flame Staff Fireball_Impact_3:
# ExplosionUtils promotes its authored EntityDamageRadius=5 above the inherited
# BlockDamageRadius=3. A future larger asset must move this contract instead of
# silently truncating its native fan.
MAX_ENTITY_ONLY_EXPLOSION_BLOCK_RADIUS = 5
MAX_ENTITY_ONLY_EXPLOSION_RAYS = 691
MAX_ENTITY_ONLY_EXPLOSION_CELLS = 15

_SOFT_TAG = jnp.uint16(block_affordance_tag_mask("soft"))
_LINE_EPSILON = jnp.float32(1.0e-10)
_DDA_TIE_EPSILON = jnp.float32(1.0e-7)


class EntityOnlyExplosionCandidates(NamedTuple):
    """Native-shaped entity candidates with explicit completeness state."""

    available: Array
    candidate_mask: Array
    distance: Array
    potential_entity_mask: Array
    sphere_ray_count: Array
    traced_ray_count: Array
    geometry_exhausted: Array
    capacity_exceeded: Array
    damage_blocks_unsupported: Array
    invalid: Array


def region_entity_only_explosion_candidates(
    runtime: RegionActionRuntimeState,
    scene: PrivilegedEntityScene,
    origin: Array,
    block_damage_radius: Array,
    entity_damage_radius: Array,
    damage_blocks: Array,
    query_mask: Array,
    candidate_mask: Array,
    *,
    radius_capacity: int = MAX_ENTITY_ONLY_EXPLOSION_BLOCK_RADIUS,
) -> EntityOnlyExplosionCandidates:
    """Reproduce ``ExplosionUtils`` entity admission for non-block damage.

    The consumer owns relationship exclusions in ``candidate_mask``. World
    owns native spatial eligibility, the BlockSphereUtil ray fan, hard-vs-soft
    block stopping, and closed Box.intersectsLine admission. Block-damaging
    explosions fail closed because native mutates blocks while iterating the
    same fan, so a static pre-query would not preserve its semantics.
    """

    if not isinstance(runtime, RegionActionRuntimeState):
        raise TypeError("runtime must be a RegionActionRuntimeState")
    if not isinstance(scene, PrivilegedEntityScene):
        raise TypeError("scene must be a PrivilegedEntityScene")
    capacity = _radius_capacity(radius_capacity)
    points = jnp.asarray(origin, dtype=jnp.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("origin must have shape [batch, queries, 3]")
    batch, queries, _ = points.shape
    entities = scene.entity_mask.shape[1]
    if scene.entity_mask.shape[0] != batch:
        raise ValueError("scene batch must match origin")

    block_radius = _array(
        block_damage_radius,
        (batch, queries),
        jnp.int32,
        "block_damage_radius",
    )
    entity_radius = jnp.asarray(entity_damage_radius, dtype=jnp.float32)
    if entity_radius.shape != (batch, queries):
        raise ValueError("entity_damage_radius must have shape [batch, queries]")
    mutates_blocks = _array(
        damage_blocks,
        (batch, queries),
        jnp.bool_,
        "damage_blocks",
    )
    requested = _array(
        query_mask,
        (batch, queries),
        jnp.bool_,
        "query_mask",
    )
    requested_candidates = _array(
        candidate_mask,
        (batch, queries, entities),
        jnp.bool_,
        "candidate_mask",
    )
    _validate_scene_shapes(scene, batch, entities)

    finite = jnp.all(jnp.isfinite(points), axis=2) & jnp.isfinite(entity_radius)
    valid_radius = (block_radius > 0) & (entity_radius > 0.0)
    scene_ready = (
        scene.available
        & scene.physical_available
        & ~scene.capacity_exceeded
    )
    invalid = requested & (~finite | ~valid_radius | ~scene_ready[:, None])
    unsupported = requested & mutates_blocks

    # ExplosionUtils replaces the block radius with Java's truncated entity
    # radius only when the latter compares larger before truncation.
    truncated_entity_radius = jnp.trunc(entity_radius).astype(jnp.int32)
    fan_radius = jnp.where(
        entity_radius > block_radius.astype(jnp.float32),
        truncated_entity_radius,
        block_radius,
    )
    capacity_exceeded = requested & (fan_radius > capacity)
    query_valid = requested & ~invalid & ~unsupported & ~capacity_exceeded
    safe_radius = jnp.clip(fan_radius, 1, capacity)

    entity_finite = jnp.all(jnp.isfinite(scene.position), axis=2) & jnp.all(
        jnp.isfinite(scene.local_bounds), axis=2
    )
    entity_bounds_valid = jnp.all(
        scene.local_bounds[..., :3] < scene.local_bounds[..., 3:],
        axis=2,
    )
    physical_entity = (
        scene.entity_mask
        & scene.physical_mask
        & scene.active
        & entity_finite
        & entity_bounds_valid
    )
    distance = jnp.linalg.norm(
        scene.position[:, None, :, :] - points[:, :, None, :],
        axis=3,
    )
    potential = (
        requested_candidates
        & physical_entity[:, None, :]
        & (distance < entity_radius[..., None])
        & query_valid[..., None]
    )

    offsets, membership = _sphere_arrays(capacity)
    ray_mask = membership[safe_radius] & query_valid[..., None]
    centers = (
        jnp.floor(points).astype(jnp.float32)[..., None, :]
        + offsets.astype(jnp.float32)[None, None, :, :]
        + jnp.float32(0.5)
    )
    nonzero_ray = jnp.any(centers != points[..., None, :], axis=3)
    ray_mask &= nonzero_ray

    world_boxes = jnp.concatenate(
        (
            scene.position + scene.local_bounds[..., :3],
            scene.position + scene.local_bounds[..., 3:],
        ),
        axis=2,
    )
    full_intersection = _segments_intersect_closed_boxes(
        points[..., None, None, :],
        centers[..., None, :],
        world_boxes[:, None, None, :, :],
    )
    relevant_ray = ray_mask & jnp.any(
        full_intersection & potential[:, :, None, :],
        axis=3,
    )

    cells, step_mask = _block_iterator_cells(points, centers)
    flat_cells = cells.reshape(batch, -1, 3)
    cell_available, hard = _runtime_hard_blocks(runtime, flat_cells)
    shape = (batch, queries, offsets.shape[0], MAX_ENTITY_ONLY_EXPLOSION_CELLS)
    cell_available = cell_available.reshape(shape)
    hard = hard.reshape(shape)
    required_step = step_mask & relevant_ray[..., None]
    ray_complete = jnp.all(~required_step | cell_available, axis=3)
    geometry_exhausted = query_valid & jnp.any(
        relevant_ray & ~ray_complete,
        axis=2,
    )

    eligible_hard = hard & step_mask
    has_hard = jnp.any(eligible_hard, axis=3)
    first_hard = jnp.argmax(eligible_hard, axis=3).astype(jnp.int32)
    first_hard_cell = jnp.take_along_axis(
        cells,
        first_hard[..., None, None],
        axis=3,
    )[..., 0, :]
    stopped_center = jnp.where(
        has_hard[..., None],
        first_hard_cell.astype(jnp.float32) + jnp.float32(0.5),
        centers,
    )
    final_intersection = _segments_intersect_closed_boxes(
        points[..., None, None, :],
        stopped_center[..., None, :],
        world_boxes[:, None, None, :, :],
    )
    complete_query = query_valid & ~geometry_exhausted
    admitted = potential & jnp.any(
        final_intersection
        & relevant_ray[..., None]
        & ray_complete[..., None],
        axis=2,
    )
    admitted &= complete_query[..., None]

    return EntityOnlyExplosionCandidates(
        available=complete_query,
        candidate_mask=admitted,
        distance=jnp.where(admitted, distance, 0.0),
        potential_entity_mask=potential,
        sphere_ray_count=jnp.sum(ray_mask, axis=2, dtype=jnp.int32),
        traced_ray_count=jnp.sum(relevant_ray, axis=2, dtype=jnp.int32),
        geometry_exhausted=geometry_exhausted,
        capacity_exceeded=capacity_exceeded,
        damage_blocks_unsupported=unsupported,
        invalid=invalid,
    )


def entity_only_explosion_contract() -> dict[str, object]:
    """Return the checkpoint-pinnable entity-only explosion contract."""

    return {
        "schema": ENTITY_ONLY_EXPLOSION_SCHEMA,
        "version": ENTITY_ONLY_EXPLOSION_VERSION,
        "server_version": "0.5.7",
        "producer": "region_entity_only_explosion_candidates",
        "native_sources": [
            "ExplosionUtils.processTargetBlocks",
            "ExplosionUtils.isValidTargetBlock",
            "ExplosionUtils.processPotentialEntity",
            "BlockSphereUtil.forEachBlock",
            "TargetUtil.getTargetBlock",
            "BlockIterator.iterate",
            "KDTree.collect",
            "Box.intersectsLine",
        ],
        "scope": {
            "damage_entities": True,
            "damage_blocks": False,
            "maximum_block_radius": MAX_ENTITY_ONLY_EXPLOSION_BLOCK_RADIUS,
            "maximum_rays": MAX_ENTITY_ONLY_EXPLOSION_RAYS,
            "maximum_cells_per_ray": MAX_ENTITY_ONLY_EXPLOSION_CELLS,
            "installed_roots": ["Flame_Staff_projectiles", "Bombs"],
            "larger_future_asset": "capacity_exceeded_fail_closed",
            "damage_blocks_true": (
                "unavailable_interleaved_native_block_mutation_required"
            ),
        },
        "candidate_semantics": {
            "spatial_prefilter": "strict_transform_distance_less_than_radius",
            "consumer_exclusions": "candidate_mask",
            "entity_shape": "world_offset_BoundingBox",
            "block_stop": "non_air_non_soft_current_block",
            "line_intersection": "closed_slab_epsilon_1e-10",
            "unavailable_relevant_ray": "fail_complete_query_closed",
        },
        "dependencies": {
            "region_block_semantic_atlas_sha256": (
                region_block_semantic_atlas_contract_sha256()
            ),
            "mutable_block_sha256": mutable_block_contract_sha256(),
            "privileged_entity_sha256": privileged_entity_contract_sha256(),
        },
        "provenance": "native_source_transcribed_with_bounded_live_admission_gate",
    }


def entity_only_explosion_contract_sha256() -> str:
    payload = json.dumps(
        entity_only_explosion_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _runtime_hard_blocks(
    runtime: RegionActionRuntimeState,
    positions: Array,
) -> tuple[Array, Array]:
    geometry = runtime.geometry
    physical = lookup_region_blocks(
        geometry.atlas,
        positions,
        geometry.environment_world_id,
        require_core=True,
    )
    semantic = lookup_region_block_semantics(
        runtime.semantic_atlas,
        geometry.atlas,
        positions,
        geometry.environment_world_id,
        require_core=True,
    )
    block_present = physical.cell_code != 0
    semantic_known = ~block_present | semantic.block_present
    available = physical.available & semantic.available & semantic_known
    hard = block_present & ((semantic.affordance_tags & _SOFT_TAG) == 0)

    state = geometry.mutable_blocks
    if state is None:
        return available, available & hard
    if not isinstance(state, MutableBlockState):
        raise TypeError("runtime mutable_blocks must be a MutableBlockState")
    batch, queries, _ = positions.shape
    cells = state.cell_mask.shape[1]
    if state.world_id.shape != (batch,):
        raise ValueError("mutable state batch must match explosion queries")

    duplicate = _duplicate_mutable_positions(state)
    identity = (
        state.synchronized
        & (state.world_id == geometry.environment_world_id)
        & ~duplicate
    )

    def overlay_slot(index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        current_available, current_hard = carry
        applies = (
            state.cell_mask[:, index, None]
            & state.geometry_override[:, index, None]
            & jnp.all(
                positions == state.cell_position[:, index, None, :],
                axis=2,
            )
        )
        stored_present = state.geometry.block_present[:, index]
        stored_known = state.geometry.exact[:, index] & (
            ~stored_present | state.geometry.affordance_valid[:, index]
        )
        stored_hard = stored_present & (
            (state.geometry.affordance_tags[:, index] & _SOFT_TAG) == 0
        )
        return (
            jnp.where(applies, stored_known[:, None], current_available),
            jnp.where(applies, stored_hard[:, None], current_hard),
        )

    available, hard = jax.lax.fori_loop(
        0,
        cells,
        overlay_slot,
        (available, hard),
    )
    available &= identity[:, None]
    return available, available & hard


def _duplicate_mutable_positions(state: MutableBlockState) -> Array:
    cells = state.cell_mask.shape[1]
    same = (
        state.cell_mask[:, :, None]
        & state.cell_mask[:, None, :]
        & jnp.all(
            state.cell_position[:, :, None, :]
            == state.cell_position[:, None, :, :],
            axis=3,
        )
    )
    upper = jnp.triu(jnp.ones((cells, cells), dtype=jnp.bool_), k=1)
    return jnp.any(same & upper[None, :, :], axis=(1, 2))


def _block_iterator_cells(start: Array, end: Array) -> tuple[Array, Array]:
    direction = end - start[..., None, :]
    cell = jnp.floor(start).astype(jnp.int32)[..., None, :]
    cell = jnp.broadcast_to(cell, direction.shape)
    step = jnp.sign(direction).astype(jnp.int32)
    moving = direction != 0.0
    boundary = jnp.where(step > 0, cell + 1, cell).astype(jnp.float32)
    safe_direction = jnp.where(moving, direction, 1.0)
    t_max = jnp.where(moving, (boundary - start[..., None, :]) / safe_direction, jnp.inf)
    t_delta = jnp.where(moving, jnp.abs(1.0 / safe_direction), jnp.inf)
    current_t = jnp.zeros(direction.shape[:-1], dtype=jnp.float32)

    def step_cell(carry, _):
        current_cell, crossing, parameter = carry
        active = parameter <= jnp.float32(1.0) + _DDA_TIE_EPSILON
        next_parameter = jnp.min(crossing, axis=3)
        advance = moving & (
            jnp.abs(crossing - next_parameter[..., None]) <= _DDA_TIE_EPSILON
        )
        next_cell = current_cell + jnp.where(advance, step, 0)
        next_crossing = crossing + jnp.where(advance, t_delta, 0.0)
        return (next_cell, next_crossing, next_parameter), (current_cell, active)

    _, (cells, mask) = jax.lax.scan(
        step_cell,
        (cell, t_max, current_t),
        xs=None,
        length=MAX_ENTITY_ONLY_EXPLOSION_CELLS,
    )
    return jnp.moveaxis(cells, 0, 3), jnp.moveaxis(mask, 0, 3)


def _segments_intersect_closed_boxes(start: Array, end: Array, boxes: Array) -> Array:
    direction = end - start
    moving = jnp.abs(direction) >= _LINE_EPSILON
    safe_direction = jnp.where(moving, direction, 1.0)
    first = (boxes[..., :3] - start) / safe_direction
    second = (boxes[..., 3:] - start) / safe_direction
    entry = jnp.where(moving, jnp.minimum(first, second), 0.0)
    exit_time = jnp.where(moving, jnp.maximum(first, second), 1.0)
    stationary_inside = (start >= boxes[..., :3]) & (start <= boxes[..., 3:])
    return (
        jnp.all(moving | stationary_inside, axis=-1)
        & (
            jnp.maximum(jnp.max(entry, axis=-1), 0.0)
            <= jnp.minimum(jnp.min(exit_time, axis=-1), 1.0)
        )
    )


@lru_cache(maxsize=MAX_ENTITY_ONLY_EXPLOSION_BLOCK_RADIUS)
def _sphere_host(radius_capacity: int) -> tuple[np.ndarray, np.ndarray]:
    rows = [_block_sphere_offsets(radius) for radius in range(1, radius_capacity + 1)]
    offsets = np.asarray(sorted(set().union(*rows)), dtype=np.int32)
    membership = np.zeros((radius_capacity + 1, len(offsets)), dtype=np.bool_)
    slot = {tuple(value): index for index, value in enumerate(offsets.tolist())}
    for radius, values in enumerate(rows, start=1):
        membership[radius, [slot[value] for value in values]] = True
    return offsets, membership


def _sphere_arrays(radius_capacity: int) -> tuple[Array, Array]:
    offsets, membership = _sphere_host(radius_capacity)
    return jnp.asarray(offsets), jnp.asarray(membership)


def _block_sphere_offsets(radius: int) -> set[tuple[int, int, int]]:
    adjusted = np.float32(radius + np.float32(0.41))
    inverse = np.float32(1.0) / np.float32(adjusted * adjusted)
    result: set[tuple[int, int, int]] = set()
    for x in range(-radius, radius + 1):
        qx = np.float32(1.0) - np.float32(x * x) * inverse
        if qx < 0.0:
            continue
        dy = math.sqrt(float(qx)) * float(adjusted)
        low_y = max(math.ceil(-dy), -radius)
        high_y = min(int(dy), radius)
        for y in range(low_y, high_y + 1):
            qxy = qx - np.float32(y * y) * inverse
            if qxy < 0.0:
                continue
            dz = math.sqrt(float(qxy)) * float(adjusted)
            low_z = max(math.ceil(-dz), -radius)
            high_z = min(int(dz), radius)
            result.update((x, y, z) for z in range(low_z, high_z + 1))
    return result


def _radius_capacity(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("radius_capacity must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError("radius_capacity must be an integer") from error
    if not 1 <= result <= MAX_ENTITY_ONLY_EXPLOSION_BLOCK_RADIUS:
        raise ValueError(
            "radius_capacity must be in [1, "
            f"{MAX_ENTITY_ONLY_EXPLOSION_BLOCK_RADIUS}]"
        )
    return result


def _array(value: Array, shape: tuple[int, ...], dtype, label: str) -> Array:
    result = jnp.asarray(value)
    if result.dtype != dtype:
        raise TypeError(f"{label} must use dtype {dtype}")
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    return result


def _validate_scene_shapes(scene: PrivilegedEntityScene, batch: int, entities: int) -> None:
    shape = (batch, entities)
    for name, tail in (
        ("entity_mask", ()),
        ("physical_mask", ()),
        ("active", ()),
        ("position", (3,)),
        ("local_bounds", (6,)),
    ):
        if getattr(scene, name).shape != shape + tail:
            raise ValueError(f"scene {name} has an invalid shape")


__all__ = [
    "ENTITY_ONLY_EXPLOSION_SCHEMA",
    "ENTITY_ONLY_EXPLOSION_VERSION",
    "MAX_ENTITY_ONLY_EXPLOSION_BLOCK_RADIUS",
    "MAX_ENTITY_ONLY_EXPLOSION_CELLS",
    "MAX_ENTITY_ONLY_EXPLOSION_RAYS",
    "EntityOnlyExplosionCandidates",
    "entity_only_explosion_contract",
    "entity_only_explosion_contract_sha256",
    "region_entity_only_explosion_candidates",
]
