"""Resolve exact placed geometry and actor aim for Region world actions."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.types import AGENT_ENTITY
from hytalegym.jax.world import (
    ActorBlockPlacementCandidates,
    MutableBlockGeometry,
    RegionActionRuntimeState,
    query_region_block_action_target,
    region_action_runtime_contract_sha256,
    region_block_semantic_atlas_contract_sha256,
    region_placement_action_runtime_contract_sha256,
)
from hytalegym.worldgen.region import (
    CAPTURE_CHUNKS_PER_AXIS,
    CAPTURE_HALO_CHUNKS,
    CHUNK_SIZE,
    CORE_CHUNKS_PER_AXIS,
    MIN_Y,
)


REGION_PLACED_GEOMETRY_SCHEMA = "hytalerl_region_placed_geometry_v1"
REGION_PLACED_GEOMETRY_VERSION = 1


def region_placed_geometry_contract_sha256() -> str:
    """Return the exact, fail-closed placed-geometry contract identity."""

    payload = {
        "schema": REGION_PLACED_GEOMETRY_SCHEMA,
        "version": REGION_PLACED_GEOMETRY_VERSION,
        "dependencies": {
            "region_action_runtime_contract_sha256": (
                region_action_runtime_contract_sha256()
            ),
            "region_block_semantic_atlas_contract_sha256": (
                region_block_semantic_atlas_contract_sha256()
            ),
            "region_placement_action_runtime_contract_sha256": (
                region_placement_action_runtime_contract_sha256()
            ),
        },
        "identity": "native_BlockType_reference_plus_rotation_zero",
        "geometry_source": (
            "direct_non_filler_exact_matching_semantic_row_in_each_loaded_"
            "Region"
        ),
        "admission": (
            "one_semantic_palette_row_and_one_exact_direct_core_instance"
        ),
        "missing_identity": "unavailable_no_fabricated_full_cube",
        "published_provenance": "native_exact_region_projection",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest().upper()


class RegionPlacedGeometryTable(NamedTuple):
    """One exact representative geometry per Region/item/trigger row."""

    available: jax.Array
    geometry: MutableBlockGeometry


def compile_region_placed_geometry_table(
    runtime: RegionActionRuntimeState,
    place_semantic_key: jax.Array,
    place_semantic_key_valid: jax.Array,
) -> RegionPlacedGeometryTable:
    """Resolve placed block geometry from the exact loaded Region corpus.

    A placed identity is admitted only when every selected Region contains a
    direct, non-filler, rotation-exact instance of that same semantic key.
    Missing identities remain unavailable; no cube or asset default is
    fabricated.
    """

    if not isinstance(runtime, RegionActionRuntimeState):
        raise TypeError("runtime must be a RegionActionRuntimeState")
    keys = np.asarray(jax.device_get(place_semantic_key), dtype=np.uint32)
    valid = np.asarray(
        jax.device_get(place_semantic_key_valid),
        dtype=np.bool_,
    )
    if keys.ndim != 3 or valid.shape != keys.shape[:2]:
        raise ValueError(
            "place semantic keys must have shape [item,trigger,words]"
        )

    physical = runtime.geometry.atlas
    semantic = runtime.semantic_atlas
    region_mask = np.asarray(jax.device_get(physical.region_mask), dtype=np.bool_)
    world_id = np.asarray(jax.device_get(physical.world_id), dtype=np.int32)
    core_min = np.asarray(
        jax.device_get(physical.core_min_chunk_xz),
        dtype=np.int32,
    )
    semantic_code = np.asarray(
        jax.device_get(semantic.cell_code),
        dtype=np.uint16,
    )
    semantic_keys = np.asarray(
        jax.device_get(semantic.semantic_key),
        dtype=np.uint32,
    )
    entry_valid = np.asarray(
        jax.device_get(semantic.entry_valid),
        dtype=np.bool_,
    )
    semantic_known = np.asarray(
        jax.device_get(semantic.section_known),
        dtype=np.bool_,
    )
    physical_known = np.asarray(
        jax.device_get(physical.section_known),
        dtype=np.bool_,
    )
    filler_known = np.asarray(
        jax.device_get(physical.filler_root_known),
        dtype=np.bool_,
    )
    filler_offset = np.asarray(
        jax.device_get(physical.filler_root_offset_packed),
        dtype=np.uint16,
    )

    region_capacity = region_mask.shape[0]
    item_capacity, trigger_capacity = valid.shape
    query_capacity = item_capacity * trigger_capacity
    positions = np.zeros((region_capacity, query_capacity, 3), dtype=np.int32)
    source_available = np.zeros(
        (region_capacity, query_capacity),
        dtype=np.bool_,
    )
    for region in range(region_capacity):
        if not region_mask[region]:
            continue
        for item in range(item_capacity):
            for trigger in range(trigger_capacity):
                slot = item * trigger_capacity + trigger
                if not valid[item, trigger]:
                    continue
                matches = np.flatnonzero(
                    entry_valid[region]
                    & np.all(
                        semantic_keys[region] == keys[item, trigger],
                        axis=1,
                    )
                )
                if matches.size != 1:
                    continue
                palette_code = int(matches[0])
                position = _first_direct_core_position(
                    semantic_code[region],
                    semantic_known[region],
                    physical_known[region],
                    filler_known[region],
                    filler_offset[region],
                    core_min[region],
                    palette_code,
                )
                if position is None:
                    continue
                positions[region, slot] = position
                source_available[region, slot] = True

    queried = query_region_block_action_target(
        semantic,
        physical,
        jnp.asarray(positions),
        jnp.asarray(world_id),
        require_core=True,
    )
    source = jnp.asarray(source_available)
    position_match = jnp.all(
        queried.action_position == jnp.asarray(positions),
        axis=2,
    )
    ready = (
        source
        & queried.available
        & position_match
        & ~queried.canonicalized_to_root
        & ~queried.direct_filler
        & queried.query.geometry.exact
        & queried.query.geometry.block_present
        & queried.query.geometry.semantic_key_valid
    )
    shape = (region_capacity, item_capacity, trigger_capacity)
    geometry = jax.tree.map(
        lambda value: value.reshape(shape + value.shape[2:]),
        queried.query.geometry,
    )
    ready = ready.reshape(shape)
    return RegionPlacedGeometryTable(
        available=ready,
        geometry=jax.tree.map(
            lambda value: jnp.where(
                ready.reshape(
                    ready.shape + (1,) * (value.ndim - ready.ndim)
                ),
                value,
                jnp.zeros_like(value),
            ),
            geometry,
        ),
    )


def select_region_placed_geometry(
    table: RegionPlacedGeometryTable,
    runtime: RegionActionRuntimeState,
    item_row: jax.Array,
    trigger_index: jax.Array,
) -> tuple[MutableBlockGeometry, jax.Array]:
    """Select one exact geometry row for each environment world."""

    if not isinstance(table, RegionPlacedGeometryTable):
        raise TypeError("table must be a RegionPlacedGeometryTable")
    world = runtime.geometry.environment_world_id
    atlas = runtime.geometry.atlas
    matches = (
        atlas.region_mask[None, :]
        & (atlas.world_id[None, :] == world[:, None])
    )
    unique = jnp.sum(matches, axis=1) == 1
    region_row = jnp.argmax(matches, axis=1).astype(jnp.int32)
    row = jnp.asarray(item_row, dtype=jnp.int32)
    trigger = jnp.asarray(trigger_index, dtype=jnp.int32)
    selected = jax.tree.map(
        lambda value: value[region_row, row, trigger][:, None, ...],
        table.geometry,
    )
    available = table.available[region_row, row, trigger] & unique
    selected = jax.tree.map(
        lambda value: jnp.where(
            available.reshape(
                (available.shape[0], 1)
                + (1,) * (value.ndim - 2)
            ),
            value,
            jnp.zeros_like(value),
        ),
        selected,
    )
    return selected, available


def select_actor_aim_block_face(
    state,
    params,
    placements: ActorBlockPlacementCandidates,
    clicked_position: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Resolve a unique protocol face from the actor's current aim ray."""

    if not isinstance(placements, ActorBlockPlacementCandidates):
        raise TypeError("placements must be ActorBlockPlacementCandidates")
    clicked = jnp.asarray(clicked_position, dtype=jnp.int32)
    if clicked.ndim != 3 or clicked.shape[1:] != (1, 3):
        raise ValueError("clicked_position must have shape [batch,1,3]")
    origin = (
        state.combat.position[:, AGENT_ENTITY]
        + jnp.asarray(params.agent_eye_offset, dtype=jnp.float32)
    )
    # You place where you are looking, so this ray is head-frame. Both head
    # scalars are per batch for the controlled actor; the target's are separate
    # fields again.
    yaw = jnp.deg2rad(state.combat.agent_head_yaw)
    pitch = jnp.deg2rad(state.combat.agent_head_pitch)
    horizontal = jnp.cos(pitch)
    direction = jnp.stack(
        (
            -jnp.sin(yaw) * horizontal,
            jnp.sin(pitch),
            -jnp.cos(yaw) * horizontal,
        ),
        axis=1,
    )
    minimum = clicked[:, 0].astype(jnp.float32)
    maximum = minimum + 1.0
    nonzero = jnp.abs(direction) > jnp.float32(1.0e-7)
    inverse = jnp.where(nonzero, 1.0 / direction, 0.0)
    first = (minimum - origin) * inverse
    second = (maximum - origin) * inverse
    inside = (origin >= minimum) & (origin < maximum)
    axis_entry = jnp.where(
        nonzero,
        jnp.minimum(first, second),
        jnp.where(inside, -jnp.inf, jnp.inf),
    )
    axis_exit = jnp.where(
        nonzero,
        jnp.maximum(first, second),
        jnp.where(inside, jnp.inf, -jnp.inf),
    )
    entry = jnp.max(axis_entry, axis=1)
    exit = jnp.min(axis_exit, axis=1)
    axis = jnp.argmax(axis_entry, axis=1)
    selected_entry = jnp.take_along_axis(
        axis_entry,
        axis[:, None],
        axis=1,
    )[:, 0]
    unique_axis = jnp.sum(
        jnp.abs(axis_entry - selected_entry[:, None]) <= jnp.float32(1.0e-6),
        axis=1,
    ) == 1
    component = jnp.take_along_axis(
        direction,
        axis[:, None],
        axis=1,
    )[:, 0]
    positive_faces = jnp.asarray((6, 2, 3), dtype=jnp.uint8)
    negative_faces = jnp.asarray((5, 1, 4), dtype=jnp.uint8)
    face = jnp.where(
        component > 0.0,
        positive_faces[axis],
        negative_faces[axis],
    )
    ray_hit = unique_axis & (entry >= 0.0) & (entry <= exit)
    candidate_match = (
        placements.candidate_mask[:, AGENT_ENTITY]
        & placements.available[:, AGENT_ENTITY, None]
        & jnp.all(
            placements.clicked_position[:, AGENT_ENTITY]
            == clicked[:, 0, None, :],
            axis=2,
        )
        & (placements.block_face[:, AGENT_ENTITY] == face[:, None])
    )
    available = ray_hit & (jnp.sum(candidate_match, axis=1) == 1)
    return jnp.where(available, face, jnp.uint8(0)), available


def _first_direct_core_position(
    semantic_code: np.ndarray,
    semantic_known: np.ndarray,
    physical_known: np.ndarray,
    filler_known: np.ndarray,
    filler_offset: np.ndarray,
    core_min_chunk_xz: np.ndarray,
    palette_code: int,
) -> tuple[int, int, int] | None:
    for relative_x in range(
        CAPTURE_HALO_CHUNKS,
        CAPTURE_HALO_CHUNKS + CORE_CHUNKS_PER_AXIS,
    ):
        for relative_z in range(
            CAPTURE_HALO_CHUNKS,
            CAPTURE_HALO_CHUNKS + CORE_CHUNKS_PER_AXIS,
        ):
            chunk_slot = (
                relative_x * CAPTURE_CHUNKS_PER_AXIS + relative_z
            )
            for section in range(semantic_code.shape[1]):
                if not (
                    semantic_known[chunk_slot, section]
                    and physical_known[chunk_slot, section]
                ):
                    continue
                direct = (
                    (semantic_code[chunk_slot, section] == palette_code)
                    & filler_known
                    & (filler_offset[chunk_slot, section] == 0)
                )
                indexes = np.flatnonzero(direct)
                if indexes.size == 0:
                    continue
                cell = int(indexes[0])
                local_y, remainder = divmod(cell, CHUNK_SIZE * CHUNK_SIZE)
                local_z, local_x = divmod(remainder, CHUNK_SIZE)
                chunk_x = (
                    int(core_min_chunk_xz[0])
                    + relative_x
                    - CAPTURE_HALO_CHUNKS
                )
                chunk_z = (
                    int(core_min_chunk_xz[1])
                    + relative_z
                    - CAPTURE_HALO_CHUNKS
                )
                return (
                    chunk_x * CHUNK_SIZE + local_x,
                    MIN_Y + section * CHUNK_SIZE + local_y,
                    chunk_z * CHUNK_SIZE + local_z,
                )
    return None


__all__ = [
    "REGION_PLACED_GEOMETRY_SCHEMA",
    "REGION_PLACED_GEOMETRY_VERSION",
    "RegionPlacedGeometryTable",
    "compile_region_placed_geometry_table",
    "region_placed_geometry_contract_sha256",
    "select_actor_aim_block_face",
    "select_region_placed_geometry",
]
