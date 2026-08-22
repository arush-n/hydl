"""Host construction and JIT-safe per-entity region/cell selection."""

from __future__ import annotations

from collections.abc import Sequence
import operator

import jax.numpy as jnp
import numpy as np

from hytalegym.geometry.contract import (
    FLUID_MOVEMENT_FEATURES,
    MOVEMENT_FEATURES,
)
from hytalegym.jax.world.region.types import RegionAtlas, RegionCellSelection
from hytalegym.worldgen.region import (
    CAPTURE_CHUNK_COUNT,
    CAPTURE_CHUNKS_PER_AXIS,
    CAPTURE_HALO_CHUNKS,
    CHUNK_SIZE,
    CORE_CHUNKS_PER_AXIS,
    HEIGHT_SECTIONS,
    MAX_CELL_PALETTE,
    MAX_SHAPE_BOXES,
    MAX_SHAPE_PALETTE,
    MIN_Y,
    SECTION_VOLUME,
    NativeRegionSnapshot,
)


def region_atlas_from_snapshots(
    snapshots: Sequence[NativeRegionSnapshot],
    *,
    world_ids: Sequence[int] | None = None,
    region_capacity: int | None = None,
    cell_palette_capacity: int | None = None,
    shape_palette_capacity: int | None = None,
    validate_overlaps: bool = True,
) -> RegionAtlas:
    """Pad validated snapshots outside JIT and reject every overflow."""

    source = list(snapshots)
    if not source:
        raise ValueError("at least one native region snapshot is required")
    if any(not isinstance(item, NativeRegionSnapshot) for item in source):
        raise TypeError("region atlas accepts only NativeRegionSnapshot values")
    ids = (
        [0] * len(source)
        if world_ids is None
        else [_host_int(x, "world ID") for x in world_ids]
    )
    if len(ids) != len(source) or any(
        value < 0 or value > np.iinfo(np.int32).max for value in ids
    ):
        raise ValueError(
            "world_ids must be non-negative int32 values matching snapshots"
        )

    regions = _capacity(region_capacity, len(source), "region")
    cell_capacity = _capacity(
        cell_palette_capacity,
        max(item.cell_palette.size for item in source),
        "cell palette",
        maximum=MAX_CELL_PALETTE,
    )
    shape_capacity = _capacity(
        shape_palette_capacity,
        max(item.shape_palette.size for item in source),
        "shape palette",
        maximum=MAX_SHAPE_PALETTE,
    )
    if validate_overlaps:
        _require_consistent_overlaps(source, ids)

    region_mask = np.zeros(regions, dtype=np.bool_)
    world_id = np.full(regions, -1, dtype=np.int32)
    core_min = np.zeros((regions, 2), dtype=np.int32)
    known = np.zeros(
        (regions, CAPTURE_CHUNK_COUNT, HEIGHT_SECTIONS),
        dtype=np.bool_,
    )
    code = np.zeros(
        (
            regions,
            CAPTURE_CHUNK_COUNT,
            HEIGHT_SECTIONS,
            SECTION_VOLUME,
        ),
        dtype=np.uint16,
    )
    filler_root_known = np.zeros(regions, dtype=np.bool_)
    filler_root = np.zeros_like(code)
    cell_size = np.zeros(regions, dtype=np.int32)
    cell_flags = np.zeros((regions, cell_capacity), dtype=np.uint16)
    cell_shape = np.zeros((regions, cell_capacity), dtype=np.uint16)
    fluid_level = np.zeros((regions, cell_capacity), dtype=np.uint8)
    fluid_fill_height = np.zeros(
        (regions, cell_capacity),
        dtype=np.float32,
    )
    support = np.zeros((regions, cell_capacity), dtype=np.int32)
    block_damage = np.zeros((regions, cell_capacity), dtype=np.int32)
    fluid_damage = np.zeros((regions, cell_capacity), dtype=np.int32)
    movement = np.zeros(
        (regions, cell_capacity, MOVEMENT_FEATURES),
        dtype=np.float32,
    )
    fluid_movement = np.zeros(
        (regions, cell_capacity, FLUID_MOVEMENT_FEATURES),
        dtype=np.float32,
    )
    shape_size = np.zeros(regions, dtype=np.int32)
    boxes = np.zeros(
        (regions, shape_capacity, MAX_SHAPE_BOXES, 6),
        dtype=np.float32,
    )
    box_mask = np.zeros(
        (regions, shape_capacity, MAX_SHAPE_BOXES),
        dtype=np.bool_,
    )

    for index, (snapshot, identifier) in enumerate(zip(source, ids)):
        cell_count = snapshot.cell_palette.size
        shape_count = snapshot.shape_palette.size
        region_mask[index] = True
        world_id[index] = identifier
        core_min[index] = snapshot.core_min_chunk_xz
        known[index] = snapshot.section_known
        code[index] = snapshot.cell_code
        filler_root_known[index] = snapshot.filler_root_offsets_available
        filler_root[index] = snapshot.filler_root_offset_packed
        cell_size[index] = cell_count
        cell_flags[index, :cell_count] = snapshot.cell_palette.flags
        cell_shape[index, :cell_count] = snapshot.cell_palette.shape_index
        fluid_level[index, :cell_count] = snapshot.cell_palette.fluid_level
        fluid_fill_height[index, :cell_count] = (
            snapshot.cell_palette.fluid_fill_height
        )
        support[index, :cell_count] = snapshot.cell_palette.support
        block_damage[index, :cell_count] = snapshot.cell_palette.block_damage
        fluid_damage[index, :cell_count] = snapshot.cell_palette.fluid_damage
        movement[index, :cell_count] = snapshot.cell_palette.movement
        fluid_movement[index, :cell_count] = snapshot.cell_palette.fluid_movement
        shape_size[index] = shape_count
        boxes[index, :shape_count] = snapshot.shape_palette.boxes
        box_mask[index, :shape_count] = snapshot.shape_palette.box_mask

    return RegionAtlas(
        *(
            jnp.asarray(value)
            for value in (
                region_mask,
                world_id,
                core_min,
                known,
                code,
                filler_root_known,
                filler_root,
                cell_size,
                cell_flags,
                cell_shape,
                fluid_level,
                fluid_fill_height,
                support,
                block_damage,
                fluid_damage,
                movement,
                fluid_movement,
                shape_size,
                boxes,
                box_mask,
            )
        )
    )


def lookup_region_cells(
    atlas: RegionAtlas,
    positions: jnp.ndarray,
    environment_world_id: jnp.ndarray,
    *,
    require_core: bool = True,
) -> RegionCellSelection:
    """Look up ``[B,E,3]`` positions with independent per-entity selection."""

    points = jnp.asarray(positions, dtype=jnp.float32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("positions must have shape (batch, entities, 3)")
    return lookup_region_blocks(
        atlas,
        jnp.floor(points).astype(jnp.int32),
        environment_world_id,
        require_core=require_core,
    )


def lookup_region_blocks(
    atlas: RegionAtlas,
    block_positions: jnp.ndarray,
    environment_world_id: jnp.ndarray,
    *,
    require_core: bool = True,
) -> RegionCellSelection:
    """Look up integer block cells without coupling entities to one tile."""

    blocks = jnp.asarray(block_positions, dtype=jnp.int32)
    if blocks.ndim != 3 or blocks.shape[-1] != 3:
        raise ValueError("block_positions must have shape (batch, queries, 3)")
    batch, queries, _ = blocks.shape
    world_ids = jnp.asarray(environment_world_id, dtype=jnp.int32)
    if world_ids.shape != (batch,):
        raise ValueError("environment_world_id must have shape (batch,)")

    chunk_x = jnp.floor_divide(blocks[..., 0], CHUNK_SIZE)
    chunk_z = jnp.floor_divide(blocks[..., 2], CHUNK_SIZE)
    chunks = jnp.stack((chunk_x, chunk_z), axis=-1)
    relative_y = blocks[..., 1] - MIN_Y
    section_y = jnp.floor_divide(relative_y, CHUNK_SIZE)
    inside_y = (section_y >= 0) & (section_y < HEIGHT_SECTIONS)
    safe_section = jnp.clip(section_y, 0, HEIGHT_SECTIONS - 1)
    extent = CORE_CHUNKS_PER_AXIS if require_core else CAPTURE_CHUNKS_PER_AXIS

    core_relative = (
        chunks[:, :, None, :]
        - atlas.core_min_chunk_xz[None, None, :, :]
    )
    capture_relative = core_relative + 1
    relative = core_relative if require_core else capture_relative
    inside_xz = jnp.all(
        (relative >= 0) & (relative < extent),
        axis=-1,
    )
    safe_capture = jnp.clip(
        capture_relative,
        0,
        CAPTURE_CHUNKS_PER_AXIS - 1,
    )
    chunk_slot = (
        safe_capture[..., 0] * CAPTURE_CHUNKS_PER_AXIS
        + safe_capture[..., 1]
    )
    region_axis = jnp.arange(atlas.region_mask.shape[0])[None, None, :]
    known = atlas.section_known[
        region_axis,
        chunk_slot,
        safe_section[:, :, None],
    ]
    compatible = atlas.region_mask[None, None, :] & (
        atlas.world_id[None, None, :] == world_ids[:, None, None]
    )
    eligible = inside_xz & inside_y[:, :, None] & known & compatible

    # Score in doubled core-local block units. Global float32 positions lose
    # chunk-scale precision near Hytale's int32 world limits.
    local_xz = blocks[..., (0, 2)] & (CHUNK_SIZE - 1)
    safe_core_relative = jnp.clip(
        core_relative,
        -CAPTURE_HALO_CHUNKS,
        CORE_CHUNKS_PER_AXIS,
    )
    center_delta_twice = (
        safe_core_relative * (2 * CHUNK_SIZE)
        + local_xz[:, :, None, :] * 2
        + 1
        - CORE_CHUNKS_PER_AXIS * CHUNK_SIZE
    )
    distance = jnp.sum(center_delta_twice**2, axis=-1)
    score = jnp.where(eligible, distance, jnp.iinfo(jnp.int32).max)
    region_index = jnp.argmin(score, axis=2).astype(jnp.int32)
    candidate_available = jnp.any(eligible, axis=2)
    safe_region = jnp.where(candidate_available, region_index, 0)
    selected_slot = jnp.take_along_axis(
        chunk_slot,
        safe_region[..., None],
        axis=2,
    )[..., 0]

    local_x = blocks[..., 0] & (CHUNK_SIZE - 1)
    local_y = relative_y & (CHUNK_SIZE - 1)
    local_z = blocks[..., 2] & (CHUNK_SIZE - 1)
    section_index = local_y * (CHUNK_SIZE * CHUNK_SIZE) + local_z * CHUNK_SIZE + local_x
    code = atlas.cell_code[
        safe_region,
        selected_slot,
        safe_section,
        section_index,
    ].astype(jnp.int32)
    code_valid = code < atlas.cell_palette_size[safe_region]
    safe_code = jnp.where(code_valid, code, 0)
    shape_index = atlas.cell_shape_index[
        safe_region,
        safe_code,
    ].astype(jnp.int32)
    shape_valid = shape_index < atlas.shape_palette_size[safe_region]
    safe_shape = jnp.where(shape_valid, shape_index, 0)
    available = candidate_available & code_valid & shape_valid
    filler_root_packed = atlas.filler_root_offset_packed[
        safe_region,
        selected_slot,
        safe_section,
        section_index,
    ].astype(jnp.int32)
    filler_root_offset = _unpack_filler_root_offset(filler_root_packed)

    def cell_value(values: jnp.ndarray) -> jnp.ndarray:
        selected = values[safe_region, safe_code]
        mask = available.reshape(available.shape + (1,) * (selected.ndim - 2))
        return jnp.where(mask, selected, jnp.zeros_like(selected))

    boxes = atlas.collision_boxes[safe_region, safe_shape]
    box_mask = atlas.collision_box_mask[safe_region, safe_shape]
    shape_requires_filler = jnp.any(
        box_mask
        & (
            jnp.any(boxes[..., :3] < 0.0, axis=-1)
            | jnp.any(boxes[..., 3:] > 1.0, axis=-1)
        ),
        axis=-1,
    )
    filler_root_available = available & (
        atlas.filler_root_known[safe_region] | ~shape_requires_filler
    )
    box_available = available[..., None, None]
    mask_available = available[..., None]
    return RegionCellSelection(
        available=available,
        region_index=jnp.where(available, region_index, -1),
        cell_code=jnp.where(available, code, 0),
        flags=cell_value(atlas.cell_flags),
        shape_index=jnp.where(available, shape_index, 0),
        filler_root_available=filler_root_available,
        filler_root_offset=jnp.where(
            filler_root_available[..., None],
            filler_root_offset,
            0,
        ),
        fluid_level=cell_value(atlas.cell_fluid_level),
        fluid_fill_height=cell_value(
            atlas.cell_fluid_fill_height
        ),
        support=cell_value(atlas.cell_support),
        block_damage=cell_value(atlas.cell_block_damage),
        fluid_damage=cell_value(atlas.cell_fluid_damage),
        movement=cell_value(atlas.cell_movement),
        fluid_movement=cell_value(atlas.cell_fluid_movement),
        collision_boxes=jnp.where(box_available, boxes, 0.0),
        collision_box_mask=jnp.where(mask_available, box_mask, False),
    )


def estimated_region_atlas_bytes(
    *,
    region_capacity: int,
    cell_palette_capacity: int = MAX_CELL_PALETTE,
    shape_palette_capacity: int = MAX_SHAPE_PALETTE,
) -> int:
    """Exact payload bytes, excluding small runtime/XLA allocator overhead."""

    regions = _capacity(region_capacity, 1, "region")
    cells = _capacity(
        cell_palette_capacity,
        1,
        "cell palette",
        maximum=MAX_CELL_PALETTE,
    )
    shapes = _capacity(
        shape_palette_capacity,
        1,
        "shape palette",
        maximum=MAX_SHAPE_PALETTE,
    )
    per_region = (
        1
        + 4
        + 2 * 4
        + CAPTURE_CHUNK_COUNT * HEIGHT_SECTIONS
        + CAPTURE_CHUNK_COUNT * HEIGHT_SECTIONS * SECTION_VOLUME * 2
        + 1
        + CAPTURE_CHUNK_COUNT * HEIGHT_SECTIONS * SECTION_VOLUME * 2
        + 4
        + cells
        * (
            2
            + 2
            + 1
            + 4
            + 4
            + 4
            + 4
            + MOVEMENT_FEATURES * 4
            + FLUID_MOVEMENT_FEATURES * 4
        )
        + 4
        + shapes * (MAX_SHAPE_BOXES * 6 * 4 + MAX_SHAPE_BOXES)
    )
    return regions * per_region


def _unpack_filler_root_offset(packed: jnp.ndarray) -> jnp.ndarray:
    def signed_axis(value: jnp.ndarray) -> jnp.ndarray:
        return jnp.where((value & 16) != 0, value - 32, value)

    return jnp.stack(
        (
            signed_axis(packed & 31),
            signed_axis((packed >> 10) & 31),
            signed_axis((packed >> 5) & 31),
        ),
        axis=-1,
    ).astype(jnp.int32)


def _capacity(
    requested: int | None,
    required: int,
    label: str,
    *,
    maximum: int | None = None,
) -> int:
    result = (
        required if requested is None else _host_int(requested, f"{label} capacity")
    )
    if result < required or result <= 0:
        raise ValueError(f"{label} capacity does not cover supplied data")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} capacity exceeds schema maximum {maximum}")
    return result


def _host_int(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error


def _require_consistent_overlaps(
    snapshots: Sequence[NativeRegionSnapshot],
    world_ids: Sequence[int],
) -> None:
    digest_cache: dict[tuple[int, int, int], str] = {}

    def digest(snapshot_index: int, slot: int, section: int) -> str:
        key = (snapshot_index, slot, section)
        if key not in digest_cache:
            digest_cache[key] = snapshots[snapshot_index].section_semantic_digest(
                slot, section
            )
        return digest_cache[key]

    for left in range(len(snapshots)):
        for right in range(left + 1, len(snapshots)):
            if world_ids[left] != world_ids[right]:
                continue
            left_min = snapshots[left].capture_min_chunk_xz
            right_min = snapshots[right].capture_min_chunk_xz
            minimum = np.maximum(left_min, right_min)
            maximum = np.minimum(
                left_min + CAPTURE_CHUNKS_PER_AXIS,
                right_min + CAPTURE_CHUNKS_PER_AXIS,
            )
            for chunk_x in range(int(minimum[0]), int(maximum[0])):
                for chunk_z in range(int(minimum[1]), int(maximum[1])):
                    left_slot = (
                        (chunk_x - int(left_min[0])) * CAPTURE_CHUNKS_PER_AXIS
                        + chunk_z
                        - int(left_min[1])
                    )
                    right_slot = (
                        (chunk_x - int(right_min[0])) * CAPTURE_CHUNKS_PER_AXIS
                        + chunk_z
                        - int(right_min[1])
                    )
                    both_known = (
                        snapshots[left].section_known[left_slot]
                        & snapshots[right].section_known[right_slot]
                    )
                    for section in np.flatnonzero(both_known):
                        if digest(left, left_slot, int(section)) != digest(
                            right,
                            right_slot,
                            int(section),
                        ):
                            raise ValueError(
                                "overlapping native region sections disagree"
                            )
