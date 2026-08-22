"""Exact block identity and action targets aligned to Region geometry."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import operator
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.mutable_blocks import (
    MUTABLE_BLOCK_PROVENANCE_NATIVE,
    MUTABLE_BLOCK_PROVENANCE_NONE,
    MutableBlockGeometry,
    MutableBlockQueryResult,
    mutable_block_contract_sha256,
)
from hytalegym.jax.world.region.atlas import lookup_region_blocks
from hytalegym.jax.world.region.types import RegionAtlas, RegionCellSelection
from hytalegym.worldgen.region import (
    CAPTURE_CHUNK_COUNT,
    CAPTURE_CHUNKS_PER_AXIS,
    BLOCK_SEMANTIC_KEY_WORDS,
    CHUNK_SIZE,
    HEIGHT_SECTIONS,
    MIN_Y,
    SECTION_VOLUME,
    NativeRegionBlockSemanticSnapshot,
    NativeRegionSnapshot,
    region_block_semantic_contract_sha256,
)


Array = jax.Array
REGION_BLOCK_SEMANTIC_ATLAS_SCHEMA = (
    "hytalerl_region_block_semantic_atlas_v1"
)
REGION_BLOCK_SEMANTIC_ATLAS_VERSION = 1

REGION_BLOCK_ACTION_DIAGNOSTIC_CLICKED_UNAVAILABLE = 1 << 0
REGION_BLOCK_ACTION_DIAGNOSTIC_SEMANTICS_UNAVAILABLE = 1 << 1
REGION_BLOCK_ACTION_DIAGNOSTIC_FILLER_ROOT_UNAVAILABLE = 1 << 2
REGION_BLOCK_ACTION_DIAGNOSTIC_FILLER_IDENTITY_UNAVAILABLE = 1 << 3


class RegionBlockSemanticAtlas(NamedTuple):
    """Fixed semantic sidecars in exactly the physical Region atlas order."""

    region_mask: Array
    section_known: Array
    cell_code: Array
    palette_size: Array
    semantic_key: Array
    asset_key: Array
    entry_valid: Array
    affordance_tags: Array
    gather_type_index: Array
    required_tool_quality: Array
    rotation_index: Array


class RegionBlockSemanticSelection(NamedTuple):
    """Per-cell portable identity selected by the physical Region lookup."""

    available: Array
    block_present: Array
    semantic_key: Array
    asset_key: Array
    affordance_tags: Array
    gather_type_index: Array
    required_tool_quality: Array
    rotation_index: Array


class RegionBlockActionQueryResult(NamedTuple):
    """Filler-aware immutable target suitable for mutable-block composition."""

    available: Array
    action_position: Array
    canonicalized_to_root: Array
    direct_filler: Array
    diagnostics: Array
    query: MutableBlockQueryResult


def region_block_semantic_atlas_without_evidence(
    physical_atlas: RegionAtlas,
    *,
    palette_capacity: int = 1,
) -> RegionBlockSemanticAtlas:
    """An atlas that declares *no* block identity for any world.

    Every mask is false, so ``region_block_semantic_selection`` returns
    ``available = physical.available & region_mask[...]`` = False everywhere and
    the block-interaction heads close. That is the same "absent, not zero"
    convention the rest of the observation uses: a false mask means the fact is
    unavailable, and a consumer must not read the zeros behind it.

    **This is not a stand-in for real semantics and must never be used to make a
    place/break surface look supported.** It exists so a Region library that
    ships complete terrain and traversal but no block-semantic sidecars can
    still run a movement-only scene. ``region-library-v3-288`` is exactly that:
    288 Regions, 288 traversal graphs, and sidecars that cannot be captured
    because the live server no longer regenerates the terrain they were recorded
    from. Without this, that corpus is unusable for training even though nothing
    in a pursuit or duel scene ever asks a block a question.

    The alternative -- fabricating palette entries so the shapes line up -- would
    publish invented block identities, which is the failure the arsenal already
    paid for once with invented melee cones.
    """

    region_capacity = int(physical_atlas.region_mask.shape[0])
    capacity = _capacity(palette_capacity, 1, "semantic palette")
    return RegionBlockSemanticAtlas(
        jnp.zeros(region_capacity, dtype=jnp.bool_),
        jnp.zeros(
            (region_capacity, CAPTURE_CHUNK_COUNT, HEIGHT_SECTIONS),
            dtype=jnp.bool_,
        ),
        jnp.zeros(
            (
                region_capacity,
                CAPTURE_CHUNK_COUNT,
                HEIGHT_SECTIONS,
                SECTION_VOLUME,
            ),
            dtype=jnp.uint16,
        ),
        jnp.zeros(region_capacity, dtype=jnp.int32),
        jnp.zeros(
            (region_capacity, capacity, BLOCK_SEMANTIC_KEY_WORDS),
            dtype=jnp.uint32,
        ),
        jnp.zeros(
            (region_capacity, capacity, BLOCK_SEMANTIC_KEY_WORDS),
            dtype=jnp.uint32,
        ),
        jnp.zeros((region_capacity, capacity), dtype=jnp.bool_),
        jnp.zeros((region_capacity, capacity), dtype=jnp.uint16),
        jnp.zeros((region_capacity, capacity), dtype=jnp.uint8),
        jnp.zeros((region_capacity, capacity), dtype=jnp.int16),
        jnp.zeros((region_capacity, capacity), dtype=jnp.int32),
    )


def region_block_semantic_atlas_from_snapshots(
    semantic_snapshots: Sequence[NativeRegionBlockSemanticSnapshot],
    physical_snapshots: Sequence[NativeRegionSnapshot],
    physical_atlas: RegionAtlas,
    *,
    world_ids: Sequence[int] | None = None,
    palette_capacity: int | None = None,
) -> RegionBlockSemanticAtlas:
    """Build an atlas only when semantic and physical evidence align exactly."""

    semantics = tuple(semantic_snapshots)
    physical = tuple(physical_snapshots)
    if not semantics or len(semantics) != len(physical):
        raise ValueError(
            "semantic and physical snapshots must be nonempty and paired"
        )
    if any(
        not isinstance(value, NativeRegionBlockSemanticSnapshot)
        for value in semantics
    ):
        raise TypeError("semantic snapshots have the wrong type")
    if any(not isinstance(value, NativeRegionSnapshot) for value in physical):
        raise TypeError("physical snapshots have the wrong type")

    ids = (
        [0] * len(semantics)
        if world_ids is None
        else [_host_int(value, "world ID") for value in world_ids]
    )
    if len(ids) != len(semantics) or any(
        value < 0 or value > np.iinfo(np.int32).max for value in ids
    ):
        raise ValueError(
            "world_ids must be nonnegative int32 values matching snapshots"
        )

    region_capacity = int(physical_atlas.region_mask.shape[0])
    if len(semantics) > region_capacity:
        raise ValueError("semantic snapshots exceed the physical atlas")
    capacity = _capacity(
        palette_capacity,
        max(len(snapshot.palette) for snapshot in semantics),
        "semantic palette",
    )
    _require_physical_atlas_alignment(physical_atlas, physical, ids)

    region_mask = np.zeros(region_capacity, dtype=np.bool_)
    section_known = np.zeros(
        (region_capacity, CAPTURE_CHUNK_COUNT, HEIGHT_SECTIONS),
        dtype=np.bool_,
    )
    code = np.zeros(
        (
            region_capacity,
            CAPTURE_CHUNK_COUNT,
            HEIGHT_SECTIONS,
            SECTION_VOLUME,
        ),
        dtype=np.uint16,
    )
    palette_size = np.zeros(region_capacity, dtype=np.int32)
    semantic_key = np.zeros(
        (region_capacity, capacity, BLOCK_SEMANTIC_KEY_WORDS),
        dtype=np.uint32,
    )
    asset_key = np.zeros_like(semantic_key)
    entry_valid = np.zeros((region_capacity, capacity), dtype=np.bool_)
    tags = np.zeros((region_capacity, capacity), dtype=np.uint16)
    gather = np.zeros((region_capacity, capacity), dtype=np.uint8)
    quality = np.zeros((region_capacity, capacity), dtype=np.int16)
    rotation = np.zeros((region_capacity, capacity), dtype=np.int32)

    for index, (semantic, source) in enumerate(zip(semantics, physical)):
        source_digest = source.semantic_digest()
        if semantic.source_region_semantic_sha256 != source_digest:
            raise ValueError(
                "semantic sidecar names a different physical Region"
            )
        if not np.array_equal(
            semantic.core_min_chunk_xz,
            source.core_min_chunk_xz,
        ) or not np.array_equal(
            semantic.section_known,
            source.section_known,
        ):
            raise ValueError(
                "semantic sidecar topology differs from physical Region"
            )
        entries = semantic.palette
        region_mask[index] = True
        section_known[index] = semantic.section_known
        code[index] = semantic.cell_code
        palette_size[index] = len(entries)
        for slot, entry in enumerate(entries):
            semantic_key[index, slot] = entry.semantic_key
            asset_key[index, slot] = entry.asset_key
            entry_valid[index, slot] = entry.valid
            tags[index, slot] = entry.affordance_tags
            gather[index, slot] = entry.gather_type_index
            quality[index, slot] = entry.required_tool_quality
            rotation[index, slot] = entry.rotation_index

    return RegionBlockSemanticAtlas(
        *(
            jnp.asarray(value)
            for value in (
                region_mask,
                section_known,
                code,
                palette_size,
                semantic_key,
                asset_key,
                entry_valid,
                tags,
                gather,
                quality,
                rotation,
            )
        )
    )


def lookup_region_block_semantics(
    semantic_atlas: RegionBlockSemanticAtlas,
    physical_atlas: RegionAtlas,
    block_positions: Array,
    environment_world_id: Array,
    *,
    require_core: bool = True,
) -> RegionBlockSemanticSelection:
    """Select semantic values through the authoritative physical tile lookup."""

    positions = jnp.asarray(block_positions, dtype=jnp.int32)
    physical = lookup_region_blocks(
        physical_atlas,
        positions,
        environment_world_id,
        require_core=require_core,
    )
    return _lookup_aligned_semantics(
        semantic_atlas,
        physical_atlas,
        positions,
        physical,
    )


def query_region_block_action_target(
    semantic_atlas: RegionBlockSemanticAtlas,
    physical_atlas: RegionAtlas,
    block_positions: Array,
    environment_world_id: Array,
    *,
    require_core: bool = True,
) -> RegionBlockActionQueryResult:
    """Resolve native filler-root rules and publish an exact immutable base."""

    positions = jnp.asarray(block_positions, dtype=jnp.int32)
    clicked = lookup_region_blocks(
        physical_atlas,
        positions,
        environment_world_id,
        require_core=require_core,
    )
    clicked_semantics = _lookup_aligned_semantics(
        semantic_atlas,
        physical_atlas,
        positions,
        clicked,
    )
    safe_region = jnp.maximum(clicked.region_index, 0)
    root_offsets_known = (
        clicked.available
        & physical_atlas.filler_root_known[safe_region]
    )
    is_filler = jnp.any(clicked.filler_root_offset != 0, axis=-1)
    root_position = positions - clicked.filler_root_offset
    root = lookup_region_blocks(
        physical_atlas,
        root_position,
        environment_world_id,
        require_core=require_core,
    )
    root_semantics = _lookup_aligned_semantics(
        semantic_atlas,
        physical_atlas,
        root_position,
        root,
    )
    filler_identity_known = (
        root_offsets_known
        & clicked_semantics.available
        & root_semantics.available
        & clicked_semantics.block_present
        & root_semantics.block_present
        & jnp.any(clicked_semantics.asset_key != 0, axis=-1)
        & jnp.any(root_semantics.asset_key != 0, axis=-1)
    )
    same_asset = jnp.all(
        clicked_semantics.asset_key == root_semantics.asset_key,
        axis=-1,
    )
    canonicalize = is_filler & filler_identity_known & same_asset
    direct_filler = is_filler & filler_identity_known & ~same_asset
    action_position = jnp.where(
        canonicalize[..., None],
        root_position,
        positions,
    )
    selected = lookup_region_blocks(
        physical_atlas,
        action_position,
        environment_world_id,
        require_core=require_core,
    )
    selected_semantics = _lookup_aligned_semantics(
        semantic_atlas,
        physical_atlas,
        action_position,
        selected,
    )

    clicked_unavailable = ~clicked.available
    semantics_unavailable = clicked.available & ~clicked_semantics.available
    root_unavailable = clicked.available & ~root_offsets_known
    identity_unavailable = (
        is_filler & root_offsets_known & ~filler_identity_known
    )
    diagnostics = (
        jnp.where(
            clicked_unavailable,
            jnp.uint32(
                REGION_BLOCK_ACTION_DIAGNOSTIC_CLICKED_UNAVAILABLE
            ),
            jnp.uint32(0),
        )
        | jnp.where(
            semantics_unavailable,
            jnp.uint32(
                REGION_BLOCK_ACTION_DIAGNOSTIC_SEMANTICS_UNAVAILABLE
            ),
            jnp.uint32(0),
        )
        | jnp.where(
            root_unavailable,
            jnp.uint32(
                REGION_BLOCK_ACTION_DIAGNOSTIC_FILLER_ROOT_UNAVAILABLE
            ),
            jnp.uint32(0),
        )
        | jnp.where(
            identity_unavailable,
            jnp.uint32(
                REGION_BLOCK_ACTION_DIAGNOSTIC_FILLER_IDENTITY_UNAVAILABLE
            ),
            jnp.uint32(0),
        )
    )
    available = (
        (diagnostics == 0)
        & selected.available
        & selected_semantics.available
    )
    query = _immutable_query(available, selected, selected_semantics)
    return RegionBlockActionQueryResult(
        available=available,
        action_position=jnp.where(
            available[..., None],
            action_position,
            0,
        ),
        canonicalized_to_root=available & canonicalize,
        direct_filler=available & direct_filler,
        diagnostics=diagnostics,
        query=query,
    )


def estimated_region_block_semantic_atlas_bytes(
    *,
    region_capacity: int,
    palette_capacity: int,
) -> int:
    """Exact array payload bytes, excluding runtime allocator overhead."""

    regions = _capacity(region_capacity, 1, "region")
    palette = _capacity(palette_capacity, 1, "semantic palette")
    section_cells = CAPTURE_CHUNK_COUNT * HEIGHT_SECTIONS * SECTION_VOLUME
    topology = regions * (
        1
        + CAPTURE_CHUNK_COUNT * HEIGHT_SECTIONS
        + section_cells * 2
        + 4
    )
    entry_bytes = BLOCK_SEMANTIC_KEY_WORDS * 4 * 2 + 1 + 2 + 1 + 2 + 4
    return topology + regions * palette * entry_bytes


def region_block_semantic_atlas_contract() -> dict[str, object]:
    return {
        "schema": REGION_BLOCK_SEMANTIC_ATLAS_SCHEMA,
        "version": REGION_BLOCK_SEMANTIC_ATLAS_VERSION,
        "server_version": "0.5.7",
        "source_contract_sha256": region_block_semantic_contract_sha256(),
        "mutable_block_contract_sha256": mutable_block_contract_sha256(),
        "selection": (
            "physical_Region_tile_is_authoritative_semantics_reuse_its_index"
        ),
        "alignment": [
            "physical_expanded_semantic_sha256",
            "core_min_chunk_xz",
            "section_known",
            "world_id_and_physical_atlas_order",
        ],
        "fixed_capacity": {
            "cell_code": "uint16_5x5x10x32768_per_region",
            "palette": "host_selected_reject_overflow",
        },
        "filler_root": {
            "same_asset_key": "canonicalize_action_to_root",
            "different_asset_key": "act_on_clicked_filler_without_root_drop",
            "missing_offset_or_identity": "fail_closed",
        },
        "immutable_base": {
            "health": "one_for_present_zero_for_absent",
            "provenance": "native",
            "runtime_block_ordinal": "unavailable",
            "geometry": "exact_physical_Region_value",
        },
        "policy_binding": (
            "staged_for_single_announced_action_surface_contract_move"
        ),
    }


def region_block_semantic_atlas_contract_sha256() -> str:
    payload = json.dumps(
        region_block_semantic_atlas_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _lookup_aligned_semantics(
    semantic_atlas: RegionBlockSemanticAtlas,
    physical_atlas: RegionAtlas,
    positions: Array,
    physical: RegionCellSelection,
) -> RegionBlockSemanticSelection:
    safe_region = jnp.maximum(physical.region_index, 0)
    chunk_x = jnp.floor_divide(positions[..., 0], CHUNK_SIZE)
    chunk_z = jnp.floor_divide(positions[..., 2], CHUNK_SIZE)
    chunks = jnp.stack((chunk_x, chunk_z), axis=-1)
    capture_relative = (
        chunks - physical_atlas.core_min_chunk_xz[safe_region] + 1
    )
    capture_inside = jnp.all(
        (capture_relative >= 0)
        & (capture_relative < CAPTURE_CHUNKS_PER_AXIS),
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
    relative_y = positions[..., 1] - MIN_Y
    section_y = jnp.floor_divide(relative_y, CHUNK_SIZE)
    inside_y = (section_y >= 0) & (section_y < HEIGHT_SECTIONS)
    safe_section = jnp.clip(section_y, 0, HEIGHT_SECTIONS - 1)
    local_x = positions[..., 0] & (CHUNK_SIZE - 1)
    local_y = relative_y & (CHUNK_SIZE - 1)
    local_z = positions[..., 2] & (CHUNK_SIZE - 1)
    cell_index = (
        local_y * CHUNK_SIZE * CHUNK_SIZE
        + local_z * CHUNK_SIZE
        + local_x
    )
    code = semantic_atlas.cell_code[
        safe_region,
        chunk_slot,
        safe_section,
        cell_index,
    ].astype(jnp.int32)
    code_valid = code < semantic_atlas.palette_size[safe_region]
    safe_code = jnp.where(code_valid, code, 0)
    available = (
        physical.available
        & semantic_atlas.region_mask[safe_region]
        & capture_inside
        & inside_y
        & semantic_atlas.section_known[
            safe_region,
            chunk_slot,
            safe_section,
        ]
        & code_valid
    )

    def entry(values: Array) -> Array:
        selected = values[safe_region, safe_code]
        mask = available.reshape(
            available.shape + (1,) * (selected.ndim - available.ndim)
        )
        return jnp.where(mask, selected, jnp.zeros_like(selected))

    valid = available & semantic_atlas.entry_valid[
        safe_region,
        safe_code,
    ]
    return RegionBlockSemanticSelection(
        available=available,
        block_present=valid,
        semantic_key=entry(semantic_atlas.semantic_key),
        asset_key=entry(semantic_atlas.asset_key),
        affordance_tags=entry(semantic_atlas.affordance_tags),
        gather_type_index=entry(semantic_atlas.gather_type_index),
        required_tool_quality=entry(
            semantic_atlas.required_tool_quality
        ),
        rotation_index=entry(semantic_atlas.rotation_index),
    )


def _immutable_query(
    available: Array,
    physical: RegionCellSelection,
    semantic: RegionBlockSemanticSelection,
) -> MutableBlockQueryResult:
    zeros_i32 = jnp.zeros_like(physical.flags, dtype=jnp.int32)

    def mask(value: Array) -> Array:
        condition = available.reshape(
            available.shape + (1,) * (value.ndim - available.ndim)
        )
        return jnp.where(condition, value, jnp.zeros_like(value))

    block_present = available & semantic.block_present
    geometry = MutableBlockGeometry(
        exact=available,
        block_present=block_present,
        runtime_block_id=zeros_i32,
        runtime_block_id_valid=jnp.zeros_like(available),
        semantic_key=mask(semantic.semantic_key),
        semantic_key_valid=block_present,
        affordance_valid=block_present,
        affordance_tags=mask(semantic.affordance_tags),
        gather_type_index=mask(semantic.gather_type_index),
        required_tool_quality=mask(semantic.required_tool_quality),
        rotation_index=mask(semantic.rotation_index),
        flags=mask(physical.flags.astype(jnp.int32)),
        fluid_level=mask(physical.fluid_level.astype(jnp.int32)),
        fluid_fill_height=mask(physical.fluid_fill_height),
        support=mask(physical.support.astype(jnp.int32)),
        block_damage=mask(physical.block_damage.astype(jnp.int32)),
        fluid_damage=mask(physical.fluid_damage.astype(jnp.int32)),
        movement=mask(physical.movement),
        fluid_movement=mask(physical.fluid_movement),
        collision_boxes=mask(physical.collision_boxes),
        collision_box_mask=mask(physical.collision_box_mask),
    )
    health_valid = block_present
    return MutableBlockQueryResult(
        available=available,
        overridden=jnp.zeros_like(available),
        block_health=jnp.where(health_valid, 1.0, 0.0).astype(jnp.float32),
        block_health_valid=health_valid,
        provenance=jnp.where(
            available,
            jnp.uint8(MUTABLE_BLOCK_PROVENANCE_NATIVE),
            jnp.uint8(MUTABLE_BLOCK_PROVENANCE_NONE),
        ),
        geometry=geometry,
        invalid=jnp.zeros_like(available),
        resync_required=jnp.zeros_like(available),
    )


def _require_physical_atlas_alignment(
    atlas: RegionAtlas,
    snapshots: Sequence[NativeRegionSnapshot],
    world_ids: Sequence[int],
) -> None:
    region_mask = np.asarray(jax.device_get(atlas.region_mask))
    expected = np.zeros_like(region_mask)
    expected[: len(snapshots)] = True
    if not np.array_equal(region_mask, expected):
        raise ValueError(
            "physical Region atlas order differs from semantic snapshots"
        )
    actual_world = np.asarray(jax.device_get(atlas.world_id))
    actual_core = np.asarray(jax.device_get(atlas.core_min_chunk_xz))
    actual_known = np.asarray(jax.device_get(atlas.section_known))
    for index, (snapshot, world_id) in enumerate(
        zip(snapshots, world_ids)
    ):
        if actual_world[index] != world_id:
            raise ValueError("physical Region world ID differs")
        if not np.array_equal(
            actual_core[index],
            snapshot.core_min_chunk_xz,
        ) or not np.array_equal(
            actual_known[index],
            snapshot.section_known,
        ):
            raise ValueError("physical Region atlas topology differs")


def _capacity(value: int | None, required: int, label: str) -> int:
    result = required if value is None else _host_int(value, label)
    if not 1 <= result <= (1 << 16):
        raise ValueError(f"{label} capacity must be in [1, 65536]")
    if result < required:
        raise ValueError(f"{label} capacity is smaller than required")
    return result


def _host_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error


__all__ = [
    "REGION_BLOCK_ACTION_DIAGNOSTIC_CLICKED_UNAVAILABLE",
    "REGION_BLOCK_ACTION_DIAGNOSTIC_FILLER_IDENTITY_UNAVAILABLE",
    "REGION_BLOCK_ACTION_DIAGNOSTIC_FILLER_ROOT_UNAVAILABLE",
    "REGION_BLOCK_ACTION_DIAGNOSTIC_SEMANTICS_UNAVAILABLE",
    "REGION_BLOCK_SEMANTIC_ATLAS_SCHEMA",
    "REGION_BLOCK_SEMANTIC_ATLAS_VERSION",
    "RegionBlockActionQueryResult",
    "RegionBlockSemanticAtlas",
    "RegionBlockSemanticSelection",
    "estimated_region_block_semantic_atlas_bytes",
    "lookup_region_block_semantics",
    "query_region_block_action_target",
    "region_block_semantic_atlas_contract",
    "region_block_semantic_atlas_contract_sha256",
    "region_block_semantic_atlas_from_snapshots",
]
