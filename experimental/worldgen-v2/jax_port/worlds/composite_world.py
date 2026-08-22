"""Exact same-reset multi-Region worlds materialized as one JAX world.

The ordinary V2 bundle intentionally treats every Region artifact as an
independent generated world.  Large authored environments need the opposite
contract: several adjacent Region cores captured without another reset must
remain tiles of one world.  This module freezes that relationship, validates
the overlapping physical and identity sidecars, and assigns every tile the
same JAX ``world_id``.

Generation remains native.  Runtime materialization is artifact-only and does
not call Hytale or synthesize terrain in JAX.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import operator
import os
from pathlib import Path, PurePosixPath
import re
import struct
import tempfile
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.region.atlas import region_atlas_from_snapshots
from hytalegym.jax.world.region.block_semantics import (
    RegionBlockSemanticAtlas,
    region_block_semantic_atlas_from_snapshots,
)
from hytalegym.jax.world.region.types import RegionAtlas
from hytalegym.worldgen.region import (
    CAPTURE_CHUNKS_PER_AXIS,
    CHUNK_SIZE,
    CORE_CHUNKS_PER_AXIS,
    MIN_Y,
    WORLD_HEIGHT,
    NativeRegionBlockSemanticSnapshot,
    NativeRegionSnapshot,
)

from ..bundles.bundle import (
    CAPABILITY_EXACT_GEOMETRY,
    CAPABILITY_STABLE_BLOCKS,
    CAPABILITY_STABLE_FLUIDS,
    CAPABILITY_STATIC_SPAWN,
)
from ..bundles.jax_loader import (
    V2FluidSemanticAtlas,
    _FluidSnapshot,
    _fluid_atlas_from_snapshots,
    _load_fluid_snapshot,
    _require_fluid_alignment,
)


COMPOSITE_WORLD_SCHEMA = "hytalerl_worldgen_v2_composite_jax_world_v1"
COMPOSITE_WORLD_VERSION = 1
COMPOSITE_WORLD_AUTHORITY = "exact_same_reset_multi_region_capture"
COMPOSITE_WORLD_RUNTIME_SOURCE = "artifact_only_no_live_generation"
COMPOSITE_CAPTURE_SCHEMA = "hytalerl_worldgen_v2_composite_capture_group_v1"
COMPOSITE_JAX_WORLD_ID = 0

_AVAILABLE_CAPABILITIES = frozenset(
    {
        CAPABILITY_EXACT_GEOMETRY,
        CAPABILITY_STABLE_BLOCKS,
        CAPABILITY_STABLE_FLUIDS,
        CAPABILITY_STATIC_SPAWN,
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "version",
        "authority",
        "runtime_source",
        "environment_id",
        "world_template",
        "worldgen_provider",
        "structure",
        "seed",
        "split",
        "capture_group_sha256",
        "tile_grid",
        "core_origin_chunk_xz",
        "spawn_position",
        "bounds",
        "capabilities",
        "tiles",
        "composite_semantic_sha256",
    }
)
_TILE_FIELDS = frozenset(
    {
        "tile_xz",
        "core_min_chunk_xz",
        "region",
        "block_semantics",
        "fluid_semantics",
    }
)
_FILE_REF_FIELDS = frozenset({"path", "file_sha256", "semantic_sha256"})
_BOUNDS_FIELDS = frozenset({"coordinate_mode", "minimum", "maximum"})
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class V2CompositeTileSource:
    """Files for one tile captured from the still-live composite world."""

    tile_xz: tuple[int, int]
    region_path: Path
    block_semantic_path: Path
    fluid_semantic_path: Path


@dataclass(frozen=True)
class V2CompositeTileArtifact:
    """One immutable tile entry parsed from a composite manifest."""

    tile_xz: tuple[int, int]
    core_min_chunk_xz: tuple[int, int]
    region_path: str
    region_file_sha256: str
    region_semantic_sha256: str
    block_semantic_path: str
    block_semantic_file_sha256: str
    block_semantic_sha256: str
    fluid_semantic_path: str
    fluid_semantic_file_sha256: str
    fluid_semantic_sha256: str


@dataclass(frozen=True)
class V2CompositeWorld:
    """Validated manifest for one generated world made from Region tiles."""

    manifest_path: Path
    artifact_root: Path
    environment_id: str
    structure: str
    seed: int
    split: str
    capture_group_sha256: str
    tile_grid: tuple[int, int]
    core_origin_chunk_xz: tuple[int, int]
    spawn_position: tuple[float, float, float]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    tiles: tuple[V2CompositeTileArtifact, ...]
    semantic_sha256: str

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        artifact_root: str | Path,
        *,
        verify_artifacts: bool = True,
    ) -> "V2CompositeWorld":
        path = Path(manifest_path).resolve()
        root = Path(artifact_root).resolve()
        if not root.is_dir():
            raise ValueError(f"artifact root does not exist: {root}")
        value = _json_object(path)
        _exact_fields(value, _MANIFEST_FIELDS, "composite world")
        if value["schema"] != COMPOSITE_WORLD_SCHEMA:
            raise ValueError("unsupported composite world schema")
        if _exact_int(value["version"], "composite version") != (
            COMPOSITE_WORLD_VERSION
        ):
            raise ValueError("unsupported composite world version")
        if value["authority"] != COMPOSITE_WORLD_AUTHORITY:
            raise ValueError("composite world authority changed")
        if value["runtime_source"] != COMPOSITE_WORLD_RUNTIME_SOURCE:
            raise ValueError("composite world must be artifact-only")
        if value["world_template"] != "hytale_generator":
            raise ValueError("composite world is not the explicit V2 template")
        if value["worldgen_provider"] != "HytaleGenerator":
            raise ValueError("composite world provider changed")

        expected_semantic = _manifest_digest(value)
        semantic = _sha256(
            value["composite_semantic_sha256"], "composite semantic"
        )
        if semantic != expected_semantic:
            raise ValueError("composite world semantic SHA-256 mismatch")

        grid = _positive_pair(value["tile_grid"], "tile_grid")
        origin = _int32_pair(
            value["core_origin_chunk_xz"], "core_origin_chunk_xz"
        )
        spawn = _position(value["spawn_position"], "spawn_position")
        bounds = _mapping(value["bounds"], "bounds")
        _exact_fields(bounds, _BOUNDS_FIELDS, "bounds")
        if bounds["coordinate_mode"] != "native_block":
            raise ValueError("composite bounds coordinate mode changed")
        bounds_min = _position(bounds["minimum"], "bounds minimum")
        bounds_max = _position(bounds["maximum"], "bounds maximum")

        capabilities = value["capabilities"]
        if not isinstance(capabilities, list) or frozenset(capabilities) != (
            _AVAILABLE_CAPABILITIES
        ) or len(capabilities) != len(_AVAILABLE_CAPABILITIES):
            raise ValueError("composite JAX capabilities changed")

        raw_tiles = value["tiles"]
        if not isinstance(raw_tiles, list) or len(raw_tiles) != grid[0] * grid[1]:
            raise ValueError("composite tile count differs from tile_grid")
        tiles = tuple(_parse_tile(raw) for raw in raw_tiles)
        _require_rectangular_tile_layout(tiles, grid, origin)

        _require_environment_bounds(bounds_min, bounds_max, spawn, grid, origin)

        result = cls(
            manifest_path=path,
            artifact_root=root,
            environment_id=_nonempty(value["environment_id"], "environment_id"),
            structure=_nonempty(value["structure"], "structure"),
            seed=_exact_int(value["seed"], "seed"),
            split=_nonempty(value["split"], "split"),
            capture_group_sha256=_sha256(
                value["capture_group_sha256"], "capture group"
            ),
            tile_grid=grid,
            core_origin_chunk_xz=origin,
            spawn_position=spawn,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            tiles=tiles,
            semantic_sha256=semantic,
        )
        if verify_artifacts:
            _load_composite_artifacts(result)
        return result


@dataclass(frozen=True)
class V2CompositeJaxWorld:
    """One exact generated world, regardless of how many Region tiles it uses."""

    physical_atlas: RegionAtlas
    block_semantic_atlas: RegionBlockSemanticAtlas
    fluid_semantic_atlas: V2FluidSemanticAtlas
    environment_world_id: jnp.ndarray
    environment_spawn_position: jnp.ndarray
    world_spawn_position: jnp.ndarray
    environment_bounds_min: jnp.ndarray
    environment_bounds_max: jnp.ndarray
    world_bounds_min: jnp.ndarray
    world_bounds_max: jnp.ndarray
    tiles: tuple[V2CompositeTileArtifact, ...]
    fluid_asset_ids: tuple[tuple[str, ...], ...]
    source_composite_semantic_sha256: str
    available_capabilities: frozenset[str]


def bind_composite_capture_identity(
    snapshot: NativeRegionSnapshot,
    *,
    capture_group_sha256: str,
    tile_xz: Sequence[int],
    tile_grid: Sequence[int],
) -> NativeRegionSnapshot:
    """Bind a Region to a same-reset capture group before publication.

    The group token must be created once immediately after native reset and
    reused for every tile.  This metadata does not change the expanded Region
    semantic digest, so already captured block/fluid sidecars remain aligned.
    """

    if not isinstance(snapshot, NativeRegionSnapshot):
        raise TypeError("snapshot must be a NativeRegionSnapshot")
    group = _sha256(capture_group_sha256, "capture group")
    tile = _nonnegative_pair(tile_xz, "tile_xz")
    grid = _positive_pair(tile_grid, "tile_grid")
    if tile[0] >= grid[0] or tile[1] >= grid[1]:
        raise ValueError("tile_xz lies outside tile_grid")
    additions: dict[str, Any] = {
        "composite_capture_schema": COMPOSITE_CAPTURE_SCHEMA,
        "composite_capture_group_sha256": group,
        "composite_same_reset": True,
        "composite_tile_xz": list(tile),
        "composite_tile_grid": list(grid),
    }
    metadata = dict(snapshot.metadata)
    conflicts = {
        key: (metadata[key], value)
        for key, value in additions.items()
        if key in metadata and metadata[key] != value
    }
    if conflicts:
        raise ValueError(f"Region has conflicting composite identity: {conflicts}")
    metadata.update(additions)
    return NativeRegionSnapshot(
        metadata=metadata,
        core_min_chunk_xz=snapshot.core_min_chunk_xz,
        section_known=snapshot.section_known,
        cell_code=snapshot.cell_code,
        cell_palette=snapshot.cell_palette,
        shape_palette=snapshot.shape_palette,
        filler_root_offset_packed=snapshot.filler_root_offset_packed,
    )


def publish_composite_world(
    destination: str | Path,
    artifact_root: str | Path,
    tile_sources: Sequence[V2CompositeTileSource],
    *,
    environment_id: str,
    split: str = "authoring",
    bounds_min: Sequence[float] | None = None,
    bounds_max: Sequence[float] | None = None,
) -> V2CompositeWorld:
    """Validate same-reset tile artifacts and atomically publish a manifest."""

    root = Path(artifact_root).resolve()
    if not root.is_dir():
        raise ValueError(f"artifact root does not exist: {root}")
    sources = tuple(tile_sources)
    if not sources or any(not isinstance(row, V2CompositeTileSource) for row in sources):
        raise ValueError("tile_sources must contain V2CompositeTileSource values")

    loaded: list[
        tuple[
            V2CompositeTileSource,
            NativeRegionSnapshot,
            NativeRegionBlockSemanticSnapshot,
            _FluidSnapshot,
        ]
    ] = []
    for source in sources:
        region_path = _contained_file(root, source.region_path)
        block_path = _contained_file(root, source.block_semantic_path)
        fluid_path = _contained_file(root, source.fluid_semantic_path)
        physical = NativeRegionSnapshot.load(region_path)
        block = NativeRegionBlockSemanticSnapshot.load(block_path)
        fluid = _load_fluid_snapshot(fluid_path)
        _require_artifact_triplet(physical, block, fluid)
        loaded.append((source, physical, block, fluid))

    metadata = loaded[0][1].metadata
    grid = _positive_pair(metadata.get("composite_tile_grid"), "tile_grid")
    group = _sha256(
        metadata.get("composite_capture_group_sha256"), "capture group"
    )
    structure = _nonempty(metadata.get("worldgen_structure"), "worldgen structure")
    seed = _exact_int(metadata.get("worldgen_seed"), "worldgen seed")
    spawn = _position(metadata.get("spawn_position"), "spawn_position")
    entries: list[dict[str, Any]] = []
    for source, physical, block, fluid in loaded:
        tile = _nonnegative_pair(source.tile_xz, "tile_xz")
        _require_capture_identity(
            physical,
            capture_group_sha256=group,
            tile_xz=tile,
            tile_grid=grid,
            structure=structure,
            seed=seed,
            spawn_position=spawn,
        )
        entries.append(
            {
                "tile_xz": list(tile),
                "core_min_chunk_xz": [int(v) for v in physical.core_min_chunk_xz],
                "region": _file_ref(
                    root,
                    source.region_path,
                    physical.semantic_artifact_digest(),
                ),
                "block_semantics": _file_ref(
                    root,
                    source.block_semantic_path,
                    block.semantic_sha256(),
                ),
                "fluid_semantics": _file_ref(
                    root,
                    source.fluid_semantic_path,
                    fluid.semantic_sha256,
                ),
            }
        )
    entries.sort(key=lambda row: (row["tile_xz"][1], row["tile_xz"][0]))
    origin_entry = next((row for row in entries if row["tile_xz"] == [0, 0]), None)
    if origin_entry is None:
        raise ValueError("composite tiles must include tile [0, 0]")
    origin = tuple(origin_entry["core_min_chunk_xz"])
    parsed_tiles = tuple(_parse_tile(row) for row in entries)
    _require_rectangular_tile_layout(parsed_tiles, grid, origin)
    _require_semantic_overlaps(
        [row[1] for row in loaded],
        [row[2] for row in loaded],
        [row[3] for row in loaded],
    )
    # Validate physical halo agreement before publication, not only when the
    # just-written manifest is loaded.  A rejected composite must never leave
    # an apparently authoritative manifest behind.
    region_atlas_from_snapshots(
        [row[1] for row in loaded],
        world_ids=(COMPOSITE_JAX_WORLD_ID,) * len(loaded),
        validate_overlaps=True,
    )

    default_min = (origin[0] * CHUNK_SIZE, MIN_Y, origin[1] * CHUNK_SIZE)
    default_max = (
        (origin[0] + grid[0] * CORE_CHUNKS_PER_AXIS) * CHUNK_SIZE,
        MIN_Y + WORLD_HEIGHT,
        (origin[1] + grid[1] * CORE_CHUNKS_PER_AXIS) * CHUNK_SIZE,
    )
    selected_min = default_min if bounds_min is None else _position(
        bounds_min, "bounds minimum"
    )
    selected_max = default_max if bounds_max is None else _position(
        bounds_max, "bounds maximum"
    )
    _require_environment_bounds(selected_min, selected_max, spawn, grid, origin)
    value: dict[str, Any] = {
        "schema": COMPOSITE_WORLD_SCHEMA,
        "version": COMPOSITE_WORLD_VERSION,
        "authority": COMPOSITE_WORLD_AUTHORITY,
        "runtime_source": COMPOSITE_WORLD_RUNTIME_SOURCE,
        "environment_id": _nonempty(environment_id, "environment_id"),
        "world_template": "hytale_generator",
        "worldgen_provider": "HytaleGenerator",
        "structure": structure,
        "seed": seed,
        "split": _nonempty(split, "split"),
        "capture_group_sha256": group,
        "tile_grid": list(grid),
        "core_origin_chunk_xz": list(origin),
        "spawn_position": list(spawn),
        "bounds": {
            "coordinate_mode": "native_block",
            "minimum": list(selected_min),
            "maximum": list(selected_max),
        },
        "capabilities": sorted(_AVAILABLE_CAPABILITIES),
        "tiles": entries,
        "composite_semantic_sha256": "",
    }
    value["composite_semantic_sha256"] = _manifest_digest(value)
    output = Path(destination).resolve()
    _atomic_json(output, value)
    return V2CompositeWorld.load(output, root, verify_artifacts=True)


def materialize_composite_jax_world(
    world: V2CompositeWorld,
    *,
    environment_count: int,
    region_capacity: int | None = None,
    cell_palette_capacity: int | None = None,
    shape_palette_capacity: int | None = None,
    block_palette_capacity: int | None = None,
    fluid_palette_capacity: int | None = None,
) -> V2CompositeJaxWorld:
    """Materialize every Region tile under one literal JAX world ID."""

    if not isinstance(world, V2CompositeWorld):
        raise TypeError("world must be a V2CompositeWorld")
    environments = _positive_int(environment_count, "environment_count")
    physical, block, fluid = _load_composite_artifacts(world)
    tile_count = len(physical)
    capacity = tile_count if region_capacity is None else _positive_int(
        region_capacity, "region_capacity"
    )
    if capacity < tile_count:
        raise ValueError("region_capacity is smaller than composite tile count")
    tile_world_ids = (COMPOSITE_JAX_WORLD_ID,) * tile_count
    physical_atlas = region_atlas_from_snapshots(
        physical,
        world_ids=tile_world_ids,
        region_capacity=capacity,
        cell_palette_capacity=cell_palette_capacity,
        shape_palette_capacity=shape_palette_capacity,
        validate_overlaps=True,
    )
    block_atlas = region_block_semantic_atlas_from_snapshots(
        block,
        physical,
        physical_atlas,
        world_ids=tile_world_ids,
        palette_capacity=block_palette_capacity,
    )
    fluid_atlas = _fluid_atlas_from_snapshots(
        fluid,
        region_capacity=capacity,
        palette_capacity=fluid_palette_capacity,
    )
    environment_world_id = jnp.zeros((environments,), dtype=jnp.int32)
    spawn = jnp.asarray([world.spawn_position], dtype=jnp.float32)
    minimum = jnp.asarray([world.bounds_min], dtype=jnp.float32)
    maximum = jnp.asarray([world.bounds_max], dtype=jnp.float32)
    return V2CompositeJaxWorld(
        physical_atlas=physical_atlas,
        block_semantic_atlas=block_atlas,
        fluid_semantic_atlas=fluid_atlas,
        environment_world_id=environment_world_id,
        environment_spawn_position=jnp.repeat(spawn, environments, axis=0),
        world_spawn_position=spawn,
        environment_bounds_min=jnp.repeat(minimum, environments, axis=0),
        environment_bounds_max=jnp.repeat(maximum, environments, axis=0),
        world_bounds_min=minimum,
        world_bounds_max=maximum,
        tiles=world.tiles,
        fluid_asset_ids=tuple(snapshot.palette for snapshot in fluid),
        source_composite_semantic_sha256=world.semantic_sha256,
        available_capabilities=_AVAILABLE_CAPABILITIES,
    )


def _load_composite_artifacts(
    world: V2CompositeWorld,
) -> tuple[
    tuple[NativeRegionSnapshot, ...],
    tuple[NativeRegionBlockSemanticSnapshot, ...],
    tuple[_FluidSnapshot, ...],
]:
    physical: list[NativeRegionSnapshot] = []
    block: list[NativeRegionBlockSemanticSnapshot] = []
    fluid: list[_FluidSnapshot] = []
    for tile in world.tiles:
        region, semantic, fluid_snapshot = load_composite_tile_artifacts(
            world, tile
        )
        physical.append(region)
        block.append(semantic)
        fluid.append(fluid_snapshot)
    _require_semantic_overlaps(physical, block, fluid)
    # Building once here makes physical overlap disagreement fail on manifest
    # load as well as at JAX materialization.
    region_atlas_from_snapshots(
        physical,
        world_ids=(COMPOSITE_JAX_WORLD_ID,) * len(physical),
        validate_overlaps=True,
    )
    return tuple(physical), tuple(block), tuple(fluid)


def load_composite_tile_artifacts(
    world: V2CompositeWorld,
    tile: V2CompositeTileArtifact,
) -> tuple[
    NativeRegionSnapshot,
    NativeRegionBlockSemanticSnapshot,
    _FluidSnapshot,
]:
    """Load and authenticate one composite tile without retaining its peers.

    The exact JAX loader still validates every overlap as a set.  A previewer
    can use this narrower operation to stream large worlds tile by tile while
    retaining all per-file hashes, semantic digests, sidecar alignment, and
    same-reset identity checks from the composite contract.
    """

    if not isinstance(world, V2CompositeWorld):
        raise TypeError("world must be a V2CompositeWorld")
    if not isinstance(tile, V2CompositeTileArtifact) or tile not in world.tiles:
        raise ValueError("tile is not part of the composite world")
    region_path = _verified_path(
        world.artifact_root, tile.region_path, tile.region_file_sha256
    )
    block_path = _verified_path(
        world.artifact_root,
        tile.block_semantic_path,
        tile.block_semantic_file_sha256,
    )
    fluid_path = _verified_path(
        world.artifact_root,
        tile.fluid_semantic_path,
        tile.fluid_semantic_file_sha256,
    )
    region = NativeRegionSnapshot.load(region_path)
    semantic = NativeRegionBlockSemanticSnapshot.load(block_path)
    fluid_snapshot = _load_fluid_snapshot(fluid_path)
    if region.semantic_artifact_digest() != tile.region_semantic_sha256:
        raise ValueError("composite Region semantic differs from manifest")
    if semantic.semantic_sha256() != tile.block_semantic_sha256:
        raise ValueError("composite block semantic differs from manifest")
    if fluid_snapshot.semantic_sha256 != tile.fluid_semantic_sha256:
        raise ValueError("composite fluid semantic differs from manifest")
    if tuple(int(v) for v in region.core_min_chunk_xz) != (
        tile.core_min_chunk_xz
    ):
        raise ValueError("composite Region core differs from manifest")
    _require_artifact_triplet(region, semantic, fluid_snapshot)
    _require_capture_identity(
        region,
        capture_group_sha256=world.capture_group_sha256,
        tile_xz=tile.tile_xz,
        tile_grid=world.tile_grid,
        structure=world.structure,
        seed=world.seed,
        spawn_position=world.spawn_position,
    )
    return region, semantic, fluid_snapshot


def _require_artifact_triplet(
    physical: NativeRegionSnapshot,
    block: NativeRegionBlockSemanticSnapshot,
    fluid: _FluidSnapshot,
) -> None:
    digest = physical.semantic_artifact_digest()
    if block.source_region_semantic_sha256 != digest:
        raise ValueError("block semantics name another composite Region")
    if not np.array_equal(block.core_min_chunk_xz, physical.core_min_chunk_xz):
        raise ValueError("block semantics and physical Region cores differ")
    if not np.array_equal(block.section_known, physical.section_known):
        raise ValueError("block semantics and physical section coverage differ")
    _require_fluid_alignment(fluid, physical)


def _require_capture_identity(
    snapshot: NativeRegionSnapshot,
    *,
    capture_group_sha256: str,
    tile_xz: Sequence[int],
    tile_grid: Sequence[int],
    structure: str,
    seed: int,
    spawn_position: Sequence[float],
) -> None:
    metadata = snapshot.metadata
    expected = {
        "composite_capture_schema": COMPOSITE_CAPTURE_SCHEMA,
        "composite_capture_group_sha256": capture_group_sha256,
        "composite_same_reset": True,
        "composite_tile_xz": list(tile_xz),
        "composite_tile_grid": list(tile_grid),
        "world_template": "hytale_generator",
        "worldgen_provider": "HytaleGenerator",
        "worldgen_structure": structure,
        "worldgen_seed": seed,
        "spawn_position": list(spawn_position),
    }
    mismatches = {
        key: (value, metadata.get(key))
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"composite same-reset identity differs: {mismatches}")


def _require_semantic_overlaps(
    physical: Sequence[NativeRegionSnapshot],
    blocks: Sequence[NativeRegionBlockSemanticSnapshot],
    fluids: Sequence[_FluidSnapshot],
) -> None:
    if not (len(physical) == len(blocks) == len(fluids)):
        raise ValueError("composite artifact columns are unpaired")
    block_keys = [_block_entry_digests(snapshot) for snapshot in blocks]
    fluid_keys = [_fluid_entry_digests(snapshot) for snapshot in fluids]
    for left in range(len(physical)):
        for right in range(left + 1, len(physical)):
            for left_slot, right_slot in _overlap_slots(
                physical[left], physical[right]
            ):
                known = blocks[left].section_known[left_slot] & (
                    blocks[right].section_known[right_slot]
                )
                for section in np.flatnonzero(known):
                    section_y = int(section)
                    left_block = block_keys[left][
                        blocks[left].cell_code[left_slot, section_y]
                    ]
                    right_block = block_keys[right][
                        blocks[right].cell_code[right_slot, section_y]
                    ]
                    if not np.array_equal(left_block, right_block):
                        raise ValueError(
                            "overlapping composite block-semantic sections disagree"
                        )
                    left_fluid = fluid_keys[left][
                        fluids[left].cell_code[left_slot, section_y]
                    ]
                    right_fluid = fluid_keys[right][
                        fluids[right].cell_code[right_slot, section_y]
                    ]
                    if not np.array_equal(left_fluid, right_fluid):
                        raise ValueError(
                            "overlapping composite fluid-semantic sections disagree"
                        )


def _overlap_slots(
    left: NativeRegionSnapshot,
    right: NativeRegionSnapshot,
) -> list[tuple[int, int]]:
    left_min = left.capture_min_chunk_xz
    right_min = right.capture_min_chunk_xz
    minimum = np.maximum(left_min, right_min)
    maximum = np.minimum(
        left_min + CAPTURE_CHUNKS_PER_AXIS,
        right_min + CAPTURE_CHUNKS_PER_AXIS,
    )
    slots: list[tuple[int, int]] = []
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
            slots.append((left_slot, right_slot))
    return slots


def _block_entry_digests(
    snapshot: NativeRegionBlockSemanticSnapshot,
) -> np.ndarray:
    result = np.empty((len(snapshot.palette), 32), dtype=np.uint8)
    for index, entry in enumerate(snapshot.palette):
        digest = hashlib.sha256()
        digest.update(np.asarray(entry.semantic_key, dtype="<u4").tobytes())
        digest.update(np.asarray(entry.asset_key, dtype="<u4").tobytes())
        digest.update(struct.pack(
            "<?HBhi",
            entry.valid,
            entry.affordance_tags,
            entry.gather_type_index,
            entry.required_tool_quality,
            entry.rotation_index,
        ))
        result[index] = np.frombuffer(digest.digest(), dtype=np.uint8)
    return result


def _fluid_entry_digests(snapshot: _FluidSnapshot) -> np.ndarray:
    return np.stack(
        [
            np.frombuffer(
                hashlib.sha256(asset_id.encode("utf-8")).digest(),
                dtype=np.uint8,
            )
            for asset_id in snapshot.palette
        ]
    )


def _parse_tile(value: Any) -> V2CompositeTileArtifact:
    raw = _mapping(value, "composite tile")
    _exact_fields(raw, _TILE_FIELDS, "composite tile")
    tile = _nonnegative_pair(raw["tile_xz"], "tile_xz")
    core = _int32_pair(raw["core_min_chunk_xz"], "core_min_chunk_xz")
    region = _parse_file_ref(raw["region"], "Region")
    block = _parse_file_ref(raw["block_semantics"], "block semantics")
    fluid = _parse_file_ref(raw["fluid_semantics"], "fluid semantics")
    return V2CompositeTileArtifact(
        tile_xz=tile,
        core_min_chunk_xz=core,
        region_path=region[0],
        region_file_sha256=region[1],
        region_semantic_sha256=region[2],
        block_semantic_path=block[0],
        block_semantic_file_sha256=block[1],
        block_semantic_sha256=block[2],
        fluid_semantic_path=fluid[0],
        fluid_semantic_file_sha256=fluid[1],
        fluid_semantic_sha256=fluid[2],
    )


def _require_rectangular_tile_layout(
    tiles: Sequence[V2CompositeTileArtifact],
    grid: tuple[int, int],
    origin: tuple[int, int],
) -> None:
    expected = {(x, z) for z in range(grid[1]) for x in range(grid[0])}
    actual = {tile.tile_xz for tile in tiles}
    if actual != expected or len(actual) != len(tiles):
        raise ValueError("composite tiles do not form one complete rectangle")
    for tile in tiles:
        expected_core = (
            origin[0] + tile.tile_xz[0] * CORE_CHUNKS_PER_AXIS,
            origin[1] + tile.tile_xz[1] * CORE_CHUNKS_PER_AXIS,
        )
        if tile.core_min_chunk_xz != expected_core:
            raise ValueError("composite Region cores are not contiguous")


def _require_environment_bounds(
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    spawn: tuple[float, float, float],
    grid: tuple[int, int],
    origin: tuple[int, int],
) -> None:
    if any(a >= b for a, b in zip(bounds_min, bounds_max, strict=True)):
        raise ValueError("composite bounds must have positive volume")
    core_min = (origin[0] * CHUNK_SIZE, MIN_Y, origin[1] * CHUNK_SIZE)
    core_max = (
        (origin[0] + grid[0] * CORE_CHUNKS_PER_AXIS) * CHUNK_SIZE,
        MIN_Y + WORLD_HEIGHT,
        (origin[1] + grid[1] * CORE_CHUNKS_PER_AXIS) * CHUNK_SIZE,
    )
    if any(
        bounds_min[index] < core_min[index]
        or bounds_max[index] > core_max[index]
        for index in range(3)
    ):
        raise ValueError("composite bounds exceed the union of Region cores")
    if any(
        spawn[index] < bounds_min[index]
        or spawn[index] > bounds_max[index]
        for index in range(3)
    ):
        raise ValueError("composite spawn lies outside environment bounds")


def _file_ref(root: Path, path: Path, semantic: str) -> dict[str, str]:
    contained = _contained_file(root, path)
    return {
        "path": contained.relative_to(root).as_posix(),
        "file_sha256": _file_sha256(contained),
        "semantic_sha256": _sha256(semantic, "artifact semantic"),
    }


def _parse_file_ref(value: Any, label: str) -> tuple[str, str, str]:
    raw = _mapping(value, f"{label} file reference")
    _exact_fields(raw, _FILE_REF_FIELDS, f"{label} file reference")
    return (
        _relative_path(raw["path"], f"{label} path"),
        _sha256(raw["file_sha256"], f"{label} file"),
        _sha256(raw["semantic_sha256"], f"{label} semantic"),
    )


def _contained_file(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"artifact is unavailable below root: {path}")
    return resolved


def _verified_path(root: Path, relative: str, expected: str) -> Path:
    path = (root / PurePosixPath(relative)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"composite artifact is unavailable: {relative}")
    if _file_sha256(path) != expected:
        raise ValueError(f"composite artifact file SHA-256 differs: {relative}")
    return path


def _relative_path(value: Any, label: str) -> str:
    text = _nonempty(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return text


def _manifest_digest(value: Mapping[str, Any]) -> str:
    stable = dict(value)
    stable["composite_semantic_sha256"] = ""
    return hashlib.sha256(_canonical_json(stable)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            output.write(_canonical_json(value) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _json_object(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"composite manifest does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("composite manifest is not valid UTF-8 JSON") from error
    return _mapping(value, "composite manifest")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], fields: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    if actual != fields:
        raise ValueError(
            f"{label} fields differ: missing={sorted(fields - actual)}, "
            f"extra={sorted(actual - fields)}"
        )


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty canonical string")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error
    return int(result)


def _positive_int(value: Any, label: str) -> int:
    result = _exact_int(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _pair(value: Any, label: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two integers")
    return (_exact_int(value[0], label), _exact_int(value[1], label))


def _positive_pair(value: Any, label: str) -> tuple[int, int]:
    result = _pair(value, label)
    if any(item <= 0 for item in result):
        raise ValueError(f"{label} entries must be positive")
    return result


def _nonnegative_pair(value: Any, label: str) -> tuple[int, int]:
    result = _pair(value, label)
    if any(item < 0 for item in result):
        raise ValueError(f"{label} entries must be nonnegative")
    return result


def _int32_pair(value: Any, label: str) -> tuple[int, int]:
    result = _pair(value, label)
    if any(item < np.iinfo(np.int32).min or item > np.iinfo(np.int32).max
           for item in result):
        raise ValueError(f"{label} entries must fit int32")
    return result


def _position(value: Any, label: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three finite numbers")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise TypeError(f"{label} entries must be numbers")
        try:
            number = float(item)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{label} entries must be numbers") from error
        if not math.isfinite(number):
            raise ValueError(f"{label} entries must be finite")
        result.append(number)
    return tuple(result)  # type: ignore[return-value]


__all__ = [
    "COMPOSITE_CAPTURE_SCHEMA",
    "COMPOSITE_JAX_WORLD_ID",
    "COMPOSITE_WORLD_AUTHORITY",
    "COMPOSITE_WORLD_RUNTIME_SOURCE",
    "COMPOSITE_WORLD_SCHEMA",
    "COMPOSITE_WORLD_VERSION",
    "V2CompositeJaxWorld",
    "V2CompositeTileArtifact",
    "V2CompositeTileSource",
    "V2CompositeWorld",
    "bind_composite_capture_identity",
    "materialize_composite_jax_world",
    "publish_composite_world",
]
