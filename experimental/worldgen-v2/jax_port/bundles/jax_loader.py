"""Selected-artifact materialization for exact WorldGen V2 JAX worlds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import operator
from pathlib import Path, PurePosixPath
import struct
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.geometry.contract import FLAG_FLUID
from hytalegym.jax.world.region.atlas import (
    lookup_region_blocks,
    region_atlas_from_snapshots,
)
from hytalegym.jax.world.region.block_semantics import (
    RegionBlockSemanticAtlas,
    region_block_semantic_atlas_from_snapshots,
)
from hytalegym.jax.world.region.traversal import (
    region_traversal_atlas_from_graphs,
)
from hytalegym.jax.world.region.types import RegionAtlas, RegionTraversalAtlas
from hytalegym.worldgen.block_actions import BLOCK_SEMANTIC_KEY_WORDS
from hytalegym.worldgen.region import (
    CAPTURE_CHUNK_COUNT,
    CAPTURE_CHUNKS_PER_AXIS,
    CHUNK_SIZE,
    HEIGHT_SECTIONS,
    MIN_Y,
    SECTION_VOLUME,
    NativeRegionBlockSemanticSnapshot,
    NativeRegionSnapshot,
)

from .bundle import (
    BUNDLE_ASSIGNMENT_METHOD,
    CAPABILITY_EXACT_GEOMETRY,
    CAPABILITY_NATIVE_TRAVERSAL,
    CAPABILITY_STABLE_BLOCKS,
    CAPABILITY_STABLE_FLUIDS,
    CAPABILITY_STATIC_SPAWN,
    CAPABILITY_STRUCTURE_INSTANCES,
    V2JaxArtifact,
    V2JaxBundle,
)
from ..structures.structure_jax import (
    V2StructureInstanceAtlas,
    structure_instance_atlas_from_snapshots,
)
from ..structures.structure_pack import V2StructurePack
from .traversal_pack import V2TraversalPack


FLUID_ATLAS_SCHEMA = "hytalerl_worldgen_v2_fluid_semantic_atlas_v1"
FLUID_ATLAS_VERSION = 1
FLUID_KEY_IDENTITY = "sha256_words_of_utf8_fluid_getId"
_FLUID_SNAPSHOT_SCHEMA = "hytalerl_region_fluid_semantic_snapshot_v1"
_FLUID_SNAPSHOT_VERSION = 1
_FLUID_SNAPSHOT_KEYS = {
    "metadata",
    "core_min_chunk_xz",
    "section_known",
    "cell_code",
    "palette_asset_id",
}
_KEY_FORMAT = f">{BLOCK_SEMANTIC_KEY_WORDS}I"


class V2FluidSemanticAtlas(NamedTuple):
    """Stable native fluid identity aligned to exact physical Regions."""

    region_mask: jax.Array
    section_known: jax.Array
    cell_code: jax.Array
    palette_size: jax.Array
    asset_key: jax.Array


class V2FluidSemanticSelection(NamedTuple):
    """Per-cell stable fluid identity; zero key means no fluid."""

    available: jax.Array
    fluid_present: jax.Array
    asset_key: jax.Array


@dataclass(frozen=True)
class V2JaxWorldSet:
    """Selected exact worlds ready to close over in a JAX runtime."""

    physical_atlas: RegionAtlas
    block_semantic_atlas: RegionBlockSemanticAtlas
    fluid_semantic_atlas: V2FluidSemanticAtlas
    traversal_atlas: RegionTraversalAtlas | None
    structure_instance_atlas: V2StructureInstanceAtlas | None
    environment_world_id: jax.Array
    environment_spawn_position: jax.Array
    world_spawn_position: jax.Array
    artifacts: tuple[V2JaxArtifact, ...]
    fluid_asset_ids: tuple[tuple[str, ...], ...]
    structure_asset_ids: tuple[tuple[str, ...], ...]
    structure_marker_asset_ids: tuple[tuple[str, ...], ...]
    source_bundle_semantic_sha256: str
    available_capabilities: frozenset[str]
    assignment_key: int


@dataclass(frozen=True)
class _FluidSnapshot:
    core_min_chunk_xz: np.ndarray
    section_known: np.ndarray
    cell_code: np.ndarray
    palette: tuple[str, ...]
    source_region_semantic_sha256: str
    evidence_bridge_sha256: str
    semantic_sha256: str


def load_selected_jax_worlds(
    bundle: V2JaxBundle,
    structure: str,
    *,
    split: str | None = None,
    seeds: Sequence[int] | None = None,
    count: int | None = None,
    selection_key: int = 0,
    environment_count: int,
    assignment_key: int = 0,
    region_capacity: int | None = None,
    cell_palette_capacity: int | None = None,
    shape_palette_capacity: int | None = None,
    block_palette_capacity: int | None = None,
    fluid_palette_capacity: int | None = None,
    traversal_pack: V2TraversalPack | None = None,
    traversal_node_capacity: int | None = None,
    traversal_column_node_capacity: int | None = None,
    structure_pack: V2StructurePack | None = None,
    structure_instance_capacity: int | None = None,
    required_capabilities: Sequence[str] = (
        CAPABILITY_EXACT_GEOMETRY,
        CAPABILITY_STABLE_BLOCKS,
        CAPABILITY_STABLE_FLUIDS,
        CAPABILITY_STATIC_SPAWN,
    ),
) -> V2JaxWorldSet:
    """Select by the global seed split and materialize only those worlds."""

    selected = bundle.select(
        structure,
        split=split,
        seeds=seeds,
        count=count,
        selection_key=selection_key,
    )
    return materialize_jax_worlds(
        bundle,
        selected,
        environment_count=environment_count,
        assignment_key=assignment_key,
        region_capacity=region_capacity,
        cell_palette_capacity=cell_palette_capacity,
        shape_palette_capacity=shape_palette_capacity,
        block_palette_capacity=block_palette_capacity,
        fluid_palette_capacity=fluid_palette_capacity,
        traversal_pack=traversal_pack,
        traversal_node_capacity=traversal_node_capacity,
        traversal_column_node_capacity=traversal_column_node_capacity,
        structure_pack=structure_pack,
        structure_instance_capacity=structure_instance_capacity,
        required_capabilities=required_capabilities,
    )


def materialize_jax_worlds(
    bundle: V2JaxBundle,
    artifacts: Sequence[V2JaxArtifact],
    *,
    environment_count: int,
    assignment_key: int = 0,
    region_capacity: int | None = None,
    cell_palette_capacity: int | None = None,
    shape_palette_capacity: int | None = None,
    block_palette_capacity: int | None = None,
    fluid_palette_capacity: int | None = None,
    traversal_pack: V2TraversalPack | None = None,
    traversal_node_capacity: int | None = None,
    traversal_column_node_capacity: int | None = None,
    structure_pack: V2StructurePack | None = None,
    structure_instance_capacity: int | None = None,
    required_capabilities: Sequence[str] = (
        CAPABILITY_EXACT_GEOMETRY,
        CAPABILITY_STABLE_BLOCKS,
        CAPABILITY_STABLE_FLUIDS,
        CAPABILITY_STATIC_SPAWN,
    ),
) -> V2JaxWorldSet:
    """Hash, load, and compile only explicitly selected exact artifacts."""

    if not isinstance(bundle, V2JaxBundle):
        raise TypeError("bundle must be a V2JaxBundle")
    if isinstance(required_capabilities, (str, bytes)):
        raise TypeError("required_capabilities must be a sequence")
    requirements = frozenset(required_capabilities)
    sidecar_capabilities = {
        CAPABILITY_NATIVE_TRAVERSAL,
        CAPABILITY_STRUCTURE_INSTANCES,
    }
    bundle.require_capabilities(tuple(requirements - sidecar_capabilities))
    selected = tuple(artifacts)
    if not selected or any(not isinstance(row, V2JaxArtifact) for row in selected):
        raise ValueError("artifacts must contain V2JaxArtifact values")
    known = {
        (row.structure, row.seed, row.region_semantic_sha256): row
        for row in bundle.artifacts
    }
    if any(
        known.get((row.structure, row.seed, row.region_semantic_sha256)) != row
        for row in selected
    ):
        raise ValueError("selected artifact does not belong to the bundle")
    if len({row.region_semantic_sha256 for row in selected}) != len(selected):
        raise ValueError("selected Region artifacts must be unique")
    traversal_graphs = None
    structure_snapshots = None
    external_capabilities: set[str] = set()
    if traversal_pack is not None:
        if not isinstance(traversal_pack, V2TraversalPack):
            raise TypeError("traversal_pack must be a V2TraversalPack")
        if traversal_pack.bundle.semantic_sha256 != bundle.semantic_sha256:
            raise ValueError("traversal pack belongs to another V2 bundle")
        traversal_graphs = traversal_pack.load_for_artifacts(selected)
        external_capabilities.add(CAPABILITY_NATIVE_TRAVERSAL)
    if structure_pack is not None:
        if not isinstance(structure_pack, V2StructurePack):
            raise TypeError("structure_pack must be a V2StructurePack")
        if structure_pack.bundle.semantic_sha256 != bundle.semantic_sha256:
            raise ValueError("structure pack belongs to another V2 bundle")
        structure_snapshots = structure_pack.load_for_artifacts(selected)
        external_capabilities.add(CAPABILITY_STRUCTURE_INSTANCES)
    if (
        CAPABILITY_NATIVE_TRAVERSAL in requirements
        and traversal_graphs is None
    ):
        raise ValueError(
            "required V2 JAX capability unavailable: native_traversal "
            "requires a traversal pack covering every selected world"
        )
    if (
        CAPABILITY_STRUCTURE_INSTANCES in requirements
        and structure_snapshots is None
    ):
        raise ValueError(
            "required V2 JAX capability unavailable: "
            "structure_instance_identity requires a marker-backed structure "
            "pack covering every selected world"
        )
    environments = _positive_int(environment_count, "environment_count")
    assignment = _nonnegative_int(assignment_key, "assignment_key")
    capacity = len(selected) if region_capacity is None else _positive_int(
        region_capacity, "region_capacity"
    )
    if capacity < len(selected):
        raise ValueError("region_capacity is smaller than selected worlds")

    physical_snapshots: list[NativeRegionSnapshot] = []
    block_snapshots: list[NativeRegionBlockSemanticSnapshot] = []
    fluid_snapshots: list[_FluidSnapshot] = []
    for row in selected:
        region_path = _verified_path(
            bundle,
            row.region_path,
            row.region_file_sha256,
        )
        block_path = _verified_path(
            bundle,
            row.block_semantic_path,
            row.block_semantic_file_sha256,
        )
        fluid_path = _verified_path(
            bundle,
            row.fluid_semantic_path,
            row.fluid_semantic_file_sha256,
        )
        physical = NativeRegionSnapshot.load(region_path)
        if physical.semantic_artifact_digest() != row.region_semantic_sha256:
            raise ValueError(
                f"selected Region semantic differs: {row.structure}/{row.seed}"
            )
        if tuple(int(value) for value in physical.core_min_chunk_xz) != (
            row.core_min_chunk_xz
        ):
            raise ValueError(
                f"selected Region core differs: {row.structure}/{row.seed}"
            )
        block = NativeRegionBlockSemanticSnapshot.load(block_path)
        if block.semantic_sha256() != row.block_semantic_sha256:
            raise ValueError(
                f"selected block semantics differ: {row.structure}/{row.seed}"
            )
        if block.source_region_semantic_sha256 != row.region_semantic_sha256:
            raise ValueError(
                f"block semantics name another Region: {row.structure}/{row.seed}"
            )
        fluid = _load_fluid_snapshot(fluid_path)
        if fluid.semantic_sha256 != row.fluid_semantic_sha256:
            raise ValueError(
                f"selected fluid semantics differ: {row.structure}/{row.seed}"
            )
        _require_fluid_alignment(fluid, physical)
        physical_snapshots.append(physical)
        block_snapshots.append(block)
        fluid_snapshots.append(fluid)

    world_ids = tuple(range(len(selected)))
    physical_atlas = region_atlas_from_snapshots(
        physical_snapshots,
        world_ids=world_ids,
        region_capacity=capacity,
        cell_palette_capacity=cell_palette_capacity,
        shape_palette_capacity=shape_palette_capacity,
        validate_overlaps=False,
    )
    block_atlas = region_block_semantic_atlas_from_snapshots(
        block_snapshots,
        physical_snapshots,
        physical_atlas,
        world_ids=world_ids,
        palette_capacity=block_palette_capacity,
    )
    fluid_atlas = _fluid_atlas_from_snapshots(
        fluid_snapshots,
        region_capacity=capacity,
        palette_capacity=fluid_palette_capacity,
    )
    traversal_atlas = None
    if traversal_graphs is not None:
        traversal_atlas = region_traversal_atlas_from_graphs(
            physical_atlas,
            tuple(traversal_graphs)
            + (None,) * (capacity - len(traversal_graphs)),
            node_capacity=traversal_node_capacity,
            column_node_capacity=traversal_column_node_capacity,
        )
    structure_atlas = None
    structure_asset_ids: tuple[tuple[str, ...], ...] = ()
    structure_marker_asset_ids: tuple[tuple[str, ...], ...] = ()
    if structure_snapshots is not None:
        structure_atlas = structure_instance_atlas_from_snapshots(
            physical_atlas,
            structure_snapshots,
            selected,
            instance_capacity=structure_instance_capacity,
        )
        structure_asset_ids = tuple(
            tuple(instance.structure_asset_id for instance in snapshot.instances)
            for snapshot in structure_snapshots
        )
        structure_marker_asset_ids = tuple(
            tuple(instance.marker_asset_id for instance in snapshot.instances)
            for snapshot in structure_snapshots
        )
    ranked_world_ids = sorted(
        world_ids,
        key=lambda world_id: (
            hashlib.sha256(
                (
                    f"{BUNDLE_ASSIGNMENT_METHOD}\0{assignment}\0"
                    f"{selected[world_id].region_semantic_sha256}"
                ).encode()
            ).digest(),
            selected[world_id].region_semantic_sha256,
        ),
    )
    environment_world_id = np.asarray(
        [ranked_world_ids[index % len(ranked_world_ids)] for index in range(environments)],
        dtype=np.int32,
    )
    world_spawn = np.asarray(
        [row.spawn_position for row in selected],
        dtype=np.float32,
    )
    return V2JaxWorldSet(
        physical_atlas=physical_atlas,
        block_semantic_atlas=block_atlas,
        fluid_semantic_atlas=fluid_atlas,
        traversal_atlas=traversal_atlas,
        structure_instance_atlas=structure_atlas,
        environment_world_id=jnp.asarray(environment_world_id),
        environment_spawn_position=jnp.asarray(world_spawn[environment_world_id]),
        world_spawn_position=jnp.asarray(world_spawn),
        artifacts=selected,
        fluid_asset_ids=tuple(snapshot.palette for snapshot in fluid_snapshots),
        structure_asset_ids=structure_asset_ids,
        structure_marker_asset_ids=structure_marker_asset_ids,
        source_bundle_semantic_sha256=bundle.semantic_sha256,
        available_capabilities=frozenset(
            set(bundle.available_capabilities) | external_capabilities
        ),
        assignment_key=assignment,
    )


def lookup_v2_fluid_identity(
    fluid_atlas: V2FluidSemanticAtlas,
    physical_atlas: RegionAtlas,
    block_positions: jax.Array,
    environment_world_id: jax.Array,
    *,
    require_core: bool = True,
) -> V2FluidSemanticSelection:
    """Select stable fluid IDs through the authoritative physical lookup."""

    blocks = jnp.asarray(block_positions, dtype=jnp.int32)
    if blocks.ndim != 3 or blocks.shape[-1] != 3:
        raise ValueError("block_positions must have shape [batch, queries, 3]")
    world_ids = jnp.asarray(environment_world_id, dtype=jnp.int32)
    if world_ids.shape != (blocks.shape[0],):
        raise ValueError("environment_world_id must have shape [batch]")
    physical = lookup_region_blocks(
        physical_atlas,
        blocks,
        world_ids,
        require_core=require_core,
    )
    safe_region = jnp.maximum(physical.region_index, 0)
    core_min = physical_atlas.core_min_chunk_xz[safe_region]
    chunk_xz = jnp.floor_divide(blocks[..., (0, 2)], CHUNK_SIZE)
    capture_relative = chunk_xz - core_min + 1
    safe_capture = jnp.clip(capture_relative, 0, CAPTURE_CHUNKS_PER_AXIS - 1)
    chunk_slot = (
        safe_capture[..., 0] * CAPTURE_CHUNKS_PER_AXIS + safe_capture[..., 1]
    )
    relative_y = blocks[..., 1] - MIN_Y
    section_y = jnp.floor_divide(relative_y, CHUNK_SIZE)
    safe_section = jnp.clip(section_y, 0, HEIGHT_SECTIONS - 1)
    local_x = blocks[..., 0] & (CHUNK_SIZE - 1)
    local_y = relative_y & (CHUNK_SIZE - 1)
    local_z = blocks[..., 2] & (CHUNK_SIZE - 1)
    section_index = (
        local_y * (CHUNK_SIZE * CHUNK_SIZE) + local_z * CHUNK_SIZE + local_x
    )
    code = fluid_atlas.cell_code[
        safe_region,
        chunk_slot,
        safe_section,
        section_index,
    ].astype(jnp.int32)
    valid_code = code < fluid_atlas.palette_size[safe_region]
    safe_code = jnp.where(valid_code, code, 0)
    available = (
        physical.available
        & fluid_atlas.region_mask[safe_region]
        & fluid_atlas.section_known[safe_region, chunk_slot, safe_section]
        & valid_code
    )
    key = fluid_atlas.asset_key[safe_region, safe_code]
    return V2FluidSemanticSelection(
        available=available,
        fluid_present=available & (code != 0),
        asset_key=jnp.where(available[..., None], key, 0),
    )


def _fluid_atlas_from_snapshots(
    snapshots: Sequence[_FluidSnapshot],
    *,
    region_capacity: int,
    palette_capacity: int | None,
) -> V2FluidSemanticAtlas:
    capacity = max(len(snapshot.palette) for snapshot in snapshots)
    if palette_capacity is not None:
        requested = _positive_int(palette_capacity, "fluid_palette_capacity")
        if requested < capacity:
            raise ValueError("fluid palette capacity is too small")
        capacity = requested
    region_mask = np.zeros(region_capacity, dtype=np.bool_)
    known = np.zeros(
        (region_capacity, CAPTURE_CHUNK_COUNT, HEIGHT_SECTIONS), dtype=np.bool_
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
    asset_key = np.zeros(
        (region_capacity, capacity, BLOCK_SEMANTIC_KEY_WORDS), dtype=np.uint32
    )
    for index, snapshot in enumerate(snapshots):
        region_mask[index] = True
        known[index] = snapshot.section_known
        code[index] = snapshot.cell_code
        palette_size[index] = len(snapshot.palette)
        for slot, asset_id in enumerate(snapshot.palette[1:], start=1):
            asset_key[index, slot] = np.asarray(
                _fluid_asset_key(asset_id), dtype=np.uint32
            )
    return V2FluidSemanticAtlas(
        region_mask=jnp.asarray(region_mask),
        section_known=jnp.asarray(known),
        cell_code=jnp.asarray(code),
        palette_size=jnp.asarray(palette_size),
        asset_key=jnp.asarray(asset_key),
    )


def _load_fluid_snapshot(path: Path) -> _FluidSnapshot:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != _FLUID_SNAPSHOT_KEYS:
            raise ValueError("fluid semantic snapshot fields differ")
        metadata = json.loads(str(archive["metadata"].item()))
        if not isinstance(metadata, dict) or metadata.get("schema") != (
            _FLUID_SNAPSHOT_SCHEMA
        ) or metadata.get("version") != _FLUID_SNAPSHOT_VERSION:
            raise ValueError("unsupported fluid semantic snapshot")
        if metadata.get("identity") != "stable_hytale_fluid_getId_string":
            raise ValueError("fluid identity contract changed")
        core = np.asarray(archive["core_min_chunk_xz"])
        known = np.asarray(archive["section_known"])
        code = np.asarray(archive["cell_code"])
        palette_array = np.asarray(archive["palette_asset_id"])
    if core.dtype != np.int32 or core.shape != (2,):
        raise ValueError("fluid snapshot core differs")
    section_shape = (CAPTURE_CHUNK_COUNT, HEIGHT_SECTIONS)
    if known.dtype != np.bool_ or known.shape != section_shape or not np.all(known):
        raise ValueError("fluid snapshot section coverage differs")
    if code.dtype != np.uint16 or code.shape != section_shape + (SECTION_VOLUME,):
        raise ValueError("fluid snapshot cell codes differ")
    if palette_array.ndim != 1 or palette_array.dtype.kind != "U":
        raise ValueError("fluid snapshot palette differs")
    palette = tuple(str(value) for value in palette_array.tolist())
    if not palette or palette[0] != "" or tuple(sorted(set(palette[1:]))) != (
        palette[1:]
    ):
        raise ValueError("fluid snapshot palette is not canonical")
    if np.any(code.astype(np.uint32) >= len(palette)):
        raise ValueError("fluid snapshot code exceeds its palette")
    source_region = _sha256(
        metadata.get("source_region_semantic_sha256"), "fluid source Region"
    )
    bridge = _sha256(metadata.get("evidence_bridge_sha256"), "fluid bridge")
    semantic = _fluid_semantic_sha256(
        core,
        known,
        code,
        palette,
        source_region,
    )
    if semantic != _sha256(metadata.get("semantic_sha256"), "fluid semantic"):
        raise ValueError("fluid snapshot semantic SHA-256 mismatch")
    return _FluidSnapshot(core, known, code, palette, source_region, bridge, semantic)


def _require_fluid_alignment(
    fluid: _FluidSnapshot,
    physical: NativeRegionSnapshot,
) -> None:
    if fluid.source_region_semantic_sha256 != physical.semantic_artifact_digest():
        raise ValueError("fluid semantics name another physical Region")
    if not np.array_equal(fluid.core_min_chunk_xz, physical.core_min_chunk_xz):
        raise ValueError("fluid and physical Region cores differ")
    if not np.array_equal(fluid.section_known, physical.section_known):
        raise ValueError("fluid and physical section coverage differs")
    flags = physical.cell_palette.flags[physical.cell_code]
    if not np.array_equal(fluid.cell_code != 0, (flags & np.uint16(FLAG_FLUID)) != 0):
        raise ValueError("stable fluid identity differs from physical FLAG_FLUID")


def _fluid_semantic_sha256(
    core: np.ndarray,
    known: np.ndarray,
    code: np.ndarray,
    palette: tuple[str, ...],
    source_region: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(_FLUID_SNAPSHOT_SCHEMA.encode("ascii"))
    digest.update(struct.pack("<I", _FLUID_SNAPSHOT_VERSION))
    digest.update(source_region.encode("ascii"))
    digest.update(core.astype("<i4").tobytes())
    digest.update(known.tobytes(order="C"))
    digest.update(struct.pack("<I", len(palette)))
    for asset_id in palette:
        encoded = asset_id.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded)))
        digest.update(encoded)
    digest.update(code.astype("<u2", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _fluid_asset_key(asset_id: str) -> tuple[int, ...]:
    if not isinstance(asset_id, str) or not asset_id or asset_id != asset_id.strip():
        raise ValueError("fluid asset ID must be a non-empty canonical string")
    return struct.unpack(_KEY_FORMAT, hashlib.sha256(asset_id.encode("utf-8")).digest())


def _verified_path(bundle: V2JaxBundle, relative: str, expected: str) -> Path:
    path = (bundle.artifact_root / PurePosixPath(relative)).resolve()
    if not path.is_relative_to(bundle.artifact_root) or not path.is_file():
        raise ValueError(f"selected artifact is unavailable: {relative}")
    if _file_sha256(path) != expected:
        raise ValueError(f"selected artifact file SHA-256 differs: {relative}")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{label} must be a SHA-256")
    return value.lower()


def _exact_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error


def _nonnegative_int(value: object, label: str) -> int:
    result = _exact_int(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _positive_int(value: object, label: str) -> int:
    result = _exact_int(value, label)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


__all__ = [
    "FLUID_ATLAS_SCHEMA",
    "FLUID_ATLAS_VERSION",
    "FLUID_KEY_IDENTITY",
    "V2FluidSemanticAtlas",
    "V2FluidSemanticSelection",
    "V2JaxWorldSet",
    "load_selected_jax_worlds",
    "lookup_v2_fluid_identity",
    "materialize_jax_worlds",
]
