"""Fixed-shape JAX metadata for marker-backed WorldGen V2 structures."""

from __future__ import annotations

import hashlib
import struct
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.region.types import RegionAtlas
from hytalegym.worldgen.block_actions import BLOCK_SEMANTIC_KEY_WORDS

from ..bundles.bundle import V2JaxArtifact
from .structure_pack import V2StructureSnapshot


STRUCTURE_KIND_NONE = 0
STRUCTURE_KIND_PREFAB = 1
STRUCTURE_KIND_STATIC_PROP = 2
STRUCTURE_COVERAGE_NONE = 0
STRUCTURE_COVERAGE_COMPLETE = 1
STRUCTURE_COVERAGE_CLIPPED = 2

_KIND_CODE = {
    "prefab": STRUCTURE_KIND_PREFAB,
    "static_prop": STRUCTURE_KIND_STATIC_PROP,
}
_COVERAGE_CODE = {
    "complete": STRUCTURE_COVERAGE_COMPLETE,
    "clipped_to_capture": STRUCTURE_COVERAGE_CLIPPED,
}
_KEY_FORMAT = f">{BLOCK_SEMANTIC_KEY_WORDS}I"


class V2StructureInstanceAtlas(NamedTuple):
    """Padded immutable placed-instance metadata aligned to Region worlds."""

    region_mask: jax.Array
    world_id: jax.Array
    instance_mask: jax.Array
    instance_key: jax.Array
    structure_asset_key: jax.Array
    marker_asset_key: jax.Array
    kind: jax.Array
    anchor_world: jax.Array
    quarter_turns: jax.Array
    bounds_min: jax.Array
    bounds_max: jax.Array
    capture_coverage: jax.Array
    geometry_present: jax.Array
    native_worldgen_id_words: jax.Array
    native_prefab_instance_id_words: jax.Array


class V2StructurePointSelection(NamedTuple):
    """All placed instances whose half-open bounds contain each query."""

    available: jax.Array
    instance_mask: jax.Array
    instance_key: jax.Array
    structure_asset_key: jax.Array
    kind: jax.Array
    anchor_world: jax.Array
    quarter_turns: jax.Array


def structure_instance_atlas_from_snapshots(
    region_atlas: RegionAtlas,
    snapshots: Sequence[V2StructureSnapshot],
    artifacts: Sequence[V2JaxArtifact],
    *,
    instance_capacity: int | None = None,
) -> V2StructureInstanceAtlas:
    """Pad exact marker snapshots without inferring unmarked structures."""

    if not isinstance(region_atlas, RegionAtlas):
        raise TypeError("region_atlas must be a RegionAtlas")
    source = tuple(snapshots)
    selected = tuple(artifacts)
    if not source or len(source) != len(selected):
        raise ValueError("structure snapshots must align with selected artifacts")
    if any(not isinstance(row, V2StructureSnapshot) for row in source):
        raise TypeError("snapshots must contain V2StructureSnapshot values")
    if any(not isinstance(row, V2JaxArtifact) for row in selected):
        raise TypeError("artifacts must contain V2JaxArtifact values")
    regions = int(region_atlas.region_mask.shape[0])
    if regions < len(source):
        raise ValueError("Region atlas is smaller than structure snapshots")
    for snapshot, artifact in zip(source, selected, strict=True):
        if snapshot.source_region_semantic_sha256 != (
            artifact.region_semantic_sha256
        ):
            raise ValueError("structure snapshot and Region artifact disagree")

    required = max((len(row.instances) for row in source), default=0)
    capacity = max(required, 1)
    if instance_capacity is not None:
        capacity = _positive_int(instance_capacity, "structure_instance_capacity")
        if capacity < required:
            raise ValueError("structure instance capacity is too small")

    region_mask = np.zeros(regions, dtype=np.bool_)
    instance_mask = np.zeros((regions, capacity), dtype=np.bool_)
    key_shape = (regions, capacity, BLOCK_SEMANTIC_KEY_WORDS)
    instance_key = np.zeros(key_shape, dtype=np.uint32)
    structure_asset_key = np.zeros(key_shape, dtype=np.uint32)
    marker_asset_key = np.zeros(key_shape, dtype=np.uint32)
    kind = np.zeros((regions, capacity), dtype=np.uint8)
    anchor_world = np.zeros((regions, capacity, 3), dtype=np.int32)
    quarter_turns = np.zeros((regions, capacity), dtype=np.uint8)
    bounds_min = np.zeros((regions, capacity, 3), dtype=np.int32)
    bounds_max = np.zeros((regions, capacity, 3), dtype=np.int32)
    capture_coverage = np.zeros((regions, capacity), dtype=np.uint8)
    geometry_present = np.zeros((regions, capacity), dtype=np.bool_)
    native_worldgen_id_words = np.zeros((regions, capacity, 2), dtype=np.uint32)
    native_prefab_instance_id_words = np.zeros(
        (regions, capacity, 2), dtype=np.uint32
    )
    for world_index, snapshot in enumerate(source):
        region_mask[world_index] = True
        for instance_index, instance in enumerate(snapshot.instances):
            instance_mask[world_index, instance_index] = True
            instance_key[world_index, instance_index] = _digest_words(
                bytes.fromhex(instance.instance_id)
            )
            structure_asset_key[world_index, instance_index] = _asset_key(
                instance.structure_asset_id
            )
            marker_asset_key[world_index, instance_index] = _asset_key(
                instance.marker_asset_id
            )
            kind[world_index, instance_index] = _KIND_CODE[instance.kind]
            anchor_world[world_index, instance_index] = instance.anchor_world
            quarter_turns[world_index, instance_index] = instance.quarter_turns
            bounds_min[world_index, instance_index] = instance.bounds_min
            bounds_max[world_index, instance_index] = instance.bounds_max
            capture_coverage[world_index, instance_index] = _COVERAGE_CODE[
                instance.capture_coverage
            ]
            geometry_present[world_index, instance_index] = (
                instance.geometry_present
            )
            native_worldgen_id_words[world_index, instance_index] = _u64_words(
                instance.native_worldgen_id,
                "native_worldgen_id",
            )
            native_prefab_instance_id_words[
                world_index, instance_index
            ] = _u64_words(
                instance.native_prefab_instance_id,
                "native_prefab_instance_id",
            )
    physical_mask = np.asarray(
        jax.device_get(region_atlas.region_mask), dtype=np.bool_
    )
    if np.any(region_mask & ~physical_mask):
        raise ValueError("structure metadata occupies an inactive Region slot")
    return V2StructureInstanceAtlas(
        region_mask=jnp.asarray(region_mask),
        world_id=region_atlas.world_id,
        instance_mask=jnp.asarray(instance_mask),
        instance_key=jnp.asarray(instance_key),
        structure_asset_key=jnp.asarray(structure_asset_key),
        marker_asset_key=jnp.asarray(marker_asset_key),
        kind=jnp.asarray(kind),
        anchor_world=jnp.asarray(anchor_world),
        quarter_turns=jnp.asarray(quarter_turns),
        bounds_min=jnp.asarray(bounds_min),
        bounds_max=jnp.asarray(bounds_max),
        capture_coverage=jnp.asarray(capture_coverage),
        geometry_present=jnp.asarray(geometry_present),
        native_worldgen_id_words=jnp.asarray(native_worldgen_id_words),
        native_prefab_instance_id_words=jnp.asarray(
            native_prefab_instance_id_words
        ),
    )


def lookup_v2_structure_instances(
    atlas: V2StructureInstanceAtlas,
    block_positions: jax.Array,
    environment_world_id: jax.Array,
) -> V2StructurePointSelection:
    """Return every authored instance containing each world-space block."""

    if not isinstance(atlas, V2StructureInstanceAtlas):
        raise TypeError("atlas must be a V2StructureInstanceAtlas")
    blocks = jnp.asarray(block_positions, dtype=jnp.int32)
    if blocks.ndim != 3 or blocks.shape[-1] != 3:
        raise ValueError("block_positions must have shape [batch, queries, 3]")
    world_ids = jnp.asarray(environment_world_id, dtype=jnp.int32)
    if world_ids.shape != (blocks.shape[0],):
        raise ValueError("environment_world_id must have shape [batch]")
    compatible = atlas.region_mask[None, :] & (
        atlas.world_id[None, :] == world_ids[:, None]
    )
    available = jnp.any(compatible, axis=1)
    region_index = jnp.argmax(compatible, axis=1)
    safe_region = jnp.where(available, region_index, 0)
    minimum = atlas.bounds_min[safe_region]
    maximum = atlas.bounds_max[safe_region]
    contains = jnp.all(
        (blocks[:, :, None, :] >= minimum[:, None, :, :])
        & (blocks[:, :, None, :] < maximum[:, None, :, :]),
        axis=3,
    )
    matches = (
        available[:, None, None]
        & atlas.instance_mask[safe_region, None, :]
        & contains
    )
    return V2StructurePointSelection(
        available=jnp.broadcast_to(available[:, None], blocks.shape[:2]),
        instance_mask=matches,
        instance_key=jnp.where(
            matches[..., None],
            atlas.instance_key[safe_region, None, :, :],
            0,
        ),
        structure_asset_key=jnp.where(
            matches[..., None],
            atlas.structure_asset_key[safe_region, None, :, :],
            0,
        ),
        kind=jnp.where(matches, atlas.kind[safe_region, None, :], 0),
        anchor_world=jnp.where(
            matches[..., None],
            atlas.anchor_world[safe_region, None, :, :],
            0,
        ),
        quarter_turns=jnp.where(
            matches, atlas.quarter_turns[safe_region, None, :], 0
        ),
    )


def _asset_key(asset_id: str) -> tuple[int, ...]:
    return _digest_words(hashlib.sha256(asset_id.encode("utf-8")).digest())


def _digest_words(digest: bytes) -> tuple[int, ...]:
    if len(digest) != 32:
        raise ValueError("structure identity digest must contain 32 bytes")
    return struct.unpack(_KEY_FORMAT, digest)


def _u64_words(value: int, label: str) -> tuple[int, int]:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if not 0 <= value < (1 << 64):
        raise ValueError(f"{label} must fit unsigned 64-bit")
    return value >> 32, value & 0xFFFFFFFF


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 1:
        raise ValueError(f"{label} must be positive")
    return value
