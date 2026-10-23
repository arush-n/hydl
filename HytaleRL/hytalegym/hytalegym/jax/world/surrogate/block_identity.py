"""Exact per-cell block identity for offline surrogate artifacts.

Identity is artifact-side: it is stored with the world and covered by the
artifact digest, but it is not part of the runtime atlas, so the runtime memory
contract is unchanged. Each stored cell kind has an index array with the same
shape as the cell-code array it mirrors. The index names the identity-table row
that the generator recorded when it placed the cell's asset.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax
import numpy as np

from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.world.surrogate.terrain import SurrogateTerrainTile
from hytalegym.jax.world.surrogate.types import SurrogateAtlas
from hytalegym.worldgen.block_actions import (
    BLOCK_SEMANTIC_KEY_WORDS as _KEY_WORDS,
    block_asset_key,
    block_semantic_key,
)
from hytalegym.worldgen.surrogate import SurrogateWorldCapacity
from hytalegym.worldgen.surrogate.block_identity import (
    AIR_IDENTITY_ROW,
    IDENTITY_INDEX_DTYPE,
    MAXIMUM_IDENTITY_ROWS,
    CompiledBlockIdentity,
    merge_block_identities,
)
from hytalegym.worldgen.surrogate.palette import ResolvedPrefabOverlay
from hytalegym.worldgen.surrogate.stateful import ResolvedStatefulPrefabOverlay
from hytalegym.worldgen.region import CAPTURE_BLOCKS_PER_AXIS

CELL_KINDS = ("surface", "subsurface", "structure", "stateful_variant")
_KIND_FIELDS = {
    "surface": "surface_identity_index",
    "subsurface": "subsurface_identity_index",
    "structure": "structure_identity_index",
    "stateful_variant": "stateful_variant_identity_index",
}
_KIND_CODE_FIELDS = {
    "surface": "surface_cell_code",
    "subsurface": "subsurface_cell_code",
    "structure": "structure_cell_code",
    "stateful_variant": "stateful_variant_cell_code",
}


class SurrogateBlockIdentity(NamedTuple):
    """Identity table rows plus per-cell row indices; numpy only (strings)."""

    identity_asset_id: np.ndarray  # (R,) unicode; row 0 is ""
    identity_asset_key: np.ndarray  # (R, 8) uint32
    identity_semantic_key: np.ndarray  # (R, 8) uint32
    identity_semantic_id: np.ndarray  # (R,) int32; 0 for air
    identity_rotation: np.ndarray  # (R,) int32
    surface_identity_index: np.ndarray  # (T, C, C) uint16
    subsurface_identity_index: np.ndarray  # (T, C, C) uint16
    structure_identity_index: np.ndarray  # (T, structure cells) uint16
    stateful_variant_identity_index: np.ndarray  # (T, blocks, states) uint16


def surrogate_block_identity_from_tiles(
    terrain: SurrogateTerrainTile,
    terrain_identity: CompiledBlockIdentity,
    overlays: Sequence[ResolvedPrefabOverlay | ResolvedStatefulPrefabOverlay | None],
    *,
    capacity: SurrogateWorldCapacity,
    tile_capacity: int,
) -> SurrogateBlockIdentity:
    """Merge recorded identity tables and remap every recorded cell index."""

    if not isinstance(terrain, SurrogateTerrainTile):
        raise TypeError("terrain must be a SurrogateTerrainTile")
    if not isinstance(terrain_identity, CompiledBlockIdentity):
        raise TypeError("terrain_identity must be a CompiledBlockIdentity")
    if not isinstance(capacity, SurrogateWorldCapacity):
        raise TypeError("capacity must be a SurrogateWorldCapacity")
    valid = np.asarray(jax.device_get(terrain.valid), dtype=np.bool_)
    tile_count = valid.size
    if tile_count == 0 or tile_capacity < tile_count:
        raise ValueError("tile capacity must cover the terrain tiles")
    source = list(overlays)
    if len(source) != tile_count:
        raise TypeError("overlays must match terrain tiles")

    recorded_surface = np.asarray(
        jax.device_get(terrain.surface_identity_index),
        dtype=IDENTITY_INDEX_DTYPE,
    )
    recorded_subsurface = np.asarray(
        jax.device_get(terrain.subsurface_identity_index),
        dtype=IDENTITY_INDEX_DTYPE,
    )
    for recorded in (recorded_surface, recorded_subsurface):
        if np.any(recorded >= len(terrain_identity.rows)):
            raise ValueError("terrain identity index exceeds its table")

    overlay_tables = [value.identity for value in source if value is not None]
    merged, remaps = merge_block_identities([terrain_identity, *overlay_tables])
    terrain_remap = remaps[0]
    overlay_remaps = iter(remaps[1:])
    tile_remaps: list[np.ndarray | None] = []
    for value in source:
        tile_remaps.append(None if value is None else next(overlay_remaps))

    column_shape = (tile_capacity, CAPTURE_BLOCKS_PER_AXIS, CAPTURE_BLOCKS_PER_AXIS)
    surface = np.zeros(column_shape, dtype=IDENTITY_INDEX_DTYPE)
    subsurface = np.zeros(column_shape, dtype=IDENTITY_INDEX_DTYPE)
    surface[:tile_count] = terrain_remap[recorded_surface]
    subsurface[:tile_count] = terrain_remap[recorded_subsurface]
    structure = np.zeros(
        (tile_capacity, capacity.structure_cell_capacity),
        dtype=IDENTITY_INDEX_DTYPE,
    )
    stateful = np.zeros(
        (tile_capacity, capacity.stateful_block_capacity, capacity.states_per_block),
        dtype=IDENTITY_INDEX_DTYPE,
    )
    for index, value in enumerate(source):
        if value is None:
            continue
        remap = tile_remaps[index]
        active = np.flatnonzero(value.source.cell_mask)
        if active.size > capacity.structure_cell_capacity:
            raise ValueError("structure overlay exceeds atlas capacity")
        structure[index, : active.size] = remap[value.cell_identity[active]]
        if not isinstance(value, ResolvedStatefulPrefabOverlay):
            continue
        recorded = value.stateful.variant_identity
        if recorded.shape != (
            capacity.stateful_block_capacity,
            capacity.states_per_block,
        ):
            raise ValueError("stateful identity differs from atlas capacity")
        for slot in np.flatnonzero(value.stateful.block_mask):
            states = int(value.stateful.state_count[slot])
            stateful[index, slot, :states] = remap[recorded[slot, :states]]

    rows = merged.rows
    if len(rows) > MAXIMUM_IDENTITY_ROWS:
        raise ValueError("block identity table exceeds the index dtype")
    return SurrogateBlockIdentity(
        identity_asset_id=np.asarray([row.asset_id for row in rows], dtype=np.str_),
        identity_asset_key=np.asarray(
            [row.asset_key for row in rows], dtype=np.uint32
        ).reshape(len(rows), _KEY_WORDS),
        identity_semantic_key=np.asarray(
            [row.semantic_key for row in rows], dtype=np.uint32
        ).reshape(len(rows), _KEY_WORDS),
        identity_semantic_id=np.asarray(
            [0 if not row.asset_id else semantic_id(row.asset_id) for row in rows],
            dtype=np.int32,
        ),
        identity_rotation=np.asarray(
            [row.rotation for row in rows], dtype=np.int32
        ),
        surface_identity_index=surface,
        subsurface_identity_index=subsurface,
        structure_identity_index=structure,
        stateful_variant_identity_index=stateful,
    )


def validate_block_identity(
    identity: SurrogateBlockIdentity,
    atlas: SurrogateAtlas,
) -> None:
    """Fail closed unless the identity agrees with its own keys and the atlas."""

    if not isinstance(identity, SurrogateBlockIdentity):
        raise TypeError("identity must be a SurrogateBlockIdentity")
    assets = identity.identity_asset_id
    if assets.dtype.kind != "U" or assets.ndim != 1 or not 1 <= assets.size <= (
        MAXIMUM_IDENTITY_ROWS
    ):
        raise ValueError("block identity asset table is malformed")
    names = [str(value) for value in assets]
    if names[AIR_IDENTITY_ROW] != "" or any(not name for name in names[1:]):
        raise ValueError("block identity row 0 must be air and others named")
    rotation = np.asarray(identity.identity_rotation)
    if rotation.shape != (len(names),) or np.any(rotation < 0):
        raise ValueError("block identity rotations are malformed")
    zero_key = [0] * _KEY_WORDS
    expected_semantic = [0] + [semantic_id(name) for name in names[1:]]
    expected_asset = [zero_key] + [list(block_asset_key(name)) for name in names[1:]]
    expected_semantic_key = [zero_key] + [
        list(block_semantic_key(name, int(rotation[index])))
        for index, name in enumerate(names[1:], start=1)
    ]
    if (
        not np.array_equal(identity.identity_semantic_id, expected_semantic)
        or not np.array_equal(identity.identity_asset_key, expected_asset)
        or not np.array_equal(identity.identity_semantic_key, expected_semantic_key)
    ):
        raise ValueError("block identity keys differ from asset names")
    for kind in CELL_KINDS:
        index = np.asarray(getattr(identity, _KIND_FIELDS[kind]))
        code = np.asarray(jax.device_get(getattr(atlas, _KIND_CODE_FIELDS[kind])))
        if index.shape != code.shape or index.dtype != IDENTITY_INDEX_DTYPE:
            raise ValueError(f"{kind} identity index differs from cell codes")
        if np.any(index >= len(names)):
            raise ValueError(f"{kind} identity index exceeds the table")
        if np.any((code != 0) & (index == AIR_IDENTITY_ROW)):
            raise ValueError(f"{kind} non-air cell has air identity")


def surrogate_cell_identity(
    identity: SurrogateBlockIdentity,
    kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (semantic_id, asset_id) for every stored cell of one kind."""

    if kind not in CELL_KINDS:
        raise ValueError(f"unknown cell kind {kind!r}")
    index = np.asarray(getattr(identity, _KIND_FIELDS[kind]))
    return identity.identity_semantic_id[index], identity.identity_asset_id[index]


__all__ = [
    "CELL_KINDS",
    "SurrogateBlockIdentity",
    "surrogate_block_identity_from_tiles",
    "surrogate_cell_identity",
    "validate_block_identity",
]
