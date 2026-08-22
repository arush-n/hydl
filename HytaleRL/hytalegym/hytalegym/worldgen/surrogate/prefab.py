"""Deterministic fixed-capacity placement of validated prefab cells."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import operator

import numpy as np

from hytalegym.worldgen.region import (
    CAPTURE_BLOCKS_PER_AXIS,
    CHUNK_SIZE,
    MIN_Y,
    WORLD_HEIGHT,
)
from hytalegym.worldgen.surrogate.assets import (
    HytaleAssetError,
    PrefabBlock,
    PrefabEntity,
    PrefabTemplate,
    decode_prefab_filler,
)
from hytalegym.worldgen.surrogate.contract import SurrogateWorldCapacity


@dataclass(frozen=True)
class PrefabPlacement:
    """One prefab anchor in absolute block coordinates."""

    template: PrefabTemplate
    anchor_world: tuple[int, int, int]
    quarter_turns: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.template, PrefabTemplate):
            raise TypeError("template must be a PrefabTemplate")
        object.__setattr__(
            self,
            "anchor_world",
            _int_tuple(self.anchor_world, 3, "anchor_world"),
        )
        turns = _integer(self.quarter_turns, "quarter_turns")
        if turns not in range(4):
            raise ValueError("quarter_turns must be in [0, 4)")
        object.__setattr__(self, "quarter_turns", turns)


def place_prefab_entity_transform(
    entity: PrefabEntity,
    template_anchor: tuple[int, int, int],
    world_anchor: tuple[int, int, int],
    quarter_turns: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Apply the inspected native half-cell entity placement transform."""

    if not isinstance(entity, PrefabEntity):
        raise TypeError("entity must be a PrefabEntity")
    source_anchor = _int_tuple(template_anchor, 3, "template_anchor")
    target_anchor = _int_tuple(world_anchor, 3, "world_anchor")
    turns = _integer(quarter_turns, "quarter_turns")
    if turns not in range(4):
        raise ValueError("quarter_turns must be in [0, 4)")
    local_x = entity.position[0] - source_anchor[0] - 0.5
    local_y = entity.position[1] - source_anchor[1]
    local_z = entity.position[2] - source_anchor[2] - 0.5
    rotated_x, rotated_z = (
        (local_x, local_z),
        (local_z, -local_x),
        (-local_x, -local_z),
        (-local_z, local_x),
    )[turns]
    placed = (
        target_anchor[0] + rotated_x + 0.5,
        target_anchor[1] + local_y,
        target_anchor[2] + rotated_z + 0.5,
    )
    rotation = (
        entity.rotation[0],
        entity.rotation[1],
        entity.rotation[2] + turns * (math.pi / 2.0),
    )
    if not all(math.isfinite(value) for value in (*placed, *rotation)):
        raise HytaleAssetError("placed prefab entity transform is not finite")
    return placed, rotation


@dataclass(frozen=True)
class PrefabCompilationPolicy:
    """Explicitly acknowledged non-geometry component omissions."""

    ignored_component_types: frozenset[str] = frozenset()
    allow_clipping: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.ignored_component_types, frozenset) or any(
            not isinstance(value, str) or not value
            for value in self.ignored_component_types
        ):
            raise TypeError("ignored_component_types must be a frozenset of names")
        if not isinstance(self.allow_clipping, bool):
            raise TypeError("allow_clipping must be boolean")


@dataclass(frozen=True)
class CompiledPrefabOverlay:
    """Sorted padded host arrays before block semantics are resolved."""

    tile_min_xz: tuple[int, int]
    cell_mask: np.ndarray
    block_key: np.ndarray
    asset_code: np.ndarray
    block_rotation: np.ndarray
    filler: np.ndarray
    support: np.ndarray
    source_code: np.ndarray
    asset_ids: tuple[str, ...]
    source_sha256: tuple[str, ...]
    ignored_component_types: tuple[str, ...]
    clipped_cell_count: int

    @property
    def cell_count(self) -> int:
        return int(np.count_nonzero(self.cell_mask))

    @property
    def logical_bytes(self) -> int:
        return sum(
            value.nbytes
            for value in (
                self.cell_mask,
                self.block_key,
                self.asset_code,
                self.block_rotation,
                self.filler,
                self.support,
                self.source_code,
            )
        )


def compile_prefab_overlay(
    placements: Sequence[PrefabPlacement],
    *,
    tile_min_xz: tuple[int, int],
    capacity: int | None = None,
    policy: PrefabCompilationPolicy | None = None,
) -> CompiledPrefabOverlay:
    """Compile prefab cells into deterministic padded arrays.

    Coordinate, block-rotation, and filler transforms reproduce the local
    0.5.7 ``PrefabRotation`` and ``FillerBlockUtil`` API. A native fixture is
    still required before calling the complete placement path certified.
    """

    source = list(placements)
    if not source:
        raise ValueError("at least one prefab placement is required")
    if any(not isinstance(value, PrefabPlacement) for value in source):
        raise TypeError("placements must contain PrefabPlacement values")
    tile_min = _int_tuple(tile_min_xz, 2, "tile_min_xz")
    if any(value % CHUNK_SIZE for value in tile_min):
        raise ValueError("tile_min_xz must be chunk aligned")
    maximum = (
        SurrogateWorldCapacity().structure_cell_capacity
        if capacity is None
        else _positive_int(capacity, "structure cell capacity")
    )
    settings = PrefabCompilationPolicy() if policy is None else policy
    if not isinstance(settings, PrefabCompilationPolicy):
        raise TypeError("policy must be a PrefabCompilationPolicy")

    records: dict[
        int,
        tuple[PrefabBlock, str],
    ] = {}
    clipped = 0
    observed_components: set[str] = set()
    for placement in source:
        template = placement.template
        if not template.static_placement_supported:
            raise HytaleAssetError("prefab contains unsupported placement fields")
        unsupported_components = set(template.component_types).difference(
            settings.ignored_component_types
        )
        if unsupported_components:
            raise HytaleAssetError(
                "prefab components require an explicit policy: "
                f"{sorted(unsupported_components)}"
            )
        observed_components.update(template.component_types)
        _validate_template_fillers(template)

        for block in template.blocks:
            offset = tuple(
                block.position[axis] - template.anchor[axis] for axis in range(3)
            )
            rotated = _rotate_y(offset, placement.quarter_turns)
            world = tuple(
                placement.anchor_world[axis] + rotated[axis] for axis in range(3)
            )
            _require_int32_tuple(world, "placed block position")
            local = (
                world[0] - tile_min[0],
                world[1] - MIN_Y,
                world[2] - tile_min[1],
            )
            inside = (
                0 <= local[0] < CAPTURE_BLOCKS_PER_AXIS
                and 0 <= local[1] < WORLD_HEIGHT
                and 0 <= local[2] < CAPTURE_BLOCKS_PER_AXIS
            )
            if not inside:
                if settings.allow_clipping:
                    clipped += 1
                    continue
                raise HytaleAssetError(
                    f"placed block {world} leaves the surrogate tile"
                )
            key = _block_key(local)
            if key in records:
                raise HytaleAssetError(
                    f"prefab placements overlap at world cell {world}"
                )
            records[key] = (
                PrefabBlock(
                    position=block.position,
                    asset_id=block.asset_id,
                    rotation=_rotate_block_rotation(
                        block.rotation,
                        placement.quarter_turns,
                    ),
                    filler=_rotate_filler(
                        block.filler,
                        placement.quarter_turns,
                    ),
                    component_types=block.component_types,
                    # The local 0.5.7 PrefabBuffer accessor rotates only
                    # coordinates, RotationTuple, and filler. Its four-bit
                    # support value is passed through unchanged.
                    support=block.support,
                ),
                template.provenance.content_sha256,
            )
            if len(records) > maximum:
                raise HytaleAssetError(
                    f"prefab overlay exceeds structure cell capacity {maximum}"
                )

    asset_ids = tuple(sorted({block.asset_id for block, _source in records.values()}))
    sources = tuple(sorted({_source for _block, _source in records.values()}))
    if len(asset_ids) > np.iinfo(np.uint16).max + 1:
        raise HytaleAssetError("prefab asset palette exceeds uint16")
    if len(sources) > np.iinfo(np.uint16).max + 1:
        raise HytaleAssetError("prefab source palette exceeds uint16")
    asset_index = {value: index for index, value in enumerate(asset_ids)}
    source_index = {value: index for index, value in enumerate(sources)}

    cell_mask = np.zeros(maximum, dtype=np.bool_)
    block_key = np.full(maximum, np.iinfo(np.int32).max, dtype=np.int32)
    asset_code = np.zeros(maximum, dtype=np.uint16)
    block_rotation = np.full(maximum, -1, dtype=np.int32)
    filler = np.full(maximum, -1, dtype=np.int32)
    support = np.zeros(maximum, dtype=np.uint8)
    source_code = np.zeros(maximum, dtype=np.uint16)
    for index, key in enumerate(sorted(records)):
        block, source_hash = records[key]
        cell_mask[index] = True
        block_key[index] = key
        asset_code[index] = asset_index[block.asset_id]
        block_rotation[index] = -1 if block.rotation is None else block.rotation
        filler[index] = -1 if block.filler is None else block.filler
        support[index] = block.support
        source_code[index] = source_index[source_hash]
    arrays = (
        cell_mask,
        block_key,
        asset_code,
        block_rotation,
        filler,
        support,
        source_code,
    )
    for value in arrays:
        value.flags.writeable = False
    return CompiledPrefabOverlay(
        tile_min_xz=tile_min,
        cell_mask=cell_mask,
        block_key=block_key,
        asset_code=asset_code,
        block_rotation=block_rotation,
        filler=filler,
        support=support,
        source_code=source_code,
        asset_ids=asset_ids,
        source_sha256=sources,
        ignored_component_types=tuple(sorted(observed_components)),
        clipped_cell_count=clipped,
    )


def _block_key(local: tuple[int, int, int]) -> int:
    return (
        local[1] * CAPTURE_BLOCKS_PER_AXIS * CAPTURE_BLOCKS_PER_AXIS
        + local[2] * CAPTURE_BLOCKS_PER_AXIS
        + local[0]
    )


def _validate_template_fillers(template: PrefabTemplate) -> None:
    blocks = {block.position: block for block in template.blocks}
    for block in template.blocks:
        if block.filler is None:
            continue
        offset = decode_prefab_filler(block.filler)
        owner_position = tuple(block.position[axis] - offset[axis] for axis in range(3))
        owner = blocks.get(owner_position)
        if owner is None:
            raise HytaleAssetError("prefab filler owner is absent")
        if owner.asset_id != block.asset_id or owner.rotation != block.rotation:
            raise HytaleAssetError("prefab filler owner block or rotation disagrees")
        if owner.filler is not None:
            raise HytaleAssetError("prefab filler must point directly to a root block")


def _rotate_y(
    value: tuple[int, int, int],
    quarter_turns: int,
) -> tuple[int, int, int]:
    x, y, z = value
    return (
        (x, y, z),
        (z, y, -x),
        (-x, y, -z),
        (-z, y, x),
    )[quarter_turns]


def _rotate_block_rotation(
    rotation: int | None,
    quarter_turns: int,
) -> int | None:
    if rotation is None:
        return None if quarter_turns == 0 else quarter_turns
    if rotation >= 64:
        raise HytaleAssetError("block rotation exceeds RotationTuple capacity")
    yaw = rotation & 3
    return (rotation & ~3) | ((yaw + quarter_turns) & 3)


def _rotate_filler(
    filler: int | None,
    quarter_turns: int,
) -> int | None:
    if filler is None or quarter_turns == 0:
        return filler
    x = _unpack_filler(filler, 0)
    z = _unpack_filler(filler, 5)
    y = _unpack_filler(filler, 10)
    rotated = _rotate_y((x, y, z), quarter_turns)
    return (rotated[0] & 31) | ((rotated[2] & 31) << 5) | ((rotated[1] & 31) << 10)


def _unpack_filler(value: int, shift: int) -> int:
    result = (value >> shift) & 31
    return result | -32 if result & 16 else result


def _require_int32_tuple(value: tuple[int, ...], label: str) -> None:
    limits = np.iinfo(np.int32)
    if any(item < limits.min or item > limits.max for item in value):
        raise HytaleAssetError(f"{label} exceeds int32")


def _int_tuple(
    value: tuple[int, ...],
    length: int,
    label: str,
) -> tuple[int, ...]:
    if not isinstance(value, tuple) or len(value) != length:
        raise TypeError(f"{label} must be a {length}-tuple")
    result = tuple(_integer(item, label) for item in value)
    _require_int32_tuple(result, label)
    return result


def _positive_int(value: int, label: str) -> int:
    result = _integer(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _integer(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must contain integers")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must contain integers") from error


__all__ = [
    "CompiledPrefabOverlay",
    "PrefabCompilationPolicy",
    "PrefabPlacement",
    "compile_prefab_overlay",
    "place_prefab_entity_transform",
]
