"""Fixed-capacity collision/LOS palettes for surrogate structure cells."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from hytalegym.geometry.contract import (
    FLUID_MOVEMENT_FEATURES,
    MAX_DETAIL_BOXES,
    MOVEMENT_FEATURES,
)
from hytalegym.worldgen.surrogate.assets import HytaleAssetError
from hytalegym.worldgen.surrogate.contract import SurrogateWorldCapacity
from hytalegym.worldgen.surrogate.prefab import CompiledPrefabOverlay
from hytalegym.worldgen.surrogate.semantics import (
    LocalBlockSemantics,
    LocalBlockSemanticsResolver,
)

ShapeKey = tuple[tuple[float, float, float, float, float, float], ...]
ReferenceKey = tuple[str, int, int, int]
CellKey = tuple[int, ShapeKey, int]


@dataclass(frozen=True)
class CompiledSemanticPalette:
    """Padded native-shaped cell and collision palettes."""

    cell_count: int
    shape_count: int
    cell_flags: np.ndarray
    cell_shape_index: np.ndarray
    cell_fluid_level: np.ndarray
    cell_support: np.ndarray
    cell_block_damage: np.ndarray
    cell_fluid_damage: np.ndarray
    cell_movement: np.ndarray
    cell_fluid_movement: np.ndarray
    shape_boxes: np.ndarray
    shape_box_mask: np.ndarray
    reference_keys: tuple[ReferenceKey, ...]
    reference_cell_code: np.ndarray

    @property
    def logical_bytes(self) -> int:
        return sum(
            value.nbytes
            for value in (
                self.cell_flags,
                self.cell_shape_index,
                self.cell_fluid_level,
                self.cell_support,
                self.cell_block_damage,
                self.cell_fluid_damage,
                self.cell_movement,
                self.cell_fluid_movement,
                self.shape_boxes,
                self.shape_box_mask,
            )
        )

    def cell_code(
        self,
        reference: str,
        rotation: int = 0,
        support: int = 0,
        filler: int = 0,
    ) -> int:
        """Return the compiled code for an exact resolved reference."""

        key = (reference, rotation, support, filler)
        try:
            index = self.reference_keys.index(key)
        except ValueError as error:
            raise KeyError(key) from error
        return int(self.reference_cell_code[index])


@dataclass(frozen=True)
class ResolvedPrefabOverlay:
    """A sparse prefab overlay with concrete semantic cell codes."""

    source: CompiledPrefabOverlay
    cell_code: np.ndarray
    state_slot: np.ndarray
    palette: CompiledSemanticPalette

    @property
    def cell_count(self) -> int:
        return self.source.cell_count


def compile_local_semantic_palette(
    semantics: Sequence[LocalBlockSemantics],
    *,
    capacity: SurrogateWorldCapacity | None = None,
) -> CompiledSemanticPalette:
    """Deduplicate exact local geometry without input-order dependence."""

    source = list(semantics)
    if any(not isinstance(value, LocalBlockSemantics) for value in source):
        raise TypeError("semantics must contain LocalBlockSemantics values")
    layout = SurrogateWorldCapacity() if capacity is None else capacity
    if not isinstance(layout, SurrogateWorldCapacity):
        raise TypeError("capacity must be a SurrogateWorldCapacity")

    reference_semantics: dict[ReferenceKey, CellKey] = {}
    for value in source:
        _require_support_value(value.support)
        key = (
            value.reference,
            value.rotation,
            value.support,
            value.filler,
        )
        semantic_key = (value.flags, _shape_key(value), value.support)
        previous = reference_semantics.setdefault(key, semantic_key)
        if previous != semantic_key:
            raise HytaleAssetError(
                f"semantic reference {key!r} resolves inconsistently"
            )
    return _compile_reference_semantics(reference_semantics, layout)


def merge_compiled_semantic_palettes(
    palettes: Sequence[CompiledSemanticPalette],
    *,
    capacity: SurrogateWorldCapacity | None = None,
) -> CompiledSemanticPalette:
    """Merge local palettes by expanded semantics, rejecting conflicts."""

    source = list(palettes)
    if not source:
        raise ValueError("at least one semantic palette is required")
    if any(not isinstance(value, CompiledSemanticPalette) for value in source):
        raise TypeError("palettes must contain CompiledSemanticPalette values")
    layout = SurrogateWorldCapacity() if capacity is None else capacity
    if not isinstance(layout, SurrogateWorldCapacity):
        raise TypeError("capacity must be a SurrogateWorldCapacity")
    reference_semantics: dict[ReferenceKey, CellKey] = {}
    for palette in source:
        _require_zero_v1_cell_extras(palette)
        for key, raw_code in zip(
            palette.reference_keys,
            palette.reference_cell_code,
            strict=True,
        ):
            code = int(raw_code)
            if not 0 <= code < palette.cell_count:
                raise HytaleAssetError("semantic reference cell code is invalid")
            shape = int(palette.cell_shape_index[code])
            if not 0 <= shape < palette.shape_count:
                raise HytaleAssetError("semantic cell shape code is invalid")
            shape_key = _compiled_shape_key(palette, shape)
            support = int(palette.cell_support[code])
            _require_support_value(support)
            semantic_key = (
                int(palette.cell_flags[code]),
                shape_key,
                support,
            )
            previous = reference_semantics.setdefault(key, semantic_key)
            if previous != semantic_key:
                raise HytaleAssetError(
                    f"semantic reference {key!r} conflicts across palettes"
                )
    return _compile_reference_semantics(reference_semantics, layout)


def semantic_palette_remap(
    source: CompiledSemanticPalette,
    target: CompiledSemanticPalette,
) -> np.ndarray:
    """Map every active source cell code into a merged target palette."""

    if not isinstance(source, CompiledSemanticPalette) or not isinstance(
        target,
        CompiledSemanticPalette,
    ):
        raise TypeError("source and target must be CompiledSemanticPalette values")
    result = np.zeros(source.cell_count, dtype=np.uint16)
    assigned = np.zeros(source.cell_count, dtype=np.bool_)
    assigned[0] = True
    for key, raw_source_code in zip(
        source.reference_keys,
        source.reference_cell_code,
        strict=True,
    ):
        source_code = int(raw_source_code)
        target_code = target.cell_code(*key)
        if assigned[source_code] and int(result[source_code]) != target_code:
            raise HytaleAssetError("semantic palette remap is inconsistent")
        result[source_code] = target_code
        assigned[source_code] = True
    if not np.all(assigned):
        raise HytaleAssetError("semantic palette contains an unreferenced cell")
    result.flags.writeable = False
    return result


def compile_role_opaque_cell_mask(
    palette: CompiledSemanticPalette,
    opaque_references: Iterable[str],
) -> np.ndarray:
    """Compile exact role opacity without guessing through palette aliases."""

    if not isinstance(palette, CompiledSemanticPalette):
        raise TypeError("palette must be a CompiledSemanticPalette")
    if isinstance(opaque_references, (str, bytes)):
        raise TypeError("opaque_references must be an iterable of asset IDs")
    requested = set(opaque_references)
    if any(not isinstance(value, str) or not value for value in requested):
        raise TypeError("opaque references must be non-empty strings")
    decisions: dict[int, bool] = {}
    for key, raw_code in zip(
        palette.reference_keys,
        palette.reference_cell_code,
        strict=True,
    ):
        code = int(raw_code)
        selected = key[0] in requested
        previous = decisions.setdefault(code, selected)
        if previous != selected:
            raise HytaleAssetError(
                "role opacity cannot distinguish semantic palette aliases "
                f"for cell code {code}"
            )
    result = np.zeros(palette.cell_flags.shape, dtype=np.bool_)
    for code, selected in decisions.items():
        result[code] = selected
    result.flags.writeable = False
    return result


def _compile_reference_semantics(
    reference_semantics: dict[ReferenceKey, CellKey],
    layout: SurrogateWorldCapacity,
) -> CompiledSemanticPalette:
    empty_shape: ShapeKey = ()
    shape_keys = sorted(
        {
            shape
            for _flags, shape, _support in reference_semantics.values()
            if shape != empty_shape
        }
    )
    if len(shape_keys) + 1 > layout.shape_palette_capacity:
        raise HytaleAssetError(
            "resolved shapes exceed surrogate shape palette capacity "
            f"{layout.shape_palette_capacity}"
        )
    shape_index = {empty_shape: 0}
    shape_index.update({shape: index + 1 for index, shape in enumerate(shape_keys)})

    air_key: CellKey = (0, empty_shape, 0)
    cell_keys = sorted(
        {semantic for semantic in reference_semantics.values() if semantic != air_key},
        key=lambda value: (value[0], shape_index[value[1]], value[2]),
    )
    if len(cell_keys) + 1 > layout.cell_palette_capacity:
        raise HytaleAssetError(
            "resolved cells exceed surrogate cell palette capacity "
            f"{layout.cell_palette_capacity}"
        )
    cell_index = {air_key: 0}
    cell_index.update({cell: index + 1 for index, cell in enumerate(cell_keys)})

    cell_capacity = layout.cell_palette_capacity
    shape_capacity = layout.shape_palette_capacity
    flags = np.zeros(cell_capacity, dtype=np.uint16)
    cell_shape = np.zeros(cell_capacity, dtype=np.uint16)
    fluid_level = np.zeros(cell_capacity, dtype=np.uint8)
    support = np.zeros(cell_capacity, dtype=np.int32)
    block_damage = np.zeros(cell_capacity, dtype=np.int32)
    fluid_damage = np.zeros(cell_capacity, dtype=np.int32)
    movement = np.zeros(
        (cell_capacity, MOVEMENT_FEATURES),
        dtype=np.float32,
    )
    fluid_movement = np.zeros(
        (cell_capacity, FLUID_MOVEMENT_FEATURES),
        dtype=np.float32,
    )
    boxes = np.zeros(
        (shape_capacity, MAX_DETAIL_BOXES, 6),
        dtype=np.float32,
    )
    box_mask = np.zeros(
        (shape_capacity, MAX_DETAIL_BOXES),
        dtype=np.bool_,
    )
    for key, index in shape_index.items():
        if index == 0:
            continue
        count = len(key)
        boxes[index, :count] = np.asarray(key, dtype=np.float32)
        box_mask[index, :count] = True
    for key, index in cell_index.items():
        flags[index] = key[0]
        cell_shape[index] = shape_index[key[1]]
        support[index] = key[2]

    reference_keys = tuple(sorted(reference_semantics))
    reference_cell_code = np.asarray(
        [cell_index[reference_semantics[key]] for key in reference_keys],
        dtype=np.uint16,
    )
    arrays = (
        flags,
        cell_shape,
        fluid_level,
        support,
        block_damage,
        fluid_damage,
        movement,
        fluid_movement,
        boxes,
        box_mask,
        reference_cell_code,
    )
    for value in arrays:
        value.flags.writeable = False
    return CompiledSemanticPalette(
        cell_count=len(cell_index),
        shape_count=len(shape_index),
        cell_flags=flags,
        cell_shape_index=cell_shape,
        cell_fluid_level=fluid_level,
        cell_support=support,
        cell_block_damage=block_damage,
        cell_fluid_damage=fluid_damage,
        cell_movement=movement,
        cell_fluid_movement=fluid_movement,
        shape_boxes=boxes,
        shape_box_mask=box_mask,
        reference_keys=reference_keys,
        reference_cell_code=reference_cell_code,
    )


def _compiled_shape_key(
    palette: CompiledSemanticPalette,
    shape: int,
) -> ShapeKey:
    mask = palette.shape_box_mask[shape]
    boxes = palette.shape_boxes[shape, mask]
    return tuple(sorted(tuple(float(item) for item in row) for row in boxes))


def _require_zero_v1_cell_extras(
    palette: CompiledSemanticPalette,
) -> None:
    active = slice(0, palette.cell_count)
    for value in (
        palette.cell_fluid_level[active],
        palette.cell_block_damage[active],
        palette.cell_fluid_damage[active],
        palette.cell_movement[active],
        palette.cell_fluid_movement[active],
    ):
        if np.any(value):
            raise HytaleAssetError(
                "surrogate palette v1 cannot merge authored cell extras"
            )


def resolve_static_prefab_overlay(
    overlay: CompiledPrefabOverlay,
    resolver: LocalBlockSemanticsResolver,
    *,
    capacity: SurrogateWorldCapacity | None = None,
) -> ResolvedPrefabOverlay:
    """Resolve a static overlay; stateful/filler mechanics are rejected."""

    if not isinstance(overlay, CompiledPrefabOverlay):
        raise TypeError("overlay must be a CompiledPrefabOverlay")
    if not isinstance(resolver, LocalBlockSemanticsResolver):
        raise TypeError("resolver must be a LocalBlockSemanticsResolver")
    resolved: list[LocalBlockSemantics] = []
    active: list[tuple[int, LocalBlockSemantics]] = []
    for index in np.flatnonzero(overlay.cell_mask):
        asset_code = int(overlay.asset_code[index])
        if not 0 <= asset_code < len(overlay.asset_ids):
            raise HytaleAssetError("prefab overlay asset code is invalid")
        rotation_value = int(overlay.block_rotation[index])
        filler_value = int(overlay.filler[index])
        semantic = resolver.resolve(
            overlay.asset_ids[asset_code],
            rotation=None if rotation_value < 0 else rotation_value,
            filler=None if filler_value < 0 else filler_value,
        )
        semantic = replace(semantic, support=int(overlay.support[index]))
        if semantic.is_door:
            raise HytaleAssetError(
                "stateful door requires the stateful structure compiler"
            )
        resolved.append(semantic)
        active.append((int(index), semantic))

    palette = compile_local_semantic_palette(resolved, capacity=capacity)
    cell_code = np.zeros(overlay.cell_mask.shape, dtype=np.uint16)
    state_slot = np.full(overlay.cell_mask.shape, -1, dtype=np.int32)
    for index, semantic in active:
        cell_code[index] = palette.cell_code(
            semantic.reference,
            semantic.rotation,
            semantic.support,
            semantic.filler,
        )
    cell_code.flags.writeable = False
    state_slot.flags.writeable = False
    return ResolvedPrefabOverlay(
        source=overlay,
        cell_code=cell_code,
        state_slot=state_slot,
        palette=palette,
    )


def _shape_key(value: LocalBlockSemantics) -> ShapeKey:
    boxes = []
    for box in value.collision_boxes:
        array = np.asarray(box.minimum + box.maximum, dtype=np.float32)
        if not np.all(np.isfinite(array)):
            raise HytaleAssetError("resolved collision box is non-finite")
        array[array == 0.0] = 0.0
        if np.any(array[:3] >= array[3:]):
            raise HytaleAssetError("resolved collision box has no volume")
        boxes.append(tuple(float(item) for item in array))
    if len(boxes) > MAX_DETAIL_BOXES:
        raise HytaleAssetError(
            f"resolved shape exceeds detail-box capacity {MAX_DETAIL_BOXES}"
        )
    return tuple(sorted(boxes))


def _require_support_value(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 15:
        raise HytaleAssetError("cell support must be an integer in [0, 15]")


__all__ = [
    "CompiledSemanticPalette",
    "ResolvedPrefabOverlay",
    "compile_role_opaque_cell_mask",
    "compile_local_semantic_palette",
    "merge_compiled_semantic_palettes",
    "resolve_static_prefab_overlay",
    "semantic_palette_remap",
]
