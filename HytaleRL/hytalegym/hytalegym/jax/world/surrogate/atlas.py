"""Host construction and JIT-safe lookup for surrogate world tiles."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.world.surrogate.capabilities import (
    SURROGATE_BASE_CAPABILITIES,
    SURROGATE_CAPABILITY_ENTITY_GEOMETRY,
    SURROGATE_CAPABILITY_ENTITY_PERCEPTION,
)
from hytalegym.jax.world.surrogate.terrain import (
    CAVE_SPAN_CAPACITY,
    SurrogateTerrainTile,
)
from hytalegym.jax.world.surrogate.types import (
    SurrogateAtlas,
)
from hytalegym.worldgen.region import (
    CAPTURE_BLOCKS_PER_AXIS,
    CHUNK_SIZE,
)
from hytalegym.worldgen.surrogate import (
    CompiledSemanticPalette,
    CompiledEntitySpawns,
    ResolvedPrefabOverlay,
    ResolvedStatefulPrefabOverlay,
    SurrogateWorldCapacity,
    entity_type_identity_words,
    merge_compiled_semantic_palettes,
    semantic_palette_remap,
)
from ._validate import (  # noqa: F401  (re-exported: callers unchanged)
    _absolute_structure_map,
    _effective_host_code,
    _inside_xz,
    _validate_entity_type_geometry,
    _validate_overlaps,
    _validate_runtime_shape,
    _validate_structure_state_slots,
    _validate_terrain_shapes,
    _world_block,
)
from ._lookup import (  # noqa: F401  (re-exported: callers unchanged)
    GENERATION_READY,
    _DOOR_NORMALS,
    _find_structure_cells,
    apply_surrogate_door_use,
    lookup_surrogate_blocks,
    lookup_surrogate_columns,
    lookup_surrogate_entities,
)
from ._runtime import (  # noqa: F401  (re-exported: callers unchanged)
    _assign_entity_runtime_slots,
    _assign_runtime_slots,
    _copy_entity_descriptors,
    _host_int,
    surrogate_runtime_from_atlas,
)

GENERATION_EMPTY = 0
Overlay = ResolvedPrefabOverlay | ResolvedStatefulPrefabOverlay | None


def surrogate_atlas_from_tiles(
    terrain: SurrogateTerrainTile,
    terrain_palette: CompiledSemanticPalette,
    overlays: Sequence[Overlay],
    *,
    entity_spawns: Sequence[CompiledEntitySpawns | None] | None = None,
    world_ids: Sequence[int],
    capacity: SurrogateWorldCapacity | None = None,
    tile_capacity: int | None = None,
) -> SurrogateAtlas:
    """Build an offline-complete atlas and reject every inconsistency."""

    if not isinstance(terrain, SurrogateTerrainTile):
        raise TypeError("terrain must be a SurrogateTerrainTile")
    if not isinstance(terrain_palette, CompiledSemanticPalette):
        raise TypeError("terrain_palette must be a CompiledSemanticPalette")
    layout = SurrogateWorldCapacity() if capacity is None else capacity
    if not isinstance(layout, SurrogateWorldCapacity):
        raise TypeError("capacity must be a SurrogateWorldCapacity")

    valid = _host_array(terrain.valid, np.bool_, "terrain valid")
    if valid.ndim != 1 or valid.size == 0:
        raise ValueError("terrain batch must be non-empty")
    tile_count = valid.size
    source_overlays = list(overlays)
    if len(source_overlays) != tile_count or any(
        value is not None
        and not isinstance(
            value,
            (ResolvedPrefabOverlay, ResolvedStatefulPrefabOverlay),
        )
        for value in source_overlays
    ):
        raise TypeError("overlays must match terrain tiles")
    source_entities = (
        [None] * tile_count if entity_spawns is None else list(entity_spawns)
    )
    if len(source_entities) != tile_count or any(
        value is not None and not isinstance(value, CompiledEntitySpawns)
        for value in source_entities
    ):
        raise TypeError("entity_spawns must match terrain tiles")
    identifiers = [_host_int(value, "world ID") for value in world_ids]
    if len(identifiers) != tile_count or any(
        value < 0 or value > np.iinfo(np.int32).max for value in identifiers
    ):
        raise ValueError("world_ids must be non-negative int32 values")
    tiles = _capacity(tile_capacity, tile_count, "tile")
    if not np.all(valid):
        raise ValueError("offline surrogate atlas rejects incomplete terrain")

    seeds = _host_array(terrain.seed_words, np.uint32, "terrain seed words")
    origins = _host_array(terrain.tile_min_xz, np.int32, "terrain origins")
    parameters = _host_array(
        terrain.parameters,
        np.float32,
        "terrain parameters",
    )
    biome_parameters = _host_array(
        terrain.biome_parameters,
        np.float32,
        "terrain biome parameters",
    )
    known = _host_array(
        terrain.column_known,
        np.bool_,
        "known terrain columns",
    )
    heights = _host_array(
        terrain.terrain_height,
        np.int16,
        "terrain heights",
    )
    surface = _host_array(
        terrain.surface_cell_code,
        np.uint16,
        "surface cell codes",
    )
    subsurface = _host_array(
        terrain.subsurface_cell_code,
        np.uint16,
        "subsurface cell codes",
    )
    biomes = _host_array(terrain.biome_code, np.uint8, "terrain biome codes")
    cave_mask = _host_array(
        terrain.cave_span_mask,
        np.bool_,
        "terrain cave span mask",
    )
    cave_minimum = _host_array(
        terrain.cave_min_y,
        np.int16,
        "terrain cave minimum",
    )
    cave_maximum = _host_array(
        terrain.cave_max_y,
        np.int16,
        "terrain cave maximum",
    )
    _validate_terrain_shapes(
        tile_count,
        seeds,
        origins,
        parameters,
        biome_parameters,
        known,
        heights,
        surface,
        subsurface,
        biomes,
        cave_mask,
        cave_minimum,
        cave_maximum,
    )
    if not np.all(known):
        raise ValueError("offline surrogate atlas rejects unknown columns")
    if np.any(origins % CHUNK_SIZE):
        raise ValueError("terrain tile origins must be chunk aligned")

    palettes = [terrain_palette] + [
        value.palette for value in source_overlays if value is not None
    ]
    palette = merge_compiled_semantic_palettes(
        palettes,
        capacity=layout,
    )
    terrain_remap = semantic_palette_remap(terrain_palette, palette)
    surface = _remap_cell_codes(surface, terrain_remap, terrain_palette)
    subsurface = _remap_cell_codes(
        subsurface,
        terrain_remap,
        terrain_palette,
    )

    tile_mask = np.zeros(tiles, dtype=np.bool_)
    world_id = np.full(tiles, -1, dtype=np.int32)
    core_min = np.zeros((tiles, 2), dtype=np.int32)
    seed_words = np.zeros((tiles, 2), dtype=np.uint32)
    generation_status = np.full(
        tiles,
        GENERATION_EMPTY,
        dtype=np.uint8,
    )
    capability_bits = np.zeros(tiles, dtype=np.uint32)
    terrain_parameters = np.zeros(
        (tiles, parameters.shape[1]),
        dtype=np.float32,
    )
    terrain_biome_parameters = np.zeros(
        (tiles,) + biome_parameters.shape[1:],
        dtype=np.float32,
    )
    column_known = np.zeros(
        (tiles, CAPTURE_BLOCKS_PER_AXIS, CAPTURE_BLOCKS_PER_AXIS),
        dtype=np.bool_,
    )
    terrain_height = np.zeros_like(column_known, dtype=np.int16)
    surface_code = np.zeros_like(column_known, dtype=np.uint16)
    subsurface_code = np.zeros_like(column_known, dtype=np.uint16)
    biome_code = np.full_like(column_known, 255, dtype=np.uint8)
    cave_shape = column_known.shape + (CAVE_SPAN_CAPACITY,)
    cave_span_mask = np.zeros(cave_shape, dtype=np.bool_)
    cave_min_y = np.zeros(cave_shape, dtype=np.int16)
    cave_max_y = np.zeros(cave_shape, dtype=np.int16)

    structure_shape = (tiles, layout.structure_cell_capacity)
    structure_mask = np.zeros(structure_shape, dtype=np.bool_)
    structure_key = np.full(
        structure_shape,
        np.iinfo(np.int32).max,
        dtype=np.int32,
    )
    structure_code = np.zeros(structure_shape, dtype=np.uint16)
    structure_state = np.full(structure_shape, -1, dtype=np.int32)

    stateful_shape = (tiles, layout.stateful_block_capacity)
    stateful_mask = np.zeros(stateful_shape, dtype=np.bool_)
    stateful_key = np.full(
        stateful_shape,
        np.iinfo(np.int32).max,
        dtype=np.int32,
    )
    stateful_identity = np.full(
        stateful_shape,
        np.iinfo(np.int32).max,
        dtype=np.int32,
    )
    stateful_runtime = np.full(stateful_shape, -1, dtype=np.int32)
    stateful_count = np.zeros(stateful_shape, dtype=np.uint8)
    stateful_initial = np.zeros(stateful_shape, dtype=np.uint8)
    stateful_yaw = np.zeros(stateful_shape, dtype=np.uint8)
    state_table_shape = stateful_shape + (layout.states_per_block,)
    stateful_variant = np.zeros(state_table_shape, dtype=np.uint16)
    transition_shape = state_table_shape + (2,)
    stateful_success = np.zeros(transition_shape, dtype=np.uint8)
    stateful_blocked = np.zeros(transition_shape, dtype=np.uint8)
    stateful_transition = np.zeros(transition_shape, dtype=np.bool_)
    stateful_partner_identity = np.full(
        stateful_shape,
        np.iinfo(np.int32).max,
        dtype=np.int32,
    )
    stateful_partner_runtime = np.full(stateful_shape, -1, dtype=np.int32)

    entity_shape = (tiles, layout.entity_capacity)
    entity_mask = np.zeros(entity_shape, dtype=np.bool_)
    entity_identity = np.zeros(entity_shape + (2,), dtype=np.uint32)
    entity_runtime = np.full(entity_shape, -1, dtype=np.int32)
    entity_position = np.zeros(entity_shape + (3,), dtype=np.float32)
    entity_rotation = np.zeros(entity_shape + (3,), dtype=np.float32)
    entity_velocity = np.zeros(entity_shape + (3,), dtype=np.float32)
    entity_kind = np.zeros(entity_shape, dtype=np.uint8)
    entity_type = np.zeros(entity_shape, dtype=np.uint16)
    entity_type_identity = np.zeros(entity_shape + (2,), dtype=np.uint32)
    entity_behavior = np.zeros(entity_shape, dtype=np.bool_)
    entity_geometry = np.zeros(entity_shape, dtype=np.bool_)
    entity_collidable = np.zeros(entity_shape, dtype=np.bool_)
    entity_blocks_los = np.zeros(entity_shape, dtype=np.bool_)
    entity_bounds = np.zeros(entity_shape + (6,), dtype=np.float32)
    entity_los_offset_supported = np.zeros(entity_shape, dtype=np.bool_)
    entity_los_offset = np.zeros(entity_shape + (3,), dtype=np.float32)
    entity_type_ids = tuple(
        sorted(
            {
                type_id
                for compiled in source_entities
                if compiled is not None
                for type_id in compiled.type_ids
            }
        )
    )
    if len(entity_type_ids) > np.iinfo(np.uint16).max:
        raise ValueError("entity type palette exceeds uint16")
    entity_type_index = {
        type_id: index for index, type_id in enumerate(entity_type_ids)
    }
    type_identities = {
        type_id: entity_type_identity_words(type_id) for type_id in entity_type_ids
    }
    if len(set(type_identities.values())) != len(type_identities):
        raise ValueError("entity type semantic identity collision")

    for index in range(tile_count):
        tile_mask[index] = True
        world_id[index] = identifiers[index]
        core_chunks = origins[index].astype(np.int64) // CHUNK_SIZE + 1
        if np.any(
            (core_chunks < np.iinfo(np.int32).min)
            | (core_chunks > np.iinfo(np.int32).max)
        ):
            raise ValueError("terrain core chunk coordinate exceeds int32")
        core_min[index] = core_chunks.astype(np.int32)
        seed_words[index] = seeds[index]
        generation_status[index] = GENERATION_READY
        terrain_parameters[index] = parameters[index]
        terrain_biome_parameters[index] = biome_parameters[index]
        column_known[index] = known[index]
        terrain_height[index] = heights[index]
        surface_code[index] = surface[index]
        subsurface_code[index] = subsurface[index]
        biome_code[index] = biomes[index]
        cave_span_mask[index] = cave_mask[index]
        cave_min_y[index] = cave_minimum[index]
        cave_max_y[index] = cave_maximum[index]
        _copy_entity_descriptors(
            index,
            origins[index],
            source_entities[index],
            entity_type_index,
            entity_mask,
            entity_identity,
            entity_position,
            entity_rotation,
            entity_velocity,
            entity_kind,
            entity_type,
            entity_type_identity,
            entity_geometry,
            entity_collidable,
            entity_blocks_los,
            entity_bounds,
            entity_los_offset_supported,
            entity_los_offset,
            layout.entity_capacity,
        )
        capability_bits[index] = SURROGATE_BASE_CAPABILITIES
        if np.all(entity_geometry[index, entity_mask[index]]):
            capability_bits[index] |= SURROGATE_CAPABILITY_ENTITY_GEOMETRY
        if np.all(
            entity_geometry[index, entity_mask[index]]
            & entity_los_offset_supported[index, entity_mask[index]]
        ):
            capability_bits[index] |= SURROGATE_CAPABILITY_ENTITY_PERCEPTION

        overlay = source_overlays[index]
        if overlay is None:
            continue
        if overlay.source.tile_min_xz != tuple(int(x) for x in origins[index]):
            raise ValueError("prefab overlay and terrain origins differ")
        active = np.flatnonzero(overlay.source.cell_mask)
        if active.size > layout.structure_cell_capacity:
            raise ValueError("structure overlay exceeds atlas capacity")
        remap = semantic_palette_remap(overlay.palette, palette)
        count = active.size
        structure_mask[index, :count] = True
        structure_key[index, :count] = overlay.source.block_key[active]
        structure_code[index, :count] = _remap_cell_codes(
            overlay.cell_code[active],
            remap,
            overlay.palette,
        )
        structure_state[index, :count] = overlay.state_slot[active]

        if not isinstance(overlay, ResolvedStatefulPrefabOverlay):
            continue
        local_slots = np.flatnonzero(overlay.stateful.block_mask)
        if local_slots.size > layout.stateful_block_capacity:
            raise ValueError("stateful definitions exceed atlas capacity")
        if not np.array_equal(
            local_slots,
            np.arange(local_slots.size, dtype=local_slots.dtype),
        ):
            raise ValueError("stateful definitions must form a compact prefix")
        for slot in local_slots:
            slot = int(slot)
            if slot >= layout.stateful_block_capacity:
                raise ValueError("stateful definition slot exceeds atlas capacity")
            states = int(overlay.stateful.state_count[slot])
            if not 1 <= states <= layout.states_per_block:
                raise ValueError("stateful definition has an invalid state count")
            stateful_mask[index, slot] = True
            stateful_key[index, slot] = overlay.stateful.block_key[slot]
            stateful_identity[index, slot] = overlay.stateful.identity_key[slot]
            stateful_count[index, slot] = states
            stateful_initial[index, slot] = overlay.stateful.initial_state[slot]
            stateful_yaw[index, slot] = overlay.stateful.yaw[slot]
            stateful_variant[index, slot, :states] = remap[
                overlay.stateful.variant_cell_code[slot, :states]
            ]
            stateful_success[index, slot, :states] = overlay.stateful.success_target[
                slot, :states
            ]
            stateful_blocked[index, slot, :states] = overlay.stateful.blocked_target[
                slot, :states
            ]
            stateful_transition[index, slot, :states] = (
                overlay.stateful.transition_mask[slot, :states]
            )
            stateful_partner_identity[index, slot] = (
                overlay.stateful.partner_identity_key[slot]
            )
        _validate_structure_state_slots(
            structure_mask[index],
            structure_key[index],
            structure_state[index],
            stateful_mask[index],
            stateful_key[index],
        )

    _validate_entity_type_geometry(
        tile_count,
        entity_mask,
        entity_type,
        entity_type_identity,
        entity_geometry,
        entity_collidable,
        entity_blocks_los,
        entity_bounds,
        entity_los_offset_supported,
        entity_los_offset,
    )
    _assign_runtime_slots(
        tile_count,
        identifiers,
        origins,
        stateful_mask,
        stateful_key,
        stateful_identity,
        stateful_partner_identity,
        stateful_runtime,
        stateful_partner_runtime,
        stateful_count,
        stateful_initial,
        stateful_yaw,
        stateful_variant,
        stateful_success,
        stateful_blocked,
        stateful_transition,
        layout.stateful_block_capacity,
    )
    _assign_entity_runtime_slots(
        tile_count,
        identifiers,
        entity_mask,
        entity_identity,
        entity_runtime,
        entity_position,
        entity_rotation,
        entity_velocity,
        entity_kind,
        entity_type,
        entity_type_identity,
        entity_geometry,
        entity_collidable,
        entity_blocks_los,
        entity_bounds,
        entity_los_offset_supported,
        entity_los_offset,
        layout.entity_capacity,
    )
    _validate_overlaps(
        tile_count,
        identifiers,
        origins,
        seeds,
        column_known,
        terrain_height,
        surface_code,
        subsurface_code,
        biome_code,
        cave_span_mask,
        cave_min_y,
        cave_max_y,
        structure_mask,
        structure_key,
        structure_code,
        structure_state,
        stateful_mask,
        stateful_key,
        stateful_runtime,
        stateful_partner_runtime,
        entity_mask,
        entity_identity,
        entity_runtime,
        entity_position,
        entity_rotation,
        entity_velocity,
        entity_kind,
        entity_type,
        entity_type_identity,
        entity_geometry,
        entity_collidable,
        entity_blocks_los,
        entity_bounds,
        entity_los_offset_supported,
        entity_los_offset,
    )

    return SurrogateAtlas(
        *(
            jnp.asarray(value)
            for value in (
                tile_mask,
                world_id,
                core_min,
                seed_words,
                generation_status,
                capability_bits,
                terrain_parameters,
                terrain_biome_parameters,
                column_known,
                terrain_height,
                surface_code,
                subsurface_code,
                biome_code,
                cave_span_mask,
                cave_min_y,
                cave_max_y,
                structure_mask,
                structure_key,
                structure_code,
                structure_state,
                stateful_mask,
                stateful_key,
                stateful_identity,
                stateful_runtime,
                stateful_count,
                stateful_initial,
                stateful_yaw,
                stateful_variant,
                stateful_success,
                stateful_blocked,
                stateful_transition,
                stateful_partner_runtime,
                entity_mask,
                entity_identity,
                entity_runtime,
                entity_position,
                entity_rotation,
                entity_velocity,
                entity_kind,
                entity_type,
                entity_type_identity,
                entity_behavior,
                entity_geometry,
                entity_collidable,
                entity_blocks_los,
                entity_bounds,
                entity_los_offset_supported,
                entity_los_offset,
                np.asarray(len(entity_type_ids), dtype=np.int32),
                np.asarray(palette.cell_count, dtype=np.int32),
                palette.cell_flags,
                palette.cell_shape_index,
                palette.cell_fluid_level,
                palette.cell_support,
                palette.cell_block_damage,
                palette.cell_fluid_damage,
                palette.cell_movement,
                palette.cell_fluid_movement,
                np.asarray(palette.shape_count, dtype=np.int32),
                palette.shape_boxes,
                palette.shape_box_mask,
            )
        )
    )


def _remap_cell_codes(
    values: np.ndarray,
    remap: np.ndarray,
    palette: CompiledSemanticPalette,
) -> np.ndarray:
    codes = np.asarray(values)
    if np.any(codes.astype(np.uint32) >= palette.cell_count):
        raise ValueError("cell code exceeds its source semantic palette")
    return remap[codes].astype(np.uint16, copy=False)


def _host_array(value: jax.Array, dtype: np.dtype, label: str) -> np.ndarray:
    result = np.asarray(jax.device_get(value))
    if result.dtype != np.dtype(dtype):
        raise TypeError(f"{label} must use dtype {np.dtype(dtype)}")
    return result.copy()


def _capacity(requested: int | None, required: int, label: str) -> int:
    result = (
        required if requested is None else _host_int(requested, f"{label} capacity")
    )
    if result < required or result <= 0:
        raise ValueError(f"{label} capacity does not cover supplied data")
    return result


__all__ = [
    "GENERATION_EMPTY",
    "GENERATION_READY",
    "apply_surrogate_door_use",
    "lookup_surrogate_blocks",
    "lookup_surrogate_columns",
    "lookup_surrogate_entities",
    "surrogate_atlas_from_tiles",
    "surrogate_runtime_from_atlas",
]
