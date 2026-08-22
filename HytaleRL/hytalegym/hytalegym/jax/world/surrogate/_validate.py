"""Leaf helpers extracted verbatim from atlas.py."""

from collections.abc import Sequence
import numpy as np
from hytalegym.jax.world.surrogate.terrain import BIOME_CAPACITY, CAVE_SPAN_CAPACITY
from hytalegym.jax.world.surrogate.types import SurrogateAtlas, SurrogateRuntimeState
from hytalegym.worldgen.region import CAPTURE_BLOCKS_PER_AXIS, MIN_Y
from hytalegym.worldgen.surrogate import TERRAIN_PARAMETER_COUNT


def _validate_entity_type_geometry(
    tile_count: int,
    mask: np.ndarray,
    type_code: np.ndarray,
    type_identity: np.ndarray,
    geometry_supported: np.ndarray,
    collidable: np.ndarray,
    blocks_los: np.ndarray,
    local_bounds: np.ndarray,
    los_offset_supported: np.ndarray,
    los_offset: np.ndarray,
) -> None:
    registry: dict[int, tuple[object, ...]] = {}
    identities: dict[int, bytes] = {}
    for tile in range(tile_count):
        for definition in np.flatnonzero(mask[tile]):
            code = int(type_code[tile, definition])
            identity = type_identity[tile, definition].tobytes()
            supported = bool(geometry_supported[tile, definition])
            bounds = local_bounds[tile, definition]
            physical = bool(
                collidable[tile, definition] or blocks_los[tile, definition]
            )
            perception = bool(los_offset_supported[tile, definition])
            if (
                not np.all(np.isfinite(bounds))
                or not np.all(np.isfinite(los_offset[tile, definition]))
                or (
                    not supported
                    and (
                        physical
                        or perception
                        or np.any(bounds != 0.0)
                        or np.any(los_offset[tile, definition] != 0.0)
                    )
                )
                or (
                    supported
                    and (physical or perception)
                    and np.any(bounds[:3] >= bounds[3:])
                )
                or (not perception and np.any(los_offset[tile, definition] != 0.0))
            ):
                raise ValueError("entity type geometry is invalid")
            previous_identity = identities.setdefault(code, identity)
            if previous_identity != identity:
                raise ValueError("entity type code maps to multiple identities")
            signature = (
                supported,
                bool(collidable[tile, definition]),
                bool(blocks_los[tile, definition]),
                local_bounds[tile, definition].tobytes(),
                perception,
                los_offset[tile, definition].tobytes(),
            )
            previous = registry.setdefault(code, signature)
            if previous != signature:
                raise ValueError("entity type geometry differs across atlas tiles")


def _validate_overlaps(
    tile_count: int,
    world_ids: Sequence[int],
    origins: np.ndarray,
    seeds: np.ndarray,
    known: np.ndarray,
    heights: np.ndarray,
    surface: np.ndarray,
    subsurface: np.ndarray,
    biomes: np.ndarray,
    cave_mask: np.ndarray,
    cave_minimum: np.ndarray,
    cave_maximum: np.ndarray,
    structure_mask: np.ndarray,
    structure_key: np.ndarray,
    structure_code: np.ndarray,
    structure_state: np.ndarray,
    stateful_mask: np.ndarray,
    stateful_key: np.ndarray,
    stateful_runtime: np.ndarray,
    stateful_partner_runtime: np.ndarray,
    entity_mask: np.ndarray,
    entity_identity: np.ndarray,
    entity_runtime: np.ndarray,
    entity_position: np.ndarray,
    entity_rotation: np.ndarray,
    entity_velocity: np.ndarray,
    entity_kind: np.ndarray,
    entity_type: np.ndarray,
    entity_type_identity: np.ndarray,
    entity_geometry: np.ndarray,
    entity_collidable: np.ndarray,
    entity_blocks_los: np.ndarray,
    entity_bounds: np.ndarray,
    entity_los_offset_supported: np.ndarray,
    entity_los_offset: np.ndarray,
) -> None:
    structures = [
        _absolute_structure_map(
            origins[index],
            structure_mask[index],
            structure_key[index],
            structure_code[index],
            structure_state[index],
        )
        for index in range(tile_count)
    ]
    stateful = [
        {
            _world_block(origins[index], int(stateful_key[index, slot])): (
                int(stateful_runtime[index, slot]),
                int(stateful_partner_runtime[index, slot]),
            )
            for slot in np.flatnonzero(stateful_mask[index])
        }
        for index in range(tile_count)
    ]
    entities = [
        {
            tuple(int(value) for value in entity_identity[index, slot]): (
                int(entity_runtime[index, slot]),
                entity_position[index, slot].tobytes(),
                entity_rotation[index, slot].tobytes(),
                entity_velocity[index, slot].tobytes(),
                int(entity_kind[index, slot]),
                int(entity_type[index, slot]),
                entity_type_identity[index, slot].tobytes(),
                bool(entity_geometry[index, slot]),
                bool(entity_collidable[index, slot]),
                bool(entity_blocks_los[index, slot]),
                entity_bounds[index, slot].tobytes(),
                bool(entity_los_offset_supported[index, slot]),
                entity_los_offset[index, slot].tobytes(),
            )
            for slot in np.flatnonzero(entity_mask[index])
        }
        for index in range(tile_count)
    ]
    for left in range(tile_count):
        for right in range(left + 1, tile_count):
            if world_ids[left] != world_ids[right]:
                continue
            minimum = np.maximum(origins[left], origins[right])
            maximum = np.minimum(
                origins[left] + CAPTURE_BLOCKS_PER_AXIS,
                origins[right] + CAPTURE_BLOCKS_PER_AXIS,
            )
            if np.any(minimum >= maximum):
                continue
            if not np.array_equal(seeds[left], seeds[right]):
                raise ValueError("overlapping tiles in one world use different seeds")
            left_slice = tuple(
                slice(
                    int(minimum[axis] - origins[left, axis]),
                    int(maximum[axis] - origins[left, axis]),
                )
                for axis in range(2)
            )
            right_slice = tuple(
                slice(
                    int(minimum[axis] - origins[right, axis]),
                    int(maximum[axis] - origins[right, axis]),
                )
                for axis in range(2)
            )
            for left_value, right_value in (
                (known[left][left_slice], known[right][right_slice]),
                (heights[left][left_slice], heights[right][right_slice]),
                (surface[left][left_slice], surface[right][right_slice]),
                (
                    subsurface[left][left_slice],
                    subsurface[right][right_slice],
                ),
                (biomes[left][left_slice], biomes[right][right_slice]),
                (
                    cave_mask[left][left_slice],
                    cave_mask[right][right_slice],
                ),
                (
                    cave_minimum[left][left_slice],
                    cave_minimum[right][right_slice],
                ),
                (
                    cave_maximum[left][left_slice],
                    cave_maximum[right][right_slice],
                ),
            ):
                if not np.array_equal(left_value, right_value):
                    raise ValueError("overlapping surrogate terrain disagrees")

            positions = set(structures[left]) | set(structures[right])
            for position in positions:
                if not _inside_xz(position, minimum, maximum):
                    continue
                left_code = _effective_host_code(
                    left,
                    position,
                    origins,
                    heights,
                    surface,
                    subsurface,
                    cave_mask,
                    cave_minimum,
                    cave_maximum,
                    structures,
                )
                right_code = _effective_host_code(
                    right,
                    position,
                    origins,
                    heights,
                    surface,
                    subsurface,
                    cave_mask,
                    cave_minimum,
                    cave_maximum,
                    structures,
                )
                if left_code != right_code:
                    raise ValueError(
                        "overlapping surrogate structure geometry disagrees"
                    )
            state_positions = set(stateful[left]) | set(stateful[right])
            for position in state_positions:
                if _inside_xz(position, minimum, maximum) and (
                    stateful[left].get(position) != stateful[right].get(position)
                ):
                    raise ValueError(
                        "overlapping surrogate stateful identities disagree"
                    )
            entity_ids = set(entities[left]) | set(entities[right])
            for identity in entity_ids:
                left_definition = entities[left].get(identity)
                right_definition = entities[right].get(identity)
                definition = left_definition or right_definition
                assert definition is not None
                position = np.frombuffer(
                    definition[1],
                    dtype=np.float32,
                )
                if (
                    minimum[0] <= position[0] < maximum[0]
                    and minimum[1] <= position[2] < maximum[1]
                    and left_definition != right_definition
                ):
                    raise ValueError(
                        "overlapping surrogate entity definitions disagree"
                    )


def _validate_structure_state_slots(
    structure_mask: np.ndarray,
    structure_key: np.ndarray,
    structure_state: np.ndarray,
    stateful_mask: np.ndarray,
    stateful_key: np.ndarray,
) -> None:
    for index in np.flatnonzero(structure_mask):
        slot = int(structure_state[index])
        if slot < 0:
            continue
        if (
            slot >= stateful_mask.size
            or not stateful_mask[slot]
            or structure_key[index] != stateful_key[slot]
        ):
            raise ValueError("structure state slot does not match its definition")


def _absolute_structure_map(
    origin: np.ndarray,
    mask: np.ndarray,
    keys: np.ndarray,
    codes: np.ndarray,
    states: np.ndarray,
) -> dict[tuple[int, int, int], tuple[int, int]]:
    return {
        _world_block(origin, int(keys[index])): (
            int(codes[index]),
            int(states[index]),
        )
        for index in np.flatnonzero(mask)
    }


def _effective_host_code(
    tile: int,
    position: tuple[int, int, int],
    origins: np.ndarray,
    heights: np.ndarray,
    surface: np.ndarray,
    subsurface: np.ndarray,
    cave_mask: np.ndarray,
    cave_minimum: np.ndarray,
    cave_maximum: np.ndarray,
    structures: list[dict[tuple[int, int, int], tuple[int, int]]],
) -> int:
    override = structures[tile].get(position)
    if override is not None:
        return override[0]
    x = position[0] - int(origins[tile, 0])
    z = position[2] - int(origins[tile, 1])
    height = int(heights[tile, x, z])
    carved = np.any(
        cave_mask[tile, x, z]
        & (position[1] >= cave_minimum[tile, x, z])
        & (position[1] < cave_maximum[tile, x, z])
    )
    if carved:
        return 0
    if position[1] == height:
        return int(surface[tile, x, z])
    if position[1] < height:
        return int(subsurface[tile, x, z])
    return 0


def _inside_xz(
    position: tuple[int, int, int],
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> bool:
    return int(minimum[0]) <= position[0] < int(maximum[0]) and int(
        minimum[1]
    ) <= position[2] < int(maximum[1])


def _world_block(origin: np.ndarray, key: int) -> tuple[int, int, int]:
    area = CAPTURE_BLOCKS_PER_AXIS**2
    y, remainder = divmod(key, area)
    z, x = divmod(remainder, CAPTURE_BLOCKS_PER_AXIS)
    return int(origin[0]) + x, MIN_Y + y, int(origin[1]) + z


def _validate_terrain_shapes(
    count: int,
    seeds: np.ndarray,
    origins: np.ndarray,
    parameters: np.ndarray,
    biome_parameters: np.ndarray,
    known: np.ndarray,
    heights: np.ndarray,
    surface: np.ndarray,
    subsurface: np.ndarray,
    biomes: np.ndarray,
    cave_mask: np.ndarray,
    cave_minimum: np.ndarray,
    cave_maximum: np.ndarray,
) -> None:
    columns = (
        count,
        CAPTURE_BLOCKS_PER_AXIS,
        CAPTURE_BLOCKS_PER_AXIS,
    )
    expected = {
        "seed words": (seeds, (count, 2)),
        "origins": (origins, (count, 2)),
        "parameters": (
            parameters,
            (count, TERRAIN_PARAMETER_COUNT),
        ),
        "biome parameters": (
            biome_parameters,
            (count, BIOME_CAPACITY, TERRAIN_PARAMETER_COUNT),
        ),
        "known columns": (known, columns),
        "heights": (heights, columns),
        "surface codes": (surface, columns),
        "subsurface codes": (subsurface, columns),
        "biome codes": (biomes, columns),
        "cave span mask": (
            cave_mask,
            columns + (CAVE_SPAN_CAPACITY,),
        ),
        "cave minimum": (
            cave_minimum,
            columns + (CAVE_SPAN_CAPACITY,),
        ),
        "cave maximum": (
            cave_maximum,
            columns + (CAVE_SPAN_CAPACITY,),
        ),
    }
    for label, (value, shape) in expected.items():
        if value.shape != shape:
            raise ValueError(f"terrain {label} must have shape {shape}")
    if not np.all(np.isfinite(parameters)) or not np.all(np.isfinite(biome_parameters)):
        raise ValueError("terrain parameters must be finite")
    active_span_valid = (
        (cave_minimum > MIN_Y)
        & (cave_maximum > cave_minimum)
        & (cave_maximum <= heights[..., None] - 3)
    )
    if np.any(cave_mask & ~active_span_valid) or np.any(
        ~cave_mask & ((cave_minimum != 0) | (cave_maximum != 0))
    ):
        raise ValueError("terrain cave spans are malformed")
    if CAVE_SPAN_CAPACITY > 1 and np.any(
        cave_mask[..., 0]
        & cave_mask[..., 1]
        & (cave_minimum[..., 1] < cave_maximum[..., 0] + 2)
    ):
        raise ValueError("terrain cave spans overlap")


def _validate_runtime_shape(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    batch: int,
) -> None:
    if not isinstance(atlas, SurrogateAtlas):
        raise TypeError("atlas must be a SurrogateAtlas")
    if not isinstance(runtime, SurrogateRuntimeState):
        raise TypeError("runtime must be a SurrogateRuntimeState")
    state_capacity = atlas.stateful_mask.shape[1]
    entity_capacity = atlas.entity_mask.shape[1]
    if (
        runtime.environment_world_id.shape != (batch,)
        or runtime.stateful_state.shape != (batch, state_capacity)
        or runtime.stateful_initialized.shape != (batch, state_capacity)
        or runtime.entity_active.shape != (batch, entity_capacity)
        or runtime.entity_initialized.shape != (batch, entity_capacity)
        or runtime.entity_identity_words.shape != (batch, entity_capacity, 2)
        or runtime.entity_position.shape != (batch, entity_capacity, 3)
        or runtime.entity_rotation.shape != (batch, entity_capacity, 3)
        or runtime.entity_velocity.shape != (batch, entity_capacity, 3)
        or runtime.entity_kind.shape != (batch, entity_capacity)
        or runtime.entity_type_code.shape != (batch, entity_capacity)
        or runtime.entity_type_identity_words.shape != (batch, entity_capacity, 2)
        or runtime.entity_behavior_supported.shape != (batch, entity_capacity)
        or runtime.entity_geometry_supported.shape != (batch, entity_capacity)
        or runtime.entity_collidable.shape != (batch, entity_capacity)
        or runtime.entity_blocks_los.shape != (batch, entity_capacity)
        or runtime.entity_local_bounds.shape != (batch, entity_capacity, 6)
        or runtime.entity_los_offset_supported.shape != (batch, entity_capacity)
        or runtime.entity_los_offset.shape != (batch, entity_capacity, 3)
        or runtime.capability_bits.shape != (batch,)
        or runtime.failure_bits.shape != (batch,)
        or runtime.unsupported_mechanics.shape != (batch,)
    ):
        raise ValueError("surrogate runtime shapes do not match atlas and batch")
