"""Leaf helpers extracted verbatim from atlas.py."""

from collections.abc import Sequence
import operator
import jax
import jax.numpy as jnp
import numpy as np
from hytalegym.jax.world.surrogate.capabilities import SURROGATE_FAILURE_WORLD_UNAVAILABLE
from hytalegym.jax.world.surrogate.types import SurrogateAtlas, SurrogateRuntimeState
from hytalegym.worldgen.surrogate import CompiledEntitySpawns, entity_type_identity_words
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


def surrogate_runtime_from_atlas(
    atlas: SurrogateAtlas,
    environment_world_id: Sequence[int],
) -> SurrogateRuntimeState:
    """Construct reset state on the host with world-absolute door identity."""

    if not isinstance(atlas, SurrogateAtlas):
        raise TypeError("atlas must be a SurrogateAtlas")
    world_ids = np.asarray(
        [_host_int(value, "environment world ID") for value in environment_world_id],
        dtype=np.int32,
    )
    if world_ids.ndim != 1 or world_ids.size == 0 or np.any(world_ids < 0):
        raise ValueError("environment_world_id must be non-negative and non-empty")
    tile_mask = np.asarray(jax.device_get(atlas.tile_mask), dtype=np.bool_)
    atlas_world = np.asarray(jax.device_get(atlas.world_id), dtype=np.int32)
    atlas_capabilities = np.asarray(
        jax.device_get(atlas.capability_bits),
        dtype=np.uint32,
    )
    stateful_mask = np.asarray(
        jax.device_get(atlas.stateful_mask),
        dtype=np.bool_,
    )
    runtime_slot = np.asarray(
        jax.device_get(atlas.stateful_runtime_slot),
        dtype=np.int32,
    )
    initial = np.asarray(
        jax.device_get(atlas.stateful_initial_state),
        dtype=np.uint8,
    )
    entity_mask = np.asarray(jax.device_get(atlas.entity_mask), dtype=np.bool_)
    entity_runtime_slot = np.asarray(
        jax.device_get(atlas.entity_runtime_slot),
        dtype=np.int32,
    )
    atlas_entity_identity = np.asarray(
        jax.device_get(atlas.entity_identity_words),
        dtype=np.uint32,
    )
    entity_initial_position = np.asarray(
        jax.device_get(atlas.entity_initial_position),
        dtype=np.float32,
    )
    entity_initial_rotation = np.asarray(
        jax.device_get(atlas.entity_initial_rotation),
        dtype=np.float32,
    )
    entity_initial_velocity = np.asarray(
        jax.device_get(atlas.entity_initial_velocity),
        dtype=np.float32,
    )
    atlas_entity_kind = np.asarray(
        jax.device_get(atlas.entity_kind),
        dtype=np.uint8,
    )
    atlas_entity_type = np.asarray(
        jax.device_get(atlas.entity_type_code),
        dtype=np.uint16,
    )
    atlas_entity_type_identity = np.asarray(
        jax.device_get(atlas.entity_type_identity_words),
        dtype=np.uint32,
    )
    atlas_entity_behavior = np.asarray(
        jax.device_get(atlas.entity_behavior_supported),
        dtype=np.bool_,
    )
    atlas_entity_geometry = np.asarray(
        jax.device_get(atlas.entity_geometry_supported),
        dtype=np.bool_,
    )
    atlas_entity_collidable = np.asarray(
        jax.device_get(atlas.entity_collidable),
        dtype=np.bool_,
    )
    atlas_entity_blocks_los = np.asarray(
        jax.device_get(atlas.entity_blocks_los),
        dtype=np.bool_,
    )
    atlas_entity_bounds = np.asarray(
        jax.device_get(atlas.entity_local_bounds),
        dtype=np.float32,
    )
    atlas_entity_los_offset_supported = np.asarray(
        jax.device_get(atlas.entity_los_offset_supported),
        dtype=np.bool_,
    )
    atlas_entity_los_offset = np.asarray(
        jax.device_get(atlas.entity_los_offset),
        dtype=np.float32,
    )
    state_capacity = stateful_mask.shape[1]
    states = np.zeros((world_ids.size, state_capacity), dtype=np.uint8)
    initialized = np.zeros_like(states, dtype=np.bool_)
    entity_capacity = entity_mask.shape[1]
    entity_active = np.zeros(
        (world_ids.size, entity_capacity),
        dtype=np.bool_,
    )
    entity_initialized = np.zeros_like(entity_active)
    entity_identity = np.zeros(
        (world_ids.size, entity_capacity, 2),
        dtype=np.uint32,
    )
    entity_position = np.zeros(
        (world_ids.size, entity_capacity, 3),
        dtype=np.float32,
    )
    entity_rotation = np.zeros_like(entity_position)
    entity_velocity = np.zeros_like(entity_position)
    entity_kind = np.zeros(
        (world_ids.size, entity_capacity),
        dtype=np.uint8,
    )
    entity_type = np.zeros(
        (world_ids.size, entity_capacity),
        dtype=np.uint16,
    )
    entity_type_identity = np.zeros(
        (world_ids.size, entity_capacity, 2),
        dtype=np.uint32,
    )
    entity_behavior = np.zeros_like(entity_active)
    entity_geometry = np.zeros_like(entity_active)
    entity_collidable = np.zeros_like(entity_active)
    entity_blocks_los = np.zeros_like(entity_active)
    entity_bounds = np.zeros(
        (world_ids.size, entity_capacity, 6),
        dtype=np.float32,
    )
    entity_los_offset_supported = np.zeros_like(entity_active)
    entity_los_offset = np.zeros(
        (world_ids.size, entity_capacity, 3),
        dtype=np.float32,
    )
    capabilities = np.zeros(world_ids.size, dtype=np.uint32)
    failure_bits = np.full(
        world_ids.size,
        SURROGATE_FAILURE_WORLD_UNAVAILABLE,
        dtype=np.uint32,
    )
    for environment, world_id in enumerate(world_ids):
        compatible = np.flatnonzero(tile_mask & (atlas_world == world_id))
        if compatible.size:
            capabilities[environment] = np.bitwise_and.reduce(
                atlas_capabilities[compatible]
            )
            failure_bits[environment] = 0
        for tile in compatible:
            for local_slot in np.flatnonzero(stateful_mask[tile]):
                slot = int(runtime_slot[tile, local_slot])
                if not 0 <= slot < state_capacity:
                    raise ValueError("atlas contains an invalid runtime state slot")
                value = initial[tile, local_slot]
                if (
                    initialized[environment, slot]
                    and states[environment, slot] != value
                ):
                    raise ValueError("overlapping door initial states disagree")
                states[environment, slot] = value
                initialized[environment, slot] = True
            for definition in np.flatnonzero(entity_mask[tile]):
                slot = int(entity_runtime_slot[tile, definition])
                if not 0 <= slot < entity_capacity:
                    raise ValueError("atlas contains an invalid entity runtime slot")
                signature = (
                    atlas_entity_identity[tile, definition],
                    entity_initial_position[tile, definition],
                    entity_initial_rotation[tile, definition],
                    entity_initial_velocity[tile, definition],
                    atlas_entity_kind[tile, definition],
                    atlas_entity_type[tile, definition],
                    atlas_entity_type_identity[tile, definition],
                    atlas_entity_behavior[tile, definition],
                    atlas_entity_geometry[tile, definition],
                    atlas_entity_collidable[tile, definition],
                    atlas_entity_blocks_los[tile, definition],
                    atlas_entity_bounds[tile, definition],
                    atlas_entity_los_offset_supported[tile, definition],
                    atlas_entity_los_offset[tile, definition],
                )
                if entity_initialized[environment, slot]:
                    current = (
                        entity_identity[environment, slot],
                        entity_position[environment, slot],
                        entity_rotation[environment, slot],
                        entity_velocity[environment, slot],
                        entity_kind[environment, slot],
                        entity_type[environment, slot],
                        entity_type_identity[environment, slot],
                        entity_behavior[environment, slot],
                        entity_geometry[environment, slot],
                        entity_collidable[environment, slot],
                        entity_blocks_los[environment, slot],
                        entity_bounds[environment, slot],
                        entity_los_offset_supported[environment, slot],
                        entity_los_offset[environment, slot],
                    )
                    if not all(
                        np.array_equal(left, right)
                        for left, right in zip(current, signature, strict=True)
                    ):
                        raise ValueError("overlapping entity definitions disagree")
                (
                    entity_identity[environment, slot],
                    entity_position[environment, slot],
                    entity_rotation[environment, slot],
                    entity_velocity[environment, slot],
                    entity_kind[environment, slot],
                    entity_type[environment, slot],
                    entity_type_identity[environment, slot],
                    entity_behavior[environment, slot],
                    entity_geometry[environment, slot],
                    entity_collidable[environment, slot],
                    entity_blocks_los[environment, slot],
                    entity_bounds[environment, slot],
                    entity_los_offset_supported[environment, slot],
                    entity_los_offset[environment, slot],
                ) = signature
                entity_active[environment, slot] = True
                entity_initialized[environment, slot] = True
    return SurrogateRuntimeState(
        environment_world_id=jnp.asarray(world_ids),
        stateful_state=jnp.asarray(states),
        stateful_initialized=jnp.asarray(initialized),
        entity_active=jnp.asarray(entity_active),
        entity_initialized=jnp.asarray(entity_initialized),
        entity_identity_words=jnp.asarray(entity_identity),
        entity_position=jnp.asarray(entity_position),
        entity_rotation=jnp.asarray(entity_rotation),
        entity_velocity=jnp.asarray(entity_velocity),
        entity_kind=jnp.asarray(entity_kind),
        entity_type_code=jnp.asarray(entity_type),
        entity_type_identity_words=jnp.asarray(entity_type_identity),
        entity_behavior_supported=jnp.asarray(entity_behavior),
        entity_geometry_supported=jnp.asarray(entity_geometry),
        entity_collidable=jnp.asarray(entity_collidable),
        entity_blocks_los=jnp.asarray(entity_blocks_los),
        entity_local_bounds=jnp.asarray(entity_bounds),
        entity_los_offset_supported=jnp.asarray(entity_los_offset_supported),
        entity_los_offset=jnp.asarray(entity_los_offset),
        capability_bits=jnp.asarray(capabilities),
        failure_bits=jnp.asarray(failure_bits),
        unsupported_mechanics=jnp.asarray(failure_bits != 0),
    )


def _assign_runtime_slots(
    tile_count: int,
    world_ids: Sequence[int],
    origins: np.ndarray,
    mask: np.ndarray,
    keys: np.ndarray,
    identity_keys: np.ndarray,
    partner_identity_keys: np.ndarray,
    runtime_slot: np.ndarray,
    partner_runtime_slot: np.ndarray,
    state_count: np.ndarray,
    initial: np.ndarray,
    yaw: np.ndarray,
    variants: np.ndarray,
    success: np.ndarray,
    blocked: np.ndarray,
    transitions: np.ndarray,
    capacity: int,
) -> None:
    definitions: dict[
        int,
        dict[tuple[int, int, int], tuple[object, ...]],
    ] = {}
    physical_definitions: dict[
        int,
        dict[tuple[int, int, int], tuple[object, ...]],
    ] = {}
    for tile in range(tile_count):
        registry = definitions.setdefault(world_ids[tile], {})
        physical_registry = physical_definitions.setdefault(world_ids[tile], {})
        for local_slot in np.flatnonzero(mask[tile]):
            local_slot = int(local_slot)
            physical_block = _world_block(
                origins[tile],
                int(keys[tile, local_slot]),
            )
            identity_block = _world_block(
                origins[tile],
                int(identity_keys[tile, local_slot]),
            )
            raw_partner = int(partner_identity_keys[tile, local_slot])
            partner_block = (
                None
                if raw_partner == np.iinfo(np.int32).max
                else _world_block(origins[tile], raw_partner)
            )
            count = int(state_count[tile, local_slot])
            transition_signature: tuple[object, ...] = (
                count,
                int(initial[tile, local_slot]),
                int(yaw[tile, local_slot]),
                partner_block,
                success[tile, local_slot, :count].tobytes(),
                blocked[tile, local_slot, :count].tobytes(),
                transitions[tile, local_slot, :count].tobytes(),
            )
            previous = registry.setdefault(
                identity_block,
                transition_signature,
            )
            if previous != transition_signature:
                raise ValueError("stateful identity transition definitions disagree")
            physical_signature = (
                identity_block,
                transition_signature,
                variants[tile, local_slot, :count].tobytes(),
            )
            previous_physical = physical_registry.setdefault(
                physical_block,
                physical_signature,
            )
            if previous_physical != physical_signature:
                raise ValueError("overlapping stateful cell definitions disagree")
            if len(registry) > capacity:
                raise ValueError(
                    "world-absolute stateful blocks exceed runtime capacity"
                )
    runtime_indices = {
        world_id: {position: slot for slot, position in enumerate(sorted(registry))}
        for world_id, registry in definitions.items()
    }
    for tile in range(tile_count):
        registry = runtime_indices[world_ids[tile]]
        for local_slot in np.flatnonzero(mask[tile]):
            identity = _world_block(
                origins[tile],
                int(identity_keys[tile, local_slot]),
            )
            runtime_slot[tile, local_slot] = registry[identity]
            raw_partner = int(partner_identity_keys[tile, local_slot])
            if raw_partner == np.iinfo(np.int32).max:
                continue
            partner = _world_block(origins[tile], raw_partner)
            try:
                partner_runtime_slot[tile, local_slot] = registry[partner]
            except KeyError as error:
                raise ValueError("double-door partner identity is absent") from error


def _copy_entity_descriptors(
    tile: int,
    origin: np.ndarray,
    compiled: CompiledEntitySpawns | None,
    type_index: dict[str, int],
    mask: np.ndarray,
    identity: np.ndarray,
    position: np.ndarray,
    rotation: np.ndarray,
    velocity: np.ndarray,
    kind: np.ndarray,
    type_code: np.ndarray,
    type_identity: np.ndarray,
    geometry_supported: np.ndarray,
    collidable: np.ndarray,
    blocks_los: np.ndarray,
    local_bounds: np.ndarray,
    los_offset_supported: np.ndarray,
    los_offset: np.ndarray,
    capacity: int,
) -> None:
    if compiled is None:
        return
    if compiled.tile_min_xz != tuple(int(value) for value in origin):
        raise ValueError("entity descriptors and terrain origins differ")
    active = np.flatnonzero(compiled.entity_mask)
    if active.size > capacity:
        raise ValueError("entity descriptors exceed atlas capacity")
    expected_identity = np.asarray(
        [
            entity_type_identity_words(compiled.type_ids[int(code)])
            for code in compiled.type_code[active]
        ],
        dtype=np.uint32,
    ).reshape(active.size, 2)
    if not np.array_equal(
        compiled.type_identity_words[active],
        expected_identity,
    ):
        raise ValueError("entity type semantic identity is inconsistent")
    count = active.size
    mask[tile, :count] = True
    identity[tile, :count] = compiled.identity_words[active]
    position[tile, :count] = compiled.position[active]
    rotation[tile, :count] = compiled.rotation[active]
    velocity[tile, :count] = compiled.velocity[active]
    kind[tile, :count] = compiled.kind[active]
    type_identity[tile, :count] = compiled.type_identity_words[active]
    geometry_supported[tile, :count] = compiled.geometry_supported[active]
    collidable[tile, :count] = compiled.collidable[active]
    blocks_los[tile, :count] = compiled.blocks_los[active]
    local_bounds[tile, :count] = compiled.local_bounds[active]
    los_offset_supported[tile, :count] = compiled.line_of_sight_offset_supported[active]
    los_offset[tile, :count] = compiled.line_of_sight_offset[active]
    type_code[tile, :count] = np.asarray(
        [
            type_index[compiled.type_ids[int(code)]]
            for code in compiled.type_code[active]
        ],
        dtype=np.uint16,
    )


def _assign_entity_runtime_slots(
    tile_count: int,
    world_ids: Sequence[int],
    mask: np.ndarray,
    identity: np.ndarray,
    runtime_slot: np.ndarray,
    position: np.ndarray,
    rotation: np.ndarray,
    velocity: np.ndarray,
    kind: np.ndarray,
    type_code: np.ndarray,
    type_identity: np.ndarray,
    geometry_supported: np.ndarray,
    collidable: np.ndarray,
    blocks_los: np.ndarray,
    local_bounds: np.ndarray,
    los_offset_supported: np.ndarray,
    los_offset: np.ndarray,
    capacity: int,
) -> None:
    registries: dict[
        int,
        dict[tuple[int, int], tuple[object, ...]],
    ] = {}
    for tile in range(tile_count):
        registry = registries.setdefault(world_ids[tile], {})
        for definition in np.flatnonzero(mask[tile]):
            definition = int(definition)
            key = tuple(int(value) for value in identity[tile, definition])
            signature = (
                position[tile, definition].tobytes(),
                rotation[tile, definition].tobytes(),
                velocity[tile, definition].tobytes(),
                int(kind[tile, definition]),
                int(type_code[tile, definition]),
                type_identity[tile, definition].tobytes(),
                bool(geometry_supported[tile, definition]),
                bool(collidable[tile, definition]),
                bool(blocks_los[tile, definition]),
                local_bounds[tile, definition].tobytes(),
                bool(los_offset_supported[tile, definition]),
                los_offset[tile, definition].tobytes(),
            )
            previous = registry.setdefault(key, signature)
            if previous != signature:
                raise ValueError("overlapping entity definitions disagree")
            if len(registry) > capacity:
                raise ValueError(f"world entities exceed runtime capacity {capacity}")
    indices = {
        world_id: {key: slot for slot, key in enumerate(sorted(registry))}
        for world_id, registry in registries.items()
    }
    for tile in range(tile_count):
        for definition in np.flatnonzero(mask[tile]):
            key = tuple(int(value) for value in identity[tile, definition])
            runtime_slot[tile, definition] = indices[world_ids[tile]][key]


def _host_int(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error
