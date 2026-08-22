"""Leaf helpers extracted verbatim from atlas.py."""

import jax
import jax.numpy as jnp
from hytalegym.jax.world.surrogate.capabilities import SURROGATE_FAILURE_DOOR_INTENT
from hytalegym.jax.world.surrogate.types import (
    SurrogateAtlas,
    SurrogateCellSelection,
    SurrogateColumnSelection,
    SurrogateDoorTransitionResult,
    SurrogateEntitySelection,
    SurrogateRuntimeState,
)
from hytalegym.worldgen.region import (
    CAPTURE_BLOCKS_PER_AXIS,
    CAPTURE_CHUNKS_PER_AXIS,
    CHUNK_SIZE,
    CORE_CHUNKS_PER_AXIS,
    MIN_Y,
    WORLD_HEIGHT,
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


GENERATION_READY = 1


_DOOR_NORMALS = jnp.asarray(
    (
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (-1.0, 0.0, 0.0),
    ),
    dtype=jnp.float32,
)


def lookup_surrogate_columns(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    block_xz: jax.Array,
    *,
    require_core: bool = True,
) -> SurrogateColumnSelection:
    """Select one complete tile for each ``[B,Q,2]`` world column."""

    if not isinstance(require_core, bool):
        raise TypeError("require_core must be boolean")
    columns = jnp.asarray(block_xz, dtype=jnp.int32)
    if columns.ndim != 3 or columns.shape[-1] != 2:
        raise ValueError("block_xz must have shape [batch, queries, 2]")
    batch, _, _ = columns.shape
    _validate_runtime_shape(atlas, runtime, batch)

    chunks = jnp.floor_divide(columns, CHUNK_SIZE)
    core_relative = chunks[:, :, None, :] - atlas.core_min_chunk_xz[None, None, :, :]
    capture_relative = core_relative + 1
    relative = core_relative if require_core else capture_relative
    extent = CORE_CHUNKS_PER_AXIS if require_core else CAPTURE_CHUNKS_PER_AXIS
    inside = jnp.all((relative >= 0) & (relative < extent), axis=-1)
    safe_capture = jnp.clip(capture_relative, 0, CAPTURE_CHUNKS_PER_AXIS - 1)
    local_chunk_xz = columns & (CHUNK_SIZE - 1)
    local_xz = safe_capture * CHUNK_SIZE + local_chunk_xz[:, :, None, :]
    tile_axis = jnp.arange(atlas.tile_mask.shape[0])[None, None, :]
    known = atlas.column_known[
        tile_axis,
        local_xz[..., 0],
        local_xz[..., 1],
    ]
    compatible = (
        atlas.tile_mask[None, None, :]
        & (atlas.generation_status[None, None, :] == GENERATION_READY)
        & (
            atlas.world_id[None, None, :]
            == runtime.environment_world_id[:, None, None]
        )
    )
    eligible = (
        inside
        & known
        & compatible
        & ~runtime.unsupported_mechanics[:, None, None]
    )
    center_delta_twice = (
        jnp.clip(core_relative, -1, CORE_CHUNKS_PER_AXIS)
        * (2 * CHUNK_SIZE)
        + local_chunk_xz[:, :, None, :] * 2
        + 1
        - CORE_CHUNKS_PER_AXIS * CHUNK_SIZE
    )
    distance = jnp.sum(center_delta_twice**2, axis=-1)
    score = jnp.where(eligible, distance, jnp.iinfo(jnp.int32).max)
    tile_index = jnp.argmin(score, axis=2).astype(jnp.int32)
    available = jnp.any(eligible, axis=2)
    safe_tile = jnp.where(available, tile_index, 0)
    selected_local = jnp.take_along_axis(
        local_xz,
        safe_tile[..., None, None],
        axis=2,
    )[..., 0, :]
    return SurrogateColumnSelection(
        available=available,
        tile_index=jnp.where(available, tile_index, -1),
        local_xz=jnp.where(available[..., None], selected_local, 0),
    )


def lookup_surrogate_blocks(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    block_positions: jax.Array,
    *,
    require_core: bool = True,
) -> SurrogateCellSelection:
    """Look up ``[B,Q,3]`` cells with independent per-query tile selection."""

    if not isinstance(require_core, bool):
        raise TypeError("require_core must be boolean")
    blocks = jnp.asarray(block_positions, dtype=jnp.int32)
    if blocks.ndim != 3 or blocks.shape[-1] != 3:
        raise ValueError("block_positions must have shape [batch, queries, 3]")
    batch, _, _ = blocks.shape
    selected_column = lookup_surrogate_columns(
        atlas,
        runtime,
        blocks[..., (0, 2)],
        require_core=require_core,
    )
    inside_y = (blocks[..., 1] >= MIN_Y) & (blocks[..., 1] < MIN_Y + WORLD_HEIGHT)
    candidate_available = selected_column.available & inside_y
    safe_tile = jnp.where(candidate_available, selected_column.tile_index, 0)
    selected_local = jnp.where(
        candidate_available[..., None],
        selected_column.local_xz,
        0,
    )
    local_x = selected_local[..., 0]
    local_z = selected_local[..., 1]
    local_y = blocks[..., 1] - MIN_Y
    key = (
        local_y * CAPTURE_BLOCKS_PER_AXIS**2
        + local_z * CAPTURE_BLOCKS_PER_AXIS
        + local_x
    )

    height = atlas.terrain_height[safe_tile, local_x, local_z]
    cave_mask = atlas.cave_span_mask[safe_tile, local_x, local_z]
    cave_minimum = atlas.cave_min_y[safe_tile, local_x, local_z]
    cave_maximum = atlas.cave_max_y[safe_tile, local_x, local_z]
    carved = jnp.any(
        cave_mask
        & (blocks[..., 1, None] >= cave_minimum)
        & (blocks[..., 1, None] < cave_maximum),
        axis=-1,
    )
    terrain_code = jnp.where(
        carved,
        jnp.uint16(0),
        jnp.where(
            local_y == height,
            atlas.surface_cell_code[safe_tile, local_x, local_z],
            jnp.where(
                local_y < height,
                atlas.subsurface_cell_code[safe_tile, local_x, local_z],
                jnp.uint16(0),
            ),
        ),
    )
    found, structure_index = _find_structure_cells(
        atlas,
        safe_tile,
        key,
    )
    structure_code = atlas.structure_cell_code[
        safe_tile,
        structure_index,
    ]
    base_code = jnp.where(found, structure_code, terrain_code).astype(jnp.int32)
    raw_definition = atlas.structure_state_slot[
        safe_tile,
        structure_index,
    ]
    state_declared = found & (raw_definition >= 0)
    definition_capacity = atlas.stateful_mask.shape[1]
    safe_definition = jnp.clip(
        raw_definition,
        0,
        definition_capacity - 1,
    )
    definition_valid = (
        state_declared
        & (raw_definition < definition_capacity)
        & atlas.stateful_mask[safe_tile, safe_definition]
        & (atlas.stateful_block_key[safe_tile, safe_definition] == key)
    )
    runtime_slot = atlas.stateful_runtime_slot[
        safe_tile,
        safe_definition,
    ]
    runtime_capacity = runtime.stateful_state.shape[1]
    safe_runtime = jnp.clip(runtime_slot, 0, runtime_capacity - 1)
    batch_axis = jnp.arange(batch)[:, None]
    current_state = runtime.stateful_state[batch_axis, safe_runtime]
    initialized = runtime.stateful_initialized[batch_axis, safe_runtime]
    state_count = atlas.stateful_state_count[
        safe_tile,
        safe_definition,
    ]
    state_valid = (
        definition_valid
        & (runtime_slot >= 0)
        & (runtime_slot < runtime_capacity)
        & initialized
        & (current_state < state_count)
    )
    safe_state = jnp.clip(
        current_state,
        0,
        atlas.stateful_variant_cell_code.shape[2] - 1,
    )
    variant_code = atlas.stateful_variant_cell_code[
        safe_tile,
        safe_definition,
        safe_state,
    ].astype(jnp.int32)
    code = jnp.where(state_declared, variant_code, base_code)
    state_complete = ~state_declared | state_valid
    code_valid = (code >= 0) & (code < atlas.cell_palette_size)
    safe_code = jnp.where(code_valid, code, 0)
    shape_index = atlas.cell_shape_index[safe_code].astype(jnp.int32)
    shape_valid = shape_index < atlas.shape_palette_size
    safe_shape = jnp.where(shape_valid, shape_index, 0)
    available = candidate_available & state_complete & code_valid & shape_valid

    def cell_value(values: jax.Array) -> jax.Array:
        selected = values[safe_code]
        mask = available.reshape(available.shape + (1,) * (selected.ndim - 2))
        return jnp.where(mask, selected, jnp.zeros_like(selected))

    boxes = atlas.collision_boxes[safe_shape]
    box_mask = atlas.collision_box_mask[safe_shape]
    return SurrogateCellSelection(
        available=available,
        tile_index=jnp.where(available, selected_column.tile_index, -1),
        cell_code=jnp.where(available, code, 0),
        flags=cell_value(atlas.cell_flags),
        shape_index=jnp.where(available, shape_index, 0),
        fluid_level=cell_value(atlas.cell_fluid_level),
        support=cell_value(atlas.cell_support),
        block_damage=cell_value(atlas.cell_block_damage),
        fluid_damage=cell_value(atlas.cell_fluid_damage),
        movement=cell_value(atlas.cell_movement),
        fluid_movement=cell_value(atlas.cell_fluid_movement),
        collision_boxes=jnp.where(
            available[..., None, None],
            boxes,
            0.0,
        ),
        collision_box_mask=jnp.where(
            available[..., None],
            box_mask,
            False,
        ),
        structure_override=available & found,
        stateful=available & state_declared,
        stateful_definition_slot=jnp.where(
            available & state_declared,
            raw_definition,
            -1,
        ),
        stateful_runtime_slot=jnp.where(
            available & state_declared,
            runtime_slot,
            -1,
        ),
    )


def lookup_surrogate_entities(
    runtime: SurrogateRuntimeState,
    runtime_slots: jax.Array,
) -> SurrogateEntitySelection:
    """Read fixed entity slots without assembling learner observations."""

    if not isinstance(runtime, SurrogateRuntimeState):
        raise TypeError("runtime must be a SurrogateRuntimeState")
    slots = jnp.asarray(runtime_slots, dtype=jnp.int32)
    if slots.ndim != 2 or slots.shape[0] != runtime.environment_world_id.shape[0]:
        raise ValueError("runtime_slots must have shape [batch, queries]")
    capacity = runtime.entity_active.shape[1]
    safe = jnp.clip(slots, 0, capacity - 1)
    batch = jnp.arange(slots.shape[0])[:, None]
    available = (
        (slots >= 0)
        & (slots < capacity)
        & runtime.entity_initialized[batch, safe]
        & ~runtime.unsupported_mechanics[:, None]
    )

    def selected(values: jax.Array) -> jax.Array:
        result = values[batch, safe]
        mask = available.reshape(
            available.shape + (1,) * (result.ndim - available.ndim)
        )
        return jnp.where(mask, result, jnp.zeros_like(result))

    return SurrogateEntitySelection(
        available=available,
        active=available & runtime.entity_active[batch, safe],
        identity_words=selected(runtime.entity_identity_words),
        position=selected(runtime.entity_position),
        rotation=selected(runtime.entity_rotation),
        velocity=selected(runtime.entity_velocity),
        kind=selected(runtime.entity_kind),
        type_code=selected(runtime.entity_type_code),
        type_identity_words=selected(runtime.entity_type_identity_words),
        behavior_supported=(available & runtime.entity_behavior_supported[batch, safe]),
        geometry_supported=(available & runtime.entity_geometry_supported[batch, safe]),
        collidable=available & runtime.entity_collidable[batch, safe],
        blocks_los=available & runtime.entity_blocks_los[batch, safe],
        local_bounds=selected(runtime.entity_local_bounds),
        los_offset_supported=(
            available & runtime.entity_los_offset_supported[batch, safe]
        ),
        los_offset=selected(runtime.entity_los_offset),
    )


def apply_surrogate_door_use(
    atlas: SurrogateAtlas,
    runtime: SurrogateRuntimeState,
    door_blocks: jax.Array,
    actor_positions: jax.Array,
    target_clear: jax.Array,
    *,
    intent_mask: jax.Array | None = None,
    opposite_target_clear: jax.Array | None = None,
    partner_target_clear: jax.Array | None = None,
    partner_opposite_target_clear: jax.Array | None = None,
) -> SurrogateDoorTransitionResult:
    """Apply native-shaped single/double-door transitions after clearance."""

    blocks = jnp.asarray(door_blocks, dtype=jnp.int32)
    actors = jnp.asarray(actor_positions, dtype=jnp.float32)
    clear = jnp.asarray(target_clear, dtype=jnp.bool_)
    if blocks.ndim != 2 or blocks.shape[1] != 3:
        raise ValueError("door_blocks must have shape [batch, 3]")
    batch = blocks.shape[0]
    if actors.shape != (batch, 3) or clear.shape != (batch,):
        raise ValueError("actor_positions and target_clear must match the door batch")
    optional_clearance = (
        opposite_target_clear,
        partner_target_clear,
        partner_opposite_target_clear,
    )
    provided = tuple(value is not None for value in optional_clearance)
    if provided[1] != provided[2]:
        raise ValueError("double-door partner clearances must be supplied together")

    def clearance(value: jax.Array | None) -> jax.Array:
        result = (
            jnp.zeros(batch, dtype=jnp.bool_)
            if value is None
            else jnp.asarray(value, dtype=jnp.bool_)
        )
        if result.shape != (batch,):
            raise ValueError("optional door clearances must have shape [batch]")
        return result

    opposite_clear = clearance(opposite_target_clear)
    partner_clear = clearance(partner_target_clear)
    partner_opposite_clear = clearance(partner_opposite_target_clear)
    intents = (
        jnp.ones(batch, dtype=jnp.bool_)
        if intent_mask is None
        else jnp.asarray(intent_mask, dtype=jnp.bool_)
    )
    if intents.shape != (batch,):
        raise ValueError("intent_mask must have shape [batch]")

    selected = lookup_surrogate_blocks(
        atlas,
        runtime,
        blocks[:, None, :],
        require_core=True,
    )
    tile = jnp.maximum(selected.tile_index[:, 0], 0)
    definition = jnp.maximum(
        selected.stateful_definition_slot[:, 0],
        0,
    )
    runtime_slot = jnp.maximum(selected.stateful_runtime_slot[:, 0], 0)
    current = runtime.stateful_state[jnp.arange(batch), runtime_slot]
    raw_partner_slot = atlas.stateful_partner_runtime_slot[tile, definition]
    partner_slot = jnp.clip(
        raw_partner_slot,
        0,
        runtime.stateful_state.shape[1] - 1,
    )
    partner_current = runtime.stateful_state[
        jnp.arange(batch),
        partner_slot,
    ]
    partner_initialized = runtime.stateful_initialized[
        jnp.arange(batch),
        partner_slot,
    ]
    opposite_state = jnp.asarray((0, 2, 1), dtype=jnp.uint8)
    partner_detected = (
        (raw_partner_slot >= 0)
        & partner_initialized
        & (current < 3)
        & (partner_current == opposite_state[jnp.minimum(current, 2)])
    )
    yaw = atlas.stateful_yaw[tile, definition]
    finite_actor = jnp.all(jnp.isfinite(actors), axis=1)
    delta = actors - (blocks.astype(jnp.float32) + 0.5)
    front = jnp.sum(delta * _DOOR_NORMALS[yaw], axis=1) < 0.0
    side = front.astype(jnp.int32)
    supported = atlas.stateful_transition_mask[
        tile,
        definition,
        current,
        side,
    ]
    state_count = atlas.stateful_state_count[tile, definition]
    success_table = atlas.stateful_success_target[tile, definition]
    blocked_table = atlas.stateful_blocked_target[tile, definition]
    transition_table = atlas.stateful_transition_mask[tile, definition]
    declared = (
        jnp.arange(transition_table.shape[1], dtype=jnp.uint8)[None, :]
        < state_count[:, None]
    )
    side_dependent = jnp.any(
        declared
        & (
            (success_table[..., 0] != success_table[..., 1])
            | (blocked_table[..., 0] != blocked_table[..., 1])
            | (transition_table[..., 0] != transition_table[..., 1])
        ),
        axis=1,
    )
    base_valid = (
        intents
        & selected.available[:, 0]
        & selected.stateful[:, 0]
        & finite_actor
        & side_dependent
        & supported
    )
    desired = atlas.stateful_success_target[
        tile,
        definition,
        current,
        side,
    ]
    opposite = opposite_state[jnp.minimum(desired, 2)]
    pair_inputs_available = jnp.asarray(provided[1])
    pair_supported = ~partner_detected | pair_inputs_available
    desired_pair_clear = clear & (~partner_detected | partner_clear)
    opposite_pair_clear = opposite_clear & (~partner_detected | partner_opposite_clear)
    chosen_clear = desired_pair_clear | opposite_pair_clear
    target = jnp.where(desired_pair_clear, desired, opposite)
    valid = base_valid & pair_supported
    accepted = valid & chosen_clear
    next_value = jnp.where(accepted, target, current)
    next_states = runtime.stateful_state.at[
        jnp.arange(batch),
        runtime_slot,
    ].set(next_value)
    partner_target = opposite_state[target]
    partner_base = next_states[jnp.arange(batch), partner_slot]
    next_partner = jnp.where(
        accepted & partner_detected,
        partner_target,
        partner_base,
    )
    next_states = next_states.at[
        jnp.arange(batch),
        partner_slot,
    ].set(next_partner)
    unsupported = intents & ~valid
    next_failure_bits = runtime.failure_bits | jnp.where(
        unsupported,
        jnp.uint32(SURROGATE_FAILURE_DOOR_INTENT),
        jnp.uint32(0),
    )
    next_runtime = runtime._replace(
        stateful_state=next_states,
        failure_bits=next_failure_bits,
        unsupported_mechanics=(next_failure_bits != 0),
    )
    return SurrogateDoorTransitionResult(
        runtime=next_runtime,
        accepted=accepted,
        state_changed=accepted & (target != current),
        partner_state_changed=(
            accepted & partner_detected & (partner_target != partner_current)
        ),
        blocked=valid & ~chosen_clear,
        unsupported=unsupported,
    )


def _find_structure_cells(
    atlas: SurrogateAtlas,
    tile: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    capacity = atlas.structure_block_key.shape[1]
    low = jnp.zeros(key.shape, dtype=jnp.int32)
    high = jnp.full(key.shape, capacity, dtype=jnp.int32)
    # ``high`` is an exclusive sentinel at ``capacity``, so the search has
    # ``capacity + 1`` boundary states. Power-of-two capacities therefore
    # need one more step than ``(capacity - 1).bit_length()``.
    for _ in range(capacity.bit_length()):
        middle = (low + high) // 2
        value = atlas.structure_block_key[
            tile,
            jnp.minimum(middle, capacity - 1),
        ]
        move_right = value < key
        low = jnp.where(move_right, middle + 1, low)
        high = jnp.where(move_right, high, middle)
    index = jnp.minimum(low, capacity - 1)
    found = (
        (low < capacity)
        & atlas.structure_mask[tile, index]
        & (atlas.structure_block_key[tile, index] == key)
    )
    return found, index
