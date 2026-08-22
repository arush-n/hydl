"""Fixed-shape mutable-block health and exact geometry overlays."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.geometry.contract import (
    FLAG_SOLID,
    FLUID_MOVEMENT_FEATURES,
    MAX_DETAIL_BOXES,
    MOVEMENT_FEATURES,
)
from hytalegym.worldgen.region.contract import (
    MAX_CHUNK_COORDINATE,
    MIN_CHUNK_COORDINATE,
    HEIGHT_SECTIONS,
)
from hytalegym.worldgen.block_affordances import (
    BLOCK_AFFORDANCE_TAG_CAPACITY,
    BLOCK_AFFORDANCE_TAGS,
    GATHER_TYPES,
    block_affordance_dictionary_sha256,
)
from hytalegym.worldgen.block_actions import BLOCK_SEMANTIC_KEY_WORDS
from ._validate import (  # noqa: F401  (re-exported: callers unchanged)
    Array,
    MutableBlockGeometry,
    MutableBlockResync,
    MutableBlockState,
    MutableBlockUpdate,
    _array_shape,
    _exact_int,
    _nonnegative_int,
    _positive_int,
    _shape_tuple,
    _validate_geometry_shapes,
    _validate_resync_shapes,
    _validate_state_shapes,
    _validate_update_shapes,
)


MUTABLE_BLOCK_SCHEMA = "hytalerl_mutable_block_geometry_v1"
MUTABLE_BLOCK_VERSION = 1

MUTABLE_BLOCK_PROVENANCE_NONE = 0
MUTABLE_BLOCK_PROVENANCE_NATIVE = 1
MUTABLE_BLOCK_PROVENANCE_SURROGATE = 2

MUTABLE_BLOCK_FAILURE_INVALID = 1 << 0
MUTABLE_BLOCK_FAILURE_STALE_BASE = 1 << 1
MUTABLE_BLOCK_FAILURE_CONFLICT = 1 << 2
MUTABLE_BLOCK_FAILURE_CAPACITY = 1 << 3
MUTABLE_BLOCK_FAILURE_REVISION_WRAP = 1 << 4

BLOCK_HEALTH_REPAIR_DELAY_SECONDS = 5.0
BLOCK_HEALTH_REPAIR_PER_SECOND = 0.1


class MutableBlockResyncResult(NamedTuple):
    accepted: Array
    rejected: Array
    duplicate_section: Array
    duplicate_cell: Array
    invalid: Array
    resync_required: Array


class MutableBlockQueryResult(NamedTuple):
    """Current exact value after overlaying mutations on a pinned base."""

    available: Array
    overridden: Array
    block_health: Array
    block_health_valid: Array
    provenance: Array
    geometry: MutableBlockGeometry
    invalid: Array
    resync_required: Array


class MutableBlockUpdateResult(NamedTuple):
    accepted: Array
    applied: Array
    before: MutableBlockQueryResult
    after: MutableBlockQueryResult
    stale_base: Array
    duplicate_position: Array
    capacity_exceeded: Array
    invalid: Array
    revision_before: Array
    revision_after: Array
    resync_required: Array


class MutableBlockHealthAdvanceResult(NamedTuple):
    advanced: Array
    repaired: Array
    invalid: Array
    revision_before: Array
    revision_after: Array
    resync_required: Array


class NativeMutableBlockCellBatch(NamedTuple):
    """Device-ready exact cells captured from one live bridge process."""

    positions: Array
    available: Array
    geometry: MutableBlockGeometry
    block_health: Array
    block_health_valid: Array
    seconds_since_damage: Array
    damage_age_valid: Array
    local_change_counter: Array
    global_change_counter: Array


def empty_mutable_block_geometry(
    prefix_shape: tuple[int, ...],
    *,
    exact: bool = False,
) -> MutableBlockGeometry:
    """Return canonical air/unavailable geometry with ``prefix_shape``."""

    shape = _shape_tuple(prefix_shape)
    zeros_i32 = jnp.zeros(shape, dtype=jnp.int32)
    false = jnp.zeros(shape, dtype=jnp.bool_)
    return MutableBlockGeometry(
        exact=jnp.full(shape, exact, dtype=jnp.bool_),
        block_present=false,
        runtime_block_id=zeros_i32,
        runtime_block_id_valid=false,
        semantic_key=jnp.zeros(
            shape + (BLOCK_SEMANTIC_KEY_WORDS,),
            dtype=jnp.uint32,
        ),
        semantic_key_valid=false,
        affordance_valid=false,
        affordance_tags=jnp.zeros(shape, dtype=jnp.uint16),
        gather_type_index=jnp.zeros(shape, dtype=jnp.uint8),
        required_tool_quality=jnp.zeros(shape, dtype=jnp.int16),
        rotation_index=zeros_i32,
        flags=zeros_i32,
        fluid_level=zeros_i32,
        fluid_fill_height=jnp.zeros(shape, dtype=jnp.float32),
        support=zeros_i32,
        block_damage=zeros_i32,
        fluid_damage=zeros_i32,
        movement=jnp.zeros(
            shape + (MOVEMENT_FEATURES,),
            dtype=jnp.float32,
        ),
        fluid_movement=jnp.zeros(
            shape + (FLUID_MOVEMENT_FEATURES,),
            dtype=jnp.float32,
        ),
        collision_boxes=jnp.zeros(
            shape + (MAX_DETAIL_BOXES, 6),
            dtype=jnp.float32,
        ),
        collision_box_mask=jnp.zeros(
            shape + (MAX_DETAIL_BOXES,),
            dtype=jnp.bool_,
        ),
    )


def mutable_block_geometry_from_native_row(
    row,
    *,
    prefix_shape: tuple[int, ...] = (1, 1),
    include_runtime_id: bool = False,
) -> MutableBlockGeometry:
    """Broadcast one bridge-certified geometry row into a device value.

    Runtime block ordinals are process-local and therefore stay invalid by
    default. Portable semantic identity and every physical field remain exact.
    """

    from hytalegym.worldgen.native_mutable_blocks import (  # local: cycle-safe
        NativeMutableBlockRow,
    )

    if not isinstance(row, NativeMutableBlockRow):
        raise TypeError("row must be NativeMutableBlockRow")
    if not isinstance(include_runtime_id, bool):
        raise TypeError("include_runtime_id must be boolean")
    shape = _shape_tuple(prefix_shape)
    geometry = empty_mutable_block_geometry(shape, exact=True)
    if not row.block_present:
        return geometry

    box_count = row.collision_boxes.shape[0]
    boxes = geometry.collision_boxes.at[
        ...,
        :box_count,
        :,
    ].set(
        jnp.asarray(row.collision_boxes, dtype=jnp.float32)
    )
    box_mask = geometry.collision_box_mask.at[
        ...,
        :box_count,
    ].set(True)

    def full(value, dtype):
        return jnp.full(shape, value, dtype=dtype)

    return geometry._replace(
        block_present=jnp.ones(shape, dtype=jnp.bool_),
        runtime_block_id=full(
            row.runtime_block_id if include_runtime_id else 0,
            jnp.int32,
        ),
        runtime_block_id_valid=jnp.full(
            shape,
            include_runtime_id,
            dtype=jnp.bool_,
        ),
        semantic_key=jnp.broadcast_to(
            jnp.asarray(row.semantic_key, dtype=jnp.uint32),
            shape + (BLOCK_SEMANTIC_KEY_WORDS,),
        ),
        semantic_key_valid=jnp.ones(shape, dtype=jnp.bool_),
        affordance_valid=jnp.ones(shape, dtype=jnp.bool_),
        affordance_tags=full(row.affordance_tags, jnp.uint16),
        gather_type_index=full(row.gather_type_index, jnp.uint8),
        required_tool_quality=full(
            row.required_tool_quality,
            jnp.int16,
        ),
        rotation_index=full(row.rotation_index, jnp.int32),
        flags=full(row.flags, jnp.int32),
        fluid_level=full(row.fluid_level, jnp.int32),
        fluid_fill_height=full(row.fluid_fill_height, jnp.float32),
        support=full(row.support, jnp.int32),
        block_damage=full(row.block_damage, jnp.int32),
        fluid_damage=full(row.fluid_damage, jnp.int32),
        movement=jnp.broadcast_to(
            jnp.asarray(row.movement, dtype=jnp.float32),
            shape + (MOVEMENT_FEATURES,),
        ),
        fluid_movement=jnp.broadcast_to(
            jnp.asarray(row.fluid_movement, dtype=jnp.float32),
            shape + (FLUID_MOVEMENT_FEATURES,),
        ),
        collision_boxes=boxes,
        collision_box_mask=box_mask,
    )


def mutable_block_geometry_from_native_cells(
    capture,
    *,
    include_runtime_id: bool = False,
) -> NativeMutableBlockCellBatch:
    """Convert a live bridge cell capture without conflating unavailable/air."""

    from hytalegym.worldgen.native_mutable_blocks import (  # cycle-safe
        NativeMutableBlockCells,
    )

    if not isinstance(capture, NativeMutableBlockCells):
        raise TypeError("capture must be NativeMutableBlockCells")
    if not isinstance(include_runtime_id, bool):
        raise TypeError("include_runtime_id must be boolean")
    geometries = [
        (
            mutable_block_geometry_from_native_row(
                cell.row,
                prefix_shape=(),
                include_runtime_id=include_runtime_id,
            )
            if cell.available and cell.row is not None
            else empty_mutable_block_geometry((), exact=False)
        )
        for cell in capture.cells
    ]
    geometry = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values, axis=0),
        *geometries,
    )
    return NativeMutableBlockCellBatch(
        positions=jnp.asarray(
            [cell.position for cell in capture.cells],
            dtype=jnp.int32,
        ),
        available=jnp.asarray(
            [cell.available for cell in capture.cells],
            dtype=jnp.bool_,
        ),
        geometry=geometry,
        block_health=jnp.asarray(
            [
                cell.row.block_health
                if cell.available and cell.row is not None
                else 0.0
                for cell in capture.cells
            ],
            dtype=jnp.float32,
        ),
        block_health_valid=jnp.asarray(
            [
                cell.row.block_health_valid
                if cell.available and cell.row is not None
                else False
                for cell in capture.cells
            ],
            dtype=jnp.bool_,
        ),
        seconds_since_damage=jnp.asarray(
            [
                cell.row.seconds_since_damage
                if cell.available and cell.row is not None
                else 0.0
                for cell in capture.cells
            ],
            dtype=jnp.float32,
        ),
        damage_age_valid=jnp.asarray(
            [
                cell.row.damage_age_valid
                if cell.available and cell.row is not None
                else False
                for cell in capture.cells
            ],
            dtype=jnp.bool_,
        ),
        local_change_counter=jnp.asarray(
            [
                cell.row.local_change_counter
                if cell.available and cell.row is not None
                else 0
                for cell in capture.cells
            ],
            dtype=jnp.int16,
        ),
        global_change_counter=jnp.asarray(
            [
                cell.row.global_change_counter
                if cell.available and cell.row is not None
                else 0
                for cell in capture.cells
            ],
            dtype=jnp.int16,
        ),
    )


def empty_mutable_block_state(
    batch_size: int,
    *,
    section_capacity: int,
    mutation_capacity: int,
) -> MutableBlockState:
    """Return an explicitly unsynchronized state with caller-owned capacities."""

    batch = _positive_int(batch_size, "batch_size")
    sections = _positive_int(section_capacity, "section_capacity")
    cells = _positive_int(mutation_capacity, "mutation_capacity")
    return MutableBlockState(
        synchronized=jnp.zeros(batch, dtype=jnp.bool_),
        world_id=jnp.full(batch, -1, dtype=jnp.int32),
        base_semantic_sha256=jnp.zeros(
            (batch, BLOCK_SEMANTIC_KEY_WORDS),
            dtype=jnp.uint32,
        ),
        resync_epoch=jnp.zeros(batch, dtype=jnp.uint32),
        mutation_revision=jnp.zeros(batch, dtype=jnp.uint32),
        base_provenance=jnp.full(
            batch,
            MUTABLE_BLOCK_PROVENANCE_NONE,
            dtype=jnp.uint8,
        ),
        section_mask=jnp.zeros((batch, sections), dtype=jnp.bool_),
        section_coordinate=jnp.zeros(
            (batch, sections, 3),
            dtype=jnp.int32,
        ),
        local_change_counter=jnp.zeros(
            (batch, sections),
            dtype=jnp.int16,
        ),
        global_change_counter=jnp.zeros(
            (batch, sections),
            dtype=jnp.int16,
        ),
        cell_mask=jnp.zeros((batch, cells), dtype=jnp.bool_),
        cell_position=jnp.zeros((batch, cells, 3), dtype=jnp.int32),
        geometry_override=jnp.zeros((batch, cells), dtype=jnp.bool_),
        health_override=jnp.zeros((batch, cells), dtype=jnp.bool_),
        block_health=jnp.zeros((batch, cells), dtype=jnp.float32),
        seconds_since_damage=jnp.zeros(
            (batch, cells),
            dtype=jnp.float32,
        ),
        cell_provenance=jnp.full(
            (batch, cells),
            MUTABLE_BLOCK_PROVENANCE_NONE,
            dtype=jnp.uint8,
        ),
        geometry=empty_mutable_block_geometry((batch, cells)),
        failure_bits=jnp.zeros(batch, dtype=jnp.uint32),
    )


def empty_mutable_block_resync(
    state: MutableBlockState,
) -> MutableBlockResync:
    """Return an inert full-resync frame matching ``state`` capacities."""

    _validate_state_shapes(state)
    batch, cells = state.cell_mask.shape
    sections = state.section_mask.shape[1]
    return MutableBlockResync(
        publish=jnp.zeros(batch, dtype=jnp.bool_),
        snapshot_complete=jnp.zeros(batch, dtype=jnp.bool_),
        world_id=jnp.full(batch, -1, dtype=jnp.int32),
        base_semantic_sha256=jnp.zeros(
            (batch, BLOCK_SEMANTIC_KEY_WORDS),
            dtype=jnp.uint32,
        ),
        resync_epoch=jnp.zeros(batch, dtype=jnp.uint32),
        base_provenance=jnp.full(
            batch,
            MUTABLE_BLOCK_PROVENANCE_NONE,
            dtype=jnp.uint8,
        ),
        section_mask=jnp.zeros((batch, sections), dtype=jnp.bool_),
        section_coordinate=jnp.zeros(
            (batch, sections, 3),
            dtype=jnp.int32,
        ),
        local_change_counter=jnp.zeros(
            (batch, sections),
            dtype=jnp.int16,
        ),
        global_change_counter=jnp.zeros(
            (batch, sections),
            dtype=jnp.int16,
        ),
        cell_mask=jnp.zeros((batch, cells), dtype=jnp.bool_),
        cell_position=jnp.zeros((batch, cells, 3), dtype=jnp.int32),
        geometry_override=jnp.zeros((batch, cells), dtype=jnp.bool_),
        health_override=jnp.zeros((batch, cells), dtype=jnp.bool_),
        block_health=jnp.zeros((batch, cells), dtype=jnp.float32),
        seconds_since_damage=jnp.zeros(
            (batch, cells),
            dtype=jnp.float32,
        ),
        cell_provenance=jnp.full(
            (batch, cells),
            MUTABLE_BLOCK_PROVENANCE_NONE,
            dtype=jnp.uint8,
        ),
        geometry=empty_mutable_block_geometry((batch, cells)),
    )


def empty_mutable_block_update(
    state: MutableBlockState,
    *,
    query_capacity: int,
) -> MutableBlockUpdate:
    """Return an inert update carrying the current base identity."""

    _validate_state_shapes(state)
    queries = _positive_int(query_capacity, "query_capacity")
    batch = state.world_id.shape[0]
    shape = (batch, queries)
    return MutableBlockUpdate(
        mask=jnp.zeros(shape, dtype=jnp.bool_),
        world_id=state.world_id,
        base_semantic_sha256=state.base_semantic_sha256,
        resync_epoch=state.resync_epoch,
        expected_revision=state.mutation_revision,
        position=jnp.zeros(shape + (3,), dtype=jnp.int32),
        geometry_changed=jnp.zeros(shape, dtype=jnp.bool_),
        health_changed=jnp.zeros(shape, dtype=jnp.bool_),
        damage_applied=jnp.zeros(shape, dtype=jnp.bool_),
        expected_health=jnp.zeros(shape, dtype=jnp.float32),
        block_health_after=jnp.zeros(shape, dtype=jnp.float32),
        provenance=jnp.full(
            shape,
            MUTABLE_BLOCK_PROVENANCE_NONE,
            dtype=jnp.uint8,
        ),
        geometry=empty_mutable_block_geometry(shape),
    )


def publish_mutable_block_resync(
    state: MutableBlockState,
    frame: MutableBlockResync,
) -> tuple[MutableBlockState, MutableBlockResyncResult]:
    """Atomically replace a base only when the complete frame is valid."""

    _validate_resync_shapes(state, frame)
    section_duplicate = _duplicate_positions(
        frame.section_coordinate,
        frame.section_mask,
    )
    cell_duplicate = _duplicate_positions(
        frame.cell_position,
        frame.cell_mask,
    )
    section_valid = (
        (frame.section_coordinate[..., 0] >= MIN_CHUNK_COORDINATE)
        & (frame.section_coordinate[..., 0] <= MAX_CHUNK_COORDINATE)
        & (frame.section_coordinate[..., 1] >= 0)
        & (frame.section_coordinate[..., 1] < HEIGHT_SECTIONS)
        & (frame.section_coordinate[..., 2] >= MIN_CHUNK_COORDINATE)
        & (frame.section_coordinate[..., 2] <= MAX_CHUNK_COORDINATE)
    )
    cell_valid = (
        (frame.cell_position[..., 1] >= 0)
        & (frame.cell_position[..., 1] < HEIGHT_SECTIONS * 32)
    )
    geometry_valid = _geometry_valid(frame.geometry)
    override_valid = frame.geometry_override | frame.health_override
    provenance_valid = _provenance_valid(frame.cell_provenance)
    health_valid = _health_valid(
        frame.geometry.block_present,
        frame.block_health,
    ) & (
        jnp.isfinite(frame.seconds_since_damage)
        & (frame.seconds_since_damage >= 0.0)
    )
    active_cells_valid = jnp.all(
        ~frame.cell_mask
        | (
            cell_valid
            & geometry_valid
            & override_valid
            & provenance_valid
            & (~frame.health_override | health_valid)
        ),
        axis=1,
    )
    base_valid = (
        frame.snapshot_complete
        & (frame.world_id >= 0)
        & jnp.any(frame.base_semantic_sha256 != 0, axis=1)
        & _provenance_valid(frame.base_provenance)
        & jnp.any(frame.section_mask, axis=1)
        & jnp.all(~frame.section_mask | section_valid, axis=1)
        & ~section_duplicate
        & ~cell_duplicate
        & active_cells_valid
    )
    accepted = frame.publish & base_valid
    rejected = frame.publish & ~base_valid

    candidate = state._replace(
        synchronized=jnp.ones_like(state.synchronized),
        world_id=frame.world_id,
        base_semantic_sha256=frame.base_semantic_sha256,
        resync_epoch=frame.resync_epoch,
        mutation_revision=jnp.zeros_like(state.mutation_revision),
        base_provenance=frame.base_provenance,
        section_mask=frame.section_mask,
        section_coordinate=frame.section_coordinate,
        local_change_counter=frame.local_change_counter,
        global_change_counter=frame.global_change_counter,
        cell_mask=frame.cell_mask,
        cell_position=frame.cell_position,
        geometry_override=frame.geometry_override,
        health_override=frame.health_override,
        block_health=frame.block_health,
        seconds_since_damage=frame.seconds_since_damage,
        cell_provenance=frame.cell_provenance,
        geometry=frame.geometry,
        failure_bits=jnp.zeros_like(state.failure_bits),
    )
    next_state = _select_state_rows(accepted, candidate, state)
    next_state = next_state._replace(
        synchronized=next_state.synchronized & ~rejected,
        failure_bits=next_state.failure_bits
        | jnp.where(
            rejected,
            jnp.uint32(MUTABLE_BLOCK_FAILURE_INVALID),
            jnp.uint32(0),
        ),
    )
    return next_state, MutableBlockResyncResult(
        accepted=accepted,
        rejected=rejected,
        duplicate_section=frame.publish & section_duplicate,
        duplicate_cell=frame.publish & cell_duplicate,
        invalid=rejected,
        resync_required=~next_state.synchronized,
    )


def query_mutable_blocks(
    state: MutableBlockState,
    *,
    world_id: Array,
    base_semantic_sha256: Array,
    resync_epoch: Array,
    position: Array,
    base_available: Array,
    base_geometry: MutableBlockGeometry,
    base_block_health: Array,
) -> MutableBlockQueryResult:
    """Overlay current mutations on exact values from the pinned full base."""

    _validate_state_shapes(state)
    points = jnp.asarray(position, dtype=jnp.int32)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("position must have shape [batch, query, 3]")
    batch, queries, _ = points.shape
    if batch != state.world_id.shape[0]:
        raise ValueError("position batch must match mutable state")
    shape = (batch, queries)
    worlds = _array_shape(world_id, (batch,), jnp.int32, "world_id")
    digest = _array_shape(
        base_semantic_sha256,
        (batch, BLOCK_SEMANTIC_KEY_WORDS),
        jnp.uint32,
        "base_semantic_sha256",
    )
    epoch = _array_shape(
        resync_epoch,
        (batch,),
        jnp.uint32,
        "resync_epoch",
    )
    base_mask = _array_shape(
        base_available,
        shape,
        jnp.bool_,
        "base_available",
    )
    base_health = _array_shape(
        base_block_health,
        shape,
        jnp.float32,
        "base_block_health",
    )
    _validate_geometry_shapes(base_geometry, shape, "base_geometry")

    slot_match = state.cell_mask[:, None, :] & jnp.all(
        state.cell_position[:, None, :, :] == points[:, :, None, :],
        axis=3,
    )
    match_count = jnp.sum(slot_match, axis=2)
    duplicate = match_count > 1
    slot = jnp.argmax(slot_match, axis=2)
    matched = match_count == 1
    stored_geometry = _gather_geometry(state.geometry, slot)
    geometry_override = _gather_slots(state.geometry_override, slot)
    health_override = _gather_slots(state.health_override, slot)
    stored_health = _gather_slots(state.block_health, slot)
    stored_provenance = _gather_slots(state.cell_provenance, slot)
    use_geometry = matched & geometry_override
    use_health = matched & health_override
    geometry = _select_geometry(use_geometry, stored_geometry, base_geometry)
    health = jnp.where(use_health, stored_health, base_health)
    provenance = jnp.where(
        matched & (geometry_override | health_override),
        stored_provenance,
        state.base_provenance[:, None],
    )
    identity = (
        state.synchronized
        & (state.world_id == worlds)
        & jnp.all(state.base_semantic_sha256 == digest, axis=1)
        & (state.resync_epoch == epoch)
    )
    geometry_valid = _geometry_valid(geometry)
    health_valid = _health_valid(geometry.block_present, health)
    invalid = duplicate | (base_mask & (~geometry_valid | ~health_valid))
    available = (
        identity[:, None]
        & base_mask
        & ~invalid
        & (points[..., 1] >= 0)
        & (points[..., 1] < HEIGHT_SECTIONS * 32)
    )
    output_geometry = _mask_geometry(geometry, available)
    block_health_valid = available & output_geometry.block_present
    return MutableBlockQueryResult(
        available=available,
        overridden=available & matched,
        block_health=jnp.where(block_health_valid, health, 0.0),
        block_health_valid=block_health_valid,
        provenance=jnp.where(
            available,
            provenance,
            jnp.uint8(MUTABLE_BLOCK_PROVENANCE_NONE),
        ),
        geometry=output_geometry,
        invalid=invalid,
        resync_required=~identity[:, None] | duplicate,
    )


def apply_mutable_block_updates(
    state: MutableBlockState,
    *,
    base_available: Array,
    base_geometry: MutableBlockGeometry,
    base_block_health: Array,
    update: MutableBlockUpdate,
) -> tuple[MutableBlockState, MutableBlockUpdateResult]:
    """Atomically upsert exact resulting values or invalidate the whole row."""

    _validate_update_shapes(state, update)
    batch, queries = update.mask.shape
    before = query_mutable_blocks(
        state,
        world_id=update.world_id,
        base_semantic_sha256=update.base_semantic_sha256,
        resync_epoch=update.resync_epoch,
        position=update.position,
        base_available=base_available,
        base_geometry=base_geometry,
        base_block_health=base_block_health,
    )
    requested = jnp.any(update.mask, axis=1)
    identity = (
        state.synchronized
        & (state.world_id == update.world_id)
        & jnp.all(
            state.base_semantic_sha256 == update.base_semantic_sha256,
            axis=1,
        )
        & (state.resync_epoch == update.resync_epoch)
        & (state.mutation_revision == update.expected_revision)
    )
    stale = requested & ~identity
    duplicate_update = _duplicate_positions(update.position, update.mask)
    duplicate_state = _duplicate_positions(
        state.cell_position,
        state.cell_mask,
    )
    duplicate = requested & (duplicate_update | duplicate_state)

    changed = update.geometry_changed | update.health_changed
    final_geometry = _select_geometry(
        update.geometry_changed,
        update.geometry,
        before.geometry,
    )
    geometry_valid = _geometry_valid(final_geometry)
    final_health_valid = _health_valid(
        final_geometry.block_present,
        update.block_health_after,
    )
    presence_changed = (
        final_geometry.block_present != before.geometry.block_present
    )
    update_valid = (
        before.available
        & changed
        & _provenance_valid(update.provenance)
        & geometry_valid
        & final_health_valid
        & (update.expected_health == before.block_health)
        & (
            update.health_changed
            | (update.block_health_after == before.block_health)
        )
        & (~presence_changed | update.health_changed)
        & (
            ~update.damage_applied
            | (
                update.health_changed
                & before.geometry.block_present
                & (
                    update.block_health_after
                    <= before.block_health
                )
            )
        )
    )
    invalid = requested & identity & jnp.any(
        update.mask & ~update_valid,
        axis=1,
    )

    existing_match = state.cell_mask[:, None, :] & jnp.all(
        state.cell_position[:, None, :, :] == update.position[:, :, None, :],
        axis=3,
    )
    new_position = update.mask & ~jnp.any(existing_match, axis=2)
    new_count = jnp.sum(new_position, axis=1)
    free_count = jnp.sum(~state.cell_mask, axis=1)
    capacity = requested & (new_count > free_count)
    revision_wrap = requested & (
        state.mutation_revision == jnp.uint32(0xFFFFFFFF)
    )
    row_accepted = (
        requested
        & identity
        & ~duplicate
        & ~invalid
        & ~capacity
        & ~revision_wrap
    )

    next_state = state
    rows = jnp.arange(batch, dtype=jnp.int32)
    for query in range(queries):
        active = row_accepted & update.mask[:, query]
        matches = next_state.cell_mask & jnp.all(
            next_state.cell_position == update.position[:, query, None, :],
            axis=2,
        )
        present = jnp.any(matches, axis=1)
        existing_slot = jnp.argmax(matches, axis=1)
        free_slot = jnp.argmax(~next_state.cell_mask, axis=1)
        slot = jnp.where(present, existing_slot, free_slot)

        next_state = next_state._replace(
            cell_mask=_write_slot(
                next_state.cell_mask,
                jnp.ones(batch, dtype=jnp.bool_),
                slot,
                active,
                rows,
            ),
            cell_position=_write_slot(
                next_state.cell_position,
                update.position[:, query],
                slot,
                active,
                rows,
            ),
            geometry_override=_write_slot(
                next_state.geometry_override,
                _gather_slots(next_state.geometry_override, slot)
                | update.geometry_changed[:, query],
                slot,
                active,
                rows,
            ),
            health_override=_write_slot(
                next_state.health_override,
                _gather_slots(next_state.health_override, slot)
                | update.health_changed[:, query],
                slot,
                active,
                rows,
            ),
            block_health=_write_slot(
                next_state.block_health,
                update.block_health_after[:, query],
                slot,
                active & update.health_changed[:, query],
                rows,
            ),
            seconds_since_damage=_write_slot(
                next_state.seconds_since_damage,
                jnp.where(
                    update.damage_applied[:, query],
                    0.0,
                    _gather_slots(
                        next_state.seconds_since_damage,
                        slot,
                    ),
                ),
                slot,
                active & update.health_changed[:, query],
                rows,
            ),
            cell_provenance=_write_slot(
                next_state.cell_provenance,
                update.provenance[:, query],
                slot,
                active,
                rows,
            ),
            geometry=_write_geometry_slot(
                next_state.geometry,
                final_geometry,
                query,
                slot,
                active,
                rows,
            ),
        )

    rejected = requested & ~row_accepted
    next_state = next_state._replace(
        synchronized=next_state.synchronized & ~rejected,
        mutation_revision=jnp.where(
            row_accepted,
            state.mutation_revision + jnp.uint32(1),
            state.mutation_revision,
        ),
        failure_bits=(
            state.failure_bits
            | jnp.where(
                invalid,
                jnp.uint32(MUTABLE_BLOCK_FAILURE_INVALID),
                jnp.uint32(0),
            )
            | jnp.where(
                stale,
                jnp.uint32(MUTABLE_BLOCK_FAILURE_STALE_BASE),
                jnp.uint32(0),
            )
            | jnp.where(
                duplicate,
                jnp.uint32(MUTABLE_BLOCK_FAILURE_CONFLICT),
                jnp.uint32(0),
            )
            | jnp.where(
                capacity,
                jnp.uint32(MUTABLE_BLOCK_FAILURE_CAPACITY),
                jnp.uint32(0),
            )
            | jnp.where(
                revision_wrap,
                jnp.uint32(MUTABLE_BLOCK_FAILURE_REVISION_WRAP),
                jnp.uint32(0),
            )
        ),
    )
    after = query_mutable_blocks(
        next_state,
        world_id=update.world_id,
        base_semantic_sha256=update.base_semantic_sha256,
        resync_epoch=update.resync_epoch,
        position=update.position,
        base_available=base_available,
        base_geometry=base_geometry,
        base_block_health=base_block_health,
    )
    return next_state, MutableBlockUpdateResult(
        accepted=row_accepted,
        applied=row_accepted[:, None] & update.mask,
        before=before,
        after=after,
        stale_base=stale,
        duplicate_position=duplicate,
        capacity_exceeded=capacity,
        invalid=invalid,
        revision_before=state.mutation_revision,
        revision_after=next_state.mutation_revision,
        resync_required=~next_state.synchronized,
    )


def advance_mutable_block_health(
    state: MutableBlockState,
    dt_seconds: Array,
) -> tuple[MutableBlockState, MutableBlockHealthAdvanceResult]:
    """Advance native 0.5.7 delayed block repair without changing geometry."""

    _validate_state_shapes(state)
    batch = state.world_id.shape[0]
    dt = jnp.asarray(dt_seconds, dtype=jnp.float32)
    if dt.ndim == 0:
        dt = jnp.broadcast_to(dt, (batch,))
    if dt.shape != (batch,):
        raise ValueError("dt_seconds must be scalar or have shape [batch]")
    requested = jnp.any(
        state.cell_mask
        & state.health_override
        & state.geometry.block_present
        & (state.block_health < 1.0),
        axis=1,
    )
    invalid = requested & (~jnp.isfinite(dt) | (dt <= 0.0))
    revision_wrap = requested & (
        state.mutation_revision == jnp.uint32(0xFFFFFFFF)
    )
    accepted = requested & state.synchronized & ~invalid & ~revision_wrap
    active = (
        accepted[:, None]
        & state.cell_mask
        & state.health_override
        & state.geometry.block_present
        & (state.block_health < 1.0)
    )
    age = jnp.where(
        active,
        state.seconds_since_damage + dt[:, None],
        state.seconds_since_damage,
    )
    healing = active & (
        age >= jnp.float32(BLOCK_HEALTH_REPAIR_DELAY_SECONDS)
    )
    health = jnp.where(
        healing,
        jnp.minimum(
            1.0,
            state.block_health
            + jnp.float32(BLOCK_HEALTH_REPAIR_PER_SECOND) * dt[:, None],
        ),
        state.block_health,
    )
    repaired = healing & (health >= 1.0)
    clear_slot = repaired & ~state.geometry_override
    next_state = state._replace(
        cell_mask=state.cell_mask & ~clear_slot,
        health_override=state.health_override & ~clear_slot,
        block_health=jnp.where(clear_slot, 0.0, health),
        seconds_since_damage=jnp.where(
            clear_slot | repaired,
            0.0,
            age,
        ),
        cell_provenance=jnp.where(
            clear_slot,
            jnp.uint8(MUTABLE_BLOCK_PROVENANCE_NONE),
            state.cell_provenance,
        ),
        mutation_revision=jnp.where(
            accepted,
            state.mutation_revision + jnp.uint32(1),
            state.mutation_revision,
        ),
        synchronized=state.synchronized & ~invalid & ~revision_wrap,
        failure_bits=(
            state.failure_bits
            | jnp.where(
                invalid,
                jnp.uint32(MUTABLE_BLOCK_FAILURE_INVALID),
                jnp.uint32(0),
            )
            | jnp.where(
                revision_wrap,
                jnp.uint32(MUTABLE_BLOCK_FAILURE_REVISION_WRAP),
                jnp.uint32(0),
            )
        ),
    )
    return next_state, MutableBlockHealthAdvanceResult(
        advanced=accepted,
        repaired=repaired,
        invalid=invalid,
        revision_before=state.mutation_revision,
        revision_after=next_state.mutation_revision,
        resync_required=~next_state.synchronized,
    )


def estimated_mutable_block_state_bytes(
    *,
    batch_size: int,
    section_capacity: int,
    mutation_capacity: int,
) -> int:
    """Return exact leaf payload bytes, excluding allocator/XLA overhead."""

    batch = _positive_int(batch_size, "batch_size")
    sections = _positive_int(section_capacity, "section_capacity")
    cells = _positive_int(mutation_capacity, "mutation_capacity")
    base_bytes = (
        1  # synchronized
        + 4  # world_id
        + 4 * BLOCK_SEMANTIC_KEY_WORDS
        + 4  # resync_epoch
        + 4  # mutation_revision
        + 1  # provenance
        + 4  # failure_bits
    )
    section_bytes = 1 + 3 * 4 + 2 + 2
    geometry_bytes = (
        1  # exact
        + 1  # block_present
        + 4  # runtime ID
        + 1  # runtime ID valid
        + 4 * BLOCK_SEMANTIC_KEY_WORDS
        + 1  # semantic key valid
        + 1  # affordance valid
        + 2  # affordance tag bits
        + 1  # stable gather type
        + 2  # required tool quality
        + 6 * 4  # rotation, flags, fluid, support, two damage values
        + 4  # fluid fill
        + MOVEMENT_FEATURES * 4
        + FLUID_MOVEMENT_FEATURES * 4
        + MAX_DETAIL_BOXES * 6 * 4
        + MAX_DETAIL_BOXES
    )
    cell_bytes = (
        1  # mask
        + 3 * 4  # position
        + 1  # geometry override
        + 1  # health override
        + 4  # health
        + 4  # damage age
        + 1  # provenance
        + geometry_bytes
    )
    return batch * (
        base_bytes + sections * section_bytes + cells * cell_bytes
    )


def mutable_block_contract() -> dict[str, object]:
    """Return the checkpoint-pinnable mutable block boundary."""

    return {
        "schema": MUTABLE_BLOCK_SCHEMA,
        "version": MUTABLE_BLOCK_VERSION,
        "server_version": "0.5.7",
        "base": {
            "identity": [
                "world_id",
                "full_snapshot_semantic_sha256",
                "resync_epoch",
            ],
            "publication": "complete_snapshot_atomic_replace",
            "partial_or_invalid": "retain_arrays_but_mark_unsynchronized",
            "sections": {
                "coordinate": ["chunk_x", "section_y", "chunk_z"],
                "change_counters": "signed_short_inequality_only",
            },
        },
        "overlay": {
            "capacity": "caller_selected_static_axis",
            "position": "absolute_block_i32_xyz",
            "duplicate_position": "reject_complete_batch_row",
            "overflow": "reject_complete_batch_row_and_require_resync",
            "revision": "uint32_equality_only_resync_before_wrap",
        },
        "geometry": {
            "result": "exact_values_not_boolean_legality",
            "block_presence": "explicit_never_inferred_from_solid_or_boxes",
            "portable_identity": (
                "sha256_words_of_canonical_asset_state_rotation"
            ),
            "runtime_block_id": "optional_process_local_diagnostic",
            "affordance_dictionary_sha256": (
                block_affordance_dictionary_sha256()
            ),
            "affordances": {
                "tags": (
                    f"uint16_known_bits_capacity_{BLOCK_AFFORDANCE_TAG_CAPACITY}"
                ),
                "gather_type": (
                    f"stable_uint8_dictionary_{len(GATHER_TYPES)}_entries"
                ),
                "required_tool_quality": (
                    "BlockBreakingDropType.getQuality_int16"
                ),
                "unknown_tag_or_gather_type": (
                    "reject_complete_geometry_value"
                ),
                "raw_asset_or_runtime_ordinal_as_policy_input": False,
            },
            "values": [
                "rotation",
                "flags",
                "fluid_level_and_fill",
                "support",
                "block_and_fluid_damage",
                f"movement_{MOVEMENT_FEATURES}",
                f"fluid_movement_{FLUID_MOVEMENT_FEATURES}",
                f"collision_boxes_{MAX_DETAIL_BOXES}",
            ],
        },
        "health": {
            "range": "normalized_[0,1]_only_when_block_present",
            "absent_value": 0.0,
            "damage_owner": "caller_resolves_amount_world_owns_state",
            "expected_before": "exact_revision_guarded",
            "repair": {
                "delay_seconds": BLOCK_HEALTH_REPAIR_DELAY_SECONDS,
                "rate_per_second": BLOCK_HEALTH_REPAIR_PER_SECOND,
                "threshold": "new_damage_age_greater_than_or_equal_to_delay",
                "full_health": "remove_health_only_overlay",
            },
        },
        "provenance": {
            "native": MUTABLE_BLOCK_PROVENANCE_NATIVE,
            "surrogate": MUTABLE_BLOCK_PROVENANCE_SURROGATE,
            "none": MUTABLE_BLOCK_PROVENANCE_NONE,
            "required_per_base_and_update": True,
        },
        "fail_closed": [
            "missing_or_stale_base_identity",
            "incomplete_snapshot",
            "duplicate_section_or_cell",
            "invalid_or_non_exact_geometry",
            "unknown_or_missing_block_affordance",
            "non_finite_or_out_of_range_health",
            "mutation_revision_mismatch_or_wrap",
            "fixed_capacity_overflow",
            "conflicting_same_tick_position",
        ],
        "native_delta_boundary": {
            "changed_positions": (
                "BlockSection.getAndClearChangedPositions_destructive_drain"
            ),
            "required_order": [
                "capture_complete_full_snapshot",
                "publish_base_and_epoch_atomically",
                "acknowledge_signed_short_counters",
                "only_then_drain_and_apply_matching_deltas",
            ],
            "counter_only_change": "not_a_block_delta_requires_light_or_resync",
        },
        "compiled_geometry_boundary": {
            "GeometryState_and_RegionGeometryState": (
                "insufficient_for_block_presence_stable_identity_and_health"
            ),
            "adapter_policy": "no_inference_fail_closed",
        },
        "certification": {
            "source": "installed_0.5.7_class_hashes",
            "array_behavior": "eager_and_jit",
            "native_runtime": "pending_staged_bridge_endpoint_and_live_fixture",
        },
        "not_modelled": [
            "block_break_or_place_legality",
            "tool_damage_resolution",
            "drop_quantity_randomization_or_inventory_commit",
            "fluid_mutation",
            "lighting_delta_payloads",
            "connected_stateful_block_propagation",
        ],
    }


def mutable_block_contract_sha256() -> str:
    payload = json.dumps(
        mutable_block_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _geometry_valid(geometry: MutableBlockGeometry) -> Array:
    boxes = geometry.collision_boxes
    mask = geometry.collision_box_mask
    finite = (
        jnp.isfinite(geometry.fluid_fill_height)
        & jnp.all(jnp.isfinite(geometry.movement), axis=-1)
        & jnp.all(jnp.isfinite(geometry.fluid_movement), axis=-1)
        & jnp.all(jnp.isfinite(boxes), axis=(-2, -1))
    )
    ordered_boxes = jnp.all(
        ~mask | jnp.all(boxes[..., :3] <= boxes[..., 3:], axis=-1),
        axis=-1,
    )
    inactive_zero = jnp.all(
        mask[..., None] | (boxes == 0.0),
        axis=(-2, -1),
    )
    semantic_key = (
        geometry.semantic_key_valid
        & jnp.any(geometry.semantic_key != 0, axis=-1)
    )
    semantic_identity = jnp.where(
        geometry.block_present,
        semantic_key,
        ~geometry.semantic_key_valid
        & jnp.all(geometry.semantic_key == 0, axis=-1),
    )
    runtime_identity = jnp.where(
        geometry.runtime_block_id_valid,
        jnp.where(
            geometry.block_present,
            geometry.runtime_block_id > 0,
            geometry.runtime_block_id == 0,
        ),
        geometry.runtime_block_id == 0,
    )
    affordance_identity = jnp.where(
        geometry.block_present,
        geometry.affordance_valid
        & (geometry.gather_type_index < len(GATHER_TYPES))
        & (geometry.required_tool_quality >= 0),
        ~geometry.affordance_valid
        & (geometry.affordance_tags == 0)
        & (geometry.gather_type_index == 0)
        & (geometry.required_tool_quality == 0),
    )
    unknown_affordance_mask = jnp.uint16(
        (~((1 << len(BLOCK_AFFORDANCE_TAGS)) - 1)) & 0xFFFF
    )
    known_affordance_bits = (
        geometry.affordance_tags & unknown_affordance_mask
    ) == 0
    collision_valid = (
        geometry.block_present
        | ~jnp.any(mask, axis=-1)
    ) & (
        ((geometry.flags & jnp.int32(FLAG_SOLID)) == 0)
        | jnp.any(mask, axis=-1)
    )
    return (
        geometry.exact
        & finite
        & (geometry.fluid_fill_height >= 0.0)
        & (geometry.fluid_fill_height <= 1.0)
        & ordered_boxes
        & inactive_zero
        & semantic_identity
        & runtime_identity
        & affordance_identity
        & known_affordance_bits
        & collision_valid
        & (geometry.rotation_index >= 0)
        & (geometry.fluid_level >= 0)
    )


def _health_valid(block_present: Array, health: Array) -> Array:
    return jnp.isfinite(health) & jnp.where(
        block_present,
        (health >= 0.0) & (health <= 1.0),
        health == 0.0,
    )


def _provenance_valid(value: Array) -> Array:
    return (
        (value == MUTABLE_BLOCK_PROVENANCE_NATIVE)
        | (value == MUTABLE_BLOCK_PROVENANCE_SURROGATE)
    )


def _duplicate_positions(position: Array, mask: Array) -> Array:
    same = jnp.all(
        position[:, :, None, :] == position[:, None, :, :],
        axis=3,
    )
    active = mask[:, :, None] & mask[:, None, :]
    upper = jnp.triu(jnp.ones(same.shape[1:], dtype=jnp.bool_), k=1)
    return jnp.any(same & active & upper[None, :, :], axis=(1, 2))


def _select_state_rows(
    mask: Array,
    selected: MutableBlockState,
    fallback: MutableBlockState,
) -> MutableBlockState:
    return jax.tree_util.tree_map(
        lambda left, right: jnp.where(
            mask.reshape((mask.shape[0],) + (1,) * (left.ndim - 1)),
            left,
            right,
        ),
        selected,
        fallback,
    )


def _select_geometry(
    mask: Array,
    selected: MutableBlockGeometry,
    fallback: MutableBlockGeometry,
) -> MutableBlockGeometry:
    return jax.tree_util.tree_map(
        lambda left, right: jnp.where(
            mask.reshape(mask.shape + (1,) * (left.ndim - mask.ndim)),
            left,
            right,
        ),
        selected,
        fallback,
    )


def _mask_geometry(
    geometry: MutableBlockGeometry,
    mask: Array,
) -> MutableBlockGeometry:
    empty = empty_mutable_block_geometry(mask.shape)
    return _select_geometry(mask, geometry, empty)


def _gather_slots(values: Array, slot: Array) -> Array:
    return jax.vmap(lambda row, selected: row[selected])(values, slot)


def _gather_geometry(
    geometry: MutableBlockGeometry,
    slot: Array,
) -> MutableBlockGeometry:
    return jax.tree_util.tree_map(
        lambda values: _gather_slots(values, slot),
        geometry,
    )


def _write_slot(
    values: Array,
    source: Array,
    slot: Array,
    active: Array,
    rows: Array,
) -> Array:
    current = values[rows, slot]
    use = active.reshape(
        (active.shape[0],) + (1,) * (current.ndim - 1)
    )
    return values.at[rows, slot].set(jnp.where(use, source, current))


def _write_geometry_slot(
    geometry: MutableBlockGeometry,
    source: MutableBlockGeometry,
    query: int,
    slot: Array,
    active: Array,
    rows: Array,
) -> MutableBlockGeometry:
    return jax.tree_util.tree_map(
        lambda values, updates: _write_slot(
            values,
            updates[:, query],
            slot,
            active,
            rows,
        ),
        geometry,
        source,
    )


__all__ = [
    "BLOCK_HEALTH_REPAIR_DELAY_SECONDS",
    "BLOCK_HEALTH_REPAIR_PER_SECOND",
    "BLOCK_SEMANTIC_KEY_WORDS",
    "MUTABLE_BLOCK_FAILURE_CAPACITY",
    "MUTABLE_BLOCK_FAILURE_CONFLICT",
    "MUTABLE_BLOCK_FAILURE_INVALID",
    "MUTABLE_BLOCK_FAILURE_REVISION_WRAP",
    "MUTABLE_BLOCK_FAILURE_STALE_BASE",
    "MUTABLE_BLOCK_PROVENANCE_NATIVE",
    "MUTABLE_BLOCK_PROVENANCE_NONE",
    "MUTABLE_BLOCK_PROVENANCE_SURROGATE",
    "MUTABLE_BLOCK_SCHEMA",
    "MUTABLE_BLOCK_VERSION",
    "MutableBlockGeometry",
    "MutableBlockHealthAdvanceResult",
    "NativeMutableBlockCellBatch",
    "MutableBlockQueryResult",
    "MutableBlockResync",
    "MutableBlockResyncResult",
    "MutableBlockState",
    "MutableBlockUpdate",
    "MutableBlockUpdateResult",
    "advance_mutable_block_health",
    "apply_mutable_block_updates",
    "empty_mutable_block_geometry",
    "empty_mutable_block_resync",
    "empty_mutable_block_state",
    "empty_mutable_block_update",
    "estimated_mutable_block_state_bytes",
    "mutable_block_contract",
    "mutable_block_contract_sha256",
    "mutable_block_geometry_from_native_row",
    "mutable_block_geometry_from_native_cells",
    "publish_mutable_block_resync",
    "query_mutable_blocks",
]
