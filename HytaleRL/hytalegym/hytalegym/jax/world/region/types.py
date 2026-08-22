"""Fixed leaves for exact multi-chunk geometry lookup."""

from __future__ import annotations

from typing import NamedTuple

import jax

Array = jax.Array


class RegionAtlas(NamedTuple):
    """Padded exact region captures; every capacity is static under JIT."""

    region_mask: Array
    world_id: Array
    core_min_chunk_xz: Array
    section_known: Array
    cell_code: Array
    filler_root_known: Array
    filler_root_offset_packed: Array
    cell_palette_size: Array
    cell_flags: Array
    cell_shape_index: Array
    cell_fluid_level: Array
    cell_fluid_fill_height: Array
    cell_support: Array
    cell_block_damage: Array
    cell_fluid_damage: Array
    cell_movement: Array
    cell_fluid_movement: Array
    shape_palette_size: Array
    collision_boxes: Array
    collision_box_mask: Array


class RegionCellSelection(NamedTuple):
    """Per-query cell semantics and the independently selected region."""

    available: Array
    region_index: Array
    cell_code: Array
    flags: Array
    shape_index: Array
    filler_root_available: Array
    filler_root_offset: Array
    fluid_level: Array
    fluid_fill_height: Array
    support: Array
    block_damage: Array
    fluid_damage: Array
    movement: Array
    fluid_movement: Array
    collision_boxes: Array
    collision_box_mask: Array


class RegionVisibilityResult(NamedTuple):
    visible: Array
    geometry_exhausted: Array
    capacity_exceeded: Array


class RegionGeometryState(NamedTuple):
    """Exact Region v1 geometry plus entity-local collision semantics.

    Region captures intentionally contain only immutable world cells. Entity
    bounds and eye offsets come from the same native reset/ruleset contract,
    while the source-cell offsets are derived on the host from the captured
    shape palettes. Their fixed shape becomes part of the compiled program.
    """

    atlas: RegionAtlas
    environment_world_id: Array
    source_lower_extension: Array
    source_upper_extension: Array
    source_offsets: Array
    source_capacity: Array
    los_step_marker: Array
    agent_bounds: Array
    target_bounds: Array
    agent_los_offset: Array
    target_los_offset: Array
    # Optional per-environment sparse physical overlay. ``None`` preserves
    # the immutable frozen-Region path; bound state must be a synchronized
    # MutableBlockState and is queried fail-closed.
    mutable_blocks: object | None = None


class RegionTraversalAtlas(NamedTuple):
    """Padded native-classified traversal graphs aligned to Region worlds."""

    graph_mask: Array
    world_id: Array
    core_min_chunk_xz: Array
    actor_bounds: Array
    maximum_climb_height: Array
    maximum_drop_height: Array
    query_distance_capacity: Array
    column_offsets: Array
    column_slot: Array
    column_node_start: Array
    column_node_count: Array
    node_mask: Array
    node_position: Array
    node_clearance: Array
    edge_mask: Array
    edge_destination: Array
    edge_cost: Array
    edge_kind: Array
    edge_flags: Array
