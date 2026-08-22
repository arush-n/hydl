"""Fixed JAX leaves for non-authoritative surrogate world geometry."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax

from hytalegym.jax.world.traversal import TraversalTokenObservation

if TYPE_CHECKING:
    from hytalegym.jax.combat.observation.v1.schema.types import InjectedWorldFeatures

Array = jax.Array


class SurrogateAtlas(NamedTuple):
    """Overlapping immutable tiles with one shared semantic palette."""

    tile_mask: Array
    world_id: Array
    core_min_chunk_xz: Array
    seed_words: Array
    generation_status: Array
    capability_bits: Array
    terrain_parameters: Array
    terrain_biome_parameters: Array
    column_known: Array
    terrain_height: Array
    surface_cell_code: Array
    subsurface_cell_code: Array
    biome_code: Array
    cave_span_mask: Array
    cave_min_y: Array
    cave_max_y: Array
    structure_mask: Array
    structure_block_key: Array
    structure_cell_code: Array
    structure_state_slot: Array
    stateful_mask: Array
    stateful_block_key: Array
    stateful_identity_key: Array
    stateful_runtime_slot: Array
    stateful_state_count: Array
    stateful_initial_state: Array
    stateful_yaw: Array
    stateful_variant_cell_code: Array
    stateful_success_target: Array
    stateful_blocked_target: Array
    stateful_transition_mask: Array
    stateful_partner_runtime_slot: Array
    entity_mask: Array
    entity_identity_words: Array
    entity_runtime_slot: Array
    entity_initial_position: Array
    entity_initial_rotation: Array
    entity_initial_velocity: Array
    entity_kind: Array
    entity_type_code: Array
    entity_type_identity_words: Array
    entity_behavior_supported: Array
    entity_geometry_supported: Array
    entity_collidable: Array
    entity_blocks_los: Array
    entity_local_bounds: Array
    entity_los_offset_supported: Array
    entity_los_offset: Array
    entity_type_count: Array
    cell_palette_size: Array
    cell_flags: Array
    cell_shape_index: Array
    cell_fluid_level: Array
    cell_support: Array
    cell_block_damage: Array
    cell_fluid_damage: Array
    cell_movement: Array
    cell_fluid_movement: Array
    shape_palette_size: Array
    collision_boxes: Array
    collision_box_mask: Array


class SurrogateRuntimeState(NamedTuple):
    """Per-environment mutable physical state."""

    environment_world_id: Array
    stateful_state: Array
    stateful_initialized: Array
    entity_active: Array
    entity_initialized: Array
    entity_identity_words: Array
    entity_position: Array
    entity_rotation: Array
    entity_velocity: Array
    entity_kind: Array
    entity_type_code: Array
    entity_type_identity_words: Array
    entity_behavior_supported: Array
    entity_geometry_supported: Array
    entity_collidable: Array
    entity_blocks_los: Array
    entity_local_bounds: Array
    entity_los_offset_supported: Array
    entity_los_offset: Array
    capability_bits: Array
    failure_bits: Array
    unsupported_mechanics: Array


class SurrogateColumnSelection(NamedTuple):
    """Per-query X/Z column coverage and independently selected tile."""

    available: Array
    tile_index: Array
    local_xz: Array


class SurrogateCellSelection(NamedTuple):
    """Per-query effective geometry and independently selected tile."""

    available: Array
    tile_index: Array
    cell_code: Array
    flags: Array
    shape_index: Array
    fluid_level: Array
    support: Array
    block_damage: Array
    fluid_damage: Array
    movement: Array
    fluid_movement: Array
    collision_boxes: Array
    collision_box_mask: Array
    structure_override: Array
    stateful: Array
    stateful_definition_slot: Array
    stateful_runtime_slot: Array


class SurrogateDoorTransitionResult(NamedTuple):
    """One batched door-use transition with explicit failure masks."""

    runtime: SurrogateRuntimeState
    accepted: Array
    state_changed: Array
    partner_state_changed: Array
    blocked: Array
    unsupported: Array


class SurrogateStateChangeResult(NamedTuple):
    """One side-independent ChangeState use with explicit diagnostics."""

    runtime: SurrogateRuntimeState
    previous_state: Array
    next_state: Array
    accepted: Array
    state_changed: Array
    unsupported: Array


class SurrogateEntitySelection(NamedTuple):
    """Fixed slot lookup over per-environment world-owned entity state."""

    available: Array
    active: Array
    identity_words: Array
    position: Array
    rotation: Array
    velocity: Array
    kind: Array
    type_code: Array
    type_identity_words: Array
    behavior_supported: Array
    geometry_supported: Array
    collidable: Array
    blocks_los: Array
    local_bounds: Array
    los_offset_supported: Array
    los_offset: Array


class SurrogateEntityUpdateBatch(NamedTuple):
    """Padded state updates for existing world-absolute entity instances."""

    mask: Array
    runtime_slot: Array
    expected_identity_words: Array
    active: Array
    position: Array
    rotation: Array
    velocity: Array


class SurrogateEntityUpdateResult(NamedTuple):
    """Atomic per-environment entity update result and diagnostics."""

    runtime: SurrogateRuntimeState
    accepted: Array
    applied: Array
    state_changed: Array
    diagnostics: Array
    request_diagnostics: Array


class SurrogateEntityVisibilityResult(NamedTuple):
    """Bounded per-observer entity visibility with native and derived rays."""

    target_mask: Array
    available: Array
    diagnostics: Array
    observer_diagnostics: Array
    capacity_exceeded: Array
    entity_slot: Array
    distance: Array
    native_available: Array
    native_block_visible: Array
    entity_blocked: Array
    target_point_visible: Array
    sample_visible: Array
    visible_fraction: Array
    visible_any: Array
    fully_visible: Array
    partially_visible: Array
    occluded: Array


class SurrogatePerceptionChannelResult(NamedTuple):
    """Fixed raw world channels with independent availability masks."""

    available: Array
    diagnostics: Array
    tile_index: Array
    channel_valid: Array
    heightmap_block_y: Array
    height_above_surface: Array
    sky_light: Array
    block_light_rgb: Array
    environment_code: Array
    tint_rgb: Array


class SurrogateSpawnMarkerCatalog(NamedTuple):
    """Fixed native marker metadata; NPC lifecycle remains unsupported."""

    marker_mask: Array
    marker_identity_words: Array
    asset_sha256_words: Array
    realtime_respawn: Array
    manual_trigger: Array
    exclusion_radius: Array
    maximum_drop_height: Array
    deactivation_distance: Array
    deactivation_seconds: Array
    choice_mask: Array
    role_present: Array
    role_identity_words: Array
    choice_weight: Array
    respawn_seconds: Array
    flock_required: Array


class SurrogateSpawnMarkerSelection(NamedTuple):
    """One bounded weighted metadata lookup, never an NPC spawn transition."""

    available: Array
    choice_index: Array
    role_present: Array
    role_identity_words: Array
    respawn_seconds: Array
    realtime_respawn: Array
    manual_trigger: Array
    flock_required: Array
    behavior_supported: Array


class SurrogateSweepResult(NamedTuple):
    """Fixed-shape swept-AABB query with explicit failure diagnostics."""

    available: Array
    clear: Array
    diagnostics: Array
    visited_cell_count: Array
    hit_block: Array
    hit_entity: Array
    hit_entity_slot: Array


class SurrogateVisibilityResult(NamedTuple):
    visible: Array
    geometry_exhausted: Array
    capacity_exceeded: Array


class SurrogateHitboxLineOfSightResult(NamedTuple):
    """Native selector LOS with non-blocking unloaded-space diagnostics."""

    clear: Array
    geometry_exhausted: Array
    capacity_exceeded: Array
    invalid: Array


class SurrogateTraversalAtlas(NamedTuple):
    """Padded immutable graph tiles aligned with a surrogate atlas."""

    graph_mask: Array
    graph_available: Array
    diagnostics: Array
    world_id: Array
    core_min_chunk_xz: Array
    actor_bounds: Array
    maximum_climb_height: Array
    maximum_safe_drop_height: Array
    drop_safety_supported: Array
    column_node_start: Array
    column_node_count: Array
    node_mask: Array
    node_position: Array
    node_clearance: Array
    node_support_key: Array
    node_flags: Array
    node_gate_runtime_slot: Array
    node_gate_state_mask: Array
    edge_mask: Array
    edge_destination: Array
    edge_cost: Array
    edge_kind: Array
    edge_flags: Array
    edge_gate_runtime_slot: Array
    edge_gate_state_mask: Array


class SurrogateWorldFeatureResult(NamedTuple):
    """Privileged world candidates plus explicit production diagnostics."""

    features: InjectedWorldFeatures
    valid: Array
    diagnostics: Array
    graph_diagnostics: Array
    terrain_source_tile_index: Array
    interaction_source_tile_index: Array
    traversal_provider_available: Array


__all__ = [
    "SurrogateAtlas",
    "SurrogateCellSelection",
    "SurrogateColumnSelection",
    "SurrogateDoorTransitionResult",
    "SurrogateStateChangeResult",
    "SurrogateEntitySelection",
    "SurrogateEntityUpdateBatch",
    "SurrogateEntityUpdateResult",
    "SurrogateEntityVisibilityResult",
    "SurrogateHitboxLineOfSightResult",
    "SurrogatePerceptionChannelResult",
    "SurrogateRuntimeState",
    "SurrogateSpawnMarkerCatalog",
    "SurrogateSpawnMarkerSelection",
    "SurrogateSweepResult",
    "SurrogateTraversalAtlas",
    "SurrogateVisibilityResult",
    "SurrogateWorldFeatureResult",
    "TraversalTokenObservation",
]
