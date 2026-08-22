"""Actor-legal block affordances aligned to existing geometry tokens."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.geometry.contract import CELL_COUNT, CELL_RADIUS
from hytalegym.jax.world.block_actions import (
    BLOCK_ACTION_DIAGNOSTIC_CATALOG_DUPLICATE,
    BLOCK_ACTION_DIAGNOSTIC_CATALOG_MISSING,
    BLOCK_ACTION_DIAGNOSTIC_INVALID,
    BLOCK_ACTION_DIAGNOSTIC_TARGET_UNAVAILABLE,
    BlockActionTargetResult,
    BlockResolvedDrops,
    block_action_target_contract_sha256,
)
from hytalegym.jax.world.perception.los import (
    GeometryProvider,
    geometry_perception_target_cell_visible_result,
    geometry_provider_los_contract_sha256,
)
from hytalegym.jax.world.mutable_blocks import (
    BLOCK_SEMANTIC_KEY_WORDS,
    MutableBlockQueryResult,
    mutable_block_contract_sha256,
)
from hytalegym.jax.world.perception import (
    NATIVE_VIEW_MAX_FULL_ANGLE_RADIANS,
    native_view_sector,
)
from hytalegym.jax.world.region.types import RegionGeometryState
from hytalegym.jax.world.tokens import (
    WORLD_TOKEN_KIND_COLLISION_EXCEPTION,
    WORLD_TOKEN_KIND_TRAVERSAL_SURFACE,
    WorldGeometryTokenObservation,
    world_geometry_token_contract_sha256,
)
from hytalegym.jax.world.types import GeometryState
from hytalegym.worldgen.block_affordances import (
    block_affordance_dictionary_sha256,
)


Array = jax.Array
ACTOR_BLOCK_AFFORDANCE_TOKEN_SCHEMA = (
    "hytalerl_actor_block_affordance_tokens_v1"
)
ACTOR_BLOCK_AFFORDANCE_TOKEN_VERSION = 1

ACTOR_BLOCK_TOKEN_DIAGNOSTIC_INVALID_KIND = jnp.uint32(1 << 16)
ACTOR_BLOCK_TOKEN_DIAGNOSTIC_INVALID_SOURCE = jnp.uint32(1 << 17)
ACTOR_BLOCK_TOKEN_DIAGNOSTIC_CELL_UNAVAILABLE = jnp.uint32(1 << 18)
ACTOR_BLOCK_TOKEN_DIAGNOSTIC_BLOCK_ABSENT = jnp.uint32(1 << 19)
ACTOR_BLOCK_TOKEN_DIAGNOSTIC_AFFORDANCE_UNAVAILABLE = jnp.uint32(1 << 20)

ACTOR_BLOCK_ACTION_CANDIDATE_SCHEMA = (
    "hytalerl_actor_block_action_candidates_v1"
)
ACTOR_BLOCK_ACTION_CANDIDATE_VERSION = 1
ACTOR_BLOCK_ACTION_DIAGNOSTIC_INVALID_ACTOR = jnp.uint32(1)
ACTOR_BLOCK_ACTION_DIAGNOSTIC_QUERY_UNAVAILABLE = jnp.uint32(1 << 1)
ACTOR_BLOCK_ACTION_DIAGNOSTIC_LOS_UNAVAILABLE = jnp.uint32(1 << 2)
ACTOR_BLOCK_ACTION_DIAGNOSTIC_AFFORDANCE_UNAVAILABLE = jnp.uint32(1 << 3)
ACTOR_BLOCK_ACTION_DIAGNOSTIC_OUTPUT_CAPACITY = jnp.uint32(1 << 4)
ACTOR_BLOCK_ACTION_DIAGNOSTIC_SOURCE_INCOMPLETE = jnp.uint32(1 << 5)
ACTOR_BLOCK_ACTION_DIAGNOSTIC_ALIAS_CONFLICT = jnp.uint32(1 << 6)

ACTOR_BLOCK_ACTION_PARAMETER_SCHEMA = (
    "hytalerl_actor_block_action_parameters_v2"
)
ACTOR_BLOCK_ACTION_PARAMETER_VERSION = 2
ACTOR_BLOCK_PARAMETER_DIAGNOSTIC_POSITION = jnp.uint32(1 << 24)
ACTOR_BLOCK_PARAMETER_DIAGNOSTIC_QUERY = jnp.uint32(1 << 25)
ACTOR_BLOCK_PARAMETER_DIAGNOSTIC_ACTION = jnp.uint32(1 << 26)
_BLOCK_ACTION_PARAMETER_FATAL_DIAGNOSTICS = jnp.uint32(
    BLOCK_ACTION_DIAGNOSTIC_INVALID
    | BLOCK_ACTION_DIAGNOSTIC_TARGET_UNAVAILABLE
    | BLOCK_ACTION_DIAGNOSTIC_CATALOG_MISSING
    | BLOCK_ACTION_DIAGNOSTIC_CATALOG_DUPLICATE
)

_SUPPORT_EPSILON = jnp.float32(1.0e-4)
_SIDE = CELL_RADIUS * 2 + 1


class ActorBlockTokenCells(NamedTuple):
    """Exact block cells associated with actor-legal geometry tokens."""

    available: Array
    diagnostics: Array
    token_mask: Array
    position: Array


class ActorBlockAffordanceTokens(NamedTuple):
    """Policy-ready affordances; verification identity remains excluded."""

    available: Array
    diagnostics: Array
    token_mask: Array
    affordance_tags: Array
    gather_type_index: Array
    required_tool_quality: Array
    provenance: Array


class ActorBlockActionCandidates(NamedTuple):
    """Actor-visible actionable blocks, including ordinary full cubes."""

    available: Array
    capacity_exceeded: Array
    diagnostics: Array
    candidate_mask: Array
    visible_position: Array
    action_position: Array
    visible_relative_position: Array
    affordance_tags: Array
    gather_type_index: Array
    required_tool_quality: Array
    provenance: Array


class ActorBlockActionParameters(NamedTuple):
    """Exact mutable values aligned to actor-visible block candidates."""

    available: Array
    diagnostics: Array
    candidate_mask: Array
    block_health: Array
    block_health_valid: Array
    break_available: Array
    break_allowed: Array
    tool_compatible: Array
    tool_power: Array
    interaction_tool_route_matched: Array
    break_removes_block: Array
    replacement_semantic_key_valid: Array
    replacement_semantic_key: Array
    harvest_available: Array
    harvest_allowed: Array
    target_harvestable: Array
    place_available: Array
    place_allowed: Array
    break_drops: BlockResolvedDrops
    harvest_drops: BlockResolvedDrops


def dense_actor_block_candidate_cells(
    actor_position: Array,
    *,
    cell_radius: int,
) -> Array:
    """Return a static dense cell stencil around each actor.

    ``cell_radius`` is deliberately caller-owned: native interaction reach
    belongs to the authored action contract, not to World.
    """

    radius = _static_cell_radius(cell_radius)
    positions = jnp.asarray(actor_position, dtype=jnp.float32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("actor_position must have shape [B, A, 3]")
    finite = jnp.all(jnp.isfinite(positions), axis=2)
    origin = jnp.floor(
        jnp.where(finite[..., None], positions, 0.0)
    ).astype(jnp.int32)
    offsets = jnp.asarray(
        [
            (x, y, z)
            for x in range(-radius, radius + 1)
            for y in range(-radius, radius + 1)
            for z in range(-radius, radius + 1)
        ],
        dtype=jnp.int32,
    )
    return origin[:, :, None, :] + offsets


def dense_actor_block_candidate_source_complete(
    actor_position: Array,
    maximum_distance: Array | float,
    *,
    cell_radius: int,
) -> Array:
    """Prove a dense stencil contains every cell center inside reach."""

    radius = jnp.float32(_static_cell_radius(cell_radius))
    positions = jnp.asarray(actor_position, dtype=jnp.float32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("actor_position must have shape [B, A, 3]")
    batch, actors, _ = positions.shape
    maximum = _actor_parameter(
        maximum_distance,
        batch,
        actors,
        "maximum_distance",
    )
    finite = jnp.all(jnp.isfinite(positions), axis=2)
    safe = jnp.where(finite[..., None], positions, 0.0)
    fraction = safe - jnp.floor(safe)
    nearest_squared = (jnp.float32(0.5) - fraction) ** 2
    other_squared = (
        jnp.sum(nearest_squared, axis=2, keepdims=True) - nearest_squared
    )
    lower_squared = (
        fraction + radius + jnp.float32(0.5)
    ) ** 2 + other_squared
    upper_squared = (
        radius + jnp.float32(1.5) - fraction
    ) ** 2 + other_squared
    nearest_omitted_squared = jnp.min(
        jnp.minimum(lower_squared, upper_squared),
        axis=2,
    )
    return (
        finite
        & jnp.isfinite(maximum)
        & (maximum > 0.0)
        & (maximum**2 < nearest_omitted_squared)
    )


def produce_actor_block_action_candidates(
    geometry: GeometryProvider,
    actor_position: Array,
    actor_eye_position: Array,
    actor_forward: Array,
    visible_position: Array,
    action_position: Array,
    targets: MutableBlockQueryResult,
    *,
    source_complete: Array,
    role_opaque_mask: Array,
    candidate_capacity: int,
    maximum_distance: Array | float,
    view_sector_full_angle_radians: Array | float,
    max_los_cells: int | None = None,
) -> ActorBlockActionCandidates:
    """Select a complete fixed-capacity row of visible actionable blocks.

    ``visible_position`` contains clicked cells used for FOV and LOS.
    ``action_position`` contains filler-canonicalized mutation cells, and
    ``targets`` is their exact mutable overlay flattened in actor-major order.
    ``source_complete`` certifies that the fixed source covers the caller's
    authored reach. Unknown coverage, cells, semantics, LOS, or capacity clear
    the complete actor row.
    """

    if not isinstance(geometry, (GeometryState, RegionGeometryState)):
        raise TypeError("geometry must be GeometryState or RegionGeometryState")
    if not isinstance(targets, MutableBlockQueryResult):
        raise TypeError("targets must be MutableBlockQueryResult")
    if (
        isinstance(candidate_capacity, bool)
        or not isinstance(candidate_capacity, int)
        or candidate_capacity <= 0
    ):
        raise ValueError("candidate_capacity must be a positive integer")

    actors = jnp.asarray(actor_position, dtype=jnp.float32)
    eyes = jnp.asarray(actor_eye_position, dtype=jnp.float32)
    forward = jnp.asarray(actor_forward, dtype=jnp.float32)
    visible_cells = jnp.asarray(visible_position, dtype=jnp.int32)
    action_cells = jnp.asarray(action_position, dtype=jnp.int32)
    if (
        actors.ndim != 3
        or actors.shape[-1] != 3
        or eyes.shape != actors.shape
        or forward.shape != actors.shape
        or visible_cells.ndim != 4
        or visible_cells.shape[:2] != actors.shape[:2]
        or visible_cells.shape[-1] != 3
        or action_cells.shape != visible_cells.shape
    ):
        raise ValueError(
            "actor values must be [B, A, 3] and block cells [B, A, S, 3]"
        )
    batch, actor_count, source_capacity = visible_cells.shape[:3]
    if candidate_capacity > source_capacity:
        raise ValueError("candidate_capacity exceeds the source stencil")
    query_shape = (batch, actor_count * source_capacity)
    if targets.available.shape != query_shape:
        raise ValueError(
            "targets must flatten target_position in actor-major order"
        )
    _validate_candidate_query_shapes(targets, query_shape)
    _validate_geometry_batch(geometry, batch)
    complete_source = jnp.asarray(source_complete)
    if (
        complete_source.dtype != jnp.bool_
        or complete_source.shape != (batch, actor_count)
    ):
        raise ValueError("source_complete must be boolean with shape [B, A]")
    role_mask = _validated_actor_role_mask(
        geometry,
        role_opaque_mask,
        batch,
        actor_count,
    )
    maximum = _actor_parameter(
        maximum_distance,
        batch,
        actor_count,
        "maximum_distance",
    )
    view_angle = _actor_parameter(
        view_sector_full_angle_radians,
        batch,
        actor_count,
        "view_sector_full_angle_radians",
    )

    forward_xz = forward[..., (0, 2)]
    forward_norm = jnp.linalg.norm(forward_xz, axis=2)
    actor_valid = (
        jnp.all(
            jnp.isfinite(actors) & jnp.isfinite(eyes) & jnp.isfinite(forward),
            axis=2,
        )
        & jnp.isfinite(maximum)
        & (maximum > 0.0)
        & jnp.isfinite(view_angle)
        & (view_angle >= 0.0)
        & (
            view_angle
            <= jnp.float32(NATIVE_VIEW_MAX_FULL_ANGLE_RADIANS)
        )
        & (forward_norm > jnp.float32(1.0e-6))
    )
    safe_actors = jnp.where(actor_valid[..., None], actors, 0.0)
    safe_eyes = jnp.where(actor_valid[..., None], eyes, 0.0)
    normalized_forward = jnp.where(
        actor_valid[..., None],
        forward_xz / jnp.maximum(forward_norm[..., None], 1.0e-6),
        jnp.asarray((0.0, 1.0), dtype=jnp.float32),
    )

    target_center = visible_cells.astype(jnp.float32) + 0.5
    relative = target_center - safe_actors[:, :, None, :]
    distance_squared = jnp.sum(relative * relative, axis=3)
    pre_sector = (
        actor_valid[..., None]
        & (distance_squared <= maximum[..., None] ** 2)
        & native_view_sector(
            relative[..., (0, 2)],
            normalized_forward[:, :, None, :],
            view_angle[:, :, None],
        )
    )

    def rows(value: Array) -> Array:
        return value.reshape(
            (batch, actor_count, source_capacity) + value.shape[2:]
        )

    query_available = rows(targets.available) & ~rows(
        targets.invalid | targets.resync_required
    )
    present = rows(targets.geometry.block_present)
    exact = rows(targets.geometry.exact)
    semantic = rows(targets.geometry.semantic_key_valid)
    semantic_key = rows(targets.geometry.semantic_key)
    affordance = rows(targets.geometry.affordance_valid)
    tags = rows(targets.geometry.affordance_tags)
    gather = rows(targets.geometry.gather_type_index)
    quality = rows(targets.geometry.required_tool_quality)
    provenance = rows(targets.provenance)

    query_unavailable = pre_sector & ~query_available
    los_mask = pre_sector & query_available & present
    visible, los_unavailable = _candidate_visibility(
        geometry,
        safe_eyes,
        target_center,
        los_mask,
        role_mask,
        max_los_cells=max_los_cells,
    )
    semantics_unavailable = visible & (
        ~exact | ~semantic | ~affordance
    )
    eligible = (
        visible
        & exact
        & semantic
        & affordance
        & (tags != 0)
    )

    maximum_int = jnp.iinfo(jnp.int32).max
    source_index = jnp.broadcast_to(
        jnp.arange(source_capacity, dtype=jnp.int32),
        eligible.shape,
    )
    grouped_values = jax.lax.sort(
        (
            jnp.where(eligible, action_cells[..., 0], maximum_int),
            jnp.where(eligible, action_cells[..., 1], maximum_int),
            jnp.where(eligible, action_cells[..., 2], maximum_int),
            jnp.where(eligible, distance_squared, jnp.inf),
            jnp.where(eligible, visible_cells[..., 0], maximum_int),
            jnp.where(eligible, visible_cells[..., 1], maximum_int),
            jnp.where(eligible, visible_cells[..., 2], maximum_int),
            source_index,
        ),
        dimension=2,
        is_stable=True,
        num_keys=7,
    )
    grouped_index = grouped_values[-1]
    grouped_eligible = _gather_actor_candidates(eligible, grouped_index)
    grouped_cells = _gather_actor_candidates(action_cells, grouped_index)
    grouped_semantic_key = _gather_actor_candidates(
        semantic_key,
        grouped_index,
    )
    grouped_tags = _gather_actor_candidates(tags, grouped_index)
    grouped_gather = _gather_actor_candidates(gather, grouped_index)
    grouped_quality = _gather_actor_candidates(quality, grouped_index)
    grouped_provenance = _gather_actor_candidates(
        provenance,
        grouped_index,
    )
    same_previous = jnp.all(
        grouped_cells[:, :, 1:, :] == grouped_cells[:, :, :-1, :],
        axis=3,
    )
    duplicate_pair = (
        same_previous
        & grouped_eligible[:, :, 1:]
        & grouped_eligible[:, :, :-1]
    )
    alias_conflict = jnp.any(
        duplicate_pair
        & (
            jnp.any(
                grouped_semantic_key[:, :, 1:, :]
                != grouped_semantic_key[:, :, :-1, :],
                axis=3,
            )
            | (
                grouped_tags[:, :, 1:]
                != grouped_tags[:, :, :-1]
            )
            | (
                grouped_gather[:, :, 1:]
                != grouped_gather[:, :, :-1]
            )
            | (
                grouped_quality[:, :, 1:]
                != grouped_quality[:, :, :-1]
            )
            | (
                grouped_provenance[:, :, 1:]
                != grouped_provenance[:, :, :-1]
            )
        ),
        axis=2,
    )
    unique = grouped_eligible.at[:, :, 1:].set(
        grouped_eligible[:, :, 1:]
        & ~(same_previous & grouped_eligible[:, :, :-1])
    )
    visible_count = jnp.sum(unique, axis=2)
    capacity_exceeded = visible_count > candidate_capacity
    grouped_distance = _gather_actor_candidates(
        distance_squared,
        grouped_index,
    )
    grouped_visible = _gather_actor_candidates(
        visible_cells,
        grouped_index,
    )
    ranked_values = jax.lax.sort(
        (
            jnp.where(unique, grouped_distance, jnp.inf),
            jnp.where(unique, grouped_cells[..., 0], maximum_int),
            jnp.where(unique, grouped_cells[..., 1], maximum_int),
            jnp.where(unique, grouped_cells[..., 2], maximum_int),
            jnp.where(unique, grouped_visible[..., 0], maximum_int),
            jnp.where(unique, grouped_visible[..., 1], maximum_int),
            jnp.where(unique, grouped_visible[..., 2], maximum_int),
            grouped_index,
        ),
        dimension=2,
        is_stable=True,
        num_keys=7,
    )
    selected_index = ranked_values[-1][:, :, :candidate_capacity]
    selected_unique = jnp.isfinite(
        ranked_values[0][:, :, :candidate_capacity]
    )

    row_available = (
        actor_valid
        & complete_source
        & ~alias_conflict
        & ~jnp.any(query_unavailable, axis=2)
        & ~jnp.any(los_unavailable, axis=2)
        & ~jnp.any(semantics_unavailable, axis=2)
        & ~capacity_exceeded
    )
    candidate_mask = selected_unique & row_available[..., None]
    diagnostics = (
        jnp.where(
            ~actor_valid,
            ACTOR_BLOCK_ACTION_DIAGNOSTIC_INVALID_ACTOR,
            jnp.uint32(0),
        )
        | jnp.where(
            ~complete_source,
            ACTOR_BLOCK_ACTION_DIAGNOSTIC_SOURCE_INCOMPLETE,
            jnp.uint32(0),
        )
        | jnp.where(
            alias_conflict,
            ACTOR_BLOCK_ACTION_DIAGNOSTIC_ALIAS_CONFLICT,
            jnp.uint32(0),
        )
        | jnp.where(
            jnp.any(query_unavailable, axis=2),
            ACTOR_BLOCK_ACTION_DIAGNOSTIC_QUERY_UNAVAILABLE,
            jnp.uint32(0),
        )
        | jnp.where(
            jnp.any(los_unavailable, axis=2),
            ACTOR_BLOCK_ACTION_DIAGNOSTIC_LOS_UNAVAILABLE,
            jnp.uint32(0),
        )
        | jnp.where(
            jnp.any(semantics_unavailable, axis=2),
            ACTOR_BLOCK_ACTION_DIAGNOSTIC_AFFORDANCE_UNAVAILABLE,
            jnp.uint32(0),
        )
        | jnp.where(
            capacity_exceeded,
            ACTOR_BLOCK_ACTION_DIAGNOSTIC_OUTPUT_CAPACITY,
            jnp.uint32(0),
        )
    )

    def selected(value: Array) -> Array:
        return _gather_actor_candidates(value, selected_index)

    def masked(value: Array) -> Array:
        value = selected(value)
        gate = candidate_mask.reshape(
            candidate_mask.shape
            + (1,) * (value.ndim - candidate_mask.ndim)
        )
        return jnp.where(gate, value, jnp.zeros_like(value))

    return ActorBlockActionCandidates(
        available=row_available,
        capacity_exceeded=capacity_exceeded,
        diagnostics=diagnostics,
        candidate_mask=candidate_mask,
        visible_position=masked(visible_cells),
        action_position=masked(action_cells),
        visible_relative_position=masked(relative),
        affordance_tags=masked(tags),
        gather_type_index=masked(gather),
        required_tool_quality=masked(quality),
        provenance=masked(provenance),
    )


def publish_actor_block_action_parameters(
    candidates: ActorBlockActionCandidates,
    queried_position: Array,
    targets: MutableBlockQueryResult,
    actions: BlockActionTargetResult,
) -> ActorBlockActionParameters:
    """Align health, tool legality, and resolved quantities to candidates.

    The caller must query ``candidates.action_position`` flattened in
    actor-major order. Position equality is checked before any value is
    published, so a stale or permuted query clears the complete actor row.
    """

    if not isinstance(candidates, ActorBlockActionCandidates):
        raise TypeError("candidates must be ActorBlockActionCandidates")
    if not isinstance(targets, MutableBlockQueryResult):
        raise TypeError("targets must be MutableBlockQueryResult")
    if not isinstance(actions, BlockActionTargetResult):
        raise TypeError("actions must be BlockActionTargetResult")
    batch, actors, capacity = candidates.candidate_mask.shape
    flat_shape = (batch, actors * capacity)
    queried = jnp.asarray(queried_position, dtype=jnp.int32)
    if queried.shape != (batch, actors, capacity, 3):
        raise ValueError(
            "queried_position must have shape [batch,actor,candidate,3]"
        )
    _validate_parameter_query_shapes(targets, actions, flat_shape)

    def rows(value: Array) -> Array:
        return value.reshape(
            (batch, actors, capacity) + value.shape[2:]
        )

    required = candidates.candidate_mask
    position_match = jnp.all(
        queried == candidates.action_position,
        axis=3,
    )
    target_available = (
        rows(targets.available)
        & rows(targets.geometry.block_present)
        & rows(targets.geometry.exact)
        & rows(targets.geometry.semantic_key_valid)
        & rows(targets.block_health_valid)
        & ~rows(targets.invalid | targets.resync_required)
        & rows(actions.target_available)
    )
    missing_position = required & ~position_match
    missing_query = required & ~target_available
    action_diagnostic = required & (
        (
            rows(actions.diagnostics)
            & _BLOCK_ACTION_PARAMETER_FATAL_DIAGNOSTICS
        )
        != 0
    )
    row_available = (
        candidates.available
        & ~jnp.any(
            missing_position | missing_query | action_diagnostic,
            axis=2,
        )
    )
    candidate_mask = required & row_available[..., None]
    diagnostics = (
        candidates.diagnostics
        | jnp.where(
            jnp.any(missing_position, axis=2),
            ACTOR_BLOCK_PARAMETER_DIAGNOSTIC_POSITION,
            jnp.uint32(0),
        )
        | jnp.where(
            jnp.any(missing_query, axis=2),
            ACTOR_BLOCK_PARAMETER_DIAGNOSTIC_QUERY,
            jnp.uint32(0),
        )
        | jnp.where(
            jnp.any(action_diagnostic, axis=2),
            ACTOR_BLOCK_PARAMETER_DIAGNOSTIC_ACTION,
            jnp.uint32(0),
        )
    )

    def masked(value: Array) -> Array:
        shaped = rows(value)
        gate = candidate_mask.reshape(
            candidate_mask.shape
            + (1,) * (shaped.ndim - candidate_mask.ndim)
        )
        return jnp.where(gate, shaped, jnp.zeros_like(shaped))

    def drops(value: BlockResolvedDrops) -> BlockResolvedDrops:
        return BlockResolvedDrops(*(masked(field) for field in value))

    return ActorBlockActionParameters(
        available=row_available,
        diagnostics=diagnostics,
        candidate_mask=candidate_mask,
        block_health=masked(targets.block_health),
        block_health_valid=masked(targets.block_health_valid),
        break_available=masked(actions.break_available),
        break_allowed=masked(actions.break_allowed),
        tool_compatible=masked(actions.tool_compatible),
        tool_power=masked(actions.tool_power),
        interaction_tool_route_matched=masked(
            actions.interaction_tool_route_matched
        ),
        break_removes_block=masked(actions.break_removes_block),
        replacement_semantic_key_valid=masked(
            actions.replacement_semantic_key_valid
        ),
        replacement_semantic_key=masked(
            actions.replacement_semantic_key
        ),
        harvest_available=masked(actions.harvest_available),
        harvest_allowed=masked(actions.harvest_allowed),
        target_harvestable=masked(actions.target_harvestable),
        place_available=masked(actions.place_available),
        place_allowed=masked(actions.place_allowed),
        break_drops=drops(actions.break_drops),
        harvest_drops=drops(actions.harvest_drops),
    )


def world_geometry_token_block_cells(
    tokens: WorldGeometryTokenObservation,
    geometry: GeometryState | RegionGeometryState,
    actor_position: Array,
) -> ActorBlockTokenCells:
    """Map selected token kinds to their exact native block cell.

    Collision exceptions retain the provider cell selected by the token
    producer. Traversal nodes map to the supporting cell immediately below
    the actor-feet position, matching the standable-surface graph.
    """

    if not isinstance(tokens, WorldGeometryTokenObservation):
        raise TypeError("tokens must be WorldGeometryTokenObservation")
    if not isinstance(geometry, (GeometryState, RegionGeometryState)):
        raise TypeError("geometry must be GeometryState or RegionGeometryState")
    positions = jnp.asarray(actor_position, dtype=jnp.float32)
    shape = tokens.token_mask.shape
    if (
        positions.shape != shape[:2] + (3,)
        or tokens.source_index.shape != shape
        or tokens.relative_position.shape != shape + (3,)
    ):
        raise ValueError("token and actor shapes are inconsistent")
    batch, actors, _ = positions.shape
    source = tokens.source_index.astype(jnp.int32)
    source_valid = (source >= 0) & (source < CELL_COUNT)
    safe_source = jnp.clip(source, 0, CELL_COUNT - 1)
    offsets = _cell_offsets(safe_source)

    if isinstance(geometry, RegionGeometryState):
        exception_cell = (
            jnp.floor(positions).astype(jnp.int32)[:, :, None, :]
            + offsets
        )
    else:
        origins = geometry.origin
        if origins.shape[0] == 1 and batch != 1:
            origins = jnp.broadcast_to(origins, (batch, 3))
        if origins.shape != (batch, 3):
            raise ValueError("geometry batch must be one or match actors")
        exception_cell = origins[:, None, None, :].astype(jnp.int32) + offsets

    traversal_position = positions[:, :, None, :] + tokens.relative_position
    traversal_cell = jnp.floor(
        traversal_position
        - jnp.asarray((0.0, _SUPPORT_EPSILON, 0.0), dtype=jnp.float32)
    ).astype(jnp.int32)
    collision = (
        tokens.token_kind == jnp.uint8(WORLD_TOKEN_KIND_COLLISION_EXCEPTION)
    )
    traversal = (
        tokens.token_kind == jnp.uint8(WORLD_TOKEN_KIND_TRAVERSAL_SURFACE)
    )
    known_kind = collision | traversal
    cell = jnp.where(collision[..., None], exception_cell, traversal_cell)
    invalid_kind = tokens.token_mask & ~known_kind
    invalid_source = tokens.token_mask & collision & ~source_valid
    invalid = invalid_kind | invalid_source
    available = tokens.available & ~jnp.any(invalid, axis=2)
    token_mask = tokens.token_mask & available[..., None]
    diagnostics = (
        tokens.diagnostics
        | jnp.where(
            jnp.any(invalid_kind, axis=2),
            ACTOR_BLOCK_TOKEN_DIAGNOSTIC_INVALID_KIND,
            jnp.uint32(0),
        )
        | jnp.where(
            jnp.any(invalid_source, axis=2),
            ACTOR_BLOCK_TOKEN_DIAGNOSTIC_INVALID_SOURCE,
            jnp.uint32(0),
        )
    )
    return ActorBlockTokenCells(
        available=available,
        diagnostics=diagnostics,
        token_mask=token_mask,
        position=jnp.where(token_mask[..., None], cell, 0),
    )


def publish_actor_block_affordance_tokens(
    cells: ActorBlockTokenCells,
    blocks: MutableBlockQueryResult,
) -> ActorBlockAffordanceTokens:
    """Publish stable tags only when every selected block value is exact."""

    if not isinstance(cells, ActorBlockTokenCells):
        raise TypeError("cells must be ActorBlockTokenCells")
    if not isinstance(blocks, MutableBlockQueryResult):
        raise TypeError("blocks must be MutableBlockQueryResult")
    batch, actors, capacity = cells.token_mask.shape
    flat_shape = (batch, actors * capacity)
    if blocks.available.shape != flat_shape:
        raise ValueError(
            "blocks must query flattened actor-token cells with shape "
            f"{flat_shape}"
        )

    def rows(value: Array) -> Array:
        return value.reshape((batch, actors, capacity) + value.shape[2:])

    available = rows(blocks.available)
    present = rows(blocks.geometry.block_present)
    exact = rows(blocks.geometry.exact)
    semantic = rows(blocks.geometry.semantic_key_valid)
    affordance = rows(blocks.geometry.affordance_valid)
    invalid = rows(blocks.invalid | blocks.resync_required)
    valid = available & present & exact & semantic & affordance & ~invalid
    required = cells.token_mask
    row_available = cells.available & jnp.all(~required | valid, axis=2)
    token_mask = required & row_available[..., None]
    missing_cell = required & (~available | invalid)
    missing_block = required & available & ~present
    missing_affordance = required & available & present & (
        ~exact | ~semantic | ~affordance
    )
    diagnostics = (
        cells.diagnostics
        | jnp.where(
            jnp.any(missing_cell, axis=2),
            ACTOR_BLOCK_TOKEN_DIAGNOSTIC_CELL_UNAVAILABLE,
            jnp.uint32(0),
        )
        | jnp.where(
            jnp.any(missing_block, axis=2),
            ACTOR_BLOCK_TOKEN_DIAGNOSTIC_BLOCK_ABSENT,
            jnp.uint32(0),
        )
        | jnp.where(
            jnp.any(missing_affordance, axis=2),
            ACTOR_BLOCK_TOKEN_DIAGNOSTIC_AFFORDANCE_UNAVAILABLE,
            jnp.uint32(0),
        )
    )

    def masked(value: Array) -> Array:
        shaped = rows(value)
        gate = token_mask.reshape(
            token_mask.shape + (1,) * (shaped.ndim - token_mask.ndim)
        )
        return jnp.where(gate, shaped, jnp.zeros_like(shaped))

    return ActorBlockAffordanceTokens(
        available=row_available,
        diagnostics=diagnostics,
        token_mask=token_mask,
        affordance_tags=masked(blocks.geometry.affordance_tags),
        gather_type_index=masked(blocks.geometry.gather_type_index),
        required_tool_quality=masked(
            blocks.geometry.required_tool_quality
        ),
        provenance=masked(blocks.provenance),
    )


def actor_block_affordance_token_contract() -> dict[str, object]:
    """Return the staged companion contract for the batched policy move."""

    return {
        "schema": ACTOR_BLOCK_AFFORDANCE_TOKEN_SCHEMA,
        "version": ACTOR_BLOCK_AFFORDANCE_TOKEN_VERSION,
        "base_actor_token_contract_sha256": (
            world_geometry_token_contract_sha256()
        ),
        "affordance_dictionary_sha256": (
            block_affordance_dictionary_sha256()
        ),
        "shape": {
            "row": ["batch", "actor"],
            "token": ["batch", "actor", "base_token_capacity"],
            "block_query": [
                "batch",
                "actor_times_base_token_capacity",
            ],
        },
        "cell_mapping": {
            "collision_exception": "exact_provider_source_cell",
            "traversal_surface": (
                "floor(actor_feet_world_position_minus_y_epsilon)"
            ),
        },
        "policy_fields": [
            "available",
            "token_mask",
            "affordance_tags",
            "gather_type_index",
            "required_tool_quality",
        ],
        "non_policy_fields": ["diagnostics", "provenance"],
        "identity": "semantic_key_required_for_verification_not_published",
        "actor_legality": (
            "inherits_selected_current_frame_FOV_and_LOS_gates_from_base"
        ),
        "coverage": (
            "base_geometry_tokens_only_not_an_exhaustive_full_cube_action_row"
        ),
        "full_cube_action_candidates_contract_sha256": (
            actor_block_action_candidate_contract_sha256()
        ),
        "fail_closed": (
            "clear_complete_actor_row_on_missing_nonexact_or_invalid_block"
        ),
        "policy_binding": (
            "staged_for_single_announced_action_surface_contract_move"
        ),
    }


def actor_block_affordance_token_contract_sha256() -> str:
    payload = json.dumps(
        actor_block_affordance_token_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def actor_block_action_candidate_contract() -> dict[str, object]:
    """Return the staged exhaustive block-action candidate contract."""

    return {
        "schema": ACTOR_BLOCK_ACTION_CANDIDATE_SCHEMA,
        "version": ACTOR_BLOCK_ACTION_CANDIDATE_VERSION,
        "mutable_block_contract_sha256": mutable_block_contract_sha256(),
        "geometry_provider_los_contract_sha256": (
            geometry_provider_los_contract_sha256()
        ),
        "affordance_dictionary_sha256": (
            block_affordance_dictionary_sha256()
        ),
        "source": {
            "cells": (
                "caller_fixed_clicked_cells_and_separate_"
                "filler_canonicalized_action_targets"
            ),
            "reach": "caller_authored_not_hardcoded_by_world",
            "mutable_state": "exact_overlay_before_publication",
            "completeness": "explicit_per_actor_required",
            "dense_stencil": (
                "available_fallback_with_exact_nearest_omitted_cell_proof"
            ),
        },
        "actor_legality": {
            "view": "native_horizontal_full_width_sector",
            "los": "certified_perception_target_cell_visible",
            "target_point": "clicked_block_cell_center",
        },
        "selection": (
            "nearest_visible_child_per_action_position_then_distance_xyz"
        ),
        "fixed_capacity": (
            "count_all_actor_legal_actionable_cells_then_clear_row_on_overflow"
        ),
        "logical_output_bytes_per_actor": (
            "6_plus_43_times_candidate_capacity"
        ),
        "policy_fields": [
            "available",
            "candidate_mask",
            "visible_relative_position",
            "affordance_tags",
            "gather_type_index",
            "required_tool_quality",
        ],
        "non_policy_fields": [
            "capacity_exceeded",
            "diagnostics",
            "visible_position",
            "action_position",
            "provenance",
        ],
        "identity": "semantic_key_required_for_query_not_published",
        "fail_closed": [
            "invalid_actor",
            "source_incomplete",
            "canonical_action_alias_conflict",
            "visible_query_unavailable",
            "los_unavailable",
            "visible_affordance_unavailable",
            "output_capacity",
        ],
        "policy_binding": (
            "staged_for_single_announced_action_surface_contract_move"
        ),
    }


def actor_block_action_candidate_contract_sha256() -> str:
    payload = json.dumps(
        actor_block_action_candidate_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def actor_block_action_parameter_contract() -> dict[str, object]:
    """Return the additive selected-candidate parameter contract."""

    return {
        "schema": ACTOR_BLOCK_ACTION_PARAMETER_SCHEMA,
        "version": ACTOR_BLOCK_ACTION_PARAMETER_VERSION,
        "candidate_contract_sha256": (
            actor_block_action_candidate_contract_sha256()
        ),
        "action_target_contract_sha256": (
            block_action_target_contract_sha256()
        ),
        "shape": {
            "row": ["batch", "actor"],
            "candidate": ["batch", "actor", "candidate_capacity"],
            "drop": [
                "batch",
                "actor",
                "candidate_capacity",
                "drop_capacity",
            ],
        },
        "alignment": (
            "requery_candidate_action_position_in_actor_major_order_"
            "and_require_exact_position_match"
        ),
        "parameters": [
            "block_health",
            "break_available_and_allowed",
            "tool_compatible_and_power",
            "interaction_tool_route_matched",
            "break_removes_block_or_exact_replacement_semantic_key",
            "harvest_available_and_allowed",
            "place_available_and_allowed",
            "resolved_item_id_quantity_max_stack_durability_metadata",
        ],
        "quantity": {
            "dtype": "int32",
            "source": "resolved_installed_BlockGathering_or_DropList_program",
            "randomized": "explicit_caller_sample_never_process_rng",
            "inventory_application": "consumer_owned",
        },
        "availability": (
            "row_for_exact_target_alignment_action_predicates_independent"
        ),
        "policy_binding": (
            "additive_companion_not_in_existing_token_or_policy_contract"
        ),
        "failure": (
            "position_mismatch_nonexact_query_or_structural_action_"
            "diagnostic_clears_complete_actor_row; verb_specific_"
            "unavailability_remains_in_independent_predicates"
        ),
    }


def actor_block_action_parameter_contract_sha256() -> str:
    payload = json.dumps(
        actor_block_action_parameter_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def estimated_actor_block_action_candidate_bytes(
    *,
    batch_size: int,
    actor_count: int,
    candidate_capacity: int,
) -> int:
    """Return exact logical output-array bytes, excluding allocator padding."""

    values = {
        "batch_size": batch_size,
        "actor_count": actor_count,
        "candidate_capacity": candidate_capacity,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return batch_size * actor_count * (6 + 78 * candidate_capacity)


def _cell_offsets(index: Array) -> Array:
    x = index // (_SIDE * _SIDE)
    remainder = index % (_SIDE * _SIDE)
    y = remainder // _SIDE
    z = remainder % _SIDE
    return (
        jnp.stack((x, y, z), axis=3).astype(jnp.int32)
        - jnp.int32(CELL_RADIUS)
    )


def _candidate_visibility(
    geometry: GeometryProvider,
    eye: Array,
    target: Array,
    mask: Array,
    role_mask: Array,
    *,
    max_los_cells: int | None,
) -> tuple[Array, Array]:
    starts = jnp.broadcast_to(eye[:, :, None, :], target.shape)
    ends = jnp.where(mask[..., None], target, starts)
    starts = jnp.transpose(starts, (1, 2, 0, 3))
    ends = jnp.transpose(ends, (1, 2, 0, 3))
    masks = jnp.swapaxes(role_mask, 0, 1)

    def actor_queries(
        actor_starts: Array,
        actor_ends: Array,
        actor_mask: Array,
    ):
        return jax.vmap(
            lambda begin, end: geometry_perception_target_cell_visible_result(
                geometry,
                begin,
                end,
                role_opaque_mask=actor_mask,
                max_cells=max_los_cells,
            )
        )(actor_starts, actor_ends)

    result = jax.vmap(actor_queries)(starts, ends, masks)

    def restore(value: Array) -> Array:
        return jnp.transpose(value, (2, 0, 1))

    visible = restore(result.visible) & mask
    failed = restore(
        result.geometry_exhausted
        | result.capacity_exceeded
        | result.invalid
        | ~result.role_opacity_applied
    ) & mask
    return visible, failed


def _validated_actor_role_mask(
    geometry: GeometryProvider,
    value: Array,
    batch: int,
    actors: int,
) -> Array:
    result = jnp.asarray(value)
    if result.dtype != jnp.bool_:
        raise TypeError("role_opaque_mask must have boolean dtype")
    expected = (
        geometry.atlas.cell_flags.shape
        if isinstance(geometry, RegionGeometryState)
        else (CELL_COUNT,)
    )
    if result.shape != (batch, actors) + expected:
        raise ValueError(
            "role_opaque_mask must have explicit [B, A, ...] provider shape"
        )
    return result


def _validate_geometry_batch(geometry: GeometryProvider, batch: int) -> None:
    size = (
        geometry.environment_world_id.shape[0]
        if isinstance(geometry, RegionGeometryState)
        else geometry.origin.shape[0]
    )
    if size not in (1, batch) or (
        isinstance(geometry, RegionGeometryState) and size != batch
    ):
        raise ValueError("geometry batch must be one or match actors")


def _validate_candidate_query_shapes(
    query: MutableBlockQueryResult,
    shape: tuple[int, int],
) -> None:
    values = (
        query.invalid,
        query.resync_required,
        query.provenance,
        query.geometry.block_present,
        query.geometry.exact,
        query.geometry.semantic_key_valid,
        query.geometry.affordance_valid,
        query.geometry.affordance_tags,
        query.geometry.gather_type_index,
        query.geometry.required_tool_quality,
    )
    if any(value.shape != shape for value in values):
        raise ValueError("target query scalar fields have inconsistent shape")
    if query.geometry.semantic_key.shape != shape + (
        BLOCK_SEMANTIC_KEY_WORDS,
    ):
        raise ValueError("target query semantic keys have inconsistent shape")


def _validate_parameter_query_shapes(
    query: MutableBlockQueryResult,
    actions: BlockActionTargetResult,
    shape: tuple[int, int],
) -> None:
    _validate_candidate_query_shapes(query, shape)
    if (
        query.available.shape != shape
        or query.block_health.shape != shape
        or query.block_health_valid.shape != shape
    ):
        raise ValueError("selected target query fields have inconsistent shape")
    scalar_actions = (
        actions.target_available,
        actions.diagnostics,
        actions.catalog_index,
        actions.break_available,
        actions.break_allowed,
        actions.tool_compatible,
        actions.tool_power,
        actions.interaction_tool_route_matched,
        actions.break_removes_block,
        actions.replacement_semantic_key_valid,
        actions.harvest_available,
        actions.harvest_allowed,
        actions.target_harvestable,
        actions.place_available,
        actions.place_allowed,
    )
    if any(value.shape != shape for value in scalar_actions):
        raise ValueError("selected action fields have inconsistent shape")
    if actions.replacement_semantic_key.shape != shape + (
        BLOCK_SEMANTIC_KEY_WORDS,
    ):
        raise ValueError(
            "selected action replacement semantic keys have "
            "inconsistent shape"
        )
    _validate_parameter_drops(actions.break_drops, shape, "break_drops")
    _validate_parameter_drops(actions.harvest_drops, shape, "harvest_drops")


def _validate_parameter_drops(
    drops: BlockResolvedDrops,
    shape: tuple[int, int],
    label: str,
) -> None:
    if not isinstance(drops, BlockResolvedDrops):
        raise TypeError(f"{label} must be BlockResolvedDrops")
    if drops.available.shape != shape or drops.mask.ndim != 3:
        raise ValueError(f"{label} has inconsistent prefix shape")
    capacity = drops.mask.shape[2]
    if capacity <= 0:
        raise ValueError(f"{label} must have positive drop capacity")
    for name in (
        "mask",
        "item_id",
        "quantity",
        "item_max_stack",
        "durability",
        "max_durability",
    ):
        if getattr(drops, name).shape != shape + (capacity,):
            raise ValueError(f"{label}.{name} has inconsistent shape")
    if (
        drops.metadata_hash.ndim != 4
        or drops.metadata_hash.shape[:3] != shape + (capacity,)
    ):
        raise ValueError(f"{label}.metadata_hash has inconsistent shape")


def _actor_parameter(
    value: Array | float,
    batch: int,
    actors: int,
    name: str,
) -> Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.ndim == 0:
        return jnp.broadcast_to(result, (batch, actors))
    if result.shape == (batch,):
        return jnp.broadcast_to(result[:, None], (batch, actors))
    if result.shape != (batch, actors):
        raise ValueError(f"{name} must be scalar or have shape [B] or [B, A]")
    return result


def _static_cell_radius(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("cell_radius must be a nonnegative integer")
    return value


def _gather_actor_candidates(value: Array, index: Array) -> Array:
    expanded = index.reshape(
        index.shape + (1,) * (value.ndim - index.ndim)
    )
    expanded = jnp.broadcast_to(
        expanded,
        index.shape + value.shape[3:],
    )
    return jnp.take_along_axis(value, expanded, axis=2)


__all__ = [
    "ACTOR_BLOCK_ACTION_CANDIDATE_SCHEMA",
    "ACTOR_BLOCK_ACTION_CANDIDATE_VERSION",
    "ACTOR_BLOCK_ACTION_PARAMETER_SCHEMA",
    "ACTOR_BLOCK_ACTION_PARAMETER_VERSION",
    "ACTOR_BLOCK_ACTION_DIAGNOSTIC_ALIAS_CONFLICT",
    "ACTOR_BLOCK_ACTION_DIAGNOSTIC_AFFORDANCE_UNAVAILABLE",
    "ACTOR_BLOCK_ACTION_DIAGNOSTIC_INVALID_ACTOR",
    "ACTOR_BLOCK_ACTION_DIAGNOSTIC_LOS_UNAVAILABLE",
    "ACTOR_BLOCK_ACTION_DIAGNOSTIC_OUTPUT_CAPACITY",
    "ACTOR_BLOCK_ACTION_DIAGNOSTIC_QUERY_UNAVAILABLE",
    "ACTOR_BLOCK_ACTION_DIAGNOSTIC_SOURCE_INCOMPLETE",
    "ACTOR_BLOCK_PARAMETER_DIAGNOSTIC_ACTION",
    "ACTOR_BLOCK_PARAMETER_DIAGNOSTIC_POSITION",
    "ACTOR_BLOCK_PARAMETER_DIAGNOSTIC_QUERY",
    "ACTOR_BLOCK_AFFORDANCE_TOKEN_SCHEMA",
    "ACTOR_BLOCK_AFFORDANCE_TOKEN_VERSION",
    "ACTOR_BLOCK_TOKEN_DIAGNOSTIC_AFFORDANCE_UNAVAILABLE",
    "ACTOR_BLOCK_TOKEN_DIAGNOSTIC_BLOCK_ABSENT",
    "ACTOR_BLOCK_TOKEN_DIAGNOSTIC_CELL_UNAVAILABLE",
    "ACTOR_BLOCK_TOKEN_DIAGNOSTIC_INVALID_KIND",
    "ACTOR_BLOCK_TOKEN_DIAGNOSTIC_INVALID_SOURCE",
    "ActorBlockActionCandidates",
    "ActorBlockActionParameters",
    "ActorBlockAffordanceTokens",
    "ActorBlockTokenCells",
    "actor_block_action_candidate_contract",
    "actor_block_action_candidate_contract_sha256",
    "actor_block_action_parameter_contract",
    "actor_block_action_parameter_contract_sha256",
    "actor_block_affordance_token_contract",
    "actor_block_affordance_token_contract_sha256",
    "dense_actor_block_candidate_cells",
    "dense_actor_block_candidate_source_complete",
    "estimated_actor_block_action_candidate_bytes",
    "produce_actor_block_action_candidates",
    "publish_actor_block_affordance_tokens",
    "publish_actor_block_action_parameters",
    "world_geometry_token_block_cells",
]
