"""Exact empty-cell placement candidates derived from visible block faces."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.world.interactions.blocks.tokens import (
    ActorBlockActionCandidates,
    actor_block_action_candidate_contract_sha256,
)
from hytalegym.jax.world.mutable_blocks import (
    MutableBlockQueryResult,
    mutable_block_contract_sha256,
)


Array = jax.Array

ACTOR_BLOCK_PLACEMENT_CANDIDATE_SCHEMA = (
    "hytalerl_actor_block_placement_candidates_v1"
)
ACTOR_BLOCK_PLACEMENT_CANDIDATE_VERSION = 1

ACTOR_BLOCK_PLACEMENT_DIAGNOSTIC_SOURCE = jnp.uint32(1)
ACTOR_BLOCK_PLACEMENT_DIAGNOSTIC_POSITION = jnp.uint32(1 << 1)
ACTOR_BLOCK_PLACEMENT_DIAGNOSTIC_QUERY = jnp.uint32(1 << 2)
ACTOR_BLOCK_PLACEMENT_DIAGNOSTIC_CAPACITY = jnp.uint32(1 << 3)

# Hytale protocol.BlockFace values and Vector3iUtil directions:
# Up, Down, North/FORWARD, South/BACKWARD, East/RIGHT, West/LEFT.
BLOCK_PLACEMENT_FACE_VALUES = jnp.asarray(
    (1, 2, 3, 4, 5, 6),
    dtype=jnp.uint8,
)
BLOCK_PLACEMENT_FACE_OFFSETS = jnp.asarray(
    (
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, -1),
        (0, 0, 1),
        (1, 0, 0),
        (-1, 0, 0),
    ),
    dtype=jnp.int32,
)


class ActorBlockPlacementQuery(NamedTuple):
    """All six face-adjacent cells for each actor-visible block candidate."""

    source_mask: Array
    clicked_position: Array
    destination_position: Array
    block_face: Array
    clicked_relative_position: Array
    destination_relative_position: Array
    source_provenance: Array


class ActorBlockPlacementCandidates(NamedTuple):
    """Bounded exact empty destinations ready for a Place request."""

    available: Array
    capacity_exceeded: Array
    diagnostics: Array
    candidate_mask: Array
    clicked_position: Array
    destination_position: Array
    block_face: Array
    clicked_relative_position: Array
    destination_relative_position: Array
    source_provenance: Array
    destination_provenance: Array


def actor_block_placement_query(
    candidates: ActorBlockActionCandidates,
) -> ActorBlockPlacementQuery:
    """Expand each visible block into protocol-face placement destinations."""

    if not isinstance(candidates, ActorBlockActionCandidates):
        raise TypeError("candidates must be ActorBlockActionCandidates")
    batch, actors, capacity = candidates.candidate_mask.shape
    face_count = BLOCK_PLACEMENT_FACE_VALUES.shape[0]
    output_shape = (batch, actors, capacity * face_count)

    def expand(value: Array) -> Array:
        repeated = jnp.repeat(value[..., None, :], face_count, axis=3)
        return repeated.reshape(output_shape + value.shape[3:])

    clicked = expand(candidates.visible_position)
    offset = jnp.broadcast_to(
        BLOCK_PLACEMENT_FACE_OFFSETS,
        (batch, actors, capacity, face_count, 3),
    ).reshape(output_shape + (3,))
    faces = jnp.broadcast_to(
        BLOCK_PLACEMENT_FACE_VALUES,
        (batch, actors, capacity, face_count),
    ).reshape(output_shape)
    mask = jnp.repeat(candidates.candidate_mask, face_count, axis=2)
    relative = expand(candidates.visible_relative_position)
    provenance = jnp.repeat(candidates.provenance, face_count, axis=2)

    def masked(value: Array) -> Array:
        gate = mask.reshape(
            mask.shape + (1,) * (value.ndim - mask.ndim)
        )
        return jnp.where(gate, value, jnp.zeros_like(value))

    return ActorBlockPlacementQuery(
        source_mask=mask,
        clicked_position=masked(clicked),
        destination_position=masked(clicked + offset),
        block_face=masked(faces),
        clicked_relative_position=masked(relative),
        destination_relative_position=masked(
            relative + offset.astype(jnp.float32)
        ),
        source_provenance=masked(provenance),
    )


def publish_actor_block_placement_candidates(
    candidates: ActorBlockActionCandidates,
    queried_destination_position: Array,
    destinations: MutableBlockQueryResult,
    *,
    candidate_capacity: int,
) -> ActorBlockPlacementCandidates:
    """Publish complete exact empty-cell destinations without face inference.

    The caller must query ``actor_block_placement_query(...).destination_position``
    flattened in actor-major order. Any required position mismatch or unknown
    destination clears the complete actor row; occupied cells are known
    non-candidates rather than unavailable evidence.
    """

    if not isinstance(candidates, ActorBlockActionCandidates):
        raise TypeError("candidates must be ActorBlockActionCandidates")
    if not isinstance(destinations, MutableBlockQueryResult):
        raise TypeError("destinations must be MutableBlockQueryResult")
    if (
        isinstance(candidate_capacity, bool)
        or not isinstance(candidate_capacity, int)
        or candidate_capacity <= 0
    ):
        raise ValueError("candidate_capacity must be a positive integer")

    query = actor_block_placement_query(candidates)
    batch, actors, source_capacity = query.source_mask.shape
    if candidate_capacity > source_capacity:
        raise ValueError("candidate_capacity exceeds expanded face capacity")
    queried = jnp.asarray(queried_destination_position, dtype=jnp.int32)
    if queried.shape != query.destination_position.shape:
        raise ValueError(
            "queried_destination_position must have shape "
            "[batch,actor,expanded_face]"
        )
    flat_shape = (batch, actors * source_capacity)
    _validate_destination_query(destinations, flat_shape)

    def rows(value: Array) -> Array:
        return value.reshape(
            (batch, actors, source_capacity) + value.shape[2:]
        )

    required = query.source_mask
    position_match = jnp.all(
        queried == query.destination_position,
        axis=3,
    )
    target_known = (
        rows(destinations.available)
        & rows(destinations.geometry.exact)
        & ~rows(destinations.invalid | destinations.resync_required)
    )
    query_failure = required & (~position_match | ~target_known)
    eligible = (
        required
        & target_known
        & ~rows(destinations.geometry.block_present)
    )
    eligible_count = jnp.sum(eligible, axis=2)
    capacity_exceeded = eligible_count > candidate_capacity
    row_available = (
        candidates.available
        & ~jnp.any(required & ~position_match, axis=2)
        & ~jnp.any(required & ~target_known, axis=2)
        & ~capacity_exceeded
    )
    diagnostics = (
        jnp.where(
            ~candidates.available,
            ACTOR_BLOCK_PLACEMENT_DIAGNOSTIC_SOURCE,
            jnp.uint32(0),
        )
        | jnp.where(
            jnp.any(required & ~position_match, axis=2),
            ACTOR_BLOCK_PLACEMENT_DIAGNOSTIC_POSITION,
            jnp.uint32(0),
        )
        | jnp.where(
            jnp.any(query_failure & position_match, axis=2),
            ACTOR_BLOCK_PLACEMENT_DIAGNOSTIC_QUERY,
            jnp.uint32(0),
        )
        | jnp.where(
            capacity_exceeded,
            ACTOR_BLOCK_PLACEMENT_DIAGNOSTIC_CAPACITY,
            jnp.uint32(0),
        )
    )

    source_index = jnp.broadcast_to(
        jnp.arange(source_capacity, dtype=jnp.int32),
        eligible.shape,
    )
    selected_index = jnp.argsort(
        jnp.where(
            eligible,
            source_index,
            jnp.iinfo(jnp.int32).max,
        ),
        axis=2,
        stable=True,
    )[..., :candidate_capacity]
    selected_eligible = _gather(eligible, selected_index)
    output_mask = selected_eligible & row_available[..., None]

    def selected(value: Array) -> Array:
        return _gather(value, selected_index)

    def masked(value: Array) -> Array:
        value = selected(value)
        gate = output_mask.reshape(
            output_mask.shape + (1,) * (value.ndim - output_mask.ndim)
        )
        return jnp.where(gate, value, jnp.zeros_like(value))

    return ActorBlockPlacementCandidates(
        available=row_available,
        capacity_exceeded=capacity_exceeded,
        diagnostics=diagnostics,
        candidate_mask=output_mask,
        clicked_position=masked(query.clicked_position),
        destination_position=masked(query.destination_position),
        block_face=masked(query.block_face),
        clicked_relative_position=masked(
            query.clicked_relative_position
        ),
        destination_relative_position=masked(
            query.destination_relative_position
        ),
        source_provenance=masked(query.source_provenance),
        destination_provenance=masked(rows(destinations.provenance)),
    )


def actor_block_placement_candidate_contract() -> dict[str, object]:
    """Return the exact actor placement-destination contract."""

    return {
        "schema": ACTOR_BLOCK_PLACEMENT_CANDIDATE_SCHEMA,
        "version": ACTOR_BLOCK_PLACEMENT_CANDIDATE_VERSION,
        "actor_block_action_candidate_contract_sha256": (
            actor_block_action_candidate_contract_sha256()
        ),
        "mutable_block_contract_sha256": mutable_block_contract_sha256(),
        "source": (
            "complete_actor_visible_block_candidates_expanded_over_six_"
            "Hytale_protocol_faces"
        ),
        "face_order": [
            ["Up", 1, [0, 1, 0]],
            ["Down", 2, [0, -1, 0]],
            ["North", 3, [0, 0, -1]],
            ["South", 4, [0, 0, 1]],
            ["East", 5, [1, 0, 0]],
            ["West", 6, [-1, 0, 0]],
        ],
        "destination": (
            "clicked_cell_plus_protocol_face_direction_exact_current_"
            "mutable_revision_and_block_absent"
        ),
        "query_order": "actor_then_source_candidate_then_protocol_face",
        "selection": "source_candidate_rank_then_protocol_face",
        "fixed_capacity": (
            "count_all_exact_empty_destinations_then_clear_row_on_overflow"
        ),
        "policy_fields": [
            "available",
            "candidate_mask",
            "clicked_relative_position",
            "destination_relative_position",
        ],
        "native_request_fields": [
            "destination_position_as_target",
            "block_face",
        ],
        "verification_fields": [
            "clicked_position",
            "source_provenance",
            "destination_provenance",
        ],
        "rotation": (
            "downstream_actor_aim_owned_not_inferred_or_fixture_defaulted"
        ),
        "failure": (
            "source_position_query_or_capacity_failure_clears_complete_"
            "actor_row; occupied_exact_destinations_are_omitted"
        ),
        "consumer": (
            "sampled_slot_must_be_requeried_before_typed_native_request"
        ),
    }


def actor_block_placement_candidate_contract_sha256() -> str:
    payload = json.dumps(
        actor_block_placement_candidate_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_destination_query(
    query: MutableBlockQueryResult,
    shape: tuple[int, int],
) -> None:
    fields = (
        ("available", query.available, jnp.bool_),
        ("provenance", query.provenance, jnp.uint8),
        ("invalid", query.invalid, jnp.bool_),
        ("resync_required", query.resync_required, jnp.bool_),
        ("geometry.block_present", query.geometry.block_present, jnp.bool_),
        ("geometry.exact", query.geometry.exact, jnp.bool_),
    )
    for name, value, dtype in fields:
        if value.shape != shape or value.dtype != dtype:
            raise TypeError(f"{name} must be {dtype} with shape {shape}")


def _gather(value: Array, index: Array) -> Array:
    suffix = value.shape[3:]
    gather_index = index.reshape(
        index.shape + (1,) * len(suffix)
    )
    return jnp.take_along_axis(
        value,
        jnp.broadcast_to(gather_index, index.shape + suffix),
        axis=2,
    )


__all__ = [
    "ACTOR_BLOCK_PLACEMENT_CANDIDATE_SCHEMA",
    "ACTOR_BLOCK_PLACEMENT_CANDIDATE_VERSION",
    "ACTOR_BLOCK_PLACEMENT_DIAGNOSTIC_CAPACITY",
    "ACTOR_BLOCK_PLACEMENT_DIAGNOSTIC_POSITION",
    "ACTOR_BLOCK_PLACEMENT_DIAGNOSTIC_QUERY",
    "ACTOR_BLOCK_PLACEMENT_DIAGNOSTIC_SOURCE",
    "BLOCK_PLACEMENT_FACE_OFFSETS",
    "BLOCK_PLACEMENT_FACE_VALUES",
    "ActorBlockPlacementCandidates",
    "ActorBlockPlacementQuery",
    "actor_block_placement_candidate_contract",
    "actor_block_placement_candidate_contract_sha256",
    "actor_block_placement_query",
    "publish_actor_block_placement_candidates",
]
