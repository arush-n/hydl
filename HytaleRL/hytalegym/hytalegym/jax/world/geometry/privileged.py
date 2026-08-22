"""Physically separate privileged geometry for critics and teachers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.framework import (
    AuthoritativeSnapshot,
    ContentDigest,
    PrivilegedSceneEnvelope,
    ProviderContract,
    SchemaBundle,
    SchemaRef,
)
from hytalegym.jax.world.tokens import (
    WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY,
    WORLD_TOKEN_PROVENANCE_SURROGATE_EXACT_GEOMETRY,
    WORLD_TOKEN_PROVENANCE_SURROGATE_TRAVERSAL,
    _VisibleCandidates,
    _actor_parameter,
    _actor_tile_index,
    _broadcast_geometry,
    _collision_exception_candidates,
    _combine_candidates,
    _pad_candidates,
    _remap_selected_edges,
    _select_candidates,
    _traversal_candidates,
)
from hytalegym.jax.world.traversal import (
    GRAPH_DIAGNOSTIC_STATEFUL_TRANSITIONS,
    TraversalTokenObservation,
)
from hytalegym.jax.world.types import GeometryState


Array = jax.Array
PRIVILEGED_GEOMETRY_SCHEMA = "hytalerl_privileged_geometry_scene_v1"
PRIVILEGED_GEOMETRY_VERSION = 1
PRIVILEGED_GEOMETRY_TOKEN_CAPACITY = 102
PRIVILEGED_GEOMETRY_MAXIMUM_DISTANCE = 4.0

PRIVILEGED_GEOMETRY_DIAGNOSTIC_INVALID_ACTOR = jnp.uint32(1)
PRIVILEGED_GEOMETRY_DIAGNOSTIC_SOURCE_UNAVAILABLE = jnp.uint32(1 << 1)
PRIVILEGED_GEOMETRY_DIAGNOSTIC_SOURCE_OVERFLOW = jnp.uint32(1 << 2)
PRIVILEGED_GEOMETRY_DIAGNOSTIC_OUTPUT_CAPACITY = jnp.uint32(1 << 3)
PRIVILEGED_GEOMETRY_DIAGNOSTIC_STATE_EVIDENCE = jnp.uint32(1 << 4)


class PrivilegedGeometryScene(NamedTuple):
    """Training-only geometry; deliberately not an actor observation type."""

    available: Array
    capacity_exceeded: Array
    diagnostics: Array
    source_available: Array
    source_overflow: Array
    source_tile_index: Array
    token_mask: Array
    token_kind: Array
    token_provenance: Array
    source_index: Array
    relative_position: Array
    clearance: Array
    semantic_flags: Array
    dynamic_blocked: Array
    edge_mask: Array
    edge_destination: Array
    edge_cost: Array
    edge_kind: Array
    edge_flags: Array
    collision_box_mask: Array
    collision_boxes_relative: Array


@dataclass(frozen=True, slots=True)
class PrivilegedGeometrySnapshot:
    """Exact world inputs copied into one authoritative tick snapshot."""

    geometry: GeometryState
    actor_position: Array
    traversal: TraversalTokenObservation
    geometry_provenance: int
    geometry_tile_index: Array | None = None
    dynamic_state_visible: Array | None = None
    edge_state_visible: Array | None = None


def produce_privileged_geometry_scene(
    geometry: GeometryState,
    actor_position: Array,
    *,
    traversal: TraversalTokenObservation,
    geometry_provenance: int,
    geometry_tile_index: Array | None = None,
    dynamic_state_visible: Array | None = None,
    edge_state_visible: Array | None = None,
    token_capacity: int = PRIVILEGED_GEOMETRY_TOKEN_CAPACITY,
    maximum_distance: float = PRIVILEGED_GEOMETRY_MAXIMUM_DISTANCE,
) -> PrivilegedGeometryScene:
    """Publish exact nearby geometry without actor FOV or LOS filtering."""

    allowed_provenance = (
        WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY,
        WORLD_TOKEN_PROVENANCE_SURROGATE_EXACT_GEOMETRY,
    )
    if (
        isinstance(geometry_provenance, bool)
        or geometry_provenance not in allowed_provenance
    ):
        raise ValueError("geometry_provenance must identify an exact provider")
    if (
        isinstance(token_capacity, bool)
        or not isinstance(token_capacity, int)
        or token_capacity <= 0
    ):
        raise ValueError("token_capacity must be a positive integer")
    if not isinstance(traversal, TraversalTokenObservation):
        raise TypeError("traversal must be a TraversalTokenObservation")

    positions = jnp.asarray(actor_position, dtype=jnp.float32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("actor_position must have shape [batch, actors, 3]")
    batch, actors, _ = positions.shape
    if traversal.available.shape != (batch, actors):
        raise ValueError("traversal and actor row shapes differ")
    world = _broadcast_geometry(geometry, batch)
    maximum = _actor_parameter(
        maximum_distance,
        batch,
        actors,
        "maximum_distance",
    )
    valid_actor = (
        jnp.all(jnp.isfinite(positions), axis=2)
        & jnp.isfinite(maximum)
        & (maximum > 0.0)
    )
    safe_position = jnp.where(valid_actor[..., None], positions, 0.0)

    exception_source, exception_source_available = (
        _collision_exception_candidates(
            world,
            safe_position,
            geometry_provenance,
        )
    )
    traversal_source, visible_edges = _traversal_candidates(
        traversal,
        safe_position,
        safe_position,
        dynamic_state_visible,
        edge_state_visible,
    )

    def within(source) -> _VisibleCandidates:
        distance_squared = jnp.sum(
            source.payload.relative_position**2,
            axis=3,
        )
        return _VisibleCandidates(
            source.mask
            & valid_actor[..., None]
            & (distance_squared <= maximum[..., None] ** 2),
            source.payload,
        )

    candidates = _combine_candidates(
        within(traversal_source),
        within(exception_source),
    )
    candidates = _pad_candidates(candidates, token_capacity)
    selected, selected_index = _select_candidates(
        candidates,
        token_capacity,
    )
    visible_count = jnp.sum(candidates.eligible, axis=2)
    capacity_exceeded = visible_count > token_capacity
    traversal_provenance_valid = (
        traversal.provenance
        == jnp.uint8(WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY)
    ) | (
        traversal.provenance
        == jnp.uint8(WORLD_TOKEN_PROVENANCE_SURROGATE_TRAVERSAL)
    )
    traversal_source_available = (
        traversal.available & traversal_provenance_valid
    )
    all_sources_available = (
        traversal_source_available & exception_source_available
    )
    source_overflow = traversal.capacity_exceeded
    state_evidence_missing = _state_evidence_missing(
        traversal,
        dynamic_state_visible,
        edge_state_visible,
    )
    row_available = (
        valid_actor
        & all_sources_available
        & ~source_overflow
        & ~state_evidence_missing
        & ~capacity_exceeded
    )
    token_mask = selected.eligible & row_available[..., None]
    traversal_capacity = traversal.token_mask.shape[2]
    edge_capacity = traversal.edge_mask.shape[3]
    edges = _remap_selected_edges(
        traversal,
        visible_edges,
        selected_index,
        token_mask,
        traversal_capacity,
        edge_capacity,
    )

    def masked(values: Array, fill: int | float = 0) -> Array:
        gate = token_mask.reshape(
            token_mask.shape + (1,) * (values.ndim - token_mask.ndim)
        )
        return jnp.where(gate, values, jnp.asarray(fill, dtype=values.dtype))

    diagnostics = (
        jnp.where(
            valid_actor,
            jnp.uint32(0),
            PRIVILEGED_GEOMETRY_DIAGNOSTIC_INVALID_ACTOR,
        )
        | jnp.where(
            all_sources_available,
            jnp.uint32(0),
            PRIVILEGED_GEOMETRY_DIAGNOSTIC_SOURCE_UNAVAILABLE,
        )
        | jnp.where(
            source_overflow,
            PRIVILEGED_GEOMETRY_DIAGNOSTIC_SOURCE_OVERFLOW,
            jnp.uint32(0),
        )
        | jnp.where(
            capacity_exceeded,
            PRIVILEGED_GEOMETRY_DIAGNOSTIC_OUTPUT_CAPACITY,
            jnp.uint32(0),
        )
        | jnp.where(
            state_evidence_missing,
            PRIVILEGED_GEOMETRY_DIAGNOSTIC_STATE_EVIDENCE,
            jnp.uint32(0),
        )
    )
    exception_tile = _actor_tile_index(
        geometry_tile_index,
        batch,
        actors,
    )
    return PrivilegedGeometryScene(
        available=row_available,
        capacity_exceeded=capacity_exceeded,
        diagnostics=diagnostics,
        source_available=jnp.stack(
            (
                traversal_source_available,
                exception_source_available,
            ),
            axis=2,
        ),
        source_overflow=jnp.stack(
            (source_overflow, jnp.zeros_like(source_overflow)),
            axis=2,
        ),
        source_tile_index=jnp.stack(
            (
                jnp.where(
                    traversal_source_available,
                    traversal.tile_index,
                    -1,
                ),
                jnp.where(
                    exception_source_available,
                    exception_tile,
                    -1,
                ),
            ),
            axis=2,
        ),
        token_mask=token_mask,
        token_kind=masked(selected.payload.kind),
        token_provenance=masked(selected.payload.provenance),
        source_index=masked(selected.payload.source_index, -1),
        relative_position=masked(selected.payload.relative_position),
        clearance=masked(selected.payload.clearance),
        semantic_flags=masked(selected.payload.semantic_flags),
        dynamic_blocked=masked(selected.payload.dynamic_blocked),
        edge_mask=edges.mask,
        edge_destination=edges.destination,
        edge_cost=edges.cost,
        edge_kind=edges.kind,
        edge_flags=edges.flags,
        collision_box_mask=masked(selected.payload.collision_box_mask),
        collision_boxes_relative=masked(
            selected.payload.collision_boxes_relative
        ),
    )


def privileged_geometry_contract() -> dict[str, object]:
    """Return the framework- and checkpoint-pinnable privileged contract."""

    return {
        "schema": PRIVILEGED_GEOMETRY_SCHEMA,
        "version": PRIVILEGED_GEOMETRY_VERSION,
        "producer": "produce_privileged_geometry_scene",
        "framework_envelope": "PrivilegedSceneEnvelope",
        "physically_distinct_from": "WorldGeometryTokenObservation",
        "arguments_absent_by_design": [
            "actor_eye_position",
            "actor_forward",
            "role_opaque_mask",
            "view_sector_full_angle_radians",
            "line_of_sight_mode",
        ],
        "default_shape": {
            "token_capacity": PRIVILEGED_GEOMETRY_TOKEN_CAPACITY,
            "maximum_distance_blocks": PRIVILEGED_GEOMETRY_MAXIMUM_DISTANCE,
            "edge_capacity": "traversal_source_static",
            "detail_box_capacity": "exact_GeometryState_static",
        },
        "sources": {
            "collision_exception": "exact_GeometryState",
            "ordinary_surface_and_edges": (
                "provenance-bearing_TraversalTokenObservation"
            ),
            "stateful_values": "explicit_privileged_evidence_required",
        },
        "selection": {
            "filters": ["finite_actor", "distance"],
            "fov": False,
            "los": False,
            "overflow": "clear_complete_privileged_row",
            "ordering": "shared_deterministic_world_candidate_order",
        },
        "capacity_evidence": {
            "artifact": (
                "artifacts/worldgen/"
                "privileged-geometry-capacity-design-v1.json"
            ),
            "region_library_semantic_sha256": (
                "f0ea0526ac1ee6184fc575c085607ee7b3f9843d78b0d4f806fff15b51c8616c"
            ),
            "corpus_maximum": 102,
            "outside_corpus": "checked_fail_closed",
        },
        "consumer": "explicit_opt_in_critic_teacher_or_debug_only",
        "actor_conversion": None,
    }


def privileged_geometry_contract_sha256() -> str:
    encoded = json.dumps(
        privileged_geometry_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def privileged_geometry_schema_ref() -> SchemaRef:
    return SchemaRef(
        name=PRIVILEGED_GEOMETRY_SCHEMA,
        version=PRIVILEGED_GEOMETRY_VERSION,
        digest=ContentDigest(
            "sha256",
            privileged_geometry_contract_sha256(),
        ),
    )


@dataclass(frozen=True, slots=True)
class PrivilegedGeometryCompiler:
    """Concrete first compiler for the framework's privileged protocol."""

    input_schema: SchemaRef
    token_capacity: int = PRIVILEGED_GEOMETRY_TOKEN_CAPACITY
    maximum_distance: float = PRIVILEGED_GEOMETRY_MAXIMUM_DISTANCE

    @property
    def contract(self) -> ProviderContract:
        return ProviderContract(
            provider_id="world.privileged-geometry-v1",
            consumes=SchemaBundle.from_mapping(
                {"snapshot": self.input_schema}
            ),
            produces=SchemaBundle.from_mapping(
                {"privileged_geometry": privileged_geometry_schema_ref()}
            ),
            required_capabilities=(
                "exact_geometry",
                "traversal_graph",
            ),
            provided_capabilities=("privileged_geometry",),
        )

    def compile(
        self,
        snapshot: AuthoritativeSnapshot[PrivilegedGeometrySnapshot],
    ) -> PrivilegedSceneEnvelope[PrivilegedGeometryScene]:
        if not isinstance(snapshot, AuthoritativeSnapshot):
            raise TypeError("snapshot must be an AuthoritativeSnapshot")
        if snapshot.header.schema != self.input_schema:
            raise ValueError("snapshot schema differs from compiler contract")
        source = snapshot.payload
        if not isinstance(source, PrivilegedGeometrySnapshot):
            raise TypeError("snapshot payload must be PrivilegedGeometrySnapshot")
        scene = produce_privileged_geometry_scene(
            source.geometry,
            source.actor_position,
            traversal=source.traversal,
            geometry_provenance=source.geometry_provenance,
            geometry_tile_index=source.geometry_tile_index,
            dynamic_state_visible=source.dynamic_state_visible,
            edge_state_visible=source.edge_state_visible,
            token_capacity=self.token_capacity,
            maximum_distance=self.maximum_distance,
        )
        return PrivilegedSceneEnvelope(
            source=snapshot.header,
            schema=privileged_geometry_schema_ref(),
            payload=scene,
        )


def _state_evidence_missing(
    traversal: TraversalTokenObservation,
    dynamic_state_visible: Array | None,
    edge_state_visible: Array | None,
) -> Array:
    stateful = (
        traversal.diagnostics
        & jnp.uint32(GRAPH_DIAGNOSTIC_STATEFUL_TRANSITIONS)
    ) != 0
    if dynamic_state_visible is None or edge_state_visible is None:
        return stateful
    return jnp.zeros_like(stateful)


__all__ = [
    "PRIVILEGED_GEOMETRY_MAXIMUM_DISTANCE",
    "PRIVILEGED_GEOMETRY_SCHEMA",
    "PRIVILEGED_GEOMETRY_TOKEN_CAPACITY",
    "PRIVILEGED_GEOMETRY_VERSION",
    "PrivilegedGeometryCompiler",
    "PrivilegedGeometryScene",
    "PrivilegedGeometrySnapshot",
    "privileged_geometry_contract",
    "privileged_geometry_contract_sha256",
    "privileged_geometry_schema_ref",
    "produce_privileged_geometry_scene",
]
