"""Actor-legal fixed-shape World geometry tokens."""

from __future__ import annotations

from functools import cache
import hashlib
import json
import math
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from hytalegym.geometry.contract import CELL_COUNT, MAX_DETAIL_BOXES
from hytalegym.jax.world.perception.los import (
    GeometryProvider,
)
from hytalegym.jax.world.perception import (
    NATIVE_VIEW_MAX_FULL_ANGLE_RADIANS,
)
from hytalegym.jax.world.region.types import RegionGeometryState
from hytalegym.jax.world.traversal import (
    TRAVERSAL_PROVENANCE_NATIVE_EXACT_GEOMETRY,
    TRAVERSAL_PROVENANCE_SURROGATE,
    TraversalTokenObservation,
)
from hytalegym.jax.world.types import GeometryState
from hytalegym.worldgen.region.contract import (
    CHUNK_SIZE,
    CORE_CHUNKS_PER_AXIS,
)
from hytalegym.worldgen.region.traversal_library import (
    region_traversal_library_semantic_sha256,
)
from ._candidates import (  # noqa: F401  (re-exported: callers unchanged)
    Array,
    WORLD_TOKEN_KIND_COLLISION_EXCEPTION,
    WORLD_TOKEN_KIND_TRAVERSAL_SURFACE,
    _CandidatePayload,
    _CandidateSource,
    _LOCAL_CELL_OFFSETS,
    _SIDE,
    _SelectedEdges,
    _UNIT_CUBE,
    _Visibility,
    _VisibleCandidates,
    _candidate_visibility,
    _cell_offsets,
    _collision_exception_candidates,
    _combine_candidates,
    _gather_candidates,
    _optional_boolean_evidence,
    _pad_candidates,
    _pre_los_candidates,
    _region_collision_exception_candidates,
    _remap_selected_edges,
    _select_candidates,
    _traversal_candidates,
    _validate_traversal,
)


WORLD_GEOMETRY_TOKEN_SCHEMA = "hytalerl_world_geometry_tokens_v3"
WORLD_GEOMETRY_TOKEN_VERSION = 3

WORLD_TOKEN_KIND_PADDING = 0
WORLD_TOKEN_PROVENANCE_UNKNOWN = 0
WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY = (
    TRAVERSAL_PROVENANCE_NATIVE_EXACT_GEOMETRY
)
WORLD_TOKEN_PROVENANCE_SURROGATE_EXACT_GEOMETRY = 2
WORLD_TOKEN_PROVENANCE_SURROGATE_TRAVERSAL = TRAVERSAL_PROVENANCE_SURROGATE

WORLD_TOKEN_SOURCE_TRAVERSAL = 0
WORLD_TOKEN_SOURCE_COLLISION_EXCEPTION = 1
WORLD_TOKEN_SOURCE_COUNT = 2

WORLD_TOKEN_DIAGNOSTIC_INVALID_ACTOR = jnp.uint32(1)
WORLD_TOKEN_DIAGNOSTIC_ROLE_OPACITY_MISSING = jnp.uint32(1 << 1)
WORLD_TOKEN_DIAGNOSTIC_SOURCE_UNAVAILABLE = jnp.uint32(1 << 2)
WORLD_TOKEN_DIAGNOSTIC_SOURCE_OVERFLOW = jnp.uint32(1 << 3)
WORLD_TOKEN_DIAGNOSTIC_LOS_UNAVAILABLE = jnp.uint32(1 << 4)
WORLD_TOKEN_DIAGNOSTIC_OUTPUT_CAPACITY = jnp.uint32(1 << 5)

_EPSILON = jnp.float32(1.0e-6)
_WORLDGEN_ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[5] / "artifacts" / "worldgen"
)
_REGION_LIBRARY_MANIFEST = (
    _WORLDGEN_ARTIFACT_ROOT / "region-library-pilot-v2" / "manifest.json"
)
_TRAVERSAL_LIBRARY_MANIFEST = (
    _WORLDGEN_ARTIFACT_ROOT
    / "region-library-pilot-v2"
    / "traversal-v2"
    / "traversal-manifest.json"
)


class WorldGeometryTokenObservation(NamedTuple):
    """Fixed-shape policy tokens plus non-policy production diagnostics."""

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


def produce_actor_world_geometry_tokens(
    geometry: GeometryProvider,
    actor_position: Array,
    actor_eye_position: Array,
    actor_forward: Array,
    *,
    role_opaque_mask: Array | None,
    geometry_provenance: int,
    traversal: TraversalTokenObservation | None = None,
    traversal_dynamic_state_visible: Array | None = None,
    traversal_edge_state_visible: Array | None = None,
    geometry_tile_index: Array | None = None,
    token_capacity: int = 32,
    maximum_distance: Array | float = 12.0,
    view_sector_full_angle_radians: Array | float = math.tau,
    max_los_cells: int | None = None,
) -> WorldGeometryTokenObservation:
    """Publish visible local geometry and optional traversal tokens.

    The producer filters every candidate by Hytale's horizontal full-width
    view sector and the certified perception LOS predicate. Runtime occupancy
    and state-gated edges stay hidden unless the caller supplies separate
    observable-state evidence. Any required source, LOS, or capacity failure
    clears the complete actor row.
    """

    if not isinstance(geometry, (GeometryState, RegionGeometryState)):
        raise TypeError("geometry must be GeometryState or RegionGeometryState")
    if (
        isinstance(token_capacity, bool)
        or not isinstance(token_capacity, int)
        or token_capacity <= 0
    ):
        raise ValueError("token_capacity must be a positive integer")
    allowed_provenance = (
        WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY,
        WORLD_TOKEN_PROVENANCE_SURROGATE_EXACT_GEOMETRY,
    )
    if isinstance(geometry_provenance, bool) or geometry_provenance not in (
        allowed_provenance
    ):
        raise ValueError("geometry_provenance must identify an exact provider")

    positions = jnp.asarray(actor_position, dtype=jnp.float32)
    eyes = jnp.asarray(actor_eye_position, dtype=jnp.float32)
    forward = jnp.asarray(actor_forward, dtype=jnp.float32)
    if (
        positions.ndim != 3
        or positions.shape[-1] != 3
        or eyes.shape != positions.shape
        or forward.shape != positions.shape
    ):
        raise ValueError(
            "actor position, eye position, and forward must have shape [B, A, 3]"
        )
    batch, actors, _ = positions.shape
    world = _broadcast_geometry(geometry, batch)
    maximum = _actor_parameter(
        maximum_distance,
        batch,
        actors,
        "maximum_distance",
    )
    view_angle = _actor_parameter(
        view_sector_full_angle_radians,
        batch,
        actors,
        "view_sector_full_angle_radians",
    )
    role_mask, role_mask_present = _actor_role_mask(
        world,
        role_opaque_mask,
        batch,
        actors,
    )
    tile_index = _actor_tile_index(geometry_tile_index, batch, actors)

    forward_xz = forward[..., (0, 2)]
    forward_norm = jnp.linalg.norm(forward_xz, axis=2)
    actor_valid = (
        jnp.all(
            jnp.isfinite(positions) & jnp.isfinite(eyes) & jnp.isfinite(forward),
            axis=2,
        )
        & jnp.isfinite(maximum)
        & (maximum > 0.0)
        & jnp.isfinite(view_angle)
        & (view_angle >= 0.0)
        & (view_angle <= jnp.float32(NATIVE_VIEW_MAX_FULL_ANGLE_RADIANS))
        & (forward_norm > _EPSILON)
    )
    safe_positions = jnp.where(actor_valid[..., None], positions, 0.0)
    safe_eyes = jnp.where(actor_valid[..., None], eyes, 0.0)
    normalized_forward = jnp.where(
        actor_valid[..., None],
        forward_xz / jnp.maximum(forward_norm[..., None], _EPSILON),
        jnp.asarray((0.0, 1.0), dtype=jnp.float32),
    )

    exception_source, exception_source_available = (
        _collision_exception_candidates(
            world,
            safe_positions,
            geometry_provenance,
        )
    )
    exception_pre_los = _pre_los_candidates(
        exception_source.mask,
        exception_source.payload.relative_position,
        actor_valid,
        maximum,
        normalized_forward,
        view_angle,
    )
    exception_los = _candidate_visibility(
        world,
        safe_eyes,
        exception_source.los_target,
        exception_pre_los,
        role_mask,
        target_cell=True,
        max_cells=max_los_cells,
    )

    traversal_capacity = 0
    edge_capacity = 1
    traversal_source_available = jnp.zeros(
        (batch, actors),
        dtype=jnp.bool_,
    )
    traversal_source_overflow = jnp.zeros_like(traversal_source_available)
    traversal_tile = jnp.full((batch, actors), -1, dtype=jnp.int32)
    traversal_los_failure = jnp.zeros_like(traversal_source_available)
    traversal_candidates: _VisibleCandidates | None = None
    traversal_edge_visible: Array | None = None
    if traversal is not None:
        traversal_capacity, edge_capacity = _validate_traversal(
            traversal,
            batch,
            actors,
        )
        traversal_provenance_valid = (
            (
                traversal.provenance
                == jnp.uint8(
                    WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY
                )
            )
            | (
                traversal.provenance
                == jnp.uint8(
                    WORLD_TOKEN_PROVENANCE_SURROGATE_TRAVERSAL
                )
            )
        )
        traversal_source_available = (
            traversal.available & traversal_provenance_valid
        )
        traversal_source_overflow = traversal.capacity_exceeded
        traversal_tile = jnp.where(
            traversal_source_available,
            traversal.tile_index,
            -1,
        )
        traversal_source, traversal_edge_visible = _traversal_candidates(
            traversal,
            safe_positions,
            safe_eyes,
            traversal_dynamic_state_visible,
            traversal_edge_state_visible,
        )
        traversal_pre_los = _pre_los_candidates(
            traversal_source.mask,
            traversal_source.payload.relative_position,
            actor_valid,
            maximum,
            normalized_forward,
            view_angle,
        )
        traversal_los = _candidate_visibility(
            world,
            safe_eyes,
            traversal_source.los_target,
            traversal_pre_los,
            role_mask,
            target_cell=False,
            max_cells=max_los_cells,
        )
        traversal_candidates = _VisibleCandidates(
            traversal_pre_los & traversal_los.visible,
            traversal_source.payload,
        )
        traversal_los_failure = jnp.any(
            traversal_pre_los & traversal_los.failed,
            axis=2,
        )

    exception_candidates = _VisibleCandidates(
        exception_pre_los & exception_los.visible,
        exception_source.payload,
    )
    exception_los_failure = jnp.any(
        exception_pre_los & exception_los.failed,
        axis=2,
    )
    candidates = _combine_candidates(
        traversal_candidates,
        exception_candidates,
    )
    candidates = _pad_candidates(candidates, token_capacity)
    selected, selected_index = _select_candidates(
        candidates,
        token_capacity,
    )

    visible_count = jnp.sum(candidates.eligible, axis=2)
    capacity_exceeded = visible_count > token_capacity
    traversal_unavailable = (
        ~traversal_source_available
        if traversal is not None
        else jnp.zeros_like(actor_valid)
    )
    source_unavailable = traversal_unavailable | ~exception_source_available
    source_overflow = (
        traversal_source_overflow
        if traversal is not None
        else jnp.zeros_like(actor_valid)
    )
    los_failure = exception_los_failure | traversal_los_failure
    row_available = (
        actor_valid
        & role_mask_present
        & ~source_unavailable
        & ~source_overflow
        & ~los_failure
        & ~capacity_exceeded
    )
    token_mask = selected.eligible & row_available[..., None]
    diagnostics = (
        jnp.where(
            actor_valid,
            jnp.uint32(0),
            WORLD_TOKEN_DIAGNOSTIC_INVALID_ACTOR,
        )
        | jnp.where(
            role_mask_present,
            jnp.uint32(0),
            WORLD_TOKEN_DIAGNOSTIC_ROLE_OPACITY_MISSING,
        )
        | jnp.where(
            source_unavailable,
            WORLD_TOKEN_DIAGNOSTIC_SOURCE_UNAVAILABLE,
            jnp.uint32(0),
        )
        | jnp.where(
            source_overflow,
            WORLD_TOKEN_DIAGNOSTIC_SOURCE_OVERFLOW,
            jnp.uint32(0),
        )
        | jnp.where(
            los_failure,
            WORLD_TOKEN_DIAGNOSTIC_LOS_UNAVAILABLE,
            jnp.uint32(0),
        )
        | jnp.where(
            capacity_exceeded,
            WORLD_TOKEN_DIAGNOSTIC_OUTPUT_CAPACITY,
            jnp.uint32(0),
        )
    )
    source_available = jnp.stack(
        (
            traversal_source_available,
            exception_source_available,
        ),
        axis=2,
    )
    source_overflow_rows = jnp.stack(
        (
            traversal_source_overflow,
            jnp.zeros_like(actor_valid),
        ),
        axis=2,
    )
    source_tiles = jnp.stack((traversal_tile, tile_index), axis=2)
    edges = _remap_selected_edges(
        traversal,
        traversal_edge_visible,
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

    return WorldGeometryTokenObservation(
        available=row_available,
        capacity_exceeded=capacity_exceeded,
        diagnostics=diagnostics,
        source_available=source_available,
        source_overflow=source_overflow_rows,
        source_tile_index=source_tiles,
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
        collision_boxes_relative=masked(selected.payload.collision_boxes_relative),
    )


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


def _actor_role_mask(
    geometry: GeometryProvider,
    value: Array | None,
    batch: int,
    actors: int,
) -> tuple[Array, Array]:
    if isinstance(geometry, RegionGeometryState):
        expected = geometry.atlas.cell_flags.shape
        shape = (batch, actors) + expected
        if value is None:
            return (
                jnp.zeros(shape, dtype=jnp.bool_),
                jnp.zeros((batch, actors), dtype=jnp.bool_),
            )
        result = jnp.asarray(value)
        if result.dtype != jnp.bool_:
            raise TypeError("role_opaque_mask must have boolean dtype")
        if result.shape == expected:
            result = jnp.broadcast_to(result, shape)
        elif result.shape == (batch,) + expected:
            result = jnp.broadcast_to(result[:, None], shape)
        elif result.shape != shape:
            raise ValueError(
                "Region role_opaque_mask must have shape [R, P], "
                "[B, R, P], or [B, A, R, P]"
            )
        return result, jnp.ones((batch, actors), dtype=jnp.bool_)

    if value is None:
        return (
            jnp.zeros((batch, actors, CELL_COUNT), dtype=jnp.bool_),
            jnp.zeros((batch, actors), dtype=jnp.bool_),
        )
    result = jnp.asarray(value)
    if result.dtype != jnp.bool_:
        raise TypeError("role_opaque_mask must have boolean dtype")
    if result.shape == (CELL_COUNT,):
        result = jnp.broadcast_to(result, (batch, actors, CELL_COUNT))
    elif result.shape == (batch, CELL_COUNT):
        result = jnp.broadcast_to(
            result[:, None, :],
            (batch, actors, CELL_COUNT),
        )
    elif result.shape != (batch, actors, CELL_COUNT):
        raise ValueError("role_opaque_mask must have shape [C], [B, C], or [B, A, C]")
    return result, jnp.ones((batch, actors), dtype=jnp.bool_)


def _actor_tile_index(
    value: Array | None,
    batch: int,
    actors: int,
) -> Array:
    if value is None:
        return jnp.full((batch, actors), -1, dtype=jnp.int32)
    result = jnp.asarray(value, dtype=jnp.int32)
    if result.shape == (batch,):
        result = jnp.broadcast_to(result[:, None], (batch, actors))
    if result.shape != (batch, actors):
        raise ValueError("geometry_tile_index must have shape [B] or [B, A]")
    return result


def _broadcast_geometry(
    geometry: GeometryProvider,
    batch: int,
) -> GeometryProvider:
    if isinstance(geometry, RegionGeometryState):
        if geometry.environment_world_id.shape != (batch,):
            raise ValueError(
                "RegionGeometryState environment batch must match actors"
            )
        return geometry
    size = geometry.origin.shape[0]
    if size not in (1, batch):
        raise ValueError("geometry batch must be one or match actors")
    if size == batch:
        return geometry
    return GeometryState(
        *(
            None
            if leaf is None
            else jnp.broadcast_to(leaf, (batch,) + leaf.shape[1:])
            for leaf in geometry
        )
    )


def world_geometry_token_contract() -> dict[str, object]:
    """Return the checkpoint-pinnable actor token contract."""

    (
        region_library_sha256,
        traversal_library_sha256,
        legal_core_node_count,
    ) = _native_region_staging_artifact_identity()
    return {
        "schema": WORLD_GEOMETRY_TOKEN_SCHEMA,
        "version": WORLD_GEOMETRY_TOKEN_VERSION,
        "status": "actor_legal_producer_implemented",
        "producers": {
            "actor_gate": "produce_actor_world_geometry_tokens",
            "native_region_composition": (
                "produce_actor_region_world_geometry_tokens"
            ),
        },
        "providers": {
            "exact_local_geometry": {
                "providers": "GeometryState or RegionGeometryState",
                "ordinary_surfaces": (
                    "dense_5x5_plus_four_block_cardinal_stencil"
                ),
                "collision_exceptions": "all_exact_non_cube_shapes",
                "surface_choice": "nearest_vertical_certified_full_cube_top",
            },
            "traversal_surface": "optional TraversalTokenObservation",
            "region_traversal": (
                "native-classified RegionTraversalAtlas query projected to "
                "TraversalTokenObservation"
            ),
            "region_exception_projection": (
                "exact actor-centred 9x9x9 Region cell projection"
            ),
        },
        "shape": {
            "row": ["batch", "actor"],
            "token": ["batch", "actor", "token_capacity"],
            "detail_box_capacity_maximum": MAX_DETAIL_BOXES,
            "token_capacity": "caller_selected_static_positive_integer",
            "edge_capacity": "traversal_source_static_or_one_when_absent",
        },
        "native_region_staging": {
            "default_capacity": 56,
            "default_query_distance_blocks": 4.0,
            "scope": (
                "zero_overflow_full_census_of_"
                f"{legal_core_node_count}_legal_core_graph_nodes"
            ),
            "source_region_library_semantic_sha256": region_library_sha256,
            "traversal_library_semantic_sha256": traversal_library_sha256,
            "other_positions_or_libraries": "capacity_checked_fail_closed",
        },
        "policy_fields": [
            "available",
            "token_mask",
            "token_kind",
            "token_provenance",
            "relative_position",
            "clearance",
            "semantic_flags",
            "dynamic_blocked",
            "edge_mask",
            "edge_destination",
            "edge_cost",
            "edge_kind",
            "edge_flags",
            "collision_box_mask",
            "collision_boxes_relative",
        ],
        "non_policy_diagnostics": [
            "capacity_exceeded",
            "diagnostics",
            "source_available",
            "source_overflow",
            "source_tile_index",
            "source_index",
        ],
        "selection": {
            "eligibility": [
                "finite_actor_and_query_parameters",
                "source_available",
                "within_caller_distance",
                "native_horizontal_full_width_view_sector",
                "certified_perception_los_with_explicit_role_mask",
                "observable_runtime_state_only",
            ],
            "ordering": [
                "squared_distance",
                "token_kind",
                "world_cell_x",
                "world_cell_y",
                "world_cell_z",
                "provider_source_index",
            ],
            "overflow": "count_all_actor_legal_candidates_then_clear_row",
            "missing_los_or_source": "clear_row",
            "empty_fully_evaluated_view": "available",
            "truncation": False,
        },
        "visibility": {
            "fov": "NPCPhysicsMath.inViewSector_horizontal_full_width",
            "los": "geometry_perception_line_of_sight_result",
            "target_geometry_cell": (
                "same_predicate_with_terminal_self_occlusion_excluded"
            ),
            "role_opacity": "explicit_per_actor_mask_required",
            "memory": "none_current_frame_only",
        },
        "runtime_state": {
            "dynamic_blocked_default": "hidden",
            "dynamic_blocked_publication": "requires_observable_state_evidence",
            "stateful_edge_default": "hidden",
            "stateful_edge_publication": "requires_observable_state_evidence",
            "static_edges": "published_when_both_endpoint_tokens_are_visible",
        },
        "provenance": {
            str(WORLD_TOKEN_PROVENANCE_UNKNOWN): "unknown_padding_only",
            str(WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY): (
                "native_exact_geometry"
            ),
            str(WORLD_TOKEN_PROVENANCE_SURROGATE_EXACT_GEOMETRY): (
                "surrogate_exact_geometry"
            ),
            str(WORLD_TOKEN_PROVENANCE_SURROGATE_TRAVERSAL): ("surrogate_traversal"),
            "per_token": True,
            "traversal_row_source": (
                "provider_supplied_native_exact_or_surrogate_traversal"
            ),
            "unknown_traversal_provider": "fail_closed",
        },
        "extent": {
            "local_geometry_state": (
                "four_block_immediate_physics_with_bounded_ordinary_surfaces"
            ),
            "navigation_scale": "requires_optional_traversal_source",
            "memory": "capacity_not_dense_query_volume",
        },
        "privilege": {
            "actor_legal": True,
            "leakage_gate": (
                "paired worlds differing only outside FOV_or_LOS_are_bit_exact"
            ),
            "unsupported_or_unavailable": "fail_closed",
        },
    }


def world_geometry_token_contract_sha256() -> str:
    """Return the checkpoint-pinnable canonical contract hash."""

    encoded = json.dumps(
        world_geometry_token_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


#: What the staging identity reads in a checkout with no captured Region
#: library. Deliberately not a hex digest, so a contract built without the
#: library can never be mistaken for one built with it, and the two hash to
#: different values under `world_geometry_token_contract_sha256`.
STAGING_LIBRARY_ABSENT = "absent_no_staged_region_library"


@cache
def _native_region_staging_artifact_identity() -> tuple[str, str, int]:
    # ABSENT IS NOT INVALID. The captured Region library and its traversal
    # graphs are ~13 MB of data derived from the shipped game, so they are not
    # published; a clone therefore legitimately has none. Raising here made
    # `import arena.training` fail outright on a fresh checkout, because the
    # observation schema asserts geometry-token field drift while its own module
    # is still executing -- the token LAYOUT is source and is present, only the
    # provenance of the library it was staged against is missing.
    #
    # Gated on the region manifest alone, and on its ABSENCE only. A manifest
    # that exists but whose graphs or traversal sidecar do not is a broken
    # working copy rather than a clean one, and still fails closed below.
    if not _REGION_LIBRARY_MANIFEST.is_file():
        return (STAGING_LIBRARY_ABSENT, STAGING_LIBRARY_ABSENT, 0)
    region = _read_json_object(_REGION_LIBRARY_MANIFEST)
    traversal = _read_json_object(_TRAVERSAL_LIBRARY_MANIFEST)
    region_sha256 = _sha256_field(
        region.get("library_semantic_sha256"),
        "Region library semantic SHA-256",
    )
    traversal_sha256 = _sha256_field(
        traversal.get("library_semantic_sha256"),
        "traversal library semantic SHA-256",
    )
    region_payload = dict(region)
    region_payload.pop("library_semantic_sha256", None)
    if _canonical_sha256(region_payload) != region_sha256:
        raise RuntimeError("Region library manifest hash changed")
    if (
        region_traversal_library_semantic_sha256(traversal)
        != traversal_sha256
    ):
        raise RuntimeError("traversal library manifest hash changed")
    if traversal.get("source_region_library_semantic_sha256") != region_sha256:
        raise RuntimeError("Region and traversal artifact identities differ")
    return (
        region_sha256,
        traversal_sha256,
        _legal_core_node_count(traversal),
    )


def _legal_core_node_count(traversal: dict[str, object]) -> int:
    entries = traversal.get("entries")
    if (
        not isinstance(entries, list)
        or isinstance(traversal.get("graph_count"), bool)
        or traversal.get("graph_count") != len(entries)
    ):
        raise RuntimeError("traversal graph count is invalid")
    graph_root = _TRAVERSAL_LIBRARY_MANIFEST.parent.resolve()
    core_width = CORE_CHUNKS_PER_AXIS * CHUNK_SIZE
    count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("traversal graph entry must be an object")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError("traversal graph path must be non-empty")
        graph_path = (graph_root / relative).resolve()
        try:
            graph_path.relative_to(graph_root)
        except ValueError as error:
            raise RuntimeError("traversal graph path escapes its library") from error
        if _file_sha256(graph_path) != _sha256_field(
            entry.get("file_sha256"),
            "traversal graph file SHA-256",
        ):
            raise RuntimeError("traversal graph file hash changed")
        with np.load(graph_path, allow_pickle=False) as graph:
            core_min = graph["core_min_chunk_xz"]
            positions = graph["node_position"]
        if (
            core_min.shape != (2,)
            or not np.issubdtype(core_min.dtype, np.integer)
            or positions.ndim != 2
            or positions.shape[1] != 3
            or not np.isfinite(positions).all()
        ):
            raise RuntimeError("traversal graph core-node arrays are invalid")
        core_origin = core_min.astype(np.int64) * CHUNK_SIZE
        block_x = np.floor(positions[:, 0]).astype(np.int64)
        block_z = np.floor(positions[:, 2]).astype(np.int64)
        count += int(
            np.count_nonzero(
                (block_x >= core_origin[0])
                & (block_x < core_origin[0] + core_width)
                & (block_z >= core_origin[1])
                & (block_z < core_origin[1] + core_width)
            )
        )
    if count <= 0:
        raise RuntimeError("traversal library has no legal core graph nodes")
    return count


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read World contract artifact {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"World contract artifact {path} must be an object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise RuntimeError(f"cannot read traversal graph {path}") from error
    return digest.hexdigest()


def _sha256_field(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise RuntimeError(f"{label} must be a 64-hex digest")
    return value.lower()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "WORLD_GEOMETRY_TOKEN_SCHEMA",
    "WORLD_GEOMETRY_TOKEN_VERSION",
    "WORLD_TOKEN_DIAGNOSTIC_INVALID_ACTOR",
    "WORLD_TOKEN_DIAGNOSTIC_LOS_UNAVAILABLE",
    "WORLD_TOKEN_DIAGNOSTIC_OUTPUT_CAPACITY",
    "WORLD_TOKEN_DIAGNOSTIC_ROLE_OPACITY_MISSING",
    "WORLD_TOKEN_DIAGNOSTIC_SOURCE_OVERFLOW",
    "WORLD_TOKEN_DIAGNOSTIC_SOURCE_UNAVAILABLE",
    "WORLD_TOKEN_KIND_COLLISION_EXCEPTION",
    "WORLD_TOKEN_KIND_PADDING",
    "WORLD_TOKEN_KIND_TRAVERSAL_SURFACE",
    "WORLD_TOKEN_PROVENANCE_NATIVE_EXACT_GEOMETRY",
    "WORLD_TOKEN_PROVENANCE_SURROGATE_EXACT_GEOMETRY",
    "WORLD_TOKEN_PROVENANCE_SURROGATE_TRAVERSAL",
    "WORLD_TOKEN_PROVENANCE_UNKNOWN",
    "WORLD_TOKEN_SOURCE_COLLISION_EXCEPTION",
    "WORLD_TOKEN_SOURCE_COUNT",
    "WORLD_TOKEN_SOURCE_TRAVERSAL",
    "WorldGeometryTokenObservation",
    "produce_actor_world_geometry_tokens",
    "world_geometry_token_contract",
    "world_geometry_token_contract_sha256",
]
