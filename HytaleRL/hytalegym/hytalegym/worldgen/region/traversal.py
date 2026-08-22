"""Exact Region surfaces and native MotionControllerWalk graph artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from hytalegym.worldgen.native_traversal import (
    NativeTraversalEdgeProbeCapture,
    native_traversal_edge_probe_contract_sha256,
    region_traversal_projection_contract_sha256,
)
from hytalegym.worldgen.region.contract import (
    CHUNK_SIZE,
)
from hytalegym.worldgen.region.snapshot import NativeRegionSnapshot
from hytalegym.worldgen.region._traversal_columns import (
    _RegionColumns,
)
from hytalegym.worldgen.region._traversal_edges import (
    _candidate_edges,
    _fluid_clear_edges,
)
from hytalegym.worldgen.region._traversal_validate import (
    _array,
    _exact_nonnegative_int,
    _finite_nonnegative,
    _nonempty,
    _positive_finite,
    _sha256_text,
    _supported_graph_version,
)

REGION_TRAVERSAL_GRAPH_SCHEMA = "hytalerl_native_region_traversal_graph_v3"
REGION_TRAVERSAL_GRAPH_VERSION = 3
REGION_TRAVERSAL_EDGE_WALK = 1
REGION_TRAVERSAL_EDGE_CLIMB = 2
REGION_TRAVERSAL_EDGE_DROP = 3
REGION_TRAVERSAL_EDGE_SAFE = 1
REGION_TRAVERSAL_EDGES_PER_NODE = 12
REGION_TRAVERSAL_DEFAULT_QUERY_DISTANCE_CAPACITY = 12.0
REGION_TRAVERSAL_HORIZONTAL_TOLERANCE = 2.0e-4
REGION_TRAVERSAL_VERTICAL_TOLERANCE = 2.0e-4

_EPSILON = 1.0e-6
_SUPPORT_EPSILON = 2.0e-4
_FLUID_POLICY = "exclude_swept_actor_aabb_with_one_cell_horizontal_halo"
_GRAPH_SCHEMAS = {
    1: "hytalerl_native_region_traversal_graph_v1",
    2: "hytalerl_native_region_traversal_graph_v2",
    REGION_TRAVERSAL_GRAPH_VERSION: REGION_TRAVERSAL_GRAPH_SCHEMA,
}
_GRAPH_METADATA_KEY = "__metadata_json__"
_GRAPH_FIELDS = frozenset(
    {
        _GRAPH_METADATA_KEY,
        "core_min_chunk_xz",
        "actor_bounds",
        "node_position",
        "node_clearance",
        "edge_mask",
        "edge_destination",
        "edge_cost",
        "edge_kind",
        "edge_flags",
    }
)
_DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
@dataclass(frozen=True)
class RegionTraversalCandidates:
    """Unpadded exact surfaces and all native-controller edge probes."""

    source_region_semantic_sha256: str
    core_min_chunk_xz: np.ndarray
    actor_bounds: np.ndarray
    maximum_climb_height: float
    maximum_drop_height: float
    query_distance_capacity: float
    node_position: np.ndarray
    node_clearance: np.ndarray
    edge_source: np.ndarray
    edge_destination: np.ndarray
    edge_kind: np.ndarray
    fluid_excluded_candidate_count: int

    def __post_init__(self) -> None:
        digest = _sha256_text(
            self.source_region_semantic_sha256,
            "source_region_semantic_sha256",
        )
        arrays = {
            "core_min_chunk_xz": _array(
                self.core_min_chunk_xz,
                np.int32,
            ),
            "actor_bounds": _array(self.actor_bounds, np.float64),
            "node_position": _array(self.node_position, np.float64),
            "node_clearance": _array(self.node_clearance, np.float64),
            "edge_source": _array(self.edge_source, np.int32),
            "edge_destination": _array(
                self.edge_destination,
                np.int32,
            ),
            "edge_kind": _array(self.edge_kind, np.uint8),
        }
        nodes = arrays["node_position"].shape[0]
        edges = arrays["edge_source"].shape[0]
        expected = {
            "core_min_chunk_xz": (2,),
            "actor_bounds": (6,),
            "node_position": (nodes, 3),
            "node_clearance": (nodes,),
            "edge_source": (edges,),
            "edge_destination": (edges,),
            "edge_kind": (edges,),
        }
        if nodes < 1 or edges < 1:
            raise ValueError("Region traversal candidates cannot be empty")
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        bounds = arrays["actor_bounds"]
        if (
            not np.all(np.isfinite(bounds))
            or np.any(bounds[:3] >= bounds[3:])
        ):
            raise ValueError("actor_bounds must be one positive finite AABB")
        numeric = (
            arrays["node_position"],
            arrays["node_clearance"],
        )
        if any(not np.all(np.isfinite(value)) for value in numeric):
            raise ValueError("Region traversal candidates must be finite")
        if np.any(arrays["node_clearance"] < 0.0):
            raise ValueError("node clearance must be non-negative")
        if np.any(arrays["edge_source"] < 0) or np.any(
            arrays["edge_source"] >= nodes
        ):
            raise ValueError("edge source exceeds the node table")
        if np.any(arrays["edge_destination"] < 0) or np.any(
            arrays["edge_destination"] >= nodes
        ):
            raise ValueError("edge destination exceeds the node table")
        if np.any(
            ~np.isin(
                arrays["edge_kind"],
                (
                    REGION_TRAVERSAL_EDGE_WALK,
                    REGION_TRAVERSAL_EDGE_CLIMB,
                    REGION_TRAVERSAL_EDGE_DROP,
                ),
            )
        ):
            raise ValueError("edge kind is unknown")
        climb = _finite_nonnegative(
            self.maximum_climb_height,
            "maximum_climb_height",
        )
        drop = _finite_nonnegative(
            self.maximum_drop_height,
            "maximum_drop_height",
        )
        query_distance = _positive_finite(
            self.query_distance_capacity,
            "query_distance_capacity",
        )
        fluid_excluded = _exact_nonnegative_int(
            self.fluid_excluded_candidate_count,
            "fluid_excluded_candidate_count",
        )
        for name, value in arrays.items():
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "source_region_semantic_sha256", digest)
        object.__setattr__(self, "maximum_climb_height", climb)
        object.__setattr__(self, "maximum_drop_height", drop)
        object.__setattr__(
            self,
            "query_distance_capacity",
            query_distance,
        )
        object.__setattr__(
            self,
            "fluid_excluded_candidate_count",
            fluid_excluded,
        )

    @property
    def edge_start_position(self) -> np.ndarray:
        return self.node_position[self.edge_source]

    @property
    def edge_target_position(self) -> np.ndarray:
        return self.node_position[self.edge_destination]


@dataclass(frozen=True)
class NativeRegionTraversalGraph:
    """One immutable native-classified graph keyed to an exact Region."""

    metadata: Mapping[str, Any]
    core_min_chunk_xz: np.ndarray
    actor_bounds: np.ndarray
    node_position: np.ndarray
    node_clearance: np.ndarray
    edge_mask: np.ndarray
    edge_destination: np.ndarray
    edge_cost: np.ndarray
    edge_kind: np.ndarray
    edge_flags: np.ndarray

    def __post_init__(self) -> None:
        metadata = dict(self.metadata)
        version = _supported_graph_version(metadata.get("version"))
        contract = region_traversal_graph_contract(version)
        if metadata.get("schema") != contract["schema"]:
            raise ValueError("unsupported Region traversal graph schema")
        if metadata.get("edge_authority") != (
            "native_MotionControllerWalk_probeMove"
        ):
            raise ValueError("Region traversal edges lack native authority")
        if metadata.get("graph_contract_sha256") != (
            region_traversal_graph_contract_sha256(version)
        ):
            raise ValueError("Region traversal graph contract changed")
        _sha256_text(
            metadata.get("source_region_semantic_sha256"),
            "source Region semantic SHA-256",
        )
        _sha256_text(
            metadata.get("native_evidence_jar_sha256"),
            "native evidence bridge SHA-256",
        )
        if metadata.get("native_edge_contract_sha256") != (
            native_traversal_edge_probe_contract_sha256()
        ):
            raise ValueError("native traversal edge contract changed")
        if metadata.get("stationary_projection_contract_sha256") != (
            region_traversal_projection_contract_sha256()
        ):
            raise ValueError("stationary traversal projection changed")
        if metadata.get("fluid_policy") != _FLUID_POLICY:
            raise ValueError("Region traversal fluid policy changed")
        if version >= 3:
            selector = np.asarray(
                metadata.get("direction_component_selector"),
                dtype=np.float64,
            )
            if (
                selector.shape != (3,)
                or not np.all(np.isfinite(selector))
                or np.any(selector < 0.0)
                or not np.any(selector > 0.0)
            ):
                raise ValueError(
                    "Region traversal direction selector is invalid"
                )
            if metadata.get("edge_cost_authority") != (
                "native_MotionControllerWalk_probeMove_return_distance"
            ):
                raise ValueError("Region traversal edge costs lack authority")
        arrays = {
            "core_min_chunk_xz": _array(
                self.core_min_chunk_xz,
                np.int32,
            ),
            "actor_bounds": _array(self.actor_bounds, np.float32),
            "node_position": _array(self.node_position, np.float32),
            "node_clearance": _array(self.node_clearance, np.float32),
            "edge_mask": _array(self.edge_mask, np.bool_),
            "edge_destination": _array(
                self.edge_destination,
                np.int32,
            ),
            "edge_cost": _array(self.edge_cost, np.float32),
            "edge_kind": _array(self.edge_kind, np.uint8),
            "edge_flags": _array(self.edge_flags, np.uint8),
        }
        nodes = arrays["node_position"].shape[0]
        edge_shape = (nodes, REGION_TRAVERSAL_EDGES_PER_NODE)
        expected = {
            "core_min_chunk_xz": (2,),
            "actor_bounds": (6,),
            "node_position": (nodes, 3),
            "node_clearance": (nodes,),
            "edge_mask": edge_shape,
            "edge_destination": edge_shape,
            "edge_cost": edge_shape,
            "edge_kind": edge_shape,
            "edge_flags": edge_shape,
        }
        if nodes < 1:
            raise ValueError("Region traversal graph cannot be empty")
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        candidate_count = _exact_nonnegative_int(
            metadata.get("candidate_count"),
            "candidate_count",
        )
        fluid_excluded = _exact_nonnegative_int(
            metadata.get("fluid_excluded_candidate_count"),
            "fluid_excluded_candidate_count",
        )
        unfiltered = _exact_nonnegative_int(
            metadata.get("unfiltered_candidate_count"),
            "unfiltered_candidate_count",
        )
        accepted = _exact_nonnegative_int(
            metadata.get("accepted_edge_count"),
            "accepted_edge_count",
        )
        if candidate_count + fluid_excluded != unfiltered:
            raise ValueError("Region traversal fluid exclusion count changed")
        if accepted != int(np.count_nonzero(arrays["edge_mask"])):
            raise ValueError("Region traversal accepted edge count changed")
        if accepted > candidate_count:
            raise ValueError("Region traversal accepts more edges than probed")
        active = arrays["edge_mask"]
        if np.any(
            active
            & (
                (arrays["edge_destination"] < 0)
                | (arrays["edge_destination"] >= nodes)
            )
        ):
            raise ValueError("active edge destination exceeds the graph")
        if np.any(~active & (arrays["edge_destination"] != -1)):
            raise ValueError("inactive edges require destination -1")
        if version >= 3 and (
            np.any(~np.isfinite(arrays["edge_cost"]))
            or np.any(active & (arrays["edge_cost"] <= 0.0))
            or np.any(~active & (arrays["edge_cost"] != 0.0))
        ):
            raise ValueError("Region traversal edge costs are invalid")
        if not np.all(np.isfinite(arrays["node_position"])) or not np.all(
            np.isfinite(arrays["node_clearance"])
        ):
            raise ValueError("Region traversal graph must be finite")
        for name, value in arrays.items():
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "metadata", metadata)
        declared = _sha256_text(
            metadata.get("graph_semantic_sha256"),
            "graph semantic SHA-256",
        )
        if self.semantic_digest() != declared:
            raise ValueError("Region traversal graph semantic SHA-256 mismatch")

    @property
    def source_region_semantic_sha256(self) -> str:
        return str(self.metadata["source_region_semantic_sha256"])

    @property
    def native_evidence_jar_sha256(self) -> str:
        return str(self.metadata["native_evidence_jar_sha256"]).upper()

    @property
    def node_count(self) -> int:
        return int(self.node_position.shape[0])

    @property
    def edge_count(self) -> int:
        return int(np.count_nonzero(self.edge_mask))

    def semantic_digest(self) -> str:
        digest = hashlib.sha256()
        version = _supported_graph_version(self.metadata.get("version"))
        excluded = {"graph_semantic_sha256"}
        if version >= 2:
            excluded.add("native_evidence_jar_sha256")
        stable = {
            key: value
            for key, value in self.metadata.items()
            if key not in excluded
        }
        digest.update(
            json.dumps(
                stable,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        for value in (
            self.core_min_chunk_xz,
            self.actor_bounds,
            self.node_position,
            self.node_clearance,
            self.edge_mask,
            self.edge_destination,
            self.edge_cost,
            self.edge_kind,
            self.edge_flags,
        ):
            contiguous = np.ascontiguousarray(value)
            digest.update(contiguous.dtype.str.encode())
            digest.update(contiguous.tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            _GRAPH_METADATA_KEY: np.asarray(
                json.dumps(
                    self.metadata,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            "core_min_chunk_xz": self.core_min_chunk_xz,
            "actor_bounds": self.actor_bounds,
            "node_position": self.node_position,
            "node_clearance": self.node_clearance,
            "edge_mask": self.edge_mask,
            "edge_destination": self.edge_destination,
            "edge_cost": self.edge_cost,
            "edge_kind": self.edge_kind,
            "edge_flags": self.edge_flags,
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as output:
                temporary = Path(output.name)
                np.savez_compressed(output, **payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> NativeRegionTraversalGraph:
        with np.load(Path(path), allow_pickle=False) as archive:
            if set(archive.files) != _GRAPH_FIELDS:
                raise ValueError("Region traversal graph fields changed")
            return cls(
                metadata=json.loads(str(archive[_GRAPH_METADATA_KEY].item())),
                core_min_chunk_xz=archive["core_min_chunk_xz"],
                actor_bounds=archive["actor_bounds"],
                node_position=archive["node_position"],
                node_clearance=archive["node_clearance"],
                edge_mask=archive["edge_mask"],
                edge_destination=archive["edge_destination"],
                edge_cost=archive["edge_cost"],
                edge_kind=archive["edge_kind"],
                edge_flags=archive["edge_flags"],
            )


def upgrade_region_traversal_graph_identity(
    graph: NativeRegionTraversalGraph,
) -> NativeRegionTraversalGraph:
    """Move legacy v1 identity to provenance-neutral v2 without changing data.

    Graph v3 corrects native Walk edge costs and therefore requires fresh
    native probe evidence; it cannot be produced by an identity-only upgrade.
    """

    if not isinstance(graph, NativeRegionTraversalGraph):
        raise TypeError("graph must be a NativeRegionTraversalGraph")
    metadata = dict(graph.metadata)
    metadata.update(
        {
            "schema": _GRAPH_SCHEMAS[2],
            "version": 2,
            "graph_contract_sha256": (
                region_traversal_graph_contract_sha256(2)
            ),
            "graph_semantic_sha256": "",
        }
    )
    provisional = NativeRegionTraversalGraph.__new__(
        NativeRegionTraversalGraph
    )
    values = {
        "metadata": metadata,
        "core_min_chunk_xz": graph.core_min_chunk_xz,
        "actor_bounds": graph.actor_bounds,
        "node_position": graph.node_position,
        "node_clearance": graph.node_clearance,
        "edge_mask": graph.edge_mask,
        "edge_destination": graph.edge_destination,
        "edge_cost": graph.edge_cost,
        "edge_kind": graph.edge_kind,
        "edge_flags": graph.edge_flags,
    }
    for name, value in values.items():
        object.__setattr__(provisional, name, value)
    metadata["graph_semantic_sha256"] = provisional.semantic_digest()
    return NativeRegionTraversalGraph(**values)


def compile_region_traversal_candidates(
    snapshot: NativeRegionSnapshot,
    *,
    actor_bounds: Sequence[float],
    maximum_climb_height: float,
    maximum_drop_height: float,
    query_distance_capacity: float = (
        REGION_TRAVERSAL_DEFAULT_QUERY_DISTANCE_CAPACITY
    ),
) -> RegionTraversalCandidates:
    """Compile every standable core surface and candidate 8-neighbour edge."""

    if not isinstance(snapshot, NativeRegionSnapshot):
        raise TypeError("snapshot must be a NativeRegionSnapshot")
    if not snapshot.filler_root_offsets_available:
        raise ValueError("Region traversal requires section-v3 filler roots")
    if not np.all(snapshot.section_known):
        raise ValueError("Region traversal requires a complete capture")
    bounds = np.asarray(actor_bounds, dtype=np.float64)
    if (
        bounds.shape != (6,)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[:3] >= bounds[3:])
    ):
        raise ValueError("actor_bounds must be one positive finite AABB")
    # The current fixed graph samples half-cell centers. Wider or asymmetric
    # actors visit neighbouring X/Z cells even while stationary and therefore
    # need a different candidate lattice.
    if (
        math.floor(0.5 + bounds[0]) != 0
        or math.floor(0.5 + bounds[2]) != 0
        or math.floor(0.5 + bounds[3] - _EPSILON) != 0
        or math.floor(0.5 + bounds[5] - _EPSILON) != 0
    ):
        raise ValueError(
            "half-cell Region traversal does not support this actor width"
        )
    climb = _finite_nonnegative(
        maximum_climb_height,
        "maximum_climb_height",
    )
    drop = _finite_nonnegative(maximum_drop_height, "maximum_drop_height")
    query_distance = _positive_finite(
        query_distance_capacity,
        "query_distance_capacity",
    )
    margin = math.ceil(query_distance) + 1
    if margin >= CHUNK_SIZE:
        raise ValueError(
            "query distance exceeds the captured one-chunk traversal halo"
        )
    geometry = _RegionColumns(snapshot, bounds, margin)
    positions, clearance = geometry.standable_nodes()
    edge_source, edge_destination, edge_kind = _candidate_edges(
        positions,
        bounds,
        climb,
    )
    unfiltered_count = int(edge_source.size)
    fluid_clear = _fluid_clear_edges(
        snapshot,
        positions,
        edge_source,
        edge_destination,
        bounds,
    )
    edge_source = edge_source[fluid_clear]
    edge_destination = edge_destination[fluid_clear]
    edge_kind = edge_kind[fluid_clear]
    return RegionTraversalCandidates(
        source_region_semantic_sha256=snapshot.semantic_artifact_digest(),
        core_min_chunk_xz=snapshot.core_min_chunk_xz,
        actor_bounds=bounds,
        maximum_climb_height=climb,
        maximum_drop_height=drop,
        query_distance_capacity=query_distance,
        node_position=positions,
        node_clearance=clearance,
        edge_source=edge_source,
        edge_destination=edge_destination,
        edge_kind=edge_kind,
        fluid_excluded_candidate_count=(
            unfiltered_count - int(edge_source.size)
        ),
    )


def native_region_traversal_graph_from_captures(
    candidates: RegionTraversalCandidates,
    captures: Sequence[NativeTraversalEdgeProbeCapture],
    *,
    native_evidence_jar_sha256: str,
    actor_profile: str,
) -> NativeRegionTraversalGraph:
    """Publish only edges accepted by complete native probe evidence."""

    if not isinstance(candidates, RegionTraversalCandidates):
        raise TypeError("candidates must be RegionTraversalCandidates")
    rows = tuple(captures)
    if not rows or any(
        not isinstance(value, NativeTraversalEdgeProbeCapture)
        for value in rows
    ):
        raise TypeError("captures must contain native edge probe results")
    starts = np.concatenate([value.start_positions for value in rows])
    targets = np.concatenate([value.target_positions for value in rows])
    horizontal = np.concatenate(
        [value.horizontal_arrival_tolerances for value in rows]
    )
    vertical = np.concatenate(
        [value.vertical_arrival_tolerances for value in rows]
    )
    reachable = np.concatenate([value.reachable for value in rows])
    final_positions = np.concatenate(
        [value.final_positions for value in rows]
    )
    travelled_distances = np.concatenate(
        [value.travelled_distances for value in rows]
    )
    expected_start = candidates.edge_start_position
    expected_target = candidates.edge_target_position
    np.testing.assert_allclose(starts, expected_start, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(targets, expected_target, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        horizontal,
        REGION_TRAVERSAL_HORIZONTAL_TOLERANCE,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        vertical,
        REGION_TRAVERSAL_VERTICAL_TOLERANCE,
        rtol=0.0,
        atol=0.0,
    )
    first = rows[0]
    for value in rows:
        if (
            value.seed != first.seed
            or value.server_version != first.server_version
            or value.worldgen_provider != first.worldgen_provider
            or value.worldgen_version != first.worldgen_version
        ):
            raise ValueError("native traversal capture provenance changed")
        np.testing.assert_allclose(
            value.actor_bounds,
            candidates.actor_bounds,
            rtol=0.0,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            value.direction_component_selector,
            first.direction_component_selector,
            rtol=0.0,
            atol=0.0,
        )
        if (
            value.maximum_climb_height
            != candidates.maximum_climb_height
            or value.maximum_drop_height != candidates.maximum_drop_height
        ):
            raise ValueError("native traversal controller profile changed")
    selector = first.direction_component_selector
    if np.any(selector < 0.0) or not np.any(selector > 0.0):
        raise ValueError("native traversal direction selector is invalid")
    expected_distances = np.linalg.norm(
        (final_positions - starts) * selector,
        axis=1,
    )
    np.testing.assert_allclose(
        travelled_distances,
        expected_distances,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    node_count = candidates.node_position.shape[0]
    edge_shape = (node_count, REGION_TRAVERSAL_EDGES_PER_NODE)
    edge_mask = np.zeros(edge_shape, dtype=np.bool_)
    edge_destination = np.full(edge_shape, -1, dtype=np.int32)
    edge_cost = np.zeros(edge_shape, dtype=np.float32)
    edge_kind = np.zeros(edge_shape, dtype=np.uint8)
    edge_flags = np.zeros(edge_shape, dtype=np.uint8)
    per_node = np.zeros(node_count, dtype=np.uint8)
    for candidate_index in np.flatnonzero(reachable):
        source = int(candidates.edge_source[candidate_index])
        slot = int(per_node[source])
        if slot >= REGION_TRAVERSAL_EDGES_PER_NODE:
            raise ValueError(
                "native traversal edges exceed fixed per-node capacity"
            )
        destination = int(candidates.edge_destination[candidate_index])
        edge_mask[source, slot] = True
        edge_destination[source, slot] = destination
        edge_cost[source, slot] = travelled_distances[candidate_index]
        edge_kind[source, slot] = candidates.edge_kind[candidate_index]
        edge_flags[source, slot] = REGION_TRAVERSAL_EDGE_SAFE
        per_node[source] += 1
    metadata: dict[str, Any] = {
        "schema": REGION_TRAVERSAL_GRAPH_SCHEMA,
        "version": REGION_TRAVERSAL_GRAPH_VERSION,
        "server_version": first.server_version,
        "worldgen_provider": first.worldgen_provider,
        "worldgen_version": first.worldgen_version,
        "seed": first.seed,
        "actor_profile": _nonempty(actor_profile, "actor_profile"),
        "source_region_semantic_sha256": (
            candidates.source_region_semantic_sha256
        ),
        "native_evidence_jar_sha256": _sha256_text(
            native_evidence_jar_sha256,
            "native_evidence_jar_sha256",
        ).upper(),
        "native_edge_contract_sha256": (
            native_traversal_edge_probe_contract_sha256()
        ),
        "graph_contract_sha256": (
            region_traversal_graph_contract_sha256()
        ),
        "stationary_projection_contract_sha256": (
            region_traversal_projection_contract_sha256()
        ),
        "edge_authority": "native_MotionControllerWalk_probeMove",
        "edge_cost_authority": (
            "native_MotionControllerWalk_probeMove_return_distance"
        ),
        "direction_component_selector": selector.tolist(),
        "candidate_count": int(reachable.size),
        "unfiltered_candidate_count": (
            int(reachable.size) + candidates.fluid_excluded_candidate_count
        ),
        "fluid_excluded_candidate_count": (
            candidates.fluid_excluded_candidate_count
        ),
        "fluid_policy": _FLUID_POLICY,
        "accepted_edge_count": int(np.count_nonzero(reachable)),
        "maximum_climb_height": candidates.maximum_climb_height,
        "maximum_drop_height": candidates.maximum_drop_height,
        "query_distance_capacity": candidates.query_distance_capacity,
        "horizontal_arrival_tolerance": (
            REGION_TRAVERSAL_HORIZONTAL_TOLERANCE
        ),
        "vertical_arrival_tolerance": REGION_TRAVERSAL_VERTICAL_TOLERANCE,
        "graph_semantic_sha256": "",
    }
    provisional = NativeRegionTraversalGraph.__new__(
        NativeRegionTraversalGraph
    )
    for name, value in {
        "metadata": metadata,
        "core_min_chunk_xz": candidates.core_min_chunk_xz,
        "actor_bounds": candidates.actor_bounds.astype(np.float32),
        "node_position": candidates.node_position.astype(np.float32),
        "node_clearance": candidates.node_clearance.astype(np.float32),
        "edge_mask": edge_mask,
        "edge_destination": edge_destination,
        "edge_cost": edge_cost,
        "edge_kind": edge_kind,
        "edge_flags": edge_flags,
    }.items():
        object.__setattr__(provisional, name, value)
    metadata["graph_semantic_sha256"] = provisional.semantic_digest()
    return NativeRegionTraversalGraph(
        metadata=metadata,
        core_min_chunk_xz=candidates.core_min_chunk_xz,
        actor_bounds=candidates.actor_bounds,
        node_position=candidates.node_position,
        node_clearance=candidates.node_clearance,
        edge_mask=edge_mask,
        edge_destination=edge_destination,
        edge_cost=edge_cost,
        edge_kind=edge_kind,
        edge_flags=edge_flags,
    )


def region_traversal_graph_contract(
    version: int = REGION_TRAVERSAL_GRAPH_VERSION,
) -> dict[str, object]:
    """Return the immutable Region graph/capture/runtime contract."""

    resolved = _supported_graph_version(version)
    contract: dict[str, object] = {
        "schema": _GRAPH_SCHEMAS[resolved],
        "version": resolved,
        "source": {
            "terrain": "exact_frozen_native_Region_section_v3",
            "stationary_nodes": (
                "filler-aware_half-cell_surface_projection"
            ),
            "moving_edges": "native_MotionControllerWalk_probeMove",
            "actor_profile": "artifact_explicit_not_global_constant",
        },
        "coverage": {
            "actor_domain": "complete_3x3_chunk_core",
            "capture": "core_plus_query_distance_and_one_edge_cell",
            "vertical": "[0,320)_all_exact_section_elevations",
            "neighbours": "eight_horizontal_columns",
            "query_distance_capacity": "artifact_explicit_at_most_one_chunk_halo",
        },
        "edge_classification": {
            "walk": "same_foot_height",
            "climb": "positive_delta_within_native_maximum_climb_height",
            "drop": "first_highest_landing_below",
            "acceptance": (
                "native_horizontal_and_vertical_target_tolerances"
            ),
            "damage_avoidance": "ProbeMoveData_default_true",
            "fluid_adjacent": _FLUID_POLICY,
            "fluid_runtime": "unsupported_fail_closed",
            "maximum_edges_per_node": REGION_TRAVERSAL_EDGES_PER_NODE,
        },
        "runtime": {
            "layout": "fixed_capacity_padded_selected_working_set",
            "selection": "per_environment_world_and_per_actor_core",
            "row": "TraversalTokenObservation",
            "provenance": "native_exact_geometry",
            "overflow": "clear_complete_actor_row",
            "unsupported_profile_or_distance": "fail_closed",
            "seed_regeneration": "never_required_for_training",
        },
    }
    if resolved >= 2:
        contract["identity"] = {
            "meaning": (
                "all_nine_graph_arrays_and_metadata_except_declared_digest_"
                "and_native_evidence_bridge"
            ),
            "provenance": (
                "native_evidence_jar_sha256_recorded_but_not_hashed"
            ),
            "recertification": (
                "unchanged_graph_meaning_retains_graph_semantic_sha256"
            ),
        }
    if resolved >= 3:
        contract["edge_cost"] = {
            "authority": (
                "native_MotionControllerWalk_probeMove_return_distance"
            ),
            "component_selector": (
                "captured_from_active_MotionControllerWalk"
            ),
            "vertical_cost": (
                "masked_when_the_captured_component_selector_y_is_zero"
            ),
            "legacy_v1_v2": (
                "three_dimensional_euclidean_cost_not_upgradeable_without_"
                "native_probe_evidence"
            ),
        }
    return contract


def region_traversal_graph_contract_sha256(
    version: int = REGION_TRAVERSAL_GRAPH_VERSION,
) -> str:
    encoded = json.dumps(
        region_traversal_graph_contract(version),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "NativeRegionTraversalGraph",
    "REGION_TRAVERSAL_EDGES_PER_NODE",
    "REGION_TRAVERSAL_DEFAULT_QUERY_DISTANCE_CAPACITY",
    "REGION_TRAVERSAL_EDGE_CLIMB",
    "REGION_TRAVERSAL_EDGE_DROP",
    "REGION_TRAVERSAL_EDGE_SAFE",
    "REGION_TRAVERSAL_EDGE_WALK",
    "REGION_TRAVERSAL_GRAPH_SCHEMA",
    "REGION_TRAVERSAL_GRAPH_VERSION",
    "REGION_TRAVERSAL_HORIZONTAL_TOLERANCE",
    "REGION_TRAVERSAL_VERTICAL_TOLERANCE",
    "RegionTraversalCandidates",
    "compile_region_traversal_candidates",
    "native_region_traversal_graph_from_captures",
    "region_traversal_graph_contract",
    "region_traversal_graph_contract_sha256",
    "upgrade_region_traversal_graph_identity",
]
