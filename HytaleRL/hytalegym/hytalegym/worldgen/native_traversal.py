"""Validated Hytale 0.5.7 CollisionModule traversal probes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Protocol

import numpy as np

NATIVE_TRAVERSAL_PROBE_SCHEMA = "hytalerl_native_traversal_probe_v1"
NATIVE_TRAVERSAL_PROBE_VERSION = 1
NATIVE_TRAVERSAL_PROBE_MAX_SAMPLES = 4096
NATIVE_TRAVERSAL_EDGE_PROBE_SCHEMA = (
    "hytalerl_native_traversal_edge_probe_v1"
)
NATIVE_TRAVERSAL_EDGE_PROBE_VERSION = 1
NATIVE_TRAVERSAL_EDGE_PROBE_MAX_SAMPLES = 4096
REGION_TRAVERSAL_PROJECTION_SCHEMA = (
    "hytalerl_region_traversal_projection_contract_v2"
)
REGION_TRAVERSAL_PROJECTION_VERSION = 2
VALIDATE_INVALID = -1
VALIDATE_CLEAR = 0
VALIDATE_ON_GROUND = 1
VALIDATE_TOUCH_CEILING = 2


class NativeTraversalProbeTransport(Protocol):
    """One bounded probe verb over a selected static native Region."""

    def capture_traversal_probe(
        self,
        positions: np.ndarray,
        upward_limits: np.ndarray,
    ) -> Mapping[str, Any]: ...

    def capture_traversal_edge_probe(
        self,
        start_positions: np.ndarray,
        target_positions: np.ndarray,
        horizontal_arrival_tolerances: np.ndarray,
        vertical_arrival_tolerances: np.ndarray,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class NativeTraversalProbeCapture:
    """Engine collision results for actor-transform sample positions."""

    server_version: str
    world_name: str
    worldgen_provider: str
    worldgen_version: str
    seed: int
    positions: np.ndarray
    upward_limits: np.ndarray
    actor_bounds: np.ndarray
    validation_codes: np.ndarray
    upward_collision_distances: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "server_version",
            "world_name",
            "worldgen_provider",
            "worldgen_version",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        arrays = {
            "positions": _array(self.positions, np.float64),
            "upward_limits": _array(self.upward_limits, np.float64),
            "actor_bounds": _array(self.actor_bounds, np.float64),
            "validation_codes": _array(self.validation_codes, np.int8),
            "upward_collision_distances": _array(
                self.upward_collision_distances,
                np.float64,
            ),
        }
        samples = arrays["positions"].shape[0]
        expected = {
            "positions": (samples, 3),
            "upward_limits": (samples,),
            "actor_bounds": (6,),
            "validation_codes": (samples,),
            "upward_collision_distances": (samples,),
        }
        if not 1 <= samples <= NATIVE_TRAVERSAL_PROBE_MAX_SAMPLES:
            raise ValueError("native traversal sample capacity exceeded")
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if any(not np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("native traversal arrays must be finite")
        bounds = arrays["actor_bounds"]
        if np.any(bounds[:3] >= bounds[3:]):
            raise ValueError("actor_bounds must be a positive AABB")
        codes = arrays["validation_codes"]
        limits = arrays["upward_limits"]
        distances = arrays["upward_collision_distances"]
        if np.any((codes < VALIDATE_INVALID) | (codes > 3)):
            raise ValueError("native traversal validation code is unknown")
        if np.any(limits < 0.0) or np.any(
            (distances < 0.0) | (distances > limits + 1.0e-9)
        ):
            raise ValueError("native traversal upward distance is invalid")
        if np.any((codes == VALIDATE_INVALID) & (distances != 0.0)):
            raise ValueError("invalid native samples cannot publish clearance")
        for name, value in arrays.items():
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    @property
    def sample_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def standable(self) -> np.ndarray:
        result = (self.validation_codes >= 0) & (
            self.validation_codes & VALIDATE_ON_GROUND
        ).astype(np.bool_)
        result.flags.writeable = False
        return result

    @property
    def clearance(self) -> np.ndarray:
        height = float(self.actor_bounds[4] - self.actor_bounds[1])
        result = np.where(
            self.validation_codes >= 0,
            height + self.upward_collision_distances,
            0.0,
        )
        result.flags.writeable = False
        return result


@dataclass(frozen=True)
class NativeTraversalEdgeProbeCapture:
    """Native MotionControllerWalk reachability for fixed graph edges."""

    server_version: str
    world_name: str
    worldgen_provider: str
    worldgen_version: str
    seed: int
    start_positions: np.ndarray
    target_positions: np.ndarray
    horizontal_arrival_tolerances: np.ndarray
    vertical_arrival_tolerances: np.ndarray
    actor_bounds: np.ndarray
    direction_component_selector: np.ndarray
    maximum_climb_height: float
    maximum_drop_height: float
    reachable: np.ndarray
    edge_blocked: np.ndarray
    final_positions: np.ndarray
    travelled_distances: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "server_version",
            "world_name",
            "worldgen_provider",
            "worldgen_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        arrays = {
            "start_positions": _array(self.start_positions, np.float64),
            "target_positions": _array(self.target_positions, np.float64),
            "horizontal_arrival_tolerances": _array(
                self.horizontal_arrival_tolerances,
                np.float64,
            ),
            "vertical_arrival_tolerances": _array(
                self.vertical_arrival_tolerances,
                np.float64,
            ),
            "actor_bounds": _array(self.actor_bounds, np.float64),
            "direction_component_selector": _array(
                self.direction_component_selector,
                np.float64,
            ),
            "reachable": _array(self.reachable, np.bool_),
            "edge_blocked": _array(self.edge_blocked, np.bool_),
            "final_positions": _array(self.final_positions, np.float64),
            "travelled_distances": _array(
                self.travelled_distances,
                np.float64,
            ),
        }
        samples = arrays["horizontal_arrival_tolerances"].shape[0]
        expected = {
            "start_positions": (samples, 3),
            "target_positions": (samples, 3),
            "horizontal_arrival_tolerances": (samples,),
            "vertical_arrival_tolerances": (samples,),
            "actor_bounds": (6,),
            "direction_component_selector": (3,),
            "reachable": (samples,),
            "edge_blocked": (samples,),
            "final_positions": (samples, 3),
            "travelled_distances": (samples,),
        }
        if not 1 <= samples <= NATIVE_TRAVERSAL_EDGE_PROBE_MAX_SAMPLES:
            raise ValueError("native traversal edge sample capacity exceeded")
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        numeric = (
            arrays["start_positions"],
            arrays["target_positions"],
            arrays["horizontal_arrival_tolerances"],
            arrays["vertical_arrival_tolerances"],
            arrays["actor_bounds"],
            arrays["direction_component_selector"],
            arrays["final_positions"],
            arrays["travelled_distances"],
        )
        if any(not np.all(np.isfinite(value)) for value in numeric):
            raise ValueError("native traversal edge arrays must be finite")
        bounds = arrays["actor_bounds"]
        if np.any(bounds[:3] >= bounds[3:]):
            raise ValueError("actor_bounds must be a positive AABB")
        climb = float(self.maximum_climb_height)
        drop = float(self.maximum_drop_height)
        if (
            not np.isfinite(climb)
            or climb < 0.0
            or not np.isfinite(drop)
            or drop < 0.0
        ):
            raise ValueError("native traversal limits must be non-negative")
        if (
            np.any(arrays["horizontal_arrival_tolerances"] < 0.0)
            or np.any(arrays["vertical_arrival_tolerances"] < 0.0)
            or np.any(arrays["travelled_distances"] < 0.0)
        ):
            raise ValueError("native traversal edge distances are invalid")
        for name, value in arrays.items():
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "maximum_climb_height", climb)
        object.__setattr__(self, "maximum_drop_height", drop)

    @property
    def sample_count(self) -> int:
        return int(self.horizontal_arrival_tolerances.shape[0])

def native_traversal_probe_request(
    positions: np.ndarray,
    upward_limits: np.ndarray,
) -> dict[str, object]:
    points = np.asarray(positions, dtype=np.float64)
    limits = np.asarray(upward_limits, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("positions must have shape [N, 3]")
    if limits.shape != (points.shape[0],):
        raise ValueError("upward_limits must have shape [N]")
    if not 1 <= points.shape[0] <= NATIVE_TRAVERSAL_PROBE_MAX_SAMPLES:
        raise ValueError("native traversal sample capacity exceeded")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(limits)):
        raise ValueError("native traversal request must be finite")
    if np.any(limits < 0.0):
        raise ValueError("upward_limits must be non-negative")
    return {
        "type": "traversal_probe",
        "positions_f64_le_xyz": np.ascontiguousarray(points, dtype="<f8").tobytes(),
        "upward_limits_f64_le": np.ascontiguousarray(limits, dtype="<f8").tobytes(),
    }


def native_traversal_edge_probe_request(
    start_positions: np.ndarray,
    target_positions: np.ndarray,
    horizontal_arrival_tolerances: np.ndarray,
    vertical_arrival_tolerances: np.ndarray,
) -> dict[str, object]:
    starts = np.asarray(start_positions, dtype=np.float64)
    targets = np.asarray(target_positions, dtype=np.float64)
    horizontal = np.asarray(
        horizontal_arrival_tolerances,
        dtype=np.float64,
    )
    vertical = np.asarray(vertical_arrival_tolerances, dtype=np.float64)
    if starts.ndim != 2 or starts.shape[1:] != (3,):
        raise ValueError("start_positions must have shape [N, 3]")
    if targets.shape != starts.shape:
        raise ValueError("target_positions must match start_positions")
    if horizontal.shape != (starts.shape[0],):
        raise ValueError("horizontal tolerances must have shape [N]")
    if vertical.shape != horizontal.shape:
        raise ValueError("vertical tolerances must match horizontal tolerances")
    if not 1 <= starts.shape[0] <= NATIVE_TRAVERSAL_EDGE_PROBE_MAX_SAMPLES:
        raise ValueError("native traversal edge sample capacity exceeded")
    if (
        not np.all(np.isfinite(starts))
        or not np.all(np.isfinite(targets))
        or not np.all(np.isfinite(horizontal))
        or not np.all(np.isfinite(vertical))
    ):
        raise ValueError("native traversal edge request must be finite")
    if np.any(horizontal < 0.0) or np.any(vertical < 0.0):
        raise ValueError("arrival_tolerances must be non-negative")
    return {
        "type": "traversal_edge_probe",
        "start_positions_f64_le_xyz": np.ascontiguousarray(
            starts,
            dtype="<f8",
        ).tobytes(),
        "target_positions_f64_le_xyz": np.ascontiguousarray(
            targets,
            dtype="<f8",
        ).tobytes(),
        "horizontal_arrival_tolerances_f64_le": np.ascontiguousarray(
            horizontal,
            dtype="<f8",
        ).tobytes(),
        "vertical_arrival_tolerances_f64_le": np.ascontiguousarray(
            vertical,
            dtype="<f8",
        ).tobytes(),
    }


def capture_native_traversal_probe(
    transport: NativeTraversalProbeTransport,
    positions: np.ndarray,
    upward_limits: np.ndarray,
) -> NativeTraversalProbeCapture:
    return parse_native_traversal_probe(
        transport.capture_traversal_probe(positions, upward_limits)
    )


def capture_native_traversal_edge_probe(
    transport: NativeTraversalProbeTransport,
    start_positions: np.ndarray,
    target_positions: np.ndarray,
    horizontal_arrival_tolerances: np.ndarray,
    vertical_arrival_tolerances: np.ndarray,
) -> NativeTraversalEdgeProbeCapture:
    return parse_native_traversal_edge_probe(
        transport.capture_traversal_edge_probe(
            start_positions,
            target_positions,
            horizontal_arrival_tolerances,
            vertical_arrival_tolerances,
        )
    )


def parse_native_traversal_probe(
    response: Mapping[str, Any],
) -> NativeTraversalProbeCapture:
    source = dict(response)
    if source.get("type") != "traversal_probe":
        raise ValueError("native response is not a traversal probe")
    if source.get("schema") != NATIVE_TRAVERSAL_PROBE_SCHEMA:
        raise ValueError("unsupported native traversal schema")
    if source.get("version") != NATIVE_TRAVERSAL_PROBE_VERSION:
        raise ValueError("unsupported native traversal version")
    count = source.get("sample_count")
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("sample_count must be an integer")
    return NativeTraversalProbeCapture(
        server_version=_text(source, "server_version"),
        world_name=_text(source, "world"),
        worldgen_provider=_text(source, "worldgen_provider"),
        worldgen_version=_text(source, "worldgen_version"),
        seed=_integer(source, "seed"),
        positions=_decode(source, "positions_f64_le_xyz", np.dtype("<f8")).reshape(
            count,
            3,
        ),
        upward_limits=_decode(
            source,
            "upward_limits_f64_le",
            np.dtype("<f8"),
        ),
        actor_bounds=_decode(
            source,
            "actor_bounds_f64_le",
            np.dtype("<f8"),
        ),
        validation_codes=_decode(
            source,
            "validation_codes_i8",
            np.dtype("i1"),
        ),
        upward_collision_distances=_decode(
            source,
            "upward_collision_distances_f64_le",
            np.dtype("<f8"),
        ),
    )


def parse_native_traversal_edge_probe(
    response: Mapping[str, Any],
) -> NativeTraversalEdgeProbeCapture:
    source = dict(response)
    if source.get("type") != "traversal_edge_probe":
        raise ValueError("native response is not a traversal edge probe")
    if source.get("schema") != NATIVE_TRAVERSAL_EDGE_PROBE_SCHEMA:
        raise ValueError("unsupported native traversal edge schema")
    if source.get("version") != NATIVE_TRAVERSAL_EDGE_PROBE_VERSION:
        raise ValueError("unsupported native traversal edge version")
    count = source.get("sample_count")
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("sample_count must be an integer")
    return NativeTraversalEdgeProbeCapture(
        server_version=_text(source, "server_version"),
        world_name=_text(source, "world"),
        worldgen_provider=_text(source, "worldgen_provider"),
        worldgen_version=_text(source, "worldgen_version"),
        seed=_integer(source, "seed"),
        start_positions=_decode(
            source,
            "start_positions_f64_le_xyz",
            np.dtype("<f8"),
        ).reshape(count, 3),
        target_positions=_decode(
            source,
            "target_positions_f64_le_xyz",
            np.dtype("<f8"),
        ).reshape(count, 3),
        horizontal_arrival_tolerances=_decode(
            source,
            "horizontal_arrival_tolerances_f64_le",
            np.dtype("<f8"),
        ),
        vertical_arrival_tolerances=_decode(
            source,
            "vertical_arrival_tolerances_f64_le",
            np.dtype("<f8"),
        ),
        actor_bounds=_decode(
            source,
            "actor_bounds_f64_le",
            np.dtype("<f8"),
        ),
        direction_component_selector=_decode(
            source,
            "direction_component_selector_f64_le_xyz",
            np.dtype("<f8"),
        ),
        maximum_climb_height=_number(source, "maximum_climb_height"),
        maximum_drop_height=_number(source, "maximum_drop_height"),
        reachable=_decode(source, "reachable_u8", np.dtype("u1")).astype(
            np.bool_
        ),
        edge_blocked=_decode(
            source,
            "edge_blocked_u8",
            np.dtype("u1"),
        ).astype(np.bool_),
        final_positions=_decode(
            source,
            "final_positions_f64_le_xyz",
            np.dtype("<f8"),
        ).reshape(count, 3),
        travelled_distances=_decode(
            source,
            "travelled_distances_f64_le",
            np.dtype("<f8"),
        ),
    )


def native_traversal_probe_contract() -> dict[str, object]:
    return {
        "schema": NATIVE_TRAVERSAL_PROBE_SCHEMA,
        "version": NATIVE_TRAVERSAL_PROBE_VERSION,
        "message_type": "traversal_probe",
        "sample_capacity": NATIVE_TRAVERSAL_PROBE_MAX_SAMPLES,
        "positions": "actor_transform_f64_le_xyz",
        "actor_bounds": "active_native_npc_collision_aabb",
        "standability": (
            "CollisionModule.validatePosition_valid_and_on_ground_bit"
        ),
        "collision_box_origin": (
            "queried_cell_minus_signed_BlockSection_filler_root_offset"
        ),
        "clearance": (
            "actor_height_plus_upward_CollisionModule.findCollisions_distance"
        ),
        "unavailable_or_overlap": "fail_closed",
        "capture_thread": "native_world_thread",
        "required_fixture": "static_region_with_selected_manifest",
    }


def native_traversal_edge_probe_contract() -> dict[str, object]:
    return {
        "schema": NATIVE_TRAVERSAL_EDGE_PROBE_SCHEMA,
        "version": NATIVE_TRAVERSAL_EDGE_PROBE_VERSION,
        "message_type": "traversal_edge_probe",
        "sample_capacity": NATIVE_TRAVERSAL_EDGE_PROBE_MAX_SAMPLES,
        "positions": "actor_transform_f64_le_xyz",
        "actor_bounds": "active_native_npc_collision_aabb",
        "controller": "active_MotionControllerWalk",
        "reachability": (
            "ProbeMoveData.canMoveTo_horizontal_and_vertical_equivalent_"
            "after_one_probeMove_call"
        ),
        "limits": [
            "MotionControllerWalk.getMaxClimbHeight",
            "MotionControllerWalk.getMaxDropHeight",
        ],
        "diagnostics": [
            "ProbeMoveData.edgeBlocked",
            "final_actor_transform_position",
            "travelled_projected_distance",
        ],
        "capture_thread": "native_world_thread",
        "required_fixture": "static_region_with_selected_manifest",
        "mutation": "probe_only_does_not_move_actor_transform",
    }


def region_traversal_projection_contract() -> dict[str, object]:
    """Describe the native-certified Region traversal projection."""

    return {
        "schema": REGION_TRAVERSAL_PROJECTION_SCHEMA,
        "version": REGION_TRAVERSAL_PROJECTION_VERSION,
        "status": "native_exact_stationary_surface_certified",
        "affected_inputs": {
            "snapshot": "hytalerl_native_region_snapshot_v1",
            "supported_section_protocols": [
                "hytalerl_native_region_section_v3",
            ],
            "rejected_section_protocols": [
                "hytalerl_native_region_section_v1",
                "hytalerl_native_region_section_v2",
            ],
        },
        "captured": [
            "rotated_collision_detail_boxes",
            "cell_flags",
            "support",
            "signed_per_cell_BlockSection_filler_root_offset_xyz",
        ],
        "native_placement": (
            "visited_cell_xyz_minus_"
            "FillerBlockUtil.unpackXYZ(getFiller(visited_cell_xyz))"
        ),
        "native_broad_phase": (
            "only_cells_visited_by_actor_AABB_or_upward_sweep;"
            "do_not_scan_neighbouring_protruding_roots"
        ),
        "availability": True,
        "legacy_failure_semantics": "clear_complete_traversal_source_row",
        "certification_fixture": (
            "native_traversal_projection_v2_recording.json"
        ),
        "certified_scope": (
            "stationary_half_cell_top_face_standability_and_"
            "coverage_bounded_vertical_clearance"
        ),
        "excluded": (
            "edge_connectivity_moving_step_drop_controller_fluids_"
            "stateful_blocks_entities_and_region_scale_graph_connectivity"
        ),
    }


def region_traversal_projection_contract_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            region_traversal_projection_contract(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def native_traversal_probe_contract_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            native_traversal_probe_contract(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def native_traversal_edge_probe_contract_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            native_traversal_edge_probe_contract(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _array(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    return np.asarray(value, dtype=dtype).copy()


def _decode(
    source: Mapping[str, Any],
    name: str,
    dtype: np.dtype[Any],
) -> np.ndarray:
    value = source.get(name)
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be binary")
    return np.frombuffer(value, dtype=dtype).copy()


def _text(source: Mapping[str, Any], name: str) -> str:
    value = source.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(source: Mapping[str, Any], name: str) -> int:
    value = source.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _number(source: Mapping[str, Any], name: str) -> float:
    value = source.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


__all__ = [
    "NATIVE_TRAVERSAL_EDGE_PROBE_MAX_SAMPLES",
    "NATIVE_TRAVERSAL_EDGE_PROBE_SCHEMA",
    "NATIVE_TRAVERSAL_EDGE_PROBE_VERSION",
    "NATIVE_TRAVERSAL_PROBE_MAX_SAMPLES",
    "NATIVE_TRAVERSAL_PROBE_SCHEMA",
    "NATIVE_TRAVERSAL_PROBE_VERSION",
    "REGION_TRAVERSAL_PROJECTION_SCHEMA",
    "REGION_TRAVERSAL_PROJECTION_VERSION",
    "NativeTraversalProbeCapture",
    "NativeTraversalProbeTransport",
    "NativeTraversalEdgeProbeCapture",
    "VALIDATE_CLEAR",
    "VALIDATE_INVALID",
    "VALIDATE_ON_GROUND",
    "VALIDATE_TOUCH_CEILING",
    "capture_native_traversal_probe",
    "capture_native_traversal_edge_probe",
    "native_traversal_edge_probe_contract",
    "native_traversal_edge_probe_contract_sha256",
    "native_traversal_edge_probe_request",
    "native_traversal_probe_contract",
    "native_traversal_probe_contract_sha256",
    "native_traversal_probe_request",
    "parse_native_traversal_probe",
    "parse_native_traversal_edge_probe",
    "region_traversal_projection_contract",
    "region_traversal_projection_contract_sha256",
]
