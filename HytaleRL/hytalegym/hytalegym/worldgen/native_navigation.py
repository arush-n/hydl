"""Bounded transport contract for native Hytale NPC A* paths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Protocol

import numpy as np

NATIVE_NAVIGATION_PATH_PROBE_SCHEMA = (
    "hytalerl_native_navigation_path_probe_v1"
)
NATIVE_NAVIGATION_PATH_PROBE_VERSION = 1
NATIVE_NAVIGATION_PATH_PROBE_MAX_SAMPLES = 32
NATIVE_NAVIGATION_MAXIMUM_PATH_LENGTH = 200
NATIVE_NAVIGATION_MAX_OPEN_NODES = 200
NATIVE_NAVIGATION_MAX_TOTAL_NODES = 900
NATIVE_NAVIGATION_MAX_NODES_PER_ITERATION = 900

NATIVE_ASTAR_PROGRESS_UNSTARTED = 0
NATIVE_ASTAR_PROGRESS_ABORTED = 1
NATIVE_ASTAR_PROGRESS_COMPUTING = 2
NATIVE_ASTAR_PROGRESS_ACCOMPLISHED = 3
NATIVE_ASTAR_PROGRESS_TERMINATED = 4
NATIVE_ASTAR_PROGRESS_OPEN_LIMIT = 5
NATIVE_ASTAR_PROGRESS_TOTAL_LIMIT = 6


class NativeNavigationPathProbeTransport(Protocol):
    """One bounded path verb over a selected static native Region."""

    def capture_navigation_path_probe(
        self,
        start_positions: np.ndarray,
        target_positions: np.ndarray,
        *,
        maximum_path_length: int,
        open_nodes_limit: int,
        total_nodes_limit: int,
        nodes_per_iteration: int,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class NativeNavigationPathProbeCapture:
    """Full native predecessor chains and terminal search diagnostics."""

    server_version: str
    world_name: str
    worldgen_provider: str
    worldgen_version: str
    seed: int
    start_positions: np.ndarray
    target_positions: np.ndarray
    actor_bounds: np.ndarray
    direction_component_selector: np.ndarray
    maximum_path_length: int
    open_nodes_limit: int
    total_nodes_limit: int
    nodes_per_iteration: int
    progress: np.ndarray
    iterations: np.ndarray
    visited_counts: np.ndarray
    open_counts: np.ndarray
    path_node_counts: np.ndarray
    path_positions: np.ndarray
    path_travel_costs: np.ndarray

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
        maximum_path_length = _bounded_int(
            self.maximum_path_length,
            1,
            NATIVE_NAVIGATION_MAXIMUM_PATH_LENGTH,
            "maximum_path_length",
        )
        open_nodes_limit = _bounded_int(
            self.open_nodes_limit,
            1,
            NATIVE_NAVIGATION_MAX_OPEN_NODES,
            "open_nodes_limit",
        )
        total_nodes_limit = _bounded_int(
            self.total_nodes_limit,
            1,
            NATIVE_NAVIGATION_MAX_TOTAL_NODES,
            "total_nodes_limit",
        )
        nodes_per_iteration = _bounded_int(
            self.nodes_per_iteration,
            1,
            NATIVE_NAVIGATION_MAX_NODES_PER_ITERATION,
            "nodes_per_iteration",
        )
        arrays = {
            "start_positions": _array(self.start_positions, np.float64),
            "target_positions": _array(self.target_positions, np.float64),
            "actor_bounds": _array(self.actor_bounds, np.float64),
            "direction_component_selector": _array(
                self.direction_component_selector,
                np.float64,
            ),
            "progress": _array(self.progress, np.int8),
            "iterations": _array(self.iterations, np.int32),
            "visited_counts": _array(self.visited_counts, np.int32),
            "open_counts": _array(self.open_counts, np.int32),
            "path_node_counts": _array(
                self.path_node_counts,
                np.int32,
            ),
            "path_positions": _array(self.path_positions, np.float64),
            "path_travel_costs": _array(
                self.path_travel_costs,
                np.float32,
            ),
        }
        samples = arrays["progress"].shape[0]
        if not 1 <= samples <= NATIVE_NAVIGATION_PATH_PROBE_MAX_SAMPLES:
            raise ValueError("native navigation sample capacity exceeded")
        expected = {
            "start_positions": (samples, 3),
            "target_positions": (samples, 3),
            "actor_bounds": (6,),
            "direction_component_selector": (3,),
            "progress": (samples,),
            "iterations": (samples,),
            "visited_counts": (samples,),
            "open_counts": (samples,),
            "path_node_counts": (samples,),
            "path_positions": (
                samples,
                maximum_path_length,
                3,
            ),
            "path_travel_costs": (
                samples,
                maximum_path_length,
            ),
        }
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if any(not np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("native navigation arrays must be finite")
        bounds = arrays["actor_bounds"]
        if np.any(bounds[:3] >= bounds[3:]):
            raise ValueError("actor_bounds must be a positive AABB")
        selector = arrays["direction_component_selector"]
        if np.any(selector < 0.0) or not np.any(selector > 0.0):
            raise ValueError("direction_component_selector is invalid")
        progress = arrays["progress"]
        if np.any(
            (progress < NATIVE_ASTAR_PROGRESS_UNSTARTED)
            | (progress > NATIVE_ASTAR_PROGRESS_TOTAL_LIMIT)
        ):
            raise ValueError("native A* progress is unknown")
        counts = arrays["path_node_counts"]
        if np.any((counts < 0) | (counts > maximum_path_length)):
            raise ValueError("native path node count exceeds capacity")
        if np.any(
            (progress == NATIVE_ASTAR_PROGRESS_ACCOMPLISHED) != (counts > 0)
        ):
            raise ValueError("only accomplished searches may publish paths")
        if np.any(arrays["iterations"] < 0):
            raise ValueError("native navigation iterations cannot be negative")
        if np.any(
            (arrays["visited_counts"] < 0)
            | (
                arrays["visited_counts"]
                > NATIVE_NAVIGATION_MAX_TOTAL_NODES + 8
            )
            | (arrays["open_counts"] < 0)
            | (
                arrays["open_counts"]
                > NATIVE_NAVIGATION_MAX_OPEN_NODES + 8
            )
        ):
            raise ValueError("native navigation counts exceed bounds")
        mask = (
            np.arange(maximum_path_length, dtype=np.int32)[None, :]
            < counts[:, None]
        )
        if np.any(arrays["path_travel_costs"][mask] < 0.0):
            raise ValueError("native path travel cost cannot be negative")
        if np.any(arrays["path_positions"][~mask] != 0.0) or np.any(
            arrays["path_travel_costs"][~mask] != 0.0
        ):
            raise ValueError("unused native path capacity must be zero")
        for name, value in arrays.items():
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "maximum_path_length",
            maximum_path_length,
        )
        object.__setattr__(self, "open_nodes_limit", open_nodes_limit)
        object.__setattr__(self, "total_nodes_limit", total_nodes_limit)
        object.__setattr__(
            self,
            "nodes_per_iteration",
            nodes_per_iteration,
        )

    @property
    def sample_count(self) -> int:
        return int(self.progress.shape[0])

    @property
    def path_mask(self) -> np.ndarray:
        result = (
            np.arange(self.maximum_path_length, dtype=np.int32)[None, :]
            < self.path_node_counts[:, None]
        )
        result.flags.writeable = False
        return result


def native_navigation_path_probe_request(
    start_positions: np.ndarray,
    target_positions: np.ndarray,
    *,
    maximum_path_length: int = NATIVE_NAVIGATION_MAXIMUM_PATH_LENGTH,
    open_nodes_limit: int = NATIVE_NAVIGATION_MAX_OPEN_NODES,
    total_nodes_limit: int = NATIVE_NAVIGATION_MAX_TOTAL_NODES,
    nodes_per_iteration: int = 50,
) -> dict[str, object]:
    starts = np.asarray(start_positions, dtype=np.float64)
    targets = np.asarray(target_positions, dtype=np.float64)
    if starts.ndim != 2 or starts.shape[1:] != (3,):
        raise ValueError("start_positions must have shape [N, 3]")
    if targets.shape != starts.shape:
        raise ValueError("target_positions must match start_positions")
    if not 1 <= starts.shape[0] <= NATIVE_NAVIGATION_PATH_PROBE_MAX_SAMPLES:
        raise ValueError("native navigation sample capacity exceeded")
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(targets)):
        raise ValueError("native navigation positions must be finite")
    return {
        "type": "navigation_path_probe",
        "start_positions_f64_le_xyz": np.ascontiguousarray(
            starts,
            dtype="<f8",
        ).tobytes(),
        "target_positions_f64_le_xyz": np.ascontiguousarray(
            targets,
            dtype="<f8",
        ).tobytes(),
        "maximum_path_length": _bounded_int(
            maximum_path_length,
            1,
            NATIVE_NAVIGATION_MAXIMUM_PATH_LENGTH,
            "maximum_path_length",
        ),
        "open_nodes_limit": _bounded_int(
            open_nodes_limit,
            1,
            NATIVE_NAVIGATION_MAX_OPEN_NODES,
            "open_nodes_limit",
        ),
        "total_nodes_limit": _bounded_int(
            total_nodes_limit,
            1,
            NATIVE_NAVIGATION_MAX_TOTAL_NODES,
            "total_nodes_limit",
        ),
        "nodes_per_iteration": _bounded_int(
            nodes_per_iteration,
            1,
            NATIVE_NAVIGATION_MAX_NODES_PER_ITERATION,
            "nodes_per_iteration",
        ),
    }


def capture_native_navigation_path_probe(
    transport: NativeNavigationPathProbeTransport,
    start_positions: np.ndarray,
    target_positions: np.ndarray,
    *,
    maximum_path_length: int = NATIVE_NAVIGATION_MAXIMUM_PATH_LENGTH,
    open_nodes_limit: int = NATIVE_NAVIGATION_MAX_OPEN_NODES,
    total_nodes_limit: int = NATIVE_NAVIGATION_MAX_TOTAL_NODES,
    nodes_per_iteration: int = 50,
) -> NativeNavigationPathProbeCapture:
    return parse_native_navigation_path_probe(
        transport.capture_navigation_path_probe(
            start_positions,
            target_positions,
            maximum_path_length=maximum_path_length,
            open_nodes_limit=open_nodes_limit,
            total_nodes_limit=total_nodes_limit,
            nodes_per_iteration=nodes_per_iteration,
        )
    )


def parse_native_navigation_path_probe(
    response: Mapping[str, Any],
) -> NativeNavigationPathProbeCapture:
    source = dict(response)
    if source.get("type") != "navigation_path_probe":
        raise ValueError("native response is not a navigation path probe")
    if source.get("schema") != NATIVE_NAVIGATION_PATH_PROBE_SCHEMA:
        raise ValueError("unsupported native navigation path schema")
    if source.get("version") != NATIVE_NAVIGATION_PATH_PROBE_VERSION:
        raise ValueError("unsupported native navigation path version")
    count = _bounded_int(
        source.get("sample_count"),
        1,
        NATIVE_NAVIGATION_PATH_PROBE_MAX_SAMPLES,
        "sample_count",
    )
    path_capacity = _bounded_int(
        source.get("maximum_path_length"),
        1,
        NATIVE_NAVIGATION_MAXIMUM_PATH_LENGTH,
        "maximum_path_length",
    )
    return NativeNavigationPathProbeCapture(
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
        maximum_path_length=path_capacity,
        open_nodes_limit=_integer(source, "open_nodes_limit"),
        total_nodes_limit=_integer(source, "total_nodes_limit"),
        nodes_per_iteration=_integer(source, "nodes_per_iteration"),
        progress=_decode(source, "progress_i8", np.dtype("i1")),
        iterations=_decode(
            source,
            "iterations_i32_le",
            np.dtype("<i4"),
        ),
        visited_counts=_decode(
            source,
            "visited_counts_i32_le",
            np.dtype("<i4"),
        ),
        open_counts=_decode(
            source,
            "open_counts_i32_le",
            np.dtype("<i4"),
        ),
        path_node_counts=_decode(
            source,
            "path_node_counts_i32_le",
            np.dtype("<i4"),
        ),
        path_positions=_decode(
            source,
            "path_positions_f64_le_xyz",
            np.dtype("<f8"),
        ).reshape(count, path_capacity, 3),
        path_travel_costs=_decode(
            source,
            "path_travel_costs_f32_le",
            np.dtype("<f4"),
        ).reshape(count, path_capacity),
    )


def native_navigation_path_probe_contract() -> dict[str, object]:
    return {
        "schema": NATIVE_NAVIGATION_PATH_PROBE_SCHEMA,
        "version": NATIVE_NAVIGATION_PATH_PROBE_VERSION,
        "message_type": "navigation_path_probe",
        "sample_capacity": NATIVE_NAVIGATION_PATH_PROBE_MAX_SAMPLES,
        "controller": "active_MotionControllerWalk",
        "pathfinder": "native_AStarWithTarget",
        "configuration": {
            "maximum_path_length_max": (
                NATIVE_NAVIGATION_MAXIMUM_PATH_LENGTH
            ),
            "open_nodes_limit_max": NATIVE_NAVIGATION_MAX_OPEN_NODES,
            "total_nodes_limit_max": NATIVE_NAVIGATION_MAX_TOTAL_NODES,
            "nodes_per_iteration_max": (
                NATIVE_NAVIGATION_MAX_NODES_PER_ITERATION
            ),
            "diagonal_moves": True,
            "optimized_build_path": False,
        },
        "goal": "exact_native_half_cell_position_index",
        "heuristic": (
            "Vector3d_double_euclidean_distance_then_float_cast"
        ),
        "successor": "MotionControllerWalk.probeMove",
        "step_cost": "MotionControllerWalk.waypointDistance",
        "result": (
            "full_predecessor_chain_with_terminal_progress_and_counts"
        ),
        "capture_thread": "native_world_thread",
        "required_fixture": "static_region_with_selected_manifest",
        "mutation": "probe_only_does_not_move_actor_transform",
        "failure": "oversize_or_unavailable_rejected_without_truncation",
        "scope": (
            "T8_node_chain_and_T9_terminal_evidence;runtime_optimized_"
            "follower_and_replan_remain_T11"
        ),
    }


def native_navigation_path_probe_contract_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            native_navigation_path_probe_contract(),
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
    payload = source.get(name)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be binary")
    if len(payload) % dtype.itemsize:
        raise ValueError(f"{name} has invalid byte length")
    return np.frombuffer(payload, dtype=dtype).copy()


def _bounded_int(
    value: object,
    minimum: int,
    maximum: int,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _integer(source: Mapping[str, Any], name: str) -> int:
    value = source.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _text(source: Mapping[str, Any], name: str) -> str:
    value = source.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


__all__ = [
    "NATIVE_ASTAR_PROGRESS_ABORTED",
    "NATIVE_ASTAR_PROGRESS_ACCOMPLISHED",
    "NATIVE_ASTAR_PROGRESS_COMPUTING",
    "NATIVE_ASTAR_PROGRESS_OPEN_LIMIT",
    "NATIVE_ASTAR_PROGRESS_TERMINATED",
    "NATIVE_ASTAR_PROGRESS_TOTAL_LIMIT",
    "NATIVE_ASTAR_PROGRESS_UNSTARTED",
    "NATIVE_NAVIGATION_MAXIMUM_PATH_LENGTH",
    "NATIVE_NAVIGATION_MAX_NODES_PER_ITERATION",
    "NATIVE_NAVIGATION_MAX_OPEN_NODES",
    "NATIVE_NAVIGATION_MAX_TOTAL_NODES",
    "NATIVE_NAVIGATION_PATH_PROBE_MAX_SAMPLES",
    "NATIVE_NAVIGATION_PATH_PROBE_SCHEMA",
    "NATIVE_NAVIGATION_PATH_PROBE_VERSION",
    "NativeNavigationPathProbeCapture",
    "NativeNavigationPathProbeTransport",
    "capture_native_navigation_path_probe",
    "native_navigation_path_probe_contract",
    "native_navigation_path_probe_contract_sha256",
    "native_navigation_path_probe_request",
    "parse_native_navigation_path_probe",
]
