"""Transport contract for exact native ``AStarBase`` successor probes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np

NATIVE_NAVIGATION_SUCCESSOR_SCHEMA = (
    "hytalerl_native_navigation_successor_probe_v1"
)
NATIVE_NAVIGATION_SUCCESSOR_VERSION = 1
NATIVE_NAVIGATION_SUCCESSOR_MAX_SAMPLES = 4096


@dataclass(frozen=True)
class NativeNavigationSuccessorCapture:
    """Raw native half/full-step decisions before A* stateful guards."""

    server_version: str
    world_name: str
    worldgen_provider: str
    worldgen_version: str
    seed: int
    start_positions: np.ndarray
    direction_indices: np.ndarray
    directions: np.ndarray
    actor_bounds: np.ndarray
    direction_component_selector: np.ndarray
    direction_distances: np.ndarray
    travelled_distances: np.ndarray
    reached_half_step: np.ndarray
    reached_full_step: np.ndarray
    valid_positions: np.ndarray
    edge_blocked: np.ndarray
    half_step_positions: np.ndarray
    successor_positions: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "server_version",
            "world_name",
            "worldgen_provider",
            "worldgen_version",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(
                self, name
            ):
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        arrays = {
            "start_positions": _array(self.start_positions, np.float64),
            "direction_indices": _array(self.direction_indices, np.int8),
            "directions": _array(self.directions, np.float64),
            "actor_bounds": _array(self.actor_bounds, np.float64),
            "direction_component_selector": _array(
                self.direction_component_selector,
                np.float64,
            ),
            "direction_distances": _array(
                self.direction_distances,
                np.float64,
            ),
            "travelled_distances": _array(
                self.travelled_distances,
                np.float64,
            ),
            "reached_half_step": _array(
                self.reached_half_step,
                np.bool_,
            ),
            "reached_full_step": _array(
                self.reached_full_step,
                np.bool_,
            ),
            "valid_positions": _array(self.valid_positions, np.bool_),
            "edge_blocked": _array(self.edge_blocked, np.bool_),
            "half_step_positions": _array(
                self.half_step_positions,
                np.float64,
            ),
            "successor_positions": _array(
                self.successor_positions,
                np.float64,
            ),
        }
        count = arrays["direction_indices"].shape[0]
        if not 1 <= count <= NATIVE_NAVIGATION_SUCCESSOR_MAX_SAMPLES:
            raise ValueError("native navigation successor capacity exceeded")
        expected = {
            "start_positions": (count, 3),
            "direction_indices": (count,),
            "directions": (count, 3),
            "actor_bounds": (6,),
            "direction_component_selector": (3,),
            "direction_distances": (count,),
            "travelled_distances": (count,),
            "reached_half_step": (count,),
            "reached_full_step": (count,),
            "valid_positions": (count,),
            "edge_blocked": (count,),
            "half_step_positions": (count, 3),
            "successor_positions": (count, 3),
        }
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        numeric = (
            arrays["start_positions"],
            arrays["directions"],
            arrays["actor_bounds"],
            arrays["direction_component_selector"],
            arrays["direction_distances"],
            arrays["travelled_distances"],
            arrays["half_step_positions"],
            arrays["successor_positions"],
        )
        if any(not np.all(np.isfinite(value)) for value in numeric):
            raise ValueError("native navigation successor values must be finite")
        if np.any(
            (arrays["direction_indices"] < 0)
            | (arrays["direction_indices"] >= 26)
        ):
            raise ValueError("native navigation direction index is invalid")
        bounds = arrays["actor_bounds"]
        if np.any(bounds[:3] >= bounds[3:]):
            raise ValueError("actor_bounds must be a positive AABB")
        selector = arrays["direction_component_selector"]
        if np.any(selector < 0.0) or not np.any(selector > 0.0):
            raise ValueError("direction_component_selector is invalid")
        distances = arrays["direction_distances"]
        if np.any(distances <= 0.0) or np.any(
            arrays["travelled_distances"] < 0.0
        ):
            raise ValueError("native navigation distances are invalid")
        np.testing.assert_allclose(
            distances,
            np.linalg.norm(arrays["directions"], axis=1),
            rtol=0.0,
            atol=0.0,
        )
        half = arrays["reached_half_step"]
        full = arrays["reached_full_step"]
        valid = arrays["valid_positions"]
        if np.any(full & ~half) or np.any(valid & ~half):
            raise ValueError("native navigation successor flags disagree")
        np.testing.assert_array_equal(
            half,
            arrays["travelled_distances"] >= distances * 0.49999995,
        )
        np.testing.assert_array_equal(
            full,
            arrays["travelled_distances"] >= distances * 0.9999999,
        )
        np.testing.assert_array_equal(
            arrays["half_step_positions"][~half],
            arrays["start_positions"][~half],
        )
        np.testing.assert_array_equal(
            arrays["successor_positions"][~half],
            arrays["start_positions"][~half],
        )
        for name, value in arrays.items():
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    @property
    def sample_count(self) -> int:
        return int(self.direction_indices.shape[0])


def native_navigation_successor_request(
    start_positions: np.ndarray,
    direction_indices: np.ndarray,
) -> dict[str, object]:
    starts = np.asarray(start_positions, dtype=np.float64)
    directions = np.asarray(direction_indices, dtype=np.int8)
    if starts.ndim != 2 or starts.shape[1:] != (3,):
        raise ValueError("start_positions must have shape [N,3]")
    if directions.shape != (starts.shape[0],):
        raise ValueError("direction_indices must have shape [N]")
    if not 1 <= starts.shape[0] <= NATIVE_NAVIGATION_SUCCESSOR_MAX_SAMPLES:
        raise ValueError("native navigation successor capacity exceeded")
    if not np.all(np.isfinite(starts)):
        raise ValueError("native navigation starts must be finite")
    if np.any((directions < 0) | (directions >= 26)):
        raise ValueError("direction_indices must be in [0,26)")
    return {
        "type": "navigation_successor_probe",
        "start_positions_f64_le_xyz": np.ascontiguousarray(
            starts,
            dtype="<f8",
        ).tobytes(),
        "direction_indices_i8": np.ascontiguousarray(
            directions,
            dtype=np.int8,
        ).tobytes(),
    }


def parse_native_navigation_successor(
    response: Mapping[str, Any],
) -> NativeNavigationSuccessorCapture:
    source = dict(response)
    if source.get("type") != "navigation_successor_probe":
        raise ValueError("native response is not a navigation successor probe")
    if source.get("schema") != NATIVE_NAVIGATION_SUCCESSOR_SCHEMA:
        raise ValueError("unsupported native navigation successor schema")
    if source.get("version") != NATIVE_NAVIGATION_SUCCESSOR_VERSION:
        raise ValueError("unsupported native navigation successor version")
    count = _integer(source, "sample_count")
    if not 1 <= count <= NATIVE_NAVIGATION_SUCCESSOR_MAX_SAMPLES:
        raise ValueError("native navigation successor capacity exceeded")
    return NativeNavigationSuccessorCapture(
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
        direction_indices=_decode(
            source,
            "direction_indices_i8",
            np.dtype("i1"),
        ),
        directions=_decode(
            source,
            "directions_f64_le_xyz",
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
        direction_distances=_decode(
            source,
            "direction_distances_f64_le",
            np.dtype("<f8"),
        ),
        travelled_distances=_decode(
            source,
            "travelled_distances_f64_le",
            np.dtype("<f8"),
        ),
        reached_half_step=_flag(source, "reached_half_step_u8"),
        reached_full_step=_flag(source, "reached_full_step_u8"),
        valid_positions=_flag(source, "valid_positions_u8"),
        edge_blocked=_flag(source, "edge_blocked_u8"),
        half_step_positions=_decode(
            source,
            "half_step_positions_f64_le_xyz",
            np.dtype("<f8"),
        ).reshape(count, 3),
        successor_positions=_decode(
            source,
            "successor_positions_f64_le_xyz",
            np.dtype("<f8"),
        ).reshape(count, 3),
    )


def native_navigation_successor_contract() -> dict[str, object]:
    return {
        "schema": NATIVE_NAVIGATION_SUCCESSOR_SCHEMA,
        "version": NATIVE_NAVIGATION_SUCCESSOR_VERSION,
        "message_type": "navigation_successor_probe",
        "sample_capacity": NATIVE_NAVIGATION_SUCCESSOR_MAX_SAMPLES,
        "source": "AStarBase.computePath_successor_branch",
        "controller": "active_MotionControllerWalk",
        "direction_order": (
            "native_AStarBase_xyz_nested_order_for_active_component_selector"
        ),
        "thresholds": {
            "half_step": 0.49999995,
            "full_step": 0.9999999,
        },
        "result": (
            "raw_probe_distance_segment_interpolated_half_key_position_"
            "selected_successor_and_isValidPosition"
        ),
        "excluded_stateful_steps": [
            "visited_lookup",
            "diagonal_visited_neighbour_guard",
            "recursive_successor_cost_propagation",
        ],
        "capture_thread": "native_world_thread",
        "required_fixture": "static_region_with_selected_manifest",
        "mutation": "probe_only_does_not_move_actor_transform",
        "failure": "oversize_or_unavailable_rejected_without_truncation",
        "provenance": "native",
    }


def native_navigation_successor_contract_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            native_navigation_successor_contract(),
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


def _flag(source: Mapping[str, Any], name: str) -> np.ndarray:
    value = _decode(source, name, np.dtype("u1"))
    if np.any(value > 1):
        raise ValueError(f"{name} must contain binary flags")
    return value.astype(np.bool_)


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
    "NATIVE_NAVIGATION_SUCCESSOR_MAX_SAMPLES",
    "NATIVE_NAVIGATION_SUCCESSOR_SCHEMA",
    "NATIVE_NAVIGATION_SUCCESSOR_VERSION",
    "NativeNavigationSuccessorCapture",
    "native_navigation_successor_contract",
    "native_navigation_successor_contract_sha256",
    "native_navigation_successor_request",
    "parse_native_navigation_successor",
]
