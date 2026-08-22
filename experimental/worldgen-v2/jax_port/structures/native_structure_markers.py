"""Checked host adapter for authored WorldGen V2 structure-marker rows."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

import numpy as np

from .structure_contract import (
    STRUCTURE_MARKER_COMPONENT_FILTER,
    STRUCTURE_MARKER_IDENTITY,
)
from .structure_pack import V2NativeStructureMarker


NATIVE_STRUCTURE_MARKER_SCHEMA = "hytalerl_worldgen_v2_structure_markers_v1"
NATIVE_STRUCTURE_MARKER_VERSION = 1
NATIVE_STRUCTURE_MARKER_MAX_CAPACITY = 256
_RESPONSE_FIELDS = frozenset(
    {
        "type",
        "schema",
        "version",
        "bridge_sha256",
        "server_version",
        "world",
        "worldgen_provider",
        "worldgen_version",
        "seed",
        "component_filter",
        "marker_identity",
        "bounds_semantics",
        "bounds_f64_le_min_max_xyz",
        "capacity",
        "requested_marker_asset_ids",
        "total_matching",
        "emitted_count",
        "overflow",
        "uuid_encoding",
        "uuid_bytes",
        "positions_f64_le_xyz",
        "rotation_units",
        "rotations_f64_le_yaw_pitch_roll",
        "marker_asset_ids",
        "native_worldgen_ids_i32_le",
        "native_prefab_instance_ids_i32_le",
    }
)
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")


@dataclass(frozen=True)
class V2NativeStructureMarkerCapture:
    bridge_sha256: str
    server_version: str
    world_name: str
    worldgen_provider: str
    worldgen_version: str
    seed: int
    bounds: np.ndarray
    capacity: int
    requested_marker_asset_ids: tuple[str, ...]
    total_matching: int
    overflow: bool
    uuid_bytes: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray
    marker_asset_ids: tuple[str, ...]
    native_worldgen_ids: np.ndarray
    native_prefab_instance_ids: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "bounds": np.asarray(self.bounds, dtype=np.float64),
            "uuid_bytes": np.asarray(self.uuid_bytes, dtype=np.uint8),
            "positions": np.asarray(self.positions, dtype=np.float64),
            "rotations": np.asarray(self.rotations, dtype=np.float64),
            "native_worldgen_ids": np.asarray(
                self.native_worldgen_ids, dtype=np.int32
            ),
            "native_prefab_instance_ids": np.asarray(
                self.native_prefab_instance_ids, dtype=np.int32
            ),
        }
        count = len(self.marker_asset_ids)
        expected = {
            "bounds": (6,),
            "uuid_bytes": (count, 16),
            "positions": (count, 3),
            "rotations": (count, 3),
            "native_worldgen_ids": (count,),
            "native_prefab_instance_ids": (count,),
        }
        for name, shape in expected.items():
            if arrays[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        for value in arrays.values():
            value.flags.writeable = False
        for name, value in arrays.items():
            object.__setattr__(self, name, value)

    @property
    def emitted_count(self) -> int:
        return len(self.marker_asset_ids)

    def complete_markers(self) -> tuple[V2NativeStructureMarker, ...]:
        if self.overflow:
            raise ValueError(
                "WorldGen structure marker capture overflowed; partial rows "
                "are never usable"
            )
        return tuple(
            V2NativeStructureMarker(
                marker_asset_id=self.marker_asset_ids[index],
                position=tuple(float(value) for value in self.positions[index]),
                yaw_radians=float(self.rotations[index, 0]),
                native_worldgen_id=int(self.native_worldgen_ids[index]),
                native_prefab_instance_id=int(
                    self.native_prefab_instance_ids[index]
                ),
            )
            for index in range(self.emitted_count)
        )


def worldgen_structure_marker_request(
    bounds: Sequence[float],
    marker_asset_ids: Sequence[str],
    *,
    capacity: int = NATIVE_STRUCTURE_MARKER_MAX_CAPACITY,
) -> dict[str, Any]:
    """Build one bounded allowlist request; it never enumerates all entities."""

    box = _bounds(bounds)
    markers = _marker_asset_id_list(marker_asset_ids)
    requested_capacity = _positive_int(capacity, "capacity")
    if requested_capacity > NATIVE_STRUCTURE_MARKER_MAX_CAPACITY:
        raise ValueError("WorldGen structure marker capacity exceeds 256")
    return {
        "type": "worldgen_structure_markers",
        "component_filter": STRUCTURE_MARKER_COMPONENT_FILTER,
        "bounds_f64_le_min_max_xyz": np.ascontiguousarray(
            box, dtype="<f8"
        ).tobytes(),
        "capacity": requested_capacity,
        "marker_asset_ids": list(markers),
    }


def parse_worldgen_structure_markers(
    response: Mapping[str, Any],
    *,
    expected_bridge_sha256: str | None = None,
    expected_seed: int | None = None,
    expected_marker_asset_ids: Sequence[str] | None = None,
) -> V2NativeStructureMarkerCapture:
    """Validate complete provenance, fixed columns, ordering, and grouping."""

    source = dict(response)
    if set(source) != _RESPONSE_FIELDS:
        raise ValueError("WorldGen structure marker response fields changed")
    if source["type"] != "worldgen_structure_markers":
        raise ValueError("native response is not a structure marker snapshot")
    if source["schema"] != NATIVE_STRUCTURE_MARKER_SCHEMA:
        raise ValueError("unsupported WorldGen structure marker schema")
    if _exact_int(source["version"], "version") != (
        NATIVE_STRUCTURE_MARKER_VERSION
    ):
        raise ValueError("unsupported WorldGen structure marker version")
    bridge = _sha256(source["bridge_sha256"], "bridge SHA-256")
    if expected_bridge_sha256 is not None and bridge != _sha256(
        expected_bridge_sha256, "expected bridge SHA-256"
    ):
        raise ValueError("WorldGen structure marker bridge differs")
    server_version = _nonempty(source["server_version"], "server_version")
    world_name = _nonempty(source["world"], "world")
    provider = _nonempty(source["worldgen_provider"], "worldgen_provider")
    worldgen_version = _nonempty(source["worldgen_version"], "worldgen_version")
    seed = _nonnegative_int(source["seed"], "seed")
    if expected_seed is not None and seed != _nonnegative_int(
        expected_seed, "expected seed"
    ):
        raise ValueError("WorldGen structure marker seed differs")
    if source["component_filter"] != STRUCTURE_MARKER_COMPONENT_FILTER:
        raise ValueError("WorldGen structure marker component filter changed")
    if source["marker_identity"] != STRUCTURE_MARKER_IDENTITY:
        raise ValueError("WorldGen structure marker identity changed")
    if source["bounds_semantics"] != (
        "half_open_min_inclusive_max_exclusive"
    ):
        raise ValueError("WorldGen structure marker bounds semantics changed")
    if source["uuid_encoding"] != "rfc4122_network_order_16_bytes_per_row":
        raise ValueError("WorldGen structure marker UUID encoding changed")
    if source["rotation_units"] != "radians_yaw_pitch_roll":
        raise ValueError("WorldGen structure marker rotation units changed")
    bounds = _bounds(
        _decode(
            source["bounds_f64_le_min_max_xyz"],
            "<f8",
            6,
            "bounds",
        )
    )
    capacity = _positive_int(source["capacity"], "capacity")
    if capacity > NATIVE_STRUCTURE_MARKER_MAX_CAPACITY:
        raise ValueError("WorldGen structure marker capacity exceeds 256")
    requested = _marker_asset_id_list(source["requested_marker_asset_ids"])
    if expected_marker_asset_ids is not None and requested != (
        _marker_asset_id_list(expected_marker_asset_ids)
    ):
        raise ValueError("WorldGen structure marker allowlist differs")
    total = _nonnegative_int(source["total_matching"], "total_matching")
    emitted = _nonnegative_int(source["emitted_count"], "emitted_count")
    overflow = source["overflow"]
    if not isinstance(overflow, bool) or overflow != (total > capacity):
        raise ValueError("WorldGen structure marker overflow flag changed")
    if emitted != (0 if overflow else total):
        raise ValueError("WorldGen structure marker emitted count changed")
    marker_asset_ids = _row_marker_asset_ids(
        source["marker_asset_ids"], emitted, requested
    )
    uuid_payload = _binary(source["uuid_bytes"], "uuid_bytes")
    if len(uuid_payload) != emitted * 16:
        raise ValueError("WorldGen structure marker UUID column length changed")
    uuids = np.frombuffer(uuid_payload, dtype=np.uint8).reshape(emitted, 16).copy()
    positions = _decode(
        source["positions_f64_le_xyz"], "<f8", emitted * 3, "positions"
    ).reshape(emitted, 3)
    rotations = _decode(
        source["rotations_f64_le_yaw_pitch_roll"],
        "<f8",
        emitted * 3,
        "rotations",
    ).reshape(emitted, 3)
    worldgen_ids = _decode(
        source["native_worldgen_ids_i32_le"],
        "<i4",
        emitted,
        "native_worldgen_ids",
    )
    prefab_ids = _decode(
        source["native_prefab_instance_ids_i32_le"],
        "<i4",
        emitted,
        "native_prefab_instance_ids",
    )
    if np.any(worldgen_ids < 0) or np.any(prefab_ids < 0):
        raise ValueError("WorldGen structure marker native IDs must be non-negative")
    if emitted and (
        not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(rotations))
    ):
        raise ValueError("WorldGen structure marker pose must be finite")
    if emitted and np.any(np.abs(rotations[:, 1:]) > 1.0e-4):
        raise ValueError("WorldGen structure markers cannot use pitch or roll")
    if emitted and (
        np.any(positions < bounds[:3]) or np.any(positions >= bounds[3:])
    ):
        raise ValueError("WorldGen structure marker lies outside query bounds")
    uuid_rows = [bytes(row) for row in uuids]
    if uuid_rows != sorted(set(uuid_rows)):
        raise ValueError("WorldGen structure marker UUIDs must be unique and sorted")
    groups = list(
        zip(worldgen_ids.tolist(), prefab_ids.tolist(), strict=True)
    )
    if len(set(groups)) != len(groups):
        raise ValueError(
            "exactly one authored marker is required per native prefab instance"
        )
    return V2NativeStructureMarkerCapture(
        bridge_sha256=bridge,
        server_version=server_version,
        world_name=world_name,
        worldgen_provider=provider,
        worldgen_version=worldgen_version,
        seed=seed,
        bounds=bounds,
        capacity=capacity,
        requested_marker_asset_ids=requested,
        total_matching=total,
        overflow=overflow,
        uuid_bytes=uuids,
        positions=positions,
        rotations=rotations,
        marker_asset_ids=marker_asset_ids,
        native_worldgen_ids=worldgen_ids,
        native_prefab_instance_ids=prefab_ids,
    )


def _bounds(value: object) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (6,) or not np.all(np.isfinite(result)):
        raise ValueError("WorldGen structure marker bounds must be finite min/max XYZ")
    if np.any(result[:3] >= result[3:]):
        raise ValueError("WorldGen structure marker bounds must be positive")
    return result


def _marker_asset_id_list(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("marker asset IDs must be a sequence")
    result = tuple(_asset_id(row, "marker asset ID") for row in value)
    if not result or len(result) > NATIVE_STRUCTURE_MARKER_MAX_CAPACITY:
        raise ValueError("marker asset IDs must contain 1..256 rows")
    if len(set(result)) != len(result):
        raise ValueError("marker asset IDs must be unique")
    return tuple(sorted(result))


def _row_marker_asset_ids(
    value: object,
    count: int,
    requested: tuple[str, ...],
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError("marker asset ID rows differ from emitted count")
    result = tuple(_asset_id(row, "marker asset ID") for row in value)
    if any(row not in requested for row in result):
        raise ValueError("response contains an unrequested marker asset ID")
    return result


def _decode(
    value: object,
    dtype: str,
    count: int,
    label: str,
) -> np.ndarray:
    payload = _binary(value, label)
    expected = count * np.dtype(dtype).itemsize
    if len(payload) != expected:
        raise ValueError(f"{label} binary length changed")
    return np.frombuffer(payload, dtype=dtype).copy()


def _binary(value: object, label: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{label} must be binary")
    return bytes(value)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256")
    return value.lower()


def _asset_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty portable string")
    return value


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _exact_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    result = _exact_int(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _positive_int(value: object, label: str) -> int:
    result = _exact_int(value, label)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result
