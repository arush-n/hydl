"""Checked adapter for native privileged entity snapshots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from hytalegym.jax.world.entities.privileged import (
    PRIVILEGED_ENTITY_CAPACITY,
    PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NONE,
    PRIVILEGED_ENTITY_PROVENANCE_NATIVE_RUNTIME,
    PrivilegedEntitySnapshot,
)


NATIVE_PRIVILEGED_ENTITY_SCHEMA = "hytalerl_privileged_entity_snapshot_v1"
NATIVE_PRIVILEGED_ENTITY_VERSION = 1
NATIVE_PRIVILEGED_ENTITY_COMPONENT_FILTER = "uuid_transform_velocity_v1"
NATIVE_PRIVILEGED_ENTITY_MAX_CAPACITY = 256

_BOUNDS_SEMANTICS = "half_open_min_inclusive_max_exclusive"
_UUID_ENCODING = "rfc4122_network_order_16_bytes_per_row"
_ROTATION_UNITS = "radians_yaw_pitch_roll"
_INSTANCE_DOMAIN = b"hytalerl-native-entity-instance-v1\x00"
_TYPE_DOMAIN = b"hytalerl-native-entity-type-v1\x00"


@dataclass(frozen=True)
class NativePrivilegedEntityCapture:
    """Complete bounded UUID/pose/velocity rows from one native world."""

    bridge_sha256: str
    server_version: str
    world: str
    worldgen_provider: str
    worldgen_version: str
    seed: int
    bounds: npt.NDArray[np.float64]
    capacity: int
    total_matching: int
    overflow: bool
    uuid_bytes: npt.NDArray[np.uint8]
    position: npt.NDArray[np.float64]
    rotation: npt.NDArray[np.float64]
    velocity: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[0-9A-Fa-f]{64}", self.bridge_sha256):
            raise ValueError("bridge_sha256 must be a SHA-256")
        for name in (
            "server_version",
            "world",
            "worldgen_provider",
            "worldgen_version",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(
                self,
                name,
            ):
                raise ValueError(f"{name} must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        capacity = _bounded_int(
            self.capacity,
            1,
            NATIVE_PRIVILEGED_ENTITY_MAX_CAPACITY,
            "capacity",
        )
        total = _bounded_int(
            self.total_matching,
            0,
            2**31 - 1,
            "total_matching",
        )
        if not isinstance(self.overflow, bool):
            raise TypeError("overflow must be boolean")
        if self.overflow != (total > capacity):
            raise ValueError("overflow must exactly reflect total_matching")
        emitted = 0 if self.overflow else total
        arrays = {
            "bounds": _array(self.bounds, np.float64, (6,)),
            "uuid_bytes": _array(
                self.uuid_bytes,
                np.uint8,
                (emitted, 16),
            ),
            "position": _array(
                self.position,
                np.float64,
                (emitted, 3),
            ),
            "rotation": _array(
                self.rotation,
                np.float64,
                (emitted, 3),
            ),
            "velocity": _array(
                self.velocity,
                np.float64,
                (emitted, 3),
            ),
        }
        bounds = arrays["bounds"]
        if (
            not np.all(np.isfinite(bounds))
            or np.any(bounds[:3] >= bounds[3:])
        ):
            raise ValueError("bounds must be a finite positive AABB")
        for name in ("position", "rotation", "velocity"):
            if not np.all(np.isfinite(arrays[name])):
                raise ValueError(f"{name} must be finite")
        position = arrays["position"]
        if emitted and np.any(
            (position < bounds[:3]) | (position >= bounds[3:])
        ):
            raise ValueError("entity position lies outside query bounds")
        identities = [bytes(row) for row in arrays["uuid_bytes"]]
        if any(
            left >= right
            for left, right in zip(identities, identities[1:])
        ):
            raise ValueError("UUID rows must be unique and strictly ordered")
        for name, value in arrays.items():
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "total_matching", total)
        object.__setattr__(
            self,
            "bridge_sha256",
            self.bridge_sha256.upper(),
        )

    @classmethod
    def from_response(
        cls,
        response: Mapping[str, Any],
    ) -> NativePrivilegedEntityCapture:
        """Decode the versioned MessagePack bridge response."""

        source = dict(response)
        if source.get("type") != "privileged_entity_snapshot":
            raise ValueError("native response is not an entity snapshot")
        if source.get("schema") != NATIVE_PRIVILEGED_ENTITY_SCHEMA:
            raise ValueError("unsupported privileged entity schema")
        if source.get("version") != NATIVE_PRIVILEGED_ENTITY_VERSION:
            raise ValueError("unsupported privileged entity version")
        if source.get("component_filter") != (
            NATIVE_PRIVILEGED_ENTITY_COMPONENT_FILTER
        ):
            raise ValueError("unsupported privileged entity component filter")
        if source.get("bounds_semantics") != _BOUNDS_SEMANTICS:
            raise ValueError("unsupported entity bounds semantics")
        if source.get("uuid_encoding") != _UUID_ENCODING:
            raise ValueError("unsupported entity UUID encoding")
        if source.get("rotation_units") != _ROTATION_UNITS:
            raise ValueError("unsupported entity rotation units")
        emitted = _bounded_int(
            source.get("emitted_count"),
            0,
            NATIVE_PRIVILEGED_ENTITY_MAX_CAPACITY,
            "emitted_count",
        )
        capacity = _bounded_int(
            source.get("capacity"),
            1,
            NATIVE_PRIVILEGED_ENTITY_MAX_CAPACITY,
            "capacity",
        )
        total = _bounded_int(
            source.get("total_matching"),
            0,
            2**31 - 1,
            "total_matching",
        )
        overflow = _boolean(source, "overflow")
        if emitted != (0 if overflow else total):
            raise ValueError("emitted_count contradicts overflow state")
        return cls(
            bridge_sha256=_text(source, "bridge_sha256"),
            server_version=_text(source, "server_version"),
            world=_text(source, "world"),
            worldgen_provider=_text(source, "worldgen_provider"),
            worldgen_version=_text(source, "worldgen_version"),
            seed=_integer(source, "seed"),
            bounds=_decode(source, "bounds_f64_le_min_max_xyz", "<f8", 6),
            capacity=capacity,
            total_matching=total,
            overflow=overflow,
            uuid_bytes=_decode(
                source,
                "uuid_bytes",
                "u1",
                emitted * 16,
            ).reshape(emitted, 16),
            position=_decode(
                source,
                "positions_f64_le_xyz",
                "<f8",
                emitted * 3,
            ).reshape(emitted, 3),
            rotation=_decode(
                source,
                "rotations_f64_le_yaw_pitch_roll",
                "<f8",
                emitted * 3,
            ).reshape(emitted, 3),
            velocity=_decode(
                source,
                "velocities_f64_le_xyz",
                "<f8",
                emitted * 3,
            ).reshape(emitted, 3),
        )

    @property
    def emitted_count(self) -> int:
        return int(self.uuid_bytes.shape[0])

    def instance_identity_words(self) -> npt.NDArray[np.uint32]:
        """Project UUIDs to the public pair and reject any collision."""

        return _checked_words(
            [bytes(row) for row in self.uuid_bytes],
            _INSTANCE_DOMAIN,
            "native entity UUID",
        )


def native_privileged_entity_request(
    bounds: npt.ArrayLike,
    *,
    capacity: int = NATIVE_PRIVILEGED_ENTITY_MAX_CAPACITY,
) -> dict[str, object]:
    """Build one bounded native snapshot request."""

    box = np.asarray(bounds, dtype=np.float64)
    if (
        box.shape != (6,)
        or not np.all(np.isfinite(box))
        or np.any(box[:3] >= box[3:])
    ):
        raise ValueError("bounds must be one finite positive AABB")
    return {
        "type": "privileged_entity_snapshot",
        "component_filter": NATIVE_PRIVILEGED_ENTITY_COMPONENT_FILTER,
        "bounds_f64_le_min_max_xyz": np.ascontiguousarray(
            box,
            dtype="<f8",
        ).tobytes(),
        "capacity": _bounded_int(
            capacity,
            1,
            NATIVE_PRIVILEGED_ENTITY_MAX_CAPACITY,
            "capacity",
        ),
    }


def native_privileged_entity_snapshot(
    capture: NativePrivilegedEntityCapture,
    *,
    type_asset_ids: Sequence[str] | None = None,
    entity_capacity: int = PRIVILEGED_ENTITY_CAPACITY,
) -> PrivilegedEntitySnapshot:
    """Adapt native spatial rows; absent type evidence remains fail-closed."""

    if not isinstance(capture, NativePrivilegedEntityCapture):
        raise TypeError("capture must be NativePrivilegedEntityCapture")
    capacity = _bounded_int(
        entity_capacity,
        1,
        PRIVILEGED_ENTITY_CAPACITY,
        "entity_capacity",
    )
    count = capture.emitted_count
    overflow = capture.overflow or count > capacity
    shape = (1, capacity)
    initialized = np.zeros(shape, dtype=np.bool_)
    active = np.zeros(shape, dtype=np.bool_)
    identity = np.zeros(shape + (2,), dtype=np.uint32)
    type_identity = np.zeros(shape + (2,), dtype=np.uint32)
    position = np.zeros(shape + (3,), dtype=np.float32)
    rotation = np.zeros(shape + (3,), dtype=np.float32)
    velocity = np.zeros(shape + (3,), dtype=np.float32)
    if not overflow:
        initialized[0, :count] = True
        active[0, :count] = True
        identity[0, :count] = capture.instance_identity_words()
        position[0, :count] = capture.position
        rotation[0, :count] = capture.rotation
        velocity[0, :count] = capture.velocity
        if type_asset_ids is not None:
            values = tuple(type_asset_ids)
            if len(values) != count:
                raise ValueError("type_asset_ids must match emitted rows")
            type_identity[0, :count] = _checked_words(
                [_asset_id(value) for value in values],
                _TYPE_DOMAIN,
                "native entity type asset",
                allow_equal_inputs=True,
            )
    unsupported = np.zeros(shape, dtype=np.bool_)
    vectors = np.zeros(shape + (3,), dtype=np.float32)
    return PrivilegedEntitySnapshot(
        source_available=jnp.asarray([True]),
        source_overflow=jnp.asarray([overflow]),
        source_provenance=PRIVILEGED_ENTITY_PROVENANCE_NATIVE_RUNTIME,
        initialized=jnp.asarray(initialized),
        active=jnp.asarray(active),
        identity_words=jnp.asarray(identity),
        type_identity_words=jnp.asarray(type_identity),
        kind=jnp.zeros(shape, dtype=jnp.uint8),
        position=jnp.asarray(position),
        rotation=jnp.asarray(rotation),
        velocity=jnp.asarray(velocity),
        behavior_supported=jnp.asarray(unsupported),
        geometry_supported=jnp.asarray(unsupported),
        collidable=jnp.asarray(unsupported),
        blocks_los=jnp.asarray(unsupported),
        local_bounds=jnp.zeros(shape + (6,), dtype=jnp.float32),
        los_offset_supported=jnp.asarray(unsupported),
        los_offset=jnp.asarray(vectors),
        behavior_provenance=jnp.full(
            shape,
            PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NONE,
            dtype=jnp.uint8,
        ),
        geometry_provenance=jnp.full(
            shape,
            PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NONE,
            dtype=jnp.uint8,
        ),
    )


def native_privileged_entity_adapter_contract() -> dict[str, object]:
    return {
        "schema": "hytalerl_native_privileged_entity_adapter_v1",
        "version": 1,
        "source": NATIVE_PRIVILEGED_ENTITY_SCHEMA,
        "component_filter": NATIVE_PRIVILEGED_ENTITY_COMPONENT_FILTER,
        "capacity": NATIVE_PRIVILEGED_ENTITY_MAX_CAPACITY,
        "uuid": {
            "wire": _UUID_ENCODING,
            "scene_projection": (
                "first_64_bits_sha256_of_domain_nul_plus_uuid_bytes_"
                "as_two_big_endian_uint32"
            ),
            "domain": _INSTANCE_DOMAIN[:-1].decode(),
            "collision": "reject_complete_capture",
            "lifetime": (
                "one_continuously_loaded_entity_only_not_reset_or_reload"
            ),
        },
        "type_identity": {
            "source": "explicit_palette_independent_asset_id_required",
            "projection": (
                "first_64_bits_sha256_of_domain_nul_plus_utf8_asset_id_"
                "as_two_big_endian_uint32"
            ),
            "missing": "scene_fails_closed",
            "native_v1_wire": "not_present",
        },
        "behavior": "unsupported",
        "geometry": "unsupported",
        "overflow": "no_partial_rows",
    }


def native_privileged_entity_adapter_contract_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            native_privileged_entity_adapter_contract(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _checked_words(
    values: Sequence[bytes],
    domain: bytes,
    label: str,
    *,
    allow_equal_inputs: bool = False,
) -> npt.NDArray[np.uint32]:
    result = np.zeros((len(values), 2), dtype=np.uint32)
    seen: dict[tuple[int, int], bytes] = {}
    for index, value in enumerate(values):
        digest = hashlib.sha256(domain + value).digest()
        words = np.frombuffer(digest[:8], dtype=">u4").astype(np.uint32)
        key = (int(words[0]), int(words[1]))
        previous = seen.get(key)
        if key == (0, 0) or (
            previous is not None
            and (not allow_equal_inputs or previous != value)
        ):
            raise ValueError(f"{label} projection collided")
        seen[key] = value
        result[index] = words
    result.flags.writeable = False
    return result


def _asset_id(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("type asset IDs must be non-empty strings")
    return value.encode()


def _array(
    value: npt.ArrayLike,
    dtype: npt.DTypeLike,
    shape: tuple[int, ...],
) -> np.ndarray:
    result = np.asarray(value, dtype=dtype).copy()
    if result.shape != shape:
        raise ValueError(f"array must have shape {shape}")
    return result


def _decode(
    source: Mapping[str, Any],
    name: str,
    dtype: npt.DTypeLike,
    count: int,
) -> np.ndarray:
    payload = source.get(name)
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError(f"{name} must be binary")
    expected = np.dtype(dtype).itemsize * count
    if len(payload) != expected:
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
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _integer(source: Mapping[str, Any], name: str) -> int:
    value = source.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _boolean(source: Mapping[str, Any], name: str) -> bool:
    value = source.get(name)
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be boolean")
    return value


def _text(source: Mapping[str, Any], name: str) -> str:
    value = source.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


__all__ = [
    "NATIVE_PRIVILEGED_ENTITY_COMPONENT_FILTER",
    "NATIVE_PRIVILEGED_ENTITY_MAX_CAPACITY",
    "NATIVE_PRIVILEGED_ENTITY_SCHEMA",
    "NATIVE_PRIVILEGED_ENTITY_VERSION",
    "NativePrivilegedEntityCapture",
    "native_privileged_entity_adapter_contract",
    "native_privileged_entity_adapter_contract_sha256",
    "native_privileged_entity_request",
    "native_privileged_entity_snapshot",
]
