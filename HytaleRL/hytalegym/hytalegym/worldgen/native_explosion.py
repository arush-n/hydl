"""Strict host contract for Hytale's native explosion entity admission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
import numpy.typing as npt


NATIVE_EXPLOSION_CANDIDATE_SCHEMA = (
    "hytalerl_native_explosion_candidate_probe_v1"
)
NATIVE_EXPLOSION_CANDIDATE_VERSION = 1
NATIVE_EXPLOSION_CANDIDATE_MAX_CAPACITY = 64
MAX_ENTITY_ONLY_EXPLOSION_BLOCK_RADIUS = 5
_NATIVE_METHOD = "ExplosionUtils.processTargetBlocks"
_JAX_EXPLOSION_PRODUCER_SHA256 = (
    "bb4aac6bedcfa5ecbb2bc84c027ba32cca735eece7f9319945e976d511316282"
)
_EXPLOSION_UTILS_CLASS_SHA256 = (
    "03FE187757565EA414053D795136044F3362246B26136B5B1A6387DDA2F6D4B5"
)


@dataclass(frozen=True, slots=True)
class NativeExplosionCandidateCapture:
    """One complete, non-damaging native explosion candidate set."""

    bridge_sha256: str
    server_version: str
    world: str
    worldgen_provider: str
    worldgen_version: str
    seed: int
    origin: npt.NDArray[np.float64]
    block_damage_radius: int
    entity_damage_radius: float
    ignore_controlled_actor: bool
    capacity: int
    total_matching: int
    overflow: bool
    uuid_bytes: bytes
    positions: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        bridge = _sha256_text(self.bridge_sha256, "bridge_sha256")
        origin = _array(self.origin, (3,), np.float64, "origin")
        positions = np.asarray(self.positions, dtype=np.float64)
        if positions.shape != (self.emitted_count, 3):
            raise ValueError("positions must match emitted UUID rows")
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions must be finite")
        if len(self.uuid_bytes) != self.emitted_count * 16:
            raise ValueError("uuid_bytes must contain 16 bytes per row")
        if len(set(self.uuid_rows())) != self.emitted_count:
            raise ValueError("candidate UUID rows must be unique")
        if tuple(self.uuid_rows()) != tuple(sorted(self.uuid_rows())):
            raise ValueError("candidate UUID rows must be ordered")
        object.__setattr__(self, "bridge_sha256", bridge)
        object.__setattr__(self, "origin", origin.copy())
        object.__setattr__(self, "positions", positions.copy())

    @property
    def emitted_count(self) -> int:
        return 0 if self.overflow else self.total_matching

    def uuid_rows(self) -> tuple[bytes, ...]:
        return tuple(
            self.uuid_bytes[offset : offset + 16]
            for offset in range(0, len(self.uuid_bytes), 16)
        )

    @classmethod
    def from_response(
        cls,
        response: Mapping[str, Any],
    ) -> NativeExplosionCandidateCapture:
        source = dict(response)
        _require_equal(source, "type", "explosion_candidate_probe")
        _require_equal(source, "schema", NATIVE_EXPLOSION_CANDIDATE_SCHEMA)
        _require_equal(source, "version", NATIVE_EXPLOSION_CANDIDATE_VERSION)
        _require_equal(source, "damage_blocks", False)
        _require_equal(source, "native_method", _NATIVE_METHOD)

        capacity = _bounded_int(
            source.get("capacity"),
            1,
            NATIVE_EXPLOSION_CANDIDATE_MAX_CAPACITY,
            "capacity",
        )
        total = _bounded_int(
            source.get("total_matching"), 0, 2**31 - 1, "total_matching"
        )
        overflow = _boolean(source.get("overflow"), "overflow")
        if overflow != (total > capacity):
            raise ValueError("overflow must exactly reflect total > capacity")
        emitted = _bounded_int(
            source.get("emitted_count"), 0, capacity, "emitted_count"
        )
        if emitted != (0 if overflow else total):
            raise ValueError("emitted_count violates complete-or-empty semantics")
        _require_equal(
            source,
            "uuid_encoding",
            "rfc4122_network_order_16_bytes_per_row",
        )
        uuid_bytes = _bytes(source.get("uuid_bytes"), "uuid_bytes")
        positions_raw = _bytes(
            source.get("positions_f64_le_xyz"),
            "positions_f64_le_xyz",
        )
        origin_raw = _bytes(source.get("origin_f64_le_xyz"), "origin_f64_le_xyz")
        if len(origin_raw) != 3 * np.dtype("<f8").itemsize:
            raise ValueError("origin_f64_le_xyz must contain one XYZ row")
        if len(positions_raw) != emitted * 3 * np.dtype("<f8").itemsize:
            raise ValueError("positions payload must match emitted_count")

        block_radius = _bounded_int(
            source.get("block_damage_radius"),
            1,
            MAX_ENTITY_ONLY_EXPLOSION_BLOCK_RADIUS,
            "block_damage_radius",
        )
        entity_radius = _finite_float(
            source.get("entity_damage_radius"), "entity_damage_radius"
        )
        if not 0.0 < entity_radius <= MAX_ENTITY_ONLY_EXPLOSION_BLOCK_RADIUS:
            raise ValueError(
                "entity_damage_radius must be in (0,"
                f"{MAX_ENTITY_ONLY_EXPLOSION_BLOCK_RADIUS}]"
            )
        return cls(
            bridge_sha256=_text(source.get("bridge_sha256"), "bridge_sha256"),
            server_version=_text(source.get("server_version"), "server_version"),
            world=_text(source.get("world"), "world"),
            worldgen_provider=_text(
                source.get("worldgen_provider"), "worldgen_provider"
            ),
            worldgen_version=_text(
                source.get("worldgen_version"), "worldgen_version"
            ),
            seed=_bounded_int(source.get("seed"), -(2**63), 2**63 - 1, "seed"),
            origin=np.frombuffer(origin_raw, dtype="<f8").copy(),
            block_damage_radius=block_radius,
            entity_damage_radius=entity_radius,
            ignore_controlled_actor=_boolean(
                source.get("ignore_controlled_actor"),
                "ignore_controlled_actor",
            ),
            capacity=capacity,
            total_matching=total,
            overflow=overflow,
            uuid_bytes=uuid_bytes,
            positions=np.frombuffer(positions_raw, dtype="<f8")
            .reshape(emitted, 3)
            .copy(),
        )


def native_explosion_candidate_request(
    origin: npt.ArrayLike,
    *,
    block_damage_radius: int,
    entity_damage_radius: float,
    ignore_controlled_actor: bool = True,
    capacity: int = NATIVE_EXPLOSION_CANDIDATE_MAX_CAPACITY,
) -> dict[str, object]:
    """Build one bounded, entity-only native explosion admission request."""

    point = _array(origin, (3,), np.float64, "origin")
    block_radius = _bounded_int(
        block_damage_radius,
        1,
        MAX_ENTITY_ONLY_EXPLOSION_BLOCK_RADIUS,
        "block_damage_radius",
    )
    entity_radius = _finite_float(entity_damage_radius, "entity_damage_radius")
    if not 0.0 < entity_radius <= MAX_ENTITY_ONLY_EXPLOSION_BLOCK_RADIUS:
        raise ValueError(
            "entity_damage_radius must be in (0,"
            f"{MAX_ENTITY_ONLY_EXPLOSION_BLOCK_RADIUS}]"
        )
    return {
        "type": "explosion_candidate_probe",
        "origin_f64_le_xyz": np.asarray(point, dtype="<f8").tobytes(),
        "block_damage_radius": block_radius,
        "entity_damage_radius": entity_radius,
        "ignore_controlled_actor": bool(ignore_controlled_actor),
        "capacity": _bounded_int(
            capacity,
            1,
            NATIVE_EXPLOSION_CANDIDATE_MAX_CAPACITY,
            "capacity",
        ),
    }


def native_explosion_candidate_contract() -> dict[str, object]:
    return {
        "schema": "hytalerl_native_explosion_candidate_host_v1",
        "version": 1,
        "wire_schema": NATIVE_EXPLOSION_CANDIDATE_SCHEMA,
        "native_method": _NATIVE_METHOD,
        "source_class": "com.hypixel.hytale.server.core.entity.ExplosionUtils",
        "source_class_sha256": _EXPLOSION_UTILS_CLASS_SHA256,
        "jax_producer_sha256": _JAX_EXPLOSION_PRODUCER_SHA256,
        "scope": {
            "damage_entities": True,
            "damage_blocks": False,
            "max_radius": MAX_ENTITY_ONLY_EXPLOSION_BLOCK_RADIUS,
            "max_candidates": NATIVE_EXPLOSION_CANDIDATE_MAX_CAPACITY,
        },
        "identity": "candidate_UUID_network_order_bytes",
        "overflow": "complete_rows_or_empty_fail_closed",
        "effect": "read_only_no_damage_no_knockback_no_block_mutation",
        "provenance": "native_private_admission_invoked_on_world_thread",
    }


def native_explosion_candidate_contract_sha256() -> str:
    payload = json.dumps(
        native_explosion_candidate_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array(value: npt.ArrayLike, shape: tuple[int, ...], dtype, label: str):
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite")
    return result


def _bounded_int(value: object, low: int, high: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{label} must be in [{low},{high}]")
    return result


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean")
    return value


def _bytes(value: object, label: str) -> bytes:
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise TypeError(f"{label} must be binary")
    return bytes(value)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _sha256_text(value: object, label: str) -> str:
    result = _text(value, label).upper()
    if len(result) != 64 or set(result) - set("0123456789ABCDEF"):
        raise ValueError(f"{label} must be a SHA-256")
    return result


def _require_equal(source: Mapping[str, Any], key: str, expected: object) -> None:
    if source.get(key) != expected:
        raise ValueError(f"{key} must equal {expected!r}")


__all__ = [
    "NATIVE_EXPLOSION_CANDIDATE_MAX_CAPACITY",
    "NATIVE_EXPLOSION_CANDIDATE_SCHEMA",
    "NATIVE_EXPLOSION_CANDIDATE_VERSION",
    "NativeExplosionCandidateCapture",
    "native_explosion_candidate_contract",
    "native_explosion_candidate_contract_sha256",
    "native_explosion_candidate_request",
]
