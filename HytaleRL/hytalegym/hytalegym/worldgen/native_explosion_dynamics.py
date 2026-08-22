"""Strict host contract for native explosion entity dynamics.

The fixture invokes ``ExplosionUtils.performExplosion`` with a server-owned
entity-only configuration.  It restores health for inspection, but callers
must reset immediately because native damage systems may have other effects.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np
import numpy.typing as npt

from hytalegym.worldgen.native_explosion_ordering import (
    EXPLOSION_UTILS_CLASS_SHA256,
    native_explosion_ordering_contract_sha256,
)


NATIVE_EXPLOSION_DYNAMICS_SCHEMA = "hytalerl_native_explosion_dynamics_probe_v1"
NATIVE_EXPLOSION_DYNAMICS_VERSION = 1
NATIVE_EXPLOSION_DYNAMICS_MAX_CAPACITY = 8
NATIVE_EXPLOSION_DYNAMICS_FIXTURE_KIND = "open_two_entity"
NATIVE_EXPLOSION_DYNAMICS_PHASE_ORDER = (
    "processTargetBlocks_then_processTargetEntities_then_effects"
)
NATIVE_EXPLOSION_DYNAMICS_ROW_ORDER = (
    "UUID_text_ascending_serialization_not_native_execution_order"
)
NATIVE_EXPLOSION_ENTITY_DAMAGE_RADIUS = 5.0
NATIVE_EXPLOSION_ENTITY_DAMAGE = 8.0
NATIVE_EXPLOSION_ENTITY_DAMAGE_FALLOFF = 2.0
_NATIVE_METHOD = "ExplosionUtils.performExplosion"
DAMAGE_SYSTEMS_CLASS_SHA256 = (
    "F5DB0A203DC80920699D1BB92EF3E3537C081D9E5A1A5E2E5FEB969A52D3F601"
)


@dataclass(frozen=True, slots=True)
class NativeExplosionEntityEffect:
    ordinal: int
    uuid: str
    position: npt.NDArray[np.float64]
    distance: float
    health_before: float
    health_after: float
    expected_raw_damage: float
    applied_health_delta: float

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("ordinal must be non-negative")
        if not self.uuid:
            raise ValueError("uuid must be present")
        position = _array(self.position, (3,), "position")
        values = (
            self.distance,
            self.health_before,
            self.health_after,
            self.expected_raw_damage,
            self.applied_health_delta,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("effect scalars must be finite")
        if self.distance < 0.0 or self.expected_raw_damage < 0.0:
            raise ValueError("distance and damage must be non-negative")
        if not math.isclose(
            self.applied_health_delta,
            self.health_before - self.health_after,
            abs_tol=1e-5,
        ):
            raise ValueError("applied_health_delta disagrees with health")
        object.__setattr__(self, "position", position.copy())


@dataclass(frozen=True, slots=True)
class NativeExplosionDynamicsCapture:
    bridge_sha256: str
    server_version: str
    world: str
    worldgen_provider: str
    worldgen_version: str
    seed: int
    world_epoch: str
    origin: npt.NDArray[np.float64]
    capacity: int
    total_matching: int
    overflow: bool
    complete: bool
    session_reset_required: bool
    failure_reason: str
    effects: tuple[NativeExplosionEntityEffect, ...]

    def __post_init__(self) -> None:
        bridge = _sha256(self.bridge_sha256, "bridge_sha256")
        origin = _array(self.origin, (3,), "origin")
        if not 1 <= self.capacity <= NATIVE_EXPLOSION_DYNAMICS_MAX_CAPACITY:
            raise ValueError("capacity is outside the v1 bound")
        if self.total_matching < 0:
            raise ValueError("total_matching must be non-negative")
        if self.overflow != (self.total_matching > self.capacity):
            raise ValueError("overflow must exactly reflect capacity")
        expected_complete = (
            not self.overflow
            and not self.failure_reason
            and len(self.effects) == self.total_matching
        )
        if self.complete != expected_complete:
            raise ValueError("complete violates complete-or-empty semantics")
        if not self.complete and self.effects:
            raise ValueError("incomplete captures must publish no effects")
        if not self.session_reset_required:
            raise ValueError("the native fixture always requires reset")
        uuids = tuple(effect.uuid for effect in self.effects)
        if uuids != tuple(sorted(set(uuids))):
            raise ValueError("effects must be uniquely UUID-ordered")
        for ordinal, effect in enumerate(self.effects):
            if effect.ordinal != ordinal:
                raise ValueError("effect ordinals must be consecutive")
            expected_distance = float(np.linalg.norm(effect.position - origin))
            if not math.isclose(effect.distance, expected_distance, abs_tol=1e-6):
                raise ValueError("effect distance disagrees with origin")
            expected_damage = native_explosion_raw_damage(effect.distance)
            if not math.isclose(
                effect.expected_raw_damage, expected_damage, abs_tol=1e-5
            ):
                raise ValueError("native expected damage disagrees with source formula")
            expected_delta = native_explosion_expected_health_delta(expected_damage)
            if not math.isclose(
                effect.applied_health_delta, expected_delta, abs_tol=1e-5
            ):
                raise ValueError("health delta disagrees with native integer rounding")
        object.__setattr__(self, "bridge_sha256", bridge)
        object.__setattr__(self, "origin", origin.copy())

    @classmethod
    def from_response(
        cls, response: Mapping[str, Any]
    ) -> NativeExplosionDynamicsCapture:
        source = dict(response)
        expected = {
            "type": "explosion_dynamics_probe",
            "schema": NATIVE_EXPLOSION_DYNAMICS_SCHEMA,
            "version": NATIVE_EXPLOSION_DYNAMICS_VERSION,
            "fixture_kind": NATIVE_EXPLOSION_DYNAMICS_FIXTURE_KIND,
            "damage_blocks": False,
            "damage_entities": True,
            "entity_damage_radius": NATIVE_EXPLOSION_ENTITY_DAMAGE_RADIUS,
            "entity_damage": NATIVE_EXPLOSION_ENTITY_DAMAGE,
            "entity_damage_falloff": NATIVE_EXPLOSION_ENTITY_DAMAGE_FALLOFF,
            "knockback": False,
            "session_reset_required": True,
            "phase_order": NATIVE_EXPLOSION_DYNAMICS_PHASE_ORDER,
            "row_order": NATIVE_EXPLOSION_DYNAMICS_ROW_ORDER,
            "native_method": _NATIVE_METHOD,
        }
        for key, value in expected.items():
            if source.get(key) != value:
                raise ValueError(f"{key} does not match the v1 contract")
        effects = tuple(_effect(row) for row in _rows(source.get("effects")))
        return cls(
            bridge_sha256=_text(source.get("bridge_sha256"), "bridge_sha256"),
            server_version=_text(source.get("server_version"), "server_version"),
            world=_text(source.get("world"), "world"),
            worldgen_provider=_text(
                source.get("worldgen_provider"), "worldgen_provider"
            ),
            worldgen_version=_text(source.get("worldgen_version"), "worldgen_version"),
            seed=_integer(source.get("seed"), "seed"),
            world_epoch=_text(source.get("world_epoch"), "world_epoch"),
            origin=_array(source.get("origin_f64_xyz"), (3,), "origin_f64_xyz"),
            capacity=_integer(source.get("capacity"), "capacity"),
            total_matching=_integer(source.get("total_matching"), "total_matching"),
            overflow=_boolean(source.get("overflow"), "overflow"),
            complete=_boolean(source.get("complete"), "complete"),
            session_reset_required=True,
            failure_reason=_optional_text(source.get("failure_reason")),
            effects=effects,
        )


def native_explosion_dynamics_request(
    world_epoch: str,
    *,
    capacity: int = NATIVE_EXPLOSION_DYNAMICS_MAX_CAPACITY,
) -> dict[str, object]:
    """Build one terminal, server-owned native dynamics request."""

    bounded = _integer(capacity, "capacity")
    if not 1 <= bounded <= NATIVE_EXPLOSION_DYNAMICS_MAX_CAPACITY:
        raise ValueError("capacity is outside the v1 bound")
    return {
        "type": "explosion_dynamics_probe",
        "schema": NATIVE_EXPLOSION_DYNAMICS_SCHEMA,
        "version": NATIVE_EXPLOSION_DYNAMICS_VERSION,
        "world_epoch": _text(world_epoch, "world_epoch"),
        "fixture_kind": NATIVE_EXPLOSION_DYNAMICS_FIXTURE_KIND,
        "capacity": bounded,
    }


def native_explosion_raw_damage(distance: float) -> float:
    """Evaluate Hytale 0.5.7's native entity-damage falloff formula."""

    value = float(distance)
    if (
        not math.isfinite(value)
        or not 0.0 <= value <= NATIVE_EXPLOSION_ENTITY_DAMAGE_RADIUS
    ):
        raise ValueError("distance must be finite and inside the fixture radius")
    return (
        NATIVE_EXPLOSION_ENTITY_DAMAGE
        * (1.0 - value / NATIVE_EXPLOSION_ENTITY_DAMAGE_RADIUS)
        ** NATIVE_EXPLOSION_ENTITY_DAMAGE_FALLOFF
    )


def native_explosion_expected_health_delta(raw_damage: float) -> float:
    """Apply ``DamageSystems.ApplyDamage``'s Java ``Math.round`` boundary."""

    value = float(raw_damage)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("raw_damage must be finite and non-negative")
    return float(math.floor(value + 0.5))


def native_explosion_dynamics_contract() -> dict[str, object]:
    return {
        "schema": "hytalerl_native_explosion_dynamics_host_v1",
        "version": 1,
        "wire_schema": NATIVE_EXPLOSION_DYNAMICS_SCHEMA,
        "native_method": _NATIVE_METHOD,
        "source_class": "com.hypixel.hytale.server.core.entity.ExplosionUtils",
        "source_class_sha256": EXPLOSION_UTILS_CLASS_SHA256,
        "damage_application_source": {
            "class": (
                "com.hypixel.hytale.server.core.modules.entity.damage.DamageSystems"
            ),
            "class_sha256": DAMAGE_SYSTEMS_CLASS_SHA256,
            "rule": "Math.round(raw_damage)_before_health_subtraction",
        },
        "ordering_contract_sha256": native_explosion_ordering_contract_sha256(),
        "configuration": {
            "damage_blocks": False,
            "damage_entities": True,
            "entity_damage_radius": NATIVE_EXPLOSION_ENTITY_DAMAGE_RADIUS,
            "entity_damage": NATIVE_EXPLOSION_ENTITY_DAMAGE,
            "entity_damage_falloff": NATIVE_EXPLOSION_ENTITY_DAMAGE_FALLOFF,
            "knockback": False,
        },
        "row_order": NATIVE_EXPLOSION_DYNAMICS_ROW_ORDER,
        "row_order_is_native_execution_order": False,
        "effect": "terminal_native_damage_then_health_restore_then_reset",
        "provenance": "native_ExplosionUtils_performExplosion",
        "fail_closed": True,
    }


def native_explosion_dynamics_contract_sha256() -> str:
    payload = json.dumps(
        native_explosion_dynamics_contract(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _effect(value: object) -> NativeExplosionEntityEffect:
    if not isinstance(value, Mapping):
        raise TypeError("effect must be a mapping")
    source = dict(value)
    if source.get("damage_cause") != "ENVIRONMENT":
        raise ValueError("damage_cause must be ENVIRONMENT")
    return NativeExplosionEntityEffect(
        ordinal=_integer(source.get("ordinal"), "ordinal"),
        uuid=_text(source.get("uuid"), "uuid"),
        position=_array(source.get("position_f64_xyz"), (3,), "position"),
        distance=_finite(source.get("distance"), "distance"),
        health_before=_finite(source.get("health_before"), "health_before"),
        health_after=_finite(source.get("health_after"), "health_after"),
        expected_raw_damage=_finite(
            source.get("expected_raw_damage"), "expected_raw_damage"
        ),
        applied_health_delta=_finite(
            source.get("applied_health_delta"), "applied_health_delta"
        ),
    )


def _rows(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("effects must be a sequence")
    return tuple(value)


def _array(value: object, shape: tuple[int, ...], label: str):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _optional_text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("failure_reason must be text")
    return value


def _sha256(value: object, label: str) -> str:
    result = _text(value, label).upper()
    if len(result) != 64 or any(
        character not in "0123456789ABCDEF" for character in result
    ):
        raise ValueError(f"{label} must be a SHA-256")
    return result


__all__ = [
    "DAMAGE_SYSTEMS_CLASS_SHA256",
    "NATIVE_EXPLOSION_DYNAMICS_FIXTURE_KIND",
    "NATIVE_EXPLOSION_DYNAMICS_MAX_CAPACITY",
    "NATIVE_EXPLOSION_DYNAMICS_PHASE_ORDER",
    "NATIVE_EXPLOSION_DYNAMICS_ROW_ORDER",
    "NATIVE_EXPLOSION_DYNAMICS_SCHEMA",
    "NATIVE_EXPLOSION_DYNAMICS_VERSION",
    "NativeExplosionDynamicsCapture",
    "NativeExplosionEntityEffect",
    "native_explosion_dynamics_contract",
    "native_explosion_dynamics_contract_sha256",
    "native_explosion_dynamics_request",
    "native_explosion_expected_health_delta",
    "native_explosion_raw_damage",
]
