"""Strict host contract for native dual-LOS and PositionCache evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

NATIVE_LOS_EVIDENCE_SCHEMA = "hytalerl_native_los_evidence_v1"
NATIVE_LOS_EVIDENCE_VERSION = 1
NATIVE_LOS_MAX_CACHE_TRIALS = 32
NATIVE_LOS_MAX_CACHE_STEPS = 128
NATIVE_LOS_PERCEPTION_CALL_SITE = (
    "EntityFilterLineOfSight.matchesEntity->PositionCache.hasLineOfSight"
)
NATIVE_LOS_HIT_CONFIRMATION_CALL_SITE = (
    "HorizontalSelector.RuntimeSelector.tick"
    "->HitDetectionExecutor.LineOfSightProvider"
)

_EXPECTED_KEYS = {
    "type",
    "schema",
    "version",
    "server_version",
    "world",
    "worldgen_provider",
    "worldgen_version",
    "seed",
    "perception_call_site",
    "hit_confirmation_call_site",
    "start",
    "loaded_end",
    "unloaded_end",
    "loaded_end_chunk_x",
    "loaded_end_chunk_z",
    "unloaded_end_chunk_x",
    "unloaded_end_chunk_z",
    "loaded_end_chunk_present",
    "unloaded_end_chunk_present_before",
    "unloaded_end_chunk_present_after",
    "perception_loaded_clear",
    "perception_unloaded_clear",
    "selector_loaded_clear",
    "selector_unloaded_clear",
    "forward_cached_visibility_after_move",
    "inverse_uncached_visibility_after_move",
    "cache_step_seconds",
    "cache_expiry_steps",
}


@dataclass(frozen=True)
class NativeLineOfSightEvidence:
    server_version: str
    world: str
    worldgen_provider: str
    worldgen_version: str
    seed: int
    start: tuple[float, float, float]
    loaded_end: tuple[float, float, float]
    unloaded_end: tuple[float, float, float]
    loaded_end_chunk: tuple[int, int]
    unloaded_end_chunk: tuple[int, int]
    loaded_end_chunk_present: bool
    unloaded_end_chunk_present_before: bool
    unloaded_end_chunk_present_after: bool
    perception_loaded_clear: bool
    perception_unloaded_clear: bool
    selector_loaded_clear: bool
    selector_unloaded_clear: bool
    forward_cached_visibility_after_move: bool
    inverse_uncached_visibility_after_move: bool
    cache_step_seconds: float
    cache_expiry_steps: tuple[int, ...]

    @classmethod
    def from_response(
        cls,
        response: Mapping[str, Any],
    ) -> NativeLineOfSightEvidence:
        if set(response) != _EXPECTED_KEYS:
            raise ValueError("native LOS response fields do not match v1")
        if response["type"] != "los_evidence":
            raise ValueError("bridge returned the wrong native LOS message type")
        if (
            response["schema"] != NATIVE_LOS_EVIDENCE_SCHEMA
            or _integer(response["version"], "version")
            != NATIVE_LOS_EVIDENCE_VERSION
        ):
            raise ValueError("unsupported native LOS evidence contract")
        if response["perception_call_site"] != NATIVE_LOS_PERCEPTION_CALL_SITE:
            raise ValueError("native perception LOS call site changed")
        if (
            response["hit_confirmation_call_site"]
            != NATIVE_LOS_HIT_CONFIRMATION_CALL_SITE
        ):
            raise ValueError("native hit-confirmation LOS call site changed")
        expiry = tuple(
            _integer(value, "cache_expiry_steps")
            for value in _sequence(response["cache_expiry_steps"], "cache_expiry_steps")
        )
        if not 1 <= len(expiry) <= NATIVE_LOS_MAX_CACHE_TRIALS:
            raise ValueError("native LOS cache trial count exceeds capacity")
        if any(not 1 <= value <= NATIVE_LOS_MAX_CACHE_STEPS for value in expiry):
            raise ValueError("native LOS cache expiry lies outside the bounded probe")
        step_seconds = _number(response["cache_step_seconds"], "cache_step_seconds")
        if step_seconds <= 0.0:
            raise ValueError("native LOS cache step must be positive")
        return cls(
            server_version=_text(response["server_version"], "server_version"),
            world=_text(response["world"], "world"),
            worldgen_provider=_text(
                response["worldgen_provider"],
                "worldgen_provider",
            ),
            worldgen_version=_text(
                response["worldgen_version"],
                "worldgen_version",
            ),
            seed=_integer(response["seed"], "seed"),
            start=_point(response["start"], "start"),
            loaded_end=_point(response["loaded_end"], "loaded_end"),
            unloaded_end=_point(response["unloaded_end"], "unloaded_end"),
            loaded_end_chunk=(
                _integer(response["loaded_end_chunk_x"], "loaded_end_chunk_x"),
                _integer(response["loaded_end_chunk_z"], "loaded_end_chunk_z"),
            ),
            unloaded_end_chunk=(
                _integer(
                    response["unloaded_end_chunk_x"],
                    "unloaded_end_chunk_x",
                ),
                _integer(
                    response["unloaded_end_chunk_z"],
                    "unloaded_end_chunk_z",
                ),
            ),
            loaded_end_chunk_present=_boolean(
                response["loaded_end_chunk_present"],
                "loaded_end_chunk_present",
            ),
            unloaded_end_chunk_present_before=_boolean(
                response["unloaded_end_chunk_present_before"],
                "unloaded_end_chunk_present_before",
            ),
            unloaded_end_chunk_present_after=_boolean(
                response["unloaded_end_chunk_present_after"],
                "unloaded_end_chunk_present_after",
            ),
            perception_loaded_clear=_boolean(
                response["perception_loaded_clear"],
                "perception_loaded_clear",
            ),
            perception_unloaded_clear=_boolean(
                response["perception_unloaded_clear"],
                "perception_unloaded_clear",
            ),
            selector_loaded_clear=_boolean(
                response["selector_loaded_clear"],
                "selector_loaded_clear",
            ),
            selector_unloaded_clear=_boolean(
                response["selector_unloaded_clear"],
                "selector_unloaded_clear",
            ),
            forward_cached_visibility_after_move=_boolean(
                response["forward_cached_visibility_after_move"],
                "forward_cached_visibility_after_move",
            ),
            inverse_uncached_visibility_after_move=_boolean(
                response["inverse_uncached_visibility_after_move"],
                "inverse_uncached_visibility_after_move",
            ),
            cache_step_seconds=step_seconds,
            cache_expiry_steps=expiry,
        )

    def semantic_digest(self) -> str:
        payload = {
            "schema": NATIVE_LOS_EVIDENCE_SCHEMA,
            "version": NATIVE_LOS_EVIDENCE_VERSION,
            "server_version": self.server_version,
            "worldgen_provider": self.worldgen_provider,
            "worldgen_version": self.worldgen_version,
            "seed": self.seed,
            "call_sites": {
                "perception": NATIVE_LOS_PERCEPTION_CALL_SITE,
                "hit_confirmation": NATIVE_LOS_HIT_CONFIRMATION_CALL_SITE,
            },
            "start": self.start,
            "loaded_end": self.loaded_end,
            "unloaded_end": self.unloaded_end,
            "loaded_end_chunk": self.loaded_end_chunk,
            "unloaded_end_chunk": self.unloaded_end_chunk,
            "loaded_end_chunk_present": self.loaded_end_chunk_present,
            "unloaded_end_chunk_present_before": (
                self.unloaded_end_chunk_present_before
            ),
            "unloaded_end_chunk_present_after": (
                self.unloaded_end_chunk_present_after
            ),
            "perception_loaded_clear": self.perception_loaded_clear,
            "perception_unloaded_clear": self.perception_unloaded_clear,
            "selector_loaded_clear": self.selector_loaded_clear,
            "selector_unloaded_clear": self.selector_unloaded_clear,
            "forward_cached_visibility_after_move": (
                self.forward_cached_visibility_after_move
            ),
            "inverse_uncached_visibility_after_move": (
                self.inverse_uncached_visibility_after_move
            ),
            "cache_step_seconds": self.cache_step_seconds,
            "cache_expiry_steps": self.cache_expiry_steps,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


def native_line_of_sight_evidence_contract() -> dict[str, object]:
    return {
        "schema": NATIVE_LOS_EVIDENCE_SCHEMA,
        "version": NATIVE_LOS_EVIDENCE_VERSION,
        "message_type": "los_evidence",
        "capture_thread": "native_world_thread_while_paused",
        "perception": {
            "call_site": NATIVE_LOS_PERCEPTION_CALL_SITE,
            "unloaded_chunk": "blocked",
        },
        "hit_confirmation": {
            "call_site": NATIVE_LOS_HIT_CONFIRMATION_CALL_SITE,
            "unloaded_chunk": "non_blocking",
        },
        "cache": {
            "owner": "PositionCache",
            "forward_inverse_independent": True,
            "step_seconds": 0.001,
            "maximum_trials": NATIVE_LOS_MAX_CACHE_TRIALS,
            "maximum_steps": NATIVE_LOS_MAX_CACHE_STEPS,
            "native_ttl_seconds_half_open_source": [0.09, 0.11],
            "jax_policy": "fresh_per_query",
        },
        "state_restoration": "target_transform_and_position_cache_finally",
    }


def native_line_of_sight_evidence_contract_sha256() -> str:
    encoded = json.dumps(
        native_line_of_sight_evidence_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sequence(value: Any, name: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a sequence")
    return value


def _point(value: Any, name: str) -> tuple[float, float, float]:
    sequence = _sequence(value, name)
    if len(sequence) != 3:
        raise ValueError(f"{name} must contain three coordinates")
    return tuple(_number(component, name) for component in sequence)  # type: ignore[return-value]


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be boolean")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be non-empty text")
    return value


__all__ = [
    "NATIVE_LOS_EVIDENCE_SCHEMA",
    "NATIVE_LOS_EVIDENCE_VERSION",
    "NATIVE_LOS_HIT_CONFIRMATION_CALL_SITE",
    "NATIVE_LOS_MAX_CACHE_STEPS",
    "NATIVE_LOS_MAX_CACHE_TRIALS",
    "NATIVE_LOS_PERCEPTION_CALL_SITE",
    "NativeLineOfSightEvidence",
    "native_line_of_sight_evidence_contract",
    "native_line_of_sight_evidence_contract_sha256",
]
