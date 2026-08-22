"""Host-side identity and stale-result validation for async inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Any

import jax
import numpy as np


class StaleInferenceReason(str, Enum):
    AGENT_MISMATCH = "agent_mismatch"
    SOURCE_TICK_IN_FUTURE = "source_tick_in_future"
    AGENT_DIED = "agent_died"
    GOAL_CHANGED = "goal_changed_incompatibly"
    TARGET_MISSING = "target_no_longer_exists"
    ACTION_MASK_CHANGED = "action_mask_changed"
    ACTION_ILLEGAL = "action_no_longer_legal"
    TOO_OLD = "result_too_old"
    POLICY_INCOMPATIBLE = "policy_version_incompatible"


@dataclass(frozen=True, slots=True)
class InferenceMetadata:
    """The five identity fields required by PLAN section 30.3."""

    agent_id: str
    source_tick: int
    policy_version: str | int
    goal_version: str | int
    action_mask_hash: str

    def __post_init__(self) -> None:
        _validate_agent_id(self.agent_id)
        _validate_tick(self.source_tick, "source_tick")
        _validate_version(self.policy_version, "policy_version")
        _validate_version(self.goal_version, "goal_version")
        _validate_sha256(self.action_mask_hash, "action_mask_hash")


@dataclass(frozen=True, slots=True)
class InferenceValidationContext:
    """Current host state needed to evaluate every PLAN stale-result rule."""

    agent_id: str
    tick: int
    policy_version: str | int
    goal_version: str | int
    action_mask_hash: str
    agent_alive: bool
    action_legal: bool
    target_exists: bool | None = None
    goal_compatible: bool = False
    policy_compatible: bool = False

    def __post_init__(self) -> None:
        _validate_agent_id(self.agent_id)
        _validate_tick(self.tick, "tick")
        _validate_version(self.policy_version, "policy_version")
        _validate_version(self.goal_version, "goal_version")
        _validate_sha256(self.action_mask_hash, "action_mask_hash")
        for name in (
            "agent_alive",
            "action_legal",
            "goal_compatible",
            "policy_compatible",
        ):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be bool")
        if self.target_exists is not None and not isinstance(
            self.target_exists,
            (bool, np.bool_),
        ):
            raise TypeError("target_exists must be bool or None")

    @classmethod
    def capture(
        cls,
        *,
        agent_id: str,
        tick: int,
        policy_version: str | int,
        goal_version: str | int,
        action_mask: Any,
        agent_alive: bool,
        action_legal: bool,
        target_exists: bool | None = None,
        goal_compatible: bool = False,
        policy_compatible: bool = False,
    ) -> "InferenceValidationContext":
        return cls(
            agent_id=agent_id,
            tick=tick,
            policy_version=policy_version,
            goal_version=goal_version,
            action_mask_hash=action_mask_hash(action_mask),
            agent_alive=agent_alive,
            action_legal=action_legal,
            target_exists=target_exists,
            goal_compatible=goal_compatible,
            policy_compatible=policy_compatible,
        )


@dataclass(frozen=True, slots=True)
class InferenceValidation:
    accepted: bool
    reasons: tuple[StaleInferenceReason, ...]
    age_ticks: int
    action_mask_changed: bool

    @property
    def fallback_required(self) -> bool:
        return not self.accepted


def action_mask_hash(action_mask: Any) -> str:
    """Hash one host-side boolean mask with shape and bit order identified."""

    mask = np.asarray(jax.device_get(action_mask))
    if mask.dtype != np.dtype(np.bool_):
        raise TypeError("action_mask must have dtype bool")
    if mask.ndim != 1 or mask.size == 0:
        raise ValueError("action_mask must be one non-empty agent row")
    digest = hashlib.sha256()
    digest.update(b"hytalerl-adk-action-mask-v1\0")
    digest.update(np.asarray(mask.shape, dtype="<i8").tobytes())
    digest.update(np.packbits(mask, bitorder="little").tobytes())
    return digest.hexdigest().upper()


def capture_inference_metadata(
    *,
    agent_id: str,
    source_tick: int,
    policy_version: str | int,
    goal_version: str | int,
    action_mask: Any,
) -> InferenceMetadata:
    return InferenceMetadata(
        agent_id=agent_id,
        source_tick=source_tick,
        policy_version=policy_version,
        goal_version=goal_version,
        action_mask_hash=action_mask_hash(action_mask),
    )


def validate_inference_result(
    metadata: InferenceMetadata,
    current: InferenceValidationContext,
    *,
    maximum_age_ticks: int,
) -> InferenceValidation:
    """Fail closed over the PLAN section 30.3 host-side rejection rules."""

    if not isinstance(metadata, InferenceMetadata):
        raise TypeError("metadata must be InferenceMetadata")
    if not isinstance(current, InferenceValidationContext):
        raise TypeError("current must be InferenceValidationContext")
    _validate_tick(maximum_age_ticks, "maximum_age_ticks")

    reasons: list[StaleInferenceReason] = []
    if metadata.agent_id != current.agent_id:
        reasons.append(StaleInferenceReason.AGENT_MISMATCH)
    age = int(current.tick) - int(metadata.source_tick)
    if age < 0:
        reasons.append(StaleInferenceReason.SOURCE_TICK_IN_FUTURE)
    if not current.agent_alive:
        reasons.append(StaleInferenceReason.AGENT_DIED)
    if metadata.goal_version != current.goal_version and not current.goal_compatible:
        reasons.append(StaleInferenceReason.GOAL_CHANGED)
    if current.target_exists is False:
        reasons.append(StaleInferenceReason.TARGET_MISSING)
    mask_changed = metadata.action_mask_hash != current.action_mask_hash
    if mask_changed:
        reasons.append(StaleInferenceReason.ACTION_MASK_CHANGED)
    if not current.action_legal:
        reasons.append(StaleInferenceReason.ACTION_ILLEGAL)
    if age > maximum_age_ticks:
        reasons.append(StaleInferenceReason.TOO_OLD)
    if (
        metadata.policy_version != current.policy_version
        and not current.policy_compatible
    ):
        reasons.append(StaleInferenceReason.POLICY_INCOMPATIBLE)
    return InferenceValidation(
        accepted=not reasons,
        reasons=tuple(reasons),
        age_ticks=age,
        action_mask_changed=mask_changed,
    )


def _validate_agent_id(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("agent_id must be a non-empty string")


def _validate_tick(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if int(value) < 0:
        raise ValueError(f"{name} must be nonnegative")


def _validate_version(value: str | int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer or non-empty string")
    if isinstance(value, (int, np.integer)):
        if int(value) < 0:
            raise ValueError(f"{name} must be nonnegative")
        return
    if isinstance(value, str) and value.strip():
        return
    raise TypeError(f"{name} must be an integer or non-empty string")


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from error


__all__ = [
    "InferenceMetadata",
    "InferenceValidation",
    "InferenceValidationContext",
    "StaleInferenceReason",
    "action_mask_hash",
    "capture_inference_metadata",
    "validate_inference_result",
]
