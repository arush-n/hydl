"""Lossless, algorithm-neutral composite-action tools.

The environment owns action semantics and joint validation.  This module only
provides a named view of the live categorical heads, explicit ``int32[B, H]``
factor transport, flat-mask handling, neutral JAX distribution operations,
and receipts that keep the environment boundary visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generic, NamedTuple, Protocol, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.policy import (
    action_factor_log_probabilities,
    sample_action_factors,
)


StructuredActionT = TypeVar("StructuredActionT")


class JointValidationUnavailable(RuntimeError):
    """Raised when an environment did not publish joint-action validation."""


class JointActionRejected(RuntimeError):
    """Raised when environment-owned joint validation rejected any row."""


class JointExecutionUnavailable(RuntimeError):
    """Raised when an environment did not publish execution evidence."""


class JointExecutionFailed(RuntimeError):
    """Raised when an environment reported that any action was not executed."""


@dataclass(frozen=True, slots=True)
class ActionHead:
    """One categorical head in the exact order published by the environment."""

    index: int
    name: str
    size: int
    offset: int


class MaskedCompositeSample(NamedTuple):
    """A sampled factor row and the evidence needed to consume it safely.

    ``log_probability`` is the sum of independent per-head categorical log
    probabilities.  ``per_head_available`` says that a head had at least one
    legal choice and finite legal logits.  An unavailable head produces a
    ``-1`` factor and a ``-inf`` row log probability rather than silently
    selecting a fallback action.
    """

    factors: jax.Array
    log_probability: jax.Array
    per_head_available: jax.Array
    per_head_selected_legal: jax.Array


@dataclass(frozen=True, slots=True)
class CompositeActionSpec:
    """Named, lossless view over a live factored action contract.

    ``raw_contract`` retains the exact object that supplied the head layout.
    It is an escape hatch for identities or fields that this general SDK does
    not interpret.
    """

    heads: tuple[ActionHead, ...]
    raw_contract: Any = None

    @classmethod
    def from_contract(
        cls,
        head_names: tuple[str, ...],
        head_sizes: tuple[int, ...],
        *,
        raw_contract: Any = None,
    ) -> CompositeActionSpec:
        """Build from exact names and sizes without hardcoded domain heads."""

        names = tuple(head_names)
        sizes = tuple(head_sizes)
        if not names or len(names) != len(sizes):
            raise ValueError(
                "action head names and sizes must be non-empty and aligned"
            )
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("action head names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("action head names must be unique")
        if any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in sizes
        ):
            raise ValueError("action head sizes must be positive integers")

        offset = 0
        heads = []
        for index, (name, size) in enumerate(zip(names, sizes, strict=True)):
            heads.append(ActionHead(index, name, size, offset))
            offset += size
        return cls(tuple(heads), raw_contract)

    @classmethod
    def from_live_contract(cls, contract: Any) -> CompositeActionSpec:
        """Read the standard head fields while retaining the live object."""

        try:
            names = tuple(contract.action_head_names)
            sizes = tuple(contract.action_head_sizes)
        except AttributeError as error:
            raise TypeError(
                "live action contract must publish action_head_names and "
                "action_head_sizes"
            ) from error
        return cls.from_contract(names, sizes, raw_contract=contract)

    @property
    def head_names(self) -> tuple[str, ...]:
        return tuple(head.name for head in self.heads)

    @property
    def head_sizes(self) -> tuple[int, ...]:
        return tuple(head.size for head in self.heads)

    @property
    def head_count(self) -> int:
        return len(self.heads)

    @property
    def logit_size(self) -> int:
        return sum(self.head_sizes)

    def head(self, name: str) -> ActionHead:
        """Return one live head by name."""

        for head in self.heads:
            if head.name == name:
                return head
        raise KeyError(f"unknown action head: {name!r}")

    def metadata(self) -> dict[str, object]:
        """Return a JSON-safe description without reducing factor transport."""

        return {
            "transport": "explicit_int32_factors",
            "factor_shape": ["batch", self.head_count],
            "logit_size": self.logit_size,
            "joint_validation_owner": "environment",
            "heads": [
                {
                    "index": head.index,
                    "name": head.name,
                    "size": head.size,
                    "offset": head.offset,
                }
                for head in self.heads
            ],
        }

    def factor_layout(self, factors: Any) -> jax.Array:
        """Check the static factor ABI without leaving a compiled JAX path."""

        values = jnp.asarray(factors)
        if values.dtype != jnp.dtype(jnp.int32):
            raise TypeError("action factors must have dtype int32")
        expected = (values.shape[0], self.head_count) if values.ndim == 2 else None
        if values.ndim != 2 or values.shape != expected:
            raise ValueError(
                f"action factors must have shape (B, {self.head_count})"
            )
        return values

    def factors_in_range(self, factors: Any) -> jax.Array:
        """Return a JAX ``bool[B, H]`` range check for explicit factors."""

        values = self.factor_layout(factors)
        sizes = jnp.asarray(self.head_sizes, dtype=jnp.int32)
        return (values >= jnp.int32(0)) & (values < sizes[None, :])

    def validate_factors(
        self,
        factors: Any,
        *,
        expected_batch: int | None = None,
    ) -> jax.Array:
        """Fail closed on dtype, layout, batch, or value range at a host edge."""

        values = self.factor_layout(factors)
        if expected_batch is not None:
            _validate_batch(expected_batch)
            if values.shape[0] != expected_batch:
                raise ValueError(
                    "action factors have the wrong batch: "
                    f"{values.shape[0]} != {expected_batch}"
                )
        valid = np.asarray(jax.device_get(self.factors_in_range(values)))
        if not bool(np.all(valid)):
            row, column = np.argwhere(~valid)[0]
            head = self.heads[int(column)]
            value = int(np.asarray(jax.device_get(values))[row, column])
            raise ValueError(
                f"action factor row {int(row)}, head {head.name!r} is out of "
                f"range: {value} not in [0, {head.size - 1}]"
            )
        return values

    def split_factors(
        self,
        factors: Any,
        *,
        validate_values: bool = True,
    ) -> dict[str, jax.Array]:
        """Split ``int32[B, H]`` factors without changing their values."""

        values = (
            self.validate_factors(factors)
            if validate_values
            else self.factor_layout(factors)
        )
        return {head.name: values[:, head.index] for head in self.heads}

    def mask_layout(self, flat_mask: Any) -> jax.Array:
        """Check the static ``bool[B, sum(head_sizes)]`` mask ABI."""

        mask = jnp.asarray(flat_mask)
        if mask.dtype != jnp.dtype(jnp.bool_):
            raise TypeError("action mask must have dtype bool")
        if mask.ndim != 2 or mask.shape[1] != self.logit_size:
            raise ValueError(
                f"action mask must have shape (B, {self.logit_size})"
            )
        return mask

    def split_mask(self, flat_mask: Any) -> dict[str, jax.Array]:
        """Split a flat mask into exact named head slices, JIT-compatibly."""

        mask = self.mask_layout(flat_mask)
        return {
            head.name: mask[:, head.offset : head.offset + head.size]
            for head in self.heads
        }

    def per_head_mask_available(self, flat_mask: Any) -> jax.Array:
        """Return whether every row has at least one choice in each head."""

        split = self.split_mask(flat_mask)
        return jnp.stack(
            tuple(jnp.any(split[head.name], axis=1) for head in self.heads),
            axis=1,
        )

    def validate_mask(
        self,
        flat_mask: Any,
        *,
        expected_batch: int | None = None,
    ) -> jax.Array:
        """Validate mask layout and per-head availability at a host edge.

        Passing this check proves only independent per-head legality.  It does
        not prove that a combination of selected head values is jointly valid.
        """

        mask = self.mask_layout(flat_mask)
        if expected_batch is not None:
            _validate_batch(expected_batch)
            if mask.shape[0] != expected_batch:
                raise ValueError(
                    "action mask has the wrong batch: "
                    f"{mask.shape[0]} != {expected_batch}"
                )
        available = np.asarray(
            jax.device_get(self.per_head_mask_available(mask))
        )
        if not bool(np.all(available)):
            row, column = np.argwhere(~available)[0]
            raise ValueError(
                "action mask has no legal choice for "
                f"row {int(row)}, head {self.heads[int(column)].name!r}"
            )
        return mask

    def selected_per_head_legal(
        self,
        factors: Any,
        flat_mask: Any,
    ) -> jax.Array:
        """Return marginal legality for each selected factor.

        This result is intentionally named ``per_head``: even an all-true row
        is not an environment-owned joint validation receipt.
        """

        values = self.factor_layout(factors)
        mask = self.mask_layout(flat_mask)
        if values.shape[0] != mask.shape[0]:
            raise ValueError("action factors and mask must have the same batch")
        in_range = self.factors_in_range(values)
        sizes = jnp.asarray(self.head_sizes, dtype=jnp.int32)
        bounded = jnp.clip(values, jnp.int32(0), sizes[None, :] - 1)
        offsets = jnp.asarray(
            tuple(head.offset for head in self.heads),
            dtype=jnp.int32,
        )
        selected = jnp.take_along_axis(
            mask,
            bounded + offsets[None, :],
            axis=1,
        )
        return in_range & selected

    def require_per_head_legal(
        self,
        factors: Any,
        flat_mask: Any,
    ) -> jax.Array:
        """Fail closed on marginal legality without claiming joint validity."""

        values = self.validate_factors(factors)
        mask = self.validate_mask(flat_mask, expected_batch=values.shape[0])
        legal = np.asarray(
            jax.device_get(self.selected_per_head_legal(values, mask))
        )
        if not bool(np.all(legal)):
            row, column = np.argwhere(~legal)[0]
            raise ValueError(
                "masked action choice at "
                f"row {int(row)}, head {self.heads[int(column)].name!r}"
            )
        return values

    def sample_masked(
        self,
        key: jax.Array,
        logits: Any,
        flat_mask: Any,
    ) -> MaskedCompositeSample:
        """Sample independent masked heads through ``hytalegym.jax.policy``.

        This method is JIT-compatible and never packs the factors into a
        scalar.  It does not claim joint legality or execution authorization.
        """

        values, mask = self._distribution_layout(logits, flat_mask)
        safe_logits, available = self._safe_masked_logits(values, mask)
        sampled, log_probability = sample_action_factors(
            key,
            safe_logits,
            self.head_sizes,
        )
        factors = jnp.where(
            available,
            sampled,
            jnp.full_like(sampled, -1),
        ).astype(jnp.int32)
        selected = self.selected_per_head_legal(factors, mask)
        row_available = jnp.all(available, axis=1)
        log_probability = jnp.where(
            row_available,
            log_probability,
            jnp.asarray(-jnp.inf, dtype=log_probability.dtype),
        )
        return MaskedCompositeSample(
            factors,
            log_probability,
            available,
            selected,
        )

    def masked_log_probability(
        self,
        logits: Any,
        factors: Any,
        flat_mask: Any,
    ) -> jax.Array:
        """Return the summed independent-head log probability for factors.

        Out-of-range, masked, or unavailable rows return ``-inf``.  The
        operation is JIT-compatible and uses the neutral Gym distribution
        primitive rather than a training-algorithm implementation.
        """

        values, mask = self._distribution_layout(logits, flat_mask)
        selected = self.factor_layout(factors)
        if selected.shape[0] != values.shape[0]:
            raise ValueError("action factors and logits must have the same batch")
        safe_logits, available = self._safe_masked_logits(values, mask)
        sizes = jnp.asarray(self.head_sizes, dtype=jnp.int32)
        bounded = jnp.clip(selected, jnp.int32(0), sizes[None, :] - 1)
        log_probability = action_factor_log_probabilities(
            safe_logits,
            bounded,
            self.head_sizes,
        )
        selected_legal = self.selected_per_head_legal(selected, mask)
        valid = jnp.all(available & selected_legal, axis=1)
        return jnp.where(
            valid,
            log_probability,
            jnp.asarray(-jnp.inf, dtype=log_probability.dtype),
        )

    def _distribution_layout(
        self,
        logits: Any,
        flat_mask: Any,
    ) -> tuple[jax.Array, jax.Array]:
        values = jnp.asarray(logits)
        if not jnp.issubdtype(values.dtype, jnp.floating):
            raise TypeError("action logits must have a floating dtype")
        if values.ndim != 2 or values.shape[1] != self.logit_size:
            raise ValueError(
                f"action logits must have shape (B, {self.logit_size})"
            )
        mask = self.mask_layout(flat_mask)
        if values.shape != mask.shape:
            raise ValueError("action logits and mask must have identical shapes")
        return values, mask

    def _safe_masked_logits(
        self,
        logits: jax.Array,
        mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        masked_heads = []
        available_heads = []
        for head in self.heads:
            head_logits = logits[:, head.offset : head.offset + head.size]
            head_mask = mask[:, head.offset : head.offset + head.size]
            legal_logits_finite = jnp.all(
                jnp.where(head_mask, jnp.isfinite(head_logits), True),
                axis=1,
            )
            available = jnp.any(head_mask, axis=1) & legal_logits_finite
            masked = jnp.where(head_mask, head_logits, -jnp.inf)
            fallback = jnp.full_like(masked, -jnp.inf).at[:, 0].set(0.0)
            masked_heads.append(
                jnp.where(available[:, None], masked, fallback)
            )
            available_heads.append(available)
        return (
            jnp.concatenate(tuple(masked_heads), axis=1),
            jnp.stack(tuple(available_heads), axis=1),
        )


@dataclass(frozen=True, slots=True)
class StructuredActionCodec(Generic[StructuredActionT]):
    """Direct adapter around an environment's exact structured action codec.

    ``decode`` returns the upstream structured object itself; no ADK projection
    or field subset is introduced.  ``raw_codec`` retains the module, object,
    or descriptor that owns the callbacks for environment-specific access.
    """

    spec: CompositeActionSpec
    encoder: Callable[[StructuredActionT], Any]
    decoder: Callable[[jax.Array], StructuredActionT]
    raw_codec: Any = None

    def encode(self, structured: StructuredActionT) -> jax.Array:
        """Encode directly to explicit factors, checking the static ABI."""

        return self.spec.factor_layout(self.encoder(structured))

    def decode(self, factors: Any) -> StructuredActionT:
        """Return the exact upstream structured action object."""

        return self.decoder(self.spec.factor_layout(factors))

    def require_lossless_round_trip(self, factors: Any) -> StructuredActionT:
        """Verify exact factor preservation at a host-side contract edge."""

        original = self.spec.validate_factors(factors)
        structured = self.decode(original)
        recovered = self.spec.validate_factors(
            self.encode(structured),
            expected_batch=original.shape[0],
        )
        if not np.array_equal(
            np.asarray(jax.device_get(original)),
            np.asarray(jax.device_get(recovered)),
        ):
            raise ValueError(
                "structured action codec did not preserve every factor"
            )
        return structured


class JointActionCapability(str, Enum):
    """Environment-owned evidence available for one composite action."""

    UNAVAILABLE = "unavailable"
    JOINT_VALIDATION = "joint_validation"
    JOINT_EXECUTION = "joint_execution"


@dataclass(frozen=True, slots=True)
class EnvironmentActionReceipt:
    """Lossless evidence returned by an environment action boundary.

    Per-head masks are intentionally absent from the capability levels: they
    are marginals, not joint validation.  ``raw_receipt`` retains all backend
    evidence that the common SDK does not interpret.
    """

    factors: jax.Array
    capability: JointActionCapability
    joint_valid: Any | None = None
    executed: Any | None = None
    reason: str | None = None
    raw_receipt: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.capability, JointActionCapability):
            raise TypeError("capability must be a JointActionCapability")
        if self.reason is not None and not isinstance(self.reason, str):
            raise TypeError("receipt reason must be a string or None")
        if self.capability is JointActionCapability.UNAVAILABLE:
            if self.joint_valid is not None or self.executed is not None:
                raise ValueError(
                    "unavailable joint capability cannot publish validation "
                    "or execution results"
                )
        elif self.joint_valid is None:
            raise ValueError("joint capability must publish joint_valid")
        if (
            self.capability is JointActionCapability.JOINT_VALIDATION
            and self.executed is not None
        ):
            raise ValueError(
                "validation-only capability cannot publish execution results"
            )
        if (
            self.capability is JointActionCapability.JOINT_EXECUTION
            and self.executed is None
        ):
            raise ValueError("joint execution capability must publish executed")

    @classmethod
    def unavailable(
        cls,
        factors: Any,
        *,
        reason: str | None = None,
        raw_receipt: Any = None,
    ) -> EnvironmentActionReceipt:
        return cls(
            jnp.asarray(factors),
            JointActionCapability.UNAVAILABLE,
            reason=reason,
            raw_receipt=raw_receipt,
        )

    @classmethod
    def validated(
        cls,
        factors: Any,
        joint_valid: Any,
        *,
        reason: str | None = None,
        raw_receipt: Any = None,
    ) -> EnvironmentActionReceipt:
        return cls(
            jnp.asarray(factors),
            JointActionCapability.JOINT_VALIDATION,
            joint_valid=joint_valid,
            reason=reason,
            raw_receipt=raw_receipt,
        )

    @classmethod
    def execution(
        cls,
        factors: Any,
        joint_valid: Any,
        executed: Any,
        *,
        reason: str | None = None,
        raw_receipt: Any = None,
    ) -> EnvironmentActionReceipt:
        return cls(
            jnp.asarray(factors),
            JointActionCapability.JOINT_EXECUTION,
            joint_valid=joint_valid,
            executed=executed,
            reason=reason,
            raw_receipt=raw_receipt,
        )

    def require_joint_validated(
        self,
        spec: CompositeActionSpec,
    ) -> jax.Array:
        """Return factors only when environment-owned validation accepted all."""

        factors = spec.validate_factors(self.factors)
        if self.capability is JointActionCapability.UNAVAILABLE:
            detail = f": {self.reason}" if self.reason else ""
            raise JointValidationUnavailable(
                "environment joint-action validation is unavailable" + detail
            )
        valid = _receipt_batch_bool(
            self.joint_valid,
            factors.shape[0],
            "joint_valid",
        )
        host_valid = np.asarray(jax.device_get(valid))
        if not bool(np.all(host_valid)):
            rows = np.flatnonzero(~host_valid).tolist()
            detail = f": {self.reason}" if self.reason else ""
            raise JointActionRejected(
                f"environment rejected composite action rows {rows}" + detail
            )
        return factors

    def require_executed(self, spec: CompositeActionSpec) -> jax.Array:
        """Return factors only when the environment also confirms execution."""

        factors = self.require_joint_validated(spec)
        if self.capability is not JointActionCapability.JOINT_EXECUTION:
            detail = f": {self.reason}" if self.reason else ""
            raise JointExecutionUnavailable(
                "environment action execution evidence is unavailable" + detail
            )
        executed = _receipt_batch_bool(
            self.executed,
            factors.shape[0],
            "executed",
        )
        host_executed = np.asarray(jax.device_get(executed))
        if not bool(np.all(host_executed)):
            rows = np.flatnonzero(~host_executed).tolist()
            detail = f": {self.reason}" if self.reason else ""
            raise JointExecutionFailed(
                f"environment did not execute composite action rows {rows}"
                + detail
            )
        return factors


class EnvironmentActionBoundary(Protocol):
    """Protocol for an environment-owned validation/execution submission."""

    @property
    def joint_action_capability(self) -> JointActionCapability:
        """Describe the strongest receipt the environment can produce."""

    def submit_composite_action(
        self,
        factors: jax.Array,
    ) -> EnvironmentActionReceipt:
        """Validate or execute explicit factors and return raw-backed evidence."""


def _validate_batch(batch: int) -> None:
    if isinstance(batch, bool) or not isinstance(batch, int):
        raise TypeError("expected batch must be an integer")
    if batch < 1:
        raise ValueError("expected batch must be positive")


def _receipt_batch_bool(value: Any, batch: int, name: str) -> jax.Array:
    result = jnp.asarray(value)
    if result.dtype != jnp.dtype(jnp.bool_):
        raise TypeError(f"receipt {name} must have dtype bool")
    if result.shape != (batch,):
        raise ValueError(f"receipt {name} must have shape ({batch},)")
    return result


__all__ = [
    "ActionHead",
    "CompositeActionSpec",
    "EnvironmentActionBoundary",
    "EnvironmentActionReceipt",
    "JointActionCapability",
    "JointActionRejected",
    "JointExecutionFailed",
    "JointExecutionUnavailable",
    "JointValidationUnavailable",
    "MaskedCompositeSample",
    "StructuredActionCodec",
]
