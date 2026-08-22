"""Backend-neutral episode-boundary signals.

``done`` says that an episode boundary occurred.  It does not say why.  Some
backends additionally publish Gymnasium's separate ``terminated`` and
``truncated`` signals, while the current structured JAX Arsenal port returns
only their already-collapsed disjunction.

The optional fields below make that information boundary explicit.  ``None``
means the producer did not publish the split; it never means ``False``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp


class EpisodeBoundary(NamedTuple):
    """One episode-boundary signal with an optional, lossless cause split.

    ``done`` is always available.  ``terminated`` and ``truncated`` are either
    both available or both ``None``.  Availability is schema-level and thus a
    Python boolean, so checking it adds no device synchronization or traced
    branch to a compiled collector.
    """

    done: Any
    terminated: Any | None
    truncated: Any | None

    @classmethod
    def from_done(cls, done: Any) -> "EpisodeBoundary":
        """Preserve a collapsed boundary without inventing its cause."""

        return cls(
            done=_boolean_signal(done, "done"),
            terminated=None,
            truncated=None,
        )

    @classmethod
    def from_split(
        cls,
        terminated: Any,
        truncated: Any,
    ) -> "EpisodeBoundary":
        """Build a boundary when the producer publishes both causes."""

        terminated_value = _boolean_signal(terminated, "terminated")
        truncated_value = _boolean_signal(truncated, "truncated")
        if terminated_value.shape != truncated_value.shape:
            raise ValueError(
                "terminated and truncated must have identical shapes"
            )
        return cls(
            done=terminated_value | truncated_value,
            terminated=terminated_value,
            truncated=truncated_value,
        )

    @property
    def split_known(self) -> bool:
        """Whether separate termination and truncation values are available."""

        terminated_known = self.terminated is not None
        truncated_known = self.truncated is not None
        if terminated_known != truncated_known:
            raise RuntimeError(
                "episode boundary is malformed: terminated and truncated "
                "must be available together"
            )
        return terminated_known

    @property
    def cause_known(self) -> bool:
        """Readable alias for :attr:`split_known`."""

        return self.split_known

    def require_split(self) -> tuple[Any, Any]:
        """Return both cause signals, failing closed when they are unknown."""

        if not self.split_known:
            raise RuntimeError(
                "terminated/truncated split is unavailable; this producer "
                "published only done"
            )
        return self.terminated, self.truncated

    def canonical(self) -> "EpisodeBoundary":
        """Validate signal schema and derive ``done`` from a known split."""

        if self.split_known:
            return EpisodeBoundary.from_split(
                self.terminated,
                self.truncated,
            )
        return EpisodeBoundary.from_done(self.done)


def _boolean_signal(value: Any, name: str):
    signal = jnp.asarray(value)
    if signal.dtype != jnp.dtype(jnp.bool_):
        raise TypeError(f"{name} must have dtype bool")
    return signal


__all__ = ["EpisodeBoundary"]
