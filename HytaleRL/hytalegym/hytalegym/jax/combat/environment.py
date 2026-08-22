"""Framework-neutral environment metadata for JAX combat consumers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionComponentSpec:
    """One independently addressable field on a structured action."""

    name: str
    kind: str
    size: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"binary", "continuous", "discrete"}:
            raise ValueError(f"unsupported action component kind: {self.kind}")
        if self.kind == "continuous":
            if self.size is not None:
                raise ValueError("continuous action components have no fixed size")
        elif self.size is None or self.size < 2:
            raise ValueError(
                "binary and discrete action components need at least two values"
            )


@dataclass(frozen=True)
class EnvironmentSpec:
    """Public observation/action identity independent of a learner framework."""

    observation_schema: str
    action_schema: str
    action_components: tuple[ActionComponentSpec, ...]
    observation_size: int | None = None
    action_size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_components", tuple(self.action_components))
        if not self.observation_schema:
            raise ValueError("observation_schema must be non-empty")
        if not self.action_schema:
            raise ValueError("action_schema must be non-empty")
        if not self.action_components:
            raise ValueError("action_components must be non-empty")
        if self.observation_size is not None and self.observation_size < 1:
            raise ValueError("observation_size must be positive when present")
        if self.action_size is not None and self.action_size < 2:
            raise ValueError("action_size must contain at least two actions")

    @property
    def is_dense(self) -> bool:
        """Whether this adapter publishes one flat observation and mask width."""

        return self.observation_size is not None and self.action_size is not None


__all__ = ["ActionComponentSpec", "EnvironmentSpec"]
