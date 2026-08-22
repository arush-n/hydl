"""Immutable, named scalar sets for data-driven weapon archetypes.

The asset catalog keeps values separate from the graph template that consumes
them.  A new weapon can therefore add another named scalar without adding a
field to every archetype record or a weapon-ID branch to the compiler.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping


@dataclass(frozen=True)
class WeaponScalarSet:
    """A canonical collection of finite, named weapon scalars."""

    entries: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        names: list[str] = []
        canonical: list[tuple[str, float]] = []
        for name, raw_value in self.entries:
            if not isinstance(name, str) or not name:
                raise ValueError("weapon scalar names must be non-empty strings")
            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise TypeError("weapon scalar values must be real numbers")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("weapon scalar values must be finite")
            names.append(name)
            canonical.append((name, value))
        if len(set(names)) != len(names):
            raise ValueError("weapon scalar names must be unique")
        if names != sorted(names):
            raise ValueError("weapon scalar entries must use canonical name order")
        if tuple(canonical) != self.entries:
            object.__setattr__(self, "entries", tuple(canonical))

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> WeaponScalarSet:
        """Build a deterministic scalar set from any mapping order."""

        return cls(tuple(sorted(values.items())))

    def value(self, name: str) -> float:
        """Return one required scalar, failing loudly when a template is incomplete."""

        for candidate, value in self.entries:
            if candidate == name:
                return value
        raise KeyError(f"weapon scalar {name!r} is not authored")

    def merged(self, **values: float) -> WeaponScalarSet:
        """Return a new set with explicit additions or replacements."""

        merged = dict(self.entries)
        merged.update(values)
        return type(self).from_mapping(merged)

    def as_dict(self) -> dict[str, float]:
        """Return a mutable copy for reporting and serialization."""

        return dict(self.entries)


__all__ = ["WeaponScalarSet"]
