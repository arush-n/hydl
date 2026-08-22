"""Parameter spaces: how one template becomes hundreds of distinct variants.

Nothing here knows what a goal or a task is. A :class:`ParamSpace` is a mapping
of name -> domain, and the domains validate, sample and enumerate. Anything
that wants to be parameterised declares a space and gets variation for free.

    space = {"ticks": Integer(100, 900), "armed": Boolean()}
    sample_params(space, rng)         # one random point
    grid_params(space, steps=4)       # 4 x 2 = 8 points, deterministic

Domains validate on construction and on use, because a parameter that silently
clamps produces two "different" variants that are secretly identical -- and a
sweep that reports 200 configurations while running 12 is worse than no sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterator, Mapping, Sequence

import math
import random


class Domain:
    """A parameter's set of permitted values."""

    def validate(self, value: Any) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def sample(self, rng: random.Random) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def enumerate(self, steps: int) -> tuple[Any, ...]:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Real(Domain):
    low: float
    high: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError("Real bounds must be finite")
        if self.low > self.high:
            raise ValueError(f"Real({self.low}, {self.high}): low exceeds high")

    def validate(self, value: Any) -> float:
        found = float(value)
        if not self.low <= found <= self.high:
            raise ValueError(
                f"{found} is outside [{self.low}, {self.high}]"
            )
        return found

    def sample(self, rng: random.Random) -> float:
        return rng.uniform(self.low, self.high)

    def enumerate(self, steps: int) -> tuple[float, ...]:
        if steps < 1:
            raise ValueError("steps must be positive")
        if steps == 1 or self.low == self.high:
            return (self.low,)
        width = (self.high - self.low) / (steps - 1)
        return tuple(self.low + width * index for index in range(steps))

    def describe(self) -> dict[str, Any]:
        return {"kind": "real", "low": self.low, "high": self.high}


@dataclass(frozen=True, slots=True)
class Integer(Domain):
    low: int
    high: int

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"Integer({self.low}, {self.high}): low exceeds high")

    def validate(self, value: Any) -> int:
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{value!r} is not an integer")
        found = int(value)
        if not self.low <= found <= self.high:
            raise ValueError(f"{found} is outside [{self.low}, {self.high}]")
        return found

    def sample(self, rng: random.Random) -> int:
        return rng.randint(self.low, self.high)

    def enumerate(self, steps: int) -> tuple[int, ...]:
        if steps < 1:
            raise ValueError("steps must be positive")
        total = self.high - self.low + 1
        if steps >= total:
            return tuple(range(self.low, self.high + 1))
        if steps == 1:
            return (self.low,)
        width = (self.high - self.low) / (steps - 1)
        # dedupe: rounding can collide on a narrow range
        seen: list[int] = []
        for index in range(steps):
            value = int(round(self.low + width * index))
            if value not in seen:
                seen.append(value)
        return tuple(seen)

    def describe(self) -> dict[str, Any]:
        return {"kind": "integer", "low": self.low, "high": self.high}


@dataclass(frozen=True, slots=True)
class Choice(Domain):
    options: tuple[Any, ...]

    def __init__(self, options: Sequence[Any]) -> None:
        values = tuple(options)
        if not values:
            raise ValueError("Choice needs at least one option")
        if len(set(map(repr, values))) != len(values):
            raise ValueError("Choice options must be distinct")
        object.__setattr__(self, "options", values)

    def validate(self, value: Any) -> Any:
        if value not in self.options:
            raise ValueError(f"{value!r} is not one of {list(self.options)}")
        return value

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(self.options)

    def enumerate(self, steps: int) -> tuple[Any, ...]:
        if steps < 1:
            raise ValueError("steps must be positive")
        return self.options[:steps]

    def describe(self) -> dict[str, Any]:
        return {"kind": "choice", "options": list(self.options)}


@dataclass(frozen=True, slots=True)
class Boolean(Domain):
    def validate(self, value: Any) -> bool:
        if not isinstance(value, bool):
            raise ValueError(f"{value!r} is not a bool")
        return value

    def sample(self, rng: random.Random) -> bool:
        return rng.random() < 0.5

    def enumerate(self, steps: int) -> tuple[bool, ...]:
        if steps < 1:
            raise ValueError("steps must be positive")
        return (False, True)[:steps] if steps < 2 else (False, True)

    def describe(self) -> dict[str, Any]:
        return {"kind": "boolean"}


#: name -> domain.
ParamSpace = Mapping[str, Domain]


def validate_params(space: ParamSpace, params: Mapping[str, Any]) -> dict[str, Any]:
    """Check a point against a space, returning the coerced values.

    Unknown names are an error rather than being ignored: a typo'd parameter
    that is silently dropped produces a variant identical to the default while
    reporting itself as distinct.
    """

    unknown = sorted(set(params) - set(space))
    if unknown:
        raise ValueError(
            f"unknown parameter(s): {unknown}; this space declares "
            f"{sorted(space) or ['<none>']}"
        )
    resolved: dict[str, Any] = {}
    problems: list[str] = []
    for name, domain in space.items():
        if name not in params:
            continue
        try:
            resolved[name] = domain.validate(params[name])
        except (TypeError, ValueError) as error:
            problems.append(f"{name}: {error}")
    if problems:
        raise ValueError("invalid parameter(s): " + "; ".join(problems))
    return resolved


def sample_params(
    space: ParamSpace, rng: random.Random | None = None
) -> dict[str, Any]:
    """One random point in the space."""

    chooser = rng or random
    return {name: domain.sample(chooser) for name, domain in space.items()}


def grid_params(space: ParamSpace, *, steps: int = 3) -> Iterator[dict[str, Any]]:
    """Every combination of ``steps`` values per axis, deterministically."""

    if not space:
        yield {}
        return
    names = sorted(space)
    axes = [space[name].enumerate(steps) for name in names]
    for combination in product(*axes):
        yield dict(zip(names, combination))


def space_size(space: ParamSpace, *, steps: int = 3) -> int:
    """How many variants a grid sweep would produce."""

    if not space:
        return 1
    total = 1
    for domain in space.values():
        total *= len(domain.enumerate(steps))
    return total


def describe_space(space: ParamSpace) -> dict[str, Any]:
    return {name: domain.describe() for name, domain in sorted(space.items())}


def label_params(params: Mapping[str, Any]) -> str:
    """A short, stable, collision-free label for one point."""

    if not params:
        return "default"
    return ",".join(f"{name}={params[name]!r}" for name in sorted(params))


__all__ = [
    "Boolean",
    "Choice",
    "Domain",
    "Integer",
    "ParamSpace",
    "Real",
    "describe_space",
    "grid_params",
    "label_params",
    "sample_params",
    "space_size",
    "validate_params",
]
