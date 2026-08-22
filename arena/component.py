"""One template + a parameter space = many distinct variants.

:class:`Component` knows nothing about goals or tasks. It pairs a declared
parameter space with a builder, and gives back validation, sampling, grid
sweeps and stable labelling. A goal component and a task template are the same
type with a different builder, so anything you pass in is supported on equal
footing with anything that shipped.

    from arena.component import Component
    from arena.params import Integer, Choice
    from arena.tasks.framework.goals import survive_for

    endurance = Component(
        name="endurance",
        space={"ticks": Integer(100, 900)},
        build=lambda ticks: survive_for(ticks=ticks),
    )

    endurance(ticks=250)                 # one variant
    list(endurance.grid(steps=5))        # five, deterministic
    endurance.sample(rng)                # one at random
    endurance.space_size(steps=5)        # 5

Grid and sample are the point: a component with three axes at four steps each
is 64 distinct variants of one game, all validated, all labelled, none of them
accidentally identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Iterator, Mapping, TypeVar

import random

from arena.params import (
    ParamSpace,
    describe_space,
    grid_params,
    label_params,
    sample_params,
    space_size,
    validate_params,
)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Variant(Generic[T]):
    """One built object plus the exact parameters that produced it."""

    value: T
    params: Mapping[str, Any]
    label: str

    def describe(self) -> dict[str, Any]:
        return {"label": self.label, "params": dict(self.params)}


@dataclass(frozen=True, slots=True)
class Component(Generic[T]):
    """A parameterised builder for anything."""

    name: str
    space: ParamSpace
    build: Callable[..., T]
    description: str = ""
    #: Values used for axes the caller does not supply.
    defaults: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("component name must be a non-empty label")
        if not callable(self.build):
            raise TypeError("build must be callable")
        # Defaults are validated once, here, rather than on every call: a bad
        # default would otherwise only surface for callers who omit that axis.
        validate_params(self.space, self.defaults)

    # -- building --------------------------------------------------------------

    def resolve(self, **params: Any) -> dict[str, Any]:
        """Merge over defaults and validate. Raises on unknown or bad values."""

        merged = dict(self.defaults)
        merged.update(params)
        return validate_params(self.space, merged)

    def variant(self, **params: Any) -> Variant[T]:
        """Build one variant, keeping the parameters that produced it."""

        resolved = self.resolve(**params)
        return Variant(
            value=self.build(**resolved),
            params=resolved,
            label=f"{self.name}[{label_params(resolved)}]",
        )

    def __call__(self, **params: Any) -> T:
        """Build and return the object itself, discarding the wrapper."""

        return self.variant(**params).value

    # -- exploring the space ---------------------------------------------------

    def grid(self, *, steps: int = 3) -> Iterator[Variant[T]]:
        """Every combination of ``steps`` values per axis, deterministically."""

        for params in grid_params(self.space, steps=steps):
            merged = dict(self.defaults)
            merged.update(params)
            resolved = validate_params(self.space, merged)
            yield Variant(
                value=self.build(**resolved),
                params=resolved,
                label=f"{self.name}[{label_params(resolved)}]",
            )

    def sample(self, rng: random.Random | None = None) -> Variant[T]:
        """One uniformly random variant."""

        params = sample_params(self.space, rng)
        return self.variant(**params)

    def samples(self, count: int, rng: random.Random | None = None) -> list[Variant[T]]:
        """``count`` random variants, deduplicated by label.

        Deduplication matters on small or discrete spaces: asking for 100
        variants of a two-valued axis should return 2, not 100 copies
        reported as distinct.
        """

        if count < 1:
            raise ValueError("count must be positive")
        chooser = rng or random.Random()
        seen: dict[str, Variant[T]] = {}
        # Bounded attempts: a small discrete space cannot fill a large request.
        for _ in range(count * 10):
            if len(seen) >= count:
                break
            variant = self.sample(chooser)
            seen.setdefault(variant.label, variant)
        return list(seen.values())

    def space_size(self, *, steps: int = 3) -> int:
        """How many variants a grid sweep would produce."""

        return space_size(self.space, steps=steps)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "space": describe_space(self.space),
            "defaults": dict(self.defaults),
            "grid_size_at_3_steps": self.space_size(steps=3),
        }


class ComponentRegistry(Generic[T]):
    """Named components, so anything passed in is first-class.

    Nothing is registered by default. Whatever you add is what exists.
    """

    def __init__(self, kind: str = "component") -> None:
        self._kind = kind
        self._items: dict[str, Component[T]] = {}

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[Component[T]]:
        return iter(self._items[name] for name in sorted(self._items))

    def add(self, component: Component[T], *, replace: bool = False) -> Component[T]:
        if not isinstance(component, Component):
            raise TypeError(f"expected a Component, got {type(component).__name__}")
        existing = self._items.get(component.name)
        if existing is not None and existing is not component and not replace:
            raise ValueError(
                f"{self._kind} {component.name!r} is already registered; pass "
                "replace=True to override it deliberately"
            )
        self._items[component.name] = component
        return component

    def get(self, name: str) -> Component[T]:
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "<none registered>"
            raise KeyError(f"no {self._kind} {name!r}; registered: {known}") from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def clear(self) -> None:
        self._items.clear()


__all__ = ["Component", "ComponentRegistry", "Variant"]
