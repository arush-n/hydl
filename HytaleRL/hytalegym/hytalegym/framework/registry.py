"""Small setup-time registry with no hardcoded provider names."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Generic, TypeVar


ValueT = TypeVar("ValueT")


class RegistryError(LookupError):
    """A registry key is duplicate, unknown, or added after freezing."""


class ComponentRegistry(Generic[ValueT]):
    """Explicit registry that cannot overwrite entries or mutate after freeze."""

    def __init__(self) -> None:
        self._entries: dict[str, ValueT] = {}
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def register(self, name: str, value: ValueT) -> None:
        key = _name(name)
        if self._frozen:
            raise RegistryError("registry is frozen")
        if key in self._entries:
            raise RegistryError(f"component {key!r} is already registered")
        self._entries[key] = value

    def resolve(self, name: str) -> ValueT:
        key = _name(name)
        try:
            return self._entries[key]
        except KeyError as error:
            raise RegistryError(f"unknown component {key!r}") from error

    def freeze(self) -> None:
        self._frozen = True

    def snapshot(self) -> Mapping[str, ValueT]:
        return MappingProxyType(dict(self._entries))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.names)

    def __len__(self) -> int:
        return len(self._entries)


def _name(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError("component name must be a non-empty trimmed string")
    if any(ord(character) < 32 for character in value):
        raise ValueError("component name cannot contain control characters")
    return value


__all__ = ["ComponentRegistry", "RegistryError"]
