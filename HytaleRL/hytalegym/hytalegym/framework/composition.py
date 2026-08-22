"""Fail-closed composition contracts for replaceable framework providers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from hytalegym.framework.contracts import (
    CompatibilityError,
    SchemaBundle,
)


@dataclass(frozen=True, slots=True)
class ProviderContract:
    """Schemas and named capabilities consumed and produced by one provider."""

    provider_id: str
    consumes: SchemaBundle = SchemaBundle()
    produces: SchemaBundle = SchemaBundle()
    required_capabilities: tuple[str, ...] = ()
    provided_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider_id",
            _name(self.provider_id, "provider ID"),
        )
        if not isinstance(self.consumes, SchemaBundle):
            raise TypeError("consumes must be a SchemaBundle")
        if not isinstance(self.produces, SchemaBundle):
            raise TypeError("produces must be a SchemaBundle")
        object.__setattr__(
            self,
            "required_capabilities",
            _capabilities(
                self.required_capabilities,
                "required capabilities",
            ),
        )
        object.__setattr__(
            self,
            "provided_capabilities",
            _capabilities(
                self.provided_capabilities,
                "provided capabilities",
            ),
        )


@dataclass(frozen=True, slots=True)
class CompositionState:
    """Exact schemas and capabilities available after provider validation."""

    schemas: SchemaBundle
    capabilities: tuple[str, ...]


def compose_provider_contracts(
    providers: Iterable[ProviderContract],
    *,
    initial_schemas: SchemaBundle | None = None,
    initial_capabilities: Iterable[str] = (),
) -> CompositionState:
    """Validate a provider chain without instantiating or executing providers."""

    schemas = SchemaBundle() if initial_schemas is None else initial_schemas
    if not isinstance(schemas, SchemaBundle):
        raise TypeError("initial_schemas must be a SchemaBundle")
    capabilities = set(_capabilities(initial_capabilities, "initial capabilities"))
    for provider in providers:
        if not isinstance(provider, ProviderContract):
            raise TypeError("providers must contain ProviderContract instances")
        schemas.require(provider.consumes, consumer=provider.provider_id)
        missing = set(provider.required_capabilities).difference(capabilities)
        if missing:
            raise CompatibilityError(
                f"{provider.provider_id} lacks capabilities {sorted(missing)}"
            )
        schemas = schemas.merge(
            provider.produces,
            producer=provider.provider_id,
        )
        capabilities.update(provider.provided_capabilities)
    return CompositionState(
        schemas=schemas,
        capabilities=tuple(sorted(capabilities)),
    )


def _capabilities(values: Iterable[str], label: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{label} must be an iterable of names")
    normalized = tuple(_name(value, label) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} cannot contain duplicates")
    return tuple(sorted(normalized))


def _name(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{label} must contain non-empty trimmed strings")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{label} cannot contain control characters")
    return value


__all__ = [
    "CompositionState",
    "ProviderContract",
    "compose_provider_contracts",
]
