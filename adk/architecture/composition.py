"""Fail-closed composition and identity for replaceable agent modules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any

from hytalegym.framework import (
    CompositionState,
    ProviderContract,
    SchemaBundle,
    compose_provider_contracts,
)


PLAN_COMPONENT_ROLES = (
    "observation_encoder",
    "belief",
    "manager",
    "actor",
    "world_model",
    "teacher",
    "critic",
    "value_heads",
)


@dataclass(frozen=True, slots=True)
class ComponentBinding:
    """One named runtime component plus reproducible configuration."""

    role: str
    component: Any = field(repr=False, compare=False)
    configuration: Mapping[str, Any] = field(default_factory=dict, compare=False)
    _configuration_json: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        role = _name(self.role, "component role")
        contract = getattr(self.component, "contract", None)
        if not isinstance(contract, ProviderContract):
            raise TypeError(
                f"component for role {role!r} must expose a ProviderContract"
            )
        configuration, encoded = _json_mapping(
            self.configuration,
            "component configuration",
        )
        object.__setattr__(self, "role", role)
        object.__setattr__(
            self,
            "configuration",
            _freeze_json(configuration),
        )
        object.__setattr__(self, "_configuration_json", encoded)

    @property
    def contract(self) -> ProviderContract:
        return self.component.contract

    def manifest(self) -> dict[str, Any]:
        contract = self.contract
        return {
            "role": self.role,
            "provider_id": contract.provider_id,
            "consumes": contract.consumes.to_dict(),
            "produces": contract.produces.to_dict(),
            "required_capabilities": list(contract.required_capabilities),
            "provided_capabilities": list(contract.provided_capabilities),
            "configuration": json.loads(self._configuration_json),
        }


@dataclass(frozen=True, slots=True)
class AgentArchitecture:
    """A validated module graph; it does not implement any PLAN model."""

    name: str
    bindings: tuple[ComponentBinding, ...]
    initial_schemas: SchemaBundle = SchemaBundle()
    initial_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = _name(self.name, "architecture name")
        bindings = tuple(self.bindings)
        if not bindings:
            raise ValueError("architecture must contain at least one component")
        if any(not isinstance(value, ComponentBinding) for value in bindings):
            raise TypeError("bindings must contain ComponentBinding values")
        roles = tuple(value.role for value in bindings)
        if len(set(roles)) != len(roles):
            raise ValueError("architecture component roles must be unique")
        if not isinstance(self.initial_schemas, SchemaBundle):
            raise TypeError("initial_schemas must be a SchemaBundle")
        capabilities = _names(
            self.initial_capabilities,
            "initial capabilities",
        )
        compose_provider_contracts(
            tuple(value.contract for value in bindings),
            initial_schemas=self.initial_schemas,
            initial_capabilities=capabilities,
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "initial_capabilities", capabilities)

    @property
    def composition(self) -> CompositionState:
        """Return the exact schemas and capabilities after validation."""

        return compose_provider_contracts(
            tuple(value.contract for value in self.bindings),
            initial_schemas=self.initial_schemas,
            initial_capabilities=self.initial_capabilities,
        )

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(value.role for value in self.bindings)

    def component(self, role: str) -> Any:
        key = _name(role, "component role")
        for binding in self.bindings:
            if binding.role == key:
                return binding.component
        raise KeyError(f"architecture has no component role {key!r}")

    def require_roles(self, *roles: str) -> None:
        required = {_name(role, "component role") for role in roles}
        missing = sorted(required.difference(self.roles))
        if missing:
            raise ValueError(f"architecture is missing roles {missing}")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "hytalerl_adk_agent_architecture_v1",
            "name": self.name,
            "initial_schemas": self.initial_schemas.to_dict(),
            "initial_capabilities": list(self.initial_capabilities),
            "components": [value.manifest() for value in self.bindings],
            "result": {
                "schemas": self.composition.schemas.to_dict(),
                "capabilities": list(self.composition.capabilities),
            },
        }

    def architecture_sha256(self) -> str:
        payload = json.dumps(
            self.manifest(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def bind_component(
    role: str,
    component: Any,
    *,
    configuration: Mapping[str, Any] | None = None,
) -> ComponentBinding:
    """Convenience constructor with an empty configuration by default."""

    return ComponentBinding(
        role=role,
        component=component,
        configuration={} if configuration is None else configuration,
    )


def compose_architecture(
    name: str,
    bindings: Iterable[ComponentBinding],
    *,
    initial_schemas: SchemaBundle | None = None,
    initial_capabilities: Iterable[str] = (),
) -> AgentArchitecture:
    """Validate and identify one ordered component graph."""

    return AgentArchitecture(
        name=name,
        bindings=tuple(bindings),
        initial_schemas=(
            SchemaBundle() if initial_schemas is None else initial_schemas
        ),
        initial_capabilities=tuple(initial_capabilities),
    )


def _json_mapping(
    value: Mapping[str, Any],
    label: str,
) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be finite JSON data") from error
    restored = json.loads(payload)
    if not isinstance(restored, dict):
        raise TypeError(f"{label} must be a mapping")
    return restored, payload


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _names(values: Iterable[str], label: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{label} must be an iterable of names")
    normalized = tuple(sorted(_name(value, label) for value in values))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} cannot contain duplicates")
    return normalized


def _name(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise TypeError(f"{label} must be a non-empty trimmed string")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{label} cannot contain control characters")
    return value


__all__ = [
    "AgentArchitecture",
    "ComponentBinding",
    "PLAN_COMPONENT_ROLES",
    "bind_component",
    "compose_architecture",
]
