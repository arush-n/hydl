"""Name -> AgentSpec. Declaration only; registering builds nothing."""

from __future__ import annotations

from adk.core.spec import AgentSpec

_REGISTRY: dict[str, AgentSpec] = {}


def register(spec: AgentSpec) -> AgentSpec:
    existing = _REGISTRY.get(spec.name)
    if existing is not None and existing != spec:
        raise ValueError(
            f"agent name already registered with a different spec: {spec.name!r}"
        )
    _REGISTRY[spec.name] = spec
    return spec


def get(name: str) -> AgentSpec:
    if name not in _REGISTRY:
        raise KeyError(f"unknown agent: {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_agents() -> list[str]:
    return sorted(_REGISTRY)
