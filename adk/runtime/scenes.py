"""Explicit JAX environment-provider composition.

The gym owns providers and dynamics.  The ADK owns one stable name and one
caller-supplied contract identity for the exact provider bundle selected by an
agent.  Runtime objects are deliberately excluded from ``AgentSpec`` so the
spec remains canonical JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from types import MappingProxyType
from typing import Any, Mapping

from hytalegym.jax.combat.observation.v3.world_tokens import (
    normalize_world_geometry_policy_config,
)
from hytalegym.jax.combat.arsenal.environment import (
    make_arsenal_environment,
    open_flat_arsenal_world_capabilities,
)


SCENE_CONTRACT_SCHEMA = "hytalerl_adk_jax_scene_v4"

# Keep this explicit and compare it to the upstream signature at import time.
# A new Gym seam must either be intentionally exposed here or fail loudly;
# silently dropping a server-derived provider is never a compatible fallback.
JAX_SCENE_PROVIDER_ARGUMENTS = (
    "geometry_provider",
    "target_navigation_provider",
    "world_capability_provider",
    "world_feature_provider",
    "world_token_provider",
    "world_light_token_provider",
    "world_geometry_config",
    "reset_provider",
    "inventory_reset_provider",
    "action_surface_provider",
    "action_surface_executor",
    "action_surface_runtime_initializer",
    "world_runtime_provider",
    "explosion_candidate_provider",
    "recipe_candidate_encoder_params",
    "opponent_ability_provider",
)


def _verify_factory_surface() -> None:
    parameters = inspect.signature(make_arsenal_environment).parameters
    actual = tuple(
        name
        for name in parameters
        if name not in {"params", "config", "maximum_turn_degrees"}
    )
    if actual != JAX_SCENE_PROVIDER_ARGUMENTS:
        raise RuntimeError(
            "Gym Arsenal environment-provider surface changed; update the "
            f"ADK scene port (expected {JAX_SCENE_PROVIDER_ARGUMENTS!r}, "
            f"found {actual!r})"
        )


_verify_factory_surface()


def _manifest_sha256(manifest: Mapping[str, object]) -> str:
    payload = json.dumps(
        dict(manifest),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest().upper()


def _normalize_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("scene contract_sha256 must be a string")
    normalized = value.strip().upper()
    if len(normalized) != 64 or any(
        character not in "0123456789ABCDEF" for character in normalized
    ):
        raise ValueError("scene contract_sha256 must contain 64 hex characters")
    return normalized


@dataclass(frozen=True, slots=True)
class JaxScene:
    """One named, contract-stamped provider bundle for the JAX environment."""

    name: str
    contract_sha256: str
    environment_kwargs: Mapping[str, Any]
    combat_params: Any | None = None
    runtime_config: Any | None = None
    runtime_capacity: Any | None = None
    expected_batch: int | None = None
    expected_loadout: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or self.name != self.name.strip()
            or self.name.count("/") != 1
            or any(not part for part in self.name.split("/", 1))
        ):
            raise ValueError("scene name must be a namespaced registry key")
        object.__setattr__(
            self,
            "contract_sha256",
            _normalize_sha256(self.contract_sha256),
        )
        if not isinstance(self.environment_kwargs, Mapping):
            raise TypeError("environment_kwargs must be a mapping")
        unknown = set(self.environment_kwargs) - set(
            JAX_SCENE_PROVIDER_ARGUMENTS
        )
        if unknown:
            raise ValueError(
                "unknown JAX scene provider arguments: "
                + ", ".join(sorted(unknown))
            )
        normalized_arguments = dict(self.environment_kwargs)
        world_geometry_config = normalized_arguments.get(
            "world_geometry_config"
        )
        if world_geometry_config is not None:
            normalized_arguments["world_geometry_config"] = (
                normalize_world_geometry_policy_config(
                    world_geometry_config
                )
            )
        object.__setattr__(
            self,
            "environment_kwargs",
            MappingProxyType(normalized_arguments),
        )
        if self.expected_batch is not None:
            if (
                isinstance(self.expected_batch, bool)
                or not isinstance(self.expected_batch, int)
                or self.expected_batch < 1
            ):
                raise ValueError("expected_batch must be a positive integer")
        if self.expected_loadout is not None and (
            not isinstance(self.expected_loadout, str)
            or not self.expected_loadout
            or self.expected_loadout != self.expected_loadout.strip()
        ):
            raise ValueError(
                "expected_loadout must be None or a non-empty string"
            )

    def metadata(self) -> dict[str, object]:
        return {
            "schema": SCENE_CONTRACT_SCHEMA,
            "name": self.name,
            "contract_sha256": self.contract_sha256,
            "configured_arguments": sorted(self.environment_kwargs),
            "active_arguments": sorted(
                name
                for name, value in self.environment_kwargs.items()
                if value is not None
            ),
            "combat_params_bound": self.combat_params is not None,
            "runtime_config_bound": self.runtime_config is not None,
            "runtime_capacity_bound": self.runtime_capacity is not None,
            "expected_batch": self.expected_batch,
            "expected_loadout": self.expected_loadout,
        }


def scene_from_region_fixture(
    *,
    name: str,
    contract_sha256: str,
    fixture: Any,
    runtime_config: Any | None = None,
    runtime_capacity: Any | None = None,
    expected_batch: int | None = None,
    expected_loadout: str | None = None,
    world_geometry_config: Any | None = None,
    explosion_candidate_provider: Any | None = None,
    recipe_candidate_encoder_params: Any | None = None,
    opponent_ability_provider: Any | None = None,
) -> JaxScene:
    """Adapt an exact server-derived Region fixture without copying it.

    The loader remains a Gym concern.  This adapter borrows the fixture's
    arrays, providers, runtime initializer, reset distribution, and adjusted
    combat parameters directly.  Loading happens once on the host before JAX
    tracing; every provider is then consumed by the pure compiled environment.
    """

    combat_params = getattr(fixture, "params", None)
    geometry_provider = getattr(fixture, "geometry_provider", None)
    if combat_params is None:
        raise ValueError("Region fixture lacks adjusted combat params")
    if geometry_provider is None:
        raise ValueError("Region fixture lacks its exact geometry provider")

    arguments: dict[str, Any] = {
        "geometry_provider": geometry_provider,
    }
    world_runtime_provider = getattr(
        fixture,
        "world_runtime_provider",
        None,
    )
    if world_runtime_provider is not None:
        # The unified provider owns capability, feature, token, and mutable
        # action-runtime views.  Binding the split providers as well would
        # create two competing sources of truth.
        arguments["world_runtime_provider"] = world_runtime_provider
    else:
        for provider_name in (
            "world_capability_provider",
            "world_feature_provider",
            "world_token_provider",
            "world_light_token_provider",
        ):
            provider = getattr(fixture, provider_name, None)
            if provider is not None:
                arguments[provider_name] = provider

    for provider_name in (
        "target_navigation_provider",
        "reset_provider",
        "inventory_reset_provider",
        "action_surface_provider",
        "action_surface_executor",
        "action_surface_runtime_initializer",
        "explosion_candidate_provider",
    ):
        provider = getattr(fixture, provider_name, None)
        if provider is not None:
            arguments[provider_name] = provider

    optional_bindings = {
        "world_geometry_config": world_geometry_config,
        "explosion_candidate_provider": explosion_candidate_provider,
        "recipe_candidate_encoder_params": recipe_candidate_encoder_params,
        "opponent_ability_provider": opponent_ability_provider,
    }
    arguments.update(
        {
            argument: value
            for argument, value in optional_bindings.items()
            if value is not None
        }
    )

    metadata = getattr(fixture, "metadata", None)
    declared_batch = expected_batch
    if isinstance(metadata, Mapping):
        candidate = metadata.get("environment_count")
        if candidate is not None:
            if (
                isinstance(candidate, bool)
                or not isinstance(candidate, int)
                or candidate < 1
            ):
                raise ValueError(
                    "Region fixture metadata environment_count is invalid"
                )
            if declared_batch is not None and candidate != declared_batch:
                raise ValueError(
                    "Region fixture metadata disagrees with expected batch"
                )
            declared_batch = candidate

    return JaxScene(
        name=name,
        contract_sha256=contract_sha256,
        environment_kwargs=arguments,
        combat_params=combat_params,
        runtime_config=runtime_config,
        runtime_capacity=runtime_capacity,
        expected_batch=declared_batch,
        expected_loadout=expected_loadout,
    )


_FAIL_CLOSED_MANIFEST = {
    "schema": SCENE_CONTRACT_SCHEMA,
    "name": "combat/fail_closed",
    "geometry": "unavailable",
    "world_capabilities": "fail_closed_default",
    "world_features": "unavailable",
    "world_tokens": "unavailable",
    "reset": "fixed_default",
    "action_surface": "fail_closed_default",
}
FAIL_CLOSED_SCENE = JaxScene(
    name=str(_FAIL_CLOSED_MANIFEST["name"]),
    contract_sha256=_manifest_sha256(_FAIL_CLOSED_MANIFEST),
    environment_kwargs={},
)

_OPEN_FLAT_MANIFEST = {
    "schema": SCENE_CONTRACT_SCHEMA,
    "name": "combat/open_flat_control",
    "geometry": "unavailable",
    "world_capabilities": "explicit_open_flat_control",
    "world_features": "unavailable",
    "world_tokens": "unavailable",
    "reset": "fixed_default",
    "action_surface": "fail_closed_default",
}
OPEN_FLAT_CONTROL_SCENE = JaxScene(
    name=str(_OPEN_FLAT_MANIFEST["name"]),
    contract_sha256=_manifest_sha256(_OPEN_FLAT_MANIFEST),
    environment_kwargs={
        "world_capability_provider": open_flat_arsenal_world_capabilities,
    },
)

_BUILTIN_SCENES = {
    FAIL_CLOSED_SCENE.name: FAIL_CLOSED_SCENE,
    OPEN_FLAT_CONTROL_SCENE.name: OPEN_FLAT_CONTROL_SCENE,
}


def resolve_scene(name: str, supplied: JaxScene | None = None) -> JaxScene:
    """Resolve a spec's scene without silently falling back to empty providers."""

    if supplied is not None:
        if not isinstance(supplied, JaxScene):
            raise TypeError("scene must be a JaxScene")
        if supplied.name != name:
            raise ValueError(
                f"AgentSpec scene {name!r} does not match supplied scene "
                f"{supplied.name!r}"
            )
        return supplied
    try:
        return _BUILTIN_SCENES[name]
    except KeyError as error:
        raise KeyError(
            f"unknown JAX scene {name!r}; supply a contract-stamped JaxScene"
        ) from error


__all__ = [
    "FAIL_CLOSED_SCENE",
    "JAX_SCENE_PROVIDER_ARGUMENTS",
    "JaxScene",
    "OPEN_FLAT_CONTROL_SCENE",
    "SCENE_CONTRACT_SCHEMA",
    "resolve_scene",
    "scene_from_region_fixture",
]
