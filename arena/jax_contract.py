"""Small Arena-facing views of the public JAX Arsenal contract."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping

from hytalegym.jax.combat.mechanics import (
    RESOURCE_AMMO,
    RESOURCE_COUNT,
    RESOURCE_MAGIC_CHARGES,
    RESOURCE_MANA,
    RESOURCE_OXYGEN,
    RESOURCE_SIGNATURE_CHARGES,
    RESOURCE_SIGNATURE_ENERGY,
    RESOURCE_STAMINA,
)
from hytalegym.jax.combat.observation import (
    COMBAT_FLOAT_FEATURES,
    COMBAT_INTEGER_FEATURES,
    ENTITY_FLOAT_FEATURES,
    ENTITY_INTEGER_FEATURES,
    HAZARD_FLOAT_FEATURES,
    HAZARD_INTEGER_FEATURES,
    INTERACTION_FLOAT_FEATURES,
    PROJECTILE_FLOAT_FEATURES,
    PROJECTILE_INTEGER_FEATURES,
    SELF_FLOAT_FEATURES,
    SELF_INTEGER_FEATURES,
    TARGET_FLOAT_FEATURES,
    TARGET_INTEGER_FEATURES,
    TERRAIN_FLOAT_FEATURES,
    TRAVERSAL_FLOAT_FEATURES,
)
from hytalegym.jax.combat.observation.v3.contract import (
    ABILITY_FLOAT_FEATURES,
    ABILITY_INTEGER_FEATURES,
    ACTOR_WORLD_FLOAT_FEATURES,
    ACTOR_WORLD_MASK_FEATURES,
    DEFENSE_FLOAT_FEATURES,
    MOVEMENT_STATE_FEATURES,
    STATUS_FLOAT_FEATURES,
    STATUS_INTEGER_FEATURES,
    WEAPON_INTEGER_FEATURES,
)
from hytalegym.jax.combat.observation.v3.inventory_tokens import (
    INVENTORY_POLICY_FLOAT_FEATURES,
)
from hytalegym.jax.combat.observation.v3.light_policy_tokens import (
    ACTOR_LIGHT_POLICY_FEATURES,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
)
from hytalegym.jax.combat.arsenal.profiles.hytale_0_5_7 import PROFILE_NAMES


def _resource_features() -> tuple[str, ...]:
    positions = {
        RESOURCE_STAMINA: "stamina",
        RESOURCE_MANA: "mana",
        RESOURCE_MAGIC_CHARGES: "magic_charges",
        RESOURCE_SIGNATURE_ENERGY: "signature_energy",
        RESOURCE_SIGNATURE_CHARGES: "signature_charges",
        RESOURCE_AMMO: "ammo",
        RESOURCE_OXYGEN: "oxygen",
    }
    if sorted(positions) != list(range(RESOURCE_COUNT)):
        raise RuntimeError("JAX resource channels are no longer contiguous")
    return tuple(positions[index] for index in range(RESOURCE_COUNT))


RESOURCE_FEATURES = _resource_features()
GROUP_FEATURES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "self_f32": tuple(SELF_FLOAT_FEATURES),
        "self_i32": tuple(SELF_INTEGER_FEATURES),
        "target_f32": tuple(TARGET_FLOAT_FEATURES),
        "target_i32": tuple(TARGET_INTEGER_FEATURES),
        "combat_f32": tuple(COMBAT_FLOAT_FEATURES),
        "combat_i32": tuple(COMBAT_INTEGER_FEATURES),
        "entity_f32": tuple(ENTITY_FLOAT_FEATURES),
        "entity_i32": tuple(ENTITY_INTEGER_FEATURES),
        "projectile_f32": tuple(PROJECTILE_FLOAT_FEATURES),
        "projectile_i32": tuple(PROJECTILE_INTEGER_FEATURES),
        "hazard_f32": tuple(HAZARD_FLOAT_FEATURES),
        "hazard_i32": tuple(HAZARD_INTEGER_FEATURES),
        "terrain_f32": tuple(TERRAIN_FLOAT_FEATURES),
        "traversal_f32": tuple(TRAVERSAL_FLOAT_FEATURES),
        "interaction_f32": tuple(INTERACTION_FLOAT_FEATURES),
        "weapon_i32": tuple(WEAPON_INTEGER_FEATURES),
        "resource_f32": RESOURCE_FEATURES,
        "resource_mask": RESOURCE_FEATURES,
        "defense_f32": tuple(DEFENSE_FLOAT_FEATURES),
        "status_f32": tuple(STATUS_FLOAT_FEATURES),
        "status_i32": tuple(STATUS_INTEGER_FEATURES),
        "ability_f32": tuple(ABILITY_FLOAT_FEATURES),
        "ability_i32": tuple(ABILITY_INTEGER_FEATURES),
        "actor_world_f32": tuple(ACTOR_WORLD_FLOAT_FEATURES),
        "actor_world_mask": tuple(ACTOR_WORLD_MASK_FEATURES),
        "movement_state_f32": tuple(MOVEMENT_STATE_FEATURES),
        "inventory_token_f32": tuple(INVENTORY_POLICY_FLOAT_FEATURES),
        "light_token_f32": tuple(ACTOR_LIGHT_POLICY_FEATURES),
    }
)

_BASE_GROUPS = (
    "self_f32",
    "self_i32",
    "target_f32",
    "target_i32",
    "combat_f32",
    "combat_i32",
    "entity_f32",
    "entity_i32",
    "projectile_f32",
    "projectile_i32",
    "hazard_f32",
    "hazard_i32",
    "terrain_f32",
    "traversal_f32",
    "interaction_f32",
)
_ROOT_GROUPS = (
    "weapon_i32",
    "resource_f32",
    "resource_mask",
    "defense_f32",
    "status_f32",
    "status_i32",
    "ability_f32",
    "ability_i32",
    "actor_world_f32",
    "actor_world_mask",
    "movement_state_f32",
)
READABLE_GROUPS = frozenset((*_BASE_GROUPS, *_ROOT_GROUPS))
_GROUP_SOURCE = MappingProxyType(
    {
        **{name: "base" for name in _BASE_GROUPS},
        **{name: "root" for name in _ROOT_GROUPS},
    }
)


def observation_groups(observation: Any, names) -> dict[str, Any]:
    """Select a static subset of named groups without materializing the rest."""

    selected = {}
    for name in names:
        source = _GROUP_SOURCE.get(name)
        if source is None:
            raise KeyError(f"observation group {name!r} is not always readable")
        holder = observation.base if source == "base" else observation
        selected[name] = getattr(holder, name)
    return selected


def all_observation_groups(observation: Any) -> dict[str, Any]:
    """Select every always-present named group without leaving JAX."""

    return observation_groups(observation, READABLE_GROUPS)


HEAD_SPANS: Mapping[str, tuple[int, int]] = MappingProxyType(
    {
        name: (sum(ARSENAL_POLICY_ACTION_HEAD_SIZES[:index]), size)
        for index, (name, size) in enumerate(
            zip(ARSENAL_POLICY_ACTION_HEAD_NAMES, ARSENAL_POLICY_ACTION_HEAD_SIZES)
        )
    }
)


def list_loadouts() -> tuple[str, ...]:
    return tuple(PROFILE_NAMES)


__all__ = [
    "GROUP_FEATURES",
    "HEAD_SPANS",
    "READABLE_GROUPS",
    "all_observation_groups",
    "list_loadouts",
    "observation_groups",
]
