"""Data-derived executability census for authored Arsenal abilities."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json

import numpy as np

from hytalegym.jax.combat.arsenal.schema.contract import (
    REQUIRE_CLEAR_FORCE_PATH,
    REQUIRE_CLEAR_PROJECTILE_FLIGHT,
    REQUIRE_ENTITY_ONLY_AREA,
    REQUIRE_STATIC_AREA_PLACEMENT,
)
from hytalegym.jax.combat.arsenal.effects.area_plan import (
    entity_only_area_support_mask,
    static_area_placement_support_mask,
)
from hytalegym.jax.combat.arsenal.effects.force_plan import (
    applied_force_collision_support_mask,
    force_sweep_support_mask,
)
from hytalegym.jax.combat.arsenal.profiles import (
    PROFILE_NAMES,
    hytale_0_5_7_loadouts,
)
from hytalegym.jax.combat.arsenal.schema.types import AbilityLoadout
from hytalegym.jax.combat.types import AGENT_ENTITY

# Exact GeometryProvider-backed projectile contact and StandardPhysics
# response are production inputs now.  This census measures JAX producer
# executability, not live native certification, so that family is no longer
# classified as absent merely because broad bounce traces remain uncertified.
_UNPRODUCED_REQUIREMENT_FAMILIES = np.uint32(0)


def ability_executability_census(
    loadout: AbilityLoadout,
    profile_names: Sequence[str],
) -> dict[str, object]:
    """Classify authored programs against current exact-geometry producers."""

    names = tuple(profile_names)
    if len(names) != loadout.ability_mask.shape[0]:
        raise ValueError("profile_names must match the loadout batch axis")
    if len(set(names)) != len(names):
        raise ValueError("profile_names must be unique")

    authored = np.asarray(loadout.ability_mask[:, AGENT_ENTITY], dtype=np.bool_)
    requirements = np.asarray(
        loadout.ability_requirements[:, AGENT_ENTITY],
        dtype=np.uint32,
    )
    resource_minimum = np.asarray(
        loadout.ability_resource_minimum[:, AGENT_ENTITY],
        dtype=np.float32,
    )
    resource_maximum = np.asarray(
        loadout.resource_maximum[:, AGENT_ENTITY, None, :],
        dtype=np.float32,
    )
    resource_initial = np.asarray(
        loadout.resource_initial[:, AGENT_ENTITY, None, :],
        dtype=np.float32,
    )
    resource_capacity_supported = authored & np.all(
        resource_maximum + np.float32(1.0e-6) >= resource_minimum,
        axis=2,
    )
    reset_resource_ready = authored & np.all(
        resource_initial + np.float32(1.0e-6) >= resource_minimum,
        axis=2,
    )
    scenario_resource_capacity_required = (
        authored & ~resource_capacity_supported
    )
    in_episode_resource_progression_required = (
        authored & resource_capacity_supported & ~reset_resource_ready
    )
    force_supported = np.asarray(
        force_sweep_support_mask(loadout)[:, AGENT_ENTITY],
        dtype=np.bool_,
    )
    force_collision_supported = np.asarray(
        applied_force_collision_support_mask(loadout)[:, AGENT_ENTITY],
        dtype=np.bool_,
    )
    entity_only_area_supported = np.asarray(
        entity_only_area_support_mask(loadout)[:, AGENT_ENTITY],
        dtype=np.bool_,
    )
    static_area_placement_supported = np.asarray(
        static_area_placement_support_mask(loadout)[:, AGENT_ENTITY],
        dtype=np.bool_,
    )
    requires_entity_only_area = (
        requirements & np.uint32(REQUIRE_ENTITY_ONLY_AREA)
    ) != 0
    requires_static_area_placement = (
        requirements & np.uint32(REQUIRE_STATIC_AREA_PLACEMENT)
    ) != 0
    world_free = authored & (requirements == 0)
    family_supported = (
        authored
        & (requirements != 0)
        & ((requirements & _UNPRODUCED_REQUIREMENT_FAMILIES) == 0)
        & (~requires_entity_only_area | entity_only_area_supported)
        & (
            ~requires_static_area_placement
            | static_area_placement_supported
        )
    )
    requires_force = (
        requirements & np.uint32(REQUIRE_CLEAR_FORCE_PATH)
    ) != 0
    conditioned_executable = family_supported & (
        ~requires_force | force_supported | force_collision_supported
    )
    blocked = authored & ~world_free & ~conditioned_executable
    producer_executable = world_free | conditioned_executable
    capacity_executable = producer_executable & resource_capacity_supported
    reset_executable = producer_executable & reset_resource_ready

    without_projectile_contact = (
        _UNPRODUCED_REQUIREMENT_FAMILIES
        | np.uint32(REQUIRE_CLEAR_PROJECTILE_FLIGHT)
    )
    prior_family_supported = (
        authored
        & (requirements != 0)
        & ((requirements & without_projectile_contact) == 0)
        & (~requires_entity_only_area | entity_only_area_supported)
        & (
            ~requires_static_area_placement
            | static_area_placement_supported
        )
    )
    prior_executable = prior_family_supported & (
        ~requires_force | force_supported | force_collision_supported
    )

    profile_counts: dict[str, dict[str, int]] = {}
    for index, name in enumerate(names):
        profile_counts[name] = {
            "authored": int(np.count_nonzero(authored[index])),
            "world_free": int(np.count_nonzero(world_free[index])),
            "conditioned_executable": int(
                np.count_nonzero(conditioned_executable[index])
            ),
            "blocked": int(np.count_nonzero(blocked[index])),
            "resource_capacity_supported": int(
                np.count_nonzero(resource_capacity_supported[index])
            ),
            "scenario_resource_capacity_required": int(
                np.count_nonzero(
                    scenario_resource_capacity_required[index]
                )
            ),
            "reset_resource_ready": int(
                np.count_nonzero(reset_resource_ready[index])
            ),
            "in_episode_resource_progression_required": int(
                np.count_nonzero(
                    in_episode_resource_progression_required[index]
                )
            ),
            "capacity_executable": int(
                np.count_nonzero(capacity_executable[index])
            ),
            "reset_executable": int(
                np.count_nonzero(reset_executable[index])
            ),
        }

    return {
        "classification": (
            "jax_exact_geometry_producer_executability_not_native_certification"
        ),
        "resource_classification": (
            "current_loadout_actor_scenario_capacity_and_initial_state"
        ),
        "profile_count": len(names),
        "authored_abilities": int(np.count_nonzero(authored)),
        "authored_events": int(
            np.count_nonzero(loadout.event_mask[:, AGENT_ENTITY])
        ),
        "world_free": int(np.count_nonzero(world_free)),
        "conditioned_executable": int(
            np.count_nonzero(conditioned_executable)
        ),
        "blocked": int(np.count_nonzero(blocked)),
        "resource_capacity_supported": int(
            np.count_nonzero(resource_capacity_supported)
        ),
        "scenario_resource_capacity_required": int(
            np.count_nonzero(scenario_resource_capacity_required)
        ),
        "reset_resource_ready": int(
            np.count_nonzero(reset_resource_ready)
        ),
        "in_episode_resource_progression_required": int(
            np.count_nonzero(in_episode_resource_progression_required)
        ),
        "capacity_executable": int(np.count_nonzero(capacity_executable)),
        "reset_executable": int(np.count_nonzero(reset_executable)),
        "family_supported_before_force_plan": int(
            np.count_nonzero(family_supported)
        ),
        "force_plan_rejections": int(
            np.count_nonzero(family_supported & requires_force & ~force_supported)
        ),
        "applied_force_collision_fallback": int(
            np.count_nonzero(
                family_supported
                & requires_force
                & ~force_supported
                & force_collision_supported
            )
        ),
        "projectile_contact_marginal_unlock": int(
            np.count_nonzero(conditioned_executable & ~prior_executable)
        ),
        "profile_counts": profile_counts,
    }


def hytale_0_5_7_ability_executability_census() -> dict[str, object]:
    """Return the current data-derived census for every shipped profile."""

    return ability_executability_census(
        hytale_0_5_7_loadouts(PROFILE_NAMES),
        PROFILE_NAMES,
    )


def hytale_0_5_7_ability_executability_census_sha256() -> str:
    """Hash the complete current census, including every profile row."""

    payload = json.dumps(
        hytale_0_5_7_ability_executability_census(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


__all__ = [
    "ability_executability_census",
    "hytale_0_5_7_ability_executability_census",
    "hytale_0_5_7_ability_executability_census_sha256",
]
