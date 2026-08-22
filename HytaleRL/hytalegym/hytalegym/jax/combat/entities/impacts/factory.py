"""Host factories for fixed-shape logical impact tensors."""

from __future__ import annotations

from functools import lru_cache
import operator

import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.arsenal import (
    ABILITY_CAPACITY,
    AREA_CAPACITY,
    EVENT_AREA,
    EVENT_FLOAT_FEATURES,
    EVENT_INTEGER_FEATURES,
    EVENT_PROJECTILE,
    PROFILE_NAMES,
    PROJECTILE_CAPACITY,
    hytale_0_5_7_loadouts,
)
from hytalegym.jax.combat.entities import ENTITY_CAPACITY
from hytalegym.jax.combat.entities.effects import (
    EntityEffectState,
    empty_entity_effect_state,
    validate_effect_layout,
)
from hytalegym.jax.combat.entities.impacts.schema.contract import (
    AREAS_PER_LAUNCH,
    AREA_PROGRAM_FAMILY_CAPACITY,
    IMPACT_CAPABILITIES,
    PROJECTILE_PROGRAM_FAMILY_CAPACITY,
    RANGED_PROJECTILES_PER_LAUNCH,
)
from hytalegym.jax.combat.entities.impacts.schema.types import (
    EntityAreaLaunchWorld,
    EntityAreaProgramBank,
    EntityImpactBindings,
    EntityImpactQueries,
    EntityImpactState,
    EntityProjectileLaunchCommands,
    EntityProjectileLaunchWorld,
    EntityProjectileProgramBank,
)
from hytalegym.jax.combat.mechanics import CombatMechanicsRules


def entity_impact_state(effects: EntityEffectState) -> EntityImpactState:
    validate_effect_layout(effects)
    batch = effects.combat.roster.active.shape[0]
    zeros = jnp.zeros((batch,), dtype=jnp.uint32)
    return EntityImpactState(
        effects=effects,
        capability_bits=jnp.full((batch,), IMPACT_CAPABILITIES, dtype=jnp.uint32),
        failure_bits=zeros,
        world_failure_bits=zeros,
    )


def empty_entity_impact_state(
    batch_size: int,
    rules: CombatMechanicsRules,
) -> EntityImpactState:
    return entity_impact_state(empty_entity_effect_state(batch_size, rules))


def empty_entity_impact_bindings(
    batch_size: int,
) -> EntityImpactBindings:
    batch = _positive_size(batch_size, "batch_size")
    projectile = (batch, PROJECTILE_CAPACITY)
    area = (batch, AREA_CAPACITY)
    return EntityImpactBindings(
        projectile_source_generation=jnp.zeros(projectile, dtype=jnp.uint32),
        projectile_interaction_kind=jnp.zeros(projectile, dtype=jnp.int32),
        projectile_damage_multiplier=jnp.ones(projectile, dtype=jnp.float32),
        projectile_knockback_yaw_degrees=jnp.zeros(projectile, dtype=jnp.float32),
        projectile_friendly_fire=jnp.ones(projectile, dtype=jnp.bool_),
        area_source_generation=jnp.zeros(area, dtype=jnp.uint32),
        area_friendly_fire=jnp.ones(area, dtype=jnp.bool_),
        overflow=jnp.zeros((batch,), dtype=jnp.bool_),
    )


def empty_entity_impact_queries(
    batch_size: int,
) -> EntityImpactQueries:
    batch = _positive_size(batch_size, "batch_size")
    projectile = (batch, PROJECTILE_CAPACITY)
    projectile_entity = projectile + (ENTITY_CAPACITY,)
    area = (batch, AREA_CAPACITY)
    return EntityImpactQueries(
        projectile_entity_hit_mask=jnp.zeros(projectile_entity, dtype=jnp.bool_),
        projectile_entity_hit_fraction=jnp.zeros(projectile_entity, dtype=jnp.float32),
        projectile_candidate_generation=jnp.zeros(projectile_entity, dtype=jnp.uint32),
        projectile_world_hit=jnp.zeros(projectile, dtype=jnp.bool_),
        projectile_world_hit_fraction=jnp.zeros(projectile, dtype=jnp.float32),
        projectile_explosion_candidate_mask=jnp.zeros(
            projectile_entity, dtype=jnp.bool_
        ),
        projectile_explosion_distance=jnp.zeros(projectile_entity, dtype=jnp.float32),
        projectile_query_valid=jnp.zeros(projectile, dtype=jnp.bool_),
        projectile_failure_bits=jnp.zeros(projectile, dtype=jnp.uint32),
        area_candidate_mask=jnp.zeros(area + (ENTITY_CAPACITY,), dtype=jnp.bool_),
        area_candidate_generation=jnp.zeros(
            area + (ENTITY_CAPACITY,), dtype=jnp.uint32
        ),
        area_query_valid=jnp.zeros(area, dtype=jnp.bool_),
        area_failure_bits=jnp.zeros(area, dtype=jnp.uint32),
    )


@lru_cache(maxsize=1)
def hytale_0_5_7_entity_projectile_programs() -> EntityProjectileProgramBank:
    """Extract projectile payloads from every pinned Arsenal profile."""

    loadout = hytale_0_5_7_loadouts(PROFILE_NAMES)
    shape = (
        PROJECTILE_PROGRAM_FAMILY_CAPACITY,
        ABILITY_CAPACITY,
        RANGED_PROJECTILES_PER_LAUNCH,
    )
    event_mask = np.zeros(shape, dtype=np.bool_)
    event_f32 = np.zeros(shape + (EVENT_FLOAT_FEATURES,), dtype=np.float32)
    event_i32 = np.zeros(shape + (EVENT_INTEGER_FEATURES,), dtype=np.int32)
    event_flags = np.zeros(shape, dtype=np.uint32)
    overflow = np.zeros((PROJECTILE_PROGRAM_FAMILY_CAPACITY,), dtype=np.bool_)
    for row in range(len(PROFILE_NAMES)):
        family = int(np.asarray(loadout.weapon_family[row, 0]))
        projectile = np.asarray(
            loadout.event_mask[row, 0]
            & (loadout.event_kind[row, 0] == EVENT_PROJECTILE)
        )
        for ability in range(ABILITY_CAPACITY):
            slots = np.flatnonzero(projectile[ability])
            if len(slots) > RANGED_PROJECTILES_PER_LAUNCH:
                overflow[family] = True
            for target, source in enumerate(slots[:RANGED_PROJECTILES_PER_LAUNCH]):
                event_mask[family, ability, target] = True
                event_f32[family, ability, target] = np.asarray(
                    loadout.event_f32[row, 0, ability, source]
                )
                event_i32[family, ability, target] = np.asarray(
                    loadout.event_i32[row, 0, ability, source]
                )
                event_flags[family, ability, target] = np.asarray(
                    loadout.event_flags[row, 0, ability, source]
                )
    return EntityProjectileProgramBank(
        event_mask=jnp.asarray(event_mask),
        event_f32=jnp.asarray(event_f32),
        event_i32=jnp.asarray(event_i32),
        event_flags=jnp.asarray(event_flags),
        overflow=jnp.asarray(overflow),
    )


def hytale_0_5_7_ranged_projectile_programs() -> EntityProjectileProgramBank:
    """Compatibility name for the full pinned projectile program bank."""

    return hytale_0_5_7_entity_projectile_programs()


@lru_cache(maxsize=1)
def hytale_0_5_7_entity_area_programs() -> EntityAreaProgramBank:
    """Extract persistent-area payloads from every pinned profile."""

    loadout = hytale_0_5_7_loadouts(PROFILE_NAMES)
    shape = (
        AREA_PROGRAM_FAMILY_CAPACITY,
        ABILITY_CAPACITY,
        AREAS_PER_LAUNCH,
    )
    event_mask = np.zeros(shape, dtype=np.bool_)
    event_f32 = np.zeros(shape + (EVENT_FLOAT_FEATURES,), dtype=np.float32)
    event_i32 = np.zeros(shape + (EVENT_INTEGER_FEATURES,), dtype=np.int32)
    event_flags = np.zeros(shape, dtype=np.uint32)
    overflow = np.zeros((AREA_PROGRAM_FAMILY_CAPACITY,), dtype=np.bool_)
    for row in range(len(PROFILE_NAMES)):
        family = int(np.asarray(loadout.weapon_family[row, 0]))
        areas = np.asarray(
            loadout.event_mask[row, 0] & (loadout.event_kind[row, 0] == EVENT_AREA)
        )
        for ability in range(ABILITY_CAPACITY):
            slots = np.flatnonzero(areas[ability])
            if len(slots) > AREAS_PER_LAUNCH:
                overflow[family] = True
            for target, source in enumerate(slots[:AREAS_PER_LAUNCH]):
                event_mask[family, ability, target] = True
                event_f32[family, ability, target] = np.asarray(
                    loadout.event_f32[row, 0, ability, source]
                )
                event_i32[family, ability, target] = np.asarray(
                    loadout.event_i32[row, 0, ability, source]
                )
                event_flags[family, ability, target] = np.asarray(
                    loadout.event_flags[row, 0, ability, source]
                )
    return EntityAreaProgramBank(
        event_mask=jnp.asarray(event_mask),
        event_f32=jnp.asarray(event_f32),
        event_i32=jnp.asarray(event_i32),
        event_flags=jnp.asarray(event_flags),
        overflow=jnp.asarray(overflow),
    )


def empty_entity_projectile_launch_commands(
    batch_size: int,
) -> EntityProjectileLaunchCommands:
    batch = _positive_size(batch_size, "batch_size")
    entity = (batch, ENTITY_CAPACITY)
    return EntityProjectileLaunchCommands(
        weapon_family=jnp.zeros(entity, dtype=jnp.int32),
        ability_slot=jnp.full(entity, -1, dtype=jnp.int32),
        requested=jnp.zeros(entity, dtype=jnp.bool_),
        damage_multiplier=jnp.ones(entity, dtype=jnp.float32),
        friendly_fire=jnp.ones(entity, dtype=jnp.bool_),
        failure_bits=jnp.zeros((batch,), dtype=jnp.uint32),
        valid=jnp.ones((batch,), dtype=jnp.bool_),
    )


def empty_entity_area_launch_commands(
    batch_size: int,
) -> EntityProjectileLaunchCommands:
    """Create an inactive event-ready persistent-area request."""

    return empty_entity_projectile_launch_commands(batch_size)


def empty_entity_projectile_launch_world(
    batch_size: int,
) -> EntityProjectileLaunchWorld:
    """Create a fail-closed transform handoff for 32 logical sources."""

    batch = _positive_size(batch_size, "batch_size")
    entity = (batch, ENTITY_CAPACITY)
    return EntityProjectileLaunchWorld(
        muzzle_position=jnp.zeros(entity + (3,), dtype=jnp.float32),
        muzzle_yaw_degrees=jnp.zeros(entity, dtype=jnp.float32),
        muzzle_pitch_degrees=jnp.zeros(entity, dtype=jnp.float32),
        source_generation=jnp.zeros(entity, dtype=jnp.uint32),
        muzzle_valid=jnp.zeros(entity, dtype=jnp.bool_),
        failure_bits=jnp.zeros(entity, dtype=jnp.uint32),
    )


def empty_entity_area_launch_world(
    batch_size: int,
) -> EntityAreaLaunchWorld:
    """Create a fail-closed placement handoff for 32 logical sources."""

    batch = _positive_size(batch_size, "batch_size")
    entity = (batch, ENTITY_CAPACITY)
    return EntityAreaLaunchWorld(
        area_center=jnp.zeros(entity + (3,), dtype=jnp.float32),
        source_yaw_degrees=jnp.zeros(entity, dtype=jnp.float32),
        source_generation=jnp.zeros(entity, dtype=jnp.uint32),
        area_valid=jnp.zeros(entity, dtype=jnp.bool_),
        failure_bits=jnp.zeros(entity, dtype=jnp.uint32),
    )


def empty_entity_ranged_launch_world(
    batch_size: int,
) -> EntityProjectileLaunchWorld:
    """Compatibility name for the shared projectile launch handoff."""

    return empty_entity_projectile_launch_world(batch_size)


def _positive_size(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{label} must be an integer") from exc
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result
