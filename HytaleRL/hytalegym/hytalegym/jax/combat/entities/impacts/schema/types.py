"""Array-only PyTrees for logical crowd projectile and area impacts."""

from __future__ import annotations

from typing import NamedTuple

import jax

from hytalegym.jax.combat.entities.effects import EntityEffectState


Array = jax.Array


class EntityImpactBindings(NamedTuple):
    """Combat-owned metadata parallel to frozen Arsenal projectile slots."""

    projectile_source_generation: Array
    projectile_interaction_kind: Array
    projectile_damage_multiplier: Array
    projectile_knockback_yaw_degrees: Array
    projectile_friendly_fire: Array
    area_source_generation: Array
    area_friendly_fire: Array
    overflow: Array


class EntityImpactQueries(NamedTuple):
    """Already-complete world results; this is not a world representation."""

    projectile_entity_hit_mask: Array
    projectile_entity_hit_fraction: Array
    projectile_candidate_generation: Array
    projectile_world_hit: Array
    projectile_world_hit_fraction: Array
    projectile_explosion_candidate_mask: Array
    projectile_explosion_distance: Array
    projectile_query_valid: Array
    projectile_failure_bits: Array
    area_candidate_mask: Array
    area_candidate_generation: Array
    area_query_valid: Array
    area_failure_bits: Array


class EntityImpactBindingInfo(NamedTuple):
    projectile_bound: Array
    area_bound: Array
    crossbow_yaw_required: Array
    expected_ranged_projectiles: Array
    observed_ranged_projectiles: Array
    overflow: Array
    valid: Array


class EntityProjectileProgramBank(NamedTuple):
    event_mask: Array
    event_f32: Array
    event_i32: Array
    event_flags: Array
    overflow: Array


RangedProjectileProgramBank = EntityProjectileProgramBank


class EntityAreaProgramBank(NamedTuple):
    event_mask: Array
    event_f32: Array
    event_i32: Array
    event_flags: Array
    overflow: Array


class EntityProjectileLaunchCommands(NamedTuple):
    weapon_family: Array
    ability_slot: Array
    requested: Array
    damage_multiplier: Array
    friendly_fire: Array
    failure_bits: Array
    valid: Array


EntityAreaLaunchCommands = EntityProjectileLaunchCommands


class EntityProjectileLaunchWorld(NamedTuple):
    muzzle_position: Array
    muzzle_yaw_degrees: Array
    muzzle_pitch_degrees: Array
    source_generation: Array
    muzzle_valid: Array
    failure_bits: Array


class EntityAreaLaunchWorld(NamedTuple):
    area_center: Array
    source_yaw_degrees: Array
    source_generation: Array
    area_valid: Array
    failure_bits: Array


class EntityProjectileLaunchInfo(NamedTuple):
    projectile_requested: Array
    projectile_spawned: Array
    projectile_slots: Array
    world_failure_bits: Array
    failure_bits: Array
    valid: Array
    binding: EntityImpactBindingInfo


class EntityAreaLaunchInfo(NamedTuple):
    area_requested: Array
    area_spawned: Array
    area_slots: Array
    world_failure_bits: Array
    failure_bits: Array
    valid: Array
    binding: EntityImpactBindingInfo


EntityRangedLaunchWorld = EntityProjectileLaunchWorld
EntityRangedLaunchInfo = EntityProjectileLaunchInfo


class EntityImpactState(NamedTuple):
    effects: EntityEffectState
    capability_bits: Array
    failure_bits: Array
    world_failure_bits: Array


class EntityImpactInfo(NamedTuple):
    projectile_triggers: Array
    projectile_direct_hits: Array
    projectile_explosion_hits: Array
    area_hits: Array
    damage_events: Array
    status_applications: Array
    crossbow_impacts: Array
    damage_applied: Array
    blocked_hits: Array
    invulnerable_hits: Array
    newly_dead: Array
    failure_bits: Array
    world_failure_bits: Array
    valid: Array
