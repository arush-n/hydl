"""Array-only types for compiled projectiles and hazards."""

from __future__ import annotations

from typing import NamedTuple

import jax

from hytalegym.jax.combat.observation.v1.schema.types import CombatSceneFeatures
from hytalegym.jax.combat.types import CombatInfo, CombatState


Array = jax.Array


class ProjectileState(NamedTuple):
    position: Array
    velocity: Array
    half_extent: Array
    age_seconds: Array
    despawn_seconds: Array
    authored_lifetime_seconds: Array
    damage: Array
    gravity: Array
    terminal_velocity: Array
    dead_time_seconds: Array
    dead_time_remaining: Array
    velocity_scale: Array
    kind: Array
    owner_entity_id: Array
    flags: Array
    hostile: Array
    active: Array
    impacted: Array
    physics_initialized: Array
    entity_collision_only: Array


class HazardState(NamedTuple):
    center: Array
    half_extent: Array
    age_seconds: Array
    duration_seconds: Array
    damage_per_second: Array
    intensity: Array
    activation: Array
    kind: Array
    owner_entity_id: Array
    flags: Array
    hostile: Array
    active: Array
    entity_overlap_only: Array


class CombatEffectsState(NamedTuple):
    projectiles: ProjectileState
    hazards: HazardState
    failure_bits: Array


class ProjectileSpawn(NamedTuple):
    requested: Array
    position: Array
    velocity: Array
    half_extent: Array
    despawn_seconds: Array
    authored_lifetime_seconds: Array
    damage: Array
    gravity: Array
    terminal_velocity: Array
    dead_time_seconds: Array
    velocity_scale: Array
    kind: Array
    owner_entity_id: Array
    flags: Array
    hostile: Array
    entity_collision_only: Array


class HazardSpawn(NamedTuple):
    requested: Array
    center: Array
    half_extent: Array
    duration_seconds: Array
    damage_per_second: Array
    intensity: Array
    activation: Array
    kind: Array
    owner_entity_id: Array
    flags: Array
    hostile: Array
    entity_overlap_only: Array


class CombatEffectCommands(NamedTuple):
    projectile: ProjectileSpawn
    hazard: HazardSpawn


class CombatEffectsEnvironmentState(NamedTuple):
    combat: CombatState
    effects: CombatEffectsState


class CombatEffectsInfo(NamedTuple):
    projectile_count: Array
    hazard_count: Array
    projectile_spawned: Array
    hazard_spawned: Array
    projectile_hits: Array
    hazard_hits: Array
    projectile_damage: Array
    hazard_damage: Array
    failure_bits: Array
    valid: Array


class CombatEffectsTransition(NamedTuple):
    state: CombatEffectsEnvironmentState
    observation: Array
    scene: CombatSceneFeatures
    reward: Array
    done: Array
    combat_info: CombatInfo
    effects_info: CombatEffectsInfo


class CombatEffectsTrajectory(NamedTuple):
    observation: Array
    scene: CombatSceneFeatures
    reward: Array
    done: Array
    combat_info: CombatInfo
    effects_info: CombatEffectsInfo
