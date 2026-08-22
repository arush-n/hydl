"""Array-only PyTrees for the fixed-capacity combat entity runtime."""

from __future__ import annotations

from typing import NamedTuple

import jax

from hytalegym.jax.combat.mechanics import (
    CombatMechanicsState,
    DamageEvents,
    DamageResolution,
    StatusApplications,
)


Array = jax.Array


class EntityRoster(NamedTuple):
    """Logical entity state; geometry and collision remain externally owned."""

    semantic_id: Array
    generation: Array
    team_id: Array
    position: Array
    velocity: Array
    yaw_degrees: Array
    health: Array
    max_health: Array
    damageable: Array
    intangible: Array
    invulnerable: Array
    dead: Array
    active: Array
    capability_bits: Array
    failure_bits: Array


class EntitySpawnCommands(NamedTuple):
    requested: Array
    semantic_id: Array
    team_id: Array
    position: Array
    velocity: Array
    yaw_degrees: Array
    health: Array
    max_health: Array
    damageable: Array
    intangible: Array
    invulnerable: Array
    resource_initial: Array


class EntityDespawnCommands(NamedTuple):
    requested: Array
    slot: Array
    generation: Array


class EntityLifecycleCommands(NamedTuple):
    spawn: EntitySpawnCommands
    despawn: EntityDespawnCommands


class EntityLifecycleInfo(NamedTuple):
    spawned: Array
    spawn_slot: Array
    spawn_generation: Array
    despawned: Array
    slot_spawned: Array
    slot_despawned: Array
    spawn_assignment: Array
    failure_bits: Array
    valid: Array


class EntitySelectorQueries(NamedTuple):
    """Preselected geometry/matcher candidates plus combat relationship rules."""

    requested: Array
    source_slot: Array
    source_generation: Array
    candidate_mask: Array
    ignore_owner: Array
    friendly_fire: Array
    max_targets: Array


class EntityTargetSelection(NamedTuple):
    target_mask: Array
    target_count: Array
    available_count: Array
    source_slot: Array
    failure_bits: Array
    row_failure_bits: Array
    valid: Array


class EntityDamagePayloads(NamedTuple):
    requested: Array
    amount: Array
    random_percentage: Array
    damage_class: Array
    cause: Array
    knockback_velocity: Array
    force_mode: Array
    air_resistance: Array
    air_resistance_max: Array
    ground_resistance: Array
    ground_resistance_max: Array
    resistance_threshold: Array
    resistance_style: Array
    dampen_y: Array
    on_hit_resource_id: Array
    on_hit_resource_delta: Array


class PackedEntityDamage(NamedTuple):
    events: DamageEvents
    event_count: Array
    failure_bits: Array
    valid: Array


class EntityStatusPayloads(NamedTuple):
    requested: Array
    effect_id: Array
    duration_seconds: Array
    cycle_cooldown_seconds: Array
    damage_per_cycle: Array
    damage_cause: Array
    healing_per_cycle: Array
    resource_id: Array
    resource_delta_per_cycle: Array
    speed_multiplier: Array
    flags: Array
    overlap_mode: Array


class PackedEntityStatuses(NamedTuple):
    applications: StatusApplications
    application_count: Array
    failure_bits: Array
    valid: Array


class EntityCombatState(NamedTuple):
    roster: EntityRoster
    mechanics: CombatMechanicsState


class EntityDamageInfo(NamedTuple):
    resolution: DamageResolution
    event_count: Array
    newly_dead: Array
    failure_bits: Array
    valid: Array
