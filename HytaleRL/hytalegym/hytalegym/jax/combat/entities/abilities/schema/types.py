"""Array-only PyTrees for logical crowd ability execution."""

from __future__ import annotations

from typing import NamedTuple

import jax

from hytalegym.jax.combat.entities.impacts.schema.types import (
    EntityAreaLaunchInfo,
    EntityProjectileLaunchInfo,
)


Array = jax.Array


class EntityAbilityProgramBank(NamedTuple):
    family_mask: Array
    weapon_id: Array
    guard_entry_cost: Array
    guard_stamina_value: Array
    guard_half_angle_degrees: Array
    guard_entry_delay_seconds: Array
    guard_exit_regen_delay_seconds: Array
    guard_required_resource_id: Array
    guard_required_resource_minimum: Array
    resource_maximum: Array
    resource_initial: Array
    ability_id: Array
    ability_mask: Array
    ability_evidence: Array
    ability_duration_seconds: Array
    ability_cooldown_seconds: Array
    ability_stamina_regen_delay_seconds: Array
    ability_scheduler_prelude_ticks: Array
    ability_resource_cost: Array
    ability_resource_commit_time_seconds: Array
    ability_resource_commit_flags: Array
    ability_resource_minimum: Array
    ability_requirements: Array
    event_mask: Array
    event_time_seconds: Array
    event_kind: Array
    event_f32: Array
    event_i32: Array
    event_flags: Array
    projectile_launch_compatible: Array
    area_launch_compatible: Array
    overflow: Array


class EntityAbilityState(NamedTuple):
    weapon_family: Array
    source_generation: Array
    active_ability_slot: Array
    ability_elapsed_seconds: Array
    ability_scheduler_tick: Array
    ability_scheduler_clock_seconds: Array
    ability_cooldown_seconds: Array
    capability_bits: Array
    failure_bits: Array
    world_failure_bits: Array


class EntityAbilityCommands(NamedTuple):
    requested_slot: Array
    interrupted: Array
    failure_bits: Array
    valid: Array


class EntityAbilityAvailability(NamedTuple):
    source_generation: Array
    available_requirement_bits: Array
    valid: Array
    failure_bits: Array


class EntityAbilityEvents(NamedTuple):
    requested: Array
    source_slot: Array
    source_generation: Array
    weapon_family: Array
    ability_slot: Array
    event_slot: Array
    kind: Array
    f32: Array
    i32: Array
    flags: Array
    overflow: Array
    failure_bits: Array
    world_failure_bits: Array
    valid: Array


class EntityAbilityStepInfo(NamedTuple):
    legal_mask: Array
    requested: Array
    accepted: Array
    interrupted: Array
    event_count: Array
    failure_bits: Array
    world_failure_bits: Array
    valid: Array


class EntityAbilityQueries(NamedTuple):
    candidate_mask: Array
    candidate_generation: Array
    friendly_fire: Array
    query_valid: Array
    failure_bits: Array


class EntityAbilityResolveInfo(NamedTuple):
    damage_events: Array
    status_applications: Array
    status_clears: Array
    healing_targets: Array
    resource_targets: Array
    force_targets: Array
    newly_dead: Array
    failure_bits: Array
    world_failure_bits: Array
    valid: Array


class EntityAbilityLaunchInfo(NamedTuple):
    projectile: EntityProjectileLaunchInfo
    area: EntityAreaLaunchInfo
    failure_bits: Array
    world_failure_bits: Array
    valid: Array


class EntityAbilityRuntimeInfo(NamedTuple):
    scheduler: EntityAbilityStepInfo
    direct: EntityAbilityResolveInfo
    launch: EntityAbilityLaunchInfo
    failure_bits: Array
    world_failure_bits: Array
    valid: Array
