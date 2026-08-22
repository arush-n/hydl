"""Array-only PyTrees for compiled Hytale combat mechanics."""

from __future__ import annotations

from typing import NamedTuple

import jax


Array = jax.Array


class CombatMechanicsRules(NamedTuple):
    """Episode-pinned rules; all entity-varying leaves are ``[B,E,...]``."""

    resource_minimum: Array
    resource_maximum: Array
    resource_regen_amount: Array
    resource_regen_interval_seconds: Array

    cause_inherits: Array
    cause_durability_loss: Array
    cause_stamina_loss: Array
    cause_bypass_resistances: Array
    equipment_damage_resistance_present: Array
    equipment_damage_resistance_inherits: Array
    equipment_damage_resistance_flat: Array
    equipment_damage_resistance_multiplier: Array
    equipment_damage_class_flat: Array
    equipment_damage_class_multiplier: Array
    guard_cause_mask: Array
    guard_damage_modifier: Array
    guard_knockback_modifier: Array
    guard_entry_cost: Array
    guard_stamina_value: Array
    guard_half_angle_degrees: Array
    guard_entry_delay_seconds: Array
    guard_exit_regen_delay_seconds: Array
    guard_required_resource_id: Array
    guard_required_resource_minimum: Array
    guard_entry_cost_tick: Array
    guard_activation_tick: Array
    guard_release_tick: Array

    dodge_execution_profile: Array
    dodge_admission_cost: Array
    dodge_spend_cost: Array
    dodge_launch_tick: Array
    dodge_cost_tick: Array
    dodge_force: Array
    dodge_invulnerability_seconds: Array
    dodge_cooldown_seconds: Array
    dodge_regen_delay_seconds: Array
    dodge_air_resistance: Array
    dodge_air_resistance_max: Array
    dodge_ground_resistance: Array
    dodge_ground_resistance_max: Array
    dodge_resistance_threshold: Array
    server_ticks_per_second: Array


class StatusState(NamedTuple):
    """Fixed-capacity active entity effects."""

    effect_id: Array
    source_entity_id: Array
    remaining_seconds: Array
    cycle_elapsed_seconds: Array
    cycle_cooldown_seconds: Array
    damage_per_cycle: Array
    damage_cause: Array
    healing_per_cycle: Array
    resource_id: Array
    resource_delta_per_cycle: Array
    speed_multiplier: Array
    damage_resistance_present: Array
    damage_resistance_flat: Array
    damage_resistance_multiplier: Array
    flags: Array
    overlap_mode: Array
    active: Array
    has_cycled: Array


class CombatMechanicsState(NamedTuple):
    """Entity-indexed resource, defense, force, and effect state."""

    resources: Array
    resource_regen_clock: Array
    oxygen_regen_remaining_seconds: Array
    breathing_damage_remaining_seconds: Array
    stamina_regen_delay_seconds: Array
    stamina_regen_delay_clock: Array
    stamina_broken: Array
    guard_held: Array
    guard_active: Array
    guard_windup_elapsed_seconds: Array
    guard_operation_tick: Array
    control_immunity: Array
    control_immunity_regen_clock: Array
    applied_velocity: Array
    external_velocity_y: Array
    applied_velocity_can_clear: Array
    applied_air_resistance: Array
    applied_air_resistance_max: Array
    applied_ground_resistance: Array
    applied_ground_resistance_max: Array
    applied_resistance_threshold: Array
    applied_resistance_style: Array
    applied_dampen_y: Array
    dodge_invulnerability_remaining_seconds: Array
    dodge_cooldown_remaining_seconds: Array
    dodge_operation_tick: Array
    dodge_pending_direction: Array
    statuses: StatusState
    failure_bits: Array


class DefenseCommands(NamedTuple):
    """Hold-to-guard and edge-triggered directional dodge commands."""

    guard_held: Array
    dodge_direction: Array
    dodge_corridor_clear: Array


class DefenseCommandInfo(NamedTuple):
    guard_started: Array
    guard_ended: Array
    dodge_requested: Array
    dodge_accepted: Array


class StatusApplications(NamedTuple):
    """At most four status applications per entity and microtick."""

    requested: Array
    effect_id: Array
    source_entity_id: Array
    duration_seconds: Array
    cycle_cooldown_seconds: Array
    damage_per_cycle: Array
    damage_cause: Array
    healing_per_cycle: Array
    resource_id: Array
    resource_delta_per_cycle: Array
    speed_multiplier: Array
    damage_resistance_present: Array
    damage_resistance_flat: Array
    damage_resistance_multiplier: Array
    flags: Array
    overlap_mode: Array


class StatusTick(NamedTuple):
    """Periodic outputs, retaining slots so source and cause stay explicit."""

    requested: Array
    source_entity_id: Array
    target_entity_id: Array
    damage: Array
    damage_cause: Array
    healing: Array
    aggregate_flags: Array
    speed_multiplier: Array


class DamageEvents(NamedTuple):
    """Sequential damage inputs with fixed event capacity."""

    requested: Array
    source_entity_id: Array
    target_entity_id: Array
    amount: Array
    random_percentage: Array
    damage_class: Array
    cause: Array
    stamina_drain_multiplier: Array
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
    on_hit_healing: Array


class DamageResolution(NamedTuple):
    requested: Array
    source_entity_id: Array
    applied_damage: Array
    blocked: Array
    invulnerable: Array
    stamina_spent: Array
    knockback_velocity: Array
    target_entity_id: Array
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
    on_hit_resource_applied: Array
    on_hit_healing: Array
