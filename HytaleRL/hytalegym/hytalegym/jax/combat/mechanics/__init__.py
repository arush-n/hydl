"""Versioned, compiled Hytale combat-mechanics primitives."""

from hytalegym.jax.combat.mechanics.schema.contract import *
from hytalegym.jax.combat.mechanics.sources.breathing import tick_breathing as tick_breathing
from hytalegym.jax.combat.mechanics.factory import (
    default_mechanics_rules,
    empty_damage_events,
    empty_defense_commands,
    empty_mechanics_state,
    empty_status_applications,
    with_damage_class_enhancement_profiles,
    with_damage_resistance_profiles,
)
from hytalegym.jax.combat.mechanics.runtime.kernel import (
    advance_applied_motion,
    advance_defense_interactions,
    advance_dodge_motion,
    apply_defense_commands,
    apply_damage_forces,
    apply_resource_delta,
    apply_statuses,
    clear_statuses,
    damp_applied_velocity,
    damp_applied_motion,
    damage_resistance_modifiers,
    guard_resource_available,
    native_npc_null_config_force_velocity,
    project_applied_motion_velocity,
    resolve_applied_motion_velocity,
    resolve_damage_events,
    resolve_dodge_launch_velocity,
    status_modifiers,
    tick_resources,
    tick_statuses,
    translate_applied_motion,
)
from hytalegym.jax.combat.mechanics.sources.role_statuses import (
    default_combat_role_ids,
    initial_role_status_applications,
    role_status_programs_by_semantic_id,
)
from hytalegym.jax.combat.mechanics.schema.types import (
    CombatMechanicsRules,
    CombatMechanicsState,
    DamageEvents,
    DamageResolution,
    DefenseCommandInfo,
    DefenseCommands,
    StatusApplications,
    StatusState,
    StatusTick,
)

__all__ = [
    name
    for name in globals()
    if name.isupper()
    or name.startswith(
        (
            "apply_",
            "advance_",
            "default_",
            "damage_",
            "damp_",
            "empty_",
            "guard_",
            "initial_",
            "native_",
            "project_",
            "resolve_",
            "role_",
            "status_",
            "tick_",
            "translate_",
            "with_",
        )
    )
    or name
    in {
        "CombatMechanicsRules",
        "CombatMechanicsState",
        "DamageEvents",
        "DamageResolution",
        "DefenseCommandInfo",
        "DefenseCommands",
        "StatusApplications",
        "StatusState",
        "StatusTick",
        "default_combat_role_ids",
    }
]
