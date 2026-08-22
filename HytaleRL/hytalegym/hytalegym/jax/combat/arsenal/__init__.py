"""Data-driven Hytale 0.5.7 weapon, magic, projectile, and area combat."""

# ruff: noqa: F401,F403

from hytalegym.jax.combat.arsenal.programs.ability import (
    ability_legal_mask,
    effective_resource_cost,
    spend_due_ability_resources,
    start_abilities,
    tick_ability_programs,
)
from hytalegym.jax.combat.arsenal.effects.area_plan import (
    StaticAreaPlacementPlan,
    entity_only_area_source_support,
    entity_only_area_support_mask,
    static_area_placement_plan,
    static_area_placement_support_mask,
)
from hytalegym.jax.combat.arsenal.effects.areas import (
    spawn_areas,
    spawn_terminal_areas,
    tick_areas,
    typed_area_transform_overlap,
)
from hytalegym.jax.combat.arsenal.effects.impact_events import (
    native_deployable_attack_mask,
)
from hytalegym.jax.combat.arsenal.catalog.census import (
    ability_executability_census,
    hytale_0_5_7_ability_executability_census,
    hytale_0_5_7_ability_executability_census_sha256,
)
from hytalegym.jax.combat.arsenal.catalog.census_publication import (
    ability_executability_census_publication_path,
    verify_current_ability_executability_census_publication,
    write_current_ability_executability_census_publication,
)
from hytalegym.jax.combat.arsenal.schema.contract import *
from hytalegym.jax.combat.arsenal.factory import (
    arsenal_runtime_capacity,
    empty_ability_loadout,
    empty_arsenal_commands,
    empty_arsenal_state,
    mechanics_rules_for_loadout,
    specialize_ability_loadout,
)
from hytalegym.jax.combat.arsenal.effects.force_plan import (
    applied_force_collision_support_mask,
    build_force_sweep_plan,
    force_sweep_support_mask,
)
from hytalegym.jax.combat.arsenal.item_programs import *
from hytalegym.jax.combat.arsenal.profiles import (
    NativeInteractionBinding,
    NativeProfileBindings,
    CORE_PROFILE_NAMES,
    CROSSBOW_FAMILY_IDS,
    NATIVE_RESOURCE_STAT_IDS,
    NATIVE_NPC_ROLE_PROFILE_NAMES,
    NATIVE_NPC_ROLE_PROFILE_SCHEMA,
    PROFILE_NAMES,
    SHORTBOW_FAMILY_IDS,
    SPEAR_PROFILE_NAMES,
    WORLD_CONDITIONAL_ABILITY_ASSETS,
    ability_loadout_content_sha256,
    hytale_0_5_7_catalog_counts,
    hytale_0_5_7_entity_loadouts,
    hytale_0_5_7_loadouts,
    hytale_0_5_7_native_profile_bindings,
    hytale_0_5_7_native_npc_role_bindings,
    hytale_0_5_7_native_npc_role_loadouts,
    hytale_0_5_7_native_npc_role_manifest,
    hytale_0_5_7_native_npc_role_program_content_sha256,
    hytale_0_5_7_program_content_sha256,
    hytale_0_5_7_runtime_capacity,
    native_ability_requested_charge_time_seconds,
    semantic_id,
)
from hytalegym.jax.combat.arsenal.projectiles.runtime import (
    projectile_entity_first_contact,
    projectile_look_direction,
    projectile_spawn_offset,
    spawn_projectiles,
    tick_projectiles,
)
from hytalegym.jax.combat.arsenal.runtime import (
    arsenal_runtime_config,
    combat_target_selection,
    reset_arsenal_batch,
    rollout_arsenal_batch,
    step_arsenal_batch,
)
from hytalegym.jax.combat.arsenal.schema.spec import (
    combat_arsenal_contract_json,
    combat_arsenal_contract_manifest,
    combat_arsenal_contract_sha256,
)
from hytalegym.jax.combat.arsenal.schema.types import *
from hytalegym.jax.combat.arsenal.schema.validation import validate_ability_loadout
from hytalegym.jax.combat.arsenal.scenarios import with_scenario_resources
from hytalegym.jax.combat.arsenal.environment import *

__all__ = [name for name in globals() if not name.startswith("_")]
