"""Canonical checkpoint and adapter manifest for the combat arsenal."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from hytalegym.jax.combat.arsenal.schema.contract import *
from hytalegym.jax.combat.arsenal.profiles import (
    PROFILE_NAMES,
    hytale_0_5_7_native_profile_bindings,
    hytale_0_5_7_program_content_sha256,
)
from hytalegym.jax.combat.arsenal.item_programs import (
    ITEM_PROGRAM_CATALOG_SCHEMA,
    ITEM_PROGRAM_CATALOG_VERSION,
    hytale_0_5_7_item_program_catalog,
)
from hytalegym.jax.combat.arsenal.schema.types import AreaState, ProjectileState
from hytalegym.jax.combat.arsenal.programs.scheduling import (
    SCHEDULER_CLOCK_EPSILON_SECONDS,
)
from hytalegym.jax.combat.block_interactions.adapters.world_binding import (
    block_interaction_world_binding_contract_sha256,
)
from hytalegym.jax.combat.mechanics import (
    COMBAT_MECHANICS_SCHEMA,
    COMBAT_MECHANICS_VERSION,
    CONTROL_IMMUNITY_INCREMENT,
    CONTROL_IMMUNITY_MAXIMUM,
    CONTROL_IMMUNITY_REGEN_AMOUNT,
    CONTROL_IMMUNITY_REGEN_INTERVAL_SECONDS,
    DAMAGE_CLASS_CHARGED,
    DAMAGE_CLASS_LIGHT,
    DAMAGE_CLASS_SIGNATURE,
    DAMAGE_CLASS_UNKNOWN,
    DODGE_AIR_RESISTANCE,
    DODGE_AIR_RESISTANCE_MAX,
    DODGE_AUTHORED_ACTION_MASK,
    DODGE_BACK,
    DODGE_FORCE,
    DODGE_FORWARD,
    DODGE_GROUND_RESISTANCE,
    DODGE_GROUND_RESISTANCE_MAX,
    DODGE_INVULNERABILITY_SECONDS,
    DODGE_LEFT,
    DODGE_LAUNCH_TICK,
    DODGE_NONE,
    DODGE_COST_TICK,
    DODGE_REGEN_DELAY_SECONDS,
    DODGE_RESISTANCE_THRESHOLD,
    DODGE_RIGHT,
    DODGE_STAMINA_ADMISSION_COST,
    DODGE_STAMINA_SPEND_COST,
    DODGE_VELOCITY_REMOVAL_SQUARED,
    GUARD_ACTIVATION_TICK,
    GUARD_ENTRY_COST_TICK,
    GUARD_RELEASE_TICK,
    HYTALE_0_5_7_ASSETS_SHA256,
    HYTALE_0_5_7_NATIVE_EVIDENCE_JAR_SHA256,
    RESOURCE_COUNT,
    STATUS_CAPACITY,
)
from hytalegym.jax.combat.inventory import inventory_contract_sha256
from hytalegym.jax.combat.entities.schema.spec import (
    HYTALE_0_5_7_SERVER_JAR_SHA256,
)
from hytalegym.jax.combat.opponents import (
    OPPONENT_MODE_CHASE,
    OPPONENT_MODE_INACTIVE,
    OPPONENT_MODE_RETURN_HOME_REQUIRED,
    OPPONENT_MODE_SEARCH_REQUIRED,
    SEARCH_TIMEOUT_MAX_SECONDS,
    SEARCH_TIMEOUT_MIN_SECONDS,
)
from hytalegym.jax.combat.types import ENTITY_COUNT
from hytalegym.jax.combat.targeting import TARGET_CANDIDATE_CAPACITY
from hytalegym.jax.world import (
    actor_world_state_contract_sha256,
    entity_only_explosion_contract_sha256,
    point_ray_first_contact_contract_sha256,
    projectile_first_contact_contract_sha256,
    swept_volume_clearance_contract_sha256,
)
from hytalegym.rulesets import (
    load_role_initial_statuses,
    native_status_projection_sha256,
    role_initial_statuses_sha256,
)


def combat_arsenal_contract_manifest(
    *,
    entity_count: int = ENTITY_COUNT,
) -> dict[str, Any]:
    """Return the stable host/device program contract."""

    _validate_entity_count(entity_count)
    entity_axis = (
        {
            "host_trees": "parameterized",
            "current_runtime": ("paired_behavior_plus_entity_indexed_opponent_memory"),
            "multi_target_selection": "not_implemented",
            "extra_entity_ai": (
                "memory_and_mode_state_only_no_autonomous_attack_or_motion"
            ),
        }
        if entity_count == ENTITY_COUNT
        else {
            "host_trees": "parameterized",
            "current_runtime": "dynamic_entity_axis_single_locked_target",
            "multi_target_selection": (
                "controller_component_distance_sorted_fixed_k_engage_one"
            ),
            "candidate_capacity": TARGET_CANDIDATE_CAPACITY,
            "candidate_evidence": ("actor_legal_fov_and_perception_los_before_ranking"),
            "engagement_capacity": 1,
            "engagement_lifetime": (
                "persisted_across_decisions_until_identity_invalid"
            ),
            "engagement_retention": ("alive_non_self_opposing_team_finite_distance"),
            "engagement_grounding": (
                "decompiled_marked_entity_support_and_asset_authored_lock_on_target"
            ),
            "search_return_home": (
                "state_transitions_implemented_motion_fail_closed_pending_"
                "world_search_candidates"
            ),
            "actor_observation": "self_plus_one_locked_target",
            "extra_entity_ai": (
                "entity_indexed_last_seen_and_pursuit_clocks_"
                "caller_commands_no_autonomous_attack"
            ),
        }
    )
    role_statuses = load_role_initial_statuses()
    item_program_catalog = hytale_0_5_7_item_program_catalog()
    return {
        "schema": ARSENAL_SCHEMA,
        "version": ARSENAL_VERSION,
        "mechanics": {
            "schema": COMBAT_MECHANICS_SCHEMA,
            "version": COMBAT_MECHANICS_VERSION,
            "resource_regeneration": {
                "initial_phase": "zero_initialized_due",
                "timer_gate": "positive_below_max_negative_above_min",
                "interval_boundary": "strict_remaining_below_zero",
                "maximum_cycles_per_tick": 1,
                "overdue_time": "retained_for_later_ticks",
                "saturation": "clock_pauses_at_matching_bound",
            },
            "guard": {
                "entry_cost_tick": GUARD_ENTRY_COST_TICK,
                "activation_tick": GUARD_ACTIVATION_TICK,
                "release_tick": GUARD_RELEASE_TICK,
                "authored_entry_delay": "ceil_seconds_times_30_ticks",
                "stage_order": "stat_regeneration_then_interaction_chain",
            },
            "dodge": {
                "direction_ids": {
                    "none": DODGE_NONE,
                    "forward": DODGE_FORWARD,
                    "back": DODGE_BACK,
                    "left": DODGE_LEFT,
                    "right": DODGE_RIGHT,
                },
                "authored_action_mask": list(DODGE_AUTHORED_ACTION_MASK),
                "unauthored_branch": (
                    "requested_noop_without_resource_force_or_invulnerability"
                ),
                "stamina_admission_cost": DODGE_STAMINA_ADMISSION_COST,
                "stamina_spend_cost": DODGE_STAMINA_SPEND_COST,
                "launch_tick": DODGE_LAUNCH_TICK,
                "cost_tick": DODGE_COST_TICK,
                "force": DODGE_FORCE,
                "invulnerability_seconds": DODGE_INVULNERABILITY_SECONDS,
                "regen_delay_seconds": DODGE_REGEN_DELAY_SECONDS,
                "air_resistance": DODGE_AIR_RESISTANCE,
                "air_resistance_max": DODGE_AIR_RESISTANCE_MAX,
                "ground_resistance": DODGE_GROUND_RESISTANCE,
                "ground_resistance_max": DODGE_GROUND_RESISTANCE_MAX,
                "resistance_threshold": DODGE_RESISTANCE_THRESHOLD,
                "velocity_removal_squared": DODGE_VELOCITY_REMOVAL_SQUARED,
            },
        },
        "batch_axis": "B",
        "entity_axis": entity_axis,
        "capacities": {
            "entities": entity_count,
            "abilities_per_entity": ABILITY_CAPACITY,
            "cooldown_charges_per_ability": ABILITY_CHARGE_CAPACITY,
            "events_per_ability": EVENT_CAPACITY,
            "learner_observation_abilities": OBSERVATION_CAPACITY,
            "projectiles": PROJECTILE_CAPACITY,
            "areas": AREA_CAPACITY,
            "impact_damage_events_per_phase": IMPACT_DAMAGE_CAPACITY,
            "statuses_per_entity": STATUS_CAPACITY,
            "resources": RESOURCE_COUNT,
        },
        "actor_resource_scenarios": {
            "stock_profile": ("native_entity_stat_defaults_no_implicit_weapon_funding"),
            "unmet_authored_requirement": ("structurally_valid_runtime_illegal"),
            "override_factory": "with_scenario_resources",
            "override_keys": "generic_resource_id",
            "override_values": "scalar_or_broadcastable_B_N",
            "maximum_and_initial": "independently_explicit",
            "native_parity": (
                "caller_applies_equivalent_native_stat_or_effect_modifier"
            ),
        },
        "runtime_specialization": {
            "ability_capacity": "derived_from_selected_loadout",
            "event_layout": "flat_per_entity",
            "event_bank_capacity": ("maximum_total_events_in_selected_profile"),
            "event_window_capacity": ("maximum_events_in_one_selected_ability"),
            "packing_order": "ability_slot_then_authored_event_order",
            "learner_observation_capacity": OBSERVATION_CAPACITY,
            "null_loadout": "bit_exact_certified_melee_step_batch",
        },
        "opponent_controller": {
            "private_not_actor_observation": True,
            "controlled_mask": "derived_from_opposing_team_or_explicit",
            "distance_metric": (
                "per_entity_runtime_component_selector_walk_xz_fly_dive_xyz"
            ),
            "modes": {
                "inactive": OPPONENT_MODE_INACTIVE,
                "chase": OPPONENT_MODE_CHASE,
                "search_required": OPPONENT_MODE_SEARCH_REQUIRED,
                "return_home_required": OPPONENT_MODE_RETURN_HOME_REQUIRED,
            },
            "last_seen_acquisition": (
                "selected_target_from_view_bounded_valid_perception_or_"
                "independent_legal_hearing"
            ),
            "sight_range": "combat_ruleset_target_chase_view_range_3d",
            "hearing": (
                "combat_ruleset_3d_range_los_independent_actor_crouch_suppression"
            ),
            "search_timeout_seconds": [
                SEARCH_TIMEOUT_MIN_SECONDS,
                SEARCH_TIMEOUT_MAX_SECONDS,
            ],
            "search_return_home_motion": (
                "fail_closed_pending_certified_world_candidate_producer"
            ),
            "autonomous_attack_choice": "not_implemented",
        },
        "inventory": {
            "contract_sha256": inventory_contract_sha256(),
            "state_axis": "batch_entity_flat_container_slots",
            "loadout_seed": "equipped_item_to_active_hotbar_slot_zero",
            "ability_and_guard_gate": "active_hotbar_item_matches_weapon_id",
            "armor_container": "distinct_dynamic_stat_catalog_binding_pending",
            "actor_observation": (
                "policy_v8_76_slots_with_independent_container_validity"
            ),
        },
        "interaction_scheduler": {
            "queue_delay_ticks": INTERACTION_QUEUE_DELAY_TICKS,
            "float32_boundary_epsilon_seconds": float(SCHEDULER_CLOCK_EPSILON_SECONDS),
            "root_lifecycle": {
                "internal_pending_handle": (
                    "active_ability_slot_is_reserved_on_legal_request"
                ),
                "authored_root_handle": (
                    "active_ability_root_slot_retains_the_requesting_root_"
                    "when_a_private_child_executes"
                ),
                "public_admission": ("hidden_until_chain_start_queue_has_drained"),
                "runtime_admission": (
                    "queue_admitted_view_plus_native_pending_busy_refusal"
                ),
                "queued_repeat": (
                    "mask_advertised_then_refused_without_rewinding_first_root"
                ),
                "elapsed_clock": ("starts_only_after_the_queued_root_is_admitted"),
                "completion": (
                    "prior_root_clock_reaches_duration_and_every_authored_"
                    "event_selector_and_resource_child_has_received_its_"
                    "terminal_scheduler_sample"
                ),
                "native_projection": ("native_active_root_is_already_queue_admitted"),
                "active_rule_scope": (
                    "single_ability_root_plus_guard_and_dodge_defense_roots"
                ),
                "rule_arbitration": {
                    "ability_interaction_type_field": ("ability_interaction_type"),
                    "guard_interrupting_mask_field": ("guard_interrupting_type_mask"),
                    "active_operation_interrupted_by_fields": (
                        "ability_interrupted_by_type_mask_and_"
                        "ability_interruptible_after_seconds"
                    ),
                    "active_operation_interrupted_by": (
                        "incoming_root_type_cancels_the_active_program_only_"
                        "after_the_authored_operation_boundary"
                    ),
                    "guard_start": (
                        "interrupts_asset_declared_primary_else_standard_input_blocked"
                    ),
                    "ability_start_while_guard_held": (
                        "ordinary_roots_standard_input_blocked_authored_"
                        "guard_input_fork_allowed"
                    ),
                    "guard_input_fork_parent_level": (
                        "implicitly_retained_for_the_fork_control_tick_then_"
                        "controlled_by_guard_held"
                    ),
                    "dodge": (
                        "default_dodge_blocks_only_dodge_and_coexists_with_"
                        "guard_and_ability"
                    ),
                    "pending_queue": "excluded_from_rule_comparison",
                },
                "not_claimed": (
                    "general_non_guard_multi_chain_forks_per_operation_"
                    "rules_tag_bypasses_or_custom_blocking_for_future_assets"
                ),
            },
            "positive_runtime_start_delay_ticks": (POSITIVE_RUNTIME_START_DELAY_TICKS),
            "non_player_selector_start_delay_ticks": (
                NON_PLAYER_SELECTOR_START_DELAY_TICKS
            ),
            "player_selector_client_sync_delay_ticks": (
                PLAYER_SELECTOR_CLIENT_SYNC_DELAY_TICKS
            ),
            "parallel_fork_start_delay_ticks": PARALLEL_FORK_START_DELAY_TICKS,
            "damage_knockback_stage_order": (
                "damage_component_then_next_tick_apply_knockback"
            ),
            "event_clock_count": EVENT_SCHEDULER_CLOCK_COUNT,
            "ability_scheduler_prelude": {
                "field": "ability_scheduler_prelude_ticks",
                "dtype": "int32",
                "semantics": (
                    "asset_resolved_outer_root_boundaries_before_selected_child"
                ),
                "application": "all_event_and_resource_clocks",
            },
            "outer_root_routing": {
                "policy_surface": (
                    "authored_root_slots_only_internal_selector_children_hidden"
                ),
                "crossbow_primary_priority": (
                    "big_arrow_charge_then_loaded_standard_then_reload"
                ),
                "crossbow_primary_cooldown_seconds": 0.3,
                "observable_active_slot": (
                    "internal_child_projected_back_to_authored_root"
                ),
                "native_execution_reconstruction": (
                    "selected_child_retained_from_admission_time_resources"
                ),
                "native_execution_evidence": (
                    "ammo_and_signature_charges_required_fail_closed"
                ),
                "crossbow_reload": (
                    "startup_then_six_inventory_backed_repeat_iterations"
                ),
                "crossbow_reload_interruption": (
                    "repeat_node_interrupted_by_primary_or_secondary_after_"
                    "the_point_eight_second_startup"
                ),
                "shared_native_root_scalar_choices": (
                    "many_authored_policy_slots_compact_to_one_unique_native_"
                    "root_and_retain_the_selected_scalar_on_the_host"
                ),
                "charging_clock_origin": (
                    "first_authored_legal_threshold_already_owned_by_the_"
                    "fixed_outer_root_prelude"
                ),
                "charging_selected_child_delay": (
                    "selected_threshold_minus_first_legal_threshold_"
                    "with_duration_boundaries_ceiled_at_30_tps"
                ),
                "requested_charge_transport": (
                    "explicit_nonnegative_outer_charging_threshold_verbatim_"
                    "negative_one_sentinel_transports_zero_never_child_event_"
                    "or_resource_clock"
                ),
                "outer_root_item_dispatch": (
                    "integer_30_tps_lifecycle_tick_before_selected_child"
                ),
                "outer_root_resource_phases": (
                    "data_driven_per_ability_resource_start_inclusive_end_"
                    "exclusive_tick_intervals"
                ),
                "stamina_regen_delay_set_phases": (
                    "data_driven_per_ability_change_stat_set_start_and_end_"
                    "lifecycle_ticks"
                ),
            },
            "fired_events_per_tick": FIRED_EVENT_CAPACITY,
            "fired_event_overflow": "sticky_failure_no_event_effects",
            "fired_event_requirements": (
                "originating_ability_bits_gate_each_other_target_payload"
            ),
            "target_other_or_self": ("other_when_requirement_available_else_self"),
            "event_delay_rule": (
                "queue_plus_positive_runtime_plus_server_selector_plus_parallel_"
                "fork_plus_player_client_sync_for_selector_rows"
            ),
            "projectile_spawn_order": ("after_current_tick_projectile_physics"),
            "ability_resource_spend": (
                "authored_graph_clock_after_queue_runtime_and_parallel_boundaries"
            ),
            "ability_cooldown_charges": {
                "source": "RootInteraction.Cooldown",
                "admission": "remaining_cooldown_zero_and_charge_count_positive",
                "deduct": "one_charge_per_accepted_root",
                "recharge": (
                    "current_charge_index_time_one_charge_per_tick_no_overshoot"
                ),
                "interrupt_recharge": "accepted_root_resets_charge_timer",
            },
            "ability_resource_cost": (
                "authored_base_amount_times_mechanism_application_count"
            ),
            "native_npc_instant_add_stat_applications": (
                NATIVE_NPC_INSTANT_ADD_STAT_APPLICATIONS
            ),
            "stamina_break_immunity": ("asset_flagged_apply_effect_before_change_stat"),
            "injected_stab_selector": {
                "execution": "authored_orthogonal_slice_each_engine_tick",
                "origin": "current_source_eye_position",
                "aim": "current_source_head_yaw_pitch",
                "target": "current_locked_target_aabb",
                "line_of_sight": "authored_test_line_of_sight_flag",
                "repeat_suppression": (
                    "event_local_hit_bit_until_ability_end_or_interrupt"
                ),
                "parameters": ("runtime_start_end_extents_and_rotation_from_event_f32"),
            },
            "progressive_stab_selector": {
                "execution": "same_authored_orthogonal_slice_kernel",
                "target_source": "ordinary_selected_combat_target",
                "external_candidate_required": False,
                "parameters": ("runtime_start_end_extents_and_rotation_from_event_f32"),
            },
            "progressive_horizontal_selector": {
                "execution": ("authored_perspective_frustum_arc_each_engine_tick"),
                "target_source": "ordinary_selected_combat_target",
                "external_candidate_required": False,
                "direction_encoding": (
                    "signed_yaw_length_positive_to_left_negative_to_right"
                ),
                "arc_width": ("two_times_end_distance_times_yaw_delta_over_float_pi"),
                "parameters": (
                    "runtime_start_end_vertical_extents_yaw_length_"
                    "and_rotation_from_event_f32"
                ),
            },
        },
        "ability_item_programs": {
            "schema": ITEM_PROGRAM_CATALOG_SCHEMA,
            "version": ITEM_PROGRAM_CATALOG_VERSION,
            "semantic_sha256": item_program_catalog.semantic_sha256,
            "assets_sha256": item_program_catalog.assets_sha256,
            "combat_asset_catalog_sha256": (
                item_program_catalog.combat_asset_catalog_sha256
            ),
            "native_evidence_schema": (item_program_catalog.native_evidence_schema),
            "native_evidence_version": (item_program_catalog.native_evidence_version),
            "binding_count": len(item_program_catalog.rows),
            "supported_binding_count": sum(
                row.supported for row in item_program_catalog.rows
            ),
            "program_capacity_per_ability": (
                item_program_catalog.program_capacity_per_ability
            ),
            "weapon_item_count": len(item_program_catalog.document["weapon_items"]),
            "durability_loss_on_hit": (
                "item_resolved_scalar_for_every_catalogued_weapon"
            ),
            "attacker_tool_durability_commit": (
                "once_per_entity_sourced_damage_inspection_whose_resolved_"
                "cause_enables_durability_loss_against_the_currently_held_"
                "catalogued_weapon"
            ),
            "item_durability_actor_eligibility": (
                "per_actor_non_creative_player_component_result_with_policy_"
                "actors_enabled_and_autonomous_npc_controller_rows_disabled_"
                "by_default_and_explicit_runtime_override_for_native_context"
            ),
            "player_selector_client_sync": (
                "per_actor_player_component_context_adds_two_engine_tick_"
                "boundaries_only_to_authored_select_interaction_rows"
            ),
            "headless_projectile_boundary": (
                "live_bridge_hit_decrements_and_live_bridge_miss_preserves_"
                "durability_despite_source_launch_projectile_first_run_"
                "attempting_a_launch_decrement"
            ),
            "dispatch": (
                "shared_prior_to_after_scheduler_clock_boundary_with_initial_"
                "item_node_before_events_and_declared_repeat_continuations_"
                "after_same_boundary_resource_events"
            ),
            "compact_repeat": (
                "one_static_program_plus_count_and_interval_no_capacity_multiplication"
            ),
            "failure_branch": (
                "branch_local_with_catalog_declared_child_event_suppression"
            ),
            "runtime_specialization": ("semantic_ability_id_no_weapon_class_branches"),
        },
        "program_shapes": {
            "player_backed_actor_mask": _field(
                ("B", entity_count),
                "bool",
            ),
            "item_durability_eligible_actor_mask": _field(
                ("B", entity_count),
                "bool",
            ),
            "opponent_controller_mask": _field(
                ("B", entity_count),
                "bool",
            ),
            "opponent_mode_state": _field(
                ("B", entity_count),
                "int32",
            ),
            "opponent_home_position_state": _field(
                ("B", entity_count, 3),
                "float32",
            ),
            "opponent_home_valid_state": _field(
                ("B", entity_count),
                "bool",
            ),
            "opponent_last_seen_position_state": _field(
                ("B", entity_count, 3),
                "float32",
            ),
            "opponent_last_seen_valid_state": _field(
                ("B", entity_count),
                "bool",
            ),
            "opponent_pursuit_elapsed_ticks_state": _field(
                ("B", entity_count),
                "int32",
            ),
            "opponent_search_elapsed_seconds_state": _field(
                ("B", entity_count),
                "float32",
            ),
            "opponent_search_timeout_seconds_state": _field(
                ("B", entity_count),
                "float32",
            ),
            "opponent_navigation_unavailable_state": _field(
                ("B", entity_count),
                "bool",
            ),
            "engagement_target_id_state": _field(
                ("B", entity_count),
                "int32",
            ),
            "ability_selector_hit_bits": _field(
                ("B", entity_count),
                "uint32",
            ),
            "ability_item_program_mask": _field(
                (
                    "B",
                    entity_count,
                    ABILITY_CAPACITY,
                    item_program_catalog.program_capacity_per_ability,
                ),
                "bool",
            ),
            "ability_item_outer_root_dispatch_scheduler_tick": _field(
                (
                    "B",
                    entity_count,
                    ABILITY_CAPACITY,
                    item_program_catalog.program_capacity_per_ability,
                ),
                "int32",
            ),
            "ability_cooldown_seconds_state": _field(
                ("B", entity_count, ABILITY_CAPACITY),
                "float32",
            ),
            "ability_charge_count_state": _field(
                ("B", entity_count, ABILITY_CAPACITY),
                "int32",
            ),
            "ability_charge_timer_seconds_state": _field(
                ("B", entity_count, ABILITY_CAPACITY),
                "float32",
            ),
            "ability_mask": _field(
                ("B", entity_count, ABILITY_CAPACITY),
                "bool",
            ),
            "ability_requested_charge_time_seconds": _field(
                ("B", entity_count, ABILITY_CAPACITY),
                "float32",
            ),
            "ability_charge_times_seconds": _field(
                (
                    "B",
                    entity_count,
                    ABILITY_CAPACITY,
                    ABILITY_CHARGE_CAPACITY,
                ),
                "float32",
            ),
            "ability_charge_capacity": _field(
                ("B", entity_count, ABILITY_CAPACITY),
                "int32",
            ),
            "ability_interrupt_recharge": _field(
                ("B", entity_count, ABILITY_CAPACITY),
                "bool",
            ),
            "ability_interrupted_by_type_mask": _field(
                ("B", entity_count, ABILITY_CAPACITY),
                "uint32",
            ),
            "ability_interruptible_after_seconds": _field(
                ("B", entity_count, ABILITY_CAPACITY),
                "float32",
            ),
            "ability_stamina_regen_delay_seconds": _field(
                ("B", entity_count, ABILITY_CAPACITY),
                "float32",
            ),
            "ability_stamina_regen_delay_start_tick": _field(
                ("B", entity_count, ABILITY_CAPACITY),
                "int32",
            ),
            "ability_stamina_regen_delay_end_tick": _field(
                ("B", entity_count, ABILITY_CAPACITY),
                "int32",
            ),
            "ability_outer_root_selector": _field(
                ("B", entity_count, ABILITY_CAPACITY),
                "bool",
            ),
            "ability_outer_root_item_dispatch_tick": _field(
                ("B", entity_count, ABILITY_CAPACITY),
                "int32",
            ),
            "ability_resource_phase_mask": _field(
                ("B", entity_count, ABILITY_CAPACITY, RESOURCE_COUNT),
                "bool",
            ),
            "ability_resource_phase_value": _field(
                ("B", entity_count, ABILITY_CAPACITY, RESOURCE_COUNT),
                "float32",
            ),
            "ability_resource_phase_inactive_value": _field(
                ("B", entity_count, ABILITY_CAPACITY, RESOURCE_COUNT),
                "float32",
            ),
            "ability_resource_phase_start_tick": _field(
                ("B", entity_count, ABILITY_CAPACITY, RESOURCE_COUNT),
                "int32",
            ),
            "ability_resource_phase_end_tick": _field(
                ("B", entity_count, ABILITY_CAPACITY, RESOURCE_COUNT),
                "int32",
            ),
            "ability_resource_cost": _field(
                ("B", entity_count, ABILITY_CAPACITY, RESOURCE_COUNT),
                "float32",
            ),
            "ability_resource_cost_kind": _field(
                ("B", entity_count, ABILITY_CAPACITY, RESOURCE_COUNT),
                "int32",
            ),
            "ability_resource_commit_time_seconds": _field(
                ("B", entity_count, ABILITY_CAPACITY, RESOURCE_COUNT),
                "float32",
            ),
            "ability_resource_commit_flags": _field(
                ("B", entity_count, ABILITY_CAPACITY, RESOURCE_COUNT),
                "uint32",
            ),
            "ability_resource_minimum": _field(
                ("B", entity_count, ABILITY_CAPACITY, RESOURCE_COUNT),
                "float32",
            ),
            "ability_event_start": _field(
                ("B", entity_count, ABILITY_CAPACITY),
                "int32",
            ),
            "ability_event_count": _field(
                ("B", entity_count, ABILITY_CAPACITY),
                "int32",
            ),
            "event_mask": _field(
                (
                    "B",
                    entity_count,
                    ABILITY_CAPACITY,
                    EVENT_CAPACITY,
                ),
                "bool",
            ),
            "event_f32": _field(
                (
                    "B",
                    entity_count,
                    ABILITY_CAPACITY,
                    EVENT_CAPACITY,
                    EVENT_FLOAT_FEATURES,
                ),
                "float32",
            ),
            "event_i32": _field(
                (
                    "B",
                    entity_count,
                    ABILITY_CAPACITY,
                    EVENT_CAPACITY,
                    EVENT_INTEGER_FEATURES,
                ),
                "int32",
            ),
            "event_flags": _field(
                (
                    "B",
                    entity_count,
                    ABILITY_CAPACITY,
                    EVENT_CAPACITY,
                ),
                "uint32",
            ),
        },
        "guard_semantics": {
            "positive_stamina_value": "guard_supported",
            "zero_stamina_value": "guard_not_authored_and_disabled",
        },
        "control_immunity": {
            "maximum": CONTROL_IMMUNITY_MAXIMUM,
            "accepted_effect_increment": CONTROL_IMMUNITY_INCREMENT,
            "regen_amount": -CONTROL_IMMUNITY_REGEN_AMOUNT,
            "regen_interval_seconds": (CONTROL_IMMUNITY_REGEN_INTERVAL_SECONDS),
            "full_gate": "reject_wrapped_control_interaction",
        },
        "event_channels": {
            "f32": list(EVENT_FLOAT_CHANNEL_NAMES),
            "i32": list(EVENT_INTEGER_CHANNEL_NAMES),
        },
        "enums": {
            "event_control_flags": {
                "value_percent": EVENT_FLAG_VALUE_PERCENT,
                "status_value_percent": EVENT_FLAG_STATUS_VALUE_PERCENT,
                "angled_damage": EVENT_FLAG_ANGLED_DAMAGE,
                "injected_selector": EVENT_FLAG_INJECTED_SELECTOR,
                "server_selector": EVENT_FLAG_SERVER_SELECTOR,
                "selector_ignores_line_of_sight": (
                    EVENT_FLAG_SELECTOR_IGNORES_LINE_OF_SIGHT
                ),
                "projectile_legacy_offset": (EVENT_FLAG_PROJECTILE_LEGACY_OFFSET),
                "projectile_pitch_adjust_offset": (
                    EVENT_FLAG_PROJECTILE_PITCH_ADJUST_OFFSET
                ),
                "entity_only_area": EVENT_FLAG_ENTITY_ONLY_AREA,
                "parallel_fork": EVENT_FLAG_PARALLEL_FORK,
                "area_friendly_fire": EVENT_FLAG_AREA_FRIENDLY_FIRE,
                "progressive_stab_selector": (EVENT_FLAG_PROGRESSIVE_STAB_SELECTOR),
                "progressive_horizontal_selector": (
                    EVENT_FLAG_PROGRESSIVE_HORIZONTAL_SELECTOR
                ),
                "projectile_standard_physics": (EVENT_FLAG_PROJECTILE_STANDARD_PHYSICS),
                "projectile_allow_rolling": EVENT_FLAG_PROJECTILE_ALLOW_ROLLING,
                "projectile_terminal_deployable_area": (
                    EVENT_FLAG_PROJECTILE_TERMINAL_DEPLOYABLE_AREA
                ),
            },
            "ability_requirements": {
                "clear_projectile_flight": REQUIRE_CLEAR_PROJECTILE_FLIGHT,
                "clear_force_path": REQUIRE_CLEAR_FORCE_PATH,
                "entity_only_area": REQUIRE_ENTITY_ONLY_AREA,
                "static_area_placement": REQUIRE_STATIC_AREA_PLACEMENT,
                "world_projectile_collision": (REQUIRE_WORLD_PROJECTILE_COLLISION),
                "line_of_sight": REQUIRE_LINE_OF_SIGHT,
                "injected_selector": REQUIRE_INJECTED_SELECTOR,
                "deployable_intended_graph_evidence": (
                    REQUIRE_DEPLOYABLE_INTENDED_GRAPH_EVIDENCE
                ),
            },
            "resource_cost_kind": {
                "none": RESOURCE_COST_NONE,
                "single_application": RESOURCE_COST_SINGLE_APPLICATION,
                "native_npc_instant_add_stat": (
                    RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT
                ),
                "native_npc_instant_add_stat_stamina_break_immune": (
                    RESOURCE_COST_NATIVE_NPC_INSTANT_ADD_STAT_STAMINA_BREAK_IMMUNE
                ),
                "single_application_stamina_break_immune": (
                    RESOURCE_COST_SINGLE_APPLICATION_STAMINA_BREAK_IMMUNE
                ),
            },
            "force_direction_mode": {
                "local": FORCE_DIRECTION_LOCAL,
                "point": FORCE_DIRECTION_POINT,
                "directional": FORCE_DIRECTION_DIRECTIONAL,
            },
            "damage_class": {
                "unknown": DAMAGE_CLASS_UNKNOWN,
                "light": DAMAGE_CLASS_LIGHT,
                "charged": DAMAGE_CLASS_CHARGED,
                "signature": DAMAGE_CLASS_SIGNATURE,
            },
            "projectile_kind": {
                "none": PROJECTILE_NONE,
                "arrow": PROJECTILE_ARROW,
                "big_arrow": PROJECTILE_BIG_ARROW,
                "fireball": PROJECTILE_FIREBALL,
                "ice_ball": PROJECTILE_ICE_BALL,
                "ice_bolt": PROJECTILE_ICE_BOLT,
                "bomb": PROJECTILE_BOMB,
                "corruption_orb": PROJECTILE_CORRUPTION_ORB,
                "kunai": PROJECTILE_KUNAI,
                "spear": PROJECTILE_SPEAR,
                "legacy_arrow": PROJECTILE_LEGACY_ARROW,
                "prototype_arrow": PROJECTILE_PROTOTYPE_ARROW,
                "gun_bullet": PROJECTILE_GUN_BULLET,
                "blunderbuss_bullet": PROJECTILE_BLUNDERBUSS_BULLET,
                "deployable_terminal": PROJECTILE_DEPLOYABLE,
            },
            "area_kind": {
                "generic": AREA_GENERIC,
                "flame_trap": AREA_FIRE_TRAP,
                "deployable_aoe": AREA_DEPLOYABLE_AOE,
            },
            "area_shape": {
                "none": AREA_SHAPE_NONE,
                "sphere": AREA_SHAPE_SPHERE,
                "cylinder": AREA_SHAPE_CYLINDER,
            },
            "deployable_attack_flags": {
                "owner": DEPLOYABLE_ATTACK_OWNER,
                "team": DEPLOYABLE_ATTACK_TEAM,
                "enemies": DEPLOYABLE_ATTACK_ENEMIES,
                "known_mask": DEPLOYABLE_ATTACK_FLAG_MASK,
                "semantics": (
                    "raw_authored_booleans_consumed_by_native_ordered_"
                    "owner_group_enemy_predicate_not_a_relationship_mask"
                ),
            },
        },
        "projectile_terminal_deployable_contract": {
            "typed_event_flag": EVENT_FLAG_PROJECTILE_TERMINAL_DEPLOYABLE_AREA,
            "event_flag_uniqueness": (
                "bit31_one_hot_unique_across_complete_uint32_control_vocabulary"
            ),
            "conditional_channel_aliases": {
                "projectile_collision_half_extent_xyz": [
                    EF_PROJECTILE_HALF_EXTENT,
                    EF_FALLOFF,
                    EF_PROJECTILE_DIRECT_DAMAGE,
                ],
                "projectile_collision_center_offset_xyz": [
                    EF_FORCE_X,
                    EF_FORCE_Y,
                    EF_FORCE_Z,
                ],
                "deployable_collision_half_extent_xyz": [
                    EF_ANGLED_DAMAGE,
                    EF_ANGLED_ANGLE_DEGREES,
                    EF_ANGLED_DISTANCE_DEGREES,
                ],
                "deployable_collision_center_offset_xyz": [
                    EF_ANGLED_FORCE_X,
                    EF_ANGLED_FORCE_Y,
                    EF_ANGLED_FORCE_Z,
                ],
                "shape": EI_FORCE_MODE,
                "sticks_vertically": EI_BLOCK_DAMAGE_RADIUS,
                "raw_attack_flags": EI_TARGET_MODE,
                "deployable_semantic_id": EI_PROJECTILE_DIRECT_DAMAGE_CAUSE,
                "count_towards_global_limit": EI_RESISTANCE_STYLE,
                "max_live_count": EI_FORCE_DIRECTION_MODE,
                "ownership": (
                    "only_when_bit31_set_named_terminal_encode_decode_helpers"
                ),
                "ordinary_rows": "byte_for_byte_legacy_channel_meanings",
            },
            "authored_terminal": {
                "server_jar_sha256": HYTALE_0_5_7_SERVER_JAR_SHA256,
                "server_jar_locator": (
                    "installed_release_package_game_latest_Server_HytaleServer.jar"
                ),
                "source_classes": [
                    "ProjectileInteraction",
                    "ProjectileModule",
                    "StandardPhysicsTickSystem",
                    "StandardPhysicsProvider",
                    "EntityRefCollisionProvider",
                    "SpawnDeployableAtHitLocationInteraction",
                    "DeployablesUtils",
                    "DeployableAoeConfig",
                    "DeployableOwnerComponent",
                    "DespawnSystem",
                ],
                "graph": (
                    "Serial(Simple_runtime_then_instant_Projectile)_"
                    "ProjectileMiss_SpawnDeployableAtHitLocation_Aoe_"
                    "then_RemoveEntity"
                ),
                "terminal_event_may_equal_root_duration": True,
                "scheduler_boundary": "ceil_seconds_times_30_ticks",
                "world_contact_impact": (
                    "after_bounce_exhaustion_only_exact_up_normal_or_"
                    "sticks_vertically_else_stopped_without_impact"
                ),
                "entity_contact_without_ProjectileHit": (
                    "inactive_slot_retained_until_strict_age_greater_than_300s"
                ),
                "projectile_lifetime_provenance": (
                    "new_ProjectileInteraction_Config_routes_to_ProjectileModule_"
                    "300s_Despawn_deprecated_LaunchProjectile_ProjectileId_60s_"
                    "route_is_not_this_graph"
                ),
                "contact_evidence": (
                    "finite_nonzero_normal_and_exact_hit_location_required"
                ),
                "rotation_mode": (
                    "asset_pinned_presentation_only_collision_uses_fixed_aabb"
                ),
            },
            "area_mechanics": {
                "shape_default": "missing_is_sphere",
                "sphere_overlap": "strict_full_xyz_distance_squared_lt_radius_squared",
                "cylinder_overlap": (
                    "inclusive_abs_y_lte_height_over_two_and_xz_distance_"
                    "squared_lte_radius_squared"
                ),
                "radius_age": "prior_wall_age_then_persist_age_plus_dt",
                "interval": "strict_clock_gt_interval_then_reset_zero",
                "live_duration": "evaluate_while_prior_age_lte_duration",
                "effect_overlap": (
                    "overwrite_one_effect_slot_duration_only_shared_cycle_clock"
                ),
                "effect_source": "unattributed_minus_one",
            },
            "excluded_pinned_deployable_entity_mechanics": {
                "hard_collision_aabb": True,
                "default_health": 100.0,
                "damageable_death_component": True,
                "count_towards_global_limit": True,
                "owner_global_limit": 1,
                "max_live_count": "asset_pinned_int32",
                "execution": (
                    "effect_only_area_state_requires_full_life_noninterference_"
                    "and_rejects_modeled_area_contact_or_second_counting_spawn"
                ),
            },
            "cooldown_group": {
                "asset_id": "DeployableUtility",
                "static_loadout": "at_most_one_typed_terminal_root_per_actor",
                "external_swap_surface": (
                    "explicit_episode_no_swap_attestation_required"
                ),
            },
        },
        "profiles": list(PROFILE_NAMES),
        "program_content_sha256": hytale_0_5_7_program_content_sha256(),
        "role_initial_statuses": {
            "contract_sha256": role_initial_statuses_sha256(),
            "roles": {
                role_id: [status.effect_id for status in statuses]
                for role_id, statuses in sorted(role_statuses.items())
            },
            "application": (
                "reset_time_complete_asset_program_fail_closed_unknown_role"
            ),
            "infinite_duration_encoding": "float32_max_clipped_observation_one",
        },
        "native_status_projection": {
            "contract_sha256": native_status_projection_sha256(),
            "unknown_status": "fail_closed",
            "visual_only": "asset_pinned_omit_before_fixed_shape_projection",
            "derived_mechanic": (
                "asset_pinned_omit_only_when_dedicated_actor_evidence_or_"
                "arsenal_mechanic_models_effect"
            ),
        },
        "world_block_interactions": {
            "consumer_contract_sha256": (
                block_interaction_world_binding_contract_sha256()
            ),
            "candidate_selection": (
                "bounded_actor_local_index_preserves_resolved_quantities"
            ),
            "runtime_publication": "policy_head_published_executor_injection_required",
        },
        "native_policy_bindings": _native_policy_binding_manifest(),
        "source_equipment_damage": {
            "profile_axis": "fixed_per_batch_entity_and_damage_class",
            "formula": ("(randomized_damage + additive) * max(0, 1 + multiplicative)"),
            "ordering": "before_guard_stamina_and_target_resistance",
        },
        "world_capabilities": {
            "actor_world_state_contract_sha256": (actor_world_state_contract_sha256()),
            "swept_volume_clearance_contract_sha256": (
                swept_volume_clearance_contract_sha256()
            ),
            "projectile_first_contact_contract_sha256": (
                projectile_first_contact_contract_sha256()
            ),
            "point_ray_first_contact_contract_sha256": (
                point_ray_first_contact_contract_sha256()
            ),
            "entity_only_explosion_contract_sha256": (
                entity_only_explosion_contract_sha256()
            ),
            "actor_world_state_available": _field(("B",), "bool"),
            "actor_controller_medium_available": _field(("B",), "bool"),
            "actor_submersion_available": _field(("B",), "bool"),
            "actor_drop_available": _field(("B",), "bool"),
            "actor_controller_in_fluid": _field(("B",), "bool"),
            "actor_feet_submerged": _field(("B",), "bool"),
            "actor_eyes_submerged": _field(("B",), "bool"),
            "actor_drop_support_found": _field(("B",), "bool"),
            "actor_drop_height": _field(("B",), "float32"),
            "target_candidate_perceptible": _field(
                ("B", entity_count, entity_count),
                "bool",
            ),
            "target_candidate_perception_valid": _field(
                ("B", entity_count, entity_count),
                "bool",
            ),
            "line_of_sight": _field(("B", entity_count), "bool"),
            "line_of_sight_valid": _field(("B", entity_count), "bool"),
            "selector_line_of_sight": _field(
                ("B", entity_count),
                "bool",
            ),
            "selector_line_of_sight_valid": _field(
                ("B", entity_count),
                "bool",
            ),
            "direct_target_selected": _field(
                ("B", entity_count),
                "bool",
            ),
            "direct_target_selection_valid": _field(
                ("B", entity_count),
                "bool",
            ),
            "muzzle_position": _field(
                ("B", entity_count, 3),
                "float32",
            ),
            "muzzle_yaw_degrees": _field(
                ("B", entity_count),
                "float32",
            ),
            "muzzle_pitch_degrees": _field(
                ("B", entity_count),
                "float32",
            ),
            "muzzle_valid": _field(("B", entity_count), "bool"),
            "clear_projectile_flight": _field(
                ("B", entity_count),
                "bool",
            ),
            "projectile_world_collision_available": _field(
                ("B", entity_count),
                "bool",
            ),
            "projectile_evidence_semantics": {
                "known_clear": ("clear_projectile_flight_and_finite_valid_muzzle"),
                "known_obstruction": (
                    "not_clear_and_projectile_world_collision_available"
                ),
                "missing_evidence": (
                    "not_clear_and_not_projectile_world_collision_available"
                ),
            },
            "clear_force_path": _field(
                ("B", entity_count, ABILITY_CAPACITY),
                "bool",
            ),
            "applied_force_collision_available": _field(
                ("B", entity_count, ABILITY_CAPACITY),
                "bool",
            ),
            "dodge_corridor_clear": _field(
                ("B", entity_count, 4),
                "bool",
            ),
            "entity_only_area": _field(("B", entity_count), "bool"),
            "static_area_placement": _field(
                ("B", entity_count),
                "bool",
            ),
            "static_area_placement_semantics": {
                "producer": "first_exact_camera_point_ray_contact",
                "maximum_distance_field": ("ability_static_placement_maximum_distance"),
                "allow_walls_field": ("ability_static_placement_allow_walls"),
                "dispatch": "unique_required_ability_no_weapon_name_branch",
                "missing_or_ambiguous": "fail_closed",
            },
            "area_center": _field(("B", entity_count, 3), "float32"),
            "deployable_intended_graph_available": _field(
                ("B", entity_count),
                "bool",
            ),
        },
        "deployable_runtime_evidence": {
            "player_proxy_launch_and_contact_lifecycle_timing_attested": _field(
                ("B", entity_count),
                "bool",
            ),
            "player_proxy_binding": (
                "actor_scoped_and_implies_same_actor_player_backed_mask"
            ),
            "player_proxy_scope": (
                "full_predicted_proxy_launch_sync_contact_chain_and_"
                "measured_lifecycle_timing_not_contact_receipt_alone"
            ),
            "projectile_dry_air_path_attested": _field(
                ("B", entity_count),
                "bool",
            ),
            "dry_air_scope": (
                "complete_projectile_path_and_lifetime_not_launch_sample"
            ),
            "spatial_roster_group_equivalence_attested": _field(
                ("B",),
                "bool",
            ),
            "spatial_roster_scope": (
                "complete_native_entity_and_player_spatial_refs_for_full_"
                "area_envelope_plus_owner_entitygroup_equivalence"
            ),
            "explicit_entity_team_id": _field(
                ("B", entity_count),
                "int32",
            ),
            "area_effect_eligible_mask": _field(
                ("B", entity_count),
                "bool",
            ),
            "area_effect_candidates_available": _field(("B",), "bool"),
            "entity_projectile_collidable_mask": _field(
                ("B", entity_count),
                "bool",
            ),
            "entity_projectile_collidable_available": _field(
                ("B",),
                "bool",
            ),
            "full_life_owner_valid_and_noninterference_attested": _field(
                ("B",),
                "bool",
            ),
            "full_life_scope": (
                "scenario_wide_originating_owner_ref_remains_valid_no_"
                "actor_body_hardcollision_contact_for_the_full_deployable_"
                "life_and_no_"
                "projectile_entity_world_melee_explosion_"
                "aoe_turret_out_of_world_or_other_attack_can_address_or_"
                "damage_spawned_deployable_and_no_second_counting_launch_"
                "by_owner_before_expiry"
            ),
            "single_profile_no_swap_cooldown_group_attested": _field(
                ("B",),
                "bool",
            ),
            "cooldown_scope": (
                "external_inventory_or_root_replacement_cannot_introduce_"
                "another_DeployableUtility_root"
            ),
            "composite_requirement_bit7": (
                "all_evidence_above_plus_current_actor_world_collision_and_"
                "entity_only_capabilities_required_before_admission"
            ),
            "defaults": "all_false_or_unavailable",
            "native_live_enablement": (
                "requires_measured_projectile_start_delay_contact_tick_chain_"
                "start_component_add_and_first_aoe_tick"
            ),
        },
        "projectile_contact_state": _projectile_state_manifest(),
        "area_state": _area_state_manifest(),
        "failure_behavior": {
            "invalid_or_overflow": ("sticky_failure_freeze_zero_reward_truncated"),
            "unsupported_world_capability": "fail_closed",
            "known_projectile_obstruction": (
                "ordinary_refusal_or_delayed_terrain_contact_never_unsupported_world"
            ),
            "unsupported_world_diagnostics": (
                "preserved_while_mechanical_state_freezes"
            ),
            "loadout_validation": "at_reset",
        },
        "scope": {
            "projectiles": (
                "entity_first_exact_static_world_contact_with_outcome_and_evidence_separate"
            ),
            "projectile_contact_outputs": (
                "fraction_point_normal_and_diagnostics_preserved_in_runtime_state"
            ),
            "crossbow_impacts": (
                "physical_contact_routes_once_through_shared_target_local_"
                "combo_guard_status_force_and_signature_program"
            ),
            "explosions": (
                "provider_bound_entity_only_physical_admission_or_explicit_"
                "transform_radius_fallback_query_major_entity_slot_order_"
                "no_block_damage_separate_direct_projectile_and_radial_"
                "environment_payloads_when_authored"
            ),
            "areas": (
                "static_injected_placement_all_live_non_owner_entities_"
                "explicit_friendly_fire_query_major_entity_slot_order"
            ),
            "impact_overflow": ("more_than_64_damage_packets_rejects_complete_row"),
            "doors": "intent_only",
            "world_collision_and_mutation": (
                "projectile_static_contact_bound_other_mutation_external"
            ),
        },
        "evidence": {
            "level": "asset_resolved_bounded_native_differential",
            "assets_sha256": HYTALE_0_5_7_ASSETS_SHA256,
            "native_evidence_jar_sha256": (HYTALE_0_5_7_NATIVE_EVIDENCE_JAR_SHA256),
            "native_differential_scope": [
                "iron_sword_left_internal_acceptance_damage_hit_tick",
                "shortbow_strength_zero_internal_acceptance_damage_hit_tick",
                "shield_bash_internal_resource_cost_and_spend_tick",
            ],
        },
    }


def combat_arsenal_contract_json(
    *,
    entity_count: int = ENTITY_COUNT,
) -> str:
    return json.dumps(
        combat_arsenal_contract_manifest(entity_count=entity_count),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def combat_arsenal_contract_sha256(
    *,
    entity_count: int = ENTITY_COUNT,
) -> str:
    return (
        hashlib.sha256(
            combat_arsenal_contract_json(
                entity_count=entity_count,
            ).encode("utf-8")
        )
        .hexdigest()
        .upper()
    )


def _field(shape: tuple[str | int, ...], dtype: str) -> dict[str, Any]:
    return {"shape": list(shape), "dtype": dtype}


def _projectile_state_manifest() -> dict[str, Any]:
    scalar_shape = ("B", PROJECTILE_CAPACITY)
    vector_shape = scalar_shape + (3,)
    vectors = {
        "position",
        "velocity",
        "half_extent",
        "collision_center_offset",
        "force_direction",
        "world_contact_point",
        "world_contact_normal",
        "terminal_deployable_collision_half_extent",
        "terminal_deployable_collision_center_offset",
    }
    floats = {
        "launch_yaw_degrees",
        "age_seconds",
        "lifetime_seconds",
        "damage",
        "direct_damage",
        "random_percentage",
        "gravity",
        "terminal_velocity",
        "bounciness",
        "bounce_limit",
        "rolling_friction_factor",
        "fuse_seconds",
        "dead_time_seconds",
        "dead_time_remaining",
        "explosion_radius",
        "explosion_falloff",
        "force_velocity_y",
        "force_magnitude",
        "air_resistance",
        "air_resistance_max",
        "ground_resistance",
        "ground_resistance_max",
        "resistance_threshold",
        "on_hit_resource_delta",
        "on_hit_healing",
        "status_duration_seconds",
        "status_cooldown_seconds",
        "status_damage",
        "status_resource_delta",
        "status_speed_multiplier",
        "terminal_area_duration_seconds",
        "terminal_area_interval_seconds",
        "terminal_area_end_radius",
        "terminal_area_height",
        "terminal_area_radius_change_seconds",
        "crossbow_combo_damage",
        "world_hit_fraction",
    }
    integers = {
        "damage_cause",
        "direct_damage_cause",
        "damage_class",
        "bounce_count_limit",
        "bounce_count",
        "block_damage_radius",
        "force_direction_mode",
        "force_mode",
        "resistance_style",
        "on_hit_resource_id",
        "status_id",
        "status_damage_cause",
        "status_resource_id",
        "status_overlap_mode",
        "terminal_deployable_id",
        "terminal_deployable_max_live_count",
        "terminal_area_shape",
        "terminal_area_attack_flags",
        "kind",
        "owner_entity_id",
        "world_segment_count",
    }
    booleans = {
        "standard_physics",
        "allow_rolling",
        "sticks_vertically",
        "on_ground",
        "damage_blocks",
        "terminal_deployable_area",
        "terminal_intended_graph_available",
        "terminal_entity_contact_inactive",
        "terminal_deployable_count_towards_global_limit",
        "crossbow_program",
        "active",
        "impacted",
        "physics_initialized",
        "entity_collision_only",
        "world_hit",
        "world_geometry_exhausted",
        "world_capacity_exceeded",
        "world_contact_invalid",
    }
    uints = {"status_flags"}
    leaves = vectors | floats | integers | booleans | uints
    if leaves != set(ProjectileState._fields):
        raise RuntimeError(
            "projectile manifest/state leaves differ: "
            f"missing={set(ProjectileState._fields) - leaves}, "
            f"extra={leaves - set(ProjectileState._fields)}"
        )
    result = {name: _field(vector_shape, "float32") for name in vectors}
    result.update({name: _field(scalar_shape, "float32") for name in floats})
    result.update({name: _field(scalar_shape, "int32") for name in integers})
    result.update({name: _field(scalar_shape, "bool") for name in booleans})
    result.update({name: _field(scalar_shape, "uint32") for name in uints})
    result["crossbow_impact_dispatch"] = {
        "selection": "asset_family_at_launch_then_persisted_until_contact",
        "target_program": "ordered_standard_combo_one_combo_two_or_big_arrow",
        "variant_scalars": (
            "standard_combo_and_big_arrow_damage_from_selected_profile"
        ),
        "entity_axis": "runtime_entity_count_adapted_to_shared_fixed_roster",
        "native_live_differential": "pending",
    }
    result["typed_terminal_semantics"] = {
        "kind": PROJECTILE_DEPLOYABLE,
        "status_flag_mask": TERMINAL_STATUS_FLAG_MASK,
        "entity_contact": (
            "inactive_retained_contact_tick_velocity_then_zero_until_strict_"
            "age_greater_than_300_seconds"
        ),
        "world_contact": (
            "exact_up_or_sticks_impacts_wall_or_slope_stops_for_current_tick_"
            "and_active_physics_may_resume"
        ),
        "area_materialization": (
            "requires_nonzero_finite_contact_normal_and_exact_hit_location"
        ),
    }
    return result


def _area_state_manifest() -> dict[str, Any]:
    scalar_shape = ("B", AREA_CAPACITY)
    vector_shape = scalar_shape + (3,)
    vectors = {
        "center",
        "collision_half_extent",
        "collision_center_offset",
        "force_direction",
    }
    floats = {
        "age_seconds",
        "duration_seconds",
        "interval_seconds",
        "interval_clock_seconds",
        "radius_change_seconds",
        "start_radius",
        "end_radius",
        "height",
        "damage",
        "random_percentage",
        "force_magnitude",
        "air_resistance",
        "air_resistance_max",
        "ground_resistance",
        "ground_resistance_max",
        "resistance_threshold",
        "status_duration_seconds",
        "status_cooldown_seconds",
        "status_damage",
        "status_healing",
        "status_resource_delta",
        "status_speed_multiplier",
    }
    integers = {
        "damage_cause",
        "damage_class",
        "force_mode",
        "resistance_style",
        "status_id",
        "status_damage_cause",
        "status_resource_id",
        "status_overlap_mode",
        "kind",
        "shape",
        "owner_entity_id",
        "target_mask",
        "deployable_attack_flags",
        "deployable_id",
        "deployable_max_live_count",
    }
    booleans = {
        "active",
        "entity_overlap_only",
        "friendly_fire",
        "deployable_count_towards_global_limit",
    }
    uints = {"status_flags"}
    leaves = vectors | floats | integers | booleans | uints
    if leaves != set(AreaState._fields):
        raise RuntimeError(
            "area manifest/state leaves differ: "
            f"missing={set(AreaState._fields) - leaves}, "
            f"extra={leaves - set(AreaState._fields)}"
        )
    result = {name: _field(vector_shape, "float32") for name in vectors}
    result.update({name: _field(scalar_shape, "float32") for name in floats})
    result.update({name: _field(scalar_shape, "int32") for name in integers})
    result.update({name: _field(scalar_shape, "bool") for name in booleans})
    result.update({name: _field(scalar_shape, "uint32") for name in uints})
    result["typed_deployable_semantics"] = {
        "kind": AREA_DEPLOYABLE_AOE,
        "execution_scope": (
            "effect_only_not_a_damageable_or_targetable_deployable_ecs_entity"
        ),
        "lifetime": (
            "evaluate_prior_wall_age_inclusive_through_expiry_equality_then_"
            "first_age_greater_than_duration_deactivates_without_aoe"
        ),
        "interval": "float32_accumulator_strictly_greater_than_authored_interval",
        "collision": (
            "pinned_aabb_is_only_a_projectile_interference_rejection_candidate_"
            "under_full_life_noninterference_attestation"
        ),
        "status_flag_mask": TERMINAL_STATUS_FLAG_MASK,
    }
    return result


def _native_policy_binding_manifest() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for profile in PROFILE_NAMES:
        binding = hytale_0_5_7_native_profile_bindings(profile)
        result[profile] = {
            "item_id": binding.item_id,
            "resource_stat_ids": list(binding.resource_stat_ids),
            "ability_slots": [
                {
                    "native_slot": native_slot,
                    "authored_slot": authored_slot,
                    "authored_slots": [
                        slot
                        for slot, mapped in enumerate(
                            binding.authored_to_native_ability_slots
                        )
                        if mapped == native_slot
                    ],
                    "interaction_id": ability.interaction_id,
                    "interaction_type": ability.interaction_type,
                }
                for native_slot, (authored_slot, ability) in enumerate(
                    zip(
                        binding.authored_ability_slots,
                        binding.abilities,
                        strict=True,
                    )
                )
            ],
            "guard": (
                None
                if binding.guard is None
                else {
                    "interaction_id": binding.guard.interaction_id,
                    "interaction_type": binding.guard.interaction_type,
                }
            ),
        }
    return result


def _validate_entity_count(entity_count: int) -> None:
    if (
        isinstance(entity_count, bool)
        or not isinstance(entity_count, int)
        or entity_count < 2
    ):
        raise ValueError("entity_count must be an integer of at least 2")


__all__ = [
    "combat_arsenal_contract_json",
    "combat_arsenal_contract_manifest",
    "combat_arsenal_contract_sha256",
]
