"""Array-only arsenal programs and runtime state."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax

from hytalegym.jax.combat.targeting import CombatTargetingRules
from hytalegym.jax.combat.inventory import InventoryLayout, InventoryState
from hytalegym.jax.combat.mechanics import (
    CombatMechanicsRules,
    CombatMechanicsState,
    DefenseCommands,
    StatusApplications,
)
from hytalegym.jax.combat.opponents import OpponentMemoryState
from hytalegym.jax.combat.types import CombatInfo, CombatState

if TYPE_CHECKING:
    from hytalegym.jax.combat.arsenal.item_programs import AbilityItemPrograms


Array = jax.Array


class ArsenalRuntimeCapacity(NamedTuple):
    """Host-visible static execution shape derived from selected programs."""

    abilities_per_entity: int
    events_per_entity: int
    events_per_ability: int


class ForceSweepPlan(NamedTuple):
    """Static source/ability/victim lanes for exact force-path queries.

    The ``ability_*`` fields retain the per-environment representation used
    to audit plan construction.  The ``program_*`` and ``sweep_*`` fields
    flatten the same authored work across the fixed runtime batch so one
    sparse environment cannot impose its rectangular capacity on every
    other environment.
    """

    ability_source_index: Array
    ability_slot_index: Array
    ability_valid: Array
    ability_affected: Array
    packed_sweep_index: Array
    program_environment_index: Array
    program_source_index: Array
    program_slot_index: Array
    program_valid: Array
    program_affected: Array
    sweep_program_index: Array
    sweep_victim_index: Array
    sweep_environment_index: Array
    sweep_valid: Array


class AbilityLoadout(NamedTuple):
    """Episode-pinned, fixed-capacity weapon programs for both entities."""

    weapon_id: Array
    weapon_family: Array
    equipped: Array
    guard_entry_cost: Array
    guard_stamina_value: Array
    guard_half_angle_degrees: Array
    guard_entry_delay_seconds: Array
    guard_exit_regen_delay_seconds: Array
    guard_required_resource_id: Array
    guard_required_resource_minimum: Array
    guard_interrupting_type_mask: Array
    resource_maximum: Array
    resource_initial: Array

    ability_id: Array
    ability_interaction_type: Array
    # Protocol input that forks this ability from an already-held Guard root.
    # -1 means an ordinary root which is blocked while Guard is held.
    ability_guard_fork_type: Array
    ability_mask: Array
    ability_evidence: Array
    ability_duration_seconds: Array
    ability_cooldown_seconds: Array
    # Client-held charge time used to select an outer Charging branch before
    # this already-selected ability begins executing.  This is intentionally
    # separate from child event/resource clocks. -1 means no outer charge
    # request and transports zero; zero is also a valid authored threshold.
    ability_requested_charge_time_seconds: Array
    ability_charge_times_seconds: Array
    ability_charge_capacity: Array
    # C1 ChargeTable: an authored ``"Type": "Charging"`` root as an ordered
    # ``(hold_threshold -> child slot)`` table, ascending by threshold, with
    # ``ability_hold_count`` live rows. Zero rows means the ability is not a
    # charging root and is requested normally. See ``programs/charge.py``.
    ability_hold_threshold_seconds: Array
    ability_hold_child_slot: Array
    ability_hold_count: Array
    # AllowIndefiniteHold: without it a Charging root finishes the tick its
    # clock reaches the top threshold instead of waiting for release.
    ability_hold_allow_indefinite: Array
    # C2 HorizontalSpeedMultiplier: what the root scales planar movement to
    # while it is being held. 1.0 is "no authored slowdown", which is also the
    # value every non-charging ability carries.
    ability_hold_speed_multiplier: Array
    # On the six nested pairs the multiplier is authored on the *inner* root
    # only, so the outer gate's first seconds run at full speed. This is when
    # the slowdown starts, measured on the same hold clock; 0.0 for a flat root.
    ability_hold_speed_multiplier_after_seconds: Array
    # C4 Continuation: an authored ApplyForce that parks the ability until the
    # world says it may resume. Bit 0 is `WaitForGround`, bit 1
    # `WaitForCollision`; 0 means the ability ends when its own clock does, the
    # behaviour every non-continuation ability keeps. All three deferred
    # world-conditional roots (Mace Groundslam, Battleaxe Downstrike, Daggers
    # Pounce) are gated on exactly this, so it is authored once here rather
    # than per weapon. See ``programs/continuation.py``.
    ability_continuation_mode: Array
    # Child slot to run when the awaited condition fires. -1 means the
    # continuation simply ends the ability. The two conditions can resolve to
    # different children -- Pounce lands differently than it clips a wall.
    ability_continuation_ground_slot: Array
    ability_continuation_collision_slot: Array
    # `GroundCheckDelay`: the launch itself leaves the actor grounded for a
    # tick or two, so an immediate check would fire the continuation before the
    # actor is airborne. 0.0 means check from the first tick.
    ability_continuation_ground_check_delay_seconds: Array
    # `RunTime`: hard ceiling on the wait. -1 means wait indefinitely; a
    # positive value ends the ability if neither condition ever fires, so a
    # continuation cannot strand an actor for the rest of the episode.
    ability_continuation_run_time_seconds: Array
    ability_interrupt_recharge: Array
    # Current-operation rule compiled from ``InteractionRules.InterruptedBy``.
    # The mask is keyed by incoming protocol ``InteractionType`` and becomes
    # active only after the corresponding scheduler boundary below.
    ability_interrupted_by_type_mask: Array
    ability_interruptible_after_seconds: Array
    ability_stamina_regen_delay_seconds: Array
    # Optional lifecycle-tick placement for an authored ChangeStat(Set)
    # node. Tick zero preserves the legacy admission-time behavior; a
    # non-negative end tick models a later authored Set(0) node. These are
    # fixed 30 TPS boundaries, not continuous rates.
    ability_stamina_regen_delay_start_tick: Array
    ability_stamina_regen_delay_end_tick: Array
    ability_scheduler_prelude_ticks: Array
    # Host-authored outer selectors may compact several policy choices onto
    # one native interaction root.  The device runtime still executes the
    # selected child, but its admission and public resource phases follow the
    # outer root rather than the child's private resource gate.
    ability_outer_root_selector: Array
    ability_outer_root_item_dispatch_tick: Array
    ability_resource_phase_mask: Array
    ability_resource_phase_value: Array
    ability_resource_phase_inactive_value: Array
    ability_resource_phase_start_tick: Array
    ability_resource_phase_end_tick: Array
    ability_resource_cost: Array
    ability_resource_cost_kind: Array
    ability_resource_commit_time_seconds: Array
    ability_resource_commit_flags: Array
    ability_resource_minimum: Array
    ability_requirements: Array
    ability_static_placement_maximum_distance: Array
    ability_static_placement_allow_walls: Array
    ability_event_start: Array
    ability_event_count: Array
    ability_event_mask: Array

    event_mask: Array
    event_time_seconds: Array
    event_kind: Array
    event_f32: Array
    event_i32: Array
    event_flags: Array

    overflow: Array


class ProjectileState(NamedTuple):
    position: Array
    velocity: Array
    # Launch-frame yaw is retained because Crossbow target-side knockback is
    # authored relative to the interaction context, not inferred from a later
    # gravity-affected velocity vector.
    launch_yaw_degrees: Array
    half_extent: Array
    # Reference-relative collision center. Most projectiles are symmetric and
    # retain zero; typed deployables preserve asymmetric model AABBs exactly.
    collision_center_offset: Array
    age_seconds: Array
    lifetime_seconds: Array
    damage: Array
    direct_damage: Array
    random_percentage: Array
    damage_cause: Array
    direct_damage_cause: Array
    damage_class: Array
    gravity: Array
    terminal_velocity: Array
    standard_physics: Array
    bounciness: Array
    bounce_limit: Array
    bounce_count_limit: Array
    bounce_count: Array
    allow_rolling: Array
    rolling_friction_factor: Array
    sticks_vertically: Array
    on_ground: Array
    fuse_seconds: Array
    dead_time_seconds: Array
    dead_time_remaining: Array
    explosion_radius: Array
    explosion_falloff: Array
    block_damage_radius: Array
    damage_blocks: Array
    force_direction: Array
    force_direction_mode: Array
    force_velocity_y: Array
    force_magnitude: Array
    force_mode: Array
    air_resistance: Array
    air_resistance_max: Array
    ground_resistance: Array
    ground_resistance_max: Array
    resistance_threshold: Array
    resistance_style: Array
    on_hit_resource_id: Array
    on_hit_resource_delta: Array
    on_hit_healing: Array
    status_id: Array
    status_duration_seconds: Array
    status_cooldown_seconds: Array
    status_damage: Array
    status_damage_cause: Array
    status_resource_id: Array
    status_resource_delta: Array
    status_speed_multiplier: Array
    status_flags: Array
    status_overlap_mode: Array
    terminal_deployable_area: Array
    terminal_intended_graph_available: Array
    terminal_entity_contact_inactive: Array
    terminal_deployable_collision_half_extent: Array
    terminal_deployable_collision_center_offset: Array
    terminal_deployable_id: Array
    terminal_deployable_count_towards_global_limit: Array
    terminal_deployable_max_live_count: Array
    terminal_area_shape: Array
    terminal_area_duration_seconds: Array
    terminal_area_interval_seconds: Array
    terminal_area_end_radius: Array
    terminal_area_height: Array
    terminal_area_radius_change_seconds: Array
    terminal_area_attack_flags: Array
    kind: Array
    # These two leaves identify the shared target-local Crossbow program and
    # preserve variant-specific third-hit damage (for example Iron 27 versus
    # Ancient Steel 42) without a per-weapon runtime branch.
    crossbow_program: Array
    crossbow_combo_damage: Array
    owner_entity_id: Array
    active: Array
    impacted: Array
    physics_initialized: Array
    entity_collision_only: Array
    world_hit: Array
    world_hit_fraction: Array
    world_contact_point: Array
    world_contact_normal: Array
    world_geometry_exhausted: Array
    world_capacity_exceeded: Array
    world_contact_invalid: Array
    world_segment_count: Array


class ArsenalExplosionCandidates(NamedTuple):
    """Provider-neutral physical admission for triggered explosions."""

    available: Array
    candidate_mask: Array
    distance: Array
    geometry_exhausted: Array
    capacity_exceeded: Array
    damage_blocks_unsupported: Array
    invalid: Array


class AreaState(NamedTuple):
    center: Array
    collision_half_extent: Array
    collision_center_offset: Array
    age_seconds: Array
    duration_seconds: Array
    interval_seconds: Array
    interval_clock_seconds: Array
    radius_change_seconds: Array
    start_radius: Array
    end_radius: Array
    height: Array
    damage: Array
    random_percentage: Array
    damage_cause: Array
    damage_class: Array
    force_direction: Array
    force_magnitude: Array
    force_mode: Array
    air_resistance: Array
    air_resistance_max: Array
    ground_resistance: Array
    ground_resistance_max: Array
    resistance_threshold: Array
    resistance_style: Array
    status_id: Array
    status_duration_seconds: Array
    status_cooldown_seconds: Array
    status_damage: Array
    status_healing: Array
    status_damage_cause: Array
    status_resource_id: Array
    status_resource_delta: Array
    status_speed_multiplier: Array
    status_flags: Array
    status_overlap_mode: Array
    kind: Array
    shape: Array
    owner_entity_id: Array
    active: Array
    entity_overlap_only: Array
    friendly_fire: Array
    target_mask: Array
    deployable_attack_flags: Array
    deployable_id: Array
    deployable_count_towards_global_limit: Array
    deployable_max_live_count: Array


class ArsenalState(NamedTuple):
    active_ability_slot: Array
    # Authored root which admitted the active execution program. This differs
    # from ``active_ability_slot`` when an outer selector chose a private
    # child, including Primary-triggered Crossbow reload.
    active_ability_root_slot: Array
    ability_elapsed_seconds: Array
    ability_scheduler_tick: Array
    ability_scheduler_clock_seconds: Array
    ability_selector_hit_bits: Array
    ability_cooldown_seconds: Array
    ability_charge_count: Array
    ability_charge_timer_seconds: Array
    # C1 ChargeTable hold clock. ``ability_hold_slot`` is the slot whose control
    # is currently held (-1 when nothing is charging) and ``ability_hold_seconds``
    # is how long it has been held. Distinct from ability_charge_timer_seconds
    # above, which is cooldown recharge. See ``programs/charge.py``.
    ability_hold_slot: Array
    ability_hold_seconds: Array
    # C4 Continuation wait clock, shaped exactly like the C1 hold clock above.
    # ``ability_continuation_slot`` is the ability parked waiting on the world
    # (-1 when nothing is waiting) and ``ability_continuation_seconds`` is how
    # long it has waited, which drives both `GroundCheckDelay` and `RunTime`.
    # See ``programs/continuation.py``.
    ability_continuation_slot: Array
    ability_continuation_seconds: Array
    projectiles: ProjectileState
    areas: AreaState
    failure_bits: Array


class ArsenalWorldCapabilities(NamedTuple):
    actor_world_state_available: Array
    actor_controller_medium_available: Array
    actor_submersion_available: Array
    actor_drop_available: Array
    actor_controller_in_fluid: Array
    actor_feet_submerged: Array
    actor_eyes_submerged: Array
    actor_drop_support_found: Array
    actor_drop_height: Array
    target_candidate_perceptible: Array
    target_candidate_perception_valid: Array
    line_of_sight: Array
    line_of_sight_valid: Array
    selector_line_of_sight: Array
    selector_line_of_sight_valid: Array
    direct_target_selected: Array
    direct_target_selection_valid: Array
    muzzle_position: Array
    muzzle_yaw_degrees: Array
    muzzle_pitch_degrees: Array
    muzzle_valid: Array
    clear_projectile_flight: Array
    projectile_world_collision_available: Array
    deployable_intended_graph_available: Array
    clear_force_path: Array
    applied_force_collision_available: Array
    dodge_corridor_clear: Array
    entity_only_area: Array
    static_area_placement: Array
    area_center: Array


class ArsenalCommands(NamedTuple):
    ability_slot: Array
    defense: DefenseCommands
    world: ArsenalWorldCapabilities
    #: Requested live hotbar slot per entity, or -1 for "leave it alone".
    #: Applied in `step_arsenal_batch` before the held item is read, because
    #: `effective_equipped` is decided by what is in the live slot.
    hotbar_slot: Array | None = None


class FiredEvents(NamedTuple):
    requested: Array
    source_entity_id: Array
    ability_event_index: Array
    selector_previous_progress: Array
    selector_current_progress: Array
    kind: Array
    f32: Array
    i32: Array
    flags: Array
    requirements: Array


class ArsenalEnvironmentState(NamedTuple):
    combat: CombatState
    mechanics: CombatMechanicsState
    arsenal: ArsenalState
    inventory: InventoryState
    opponent_memory: OpponentMemoryState


class ArsenalInfo(NamedTuple):
    # Raw compatibility-core attack request observed by ``_prepare_step``.
    # Authored Arsenal abilities use ``ability_requested`` instead; keeping
    # this bit separate prevents the two execution paths from being conflated.
    legacy_attack_requested: Array
    ability_requested: Array
    # Per-request admission event from start_abilities, not active-program
    # occupancy. ``active_ability_slot >= 0`` is the separate occupancy row.
    ability_accepted: Array
    # Policy/native authored root slot. The scheduler may privately execute a
    # selector child, but that child is never a public action or evidence slot.
    active_ability_slot: Array
    event_count: Array
    projectile_count: Array
    area_count: Array
    damage_dealt: Array
    damage_received: Array
    entity_damage_dealt: Array
    entity_damage_received: Array
    blocked_hits: Array
    invulnerable_hits: Array
    failure_bits: Array
    valid: Array


class ArsenalTransition(NamedTuple):
    state: ArsenalEnvironmentState
    observation: Array
    reward: Array
    terminated: Array
    truncated: Array
    combat_info: CombatInfo
    arsenal_info: ArsenalInfo


class ArsenalTrajectory(NamedTuple):
    observation: Array
    reward: Array
    terminated: Array
    truncated: Array
    combat_info: CombatInfo
    arsenal_info: ArsenalInfo


class ArsenalRuntimeConfig(NamedTuple):
    mechanics_rules: CombatMechanicsRules
    initial_status_applications: StatusApplications
    loadout: AbilityLoadout
    validation_failure_bits: Array
    force_sweep_plan: ForceSweepPlan
    applied_force_collision_support: Array
    targeting_rules: CombatTargetingRules
    deployable_spatial_roster_and_group_equivalence_attested: Array
    player_proxy_deployable_launch_and_contact_lifecycle_timing_attested: Array
    deployable_projectile_dry_air_path_attested: Array
    # Scenario-wide proof: no attack/damage/contact can address spawned
    # deployables and no same-owner second counting deployable can launch.
    deployable_full_life_owner_valid_and_noninterference_attested: Array
    deployable_single_profile_no_swap_cooldown_group_attested: Array
    deployable_area_effect_eligible_mask: Array
    deployable_area_effect_candidates_available: Array
    entity_projectile_collidable_mask: Array
    entity_projectile_collidable_available: Array
    inventory_layout: InventoryLayout
    item_programs: "AbilityItemPrograms"
    player_backed_actor_mask: Array
    item_durability_eligible_actor_mask: Array
    opponent_controller_mask: Array
