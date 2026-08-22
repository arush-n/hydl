"""One compiled combat decision with data-driven Hytale arsenal programs."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.programs.ability import (
    apply_ability_stamina_regen_delay_phases,
    apply_ability_resource_phases,
    ability_lifecycle_legality_view,
    spend_due_ability_resources,
    start_abilities,
    tick_ability_programs,
)
from hytalegym.jax.combat.arsenal.schema.contract import (
    ABILITY_CAPACITY,
    AREA_CAPACITY,
    ARSENAL_FAILURE_INVALID_STATE,
    ARSENAL_FAILURE_UNSUPPORTED_WORLD,
    EVENT_FLAG_PROJECTILE_TERMINAL_DEPLOYABLE_AREA,
    PROJECTILE_CAPACITY,
)
from hytalegym.jax.combat.arsenal.effects.areas import spawn_areas, tick_areas
from hytalegym.jax.combat.arsenal.effects.events import process_direct_events
from hytalegym.jax.combat.arsenal.factory import (
    arsenal_runtime_capacity,
    empty_arsenal_state,
    mechanics_rules_for_loadout,
    specialize_ability_loadout,
)
from hytalegym.jax.combat.arsenal.effects.force_plan import (
    applied_force_collision_support_mask,
    build_force_sweep_plan,
)
from hytalegym.jax.combat.arsenal.projectiles.runtime import (
    ArsenalExplosionCandidateProvider,
    spawn_projectiles,
    tick_projectiles,
)
from hytalegym.jax.combat.arsenal.projectiles.crossbow import (
    apply_crossbow_projectile_impacts,
    empty_crossbow_projectile_impact_commands,
)
from hytalegym.jax.combat.arsenal.item_programs import (
    complete_resource_bound_item_programs,
    defer_item_program_completion,
    execute_due_item_programs,
    pack_item_programs_for_loadout,
)
from hytalegym.jax.combat.arsenal.programs.interaction_rules import (
    guard_rule_resolution,
)
from hytalegym.jax.combat.arsenal.programs.outer_roots import (
    project_observable_active_slot,
)
from hytalegym.jax.combat.arsenal.programs.charge import resolve_charge_hold
from hytalegym.jax.combat.arsenal.programs.scheduling import advance_scheduler_clocks
from hytalegym.jax.combat.arsenal.hazards import environment_damage_events
from hytalegym.jax.combat.arsenal.schema.types import (
    ArsenalCommands,
    ArsenalEnvironmentState,
    ArsenalInfo,
    ArsenalRuntimeConfig,
    ArsenalTrajectory,
    ArsenalTransition,
)
from hytalegym.jax.combat.arsenal.schema.validation import validate_ability_loadout
from hytalegym.jax.combat.env import (
    TargetNavigationProvider,
    _combat_info,
    _finish_geometry_reset,
    _force_pushed_agent_motion,
    _microtick,
    _motion_delta,
    _prepare_step,
    _resolve_aabb_motion_result,
    _step_batch,
    observe_batch,
    reset_batch,
    reset_batch_geometry_at,
    reset_batch_region_at,
)
from hytalegym.jax.combat.inventory import (
    CONTAINER_HOTBAR,
    HOTBAR_CAPACITY,
    InventoryState,
    default_inventory_layout,
    inventory_from_loadout,
    inventory_state_failure_bits,
    item_stack_at,
    set_active_hotbar_slot,
)
from hytalegym.jax.combat.inventory.runtime.item_interactions import (
    apply_held_weapon_hit_durability_loss,
    initialize_held_item_durability,
)
from hytalegym.jax.combat.mechanics import (
    DAMAGE_PHYSICAL,
    STATUS_FLAG_DISABLE_ABILITIES,
    STATUS_FLAG_DISABLE_MOVEMENT,
    advance_defense_interactions,
    apply_damage_forces,
    apply_defense_commands,
    apply_statuses,
    damp_applied_motion,
    empty_damage_events,
    empty_mechanics_state,
    empty_status_applications,
    initial_role_status_applications,
    project_applied_motion_velocity,
    resolve_applied_motion_velocity,
    resolve_damage_events,
    status_modifiers,
    tick_breathing,
    tick_resources,
    tick_statuses,
    translate_applied_motion,
)
from hytalegym.jax.combat.opponents import (
    pairwise_hearing_evidence,
    reset_opponent_memory,
    step_opponent_memory,
)
from hytalegym.jax.combat.targeting import (
    CombatTargetSelection,
    TARGET_CANDIDATE_CAPACITY,
    TEAM_NONE,
    default_combat_targeting_rules,
    select_combat_targets,
)
from hytalegym.jax.combat.types import (
    ACTION_ATTACK,
    ACTION_BACK,
    ACTION_FORWARD,
    ACTION_JUMP,
    ACTION_LEFT,
    ACTION_RIGHT,
    AGENT_ENTITY,
    ENTITY_COUNT,
    MAX_MICROTICKS,
    TARGET_ENTITY,
    CombatParams,
    CombatState,
    RewardComponents,
    compose_reward,
    sum_reward_components,
)
from hytalegym.jax.world import (
    GeometryProvider,
    GeometryState,
    RegionGeometryState,
)
from hytalegym.jax.world.entities.movement_states import MOVEMENT_STATE_ORDER


NO_QUEUE_TICK = jnp.int32(2_147_483_647)
_ACTOR_CROUCHING_INDEX = MOVEMENT_STATE_ORDER.index("crouching")
# Same immediate-support distance used by World's public local and Region
# ``aabb_grounded`` entry points. This path supplies target bounds explicitly,
# which those actor-bounds convenience entry points do not accept.
_APPLIED_MOTION_GROUND_PROBE_DISTANCE = jnp.float32(0.002)
_compiled_validate_ability_loadout = jax.jit(
    validate_ability_loadout,
    inline=False,
)


def arsenal_runtime_config(
    loadout,
    *,
    specialize: bool = True,
    entity_team_id: jax.Array | None = None,
    deployable_spatial_roster_and_group_equivalence_attested: (
        jax.Array | bool
    ) = False,
    player_proxy_deployable_launch_and_contact_lifecycle_timing_attested_actor_mask: (
        jax.Array | None
    ) = None,
    deployable_projectile_dry_air_path_attested_actor_mask: (jax.Array | None) = None,
    deployable_full_life_owner_valid_and_noninterference_attested: (
        jax.Array | bool
    ) = False,
    deployable_single_profile_no_swap_cooldown_group_attested: (
        jax.Array | bool
    ) = False,
    deployable_area_effect_eligible_mask: jax.Array | None = None,
    entity_projectile_collidable_mask: jax.Array | None = None,
    distance_component_selector: jax.Array | None = None,
    sensor_range: jax.Array | float | None = None,
    backpack_capacity: int = 0,
    entity_role_ids: Sequence[str] | None = None,
    opponent_controller_mask: jax.Array | None = None,
    item_durability_eligible_actor_mask: jax.Array | bool | None = None,
    player_backed_actor_mask: jax.Array | bool | None = None,
) -> ArsenalRuntimeConfig:
    """Build episode-pinned rules, targeting, and specialized execution shape."""

    if not isinstance(specialize, bool):
        raise TypeError("specialize must be a bool")
    validation_failure_bits = _compiled_validate_ability_loadout(loadout)
    terminal_event = jnp.asarray(loadout.event_mask, dtype=jnp.bool_) & (
        (
            jnp.asarray(loadout.event_flags, dtype=jnp.uint32)
            & jnp.uint32(EVENT_FLAG_PROJECTILE_TERMINAL_DEPLOYABLE_AREA)
        )
        != 0
    )
    terminal_root_count = jnp.sum(
        jnp.any(terminal_event, axis=-1),
        axis=-1,
        dtype=jnp.int32,
    )
    if bool(jnp.any(terminal_root_count > 1)):
        raise ValueError(
            "each actor may pin at most one typed DeployableUtility root; "
            "shared cooldown arbitration and item swaps are not executable"
        )
    runtime_loadout = specialize_ability_loadout(loadout) if specialize else loadout
    # Validate the static shape even though the complete semantic validation
    # remains reset-time and device compatible.
    arsenal_runtime_capacity(runtime_loadout)
    batch, entity_count = runtime_loadout.weapon_id.shape
    force_collision_support = jnp.asarray(
        applied_force_collision_support_mask(runtime_loadout),
        dtype=jnp.bool_,
    )
    runtime_ability_capacity = force_collision_support.shape[2]
    if runtime_ability_capacity > ABILITY_CAPACITY:
        raise ValueError(
            "runtime ability capacity exceeds the public capability ABI: "
            f"{runtime_ability_capacity} > {ABILITY_CAPACITY}"
        )
    force_collision_support = jnp.pad(
        force_collision_support,
        (
            (0, 0),
            (0, 0),
            (0, ABILITY_CAPACITY - runtime_ability_capacity),
        ),
        constant_values=False,
    )
    if entity_team_id is not None:
        explicit_teams = jnp.asarray(entity_team_id)
        if explicit_teams.dtype != jnp.int32:
            raise TypeError(
                f"entity_team_id must have dtype int32, got {explicit_teams.dtype}"
            )
    targeting_rules = default_combat_targeting_rules(
        batch,
        entity_count,
        sensor_range=(jnp.float32(jnp.inf) if sensor_range is None else sensor_range),
        team_id=entity_team_id,
        distance_component_selector=distance_component_selector,
    )
    group_attested = _bool_evidence_array(
        deployable_spatial_roster_and_group_equivalence_attested,
        name="deployable_spatial_roster_and_group_equivalence_attested",
    )
    if group_attested.shape == ():
        group_attested = jnp.broadcast_to(group_attested, (batch,))
    elif group_attested.shape != (batch,):
        raise ValueError(
            "deployable_spatial_roster_and_group_equivalence_attested "
            "must be scalar or "
            f"have shape ({batch},)"
        )
    if entity_team_id is None and bool(jnp.any(group_attested)):
        raise ValueError(
            "deployable_spatial_roster_and_group_equivalence_attested "
            "requires explicit "
            "entity_team_id evidence"
        )
    contact_chain_available = _explicit_actor_evidence_mask(
        player_proxy_deployable_launch_and_contact_lifecycle_timing_attested_actor_mask,
        batch=batch,
        entity_count=entity_count,
        name=(
            "player_proxy_deployable_launch_and_contact_lifecycle_timing_"
            "attested_"
            "actor_mask"
        ),
    )
    dry_air_path_attested = _explicit_actor_evidence_mask(
        deployable_projectile_dry_air_path_attested_actor_mask,
        batch=batch,
        entity_count=entity_count,
        name="deployable_projectile_dry_air_path_attested_actor_mask",
    )
    full_life_noninterference = _bool_evidence_array(
        deployable_full_life_owner_valid_and_noninterference_attested,
        name=("deployable_full_life_owner_valid_and_noninterference_attested"),
    )
    if full_life_noninterference.shape == ():
        full_life_noninterference = jnp.broadcast_to(
            full_life_noninterference,
            (batch,),
        )
    elif full_life_noninterference.shape != (batch,):
        raise ValueError(
            "deployable_full_life_owner_valid_and_noninterference_attested "
            "must be scalar "
            f"or bool[{batch}]"
        )
    cooldown_group_attested = _bool_evidence_array(
        deployable_single_profile_no_swap_cooldown_group_attested,
        name="deployable_single_profile_no_swap_cooldown_group_attested",
    )
    if cooldown_group_attested.shape == ():
        cooldown_group_attested = jnp.broadcast_to(
            cooldown_group_attested,
            (batch,),
        )
    elif cooldown_group_attested.shape != (batch,):
        raise ValueError(
            "deployable_single_profile_no_swap_cooldown_group_attested "
            f"must be scalar or bool[{batch}]"
        )
    if deployable_area_effect_eligible_mask is None:
        area_effect_eligible = jnp.zeros(
            (batch, entity_count),
            dtype=jnp.bool_,
        )
        area_candidates_available = jnp.zeros((batch,), dtype=jnp.bool_)
    else:
        area_effect_eligible = _explicit_actor_evidence_mask(
            deployable_area_effect_eligible_mask,
            batch=batch,
            entity_count=entity_count,
            name="deployable_area_effect_eligible_mask",
        )
        area_candidates_available = jnp.ones((batch,), dtype=jnp.bool_)
    if entity_projectile_collidable_mask is None:
        projectile_collidable = jnp.zeros(
            (batch, entity_count),
            dtype=jnp.bool_,
        )
        projectile_collidable_available = jnp.zeros(
            (batch,),
            dtype=jnp.bool_,
        )
    else:
        projectile_collidable = _explicit_actor_evidence_mask(
            entity_projectile_collidable_mask,
            batch=batch,
            entity_count=entity_count,
            name="entity_projectile_collidable_mask",
        )
        projectile_collidable_available = jnp.ones(
            (batch,),
            dtype=jnp.bool_,
        )
    if opponent_controller_mask is None:
        entity_id = jnp.arange(entity_count, dtype=jnp.int32)[None, :]
        actor_team = targeting_rules.team_id[:, AGENT_ENTITY, None]
        controller_mask = (entity_id != AGENT_ENTITY) & (
            (actor_team == jnp.int32(TEAM_NONE))
            | (targeting_rules.team_id != actor_team)
        )
    else:
        controller_mask = jnp.asarray(
            opponent_controller_mask,
            dtype=jnp.bool_,
        )
        if controller_mask.shape == (entity_count,):
            controller_mask = jnp.broadcast_to(
                controller_mask[None, :],
                (batch, entity_count),
            )
        if controller_mask.shape != (batch, entity_count):
            raise ValueError(
                "opponent_controller_mask must have shape "
                f"({entity_count},) or ({batch}, {entity_count})"
            )
        controller_mask = controller_mask.at[:, AGENT_ENTITY].set(False)
    player_backed = _runtime_actor_mask(
        player_backed_actor_mask,
        default=False,
        batch=batch,
        entity_count=entity_count,
        name="player_backed_actor_mask",
    )
    if bool(jnp.any(contact_chain_available & ~player_backed)):
        raise ValueError(
            "player-proxy deployable contact evidence requires the same "
            "actor to be player-backed"
        )
    if item_durability_eligible_actor_mask is None:
        # The standard Gym matchup treats policy-owned actors as non-Creative
        # Players and autonomous controller rows as role-only NPCs.  Upload
        # adapters that retain an NPC actor (or install/remove Hytale's
        # synthetic Player context) must pass the exact actor mask explicitly.
        durability_eligible = ~controller_mask
    else:
        durability_eligible = _runtime_actor_mask(
            item_durability_eligible_actor_mask,
            default=False,
            batch=batch,
            entity_count=entity_count,
            name="item_durability_eligible_actor_mask",
        )
    return ArsenalRuntimeConfig(
        mechanics_rules=mechanics_rules_for_loadout(runtime_loadout),
        initial_status_applications=initial_role_status_applications(
            batch,
            entity_count,
            entity_role_ids,
        ),
        loadout=runtime_loadout,
        validation_failure_bits=validation_failure_bits,
        force_sweep_plan=build_force_sweep_plan(runtime_loadout),
        applied_force_collision_support=force_collision_support,
        targeting_rules=targeting_rules,
        deployable_spatial_roster_and_group_equivalence_attested=(group_attested),
        player_proxy_deployable_launch_and_contact_lifecycle_timing_attested=(
            contact_chain_available
        ),
        deployable_projectile_dry_air_path_attested=(dry_air_path_attested),
        deployable_full_life_owner_valid_and_noninterference_attested=(
            full_life_noninterference
        ),
        deployable_single_profile_no_swap_cooldown_group_attested=(
            cooldown_group_attested
        ),
        deployable_area_effect_eligible_mask=area_effect_eligible,
        deployable_area_effect_candidates_available=(area_candidates_available),
        entity_projectile_collidable_mask=projectile_collidable,
        entity_projectile_collidable_available=(projectile_collidable_available),
        inventory_layout=default_inventory_layout(
            backpack_capacity=backpack_capacity,
        ),
        item_programs=pack_item_programs_for_loadout(runtime_loadout),
        player_backed_actor_mask=player_backed,
        item_durability_eligible_actor_mask=durability_eligible,
        opponent_controller_mask=controller_mask,
    )


def _runtime_actor_mask(
    value: jax.Array | bool | None,
    *,
    default: jax.Array | bool,
    batch: int,
    entity_count: int,
    name: str,
) -> jax.Array:
    """Normalize one runtime-static actor context without role-name cases."""

    result = jnp.asarray(default if value is None else value)
    if result.dtype != jnp.bool_:
        raise TypeError(f"{name} must have dtype bool, got {result.dtype}")
    if result.shape == ():
        result = jnp.broadcast_to(result, (batch, entity_count))
    elif result.shape == (entity_count,):
        result = jnp.broadcast_to(result[None, :], (batch, entity_count))
    elif result.shape != (batch, entity_count):
        raise ValueError(
            f"{name} must be scalar or have shape ({entity_count},) or "
            f"({batch}, {entity_count})"
        )
    return result


def _explicit_actor_evidence_mask(
    value: jax.Array | None,
    *,
    batch: int,
    entity_count: int,
    name: str,
) -> jax.Array:
    """Normalize actor-scoped evidence without a global-true shortcut."""

    if value is None:
        return jnp.zeros((batch, entity_count), dtype=jnp.bool_)
    result = _bool_evidence_array(value, name=name)
    if result.shape == (entity_count,):
        result = jnp.broadcast_to(result[None, :], (batch, entity_count))
    elif result.shape != (batch, entity_count):
        raise ValueError(
            f"{name} must have shape ({entity_count},) or "
            f"({batch}, {entity_count}); scalar evidence is forbidden"
        )
    return result


def _bool_evidence_array(
    value: jax.Array | bool,
    *,
    name: str,
) -> jax.Array:
    """Preserve evidence typing instead of accepting truthy numeric inputs."""

    result = jnp.asarray(value)
    if result.dtype != jnp.bool_:
        raise TypeError(f"{name} must have dtype bool, got {result.dtype}")
    return result


def reset_arsenal_batch(
    keys: jax.Array,
    params: CombatParams,
    config: ArsenalRuntimeConfig,
    geometry: GeometryProvider | None = None,
    *,
    agent_position: jax.Array | None = None,
    target_position: jax.Array | None = None,
    entity_position: jax.Array | None = None,
    entity_health: jax.Array | None = None,
    entity_yaw: jax.Array | None = None,
    initial_inventory: InventoryState | None = None,
    projectile_capacity: int = PROJECTILE_CAPACITY,
    area_capacity: int = AREA_CAPACITY,
) -> tuple[ArsenalEnvironmentState, jax.Array]:
    """Reset calibrated combat plus exact resource/arsenal state.

    The default pair keeps the historical convenience coordinates. A dynamic
    entity axis uses explicit ``[batch, entity, ...]`` state; omitted extra
    slots remain inactive padding with zero health.
    """

    if (agent_position is None) != (target_position is None):
        raise ValueError("agent_position and target_position must be supplied together")
    if (entity_position is None) != (entity_health is None):
        raise ValueError("entity_position and entity_health must be supplied together")
    if entity_position is not None and agent_position is not None:
        raise ValueError(
            "entity_position cannot be combined with agent/target positions"
        )
    batch = keys.shape[0]
    loadout_shape = config.loadout.weapon_id.shape
    if loadout_shape[0] != batch:
        raise ValueError("loadout batch does not match reset keys")
    entity_count = loadout_shape[1]
    if entity_count < 2:
        raise ValueError("combat runtime requires at least two entity slots")
    if config.targeting_rules.team_id.shape != (batch, entity_count):
        raise ValueError("targeting rules do not match loadout entity axis")

    if entity_position is not None:
        if entity_position.shape != (batch, entity_count, 3):
            raise ValueError(
                f"entity_position must have shape ({batch}, {entity_count}, 3)"
            )
        if entity_health.shape != (batch, entity_count):
            raise ValueError(f"entity_health must have shape ({batch}, {entity_count})")
        if entity_yaw is not None and entity_yaw.shape != (
            batch,
            entity_count,
        ):
            raise ValueError(f"entity_yaw must have shape ({batch}, {entity_count})")
        combat, _ = reset_batch(
            keys,
            params,
            entity_count=entity_count,
        )
        resolved_yaw = (
            combat.yaw
            if entity_yaw is None
            else jnp.asarray(entity_yaw, dtype=jnp.float32)
        )
        combat = combat._replace(
            position=jnp.asarray(entity_position, dtype=jnp.float32),
            health=jnp.asarray(entity_health, dtype=jnp.float32),
            yaw=resolved_yaw,
            # Entity zero still runs through the shipped scalar controller.
            # Its desired heading must start from the same caller-supplied yaw
            # as the entity bank or its first movement edge uses a stale reset
            # heading while nonzero policy actors use the requested heading.
            desired_yaw=(
                combat.desired_yaw
                if entity_yaw is None
                else resolved_yaw[:, AGENT_ENTITY]
            ),
            # Travel rides the body, so the body's commanded heading is the one
            # the movement edge actually reads. Seeding only the camera above
            # would reintroduce exactly the stale-reset-heading bug that comment
            # describes, just one seam over.
            desired_body_yaw=(
                combat.desired_body_yaw
                if entity_yaw is None
                else resolved_yaw[:, AGENT_ENTITY]
            ),
            agent_fall_start_y=jnp.asarray(
                entity_position[:, AGENT_ENTITY, 1],
                dtype=jnp.float32,
            ),
        )
        if geometry is None:
            observation = observe_batch(combat, params)
        else:
            combat, observation = _finish_geometry_reset(
                combat,
                params,
                geometry,
            )
    elif agent_position is None and entity_count == ENTITY_COUNT:
        combat, observation = reset_batch(keys, params)
        if geometry is not None:
            combat, observation = _finish_geometry_reset(
                combat,
                params,
                geometry,
            )
    elif agent_position is not None and entity_count != ENTITY_COUNT:
        raise ValueError(
            "dynamic entity axes require entity_position and entity_health"
        )
    elif isinstance(geometry, GeometryState):
        combat, observation = reset_batch_geometry_at(
            keys,
            params,
            geometry,
            agent_position,
            target_position,
        )
    elif isinstance(geometry, RegionGeometryState):
        combat, observation = reset_batch_region_at(
            keys,
            params,
            geometry,
            agent_position,
            target_position,
        )
    else:
        if agent_position is not None:
            raise ValueError(
                "explicit reset positions require an exact geometry provider"
            )
        combat, observation = reset_batch(
            keys,
            params,
            entity_count=entity_count,
        )
    mechanics = empty_mechanics_state(
        batch,
        config.mechanics_rules,
        initial_resources=config.loadout.resource_initial,
    )
    mechanics = apply_statuses(
        mechanics,
        config.initial_status_applications,
    )
    if initial_inventory is None:
        inventory = inventory_from_loadout(
            config.loadout,
            config.inventory_layout,
        )
        inventory = initialize_held_item_durability(
            inventory,
            config.inventory_layout,
            config.item_programs.weapon_max_durability,
        )
    else:
        if not isinstance(initial_inventory, InventoryState):
            raise TypeError("initial_inventory must be an InventoryState")
        expected_inventory_shape = (
            batch,
            entity_count,
            config.inventory_layout.container_id.shape[0],
        )
        if initial_inventory.item_id.shape != expected_inventory_shape:
            raise ValueError(
                "initial_inventory item axis must have shape "
                f"{expected_inventory_shape}"
            )
        inventory = initial_inventory._replace(
            failure_bits=(
                initial_inventory.failure_bits
                | inventory_state_failure_bits(
                    initial_inventory,
                    config.inventory_layout,
                )
            )
        )
    if entity_count != ENTITY_COUNT:
        actor_target_id = combat_target_selection(
            combat,
            params,
            config,
        ).engagement_target_id[:, AGENT_ENTITY]
        observation = observe_batch(
            combat,
            params,
            geometry,
            target_entity_id=actor_target_id,
        )
    return (
        ArsenalEnvironmentState(
            combat=combat,
            mechanics=mechanics,
            arsenal=empty_arsenal_state(
                batch,
                ability_capacity=config.loadout.ability_mask.shape[2],
                projectile_capacity=projectile_capacity,
                area_capacity=area_capacity,
                entity_count=loadout_shape[1],
            )._replace(
                ability_charge_count=config.loadout.ability_charge_capacity,
                failure_bits=config.validation_failure_bits,
            ),
            inventory=inventory,
            opponent_memory=reset_opponent_memory(
                keys,
                combat.position,
                combat.health,
                config.opponent_controller_mask,
            ),
        ),
        observation,
    )


def step_arsenal_batch(
    state: ArsenalEnvironmentState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    commands: ArsenalCommands,
    config: ArsenalRuntimeConfig,
    geometry: GeometryProvider | None = None,
    target_navigation_provider: TargetNavigationProvider | None = None,
    explosion_candidate_provider: (ArsenalExplosionCandidateProvider | None) = None,
    motion_delta_seconds: jax.Array | None = None,
    compiled_microticks: int | None = None,
    compiled_null_loadout: bool | None = None,
    compiled_has_item_programs: bool | None = None,
    compiled_has_projectiles: bool | None = None,
    compiled_has_areas: bool | None = None,
) -> ArsenalTransition:
    """Execute one fully compiled policy decision with 2–4 microticks."""

    # ``motion_delta_seconds`` is a fidelity-only replay input. It replaces
    # physical microtick duration without rewriting contract-pinned nominal
    # or loaded timing values; ordinary production callers leave it unset.
    batch, entity_count = state.combat.position.shape[:2]
    if compiled_microticks is not None and not (
        1 <= compiled_microticks <= MAX_MICROTICKS
    ):
        raise ValueError(f"compiled_microticks must be between 1 and {MAX_MICROTICKS}")
    if compiled_null_loadout is not None and not isinstance(
        compiled_null_loadout,
        bool,
    ):
        raise TypeError("compiled_null_loadout must be boolean or None")
    for value, name in (
        (compiled_has_item_programs, "compiled_has_item_programs"),
        (compiled_has_projectiles, "compiled_has_projectiles"),
        (compiled_has_areas, "compiled_has_areas"),
    ):
        if value is not None and not isinstance(value, bool):
            raise TypeError(f"{name} must be boolean or None")
    if motion_delta_seconds is not None:
        motion_delta_seconds = jnp.asarray(
            motion_delta_seconds,
            dtype=jnp.float32,
        )
        expected_shape = (batch, MAX_MICROTICKS)
        if motion_delta_seconds.shape != expected_shape:
            raise ValueError(f"motion_delta_seconds must have shape {expected_shape}")
    if actions.shape[0] != batch or commands.ability_slot.shape != (
        batch,
        entity_count,
    ):
        raise ValueError("actions or commands do not match combat batch")
    if config.loadout.weapon_id.shape != (batch, entity_count):
        raise ValueError("loadout does not match combat entity axis")
    if compiled_null_loadout is True:
        return _step_null_arsenal_batch(
            state,
            actions,
            keys,
            params,
            config,
            geometry,
            target_navigation_provider,
            motion_delta_seconds,
        )
    if compiled_null_loadout is False:
        return _step_arsenal_program_batch(
            state,
            actions,
            keys,
            params,
            commands,
            config,
            geometry,
            target_navigation_provider,
            explosion_candidate_provider,
            motion_delta_seconds,
            compiled_microticks,
            compiled_has_item_programs,
            compiled_has_projectiles,
            compiled_has_areas,
        )
    null_loadout = (
        ~jnp.any(config.loadout.equipped)
        & ~jnp.any(config.loadout.ability_mask)
        & jnp.all(config.validation_failure_bits == jnp.uint32(0))
    )
    return jax.lax.cond(
        null_loadout,
        lambda _: _step_null_arsenal_batch(
            state,
            actions,
            keys,
            params,
            config,
            geometry,
            target_navigation_provider,
            motion_delta_seconds,
        ),
        lambda _: _step_arsenal_program_batch(
            state,
            actions,
            keys,
            params,
            commands,
            config,
            geometry,
            target_navigation_provider,
            explosion_candidate_provider,
            motion_delta_seconds,
            compiled_microticks,
            compiled_has_item_programs,
            compiled_has_projectiles,
            compiled_has_areas,
        ),
        operand=None,
    )


def _step_arsenal_program_batch(
    state: ArsenalEnvironmentState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    commands: ArsenalCommands,
    config: ArsenalRuntimeConfig,
    geometry: GeometryProvider | None,
    target_navigation_provider: TargetNavigationProvider | None,
    explosion_candidate_provider: (ArsenalExplosionCandidateProvider | None),
    motion_delta_seconds: jax.Array | None,
    compiled_microticks: int | None,
    compiled_has_item_programs: bool | None,
    compiled_has_projectiles: bool | None,
    compiled_has_areas: bool | None,
) -> ArsenalTransition:
    batch, entity_count = state.combat.position.shape[:2]
    inventory_failure = inventory_state_failure_bits(
        state.inventory,
        config.inventory_layout,
    )
    inventory = state.inventory._replace(failure_bits=inventory_failure)
    # Weapon switching, applied BEFORE the held item is read: `effective_equipped`
    # below compares the live slot's item against the loadout weapon, so a switch
    # that landed after this point would not take effect until the next tick.
    # A -1 request means "no switch"; resolving it to the current slot keeps the
    # one shipped implementation and never trips its atomic invalid-request path.
    if commands.hotbar_slot is not None:
        requested_slot = jnp.asarray(commands.hotbar_slot, dtype=jnp.int32)
        switching = (requested_slot >= 0) & (requested_slot < HOTBAR_CAPACITY)
        inventory = set_active_hotbar_slot(
            inventory,
            jnp.where(switching, requested_slot, inventory.active_hotbar_slot),
        )
    state = state._replace(inventory=inventory)
    active_hotbar = item_stack_at(
        inventory,
        config.inventory_layout,
        CONTAINER_HOTBAR,
        inventory.active_hotbar_slot,
    )
    effective_equipped = (
        config.loadout.equipped
        & (active_hotbar.item_id == config.loadout.weapon_id)
        & (active_hotbar.quantity > 0)
    )
    runtime_loadout = config.loadout._replace(equipped=effective_equipped)
    opponent_mask = _agent_opponent_mask(config)
    actor_crouching = state.combat.agent_walk_movement_state.values[
        :,
        _ACTOR_CROUCHING_INDEX,
    ]
    actor_audible = (
        ~params.target_chase_hearing_suppressed_by_crouching | ~actor_crouching
    )
    target_audible = (
        jnp.zeros_like(
            state.combat.health,
            dtype=jnp.bool_,
        )
        .at[:, AGENT_ENTITY]
        .set(actor_audible)
    )
    candidate_heard = pairwise_hearing_evidence(
        state.combat.position,
        config.opponent_controller_mask,
        target_audible,
        hearing_range=params.target_chase_hearing_range,
    )
    candidate_evidence = (
        None
        if entity_count == ENTITY_COUNT
        else (
            (
                commands.world.target_candidate_perceptible
                & commands.world.target_candidate_perception_valid
            )
            | candidate_heard
        )
    )
    decision_targeting = combat_target_selection(
        state.combat,
        params,
        config,
        candidate_evidence=candidate_evidence,
    )
    engaged_combat = state.combat._replace(
        engagement_target_id=decision_targeting.engagement_target_id,
    )
    alive = state.combat.health > 0.0
    admitted_arsenal = ability_lifecycle_legality_view(
        state.arsenal,
        runtime_loadout,
    )
    guard_rules = guard_rule_resolution(
        admitted_arsenal,
        runtime_loadout,
    )
    # C1 ChargeTable. An authored "Type": "Charging" root holds while its slot
    # stays requested and runs a threshold-selected child on release. Resolving
    # it here, at the single point where the command becomes a slot, keeps every
    # downstream consumer (guard fork, admission, scheduler) reading one
    # effective slot. Abilities with no authored table pass through unchanged,
    # so non-charging weapons are bit-identical to before this stage existed.
    # The hold runs on InteractionManager's fixed tick, not the measured delta.
    hold_slot, hold_seconds, requested_slot = resolve_charge_hold(
        state.arsenal.ability_hold_slot,
        state.arsenal.ability_hold_seconds,
        commands.ability_slot,
        runtime_loadout,
        jnp.asarray(params.nominal_dt, dtype=jnp.float32),
    )
    state = state._replace(
        arsenal=state.arsenal._replace(
            ability_hold_slot=hold_slot,
            ability_hold_seconds=hold_seconds,
        )
    )
    ability_capacity = runtime_loadout.ability_mask.shape[2]
    safe_requested_slot = jnp.clip(
        requested_slot,
        0,
        ability_capacity - 1,
    )
    selected_guard_fork_type = jnp.take_along_axis(
        runtime_loadout.ability_guard_fork_type,
        safe_requested_slot[..., None],
        axis=2,
    )[..., 0]
    guard_fork_requested = (
        (requested_slot >= 0)
        & (requested_slot < ability_capacity)
        & (selected_guard_fork_type >= 0)
        & state.mechanics.guard_held
    )
    defense = commands.defense._replace(
        guard_held=(
            (commands.defense.guard_held | guard_fork_requested)
            & effective_equipped
            & (state.mechanics.guard_held | guard_rules.can_start)
        )
    )
    mechanics, _defense_info = apply_defense_commands(
        state.mechanics,
        defense,
        state.combat.yaw,
        alive,
        config.mechanics_rules,
        params,
    )
    guard_root_started = mechanics.guard_held & ~state.mechanics.guard_held
    interrupted = guard_root_started & guard_rules.interrupts_active
    prestart_arsenal = state.arsenal._replace(
        active_ability_slot=jnp.where(
            interrupted,
            jnp.int32(-1),
            state.arsenal.active_ability_slot,
        ),
        active_ability_root_slot=jnp.where(
            interrupted,
            jnp.int32(-1),
            state.arsenal.active_ability_root_slot,
        ),
        ability_elapsed_seconds=jnp.where(
            interrupted,
            jnp.float32(0.0),
            state.arsenal.ability_elapsed_seconds,
        ),
        ability_scheduler_tick=jnp.where(
            interrupted,
            jnp.int32(0),
            state.arsenal.ability_scheduler_tick,
        ),
        ability_scheduler_clock_seconds=jnp.where(
            interrupted[..., None],
            jnp.float32(0.0),
            state.arsenal.ability_scheduler_clock_seconds,
        ),
        ability_selector_hit_bits=jnp.where(
            interrupted,
            jnp.uint32(0),
            state.arsenal.ability_selector_hit_bits,
        ),
    )
    arsenal, mechanics, ability_requested, ability_accepted = start_abilities(
        prestart_arsenal,
        mechanics,
        requested_slot,
        runtime_loadout,
        commands.world,
        alive,
        config.mechanics_rules,
        admission_mechanics=state.mechanics,
    )
    start_valid = (
        (mechanics.failure_bits == 0)
        & (arsenal.failure_bits == 0)
        & (inventory.failure_bits == 0)
    )
    # Armed Arsenal owns attacks through its authored ability scheduler.  Do
    # not mirror an ability request onto the legacy ACTION_ATTACK bit: that
    # would advance two independent attack programs for one policy decision.
    # The null-loadout compatibility path retains the legacy role cycle.
    edge_actions = actions.at[:, ACTION_ATTACK].set(jnp.float32(0.0))
    flags, speed = status_modifiers(mechanics.statuses)
    agent_disabled = (
        flags[:, AGENT_ENTITY] & jnp.uint32(STATUS_FLAG_DISABLE_MOVEMENT)
    ) != 0
    edge_actions = _mask_agent_movement(edge_actions, agent_disabled)
    prepared, legacy_attack_requested, _, scan_inputs = _prepare_step(
        engaged_combat,
        edge_actions,
        keys,
        params,
        geometry,
        None,
        initially_terminated=_arsenal_stopped(
            engaged_combat,
            opponent_mask,
        ),
    )
    policy_desired_velocity = prepared.desired_velocity
    prepared = _select_tree(start_valid, prepared, state.combat)
    mechanics = _select_tree(start_valid, mechanics, state.mechanics)._replace(
        failure_bits=mechanics.failure_bits
    )
    arsenal = _select_tree(start_valid, arsenal, state.arsenal)._replace(
        failure_bits=arsenal.failure_bits
    )
    carry = ArsenalEnvironmentState(
        prepared,
        mechanics,
        arsenal,
        inventory,
        state.opponent_memory,
    )

    def scan_tick(current, inputs):
        (
            microtick_index,
            pause_draw,
            strafe_frequency_draw,
            strafe_duration_draw,
            strafe_direction_draw,
        ) = inputs

        def damage_keys(namespace: int) -> jax.Array:
            return jax.vmap(
                lambda key: jax.random.fold_in(
                    jax.random.fold_in(key, microtick_index),
                    namespace,
                )
            )(keys)

        can_tick = (
            (microtick_index < params.microticks)
            & ~_arsenal_stopped(current.combat, opponent_mask)
            & (current.mechanics.failure_bits == 0)
            & (current.arsenal.failure_bits == 0)
            & (current.inventory.failure_bits == 0)
        )
        motion_dt = (
            _motion_delta(
                current.combat.tick_count + 1,
                current.combat.motion_timing_profile,
                params,
            )
            if motion_delta_seconds is None
            else motion_delta_seconds[:, microtick_index]
        )
        # Native interaction operations use World.tickStepNanos (fixed 30 Hz)
        # even when physics, statuses, and CooldownHandler consume a measured
        # engine delta. Keep those clocks separate during both production
        # timing-profile selection and fidelity replay.
        interaction_dt = jnp.broadcast_to(
            jnp.asarray(params.nominal_dt, dtype=jnp.float32),
            motion_dt.shape,
        )
        # InteractionManager receives the fixed World tick step regardless of
        # whether a graph contains a player GameMode gate. Measured engine dt
        # belongs to motion, statuses, and cooldowns; allowing an inventory
        # node to switch the whole ability graph onto that clock creates
        # profile-dependent timing drift that does not exist server-side.
        ability_dt = jnp.broadcast_to(
            interaction_dt[:, None],
            current.arsenal.active_ability_slot.shape,
        )
        regenerated_mechanics = tick_resources(
            current.mechanics,
            config.mechanics_rules,
            motion_dt,
            jnp.ones_like(current.mechanics.resources, dtype=jnp.bool_),
        )
        (
            mechanics_candidate,
            defended_combat,
            incoming_components,
            incoming_damage,
            incoming_blocked,
            incoming_invulnerable,
            incoming_entity_dealt,
            incoming_entity_received,
        ) = _defend_pending_target_damage(
            current.combat,
            regenerated_mechanics,
            params,
            config.mechanics_rules,
            config.targeting_rules.team_id,
        )
        mechanics_candidate, status_tick = tick_statuses(
            mechanics_candidate,
            motion_dt,
            config.mechanics_rules,
        )
        status_damage = _status_damage_events(status_tick)
        mechanics_candidate = advance_defense_interactions(
            mechanics_candidate,
            defended_combat.yaw,
            defended_combat.health > 0.0,
            motion_dt,
            config.mechanics_rules,
            params,
        )
        entity_count = current.arsenal.active_ability_slot.shape[1]
        scheduler_after = advance_scheduler_clocks(
            current.arsenal.active_ability_slot >= 0,
            current.arsenal.ability_scheduler_tick,
            current.arsenal.ability_scheduler_clock_seconds,
            ability_dt,
        )
        if compiled_has_item_programs is False:
            item_arsenal = current.arsenal
            item_inventory = current.inventory
        else:
            item_execution = execute_due_item_programs(
                current.arsenal,
                current.inventory,
                config.inventory_layout,
                config.item_programs,
                scheduler_after,
                resources=mechanics_candidate.resources,
                resource_maximum=config.mechanics_rules.resource_maximum,
                continuation_phase=False,
                player_backed_actor_mask=config.player_backed_actor_mask,
            )
            suppress_children = item_execution.suppress_all_events
            delayed_failure = suppress_children & (
                item_execution.failure_delay_seconds > jnp.float32(0.0)
            )
            immediate_suppression = suppress_children & ~delayed_failure
            deferred_item_arsenal = defer_item_program_completion(
                item_execution.arsenal,
                delayed_failure,
                item_execution.failure_delay_seconds,
                runtime_loadout.ability_duration_seconds,
                ability_dt,
                advance_occurs_this_tick=True,
            )
            item_arsenal = deferred_item_arsenal._replace(
                active_ability_slot=jnp.where(
                    immediate_suppression,
                    jnp.int32(-1),
                    deferred_item_arsenal.active_ability_slot,
                ),
                active_ability_root_slot=jnp.where(
                    immediate_suppression,
                    jnp.int32(-1),
                    deferred_item_arsenal.active_ability_root_slot,
                ),
                ability_elapsed_seconds=jnp.where(
                    immediate_suppression,
                    jnp.float32(0.0),
                    deferred_item_arsenal.ability_elapsed_seconds,
                ),
                ability_scheduler_tick=jnp.where(
                    immediate_suppression,
                    jnp.int32(0),
                    deferred_item_arsenal.ability_scheduler_tick,
                ),
                ability_scheduler_clock_seconds=jnp.where(
                    immediate_suppression[..., None],
                    jnp.float32(0.0),
                    deferred_item_arsenal.ability_scheduler_clock_seconds,
                ),
                ability_selector_hit_bits=jnp.where(
                    immediate_suppression,
                    jnp.uint32(0),
                    deferred_item_arsenal.ability_selector_hit_bits,
                ),
            )
            item_inventory = item_execution.inventory
        mechanics_candidate = spend_due_ability_resources(
            item_arsenal,
            mechanics_candidate,
            runtime_loadout,
            config.mechanics_rules,
            ability_dt,
            config.player_backed_actor_mask,
        )
        status_combat = _apply_status_healing(
            defended_combat,
            status_tick.healing,
            params,
        )
        (
            mechanics_candidate,
            status_combat,
            status_components,
            status_dealt,
            status_received,
            status_blocked,
            status_invulnerable,
            status_entity_dealt,
            status_entity_received,
        ) = _resolve_damage(
            status_combat,
            mechanics_candidate,
            status_damage,
            config.mechanics_rules,
            opponent_mask,
            config.targeting_rules.team_id,
            params,
            random_keys=damage_keys(0),
        )
        item_inventory = _apply_attacker_tool_durability(
            item_inventory,
            config,
            _attacker_tool_hit_counts(
                status_damage,
                config.mechanics_rules,
                entity_count,
            ),
        )
        resources_before_direct_events = mechanics_candidate.resources
        # C4 reads the grounded flag carried *into* this tick: the motion
        # integrator that recomputes it runs further below, so consulting it
        # here would make a continuation depend on execution order within the
        # tick. `agent_grounded` describes AGENT_ENTITY alone, so it is placed
        # on that row rather than broadcast across entities it does not
        # describe; other rows stay False until a per-entity grounded signal
        # exists in this state tree.
        continuation_grounded = (
            jnp.zeros_like(item_arsenal.active_ability_slot, dtype=jnp.bool_)
            .at[:, AGENT_ENTITY]
            .set(status_combat.agent_grounded)
        )
        arsenal_candidate, fired = tick_ability_programs(
            item_arsenal,
            runtime_loadout,
            ability_dt,
            motion_dt,
            config.player_backed_actor_mask,
            entity_grounded=continuation_grounded,
        )
        (
            mechanics_candidate,
            arsenal_candidate,
            direct_health,
            direct_damage,
            event_count,
        ) = process_direct_events(
            status_combat,
            mechanics_candidate,
            arsenal_candidate,
            fired,
            commands.world,
            params,
            config.mechanics_rules,
            engagement_target_id=decision_targeting.engagement_target_id,
        )
        # Public outer roots can expose a temporary stat while their selected
        # child is pending. Apply it after the child graph's own events so the
        # root-owned value is authoritative for the completed native tick.
        mechanics_candidate = apply_ability_resource_phases(
            arsenal_candidate,
            mechanics_candidate,
            runtime_loadout,
        )
        mechanics_candidate = apply_ability_stamina_regen_delay_phases(
            current.arsenal,
            mechanics_candidate,
            runtime_loadout,
        )
        # Native Crossbow reload commits each Ammo stat event before the next
        # Repeat iteration consumes an arrow at the same authored boundary.
        # Keep that ordering explicit: the initial item node runs before the
        # ability graph above, while compact Repeat continuations run after
        # the graph has committed this tick's resource event.
        if compiled_has_item_programs is False:
            next_inventory = item_inventory
        else:
            continuation_execution = execute_due_item_programs(
                item_arsenal,
                item_inventory,
                config.inventory_layout,
                config.item_programs,
                scheduler_after,
                resources=mechanics_candidate.resources,
                resource_maximum=config.mechanics_rules.resource_maximum,
                continuation_phase=True,
                player_backed_actor_mask=config.player_backed_actor_mask,
            )
            continuation_suppresses = continuation_execution.suppress_all_events
            arsenal_candidate = arsenal_candidate._replace(
                active_ability_slot=jnp.where(
                    continuation_suppresses,
                    jnp.int32(-1),
                    arsenal_candidate.active_ability_slot,
                ),
                active_ability_root_slot=jnp.where(
                    continuation_suppresses,
                    jnp.int32(-1),
                    arsenal_candidate.active_ability_root_slot,
                ),
                ability_elapsed_seconds=jnp.where(
                    continuation_suppresses,
                    jnp.float32(0.0),
                    arsenal_candidate.ability_elapsed_seconds,
                ),
                ability_scheduler_tick=jnp.where(
                    continuation_suppresses,
                    jnp.int32(0),
                    arsenal_candidate.ability_scheduler_tick,
                ),
                ability_scheduler_clock_seconds=jnp.where(
                    continuation_suppresses[..., None],
                    jnp.float32(0.0),
                    arsenal_candidate.ability_scheduler_clock_seconds,
                ),
                ability_selector_hit_bits=jnp.where(
                    continuation_suppresses,
                    jnp.uint32(0),
                    arsenal_candidate.ability_selector_hit_bits,
                ),
            )
            arsenal_candidate = complete_resource_bound_item_programs(
                arsenal_candidate,
                config.item_programs,
                resources_before_direct_events,
                mechanics_candidate.resources,
                config.mechanics_rules.resource_maximum,
                runtime_loadout.ability_duration_seconds,
                ability_dt,
            )
            next_inventory = continuation_execution.inventory
        direct_combat = status_combat._replace(health=direct_health)
        # Consume explicit ApplyForce events and force state retained from the
        # preceding tick before resolving this tick's damage packet. Native
        # damage writes KnockbackComponent through a command buffer after the
        # current motion consumer, so damage-authored force remains staged for
        # the following tick while a direct ApplyForce keeps its measured
        # same-tick boundary.
        applied_motion_velocity = mechanics_candidate.applied_velocity
        applied_motion_active = jnp.any(
            applied_motion_velocity != jnp.float32(0.0),
            axis=2,
        ) | (mechanics_candidate.external_velocity_y != jnp.float32(0.0))
        applied_motion_velocity = project_applied_motion_velocity(mechanics_candidate)
        grounded_before, ground_before_exhausted = _applied_motion_grounded(
            direct_combat.position,
            applied_motion_active,
            geometry,
            params,
        )
        # The provider probe is the physical contact authority for every NPC.
        # ``agent_grounded`` can lag while an applied-force branch owns motion;
        # substituting it here lets downward gravity accumulate on solid ground.
        mechanics_candidate, moved_position = translate_applied_motion(
            mechanics_candidate,
            direct_combat.position,
            grounded_before,
            motion_dt,
            params,
        )
        moved_position, motion_geometry_exhausted = _resolve_applied_motion_geometry(
            direct_combat.position,
            moved_position,
            applied_motion_active,
            geometry,
        )
        if geometry is None:
            entity_id = jnp.arange(entity_count, dtype=jnp.int32)[None, :]
            clamp_to_floor = (
                applied_motion_active
                & (entity_id != jnp.int32(AGENT_ENTITY))
                & (moved_position[..., 1] < params.floor_y)
            )
            moved_position = moved_position.at[..., 1].set(
                jnp.where(
                    clamp_to_floor,
                    params.floor_y,
                    moved_position[..., 1],
                )
            )
        applied_motion_velocity = resolve_applied_motion_velocity(
            applied_motion_velocity,
            direct_combat.position,
            moved_position,
            motion_dt,
        )
        grounded_after, ground_after_exhausted = _applied_motion_grounded(
            moved_position,
            applied_motion_active,
            geometry,
            params,
        )
        mechanics_candidate = damp_applied_motion(
            mechanics_candidate,
            grounded_after,
            motion_dt,
            config.mechanics_rules,
            params,
        )
        motion_geometry_exhausted = (
            motion_geometry_exhausted | ground_before_exhausted | ground_after_exhausted
        )
        direct_combat = direct_combat._replace(position=moved_position)
        (
            mechanics_candidate,
            direct_combat,
            direct_components,
            direct_dealt,
            direct_received,
            direct_blocked,
            direct_invulnerable,
            direct_entity_dealt,
            direct_entity_received,
        ) = _resolve_damage(
            direct_combat,
            mechanics_candidate,
            direct_damage,
            config.mechanics_rules,
            opponent_mask,
            config.targeting_rules.team_id,
            params,
            random_keys=damage_keys(1),
        )
        next_inventory = _apply_attacker_tool_durability(
            next_inventory,
            config,
            _attacker_tool_hit_counts(
                direct_damage,
                config.mechanics_rules,
                entity_count,
            ),
        )
        if compiled_has_areas is False:
            area_spawned = jnp.zeros((batch,), dtype=jnp.int32)
        else:
            arsenal_candidate, area_spawned = spawn_areas(
                arsenal_candidate,
                fired,
                commands.world,
            )
        arsenal_candidate = arsenal_candidate._replace(
            failure_bits=jnp.where(
                motion_geometry_exhausted,
                arsenal_candidate.failure_bits
                | jnp.uint32(ARSENAL_FAILURE_UNSUPPORTED_WORLD),
                arsenal_candidate.failure_bits,
            )
        )
        motion_combat = direct_combat
        current_flags, current_speed = status_modifiers(mechanics_candidate.statuses)
        movement_disabled = (
            current_flags & jnp.uint32(STATUS_FLAG_DISABLE_MOVEMENT)
        ) != 0
        motion_combat = motion_combat._replace(
            desired_velocity=jnp.where(
                (
                    movement_disabled[:, AGENT_ENTITY]
                    | applied_motion_active[:, AGENT_ENTITY]
                )[:, None],
                jnp.float32(0.0),
                policy_desired_velocity * current_speed[:, AGENT_ENTITY, None],
            )
        )
        base_candidate, base_components = _microtick(
            motion_combat,
            pause_draw,
            strafe_frequency_draw,
            strafe_duration_draw,
            strafe_direction_draw,
            params,
            geometry,
            None,
            target_navigation_provider,
            motion_dt,
        )
        pending_hit = current.combat.target_damage_pending
        base_candidate = base_candidate._replace(
            ticks_since_agent_damage=jnp.where(
                pending_hit,
                jnp.int32(0),
                base_candidate.ticks_since_agent_damage,
            ),
            target_damage_applied_this_tick=(
                base_candidate.target_damage_applied_this_tick | pending_hit
            ),
        )
        base_candidate = _apply_target_controls(
            motion_combat,
            base_candidate,
            current_flags,
            current_speed,
            runtime_loadout.equipped[:, TARGET_ENTITY],
        )
        opponent_memory_candidate = step_opponent_memory(
            current.opponent_memory,
            base_candidate.position,
            base_candidate.health,
            decision_targeting,
            commands.world.target_candidate_perceptible,
            commands.world.target_candidate_perception_valid,
            candidate_heard,
            config.opponent_controller_mask,
            distance_component_selector=(
                config.targeting_rules.distance_component_selector
            ),
            chase_stop_distance=params.target_chase_stop_distance,
            chase_view_range=params.target_chase_view_range,
            delta_seconds=motion_dt,
        )
        base_candidate = base_candidate._replace(
            target_navigation_unsupported=(
                base_candidate.target_navigation_unsupported
                | jnp.any(
                    opponent_memory_candidate.navigation_unavailable
                    & config.opponent_controller_mask,
                    axis=1,
                )
            )
        )
        # MotionControllerWalk takes its external-force branch whenever either
        # forceVelocity or a split velocity is active. That branch suppresses
        # ordinary steering and publishes the applied translation itself.
        base_candidate = base_candidate._replace(
            position=jnp.where(
                applied_motion_active[..., None],
                motion_combat.position,
                base_candidate.position,
            ),
            velocity=jnp.where(
                applied_motion_active[..., None],
                applied_motion_velocity,
                base_candidate.velocity,
            ),
            yaw=jnp.where(
                applied_motion_active,
                motion_combat.yaw,
                base_candidate.yaw,
            ),
        )
        environment_packet = environment_damage_events(
            geometry,
            base_candidate.position,
            base_candidate.health,
        )
        base_candidate = base_candidate._replace(
            geometry_exhausted=(
                base_candidate.geometry_exhausted
                | environment_packet.geometry_exhausted
            )
        )
        (
            mechanics_candidate,
            environment_combat,
            environment_components,
            environment_dealt,
            environment_received,
            environment_blocked,
            environment_invulnerable,
            environment_entity_dealt,
            environment_entity_received,
        ) = _resolve_damage(
            base_candidate,
            mechanics_candidate,
            environment_packet.events,
            config.mechanics_rules,
            opponent_mask,
            config.targeting_rules.team_id,
            params,
        )
        if compiled_has_projectiles is False:
            projectile_damage = empty_damage_events(
                batch,
                state.arsenal.projectiles.active.shape[1],
            )
            projectile_status = empty_status_applications(
                batch,
                entity_count=entity_count,
            )
            projectile_hits = jnp.zeros((batch,), dtype=jnp.int32)
            projectile_spawned = jnp.zeros((batch,), dtype=jnp.int32)
            crossbow_commands = empty_crossbow_projectile_impact_commands(
                batch,
                state.arsenal.projectiles.active.shape[1],
            )
        else:
            (
                arsenal_candidate,
                projectile_damage,
                projectile_status,
                projectile_hits,
                crossbow_commands,
            ) = tick_projectiles(
                environment_combat,
                arsenal_candidate,
                interaction_dt,
                params,
                config.mechanics_rules,
                entity_team_id=config.targeting_rules.team_id,
                entity_projectile_collidable_mask=(
                    config.entity_projectile_collidable_mask
                ),
                friendly_fire=True,
                geometry=geometry,
                explosion_candidate_provider=explosion_candidate_provider,
                include_crossbow_program=True,
            )
            # CommandBuffer materializes projectiles after physics this tick.
            arsenal_candidate, projectile_spawned = spawn_projectiles(
                arsenal_candidate,
                fired,
                commands.world,
                config.loadout,
            )
        maximum_health = (
            jnp.full_like(
                environment_combat.health,
                params.target_max_health,
            )
            .at[:, AGENT_ENTITY]
            .set(params.agent_max_health)
        )
        (
            environment_combat,
            mechanics_candidate,
            crossbow_info,
            resolved_crossbow_commands,
        ) = apply_crossbow_projectile_impacts(
            environment_combat,
            mechanics_candidate,
            config.mechanics_rules,
            crossbow_commands,
            team_id=config.targeting_rules.team_id,
            maximum_health=maximum_health,
        )
        next_inventory = _apply_attacker_tool_durability(
            next_inventory,
            config,
            _crossbow_attacker_tool_hit_counts(
                resolved_crossbow_commands,
                crossbow_info,
                entity_count,
            ),
        )
        arsenal_candidate = arsenal_candidate._replace(
            failure_bits=jnp.where(
                crossbow_info.valid,
                arsenal_candidate.failure_bits,
                arsenal_candidate.failure_bits
                | jnp.uint32(ARSENAL_FAILURE_INVALID_STATE),
            )
        )
        if compiled_has_areas is False:
            area_damage = empty_damage_events(
                batch,
                state.arsenal.areas.active.shape[1],
            )
            area_status = empty_status_applications(
                batch,
                entity_count=entity_count,
            )
            area_hits = jnp.zeros((batch,), dtype=jnp.int32)
        else:
            (
                arsenal_candidate,
                area_damage,
                area_status,
                area_hits,
            ) = tick_areas(
                environment_combat,
                arsenal_candidate,
                motion_dt,
                params,
                config.mechanics_rules,
                entity_team_id=config.targeting_rules.team_id,
                deployable_spatial_roster_and_group_equivalence_attested=(
                    config.deployable_spatial_roster_and_group_equivalence_attested
                ),
                deployable_area_effect_eligible_mask=(
                    config.deployable_area_effect_eligible_mask
                ),
                deployable_area_effect_candidates_available=(
                    config.deployable_area_effect_candidates_available
                ),
            )
        if compiled_has_projectiles is False and compiled_has_areas is False:
            # The breathing resolver below still publishes terminal awards.
            effect_combat = environment_combat
            effect_dealt = effect_received = jnp.zeros((batch,), dtype=jnp.float32)
            effect_blocked = effect_invulnerable = jnp.zeros((batch,), dtype=jnp.int32)
            effect_components = RewardComponents(
                target_damage=effect_dealt,
                agent_damage=effect_received,
                completion=jnp.zeros((batch,), dtype=jnp.bool_),
                death=jnp.zeros((batch,), dtype=jnp.bool_),
            )
            effect_entity_dealt = effect_entity_received = jnp.zeros_like(
                environment_combat.health
            )
        else:
            combined_damage = _concat_damage(
                projectile_damage,
                area_damage,
            )
            (
                mechanics_candidate,
                effect_combat,
                effect_components,
                effect_dealt,
                effect_received,
                effect_blocked,
                effect_invulnerable,
                effect_entity_dealt,
                effect_entity_received,
            ) = _resolve_damage(
                environment_combat,
                mechanics_candidate,
                combined_damage,
                config.mechanics_rules,
                opponent_mask,
                config.targeting_rules.team_id,
                params,
                random_keys=damage_keys(2),
            )
            next_inventory = _apply_attacker_tool_durability(
                next_inventory,
                config,
                _attacker_tool_hit_counts(
                    combined_damage,
                    config.mechanics_rules,
                    entity_count,
                ),
            )
        (
            crossbow_entity_dealt,
            crossbow_entity_received,
        ) = _crossbow_damage_totals(
            resolved_crossbow_commands,
            crossbow_info,
            config.targeting_rules.team_id,
            entity_count,
        )
        crossbow_target_damage = crossbow_entity_dealt[:, AGENT_ENTITY]
        crossbow_agent_damage = crossbow_entity_received[:, AGENT_ENTITY]
        effect_components = effect_components._replace(
            target_damage=(effect_components.target_damage + crossbow_target_damage),
            agent_damage=(effect_components.agent_damage + crossbow_agent_damage),
        )
        effect_dealt = effect_dealt + crossbow_target_damage
        effect_received = effect_received + crossbow_agent_damage
        effect_blocked = effect_blocked + jnp.sum(
            crossbow_info.blocked.astype(jnp.int32),
            axis=1,
        )
        effect_invulnerable = effect_invulnerable + jnp.sum(
            crossbow_info.invulnerable.astype(jnp.int32),
            axis=1,
        )
        effect_entity_dealt = effect_entity_dealt + crossbow_entity_dealt
        effect_entity_received = effect_entity_received + crossbow_entity_received
        effect_combat = effect_combat._replace(
            ticks_since_agent_damage=jnp.where(
                crossbow_agent_damage > 0.0,
                jnp.int32(0),
                effect_combat.ticks_since_agent_damage,
            ),
            target_damage_applied_this_tick=(
                effect_combat.target_damage_applied_this_tick
                | (crossbow_agent_damage > 0.0)
            ),
        )
        if compiled_has_projectiles is not False:
            mechanics_candidate = apply_statuses(
                mechanics_candidate,
                projectile_status,
            )
        if compiled_has_areas is not False:
            mechanics_candidate = apply_statuses(
                mechanics_candidate,
                area_status,
            )
        # World currently supplies submersion for the policy actor.  Keep the
        # mechanics input entity-shaped so future solid-material and target
        # evidence can bind without profile- or weapon-specific branches.
        cannot_breathe = (
            jnp.zeros_like(
                effect_combat.health,
                dtype=jnp.bool_,
            )
            .at[:, AGENT_ENTITY]
            .set(
                commands.world.actor_submersion_available
                & commands.world.actor_eyes_submerged
            )
        )
        mechanics_candidate, breathing_damage = tick_breathing(
            mechanics_candidate,
            config.mechanics_rules,
            motion_dt,
            cannot_breathe,
            cannot_breathe,
            effect_combat.health > 0.0,
        )
        (
            mechanics_candidate,
            breathing_combat,
            breathing_components,
            breathing_dealt,
            breathing_received,
            breathing_blocked,
            breathing_invulnerable,
            breathing_entity_dealt,
            breathing_entity_received,
        ) = _resolve_damage(
            effect_combat,
            mechanics_candidate,
            breathing_damage,
            config.mechanics_rules,
            opponent_mask,
            config.targeting_rules.team_id,
            params,
        )
        candidate = ArsenalEnvironmentState(
            breathing_combat,
            mechanics_candidate,
            arsenal_candidate,
            next_inventory,
            opponent_memory_candidate,
        )
        candidate_valid = (
            (candidate.mechanics.failure_bits == 0)
            & (candidate.arsenal.failure_bits == 0)
            & (candidate.inventory.failure_bits == 0)
        )
        accepted_tick = can_tick & candidate_valid
        next_state = _select_tree(accepted_tick, candidate, current)
        next_state = next_state._replace(
            mechanics=next_state.mechanics._replace(
                failure_bits=jnp.where(
                    can_tick,
                    candidate.mechanics.failure_bits,
                    current.mechanics.failure_bits,
                )
            ),
            arsenal=next_state.arsenal._replace(
                failure_bits=jnp.where(
                    can_tick,
                    candidate.arsenal.failure_bits,
                    current.arsenal.failure_bits,
                )
            ),
        )
        components = RewardComponents(
            target_damage=(
                base_components.target_damage
                + incoming_components.target_damage
                + status_components.target_damage
                + direct_components.target_damage
                + environment_components.target_damage
                + effect_components.target_damage
                + breathing_components.target_damage
            ),
            agent_damage=(
                base_components.agent_damage
                + incoming_components.agent_damage
                + status_components.agent_damage
                + direct_components.agent_damage
                + environment_components.agent_damage
                + effect_components.agent_damage
                + breathing_components.agent_damage
            ),
            completion=(
                base_components.completion
                | incoming_components.completion
                | status_components.completion
                | direct_components.completion
                | environment_components.completion
                | effect_components.completion
                | breathing_components.completion
            ),
            death=(
                base_components.death
                | incoming_components.death
                | status_components.death
                | direct_components.death
                | environment_components.death
                | effect_components.death
                | breathing_components.death
            ),
        )
        components = jax.tree_util.tree_map(
            lambda value: jnp.where(
                accepted_tick,
                value,
                jnp.zeros_like(value),
            ),
            components,
        )
        # Legacy low-level melee remains entity-zero-only. Preserve that
        # compatibility reward on row zero; all Arsenal/mechanics packets use
        # their recorded source and target IDs below.
        base_entity_dealt = (
            jnp.zeros_like(current.combat.health)
            .at[:, AGENT_ENTITY]
            .set(base_components.target_damage)
        )
        base_entity_received = (
            jnp.zeros_like(current.combat.health)
            .at[:, AGENT_ENTITY]
            .set(base_components.agent_damage)
        )
        entity_damage_dealt = (
            base_entity_dealt
            + incoming_entity_dealt
            + status_entity_dealt
            + direct_entity_dealt
            + environment_entity_dealt
            + effect_entity_dealt
            + breathing_entity_dealt
        )
        entity_damage_received = (
            base_entity_received
            + incoming_entity_received
            + status_entity_received
            + direct_entity_received
            + environment_entity_received
            + effect_entity_received
            + breathing_entity_received
        )
        outputs = (
            components,
            jnp.where(accepted_tick, event_count, jnp.int32(0)),
            jnp.where(accepted_tick, projectile_spawned, jnp.int32(0)),
            jnp.where(accepted_tick, area_spawned, jnp.int32(0)),
            jnp.where(accepted_tick, projectile_hits, jnp.int32(0)),
            jnp.where(accepted_tick, area_hits, jnp.int32(0)),
            jnp.where(
                accepted_tick,
                status_dealt
                + direct_dealt
                + environment_dealt
                + effect_dealt
                + breathing_dealt,
                jnp.float32(0.0),
            ),
            jnp.where(
                accepted_tick,
                incoming_damage
                + status_received
                + direct_received
                + environment_received
                + effect_received
                + breathing_received,
                jnp.float32(0.0),
            ),
            jnp.where(
                accepted_tick[:, None],
                entity_damage_dealt,
                jnp.float32(0.0),
            ),
            jnp.where(
                accepted_tick[:, None],
                entity_damage_received,
                jnp.float32(0.0),
            ),
            jnp.where(
                accepted_tick,
                incoming_blocked
                + status_blocked
                + direct_blocked
                + environment_blocked
                + effect_blocked
                + breathing_blocked,
                jnp.int32(0),
            ),
            jnp.where(
                accepted_tick,
                incoming_invulnerable
                + status_invulnerable
                + direct_invulnerable
                + environment_invulnerable
                + effect_invulnerable
                + breathing_invulnerable,
                jnp.int32(0),
            ),
        )
        return next_state, outputs

    if compiled_microticks == 1:
        final, first_outputs = scan_tick(
            carry,
            jax.tree.map(lambda value: value[0], scan_inputs),
        )
        outputs = jax.tree.map(
            lambda value: jnp.expand_dims(value, axis=0),
            first_outputs,
        )
    else:
        compiled_inputs = (
            scan_inputs
            if compiled_microticks is None
            else jax.tree.map(
                lambda value: value[:compiled_microticks],
                scan_inputs,
            )
        )
        final, outputs = jax.lax.scan(scan_tick, carry, compiled_inputs)
    (
        micro_components,
        event_count,
        projectile_spawned,
        area_spawned,
        projectile_hits,
        area_hits,
        damage_dealt,
        damage_received,
        entity_damage_dealt,
        entity_damage_received,
        blocked_hits,
        invulnerable_hits,
    ) = outputs
    failure_bits = (
        final.arsenal.failure_bits
        | final.mechanics.failure_bits
        | final.inventory.failure_bits
    )
    valid = failure_bits == jnp.uint32(0)
    terminated = _arsenal_terminated(final.combat, opponent_mask) & valid
    truncated = (~valid | _arsenal_geometry_truncated(final.combat)) & ~terminated
    if entity_count == ENTITY_COUNT:
        observation = observe_batch(final.combat, params, geometry)
    else:
        observation = observe_batch(
            final.combat,
            params,
            geometry,
            target_entity_id=(decision_targeting.engagement_target_id[:, AGENT_ENTITY]),
        )
    reward_components = sum_reward_components(micro_components)
    reward_components = jax.tree_util.tree_map(
        lambda value: jnp.where(valid, value, jnp.zeros_like(value)),
        reward_components,
    )
    combat_info = _combat_info(
        final.combat,
        ability_requested[:, AGENT_ENTITY] & valid,
        ability_accepted[:, AGENT_ENTITY] & valid,
        params,
        geometry,
        reward_components,
    )
    arsenal_info = ArsenalInfo(
        legacy_attack_requested=legacy_attack_requested & valid,
        ability_requested=ability_requested & valid[:, None],
        ability_accepted=ability_accepted & valid[:, None],
        active_ability_slot=project_observable_active_slot(
            runtime_loadout,
            final.arsenal.active_ability_slot,
            final.arsenal.active_ability_root_slot,
        ),
        event_count=jnp.sum(event_count, axis=0, dtype=jnp.int32),
        projectile_count=jnp.sum(
            final.arsenal.projectiles.active.astype(jnp.int32),
            axis=1,
        ),
        area_count=jnp.sum(
            final.arsenal.areas.active.astype(jnp.int32),
            axis=1,
        ),
        damage_dealt=jnp.sum(damage_dealt, axis=0, dtype=jnp.float32),
        damage_received=jnp.sum(
            damage_received,
            axis=0,
            dtype=jnp.float32,
        ),
        entity_damage_dealt=jnp.sum(
            entity_damage_dealt,
            axis=0,
            dtype=jnp.float32,
        ),
        entity_damage_received=jnp.sum(
            entity_damage_received,
            axis=0,
            dtype=jnp.float32,
        ),
        blocked_hits=jnp.sum(blocked_hits, axis=0, dtype=jnp.int32),
        invulnerable_hits=jnp.sum(
            invulnerable_hits,
            axis=0,
            dtype=jnp.int32,
        ),
        failure_bits=failure_bits,
        valid=valid,
    )
    return ArsenalTransition(
        state=final,
        observation=observation,
        reward=jnp.where(
            valid,
            compose_reward(reward_components, params),
            jnp.float32(0.0),
        ),
        terminated=terminated,
        truncated=truncated,
        combat_info=combat_info,
        arsenal_info=arsenal_info,
    )


def _step_null_arsenal_batch(
    state: ArsenalEnvironmentState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    config: ArsenalRuntimeConfig,
    geometry: GeometryProvider | None,
    target_navigation_provider: TargetNavigationProvider | None,
    motion_delta_seconds: jax.Array | None,
) -> ArsenalTransition:
    """Preserve melee behavior while applying shared World consequences."""

    combat, observation, reward, _done, combat_info = _step_batch(
        state.combat,
        actions,
        keys,
        params,
        geometry,
        target_navigation_provider=target_navigation_provider,
        motion_delta_seconds=motion_delta_seconds,
    )
    environment_packet = environment_damage_events(
        geometry,
        combat.position,
        combat.health,
    )
    combat = combat._replace(
        geometry_exhausted=(
            combat.geometry_exhausted | environment_packet.geometry_exhausted
        )
    )
    (
        mechanics,
        combat,
        environment_components,
        environment_dealt,
        environment_received,
        environment_blocked,
        environment_invulnerable,
        environment_entity_dealt,
        environment_entity_received,
    ) = _resolve_damage(
        combat,
        state.mechanics,
        environment_packet.events,
        config.mechanics_rules,
        config.opponent_controller_mask,
        config.targeting_rules.team_id,
        params,
    )
    reward_components = jax.tree_util.tree_map(
        lambda prior, environment: prior + environment,
        combat_info.reward_components,
        environment_components,
    )
    observation = observe_batch(combat, params, geometry)
    reward = compose_reward(reward_components, params)
    combat_info = _combat_info(
        combat,
        combat_info.attack_requested,
        combat_info.attack_accepted,
        params,
        geometry,
        reward_components,
    )
    inventory_failure = inventory_state_failure_bits(
        state.inventory,
        config.inventory_layout,
    )
    failure_bits = (
        state.arsenal.failure_bits | mechanics.failure_bits | inventory_failure
    )
    valid = failure_bits == jnp.uint32(0)
    batch, entity_count = combat.position.shape[:2]
    entity = (batch, entity_count)
    terminated = _arsenal_terminated(combat, _agent_opponent_mask(config)) & valid
    truncated = (~valid | _arsenal_geometry_truncated(combat)) & ~terminated
    return ArsenalTransition(
        state=state._replace(
            combat=combat,
            mechanics=mechanics,
            inventory=state.inventory._replace(
                failure_bits=inventory_failure,
            ),
        ),
        observation=observation,
        reward=jnp.where(valid, reward, jnp.float32(0.0)),
        terminated=terminated,
        truncated=truncated,
        combat_info=combat_info,
        arsenal_info=ArsenalInfo(
            legacy_attack_requested=combat_info.attack_requested & valid,
            ability_requested=jnp.zeros(entity, dtype=jnp.bool_),
            ability_accepted=jnp.zeros(entity, dtype=jnp.bool_),
            active_ability_slot=project_observable_active_slot(
                config.loadout,
                state.arsenal.active_ability_slot,
                state.arsenal.active_ability_root_slot,
            ),
            event_count=jnp.zeros((batch,), dtype=jnp.int32),
            projectile_count=jnp.zeros((batch,), dtype=jnp.int32),
            area_count=jnp.zeros((batch,), dtype=jnp.int32),
            damage_dealt=environment_dealt,
            damage_received=environment_received,
            entity_damage_dealt=environment_entity_dealt,
            entity_damage_received=environment_entity_received,
            blocked_hits=environment_blocked,
            invulnerable_hits=environment_invulnerable,
            failure_bits=failure_bits,
            valid=valid,
        ),
    )


def rollout_arsenal_batch(
    state: ArsenalEnvironmentState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    commands: ArsenalCommands,
    config: ArsenalRuntimeConfig,
    geometry: GeometryProvider | None = None,
    target_navigation_provider: TargetNavigationProvider | None = None,
) -> tuple[ArsenalEnvironmentState, ArsenalTrajectory]:
    """Scan a complete fixed-shape arsenal rollout without Python stepping."""

    def scan_step(current, inputs):
        action, key, command = inputs
        transition = step_arsenal_batch(
            current,
            action,
            key,
            params,
            command,
            config,
            geometry,
            target_navigation_provider,
        )
        outputs = ArsenalTrajectory(
            observation=transition.observation,
            reward=transition.reward,
            terminated=transition.terminated,
            truncated=transition.truncated,
            combat_info=transition.combat_info,
            arsenal_info=transition.arsenal_info,
        )
        return transition.state, outputs

    return jax.lax.scan(scan_step, state, (actions, keys, commands))


def combat_target_selection(
    combat: CombatState,
    params: CombatParams,
    config: ArsenalRuntimeConfig,
    *,
    candidate_evidence: jax.Array | None = None,
) -> CombatTargetSelection:
    """Apply controller metrics after optional actor-legal candidate evidence."""

    sensor_range = config.targeting_rules.sensor_range
    if sensor_range.shape[1] != ENTITY_COUNT:
        sensor_range = jnp.minimum(
            sensor_range,
            jnp.broadcast_to(
                params.sensor_range,
                sensor_range.shape,
            ),
        )
    rules = config.targeting_rules._replace(sensor_range=sensor_range)
    return select_combat_targets(
        combat.position,
        combat.health,
        rules,
        candidate_capacity=TARGET_CANDIDATE_CAPACITY,
        candidate_evidence=candidate_evidence,
        locked_target_id=combat.engagement_target_id,
    )


def _agent_opponent_mask(config: ArsenalRuntimeConfig) -> jax.Array:
    teams = config.targeting_rules.team_id
    source_team = teams[:, AGENT_ENTITY, None]
    entity_id = jnp.arange(teams.shape[1], dtype=jnp.int32)[None, :]
    return (entity_id != AGENT_ENTITY) & (
        (source_team == jnp.int32(TEAM_NONE)) | (teams != source_team)
    )


def _arsenal_terminated(
    combat: CombatState,
    opponent_mask: jax.Array,
) -> jax.Array:
    """Return natural combat endings only."""

    return (combat.health[:, AGENT_ENTITY] <= 0.0) | ~jnp.any(
        opponent_mask & (combat.health > 0.0), axis=1
    )


def _arsenal_geometry_truncated(combat: CombatState) -> jax.Array:
    # A rejected NPC navigation step is recoverable: that actor stays put for
    # the step and the sticky bit remains observable. Geometry exhaustion is
    # different because no actor transition can be certified beyond the atlas.
    return combat.geometry_exhausted


def _arsenal_stopped(
    combat: CombatState,
    opponent_mask: jax.Array,
) -> jax.Array:
    """Stop unsafe execution without mislabelling support faults as deaths."""

    return _arsenal_terminated(combat, opponent_mask) | _arsenal_geometry_truncated(
        combat
    )


def _defend_pending_target_damage(combat, mechanics, params, rules, team_id):
    batch = combat.position.shape[0]
    pending = combat.target_damage_pending
    profile = jnp.clip(
        combat.last_target_attack_index,
        0,
        params.target_damage.shape[0] - 1,
    )
    raw = jnp.where(
        pending,
        params.target_damage[profile],
        jnp.float32(0.0),
    )
    events = empty_damage_events(batch, 1)._replace(
        requested=pending[:, None],
        source_entity_id=jnp.full(
            (batch, 1),
            TARGET_ENTITY,
            dtype=jnp.int32,
        ),
        target_entity_id=jnp.full(
            (batch, 1),
            AGENT_ENTITY,
            dtype=jnp.int32,
        ),
        amount=raw[:, None],
        cause=jnp.full((batch, 1), DAMAGE_PHYSICAL, dtype=jnp.int32),
        knockback_velocity=combat.pending_knockback_velocity[:, None, :],
    )
    mechanics, health, resolution = resolve_damage_events(
        mechanics,
        combat.health,
        combat.position,
        combat.yaw,
        events,
        rules,
    )
    actual = resolution.applied_damage[:, 0]
    entity_dealt, entity_received = _entity_damage_totals(
        resolution,
        team_id,
        combat.health.shape[1],
    )
    knockback, move_speed = _force_pushed_agent_motion(
        resolution.knockback_velocity[:, 0],
        combat.agent_move_speed,
        combat.yaw[:, AGENT_ENTITY],
        params,
    )
    knockback = knockback.at[:, 1].add(combat.agent_applied_vertical_velocity)
    velocity = combat.velocity.at[:, AGENT_ENTITY].set(
        jnp.where(
            pending[:, None],
            knockback,
            combat.velocity[:, AGENT_ENTITY],
        )
    )
    combat = combat._replace(
        health=health,
        velocity=velocity,
        agent_move_speed=jnp.where(
            pending,
            move_speed,
            combat.agent_move_speed,
        ),
        target_damage_pending=jnp.where(pending, False, combat.target_damage_pending),
        pending_knockback_velocity=jnp.where(
            pending[:, None],
            jnp.float32(0.0),
            combat.pending_knockback_velocity,
        ),
        ticks_since_agent_damage=jnp.where(
            pending,
            jnp.int32(0),
            combat.ticks_since_agent_damage,
        ),
        vertical_impulse_applied=jnp.where(
            pending,
            True,
            combat.vertical_impulse_applied,
        ),
        grounded_with_residual_velocity=jnp.where(
            pending,
            False,
            combat.grounded_with_residual_velocity,
        ),
        knockback_control_lock=jnp.where(
            pending,
            True,
            combat.knockback_control_lock,
        ),
        agent_grounded=jnp.where(
            pending,
            False,
            combat.agent_grounded,
        ),
    )
    return (
        mechanics,
        combat,
        RewardComponents(
            target_damage=jnp.zeros_like(actual),
            agent_damage=actual,
            completion=jnp.zeros_like(pending),
            death=(health[:, AGENT_ENTITY] <= 0.0) & ~combat.death_penalty_awarded,
        ),
        actual,
        resolution.blocked[:, 0].astype(jnp.int32),
        resolution.invulnerable[:, 0].astype(jnp.int32),
        entity_dealt,
        entity_received,
    )


def _resolve_damage(
    combat,
    mechanics,
    events,
    rules,
    opponent_mask,
    team_id,
    params,
    *,
    random_keys=None,
):
    before = combat.health
    mechanics, damage_health, resolution = resolve_damage_events(
        mechanics,
        before,
        combat.position,
        combat.yaw,
        events,
        rules,
        random_keys=random_keys,
    )
    mechanics = apply_damage_forces(mechanics, resolution)
    entity_dealt, entity_received = _entity_damage_totals(
        resolution,
        team_id,
        before.shape[1],
    )
    # Reward damage to opponents only when the policy actor actually sourced
    # it. Null-source Environment packets and damage dealt by another actor
    # remain visible in received totals without becoming free policy credit.
    target_damage = entity_dealt[:, AGENT_ENTITY]
    agent_damage = before[:, AGENT_ENTITY] - damage_health[:, AGENT_ENTITY]
    source = jnp.clip(
        resolution.source_entity_id,
        0,
        before.shape[1] - 1,
    )
    healing = jnp.sum(
        jax.nn.one_hot(
            source,
            before.shape[1],
            dtype=jnp.float32,
        )
        * resolution.on_hit_healing[..., None],
        axis=1,
        dtype=jnp.float32,
    )
    maximum_health = (
        jnp.full_like(
            damage_health,
            params.target_max_health,
        )
        .at[:, AGENT_ENTITY]
        .set(params.agent_max_health)
    )
    health = jnp.minimum(maximum_health, damage_health + healing)
    completion = (
        ~jnp.any(opponent_mask & (health > 0.0), axis=1) & ~combat.completion_awarded
    )
    death = (health[:, AGENT_ENTITY] <= 0.0) & ~combat.death_penalty_awarded
    combat = combat._replace(
        health=health,
        ticks_since_agent_damage=jnp.where(
            agent_damage > 0.0,
            jnp.int32(0),
            combat.ticks_since_agent_damage,
        ),
        target_damage_applied_this_tick=(
            combat.target_damage_applied_this_tick | (agent_damage > 0.0)
        ),
        completion_awarded=combat.completion_awarded | completion,
        death_penalty_awarded=combat.death_penalty_awarded | death,
    )
    components = RewardComponents(
        target_damage=target_damage,
        agent_damage=agent_damage,
        completion=completion,
        death=death,
    )
    return (
        mechanics,
        combat,
        components,
        target_damage,
        agent_damage,
        jnp.sum(resolution.blocked.astype(jnp.int32), axis=1),
        jnp.sum(resolution.invulnerable.astype(jnp.int32), axis=1),
        entity_dealt,
        entity_received,
    )


def _entity_damage_totals(
    resolution,
    team_id: jax.Array,
    entity_count: int,
) -> tuple[jax.Array, jax.Array]:
    """Reduce resolved packets without losing source/victim attribution."""

    teams = jnp.asarray(team_id, dtype=jnp.int32)
    batch = resolution.applied_damage.shape[0]
    if teams.shape != (batch, entity_count):
        raise ValueError("team_id must match the damage entity axis")
    source_valid = (
        resolution.requested
        & (resolution.source_entity_id >= 0)
        & (resolution.source_entity_id < entity_count)
    )
    target_valid = (
        resolution.requested
        & (resolution.target_entity_id >= 0)
        & (resolution.target_entity_id < entity_count)
    )
    source = jnp.clip(resolution.source_entity_id, 0, entity_count - 1)
    target = jnp.clip(resolution.target_entity_id, 0, entity_count - 1)
    batch_index = jnp.arange(batch, dtype=jnp.int32)[:, None]
    source_team = teams[batch_index, source]
    target_team = teams[batch_index, target]
    enemy = (
        (source_team == jnp.int32(TEAM_NONE))
        | (target_team == jnp.int32(TEAM_NONE))
        | (source_team != target_team)
    )
    applied = jnp.maximum(
        jnp.asarray(resolution.applied_damage, dtype=jnp.float32),
        jnp.float32(0.0),
    )
    dealt = (
        jnp.zeros((batch, entity_count), dtype=jnp.float32)
        .at[
            batch_index,
            source,
        ]
        .add(jnp.where(source_valid & target_valid & enemy, applied, 0.0))
    )
    received = (
        jnp.zeros((batch, entity_count), dtype=jnp.float32)
        .at[
            batch_index,
            target,
        ]
        .add(jnp.where(target_valid, applied, 0.0))
    )
    return dealt, received


def _apply_attacker_tool_durability(
    inventory: InventoryState,
    config: ArsenalRuntimeConfig,
    hit_count: jax.Array,
) -> InventoryState:
    """Apply asset-derived loss to the item held when native damage is inspected."""

    eligible = jnp.asarray(
        config.item_durability_eligible_actor_mask,
        dtype=jnp.bool_,
    )
    hit_count = jnp.asarray(hit_count, dtype=jnp.int32)
    if eligible.shape != hit_count.shape:
        raise ValueError(
            "item durability eligibility and hit counts must share [B,E] shape"
        )
    return apply_held_weapon_hit_durability_loss(
        inventory,
        config.inventory_layout,
        config.item_programs.weapon_item_id_table,
        config.item_programs.weapon_item_durability_loss_on_hit_table,
        jnp.where(eligible, hit_count, jnp.int32(0)),
    )


def _attacker_tool_hit_counts(
    events,
    rules,
    entity_count: int,
) -> jax.Array:
    """Count native ``DamageAttackerTool`` inspections by source entity.

    The engine inspector runs for a requested entity-sourced damage event when
    the resolved ``DamageCause.DurabilityLoss`` flag is true. It does not wait
    for positive applied damage, so a blocked or invulnerable impact still
    wears the attacker's currently held tool. The packaged cause table already
    materializes JSON ``Parent`` inheritance; no runtime parent walk belongs
    here.
    """

    requested = jnp.asarray(events.requested, dtype=jnp.bool_)
    source_id = jnp.asarray(events.source_entity_id, dtype=jnp.int32)
    target_id = jnp.asarray(events.target_entity_id, dtype=jnp.int32)
    cause_id = jnp.asarray(events.cause, dtype=jnp.int32)
    if requested.ndim != 2:
        raise ValueError("damage events must have shape [B,K]")
    if any(
        value.shape != requested.shape for value in (source_id, target_id, cause_id)
    ):
        raise ValueError("damage event identity leaves must match requested")
    batch = requested.shape[0]
    cause_flags = jnp.asarray(rules.cause_durability_loss, dtype=jnp.bool_)
    if cause_flags.ndim != 3 or cause_flags.shape[:2] != (batch, entity_count):
        raise ValueError("cause_durability_loss must have shape [B,E,DAMAGE_COUNT]")
    cause_count = cause_flags.shape[2]
    source_valid = (source_id >= 0) & (source_id < entity_count)
    target_valid = (target_id >= 0) & (target_id < entity_count)
    cause_valid = (cause_id >= 0) & (cause_id < cause_count)
    source = jnp.clip(source_id, 0, entity_count - 1)
    target = jnp.clip(target_id, 0, entity_count - 1)
    cause = jnp.clip(cause_id, 0, cause_count - 1)
    batch_index = jnp.arange(batch, dtype=jnp.int32)[:, None]
    loses_durability = cause_flags[batch_index, target, cause]
    inspected = requested & source_valid & target_valid & cause_valid & loses_durability
    return (
        jnp.zeros((batch, entity_count), dtype=jnp.int32)
        .at[
            batch_index,
            source,
        ]
        .add(inspected.astype(jnp.int32))
    )


def _crossbow_attacker_tool_hit_counts(
    commands,
    info,
    entity_count: int,
) -> jax.Array:
    """Count admitted native Projectile damage inspections by source entity.

    The shared Crossbow interaction adapter has already converted physical
    projectile contacts into bounded impact commands. Its authored native
    damage cause is ``Projectile``, whose resolved asset sets
    ``DurabilityLoss=true``. Count the contact even when guard or
    invulnerability reduces applied damage to zero, matching the inspect-stage
    Java system.
    """

    requested = jnp.asarray(commands.requested, dtype=jnp.bool_)
    source_id = jnp.asarray(commands.source_slot, dtype=jnp.int32)
    if requested.ndim != 2 or source_id.shape != requested.shape:
        raise ValueError("Crossbow impact commands must have shape [B,K]")
    batch = requested.shape[0]
    valid = jnp.asarray(info.valid, dtype=jnp.bool_)
    if valid.shape != (batch,):
        raise ValueError("Crossbow impact validity must have shape [B]")
    source_valid = (source_id >= 0) & (source_id < entity_count)
    source = jnp.clip(source_id, 0, entity_count - 1)
    batch_index = jnp.arange(batch, dtype=jnp.int32)[:, None]
    inspected = requested & valid[:, None] & source_valid
    return (
        jnp.zeros((batch, entity_count), dtype=jnp.int32)
        .at[
            batch_index,
            source,
        ]
        .add(inspected.astype(jnp.int32))
    )


def _crossbow_damage_totals(
    commands,
    info,
    team_id: jax.Array,
    entity_count: int,
) -> tuple[jax.Array, jax.Array]:
    """Reduce the shared target-local program into Arsenal reward facts."""

    teams = jnp.asarray(team_id, dtype=jnp.int32)
    batch = info.damage_applied.shape[0]
    if teams.shape != (batch, entity_count):
        raise ValueError("team_id must match the Crossbow entity axis")
    source_valid = (
        commands.requested
        & (commands.source_slot >= 0)
        & (commands.source_slot < entity_count)
    )
    target_valid = (
        commands.requested
        & (commands.target_slot >= 0)
        & (commands.target_slot < entity_count)
    )
    source = jnp.clip(commands.source_slot, 0, entity_count - 1)
    target = jnp.clip(commands.target_slot, 0, entity_count - 1)
    batch_index = jnp.arange(batch, dtype=jnp.int32)[:, None]
    source_team = teams[batch_index, source]
    target_team = teams[batch_index, target]
    enemy = (
        (source_team == jnp.int32(TEAM_NONE))
        | (target_team == jnp.int32(TEAM_NONE))
        | (source_team != target_team)
    )
    applied = jnp.maximum(info.damage_applied, jnp.float32(0.0))
    dealt = (
        jnp.zeros((batch, entity_count), dtype=jnp.float32)
        .at[
            batch_index,
            source,
        ]
        .add(jnp.where(source_valid & target_valid & enemy, applied, 0.0))
    )
    received = (
        jnp.zeros((batch, entity_count), dtype=jnp.float32)
        .at[
            batch_index,
            target,
        ]
        .add(jnp.where(target_valid, applied, 0.0))
    )
    return dealt, received


def _resolve_applied_motion_geometry(
    position: jax.Array,
    candidate_position: jax.Array,
    active: jax.Array,
    geometry: GeometryProvider | None,
) -> tuple[jax.Array, jax.Array]:
    """Slide active external-force translation through exact world geometry.

    Native ``MotionControllerWalk`` projects the remaining translation onto
    the first collision plane while retaining its persistent external
    velocity for later damping. The public World motion primitive implements
    that positional sweep; ``advance_applied_motion`` continues to own the
    independently sourced velocity lifetime.
    """

    batch, entity_count = position.shape[:2]
    if geometry is None:
        return candidate_position, jnp.zeros((batch,), dtype=jnp.bool_)

    def broadcast_bounds(bounds: jax.Array) -> jax.Array:
        if bounds.shape[0] == batch:
            return bounds
        return jnp.broadcast_to(bounds, (batch, 6))

    agent_bounds = broadcast_bounds(geometry.agent_bounds)
    target_bounds = broadcast_bounds(geometry.target_bounds)
    entity_bounds = (
        jnp.broadcast_to(
            target_bounds[:, None, :],
            (batch, entity_count, 6),
        )
        .at[:, AGENT_ENTITY]
        .set(agent_bounds)
    )

    def resolve(_):
        motion = jax.vmap(
            lambda current, displacement, bounds: (
                _resolve_aabb_motion_result(
                    geometry,
                    current,
                    displacement,
                    bounds,
                    # Native Walk advances to CollisionModule's exact
                    # collisionStart. The generic World safety skin leaves a
                    # visible 1e-5 gap and is not part of this controller.
                    skin_distance=0.0,
                )
            ),
            in_axes=(1, 1, 1),
            out_axes=1,
        )(
            position,
            candidate_position - position,
            entity_bounds,
        )
        return (
            jnp.where(active[..., None], motion.position, candidate_position),
            jnp.any(active & motion.geometry_exhausted, axis=1),
        )

    return jax.lax.cond(
        jnp.any(active),
        resolve,
        lambda _: (
            candidate_position,
            jnp.zeros((batch,), dtype=jnp.bool_),
        ),
        operand=None,
    )


def _applied_motion_grounded(
    position: jax.Array,
    active: jax.Array,
    geometry: GeometryProvider | None,
    params: CombatParams,
) -> tuple[jax.Array, jax.Array]:
    """Probe post-collision support with each entity's own native bounds."""

    batch, entity_count = position.shape[:2]
    if geometry is None:
        return (
            position[..., 1] <= params.floor_y + jnp.float32(1.0e-6),
            jnp.zeros((batch,), dtype=jnp.bool_),
        )

    def broadcast_bounds(bounds: jax.Array) -> jax.Array:
        if bounds.shape[0] == batch:
            return bounds
        return jnp.broadcast_to(bounds, (batch, 6))

    agent_bounds = broadcast_bounds(geometry.agent_bounds)
    target_bounds = broadcast_bounds(geometry.target_bounds)
    entity_bounds = (
        jnp.broadcast_to(
            target_bounds[:, None, :],
            (batch, entity_count, 6),
        )
        .at[:, AGENT_ENTITY]
        .set(agent_bounds)
    )
    displacement = (
        jnp.zeros_like(position).at[..., 1].set(-_APPLIED_MOTION_GROUND_PROBE_DISTANCE)
    )

    def resolve(_):
        motion = jax.vmap(
            lambda current, bounds, delta: _resolve_aabb_motion_result(
                geometry,
                current,
                delta,
                bounds,
            ),
            in_axes=(1, 1, 1),
            out_axes=1,
        )(
            position,
            entity_bounds,
            displacement,
        )
        return (
            motion.grounded,
            jnp.any(active & motion.geometry_exhausted, axis=1),
        )

    return jax.lax.cond(
        jnp.any(active),
        resolve,
        lambda _: (
            jnp.zeros_like(active),
            jnp.zeros((batch,), dtype=jnp.bool_),
        ),
        operand=None,
    )


def _status_damage_events(tick):
    batch = tick.requested.shape[0]
    count = tick.requested.shape[1] * tick.requested.shape[2]
    result = empty_damage_events(batch, count)
    return result._replace(
        requested=(tick.requested & (tick.damage > 0.0)).reshape(batch, count),
        source_entity_id=tick.source_entity_id.reshape(batch, count),
        target_entity_id=tick.target_entity_id.reshape(batch, count),
        amount=tick.damage.reshape(batch, count),
        cause=tick.damage_cause.reshape(batch, count),
    )


def _apply_status_healing(combat, healing, params):
    maximum = (
        jnp.full(
            combat.health.shape,
            params.target_max_health,
            dtype=jnp.float32,
        )
        .at[:, AGENT_ENTITY]
        .set(params.agent_max_health)
    )
    total = jnp.sum(healing, axis=2)
    return combat._replace(
        health=jnp.where(
            total != jnp.float32(0.0),
            jnp.minimum(maximum, combat.health + total),
            combat.health,
        )
    )


def _concat_damage(*events):
    return type(events[0])(
        *(
            jnp.concatenate(
                [getattr(event, field) for event in events],
                axis=1,
            )
            for field in events[0]._fields
        )
    )


def _apply_target_controls(
    before: CombatState,
    after: CombatState,
    flags,
    speed,
    arsenal_controlled,
):
    ability_disabled = (
        flags[:, TARGET_ENTITY] & jnp.uint32(STATUS_FLAG_DISABLE_ABILITIES)
    ) != 0
    # Entity-effect MovementEffects gate client input. Native NPC motion
    # controllers keep steering under DisableAll (Root), while NPCEntity does
    # consume the authored horizontal-speed multiplier. The policy-owned
    # agent is masked separately before _prepare_step.
    target_speed = speed[:, TARGET_ENTITY]
    target_position = (
        before.position[:, TARGET_ENTITY]
        + (after.position[:, TARGET_ENTITY] - before.position[:, TARGET_ENTITY])
        * target_speed[:, None]
    )
    position = after.position.at[:, TARGET_ENTITY].set(target_position)
    velocity = after.velocity.at[:, TARGET_ENTITY].set(
        after.velocity[:, TARGET_ENTITY] * target_speed[:, None]
    )
    suppress = ability_disabled | arsenal_controlled
    next_queue = jnp.where(
        arsenal_controlled,
        NO_QUEUE_TICK,
        after.tick_count + 1,
    )
    return after._replace(
        position=position,
        velocity=velocity,
        target_attack_index=jnp.where(
            suppress,
            jnp.int32(-1),
            after.target_attack_index,
        ),
        last_target_attack_index=jnp.where(
            suppress,
            jnp.int32(-1),
            after.last_target_attack_index,
        ),
        target_attack_elapsed_ticks=jnp.where(
            suppress,
            jnp.int32(0),
            after.target_attack_elapsed_ticks,
        ),
        target_attack_cooldown_seconds=jnp.where(
            suppress,
            jnp.float32(0.0),
            after.target_attack_cooldown_seconds,
        ),
        target_next_attack_queue_tick=jnp.where(
            suppress,
            next_queue,
            after.target_next_attack_queue_tick,
        ),
        target_attack_queued=jnp.where(
            suppress,
            False,
            after.target_attack_queued,
        ),
        target_attack_hit_applied=jnp.where(
            suppress,
            False,
            after.target_attack_hit_applied,
        ),
        target_damage_pending=jnp.where(
            suppress,
            False,
            after.target_damage_pending,
        ),
        pending_knockback_velocity=jnp.where(
            suppress[:, None],
            jnp.float32(0.0),
            after.pending_knockback_velocity,
        ),
    )


def _mask_agent_movement(actions, disabled):
    mask = jnp.zeros((actions.shape[1],), dtype=jnp.bool_)
    mask = mask.at[
        jnp.asarray(
            (
                ACTION_FORWARD,
                ACTION_BACK,
                ACTION_LEFT,
                ACTION_RIGHT,
                ACTION_JUMP,
            ),
            dtype=jnp.int32,
        )
    ].set(True)
    return jnp.where(disabled[:, None] & mask[None, :], 0.0, actions)


def _select_tree(mask, selected, fallback):
    return jax.tree_util.tree_map(
        lambda yes, no: jnp.where(
            mask.reshape((mask.shape[0],) + (1,) * (yes.ndim - 1)),
            yes,
            no,
        ),
        selected,
        fallback,
    )
