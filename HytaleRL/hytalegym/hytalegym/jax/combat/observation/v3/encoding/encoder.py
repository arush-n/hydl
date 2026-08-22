"""Pure-JAX encoder for learner combat observation version 3."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.programs.ability import (
    ability_lifecycle_legality_view,
    ability_legal_mask,
    effective_resource_cost,
)
from hytalegym.jax.combat.arsenal.programs.outer_roots import (
    policy_ability_mask,
    project_observable_active_slot,
)
from hytalegym.jax.combat.arsenal.runtime import combat_target_selection
from hytalegym.jax.combat.arsenal.programs.interaction_rules import (
    guard_rule_resolution,
)
from hytalegym.jax.combat.arsenal.schema.contract import OBSERVATION_CAPACITY
from hytalegym.jax.combat.arsenal.scene import (
    arsenal_scene,
    merge_arsenal_scene,
)
from hytalegym.jax.combat.arsenal.schema.types import (
    ArsenalEnvironmentState,
    ArsenalRuntimeConfig,
    ArsenalWorldCapabilities,
)
from hytalegym.jax.combat.mechanics import (
    CONTROL_IMMUNITY_MAXIMUM,
    DODGE_AUTHORED_ACTION_MASK,
    RESOURCE_STAMINA,
    STATUS_FLAG_DISABLE_ABILITIES,
    STATUS_FLAG_DISABLE_MOVEMENT,
    status_modifiers,
    guard_resource_available,
    project_applied_motion_velocity,
)
from hytalegym.jax.combat.observation.v1.runtime.encoder import (
    combat_scene_from_state,
    encode_learner_observation,
)
from hytalegym.jax.combat.observation.v1.schema.contract import (
    ACTION_DOOR_OPEN,
    COMBAT_FLOAT_FEATURES,
    DOOR_INTENT_COUNT,
    SELF_FLOAT_FEATURES,
    SELF_INTEGER_FEATURES,
)
from hytalegym.jax.combat.observation.v1.schema.types import InjectedWorldFeatures
from hytalegym.jax.combat.observation.v3.schema.contract import (
    LEARNER_OBSERVATION_V3_VERSION,
    MOVEMENT_STATE_FEATURES,
    OBSERVATION_FAILURE_ARSENAL,
    OBSERVATION_FAILURE_LOADOUT,
    OBSERVATION_FAILURE_MECHANICS,
)
from hytalegym.jax.combat.observation.v3.schema.types import (
    LearnerCombatObservationV3,
    LearnerObservationV3ActorEvidence,
)
from hytalegym.jax.combat.observation.v3.encoding.attack_compatibility import (
    project_arsenal_target_attack_compatibility,
)
from hytalegym.jax.combat.observation.v3.tokens.world import (
    WorldGeometryPolicyTokens,
    empty_world_geometry_policy_tokens,
    mask_world_geometry_policy_tokens,
)
from hytalegym.jax.combat.skills import (
    SKILL_APPROACH_ATTACK,
    SKILL_ATTACK,
    SKILL_COUNT,
    SKILL_IDLE,
    SKILL_RETREAT_ATTACK,
)
from hytalegym.jax.combat.types import (
    AGENT_ENTITY,
    ENTITY_COUNT,
    PLAYER_STAMINA_MAXIMUM,
    CombatParams,
)
from hytalegym.jax.combat.env import observe_batch
from hytalegym.jax.world import MOVEMENT_STATE_ORDER


STATUS_DURATION_SCALE_SECONDS = jnp.float32(30.0)
_SELF_ATTACK_EXECUTING = SELF_FLOAT_FEATURES.index("attack_executing")
_SELF_ATTACK_COOLDOWN = SELF_FLOAT_FEATURES.index("attack_cooldown")
_SELF_PENDING_ATTACK = SELF_INTEGER_FEATURES.index("pending_attack_index")
_COMBAT_AGENT_ATTACK_EXECUTING = COMBAT_FLOAT_FEATURES.index(
    "agent_attack_executing"
)
if tuple(MOVEMENT_STATE_FEATURES) != tuple(MOVEMENT_STATE_ORDER):
    raise RuntimeError("learner-v3 movement-state order differs from World")


def encode_learner_observation_v3(
    state: ArsenalEnvironmentState,
    params: CombatParams,
    world_features: InjectedWorldFeatures,
    world_capabilities: ArsenalWorldCapabilities,
    config: ArsenalRuntimeConfig,
    *,
    world_geometry: WorldGeometryPolicyTokens | None = None,
) -> LearnerCombatObservationV3:
    """Project privileged simulator state, then run the shared actor encoder."""

    evidence = project_learner_observation_v3_actor_evidence(
        state,
        params,
        world_features,
        world_capabilities,
        config,
        world_geometry=world_geometry,
    )
    return encode_learner_observation_v3_from_actor_evidence(evidence)


def project_learner_observation_v3_actor_evidence(
    state: ArsenalEnvironmentState,
    params: CombatParams,
    world_features: InjectedWorldFeatures,
    world_capabilities: ArsenalWorldCapabilities,
    config: ArsenalRuntimeConfig,
    *,
    world_geometry: WorldGeometryPolicyTokens | None = None,
) -> LearnerObservationV3ActorEvidence:
    """Discard privileged state before the shared v3 encoder body."""

    if world_geometry is None:
        world_geometry = empty_world_geometry_policy_tokens(
            state.combat.health.shape[0]
        )
    rules, loadout = config.mechanics_rules, config.loadout
    observable_arsenal = ability_lifecycle_legality_view(
        state.arsenal,
        loadout,
    )
    observable_active_slot = project_observable_active_slot(
        loadout,
        observable_arsenal.active_ability_slot,
        observable_arsenal.active_ability_root_slot,
    )
    # MotionControllerBase.canSteer() is false while either legacy external
    # velocity or a configured applied-velocity row remains active.  Arsenal
    # force state is held in the mechanics tree, so project that engine fact
    # into the legacy combat column instead of relying only on the separate
    # damage-knockback controller state.
    motion_force_velocity = project_applied_motion_velocity(state.mechanics)
    applied_force_active = jnp.any(
        motion_force_velocity[:, AGENT_ENTITY] != jnp.float32(0.0),
        axis=1,
    )
    observable_combat = state.combat._replace(
        knockback_control_lock=(
            state.combat.knockback_control_lock | applied_force_active
        )
    )
    observable_state = state._replace(
        combat=observable_combat,
        arsenal=observable_arsenal,
    )
    targeting = combat_target_selection(
        observable_combat,
        params,
        config,
        candidate_evidence=(
            None
            if state.combat.health.shape[1] == ENTITY_COUNT
            else (
                world_capabilities.target_candidate_perceptible
                & world_capabilities.target_candidate_perception_valid
            )
        ),
    )
    actor_target_id = (
        None
        if state.combat.health.shape[1] == ENTITY_COUNT
        else targeting.engagement_target_id[:, AGENT_ENTITY]
    )
    combat_observation = observe_batch(
        observable_combat,
        params,
        perception_line_of_sight=(
            world_capabilities.line_of_sight & world_capabilities.line_of_sight_valid
        )[:, AGENT_ENTITY],
        target_entity_id=actor_target_id,
    )
    base_scene = combat_scene_from_state(
        observable_combat,
        params,
        combat_observation,
        target_entity_id=actor_target_id,
    )
    scene = merge_arsenal_scene(
        base_scene,
        arsenal_scene(observable_combat, state.arsenal, params),
    )
    base = encode_learner_observation(
        observable_combat,
        params,
        combat_observation,
        scene,
        world_features,
        target_entity_id=actor_target_id,
    )
    base = _project_arsenal_attack_compatibility(
        base,
        observable_active_slot[:, AGENT_ENTITY],
        params,
    )
    base = project_arsenal_target_attack_compatibility(
        base,
        observable_state,
        config,
        targeting.engagement_target_id[:, AGENT_ENTITY],
    )
    resource_span = rules.resource_maximum - rules.resource_minimum
    resource_mask = resource_span > 0.0
    resource_f32 = jnp.where(
        resource_mask,
        (state.mechanics.resources - rules.resource_minimum)
        / jnp.maximum(resource_span, jnp.finfo(jnp.float32).tiny),
        jnp.float32(0.0),
    )
    applied_speed = jnp.linalg.norm(motion_force_velocity, axis=2)
    # Actor zero's sprint budget, as a signed fraction of the full bar. It goes
    # negative in the Stamina_Broken region (PLAYER_STAMINA_MINIMUM is -4.0), and
    # that sign is the useful part: it is the difference between "nearly out" and
    # "already cut off and serving a longer regen delay".
    #
    # `CombatState` carries this for actor zero only -- the other entities keep
    # theirs on `EntityLocomotionState`, which this encoder does not receive. The
    # remaining rows publish a full bar rather than a real value, because an
    # opponent's sprint budget is not observable in Hytale and inventing one
    # would hand the policy privileged information.
    self_locomotion_stamina = jnp.clip(
        state.combat.agent_locomotion_stamina / jnp.float32(PLAYER_STAMINA_MAXIMUM),
        -1.0,
        1.0,
    )
    locomotion_stamina_fraction = (
        jnp.ones_like(state.combat.health)
        .at[:, AGENT_ENTITY]
        .set(self_locomotion_stamina)
    )
    defense_f32 = jnp.stack(
        (
            state.mechanics.guard_active.astype(jnp.float32),
            state.mechanics.stamina_broken.astype(jnp.float32),
            state.mechanics.dodge_invulnerability_remaining_seconds
            / jnp.maximum(rules.dodge_invulnerability_seconds, 1.0e-6),
            # ``dodge_force`` is a scale, not a bound: this vector's Y channel
            # accumulates gravity, so player-scale jump and run speeds can push
            # the ratio past one. Clipped to keep the published column a true
            # fraction, the same way the regen-delay column below is clipped.
            jnp.clip(applied_speed / jnp.maximum(rules.dodge_force, 1.0e-6), 0.0, 1.0),
            jnp.clip(
                state.mechanics.stamina_regen_delay_seconds / 2.0,
                -1.0,
                0.0,
            ),
            state.mechanics.control_immunity / jnp.float32(CONTROL_IMMUNITY_MAXIMUM),
            (state.combat.health > 0.0).astype(jnp.float32),
            locomotion_stamina_fraction,
        ),
        axis=2,
    )
    status = state.mechanics.statuses
    max_health = (
        jnp.full(
            state.combat.health.shape,
            params.target_max_health,
            dtype=jnp.float32,
        )
        .at[:, AGENT_ENTITY]
        .set(params.agent_max_health)
    )
    status_resource_span = jnp.take_along_axis(
        resource_span,
        jnp.clip(status.resource_id, 0, resource_span.shape[2] - 1),
        axis=2,
    )
    cycle_progress = jnp.where(
        status.cycle_cooldown_seconds > 0.0,
        status.cycle_elapsed_seconds
        / jnp.maximum(status.cycle_cooldown_seconds, 1.0e-6),
        jnp.float32(0.0),
    )
    status_f32 = jnp.stack(
        (
            status.remaining_seconds / STATUS_DURATION_SCALE_SECONDS,
            cycle_progress,
            status.damage_per_cycle / max_health[..., None],
            status.healing_per_cycle / max_health[..., None],
            status.resource_delta_per_cycle / jnp.maximum(status_resource_span, 1.0),
            status.speed_multiplier / 2.0,
        ),
        axis=3,
    )
    status_i32 = jnp.stack(
        (
            status.effect_id,
            status.source_entity_id,
            status.damage_cause,
            status.resource_id,
            status.overlap_mode,
        ),
        axis=3,
    )
    status_mask = status.active
    status_f32 = jnp.where(
        status_mask[..., None],
        jnp.clip(status_f32, -1.0, 1.0),
        jnp.float32(0.0),
    )
    status_i32 = jnp.where(status_mask[..., None], status_i32, 0)
    status_flags = jnp.where(status_mask, status.flags, jnp.uint32(0))

    ability_duration_scale = jnp.maximum(
        jnp.max(
            jnp.where(
                loadout.ability_mask,
                loadout.ability_duration_seconds,
                jnp.float32(0.0),
            ),
            axis=2,
            keepdims=True,
        ),
        jnp.float32(1.0),
    )
    cooldown_denominator = jnp.maximum(
        loadout.ability_cooldown_seconds,
        jnp.float32(1.0),
    )
    slot = jnp.arange(loadout.ability_mask.shape[2])[None, None, :]
    active_slot = observable_active_slot[..., None]
    internal_slot = jnp.clip(
        observable_arsenal.active_ability_slot,
        0,
        loadout.ability_mask.shape[2] - 1,
    )
    internal_duration = jnp.take_along_axis(
        loadout.ability_duration_seconds,
        internal_slot[..., None],
        axis=2,
    )[..., 0]
    active_progress_value = jnp.where(
        observable_arsenal.active_ability_slot >= 0,
        state.arsenal.ability_elapsed_seconds
        / jnp.maximum(internal_duration, 1.0e-6),
        jnp.float32(0.0),
    )
    active_progress = jnp.where(
        slot == active_slot,
        active_progress_value[..., None],
        jnp.float32(0.0),
    )
    effective_cost = effective_resource_cost(
        loadout.ability_resource_cost,
        loadout.ability_resource_cost_kind,
    )
    affordable = jnp.all(
        state.mechanics.resources[:, :, None, :] + 1.0e-6
        >= loadout.ability_resource_minimum,
        axis=3,
    )
    # Authored outer roots may establish a transient resource only after
    # admission (for example, Shortbow's nocked Ammo stat).  Report the public
    # root as affordable; its inventory and root-condition gates remain
    # fail-closed in the action mask and runtime.
    affordable |= loadout.ability_outer_root_selector
    normalized_cost = jnp.where(
        resource_mask[:, :, None, :],
        effective_cost / jnp.maximum(resource_span[:, :, None, :], 1.0e-6),
        jnp.float32(0.0),
    )
    ability_f32 = jnp.concatenate(
        (
            (loadout.ability_duration_seconds / ability_duration_scale)[..., None],
            (state.arsenal.ability_cooldown_seconds / cooldown_denominator)[..., None],
            active_progress[..., None],
            affordable.astype(jnp.float32)[..., None],
            normalized_cost,
        ),
        axis=3,
    )
    ability_i32 = jnp.stack(
        (
            loadout.ability_id,
            loadout.ability_evidence,
            loadout.ability_requirements.astype(jnp.int32),
        ),
        axis=3,
    )
    ability_mask = (
        policy_ability_mask(loadout)
        & loadout.equipped[..., None]
        & ~loadout.overflow[..., None]
    )
    ability_f32 = jnp.where(
        ability_mask[..., None],
        jnp.clip(ability_f32, -1.0, 1.0),
        jnp.float32(0.0),
    )
    ability_i32 = jnp.where(ability_mask[..., None], ability_i32, 0)
    legal = ability_legal_mask(
        observable_arsenal,
        state.mechanics,
        loadout,
        world_capabilities,
        state.combat.health > 0.0,
    )
    ability_f32 = _pad_ability_axis(ability_f32)
    ability_i32 = _pad_ability_axis(ability_i32)
    ability_mask = _pad_ability_axis(ability_mask)
    legal = _pad_ability_axis(legal)
    actor_world_mask = jnp.stack(
        (
            world_capabilities.actor_controller_medium_available,
            world_capabilities.actor_submersion_available,
            world_capabilities.actor_drop_available,
        ),
        axis=1,
    )
    actor_world_f32 = jnp.stack(
        (
            jnp.where(
                world_capabilities.actor_controller_medium_available,
                world_capabilities.actor_controller_in_fluid,
                False,
            ),
            jnp.where(
                world_capabilities.actor_submersion_available,
                world_capabilities.actor_feet_submerged,
                False,
            ),
            jnp.where(
                world_capabilities.actor_submersion_available,
                world_capabilities.actor_eyes_submerged,
                False,
            ),
            jnp.where(
                world_capabilities.actor_drop_available,
                world_capabilities.actor_drop_support_found,
                False,
            ),
            jnp.where(
                world_capabilities.actor_drop_available,
                world_capabilities.actor_drop_height
                / jnp.maximum(
                    params.agent_walk_max_drop_height,
                    jnp.finfo(jnp.float32).tiny,
                ),
                jnp.float32(0.0),
            ),
        ),
        axis=1,
    ).astype(jnp.float32)
    actor_world_f32 = jnp.clip(actor_world_f32, 0.0, 1.0)
    movement_state_f32 = (
        state.combat.agent_walk_movement_state.values.astype(jnp.float32)
    )
    movement_state_mask = state.combat.agent_walk_movement_state.available

    loadout_failure = jnp.any(loadout.overflow, axis=1)
    skill_mask = base.action_mask[:, :SKILL_COUNT]
    for action_id in (
        SKILL_ATTACK,
        SKILL_APPROACH_ATTACK,
        SKILL_RETREAT_ATTACK,
    ):
        skill_mask = skill_mask.at[:, action_id].set(False)
    aggregate_flags, _ = status_modifiers(state.mechanics.statuses)
    movement_enabled = (
        aggregate_flags[:, AGENT_ENTITY] & jnp.uint32(STATUS_FLAG_DISABLE_MOVEMENT)
    ) == 0
    abilities_enabled = (
        aggregate_flags[:, AGENT_ENTITY] & jnp.uint32(STATUS_FLAG_DISABLE_ABILITIES)
    ) == 0
    guard_rules = guard_rule_resolution(observable_arsenal, loadout)
    guard_mask = (
        (state.combat.health[:, AGENT_ENTITY] > 0.0)
        & (rules.guard_stamina_value[:, AGENT_ENTITY] > 0.0)
        & guard_resource_available(
            state.mechanics.resources,
            rules,
        )[:, AGENT_ENTITY]
        & ~state.mechanics.stamina_broken[:, AGENT_ENTITY]
        & (
            state.mechanics.guard_active[:, AGENT_ENTITY]
            | (
                movement_enabled
                & (
                    state.mechanics.resources[:, AGENT_ENTITY, RESOURCE_STAMINA]
                    >= rules.guard_entry_cost[:, AGENT_ENTITY]
                )
                & guard_rules.can_start[:, AGENT_ENTITY]
            )
        )
    )
    dodge_mask = (
        world_capabilities.dodge_corridor_clear[:, AGENT_ENTITY]
        & jnp.asarray(DODGE_AUTHORED_ACTION_MASK, dtype=jnp.bool_)[None, :]
        & movement_enabled[:, None]
        & (state.combat.health[:, AGENT_ENTITY] > 0.0)[:, None]
        & (
            state.mechanics.resources[:, AGENT_ENTITY, RESOURCE_STAMINA]
            >= rules.dodge_admission_cost[:, AGENT_ENTITY]
        )[:, None]
        & (
            state.mechanics.dodge_cooldown_remaining_seconds[:, AGENT_ENTITY]
            <= jnp.float32(0.0)
        )[:, None]
    )
    jump_mask = (
        movement_enabled
        & (state.combat.health[:, AGENT_ENTITY] > 0.0)
        & (
            state.combat.agent_grounded
            # Holding the button while rising is what makes a Hytale jump
            # variable-height, so the head has to stay legal mid-ascent. This
            # cannot buy a second impulse: the motion path only applies one
            # while grounded, and releasing mid-ascent is what cuts the arc.
            | (state.combat.velocity[:, AGENT_ENTITY, 1] > 0.0)
        )
    )
    safe_skill = jnp.zeros_like(skill_mask).at[:, SKILL_IDLE].set(True)
    skill_mask = jnp.where(
        movement_enabled[:, None],
        skill_mask,
        safe_skill,
    )
    door_mask = (
        base.interaction_door_intent_mask
        & base.interaction_mask[..., None]
        & abilities_enabled[:, None, None]
    )
    weapon_i32 = jnp.stack(
        (
            loadout.weapon_id,
            loadout.weapon_family,
            observable_active_slot,
        ),
        axis=2,
    )
    return LearnerObservationV3ActorEvidence(
        base=base,
        weapon_i32=weapon_i32,
        resource_f32=resource_f32,
        resource_mask=resource_mask,
        defense_f32=defense_f32,
        status_f32=status_f32,
        status_i32=status_i32,
        status_flags=status_flags,
        status_mask=status_mask,
        ability_f32=ability_f32,
        ability_i32=ability_i32,
        ability_mask=ability_mask,
        ability_legal=legal,
        actor_world_f32=actor_world_f32,
        actor_world_mask=actor_world_mask,
        movement_state_f32=movement_state_f32,
        movement_state_mask=movement_state_mask,
        world_geometry=world_geometry,
        skill_action_mask=skill_mask,
        jump_action_mask=jump_mask,
        guard_action_mask=guard_mask,
        dodge_action_mask=dodge_mask,
        door_action_mask=door_mask,
        loadout_failure=loadout_failure,
        mechanics_failure_bits=state.mechanics.failure_bits,
        arsenal_failure_bits=state.arsenal.failure_bits,
    )


def _project_arsenal_attack_compatibility(base, active_slot, params):
    """Populate legacy attack columns from the authoritative v3 ability state.

    Native actor evidence exposes a one-second compatibility sentinel while
    ``CombatSupport`` executes an interaction and the active authored slot as
    the pending index.  The v3 per-ability rows remain the authoritative source
    for exact elapsed time and cooldowns; these three legacy columns only keep
    the shared 24-value prefix live and source-aligned.
    """

    active = active_slot >= 0
    active_f32 = active.astype(jnp.float32)
    pause_scale = jnp.maximum(
        jnp.asarray(
            params.agent_attack_pause_max_seconds,
            dtype=jnp.float32,
        ),
        jnp.float32(1.0e-6),
    )
    compatibility_cooldown = jnp.where(
        active,
        jnp.float32(1.0) / pause_scale,
        jnp.float32(0.0),
    )
    return base._replace(
        self_f32=(
            base.self_f32.at[:, _SELF_ATTACK_EXECUTING]
            .set(active_f32)
            .at[:, _SELF_ATTACK_COOLDOWN]
            .set(compatibility_cooldown)
        ),
        self_i32=base.self_i32.at[:, _SELF_PENDING_ATTACK].set(
            jnp.where(active, active_slot, jnp.int32(-1))
        ),
        combat_f32=base.combat_f32.at[
            :, _COMBAT_AGENT_ATTACK_EXECUTING
        ].set(active_f32),
    )


def encode_learner_observation_v3_from_actor_evidence(
    evidence: LearnerObservationV3ActorEvidence,
) -> LearnerCombatObservationV3:
    """Encode v3 from actor evidence with no privileged simulator state."""

    mechanics_failure = evidence.mechanics_failure_bits != 0
    arsenal_failure = evidence.arsenal_failure_bits != 0
    failure_bits = evidence.base.overflow_bits
    failure_bits |= jnp.where(
        evidence.loadout_failure,
        jnp.uint32(OBSERVATION_FAILURE_LOADOUT),
        jnp.uint32(0),
    )
    failure_bits |= jnp.where(
        mechanics_failure,
        jnp.uint32(OBSERVATION_FAILURE_MECHANICS),
        jnp.uint32(0),
    )
    failure_bits |= jnp.where(
        arsenal_failure,
        jnp.uint32(OBSERVATION_FAILURE_ARSENAL),
        jnp.uint32(0),
    )
    valid = failure_bits == jnp.uint32(0)
    safe_skill = jnp.zeros_like(evidence.skill_action_mask).at[:, SKILL_IDLE].set(True)
    skill_mask = jnp.where(
        valid[:, None],
        evidence.skill_action_mask,
        safe_skill,
    )
    legal = evidence.ability_legal & valid[:, None, None]
    jump_mask = evidence.jump_action_mask & valid
    guard_mask = evidence.guard_action_mask & valid
    dodge_mask = evidence.dodge_action_mask & valid[:, None]
    door_mask = evidence.door_action_mask & valid[:, None, None]
    entity_valid = valid[:, None, None]
    base = _mask_invalid_base(evidence.base, valid)
    base_action_mask = base.action_mask.at[:, :SKILL_COUNT].set(skill_mask)
    door_legal = jnp.any(door_mask, axis=1)
    base_action_mask = base_action_mask.at[
        :, ACTION_DOOR_OPEN : ACTION_DOOR_OPEN + DOOR_INTENT_COUNT
    ].set(door_legal)
    base = base._replace(action_mask=base_action_mask)
    world_geometry = mask_world_geometry_policy_tokens(
        evidence.world_geometry,
        valid,
    )
    return LearnerCombatObservationV3(
        schema_version=jnp.full(
            valid.shape,
            LEARNER_OBSERVATION_V3_VERSION,
            dtype=jnp.int32,
        ),
        base=base,
        weapon_i32=jnp.where(entity_valid, evidence.weapon_i32, 0),
        resource_f32=jnp.where(entity_valid, evidence.resource_f32, 0.0),
        resource_mask=evidence.resource_mask & valid[:, None, None],
        defense_f32=jnp.where(entity_valid, evidence.defense_f32, 0.0),
        status_f32=jnp.where(
            valid[:, None, None, None],
            evidence.status_f32,
            0.0,
        ),
        status_i32=jnp.where(
            valid[:, None, None, None],
            evidence.status_i32,
            0,
        ),
        status_flags=jnp.where(
            valid[:, None, None],
            evidence.status_flags,
            jnp.uint32(0),
        ),
        status_mask=evidence.status_mask & valid[:, None, None],
        ability_f32=jnp.where(
            valid[:, None, None, None],
            evidence.ability_f32,
            0.0,
        ),
        ability_i32=jnp.where(
            valid[:, None, None, None],
            evidence.ability_i32,
            0,
        ),
        ability_mask=evidence.ability_mask & valid[:, None, None],
        ability_legal=legal,
        actor_world_f32=jnp.where(
            valid[:, None],
            evidence.actor_world_f32,
            0.0,
        ),
        actor_world_mask=evidence.actor_world_mask & valid[:, None],
        movement_state_f32=jnp.where(
            evidence.movement_state_mask & valid[:, None],
            evidence.movement_state_f32,
            jnp.float32(0.0),
        ),
        movement_state_mask=(
            evidence.movement_state_mask & valid[:, None]
        ),
        world_geometry=world_geometry,
        skill_action_mask=skill_mask,
        ability_action_mask=legal[:, AGENT_ENTITY],
        jump_action_mask=jump_mask,
        guard_action_mask=guard_mask,
        dodge_action_mask=dodge_mask,
        door_action_mask=door_mask,
        valid=valid,
        failure_bits=failure_bits,
        mechanics_failure_bits=evidence.mechanics_failure_bits,
        arsenal_failure_bits=evidence.arsenal_failure_bits,
    )


def _mask_invalid_base(base, valid):
    masked = jax.tree_util.tree_map(
        lambda value: jnp.where(
            valid.reshape((valid.shape[0],) + (1,) * (value.ndim - 1)),
            value,
            jnp.zeros_like(value),
        ),
        base,
    )
    safe_action = jnp.zeros_like(base.action_mask).at[:, SKILL_IDLE].set(True)
    return masked._replace(
        schema_version=base.schema_version,
        action_mask=jnp.where(valid[:, None], base.action_mask, safe_action),
        valid=valid,
        overflow_bits=base.overflow_bits,
    )


def _pad_ability_axis(value):
    runtime_capacity = value.shape[2]
    if runtime_capacity > OBSERVATION_CAPACITY:
        raise ValueError(
            "learner observation ability capacity is smaller than runtime: "
            f"{OBSERVATION_CAPACITY} < {runtime_capacity}"
        )
    padding = OBSERVATION_CAPACITY - runtime_capacity
    if padding == 0:
        return value
    widths = [(0, 0)] * value.ndim
    widths[2] = (0, padding)
    return jnp.pad(value, widths)


__all__ = [
    "encode_learner_observation_v3",
    "encode_learner_observation_v3_from_actor_evidence",
    "project_learner_observation_v3_actor_evidence",
]
