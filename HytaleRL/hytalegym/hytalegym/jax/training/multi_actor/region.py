"""Actor-major Region evidence over one persistent shared World runtime."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.environment import (
    ACTION_SURFACE_REJECT_EXECUTOR_DENIED,
    ACTION_SURFACE_REJECT_EXECUTOR_UNAVAILABLE,
    ACTION_SURFACE_REJECT_TARGET_RECHECK,
    ArsenalActionSurfaceExecution,
    ArsenalActionSurfaceExecutor,
    ArsenalActionSurfaceEvidence,
    ArsenalActionSurfaceLifecycle,
    ArsenalActionSurfaceProvider,
    ArsenalActionSurfaceRuntime,
    ArsenalActionSurfaceViews,
    ArsenalWorldRuntimeViews,
    action_surface_lifecycle_evidence,
    action_surface_verb_requests,
    fail_closed_arsenal_world_capabilities,
)
from hytalegym.jax.combat.arsenal.factory import empty_arsenal_commands
from hytalegym.jax.combat.arsenal.projectiles.runtime import (
    ArsenalExplosionCandidateProvider,
)
from hytalegym.jax.combat.arsenal.schema.types import (
    ArsenalEnvironmentState,
    ArsenalRuntimeConfig,
    ArsenalTransition,
    ArsenalWorldCapabilities,
)
from hytalegym.jax.combat.block_interactions import (
    InteractionMovementConstraints,
    apply_interaction_movement_constraints,
    interaction_movement_constraints,
)
from hytalegym.jax.combat.observation.v3.encoding.encoder import (
    project_learner_observation_v3_actor_evidence,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.blocks import (
    empty_block_action_candidate_policy_view,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY,
)
from hytalegym.jax.combat.observation.v3.policy.actions import (
    decode_learner_arsenal_action,
)
from hytalegym.jax.combat.observation.v3.policy.surface import (
    StagedActionSurfaceDecode,
    action_surface_layout,
    decode_staged_action_surface_factors,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.contract import (
    recipe_candidate_encoding_parameter_seed,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.encoder import (
    RecipeCandidateEncoderParams,
    encode_recipe_candidates,
    initialize_recipe_candidate_encoder,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.recipes import (
    empty_recipe_candidate_policy_view,
)
from hytalegym.jax.combat.observation.v3.tokens.inventory import (
    InventoryPolicyTokens,
    inventory_policy_tokens_from_state,
)
from hytalegym.jax.combat.observation.v3.tokens.light import (
    ActorLightPolicyTokens,
    empty_actor_light_policy_tokens,
)
from hytalegym.jax.combat.observation.v3.schema.types import (
    LearnerCombatObservationV3,
    LearnerObservationV3ActorEvidence,
)
from hytalegym.jax.combat.types import CombatParams, RewardComponents
from hytalegym.jax.combat import TargetNavigationProvider
from hytalegym.jax.world import GeometryState, RegionGeometryState

from .assignment import PolicyActorAssignment
from .controls import PolicyActorControls, policy_actor_controls
from .locomotion import EntityLocomotionState
from .observations import (
    _assemble_policy_actor_observation,
    _mask_non_actor_entity_rows,
    encode_policy_actor_observations,
)
from .views import (
    actor_first_arsenal_view,
    actor_first_block_interaction_view,
)
from .runtime import (
    MultiActorArenaState,
    step_multi_actor_arena,
)


ActorMajorWorldRuntimeProvider = Callable[
    [ArsenalEnvironmentState, object, CombatParams, ArsenalRuntimeConfig],
    ArsenalWorldRuntimeViews,
]


class PolicyActorRegionObservation(NamedTuple):
    """Policy rows plus actor-local Region evidence retained for execution."""

    evidence: LearnerObservationV3ActorEvidence
    structured: LearnerCombatObservationV3
    inventory: InventoryPolicyTokens
    dense: jax.Array
    action_mask: jax.Array
    capabilities: ArsenalWorldCapabilities
    action_surface: ArsenalActionSurfaceEvidence
    light_tokens: ActorLightPolicyTokens
    interaction_movement: InteractionMovementConstraints
    physical_geometry: GeometryState | RegionGeometryState


class PolicyActorRegionActions(NamedTuple):
    """Decoded combat controls and retained World actions on ``[B,P]``."""

    controls: PolicyActorControls
    surface: StagedActionSurfaceDecode

    @property
    def valid(self) -> jax.Array:
        return self.controls.valid


class PolicyActorRegionExecution(NamedTuple):
    """Actor-slot lifecycle and the committed shared mutable runtime."""

    state: ArsenalEnvironmentState
    runtime: ArsenalActionSurfaceRuntime
    action_valid: jax.Array
    request_legal: jax.Array
    selected_target_rechecked: jax.Array
    state_commit: jax.Array
    lifecycle: ArsenalActionSurfaceLifecycle


class MultiActorRegionArenaTransition(NamedTuple):
    """Shared combat transition plus actor-owned Region verb evidence."""

    state: MultiActorArenaState
    arsenal: ArsenalTransition
    reward_components: RewardComponents
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    controlled: jax.Array
    action_valid: jax.Array
    action_surface: PolicyActorRegionExecution


def observe_policy_actors_region(
    state: ArsenalEnvironmentState,
    params: CombatParams,
    config: ArsenalRuntimeConfig,
    locomotion: EntityLocomotionState,
    assignment: PolicyActorAssignment,
    action_runtime: ArsenalActionSurfaceRuntime,
    *,
    actor_world_runtime_provider: ActorMajorWorldRuntimeProvider,
    action_surface_provider: ArsenalActionSurfaceProvider | None,
    recipe_candidate_encoder_params: RecipeCandidateEncoderParams | None = None,
) -> PolicyActorRegionObservation:
    """Project independent legal Region rows for every policy-owned actor.

    The caller-owned provider receives an actor-first state *and* its matching
    actor-first runtime config. This extra config argument is intentional: a
    single-actor provider closed over the original entity order is not safe for
    heterogeneous loadouts. The shared mutable World remains unpermuted.
    """

    if not isinstance(action_runtime, ArsenalActionSurfaceRuntime):
        raise TypeError("action_runtime must be ArsenalActionSurfaceRuntime")
    if not callable(actor_world_runtime_provider):
        raise TypeError("actor_world_runtime_provider must be callable")
    if action_surface_provider is not None and not callable(action_surface_provider):
        raise TypeError("action_surface_provider must be callable or None")
    batch, entity_count = state.combat.health.shape
    if assignment.actor_index.shape[0] != batch:
        raise ValueError("assignment batch does not match arena state")
    if any(
        leaf.shape[:2] != (batch, entity_count)
        for leaf in jax.tree_util.tree_leaves(locomotion)
    ):
        raise ValueError("locomotion leaves must begin with [B,N]")
    encoder = (
        initialize_recipe_candidate_encoder(
            jax.random.key(recipe_candidate_encoding_parameter_seed())
        )
        if recipe_candidate_encoder_params is None
        else recipe_candidate_encoder_params
    )
    placeholder = fail_closed_arsenal_world_capabilities(state, params)
    evidence_rows = []
    actor_inventory_states = []
    capability_rows = []
    surface_rows = []
    light_rows = []
    movement_rows = []
    physical_geometry = None
    geometry_signature = None

    for slot in range(assignment.actor_index.shape[1]):
        actor = jnp.where(
            assignment.active[:, slot],
            assignment.actor_index[:, slot],
            jnp.int32(0),
        )
        actor_state, actor_config, _ = actor_first_arsenal_view(
            state,
            config,
            placeholder,
            locomotion,
            actor,
        )
        world_views = actor_world_runtime_provider(
            actor_state,
            action_runtime.world,
            params,
            actor_config,
        )
        _validate_actor_world_views(world_views, batch)
        signature = _geometry_signature(world_views.physical_geometry)
        if physical_geometry is None:
            physical_geometry = world_views.physical_geometry
            geometry_signature = signature
        elif signature != geometry_signature:
            raise ValueError(
                "actor World providers must share one physical geometry layout"
            )
        light = world_views.light_tokens
        if light is None:
            light = empty_actor_light_policy_tokens(
                batch,
                world_views.tokens.token_mask.shape[1],
            )
        elif not isinstance(light, ActorLightPolicyTokens):
            raise TypeError("actor World light_tokens has the wrong type")
        actor_runtime = action_runtime._replace(
            block_interactions=actor_first_block_interaction_view(
                state,
                action_runtime.block_interactions,
                actor,
            )
        )
        surface_views = (
            _empty_action_surface_views(batch)
            if action_surface_provider is None
            else action_surface_provider(actor_state, actor_runtime, params)
        )
        surface = _action_surface_evidence(surface_views, encoder, batch)
        evidence = project_learner_observation_v3_actor_evidence(
            actor_state,
            params,
            world_views.features,
            world_views.capabilities,
            actor_config,
            world_geometry=world_views.tokens,
        )
        if assignment.actor_index.shape[1] > 1:
            evidence = _mask_non_actor_entity_rows(evidence)
        evidence_rows.append(evidence)
        actor_inventory_states.append(actor_state.inventory)
        capability_rows.append(world_views.capabilities)
        surface_rows.append(surface)
        light_rows.append(light)
        movement_rows.append(
            interaction_movement_constraints(
                actor_runtime.block_interactions,
                actor_index=0,
            )
        )

    if physical_geometry is None:
        raise ValueError("assignment must contain at least one policy slot")
    evidence = _stack_actor_rows(evidence_rows)
    structured = encode_policy_actor_observations(evidence, assignment)
    inventory = _stack_actor_rows(
        [
            inventory_policy_tokens_from_state(
                actor_inventory,
                config.inventory_layout,
                actor_valid=structured.valid[:, slot],
            )
            for slot, actor_inventory in enumerate(actor_inventory_states)
        ]
    )
    capabilities = _stack_actor_rows(capability_rows)
    action_surface = _stack_actor_rows(surface_rows)
    light_tokens = _stack_actor_rows(light_rows)
    movement = _stack_actor_rows(movement_rows)
    policy = _assemble_policy_actor_observation(
        evidence,
        structured,
        inventory,
        assignment,
        action_surface=action_surface,
        light_tokens=light_tokens,
        interaction_movement=movement,
    )
    return PolicyActorRegionObservation(
        evidence=policy.evidence,
        structured=policy.structured,
        inventory=policy.inventory,
        dense=policy.dense,
        action_mask=policy.action_mask,
        capabilities=capabilities,
        action_surface=action_surface,
        light_tokens=light_tokens,
        interaction_movement=movement,
        physical_geometry=physical_geometry,
    )


def decode_policy_actor_factors_region(
    state: ArsenalEnvironmentState,
    observation: PolicyActorRegionObservation,
    action_factors: jax.Array,
    locomotion: EntityLocomotionState,
    assignment: PolicyActorAssignment,
    action_runtime: ArsenalActionSurfaceRuntime,
    *,
    maximum_turn_degrees: float = 45.0,
) -> PolicyActorRegionActions:
    """Decode every Region factor without dropping selected World verbs."""

    actor_shape = assignment.actor_index.shape
    factors = jnp.asarray(action_factors)
    expected = actor_shape + (len(ARSENAL_POLICY_ACTION_HEAD_SIZES),)
    if factors.dtype != jnp.int32 or factors.shape != expected:
        raise TypeError(f"action_factors must be int32 with shape {expected}")
    if observation.action_mask.shape[:2] != actor_shape:
        raise ValueError("observation action mask must begin with [B,P]")
    if not isinstance(action_runtime, ArsenalActionSurfaceRuntime):
        raise TypeError("action_runtime must be ArsenalActionSurfaceRuntime")
    layout = action_surface_layout(
        observation.action_surface.block_candidates.candidate_mask.shape[-1]
    )
    rows = []
    surfaces = []
    for slot in range(actor_shape[1]):
        actor = jnp.where(
            assignment.active[:, slot],
            assignment.actor_index[:, slot],
            jnp.int32(0),
        )
        surface = decode_staged_action_surface_factors(
            factors[:, slot],
            observation.action_mask[:, slot],
            layout=layout,
            maximum_turn_degrees=maximum_turn_degrees,
        )
        actor_observation = jax.tree_util.tree_map(
            lambda value: value[:, slot],
            observation.structured,
        )
        actor_capabilities = jax.tree_util.tree_map(
            lambda value: value[:, slot],
            observation.capabilities,
        )
        # The world-move override resolves its compass into body-relative
        # channels, so it needs the BODY's commanded heading, not the camera's.
        desired_body_yaw = jnp.take_along_axis(
            locomotion.desired_body_yaw,
            actor[:, None],
            axis=1,
        )[:, 0]
        decoded = decode_learner_arsenal_action(
            actor_observation,
            surface.base,
            actor_capabilities,
            maximum_turn_degrees=maximum_turn_degrees,
            desired_body_yaw_degrees=desired_body_yaw,
        )
        actor_interactions = actor_first_block_interaction_view(
            state,
            action_runtime.block_interactions,
            actor,
        )
        low_level = apply_interaction_movement_constraints(
            actor_interactions,
            decoded.low_level_action,
            actor_index=0,
        )
        rows.append(
            (
                low_level,
                decoded.commands.ability_slot[:, 0],
                decoded.commands.defense.guard_held[:, 0],
                decoded.commands.defense.dodge_direction[:, 0],
                decoded.commands.defense.dodge_corridor_clear[:, 0],
                decoded.valid & surface.action_legal,
            )
        )
        surfaces.append(surface)
    controls = policy_actor_controls(
        *(jnp.stack([row[index] for row in rows], axis=1) for index in range(6)),
        assignment,
    )
    return PolicyActorRegionActions(
        controls=controls,
        surface=_stack_actor_rows(surfaces),
    )


def execute_policy_actor_region_actions(
    state: ArsenalEnvironmentState,
    locomotion: EntityLocomotionState,
    action_runtime: ArsenalActionSurfaceRuntime,
    actions: PolicyActorRegionActions,
    observation: PolicyActorRegionObservation,
    assignment: PolicyActorAssignment,
    keys: jax.Array,
    params: CombatParams,
    config: ArsenalRuntimeConfig,
    executor: ArsenalActionSurfaceExecutor | None,
) -> PolicyActorRegionExecution:
    """Execute actor World verbs in deterministic entity-ID order.

    The shipped Region executor mutates only the actor's inventory row, the
    actor's block-interaction row, and the shared World runtime. Those three
    effects are merged explicitly. Entity-ID ordering makes simultaneous
    conflicting requests independent of policy-slot assignment order, while
    every selected target is still requeried against the current runtime.
    """

    if executor is not None and not callable(executor):
        raise TypeError("executor must be callable or None")
    if not isinstance(action_runtime, ArsenalActionSurfaceRuntime):
        raise TypeError("action_runtime must be ArsenalActionSurfaceRuntime")
    batch, entity_count = state.combat.health.shape
    actor_shape = assignment.actor_index.shape
    if actor_shape[0] != batch:
        raise ValueError("assignment batch does not match arena state")
    step_keys = jnp.asarray(keys)
    if step_keys.shape[0] != batch:
        raise ValueError("keys must preserve the arena batch axis")
    priority = jnp.where(
        assignment.active,
        assignment.actor_index,
        jnp.int32(entity_count) + jnp.arange(actor_shape[1], dtype=jnp.int32)[None, :],
    )
    slot_order = jnp.argsort(priority, axis=1).astype(jnp.int32)
    current_state = state
    current_runtime = action_runtime
    request_legal = jnp.ones(actor_shape, dtype=jnp.bool_)
    target_rechecked = jnp.ones(actor_shape, dtype=jnp.bool_)
    state_commit = jnp.zeros(actor_shape, dtype=jnp.bool_)
    action_valid = actions.controls.valid
    lifecycle = _empty_actor_lifecycle(actor_shape)
    placeholder = fail_closed_arsenal_world_capabilities(state, params)

    for rank in range(actor_shape[1]):
        slot = slot_order[:, rank]
        active = _gather_slot(assignment.active, slot)
        actor = jnp.where(
            active,
            _gather_slot(assignment.actor_index, slot),
            jnp.int32(0),
        )
        actor_state, actor_config, _ = actor_first_arsenal_view(
            current_state,
            config,
            placeholder,
            locomotion,
            actor,
        )
        actor_runtime = current_runtime._replace(
            block_interactions=actor_first_block_interaction_view(
                current_state,
                current_runtime.block_interactions,
                actor,
            )
        )
        actor_surface = jax.tree_util.tree_map(
            lambda value: _gather_slot(value, slot),
            actions.surface,
        )
        actor_evidence = jax.tree_util.tree_map(
            lambda value: _gather_slot(value, slot),
            observation.action_surface,
        )
        actor_keys = jax.vmap(jax.random.fold_in)(step_keys, actor)
        requests = action_surface_verb_requests(actor_surface)
        requested = jnp.any(requests, axis=1)
        if executor is None:
            result = ArsenalActionSurfaceExecution(
                state=actor_state,
                runtime=actor_runtime,
                request_legal=~requested,
                selected_target_rechecked=~requested,
                state_commit=jnp.zeros((batch,), dtype=jnp.bool_),
                lifecycle=action_surface_lifecycle_evidence(
                    actor_surface,
                    reject_reason=jnp.full(
                        requests.shape,
                        ACTION_SURFACE_REJECT_EXECUTOR_UNAVAILABLE,
                        dtype=jnp.uint32,
                    ),
                ),
            )
        else:
            result = executor(
                actor_state,
                actor_runtime,
                actor_surface,
                actor_evidence,
                actor_keys,
                params,
                actor_config,
            )
            _validate_actor_execution(result, batch)
        lifecycle_accepted = jnp.all(
            ~requests | result.lifecycle.accepted,
            axis=1,
        )
        target_required = requested
        recheck_ok = ~target_required | result.selected_target_rechecked
        legal = (
            actor_surface.action_legal
            & result.request_legal
            & lifecycle_accepted
            & recheck_ok
        )
        commit = active & legal & result.state_commit
        current_state = current_state._replace(
            inventory=_merge_actor_first_entity_row(
                current_state.inventory,
                result.state.inventory,
                actor,
                commit,
            )
        )
        current_runtime = current_runtime._replace(
            block_interactions=_merge_actor_first_entity_row(
                current_runtime.block_interactions,
                result.runtime.block_interactions,
                actor,
                commit,
            ),
            world=_select_batch_rows(
                commit,
                result.runtime.world,
                current_runtime.world,
            ),
        )
        slot_legal = active & legal
        request_legal = _scatter_slot(
            request_legal,
            slot,
            active & result.request_legal,
            active,
        )
        target_rechecked = _scatter_slot(
            target_rechecked,
            slot,
            active & result.selected_target_rechecked,
            active,
        )
        state_commit = _scatter_slot(
            state_commit,
            slot,
            commit,
            active,
        )
        action_valid = _scatter_slot(
            action_valid,
            slot,
            slot_legal,
            active,
        )
        lifecycle = jax.tree_util.tree_map(
            lambda full, value: _scatter_slot(
                full,
                slot,
                value,
                active,
            ),
            lifecycle,
            _canonical_actor_lifecycle(
                result.lifecycle,
                requests,
                legal,
                recheck_ok,
            ),
        )

    return PolicyActorRegionExecution(
        state=current_state,
        runtime=current_runtime,
        action_valid=action_valid,
        request_legal=request_legal,
        selected_target_rechecked=target_rechecked,
        state_commit=state_commit,
        lifecycle=lifecycle,
    )


def step_multi_actor_region_arena(
    state: MultiActorArenaState,
    actions: PolicyActorRegionActions,
    observation: PolicyActorRegionObservation,
    assignment: PolicyActorAssignment,
    keys: jax.Array,
    params: CombatParams,
    config: ArsenalRuntimeConfig,
    world_capabilities: ArsenalWorldCapabilities,
    executor: ArsenalActionSurfaceExecutor | None,
    *,
    scripted_actions: jax.Array | None = None,
    target_navigation_provider: TargetNavigationProvider | None = None,
    explosion_candidate_provider: ArsenalExplosionCandidateProvider | None = None,
) -> MultiActorRegionArenaTransition:
    """Advance combat and commit actor-major Region actions in one arena tick."""

    if state.action_surface_runtime is None:
        raise ValueError("Region arena state requires an action-surface runtime")
    batch, entity_count = state.arsenal.combat.health.shape
    commands = empty_arsenal_commands(
        batch,
        entity_count=entity_count,
    )._replace(world=world_capabilities)
    combat = step_multi_actor_arena(
        state,
        actions.controls,
        assignment,
        keys,
        params,
        commands,
        config,
        scripted_actions=scripted_actions,
        geometry=observation.physical_geometry,
        target_navigation_provider=target_navigation_provider,
        explosion_candidate_provider=explosion_candidate_provider,
    )
    execution = execute_policy_actor_region_actions(
        combat.state.arsenal,
        combat.state.locomotion,
        state.action_surface_runtime,
        actions,
        observation,
        assignment,
        keys,
        params,
        config,
        executor,
    )
    next_state = combat.state._replace(
        arsenal=execution.state,
        action_surface_runtime=execution.runtime,
    )
    return MultiActorRegionArenaTransition(
        state=next_state,
        arsenal=combat.arsenal._replace(state=execution.state),
        reward_components=combat.reward_components,
        reward=combat.reward,
        terminated=combat.terminated,
        truncated=combat.truncated,
        controlled=combat.controlled,
        action_valid=execution.action_valid,
        action_surface=execution,
    )


def _validate_actor_execution(
    result: ArsenalActionSurfaceExecution,
    batch: int,
) -> None:
    if not isinstance(result, ArsenalActionSurfaceExecution):
        raise TypeError("executor must return ArsenalActionSurfaceExecution")
    for name in (
        "request_legal",
        "selected_target_rechecked",
        "state_commit",
    ):
        value = getattr(result, name)
        if value.shape != (batch,) or value.dtype != jnp.dtype(jnp.bool_):
            raise ValueError(f"execution {name} must be bool[B]")
    if not isinstance(result.lifecycle, ArsenalActionSurfaceLifecycle):
        raise TypeError("execution lifecycle has the wrong type")
    if any(
        value.shape != (batch, 3)
        for value in jax.tree_util.tree_leaves(result.lifecycle)
    ):
        raise ValueError("execution lifecycle leaves must have shape [B,3]")


def _empty_actor_lifecycle(actor_shape) -> ArsenalActionSurfaceLifecycle:
    shape = actor_shape + (3,)
    boolean = jnp.zeros(shape, dtype=jnp.bool_)
    return ArsenalActionSurfaceLifecycle(
        requested=boolean,
        accepted=boolean,
        started=boolean,
        finished=boolean,
        rejected=boolean,
        cancelled=boolean,
        reject_reason=jnp.zeros(shape, dtype=jnp.uint32),
    )


def _canonical_actor_lifecycle(
    lifecycle: ArsenalActionSurfaceLifecycle,
    requests: jax.Array,
    legal: jax.Array,
    recheck_ok: jax.Array,
) -> ArsenalActionSurfaceLifecycle:
    legal_rows = legal[:, None]
    requested = jnp.asarray(requests, dtype=jnp.bool_)
    accepted = lifecycle.accepted & requested & legal_rows
    rejected = requested & ~accepted
    reason = jnp.where(
        ~requested,
        jnp.uint32(0),
        jnp.where(
            lifecycle.reject_reason == ACTION_SURFACE_REJECT_EXECUTOR_UNAVAILABLE,
            ACTION_SURFACE_REJECT_EXECUTOR_UNAVAILABLE,
            jnp.where(
                ~recheck_ok[:, None],
                ACTION_SURFACE_REJECT_TARGET_RECHECK,
                jnp.where(
                    rejected,
                    jnp.where(
                        lifecycle.reject_reason == 0,
                        ACTION_SURFACE_REJECT_EXECUTOR_DENIED,
                        lifecycle.reject_reason,
                    ),
                    jnp.uint32(0),
                ),
            ),
        ),
    )
    return ArsenalActionSurfaceLifecycle(
        requested=requested,
        accepted=accepted,
        started=lifecycle.started & accepted,
        finished=lifecycle.finished & accepted,
        rejected=rejected,
        cancelled=lifecycle.cancelled & requested,
        reject_reason=reason,
    )


def _gather_slot(value: jax.Array, slot: jax.Array) -> jax.Array:
    batch = value.shape[0]
    return value[jnp.arange(batch, dtype=jnp.int32), slot]


def _scatter_slot(
    full: jax.Array,
    slot: jax.Array,
    value: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    batch_index = jnp.arange(full.shape[0], dtype=jnp.int32)
    current = full[batch_index, slot]
    selector = mask.reshape(mask.shape + (1,) * (value.ndim - 1))
    selected = jnp.where(selector, value, current)
    return full.at[batch_index, slot].set(selected)


def _merge_actor_first_entity_row(
    full,
    actor_first,
    actor: jax.Array,
    commit: jax.Array,
):
    """Merge only row zero from an actor-first executor result."""

    batch_index = jnp.arange(actor.shape[0], dtype=jnp.int32)
    entity_count = (
        full.phase.shape[1] if hasattr(full, "phase") else full.item_id.shape[1]
    )

    def merge(original, view):
        if original.shape != view.shape or original.dtype != view.dtype:
            raise ValueError("actor executor changed an entity-state layout")
        if original.ndim < 1 or original.shape[0] != actor.shape[0]:
            raise ValueError("actor executor state must begin with [B]")
        if original.ndim < 2 or original.shape[1] != entity_count:
            expanded = commit.reshape(commit.shape + (1,) * (original.ndim - 1))
            return jnp.where(expanded, view, original)
        current = original[batch_index, actor]
        candidate = view[:, 0]
        selector = commit.reshape(commit.shape + (1,) * (candidate.ndim - 1))
        selected = jnp.where(selector, candidate, current)
        return original.at[batch_index, actor].set(selected)

    return jax.tree_util.tree_map(merge, full, actor_first)


def _select_batch_rows(mask: jax.Array, when_true, when_false):
    selector = jnp.asarray(mask, dtype=jnp.bool_)

    def select(true_value, false_value):
        if true_value.shape != false_value.shape:
            raise ValueError("actor executor changed a World-state layout")
        if true_value.dtype != false_value.dtype:
            raise ValueError("actor executor changed a World-state dtype")
        if true_value.ndim == 0 or true_value.shape[0] != selector.shape[0]:
            return false_value
        expanded = selector.reshape(selector.shape + (1,) * (true_value.ndim - 1))
        return jnp.where(expanded, true_value, false_value)

    return jax.tree_util.tree_map(select, when_true, when_false)


def _action_surface_evidence(
    views: ArsenalActionSurfaceViews,
    encoder: RecipeCandidateEncoderParams,
    batch: int,
) -> ArsenalActionSurfaceEvidence:
    if not isinstance(views, ArsenalActionSurfaceViews):
        raise TypeError("action_surface_provider returned the wrong type")
    if views.block_candidates.available.shape != (batch,):
        raise ValueError("block candidate view batch mismatch")
    if views.recipe_candidates.available.shape != (batch,):
        raise ValueError("recipe candidate view batch mismatch")
    if views.use_available.shape != (batch,):
        raise ValueError("use availability batch mismatch")
    if views.block_trigger_available.shape != (batch, 2):
        raise ValueError("block trigger availability must have shape [B,2]")
    return ArsenalActionSurfaceEvidence(
        block_candidates=views.block_candidates,
        recipe_candidates=views.recipe_candidates,
        recipe_encoding=encode_recipe_candidates(
            encoder,
            views.recipe_candidates,
        ),
        use_available=jnp.asarray(views.use_available, dtype=jnp.bool_),
        block_trigger_available=jnp.asarray(
            views.block_trigger_available,
            dtype=jnp.bool_,
        ),
    )


def _empty_action_surface_views(batch: int) -> ArsenalActionSurfaceViews:
    return ArsenalActionSurfaceViews(
        block_candidates=empty_block_action_candidate_policy_view(
            batch,
            ARSENAL_POLICY_DEFAULT_BLOCK_CANDIDATE_CAPACITY,
        ),
        recipe_candidates=empty_recipe_candidate_policy_view(batch),
        use_available=jnp.zeros((batch,), dtype=jnp.bool_),
        block_trigger_available=jnp.zeros((batch, 2), dtype=jnp.bool_),
    )


def _validate_actor_world_views(
    views: ArsenalWorldRuntimeViews,
    batch: int,
) -> None:
    if not isinstance(views, ArsenalWorldRuntimeViews):
        raise TypeError(
            "actor_world_runtime_provider must return ArsenalWorldRuntimeViews"
        )
    if not isinstance(
        views.physical_geometry,
        (GeometryState, RegionGeometryState),
    ):
        raise TypeError("actor World physical_geometry must be exact geometry")
    if not isinstance(views.capabilities, ArsenalWorldCapabilities):
        raise TypeError("actor World capabilities has the wrong type")
    leaves = jax.tree_util.tree_leaves(views.capabilities)
    if not leaves or any(value.shape[0] != batch for value in leaves):
        raise ValueError("actor World capability leaves must begin with [B]")
    if views.tokens.token_mask.shape[0] != batch:
        raise ValueError("actor World token batch mismatch")


def _geometry_signature(geometry) -> tuple[tuple[tuple[int, ...], str], ...]:
    return tuple(
        (tuple(value.shape), str(value.dtype))
        for value in jax.tree_util.tree_leaves(geometry)
    )


def _stack_actor_rows(rows):
    return jax.tree_util.tree_map(
        lambda *values: jnp.stack(values, axis=1),
        *rows,
    )


__all__ = [
    "ActorMajorWorldRuntimeProvider",
    "MultiActorRegionArenaTransition",
    "PolicyActorRegionActions",
    "PolicyActorRegionExecution",
    "PolicyActorRegionObservation",
    "decode_policy_actor_factors_region",
    "execute_policy_actor_region_actions",
    "observe_policy_actors_region",
    "step_multi_actor_region_arena",
]
