"""Bounded shared-arena composition for multiple policy-owned entities."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat import TargetNavigationProvider
from hytalegym.jax.combat.runtime.reset import observe_batch
from hytalegym.jax.combat.arsenal.environment import (
    ArsenalActionSurfaceRuntime,
    GeometryProvider,
)
from hytalegym.jax.combat.arsenal.compilation import (
    plan_arsenal_compilation,
)
from hytalegym.jax.combat.arsenal.projectiles.runtime import (
    ArsenalExplosionCandidateProvider,
)
from hytalegym.jax.combat.arsenal.runtime import step_arsenal_batch
from hytalegym.jax.combat.arsenal.schema.types import (
    ArsenalCommands,
    ArsenalEnvironmentState,
    ArsenalInfo,
    ArsenalRuntimeConfig,
    ArsenalTransition,
)
from hytalegym.jax.combat.types import (
    ACTION_SIZE,
    AGENT_ENTITY,
    CombatInfo,
    CombatParams,
    RewardComponents,
)

from .assignment import (
    PolicyActorAssignment,
    gather_policy_actor_rows,
)
from .controls import (
    PolicyActorControls,
    merge_policy_actor_commands,
    scatter_policy_actor_controls,
)
from .locomotion import (
    EntityLocomotionState,
    initialize_entity_locomotion_state,
    synchronize_entity_zero_locomotion,
    tick_entity_policy_locomotion,
)
from .reward import (
    MultiActorRewardMemory,
    compose_per_actor_reward,
    empty_multi_actor_reward_memory,
    per_actor_reward_components_from_totals,
)
from .state import (
    PackedEntityLocomotionState,
    pack_entity_locomotion_state,
    unpack_entity_locomotion_state,
)


class MultiActorArenaState(NamedTuple):
    """Shared simulator state plus policy-owned controller/reward memory."""

    arsenal: ArsenalEnvironmentState
    packed_locomotion: PackedEntityLocomotionState
    reward_memory: MultiActorRewardMemory
    action_surface_runtime: ArsenalActionSurfaceRuntime | None = None

    @property
    def locomotion(self) -> EntityLocomotionState:
        """Return the named kernel view of the packed persistent state."""

        return unpack_entity_locomotion_state(self.packed_locomotion)


class MultiActorArenaTransition(NamedTuple):
    """One bounded shared-arena step with actor-major reward facts.

    The canonical Arsenal state is stored only in ``state.arsenal``.  Keeping
    a second copy in a nested ``ArsenalTransition`` makes every state leaf a
    duplicate XLA output, which is especially costly for the projectile,
    inventory, and interaction banks.  The compatibility property below
    reconstructs that view without adding duplicate leaves to the compiled
    result.
    """

    state: MultiActorArenaState
    arsenal_observation: jax.Array
    arsenal_reward: jax.Array
    arsenal_terminated: jax.Array
    arsenal_truncated: jax.Array
    combat_info: CombatInfo
    arsenal_info: ArsenalInfo
    reward_components: RewardComponents
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    controlled: jax.Array

    @property
    def arsenal(self) -> ArsenalTransition:
        """Expose the legacy transition view without duplicating its state."""

        return ArsenalTransition(
            state=self.state.arsenal,
            observation=self.arsenal_observation,
            reward=self.arsenal_reward,
            terminated=self.arsenal_terminated,
            truncated=self.arsenal_truncated,
            combat_info=self.combat_info,
            arsenal_info=self.arsenal_info,
        )


def _hold_speed_multiplier(
    ability_hold_slot: jax.Array,
    ability_hold_seconds: jax.Array,
    ability_hold_speed_multiplier: jax.Array,
    ability_hold_speed_multiplier_after_seconds: jax.Array,
) -> jax.Array:
    """Resolve C2 ``HorizontalSpeedMultiplier`` for the hold in flight.

    ``ability_hold_slot`` is ``[B,N]`` and is ``-1`` when nothing is held, so
    the gather is clamped and masked back to 1.0 rather than reading slot -1.
    Reading it *after* ``step_arsenal_batch`` is deliberate: the charge program
    has already opened or closed this tick's hold, so a release stops paying
    the slowdown on the same tick the child fires.

    A nested pair authors the multiplier on its inner root, so the slowdown
    starts only once the hold clock passes the outer gate. Flat roots carry
    ``0.0`` here and are slowed from the first tick.
    """

    def gather(table: jax.Array) -> jax.Array:
        safe = jnp.clip(ability_hold_slot, 0, table.shape[-1] - 1)
        return jnp.take_along_axis(table, safe[..., None], axis=-1)[..., 0]

    authored = gather(ability_hold_speed_multiplier)
    starts_after = gather(ability_hold_speed_multiplier_after_seconds)
    active = (ability_hold_slot >= 0) & (ability_hold_seconds >= starts_after)
    return jnp.where(active, authored, jnp.float32(1.0))


def initialize_multi_actor_arena_state(
    state: ArsenalEnvironmentState,
    params: CombatParams,
    action_surface_runtime: ArsenalActionSurfaceRuntime | None = None,
) -> MultiActorArenaState:
    """Attach entity-indexed controller and terminal memory at reset."""

    batch, entity_count = state.combat.health.shape
    return MultiActorArenaState(
        arsenal=state,
        packed_locomotion=pack_entity_locomotion_state(
            initialize_entity_locomotion_state(state.combat, params)
        ),
        reward_memory=empty_multi_actor_reward_memory(batch, entity_count),
        action_surface_runtime=action_surface_runtime,
    )


def step_multi_actor_arena(
    state: MultiActorArenaState,
    controls: PolicyActorControls,
    assignment: PolicyActorAssignment,
    keys: jax.Array,
    params: CombatParams,
    commands: ArsenalCommands,
    config: ArsenalRuntimeConfig,
    *,
    scripted_actions: jax.Array | None = None,
    geometry: GeometryProvider | None = None,
    target_navigation_provider: TargetNavigationProvider | None = None,
    explosion_candidate_provider: ArsenalExplosionCandidateProvider | None = None,
) -> MultiActorArenaTransition:
    """Compose policy controls, combat, locomotion, and actor-owned rewards.

    This first production boundary requires one microtick per call. It is an
    explicit guard, not an approximation: applying a yaw edge once around a
    multi-microtick inner scan requires moving the entity locomotion bank into
    that scan. Callers cannot silently train on a different control cadence.
    """

    # ``jax.extend.core`` does not export this guard on JAX 0.9.x.
    microticks = jax.core.concrete_or_error(
        int,
        params.microticks,
        "multi-actor microticks must be statically known",
    )
    if microticks != 1:
        raise ValueError("multi-actor shared-arena v1 requires microticks=1")
    batch, entity_count = state.arsenal.combat.health.shape
    if assignment.actor_index.shape[0] != batch:
        raise ValueError("assignment batch does not match arena state")
    entity_controls = scatter_policy_actor_controls(
        controls,
        assignment,
        entity_count=entity_count,
    )
    # A row cannot be both scripted by the native-role surrogate and owned by
    # a policy. The policy side fails closed instead of racing two controllers.
    policy_controlled = (
        entity_controls.controlled & ~config.opponent_controller_mask
    )
    entity_controls = entity_controls._replace(controlled=policy_controlled)
    merged_commands = merge_policy_actor_commands(commands, entity_controls)
    if scripted_actions is None:
        scripted_actions = jnp.zeros((batch, ACTION_SIZE), dtype=jnp.float32)
    scripted_actions = jnp.asarray(scripted_actions, dtype=jnp.float32)
    if scripted_actions.shape != (batch, ACTION_SIZE):
        raise ValueError(
            f"scripted_actions must have shape {(batch, ACTION_SIZE)}"
        )
    actions = jnp.where(
        policy_controlled[:, AGENT_ENTITY, None],
        entity_controls.low_level_action[:, AGENT_ENTITY],
        scripted_actions,
    )
    compile_plan = plan_arsenal_compilation(params, config)
    transition = step_arsenal_batch(
        state.arsenal,
        actions,
        keys,
        params,
        merged_commands,
        config,
        geometry,
        target_navigation_provider,
        explosion_candidate_provider,
        compiled_microticks=compile_plan.microticks,
        compiled_null_loadout=compile_plan.null_loadout,
        compiled_has_item_programs=compile_plan.has_item_programs,
        compiled_has_projectiles=compile_plan.has_projectiles,
        compiled_has_areas=compile_plan.has_areas,
    )
    locomotion = synchronize_entity_zero_locomotion(
        transition.state.combat,
        state.locomotion,
    )
    nonzero_controlled = policy_controlled.at[:, AGENT_ENTITY].set(False)
    nonzero_controls = entity_controls._replace(controlled=nonzero_controlled)
    combat, locomotion = tick_entity_policy_locomotion(
        transition.state.combat,
        locomotion,
        nonzero_controls,
        transition.state.combat.last_motion_delta_seconds,
        params,
        geometry,
        hold_speed_multiplier=_hold_speed_multiplier(
            transition.state.arsenal.ability_hold_slot,
            transition.state.arsenal.ability_hold_seconds,
            config.loadout.ability_hold_speed_multiplier,
            config.loadout.ability_hold_speed_multiplier_after_seconds,
        ),
    )
    nonzero = jnp.arange(entity_count, dtype=jnp.int32)[None, :] != AGENT_ENTITY
    ticks_since_damage = jnp.where(
        transition.arsenal_info.entity_damage_received > jnp.float32(0.0),
        jnp.int32(0),
        locomotion.ticks_since_damage + jnp.int32(1),
    )
    locomotion = locomotion._replace(
        ticks_since_damage=jnp.where(
            nonzero,
            ticks_since_damage,
            locomotion.ticks_since_damage,
        )
    )
    arsenal_state = transition.state._replace(combat=combat)
    actor_components, reward_memory = per_actor_reward_components_from_totals(
        transition.arsenal_info.entity_damage_dealt,
        transition.arsenal_info.entity_damage_received,
        config.targeting_rules.team_id,
        combat.health,
        state.reward_memory,
    )
    valid = transition.arsenal_info.valid
    actor_components = jax.tree_util.tree_map(
        lambda value: jnp.where(valid[:, None], value, jnp.zeros_like(value)),
        actor_components,
    )
    reward_memory = jax.tree_util.tree_map(
        lambda new, old: jnp.where(valid[:, None], new, old),
        reward_memory,
        state.reward_memory,
    )
    policy_components = jax.tree_util.tree_map(
        lambda value: gather_policy_actor_rows(
            value,
            assignment,
            fill_value=False if value.dtype == jnp.bool_ else 0.0,
        ),
        actor_components,
    )
    reward = compose_per_actor_reward(policy_components, params)
    active = assignment.active & controls.valid
    reward = jnp.where(active, reward, jnp.float32(0.0))
    terminated = (
        policy_components.completion | policy_components.death
    ) & active
    truncated = transition.truncated[:, None] & active
    target_id = combat.engagement_target_id[:, AGENT_ENTITY]
    observation = observe_batch(
        combat,
        params,
        geometry,
        target_entity_id=target_id,
    )
    transition = transition._replace(
        state=arsenal_state,
        observation=observation,
        reward=reward[:, 0] if reward.shape[1] > 0 else transition.reward,
    )
    return MultiActorArenaTransition(
        state=MultiActorArenaState(
            arsenal=arsenal_state,
            packed_locomotion=pack_entity_locomotion_state(locomotion),
            reward_memory=reward_memory,
            action_surface_runtime=state.action_surface_runtime,
        ),
        arsenal_observation=transition.observation,
        arsenal_reward=transition.reward,
        arsenal_terminated=transition.terminated,
        arsenal_truncated=transition.truncated,
        combat_info=transition.combat_info,
        arsenal_info=transition.arsenal_info,
        reward_components=policy_components,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        controlled=policy_controlled,
    )


__all__ = [
    "MultiActorArenaState",
    "MultiActorArenaTransition",
    "initialize_multi_actor_arena_state",
    "step_multi_actor_arena",
]
