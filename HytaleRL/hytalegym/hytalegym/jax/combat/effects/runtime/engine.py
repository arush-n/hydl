"""Compiled combat rollout with an isolated projectile/hazard state."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.effects.factory import empty_effects_state
from hytalegym.jax.combat.effects.runtime.kernel import (
    apply_effect_commands,
    tick_effects,
)
from hytalegym.jax.combat.effects.runtime.scene import (
    combat_effects_scene,
    merge_combat_effects_scene,
)
from hytalegym.jax.combat.effects.schema.types import (
    CombatEffectCommands,
    CombatEffectsEnvironmentState,
    CombatEffectsInfo,
    CombatEffectsTrajectory,
    CombatEffectsTransition,
)
from hytalegym.jax.combat.env import (
    _combat_info,
    _microtick,
    _prepare_step,
    _select_state,
    _terminated,
    observe_batch,
    reset_batch,
)
from hytalegym.jax.combat.loadout.schema.types import MeleeLoadoutBatch
from hytalegym.jax.combat.observation.v1.runtime.encoder import combat_scene_from_state
from hytalegym.jax.combat.observation.v1.schema.types import CombatSceneFeatures
from hytalegym.jax.combat.types import (
    CombatParams,
    RewardComponents,
    compose_reward,
    sum_reward_components,
)


def reset_combat_effects_batch(
    keys: jax.Array,
    params: CombatParams,
) -> tuple[
    CombatEffectsEnvironmentState,
    jax.Array,
    CombatSceneFeatures,
]:
    """Reset calibrated flat combat plus empty fixed-capacity effects."""

    combat, observation = reset_batch(keys, params)
    effects = empty_effects_state(combat.position.shape[0])
    base_scene = combat_scene_from_state(combat, params, observation)
    scene = merge_combat_effects_scene(
        base_scene,
        combat_effects_scene(combat, effects, params),
    )
    return CombatEffectsEnvironmentState(combat, effects), observation, scene


def step_batch_effects(
    state: CombatEffectsEnvironmentState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    commands: CombatEffectCommands,
) -> CombatEffectsTransition:
    """Step calibrated flat combat with entity-only effects."""

    return _step_batch_effects(state, actions, keys, params, commands, None)


def step_batch_melee_loadout_effects(
    state: CombatEffectsEnvironmentState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    commands: CombatEffectCommands,
    loadout: MeleeLoadoutBatch,
) -> CombatEffectsTransition:
    """Step one batch with both pinned melee profiles and effects."""

    if loadout.weapon_id.shape != (state.combat.position.shape[0],):
        raise ValueError("loadout batch does not match combat state")
    return _step_batch_effects(
        state,
        actions,
        keys,
        params,
        commands,
        loadout,
    )


def _step_batch_effects(
    state: CombatEffectsEnvironmentState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    commands: CombatEffectCommands,
    loadout: MeleeLoadoutBatch | None,
) -> CombatEffectsTransition:
    spawned, projectile_spawned, hazard_spawned = apply_effect_commands(
        state.effects,
        commands,
    )
    applied, attack_requested, attack_accepted, scan_inputs = _prepare_step(
        state.combat,
        actions,
        keys,
        params,
        None,
        loadout,
    )
    effects_valid = spawned.failure_bits == jnp.uint32(0)
    applied = _select_state(effects_valid, applied, state.combat)
    attack_requested &= effects_valid
    attack_accepted &= effects_valid
    initial = CombatEffectsEnvironmentState(state.combat, state.effects)
    carry = CombatEffectsEnvironmentState(applied, spawned)

    def scan_tick(current, inputs):
        (
            microtick_index,
            pause_draw,
            strafe_frequency_draw,
            strafe_duration_draw,
            strafe_direction_draw,
        ) = inputs
        can_tick = (
            (microtick_index < params.microticks)
            & ~_terminated(current.combat)
            & (current.effects.failure_bits == jnp.uint32(0))
        )
        combat_candidate, combat_components = _microtick(
            current.combat,
            pause_draw,
            strafe_frequency_draw,
            strafe_duration_draw,
            strafe_direction_draw,
            params,
            None,
            loadout,
        )
        (
            combat_candidate,
            effects_candidate,
            effect_components,
            projectile_hits,
            hazard_hits,
            projectile_damage,
            hazard_damage,
        ) = tick_effects(
            combat_candidate,
            current.effects,
            combat_candidate.last_motion_delta_seconds,
            params,
        )
        candidate = CombatEffectsEnvironmentState(
            combat_candidate,
            effects_candidate,
        )
        next_state = _select_tree(can_tick, candidate, current)
        components = RewardComponents(
            target_damage=(
                combat_components.target_damage + effect_components.target_damage
            ),
            agent_damage=(
                combat_components.agent_damage + effect_components.agent_damage
            ),
            completion=(combat_components.completion | effect_components.completion),
            death=combat_components.death | effect_components.death,
        )
        components = jax.tree_util.tree_map(
            lambda value: jnp.where(can_tick, value, jnp.zeros_like(value)),
            components,
        )
        outputs = (components,) + tuple(
            jnp.where(can_tick, value, jnp.zeros_like(value))
            for value in (
                projectile_hits,
                hazard_hits,
                projectile_damage,
                hazard_damage,
            )
        )
        return next_state, outputs

    final, output = jax.lax.scan(scan_tick, carry, scan_inputs)
    (
        micro_components,
        projectile_hits,
        hazard_hits,
        projectile_damage,
        hazard_damage,
    ) = output
    failure_bits = final.effects.failure_bits
    valid = failure_bits == jnp.uint32(0)
    final_combat = _select_state(valid, final.combat, initial.combat)
    final_effects = _select_tree(valid, final.effects, initial.effects)
    final_effects = final_effects._replace(failure_bits=failure_bits)
    final = CombatEffectsEnvironmentState(final_combat, final_effects)
    reward_components = sum_reward_components(micro_components)
    reward_components = jax.tree_util.tree_map(
        lambda value: jnp.where(valid, value, jnp.zeros_like(value)),
        reward_components,
    )
    reward = jnp.where(
        valid,
        compose_reward(reward_components, params),
        jnp.float32(0.0),
    )
    observation = observe_batch(final.combat, params)
    base_scene = combat_scene_from_state(
        final.combat,
        params,
        observation,
    )
    scene = merge_combat_effects_scene(
        base_scene,
        combat_effects_scene(final.combat, final.effects, params),
    )
    done = _terminated(final.combat) | ~valid
    combat_info = _combat_info(
        final.combat,
        attack_requested & valid,
        attack_accepted & valid,
        params,
        None,
        reward_components=reward_components,
    )
    effects_info = CombatEffectsInfo(
        projectile_count=jnp.sum(
            final.effects.projectiles.active.astype(jnp.int32),
            axis=1,
        ),
        hazard_count=jnp.sum(
            final.effects.hazards.active.astype(jnp.int32),
            axis=1,
        ),
        projectile_spawned=projectile_spawned & valid,
        hazard_spawned=hazard_spawned & valid,
        projectile_hits=jnp.where(
            valid,
            jnp.sum(projectile_hits, axis=0, dtype=jnp.int32),
            jnp.int32(0),
        ),
        hazard_hits=jnp.where(
            valid,
            jnp.sum(hazard_hits, axis=0, dtype=jnp.int32),
            jnp.int32(0),
        ),
        projectile_damage=jnp.where(
            valid,
            jnp.sum(projectile_damage, axis=0, dtype=jnp.float32),
            jnp.float32(0.0),
        ),
        hazard_damage=jnp.where(
            valid,
            jnp.sum(hazard_damage, axis=0, dtype=jnp.float32),
            jnp.float32(0.0),
        ),
        failure_bits=failure_bits,
        valid=valid,
    )
    return CombatEffectsTransition(
        state=final,
        observation=observation,
        scene=scene,
        reward=reward,
        done=done,
        combat_info=combat_info,
        effects_info=effects_info,
    )


def rollout_batch_effects(
    state: CombatEffectsEnvironmentState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    commands: CombatEffectCommands,
) -> tuple[CombatEffectsEnvironmentState, CombatEffectsTrajectory]:
    """Collect one complete effect-aware trajectory inside ``lax.scan``."""

    return _rollout_batch_effects(
        state,
        actions,
        keys,
        params,
        commands,
        None,
    )


def rollout_batch_melee_loadout_effects(
    state: CombatEffectsEnvironmentState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    commands: CombatEffectCommands,
    loadout: MeleeLoadoutBatch,
) -> tuple[CombatEffectsEnvironmentState, CombatEffectsTrajectory]:
    """Collect an effect-aware trajectory with pinned melee profiles."""

    return _rollout_batch_effects(
        state,
        actions,
        keys,
        params,
        commands,
        loadout,
    )


def _rollout_batch_effects(
    state: CombatEffectsEnvironmentState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    commands: CombatEffectCommands,
    loadout: MeleeLoadoutBatch | None,
) -> tuple[CombatEffectsEnvironmentState, CombatEffectsTrajectory]:
    def rollout_step(current, inputs):
        action, key, command = inputs
        transition = _step_batch_effects(
            current,
            action,
            key,
            params,
            command,
            loadout,
        )
        outputs = (
            transition.observation,
            transition.scene,
            transition.reward,
            transition.done,
            transition.combat_info,
            transition.effects_info,
        )
        return transition.state, outputs

    final, output = jax.lax.scan(
        rollout_step,
        state,
        (actions, keys, commands),
    )
    observation, scene, reward, done, combat_info, effects_info = output
    return final, CombatEffectsTrajectory(
        observation=observation,
        scene=scene,
        reward=reward,
        done=done,
        combat_info=combat_info,
        effects_info=effects_info,
    )


def _select_tree(mask: jax.Array, selected, fallback):
    return jax.tree_util.tree_map(
        lambda yes, no: jnp.where(
            mask.reshape((mask.shape[0],) + (1,) * (yes.ndim - 1)),
            yes,
            no,
        ),
        selected,
        fallback,
    )
