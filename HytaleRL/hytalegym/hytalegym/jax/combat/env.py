"""Pure batched JAX combat transition matching Java combat model version 9."""

# This module remains the compatibility import surface after its implementation
# was split into private shards; the imports below are intentional re-exports.
# ruff: noqa: F401

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.geometry.contract import CELL_RADIUS
from hytalegym.jax.world import (
    geometry_hitbox_line_of_sight_result,
    geometry_perception_line_of_sight_result,
)
from hytalegym.jax.world.geometry import (
    aabb_grounded as local_aabb_grounded,
    aabb_support_sweep as local_aabb_support_sweep,
    movement_medium_result as local_movement_medium_result,
    resolve_aabb_motion as local_resolve_aabb_motion,
)
from hytalegym.jax.world.geometry.atlas import select_geometry_tile
from hytalegym.jax.world.region.geometry import (
    region_aabb_core_available,
    region_aabb_grounded,
    region_aabb_support_sweep,
    region_movement_medium_result,
    region_resolve_aabb_motion,
)
from hytalegym.jax.world.region.types import RegionGeometryState
from hytalegym.jax.world.entities.movement_states import MOVEMENT_STATE_ORDER
from hytalegym.jax.world.types import GeometryAtlas, GeometryState
from hytalegym.jax.combat.loadout.schema.types import MeleeLoadoutBatch
from hytalegym.jax.combat.motion.navigation import (
    TargetNavigationStep,
    validate_target_navigation_step,
)
from hytalegym.jax.combat.motion.movement_state import (
    actor_walk_movement_state_result,
    empty_actor_walk_movement_state,
    hytale_0_5_7_actor_walk_movement_config,
)
from hytalegym.jax.combat.types import (
    ACTION_ATTACK,
    ACTION_BACK,
    ACTION_FORWARD,
    ACTION_JUMP,
    ACTION_LEFT,
    ACTION_PITCH_DELTA,
    ACTION_RIGHT,
    ACTION_YAW_DELTA,
    AGENT_ENTITY,
    ENTITY_COUNT,
    MAX_MICROTICKS,
    TARGET_ENTITY,
    CombatInfo,
    CombatParams,
    CombatState,
    CombatTrajectory,
    RewardComponents,
    compose_reward,
    sum_reward_components,
)
from .runtime.math import (  # noqa: F401  (re-exported: callers unchanged)
    _approach,
    _approach_angle,
    _approach_target_angle,
    _canonical_bearing,
    _motion_delta,
    _normalize_degrees,
    _randint_batch,
    _rotation_x,
    _rotation_y,
    _rotation_z,
    _round_to_scale,
    _select_state,
    _target_distance,
    _target_facing_error,
    _target_head_facing_error,
    _terminated,
    _uniform_batch,
)
from .opponents.legacy.controller import (  # noqa: F401  (re-exported: callers unchanged)
    COOLDOWN,
    DIRECTIONAL_KNOCKBACK_FALLBACK_SQUARED_DISTANCE,
    GeometryProvider,
    IDLE,
    NO_QUEUE_TICK,
    RECOVERY,
    SWEEP,
    TargetNavigationProvider,
    WINDUP,
    _ACTOR_CROUCHING_INDEX,
    _combat_hitbox_line_of_sight,
    _combat_perception_line_of_sight,
    _combat_positions_perception_line_of_sight,
    _combat_window_exhausted,
    _movement_medium_result,
    _pending_target_knockback,
    _pursue_speed_scale,
    _resolve_aabb_motion_result,
    _selector_sweep_window,
    _target_detects_actor,
    _target_navigation_rejected,
    _target_phase,
    _target_selector_intersects,
    _tick_target_attack,
    _tick_target_motion,
)
from .runtime.agent import (  # noqa: F401  (re-exported: callers unchanged)
    _ACTOR_WALK_MOVEMENT_CONFIG,
    _aabb_grounded_result,
    _aabb_support_sweep_result,
    _damp_force_velocity,
    _force_pushed_agent_motion,
    _gather_loadout_profile,
    _loadout_attack_count,
    _loadout_profile_index,
    _loadout_row_valid,
    _refresh_agent_walk_movement_state,
    _tick_agent_attack,
    _tick_agent_geometry_motion,
    _tick_agent_motion,
    _tick_agent_vertical,
)
from .runtime.reset import (  # noqa: F401  (re-exported: callers unchanged)
    ActorEvidence,
    TargetDetection,
    TargetEvidence,
    _encode_actor_observation,
    _finish_atlas_reset,
    _finish_geometry_reset,
    _gather_entity_axis,
    _target_detection_from_positions,
    observe_batch,
    reset_batch,
    reset_batch_geometry,
    reset_batch_geometry_at,
    reset_batch_geometry_atlas,
    reset_batch_geometry_atlas_at,
    reset_batch_region,
    reset_batch_region_at,
    target_evidence,
)
from .runtime.step import (  # noqa: F401  (re-exported: callers unchanged)
    _apply_action,
    _apply_pending_target_damage,
    _combat_info,
    _geometry_grounded,
    _microtick,
    _tick_regeneration,
    active_combat_state,
)


# DirectionalKnockback.java:39-45; this branch precedes normalization.
def step_batch(
    state: CombatState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
) -> tuple[CombatState, jax.Array, jax.Array, jax.Array, CombatInfo]:
    """Execute a complete batched policy decision inside one JAX program.

    The edge-triggered action is applied once. Movement intent persists while
    ``lax.scan`` advances up to four engine microticks, stopping each finished
    environment independently. This legacy 24-value/9-skill compatibility
    core has no guard or dodge state and is not a defensive-fidelity surface;
    use the Arsenal environment for defensive curricula.
    """

    return _step_batch(state, actions, keys, params, None)


def step_batch_melee_loadout(
    state: CombatState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    loadout: MeleeLoadoutBatch,
) -> tuple[CombatState, jax.Array, jax.Array, jax.Array, CombatInfo]:
    """Step flat combat with one episode-pinned melee loadout per row."""

    batch = state.position.shape[0]
    if loadout.weapon_id.shape != (batch,):
        raise ValueError("loadout batch does not match combat state")
    return _step_batch(
        state,
        actions,
        keys,
        params,
        None,
        loadout,
    )


def step_batch_geometry(
    state: CombatState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    geometry: GeometryState,
    *,
    target_navigation_provider: TargetNavigationProvider | None = None,
) -> tuple[CombatState, jax.Array, jax.Array, jax.Array, CombatInfo]:
    """Step combat with versioned terrain collision and block LOS enabled."""

    return _step_batch(
        state,
        actions,
        keys,
        params,
        geometry,
        target_navigation_provider=target_navigation_provider,
    )


def step_batch_region(
    state: CombatState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    geometry: RegionGeometryState,
    *,
    target_navigation_provider: TargetNavigationProvider | None = None,
) -> tuple[CombatState, jax.Array, jax.Array, jax.Array, CombatInfo]:
    """Step combat with direct Region v1 collision, support, medium, and LOS."""

    return _step_batch(
        state,
        actions,
        keys,
        params,
        geometry,
        target_navigation_provider=target_navigation_provider,
    )


def step_batch_geometry_atlas(
    state: CombatState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    atlas: GeometryAtlas,
    environment_world_id: jax.Array,
    *,
    target_navigation_provider: TargetNavigationProvider | None = None,
) -> tuple[CombatState, jax.Array, jax.Array, jax.Array, CombatInfo]:
    """Select an overlapping exact tile and execute one compiled decision."""

    selected = select_geometry_tile(
        atlas,
        state.position,
        environment_world_id,
    )
    state = state._replace(
        geometry_exhausted=state.geometry_exhausted | ~selected.available
    )
    return _step_batch(
        state,
        actions,
        keys,
        params,
        selected.geometry,
        target_navigation_provider=target_navigation_provider,
    )


def _step_batch(
    state: CombatState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    geometry: GeometryProvider | None,
    agent_loadout: MeleeLoadoutBatch | None = None,
    target_navigation_provider: TargetNavigationProvider | None = None,
    motion_delta_seconds: jax.Array | None = None,
) -> tuple[CombatState, jax.Array, jax.Array, jax.Array, CombatInfo]:
    """Advance one decision, optionally replaying measured microtick deltas.

    ``motion_delta_seconds`` is a fidelity-only, prevalidated float32
    ``[batch, MAX_MICROTICKS]`` input. Omitting it retains the ruleset timing
    profile exactly.
    """

    batch = state.position.shape[0]
    if motion_delta_seconds is not None:
        motion_delta_seconds = jnp.asarray(
            motion_delta_seconds,
            dtype=jnp.float32,
        )
        expected_shape = (batch, MAX_MICROTICKS)
        if motion_delta_seconds.shape != expected_shape:
            raise ValueError(
                "motion_delta_seconds must have shape "
                f"{expected_shape}"
            )
    applied, attack_requested, attack_accepted, scan_inputs = _prepare_step(
        state,
        actions,
        keys,
        params,
        geometry,
        agent_loadout,
    )

    def scan_tick(
        current: CombatState,
        inputs: tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
    ) -> tuple[CombatState, RewardComponents]:
        (
            microtick_index,
            pause_draw,
            strafe_frequency_draw,
            strafe_duration_draw,
            strafe_direction_draw,
        ) = inputs
        can_tick = (microtick_index < params.microticks) & ~_terminated(current)
        candidate, components = _microtick(
            current,
            pause_draw,
            strafe_frequency_draw,
            strafe_duration_draw,
            strafe_direction_draw,
            params,
            geometry,
            agent_loadout,
            target_navigation_provider,
            (
                None
                if motion_delta_seconds is None
                else motion_delta_seconds[:, microtick_index]
            ),
        )
        next_state = _select_state(can_tick, candidate, current)
        components = jax.tree_util.tree_map(
            lambda value: jnp.where(can_tick, value, jnp.zeros_like(value)),
            components,
        )
        return next_state, components

    final_state, micro_components = jax.lax.scan(
        scan_tick,
        applied,
        scan_inputs,
    )
    reward_components = sum_reward_components(micro_components)
    reward = compose_reward(reward_components, params)
    done = _terminated(final_state)
    observation = observe_batch(final_state, params, geometry)
    info = _combat_info(
        final_state,
        attack_requested,
        attack_accepted,
        params,
        geometry,
        reward_components,
    )
    return final_state, observation, reward, done, info


def _prepare_step(
    state: CombatState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    geometry: GeometryProvider | None,
    agent_loadout: MeleeLoadoutBatch | None = None,
    *,
    initially_terminated: jax.Array | None = None,
) -> tuple[
    CombatState,
    jax.Array,
    jax.Array,
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
]:
    """Apply the edge action and prepare deterministic microtick inputs.

    ``initially_terminated`` lets roster-aware callers replace the legacy
    two-entity ``TARGET_ENTITY`` terminal predicate. Omitting it preserves the
    certified compatibility path exactly.
    """

    split_keys = jax.vmap(lambda key: jax.random.split(key, MAX_MICROTICKS + 1))(keys)
    if agent_loadout is None:
        agent_pause_draw = _uniform_batch(
            split_keys[:, 0],
            params.agent_attack_pause_min_seconds,
            params.agent_attack_pause_max_seconds,
        )
    else:
        pause_index = _loadout_profile_index(
            agent_loadout,
            state.attack_sequence_index,
        )
        pause_minimum = _gather_loadout_profile(
            agent_loadout.cooldown_min_seconds,
            pause_index,
        )
        pause_maximum = _gather_loadout_profile(
            agent_loadout.cooldown_max_seconds,
            pause_index,
        )
        pause_unit = _uniform_batch(
            split_keys[:, 0],
            jnp.float32(0.0),
            jnp.float32(1.0),
        )
        agent_pause_draw = pause_minimum + (pause_maximum - pause_minimum) * pause_unit
    target_pause_draws = jax.vmap(
        lambda env_keys: jax.vmap(
            lambda key: jax.random.uniform(
                key,
                (),
                minval=params.target_attack_pause_min_seconds,
                maxval=params.target_attack_pause_max_seconds,
                dtype=jnp.float32,
            )
        )(env_keys)
    )(split_keys[:, 1:])
    target_strafe_frequency_draws = jax.vmap(
        lambda env_keys: jax.vmap(
            lambda key: jax.random.uniform(
                jax.random.fold_in(key, 1),
                (),
                minval=params.target_strafe_frequency_min_seconds,
                maxval=params.target_strafe_frequency_max_seconds,
                dtype=jnp.float32,
            )
        )(env_keys)
    )(split_keys[:, 1:])
    target_strafe_duration_draws = jax.vmap(
        lambda env_keys: jax.vmap(
            lambda key: jax.random.uniform(
                jax.random.fold_in(key, 2),
                (),
                minval=params.target_strafe_duration_min_seconds,
                maxval=params.target_strafe_duration_max_seconds,
                dtype=jnp.float32,
            )
        )(env_keys)
    )(split_keys[:, 1:])
    target_strafe_direction_draws = jax.vmap(
        lambda env_keys: jax.vmap(
            lambda key: jnp.where(
                jax.random.bernoulli(jax.random.fold_in(key, 3)),
                jnp.int32(1),
                jnp.int32(-1),
            )
        )(env_keys)
    )(split_keys[:, 1:])

    if initially_terminated is None:
        initially_done = _terminated(state)
    else:
        initially_done = jnp.asarray(initially_terminated, dtype=jnp.bool_)
        if initially_done.shape != state.tick_count.shape:
            raise ValueError(
                "initially_terminated must have one value per environment"
            )
    applied, attack_requested, attack_accepted = _apply_action(
        state,
        actions,
        agent_pause_draw,
        params,
        geometry,
        agent_loadout,
    )
    applied = _select_state(~initially_done, applied, state)
    attack_requested = attack_requested & ~initially_done
    attack_accepted = attack_accepted & ~initially_done

    scan_inputs = (
        jnp.arange(MAX_MICROTICKS, dtype=jnp.int32),
        jnp.swapaxes(target_pause_draws, 0, 1),
        jnp.swapaxes(target_strafe_frequency_draws, 0, 1),
        jnp.swapaxes(target_strafe_duration_draws, 0, 1),
        jnp.swapaxes(target_strafe_direction_draws, 0, 1),
    )
    return applied, attack_requested, attack_accepted, scan_inputs


def rollout_batch(
    state: CombatState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
) -> tuple[CombatState, CombatTrajectory]:
    """Collect a time-major trajectory with no Python policy-step boundary.

    ``actions`` has shape ``[T, B, ACTION_SIZE]`` and ``keys`` contains one
    key per time/environment pair. This Phase-1 collector intentionally leaves
    policy inference outside; the recurrent policy will replace the supplied
    action tensor inside this same scan in Phase 3.
    """

    return _rollout_batch(state, actions, keys, params, None, None)


def rollout_batch_melee_loadout(
    state: CombatState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    loadout: MeleeLoadoutBatch,
) -> tuple[CombatState, CombatTrajectory]:
    """Collect one compiled flat trajectory with episode-pinned loadouts."""

    batch = state.position.shape[0]
    if loadout.weapon_id.shape != (batch,):
        raise ValueError("loadout batch does not match combat state")
    return _rollout_batch(
        state,
        actions,
        keys,
        params,
        None,
        loadout,
    )


def rollout_batch_geometry(
    state: CombatState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    geometry: GeometryState,
    *,
    target_navigation_provider: TargetNavigationProvider | None = None,
) -> tuple[CombatState, CombatTrajectory]:
    """Collect a compiled trajectory with geometry-aware transitions."""

    return _rollout_batch(
        state,
        actions,
        keys,
        params,
        geometry,
        None,
        target_navigation_provider,
    )


def rollout_batch_region(
    state: CombatState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    geometry: RegionGeometryState,
    *,
    target_navigation_provider: TargetNavigationProvider | None = None,
) -> tuple[CombatState, CombatTrajectory]:
    """Collect one compiled trajectory directly against Region v1."""

    return _rollout_batch(
        state,
        actions,
        keys,
        params,
        geometry,
        None,
        target_navigation_provider,
    )


def rollout_batch_geometry_atlas(
    state: CombatState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    atlas: GeometryAtlas,
    environment_world_id: jax.Array,
    *,
    target_navigation_provider: TargetNavigationProvider | None = None,
) -> tuple[CombatState, CombatTrajectory]:
    """Collect a trajectory while recentering among exact tiles inside scan."""

    def rollout_step(current, inputs):
        action, key = inputs
        next_state, observation, reward, done, info = step_batch_geometry_atlas(
            current,
            action,
            key,
            params,
            atlas,
            environment_world_id,
            target_navigation_provider=target_navigation_provider,
        )
        return next_state, (observation, reward, done, info)

    final_state, output = jax.lax.scan(
        rollout_step,
        state,
        (actions, keys),
    )
    observation, reward, done, info = output
    return final_state, CombatTrajectory(
        observation=observation,
        reward=reward,
        done=done,
        info=info,
    )
def _rollout_batch(
    state: CombatState,
    actions: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    geometry: GeometryProvider | None,
    agent_loadout: MeleeLoadoutBatch | None,
    target_navigation_provider: TargetNavigationProvider | None = None,
) -> tuple[CombatState, CombatTrajectory]:
    def rollout_step(
        current: CombatState,
        inputs: tuple[jax.Array, jax.Array],
    ) -> tuple[
        CombatState,
        tuple[jax.Array, jax.Array, jax.Array, CombatInfo],
    ]:
        action, key = inputs
        next_state, observation, reward, done, info = _step_batch(
            current,
            action,
            key,
            params,
            geometry,
            agent_loadout,
            target_navigation_provider,
        )
        return next_state, (observation, reward, done, info)

    final_state, output = jax.lax.scan(
        rollout_step,
        state,
        (actions, keys),
    )
    observation, reward, done, info = output
    return final_state, CombatTrajectory(
        observation=observation,
        reward=reward,
        done=done,
        info=info,
    )
