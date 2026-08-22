"""Leaf helpers extracted verbatim from env.py."""

import jax
import jax.numpy as jnp
from hytalegym.geometry.contract import CELL_RADIUS
from hytalegym.jax.world.region.geometry import region_aabb_core_available
from hytalegym.jax.world.region.types import RegionGeometryState
from hytalegym.jax.combat.types import AGENT_ENTITY, TARGET_ENTITY, CombatParams, CombatState
from .motion import (  # noqa: F401  (re-exported: callers unchanged)
    GeometryProvider,
    TargetNavigationProvider,
    _ACTOR_CROUCHING_INDEX,
    _combat_perception_line_of_sight,
    _combat_positions_perception_line_of_sight,
    _movement_medium_result,
    _pursue_speed_scale,
    _resolve_aabb_motion_result,
    _target_detects_actor,
    _target_navigation_rejected,
    _tick_target_motion,
    _tick_target_vertical_motion,
)
from .attack import (  # noqa: F401  (re-exported: callers unchanged)
    DIRECTIONAL_KNOCKBACK_FALLBACK_SQUARED_DISTANCE,
    NO_QUEUE_TICK,
    _combat_hitbox_line_of_sight,
    _pending_target_knockback,
    _selector_sweep_window,
    _target_selector_intersects,
    _tick_target_attack,
)


IDLE = 0


WINDUP = 1


SWEEP = 2


RECOVERY = 3


COOLDOWN = 4


def _combat_window_exhausted(
    state: CombatState,
    geometry: GeometryProvider,
) -> jax.Array:
    """Flag geometry that cannot prove both combatants remain covered."""

    batch = state.position.shape[0]
    if isinstance(geometry, RegionGeometryState):
        agent_bounds = (
            geometry.agent_bounds
            if geometry.agent_bounds.shape[0] == batch
            else jnp.broadcast_to(geometry.agent_bounds, (batch, 6))
        )
        target_bounds = (
            geometry.target_bounds
            if geometry.target_bounds.shape[0] == batch
            else jnp.broadcast_to(geometry.target_bounds, (batch, 6))
        )
        return ~(
            region_aabb_core_available(
                geometry,
                state.position[:, AGENT_ENTITY],
                agent_bounds,
            )
            & region_aabb_core_available(
                geometry,
                state.position[:, TARGET_ENTITY],
                target_bounds,
            )
        )
    origin = (
        geometry.origin
        if geometry.origin.shape[0] == batch
        else jnp.broadcast_to(geometry.origin, (batch, 3))
    )
    relative = jnp.floor(state.position).astype(jnp.int32) - origin[:, None, :]
    return jnp.any(
        (relative < -CELL_RADIUS) | (relative > CELL_RADIUS),
        axis=(1, 2),
    )


def _target_phase(
    state: CombatState,
    params: CombatParams,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    target_alive = state.health[:, TARGET_ENTITY] > 0.0
    active = jnp.broadcast_to(params.target_active, target_alive.shape)
    has_root = active & target_alive & (state.target_attack_index >= 0)
    clipped_index = jnp.clip(
        state.target_attack_index,
        0,
        params.target_windup_ticks.shape[0] - 1,
    )
    elapsed = state.target_attack_elapsed_ticks
    windup = params.target_windup_ticks[clipped_index]
    sweep = params.target_sweep_ticks[clipped_index]
    recovery = params.target_recovery_ticks[clipped_index]
    root_phase = jnp.where(
        elapsed <= 0,
        IDLE,
        jnp.where(
            elapsed <= windup,
            WINDUP,
            jnp.where(elapsed <= windup + sweep, SWEEP, RECOVERY),
        ),
    )
    root_progress = jnp.where(
        root_phase == WINDUP,
        elapsed / jnp.maximum(windup, 1),
        jnp.where(
            root_phase == SWEEP,
            (elapsed - windup) / jnp.maximum(sweep, 1),
            jnp.where(
                root_phase == RECOVERY,
                (elapsed - windup - sweep) / jnp.maximum(recovery, 1),
                0.0,
            ),
        ),
    ).astype(jnp.float32)
    root_progress = jnp.clip(root_progress, 0.0, 1.0)

    queued = active & target_alive & state.target_attack_queued
    cooling = (
        active
        & target_alive
        & (state.target_attack_cooldown_seconds > 0.0)
        & (state.last_target_attack_index >= 0)
    )
    phase = jnp.where(
        has_root,
        root_phase,
        jnp.where(queued | cooling, COOLDOWN, IDLE),
    ).astype(jnp.int32)
    progress = jnp.where(
        has_root,
        root_progress,
        jnp.where(queued | cooling, 1.0, 0.0),
    ).astype(jnp.float32)
    reported_index = jnp.where(
        has_root,
        state.target_attack_index,
        jnp.where(cooling, state.last_target_attack_index, -1),
    ).astype(jnp.int32)
    reported_elapsed = jnp.where(
        has_root,
        elapsed,
        0,
    ).astype(jnp.int32)
    return phase, progress, reported_index, reported_elapsed
