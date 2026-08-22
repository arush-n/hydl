"""Combat binding for World's source-backed NPC Walk movement-state producer."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.types import AGENT_ENTITY, CombatParams, CombatState
from hytalegym.jax.world import (
    MOTION_KIND_ASCENDING,
    MOTION_KIND_DROPPING,
    MOTION_KIND_MOVING,
    MOTION_KIND_STANDING,
    NpcWalkMovementState,
    NpcWalkMovementStateResult,
    empty_npc_walk_movement_state,
    npc_walk_movement_state_result,
)
from hytalegym.rulesets.movement_projection import (
    load_actor_walk_movement_projection,
)


class ActorWalkMovementConfig(NamedTuple):
    """Device-ready, role-derived inputs for one Walk controller."""

    component_selector: jax.Array
    run_threshold: jax.Array
    run_threshold_range: jax.Array
    hover_height: jax.Array
    ascent_animation_type: jax.Array
    descent_animation_type: jax.Array
    minimum_descent_animation_height: jax.Array


def hytale_0_5_7_actor_walk_movement_config() -> ActorWalkMovementConfig:
    """Compile the asset-pinned Kweebec Walk projection for JAX."""

    source = load_actor_walk_movement_projection()
    return ActorWalkMovementConfig(
        component_selector=jnp.asarray(
            source.component_selector,
            dtype=jnp.float32,
        ),
        run_threshold=jnp.asarray(source.run_threshold, dtype=jnp.float32),
        run_threshold_range=jnp.asarray(
            source.run_threshold_range,
            dtype=jnp.float32,
        ),
        hover_height=jnp.asarray(source.hover_height, dtype=jnp.float32),
        ascent_animation_type=jnp.asarray(
            source.ascent_animation_type,
            dtype=jnp.int32,
        ),
        descent_animation_type=jnp.asarray(
            source.descent_animation_type,
            dtype=jnp.int32,
        ),
        minimum_descent_animation_height=jnp.asarray(
            source.minimum_descent_animation_height,
            dtype=jnp.float32,
        ),
    )


def empty_actor_walk_movement_state(batch_size: int) -> NpcWalkMovementState:
    """Return the public World's canonical unavailable recurrent state."""

    return empty_npc_walk_movement_state(batch_size)


def actor_walk_movement_state_result(
    previous: NpcWalkMovementState,
    combat: CombatState,
    params: CombatParams,
    *,
    controller_in_fluid: jax.Array,
    controller_available: jax.Array,
    config: ActorWalkMovementConfig,
) -> NpcWalkMovementStateResult:
    """Classify one simulated actor tick and run the public Walk producer.

    Combat derives only motion facts it actually simulates. Client/action-owned
    fields stay unavailable in ``previous`` and are therefore never fabricated.
    """

    batch = combat.health.shape[0]
    velocity = combat.velocity[:, AGENT_ENTITY]
    planar_speed = jnp.linalg.norm(velocity[:, (0, 2)], axis=1)
    moving = planar_speed > jnp.float32(1.0e-9)
    external_motion = (
        jnp.abs(combat.agent_applied_vertical_velocity) > jnp.float32(1.0e-9)
    ) | combat.grounded_with_residual_velocity
    ascending = (
        ~combat.agent_grounded
        & (velocity[:, 1] > jnp.float32(1.0e-9))
        & ~external_motion
    )
    dropping = external_motion | (~combat.agent_grounded & ~ascending)
    motion_kind = jnp.where(
        moving,
        jnp.int32(MOTION_KIND_MOVING),
        jnp.int32(MOTION_KIND_STANDING),
    )
    motion_kind = jnp.where(
        ascending,
        jnp.int32(MOTION_KIND_ASCENDING),
        motion_kind,
    )
    motion_kind = jnp.where(
        dropping,
        jnp.int32(MOTION_KIND_DROPPING),
        motion_kind,
    )
    steering = jnp.stack(
        (
            combat.desired_velocity[:, 0],
            jnp.zeros((batch,), dtype=jnp.float32),
            combat.desired_velocity[:, 1],
        ),
        axis=1,
    )
    fall_height = jnp.maximum(
        jnp.float32(0.0),
        combat.agent_fall_start_y - combat.position[:, AGENT_ENTITY, 1],
    )
    # The bridge's jump verb enters Walk through external ``setVelocity``.
    # Native labels that force-pushed arc DROPPING; the controller-owned
    # ``jumping`` flag is written only by Walk's obstacle ``tryClimb`` path.
    jumping = jnp.zeros((batch,), dtype=jnp.bool_)
    return npc_walk_movement_state_result(
        previous,
        velocity,
        steering,
        motion_kind,
        controller_available=controller_available,
        in_fluid=controller_in_fluid,
        on_ground=combat.agent_grounded,
        hover_height=config.hover_height,
        controller_jumping=jumping,
        component_selector=config.component_selector,
        maximum_horizontal_speed=params.agent_max_speed,
        fast_motion_threshold=config.run_threshold,
        fast_motion_threshold_range=config.run_threshold_range,
        ascent_animation_type=config.ascent_animation_type,
        descent_animation_type=config.descent_animation_type,
        predicted_fall_height=fall_height,
        minimum_descent_animation_height=(
            config.minimum_descent_animation_height
        ),
    )


__all__ = [
    "ActorWalkMovementConfig",
    "actor_walk_movement_state_result",
    "empty_actor_walk_movement_state",
    "hytale_0_5_7_actor_walk_movement_config",
]
