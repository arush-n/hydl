"""Pinned Hytale 0.5.7 effect-command profiles with explicit provenance."""

from __future__ import annotations

import jax.numpy as jnp

from hytalegym.jax.combat.effects.schema.contract import (
    OUTLANDER_ARROW_AUTHORED_LIFETIME_SECONDS,
    OUTLANDER_ARROW_DAMAGE,
    OUTLANDER_ARROW_DEAD_TIME_SECONDS,
    OUTLANDER_ARROW_DEPTH_OFFSET,
    OUTLANDER_ARROW_GRAVITY,
    OUTLANDER_ARROW_MODEL_HALF_EXTENT,
    OUTLANDER_ARROW_MUZZLE_VELOCITY,
    OUTLANDER_ARROW_SERVER_DESPAWN_SECONDS,
    OUTLANDER_ARROW_TERMINAL_VELOCITY,
    OUTLANDER_ARROW_VERTICAL_OFFSET,
    PROJECTILE_KIND_OUTLANDER_HUNTER_ARROW,
)
from hytalegym.jax.combat.effects.factory import empty_effect_commands
from hytalegym.jax.combat.effects.schema.types import CombatEffectCommands
from hytalegym.jax.combat.types import TARGET_ENTITY


def outlander_hunter_arrow_commands(
    look_position: jnp.ndarray,
    yaw_radians: jnp.ndarray,
    pitch_radians: jnp.ndarray,
    *,
    clear_flight_certified: bool = False,
) -> CombatEffectCommands:
    """Build the legacy Outlander arrow launch command.

    The default fails closed because the arrow normally needs world collision.
    Set ``clear_flight_certified`` only for a fixture whose entire swept
    corridor is already known to contain no collidable world geometry.
    """

    position = jnp.asarray(look_position, dtype=jnp.float32)
    yaw = jnp.asarray(yaw_radians, dtype=jnp.float32)
    pitch = jnp.asarray(pitch_radians, dtype=jnp.float32)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("look_position must have shape (batch, 3)")
    batch = position.shape[0]
    if yaw.shape != (batch,) or pitch.shape != (batch,):
        raise ValueError("yaw_radians and pitch_radians must have shape (batch,)")

    heading_x = -jnp.sin(yaw)
    heading_z = -jnp.cos(yaw)
    pitch_xz = jnp.cos(pitch)
    direction = jnp.stack(
        (heading_x * pitch_xz, jnp.sin(pitch), heading_z * pitch_xz),
        axis=1,
    )
    start = (
        position
        + direction * jnp.float32(OUTLANDER_ARROW_DEPTH_OFFSET)
        + jnp.asarray(
            [0.0, -OUTLANDER_ARROW_VERTICAL_OFFSET, 0.0],
            dtype=jnp.float32,
        )
    )
    commands = empty_effect_commands(batch)
    projectile = commands.projectile._replace(
        requested=jnp.ones((batch,), dtype=jnp.bool_),
        position=start,
        velocity=direction * jnp.float32(OUTLANDER_ARROW_MUZZLE_VELOCITY),
        half_extent=jnp.full(
            (batch, 3),
            OUTLANDER_ARROW_MODEL_HALF_EXTENT,
            dtype=jnp.float32,
        ),
        despawn_seconds=jnp.full(
            (batch,),
            OUTLANDER_ARROW_SERVER_DESPAWN_SECONDS,
            dtype=jnp.float32,
        ),
        authored_lifetime_seconds=jnp.full(
            (batch,),
            OUTLANDER_ARROW_AUTHORED_LIFETIME_SECONDS,
            dtype=jnp.float32,
        ),
        damage=jnp.full(
            (batch,),
            OUTLANDER_ARROW_DAMAGE,
            dtype=jnp.float32,
        ),
        gravity=jnp.full(
            (batch,),
            OUTLANDER_ARROW_GRAVITY,
            dtype=jnp.float32,
        ),
        terminal_velocity=jnp.full(
            (batch,),
            OUTLANDER_ARROW_TERMINAL_VELOCITY,
            dtype=jnp.float32,
        ),
        dead_time_seconds=jnp.full(
            (batch,),
            OUTLANDER_ARROW_DEAD_TIME_SECONDS,
            dtype=jnp.float32,
        ),
        velocity_scale=jnp.full(
            (batch,),
            OUTLANDER_ARROW_MUZZLE_VELOCITY,
            dtype=jnp.float32,
        ),
        kind=jnp.full(
            (batch,),
            PROJECTILE_KIND_OUTLANDER_HUNTER_ARROW,
            dtype=jnp.int32,
        ),
        owner_entity_id=jnp.full(
            (batch,),
            TARGET_ENTITY,
            dtype=jnp.int32,
        ),
        hostile=jnp.ones((batch,), dtype=jnp.bool_),
        entity_collision_only=jnp.full(
            (batch,),
            clear_flight_certified,
            dtype=jnp.bool_,
        ),
    )
    return commands._replace(projectile=projectile)
