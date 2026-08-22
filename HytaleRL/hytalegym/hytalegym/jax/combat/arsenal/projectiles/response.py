"""Pure StandardPhysics response for one projectile/block contact."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class StandardProjectileContactResponse(NamedTuple):
    """Vectorized result of one source-derived block-contact response."""

    position: jax.Array
    velocity: jax.Array
    reflected_velocity: jax.Array
    bounce_count: jax.Array
    bounced: jax.Array
    rolling: jax.Array
    stopped: jax.Array
    impacted: jax.Array
    on_ground: jax.Array


def standard_projectile_contact_response(
    *,
    contact_mask: jax.Array,
    end: jax.Array,
    contact_point: jax.Array,
    contact_normal: jax.Array,
    velocity: jax.Array,
    dt_seconds: jax.Array,
    bounciness: jax.Array,
    bounce_limit: jax.Array,
    bounce_count_limit: jax.Array,
    bounce_count: jax.Array,
    allow_rolling: jax.Array,
    rolling_friction_factor: jax.Array,
    sticks_vertically: jax.Array | bool = False,
) -> StandardProjectileContactResponse:
    """Apply Hytale's single-plane StandardPhysics contact transition.

    The native provider can consume a tangent remainder through more than one
    surface in a tick. This kernel preserves that remainder on the first
    plane; the caller remains responsible for any additional collision query.
    """

    normal_squared = jnp.sum(contact_normal * contact_normal, axis=-1)
    safe_normal_squared = jnp.maximum(
        normal_squared,
        jnp.finfo(jnp.float32).tiny,
    )
    normal_projection = (
        jnp.sum(velocity * contact_normal, axis=-1) / safe_normal_squared
    )
    reflected = velocity - (
        jnp.float32(2.0) * normal_projection[..., None] * contact_normal
    )

    next_bounce_count = jnp.where(
        contact_mask,
        bounce_count + jnp.int32(1),
        bounce_count,
    )
    within_count = (bounce_count_limit == jnp.int32(-1)) | (
        next_bounce_count <= bounce_count_limit
    )
    post_bounce_velocity = jnp.where(
        within_count[..., None],
        reflected * bounciness[..., None],
        reflected,
    )
    post_bounce_displacement = (
        jnp.linalg.norm(post_bounce_velocity, axis=-1) * dt_seconds
    )
    bounced = contact_mask & within_count & ~(post_bounce_displacement < bounce_limit)
    hit_ground = jnp.all(
        contact_normal == jnp.asarray((0.0, 1.0, 0.0), dtype=jnp.float32),
        axis=-1,
    )
    rolling = contact_mask & ~bounced & allow_rolling
    impacted = contact_mask & ~bounced & ~rolling & (sticks_vertically | hit_ground)
    stopped = contact_mask & ~bounced & ~rolling & ~impacted

    rolling_velocity = post_bounce_velocity.at[..., 1].set(jnp.float32(0.0))
    rolling_velocity *= rolling_friction_factor[..., None]
    next_velocity = jnp.where(
        bounced[..., None],
        post_bounce_velocity,
        jnp.where(
            rolling[..., None],
            rolling_velocity,
            jnp.where((impacted | stopped)[..., None], jnp.float32(0.0), velocity),
        ),
    )

    remaining = end - contact_point
    remaining_projection = (
        jnp.sum(remaining * contact_normal, axis=-1) / safe_normal_squared
    )
    tangent_remaining = remaining - (remaining_projection[..., None] * contact_normal)
    continued_position = contact_point + tangent_remaining
    next_position = jnp.where(
        (bounced | rolling)[..., None],
        continued_position,
        jnp.where((impacted | stopped)[..., None], contact_point, end),
    )
    on_ground = (rolling | impacted) & hit_ground
    return StandardProjectileContactResponse(
        position=next_position,
        velocity=next_velocity,
        reflected_velocity=reflected,
        bounce_count=next_bounce_count,
        bounced=bounced,
        rolling=rolling,
        stopped=stopped,
        impacted=impacted,
        on_ground=on_ground,
    )


__all__ = [
    "StandardProjectileContactResponse",
    "standard_projectile_contact_response",
]
