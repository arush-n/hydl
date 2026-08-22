"""Leaf helpers extracted verbatim from env.py."""

import jax
import jax.numpy as jnp
from hytalegym.jax.combat.types import AGENT_ENTITY, TARGET_ENTITY, CombatParams, CombatState


def _rotation_x(angle: jax.Array) -> jax.Array:
    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)
    zeros = jnp.zeros_like(angle)
    ones = jnp.ones_like(angle)
    return jnp.stack(
        (
            jnp.stack((ones, zeros, zeros), axis=1),
            jnp.stack((zeros, cosine, -sine), axis=1),
            jnp.stack((zeros, sine, cosine), axis=1),
        ),
        axis=1,
    )


def _rotation_y(angle: jax.Array) -> jax.Array:
    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)
    zeros = jnp.zeros_like(angle)
    ones = jnp.ones_like(angle)
    return jnp.stack(
        (
            jnp.stack((cosine, zeros, sine), axis=1),
            jnp.stack((zeros, ones, zeros), axis=1),
            jnp.stack((-sine, zeros, cosine), axis=1),
        ),
        axis=1,
    )


def _rotation_z(angle: jax.Array) -> jax.Array:
    cosine = jnp.cos(angle)
    sine = jnp.sin(angle)
    zeros = jnp.zeros_like(angle)
    ones = jnp.ones_like(angle)
    return jnp.stack(
        (
            jnp.stack((cosine, -sine, zeros), axis=1),
            jnp.stack((sine, cosine, zeros), axis=1),
            jnp.stack((zeros, zeros, ones), axis=1),
        ),
        axis=1,
    )


def _target_facing_error(state: CombatState) -> jax.Array:
    offset = state.position[:, AGENT_ENTITY] - state.position[:, TARGET_ENTITY]
    distance = jnp.hypot(offset[:, 0], offset[:, 2])
    bearing = _canonical_bearing(offset[:, 0], offset[:, 2])
    return jnp.where(
        distance <= 1.0e-9,
        0.0,
        _normalize_degrees(bearing - state.yaw[:, TARGET_ENTITY]),
    ).astype(jnp.float32)


def _target_head_facing_error(state: CombatState) -> jax.Array:
    offset = state.position[:, AGENT_ENTITY] - state.position[:, TARGET_ENTITY]
    distance = jnp.hypot(offset[:, 0], offset[:, 2])
    bearing = _canonical_bearing(offset[:, 0], offset[:, 2])
    return jnp.where(
        distance <= 1.0e-9,
        0.0,
        _normalize_degrees(bearing - state.target_head_yaw),
    ).astype(jnp.float32)


def _target_distance(state: CombatState) -> jax.Array:
    offset = (
        state.position[:, AGENT_ENTITY, (0, 2)]
        - state.position[:, TARGET_ENTITY, (0, 2)]
    )
    return jnp.linalg.norm(offset, axis=1)


def _motion_delta(
    tick_count: jax.Array,
    profile: jax.Array,
    params: CombatParams,
) -> jax.Array:
    loaded_tick = jnp.where(
        profile == 1,
        tick_count % 2 == 1,
        tick_count % 2 == 0,
    )
    return jnp.where(
        (profile != 0) & loaded_tick,
        params.loaded_dt,
        params.nominal_dt,
    ).astype(jnp.float32)


def _terminated(state: CombatState) -> jax.Array:
    return (
        (state.health[:, TARGET_ENTITY] <= 0.0)
        | (state.health[:, AGENT_ENTITY] <= 0.0)
        | state.geometry_exhausted
    )


def _select_state(
    mask: jax.Array,
    selected: CombatState,
    fallback: CombatState,
) -> CombatState:
    return jax.tree_util.tree_map(
        lambda yes, no: jnp.where(
            mask.reshape((mask.shape[0],) + (1,) * (yes.ndim - 1)),
            yes,
            no,
        ),
        selected,
        fallback,
    )


def _randint_batch(
    keys: jax.Array,
    minimum: int | jax.Array,
    maximum: int | jax.Array,
) -> jax.Array:
    return jax.vmap(
        lambda key: jax.random.randint(
            key,
            (),
            minimum,
            maximum,
            dtype=jnp.int32,
        )
    )(keys)


def _uniform_batch(
    keys: jax.Array,
    minimum: float | jax.Array,
    maximum: float | jax.Array,
) -> jax.Array:
    return jax.vmap(
        lambda key: jax.random.uniform(
            key,
            (),
            minval=minimum,
            maxval=maximum,
            dtype=jnp.float32,
        )
    )(keys)


def _approach(
    value: jax.Array,
    target: jax.Array,
    amount: jax.Array,
) -> jax.Array:
    return jnp.where(
        value < target,
        jnp.minimum(value + amount, target),
        jnp.maximum(value - amount, target),
    )


def _approach_angle(
    value: jax.Array,
    target: jax.Array,
    amount: jax.Array,
) -> jax.Array:
    difference = _normalize_degrees(target - value)
    return jnp.where(
        jnp.abs(difference) <= amount,
        target,
        _normalize_degrees(value + jnp.sign(difference) * amount),
    )


def _approach_target_angle(
    value: jax.Array,
    target: jax.Array,
    amount: jax.Array,
) -> jax.Array:
    difference = _normalize_degrees(target - value)
    difference = jnp.where(
        jnp.abs(difference + 180.0) < 1.0e-5,
        180.0,
        difference,
    )
    return jnp.where(
        jnp.abs(difference) <= amount,
        target,
        value + jnp.sign(difference) * amount,
    )


def _canonical_bearing(dx: jax.Array, dz: jax.Array) -> jax.Array:
    bearing = jnp.rad2deg(jnp.arctan2(-dx, -dz))
    return jnp.where(
        jnp.abs(jnp.abs(bearing) - 180.0) < 1.0e-5,
        180.0,
        bearing,
    )


def _normalize_degrees(value: jax.Array) -> jax.Array:
    return jnp.mod(value + 180.0, 360.0) - 180.0


def _round_to_scale(value: jax.Array, scale: jax.Array) -> jax.Array:
    return jnp.floor(value * scale + 0.5) / scale


def _unit_fraction(value: jax.Array, maximum: jax.Array) -> jax.Array:
    """Normalize a bounded quantity with exact zero/full boundary values.

    XLA may lower float32 division to reciprocal multiplication inside a fused
    program.  At ``value == maximum`` that can produce the adjacent value below
    one even though eager projection returns exactly one.  Learner rows are a
    bit-exact transport contract, so publish the semantic boundary explicitly
    and clamp damaged/overfilled values into the same unit interval.
    """

    safe_maximum = jnp.maximum(
        jnp.abs(maximum),
        jnp.float32(1.0e-6),
    )
    fraction = jnp.clip(value / safe_maximum, 0.0, 1.0)
    return jnp.where(
        value >= safe_maximum,
        jnp.float32(1.0),
        fraction,
    )
