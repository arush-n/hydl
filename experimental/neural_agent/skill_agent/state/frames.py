"""The one spatial convention of ``skill_tokens_v3`` (``docs/PERCEPTION.md`` section 10, owner 2026-10-20 night: the
body and the head frame on every spatial token).

Every spatial token carries the same ``where`` block, computed here and nowhere else:

    forward, right, up          its position in the agent's body frame, in blocks over POSITION_SCALE_BLOCKS
    log_distance                log1p(distance) / log1p(DISTANCE_SCALE_BLOCKS), the 3-D distance from the agent
    body_bearing  sin, cos      its bearing in the body frame, positive to the right
    head_bearing  sin, cos      its bearing in the head's frame: the body bearing plus the head's yaw to the left
    head_elevation sin, cos     its elevation from the eye, less the head's pitch (positive up)

Frames follow the gym's: for a body yaw ``yaw`` (radians) the body's forward is (-sin yaw, 0, -cos yaw) and its right
is (cos yaw, 0, -sin yaw) in world (x, y, z), so a larger yaw turns the body to the left (``scripted_bodies.body_frame``
uses the same). Rows the gym publishes in world axes (projectiles, hazards, geometry) go through :func:`world_to_body`
first; entity rows are already in the body frame.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.contract.tokens_v3 import WHERE_FEATURES

POSITION_SCALE_BLOCKS: float = 16.0
"""Positions are blocks over this: the sensor range, so a seen thing reads within about [-1, 1]."""
DISTANCE_SCALE_BLOCKS: float = 64.0
"""``log_distance`` is 1 at this many blocks."""


def world_to_body(offset: jax.Array, body_yaw: jax.Array) -> jax.Array:
    """World-axis offsets float32[..., 3] (x, y, z) as (forward, right, up) in the body frame of ``body_yaw`` (radians,
    broadcast against the offsets' leading axes)."""

    sin, cos = jnp.sin(body_yaw), jnp.cos(body_yaw)
    x, y, z = offset[..., 0], offset[..., 1], offset[..., 2]
    return jnp.stack((-x * sin - z * cos, x * cos - z * sin, y), axis=-1)


def body_to_world(position: jax.Array, body_yaw: jax.Array) -> jax.Array:
    """The inverse of :func:`world_to_body`: body-frame (forward, right, up) float32[..., 3] as a world-axis offset
    (x, y, z)."""

    sin, cos = jnp.sin(body_yaw), jnp.cos(body_yaw)
    forward, right, up = position[..., 0], position[..., 1], position[..., 2]
    return jnp.stack((-forward * sin + right * cos, up, -forward * cos - right * sin), axis=-1)


def where(position: jax.Array, head_yaw_left: jax.Array, head_pitch: jax.Array, eye_rise: jax.Array) -> jax.Array:
    """The ``where`` block float32[..., 10] of body-frame positions ``position`` float32[..., 3] (forward, right, up,
    blocks, from the agent's feet).

    ``head_yaw_left``: the head's yaw less the body's, radians (positive: the head is turned left). ``head_pitch``:
    radians, positive up. ``eye_rise``: the height of the token's reference point above the agent's eye, less its
    ``up`` (so ``up + eye_rise`` is the rise from the eye), broadcast like ``up``. All three broadcast against the
    leading axes of ``position``.
    """

    forward, right, up = position[..., 0], position[..., 1], position[..., 2]
    planar = jnp.hypot(forward, right)
    distance = jnp.sqrt(planar * planar + up * up)
    body = jnp.arctan2(right, forward)
    head = body + head_yaw_left
    elevation = jnp.arctan2(up + eye_rise, planar) - head_pitch
    scale = jnp.float32(POSITION_SCALE_BLOCKS)
    return jnp.stack((forward / scale, right / scale, up / scale,
                      jnp.log1p(distance) / jnp.float32(math.log1p(DISTANCE_SCALE_BLOCKS)),
                      jnp.sin(body), jnp.cos(body), jnp.sin(head), jnp.cos(head),
                      jnp.sin(elevation), jnp.cos(elevation)), axis=-1).astype(jnp.float32)


__all__ = ["DISTANCE_SCALE_BLOCKS", "POSITION_SCALE_BLOCKS", "WHERE_FEATURES", "body_to_world", "where",
           "world_to_body"]
