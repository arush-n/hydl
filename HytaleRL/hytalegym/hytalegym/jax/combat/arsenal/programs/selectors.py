"""Fixed-shape geometry kernels for authored interaction selectors."""

from __future__ import annotations

import jax
import jax.numpy as jnp


_CUBE_BITS = jnp.asarray(
    (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ),
    dtype=jnp.float32,
)

_FRUSTUM_FACE_INDICES = jnp.asarray(
    (
        (0, 1, 3),
        (4, 6, 7),
        (0, 2, 6),
        (1, 5, 7),
        (0, 4, 5),
        (2, 3, 7),
    ),
    dtype=jnp.int32,
)
_FRUSTUM_EDGE_INDICES = jnp.asarray(
    (
        (0, 1),
        (2, 0),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ),
    dtype=jnp.int32,
)


def stab_selector_intersects_aabb(
    selector_origin: jax.Array,
    head_yaw_degrees: jax.Array,
    head_pitch_degrees: jax.Array,
    target_position: jax.Array,
    target_bounds: jax.Array,
    previous_progress: jax.Array,
    current_progress: jax.Array,
    start_distance: jax.Array | float,
    end_distance: jax.Array | float,
    extend_left: jax.Array | float,
    extend_right: jax.Array | float,
    extend_bottom: jax.Array | float,
    extend_top: jax.Array | float,
    yaw_offset_degrees: jax.Array | float,
    pitch_offset_degrees: jax.Array | float,
    roll_offset_degrees: jax.Array | float,
) -> jax.Array:
    """Test one authored ``StabSelector`` slice against world-space AABBs.

    The server advances an orthographic selector from ``StartDistance`` to
    ``EndDistance`` over its runtime. ``previous_progress`` and
    ``current_progress`` delimit the exact slice for one selector tick.
    Inputs are mechanism data; this kernel contains no weapon/profile branch.

    This reproduces the cube-surface behavior of ``HitDetectionExecutor``:
    ordinary oriented-box/AABB overlap is selected, except when the complete
    selector volume is enclosed by the target and no target cube surface
    enters the selector.
    """

    selector_origin = jnp.asarray(selector_origin, dtype=jnp.float32)
    target_position = jnp.asarray(target_position, dtype=jnp.float32)
    target_bounds = jnp.asarray(target_bounds, dtype=jnp.float32)
    if selector_origin.ndim != 2 or selector_origin.shape[1] != 3:
        raise ValueError("selector_origin must have shape (B, 3)")
    batch = selector_origin.shape[0]
    if target_position.shape != (batch, 3):
        raise ValueError("target_position must have shape (B, 3)")
    if target_bounds.shape != (batch, 6):
        raise ValueError("target_bounds must have shape (B, 6)")

    yaw = _batch_scalar(head_yaw_degrees, batch, "head_yaw_degrees")
    pitch = _batch_scalar(head_pitch_degrees, batch, "head_pitch_degrees")
    previous = _batch_scalar(previous_progress, batch, "previous_progress")
    current = _batch_scalar(current_progress, batch, "current_progress")
    start = _batch_scalar(start_distance, batch, "start_distance")
    end = _batch_scalar(end_distance, batch, "end_distance")
    left = _batch_scalar(extend_left, batch, "extend_left")
    right = _batch_scalar(extend_right, batch, "extend_right")
    bottom = _batch_scalar(extend_bottom, batch, "extend_bottom")
    top = _batch_scalar(extend_top, batch, "extend_top")
    yaw_offset = _batch_scalar(
        yaw_offset_degrees,
        batch,
        "yaw_offset_degrees",
    )
    pitch_offset = _batch_scalar(
        pitch_offset_degrees,
        batch,
        "pitch_offset_degrees",
    )
    roll_offset = _batch_scalar(
        roll_offset_degrees,
        batch,
        "roll_offset_degrees",
    )

    distance = end - start
    near = start + previous * distance
    far = start + current * distance
    selector_corners = _orthogonal_corners(
        near,
        far,
        left,
        right,
        bottom,
        top,
    )
    world_to_selector = _world_to_selector_rotation(
        yaw,
        pitch,
        yaw_offset,
        pitch_offset,
        roll_offset,
    )

    target_min = target_position + target_bounds[:, :3]
    target_max = target_position + target_bounds[:, 3:]
    target_corners_world = (
        target_min[:, None, :]
        + (target_max - target_min)[:, None, :] * _CUBE_BITS[None, :, :]
    )
    target_corners = jnp.einsum(
        "bij,bvj->bvi",
        world_to_selector,
        target_corners_world - selector_origin[:, None, :],
    )

    selector_axes = jnp.broadcast_to(
        jnp.eye(3, dtype=jnp.float32)[None, :, :],
        (batch, 3, 3),
    )
    target_axes = jnp.swapaxes(world_to_selector, 1, 2)
    edge_cross_axes = jnp.cross(
        selector_axes[:, :, None, :],
        target_axes[:, None, :, :],
    ).reshape((batch, 9, 3))
    axes = jnp.concatenate(
        (selector_axes, target_axes, edge_cross_axes),
        axis=1,
    )
    axis_norm = jnp.linalg.norm(axes, axis=2)
    valid_axis = axis_norm > jnp.float32(1.0e-8)
    normalized_axes = axes / jnp.maximum(
        axis_norm[:, :, None],
        jnp.float32(1.0e-8),
    )
    target_projection = jnp.einsum(
        "bvi,bai->bav",
        target_corners,
        normalized_axes,
    )
    selector_projection = jnp.einsum(
        "bvi,bai->bav",
        selector_corners,
        normalized_axes,
    )
    separated = (
        (jnp.max(target_projection, axis=2) < jnp.min(selector_projection, axis=2))
        | (jnp.max(selector_projection, axis=2) < jnp.min(target_projection, axis=2))
    ) & valid_axis
    volumes_intersect = ~jnp.any(separated, axis=1)

    selector_corners_world = selector_origin[:, None, :] + jnp.einsum(
        "bvi,bij->bvj",
        selector_corners,
        world_to_selector,
    )
    selector_fully_inside_target = jnp.all(
        (selector_corners_world >= target_min[:, None, :])
        & (selector_corners_world <= target_max[:, None, :]),
        axis=(1, 2),
    )
    parameters = jnp.stack(
        (
            yaw,
            pitch,
            previous,
            current,
            start,
            end,
            left,
            right,
            bottom,
            top,
            yaw_offset,
            pitch_offset,
            roll_offset,
        ),
        axis=1,
    )
    finite = (
        jnp.all(jnp.isfinite(selector_origin), axis=1)
        & jnp.all(jnp.isfinite(target_position), axis=1)
        & jnp.all(jnp.isfinite(target_bounds), axis=1)
        & jnp.all(jnp.isfinite(parameters), axis=1)
    )
    valid = (
        finite
        & jnp.all(target_max >= target_min, axis=1)
        & (start >= 0.0)
        & (end > start)
        & (left >= 0.0)
        & (right >= 0.0)
        & (bottom >= 0.0)
        & (top >= 0.0)
        & (previous >= 0.0)
        & (current <= 1.0)
        & (current > previous)
    )
    return valid & volumes_intersect & ~selector_fully_inside_target


def horizontal_selector_intersects_aabb(
    selector_origin: jax.Array,
    head_yaw_degrees: jax.Array,
    head_pitch_degrees: jax.Array,
    target_position: jax.Array,
    target_bounds: jax.Array,
    previous_progress: jax.Array,
    current_progress: jax.Array,
    start_distance: jax.Array | float,
    end_distance: jax.Array | float,
    extend_bottom: jax.Array | float,
    extend_top: jax.Array | float,
    signed_yaw_length_degrees: jax.Array | float,
    yaw_start_offset_degrees: jax.Array | float,
    pitch_offset_degrees: jax.Array | float,
    roll_offset_degrees: jax.Array | float,
) -> jax.Array:
    """Test one authored ``HorizontalSelector`` arc slice against AABBs.

    The sign of ``signed_yaw_length_degrees`` carries the authored direction:
    positive is ``ToLeft`` and negative is ``ToRight``. The magnitude remains
    the asset's arc length. This mirrors the server's per-tick perspective
    frustum without introducing a weapon-specific branch.
    """

    selector_origin = jnp.asarray(selector_origin, dtype=jnp.float32)
    target_position = jnp.asarray(target_position, dtype=jnp.float32)
    target_bounds = jnp.asarray(target_bounds, dtype=jnp.float32)
    if selector_origin.ndim != 2 or selector_origin.shape[1] != 3:
        raise ValueError("selector_origin must have shape (B, 3)")
    batch = selector_origin.shape[0]
    if target_position.shape != (batch, 3):
        raise ValueError("target_position must have shape (B, 3)")
    if target_bounds.shape != (batch, 6):
        raise ValueError("target_bounds must have shape (B, 6)")

    yaw = _batch_scalar(head_yaw_degrees, batch, "head_yaw_degrees")
    pitch = _batch_scalar(head_pitch_degrees, batch, "head_pitch_degrees")
    previous = _batch_scalar(previous_progress, batch, "previous_progress")
    current = _batch_scalar(current_progress, batch, "current_progress")
    start = _batch_scalar(start_distance, batch, "start_distance")
    end = _batch_scalar(end_distance, batch, "end_distance")
    bottom = _batch_scalar(extend_bottom, batch, "extend_bottom")
    top = _batch_scalar(extend_top, batch, "extend_top")
    signed_yaw_length = _batch_scalar(
        signed_yaw_length_degrees,
        batch,
        "signed_yaw_length_degrees",
    )
    yaw_start = _batch_scalar(
        yaw_start_offset_degrees,
        batch,
        "yaw_start_offset_degrees",
    )
    pitch_offset = _batch_scalar(
        pitch_offset_degrees,
        batch,
        "pitch_offset_degrees",
    )
    roll_offset = _batch_scalar(
        roll_offset_degrees,
        batch,
        "roll_offset_degrees",
    )

    direction = jnp.where(
        signed_yaw_length >= jnp.float32(0.0),
        jnp.float32(1.0),
        jnp.float32(-1.0),
    )
    yaw_length = jnp.abs(signed_yaw_length)
    delta_progress = current - previous
    yaw_delta_radians = jnp.deg2rad(yaw_length * delta_progress)
    # HorizontalSelector.java computes this denominator through
    # ``(float) Math.PI`` even though the surrounding values are doubles.
    far_half_width = jnp.float32(2.0) * end * yaw_delta_radians / jnp.float32(3.1415927)
    stretch = start / end
    near_half_width = far_half_width * stretch
    selector_corners = _frustum_corners(
        start,
        end,
        near_half_width,
        far_half_width,
        bottom * stretch,
        top * stretch,
        bottom,
        top,
    )
    yaw_offset = (yaw_length * current + yaw_start) * direction
    world_to_selector = _world_to_selector_rotation(
        yaw,
        pitch,
        yaw_offset,
        pitch_offset,
        roll_offset,
    )

    target_min = target_position + target_bounds[:, :3]
    target_max = target_position + target_bounds[:, 3:]
    target_corners_world = (
        target_min[:, None, :]
        + (target_max - target_min)[:, None, :] * _CUBE_BITS[None, :, :]
    )
    target_corners = jnp.einsum(
        "bij,bvj->bvi",
        world_to_selector,
        target_corners_world - selector_origin[:, None, :],
    )

    face_corners = selector_corners[:, _FRUSTUM_FACE_INDICES, :]
    frustum_face_axes = jnp.cross(
        face_corners[:, :, 1, :] - face_corners[:, :, 0, :],
        face_corners[:, :, 2, :] - face_corners[:, :, 0, :],
    )
    target_axes = jnp.swapaxes(world_to_selector, 1, 2)
    frustum_edges = (
        selector_corners[:, _FRUSTUM_EDGE_INDICES[:, 1], :]
        - selector_corners[:, _FRUSTUM_EDGE_INDICES[:, 0], :]
    )
    edge_cross_axes = jnp.cross(
        frustum_edges[:, :, None, :],
        target_axes[:, None, :, :],
    ).reshape((batch, 18, 3))
    axes = jnp.concatenate(
        (frustum_face_axes, target_axes, edge_cross_axes),
        axis=1,
    )
    axis_norm = jnp.linalg.norm(axes, axis=2)
    valid_axis = axis_norm > jnp.float32(1.0e-8)
    normalized_axes = axes / jnp.maximum(
        axis_norm[:, :, None],
        jnp.float32(1.0e-8),
    )
    target_projection = jnp.einsum(
        "bvi,bai->bav",
        target_corners,
        normalized_axes,
    )
    selector_projection = jnp.einsum(
        "bvi,bai->bav",
        selector_corners,
        normalized_axes,
    )
    separated = (
        (jnp.max(target_projection, axis=2) < jnp.min(selector_projection, axis=2))
        | (jnp.max(selector_projection, axis=2) < jnp.min(target_projection, axis=2))
    ) & valid_axis
    volumes_intersect = ~jnp.any(separated, axis=1)

    selector_corners_world = selector_origin[:, None, :] + jnp.einsum(
        "bvi,bij->bvj",
        selector_corners,
        world_to_selector,
    )
    selector_fully_inside_target = jnp.all(
        (selector_corners_world >= target_min[:, None, :])
        & (selector_corners_world <= target_max[:, None, :]),
        axis=(1, 2),
    )
    parameters = jnp.stack(
        (
            yaw,
            pitch,
            previous,
            current,
            start,
            end,
            bottom,
            top,
            signed_yaw_length,
            yaw_start,
            pitch_offset,
            roll_offset,
        ),
        axis=1,
    )
    finite = (
        jnp.all(jnp.isfinite(selector_origin), axis=1)
        & jnp.all(jnp.isfinite(target_position), axis=1)
        & jnp.all(jnp.isfinite(target_bounds), axis=1)
        & jnp.all(jnp.isfinite(parameters), axis=1)
    )
    valid = (
        finite
        & jnp.all(target_max >= target_min, axis=1)
        & (start > 0.0)
        & (end > start)
        & (bottom >= 0.0)
        & (top >= 0.0)
        & (yaw_length > 0.0)
        & (previous >= 0.0)
        & (current <= 1.0)
        & (current > previous)
    )
    return valid & volumes_intersect & ~selector_fully_inside_target


def _batch_scalar(
    value: jax.Array | float,
    batch: int,
    name: str,
) -> jax.Array:
    array = jnp.asarray(value, dtype=jnp.float32)
    if array.ndim == 0:
        return jnp.broadcast_to(array, (batch,))
    if array.shape != (batch,):
        raise ValueError(f"{name} must be scalar or have shape (B,)")
    return array


def _orthogonal_corners(
    near: jax.Array,
    far: jax.Array,
    left: jax.Array,
    right: jax.Array,
    bottom: jax.Array,
    top: jax.Array,
) -> jax.Array:
    return jnp.stack(
        (
            jnp.stack((-left, -bottom, -near), axis=1),
            jnp.stack((right, -bottom, -near), axis=1),
            jnp.stack((-left, top, -near), axis=1),
            jnp.stack((right, top, -near), axis=1),
            jnp.stack((-left, -bottom, -far), axis=1),
            jnp.stack((right, -bottom, -far), axis=1),
            jnp.stack((-left, top, -far), axis=1),
            jnp.stack((right, top, -far), axis=1),
        ),
        axis=1,
    )


def _frustum_corners(
    near: jax.Array,
    far: jax.Array,
    near_half_width: jax.Array,
    far_half_width: jax.Array,
    near_bottom: jax.Array,
    near_top: jax.Array,
    far_bottom: jax.Array,
    far_top: jax.Array,
) -> jax.Array:
    return jnp.stack(
        (
            jnp.stack((-near_half_width, -near_bottom, -near), axis=1),
            jnp.stack((near_half_width, -near_bottom, -near), axis=1),
            jnp.stack((-near_half_width, near_top, -near), axis=1),
            jnp.stack((near_half_width, near_top, -near), axis=1),
            jnp.stack((-far_half_width, -far_bottom, -far), axis=1),
            jnp.stack((far_half_width, -far_bottom, -far), axis=1),
            jnp.stack((-far_half_width, far_top, -far), axis=1),
            jnp.stack((far_half_width, far_top, -far), axis=1),
        ),
        axis=1,
    )


def _world_to_selector_rotation(
    yaw_degrees: jax.Array,
    pitch_degrees: jax.Array,
    yaw_offset_degrees: jax.Array,
    pitch_offset_degrees: jax.Array,
    roll_offset_degrees: jax.Array,
) -> jax.Array:
    selector_rotation = jnp.matmul(
        _rotation_x(-jnp.deg2rad(pitch_offset_degrees)),
        jnp.matmul(
            _rotation_y(-jnp.deg2rad(yaw_offset_degrees)),
            _rotation_z(-jnp.deg2rad(roll_offset_degrees)),
        ),
    )
    yaw = jnp.deg2rad(yaw_degrees)
    pitch = jnp.deg2rad(pitch_degrees)
    cos_pitch = jnp.cos(pitch)
    forward = jnp.stack(
        (
            -jnp.sin(yaw) * cos_pitch,
            jnp.sin(pitch),
            -jnp.cos(yaw) * cos_pitch,
        ),
        axis=1,
    )
    world_up = jnp.broadcast_to(
        jnp.asarray((0.0, 1.0, 0.0), dtype=jnp.float32),
        forward.shape,
    )
    right = jnp.cross(forward, world_up)
    right = right / jnp.maximum(
        jnp.linalg.norm(right, axis=1, keepdims=True),
        jnp.float32(1.0e-9),
    )
    camera_up = jnp.cross(right, forward)
    camera_rotation = jnp.stack(
        (right, camera_up, -forward),
        axis=1,
    )
    return jnp.matmul(selector_rotation, camera_rotation)


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


__all__ = [
    "horizontal_selector_intersects_aabb",
    "stab_selector_intersects_aabb",
]
