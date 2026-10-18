"""0.7.2 line-of-sight rule. Switch ``Constructs.line_of_sight``; phase 3 (2026-10-18).

Gen 5 (``non_transparent_blocks``): an opaque cell stops sight across its whole unit cell
(``jax/world/geometry/__init__.py::line_of_sight_result``). Gen 7 (``detail_hitbox``): an opaque
cell occludes only where the ray hits one of its detail boxes, the collision boxes the capture
records for that cell (``GeometryState.collision_boxes``: up to nine cell-local boxes, unit cube
for a full block, partial boxes such as a 0.5 m slab for a half block). FIDELITY-0.7.2.md row 2b.

Rule, per opaque cell k of the 729-cell window:
* the cell has boxes: occluding iff the ray's segment meets a valid box of k;
* the cell has no boxes (the capture records boxes for Solid cells only): occluding as in gen 5,
  the conservative choice; an asset census decides whether such cells occur in 0.7.2 (open).
Full cubes reduce to the gen-5 slab test bit for bit (box = [0,0,0,1,1,1] added to the cell
origin), so a scene of full blocks keeps gen-5 answers.

The construct is the render's hook: ``construct(original, profile)`` returns ``original`` unless the
profile selects ``detail_hitbox``. The detail function reads the rendered module's own constants
(``original.__globals__``), so it uses the same window, offsets and epsilon as gen 5.
"""

from __future__ import annotations

from typing import Any

SWITCH = "line_of_sight"
IMPLEMENTED = True
DETAIL = "detail_hitbox"
GEN5 = "non_transparent_blocks"

_NEEDED = ("jnp", "VisibilityResult", "_broadcast_geometry", "_CELL_OFFSETS", "CELL_RADIUS",
           "FLAG_OPAQUE", "_FLOAT_EPSILON")


def construct(original: Any, profile: Any) -> Any:
    """Return ``original`` for gen 5; the detail-hitbox line of sight when the switch selects it."""

    value = profile.constructs.line_of_sight
    if value == GEN5:
        return original
    if value != DETAIL:
        raise ValueError(f"unknown line_of_sight switch {value!r}")
    env = original.__globals__
    missing = [name for name in _NEEDED if name not in env]
    if missing:
        raise NotImplementedError(f"rendered module lacks {missing}; cannot build the detail rule")
    return _detail_line_of_sight_result(env)


def _detail_line_of_sight_result(env: dict) -> Any:
    jnp = env["jnp"]
    visibility_result = env["VisibilityResult"]
    broadcast = env["_broadcast_geometry"]
    cell_offsets = env["_CELL_OFFSETS"]
    radius = env["CELL_RADIUS"]
    flag_opaque = env["FLAG_OPAQUE"]
    epsilon = env["_FLOAT_EPSILON"]

    def line_of_sight_result(geometry: Any, start: Any, end: Any) -> Any:
        start = jnp.asarray(start, dtype=jnp.float32)
        end = jnp.asarray(end, dtype=jnp.float32)
        if start.ndim != 2 or start.shape[1] != 3 or end.shape != start.shape:
            raise ValueError("start and end must have shape [batch, 3]")

        world = broadcast(geometry, start.shape[0])
        cell_minimum = world.origin[:, None, :].astype(jnp.float32) + cell_offsets[None, :, :]
        opaque = world.cell_mask & ((world.flags & jnp.int32(flag_opaque)) != 0)

        # [B, 729, slots, 6] cell-local boxes and [B, 729, slots] validity (full cube in slot 0).
        boxes = world.collision_boxes.astype(jnp.float32)
        box_valid = world.collision_box_mask
        box_minimum = cell_minimum[:, :, None, :] + boxes[..., :3]
        box_maximum = cell_minimum[:, :, None, :] + boxes[..., 3:]

        direction = end - start
        expanded_start = start[:, None, None, :]
        expanded_direction = direction[:, None, None, :]
        nonzero = jnp.abs(expanded_direction) > epsilon
        inverse = jnp.where(nonzero, 1.0 / expanded_direction, 0.0)
        first = (box_minimum - expanded_start) * inverse
        second = (box_maximum - expanded_start) * inverse
        axis_entry = jnp.minimum(first, second)
        axis_exit = jnp.maximum(first, second)
        inside_slab = (expanded_start >= box_minimum) & (expanded_start < box_maximum)
        axis_entry = jnp.where(nonzero, axis_entry, jnp.where(inside_slab, -jnp.inf, jnp.inf))
        axis_exit = jnp.where(nonzero, axis_exit, jnp.where(inside_slab, jnp.inf, -jnp.inf))
        entry = jnp.max(axis_entry, axis=3)
        exit_ = jnp.min(axis_exit, axis=3)
        clipped_entry = jnp.maximum(entry, 0.0)
        clipped_exit = jnp.minimum(exit_, 1.0)
        box_hit = box_valid & (clipped_entry < clipped_exit - epsilon)
        any_box_hit = jnp.any(box_hit, axis=2)
        has_boxes = jnp.any(box_valid, axis=2)
        intersects = opaque & jnp.where(has_boxes, any_box_hit, True)

        segment_minimum = jnp.floor(jnp.minimum(start, end)).astype(jnp.int32) - world.origin
        segment_maximum = jnp.floor(jnp.maximum(start, end) - epsilon).astype(jnp.int32) - world.origin
        complete = jnp.all((segment_minimum >= -radius) & (segment_maximum <= radius), axis=1)
        return visibility_result(
            visible=complete & ~jnp.any(intersects, axis=1),
            geometry_exhausted=~complete,
        )

    return line_of_sight_result
