"""Certified LOS predicates for the geometry providers used by compiled combat."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.geometry.contract import (
    CELL_COUNT,
    CELL_RADIUS,
    FLAG_OPAQUE,
    FLAG_SOLID,
)
from hytalegym.jax.world.region.mutable_geometry import (
    lookup_region_geometry_blocks,
)
from hytalegym.jax.world.region.types import RegionGeometryState
from hytalegym.jax.world.types import (
    GeometryState,
    _gather_collision_shapes,
)


Array = jax.Array
GeometryProvider = GeometryState | RegionGeometryState
GEOMETRY_PROVIDER_LOS_SCHEMA = "hytalerl_geometry_provider_los_v1"
GEOMETRY_PROVIDER_LOS_VERSION = 2
MAX_LOCAL_GEOMETRY_LOS_CELLS = 6 * CELL_RADIUS + 1
_BOUNDARY_EPSILON = jnp.float32(1.0e-6)


class GeometryPerceptionLineOfSightResult(NamedTuple):
    """NPC opacity LOS with explicit role-mask provenance."""

    visible: Array
    geometry_exhausted: Array
    capacity_exceeded: Array
    invalid: Array
    role_opacity_applied: Array


class GeometryHitboxLineOfSightResult(NamedTuple):
    """Selector detail-box LOS with non-blocking coverage diagnostics."""

    clear: Array
    geometry_exhausted: Array
    capacity_exceeded: Array
    invalid: Array


class _CellSelection(NamedTuple):
    available: Array
    flags: Array
    collision_boxes: Array
    collision_box_mask: Array
    role_opaque: Array


def geometry_perception_line_of_sight_result(
    geometry: GeometryProvider,
    start: Array,
    end: Array,
    *,
    role_opaque_mask: Array | None,
    max_cells: int | None = None,
) -> GeometryPerceptionLineOfSightResult:
    """Trace native NPC opacity LOS through local or Region geometry.

    ``role_opaque_mask`` is deliberately required. Passing ``None`` requests
    only the role-independent ``FLAG_OPAQUE`` predicate and reports
    ``role_opacity_applied=False``. For ``GeometryState`` a supplied mask is
    per captured local cell; for ``RegionGeometryState`` it is per
    ``[region, cell-palette]``. This keeps role opacity exact without treating
    discarded runtime block identity as recoverable semantics.
    """

    result, role_applied = _geometry_line_of_sight_result(
        geometry,
        start,
        end,
        role_opaque_mask=role_opaque_mask,
        hitbox=False,
        ignore_terminal_opacity=False,
        max_cells=max_cells,
    )
    return GeometryPerceptionLineOfSightResult(
        visible=result[0],
        geometry_exhausted=result[1],
        capacity_exceeded=result[2],
        invalid=result[3],
        role_opacity_applied=role_applied,
    )


def geometry_perception_target_cell_visible_result(
    geometry: GeometryProvider,
    start: Array,
    target: Array,
    *,
    role_opaque_mask: Array | None,
    max_cells: int | None = None,
) -> GeometryPerceptionLineOfSightResult:
    """Trace NPC perception LOS to a visible geometry cell.

    Intervening cells use the certified native opacity and role-mask
    predicates. Only the terminal cell is excluded from blocking so a token
    representing that cell does not occlude itself. Coverage, invalid input,
    and walk-capacity handling remain fail closed.
    """

    result, role_applied = _geometry_line_of_sight_result(
        geometry,
        start,
        target,
        role_opaque_mask=role_opaque_mask,
        hitbox=False,
        ignore_terminal_opacity=True,
        max_cells=max_cells,
    )
    return GeometryPerceptionLineOfSightResult(
        visible=result[0],
        geometry_exhausted=result[1],
        capacity_exceeded=result[2],
        invalid=result[3],
        role_opacity_applied=role_applied,
    )


def geometry_hitbox_line_of_sight_result(
    geometry: GeometryProvider,
    start: Array,
    end: Array,
    *,
    max_cells: int | None = None,
) -> GeometryHitboxLineOfSightResult:
    """Trace selector LOS against solid closed detail boxes.

    Unavailable cells remain non-blocking, exactly matching
    ``HorizontalSelector``, while ``geometry_exhausted`` diagnoses the lost
    coverage. Invalid input and fixed walk-capacity overflow fail closed.
    """

    result, _ = _geometry_line_of_sight_result(
        geometry,
        start,
        end,
        role_opaque_mask=None,
        hitbox=True,
        ignore_terminal_opacity=False,
        max_cells=max_cells,
    )
    return GeometryHitboxLineOfSightResult(
        clear=result[0],
        geometry_exhausted=result[1],
        capacity_exceeded=result[2],
        invalid=result[3],
    )


def geometry_provider_los_contract() -> dict[str, object]:
    """Return the checkpoint-pinnable provider reconciliation contract."""

    return {
        "schema": GEOMETRY_PROVIDER_LOS_SCHEMA,
        "version": GEOMETRY_PROVIDER_LOS_VERSION,
        "providers": ["GeometryState", "RegionGeometryState"],
        "native_evidence_schema": "hytalerl_native_los_evidence_v1",
        "perception": {
            "function": "geometry_perception_line_of_sight_result",
            "predicate": "opacity_plus_explicit_role_mask",
            "unloaded_chunk": "blocked",
            "role_mask_required_argument": True,
            "none_semantics": "base_opacity_only_not_native_role_parity",
            "geometry_state_mask": "boolean_per_local_cell",
            "region_geometry_state_mask": "boolean_per_region_cell_palette",
            "mask_derivation": (
                "external_required_because_geometry_providers_discard_"
                "stable_block_identity"
            ),
        },
        "perception_target_cell": {
            "function": "geometry_perception_target_cell_visible_result",
            "predicate": "perception_with_terminal_cell_self_occlusion_excluded",
            "intervening_cells": "identical_to_perception",
            "missing_terminal_coverage": "blocked",
        },
        "hit_confirmation": {
            "function": "geometry_hitbox_line_of_sight_result",
            "predicate": "solid_rotated_detail_box_intersects_closed_line",
            "unloaded_chunk": "non_blocking_with_diagnostic",
        },
        "invalid_or_capacity_overflow": "fail_closed",
        "cache_policy": "fresh_per_query_declared_native_divergence",
    }


def geometry_provider_los_contract_sha256() -> str:
    payload = json.dumps(
        geometry_provider_los_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _geometry_line_of_sight_result(
    geometry: GeometryProvider,
    start: Array,
    end: Array,
    *,
    role_opaque_mask: Array | None,
    hitbox: bool,
    ignore_terminal_opacity: bool,
    max_cells: int | None,
) -> tuple[tuple[Array, Array, Array, Array], Array]:
    points = jnp.asarray(start, dtype=jnp.float32)
    terminal_points = jnp.asarray(end, dtype=jnp.float32)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or terminal_points.shape != points.shape
    ):
        raise ValueError("start and end must have shape [batch, 3]")
    batch = points.shape[0]
    finite = jnp.all(
        jnp.isfinite(points) & jnp.isfinite(terminal_points),
        axis=1,
    )
    safe_start = jnp.where(finite[:, None], points, 0.0)
    safe_end = jnp.where(finite[:, None], terminal_points, 0.0)
    start_block = jnp.floor(safe_start).astype(jnp.int32)
    end_block = jnp.floor(safe_end).astype(jnp.int32)
    start_fraction = safe_start - start_block.astype(jnp.float32)
    end_fraction = safe_end - end_block.astype(jnp.float32)

    maximum = _provider_max_cells(geometry)
    steps = maximum if max_cells is None else max_cells
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or not 1 <= steps <= maximum
    ):
        raise ValueError(f"max_cells must be an integer in [1, {maximum}]")
    role_mask, role_applied = _prepare_role_mask(
        geometry,
        role_opaque_mask,
        batch,
    )

    block_delta = end_block - start_block
    delta_overflow = (
        ((start_block < 0) & (end_block >= 0) & (block_delta < 0))
        | ((start_block >= 0) & (end_block < 0) & (block_delta > 0))
        | (block_delta == jnp.iinfo(jnp.int32).min)
    )
    safe_delta = jnp.where(delta_overflow, 0, block_delta)
    direction = safe_delta.astype(jnp.float32) + end_fraction - start_fraction
    step = jnp.sign(direction).astype(jnp.int32)
    current = start_block - ((step < 0) & (start_fraction <= _BOUNDARY_EPSILON)).astype(
        jnp.int32
    )
    terminal = end_block - ((step > 0) & (end_fraction <= _BOUNDARY_EPSILON)).astype(
        jnp.int32
    )
    terminal_delta = terminal - current
    terminal_overflow = (
        ((current < 0) & (terminal >= 0) & (terminal_delta < 0))
        | ((current >= 0) & (terminal < 0) & (terminal_delta > 0))
        | (terminal_delta == jnp.iinfo(jnp.int32).min)
    )
    safe_terminal_delta = jnp.where(terminal_overflow, 0, terminal_delta)
    capacity_exceeded = (
        jnp.sum(jnp.abs(safe_terminal_delta), axis=1) + 1 > steps
    ) | jnp.any(delta_overflow | terminal_overflow, axis=1)
    invalid = ~finite

    next_boundary = jnp.where(step > 0, current + 1, current)
    nonzero = step != 0
    safe_direction = jnp.where(nonzero, direction, 1.0)
    boundary_time = jnp.where(
        nonzero,
        ((next_boundary - start_block).astype(jnp.float32) - start_fraction)
        / safe_direction,
        jnp.inf,
    )
    time_delta = jnp.where(
        nonzero,
        jnp.abs(1.0 / safe_direction),
        jnp.inf,
    )
    initial = (
        current,
        boundary_time,
        jnp.zeros(batch, dtype=jnp.bool_),
        ~capacity_exceeded & ~invalid,
        capacity_exceeded | invalid,
    )
    relative_end = safe_delta.astype(jnp.float32) + end_fraction

    def visit(
        _: int,
        state: tuple[Array, ...],
    ) -> tuple[Array, ...]:
        cell, times, blocked, complete, done = state
        active = ~done
        selected = _lookup_cells(geometry, cell, role_mask)
        if hitbox:
            relative_cell = (cell - start_block).astype(jnp.float32)
            boxes = (
                selected.collision_boxes
                + jnp.concatenate(
                    (relative_cell, relative_cell),
                    axis=1,
                )[:, None, :]
            )
            box_hits = _segment_intersects_closed_boxes(
                start_fraction[:, None, :],
                relative_end[:, None, :],
                boxes,
            )
            solid = (selected.flags & jnp.int32(FLAG_SOLID)) != 0
            blocked_here = (
                active
                & selected.available
                & solid
                & jnp.any(box_hits & selected.collision_box_mask, axis=1)
            )
            next_done = done | (active & (blocked_here | _reached(cell, terminal)))
        else:
            opaque = (
                (selected.flags & jnp.int32(FLAG_OPAQUE)) != 0
            ) | selected.role_opaque
            terminal_here = _reached(cell, terminal)
            blocked_here = (
                active
                & selected.available
                & opaque
                & ~(ignore_terminal_opacity & terminal_here)
            )
            next_done = done | (
                active & (blocked_here | ~selected.available | terminal_here)
            )
        next_complete = complete & (~active | selected.available)
        next_blocked = blocked | blocked_here
        minimum_time = jnp.min(times, axis=1, keepdims=True)
        tied = jnp.abs(times - minimum_time) <= _BOUNDARY_EPSILON
        advance = active & ~next_done
        axes = tied & advance[:, None]
        return (
            cell + jnp.where(axes, step, 0),
            times + jnp.where(axes, time_delta, 0.0),
            next_blocked,
            next_complete,
            next_done,
        )

    _, _, blocked, complete, done = jax.lax.fori_loop(
        0,
        steps,
        visit,
        initial,
    )
    walk_complete = done & ~capacity_exceeded & ~invalid
    clear = walk_complete & ~blocked
    if not hitbox:
        clear &= complete
    exhausted = ~(complete & walk_complete)
    return (
        (clear, exhausted, capacity_exceeded, invalid),
        jnp.full(batch, role_applied, dtype=jnp.bool_),
    )


def _lookup_cells(
    geometry: GeometryProvider,
    cell: Array,
    role_mask: Array | None,
) -> _CellSelection:
    if isinstance(geometry, RegionGeometryState):
        selected = lookup_region_geometry_blocks(
            geometry,
            cell[:, None, :],
            require_core=False,
        )
        available = selected.available[:, 0]
        role_opaque = jnp.zeros_like(available)
        if role_mask is not None:
            region = jnp.maximum(selected.region_index[:, 0], 0)
            code = selected.cell_code[:, 0]
            overridden = code < 0
            safe_code = jnp.maximum(code, 0)
            override_air = (
                overridden
                & (selected.flags[:, 0] == 0)
                & (selected.fluid_level[:, 0] == 0)
                & ~jnp.any(selected.collision_box_mask[:, 0], axis=1)
            )
            override_base_opaque = overridden & (
                (
                    selected.flags[:, 0]
                    & jnp.int32(FLAG_OPAQUE)
                )
                != 0
            )
            # Per-role opacity is not part of mutable geometry. Removed air
            # is provably clear and globally opaque geometry is provably
            # blocked. A transparent present override remains unavailable
            # rather than borrowing the immutable palette's role bit.
            available &= ~overridden | override_air | override_base_opaque
            role_opaque = (
                available
                & ~overridden
                & role_mask[
                    jnp.arange(cell.shape[0]),
                    region,
                    safe_code,
                ]
            )
        filler_translation = jnp.concatenate(
            (
                -selected.filler_root_offset[:, 0],
                -selected.filler_root_offset[:, 0],
            ),
            axis=1,
        )[:, None, :]
        return _CellSelection(
            available=available,
            flags=selected.flags[:, 0],
            collision_boxes=(
                selected.collision_boxes[:, 0]
                + filler_translation.astype(
                    selected.collision_boxes.dtype
                )
            ),
            collision_box_mask=selected.collision_box_mask[:, 0],
            role_opaque=role_opaque,
        )

    world = _broadcast_local_geometry(geometry, cell.shape[0])
    relative = cell - world.origin
    inside = jnp.all(
        (relative >= -CELL_RADIUS) & (relative <= CELL_RADIUS),
        axis=1,
    )
    index = jnp.clip(_local_cell_index(relative), 0, CELL_COUNT - 1)

    def gather(values: Array) -> Array:
        return jax.vmap(lambda row, selected: row[selected])(values, index)

    present = inside & gather(world.cell_mask)
    flags = jnp.where(present, gather(world.flags), jnp.int32(0))
    boxes, box_mask = _gather_collision_shapes(world, index[:, None])
    boxes = boxes[:, 0]
    box_mask = box_mask[:, 0] & present[:, None]
    role_opaque = jnp.zeros_like(inside)
    if role_mask is not None:
        role_opaque = present & gather(role_mask)
    return _CellSelection(
        available=inside,
        flags=flags,
        collision_boxes=boxes,
        collision_box_mask=box_mask,
        role_opaque=role_opaque,
    )


def _prepare_role_mask(
    geometry: GeometryProvider,
    value: Array | None,
    batch: int,
) -> tuple[Array | None, bool]:
    if value is None:
        return None, False
    mask = jnp.asarray(value)
    if mask.dtype != jnp.bool_:
        raise TypeError("role_opaque_mask must have boolean dtype")
    if isinstance(geometry, RegionGeometryState):
        expected = geometry.atlas.cell_flags.shape
        if mask.shape == expected:
            mask = mask[None, ...]
        if mask.shape not in ((1,) + expected, (batch,) + expected):
            raise ValueError(
                "Region role_opaque_mask must have shape "
                "[regions, cell palette] or [batch, regions, cell palette]"
            )
        return jnp.broadcast_to(mask, (batch,) + expected), True
    if mask.shape == (CELL_COUNT,):
        mask = mask[None, ...]
    if mask.shape not in ((1, CELL_COUNT), (batch, CELL_COUNT)):
        raise ValueError(
            "GeometryState role_opaque_mask must have shape "
            "[local cells] or [batch, local cells]"
        )
    return jnp.broadcast_to(mask, (batch, CELL_COUNT)), True


def _provider_max_cells(geometry: GeometryProvider) -> int:
    if isinstance(geometry, RegionGeometryState):
        return geometry.los_step_marker.shape[0]
    if isinstance(geometry, GeometryState):
        return MAX_LOCAL_GEOMETRY_LOS_CELLS
    raise TypeError("geometry must be GeometryState or RegionGeometryState")


def _broadcast_local_geometry(
    geometry: GeometryState,
    batch: int,
) -> GeometryState:
    size = geometry.origin.shape[0]
    if size not in (1, batch):
        raise ValueError("geometry batch must be one or match the LOS batch")
    if size == batch:
        return geometry
    return GeometryState(
        *(
            None
            if leaf is None
            else jnp.broadcast_to(leaf, (batch,) + leaf.shape[1:])
            for leaf in geometry
        )
    )


def _local_cell_index(relative: Array) -> Array:
    side = jnp.int32(CELL_RADIUS * 2 + 1)
    return (
        (relative[:, 0] + CELL_RADIUS) * side * side
        + (relative[:, 1] + CELL_RADIUS) * side
        + relative[:, 2]
        + CELL_RADIUS
    )


def _segment_intersects_closed_boxes(
    start: Array,
    end: Array,
    boxes: Array,
) -> Array:
    delta = end - start
    moving = jnp.abs(delta) >= jnp.float32(1.0e-10)
    safe_delta = jnp.where(moving, delta, 1.0)
    first = (boxes[..., :3] - start) / safe_delta
    second = (boxes[..., 3:] - start) / safe_delta
    entry = jnp.where(moving, jnp.minimum(first, second), 0.0)
    exit_time = jnp.where(moving, jnp.maximum(first, second), 1.0)
    stationary_inside = (start >= boxes[..., :3]) & (start <= boxes[..., 3:])
    return jnp.all(moving | stationary_inside, axis=-1) & (
        jnp.maximum(jnp.max(entry, axis=-1), 0.0)
        <= jnp.minimum(jnp.min(exit_time, axis=-1), 1.0)
    )


def _reached(cell: Array, terminal: Array) -> Array:
    return jnp.all(cell == terminal, axis=1)


__all__ = [
    "GEOMETRY_PROVIDER_LOS_SCHEMA",
    "GEOMETRY_PROVIDER_LOS_VERSION",
    "GeometryHitboxLineOfSightResult",
    "GeometryPerceptionLineOfSightResult",
    "GeometryProvider",
    "MAX_LOCAL_GEOMETRY_LOS_CELLS",
    "geometry_hitbox_line_of_sight_result",
    "geometry_perception_line_of_sight_result",
    "geometry_perception_target_cell_visible_result",
    "geometry_provider_los_contract",
    "geometry_provider_los_contract_sha256",
]
