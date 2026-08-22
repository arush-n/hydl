"""Fixed-array state for compiled Hytale geometry mechanics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.geometry.contract import (
    CELL_COUNT,
    FLAG_FLUID,
    FLUID_MOVEMENT_FEATURES,
    MAX_DETAIL_BOXES,
    MOVEMENT_FEATURES,
)


Array = jax.Array
COLLISION_SHAPE_FULL_CUBE = -2
COLLISION_SHAPE_NONE = -1
_UNIT_CUBE_NUMPY = np.asarray((0, 0, 0, 1, 1, 1), dtype=np.float32)
_UNIT_CUBE = jnp.asarray(_UNIT_CUBE_NUMPY)


class GeometryState(NamedTuple):
    """Batched geometry leaves used by compiled physics and sensing.

    Runtime IDs and transient contacts remain in the Python observation.
    Immutable collision, LOS, support, damage, and authored movement settings
    are retained here so compiled transitions never need to recover semantics
    from version-scoped asset IDs. Batch is always the first axis.
    """

    origin: Array
    cell_mask: Array
    flags: Array
    fluid_level: Array
    support: Array
    block_damage: Array
    fluid_damage: Array
    movement: Array
    fluid_movement: Array
    collision_shape_index: Array
    collision_full_cube_cell: Array
    collision_exception_cell: Array
    collision_exception_box_index: Array
    collision_exception_boxes: Array
    collision_exception_box_cell: Array
    collision_world_order: Array
    agent_bounds: Array
    target_bounds: Array
    agent_los_offset: Array
    target_los_offset: Array
    # Added in geometry v5. ``None`` keeps legacy hand-built fixtures
    # constructible; exact body sensing treats a missing leaf as unavailable.
    fluid_fill_height: Array | None = None

    @property
    def collision_boxes(self) -> Array:
        """Materialize the legacy dense view only for compatibility/debugging."""

        boxes, _ = _dense_collision_shapes(self)
        return boxes

    @property
    def collision_box_mask(self) -> Array:
        """Materialize the legacy dense mask only for compatibility/debugging."""

        _, mask = _dense_collision_shapes(self)
        return mask


def _gather_collision_shapes(
    geometry: GeometryState,
    indices: Array,
) -> tuple[Array, Array]:
    """Materialize collision slots only for selected local cells."""

    selected_shape = jax.vmap(lambda row, selected: row[selected])(
        geometry.collision_shape_index,
        indices,
    )
    safe_exception = jnp.clip(
        selected_shape,
        0,
        geometry.collision_exception_box_index.shape[1] - 1,
    )
    box_index = jax.vmap(lambda row, selected: row[selected])(
        geometry.collision_exception_box_index,
        safe_exception,
    )
    safe_box_index = jnp.clip(
        box_index,
        0,
        geometry.collision_exception_boxes.shape[1] - 1,
    )
    boxes = jax.vmap(lambda row, selected: row[selected])(
        geometry.collision_exception_boxes,
        safe_box_index,
    )
    is_exception = selected_shape >= jnp.int16(0)
    boxes = jnp.where(is_exception[..., None, None], boxes, 0.0)
    mask = (box_index >= 0) & is_exception[..., None]

    is_full_cube = selected_shape == jnp.int16(COLLISION_SHAPE_FULL_CUBE)
    first_slot = jnp.arange(boxes.shape[-2]) == 0
    full_cube_slot = is_full_cube[..., None] & first_slot
    boxes = jnp.where(
        full_cube_slot[..., None],
        _UNIT_CUBE,
        boxes,
    )
    return boxes, mask | full_cube_slot


def _dense_collision_shapes(geometry: GeometryState) -> tuple[Array, Array]:
    indices = jnp.broadcast_to(
        jnp.arange(CELL_COUNT, dtype=jnp.int32),
        (geometry.origin.shape[0], CELL_COUNT),
    )
    return _gather_collision_shapes(geometry, indices)


class GeometryAtlas(NamedTuple):
    """Fixed-capacity native tile bank shared by a compiled environment batch."""

    geometry: GeometryState
    tile_mask: Array
    world_id: Array


def _fixed_capacity(
    requested: int | None,
    required: int,
    name: str,
    maximum: int,
) -> int:
    if requested is None:
        return max(required, 1)
    if isinstance(requested, (bool, np.bool_)) or not isinstance(
        requested,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer")
    capacity = int(requested)
    if capacity <= 0 or capacity > maximum:
        raise ValueError(f"{name} must be in [1, {maximum}]")
    if capacity < required:
        raise ValueError(
            f"{name}={capacity} cannot hold {required} required entries"
        )
    return capacity


def geometry_state_from_numpy(
    frames: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    batch_size: int | None = None,
    require_exact: bool = True,
    collision_full_cube_capacity: int | None = None,
    collision_exception_capacity: int | None = None,
    collision_exception_box_capacity: int | None = None,
) -> GeometryState:
    """Convert validated NumPy geometry frames into a batched JAX PyTree.

    Passing one frame with ``batch_size`` broadcasts immutable fixture geometry
    without changing the wire contract. The wire always retains all nine native
    detail-box slots. The JAX state represents a canonical slot-zero unit cube
    with no float payload and stores every other active shape in a sparse,
    fixed-capacity exception table. Exception slots retain native ordering,
    holes, and the nine-slot ceiling. Optional capacities fail closed when too
    small; otherwise they are derived from the complete loaded run.
    Exact native shapes are required by default; callers must explicitly set
    ``require_exact=False`` to run an approximate simulator frame. Conversion
    deliberately happens outside compiled rollouts; no Python dictionaries
    enter a JIT function.
    """

    if isinstance(frames, Mapping):
        source = [frames]
    else:
        source = list(frames)
        if not source:
            raise ValueError("at least one geometry frame is required")
    if batch_size is not None:
        if len(source) != 1:
            raise ValueError("batch_size broadcasting requires exactly one frame")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        source = source * batch_size

    detail_box_capacity = 1
    encoded_shapes: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    fluid_fill_rows: list[np.ndarray] = []
    required_full_cube_capacity = 0
    required_exception_capacity = 0
    required_exception_box_capacity = 0
    for index, frame in enumerate(source):
        available = bool(int(frame.get("available", 0)))
        exact = bool(int(frame.get("exact_collision_shapes", 0)))
        if not available:
            raise ValueError(f"geometry frame {index} is explicitly unavailable")
        if require_exact and not exact:
            raise ValueError(
                f"geometry frame {index} does not contain exact collision shapes"
            )

        origin = np.asarray(frame["origin"])
        cell_mask = np.asarray(frame["cell_mask"])
        flags = np.asarray(frame["flags"])
        fluid_level = np.asarray(frame["fluid_level"])
        if "fluid_fill_height" in frame:
            fluid_fill_height = np.asarray(
                frame["fluid_fill_height"],
                dtype=np.float32,
            )
        else:
            fluid_fill_height = np.zeros(CELL_COUNT, dtype=np.float32)
            fluid_fill_height[
                (flags.astype(np.int32) & FLAG_FLUID) != 0
            ] = -1.0
        support = np.asarray(frame["support"])
        block_damage = np.asarray(frame["block_damage"])
        fluid_damage = np.asarray(frame["fluid_damage"])
        movement = np.asarray(frame["movement"], dtype=np.float32)
        fluid_movement = np.asarray(
            frame["fluid_movement"], dtype=np.float32
        )
        boxes = np.asarray(frame["collision_boxes"], dtype=np.float32)
        box_mask = np.asarray(frame["collision_box_mask"])
        agent_bounds = np.asarray(frame["agent_bounds"], dtype=np.float32)
        target_bounds = np.asarray(frame["target_bounds"], dtype=np.float32)
        agent_los_offset = np.asarray(
            frame["agent_los_offset"], dtype=np.float32
        )
        target_los_offset = np.asarray(
            frame["target_los_offset"], dtype=np.float32
        )
        if origin.shape != (3,):
            raise ValueError(f"geometry frame {index} origin must have shape (3,)")
        integer_cells = (
            cell_mask,
            flags,
            fluid_level,
            support,
            block_damage,
            fluid_damage,
        )
        if any(values.shape != (CELL_COUNT,) for values in integer_cells):
            raise ValueError(f"geometry frame {index} has an invalid cell shape")
        if fluid_fill_height.shape != (CELL_COUNT,):
            raise ValueError(
                f"geometry frame {index} has an invalid fluid-fill shape"
            )
        if movement.shape != (CELL_COUNT, MOVEMENT_FEATURES):
            raise ValueError(
                f"geometry frame {index} has an invalid movement shape"
            )
        if fluid_movement.shape != (CELL_COUNT, FLUID_MOVEMENT_FEATURES):
            raise ValueError(
                f"geometry frame {index} has an invalid fluid-movement shape"
            )
        if boxes.shape != (CELL_COUNT, MAX_DETAIL_BOXES, 6):
            raise ValueError(f"geometry frame {index} has an invalid box shape")
        if box_mask.shape != (CELL_COUNT, MAX_DETAIL_BOXES):
            raise ValueError(f"geometry frame {index} has an invalid box-mask shape")
        if agent_bounds.shape != (6,) or target_bounds.shape != (6,):
            raise ValueError(
                f"geometry frame {index} entity bounds must each have shape (6,)"
            )
        if agent_los_offset.shape != (3,) or target_los_offset.shape != (3,):
            raise ValueError(
                f"geometry frame {index} LOS offsets must have shape (3,)"
            )
        if (
            not np.all(np.isfinite(boxes))
            or not np.all(np.isfinite(movement))
            or not np.all(np.isfinite(fluid_movement))
            or not np.all(np.isfinite(agent_bounds))
            or not np.all(np.isfinite(target_bounds))
            or not np.all(np.isfinite(agent_los_offset))
            or not np.all(np.isfinite(target_los_offset))
            or not np.all(np.isfinite(fluid_fill_height))
        ):
            raise ValueError(f"geometry frame {index} contains non-finite geometry")
        fluid_cells = (flags.astype(np.int32) & FLAG_FLUID) != 0
        if np.any(
            (fluid_fill_height[cell_mask.astype(bool)] < -1.0)
            | (fluid_fill_height[cell_mask.astype(bool)] > 1.0)
        ):
            raise ValueError(
                f"geometry frame {index} has an invalid retained fluid fill"
            )
        if np.any(
            (fluid_fill_height < 0.0)
            & cell_mask.astype(bool)
            & ~fluid_cells
        ):
            raise ValueError(
                f"geometry frame {index} marks non-fluid fill unavailable"
            )
        if np.any(fluid_movement[fluid_cells, 3] <= 0.0):
            raise ValueError(
                f"geometry frame {index} has an invalid fluid speed multiplier"
            )
        fluid_fill_rows.append(fluid_fill_height)
        active = box_mask.astype(bool)
        full_cube = (
            active[:, 0]
            & ~np.any(active[:, 1:], axis=1)
            & np.all(boxes[:, 0] == _UNIT_CUBE_NUMPY, axis=1)
        )
        full_cube_cells = np.flatnonzero(full_cube)
        exception_cells = np.flatnonzero(np.any(active, axis=1) & ~full_cube)
        exception_boxes = boxes[exception_cells]
        exception_mask = active[exception_cells]
        encoded_shapes.append(
            (
                full_cube_cells,
                exception_cells,
                exception_boxes,
                exception_mask,
            )
        )
        required_full_cube_capacity = max(
            required_full_cube_capacity,
            int(full_cube_cells.size),
        )
        required_exception_capacity = max(
            required_exception_capacity,
            int(exception_cells.size),
        )
        required_exception_box_capacity = max(
            required_exception_box_capacity,
            int(np.count_nonzero(exception_mask)),
        )
        active_slots = np.flatnonzero(np.any(exception_mask, axis=0))
        if active_slots.size:
            detail_box_capacity = max(
                detail_box_capacity,
                int(active_slots[-1]) + 1,
            )
        if np.any(boxes[..., :3][active] > boxes[..., 3:][active]):
            raise ValueError(f"geometry frame {index} has an inverted collision box")
        if np.any(agent_bounds[:3] > agent_bounds[3:]):
            raise ValueError(f"geometry frame {index} has inverted agent bounds")
        if np.any(target_bounds[:3] > target_bounds[3:]):
            raise ValueError(f"geometry frame {index} has inverted target bounds")

    full_cube_capacity = _fixed_capacity(
        collision_full_cube_capacity,
        required_full_cube_capacity,
        "collision_full_cube_capacity",
        CELL_COUNT,
    )
    exception_capacity = _fixed_capacity(
        collision_exception_capacity,
        required_exception_capacity,
        "collision_exception_capacity",
        CELL_COUNT,
    )
    exception_box_capacity = _fixed_capacity(
        collision_exception_box_capacity,
        required_exception_box_capacity,
        "collision_exception_box_capacity",
        CELL_COUNT * MAX_DETAIL_BOXES,
    )

    def stack(name: str, dtype: np.dtype) -> Array:
        return jnp.asarray(
            np.stack([np.asarray(frame[name], dtype=dtype) for frame in source]),
            dtype=dtype,
        )

    shape_index_rows = []
    full_cube_rows = []
    exception_cell_rows = []
    exception_box_index_rows = []
    exception_box_rows = []
    exception_box_cell_rows = []
    world_order_rows = []
    for full_cells, exception_cells, boxes, mask in encoded_shapes:
        shape_index = np.full(
            CELL_COUNT,
            COLLISION_SHAPE_NONE,
            dtype=np.int16,
        )
        shape_index[full_cells] = COLLISION_SHAPE_FULL_CUBE
        shape_index[exception_cells] = np.arange(
            exception_cells.size,
            dtype=np.int16,
        )
        full_row = np.full(full_cube_capacity, -1, dtype=np.int16)
        full_row[: full_cells.size] = full_cells
        exception_cell_row = np.full(
            exception_capacity,
            -1,
            dtype=np.int16,
        )
        exception_cell_row[: exception_cells.size] = exception_cells
        active_pairs = np.argwhere(mask[:, :detail_box_capacity])
        exception_box_index_row = np.full(
            (exception_capacity, detail_box_capacity),
            -1,
            dtype=np.int16,
        )
        exception_box_row = np.zeros(
            (exception_box_capacity, 6),
            dtype=np.float32,
        )
        exception_box_cell_row = np.full(
            exception_box_capacity,
            -1,
            dtype=np.int16,
        )
        for flat_index, (exception_index, slot) in enumerate(active_pairs):
            exception_box_index_row[exception_index, slot] = flat_index
            exception_box_row[flat_index] = boxes[exception_index, slot]
            exception_box_cell_row[flat_index] = exception_cells[
                exception_index
            ]
        shape_index_rows.append(shape_index)
        full_cube_rows.append(full_row)
        exception_cell_rows.append(exception_cell_row)
        exception_box_index_rows.append(exception_box_index_row)
        exception_box_rows.append(exception_box_row)
        exception_box_cell_rows.append(exception_box_cell_row)
        full_keys = np.where(
            full_row >= 0,
            full_row.astype(np.int32) * detail_box_capacity,
            np.iinfo(np.int32).max,
        )
        exception_keys = np.full(
            exception_box_capacity,
            np.iinfo(np.int32).max,
            dtype=np.int32,
        )
        exception_keys[: active_pairs.shape[0]] = (
            exception_cells[active_pairs[:, 0]].astype(np.int32)
            * detail_box_capacity
            + active_pairs[:, 1].astype(np.int32)
        )
        world_order_rows.append(
            np.argsort(
                np.concatenate((full_keys, exception_keys.reshape(-1))),
                kind="stable",
            ).astype(np.int16)
        )

    state = GeometryState(
        origin=stack("origin", np.int32),
        cell_mask=stack("cell_mask", np.bool_),
        flags=stack("flags", np.int32),
        fluid_level=stack("fluid_level", np.int32),
        support=stack("support", np.int32),
        block_damage=stack("block_damage", np.int32),
        fluid_damage=stack("fluid_damage", np.int32),
        movement=stack("movement", np.float32),
        fluid_movement=stack("fluid_movement", np.float32),
        collision_shape_index=jnp.asarray(
            np.stack(shape_index_rows),
            dtype=jnp.int16,
        ),
        collision_full_cube_cell=jnp.asarray(
            np.stack(full_cube_rows),
            dtype=jnp.int16,
        ),
        collision_exception_cell=jnp.asarray(
            np.stack(exception_cell_rows),
            dtype=jnp.int16,
        ),
        collision_exception_box_index=jnp.asarray(
            np.stack(exception_box_index_rows),
            dtype=jnp.int16,
        ),
        collision_exception_boxes=jnp.asarray(
            np.stack(exception_box_rows),
            dtype=jnp.float32,
        ),
        collision_exception_box_cell=jnp.asarray(
            np.stack(exception_box_cell_rows),
            dtype=jnp.int16,
        ),
        collision_world_order=jnp.asarray(
            np.stack(world_order_rows),
            dtype=jnp.int16,
        ),
        agent_bounds=stack("agent_bounds", np.float32),
        target_bounds=stack("target_bounds", np.float32),
        agent_los_offset=stack("agent_los_offset", np.float32),
        target_los_offset=stack("target_los_offset", np.float32),
        fluid_fill_height=jnp.asarray(
            np.stack(fluid_fill_rows),
            dtype=jnp.float32,
        ),
    )
    if state.cell_mask.shape[1:] != (CELL_COUNT,):
        raise ValueError("geometry cell_mask has the wrong fixed shape")
    if state.collision_shape_index.shape[1:] != (CELL_COUNT,):
        raise ValueError("geometry collision_shape_index has the wrong shape")
    if state.collision_full_cube_cell.shape[1:] != (full_cube_capacity,):
        raise ValueError("geometry full-cube table has the wrong shape")
    if state.collision_exception_cell.shape[1:] != (exception_capacity,):
        raise ValueError("geometry exception-cell table has the wrong shape")
    if (
        state.collision_exception_box_index.shape[1]
        != exception_capacity
        or state.collision_exception_box_index.shape[2] < 1
        or state.collision_exception_box_index.shape[2] > MAX_DETAIL_BOXES
    ):
        raise ValueError("geometry exception box index has the wrong shape")
    if (
        state.collision_exception_boxes.shape[1:]
        != (exception_box_capacity, 6)
        or state.collision_exception_box_cell.shape[1:]
        != (exception_box_capacity,)
    ):
        raise ValueError("geometry exception box table has the wrong shape")
    world_box_capacity = (
        full_cube_capacity + exception_box_capacity
    )
    if state.collision_world_order.shape[1:] != (world_box_capacity,):
        raise ValueError("geometry collision order has the wrong fixed shape")
    return state


def geometry_atlas_from_numpy(
    frames: Sequence[Mapping[str, Any]],
    *,
    world_ids: Sequence[int] | None = None,
    capacity: int | None = None,
    require_exact: bool = True,
    collision_full_cube_capacity: int | None = None,
    collision_exception_capacity: int | None = None,
    collision_exception_box_capacity: int | None = None,
) -> GeometryAtlas:
    """Build a padded fixed-capacity atlas outside JIT.

    Atlas tiles are not merged or reinterpreted. Callers must capture
    overlapping complete native frames; compiled selection chooses one exact
    frame whose interior covers all required entities.
    """

    source = list(frames)
    if not source:
        raise ValueError("at least one geometry atlas frame is required")
    maximum = len(source) if capacity is None else int(capacity)
    if maximum < len(source) or maximum <= 0:
        raise ValueError("atlas capacity must cover every supplied frame")
    ids = [0] * len(source) if world_ids is None else list(world_ids)
    if len(ids) != len(source):
        raise ValueError("world_ids must match the number of atlas frames")
    if any(int(value) < 0 for value in ids):
        raise ValueError("world_ids must be non-negative")

    geometry = geometry_state_from_numpy(
        source,
        require_exact=require_exact,
        collision_full_cube_capacity=collision_full_cube_capacity,
        collision_exception_capacity=collision_exception_capacity,
        collision_exception_box_capacity=collision_exception_box_capacity,
    )
    padding = maximum - len(source)

    def pad_leaf(leaf: Array | None) -> Array | None:
        if leaf is None:
            return None
        if padding == 0:
            return leaf
        pad_shape = (padding,) + leaf.shape[1:]
        return jnp.concatenate(
            (leaf, jnp.zeros(pad_shape, dtype=leaf.dtype)),
            axis=0,
        )

    return GeometryAtlas(
        geometry=GeometryState(*(pad_leaf(leaf) for leaf in geometry)),
        tile_mask=jnp.arange(maximum) < len(source),
        world_id=jnp.asarray(ids + [-1] * padding, dtype=jnp.int32),
    )
