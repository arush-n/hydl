"""Direct Region v1 collision, support, medium, and LOS queries.

The kernels never expand a Region into a local frame. They visit the same
fixed-capacity cells covered by the actor's swept AABB; filler cells translate
their shared root shape exactly as native ``CollisionConfig`` does.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.geometry.contract import (
    FLAG_FLUID,
    FLAG_HAS_FLUID_MOVEMENT_SETTINGS,
    FLAG_SOLID,
)
from hytalegym.jax.world.geometry import (
    AabbFirstContactResult,
    CellEnvironmentResult,
    GroundResult,
    MotionResult,
    MovementMediumResult,
    SupportSweepResult,
    VisibilityResult,
    _aabb_first_contact_with_boxes,
    _resolve_motion_with_boxes,
    _support_sweep_with_boxes,
    _sweep_once,
)
from hytalegym.jax.world.region.mutable_geometry import (
    lookup_region_geometry_blocks,
    lookup_region_geometry_cells,
)
from hytalegym.jax.world.region.los import region_line_of_sight_result
from hytalegym.jax.world.region.los import MAX_REGION_LOS_CELLS
from hytalegym.jax.world.region.types import RegionAtlas, RegionGeometryState
from hytalegym.worldgen.region import CERTIFIED_SOURCE_REACH_BOUND_BLOCKS


Array = jax.Array
_FLOAT_EPSILON = jnp.float32(1.0e-7)
_LOCAL_SWEEP_SIZE = 3

#: Swept-AABB working set a *combat* runtime needs, as opposed to the bare
#: three-cell native default above. A 1.805-block actor plus one 30 Hz
#: fall/knockback step touches four voxel rows; three falsely exhausted valid
#: Region motion mid-duel.
#:
#: Measured against a 16-world Region atlas: exhaustion is a step-size cliff,
#: and each capacity ``c`` tolerates a step of roughly ``c - 2`` blocks.
#:
#:   capacity 3 -> 1.25-block step exhausts 98.4% of lanes, 1.5 exhausts 100%
#:   capacity 4 -> clean to 2.0, 2.5 exhausts
#:   capacity 5 -> clean to 3.0     capacity 6 -> clean to 4.0
#:
#: ``geometry_exhausted`` is sticky and truncates the episode, so a single
#: oversized step ends a run permanently. Anything binding this geometry to a
#: combat runtime must pass this rather than take the default -- keeping it
#: here means the two constructors that do so cannot drift apart again.
REGION_COMBAT_SWEEP_CELL_CAPACITY = (4, 4, 4)
_FLOAT32_EXACT_INTEGER_LIMIT = 1 << 24
_CORE_CELL_OFFSETS = jnp.asarray(
    [
        (dx, dy, dz)
        for dx in range(_LOCAL_SWEEP_SIZE)
        for dy in range(_LOCAL_SWEEP_SIZE)
        for dz in range(_LOCAL_SWEEP_SIZE)
    ],
    dtype=jnp.int32,
)


def region_geometry_from_atlas(
    atlas: RegionAtlas,
    environment_world_id: Array | Sequence[int],
    *,
    agent_bounds: Array | Sequence[float] | Sequence[Sequence[float]],
    target_bounds: Array | Sequence[float] | Sequence[Sequence[float]],
    agent_los_offset: Array | Sequence[float] | Sequence[Sequence[float]],
    target_los_offset: Array | Sequence[float] | Sequence[Sequence[float]],
    max_los_distance: float,
    sweep_cell_capacity: int | Sequence[int] = _LOCAL_SWEEP_SIZE,
) -> RegionGeometryState:
    """Bind an immutable Region atlas to exact entity collision semantics.

    This constructor is deliberately host-side. It validates the active shape
    palettes once and installs the native three-cell-per-axis swept-AABB
    capacity. ``sweep_cell_capacity`` lets a bound combat runtime widen that
    fixed working set for certified per-tick motion. Protruding geometry is
    reached through captured filler cells, never by scanning unrelated
    neighbouring roots.
    """

    world_ids = np.asarray(environment_world_id, dtype=np.int32)
    if world_ids.ndim != 1 or world_ids.size == 0:
        raise ValueError("environment_world_id must have shape (batch,)")

    region_mask = np.asarray(jax.device_get(atlas.region_mask), dtype=np.bool_)
    filler_root_known = np.asarray(
        jax.device_get(atlas.filler_root_known),
        dtype=np.bool_,
    )
    if filler_root_known.shape != region_mask.shape:
        raise ValueError("atlas filler-root availability has an invalid shape")
    boxes = np.asarray(
        jax.device_get(atlas.collision_boxes),
        dtype=np.float32,
    )
    box_mask = np.asarray(
        jax.device_get(atlas.collision_box_mask),
        dtype=np.bool_,
    )
    if boxes.ndim != 4 or boxes.shape[-1] != 6:
        raise ValueError("atlas collision boxes have an invalid shape")
    if box_mask.shape != boxes.shape[:-1]:
        raise ValueError("atlas collision box mask has an invalid shape")
    if region_mask.shape != (boxes.shape[0],):
        raise ValueError("atlas region mask has an invalid shape")

    active = box_mask & region_mask[:, None, None]
    protruding = active & (
        np.any(boxes[..., :3] < 0.0, axis=-1)
        | np.any(boxes[..., 3:] > 1.0, axis=-1)
    )
    if np.any(
        protruding
        & ~filler_root_known[:, None, None]
    ):
        raise ValueError(
            "protruding Region geometry requires native filler-root offsets"
        )
    if np.any(active):
        active_boxes = boxes[active]
        if not np.all(np.isfinite(active_boxes)):
            raise ValueError("atlas contains non-finite active collision boxes")
        if np.any(active_boxes[:, :3] > active_boxes[:, 3:]):
            raise ValueError("atlas contains inverted active collision boxes")
        shape_minimum = np.min(active_boxes[:, :3], axis=0)
        shape_maximum = np.max(active_boxes[:, 3:], axis=0)
    else:
        shape_minimum = np.zeros(3, dtype=np.float32)
        shape_maximum = np.ones(3, dtype=np.float32)
    observed_source_reach = float(
        np.max(
            np.maximum.reduce(
                (
                    np.zeros(3, dtype=np.float32),
                    -shape_minimum,
                    shape_maximum - np.float32(1.0),
                )
            )
        )
    )
    if observed_source_reach > CERTIFIED_SOURCE_REACH_BOUND_BLOCKS:
        raise ValueError(
            "atlas collision source reach exceeds the certified Region halo "
            f"bound ({observed_source_reach} > "
            f"{CERTIFIED_SOURCE_REACH_BOUND_BLOCKS})"
        )

    lower_extension = np.zeros(3, dtype=np.int32)
    upper_extension = np.zeros(3, dtype=np.int32)
    source_capacity = _host_sweep_cell_capacity(sweep_cell_capacity)
    offsets = np.asarray(
        [
            (x, y, z)
            for x in range(int(source_capacity[0]))
            for y in range(int(source_capacity[1]))
            for z in range(int(source_capacity[2]))
        ],
        dtype=np.int32,
    )
    los_distance = float(max_los_distance)
    if not np.isfinite(los_distance) or los_distance < 0.0:
        raise ValueError("max_los_distance must be finite and non-negative")
    agent_eye = _host_entity_leaf(
        agent_los_offset,
        world_ids.size,
        3,
        "agent_los_offset",
    )
    target_eye = _host_entity_leaf(
        target_los_offset,
        world_ids.size,
        3,
        "target_los_offset",
    )
    eye_delta = np.max(
        np.linalg.norm(
            np.broadcast_to(agent_eye, (world_ids.size, 3))
            - np.broadcast_to(target_eye, (world_ids.size, 3)),
            axis=1,
        )
    )
    # Any segment of Euclidean length D crosses at most
    # ceil(sqrt(3) * D) + 1 voxel cells. One additional cell covers the
    # half-open endpoint ownership adjustment.
    los_capacity = min(
        MAX_REGION_LOS_CELLS,
        max(
            1,
            int(np.ceil(np.sqrt(3.0) * (los_distance + eye_delta))) + 2,
        ),
    )

    return RegionGeometryState(
        atlas=atlas,
        environment_world_id=jnp.asarray(world_ids, dtype=jnp.int32),
        source_lower_extension=jnp.asarray(
            lower_extension,
            dtype=jnp.int32,
        ),
        source_upper_extension=jnp.asarray(
            upper_extension,
            dtype=jnp.int32,
        ),
        source_offsets=jnp.asarray(offsets, dtype=jnp.int32),
        source_capacity=jnp.asarray(source_capacity, dtype=jnp.int32),
        los_step_marker=jnp.zeros((los_capacity,), dtype=jnp.uint8),
        agent_bounds=_entity_leaf(
            agent_bounds,
            world_ids.size,
            6,
            "agent_bounds",
        ),
        target_bounds=_entity_leaf(
            target_bounds,
            world_ids.size,
            6,
            "target_bounds",
        ),
        agent_los_offset=jnp.asarray(agent_eye, dtype=jnp.float32),
        target_los_offset=jnp.asarray(target_eye, dtype=jnp.float32),
        mutable_blocks=None,
    )


def region_cell_environment_result(
    geometry: RegionGeometryState,
    position: Array,
) -> CellEnvironmentResult:
    """Gather complete immutable semantics for one cell per environment."""

    points = jnp.asarray(position, dtype=jnp.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("position must have shape [batch, 3]")
    _require_batch(geometry, points.shape[0])
    selected = lookup_region_geometry_cells(
        geometry,
        points[:, None, :],
        require_core=True,
    )
    available = selected.available[:, 0]
    return CellEnvironmentResult(
        present=available,
        flags=selected.flags[:, 0],
        fluid_level=selected.fluid_level[:, 0],
        fluid_fill_height=selected.fluid_fill_height[:, 0],
        support=selected.support[:, 0],
        block_damage=selected.block_damage[:, 0],
        fluid_damage=selected.fluid_damage[:, 0],
        movement=selected.movement[:, 0],
        fluid_movement=selected.fluid_movement[:, 0],
        geometry_exhausted=~available,
    )


def region_movement_medium_result(
    geometry: RegionGeometryState,
    position: Array,
) -> MovementMediumResult:
    """Return native FluidFX movement settings at each entity's feet."""

    cell = region_cell_environment_result(geometry, position)
    in_fluid = (
        cell.present
        & ((cell.flags & jnp.int32(FLAG_FLUID)) != 0)
    )
    fluid = cell.fluid_movement
    return MovementMediumResult(
        swim_up_speed=jnp.where(in_fluid, fluid[:, 0], 0.0),
        swim_down_speed=jnp.where(in_fluid, fluid[:, 1], 0.0),
        sink_speed=jnp.where(in_fluid, fluid[:, 2], 0.0),
        horizontal_speed_multiplier=jnp.where(
            in_fluid,
            fluid[:, 3],
            jnp.float32(1.0),
        ),
        field_of_view_multiplier=jnp.where(
            in_fluid,
            fluid[:, 4],
            jnp.float32(1.0),
        ),
        entry_velocity_multiplier=jnp.where(
            in_fluid,
            fluid[:, 5],
            jnp.float32(1.0),
        ),
        fluid_level=jnp.where(in_fluid, cell.fluid_level, jnp.int32(0)),
        in_fluid=in_fluid,
        has_authored_settings=(
            in_fluid
            & (
                (
                    cell.flags
                    & jnp.int32(FLAG_HAS_FLUID_MOVEMENT_SETTINGS)
                )
                != 0
            )
        ),
        geometry_exhausted=cell.geometry_exhausted,
    )


def region_line_of_sight_geometry_result(
    geometry: RegionGeometryState,
    start: Array,
    end: Array,
) -> VisibilityResult:
    """Return Region LOS using the same result surface as local geometry."""

    points = jnp.asarray(start, dtype=jnp.float32)
    terminal = jnp.asarray(end, dtype=jnp.float32)
    if points.ndim != 2 or points.shape[1] != 3 or terminal.shape != points.shape:
        raise ValueError("start and end must have shape [batch, 3]")
    _require_batch(geometry, points.shape[0])
    result = region_line_of_sight_result(
        geometry.atlas,
        points,
        terminal,
        geometry.environment_world_id,
        max_cells=geometry.los_step_marker.shape[0],
        geometry=geometry,
    )
    return VisibilityResult(
        visible=result.visible,
        geometry_exhausted=(
            result.geometry_exhausted | result.capacity_exceeded
        ),
    )


def region_aabb_core_available(
    geometry: RegionGeometryState,
    position: Array,
    bounds: Array,
) -> Array:
    """Prove that every cell touched by each AABB lies in a Region core."""

    points = jnp.asarray(position, dtype=jnp.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("position must have shape [batch, 3]")
    batch = points.shape[0]
    _require_batch(geometry, batch)
    entity_bounds = _broadcast_bounds(bounds, batch)
    minimum = jnp.floor(points + entity_bounds[:, :3]).astype(jnp.int32)
    maximum = jnp.floor(
        points + entity_bounds[:, 3:] - _FLOAT_EPSILON
    ).astype(jnp.int32)
    extent = maximum - minimum + 1
    blocks = minimum[:, None, :] + _CORE_CELL_OFFSETS[None, :, :]
    mask = jnp.all(blocks <= maximum[:, None, :], axis=2)
    selected = lookup_region_geometry_blocks(
        geometry,
        blocks,
        require_core=True,
    )
    precise = jnp.all(
        (blocks > jnp.int32(-_FLOAT32_EXACT_INTEGER_LIMIT))
        & (blocks < jnp.int32(_FLOAT32_EXACT_INTEGER_LIMIT)),
        axis=(1, 2),
    )
    return (
        jnp.all(extent <= jnp.int32(_LOCAL_SWEEP_SIZE), axis=1)
        & ~jnp.any(mask & ~selected.available, axis=1)
        & precise
    )


def region_aabb_grounded(
    geometry: RegionGeometryState,
    position: Array,
    *,
    probe_distance: float = 0.002,
) -> GroundResult:
    """Test immediate downward support against directly gathered Region boxes."""

    points = jnp.asarray(position, dtype=jnp.float32)
    batch = points.shape[0]
    displacement = jnp.zeros_like(points).at[:, 1].set(
        -jnp.float32(probe_distance)
    )
    minimum, maximum, valid, exhausted = _region_motion_boxes(
        geometry,
        points,
        displacement,
        _broadcast_bounds(geometry.agent_bounds, batch),
    )
    _, _, normal, hit = _sweep_once(
        points,
        displacement,
        _broadcast_bounds(geometry.agent_bounds, batch),
        minimum,
        maximum,
        valid,
    )
    return GroundResult(
        grounded=hit & (normal[:, 1] > 0.5),
        geometry_exhausted=exhausted,
    )


def region_aabb_support_sweep(
    geometry: RegionGeometryState,
    position: Array,
    displacement: Array,
    bounds: Array | None = None,
    *,
    probe_distance: float = 0.002,
) -> SupportSweepResult:
    """Find the connected support interval for Region-backed horizontal motion."""

    points = jnp.asarray(position, dtype=jnp.float32)
    delta = jnp.asarray(displacement, dtype=jnp.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("position must have shape [batch, 3]")
    if delta.shape != points.shape:
        raise ValueError("displacement must match position")
    batch = points.shape[0]
    entity_bounds = _broadcast_bounds(
        geometry.agent_bounds if bounds is None else bounds,
        batch,
    )
    horizontal = delta.at[:, 1].set(0.0)
    candidate = horizontal.at[:, 1].set(-jnp.float32(probe_distance))
    minimum, maximum, valid, exhausted = _region_motion_boxes(
        geometry,
        points,
        candidate,
        entity_bounds,
    )
    return _support_sweep_with_boxes(
        points,
        horizontal,
        entity_bounds,
        minimum,
        maximum,
        valid,
        probe_distance,
        exhausted,
    )


def region_aabb_first_contact_result(
    geometry: RegionGeometryState,
    position: Array,
    displacement: Array,
    bounds: Array,
    *,
    include_closed_endpoint: bool = False,
) -> AabbFirstContactResult:
    """Return the first exact Region contact without slide response.

    The default retains half-open motion broad-phase semantics. Point-ray
    callers request a closed destination so a face exactly at maximum range
    remains eligible.
    """

    points = jnp.asarray(position, dtype=jnp.float32)
    delta = jnp.asarray(displacement, dtype=jnp.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("position must have shape [batch, 3]")
    if delta.shape != points.shape:
        raise ValueError("displacement must match position")
    batch = points.shape[0]
    _require_batch(geometry, batch)
    entity_bounds = _broadcast_bounds(bounds, batch)
    minimum, maximum, valid, exhausted = _region_motion_boxes(
        geometry,
        points,
        delta,
        entity_bounds,
        include_closed_endpoint=include_closed_endpoint,
    )
    return _aabb_first_contact_with_boxes(
        points,
        delta,
        entity_bounds,
        minimum,
        maximum,
        valid,
        exhausted,
    )


def region_resolve_aabb_motion(
    geometry: RegionGeometryState,
    position: Array,
    displacement: Array,
    bounds: Array | None = None,
    stop_on_vertical_collision: Array | None = None,
    skin_distance: float = 1.0e-5,
) -> MotionResult:
    """Continuously sweep an entity AABB through exact Region collision boxes."""

    points = jnp.asarray(position, dtype=jnp.float32)
    delta = jnp.asarray(displacement, dtype=jnp.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("position must have shape [batch, 3]")
    if delta.shape != points.shape:
        raise ValueError("displacement must match position")
    batch = points.shape[0]
    if (
        isinstance(skin_distance, bool)
        or not np.isfinite(skin_distance)
        or skin_distance < 0.0
    ):
        raise ValueError("skin_distance must be finite and non-negative")
    entity_bounds = _broadcast_bounds(
        geometry.agent_bounds if bounds is None else bounds,
        batch,
    )
    if stop_on_vertical_collision is None:
        stop_vertical = jnp.zeros((batch,), dtype=jnp.bool_)
    else:
        stop_vertical = jnp.asarray(
            stop_on_vertical_collision,
            dtype=jnp.bool_,
        )
        if stop_vertical.ndim == 0:
            stop_vertical = jnp.broadcast_to(stop_vertical, (batch,))
        if stop_vertical.shape != (batch,):
            raise ValueError(
                "stop_on_vertical_collision must have shape [batch]"
            )
    minimum, maximum, valid, exhausted = _region_motion_boxes(
        geometry,
        points,
        delta,
        entity_bounds,
    )
    return _resolve_motion_with_boxes(
        points,
        delta,
        entity_bounds,
        minimum,
        maximum,
        valid,
        exhausted,
        stop_vertical,
        jnp.float32(skin_distance),
    )


def _region_motion_boxes(
    geometry: RegionGeometryState,
    position: Array,
    displacement: Array,
    bounds: Array,
    *,
    include_closed_endpoint: bool = False,
) -> tuple[Array, Array, Array, Array]:
    batch = position.shape[0]
    _require_batch(geometry, batch)
    start_minimum = position + bounds[:, :3]
    start_maximum = position + bounds[:, 3:]
    end_minimum = start_minimum + displacement
    end_maximum = start_maximum + displacement
    swept_minimum = jnp.minimum(start_minimum, end_minimum)
    swept_maximum = jnp.maximum(start_maximum, end_maximum)
    minimum_cell = jnp.floor(swept_minimum).astype(jnp.int32)
    candidate_maximum = (
        swept_maximum
        if include_closed_endpoint
        else swept_maximum - _FLOAT_EPSILON
    )
    maximum_cell = jnp.floor(candidate_maximum).astype(jnp.int32)
    target_extent = maximum_cell - minimum_cell + 1
    target_capacity_exceeded = jnp.any(
        target_extent > geometry.source_capacity,
        axis=1,
    )

    source_minimum = minimum_cell
    source_maximum = maximum_cell
    source_extent = source_maximum - source_minimum + 1
    source_capacity_exceeded = jnp.any(
        source_extent > geometry.source_capacity,
        axis=1,
    )
    source_blocks = (
        source_minimum[:, None, :] + geometry.source_offsets[None, :, :]
    )
    source_mask = jnp.all(
        source_blocks <= source_maximum[:, None, :],
        axis=2,
    )
    selected = lookup_region_geometry_blocks(
        geometry,
        source_blocks,
        require_core=False,
    )
    root_blocks = (
        source_blocks
        - selected.filler_root_offset
    )
    base = root_blocks[:, :, None, :].astype(jnp.float32)
    minimum = selected.collision_boxes[..., :3] + base
    maximum = selected.collision_boxes[..., 3:] + base
    solid = (selected.flags & jnp.int32(FLAG_SOLID)) != 0
    valid = (
        source_mask[:, :, None]
        & selected.available[:, :, None]
        & selected.filler_root_available[:, :, None]
        & selected.collision_box_mask
        & solid[:, :, None]
    )

    core_blocks = (
        minimum_cell[:, None, :] + geometry.source_offsets[None, :, :]
    )
    core_mask = jnp.all(
        core_blocks <= maximum_cell[:, None, :],
        axis=2,
    )
    core = lookup_region_geometry_blocks(
        geometry,
        core_blocks,
        require_core=True,
    )
    source_precise = jnp.all(
        (
            source_blocks
            > jnp.int32(-_FLOAT32_EXACT_INTEGER_LIMIT)
        )
        & (
            source_blocks
            < jnp.int32(_FLOAT32_EXACT_INTEGER_LIMIT)
        ),
        axis=(1, 2),
    )
    exhausted = (
        target_capacity_exceeded
        | source_capacity_exceeded
        | jnp.any(source_mask & ~selected.available, axis=1)
        | jnp.any(core_mask & ~core.available, axis=1)
        | ~source_precise
    )
    return (
        minimum.reshape(batch, -1, 3),
        maximum.reshape(batch, -1, 3),
        valid.reshape(batch, -1),
        exhausted,
    )


def _entity_leaf(
    value: Array | Sequence[float] | Sequence[Sequence[float]],
    batch: int,
    width: int,
    label: str,
) -> Array:
    return jnp.asarray(
        _host_entity_leaf(value, batch, width, label),
        dtype=jnp.float32,
    )


def _host_sweep_cell_capacity(value: int | Sequence[int]) -> np.ndarray:
    if isinstance(value, bool):
        raise TypeError("sweep_cell_capacity must contain integers")
    capacity = np.asarray(value)
    if capacity.ndim == 0:
        capacity = np.repeat(capacity, 3)
    if capacity.shape != (3,) or capacity.dtype.kind not in "iu":
        raise TypeError("sweep_cell_capacity must be an integer or three integers")
    if np.any(capacity < 1):
        raise ValueError("sweep_cell_capacity must be positive")
    return capacity.astype(np.int32)


def _host_entity_leaf(
    value: Array | Sequence[float] | Sequence[Sequence[float]],
    batch: int,
    width: int,
    label: str,
) -> np.ndarray:
    leaf = np.asarray(value, dtype=np.float32)
    if leaf.shape == (width,):
        leaf = leaf[None, :]
    if leaf.shape not in ((1, width), (batch, width)):
        raise ValueError(
            f"{label} must have shape ({width},), (1, {width}), "
            f"or ({batch}, {width})"
        )
    if not np.all(np.isfinite(leaf)):
        raise ValueError(f"{label} must contain only finite values")
    if width == 6 and np.any(leaf[:, :3] > leaf[:, 3:]):
        raise ValueError(f"{label} contains inverted bounds")
    return leaf


def _broadcast_bounds(bounds: Array, batch: int) -> Array:
    values = jnp.asarray(bounds, dtype=jnp.float32)
    if values.ndim == 1 and values.shape[0] == 6:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("bounds must have shape [batch|1, 6]")
    if values.shape[0] not in (1, batch):
        raise ValueError("bounds batch must be one or match position")
    if values.shape[0] == 1 and batch != 1:
        values = jnp.broadcast_to(values, (batch, 6))
    return values


def _require_batch(geometry: RegionGeometryState, batch: int) -> None:
    if geometry.environment_world_id.shape != (batch,):
        raise ValueError(
            "Region geometry world IDs must match the environment batch"
        )
