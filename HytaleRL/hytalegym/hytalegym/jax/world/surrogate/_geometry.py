"""Leaf helpers extracted verbatim from traversal.py."""

import jax
import numpy as np
from hytalegym.worldgen.region import CAPTURE_BLOCKS_PER_AXIS, CHUNK_SIZE, CORE_BLOCKS_PER_AXIS


_EPSILON = 1.0e-6


_SUPPORT_EPSILON = 2.0e-4


def _swept_aabb_hits_box(
    start: np.ndarray,
    end: np.ndarray,
    bounds: np.ndarray,
    box: np.ndarray,
) -> bool:
    lower = 0.0
    upper = 1.0
    for axis in range(3):
        minimum = float(box[axis] - bounds[axis + 3]) + _EPSILON
        maximum = float(box[axis + 3] - bounds[axis]) - _EPSILON
        delta = float(end[axis] - start[axis])
        if abs(delta) <= _EPSILON:
            if not minimum < float(start[axis]) < maximum:
                return False
            continue
        first = (minimum - float(start[axis])) / delta
        second = (maximum - float(start[axis])) / delta
        lower = max(lower, min(first, second))
        upper = min(upper, max(first, second))
        if lower >= upper - _EPSILON:
            return False
    return lower < 1.0 - _EPSILON and upper > _EPSILON


def _segment_hits_box(
    start: np.ndarray,
    end: np.ndarray,
    bounds: np.ndarray,
    box: np.ndarray,
) -> bool:
    if not (
        start[1] + bounds[4] > box[1] + _EPSILON
        and start[1] + bounds[1] < box[4] - _EPSILON
    ):
        return False
    interval = _segment_interval(
        start,
        end,
        (
            box[0] - bounds[3],
            box[3] - bounds[0],
            box[2] - bounds[5],
            box[5] - bounds[2],
        ),
    )
    return interval is not None and interval[0] < interval[1] - _EPSILON


def _movement_bounds(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    start_position = np.asarray(
        (start[0], start[1] - float(bounds[1]), start[2]),
        dtype=np.float64,
    )
    end_position = np.asarray(
        (end[0], end[1] - float(bounds[1]), end[2]),
        dtype=np.float64,
    )
    minimum = np.minimum(
        start_position + bounds[:3],
        end_position + bounds[:3],
    )
    maximum = np.maximum(
        start_position + bounds[3:],
        end_position + bounds[3:],
    )
    minimum[1] -= _SUPPORT_EPSILON
    return minimum, maximum


def _gate_allows(
    gate: tuple[int, int],
    runtime_slot: int,
    state: int,
) -> bool:
    gate_slot, state_mask = gate
    return gate_slot < 0 or (
        gate_slot == runtime_slot and bool(state_mask & (1 << state))
    )


def _support_interval(
    start: np.ndarray,
    end: np.ndarray,
    bounds: np.ndarray,
    box: np.ndarray,
) -> tuple[float, float] | None:
    return _segment_interval(
        start,
        end,
        (
            box[0] - bounds[3],
            box[3] - bounds[0],
            box[2] - bounds[5],
            box[5] - bounds[2],
        ),
    )


def _segment_interval(
    start: np.ndarray,
    end: np.ndarray,
    limits: tuple[float, float, float, float],
) -> tuple[float, float] | None:
    lower = 0.0
    upper = 1.0
    for axis, minimum, maximum in (
        (0, limits[0], limits[1]),
        (2, limits[2], limits[3]),
    ):
        delta = float(end[axis] - start[axis])
        if abs(delta) <= _EPSILON:
            if not minimum + _EPSILON < start[axis] < maximum - _EPSILON:
                return None
            continue
        first = (minimum - float(start[axis])) / delta
        second = (maximum - float(start[axis])) / delta
        lower = max(lower, min(first, second))
        upper = min(upper, max(first, second))
        if lower >= upper - _EPSILON:
            return None
    return max(0.0, lower), min(1.0, upper)


def _intervals_cover_unit(intervals: list[tuple[float, float]]) -> bool:
    reach = 0.0
    for lower, upper in sorted(intervals):
        if lower > reach + _EPSILON:
            return False
        reach = max(reach, upper)
        if reach >= 1.0 - _EPSILON:
            return True
    return False


def _strict_overlap(
    left_minimum: np.ndarray,
    left_maximum: np.ndarray,
    right_minimum: np.ndarray,
    right_maximum: np.ndarray,
) -> bool:
    return bool(
        np.all(left_maximum > right_minimum + _EPSILON)
        and np.all(left_minimum < right_maximum - _EPSILON)
    )


def _strict_overlap_2d(
    left_minimum: np.ndarray,
    left_maximum: np.ndarray,
    right_minimum: np.ndarray,
    right_maximum: np.ndarray,
) -> bool:
    return bool(
        left_maximum[0] > right_minimum[0] + _EPSILON
        and left_minimum[0] < right_maximum[0] - _EPSILON
        and left_maximum[2] > right_minimum[2] + _EPSILON
        and left_minimum[2] < right_maximum[2] - _EPSILON
    )


def _face_intersects_core(
    minimum_x: float,
    maximum_x: float,
    minimum_z: float,
    maximum_z: float,
    origin: np.ndarray,
) -> bool:
    core_minimum_x = float(origin[0] + CHUNK_SIZE)
    core_maximum_x = core_minimum_x + CORE_BLOCKS_PER_AXIS
    core_minimum_z = float(origin[1] + CHUNK_SIZE)
    core_maximum_z = core_minimum_z + CORE_BLOCKS_PER_AXIS
    return (
        maximum_x > core_minimum_x
        and minimum_x < core_maximum_x
        and maximum_z > core_minimum_z
        and minimum_z < core_maximum_z
    )


def _block_key(x: int, y: int, z: int) -> int:
    return y * CAPTURE_BLOCKS_PER_AXIS**2 + z * CAPTURE_BLOCKS_PER_AXIS + x


def _decode_key(key: int) -> tuple[int, int, int]:
    y, remainder = divmod(key, CAPTURE_BLOCKS_PER_AXIS**2)
    z, x = divmod(remainder, CAPTURE_BLOCKS_PER_AXIS)
    return x, y, z


def _leaf(value: jax.Array, dtype: np.dtype) -> np.ndarray:
    result = np.asarray(jax.device_get(value))
    if result.dtype != np.dtype(dtype):
        raise TypeError(f"atlas leaf must use dtype {np.dtype(dtype)}")
    return result


def _host_index(value: int, capacity: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("tile_index must be an integer")
    if not 0 <= value < capacity:
        raise ValueError("tile_index exceeds atlas capacity")
    return value
