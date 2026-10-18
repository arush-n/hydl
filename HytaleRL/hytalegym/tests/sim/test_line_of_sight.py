"""Gen-7 detail-hitbox line of sight on synthetic 729-cell windows (no bridge, no render).

The construct is built the way the render builds it: the hook receives the gen-5 function and reads
the module constants from that function's globals. Reference: an independent float64 slab test per
opaque cell (the gen-5 rule), so full cubes must agree with it on every ray.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.sim.gen7.mechanics import line_of_sight as los

RADIUS = 4
EPS = 1.0e-9
OFFSETS = np.asarray(list(itertools.product(range(-RADIUS, RADIUS + 1), repeat=3)), dtype=np.float32)


def cell(dx: int, dy: int, dz: int) -> int:
    return (dx + RADIUS) * 81 + (dy + RADIUS) * 9 + (dz + RADIUS)


class VisibilityResult(NamedTuple):
    visible: object
    geometry_exhausted: object


def _namespace() -> dict:
    ns = {
        "jnp": jnp,
        "VisibilityResult": VisibilityResult,
        "_broadcast_geometry": lambda geometry, batch: geometry,
        "_CELL_OFFSETS": jnp.asarray(OFFSETS),
        "CELL_RADIUS": RADIUS,
        "FLAG_OPAQUE": 1,
        "_FLOAT_EPSILON": EPS,
    }
    exec("def line_of_sight_result(geometry, start, end):\n    raise AssertionError('gen-5 not called')\n", ns)
    return ns


def _switch(value: str):
    return SimpleNamespace(constructs=SimpleNamespace(line_of_sight=value))


def _detail():
    ns = _namespace()
    return los.construct(ns["line_of_sight_result"], _switch("detail_hitbox"))


def _geometry(opaque, boxes=None):
    flags = np.zeros((1, 729), np.int32)
    box_values = np.zeros((1, 729, 9, 6), np.float32)
    box_valid = np.zeros((1, 729, 9), bool)
    for index in opaque:
        flags[0, index] = 1
    for index, rows in (boxes or {}).items():
        for slot, row in enumerate(rows):
            box_values[0, index, slot] = row
            box_valid[0, index, slot] = True
    return SimpleNamespace(
        origin=jnp.zeros((1, 3), jnp.int32),
        cell_mask=jnp.ones((1, 729), bool),
        flags=jnp.asarray(flags),
        collision_boxes=jnp.asarray(box_values),
        collision_box_mask=jnp.asarray(box_valid),
    )


def _visible(function, geometry, start, end) -> bool:
    result = function(geometry, jnp.asarray([start], jnp.float32), jnp.asarray([end], jnp.float32))
    assert not bool(result.geometry_exhausted[0]) or not bool(result.visible[0])
    return bool(result.visible[0])


def _reference(opaque_boxes, start, end) -> bool:
    """Gen-5 rule in float64: a ray is blocked if it meets the open interior of any opaque box."""
    start, direction = np.asarray(start, np.float64), np.asarray(end, np.float64) - np.asarray(start, np.float64)
    for lo, hi in opaque_boxes:
        lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        entry, exit_ = -np.inf, np.inf
        for axis in range(3):
            if abs(direction[axis]) > EPS:
                t1 = (lo[axis] - start[axis]) / direction[axis]
                t2 = (hi[axis] - start[axis]) / direction[axis]
                entry = max(entry, min(t1, t2))
                exit_ = min(exit_, max(t1, t2))
            elif not (lo[axis] <= start[axis] < hi[axis]):
                entry, exit_ = np.inf, -np.inf
        if max(entry, 0.0) < min(exit_, 1.0) - EPS:
            return False
    return True


def test_gen5_profile_returns_the_original_function():
    ns = _namespace()
    assert los.construct(ns["line_of_sight_result"], _switch("non_transparent_blocks")) is ns["line_of_sight_result"]


def test_unknown_switch_value_is_refused():
    ns = _namespace()
    with pytest.raises(ValueError):
        los.construct(ns["line_of_sight_result"], _switch("nonsense"))


def test_full_cube_occluder_blocks_like_the_gen5_cell_rule():
    detail = _detail()
    geometry = _geometry(opaque=[cell(1, 0, 0)], boxes={cell(1, 0, 0): [[0, 0, 0, 1, 1, 1]]})
    assert not _visible(detail, geometry, (-1.5, 0.5, 0.5), (3.5, 0.5, 0.5))
    assert _visible(detail, geometry, (-1.5, 1.5, 0.5), (3.5, 1.5, 0.5))


def test_random_full_cube_windows_agree_with_the_float64_reference():
    rng = np.random.default_rng(570057)
    detail = _detail()
    cells = sorted(set(int(c) for c in rng.integers(0, 729, size=60)))
    opaque_boxes = []
    for index in cells:
        opaque_boxes.append((tuple(OFFSETS[index]), tuple(OFFSETS[index] + 1.0)))
    geometry = _geometry(opaque=cells, boxes={c: [[0, 0, 0, 1, 1, 1]] for c in cells})
    mismatches = 0
    for _ in range(300):
        start, end = rng.uniform(-3.5, 3.5, size=3), rng.uniform(-3.5, 3.5, size=3)
        mismatches += _visible(detail, geometry, start, end) != _reference(opaque_boxes, start, end)
    assert mismatches == 0


def test_half_block_lets_a_ray_pass_over_its_empty_upper_half():
    detail = _detail()
    half = _geometry(opaque=[cell(0, 0, 0)], boxes={cell(0, 0, 0): [[0, 0, 0, 1, 0.5, 1]]})
    full = _geometry(opaque=[cell(0, 0, 0)], boxes={cell(0, 0, 0): [[0, 0, 0, 1, 1, 1]]})
    assert _visible(detail, half, (-1.0, 0.75, 0.5), (2.0, 0.75, 0.5))
    assert not _visible(detail, full, (-1.0, 0.75, 0.5), (2.0, 0.75, 0.5))
    assert not _visible(detail, half, (-1.0, 0.25, 0.5), (2.0, 0.25, 0.5))


def test_opaque_cell_without_boxes_occludes_as_in_gen5():
    detail = _detail()
    geometry = _geometry(opaque=[cell(0, 0, 0)])
    assert not _visible(detail, geometry, (-1.0, 0.75, 0.5), (2.0, 0.75, 0.5))


def test_segment_leaving_the_window_is_exhausted_and_not_visible():
    detail = _detail()
    geometry = _geometry(opaque=[])
    result = detail(geometry, jnp.asarray([[-6.0, 0.5, 0.5]]), jnp.asarray([[0.0, 0.5, 0.5]]))
    assert bool(result.geometry_exhausted[0]) and not bool(result.visible[0])
