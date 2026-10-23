"""Automatic step-up (gen 7, row 1f, lane_traversal 2026-10-23): a grounded npc_walk move blocked by a one-block ledge lifts
and settles on the ledge top; a two-block wall, a blocked headroom and a free move are unchanged.

Geometry fixture as tests/jax/test_geometry.py: floor top at y = 65, agent AABB half-width 0.3 and height 1.8. A ledge
cell at (1, 0, 0) occupies y 65..66 at x 1..2, so a move from x = 0.5 by 0.6 is blocked at x = 0.7. Not differential-gated
against the server (FIDELITY-0.7.2.md row 1d: no up-step fixture); these tests check the claimed geometry only.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.geometry.contract import FLAG_OPAQUE, FLAG_SOLID, cell_index, empty_geometry
from hytalegym.jax.world import geometry_state_from_numpy, resolve_aabb_motion
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import auto_step_up
from hytalegym.sim.gen7.profile import PROFILE as GEN7

from . import support  # noqa: F401  (keeps the sim test package import order the same as its siblings)

CLIMB = 1.3


def _solid(frame, x, y, z) -> None:
    cell = cell_index(x, y, z)
    frame["cell_mask"][cell] = 1
    frame["flags"][cell] = FLAG_SOLID | FLAG_OPAQUE
    frame["collision_boxes"][cell, 0] = [0, 0, 0, 1, 1, 1]
    frame["collision_box_mask"][cell, 0] = 1


def _frame(*, ledge_cells=(), ceiling_cells=()):
    frame = empty_geometry()
    frame["available"] = 1
    frame["exact_collision_shapes"] = 1
    frame["origin"][:] = [0, 65, 0]
    frame["agent_bounds"][:] = [-0.3, 0.0, -0.3, 0.3, 1.8, 0.3]
    frame["agent_los_offset"][:] = [0.0, 0.9, 0.0]
    for x in range(-2, 4):
        _solid(frame, x, -1, 0)
    for x, y, z in list(ledge_cells) + list(ceiling_cells):
        _solid(frame, x, y, z)
    return geometry_state_from_numpy(frame)


def _rows(*values):
    return jnp.asarray(values, dtype=jnp.float32)


def test_one_block_ledge_is_stepped_onto_and_two_block_wall_is_not() -> None:
    position = _rows([0.5, 65.0, 0.0])
    horizontal = _rows([0.6, 0.0, 0.0])
    one = _frame(ledge_cells=[(1, 0, 0)])
    target, accepted, exhausted = auto_step_up.step_target(jnp, resolve_aabb_motion, one, position, horizontal, CLIMB)
    assert bool(accepted[0]) and not bool(exhausted[0])
    assert float(target[0, 0]) == pytest.approx(1.1, abs=2.0e-3)
    assert float(target[0, 1]) == pytest.approx(66.0, abs=2.0e-3)

    two = _frame(ledge_cells=[(1, 0, 0), (1, 1, 0)])
    target, accepted, _ = auto_step_up.step_target(jnp, resolve_aabb_motion, two, position, horizontal, CLIMB)
    assert not bool(accepted[0])


def test_a_ceiling_blocks_the_lift_and_a_free_move_is_not_a_step() -> None:
    position = _rows([0.5, 65.0, 0.0])
    horizontal = _rows([0.6, 0.0, 0.0])
    # Head top at 66.8 now and 68.1 after the lift: a ceiling at y 67..68 stops the lift short.
    ceiling = _frame(ledge_cells=[(1, 0, 0)], ceiling_cells=[(0, 2, 0)])
    _, accepted, _ = auto_step_up.step_target(jnp, resolve_aabb_motion, ceiling, position, horizontal, CLIMB)
    assert not bool(accepted[0])

    open_floor = _frame()
    _, accepted, _ = auto_step_up.step_target(jnp, resolve_aabb_motion, open_floor, position, _rows([0.1, 0.0, 0.0]), CLIMB)
    assert not bool(accepted[0])


def _fake_motion_stage(resolve):
    """A stand-in for ``_tick_agent_geometry_motion``: one blocked, grounded sweep. Its globals carry what the construct reads."""

    namespace = {"jnp": jnp, "AGENT_MOTION_NPC_WALK": 1, "_resolve_aabb_motion_result": resolve}
    code = (
        "def _tick_agent_geometry_motion(geometry, position, velocity, impulse, residual, control_lock, grounded_at_start,\n"
        "        applied_vertical_velocity, fall_speed, fall_start_y, move_speed, in_fluid, motion_delta, params):\n"
        "    disp = jnp.stack((velocity[:, 0] * motion_delta, jnp.zeros_like(motion_delta), velocity[:, 2] * motion_delta), axis=1)\n"
        "    m = _resolve_aabb_motion_result(geometry, position, disp)\n"
        "    return (m.position, velocity, impulse, residual, control_lock, m.grounded, m.geometry_exhausted,\n"
        "            fall_speed, fall_start_y, move_speed)\n"
    )
    exec(code, namespace)
    return namespace["_tick_agent_geometry_motion"]


def _profile(switch: bool):
    return SimpleNamespace(constructs=SimpleNamespace(auto_step_up=switch))


def test_the_wrapped_motion_stage_steps_up_and_leaves_gen5_unchanged() -> None:
    original = _fake_motion_stage(resolve_aabb_motion)
    assert auto_step_up.construct(original, _profile(False)) is original
    wrapped = auto_step_up.construct(original, _profile(True))
    geometry = _frame(ledge_cells=[(1, 0, 0)])
    params = SimpleNamespace(agent_motion_model=jnp.asarray([1], dtype=jnp.int32),
                             agent_walk_max_climb_height=jnp.float32(CLIMB))
    flag = jnp.asarray([False])
    args = (geometry, _rows([0.5, 65.0, 0.0]), _rows([0.6, 0.0, 0.0]), flag, flag, flag, jnp.asarray([True]),
            jnp.zeros((1,), jnp.float32), jnp.zeros((1,), jnp.float32), jnp.full((1,), 65.0, jnp.float32),
            jnp.full((1,), 5.5, jnp.float32), flag, jnp.asarray([1.0], jnp.float32), params)
    eager = wrapped(*args)
    compiled = jax.jit(lambda *arrays: wrapped(*arrays, params))(*args[:-1])
    for out in (eager, compiled):
        position, velocity, *_rest, grounded, _exhausted, _fall, _start, _speed = out
        assert float(position[0, 1]) == pytest.approx(66.0, abs=2.0e-3)
        assert float(position[0, 0]) == pytest.approx(1.1, abs=2.0e-3)
        assert float(velocity[0, 1]) == 0.0
        assert bool(grounded[0])
    plain = original(*args)
    assert float(plain[0][0, 1]) == pytest.approx(65.0, abs=2.0e-3)


def test_gen_profiles_switch_and_the_hook_is_registered() -> None:
    assert GEN5.constructs.auto_step_up is False
    assert GEN7.constructs.auto_step_up is True
    from hytalegym.sim.gen7.overlay import HOOKS

    hook = next(h for h in HOOKS if h.id == "auto_step_up")
    assert (hook.path, hook.function, hook.switch, hook.fidelity_row) == (
        "combat/runtime/agent.py", "_tick_agent_geometry_motion", "auto_step_up", "1f")
    assert hook.construct == "hytalegym.sim.gen7.mechanics.auto_step_up:construct"
    assert np.isclose(auto_step_up.TOLERANCE, 1.0e-3)
