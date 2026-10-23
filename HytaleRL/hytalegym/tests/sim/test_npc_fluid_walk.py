"""Water for the npc_walk body (gen 7, row 10c; lane_swim, 2026-10-23).

0.7.2 MotionControllerWalk reads no fluid vertical cap (javap: computeMove stores ``maxVerticalSpeedFluid`` and never reloads it;
``computeNewFallSpeed`` has no fluid branch), so a walk NPC in water falls at the dry cap. Gen 5 capped the in-fluid fall at
``max_sink_speed_fluid`` (4.0). These tests check the claimed behaviour only: the switch, the parameter substitution, and a
rendered drop through a pool. They are not differential-gated against the server (no 0.7.2 water fixture exists).
"""

from __future__ import annotations

from collections import namedtuple
from importlib import import_module
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from hytalegym.geometry.contract import (
    FLAG_FLUID,
    FLAG_HAS_FLUID_MOVEMENT_SETTINGS,
    FLAG_OPAQUE,
    FLAG_SOLID,
    cell_index,
    empty_geometry,
)
from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import npc_fluid_walk
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY
from hytalegym.sim.gen7.profile import PROFILE as GEN7

from . import support

ORIGIN_Y = 65
DT = 1.0 / 30.0
WATER = [2.5, -2.5, -1.35, 0.6, 1.0, 1.0]  # 0.7.2 FluidFX Water: swim up, swim down, sink, horizontal, fov, entry
MODEL_PLAYER, MODEL_NPC_WALK = 0, 1

Params = namedtuple("Params", "agent_motion_model agent_walk_max_fall_speed agent_walk_max_sink_speed_fluid")


def test_switch_values_and_the_hook_are_declared() -> None:
    assert GEN5.constructs.swimming == "fluidfx_v0_5"
    assert GEN7.constructs.swimming == "npc_walk_072"
    assert npc_fluid_walk.SWITCH == "swimming" and npc_fluid_walk.IMPLEMENTED is True
    from hytalegym.sim.gen7.overlay import HOOKS

    hook = next(h for h in HOOKS if h.id == "npc_fluid_walk")
    assert (hook.path, hook.function, hook.switch, hook.fidelity_row) == (
        "combat/runtime/agent.py", "_tick_agent_geometry_motion", "swimming", "10c")
    assert hook.construct == "hytalegym.sim.gen7.mechanics.npc_fluid_walk:construct"


def test_other_swimming_values_return_the_original_stage() -> None:
    def original(*args):
        return args

    gen5_like = SimpleNamespace(constructs=SimpleNamespace(swimming="fluidfx_v0_5"))
    assert npc_fluid_walk.construct(original, gen5_like) is original


def _stand_in_stage():
    """A stand-in for ``_tick_agent_geometry_motion``: it returns the in-fluid cap it was given, and the dry cap."""

    namespace = {"jnp": jnp, "AGENT_MOTION_NPC_WALK": MODEL_NPC_WALK}
    code = (
        "def _tick_agent_geometry_motion(geometry, position, velocity, impulse, residual, control_lock, grounded_at_start,\n"
        "        applied_vertical_velocity, fall_speed, fall_start_y, move_speed, in_fluid, motion_delta, params):\n"
        "    return (params.agent_walk_max_sink_speed_fluid, params.agent_walk_max_fall_speed, in_fluid)\n"
    )
    exec(code, namespace)
    return namespace["_tick_agent_geometry_motion"]


def _call(stage, model):
    params = Params(jnp.int32(model), jnp.float32(15.0), jnp.float32(4.0))
    flag = jnp.asarray([True])
    zeros = jnp.zeros((1,), jnp.float32)
    return stage(None, None, None, None, None, None, None, zeros, zeros, zeros, zeros, flag, zeros + DT, params)


def test_npc_walk_in_fluid_gets_the_dry_cap_and_the_player_model_keeps_the_gen5_cap() -> None:
    profile = SimpleNamespace(constructs=SimpleNamespace(swimming="npc_walk_072"))
    stage = npc_fluid_walk.construct(_stand_in_stage(), profile)
    sink, dry, _ = _call(stage, MODEL_NPC_WALK)
    assert float(sink) == pytest.approx(15.0)
    assert float(dry) == pytest.approx(15.0)
    sink, _, _ = _call(stage, MODEL_PLAYER)
    assert float(sink) == pytest.approx(4.0)


def test_the_registry_routes_swimming_to_the_construct() -> None:
    from hytalegym.sim.gen7 import mechanics

    assert mechanics.CONSTRUCT_MODULES["swimming"] == "npc_fluid_walk"
    assert mechanics.require_construct(GEN7, "swimming") is npc_fluid_walk


# ---- rendered drop through a pool -------------------------------------------------------------------------------------------


def _solid(frame, x, y, z):
    cell = cell_index(x, y, z)
    frame["cell_mask"][cell] = 1
    frame["flags"][cell] = FLAG_SOLID | FLAG_OPAQUE
    frame["collision_boxes"][cell, 0] = [0, 0, 0, 1, 1, 1]
    frame["collision_box_mask"][cell, 0] = 1


def _water(frame, x, y, z):
    cell = cell_index(x, y, z)
    frame["cell_mask"][cell] = 1
    frame["flags"][cell] = FLAG_FLUID | FLAG_HAS_FLUID_MOVEMENT_SETTINGS
    frame["runtime_fluid_id"][cell] = 1
    frame["fluid_level"][cell] = 1
    frame["fluid_fill_height"][cell] = 1.0
    frame["fluid_movement"][cell] = WATER


def _pool_frame():
    frame = empty_geometry()
    frame["available"] = 1
    frame["exact_collision_shapes"] = 1
    frame["origin"][:] = [0, ORIGIN_Y, 0]
    frame["agent_bounds"][:] = [-0.3, 0.0, -0.3, 0.3, 1.8, 0.3]
    frame["agent_los_offset"][:] = [0.0, 0.9, 0.0]
    for x in range(-3, 5):
        for z in range(-3, 4):
            _solid(frame, x, -1, z)
    for x in range(0, 5):
        for z in range(-1, 2):
            for y in range(0, 3):
                _water(frame, x, y, z)
    return frame


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    out = tmp_path_factory.mktemp("render") / "gen7"
    R.render(GEN7_OVERLAY, out, source=HeadSource())
    package = support.mount_fresh(out).__name__
    return {
        "agent": import_module(f"{package}.combat.runtime.agent"),
        "types": import_module(f"{package}.combat.types"),
        "world": import_module(f"{package}.world"),
        "geometry": import_module(f"{package}.world.geometry"),
        "breathing": import_module(f"{package}.combat.mechanics.sources.breathing"),
        "factory": import_module(f"{package}.combat.mechanics.factory"),
    }


def _drop(engine, ticks=90):
    """Drop the npc_walk body from 3 blocks above the pool floor into the water; return the vertical speed per tick."""

    world, types, agent = engine["world"], engine["types"], engine["agent"]
    geometry = world.geometry_state_from_numpy(_pool_frame())
    params = types.default_combat_params(motion_model="npc_walk")
    pos = jnp.asarray([[2.5, ORIGIN_Y + 3.0, 0.0]], dtype=jnp.float32)
    vel = jnp.zeros((1, 3), jnp.float32)
    flag = jnp.asarray([False])
    fall = jnp.zeros((1,), jnp.float32)
    fall_start = jnp.full((1,), ORIGIN_Y + 3.0, jnp.float32)
    move = jnp.zeros((1,), jnp.float32)
    rows = []
    for _ in range(ticks):
        in_fluid = bool(engine["geometry"].movement_medium_result(geometry, pos).in_fluid[0])
        out = agent._tick_agent_geometry_motion(
            geometry, pos, vel, flag, flag, flag, jnp.asarray([False]), jnp.zeros((1,), jnp.float32),
            fall, fall_start, move, jnp.asarray([in_fluid]), jnp.asarray([DT], jnp.float32), params,
        )
        pos, vel, grounded = out[0], out[1], bool(out[5][0])
        fall, fall_start, move = out[7], out[8], out[9]
        rows.append((in_fluid, float(vel[0, 1]), float(pos[0, 1]), grounded))
        if grounded and len(rows) > 1:
            break
    return rows


def test_rendered_npc_walk_falls_through_the_pool_past_the_gen5_sink_cap(engine) -> None:
    rows = _drop(engine)
    wet = [vy for in_fluid, vy, _, _ in rows if in_fluid]
    assert len(wet) >= 10
    # Gen 5 holds the in-fluid fall at -4.0; the 0.7.2 walk has no such cap, so the fall passes it while in the water.
    assert min(wet) < -4.5
    assert rows[-1][3], "the body lands on the pool floor"


def test_rendered_horizontal_fluid_multiplier_and_oxygen_are_unchanged(engine) -> None:
    geometry = engine["world"].geometry_state_from_numpy(_pool_frame())
    medium = engine["geometry"].movement_medium_result(geometry, jnp.asarray([[2.5, ORIGIN_Y, 0.0]], jnp.float32))
    assert float(medium.horizontal_speed_multiplier[0]) == pytest.approx(0.6)
    breathing, factory = engine["breathing"], engine["factory"]
    rules = factory.default_mechanics_rules(1)
    state = factory.empty_mechanics_state(1, rules)
    entities = state.resources.shape[1]
    cannot = jnp.zeros((1, entities), jnp.bool_).at[:, 0].set(True)
    alive = jnp.ones((1, entities), jnp.bool_)
    seen = []
    for _ in range(int(20 * 30)):
        state, events = breathing.tick_breathing(state, rules, jnp.asarray([DT], jnp.float32), cannot, cannot, alive)
        if bool(events.requested[0, 0]):
            seen.append(float(events.amount[0, 0]))
    # Oxygen falls 3 per 0.5 s while submerged, reaches 0 near 16.7 s, then drowning deals 10 per second (the probe agrees).
    assert float(state.resources[0, 0, 6]) == pytest.approx(0.0)
    assert seen and set(seen) == {10.0}
    assert len(seen) >= 3
