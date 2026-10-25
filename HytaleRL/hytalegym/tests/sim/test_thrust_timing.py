"""Sword thrust timing at 0.7.2 (row 5f, 2026-10-22): the dash and stab events and the hold speed; the total stays 0.601.

(1) the adapter ``sword_thrust_timing`` and ``melee_hold_constants`` read both tables; (2) the gen-7 render's thrust
ability carries the 0.7.2 values; (3) the gen-5 render keeps 0.601 / 0.05 / 0.8; (4) on the gen-7 render, slot 3 requested
on steps 1..30 (1-based, one tick is 1/30 s) becomes active on step 23, and the agent's first moving step and planar
displacement are pinned (measured on this render; the numbers are printed with ``pytest -s``).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.sim.common.content import adapters as A
from hytalegym.sim.common.content import load
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen5.overlay import OVERLAY as GEN5
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7

from .support import mount_fresh, render_for_test, unmount

THRUST = "Weapon_Sword_Primary_Thrust"
THRUST_SLOT = 3
REQUEST_UNTIL = 30
TICKS = 60
#: Pinned from the gen-7 render of this tree, measured with this loop on 2026-10-22 (render values, not live): the last step
#: the thrust slot is active, the first step the agent moves and the planar displacement from the start at steps 30 and 60.
#: The 0.043 s dash and stab events give the same first moving step and displacement as the 0.05 s events they replace
#: (27, 1.118073, 2.455744 in both renders). Last active 43 is the 0.601 s total (the 0.533 asset sum gives 40); live the
#: slot clears at r+44 (request r = step 1 here, so step 44 is r+43; mechanics_072.md row 5f).
PIN_LAST_ACTIVE = 43
PIN_FIRST_MOVING = 27
PIN_AT_30 = 1.118073
PIN_AT_60 = 2.455744


def _thrust(profile) -> dict:
    return next(a for a in profile._iron_sword()["abilities"] if a["asset"] == THRUST)


def _profile(package: str):
    return import_module(package + ".combat.arsenal.profiles.hytale_0_5_7")


@contextmanager
def _mounted(out):
    """Mount a render for one test and drop it afterwards, so later tests can mount their own gen-7 render."""

    package = mount_fresh(out).__name__
    try:
        yield package
    finally:
        unmount(package)


@pytest.fixture(scope="module")
def gen7_render(tmp_path_factory):
    out = tmp_path_factory.mktemp("thrust_timing") / "gen7"
    out = render_for_test(GEN7, out, source=HeadSource())
    return out


@pytest.fixture(scope="module")
def gen5_render(tmp_path_factory):
    out = tmp_path_factory.mktemp("thrust_timing") / "gen5"
    out = render_for_test(GEN5, out, source=HeadSource())
    return out


def test_adapters_read_the_thrust_timing_on_both_tables() -> None:
    assert A.sword_thrust_timing(load("0.5.7")) == {"event_seconds": 0.05}
    assert A.sword_thrust_timing(load("0.7.2")) == {"event_seconds": 0.043}
    assert A.melee_hold_constants(load("0.5.7"))["sword_thrust_hold_speed_multiplier"] == 0.8
    assert A.melee_hold_constants(load("0.7.2"))["sword_thrust_hold_speed_multiplier"] == 0.75


def test_gen7_render_thrust_carries_the_072_timing(gen7_render) -> None:
    with _mounted(gen7_render) as package:
        thrust = _thrust(_profile(package))
    assert thrust["duration"] == 0.601
    assert [event["time"] for event in thrust["events"]] == [0.043, 0.043]
    assert thrust["hold_speed_multiplier"] == 0.75


def test_gen5_render_thrust_keeps_the_057_timing(gen5_render) -> None:
    with _mounted(gen5_render) as package:
        thrust = _thrust(_profile(package))
    assert thrust["duration"] == 0.601
    assert [event["time"] for event in thrust["events"]] == [0.05, 0.05]
    assert thrust["hold_speed_multiplier"] == 0.8


def _run_thrust(package: str) -> list[tuple[int, int, float]]:
    """(step, active slot, planar displacement from the start) for slot 3 requested on steps 1..30."""

    combat = import_module(package + ".combat")
    arsenal = import_module(package + ".combat.arsenal")
    environment = import_module(package + ".combat.arsenal.environment")
    params = combat.default_combat_params(microticks=1, target_active=False)
    loadout = arsenal.hytale_0_5_7_loadouts(["iron_sword"], target_profiles=["iron_sword"])
    config = arsenal.arsenal_runtime_config(loadout, specialize=False, sensor_range=params.sensor_range)
    position = np.zeros((1, 2, 3), np.float32)
    position[:, 1, 2] = -12.0  # the target stands well out of reach
    health = jnp.asarray([[params.agent_max_health, params.target_max_health]], jnp.float32)
    initial, _ = arsenal.reset_arsenal_batch(jax.random.split(jax.random.key(570057), 1), params, config,
                                             entity_position=jnp.asarray(position), entity_health=health,
                                             entity_yaw=jnp.zeros((1, 2), jnp.float32))
    world = environment.open_flat_arsenal_world_capabilities(initial, params, config=config)
    action = jnp.zeros((1, combat.ACTION_SIZE), jnp.float32)

    @jax.jit
    def step(state, slot, key):
        command = arsenal.empty_arsenal_commands(1)
        command = command._replace(ability_slot=command.ability_slot.at[:, 0].set(slot), world=world)
        return arsenal.step_arsenal_batch(state, action, key, params, command, config).state

    state = initial
    start = np.asarray(state.combat.position)[0, 0].copy()
    rows = []
    for tick in range(1, TICKS + 1):
        slot = THRUST_SLOT if tick <= REQUEST_UNTIL else -1
        state = step(state, jnp.int32(slot), jax.random.split(jax.random.key(tick), 1))
        pos = np.asarray(state.combat.position)[0, 0]
        active = int(np.asarray(state.arsenal.active_ability_slot)[0, 0])
        rows.append((tick, active, math.hypot(pos[0] - start[0], pos[2] - start[2])))
    return rows


def test_gen7_thrust_starts_on_step_23_and_moves_from_the_dash(gen7_render) -> None:
    with _mounted(gen7_render) as package:
        rows = _run_thrust(package)
    first_active = next(step for step, slot, _ in rows if slot == THRUST_SLOT)
    first_moving = next(step for step, _, distance in rows if distance > 1e-3)
    at_30 = rows[REQUEST_UNTIL - 1][2]
    at_60 = rows[-1][2]
    last_active = max(step for step, slot, _ in rows if slot == THRUST_SLOT)
    print(f"\nthrust first active step {first_active}, last active step {last_active}; first moving step {first_moving}; "
          f"planar displacement at step 30 {at_30:.6f}, at step 60 {at_60:.6f}")
    assert first_active == 23
    assert last_active == PIN_LAST_ACTIVE
    assert first_moving == PIN_FIRST_MOVING
    assert at_30 == pytest.approx(PIN_AT_30, abs=1e-5)
    assert at_60 == pytest.approx(PIN_AT_60, abs=1e-5)
