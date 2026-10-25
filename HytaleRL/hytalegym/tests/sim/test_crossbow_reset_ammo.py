"""Crossbow reset ammo (gen 7, FIDELITY section (ax)): the iron crossbow starts empty and the first press loads one bolt.

Live 0.7.2 (jar 3865F45F, slot 0, one Weapon_Arrow_Crude in hotbar slot 1; receipts crossbow_standard_slot0_jar14.json and
_press2.json): the press at tick 55 finds Ammo 0, the reload takes the arrow and raises Ammo to 1 at tick 91 (+36), and the
press at tick 105 fires (Ammo 1 -> 0). The arrow is seeded as the bow lane's arrow_rows.py does (hotbar slot 1, x1).

Measured on the gen-7 render (jitted replay, seed 570057, target 8.1553 blocks away): reset Ammo 0; the press at 55 routes to
the reload (root 0, active slot 4; live reads slot 0) and the arrow is removed by tick 85; Ammo 1 at tick 91. Without an
arrow the press does not load (Ammo 0 through tick 110). Gen 5 keeps Ammo 6 and fires on the press at 55.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen5.overlay import OVERLAY as GEN5
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7

from .support import mount_fresh, render_for_test, unmount

PRESS_ONE = 55
PRESS_TWO = 105
TICKS = 110
SLOT_PRIMARY = 0
LIVE_LOAD_TICK = 91
RESOURCE_AMMO = 5
AGENT_ENTITY = 0


@contextmanager
def _mounted(out):
    package = mount_fresh(out).__name__
    try:
        yield package
    finally:
        unmount(package)


@pytest.fixture(scope="module")
def renders(tmp_path_factory):
    root = tmp_path_factory.mktemp("crossbow_reset_ammo")
    return {
        "gen7": render_for_test(GEN7, root / "gen7", source=HeadSource()),
        "gen5": render_for_test(GEN5, root / "gen5", source=HeadSource()),
    }


def _run(package: str, presses: tuple[int, ...], arrow: bool = True) -> dict:
    """Step the iron crossbow for TICKS steps; slot 0 is requested on each press step only."""

    combat = import_module(package + ".combat")
    arsenal = import_module(package + ".combat.arsenal")
    environment = import_module(package + ".combat.arsenal.environment")
    from hytalegym.jax.combat.contracts.semantic import semantic_id
    from hytalegym.jax.combat.inventory import CONTAINER_HOTBAR

    params = combat.default_combat_params(microticks=1, target_active=False)
    loadout = arsenal.hytale_0_5_7_loadouts(["iron_crossbow"], target_profiles=["iron_sword"])
    config = arsenal.arsenal_runtime_config(loadout, specialize=False, sensor_range=params.sensor_range)
    position = np.zeros((1, 2, 3), np.float32)
    position[:, 1, 2] = -8.1553  # the live separation (agent z 6.155, target z -2.0)
    health = jnp.asarray([[params.agent_max_health, params.target_max_health]], jnp.float32)
    state, _ = arsenal.reset_arsenal_batch(jax.random.split(jax.random.key(570057), 1), params, config,
                                           entity_position=jnp.asarray(position), entity_health=health,
                                           entity_yaw=jnp.zeros((1, 2), jnp.float32))
    if arrow:
        # one Weapon_Arrow_Crude x1 in hotbar slot 1 (live hotbar_options), seeded as lane bow_fire's arrow_rows.py does
        layout = config.inventory_layout
        hot1 = (layout.container_id == jnp.int32(CONTAINER_HOTBAR)) & (layout.container_slot == jnp.int32(1))
        index = int(jnp.argmax(hot1.astype(jnp.int32)))
        inv = state.inventory
        state = state._replace(inventory=inv._replace(
            item_id=inv.item_id.at[:, :, index].set(jnp.int32(semantic_id("Weapon_Arrow_Crude"))),
            quantity=inv.quantity.at[:, :, index].set(jnp.int32(1))))
    world = environment.open_flat_arsenal_world_capabilities(state, params, config=config)
    action = jnp.zeros((1, combat.ACTION_SIZE), jnp.float32)

    @jax.jit
    def step(state, slot, key):
        command = arsenal.empty_arsenal_commands(1)
        command = command._replace(ability_slot=command.ability_slot.at[:, 0].set(slot), world=world)
        transition = arsenal.step_arsenal_batch(state, action, key, params, command, config)
        return transition.state, transition.arsenal_info

    trace = {
        "reset_ammo": float(np.asarray(state.mechanics.resources)[0, AGENT_ENTITY, RESOURCE_AMMO]),
        "ammo": [], "active_slot": [], "projectiles": [], "damage": [],
    }
    for tick in range(1, TICKS + 1):
        slot = SLOT_PRIMARY if tick in presses else -1
        state, info = step(state, jnp.int32(slot), jax.random.split(jax.random.key(tick), 1))
        trace["ammo"].append(float(np.asarray(state.mechanics.resources)[0, AGENT_ENTITY, RESOURCE_AMMO]))
        trace["active_slot"].append(int(np.asarray(state.arsenal.active_ability_slot)[0, 0]))
        trace["projectiles"].append(int(np.asarray(info.projectile_count)[0]))
        trace["damage"].append(float(np.asarray(info.damage_dealt)[0]))
    return trace


@pytest.fixture(scope="module")
def gen7_trace(renders):
    with _mounted(renders["gen7"]) as package:
        return _run(package, (PRESS_ONE, PRESS_TWO))


def test_gen7_reset_ammo_is_zero(gen7_trace) -> None:
    assert gen7_trace["reset_ammo"] == 0.0


def test_gen7_first_press_fires_nothing_and_routes_to_the_reload(gen7_trace) -> None:
    assert gen7_trace["projectiles"][PRESS_ONE - 1] == 0
    # Reported: the reload (slot 4) is active from the press; live reads slot 0 during the reload.
    assert gen7_trace["active_slot"][PRESS_ONE] == 4


def test_gen7_first_press_loads_one_bolt_at_the_live_tick(gen7_trace) -> None:
    load_tick = next(index + 1 for index, value in enumerate(gen7_trace["ammo"]) if value >= 1.0)
    print(f"\nSIM load tick (Ammo 0 -> 1): {load_tick}; live {LIVE_LOAD_TICK}; press {PRESS_ONE}")
    assert abs(load_tick - LIVE_LOAD_TICK) <= 1


def test_gen7_second_press_spawns_a_bolt_with_the_loaded_ammo(gen7_trace) -> None:
    # Measured: the second press (tick 105) is accepted and a projectile is in flight from tick 107 (live spawn 105).
    # Ammo is still 1 at tick 110 and the hit is after tick 110: the spend and the hit are not asserted here (open).
    ammo = gen7_trace["ammo"]
    assert ammo[PRESS_TWO - 2] == 1.0
    assert gen7_trace["projectiles"][PRESS_TWO] == 0
    assert gen7_trace["projectiles"][PRESS_TWO + 2] >= 1


def test_gen7_without_an_arrow_the_press_does_not_load(renders) -> None:
    with _mounted(renders["gen7"]) as package:
        trace = _run(package, (PRESS_ONE,), arrow=False)
    assert trace["ammo"] == [0.0] * TICKS
    assert sum(trace["projectiles"]) == 0


def test_gen5_keeps_six_bolts_and_fires_on_the_first_press(renders) -> None:
    with _mounted(renders["gen5"]) as package:
        trace = _run(package, (PRESS_ONE,))
    assert trace["reset_ammo"] == 6.0
    assert trace["projectiles"][PRESS_ONE] >= 1  # the shot is spawned on the tick after the press
    assert trace["ammo"][PRESS_ONE + 9] == 5.0  # Ammo 6 -> 5 by tick 65 (measured)
