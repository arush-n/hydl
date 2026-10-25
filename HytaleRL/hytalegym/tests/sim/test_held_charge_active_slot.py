"""A held charge reports its slot as active from the press (row 5f; gen-7 Patch ``held_charge_reports_active_slot``).

Live (jar 3865F45F, ``gen7_probes/thrust_lunge/thrust_lunge_Kweebec_Razorleaf_jar14_{edge,hold10,hold30}.json``): the
sword's slot 3 reads active from the press tick until the swing or thrust ends, with no gap. The engine keeps a held
charge in ``ability_hold_slot`` with ``active_ability_slot`` -1 until the child is admitted, and hides the admitted child
for its start-queue tick. On gen 7 the observation now reports the held slot, and the charged child's slot on that tick.
Legality is untouched: the held slot stays legal while it is held, in both renders.

A short press releases into the sword's swing, which the engine admits as slot 0. Gen-7 Patches ``charge_release_root``
and ``charge_release_root_admitted`` record slot 3 as that child's root after its admission, so the swing reads 3 as live
does, while its cooldown, cost and stamina regen delay stay the swing's (live: no regen pause after a quick press).
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

THRUST_SLOT = 3
TICKS = 60


@contextmanager
def _mounted(out):
    package = mount_fresh(out).__name__
    try:
        yield package
    finally:
        unmount(package)


@pytest.fixture(scope="module")
def renders(tmp_path_factory):
    root = tmp_path_factory.mktemp("held_charge_active_slot")
    return {
        "gen7": render_for_test(GEN7, root / "gen7", source=HeadSource()),
        "gen5": render_for_test(GEN5, root / "gen5", source=HeadSource()),
    }


def _run(package: str, request_until: int) -> list[tuple[int, int, int, bool]]:
    """(step, state active slot, observed active slot, slot 3 legal) for slot 3 requested on steps 1..request_until."""

    combat = import_module(package + ".combat")
    arsenal = import_module(package + ".combat.arsenal")
    environment = import_module(package + ".combat.arsenal.environment")
    encoder = import_module(package + ".combat.observation.v3.encoding.encoder")
    params = combat.default_combat_params(microticks=1, target_active=False)
    loadout = arsenal.hytale_0_5_7_loadouts(["iron_sword"], target_profiles=["iron_sword"])
    config = arsenal.arsenal_runtime_config(loadout, specialize=False, sensor_range=params.sensor_range)
    position = np.zeros((1, 2, 3), np.float32)
    position[:, 1, 2] = -12.0  # the target stands well out of reach
    health = jnp.asarray([[params.agent_max_health, params.target_max_health]], jnp.float32)
    state, _ = arsenal.reset_arsenal_batch(jax.random.split(jax.random.key(570057), 1), params, config,
                                           entity_position=jnp.asarray(position), entity_health=health,
                                           entity_yaw=jnp.zeros((1, 2), jnp.float32))
    world = environment.open_flat_arsenal_world_capabilities(state, params, config=config)
    action = jnp.zeros((1, combat.ACTION_SIZE), jnp.float32)
    features = combat.empty_injected_world_features(1)

    @jax.jit
    def step(state, slot, key):
        command = arsenal.empty_arsenal_commands(1)
        command = command._replace(ability_slot=command.ability_slot.at[:, 0].set(slot), world=world)
        state = arsenal.step_arsenal_batch(state, action, key, params, command, config).state
        caps = environment.open_flat_arsenal_world_capabilities(state, params, config=config)
        evidence = encoder.project_learner_observation_v3_actor_evidence(state, params, features, caps, config)
        return state, evidence.weapon_i32[0, 0, 2], evidence.ability_legal[0, 0, THRUST_SLOT]

    rows = []
    for tick in range(1, TICKS + 1):
        slot = THRUST_SLOT if tick <= request_until else -1
        state, observed, legal = step(state, jnp.int32(slot), jax.random.split(jax.random.key(tick), 1))
        active = int(np.asarray(state.arsenal.active_ability_slot)[0, 0])
        rows.append((tick, active, int(observed), bool(legal)))
    return rows


def _end(rows) -> int:
    """The last step the state's active slot is the thrust slot (a held thrust; a short press runs the swing, slot 0)."""

    return max(step for step, active, _, _ in rows if active == THRUST_SLOT)


def test_gen7_reports_the_held_slot_through_a_full_thrust_with_no_gap(renders) -> None:
    with _mounted(renders["gen7"]) as package:
        rows = _run(package, 30)
    end = _end(rows)
    observed = [o for step, _, o, _ in rows if step <= end]
    assert observed == [THRUST_SLOT] * end, rows[: end + 1]
    assert all(o == -1 for step, _, o, _ in rows if step > end)


@pytest.mark.parametrize("request_until", [1, 10])
def test_gen7_reports_the_held_slot_on_every_held_tick(renders, request_until: int) -> None:
    with _mounted(renders["gen7"]) as package:
        rows = _run(package, request_until)
    assert [o for step, _, o, _ in rows if step <= request_until] == [THRUST_SLOT] * request_until


def test_gen7_short_press_swing_reads_the_charging_slot_as_live(renders) -> None:
    with _mounted(renders["gen7"]) as package:
        rows = _run(package, 1)
    busy = [o for step, active, o, _ in rows if step == 1 or active >= 0]
    assert busy == [THRUST_SLOT] * len(busy)


def test_gen7_keeps_the_held_slot_legal_while_it_is_held(renders) -> None:
    with _mounted(renders["gen7"]) as package:
        rows = _run(package, 30)
    admitted = min(step for step, active, _, _ in rows if active == THRUST_SLOT)
    assert all(legal for step, _, _, legal in rows if step < admitted)


def test_gen7_short_press_keeps_the_swings_stamina_regen(renders) -> None:
    """The quick press's swing keeps its own admission: no stamina regen delay (slot 3's -2 s applies to the thrust)."""

    with _mounted(renders["gen7"]) as package:
        delays = _delays(package, 1)
    assert all(d == 0.0 for d in delays)


def _delays(package: str, request_until: int) -> list[float]:
    """The agent's stamina regen delay after each step, for slot 3 requested on steps 1..request_until."""

    combat = import_module(package + ".combat")
    arsenal = import_module(package + ".combat.arsenal")
    environment = import_module(package + ".combat.arsenal.environment")
    params = combat.default_combat_params(microticks=1, target_active=False)
    loadout = arsenal.hytale_0_5_7_loadouts(["iron_sword"], target_profiles=["iron_sword"])
    config = arsenal.arsenal_runtime_config(loadout, specialize=False, sensor_range=params.sensor_range)
    position = np.zeros((1, 2, 3), np.float32)
    position[:, 1, 2] = -12.0
    health = jnp.asarray([[params.agent_max_health, params.target_max_health]], jnp.float32)
    state, _ = arsenal.reset_arsenal_batch(jax.random.split(jax.random.key(570057), 1), params, config,
                                           entity_position=jnp.asarray(position), entity_health=health,
                                           entity_yaw=jnp.zeros((1, 2), jnp.float32))
    action = jnp.zeros((1, combat.ACTION_SIZE), jnp.float32)

    @jax.jit
    def step(state, slot, key):
        world = environment.open_flat_arsenal_world_capabilities(state, params, config=config)
        command = arsenal.empty_arsenal_commands(1)
        command = command._replace(ability_slot=command.ability_slot.at[:, 0].set(slot), world=world)
        return arsenal.step_arsenal_batch(state, action, key, params, command, config).state

    delays = []
    for tick in range(1, TICKS + 1):
        slot = THRUST_SLOT if tick <= request_until else -1
        state = step(state, jnp.int32(slot), jax.random.split(jax.random.key(tick), 1))
        delays.append(float(np.asarray(state.mechanics.stamina_regen_delay_seconds)[0, 0]))
    return delays


def test_gen5_is_unchanged_the_held_slot_reads_inactive(renders) -> None:
    with _mounted(renders["gen5"]) as package:
        rows = _run(package, 30)
    admitted = min(step for step, active, _, _ in rows if active == THRUST_SLOT)
    assert admitted > 1
    assert all(o == -1 for step, _, o, _ in rows if step <= admitted)
