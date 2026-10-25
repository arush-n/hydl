"""The learner pays a native NPC instant stat cost once on gen 7; other entities pay twice (row 8f; gen-7 Patches
``learner_cost_single_application`` and ``learner_cost_single_application_observed``; sim/gen7/mechanics/learner_cost.py).

Live (jar 3865F45F, ``gen7_probes/thrust_lunge/thrust_lunge_Kweebec_Razorleaf_jar14_hold30.json``): the iron sword's
thrust, authored 4 stamina (kind 3, native NPC instant add, break immune), takes the learner from 10 to 6. The bridge leases
remote-client mode for its policy-controlled chains, so 0.7.2 skips the simulation pass; an autonomous NPC runs both
passes and pays 8. Gen 5 keeps two applications for every entity.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen5.overlay import OVERLAY as GEN5
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7

from .support import mount_fresh, unmount

THRUST_SLOT = 3
SWING_SLOT = 0
HOLD_TICKS = 30
TICKS = 45
AUTHORED = 4.0


@contextmanager
def _mounted(out):
    package = mount_fresh(out).__name__
    try:
        yield package
    finally:
        unmount(package)


@pytest.fixture(scope="module")
def renders(tmp_path_factory):
    root = tmp_path_factory.mktemp("learner_cost")
    R.render(GEN7, root / "gen7", source=HeadSource())
    R.render(GEN5, root / "gen5", source=HeadSource())
    return root


def _setup(package: str):
    combat = import_module(package + ".combat")
    arsenal = import_module(package + ".combat.arsenal")
    environment = import_module(package + ".combat.arsenal.environment")
    encoder = import_module(package + ".combat.observation.v3.encoding.encoder")
    params = combat.default_combat_params(microticks=1, target_active=False)
    loadout = arsenal.hytale_0_5_7_loadouts(["iron_sword"], target_profiles=["iron_sword"])
    config = arsenal.arsenal_runtime_config(loadout, specialize=False, sensor_range=params.sensor_range)
    position = np.zeros((1, 2, 3), np.float32)
    position[:, 1, 2] = -12.0  # out of reach: no hit lands
    health = jnp.asarray([[params.agent_max_health, params.target_max_health]], jnp.float32)
    state, _ = arsenal.reset_arsenal_batch(jax.random.split(jax.random.key(570057), 1), params, config,
                                           entity_position=jnp.asarray(position), entity_health=health,
                                           entity_yaw=jnp.asarray([[0.0, 180.0]], jnp.float32))
    return combat, arsenal, environment, encoder, params, config, state


def _min_stamina(package: str, entity: int) -> float:
    """The lowest stamina of ``entity`` while it holds the thrust slot for HOLD_TICKS ticks from a full bar."""

    combat, arsenal, environment, _, params, config, state = _setup(package)
    action = jnp.zeros((1, combat.ACTION_SIZE), jnp.float32)

    @jax.jit
    def step(state, slot, key):
        world = environment.open_flat_arsenal_world_capabilities(state, params, config=config)
        command = arsenal.empty_arsenal_commands(1)
        command = command._replace(ability_slot=command.ability_slot.at[:, entity].set(slot), world=world)
        return arsenal.step_arsenal_batch(state, action, key, params, command, config).state

    lowest = float(state.mechanics.resources[0, entity, 0])
    assert lowest == pytest.approx(10.0)
    for tick in range(1, TICKS + 1):
        slot = THRUST_SLOT if tick <= HOLD_TICKS else -1
        state = step(state, jnp.int32(slot), jax.random.split(jax.random.key(tick), 1))
        lowest = min(lowest, float(state.mechanics.resources[0, entity, 0]))
    return lowest


def _observed_cost_row(package: str) -> np.ndarray:
    """The learner's ability_f32 rows (slot x feature) at reset."""

    combat, _, environment, encoder, params, config, state = _setup(package)
    caps = environment.open_flat_arsenal_world_capabilities(state, params, config=config)
    evidence = encoder.project_learner_observation_v3_actor_evidence(
        state, params, combat.empty_injected_world_features(1), caps, config)
    return np.asarray(evidence.ability_f32)[0, 0]


def test_the_learner_pays_the_thrust_once_on_gen7_and_twice_on_gen5(renders):
    with _mounted(renders / "gen7") as package:
        assert _min_stamina(package, 0) == pytest.approx(10.0 - AUTHORED)
    with _mounted(renders / "gen5") as package:
        assert _min_stamina(package, 0) == pytest.approx(10.0 - 2 * AUTHORED)


def test_an_npc_entity_still_pays_twice_on_gen7(renders):
    with _mounted(renders / "gen7") as package:
        assert _min_stamina(package, 1) == pytest.approx(10.0 - 2 * AUTHORED)


def test_the_learner_observes_the_cost_it_pays(renders):
    with _mounted(renders / "gen7") as package:
        gen7 = _observed_cost_row(package)
    with _mounted(renders / "gen5") as package:
        gen5 = _observed_cost_row(package)
    assert np.array_equal(gen7[SWING_SLOT], gen5[SWING_SLOT])  # the swing costs nothing on either line
    changed = np.flatnonzero(gen7[THRUST_SLOT] != gen5[THRUST_SLOT])
    assert changed.size == 1  # only the cost column
    column = int(changed[0])
    assert gen7[THRUST_SLOT, column] == pytest.approx(gen5[THRUST_SLOT, column] / 2.0)
