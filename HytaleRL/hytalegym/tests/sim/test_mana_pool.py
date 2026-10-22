"""Mana pool construct (gen 7, row 7a, phase 4b): gen 5 keeps the factory; gen 7 reads 100 / 100 and +1 per 0.3 s."""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.jax.combat.mechanics import factory
from hytalegym.jax.combat.types import AGENT_ENTITY
from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import mana_pool
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY
from hytalegym.sim.gen7.profile import PROFILE as GEN7

from .support import mount_fresh, unmount


def test_gen5_returns_the_original_factory() -> None:
    assert mana_pool.construct(factory.default_mechanics_rules, GEN5) is factory.default_mechanics_rules


def test_gen7_mana_maximum_and_regeneration_come_from_the_0_7_2_table() -> None:
    rules = mana_pool.construct(factory.default_mechanics_rules, GEN7)(1)
    mana = factory.RESOURCE_MANA
    assert float(rules.resource_maximum[0, 0, mana]) == 100.0
    assert float(rules.resource_regen_amount[0, 0, mana]) == pytest.approx(1.0)
    assert float(rules.resource_regen_interval_seconds[0, 0, mana]) == pytest.approx(0.3, rel=1e-6)
    # Other resources keep the gen-5 factory values (stamina, magic charges and their intervals).
    assert float(rules.resource_regen_interval_seconds[0, 0, factory.RESOURCE_MAGIC_CHARGES]) == pytest.approx(2.0)


def test_gen7_caller_maximum_is_respected() -> None:
    rules = mana_pool.construct(factory.default_mechanics_rules, GEN7)(1, mana_maximum=37.0)
    assert float(rules.resource_maximum[0, 0, factory.RESOURCE_MANA]) == 37.0


# ---------------------------------------------------------------------------------------------------------------------
# Task #107 item 2 (2026-10-22): the reset. The loadout carried no mana, so ``arsenal_runtime_config`` rebuilt the rules with
# maximum 0 and the reset read 0 / 0 on the single-agent env and the duel's reset path. ``construct_loadout`` fills the stat on
# the loadout; the render below reads the reset on both paths.
# ---------------------------------------------------------------------------------------------------------------------

MANA = 1  # RESOURCE_MANA (jax/combat/mechanics/schema/contract.py)
PROFILE_LOADOUT = "iron_sword"


@pytest.fixture(scope="module")
def render(tmp_path_factory):
    target = tmp_path_factory.mktemp("mana_pool_render") / "gen7"
    R.render(GEN7_OVERLAY, target, source=HeadSource())
    package = mount_fresh(target).__name__
    yield package
    unmount(package)


def _modules(package: str) -> SimpleNamespace:
    return SimpleNamespace(
        combat=import_module(package + ".combat"),
        arsenal=import_module(package + ".combat.arsenal"),
        environment=import_module(package + ".combat.arsenal.environment"),
        runtime=import_module(package + ".combat.arsenal.runtime"),
    )


def test_gen7_single_agent_reset_reads_mana_100_for_the_agent_and_the_target(render) -> None:
    m = _modules(render)
    params = m.combat.default_combat_params(microticks=1)
    config = m.arsenal.arsenal_runtime_config(m.arsenal.hytale_0_5_7_loadouts([PROFILE_LOADOUT], target_profiles=[PROFILE_LOADOUT]))
    env = m.environment.make_arsenal_environment(params, config, maximum_turn_degrees=45.0)
    carry, observation = env.reset(jax.random.split(jax.random.key(0), 1))
    assert np.asarray(carry.runtime.mechanics.resources)[0, :, MANA].tolist() == [100.0, 100.0]
    assert np.asarray(config.mechanics_rules.resource_maximum)[0, :, MANA].tolist() == [100.0, 100.0]
    assert float(np.asarray(observation.resource_f32)[0, AGENT_ENTITY, MANA]) == 1.0
    assert bool(np.asarray(observation.resource_mask)[0, AGENT_ENTITY, MANA])


def test_gen7_duel_reset_path_reads_mana_100_for_both_entities(render) -> None:
    # The duel's reset (experimental/neural_agent/tasks/duel_arena.py make_duel_arena) takes the same calls: the config from
    # hytale_0_5_7_loadouts, arsenal_runtime_config with the legal observation, and reset_arsenal_batch with the bodies placed.
    m = _modules(render)
    params = m.combat.default_combat_params(microticks=1)._replace(target_active=jnp.asarray(False, dtype=jnp.bool_))
    config = m.arsenal.arsenal_runtime_config(
        m.arsenal.hytale_0_5_7_loadouts([PROFILE_LOADOUT], target_profiles=[PROFILE_LOADOUT]),
        opponent_controller_mask=jnp.zeros((1, 2), dtype=jnp.bool_),
        legal_observation=True,
    )
    position = jnp.asarray([[[0.0, 0.0, 0.0], [0.0, 0.0, -2.5]]], jnp.float32)
    health = jnp.asarray([[params.agent_max_health, params.target_max_health]], jnp.float32)
    state, _ = m.runtime.reset_arsenal_batch(
        jax.random.split(jax.random.key(0), 1), params, config,
        entity_position=position, entity_health=health, entity_yaw=jnp.zeros((1, 2), jnp.float32),
    )
    assert np.asarray(state.mechanics.resources)[0, :, MANA].tolist() == [100.0, 100.0]


def test_a_loadout_that_names_its_own_mana_maximum_keeps_it(render) -> None:
    m = _modules(render)
    loadout = m.arsenal.hytale_0_5_7_loadouts([PROFILE_LOADOUT], target_profiles=[PROFILE_LOADOUT])
    loadout = loadout._replace(
        resource_maximum=loadout.resource_maximum.at[..., MANA].set(37.0),
        resource_initial=loadout.resource_initial.at[..., MANA].set(20.0),
    )
    config = m.arsenal.arsenal_runtime_config(loadout)
    assert np.asarray(config.mechanics_rules.resource_maximum)[0, :, MANA].tolist() == [37.0, 37.0]
    assert np.asarray(config.loadout.resource_initial)[0, :, MANA].tolist() == [20.0, 20.0]


def test_gen5_loadout_hook_is_the_original_and_gen5_reads_no_mana() -> None:
    from hytalegym.jax.combat.arsenal import arsenal_runtime_config, hytale_0_5_7_loadouts

    def original(*args, **kwargs):
        return args, kwargs

    assert mana_pool.construct_loadout(original, GEN5) is original
    config = arsenal_runtime_config(hytale_0_5_7_loadouts([PROFILE_LOADOUT], target_profiles=[PROFILE_LOADOUT]))
    assert np.asarray(config.mechanics_rules.resource_maximum)[0, :, MANA].tolist() == [0.0, 0.0]
