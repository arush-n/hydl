"""Mana pool construct (gen 7, row 7a, phase 4b): gen 5 keeps the factory; gen 7 reads 100 / 100 and +1 per 0.3 s."""

from __future__ import annotations

import pytest

from hytalegym.jax.combat.mechanics import factory
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import mana_pool
from hytalegym.sim.gen7.profile import PROFILE as GEN7


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
