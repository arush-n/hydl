"""Head yaw at reset (gen 7, FIDELITY section (ak), row 5g; construct ``sim/gen7/mechanics/reset_head_heading.py``, switch
``Constructs.reset_head_heading``).

Both renders come from committed HEAD (``HeadSource``); the gen-7 overlay carries the Hook and the gen-5 overlay does not. The
reset is the entity-position path of ``reset_arsenal_batch`` for the fixture's pair: the agent at the origin and the target
2.5 blocks ahead at ``(0, 0, -2.5)``, with the body yaws given by each test (a target facing the agent is 180 degrees).
"""

from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.common.profile import Constructs
from hytalegym.sim.gen5.overlay import OVERLAY as GEN5_OVERLAY
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7.mechanics import reset_head_heading
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7_OVERLAY
from hytalegym.sim.gen7.profile import PROFILE as GEN7

from .support import mount_fresh, render_for_test, unmount

SEED = 570057
LOADOUT = "iron_sword"


class _Engine:
    """One render's Arsenal reset for the iron sword pair."""

    def __init__(self, package: str) -> None:
        self.package = package
        self.combat = import_module(package + ".combat")
        self.arsenal = import_module(package + ".combat.arsenal")
        self.observe = import_module(package + ".combat.runtime.reset").observe_batch
        self.params = self.combat.default_combat_params(microticks=1, target_active=False)
        self.config = self.arsenal.arsenal_runtime_config(
            self.arsenal.hytale_0_5_7_loadouts((LOADOUT,)),
            specialize=False,
            sensor_range=self.params.sensor_range,
        )

    def reset(self, entity_yaw):
        position = jnp.asarray([[[0.0, 0.0, 0.0], [0.0, 0.0, -2.5]]], dtype=jnp.float32)
        health = jnp.asarray([[self.params.agent_max_health, self.params.target_max_health]], dtype=jnp.float32)
        yaw = None if entity_yaw is None else jnp.asarray([entity_yaw], dtype=jnp.float32)
        return self.arsenal.reset_arsenal_batch(
            jax.random.split(jax.random.key(SEED), 1),
            self.params,
            self.config,
            entity_position=position,
            entity_health=health,
            entity_yaw=yaw,
        )


@pytest.fixture(scope="module")
def engines(tmp_path_factory):
    built = {}
    for name, overlay in (("gen7", GEN7_OVERLAY), ("gen5", GEN5_OVERLAY)):
        target = tmp_path_factory.mktemp("render") / name
        target = render_for_test(overlay, target, source=HeadSource())
        built[name] = _Engine(mount_fresh(target).__name__)
    yield built
    for engine in built.values():
        unmount(engine.package)


# ---------------------------------------------------------------------------------------------------------------------
# Switch plumbing: gen 5 returns the original, an unknown value fails closed, no entity_yaw returns the original result.
# ---------------------------------------------------------------------------------------------------------------------


def test_switch_defaults_name_gen5_none_and_gen7_body_heading() -> None:
    assert Constructs().reset_head_heading == "none"
    assert GEN5.constructs.reset_head_heading == "none"
    assert GEN7.constructs.reset_head_heading == "body_heading"


def test_gen5_returns_the_original_reset() -> None:
    def original(*args, **kwargs):
        return args, kwargs

    assert reset_head_heading.construct(original, GEN5) is original


def test_an_unknown_switch_value_fails_closed() -> None:
    profile = SimpleNamespace(constructs=SimpleNamespace(reset_head_heading="bogus"))
    with pytest.raises(ValueError, match="reset_head_heading"):
        reset_head_heading.construct(lambda *args, **kwargs: None, profile)


def test_without_entity_yaw_the_wrapper_returns_the_original_result() -> None:
    sentinel = (object(), object())
    wrapper = reset_head_heading.construct(lambda *args, **kwargs: sentinel, GEN7)
    assert wrapper(None, None, None, entity_position=object(), entity_yaw=None) is sentinel
    assert wrapper(None, None, None) is sentinel
    assert wrapper(None, None, None, entity_yaw=object()) is sentinel


# ---------------------------------------------------------------------------------------------------------------------
# The reset on the two renders.
# ---------------------------------------------------------------------------------------------------------------------


def test_gen5_keeps_both_heads_at_zero_for_a_target_facing_the_agent(engines) -> None:
    state, _ = engines["gen5"].reset([0.0, 180.0])
    combat = state.combat
    assert float(combat.agent_head_yaw[0]) == 0.0
    assert float(combat.target_head_yaw[0]) == 0.0
    assert np.asarray(combat.yaw)[0].tolist() == [0.0, 180.0]


def test_gen7_seeds_the_target_head_on_its_body_heading_and_leaves_the_pitch(engines) -> None:
    state, _ = engines["gen7"].reset([0.0, 180.0])
    combat = state.combat
    assert float(combat.agent_head_yaw[0]) == 0.0
    assert float(combat.target_head_yaw[0]) == 180.0
    assert np.asarray(combat.yaw)[0].tolist() == [0.0, 180.0]
    assert float(combat.agent_head_pitch[0]) == 0.0
    assert float(combat.target_head_pitch[0]) == 0.0


@pytest.mark.parametrize("entity_yaw", [[45.0, 180.0], [-90.0, 90.0]])
def test_gen7_seeds_each_head_on_its_own_body_yaw(engines, entity_yaw) -> None:
    state, _ = engines["gen7"].reset(entity_yaw)
    assert float(state.combat.agent_head_yaw[0]) == entity_yaw[0]
    assert float(state.combat.target_head_yaw[0]) == entity_yaw[1]


def test_the_returned_observation_agrees_with_the_state(engines) -> None:
    engine = engines["gen7"]
    state, observation = engine.reset([0.0, 180.0])
    np.testing.assert_array_equal(np.asarray(observation), np.asarray(engine.observe(state.combat, engine.params)))


def test_without_entity_yaw_gen7_matches_gen5(engines) -> None:
    gen5_state, gen5_observation = engines["gen5"].reset(None)
    gen7_state, gen7_observation = engines["gen7"].reset(None)
    for name in ("agent_head_yaw", "target_head_yaw", "agent_head_pitch", "target_head_pitch", "yaw"):
        np.testing.assert_array_equal(
            np.asarray(getattr(gen7_state.combat, name)), np.asarray(getattr(gen5_state.combat, name)), err_msg=name
        )
    np.testing.assert_array_equal(np.asarray(gen7_observation), np.asarray(gen5_observation))
