"""The three learner-encoder constructs on together: ``perception_v2``, ``game_clock`` and ``hearing``.

Each hooks ``encode_learner_observation_v3`` and appends one field. ``perception_v2`` finds the rendered package by
unwrapping the encoder it was given (``_rendered_root``), so every wrapper it can receive must keep ``__wrapped__``
(reported by the skill layer, 2026-10-24: ``--construct perception_v2 --construct game_clock`` raised before the first
tick). The render here turns all three on and encodes one observation.
"""

from __future__ import annotations

import sys
import types
from dataclasses import replace
from importlib import import_module

import jax
import jax.numpy as jnp
import pytest

from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.gen7 import overlay as gen7_overlay
from hytalegym.sim.gen7.profile import PROFILE as GEN7

from .support import mount_fresh, unmount

ADDED = ("perception_v2", "game_clock", "heard_sounds")


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    name = "encoder_construct_chain_profile"
    shim = types.ModuleType(name)
    shim.PROFILE = replace(
        GEN7, constructs=replace(GEN7.constructs, perception_v2=True, game_clock=True, hearing=True)
    )
    sys.modules[name] = shim
    try:
        target = tmp_path_factory.mktemp("render_chain") / "gen7"
        R.render(replace(gen7_overlay.OVERLAY, profile=f"{name}:PROFILE"), target, source=HeadSource())
        mounted = mount_fresh(target).__name__
        yield mounted
        unmount(mounted)
    finally:
        sys.modules.pop(name, None)


def test_all_three_encoder_constructs_encode_together(package) -> None:
    combat = import_module(package + ".combat")
    arsenal = import_module(package + ".combat.arsenal")
    encoder = import_module(package + ".combat.observation.v3.encoding.encoder")
    params = combat.default_combat_params(microticks=1, target_active=False)
    loadout = arsenal.hytale_0_5_7_loadouts("iron_sword", target_profiles="iron_sword")
    config = arsenal.arsenal_runtime_config(loadout, specialize=False, sensor_range=params.sensor_range)
    state, _ = arsenal.reset_arsenal_batch(
        jax.random.split(jax.random.key(570057), 1),
        params,
        config,
        entity_position=jnp.asarray([[[0.0, 0.0, 0.0], [0.0, 0.0, -2.5]]], dtype=jnp.float32),
        entity_health=jnp.asarray([[params.agent_max_health, params.target_max_health]], dtype=jnp.float32),
        entity_yaw=jnp.zeros((1, 2), dtype=jnp.float32),
    )
    caps = arsenal.environment.open_flat_arsenal_world_capabilities(state, params, config=config)

    observation = encoder.encode_learner_observation_v3(
        state, params, combat.empty_injected_world_features(1), caps, config
    )

    assert sorted(observation._fields[-3:]) == sorted(ADDED)
    assert len(set(observation._fields)) == len(observation._fields)
    assert int(observation.game_clock.time_of_day[0]) == 5500
    assert observation.perception_v2 is not None
    assert observation.heard_sounds is not None
