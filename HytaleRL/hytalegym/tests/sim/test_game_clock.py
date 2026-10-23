"""Gen-7 game clock (S14; construct ``sim/gen7/mechanics/game_clock.py``, switch ``Constructs.game_clock``, default False).

The clock is a pure function of the tick count and the step, so the arithmetic is tested on arrays: the rate, the day and
night boundaries, time to next across midnight, and the live time_of_day convention. The render is tested twice, with the
switch off and on: the on learner observation gains one field, and its v3 fields and flat policy vector do not change.
"""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import HeadSource
from hytalegym.sim.common.profile import Constructs
from hytalegym.sim.gen5.profile import PROFILE as GEN5
from hytalegym.sim.gen7 import overlay as gen7_overlay
from hytalegym.sim.gen7.mechanics import game_clock
from hytalegym.sim.gen7.profile import PROFILE as GEN7

from .support import mount_fresh, unmount

STEP = 1.0 / 30.0
START = 19800  # 05:30, the live WorldConfig default
DAY = 86400
SUNRISE = 17280
NIGHTFALL = 69120
LIVE_START_TIME_OF_DAY = 5500  # 19800 / 86400 * 24000


def _at(tick: int):
    """The clock of one batch at ``tick`` ticks since reset."""

    return game_clock.clock_fields(jnp.asarray([tick], dtype=jnp.int32), STEP)


def _second(tick: int) -> int:
    return (START + tick) % DAY


# ---------------------------------------------------------------------------------------------------------------------
# The rate and the live time_of_day convention.
# ---------------------------------------------------------------------------------------------------------------------


def test_one_tick_is_one_game_second_at_the_gen7_step() -> None:
    assert float(_at(1).game_seconds_elapsed[0]) == pytest.approx(1.0, abs=1e-6)
    assert float(_at(30).game_seconds_elapsed[0]) == pytest.approx(30.0, abs=1e-4)  # one real second


def test_the_start_is_0530_and_the_live_time_of_day_is_5500() -> None:
    clock = _at(0)
    assert int(clock.time_of_day[0]) == LIVE_START_TIME_OF_DAY
    assert float(clock.game_seconds_elapsed[0]) == 0.0


def test_time_of_day_advances_1000_units_per_game_hour_over_N_ticks() -> None:
    # 3600 ticks are one game hour at 30 Hz: 1000 units of the live 24000-unit day.
    for tick in (0, 3600, 7200):
        expected = round(_second(tick) * 24000 / DAY)
        assert int(_at(tick).time_of_day[0]) == expected
    assert int(_at(3600).time_of_day[0]) - int(_at(0).time_of_day[0]) == 1000


def test_one_game_day_of_ticks_returns_the_clock_to_the_start() -> None:
    # 2880 real seconds at 30 Hz = 86400 ticks = one game day (WorldTimeResource: 86400 / 2880 = 30 game s per real s).
    clock = _at(DAY)
    assert int(clock.time_of_day[0]) == LIVE_START_TIME_OF_DAY
    assert float(clock.game_seconds_elapsed[0]) == pytest.approx(DAY, abs=1e-2)


def test_a_batch_of_ticks_is_computed_per_row() -> None:
    ticks = jnp.asarray([0, 49320, 83880], dtype=jnp.int32)  # 05:30 (day), 19:12 (nightfall), 04:48 (sunrise)
    clock = game_clock.clock_fields(ticks, STEP)
    assert clock.time_of_day.shape == (3,)
    assert clock.is_day.tolist() == [True, False, True]


# ---------------------------------------------------------------------------------------------------------------------
# The day and night boundaries, time to next, and the weather reserve.
# ---------------------------------------------------------------------------------------------------------------------


def test_the_day_starts_at_the_authored_sunrise() -> None:
    assert _second(83879) == SUNRISE - 1 and _second(83880) == SUNRISE
    assert not bool(_at(83879).is_day[0])
    assert bool(_at(83880).is_day[0])


def test_the_night_starts_at_the_authored_nightfall() -> None:
    assert _second(49319) == NIGHTFALL - 1 and _second(49320) == NIGHTFALL
    assert bool(_at(49319).is_day[0])
    assert not bool(_at(49320).is_day[0])


def test_time_to_the_next_sunrise_and_nightfall_across_midnight() -> None:
    late = _at(66599)  # second 86399, one second before midnight
    assert _second(66599) == DAY - 1
    assert float(late.seconds_to_sunrise[0]) == pytest.approx(SUNRISE + 1)
    assert float(late.seconds_to_nightfall[0]) == pytest.approx(NIGHTFALL + 1)
    midnight = _at(66600)  # second 0
    assert _second(66600) == 0
    assert float(midnight.seconds_to_sunrise[0]) == pytest.approx(SUNRISE)
    assert float(midnight.seconds_to_nightfall[0]) == pytest.approx(NIGHTFALL)


def test_seconds_to_the_next_event_are_strictly_after_now() -> None:
    assert float(_at(83880).seconds_to_sunrise[0]) == pytest.approx(DAY)  # at sunrise itself
    assert float(_at(49320).seconds_to_nightfall[0]) == pytest.approx(DAY)  # at nightfall itself


def test_weather_is_published_as_unknown() -> None:
    clock = game_clock.clock_fields(jnp.asarray([0, 5], dtype=jnp.int32), STEP)
    assert clock.weather_index.tolist() == [game_clock.WEATHER_UNKNOWN] * 2
    assert clock.weather_known.tolist() == [False, False]


def test_the_authored_constants_match_the_0_7_2_sources() -> None:
    # Default.json World: 1728 + 1152 real seconds; WorldTimeResource: sunrise = NIGHTTIME_SECONDS / 2 = 17280.
    assert game_clock.GAME_SECONDS_PER_REAL_SECOND == 30.0
    assert game_clock.SUNRISE_SECONDS == SUNRISE
    assert game_clock.NIGHTFALL_SECONDS == NIGHTFALL
    assert game_clock.START_SECONDS_OF_DAY == START


# ---------------------------------------------------------------------------------------------------------------------
# The switch and the render.
# ---------------------------------------------------------------------------------------------------------------------


def test_the_switch_is_false_in_the_defaults_and_in_both_profiles() -> None:
    assert Constructs().game_clock is False
    assert GEN5.constructs.game_clock is False
    assert GEN7.constructs.game_clock is False


def test_the_construct_returns_the_original_when_the_switch_is_off() -> None:
    def original(*args, **kwargs):
        return args, kwargs

    assert game_clock.construct(original, GEN7) is original


def test_a_non_bool_switch_fails_closed() -> None:
    with pytest.raises(ValueError, match="game_clock"):
        game_clock.construct(lambda *a, **k: None, SimpleNamespace(constructs=SimpleNamespace(game_clock="yes")))


class _Engine:
    """One mounted gen-7 render: the agent at the origin, the target 2.5 blocks away, the iron-sword loadout."""

    def __init__(self, package: str) -> None:
        self.package = package
        self.combat = importlib.import_module(package + ".combat")
        self.arsenal = importlib.import_module(package + ".combat.arsenal")
        self.encoder = importlib.import_module(package + ".combat.observation.v3.encoding.encoder")
        self.layout = importlib.import_module(package + ".combat.observation.v3.policy.layout")
        self.params = self.combat.default_combat_params(microticks=1, target_active=False)
        loadout = self.arsenal.hytale_0_5_7_loadouts("iron_sword", target_profiles="iron_sword")
        self.config = self.arsenal.arsenal_runtime_config(loadout, specialize=False, sensor_range=self.params.sensor_range)

    def reset(self):
        position = jnp.asarray([[[0.0, 0.0, 0.0], [0.0, 0.0, -2.5]]], dtype=jnp.float32)
        health = jnp.asarray([[self.params.agent_max_health, self.params.target_max_health]], dtype=jnp.float32)
        state, _ = self.arsenal.reset_arsenal_batch(
            jax.random.split(jax.random.key(570057), 1),
            self.params,
            self.config,
            entity_position=position,
            entity_health=health,
            entity_yaw=jnp.zeros((1, 2), dtype=jnp.float32),
        )
        return state

    def observe(self, state):
        features = self.combat.empty_injected_world_features(1)
        caps = self.arsenal.environment.open_flat_arsenal_world_capabilities(state, self.params, config=self.config)
        return self.encoder.encode_learner_observation_v3(state, self.params, features, caps, self.config)


@pytest.fixture(scope="module")
def engines(tmp_path_factory):
    """The gen-7 render with the clock off (the committed default) and with it on, each mounted under its own name."""

    name = "game_clock_on_profile"
    shim = types.ModuleType(name)
    shim.PROFILE = replace(GEN7, constructs=replace(GEN7.constructs, game_clock=True))
    sys.modules[name] = shim
    try:
        off_dir = tmp_path_factory.mktemp("render_off") / "gen7"
        on_dir = tmp_path_factory.mktemp("render_on") / "gen7"
        off_manifest = R.render(gen7_overlay.OVERLAY, off_dir, source=HeadSource())
        on_manifest = R.render(replace(gen7_overlay.OVERLAY, profile=f"{name}:PROFILE"), on_dir, source=HeadSource())
        off = _Engine(mount_fresh(off_dir).__name__)
        on = _Engine(mount_fresh(on_dir).__name__)
        yield SimpleNamespace(off=off, on=on, off_manifest=off_manifest, on_manifest=on_manifest)
        unmount(off.package)
        unmount(on.package)
    finally:
        sys.modules.pop(name, None)


def test_the_render_installs_the_hook_in_both_renders_and_the_off_one_is_the_original(engines) -> None:
    # The hook row is active in both renders; the construct returns the original when the switch is off (checked by
    # test_the_switch_off_observation_has_no_clock_field), as perception_v2's row does.
    for manifest in (engines.off_manifest, engines.on_manifest):
        applied = [entry for entry in manifest["applied"] if entry["id"] == "game_clock"]
        assert len(applied) == 1
        assert applied[0]["construct"] == "hytalegym.sim.gen7.mechanics.game_clock:construct"
        assert applied[0]["function"] == "encode_learner_observation_v3"


def test_the_switch_off_observation_has_no_clock_field(engines) -> None:
    observation = engines.off.observe(engines.off.reset())
    assert type(observation).__name__ == "LearnerCombatObservationV3"
    assert not hasattr(observation, "game_clock")


def test_the_switch_on_adds_one_field_last_and_keeps_every_v3_field(engines) -> None:
    off = engines.off.observe(engines.off.reset())
    on = engines.on.observe(engines.on.reset())
    assert on._fields == off._fields + ("game_clock",)
    for name in off._fields:
        left = jax.tree_util.tree_leaves(getattr(on, name))
        right = jax.tree_util.tree_leaves(getattr(off, name))
        assert len(left) == len(right), name
        for a, b in zip(left, right):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)


def test_the_rendered_clock_reads_the_state_tick_count(engines) -> None:
    state = engines.on.reset()
    assert int(state.combat.tick_count[0]) == 0
    at_nightfall = state._replace(combat=state.combat._replace(tick_count=jnp.asarray([49320], dtype=jnp.int32)))
    clock = engines.on.observe(at_nightfall).game_clock
    assert clock.is_day.tolist() == [False]
    assert int(clock.time_of_day[0]) == round(NIGHTFALL * 24000 / DAY)
    assert float(clock.game_seconds_elapsed[0]) == pytest.approx(49320.0)


def test_the_flat_policy_vector_is_unchanged_by_the_switch(engines) -> None:
    off_flat = engines.off.layout.arsenal_policy_observation(engines.off.observe(engines.off.reset()))
    on_flat = engines.on.layout.arsenal_policy_observation(engines.on.observe(engines.on.reset()))
    assert off_flat.shape[1] == engines.off.layout.ARSENAL_POLICY_OBSERVATION_SIZE
    np.testing.assert_array_equal(np.asarray(on_flat), np.asarray(off_flat))
