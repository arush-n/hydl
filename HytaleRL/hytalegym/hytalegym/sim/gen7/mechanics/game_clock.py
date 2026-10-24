"""Gen-7 game clock (S14): time of day, time to the next sunrise and nightfall, elapsed game seconds, weather (unknown).

Switch ``Constructs.game_clock``, default False. Hook: ``encode_learner_observation_v3`` (sim/gen7/overlay.py row
``game_clock``). When the switch is on the learner observation carries one more field, ``game_clock``, after the fields it
already has (``perception_v2`` when that is on). The flat policy vector and the contract hashes do not read it.

Grounding (DESIGN.md in the lane folder; CONFIRMED = a named source or bytecode line):
- 30 game seconds per real second: ``WorldTimeResource.getSecondsPerTick`` = 86400 / (day + night); Default.json World
  DaytimeDurationSeconds 1728 + NighttimeDurationSeconds 1152 = 2880 real seconds per day.
- Sunrise 17280 s and day length 51840 game s (= 1728 real s at 30 x): ``WorldTimeResource`` static init (SUNRISE_SECONDS,
  DAYTIME_SECONDS). Nightfall = sunrise + day = 69120 s.
- Start 05:30 (19800 s): ``WorldConfig.<init>`` gameTime = ZERO_YEAR + 5 h + 30 min.
- time_of_day: live convention, ``NativeEnvironmentSession`` timeOfDay = round(getDayProgress * 24000), clamped.
- One tick = one game second at the gen-7 step (1/30 s), from ``profile.step_duration.nominal_seconds`` x 30.
- Weather is not modelled: the 0.7.2 pick depends on the full LocalDateTime sampled at the first tick of each game hour
  (WeatherSystem$TickingSystem), so it is published as unknown (weather_index -1, weather_known False).
- Live: NativeEnvironmentSession pauses the game clock (setGameTimePaused(true), line 2826). Owner 2026-10-23: live
  unpaused in the next jar; until then live reads 05:30. The sim runs at the 0.7.2 rate from 05:30 regardless.
"""

from __future__ import annotations

from functools import lru_cache, wraps
from typing import Any, NamedTuple

import jax.numpy as jnp

SWITCH = "game_clock"
IMPLEMENTED = True

#: Game seconds per real second (0.7.2: 86400 / 2880, WorldTimeResource.getSecondsPerTick).
GAME_SECONDS_PER_REAL_SECOND = 30.0
GAME_SECONDS_PER_DAY = 86400
#: The live time_of_day scale: 24000 units per game day (NativeEnvironmentSession, round(dayProgress * 24000)).
TIME_OF_DAY_UNITS = 24000
#: Start of a new world: 05:30 (WorldConfig.<init>, ZERO_YEAR + 5 h 30 min).
START_SECONDS_OF_DAY = 19800
#: Sunrise (WorldTimeResource static init: NIGHTTIME_SECONDS / 2 = 34560 / 2).
SUNRISE_SECONDS = 17280
#: Day length: Default.json World.DaytimeDurationSeconds (1728 real s) at 30 game s per real second = 51840 game s.
DAY_SECONDS = 1728 * 30
NIGHTFALL_SECONDS = SUNRISE_SECONDS + DAY_SECONDS
#: Weather is reserved: the 0.7.2 pick is not modelled (see the module docstring).
WEATHER_UNKNOWN = -1


class GameClock(NamedTuple):
    """The clock's arrays, each [B]. Seconds are game seconds."""

    time_of_day: Any  # int32: live convention, 0..24000
    is_day: Any  # bool: sunrise <= second_of_day < nightfall
    seconds_to_sunrise: Any  # float32: to the next sunrise, strictly after now
    seconds_to_nightfall: Any  # float32: to the next nightfall, strictly after now
    game_seconds_elapsed: Any  # float32: since reset (tick_count x game seconds per tick)
    weather_index: Any  # int32: WEATHER_UNKNOWN
    weather_known: Any  # bool: False


def _seconds_until(target: float, second_of_day):
    """Game seconds from ``second_of_day`` to the next ``target`` strictly after it (a full day when it is due now)."""

    delta = jnp.mod(jnp.float32(target) - second_of_day, jnp.float32(GAME_SECONDS_PER_DAY))
    return jnp.where(delta <= 0.0, jnp.float32(GAME_SECONDS_PER_DAY), delta)


def clock_at(second, time_of_day, elapsed) -> GameClock:
    """The clock from the second of the day (float32 [B]), time_of_day (int32 [B]) and game seconds since reset
    (float32 [B]). The simulator derives all three from the tick count (``clock_fields``); live reads time_of_day from the
    bridge (``hytalegym/envs/live_game_clock.py``)."""

    batch = second.shape[0]
    return GameClock(
        time_of_day=time_of_day,
        is_day=(second >= SUNRISE_SECONDS) & (second < NIGHTFALL_SECONDS),
        seconds_to_sunrise=_seconds_until(SUNRISE_SECONDS, second),
        seconds_to_nightfall=_seconds_until(NIGHTFALL_SECONDS, second),
        game_seconds_elapsed=elapsed,
        weather_index=jnp.full((batch,), WEATHER_UNKNOWN, dtype=jnp.int32),
        weather_known=jnp.zeros((batch,), dtype=jnp.bool_),
    )


def clock_fields(tick_count, step_seconds: float) -> GameClock:
    """The clock of one batch from its tick counts (``state.combat.tick_count``, int [B]) and the step (seconds)."""

    per_tick = jnp.float32(step_seconds * GAME_SECONDS_PER_REAL_SECOND)
    elapsed = jnp.asarray(tick_count).astype(jnp.float32) * per_tick
    second = jnp.mod(jnp.float32(START_SECONDS_OF_DAY) + elapsed, jnp.float32(GAME_SECONDS_PER_DAY))
    time_of_day = jnp.clip(
        jnp.round(second * jnp.float32(TIME_OF_DAY_UNITS / GAME_SECONDS_PER_DAY)),
        0,
        TIME_OF_DAY_UNITS,
    ).astype(jnp.int32)
    return clock_at(second, time_of_day, elapsed)


@lru_cache(maxsize=None)
def observation_type(base: type) -> type:
    """``base`` with one more field, ``game_clock``, last. One class per observation class."""

    fields = [(name, Any) for name in base._fields] + [("game_clock", Any)]
    return NamedTuple(f"{base.__name__}WithGameClock", fields)


def construct(original, profile):
    """Return ``original`` when the switch is off; the encoder with the clock field when it is on (the render's hook)."""

    value = profile.constructs.game_clock
    if value is False:
        return original
    if value is not True:
        raise ValueError(f"game_clock must be a bool, got {value!r}")
    step_seconds = float(profile.step_duration.nominal_seconds)

    @wraps(original)  # perception_v2 unwraps the encoder it receives to find the rendered package
    def encode_with_game_clock(state, params, world_features, world_capabilities, config, *, world_geometry=None):
        observation = original(
            state, params, world_features, world_capabilities, config, world_geometry=world_geometry
        )
        clock = clock_fields(state.combat.tick_count, step_seconds)
        return observation_type(type(observation))(*observation, clock)

    return encode_with_game_clock
