"""The S14 game clock on the live learner observation (opt-in ``HytaleEnv(learner_game_clock=True)``).

The bridge publishes ``time_of_day`` each step (0..24000, ``NativeEnvironmentSession``: round(dayProgress x 24000)); jar
``3865F45F`` and later run the clock. This turns it into the simulator's ``GameClock`` (``sim/gen7/mechanics/game_clock.py``,
built by the same ``clock_at``), appended last as ``game_clock``, so a consumer reads live as it reads the gen-7 sim.

Live differences:
- ``time_of_day`` is the bridge's value, not derived from a tick count.
- ``game_seconds_elapsed`` is the wrap-aware change in time_of_day since reset, x 3.6 game seconds a unit, so it moves in
  3.6 s steps (about 3.6 ticks at 30 game seconds per real second). A paused clock leaves it at 0.
- Weather is unknown, as in the sim.
"""

from __future__ import annotations

import jax.numpy as jnp

from hytalegym.sim.gen7.mechanics.game_clock import (
    GAME_SECONDS_PER_DAY,
    TIME_OF_DAY_UNITS,
    GameClock,
    clock_at,
    observation_type,
)

SECONDS_PER_UNIT = GAME_SECONDS_PER_DAY / TIME_OF_DAY_UNITS  # 3.6 game seconds per time_of_day unit


class LiveGameClock:
    """One live environment's clock: the last time_of_day reading and the game seconds since reset."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last: int | None = None
        self._elapsed = 0.0

    def update(self, time_of_day: int) -> GameClock:
        """The clock for this step's ``time_of_day``; advances the elapsed seconds by the change since the last step."""

        if isinstance(time_of_day, bool) or not 0 <= int(time_of_day) <= TIME_OF_DAY_UNITS:
            raise ValueError(f"time_of_day must be in 0..{TIME_OF_DAY_UNITS}, got {time_of_day!r}")
        value = int(time_of_day)
        if self._last is not None:
            self._elapsed += ((value - self._last) % TIME_OF_DAY_UNITS) * SECONDS_PER_UNIT
        self._last = value
        return clock_at(
            jnp.asarray([value * SECONDS_PER_UNIT], dtype=jnp.float32),
            jnp.asarray([value], dtype=jnp.int32),
            jnp.asarray([self._elapsed], dtype=jnp.float32),
        )


def with_game_clock(observation, clock: GameClock):
    """``observation`` with ``clock`` appended as its last field, ``game_clock`` (the sim construct's type)."""

    return observation_type(type(observation))(*observation, clock)
