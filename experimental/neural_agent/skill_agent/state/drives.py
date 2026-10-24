"""Sleep pressure and hunger on the game clock (``docs/VIRTUAL_HUMAN.md`` sections 16.1, 16.5, 20.2 and 20.6).

Two drives, each in [0, 1], that the state layer keeps and publishes as the variables ``sleep_pressure`` and ``hunger``
(``contract/variables.py``, group "drives"). The body reads neither: the decision layer and the LLM do, and their effect on
the body is a regulator, future (16.1, 19.7).

- **Sleep pressure S** is process S of the two-process model (Daan, Beersma and Borbely 1984). It rises toward 1 with a
  time constant of ``SLEEP_RISE_HOURS`` while awake, and falls with ``SLEEP_FALL_HOURS`` while asleep, the agent's
  movement state ``sleeping``. A clock jump counts the same way: a jump while asleep is a skipped night.
- **Hunger H** rises with game time (``HUNGER_PER_GAME_HOUR``) and with activity, as a game's exhaustion does
  (Minecraft's weights, 20.6). Each meal, an ``ate`` event (16.5), lowers it by ``FOOD_SATISFACTION``.

The game time is the clock the simulator publishes on its learner observation (``game_clock``, S14): the elapsed game
seconds when the channel has them, else the live ``time_of_day`` (24,000 units a game day). Without a clock the elapsed
time is 0, and both drives hold still, activity included (16.1). A reset (an episode boundary) restores S = 0 and H = 0,
since a new world starts at 05:30, after the night, rested and fed. A respawn keeps both (16.2).

The constants marked [E] are estimates for the owner to rule on. Each is a named module constant.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.state import action_history, events

NAMES: tuple[str, ...] = ("sleep_pressure", "hunger")
"""The variables the drives publish (``contract/variables.py``), in the state's order."""
SLEEP_RISE_HOURS: float = 18.2
"""Awake: S rises toward 1 with this time constant, in game hours (Daan, Beersma and Borbely 1984; 16.1)."""
SLEEP_FALL_HOURS: float = 4.2
"""Asleep: S falls with this time constant, in game hours (Daan, Beersma and Borbely 1984; 16.1)."""
HUNGER_PER_GAME_HOUR: float = 0.5 / 24
"""[E] Hunger rising per game hour: a full game day idle makes the agent about half hungry, so a day of play sends it
looking for food once or twice (16.5)."""
EXHAUSTION_TO_HUNGER: float = 1.0 / 80
"""[E] Hunger per point of exhaustion. Minecraft: 4.0 exhaustion costs one point of its 20-point bar (Minecraft Wiki
"Hunger", read 2026-10-22, recalled; saturation ignored; 20.6)."""
FOOD_SATISFACTION: float = 0.25
"""[E] Hunger a meal lowers by. Minecraft's bread restores 5 of 20 points (recalled). The item registry has no food tier,
so every meal counts the same, and a consumable that is not food (a potion) counts as a meal too (open, not fixed)."""
SPRINT_EXHAUSTION: float = 0.1
"""Exhaustion per block sprinted (Minecraft Wiki "Hunger", read 2026-10-22; 20.6)."""
SWIM_EXHAUSTION: float = 0.01
"""Exhaustion per block swum (Minecraft Wiki "Hunger"; 20.6)."""
JUMP_EXHAUSTION: float = 0.05
"""Exhaustion per jump (Minecraft Wiki "Hunger"; its 0.2 for a jump while sprinting is left out, 20.6)."""
HIT_EXHAUSTION: float = 0.1
"""Exhaustion per hit dealt or taken (Minecraft Wiki "Hunger"; 20.6)."""
LIVE_UNITS_PER_DAY: float = 24000.0
"""``time_of_day`` runs over 24,000 units a game day (the live convention, ``game_clock.py``)."""
GAME_SECONDS_PER_LIVE_UNIT: float = 86400.0 / LIVE_UNITS_PER_DAY
"""Game seconds in one ``time_of_day`` unit: 3.6."""
UNITS_PER_GAME_HOUR: float = 1000.0
"""``time_of_day`` units in one game hour."""
SUNRISE_UNITS: float = 4800.0
"""Sunrise in ``time_of_day`` units: 17280 game seconds of 86400 (``game_clock.py``, ``SUNRISE_SECONDS``)."""
NIGHTFALL_UNITS: float = 19200.0
"""Nightfall in ``time_of_day`` units: 69120 game seconds of 86400 (``game_clock.py``, ``NIGHTFALL_SECONDS``)."""
_JUMP_HEAD: int = [head.name for head in action_history.HEADS].index("jump_off_on")
_JUMP_ON: int = 1  # the policy's choice for a jump (``hytalegym`` policy/surface.py: ``factor["jump_off_on"] == 1``)


class DriveState(NamedTuple):
    """The drives of B agents, and the clock's last reading (the elapsed time is the change from it)."""

    sleep_pressure: jax.Array  # float32[B]: S, in [0, 1]
    hunger: jax.Array  # float32[B]: H, in [0, 1]
    seconds: jax.Array  # float32[B]: the last game_seconds_elapsed reading (0 before one)
    time_of_day: jax.Array  # float32[B]: the last time_of_day reading (0 before one)
    has_clock: jax.Array  # bool[B]: the last tick carried a clock reading


class Activity(NamedTuple):
    """This tick's activity, each float32[B]: the blocks sprinted and swum, the jumps, the hits dealt and taken, and the
    meals (see :func:`activity`)."""

    blocks_sprinted: jax.Array
    blocks_swum: jax.Array
    jumps: jax.Array
    hits: jax.Array
    meals: jax.Array


def initial(batch: int) -> DriveState:
    """Rested and fed (S and H 0), with no clock reading yet."""

    zero = jnp.zeros((batch,), jnp.float32)
    return DriveState(zero, zero, zero, zero, jnp.zeros((batch,), jnp.bool_))


def elapsed(state: DriveState, clock: Any) -> jax.Array:
    """float32[B]: the game seconds since the last tick (16.1). The change in ``game_seconds_elapsed`` when the clock has
    it (0 if it went down); else the change in ``time_of_day`` taken as the way round the day,
    ``((tod - last) mod 24000) * 3.6``. 0 on a tick with no clock reading before it."""

    seconds = getattr(clock, "game_seconds_elapsed", None)
    if seconds is not None:
        change = jnp.maximum(jnp.asarray(seconds, jnp.float32) - state.seconds, 0.0)
    else:
        units = jnp.mod(jnp.asarray(clock.time_of_day, jnp.float32) - state.time_of_day, LIVE_UNITS_PER_DAY)
        change = units * GAME_SECONDS_PER_LIVE_UNIT
    return jnp.where(state.has_clock, change, 0.0)


def _count(log: events.EventLog, now: jax.Array, kinds: tuple[str, ...]) -> jax.Array:
    """float32[B]: the events of ``kinds`` that the log stamped this tick."""

    codes = jnp.asarray([events.EVENT_KINDS.index(kind) for kind in kinds], jnp.int32)
    stamped = jnp.isin(log.kind, codes) & (log.tick == now[:, None])
    return jnp.sum(stamped, axis=1).astype(jnp.float32)


def activity(logged: events.EventState, before: jax.Array, history: action_history.ActionHistory,
             reset: jax.Array | None = None) -> Activity:
    """This tick's activity, after the event log (``state/events.py``). ``logged`` is the log after this tick; ``before``
    float32[B, len(COUNTERS)] the counters before it. Blocks sprinted and swum are this tick's increments of those
    counters (from zero on a reset, whose counters start over). A jump is a head of the action history that changed to
    the jump choice this tick (one per press; none on a reset). Hits are this tick's damage events, dealt or taken, and
    meals its ``ate`` events."""

    base = before if reset is None else jnp.where(reset[:, None], 0.0, before)
    grown = logged.counters - base
    jumped = history.changed[:, _JUMP_HEAD] & (history.last_choice[:, _JUMP_HEAD] == _JUMP_ON)
    if reset is not None:
        jumped = jumped & ~reset
    return Activity(
        blocks_sprinted=grown[:, events.COUNTERS.index("distance_sprinted")],
        blocks_swum=grown[:, events.COUNTERS.index("distance_swum")],
        jumps=jumped.astype(jnp.float32),
        hits=_count(logged.log, logged.now, ("damage_dealt", "damage_taken")),
        meals=_count(logged.log, logged.now, ("ate",)),
    )


def update(state: DriveState, clock: Any, *, asleep: jax.Array, activity: Activity,
           reset: jax.Array | None = None) -> DriveState:
    """The drives after this tick (the module docstring). ``clock``: the observation's ``game_clock``, or None when there
    is none, and then nothing moves. ``asleep`` bool[B]: the self's ``sleeping`` movement state. ``reset`` bool[B]: the
    lanes that start an episode, which restore S = 0 and H = 0 and count no elapsed time on their first tick."""

    batch = state.sleep_pressure.shape[0]
    if reset is not None:
        state = state._replace(sleep_pressure=jnp.where(reset, 0.0, state.sleep_pressure),
                               hunger=jnp.where(reset, 0.0, state.hunger), has_clock=state.has_clock & ~reset)
    if clock is None:
        return state._replace(has_clock=jnp.zeros((batch,), jnp.bool_))
    hours = elapsed(state, clock) / 3600.0
    # 1 - (1 - S) * exp(-h / tau), written as S + (1 - S) * (1 - exp(-h / tau)): the same value, without the cancellation
    # that loses the digits of a small S in float32 (expm1 keeps 1 - exp(-h / tau) exact for small h).
    awake = state.sleep_pressure + (1.0 - state.sleep_pressure) * -jnp.expm1(-hours / SLEEP_RISE_HOURS)
    asleep_now = state.sleep_pressure * jnp.exp(-hours / SLEEP_FALL_HOURS)
    sleep = jnp.clip(jnp.where(asleep, asleep_now, awake), 0.0, 1.0)
    exhaustion = (SPRINT_EXHAUSTION * activity.blocks_sprinted + SWIM_EXHAUSTION * activity.blocks_swum
                  + JUMP_EXHAUSTION * activity.jumps + HIT_EXHAUSTION * activity.hits)
    hunger = jnp.clip(state.hunger + HUNGER_PER_GAME_HOUR * hours + EXHAUSTION_TO_HUNGER * exhaustion
                      - FOOD_SATISFACTION * activity.meals, 0.0, 1.0)
    seconds = jnp.broadcast_to(jnp.asarray(getattr(clock, "game_seconds_elapsed", state.seconds), jnp.float32), (batch,))
    time_of_day = jnp.broadcast_to(jnp.asarray(clock.time_of_day, jnp.float32), (batch,))
    return DriveState(sleep, hunger, seconds, time_of_day, jnp.ones((batch,), jnp.bool_))


def is_day(time_of_day: float) -> bool:
    """Day from a ``time_of_day`` reading: from sunrise to nightfall, as the simulator's ``is_day``."""

    return SUNRISE_UNITS <= float(time_of_day) < NIGHTFALL_UNITS


__all__ = ["Activity", "DriveState", "NAMES", "UNITS_PER_GAME_HOUR", "activity", "elapsed", "initial", "is_day", "update"]
