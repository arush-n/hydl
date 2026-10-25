"""Sleep pressure and hunger on the game clock (``docs/VIRTUAL_HUMAN.md`` sections 16.1, 16.5, 20.2, 20.6 and 20.9).

Two drives, each in [0, 1], that the state layer keeps and publishes as the variables ``sleep_pressure`` and ``hunger``
(``contract/variables.py``, group "drives"). The body reads neither: the decision layer and the LLM do, and their effect on
the body is a regulator, future (16.1, 19.7).

- **Sleep pressure S** is process S of the two-process model (Daan, Beersma and Borbely 1984). It rises toward 1 with a
  time constant of ``SLEEP_RISE_HOURS`` while awake, and falls with ``SLEEP_FALL_HOURS`` while asleep, the agent's
  movement state ``sleeping``. A clock jump counts the same way: a jump while asleep is a skipped night. It needs a clock:
  without one it holds still.
- **Hunger** is Minecraft Java Edition's food model (the Minecraft Wiki's "Hunger" and "Food" pages, read 2026-10-24;
  20.9). The state is ``food`` (0 to 20 points), ``saturation`` (0 to food: the points that drain first) and
  ``exhaustion`` (0 to 4: activity not yet paid for). Activity adds exhaustion: 0.1 a block sprinted, 0.01 a block swum,
  0.05 a jump (0.2 for a jump on a sprinting tick), 0.1 a hit dealt or taken, nothing for walking. Each whole 4.0 takes a
  point, from the saturation first, then the food. A meal (an ``ate`` event, 16.5) restores its item's food points and
  then its saturation (``FOODS``). Nothing drains hunger over time, as in Minecraft. The published variable ``hunger`` is
  ``1 - food / 20``: 0 fed, 1 starving. It moves without a clock, since it needs none.

The game time is the clock the simulator publishes on its learner observation (``game_clock``, S14): the elapsed game
seconds when the channel has them, else the live ``time_of_day`` (24,000 units a game day). Without a clock the elapsed
time is 0, so sleep pressure holds still while hunger follows the activity. A reset (an episode boundary) restores S = 0,
food 20, saturation 5 and exhaustion 0, the start of a Minecraft world, since a new world starts at 05:30, after the
night, rested and fed. A respawn keeps all of them (16.2). This differs from Minecraft, which restores food and
saturation on respawn: the owner's ruling for this agent, "hunger carries on" (16.2).

Not modelled (the regulators, future, 16.1, or no game event yet):
- natural regeneration's exhaustion (6.0 per HP healed), since Hytale heals without hunger;
- block breaking (0.005 per block), since no break events exist yet;
- the Hunger status effect;
- starvation, and losing the sprint at 6 food or below: effects on the body.

The constants marked [E] are estimates for the owner to rule on. Each is a named module constant. The choice of a Minecraft
food for each Hytale food is an estimate too (Hytale foods have no hunger values); the numbers are Minecraft's.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from experimental.neural_agent.skill_agent.state import action_history, events
from hytalegym.jax.combat.contracts.semantic import semantic_id

NAMES: tuple[str, ...] = ("sleep_pressure", "hunger")
"""The variables the drives publish (``contract/variables.py``), in the state's order."""
SLEEP_RISE_HOURS: float = 18.2
"""Awake: S rises toward 1 with this time constant, in game hours (Daan, Beersma and Borbely 1984; 16.1)."""
SLEEP_FALL_HOURS: float = 4.2
"""Asleep: S falls with this time constant, in game hours (Daan, Beersma and Borbely 1984; 16.1)."""
FOOD_MAX: float = 20.0
"""The food bar's points (Minecraft Wiki "Hunger", read 2026-10-24); ``hunger`` is ``1 - food / FOOD_MAX``."""
FOOD_START: float = 20.0
"""A new world starts with a full food bar (Minecraft Wiki "Hunger", read 2026-10-24)."""
SATURATION_START: float = 5.0
"""A new world starts with 5 saturation (Minecraft Wiki "Hunger", read 2026-10-24)."""
EXHAUSTION_PER_POINT: float = 4.0
"""Exhaustion that takes one point, from the saturation first, then the food (Minecraft Wiki "Hunger", read 2026-10-24)."""
SPRINT_EXHAUSTION: float = 0.1
"""Exhaustion per block sprinted (Minecraft Wiki "Hunger", read 2026-10-24; 20.9)."""
SWIM_EXHAUSTION: float = 0.01
"""Exhaustion per block swum (Minecraft Wiki "Hunger", read 2026-10-24; 20.9)."""
JUMP_EXHAUSTION: float = 0.05
"""Exhaustion per jump (Minecraft Wiki "Hunger", read 2026-10-24; 20.9)."""
SPRINT_JUMP_EXHAUSTION: float = 0.2
"""Exhaustion per jump on a tick with a sprinted block (Minecraft Wiki "Hunger", read 2026-10-24; 20.9)."""
HIT_EXHAUSTION: float = 0.1
"""Exhaustion per hit dealt or taken (Minecraft Wiki "Hunger", read 2026-10-24; 20.9)."""
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
FOODS: dict[str, tuple[int, float, str]] = {
    # Hytale asset id: (food points, saturation, the Minecraft food it follows). Every value is Minecraft Java's (Minecraft
    # Wiki "Food", read 2026-10-24); the choice of Minecraft food is our estimate [E], since Hytale foods have no values.
    "Food_Bread": (5, 6.0, "bread"),
    "Food_Cheese": (5, 6.0, "bread"),  # a staple; Minecraft has no cheese
    "Food_Fish_Grilled": (5, 6.0, "cooked cod"),
    "Food_Wildmeat_Cooked": (6, 9.6, "cooked mutton"),
    "Food_Vegetable_Cooked": (5, 6.0, "baked potato"),
    "Food_Kebab_Meat": (8, 12.8, "steak"),
    "Food_Kebab_Mushroom": (6, 7.2, "mushroom stew"),
    "Food_Kebab_Vegetable": (6, 7.2, "beetroot soup"),
    "Food_Kebab_Fruit": (4, 2.4, "apple"),
    "Food_Pie_Apple": (8, 4.8, "pumpkin pie"),
    "Food_Pie_Meat": (8, 4.8, "pumpkin pie"),
    "Food_Pie_Pumpkin": (8, 4.8, "pumpkin pie"),
    "Food_Salad_Berry": (6, 7.2, "mushroom stew"),
    "Food_Salad_Caesar": (6, 7.2, "mushroom stew"),
    "Food_Salad_Mushroom": (6, 7.2, "mushroom stew"),
    "Food_Popcorn": (2, 0.4, "cookie"),
    "Food_Candy_Cane": (2, 0.4, "cookie"),
    "Food_Beef_Raw": (3, 1.8, "raw beef"),
    "Food_Wildmeat_Raw": (3, 1.8, "raw beef"),
    "Food_Pork_Raw": (3, 1.8, "raw porkchop"),
    "Food_Chicken_Raw": (2, 1.2, "raw chicken"),
    "Food_Egg": (2, 1.2, "raw chicken"),  # Minecraft eggs are not eaten
    "Food_Fish_Raw": (2, 0.4, "raw cod"),
    "Food_Fish_Raw_Uncommon": (2, 0.4, "raw cod"),
    "Food_Fish_Raw_Rare": (2, 0.4, "raw cod"),
    "Food_Fish_Raw_Epic": (2, 0.4, "raw cod"),
    "Food_Fish_Raw_Legendary": (2, 0.4, "raw cod"),
    "Plant_Crop_Carrot_Item": (3, 3.6, "carrot"),
    "Plant_Crop_Potato_Item": (1, 0.6, "potato"),
    "Plant_Crop_Aubergine_Item": (3, 3.6, "carrot"),
    "Plant_Crop_Cauliflower_Item": (3, 3.6, "carrot"),
    "Plant_Crop_Lettuce_Item": (3, 3.6, "carrot"),
    "Plant_Crop_Onion_Item": (3, 3.6, "carrot"),
    "Plant_Crop_Tomato_Item": (3, 3.6, "carrot"),
    "Plant_Crop_Turnip_Item": (3, 3.6, "carrot"),
    "Plant_Crop_Corn_Item": (3, 3.6, "carrot"),
    "Plant_Crop_Chilli_Item": (2, 1.2, "melon slice"),  # a small raw bite (our estimate)
    "Plant_Crop_Rice_Item": (2, 1.2, "melon slice"),  # a small raw bite (our estimate)
    "Plant_Crop_Pumpkin_Item": (3, 3.6, "carrot"),  # a raw vegetable, as the others
    "Plant_Fruit_Apple": (4, 2.4, "apple"),
    "Plant_Fruit_Azure": (4, 2.4, "apple"),
    "Plant_Fruit_Coconut": (4, 2.4, "apple"),
    "Plant_Fruit_Mango": (4, 2.4, "apple"),
    "Plant_Fruit_Spiral": (4, 2.4, "apple"),
    "Plant_Fruit_Windwillow": (4, 2.4, "apple"),
    "Plant_Fruit_Berries_Red": (2, 0.4, "sweet berries"),
    "Plant_Fruit_Pinkberry": (2, 0.4, "sweet berries"),
    # Minecraft's poisonous foods still feed (the poisonous potato: 2, 1.2, with a 60% chance of Poison I for 5 s), and
    # Hytale's poison fruit heals on eating it (Template_Fruit's Food_Instant_Heal_T1); its poison is not modelled.
    "Plant_Fruit_Poison": (2, 1.2, "poisonous potato"),
}
"""The items that restore food when eaten (16.5, 20.9). An item not here restores nothing."""
_GLOWING = "a consumable that applies Poison, not food (Minecraft's raw mushrooms are not eaten)"
NOT_FOOD: dict[str, str] = {
    # Hytale asset id: why it is not in FOODS.
    "Plant_Crop_Mushroom_Glowing_Blue": _GLOWING,
    "Plant_Crop_Mushroom_Glowing_Green": _GLOWING,
    "Plant_Crop_Mushroom_Glowing_Orange": _GLOWING,
    "Plant_Crop_Mushroom_Glowing_Purple": _GLOWING,
    "Plant_Crop_Mushroom_Glowing_Red": _GLOWING,
    "Plant_Crop_Mushroom_Glowing_Violet": _GLOWING,
    "Plant_Crop_Cotton_Item": "an ingredient with no Consumable flag in its item file or parent; not eaten (Minecraft has "
                              "no cotton food)",
    "Plant_Crop_Wheat_Item": "an ingredient with Consumable false in its item file; not eaten (Minecraft's wheat is not "
                            "eaten)",
}
"""The game's food-like items that restore no food, with the reason (the table covers every ``Food_`` item, every
``Plant_Fruit_`` item and every ``Plant_Crop_*_Item``: the test in ``tests/.../test_sa_drives.py``)."""
_FOOD_KEYS = jnp.asarray([semantic_id(asset) for asset in FOODS], jnp.int32)
"""The semantic ids of ``FOODS``' items, in its order: the ``ate`` events' ``detail`` is matched against these."""
_FOOD_POINTS = jnp.asarray([food for food, _, _ in FOODS.values()], jnp.float32)
_FOOD_SATURATION = jnp.asarray([saturation for _, saturation, _ in FOODS.values()], jnp.float32)
_JUMP_HEAD: int = [head.name for head in action_history.HEADS].index("jump_off_on")
_JUMP_ON: int = 1  # the policy's choice for a jump (``hytalegym`` policy/surface.py: ``factor["jump_off_on"] == 1``)
_EAT: int = events.EVENT_KINDS.index("ate")


class DriveState(NamedTuple):
    """The drives of B agents, and the clock's last reading (the elapsed time is the change from it)."""

    sleep_pressure: jax.Array  # float32[B]: S, in [0, 1]
    food: jax.Array  # float32[B]: food points, in [0, 20]
    saturation: jax.Array  # float32[B]: saturation points, in [0, food]
    exhaustion: jax.Array  # float32[B]: exhaustion, in [0, 4)
    seconds: jax.Array  # float32[B]: the last game_seconds_elapsed reading (0 before one)
    time_of_day: jax.Array  # float32[B]: the last time_of_day reading (0 before one)
    has_clock: jax.Array  # bool[B]: the last tick carried a clock reading

    @property
    def hunger(self) -> jax.Array:
        """float32[B]: H, the published variable: 0 fed, 1 starving (``1 - food / FOOD_MAX``)."""

        return 1.0 - self.food / FOOD_MAX


class Activity(NamedTuple):
    """This tick's activity: the blocks sprinted and swum, the jumps and the hits dealt and taken (each float32[B]), and the
    meals, the ``ate`` events in the order the log holds them (``meal_items`` int32[B, E] the items' semantic ids, and
    ``meal_counts`` float32[B, E] the counts eaten; 0 where there is no meal). See :func:`activity`."""

    blocks_sprinted: jax.Array
    blocks_swum: jax.Array
    jumps: jax.Array
    hits: jax.Array
    meal_items: jax.Array
    meal_counts: jax.Array


def initial(batch: int) -> DriveState:
    """Rested and fed (S 0, food 20, saturation 5, exhaustion 0), with no clock reading yet."""

    zero = jnp.zeros((batch,), jnp.float32)
    return DriveState(zero, jnp.full((batch,), FOOD_START, jnp.float32),
                      jnp.full((batch,), SATURATION_START, jnp.float32), zero, zero, zero,
                      jnp.zeros((batch,), jnp.bool_))


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
    the jump choice this tick (one per press; none on a reset). Hits are this tick's damage events, dealt or taken. Meals
    are this tick's ``ate`` events, oldest first (the ring's order, from ``events.recent``)."""

    base = before if reset is None else jnp.where(reset[:, None], 0.0, before)
    grown = logged.counters - base
    jumped = history.changed[:, _JUMP_HEAD] & (history.last_choice[:, _JUMP_HEAD] == _JUMP_ON)
    if reset is not None:
        jumped = jumped & ~reset
    ring = events.recent(logged, events.EVENT_SLOTS)
    eaten = (ring.kind == _EAT) & (ring.tick == logged.now[:, None])
    return Activity(
        blocks_sprinted=grown[:, events.COUNTERS.index("distance_sprinted")],
        blocks_swum=grown[:, events.COUNTERS.index("distance_swum")],
        jumps=jumped.astype(jnp.float32),
        hits=_count(logged.log, logged.now, ("damage_dealt", "damage_taken")),
        meal_items=jnp.where(eaten, ring.detail, 0),
        meal_counts=jnp.where(eaten, ring.amount, 0.0),
    )


def _meals(food: jax.Array, saturation: jax.Array, activity: Activity) -> tuple[jax.Array, jax.Array]:
    """Each meal in order (16.5, 20.9): its item's food points times the count, capped at 20 food, then its saturation
    times the count, capped at the food. An item not in ``FOODS`` restores nothing."""

    match = activity.meal_items[:, :, None] == _FOOD_KEYS[None, None, :]  # [B, E, K]: the item is in the table
    points = jnp.sum(jnp.where(match, _FOOD_POINTS, 0.0), axis=2)  # [B, E]
    fill = jnp.sum(jnp.where(match, _FOOD_SATURATION, 0.0), axis=2)

    def meal(carry, eaten):
        food, saturation = carry
        count, food_points, saturation_points = eaten
        food = jnp.minimum(FOOD_MAX, food + count * food_points)
        return (food, jnp.minimum(food, saturation + count * saturation_points)), None

    (food, saturation), _ = jax.lax.scan(meal, (food, saturation),
                                         (activity.meal_counts.T, points.T, fill.T))
    return food, saturation


def _eat_and_burn(state: DriveState, activity: Activity) -> tuple[jax.Array, jax.Array, jax.Array]:
    """food, saturation and exhaustion after this tick: this tick's exhaustion is added, each whole 4.0 takes a point (the
    saturation first, then the food), and then the meals restore."""

    cost = (SPRINT_EXHAUSTION * activity.blocks_sprinted + SWIM_EXHAUSTION * activity.blocks_swum
            + activity.jumps * jnp.where(activity.blocks_sprinted > 0.0, SPRINT_JUMP_EXHAUSTION, JUMP_EXHAUSTION)
            + HIT_EXHAUSTION * activity.hits)
    exhaustion = state.exhaustion + cost
    points = jnp.floor(exhaustion / EXHAUSTION_PER_POINT)
    from_saturation = jnp.minimum(points, state.saturation)
    saturation = state.saturation - from_saturation
    food = jnp.maximum(state.food - (points - from_saturation), 0.0)
    exhaustion = exhaustion - points * EXHAUSTION_PER_POINT
    food, saturation = _meals(food, saturation, activity)
    return food, saturation, exhaustion


def update(state: DriveState, clock: Any, *, asleep: jax.Array, activity: Activity,
           reset: jax.Array | None = None) -> DriveState:
    """The drives after this tick (the module docstring). ``clock``: the observation's ``game_clock``, or None when there
    is none: then sleep pressure holds still, and hunger still follows the activity. ``asleep`` bool[B]: the self's
    ``sleeping`` movement state. ``reset`` bool[B]: the lanes that start an episode, which restore S = 0 and the start of
    the food, and count no elapsed time on their first tick."""

    batch = state.sleep_pressure.shape[0]
    if reset is not None:
        state = state._replace(sleep_pressure=jnp.where(reset, 0.0, state.sleep_pressure),
                               food=jnp.where(reset, FOOD_START, state.food),
                               saturation=jnp.where(reset, SATURATION_START, state.saturation),
                               exhaustion=jnp.where(reset, 0.0, state.exhaustion),
                               has_clock=state.has_clock & ~reset)
    food, saturation, exhaustion = _eat_and_burn(state, activity)
    if clock is None:
        return state._replace(food=food, saturation=saturation, exhaustion=exhaustion,
                              has_clock=jnp.zeros((batch,), jnp.bool_))
    hours = elapsed(state, clock) / 3600.0
    # 1 - (1 - S) * exp(-h / tau), written as S + (1 - S) * (1 - exp(-h / tau)): the same value, without the cancellation
    # that loses the digits of a small S in float32 (expm1 keeps 1 - exp(-h / tau) exact for small h).
    awake = state.sleep_pressure + (1.0 - state.sleep_pressure) * -jnp.expm1(-hours / SLEEP_RISE_HOURS)
    asleep_now = state.sleep_pressure * jnp.exp(-hours / SLEEP_FALL_HOURS)
    sleep = jnp.clip(jnp.where(asleep, asleep_now, awake), 0.0, 1.0)
    seconds = jnp.broadcast_to(jnp.asarray(getattr(clock, "game_seconds_elapsed", state.seconds), jnp.float32), (batch,))
    time_of_day = jnp.broadcast_to(jnp.asarray(clock.time_of_day, jnp.float32), (batch,))
    return DriveState(sleep, food, saturation, exhaustion, seconds, time_of_day, jnp.ones((batch,), jnp.bool_))


def is_day(time_of_day: float) -> bool:
    """Day from a ``time_of_day`` reading: from sunrise to nightfall, as the simulator's ``is_day``."""

    return SUNRISE_UNITS <= float(time_of_day) < NIGHTFALL_UNITS


__all__ = ["Activity", "DriveState", "FOODS", "NAMES", "NOT_FOOD", "UNITS_PER_GAME_HOUR", "activity", "elapsed", "initial",
           "is_day", "update"]
