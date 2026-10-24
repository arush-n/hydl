"""The views of the state (``docs/VIRTUAL_HUMAN.md`` section 25; its grounding is 25.5): one agent's state read as typed
facts, and as the words the LLM is told. Both run on the host, off the tick.

:func:`facts` reads one lane into a JSON-safe dict: the persona's traits, the feelings, the counters, the clock and the
drives (``state/drives.py``); what the agent knows of others (``runtime/state_tick.known_entities``, by the ids the
layer perceives); the people it has met, each with the relationship the decision layer wrote about it and its
interaction facts (``state/social.py``, section 26);
where it has been (the long-term places and routes, each place with its distance and bearing from the agent now); what
happened lately (the event log, newest first); and what the top layers wrote, with its writer and its age. The decision
layer reads the state through these facts, and no model rates them.

:func:`words` says the same facts in short plain sentences, from fixed templates. No model is in the loop, so the words
are exactly the facts and can be tested (25.4, ruling 1; a model's summary of the facts is a later option beside them).
A budget caps their length. The items that fit are chosen as the Generative Agents retrieve memories (25.5): each
candidate (an entity, a person, a place or an event) is scored by recency, importance and relevance, each component
min-max scaled to [0, 1] across the candidates, all three weighted equally (25.4, ruling 2).

``runtime/state_tick`` imports the state, so :func:`facts` imports it inside the function.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from experimental.neural_agent.skill_agent.contract import variables
from experimental.neural_agent.skill_agent.state import affect as feeling
from experimental.neural_agent.skill_agent.state import clock, drives, events, layer, registry, social, written
from experimental.neural_agent.skill_agent.state.frames import world_to_body
from experimental.neural_agent.skill_agent.state.places import PLACE_KINDS

TRAITS: tuple[str, ...] = tuple(v.name for v in variables.VARIABLES if v.section == "self" and v.source == "birth")
"""The persona's traits: the ``self`` variables set at birth (section 15.6)."""
_COUNTER_WORDS: Mapping[str, str] = {
    "distance_walked": "blocks walked", "distance_sprinted": "blocks sprinted", "distance_swum": "blocks swum",
    "distance_climbed": "blocks climbed", "damage_dealt": "damage dealt", "damage_taken": "damage taken",
    "items_gained": "items gained", "food_eaten": "food eaten", "deaths": "deaths",
}
"""Each counter's words (``events.COUNTERS``)."""


def _writer(index: int) -> str | None:
    return variables.WRITERS[index] if index >= 0 else None


def _entries(entries: written.Entries, declared: tuple[variables.Variable, ...], lane: int, clock: int,
             dt_seconds: float, *, place: bool) -> list[dict[str, Any]]:
    """The valid entries of one table for one lane: each with its ``remember`` field, its writer and its age. A place's
    entry is keyed by its position, an entity's by its id."""

    remember = [field.name for field in declared].index("remember")
    values = np.asarray(entries.values)[lane]
    valid = np.asarray(entries.valid)[lane]
    key = np.asarray(entries.world if place else entries.key_id)[lane]
    written_at = np.asarray(entries.written_at)[lane]
    writer = np.asarray(entries.writer)[lane]
    out = []
    for slot in np.flatnonzero(valid):
        head = {"position": [float(x) for x in key[slot]]} if place else {"id": int(key[slot])}
        out.append({**head, "remember": float(values[slot, remember]), "writer": _writer(int(writer[slot])),
                    "age_s": float((clock - int(written_at[slot])) * dt_seconds)})
    return out


_CLOSENESS_WORDS: tuple[str, ...] = ("distant", "rather close", "close", "very close")
"""A closeness band's words, in the order of ``social.CLOSENESS_EDGES``."""
_ROLE_WORDS: tuple[tuple[str, str], ...] = (
    ("friend", "friend"), ("family", "family"), ("leader", "leader"), ("follower", "follower"), ("rival", "rival"),
    ("trader", "trader"), ("ingroup", "in my group"))
"""Each role flag and its word; a flag reads as on above one half."""
_TRUST_WORDS: tuple[tuple[str, str, str], ...] = (
    ("ability", "able", "unable"), ("benevolence", "kind", "unkind"), ("integrity", "honest", "dishonest"))
"""Each part of trustworthiness and its words above 0 and below it."""
_STARTED_WORDS: Mapping[str, str] = {"me": "I struck first", "them": "they struck first", "both": "both struck at once"}


def _interaction(columns: Mapping[str, np.ndarray], slot: int | None, clock: int, dt_seconds: float) -> dict[str, Any]:
    """The interaction facts of one record (its slot in ``columns``), or the facts of a person with no record: not in
    view, no meetings, no fights, no last interaction."""

    if slot is None:
        return {"in_view": False, "first_seen_s_ago": None, "last_seen_s_ago": None, "meetings": 0,
                "seconds_in_view": 0.0, "hits_taken": 0, "hits_dealt": 0, "damage_taken": 0.0, "damage_dealt": 0.0,
                "spared": 0, "started": None, "last_interaction_s_ago": None, "last_kind": None}
    started, last_interaction, last_kind = (int(columns[name][slot]) for name in
                                            ("started", "last_interaction", "last_kind"))
    return {
        "in_view": int(columns["last"][slot]) == clock,
        "first_seen_s_ago": (clock - int(columns["first"][slot])) * dt_seconds,
        "last_seen_s_ago": (clock - int(columns["last"][slot])) * dt_seconds,
        "meetings": int(columns["meetings"][slot]),
        "seconds_in_view": int(columns["ticks_in_view"][slot]) * dt_seconds,
        "hits_taken": int(columns["hits_taken"][slot]), "hits_dealt": int(columns["hits_dealt"][slot]),
        "damage_taken": float(columns["damage_taken"][slot]), "damage_dealt": float(columns["damage_dealt"][slot]),
        "spared": int(columns["spared"][slot]), "started": social.STARTED[started] if started >= 0 else None,
        "last_interaction_s_ago": (clock - last_interaction) * dt_seconds if last_interaction >= 0 else None,
        "last_kind": events.EVENT_KINDS[last_kind] if last_kind >= 0 else None,
    }


def _people(state: layer.AgentState, lane: int, here: Any, yaw: Any, clock: int, clock_written: int,
            dt_seconds: float) -> list[dict[str, Any]]:
    """The people of one lane (``state/social.py``, section 26): each record, then each entity the decision layer wrote
    about that has no record. Each has its written relationship (every entry field by name, its writer and its age; None
    without an entry), the rule's feelings toward it (``felt_by_rule``, labelled provisional), its interaction facts
    (:func:`_interaction`), its world ``position`` (a list of 3 floats, or None when not known), and its distance from
    ``here`` and bearing (degrees, 0 ahead, positive to the right, needing ``yaw`` as well) when its position is known.
    The ages are ticks times ``dt_seconds``, as the places' and the events' are."""

    rec = state.social.records
    columns = {name: np.asarray(getattr(rec, name))[lane] for name in (
        "key_id", "valid", "last", "first", "meetings", "ticks_in_view", "hits_taken", "hits_dealt", "damage_taken",
        "damage_dealt", "spared", "started", "last_interaction", "last_kind", "world", "world_known")}
    entries = state.written.entity
    entry = {name: np.asarray(getattr(entries, name))[lane] for name in ("key_id", "valid", "values", "written_at",
                                                                       "writer")}
    slot_of = {int(columns["key_id"][slot]): int(slot) for slot in np.flatnonzero(columns["valid"])}
    written_only = sorted({int(entry["key_id"][slot]) for slot in np.flatnonzero(entry["valid"])} - slot_of.keys())
    ids = list(slot_of) + written_only
    if not ids:
        return []
    table = np.full((state.events.now.shape[0], len(ids)), -1, np.int32)
    table[lane] = ids
    felt = np.asarray(feeling.feelings_for(state.affect, table))[lane]  # float32[P, 4]: hostility, fear, trust, anger
    fields = [field.name for field in variables.ENTITY_ENTRIES]
    people = []
    for row, ident in enumerate(ids):
        relationship = None
        held = np.flatnonzero(entry["valid"] & (entry["key_id"] == ident))
        if held.size:
            slot = int(held[0])
            relationship = {name: float(entry["values"][slot, index]) for index, name in enumerate(fields)}
            relationship["writer"] = _writer(int(entry["writer"][slot]))
            relationship["written_s_ago"] = float((clock_written - int(entry["written_at"][slot])) * dt_seconds)
        slot = slot_of.get(ident)
        known = slot is not None and bool(columns["world_known"][slot])
        position = [float(x) for x in columns["world"][slot]] if known else None
        distance = bearing = None
        if known and here is not None:
            offset = np.asarray(position, np.float64) - here
            distance = float(np.linalg.norm(offset))
            if yaw is not None:
                body = world_to_body(offset, yaw)
                bearing = float(np.degrees(np.arctan2(body[1], body[0])))
        people.append({
            "id": ident, "relationship": relationship,
            "felt_by_rule": {"provisional": True, "hostility": float(felt[row, 0]), "fear": float(felt[row, 1]),
                             "trust": float(felt[row, 2]), "anger": float(felt[row, 3])},
            **_interaction(columns, slot, clock, dt_seconds), "position": position, "distance": distance,
            "bearing": bearing})
    return people


def facts(state: layer.AgentState, lane: int, *, own_position: Any = None, body_yaw_degrees: Any = None,
          dt_seconds: float = clock.DEFAULT_TICK_SECONDS) -> dict[str, Any]:
    """One lane's state as facts (the module docstring), JSON-safe.

    Args:
        state: The state of B agents.
        lane: The agent to read.
        own_position: float[B, 3], the agents' world positions, or None.
        body_yaw_degrees: The agents' body yaws in degrees (float[B], or one float for every lane), or None. The places
            carry their distance, ahead and right only when both this and ``own_position`` are given, and the entities
            their distance only with ``own_position``. A person has its distance with ``own_position`` (when its world
            position is known) and its bearing with this as well.
        dt_seconds: One tick, in seconds: the ages (``*_s``) are ticks times it.

    Returns:
        ``{"self", "feeling", "entities", "places", "routes", "events", "written", "people", "drives"}`` as the module
        docstring lists them.
    """

    from experimental.neural_agent.skill_agent.runtime import state_tick  # lazily: runtime imports the state

    values = np.asarray(state.variables)

    def read(name: str) -> float:
        return float(registry.get(values, name)[lane])

    def pick(value: Any) -> np.ndarray:  # this lane's row of a batch, or a value shared by every lane
        array = np.asarray(value, np.float64)
        return array[lane] if array.ndim else array

    here = None if own_position is None else pick(own_position)
    yaw = None if body_yaw_degrees is None else np.radians(pick(body_yaw_degrees))

    placed = state.places
    now = int(np.asarray(placed.now)[lane])
    store = placed.places
    world = np.asarray(store.world)[lane]
    body = None if here is None or yaw is None else np.asarray(world_to_body(world - here, yaw), np.float64)
    kinds, strength, marked = (np.asarray(x)[lane] for x in (store.kinds, store.strength, store.marked))
    visits, last, valid = (np.asarray(x)[lane] for x in (store.visits, store.last, store.valid))
    place_list = []
    for slot in np.flatnonzero(valid):
        offset = None if body is None else body[slot]
        place_list.append({
            "slot": int(slot),
            "kinds": [name for name, held in zip(PLACE_KINDS, kinds[slot]) if held],
            "strength": float(strength[slot]),
            "marked": bool(marked[slot]),
            "visits": int(visits[slot]),
            "since_s": float((now - int(last[slot])) * dt_seconds),
            "position": [float(x) for x in world[slot]],
            "distance": None if offset is None else float(np.linalg.norm(world[slot] - here)),
            "ahead": None if offset is None else float(offset[0]),
            "right": None if offset is None else float(offset[1]),
        })

    route = placed.routes
    source, target, best, count, route_last, route_valid = (
        np.asarray(x)[lane]
        for x in (route.source, route.target, route.best_ticks, route.count, route.last, route.valid))
    route_list = [{"from": int(source[slot]), "to": int(target[slot]), "best_s": float(best[slot]) * dt_seconds,
                   "count": int(count[slot]), "since_s": float((now - int(route_last[slot])) * dt_seconds)}
                  for slot in np.flatnonzero(route_valid)]

    clock = int(np.asarray(state.events.now)[lane])
    log = events.recent(state.events, events.EVENT_SLOTS)  # oldest first; reversed below
    kind, when, other, amount, inferred, held = (
        np.asarray(x)[lane] for x in (log.kind, log.tick, log.other, log.amount, log.inferred, log.valid))
    event_list = [{"kind": events.EVENT_KINDS[int(kind[slot])], "age_s": float((clock - int(when[slot])) * dt_seconds),
                   "amount": float(amount[slot]), "inferred": bool(inferred[slot]), "other": int(other[slot])}
                  for slot in range(events.EVENT_SLOTS - 1, -1, -1) if held[slot]]

    clock_written = int(np.asarray(state.written.now)[lane])
    clocked = bool(np.asarray(state.drives.has_clock)[lane])
    time_of_day = float(np.asarray(state.drives.time_of_day)[lane])
    return {
        "self": {"traits": {name: read(name) for name in TRAITS},
                 "counters": {name: read(name) for name in events.COUNTERS}},
        "feeling": {"alarm": read("fear"), "anger": read("anger"), "startle": read("startle"),
                    "valence": read("valence"), "coping": read("coping"),
                    "mood": {"pleasure": read("mood_pleasure"), "arousal": read("mood_arousal"),
                             "dominance": read("mood_dominance")}},
        "entities": state_tick.known_entities(state, own_position, dt_seconds=dt_seconds)[lane],
        "places": place_list,
        "routes": route_list,
        "events": event_list,
        "written": {
            "entities": _entries(state.written.entity, variables.ENTITY_ENTRIES, lane, clock_written, dt_seconds,
                                 place=False),
            "places": _entries(state.written.place, variables.PLACE_ENTRIES, lane, clock_written, dt_seconds,
                               place=True),
        },
        "people": _people(state, lane, here, yaw, clock, clock_written, dt_seconds),
        "drives": {"time_of_day_hours": time_of_day / drives.UNITS_PER_GAME_HOUR if clocked else None,
                   "is_day": drives.is_day(time_of_day) if clocked else None,
                   "sleep_pressure": read("sleep_pressure"), "hunger": read("hunger")},
    }


def _number(value: float) -> str:
    """One decimal at most, without the trailing zeros: 4.0 reads as 4, 12.34 as 12.3."""

    return f"{round(float(value), 1):g}"


def _two(value: float) -> str:
    """Two decimals, with a value that rounds to zero written as 0.00 (not -0.00): round first, then add 0.0, which
    turns the negative zero that rounding leaves into a positive one."""

    return f"{round(float(value), 2) + 0.0:.2f}"


def _feeling_line(feeling: Mapping[str, Any]) -> str:
    mood = feeling["mood"]
    return (f"Feeling: alarm {_two(feeling['alarm'])}, anger {_two(feeling['anger'])}, "
            f"startle {_two(feeling['startle'])}, valence {_two(feeling['valence'])}, "
            f"coping {_two(feeling['coping'])}. Mood: pleasure {_two(mood['pleasure'])}, "
            f"arousal {_two(mood['arousal'])}, dominance {_two(mood['dominance'])}.")


def _counters_line(counters: Mapping[str, float]) -> str:
    shown = [f"{_COUNTER_WORDS[name]} {_number(value)}" for name, value in counters.items() if value != 0.0]
    return f"Counters: {', '.join(shown)}." if shown else ""


def _drives_line(drives_facts: Mapping[str, Any] | None) -> str:
    """The clock and the drives in one sentence, with a clock only: the game hour, day or night, sleep pressure and
    hunger (16.1; the top layers read them, the body does not). Empty without a clock, or for facts made before the
    drives."""

    if drives_facts is None or drives_facts["time_of_day_hours"] is None:
        return ""
    hour, minute = divmod(round(drives_facts["time_of_day_hours"] * 60) % (24 * 60), 60)
    part = "day" if drives_facts["is_day"] else "night"
    return (f"It is {hour:02d}:{minute:02d} ({part}); sleep pressure {_two(drives_facts['sleep_pressure'])}, "
            f"hunger {_two(drives_facts['hunger'])}.")


def _entity_line(entity: Mapping[str, Any]) -> str:
    detail = f"hostile {entity['hostile']:.1f}, health {entity['health']:.0%}"
    if not entity["alive"]:
        detail += ", dead"
    where = "" if entity["distance"] is None else f" is {_number(entity['distance'])} blocks away"
    return f"Entity {entity['id']} ({detail}){where}, seen {entity['age_s']:.1f} s ago."


def _place_line(place: Mapping[str, Any]) -> str:
    label = f"A {' and '.join(place['kinds'])} place" if place["kinds"] else "A place"
    where = ""
    if place["ahead"] is not None:
        ahead, right = place["ahead"], place["right"]
        if abs(ahead) < 1.0 and abs(right) < 1.0:  # within a block of where the agent stands
            where = " is here"
        else:
            where = (f" is {_number(abs(ahead))} blocks {'ahead' if ahead >= 0 else 'behind'} and "
                     f"{_number(abs(right))} to the {'right' if right >= 0 else 'left'}")
    times = "time" if place["visits"] == 1 else "times"
    return f"{label} (strength {place['strength']:.1f}){where}; visited {place['visits']} {times}."


def _event_line(event: Mapping[str, Any]) -> str:
    kind, amount, other = event["kind"], event["amount"], event["other"]
    if kind == "damage_taken":
        what = f"damage taken, {amount:.0%} of health" + (f", from entity {other}" if other >= 0 else "")
    elif kind == "damage_dealt":
        what = f"damage dealt, {amount:.0%} of its health" + (f", to entity {other}" if other >= 0 else "")
    elif kind == "item_gained":
        what = f"items gained: {_number(amount)}"
    elif kind == "item_lost":
        what = f"items lost: {_number(amount)}"
    elif kind == "ate":
        what = f"ate, {_number(amount)} consumed"
    elif kind == "request_done":
        what = "a request finished"
    elif kind == "request_failed":
        what = "a request failed"
    elif kind == "request_aborted":
        what = "a request was aborted"
    elif kind == "death":
        what = "died"
    else:
        what = kind
    if event["inferred"]:
        what += " (inferred)"
    return f"{event['age_s']:.1f} s ago: {what}."


def _person_line(person: Mapping[str, Any]) -> str:
    """A person as one line, e.g. ``12: close; friend; able, kind; met 3 times, last 40 s ago 8 blocks away; fought: 2
    hits taken, 1 dealt, they struck first``. Only what is known is said: an entry's closeness, roles and trust parts (a
    part at 0 is left out), and a record's meetings and fights."""

    parts: list[str] = []
    relationship = person["relationship"]
    if relationship is not None:
        parts.append(_CLOSENESS_WORDS[sum(relationship["closeness"] >= edge for edge in social.CLOSENESS_EDGES)])
        parts.extend(word for name, word in _ROLE_WORDS if relationship[name] > 0.5)
        trust = [positive if relationship[name] > 0 else negative
                 for name, positive, negative in _TRUST_WORDS if relationship[name] != 0]
        if trust:
            parts.append(", ".join(trust))
    if person["last_seen_s_ago"] is not None:
        times = "time" if person["meetings"] == 1 else "times"
        met = f"met {person['meetings']} {times}, last {_number(person['last_seen_s_ago'])} s ago"
        if person["distance"] is not None:
            met += f" {_number(person['distance'])} blocks away"
        parts.append(met)
        taken, dealt = person["hits_taken"], person["hits_dealt"]
        if taken or dealt:
            fought = f"fought: {taken} {'hit' if taken == 1 else 'hits'} taken, {dealt} dealt"
            if person["started"] is not None:
                fought += f", {_STARTED_WORDS[person['started']]}"
            parts.append(fought)
    return f"{person['id']}: " + "; ".join(parts)


def _near(distance: float | None) -> float:
    """Relevance: the nearer the better, and 0 for a thing whose distance is not known."""

    return 0.0 if distance is None else -float(distance)


def _scale(column: np.ndarray) -> np.ndarray:
    """Min-max scaled to [0, 1] across the candidates; a constant column scales to 0."""

    low, high = column.min(), column.max()
    return np.zeros_like(column) if high == low else (column - low) / (high - low)


def _ranked(facts: Mapping[str, Any]) -> list[str]:
    """The candidate lines, best first: each entity, person, place and event with its recency, importance and relevance
    (25.5), scaled and summed with equal weights. Ties keep the order the candidates are listed in. A person's recency
    is its last sighting, or, for one never seen (a written entry only), the age of its written entry."""

    items: list[tuple[str, float, float, float]] = []  # (line, recency, importance, relevance)
    for entity in facts["entities"]:
        items.append((_entity_line(entity), -entity["age_s"], entity["hostile"], _near(entity["distance"])))
    for person in facts["people"]:
        relationship = person["relationship"]
        closeness = 0.0 if relationship is None else relationship["closeness"]
        seen_s_ago = relationship["written_s_ago"] if person["last_seen_s_ago"] is None else person["last_seen_s_ago"]
        importance = closeness + person["hits_taken"] + person["hits_dealt"]
        items.append((_person_line(person), -seen_s_ago, importance,
                      1.0 if person["in_view"] else _near(person["distance"])))
    for place in facts["places"]:
        importance = place["strength"] + (1.0 if place["marked"] else 0.0)
        items.append((_place_line(place), -place["since_s"], importance, _near(place["distance"])))
    for event in facts["events"]:
        items.append((_event_line(event), -event["age_s"], event["amount"], 0.0))
    if not items:
        return []
    scores = np.zeros(len(items))
    for component in (1, 2, 3):
        scores += _scale(np.asarray([item[component] for item in items], np.float64))
    order = sorted(range(len(items)), key=lambda index: -scores[index])
    return [items[index][0] for index in order]


def words(facts: Mapping[str, Any], budget: int = 1200) -> str:
    """The facts as plain sentences, at most ``budget`` characters long (the module docstring).

    The feeling line comes first, then the counters line when a counter is non-zero; both are always written, so a
    budget below their length is exceeded by them alone. Then the clock and the drives, one sentence with a clock only,
    and the candidates, best first, each on its own line, stopping at the first one that would take the text past the
    budget.
    """

    lines = [_feeling_line(facts["feeling"])]
    counters = _counters_line(facts["self"]["counters"])
    if counters:
        lines.append(counters)
    drives_line = _drives_line(facts.get("drives"))
    if drives_line:
        lines.append(drives_line)
    text = "\n".join(lines)
    for line in _ranked(facts):
        grown = f"{text}\n{line}"
        if len(grown) > budget:
            break
        text = grown
    return text


__all__ = ["TRAITS", "facts", "words"]
