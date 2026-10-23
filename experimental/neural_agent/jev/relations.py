"""The decision layer's relationship writes (``docs/MIDDLE_LAYER.md`` section 11; the skill layer's ``VIRTUAL_HUMAN.md``
section 26 supplies the facts and the fields).

At each host decision, per agent, :meth:`RelationRules.plan` compares every individual's interaction facts
(``skill_agent/state/social.py``) with what they were at the last settled decision, and writes the relationship fields
that change. Every rule reads the device's values (the written entries) and the facts, and each write is an absolute
value, so the device state is the only truth the host keeps; the host keeps only the snapshots it compares against.

Rules (section 11; [E] = starting values to fit):
- **Fights**: per window, ``d`` = the damage they dealt me (the event log's amount, a fraction of my maximum health). If
  they struck first in the meeting: ``hostility += 0.5 d`` and ``anger += 0.8 d``. ``fear += d * r`` where ``r`` is their
  damage over mine in the meeting, clamped to ``[0, FEAR_RATIO_CLAMP]`` [E]. A hit I dealt while I struck first:
  ``hostility += 0.2 d`` for my damage. ``ability`` moves ``ABILITY_RATE`` [E] of the way toward ``2 s - 1``, where ``s``
  is their share of the hits between us.
- **Peace**: a meeting that has run over 60 s with no blow yet credits ``closeness += 1/30`` once (the meeting count is
  the mark). ``benevolence += 0.05`` per completed spared chance.
- **Decay**: ``anger`` halves every 60 s out of contact, ``fear`` every 120 s out of view, ``hostility`` every 30 min;
  the seconds are ticks times ``tick_seconds``. Closeness, the trust parts and the roles do not decay. Contact and view
  are both read as "not perceived this tick", the one signal the records hold.
- **Roles**: ``rival`` after 3 meetings with fights; ``friend`` when closeness > 1/2 and hostility < 0.1; both cleared in
  reverse. ``ingroup`` is left alone (no scenario team is published).
- **Directives**: a goal's relationship directive sets its field, and holds against the rules while the goal is in the
  executive section. ``family``, ``leader``, ``follower``, ``trader``, ``owe_them`` and ``they_owe`` are written only
  by directive. ``integrity`` is never written.
- **Table full**: a write refused as ``full`` (a 151st individual) ejects the record with the lowest closeness, no role
  flag, no ``remember`` entry and the oldest last contact, one field at a time until the entry frees, then retries the
  refused writes at the next decision.

A group is one individual's changes at one decision. The group is written whole when its largest change is at least
``MIN_CHANGE`` (0.05, section 11), or when it carries a peace credit, which is 1/30 by itself (a deliberate exception,
reported to the coordinator). A group that is not written keeps its snapshot, so its changes accumulate until they
qualify. A group is written when it fits the agent's budget (the remember writes come first; the largest changes go
first), otherwise it waits too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from experimental.neural_agent.contracts import ADDRESSEE_ALL
from experimental.neural_agent.shared_state.sections.executive import ExecutiveSection
from experimental.neural_agent.shared_state.skill_writes import FULL_REASON
from experimental.neural_agent.skill_agent.contract import variables
from experimental.neural_agent.skill_agent.state import clock, social

MIN_CHANGE: float = 0.05
"""A change under this is not written (section 11)."""
HOSTILITY_FROM_STRIKE: float = 0.5
ANGER_FROM_STRIKE: float = 0.8
HOSTILITY_FROM_MY_STRIKE: float = 0.2
FEAR_RATIO_CLAMP: float = 2.0
"""The clamp on their damage over mine in the fear rule [E]."""
ABILITY_RATE: float = 0.1
"""The fraction of the way ``ability`` moves toward its target per window with hits [E]."""
PEACE_CLOSENESS: float = 1 / 30
PEACE_SECONDS: float = 60.0
SPARED_BENEVOLENCE: float = 0.05
ANGER_HALF_LIFE_SECONDS: float = 60.0
FEAR_HALF_LIFE_SECONDS: float = 120.0
HOSTILITY_HALF_LIFE_SECONDS: float = 30 * 60.0
RIVAL_MEETINGS: int = 3
FRIEND_CLOSENESS: float = 0.5
FRIEND_HOSTILITY: float = 0.1
ROLE_FIELDS: tuple[str, ...] = ("friend", "family", "leader", "follower", "rival", "trader", "ingroup")
RANGES: dict[str, tuple[float, float]] = {v.name: (v.low, v.high) for v in variables.ENTITY_ENTRIES}
WRITABLE: tuple[str, ...] = tuple(v.name for v in variables.ENTITY_ENTRIES if v.name not in ("remember", "integrity"))
"""The fields the decision layer writes: every relationship field but ``integrity`` (waits) and ``remember`` (the
remember policy's)."""
_FIELDS: tuple[str, ...] = tuple(v.name for v in variables.ENTITY_ENTRIES)
_STARTED_ME: int = social.STARTED.index("me")
_STARTED_THEM: int = social.STARTED.index("them")
_COUNTERS: tuple[str, ...] = ("hits_taken", "hits_dealt", "damage_taken", "damage_dealt", "spared", "meetings")
_SOCIAL_COLUMNS: tuple[str, ...] = (
    "key_id", "valid", "first", "last", "meetings", "meeting_start", "started", "hits_taken", "hits_dealt",
    "damage_taken", "damage_dealt", "spared",
)


@dataclass(frozen=True)
class LaneView:
    """One lane's device state as host values: its clock, its social records and its live entity entries."""

    now: int
    social: Mapping[int, Mapping[str, Any]]  # key -> the record's columns (``_SOCIAL_COLUMNS`` but the key)
    entries: Mapping[int, Mapping[str, float]]  # key -> every entity field by name


def read_device(state: Any, lane: int) -> LaneView:
    """The host view of one lane of an ``AgentState`` (``skill_agent/state/layer.py``): its social records and its
    written entity entries, read from the device arrays. ``state`` needs ``social``, ``written`` and ``events``."""

    records = state.social.records
    columns = {name: np.asarray(getattr(records, name))[lane] for name in _SOCIAL_COLUMNS}
    people = {}
    for slot in np.flatnonzero(columns["valid"]):
        people[int(columns["key_id"][slot])] = {
            name: columns[name][slot].item() for name in _SOCIAL_COLUMNS if name not in ("key_id", "valid")}
    entity = state.written.entity
    keys = np.asarray(entity.key_id)[lane]
    valid = np.asarray(entity.valid)[lane]
    values = np.asarray(entity.values)[lane]
    entries = {int(keys[slot]): {name: float(values[slot, index]) for index, name in enumerate(_FIELDS)}
               for slot in np.flatnonzero(valid)}
    return LaneView(now=int(np.asarray(state.events.now)[lane]), social=people, entries=entries)


def directives_of(section: ExecutiveSection, lane: int) -> dict[int, dict[str, float]]:
    """The relationship directives of the goals addressed to ``lane`` (``all``, ``lane:<n>`` or ``squad:<name>``): entity
    -> field -> value. A field named by two goals takes the later one's value."""

    squads = {squad.name: set(squad.lanes) for squad in section.squads}
    out: dict[int, dict[str, float]] = {}
    for goal in section.goals:
        addressee = goal.addressee
        if addressee == ADDRESSEE_ALL:
            addressed = True
        elif addressee.startswith("lane:"):
            addressed = addressee[len("lane:"):] == str(lane)
        else:
            addressed = lane in squads.get(addressee.removeprefix("squad:"), set())
        if not addressed:
            continue
        for item in goal.relation:
            out.setdefault(item.entity, {})[item.name] = item.value
    return out


@dataclass
class _Seen:
    """The counters a record had at its last settled decision, and when each decay was last accounted."""

    first: int
    counters: dict[str, float]
    credited: int  # the meeting count whose peace was credited
    decay_at: int  # the device tick hostility's decay was accounted to
    out_at: int  # the device tick anger's and fear's decay was accounted to


@dataclass
class _Group:
    """One individual's changes at one decision."""

    key: int
    magnitude: float
    changes: dict[str, float]
    credit: bool
    in_view: bool
    counters: dict[str, float]
    credited: int  # the meeting count whose peace stands credited once this group is written
    first: int
    now: int

    @property
    def writable(self) -> bool:
        return bool(self.changes) and (self.magnitude >= MIN_CHANGE or self.credit)


def _clamp(name: str, value: float) -> float:
    low, high = RANGES[name]
    return min(max(float(value), low), high)


class RelationRules:
    """The relationship rules of every lane, with their snapshots, the eject-and-retry state and the write refusals."""

    def __init__(self, lanes: int) -> None:
        self.lanes = lanes
        self._seen: list[dict[int, _Seen]] = [{} for _ in range(lanes)]
        self._retry: list[list[tuple[int, str, float]]] = [[] for _ in range(lanes)]
        self._ejecting: list[int | None] = [None] * lanes
        self.ejections = [0] * lanes  # entries freed by an eject
        self.refused = [0] * lanes  # writes refused for a reason other than full
        self.ignored_directives = 0  # directives naming a field the decision layer does not write

    def settle(self, lane: int, write: tuple[int, str, float], reason: str | None) -> None:
        """Take a write's result: a refusal as full is retried after an eject; any other refusal is counted."""

        if reason is None:
            return
        if reason == FULL_REASON:
            if write not in self._retry[lane]:
                self._retry[lane].append(write)
        else:
            self.refused[lane] += 1

    def plan(self, lane: int, view: LaneView, directives: Mapping[int, Mapping[str, float]], budget: int,
             tick_seconds: float) -> list[tuple[int, str, float]]:
        """The writes of one lane at this decision, as (entity, field, value), at most ``budget``, eject and retry
        first, then each individual's group, the largest change first."""

        out: list[tuple[int, str, float]] = []
        left = self._eject_and_retry(lane, view, directives, budget, out)
        groups = self._groups(lane, view, directives, tick_seconds)
        seen = self._seen[lane]
        live = set(view.social) | set(view.entries)
        for group in sorted(groups, key=lambda g: (-g.magnitude, g.key)):
            if not group.changes:
                self._store(lane, group)
            elif group.writable and len(group.changes) <= left:
                out.extend((group.key, name, value) for name, value in group.changes.items())
                left -= len(group.changes)
                self._store(lane, group)
            elif group.in_view:
                # Held, but the decay of anger and fear is not owed to a time in view: account it as it passes.
                prior = seen.get(group.key)
                if prior is not None:
                    prior.out_at = group.now
        for key in [k for k in seen if k not in live]:
            del seen[key]
        return out

    # ------------------------------------------------------------------ eject and retry

    def _eject_and_retry(self, lane: int, view: LaneView, directives: Mapping[int, Mapping[str, float]], budget: int,
                         out: list[tuple[int, str, float]]) -> int:
        """Eject one entry at a time while a write is retried; retry the refused writes once nothing is ejected. The
        writes go into ``out``; returns the budget left."""

        left = budget
        victim = self._ejecting[lane]
        freed = victim is not None and victim not in view.entries
        if freed:
            self._ejecting[lane] = victim = None
            self.ejections[lane] += 1
        if victim is None and self._retry[lane] and not freed:
            victim = self._choose_victim(lane, view, directives)
            self._ejecting[lane] = victim
            if victim is None:
                return left  # the table is full and nothing may be ejected: the refused writes wait
        if victim is not None:
            for name in WRITABLE:
                if left > 0 and view.entries[victim][name] != 0.0:
                    out.append((victim, name, 0.0))
                    left -= 1
            return left
        kept = []
        for write in self._retry[lane]:
            if left > 0:
                out.append(write)
                left -= 1
            else:
                kept.append(write)
        self._retry[lane] = kept
        return left

    def _choose_victim(self, lane: int, view: LaneView, directives: Mapping[int, Mapping[str, float]]) -> int | None:
        """The entry to eject: lowest closeness, no role flag, no remember entry, then the oldest last contact."""

        retrying = {key for key, _, _ in self._retry[lane]}
        candidates = []
        for key, row in view.entries.items():
            if key in directives or key in retrying or row["remember"] > 0.0:
                continue
            if any(row[name] > 0.5 for name in ROLE_FIELDS):
                continue
            last = view.social[key]["last"] if key in view.social else -1
            candidates.append((row["closeness"], last, key))
        return min(candidates)[2] if candidates else None

    # ------------------------------------------------------------------ rules

    def _groups(self, lane: int, view: LaneView, directives: Mapping[int, Mapping[str, float]],
                tick_seconds: float) -> list[_Group]:
        seen = self._seen[lane]
        victim = self._ejecting[lane]
        groups = []
        for key in sorted(set(view.social) | set(view.entries) | set(directives)):
            if key == victim:
                continue
            record = view.social.get(key)
            prior = seen.get(key)
            if record is not None and (prior is None or prior.first != record["first"]):
                # A new record starts at zero: its first facts, this tick's included, are the first window's.
                prior = _Seen(first=record["first"], counters={name: 0.0 for name in _COUNTERS}, credited=0,
                              decay_at=view.now, out_at=view.now)
            if prior is None:
                prior = _Seen(first=-1, counters={name: 0.0 for name in _COUNTERS}, credited=0,
                              decay_at=view.now, out_at=view.now)
            groups.append(self._group(key, record, prior, view, directives.get(key, {}), tick_seconds))
        return groups

    def _group(self, key: int, record: Mapping[str, Any] | None, prior: _Seen, view: LaneView,
               directed: Mapping[str, float], tick_seconds: float) -> _Group:
        now = view.now
        row = view.entries.get(key, {})
        old = {name: float(row.get(name, 0.0)) for name in WRITABLE}
        new = dict(old)
        counters = _counters(record) if record is not None else dict(prior.counters)
        delta = {name: counters[name] - prior.counters[name] for name in _COUNTERS}
        in_view = record is not None and record["last"] == now
        credit = False
        if delta["damage_taken"] > 0.0:
            d = delta["damage_taken"]
            if record is not None and record["started"] == _STARTED_THEM:
                new["hostility"] += HOSTILITY_FROM_STRIKE * d
                new["anger"] += ANGER_FROM_STRIKE * d
            ratio = (min(counters["damage_taken"] / counters["damage_dealt"], FEAR_RATIO_CLAMP)
                     if counters["damage_dealt"] > 0.0 else FEAR_RATIO_CLAMP)
            new["fear"] += d * ratio
        if delta["damage_dealt"] > 0.0 and record is not None and record["started"] == _STARTED_ME:
            new["hostility"] += HOSTILITY_FROM_MY_STRIKE * delta["damage_dealt"]
        hits = counters["hits_taken"] + counters["hits_dealt"]
        if delta["hits_taken"] + delta["hits_dealt"] > 0.0 and hits > 0.0:
            target = 2.0 * counters["hits_taken"] / hits - 1.0
            new["ability"] += ABILITY_RATE * (target - new["ability"])
        if delta["spared"] > 0.0:
            new["benevolence"] += SPARED_BENEVOLENCE * delta["spared"]
        if (record is not None and record["started"] == -1 and record["meetings"] != prior.credited
                and (now - record["meeting_start"]) * tick_seconds > PEACE_SECONDS
                and now - record["last"] <= clock.about(social.MEETING_GAP_SECONDS, tick_seconds)):
            new["closeness"] += PEACE_CLOSENESS
            credit = True
        seconds = (now - prior.decay_at) * tick_seconds
        if seconds > 0.0:
            new["hostility"] *= 0.5 ** (seconds / HOSTILITY_HALF_LIFE_SECONDS)
        out_seconds = (now - prior.out_at) * tick_seconds
        if out_seconds > 0.0 and not in_view:
            new["anger"] *= 0.5 ** (out_seconds / ANGER_HALF_LIFE_SECONDS)
            new["fear"] *= 0.5 ** (out_seconds / FEAR_HALF_LIFE_SECONDS)
        rival = record is not None and record["meetings"] >= RIVAL_MEETINGS and hits > 0.0
        new["rival"] = 1.0 if rival else 0.0
        new["friend"] = 1.0 if (new["closeness"] > FRIEND_CLOSENESS and new["hostility"] < FRIEND_HOSTILITY) else 0.0
        for name in WRITABLE:
            new[name] = _clamp(name, new[name])
        for name, value in directed.items():
            if name in WRITABLE:
                new[name] = _clamp(name, value)
            else:
                self.ignored_directives += 1
        changes = {name: new[name] for name in WRITABLE if abs(new[name] - old[name]) > 1e-9}
        magnitude = max((abs(new[name] - old[name]) for name in changes), default=0.0)
        credited = int(record["meetings"]) if credit else prior.credited
        return _Group(key=key, magnitude=magnitude, changes=changes, credit=credit, in_view=in_view,
                      counters=counters, credited=credited,
                      first=record["first"] if record is not None else prior.first, now=now)

    def _store(self, lane: int, group: _Group) -> None:
        """Take a resolved group's counters as the new snapshot: its writes went out, or nothing was owed."""

        self._seen[lane][group.key] = _Seen(first=group.first, counters=dict(group.counters),
                                            credited=group.credited, decay_at=group.now, out_at=group.now)


def _counters(record: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(record[name]) for name in _COUNTERS}


__all__ = [
    "ABILITY_RATE", "ANGER_FROM_STRIKE", "FEAR_RATIO_CLAMP", "HOSTILITY_FROM_MY_STRIKE",
    "HOSTILITY_FROM_STRIKE", "LaneView", "MIN_CHANGE", "PEACE_CLOSENESS", "RelationRules", "ROLE_FIELDS",
    "WRITABLE", "directives_of", "read_device",
]
