"""The recording of a run (``docs/VIRTUAL_HUMAN.md`` section 25; its grounding is 25.5): the state kept as event
sourcing keeps an application's state, as snapshots plus the events between them (25.2).

:class:`Recorder` writes, in one folder per run, ``snapshots/tick_{tick:08d}.npz`` with :func:`state.layer.save` every
``every`` ticks and at the end of an episode (``final``), the whole state. Between two calls it appends to
``stream.jsonl`` one JSON line per change: each new event of the log, each written entry that appeared, changed or was
freed, each place that formed or was evicted, and each person met (a record formed) or forgotten (a record was replaced
or cleared; section 26, ``state/social.py``). A social line is ``{"type": "social", "kind": "met" | "forgot", "id", "tick",
"lane"}``. The overhead is fixed: a snapshot now and then, and lines only when something happens.

:func:`read` gives the state at a tick: the latest snapshot at or before it, and the stream's lines after that snapshot
up to the tick. The whole state is exact at the snapshots. Between them the stream gives the recorded parts (the events,
the written entries, the places and the people), since rebuilding everything would need the raw observations, which are
not recorded. Deterministic replay of the world and the body stays the host's (25.2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experimental.neural_agent.skill_agent.contract import variables
from experimental.neural_agent.skill_agent.state import events, layer
from experimental.neural_agent.skill_agent.state.places import PLACE_KINDS

_TABLES: tuple[tuple[str, tuple[variables.Variable, ...]], ...] = (
    ("entity", variables.ENTITY_ENTRIES), ("place", variables.PLACE_ENTRIES))
"""The written tables and their declared fields (``state/written.py``)."""


def _arrays(state: layer.AgentState) -> dict[str, np.ndarray]:
    """The arrays the detectors compare, on the host: the log's clock, the place store's slots, each written table's
    slots (its key, its fields, the tick of its last write and its writer), and the people's records (key, first tick,
    valid)."""

    store, records = state.places.places, state.social.records
    arrays = {
        "clock": np.asarray(state.events.now),
        "store_valid": np.asarray(store.valid), "store_first": np.asarray(store.first),
        "store_world": np.asarray(store.world), "store_kinds": np.asarray(store.kinds),
        "store_strength": np.asarray(store.strength),
        "social_valid": np.asarray(records.valid), "social_key": np.asarray(records.key_id),
        "social_first": np.asarray(records.first),
    }
    for name, entries in (("entity", state.written.entity), ("place", state.written.place)):
        prefix = f"written_{name}"
        arrays.update({
            f"{prefix}_valid": np.asarray(entries.valid),
            f"{prefix}_key": np.asarray(entries.world if name == "place" else entries.key_id),
            f"{prefix}_values": np.asarray(entries.values),
            f"{prefix}_written_at": np.asarray(entries.written_at),
            f"{prefix}_writer": np.asarray(entries.writer),
        })
    return arrays


def _empty(current: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The arrays of a state with nothing in it: the first call records everything as new."""

    return {name: np.zeros_like(value) for name, value in current.items()}


def _flat(array: np.ndarray) -> np.ndarray:
    """[B, K, -1]: a table's keys as one row per slot, so a key compares the same for an id and for a position."""

    return array.reshape(array.shape[0], array.shape[1], -1)


def _event_lines(state: layer.AgentState, tick: int, previous: dict[str, np.ndarray],
                 current: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """The log's events not yet recorded, oldest first. An event is new when its tick is after the clock recorded last
    time; a clock that went back is a new episode, whose events are all new."""

    threshold = np.where(current["clock"] < previous["clock"], -1, previous["clock"])
    log = events.recent(state.events, events.EVENT_SLOTS)
    kind, when, other, amount, world, inferred, held = (
        np.asarray(x) for x in (log.kind, log.tick, log.other, log.amount, log.world, log.inferred, log.valid))
    return [{"tick": tick, "lane": int(lane), "type": "event", "kind": events.EVENT_KINDS[int(kind[lane, slot])],
             "amount": float(amount[lane, slot]), "inferred": bool(inferred[lane, slot]),
             "other": int(other[lane, slot]), "world": world[lane, slot].tolist()}
            for lane, slot in zip(*np.nonzero(held & (when > threshold[:, None])))]


def _entry_lines(tick: int, previous: dict[str, np.ndarray], current: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """The written entries that appeared, changed or were freed since the last call. A slot whose key moved counts as
    freed (its old entry) and appeared (its new one). A write is a change even when it leaves the fields as they were,
    since the tick of the last write moves."""

    lines = []
    for name, declared in _TABLES:
        prefix = f"written_{name}"
        fields = [field.name for field in declared]
        was, now = previous[f"{prefix}_valid"], current[f"{prefix}_valid"]
        same_key = np.all(_flat(previous[f"{prefix}_key"]) == _flat(current[f"{prefix}_key"]), axis=-1)
        moved = now & (~was | ~same_key | (current[f"{prefix}_written_at"] != previous[f"{prefix}_written_at"]))
        freed = was & (~now | ~same_key)
        for source, mask, gone in ((previous, freed, True), (current, moved, False)):
            for lane, slot in zip(*np.nonzero(mask)):
                lines.append({
                    "tick": tick, "lane": int(lane), "type": "write", "table": name,
                    "key": source[f"{prefix}_key"][lane, slot].tolist(),
                    "fields": dict(zip(fields, source[f"{prefix}_values"][lane, slot].tolist())),
                    "writer": variables.WRITERS[int(source[f"{prefix}_writer"][lane, slot])]
                    if source[f"{prefix}_writer"][lane, slot] >= 0 else None,
                    "freed": gone})
    return lines


def _place_lines(tick: int, previous: dict[str, np.ndarray], current: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """The places that formed or were evicted since the last call. A slot that holds a different place now (its first
    tick moved) evicted the old one and formed the new one."""

    moved = current["store_first"] != previous["store_first"]
    formed = current["store_valid"] & (~previous["store_valid"] | moved)
    evicted = previous["store_valid"] & (~current["store_valid"] | moved)
    lines = []
    for source, mask, gone in ((previous, evicted, True), (current, formed, False)):
        for lane, slot in zip(*np.nonzero(mask)):
            lines.append({
                "tick": tick, "lane": int(lane), "type": "place", "slot": int(slot),
                "kinds": [name for name, held in zip(PLACE_KINDS, source["store_kinds"][lane, slot]) if held],
                "strength": float(source["store_strength"][lane, slot]),
                "position": source["store_world"][lane, slot].tolist(), "evicted": gone})
    return lines


def _social_lines(tick: int, previous: dict[str, np.ndarray], current: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """The people met (a record formed) and forgotten (a record was replaced or cleared) since the last call, found as
    the places are: a slot whose key or first tick moved holds another person now. A lane whose clock went back began a
    new episode, so each of its records is forgotten, and a record formed again in the same slot is met again."""

    restarted = (current["clock"] < previous["clock"])[:, None]
    changed = ((current["social_key"] != previous["social_key"])
               | (current["social_first"] != previous["social_first"]) | restarted)
    formed = current["social_valid"] & (~previous["social_valid"] | changed)
    forgotten = previous["social_valid"] & (~current["social_valid"] | changed)
    lines = []
    for source, mask, kind in ((previous, forgotten, "forgot"), (current, formed, "met")):
        for lane, slot in zip(*np.nonzero(mask)):
            lines.append({"tick": tick, "lane": int(lane), "type": "social", "kind": kind,
                          "id": int(source["social_key"][lane, slot])})
    return lines


class Recorder:
    """The recording of one run in ``folder``: a snapshot every ``every`` ticks, and the stream of changes between
    calls. Call :meth:`record` with each tick's state (each tick, or at the decisions), with ``final=True`` at an
    episode's end."""

    def __init__(self, folder: str | Path, every: int = 300) -> None:
        self.folder = Path(folder)
        self.every = every
        self.snapshots = self.folder / "snapshots"
        self.stream = self.folder / "stream.jsonl"
        self.snapshots.mkdir(parents=True, exist_ok=True)
        self._previous: dict[str, np.ndarray] | None = None

    def record(self, state: layer.AgentState, tick: int, *, final: bool = False) -> None:
        """Append the changes since the previous call to the stream; write a snapshot when ``tick`` is a multiple of
        ``every`` or ``final`` is set."""

        current = _arrays(state)
        previous = _empty(current) if self._previous is None else self._previous
        lines = (_event_lines(state, tick, previous, current) + _entry_lines(tick, previous, current)
                 + _place_lines(tick, previous, current) + _social_lines(tick, previous, current))
        if lines:
            with self.stream.open("a") as handle:
                handle.write("".join(json.dumps(line) + "\n" for line in lines))
        if final or tick % self.every == 0:
            layer.save(self.snapshots / f"tick_{tick:08d}.npz", state)
        self._previous = current


def read(folder: str | Path, tick: int, like: layer.AgentState
         ) -> tuple[layer.AgentState, int, list[dict[str, Any]]]:
    """The state at ``tick`` of the run recorded in ``folder``: the latest snapshot at or before it, loaded with
    :func:`state.layer.load` as ``like``; the stream's lines after that snapshot's tick, up to ``tick``, oldest first.

    The whole state is exact at snapshots. Between them the stream gives the recorded parts (events, written entries and
    places), since rebuilding everything would need the raw observations, which are not recorded.

    Returns:
        ``(snapshot_state, snapshot_tick, changes)``.

    Raises:
        FileNotFoundError: If no snapshot was written at or before ``tick``.
    """

    root = Path(folder)
    snapshots = sorted((int(path.stem.split("_")[1]), path) for path in (root / "snapshots").glob("tick_*.npz"))
    earlier = [(snapshot_tick, path) for snapshot_tick, path in snapshots if snapshot_tick <= tick]
    if not earlier:
        raise FileNotFoundError(f"no snapshot at or before tick {tick} in {root}")
    snapshot_tick, path = earlier[-1]
    changes: list[dict[str, Any]] = []
    stream = root / "stream.jsonl"
    if stream.exists():
        for line in stream.read_text().splitlines():
            if not line.strip():
                continue
            change = json.loads(line)
            if snapshot_tick < change["tick"] <= tick:
                changes.append(change)
    return layer.load(path, like), snapshot_tick, changes


__all__ = ["Recorder", "read"]
