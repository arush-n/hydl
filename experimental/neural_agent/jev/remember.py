"""The remember policy of the decision layer: what it writes into an agent's written entries, and when it takes that back.

Rulings of the owner (2026-10-22), built here and driven by ``jev/programs.py`` (module docstring, item 10):

- A program step whose request targets an entity or a place writes ``remember = p`` on that key when the step's request
  is issued. ``p`` is the goal's priority clipped to [0, 1], the mapping the request's urgency already uses
  (``programs.py`` item 9). A goal with no slot has priority 0, and writes nothing: a write of 0 ejects the entry.
- The program run that wrote a key ends by done, aborted or replaced, and then the key is written ``remember = 0``, unless
  the program keeps it (``Program.keep``, default False). A kept entry is no longer owned by the run; it stays held.
- A write refused as full (no free entry on the device): the lowest-valued entry this layer wrote that no live request
  refers to is ejected, and the refused write is retried at the next decision. When every held entry is referenced,
  :meth:`RememberLedger.on_result` returns True and the caller asks the LLM instead.

The ledger holds what this layer intends and wrote. The device table belongs to the skill layer; its results come back
as write statuses, which the caller passes to :meth:`RememberLedger.on_result`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Iterable

Key = tuple[str, Any]
"""A written entry's key: ``("entity", id)`` or ``("place", (x, y, z))``."""
REMEMBER: str = "remember"
FULL_REASON: str = "full"


def remember_value(priority: float) -> float:
    """The value of ``remember`` for a goal of ``priority``: the priority clipped to [0, 1]."""

    return min(1.0, max(0.0, float(priority)))


def key_of(request: Any) -> Key | None:
    """The key a request targets: its entity, else its position, else None (the request names neither)."""

    args = request.args
    if args.entity is not None:
        return ("entity", int(args.entity))
    if args.position is not None:
        return ("place", tuple(float(value) for value in args.position))
    return None


@dataclass(frozen=True)
class WriteOp:
    """One write the decision layer intends at its next decision. ``value`` 0 ejects the entry.

    ``run`` is the program run that owns the entry (None once a kept run has ended). ``serial`` identifies the op.
    """

    serial: int
    lane: int
    key: Key
    value: float
    run: int | None
    goal_slot: int
    executive_version: int
    program_step: int


class RememberLedger:
    """The decision layer's remember entries, per lane, and the writes waiting to go out.

    Args:
        lanes: The number of lanes B.
    """

    def __init__(self, lanes: int) -> None:
        self.lanes = lanes
        self._held: list[dict[Key, WriteOp]] = [{} for _ in range(lanes)]
        self._outbox: list[list[WriteOp]] = [[] for _ in range(lanes)]
        self._retry: list[list[WriteOp]] = [[] for _ in range(lanes)]
        self._serial = 0

    def _op(self, lane: int, key: Key, value: float, run: int | None, goal_slot: int, version: int, step: int) -> WriteOp:
        self._serial += 1
        return WriteOp(self._serial, lane, key, value, run, goal_slot, version, step)

    def held(self, lane: int) -> dict[Key, float]:
        """The entries this layer holds for ``lane`` (key -> value), in the order they were written."""

        return {key: op.value for key, op in self._held[lane].items()}

    def hold(self, lane: int, run: int, key: Key, priority: float, goal_slot: int, executive_version: int,
             program_step: int) -> None:
        """Write ``remember = remember_value(priority)`` on ``key`` for the run's step, unless it is held already."""

        value = remember_value(priority)
        if value <= 0.0:
            return
        current = self._held[lane].get(key)
        if current is not None and current.value == value and current.run == run:
            return
        op = self._op(lane, key, value, run, goal_slot, executive_version, program_step)
        self._held[lane][key] = op
        self._outbox[lane].append(op)

    def release(self, lane: int, run: int, keep: bool) -> None:
        """The program ``run`` of ``lane`` ended. Eject the entries it owns, unless it keeps them. Drop its retries."""

        self._retry[lane] = [op for op in self._retry[lane] if op.run != run]
        held = self._held[lane]
        for key, op in list(held.items()):
            if op.run != run:
                continue
            if keep:
                continue  # kept: the entry stays held, and no run owns it any more (run ids are never reused)
            del held[key]
            self._outbox[lane].append(
                self._op(lane, key, 0.0, run, op.goal_slot, op.executive_version, op.program_step)
            )

    def on_result(self, lane: int, op: WriteOp, reason: str | None, referenced: Iterable[Hashable]) -> bool:
        """Apply the result of a write this layer issued. Return True when the LLM must be asked.

        ``reason`` is None when the write applied. Otherwise it is the refusal's reason; ``"full"`` ejects the lowest
        unreferenced entry and queues ``op`` for a retry at the next decision. ``referenced`` holds the keys that live
        requests of ``lane`` refer to. When every held entry is referenced, nothing is ejected and True is returned.
        """

        if reason is None or op.value == 0.0:
            return False  # applied, or an eject: nothing to take back, nothing to retry
        held = self._held[lane]
        if held.get(op.key) is op:
            del held[op.key]  # not applied, so it is not held
        if reason != FULL_REASON:
            return False
        live = set(referenced)
        candidates = [(key, held_op) for key, held_op in held.items() if key not in live]
        if not candidates:
            return True
        key, victim = min(candidates, key=lambda item: item[1].value)
        del held[key]
        self._outbox[lane].append(
            self._op(lane, key, 0.0, victim.run, victim.goal_slot, victim.executive_version, victim.program_step)
        )
        self._retry[lane].append(op)
        return False

    def requeue(self, lane: int, run: int) -> None:
        """At a decision: the writes refused at the last one go out again, unless their run has ended or the key is
        held by a newer write."""

        pending, self._retry[lane] = self._retry[lane], []
        held = self._held[lane]
        for op in pending:
            if op.run != run or op.key in held:
                continue
            again = self._op(lane, op.key, op.value, op.run, op.goal_slot, op.executive_version, op.program_step)
            held[op.key] = again
            self._outbox[lane].append(again)

    def take(self, lane: int, limit: int) -> list[WriteOp]:
        """Remove and return up to ``limit`` writes of ``lane`` waiting to go out, oldest first."""

        ready, self._outbox[lane] = self._outbox[lane][:limit], self._outbox[lane][limit:]
        return ready
