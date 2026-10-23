"""The writes section of S_t: the decision layer's structured writes into the agents' written entries, and the
results of those writes (MIDDLE_LAYER.md section 5; the owner's rulings of 2026-10-22).

Writer: the decision layer (``"jev"``), from its fast path, one record per write. The section holds the writes of the
decision just made, as ``requests`` holds the live requests. The skill layer applies them at the tick they name
(``shared_state/skill_writes.py``), and each write's result comes back in the skills section as a :class:`WriteStatus`.

A record is one write to a field of an agent's written entries (``skill_agent/state/written.py``): ``name`` (today
``remember``), a ``value``, and one key: an entity id or a world position, never both. The section refuses the shape:
a key that is neither or both, a malformed key, a non-finite value, a second record of one id, and more than
``WRITES_PER_AGENT_TICK`` records of one agent at one tick. The field's checks (its writer, its range) belong to the
skill layer's host check (``written.lower``); a refusal there comes back as a status, so the section does not refuse
them.
"""

from __future__ import annotations

import collections
import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping

from experimental.neural_agent.contracts import (
    MAX_GOALS,
    _finite_number,
    _non_negative_int,
    _text,
)
from experimental.neural_agent.executive.store import validate_against_schema

from .._common import WRITES_PER_AGENT_TICK, _instances
from ..schema import _WRITES_SECTION_SCHEMA

WRITE_WRITER: str = "decision"
"""The writer of every write record: the decision layer, which owns the written fields it writes."""
WRITE_KINDS: tuple[str, ...] = ("write_applied", "write_refused")
"""The two kinds of a write's status on the skills stream."""


# ---- writes section: written by the JEV (the decision layer) ----


@dataclass(frozen=True, kw_only=True)
class WriteRecord:
    """One write of the decision layer into an agent's written entries (MIDDLE_LAYER.md section 5).

    Attributes:
        id: Non-negative, monotonic per agent. It names the write in its status.
        agent: The agent (lane) the write is for.
        tick: The tick the write is issued at, and applied at.
        writer: ``"decision"``.
        name: The field, for example ``"remember"``. Checked by the skill layer, not here.
        value: The value, finite. Its range is checked by the skill layer, not here.
        entity: The entity id the write is about, or None. Exactly one of ``entity`` and ``place`` is given.
        place: The world position (x, y, z) the write is about, or None.
        goal_slot: The executive slot the write serves, or -1 for none.
        executive_version: The E_t version it was derived from.
        program_step: The step of the behaviour program it serves.

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If the writer is not ``"decision"``, the key is neither or both of entity and place, a number is
            not finite, or an index is out of range.
    """

    id: int
    agent: int
    tick: int
    writer: str = WRITE_WRITER
    name: str
    value: float
    entity: int | None = None
    place: tuple[float, float, float] | None = None
    goal_slot: int = -1
    executive_version: int = 0
    program_step: int = 0

    def __post_init__(self) -> None:
        """Check the fields and normalize the numbers to Python types."""

        for name in ("id", "agent", "tick", "executive_version", "program_step"):
            object.__setattr__(self, name, _non_negative_int(getattr(self, name), f"WriteRecord.{name}"))
        if _text(self.writer, "WriteRecord.writer") != WRITE_WRITER:
            raise ValueError(f"WriteRecord.writer must be {WRITE_WRITER!r}, got {self.writer!r}")
        _text(self.name, "WriteRecord.name")
        object.__setattr__(self, "value", _finite_number(self.value, "WriteRecord.value"))
        if (self.entity is None) == (self.place is None):
            raise ValueError("WriteRecord needs exactly one key: an entity id or a place")
        if self.entity is not None:
            if isinstance(self.entity, bool):
                raise TypeError("WriteRecord.entity must be an integer, got bool")
            object.__setattr__(self, "entity", _non_negative_int(self.entity, "WriteRecord.entity"))
        else:
            place = tuple(self.place)
            if len(place) != 3:
                raise ValueError(f"WriteRecord.place must hold 3 coordinates, got {len(place)}")
            object.__setattr__(
                self, "place", tuple(_finite_number(value, "WriteRecord.place") for value in place)
            )
        slot = self.goal_slot
        if isinstance(slot, bool):
            raise TypeError("WriteRecord.goal_slot must be an integer, got bool")
        slot = -1 if slot == -1 else _non_negative_int(slot, "WriteRecord.goal_slot")
        if slot >= MAX_GOALS:
            raise ValueError(f"WriteRecord.goal_slot must be -1 (no slot) or below {MAX_GOALS}, got {slot}")
        object.__setattr__(self, "goal_slot", slot)


@dataclass(frozen=True)
class WritesSection:
    """The decision layer's writes of this decision, one record per write (MIDDLE_LAYER.md section 5).

    Attributes:
        records: The records of every agent, stored sorted by (agent, id). A record id appears once per agent. At
            most ``WRITES_PER_AGENT_TICK`` (8) records of one agent share a tick: a ninth is refused, not dropped.
    """

    records: tuple[WriteRecord, ...] = ()

    def __post_init__(self) -> None:
        """Validate the records, sort them and check the per-agent, per-tick bound."""

        records = _instances(self.records, WriteRecord, "WritesSection.records")
        ordered = sorted(records, key=lambda record: (record.agent, record.id))
        keys = [(record.agent, record.id) for record in ordered]
        if len(set(keys)) != len(keys):
            raise ValueError("WritesSection.records repeats a write id for one agent")
        per_tick = collections.Counter((record.agent, record.tick) for record in ordered)
        worst = max(per_tick.values(), default=0)
        if worst > WRITES_PER_AGENT_TICK:
            raise ValueError(
                f"WritesSection holds {worst} writes of one agent at one tick; the limit is {WRITES_PER_AGENT_TICK}"
            )
        object.__setattr__(self, "records", tuple(ordered))

    def for_agent(self, agent: int) -> tuple[WriteRecord, ...]:
        """Return the records of ``agent``, oldest id first."""

        return tuple(record for record in self.records if record.agent == agent)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-shaped mapping: a place is a list of three numbers, an absent key is null."""

        return {"records": [_record_to_dict(record) for record in self.records]}

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> WritesSection:
        """Parse the JSON-shaped mapping, schema first."""

        validate_against_schema(mapping, _WRITES_SECTION_SCHEMA, "$.writes")
        return cls(records=tuple(_record_from_dict(item) for item in mapping["records"]))


def _record_to_dict(record: WriteRecord) -> dict[str, Any]:
    """Return the JSON form of a record."""

    return {
        "id": record.id,
        "agent": record.agent,
        "tick": record.tick,
        "writer": record.writer,
        "name": record.name,
        "value": record.value,
        "entity": record.entity,
        "place": None if record.place is None else list(record.place),
        "goal_slot": record.goal_slot,
        "executive_version": record.executive_version,
        "program_step": record.program_step,
    }


def _record_from_dict(item: Mapping[str, Any]) -> WriteRecord:
    """Build a record from its JSON form (already checked against the schema)."""

    place = item["place"]
    return WriteRecord(
        id=item["id"],
        agent=item["agent"],
        tick=item["tick"],
        writer=item["writer"],
        name=item["name"],
        value=item["value"],
        entity=item["entity"],
        place=None if place is None else tuple(place),
        goal_slot=item["goal_slot"],
        executive_version=item["executive_version"],
        program_step=item["program_step"],
    )


# ---- results of the writes, on the skills stream: written by the skill layer ----


@dataclass(frozen=True, kw_only=True)
class WriteStatus:
    """The result of one write, streamed back in the skills section (written by the skill layer).

    Attributes:
        agent: The agent the write was for.
        write_id: The id of the :class:`WriteRecord` this result is about.
        tick: The tick the write was applied at.
        kind: ``"write_applied"`` or ``"write_refused"``.
        reason: Empty for ``write_applied``. For ``write_refused``, the skill layer's host reason (``written.lower``:
            ``"unknown field"``, ``"wrong writer"``, ``"out of range"``, ``"bad key"``, ``"two keys"``,
            ``"too many writes this tick"``) or ``"full"`` when the device table had no free entry.

    Raises:
        ValueError: If the kind is unknown, or the reason is empty for a refusal or set for an application.
    """

    agent: int
    write_id: int
    tick: int
    kind: str
    reason: str = ""

    def __post_init__(self) -> None:
        """Check the kind and the reason against it."""

        for name in ("agent", "write_id", "tick"):
            object.__setattr__(self, name, _non_negative_int(getattr(self, name), f"WriteStatus.{name}"))
        if self.kind not in WRITE_KINDS:
            raise ValueError(f"WriteStatus.kind must be one of {WRITE_KINDS}, got {self.kind!r}")
        if not isinstance(self.reason, str):
            raise TypeError(f"WriteStatus.reason must be a string, got {type(self.reason).__name__}")
        if self.kind == "write_refused" and not self.reason:
            raise ValueError("a write_refused status needs its reason")
        if self.kind == "write_applied" and self.reason:
            raise ValueError("a write_applied status carries no reason")


def write_status_to_dict(status: WriteStatus) -> dict[str, Any]:
    """Return the JSON form of a write status."""

    return dataclasses.asdict(status)


def write_status_from_dict(item: Mapping[str, Any]) -> WriteStatus:
    """Build a write status from its JSON form (already checked against the schema)."""

    return WriteStatus(**item)
