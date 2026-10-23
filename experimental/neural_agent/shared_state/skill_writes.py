"""The adapter between the writes section and the skill layer's write path (``skill_agent/state/written.py``).

The runtime glue. It is not imported by the rest of ``shared_state``, because it imports the lower layer (and jax)
by design. The skill layer's runtime calls it once per tick:

1. ``lists = writes_for_tick(document, tick, agents)`` gives each agent's :class:`Write` objects of the tick.
2. ``device, reasons = written.lower(lists)`` checks them on the host (the reasons are the host's, or None).
3. ``state = written.apply(state, device)`` applies them between ticks; ``state.applied`` says which took effect.
4. ``record_write_results(document, tick, reasons, state.applied)`` records each write's status in the skills section.

A write refused by the host, or refused by the device as "full" (no free entry), is a ``write_refused`` status; the
decision layer reads it at its next decision (``jev/programs.py``).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy as np

from experimental.neural_agent.shared_state.document import SharedState
from experimental.neural_agent.shared_state.sections.writes import WriteStatus
from experimental.neural_agent.skill_agent.state.written import Write

from ._common import SECTION_WRITERS

FULL_REASON: str = "full"
"""The reason of a write the device refused: the table had no free entry for its key."""


def _records_of(document: SharedState, tick: int, agents: int) -> list[list[Any]]:
    """The records of ``tick`` per agent, oldest id first. Refuses a record of an agent outside ``agents``."""

    lists: list[list[Any]] = [[] for _ in range(agents)]
    for record in document.writes.records:
        if record.tick != tick:
            continue
        if record.agent >= agents:
            raise ValueError(f"write {record.id} is for agent {record.agent}, but there are {agents} agents")
        lists[record.agent].append(record)
    return lists


def writes_for_tick(document: SharedState, tick: int, agents: int) -> list[list[Write]]:
    """Return one list of :class:`Write` per agent: the writes the decision layer issued at ``tick``.

    Args:
        document: The S_t document.
        tick: The tick whose writes to return (the tick the decision was made at).
        agents: The number of agents B; the result has B lists.

    Raises:
        ValueError: If ``agents`` is below 1, or a write of ``tick`` names an agent at or above ``agents``.
    """

    if agents < 1:
        raise ValueError(f"agents must be at least 1, got {agents}")
    return [
        [
            Write(writer=record.writer, name=record.name, value=record.value, entity=record.entity, place=record.place)
            for record in records
        ]
        for records in _records_of(document, tick, agents)
    ]


def record_write_results(
    document: SharedState, tick: int, reasons: Sequence[Sequence[str | None]], applied: Any
) -> int:
    """Record the status of each write of ``tick`` in the skills section; return the new document version.

    Args:
        document: The S_t document.
        tick: The tick the writes were applied at.
        reasons: Per agent, one entry per write of :func:`writes_for_tick` in the same order: the host's refusal
            reason from ``written.lower``, or None when the host accepted it.
        applied: bool[B, W]: the device's ``Written.applied`` after ``written.apply``. An accepted write takes the
            next slot of its agent, so its flag is ``applied[agent, slot]``.

    Returns:
        The document version after the skills section is written.

    Raises:
        ValueError: If ``reasons`` does not match the records of ``tick``, ``applied`` does not have a row per agent,
            or an agent has more accepted writes than slots.
    """

    agents = len(reasons)
    flags = np.asarray(applied, dtype=bool)
    if flags.ndim != 2 or flags.shape[0] != agents:
        raise ValueError(f"applied must be bool[{agents}, W], got shape {flags.shape}")
    statuses: list[WriteStatus] = []
    for agent, records in enumerate(_records_of(document, tick, agents)):
        agent_reasons = list(reasons[agent])
        if len(agent_reasons) != len(records):
            raise ValueError(
                f"agent {agent} has {len(records)} writes at tick {tick} but {len(agent_reasons)} host reasons"
            )
        slot = 0
        for record, reason in zip(records, agent_reasons):
            if reason is not None:
                statuses.append(WriteStatus(agent=agent, write_id=record.id, tick=tick, kind="write_refused", reason=reason))
                continue
            if slot >= flags.shape[1]:
                raise ValueError(f"agent {agent} has more accepted writes than the {flags.shape[1]} device slots")
            took = bool(flags[agent, slot])
            slot += 1
            if took:
                statuses.append(WriteStatus(agent=agent, write_id=record.id, tick=tick, kind="write_applied"))
            else:
                statuses.append(
                    WriteStatus(agent=agent, write_id=record.id, tick=tick, kind="write_refused", reason=FULL_REASON)
                )
    skills = document.skills
    updated = dataclasses.replace(skills, write_statuses=skills.write_statuses + tuple(statuses))
    return document.write("skills", updated, writer=SECTION_WRITERS["skills"], tick=tick)
