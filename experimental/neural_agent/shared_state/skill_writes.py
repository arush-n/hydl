"""The adapter between the writes section and the skill layer's write path (``skill_agent/state/written.py``).

The runtime glue. It is not imported by the rest of ``shared_state``, because it imports the lower layer (and jax)
by design. The skill layer's runtime calls it once per tick:

1. ``lists = writes_for_tick(document, tick, agents)`` gives each agent's :class:`Write` objects of the tick.
2. ``device, reasons = written.lower(lists)`` checks them on the host (the reasons are the host's, or None).
3. ``state = written.apply(state, device)`` applies them between ticks; ``state.applied`` says which took effect.
4. ``record_write_results(document, tick, reasons, state.applied)`` records each write's status in the skills section.

A write refused by the host, or refused by the device as "full" (no free entry), is a ``write_refused`` status; the
decision layer reads it at its next decision (``jev/programs.py``).

Every tick the state tick also takes the requests that ended (``state_tick.tick(outcomes=, outcome_verbs=)``, the
skill layer's event log, step 4): :func:`outcomes_for_tick` builds both int32[B, 4] arrays from the tick's terminal
statuses in the skills section (``done`` with its ``SkillOutcome``, or ``aborted``).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from experimental.neural_agent.contracts import SkillOutcome
from experimental.neural_agent.shared_state.document import SharedState
from experimental.neural_agent.shared_state.sections.skills import AgentEntities, KnownEntity
from experimental.neural_agent.shared_state.sections.writes import WriteStatus
from experimental.neural_agent.skill_agent.state.written import Write

from ._common import KNOWN_ENTITY_LIMIT, SECTION_WRITERS

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


def _store_skills(
    document: SharedState, tick: int, statuses: Sequence[WriteStatus], known_entities: tuple[AgentEntities, ...] | None
) -> int:
    """Write the skills section at ``tick``: the statuses appended, and ``known_entities`` replaced when given."""

    skills = document.skills
    changes: dict[str, Any] = {"write_statuses": skills.write_statuses + tuple(statuses)}
    if known_entities is not None:
        changes["known_entities"] = known_entities
    return document.write("skills", dataclasses.replace(skills, **changes), writer=SECTION_WRITERS["skills"], tick=tick)


def record_write_results(
    document: SharedState, tick: int, reasons: Sequence[Sequence[str | None]], applied: Any,
    known_entities: tuple[AgentEntities, ...] | None = None,
) -> int:
    """Record the status of each write of ``tick`` in the skills section; return the new document version.

    Args:
        document: The S_t document.
        tick: The tick the writes were applied at.
        reasons: Per agent, one entry per write of :func:`writes_for_tick` in the same order: the host's refusal
            reason from ``written.lower``, or None when the host accepted it.
        applied: bool[B, W]: the device's ``Written.applied`` after ``written.apply``. An accepted write takes the
            next slot of its agent, so its flag is ``applied[agent, slot]``.
        known_entities: The lanes' known entities of ``tick`` (``AgentEntities``, one per lane), written into the same
            document version when given.

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
    return _store_skills(document, tick, statuses, known_entities)


OUTCOME_SLOTS: int = 4
"""Columns of the outcome arrays: the combiner's active-request table, 4 per agent."""
OUTCOME_NONE, OUTCOME_DONE, OUTCOME_FAILED, OUTCOME_ABORTED = 0, 1, 2, 3
UNMAPPED_VERB: int = -1
"""``outcome_verbs`` for an ended request whose verb is not in ``verb_index`` (0 is a real verb, ``walk_to``)."""
_DONE_CODES = {
    SkillOutcome.COMPLETED: OUTCOME_DONE,
    SkillOutcome.FAILED: OUTCOME_FAILED,
    SkillOutcome.TIMEOUT: OUTCOME_FAILED,
    SkillOutcome.INTERRUPTED: OUTCOME_ABORTED,
}


def outcomes_for_tick(
    document: SharedState,
    tick: int,
    agents: int,
    verb_index: Any,
    verbs: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(outcomes, outcome_verbs)``, int32[B, 4] each, for the requests that ended at ``tick``.

    A column per ended request of the agent, in request-id order: 1 done (``done`` with ``COMPLETED``), 2 failed
    (``done`` with ``FAILED`` or ``TIMEOUT``), 3 aborted (``aborted``, or ``done`` with ``INTERRUPTED``); 0 where nothing
    ended. ``outcome_verbs`` holds the ended request's verb index beside it (0 where nothing ended, -1 for a verb
    outside ``verb_index``).

    Args:
        document: The S_t document; its skills section holds the statuses.
        tick: The tick whose ended requests to report.
        agents: The number of agents B.
        verb_index: Maps a verb (a skill id) to the index the state tick logs: the skill layer's neural-module vocabulary,
            ``{name: VERBS.index(name)}`` with ``skill_agent.neural_skills.spec.VERBS`` (append-only). A verb outside
            it is logged as ``UNMAPPED_VERB`` (-1).
        verbs: Optional mapping ``(agent, request_id) -> verb`` for requests no longer in the requests section; by
            default the verb is read from ``document.requests.live``.

    Raises:
        ValueError: If more than ``OUTCOME_SLOTS`` requests of one agent end on one tick, a status names an agent
            outside ``agents``, or an ended request's verb is unknown.
    """

    if agents < 1:
        raise ValueError(f"agents must be at least 1, got {agents}")
    known = {(request.agent, request.request_id): request.verb for request in document.requests.live}
    if verbs is not None:
        known.update({(int(a), int(r)): v for (a, r), v in dict(verbs).items()})
    ended: list[list[tuple[int, int, str]]] = [[] for _ in range(agents)]
    for status in document.skills.statuses:
        if status.tick != tick or status.kind not in ("done", "aborted"):
            continue
        if not 0 <= status.agent < agents:
            raise ValueError(f"status of request {status.request_id} names agent {status.agent}; there are {agents}")
        code = OUTCOME_ABORTED if status.kind == "aborted" else _DONE_CODES[SkillOutcome(status.outcome)]
        verb = known.get((status.agent, status.request_id))
        if verb is None:
            raise ValueError(f"request {status.request_id} of agent {status.agent} ended at tick {tick}, verb unknown")
        ended[status.agent].append((status.request_id, code, verb))
    index = dict(verb_index)
    outcomes = np.zeros((agents, OUTCOME_SLOTS), dtype=np.int32)
    outcome_verbs = np.zeros((agents, OUTCOME_SLOTS), dtype=np.int32)
    for agent, rows in enumerate(ended):
        if len(rows) > OUTCOME_SLOTS:
            raise ValueError(f"agent {agent} ended {len(rows)} requests at tick {tick}; the table has {OUTCOME_SLOTS}")
        for column, (_, code, verb) in enumerate(sorted(rows)):
            outcomes[agent, column] = code
            outcome_verbs[agent, column] = int(index.get(verb, UNMAPPED_VERB))
    return outcomes, outcome_verbs


@dataclasses.dataclass
class StateTickBinding:
    """The skill layer's state tick, bound to a session (``session.run_session(state_binding=)``, task #136).

    The session calls :meth:`on_tick` every tick, after the decision and the combiner's statuses of that tick are in
    S_t. On a tick the host decided, the decision's writes go in (:func:`writes_for_tick`); every tick, the requests that
    ended (:func:`outcomes_for_tick`). The device side is the skill layer's
    ``skill_agent.runtime.state_tick.step_lanes``; after a tick with writes, their results are recorded
    (:func:`record_write_results`).

    Attributes:
        trunk: The skill layer's trunk.
        state: The agents' state (``skill_agent.state.layer.AgentState``); replaced every tick.
        observe: ``observe(transition) -> (observation, last_action, reset, respawn)``: what the trunk reads this tick,
            int32[B, 21] factors sent last tick, bool[B] new episode, bool[B] respawned (None for none). Supplied by
            the caller: only the skill layer knows its observation.
        verb_index: Maps a verb (skill id) to the index the event log records (the skill layer's vocabulary).
        step_kwargs: The senses' inputs passed through to ``layer.step`` (tables, light, action_surface,
            block_reach_blocks).
        view: The last tick's ``StateView`` (None before the first tick).
        own_position: ``own_position(transition)`` gives the agents' positions, float[B, 3]. When given, every tick writes
            each lane's known entities (``known_entities``); None (the default) writes none.
        tick_seconds: The tick's length in seconds, the producer's ``dt_seconds`` (0.045 by default).
    """

    trunk: Any
    state: Any
    observe: Callable[[Any], tuple[Any, Any, Any, Any]]
    verb_index: Mapping[str, int]
    step_kwargs: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    view: Any = None
    own_position: Callable[[Any], Any] | None = None
    tick_seconds: float = 0.045
    # (agent, request_id) -> verb of every request seen live: a decision can drop a request from the requests section on
    # the tick its aborted status arrives, so the verb is remembered from the ticks it was live. Pruned once it ends.
    _verbs: dict[tuple[int, int], str] = dataclasses.field(default_factory=dict, repr=False)

    def on_tick(self, document: SharedState, tick: int, decided: bool, transition: Any) -> None:
        """Run one state tick for ``tick``; ``decided`` is True when the host decision model ran on this tick."""

        from experimental.neural_agent.skill_agent.runtime.state_tick import known_entities, step_lanes

        agents = int(np.asarray(self.state.entity_ids).shape[0])
        lists = writes_for_tick(document, tick, agents) if decided else None
        if lists is not None and not any(lists):
            lists = None  # no write this decision: entries persist, nothing to send or record
        for request in document.requests.live:
            self._verbs[(int(request.agent), int(request.request_id))] = request.verb
        outcomes, outcome_verbs = outcomes_for_tick(document, tick, agents, self.verb_index, verbs=self._verbs)
        for status in document.skills.statuses:
            if status.tick == tick and status.kind in ("done", "aborted"):
                self._verbs.pop((int(status.agent), int(status.request_id)), None)
        observation, last_action, reset, respawn = self.observe(transition)
        step = dict(self.step_kwargs)
        step.setdefault("tick_seconds", self.tick_seconds)  # the skill layer's rules tick at the same length (VIRTUAL_HUMAN.md 27.7)
        if respawn is not None:
            step["respawn"] = respawn
        self.view, self.state, reasons = step_lanes(
            self.trunk, self.state, observation, last_action=last_action, reset=reset, write_lists=lists,
            outcomes=outcomes, outcome_verbs=outcome_verbs, **step,
        )
        known = None
        if self.own_position is not None:
            lanes = known_entities(self.state, own_position=self.own_position(transition), dt_seconds=self.tick_seconds)
            known = tuple(
                AgentEntities(
                    agent=agent, tick=tick, entities=tuple(KnownEntity(**item) for item in lane[:KNOWN_ENTITY_LIMIT])
                )
                for agent, lane in enumerate(lanes)
            )
        if lists is not None:
            record_write_results(document, tick, reasons, self.state.written.applied, known_entities=known)
        elif known is not None:
            _store_skills(document, tick, (), known)
