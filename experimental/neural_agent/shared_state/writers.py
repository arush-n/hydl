"""Session-side writers of the scratchpad S_t (``shared_state.py``): belief, events, skills, decision.

The LLM's section is written by the executive harness (``executive/harness.py``, on each accepted
proposal). This module holds the writer the session loop (``session.py``) calls at the host cadence
and on each goal report. Design: ``docs/SYSTEM_ONE.md`` §8.10, one writer per section.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np

from experimental.neural_agent.contracts import EVENT_NAMES, GoalPayload, GoalReport, SharedBeliefState
from experimental.neural_agent.shared_state import (
    SECTION_WRITERS,
    BeliefSection,
    DecisionRecord,
    DecisionSection,
    EventsSection,
    GoalInForce,
    InformationRequest,
    SharedState,
    SkillsSection,
)


def information_request_record(item: Any, tick: int) -> InformationRequest:
    """Return the :class:`InformationRequest` for one entry of a host decider's ``information_requests``.

    Accepted forms: an InformationRequest (returned as is); the typed decider's dict with ``version``,
    ``slot``, ``when``, ``kind``, ``argument`` and ``confidence`` (``jev/typed_rule.py``); the executor's
    ``LaneInformationRequest`` with ``tick``, ``lane``, ``executive_version``, ``program_step``, ``verb`` and
    ``reason`` (``jev/programs.py``).

    Raises:
        TypeError: For any other form.
    """

    if isinstance(item, InformationRequest):
        return item
    if isinstance(item, Mapping):
        kind = item.get("kind")
        reason = (
            "the condition could not be read; write the typed condition (kind, argument) for this goal"
            if kind is None
            else "the reading is unsure; confirm it by writing the typed condition, or reword the goal"
        )
        return InformationRequest(
            tick=tick,
            source="typed_rule",
            executive_version=int(item["version"]),
            slot=item.get("slot"),
            lane=None,
            text=str(item.get("when", "")),
            kind=kind,
            argument=item.get("argument"),
            confidence=item.get("confidence"),
            reason=reason,
        )
    if hasattr(item, "verb") and hasattr(item, "reason") and hasattr(item, "program_step"):
        return InformationRequest(
            tick=int(getattr(item, "tick", tick)),
            source="program",
            executive_version=int(item.executive_version),
            slot=None,
            lane=int(item.lane),
            text=str(item.verb),
            kind=None,
            argument=None,
            confidence=None,
            reason=f"program step {int(item.program_step)} was rejected repeatedly: {item.reason}",
        )
    raise TypeError(f"not an information request: {type(item).__name__}")


class SessionScribe:
    """Writes what the session loop knows into S_t, as each section's owner.

    ``record_tick`` writes, at the host cadence: the belief section (when the step's belief is a
    SharedBeliefState; the ADK test stubs carry a bare array, and then the section is left alone),
    the events section (the step's active events per lane), the skills section's active skill per
    lane (outcomes and runtime summaries are kept), and the decision section (the goal in force per
    lane from the carry's GoalPayload, plus one DecisionRecord per lane whose skill changed on this
    tick). ``push_goal_report`` appends a report to the skills section's bounded outcomes; it is a
    ``ReportSink``. Every write is one document version.
    """

    def __init__(self, state: SharedState, skill_ids: Sequence[str]) -> None:
        if not isinstance(state, SharedState):
            raise TypeError(f"state must be a SharedState, got {type(state).__name__}")
        self._state = state
        self._skill_ids = tuple(str(skill) for skill in skill_ids)
        self.writes = 0

    @property
    def state(self) -> SharedState:
        return self._state

    def record_tick(
        self, tick: int, transition: Any, hierarchy_state: Any, information_requests: Sequence[Any] = ()
    ) -> None:
        """Write the belief, events, skills and decision sections for ``tick``.

        Args:
            tick: The session clock tick of this write.
            transition: The HierarchicalLoopTransition just recorded; ``transition.hierarchy_state``
                is the state before the step and ``transition.hierarchy_step`` the step.
            hierarchy_state: The carried state after the step and after any host decision.
            information_requests: What the host decider appended to its ``information_requests`` since
                the last write, in any of the forms :func:`information_request_record` accepts; they join
                the decision section's bounded list of questions to the LLM.
        """

        step = transition.hierarchy_step
        belief = step.belief_update.belief
        if isinstance(belief, SharedBeliefState):
            self._write("belief", BeliefSection.from_belief(belief), tick)
        events = np.asarray(step.goal_runtime.active_events, dtype=bool)
        if events.ndim != 2 or events.shape[1] != len(EVENT_NAMES):
            raise ValueError(f"active_events must be bool[B, {len(EVENT_NAMES)}], got shape {events.shape}")
        names = tuple(tuple(EVENT_NAMES[event] for event in np.flatnonzero(row)) for row in events)
        self._write("events", EventsSection(active=names), tick)
        payload = hierarchy_state.carry.goal.payload
        if not isinstance(payload, GoalPayload):
            return
        skills = self._names(payload.skill_index)
        previous = self._state.skills
        self._write(
            "skills",
            SkillsSection(
                active=skills,
                runtime=previous.runtime,
                outcomes=previous.outcomes,
                outcome_limit=previous.outcome_limit,
                statuses=previous.statuses,
                status_limit=previous.status_limit,
            ),
            tick,
        )
        slots = np.asarray(payload.goal_slot).astype(int)
        versions = np.asarray(payload.executive_version).astype(int)
        params = np.asarray(payload.params, dtype=np.float64)
        in_force = tuple(
            GoalInForce(
                skill_id=skills[lane],
                slot=int(slots[lane]),
                executive_version=int(versions[lane]),
                params=tuple(float(value) for value in params[lane]),
            )
            for lane in range(len(skills))
        )
        decision = self._state.decision
        recent = decision.recent
        before = transition.hierarchy_state.carry.goal.payload
        if isinstance(before, GoalPayload):
            changed = np.asarray(before.skill_index).astype(int) != np.asarray(payload.skill_index).astype(int)
            recent = recent + tuple(
                DecisionRecord(tick=tick, lane=int(lane), skill_id=skills[lane], replace=True)
                for lane in np.flatnonzero(changed)
            )
        questions = decision.information_requests + tuple(
            information_request_record(item, tick) for item in information_requests
        )
        self._write(
            "decision",
            DecisionSection(
                in_force=in_force,
                recent=recent,
                recent_limit=decision.recent_limit,
                information_requests=questions,
                information_limit=decision.information_limit,
            ),
            tick,
        )

    def push_goal_report(self, report: GoalReport) -> None:
        """Append ``report`` to the skills section's outcomes (the ``ReportSink`` protocol)."""

        previous = self._state.skills
        self._write(
            "skills",
            SkillsSection(
                active=previous.active,
                runtime=previous.runtime,
                outcomes=previous.outcomes + (report,),
                outcome_limit=previous.outcome_limit,
                statuses=previous.statuses,
                status_limit=previous.status_limit,
            ),
            report.tick,
        )

    def _names(self, skill_index: Any) -> tuple[str, ...]:
        index = np.asarray(skill_index).astype(int)
        return tuple(self._skill_ids[i] if 0 <= i < len(self._skill_ids) else f"skill_{i}" for i in index)

    def _write(self, section: str, value: Any, tick: int) -> None:
        self._state.write(section, value, writer=SECTION_WRITERS[section], tick=tick)
        self.writes += 1


__all__ = ["SessionScribe"]
