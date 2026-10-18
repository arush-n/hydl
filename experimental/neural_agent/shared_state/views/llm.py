"""The text view the LLM reads from S_t.

Split out of the single ``shared_state.py`` on 2026-10-18 without changes; the package docstring
(``shared_state/__init__.py``) is the design record.
"""

from __future__ import annotations




from experimental.neural_agent.contracts import (
    BELIEF_FEATURE_NAMES,
)
from experimental.neural_agent.requests import STATUS_KINDS

from .._common import (
    LLM_REJECTIONS,
    _TEXT_FIELDS,
)
from ..sections.executive import (
    ExecutiveSection,
)
from ..sections.belief import (
    BeliefSection,
)
from ..sections.decision import (
    DecisionSection,
)
from ..sections.requests import (
    RequestsSection,
)
from ..sections.skills import (
    SkillsSection,
)
from ..sections.events import (
    EventsSection,
)
from ..document import (
    SharedState,
)


def _number(value: float) -> str:
    """Format a number compactly for the LLM view (three significant digits)."""

    return f"{value:.3g}"


def view_for_llm(state: SharedState) -> str:
    """Render S_t as compact text for the LLM.

    The text holds the LLM's own last goals (typed fields and words), facts, budget and notes; the
    belief per lane (unknown features named as unknown; the latent only as its norm); the decision
    in force per lane with its confidence and entropy; the live requests per agent; the count of
    each status kind in the status stream and the newest ``LLM_REJECTIONS`` rejections with their
    reasons; the skill outcomes it has not seen (tick after its E_t's ``issued_tick``, or all of
    them before its first version); and the events of the last window. No array is dumped.
    """

    version, tick, sections = state._snapshot()
    executive: ExecutiveSection = sections["executive"]
    belief: BeliefSection = sections["belief"]
    decision: DecisionSection = sections["decision"]
    requests: RequestsSection = sections["requests"]
    skills: SkillsSection = sections["skills"]
    events: EventsSection = sections["events"]
    budget = executive.budget
    lines = [
        f"Shared state S_t, version {version}, tick {tick}.",
        f"Your executive section: E_t version {executive.version}, issued at tick {executive.issued_tick}.",
    ]
    if not executive.goals:
        lines.append("  goals: none")
    for slot, goal in enumerate(executive.goals):
        deadline = "none" if goal.deadline_tick is None else str(goal.deadline_tick)
        lines.append(
            f"  goal {slot}: {goal.skill_id} priority={goal.priority:g} params={list(goal.params)} "
            f"constraints={list(goal.constraints)} horizon_seconds={goal.horizon_seconds:g} deadline_tick={deadline}"
        )
        lines.extend(f"    {key}: {getattr(goal, key)}" for key in _TEXT_FIELDS if getattr(goal, key))
    lines.append(f"  facts: {'; '.join(executive.facts) if executive.facts else 'none'}")
    lines.append(
        f"  budget: jev_latency_ms={budget.jev_latency_ms:g} search_expansions={budget.search_expansions} "
        f"executive_calls_per_minute={budget.executive_calls_per_minute:g}"
    )
    lines.append(f"  notes: {executive.notes or 'none'}")

    lines.append("Belief per lane (unknown means not published; it is not zero):")
    if not belief.lanes:
        lines.append("  none")
    for lane, row in enumerate(belief.lanes):
        values = " ".join(
            f"{name}={_number(value) if known else 'unknown'}"
            for name, value, known in zip(BELIEF_FEATURE_NAMES, row.features, row.known)
        )
        lines.append(f"  lane {lane}: {values} latent_norm={_number(row.latent_norm)}")

    lines.append("Decision in force per lane:")
    if not decision.in_force:
        lines.append("  none")
    for lane, goal in enumerate(decision.in_force):
        confidence = "unknown" if goal.confidence is None else _number(goal.confidence)
        entropy = "unknown" if goal.entropy is None else _number(goal.entropy)
        lines.append(
            f"  lane {lane}: {goal.skill_id} slot={goal.slot} executive_version={goal.executive_version} "
            f"confidence={confidence} entropy={entropy}"
        )

    if decision.information_requests:
        lines.append(
            f"Questions from the decision layer (newest last, {len(decision.information_requests)} kept of "
            f"{decision.information_limit}); answer by writing the typed condition or rewording the goal:"
        )
        for item in decision.information_requests:
            where = "no slot" if item.slot is None else f"goal slot {item.slot}"
            lane = "" if item.lane is None else f", lane {item.lane}"
            if item.kind is None:
                reading = "no reading"
            else:
                reading = f"read as {item.kind}" + ("" if item.argument is None else f" {_number(item.argument)}")
            if item.confidence is not None:
                reading += f" (confidence {_number(item.confidence)})"
            lines.append(
                f"  tick {item.tick}, E_t version {item.executive_version}, {where}{lane}, {item.source}: "
                f"{item.text!r}: {reading}. {item.reason}"
            )

    agents = sorted({request.agent for request in requests.live})
    per_agent = ", ".join(f"agent {agent}: {len(requests.for_agent(agent))}" for agent in agents)
    lines.append(f"Live requests to the skills: {len(requests.live)}" + (f" ({per_agent})." if agents else "."))
    counts = ", ".join(
        f"{kind} {sum(1 for status in skills.statuses if status.kind == kind)}" for kind in STATUS_KINDS
    )
    lines.append(
        f"Request statuses in the stream ({len(skills.statuses)} kept, newest {skills.status_limit}): {counts}."
    )
    rejections = [status for status in skills.statuses if status.kind == "rejected"][-LLM_REJECTIONS:]
    lines.append("Last rejections (newest last):")
    if not rejections:
        lines.append("  none")
    lines.extend(
        f"  tick {status.tick} agent {status.agent} request {status.request_id}: {status.reason}"
        for status in rejections
    )

    unseen = [
        report
        for report in skills.outcomes
        if executive.version == 0 or report.tick > executive.issued_tick
    ]
    lines.append(f"Skill outcomes since your version {executive.version}:")
    if not unseen:
        lines.append("  none")
    for report in unseen:
        detail = f": {report.detail}" if report.detail else ""
        lines.append(
            f"  tick {report.tick} lane {report.lane}: {report.skill_id} {report.outcome.name} "
            f"after {report.duration_ticks} ticks (E_t version {report.executive_version}){detail}"
        )

    lines.append(f"Events in the last {events.window_ticks} ticks:")
    seen = [(lane, names) for lane, names in enumerate(events.active) if names]
    if not seen:
        lines.append("  none")
    lines.extend(f"  lane {lane}: {', '.join(names)}" for lane, names in seen)
    return "\n".join(lines)


