"""The briefing: what the LLM reads from S_t, as typed rows before any rendering.

docs/SYSTEM_ONE.md section 8.10 gives the LLM a text view of S_t. The first view listed every section as
it is stored (numbers per lane, counts per status kind). This module builds the view the owner asked for
on 2026-10-18 instead: a briefing organised by what the LLM has to decide about, every block stamped with
the tick and version it describes, and a delta block that says what happened since the LLM's own last
version. Rendering (markdown, json, compact) is ``render.py``'s job; this module only decides the content,
so the three formats carry the same facts.

Contents, in order:
    stamp        the document version and tick
    executive    the LLM's own last version: goals (typed fields and words), facts, notes, budget
    npcs         one row per lane: the situation in words from the belief, the goal in force with its
                 typed condition, the current request with its latest status, the last outcome, the events
    questions    the decision layer's questions to the LLM (newest last)
    rejections   the newest rejected requests with their reasons
    status_counts  the status stream by kind
    delta        since the LLM's version: outcomes, questions, rejections and goal changes it has not seen

Rules: nothing in a row is invented; every value traces to one section field, and a feature the belief
does not know is named as unknown, never zero. Tick counters come out of the belief as counter over
TICKS_SATURATION and are multiplied back here. Bearing words come from the bearing's sine and cosine in
eight 45-degree sectors. Seconds are a rendering choice (``render.ViewOptions.tick_seconds``), so the
briefing keeps ticks.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Sequence

from experimental.neural_agent.contracts import TICKS_SATURATION, feature_index
from experimental.neural_agent.requests import STATUS_KINDS, ActionRequest, RequestStatus

from .._common import LLM_REJECTIONS
from ..document import SharedState
from ..sections.belief import BeliefLane
from ..sections.decision import InformationRequest
from ..sections.executive import GoalEntry

BEARINGS: tuple[str, ...] = (
    "ahead", "ahead-left", "left", "behind-left", "behind", "behind-right", "right", "ahead-right",
)
"""Eight sectors of 45 degrees from the bearing angle atan2(sin, cos); positive angles turn left."""

_FEATURE = {
    name: feature_index(name)
    for name in (
        "target_visible", "target_distance_blocks", "target_bearing_sin", "target_bearing_cos",
        "self_health_fraction", "target_health_fraction", "ticks_since_damage", "ticks_since_target_seen",
        "self_speed_blocks_per_tick", "closing_speed_blocks_per_tick",
    )
}


@dataclass(frozen=True)
class Situation:
    """One lane's belief, in the words a reader needs; None means the belief does not know the feature."""

    target_visible: bool | None
    distance_blocks: float | None
    bearing: str | None
    self_health: float | None
    target_health: float | None
    ticks_since_hit: int | None
    ticks_since_target_seen: int | None
    speed_blocks_per_tick: float | None
    closing_blocks_per_tick: float | None
    unknown: tuple[str, ...]


@dataclass(frozen=True)
class GoalRow:
    """One executive goal: the typed fields and the LLM's words."""

    slot: int
    skill_id: str
    priority: float
    condition: str
    horizon_seconds: float
    deadline_tick: int | None
    constraints: tuple[str, ...]
    params: tuple[float, ...]
    intent: str
    when: str
    note: str
    addressee: str
    objective: str | None


@dataclass(frozen=True)
class ObjectiveRow:
    """One executive objective with the goals that serve it and the outcomes of those goals' skills."""

    id: str
    text: str
    status: str
    deadline_tick: int | None
    goal_slots: tuple[int, ...]
    outcomes: dict[str, int]


@dataclass(frozen=True)
class InForceRow:
    """The goal in force in one lane."""

    skill_id: str
    slot: int
    executive_version: int
    condition: str
    confidence: float | None


@dataclass(frozen=True)
class RequestRow:
    """The lane's live request to the skills and its latest status."""

    request_id: int
    verb: str
    args: dict[str, Any]
    urgency: float
    issued_tick: int
    status: str | None
    status_tick: int | None
    signal: str
    reason: str


@dataclass(frozen=True)
class OutcomeRow:
    """One goal outcome report."""

    tick: int
    lane: int
    skill_id: str
    outcome: str
    duration_ticks: int
    executive_version: int
    detail: str


@dataclass(frozen=True)
class QuestionRow:
    """One question from the decision layer, with its reading spelled out."""

    tick: int
    source: str
    executive_version: int
    slot: int | None
    lane: int | None
    text: str
    reading: str
    reason: str


@dataclass(frozen=True)
class RejectionRow:
    tick: int
    lane: int
    request_id: int
    reason: str


@dataclass(frozen=True)
class NpcRow:
    """Everything about one lane (one NPC) that the LLM decides about."""

    lane: int
    situation: Situation
    goal: InForceRow | None
    request: RequestRow | None
    last_outcome: OutcomeRow | None
    events: tuple[str, ...]
    squads: tuple[str, ...]


@dataclass(frozen=True)
class ExecutiveRow:
    version: int
    issued_tick: int
    goals: tuple[GoalRow, ...]
    facts: tuple[str, ...]
    notes: str
    budget: dict[str, float]
    squads: tuple[tuple[str, tuple[int, ...]], ...]
    objectives: tuple[ObjectiveRow, ...]


@dataclass(frozen=True)
class Delta:
    """What happened after the LLM's own version was issued (tick > issued_tick)."""

    since_version: int
    since_tick: int
    outcomes: tuple[OutcomeRow, ...]
    questions: int
    rejections: int
    statuses: int
    lanes_with_goal_from_older_version: tuple[int, ...]


@dataclass(frozen=True)
class Briefing:
    version: int
    tick: int
    executive: ExecutiveRow
    npcs: tuple[NpcRow, ...]
    questions: tuple[QuestionRow, ...]
    rejections: tuple[RejectionRow, ...]
    status_counts: dict[str, int]
    delta: Delta

    def to_dict(self) -> dict[str, Any]:
        """The briefing as JSON-shaped data (tuples as lists, dataclasses as objects)."""

        return dataclasses.asdict(self)


def condition_text(entry: GoalEntry | None) -> str:
    """The typed condition of a goal as one token: ``always``, ``distance_below 3``, or the label hint."""

    if entry is None:
        return "always"
    if entry.condition is not None:
        kind, argument = entry.condition.kind, entry.condition.argument
        return kind if argument is None else f"{kind} {argument:g}"
    for label in entry.constraints:
        return label
    return "always"


def bearing_word(sin: float, cos: float) -> str:
    angle = math.atan2(sin, cos)
    sector = int(((angle + math.pi / 8) % (2 * math.pi)) // (math.pi / 4))
    return BEARINGS[sector % 8]


def situation(lane: BeliefLane) -> Situation:
    """Read one belief lane into words; unknown features stay None and are listed by name."""

    def value(name: str) -> float | None:
        index = _FEATURE[name]
        return float(lane.features[index]) if lane.known[index] else None

    def ticks(name: str) -> int | None:
        raw = value(name)
        return None if raw is None else int(round(raw * TICKS_SATURATION))

    visible = value("target_visible")
    sin, cos = value("target_bearing_sin"), value("target_bearing_cos")
    unknown = tuple(name for name in _FEATURE if value(name) is None)
    return Situation(
        target_visible=None if visible is None else visible > 0.5,
        distance_blocks=value("target_distance_blocks"),
        bearing=None if sin is None or cos is None else bearing_word(sin, cos),
        self_health=value("self_health_fraction"),
        target_health=value("target_health_fraction"),
        ticks_since_hit=ticks("ticks_since_damage"),
        ticks_since_target_seen=ticks("ticks_since_target_seen"),
        speed_blocks_per_tick=value("self_speed_blocks_per_tick"),
        closing_blocks_per_tick=value("closing_speed_blocks_per_tick"),
        unknown=unknown,
    )


def _args(request: ActionRequest) -> dict[str, Any]:
    args = request.args
    present: dict[str, Any] = {}
    if args.position is not None:
        present["position"] = [round(v, 2) for v in args.position]
    if args.direction is not None:
        present["direction"] = [round(v, 3) for v in args.direction]
    if args.entity is not None:
        present["entity"] = args.entity
    if args.cell is not None:
        present["cell"] = list(args.cell)
    if args.face is not None:
        present["face"] = args.face
    if any(v is not None for v in args.scalars):
        present["scalars"] = list(args.scalars)
    return present


def _request_row(request: ActionRequest, statuses: Sequence[RequestStatus]) -> RequestRow:
    latest = None
    for status in statuses:
        if status.agent == request.agent and status.request_id == request.request_id:
            latest = status  # statuses are newest last
    return RequestRow(
        request_id=request.request_id,
        verb=request.verb,
        args=_args(request),
        urgency=request.urgency,
        issued_tick=request.issued_tick,
        status=None if latest is None else latest.kind,
        status_tick=None if latest is None else latest.tick,
        signal="" if latest is None else latest.signal,
        reason="" if latest is None else latest.reason,
    )


def _reading(item: InformationRequest) -> str:
    if item.kind is None:
        reading = "no reading"
    else:
        reading = item.kind if item.argument is None else f"{item.kind} {item.argument:g}"
    if item.confidence is not None:
        reading += f" (confidence {item.confidence:.2f})"
    return reading


def briefing(state: SharedState, *, lanes: Sequence[int] | None = None, rejections: int = LLM_REJECTIONS) -> Briefing:
    """Build the briefing from one consistent snapshot of ``state``.

    Args:
        state: The scratchpad.
        lanes: The lanes to brief on (a squad's members), or None for every lane the belief has.
        rejections: How many of the newest rejected requests to keep.

    Raises:
        ValueError: If a requested lane does not exist.
    """

    version, tick, sections = state._snapshot()
    executive = sections["executive"]
    belief = sections["belief"]
    decision = sections["decision"]
    requests = sections["requests"]
    skills = sections["skills"]
    events = sections["events"]

    count = max(len(belief.lanes), len(decision.in_force), len(skills.active), len(events.active))
    chosen = tuple(range(count)) if lanes is None else tuple(int(lane) for lane in lanes)
    bad = [lane for lane in chosen if lane < 0 or lane >= count]
    if bad:
        raise ValueError(f"lanes {bad} do not exist; the scratchpad has {count}")

    readings = {r.slot: r for r in decision.readings if r.executive_version == executive.version}

    def understood(slot: int, goal: GoalEntry) -> str:
        reading = readings.get(slot)
        if reading is None:
            return condition_text(goal)
        text = reading.condition
        if reading.source == "words":
            text += f" (read from your words, confidence {reading.confidence:.2f})"
        return text

    goals = tuple(
        GoalRow(
            slot=slot,
            skill_id=goal.skill_id,
            priority=goal.priority,
            condition=understood(slot, goal),
            horizon_seconds=goal.horizon_seconds,
            deadline_tick=goal.deadline_tick,
            constraints=tuple(goal.constraints),
            params=tuple(goal.params),
            intent=goal.intent,
            when=goal.when,
            note=goal.note,
            addressee=goal.addressee,
            objective=goal.objective,
        )
        for slot, goal in enumerate(executive.goals)
    )
    objective_rows = []
    for item in executive.objectives:
        slots = tuple(slot for slot, goal in enumerate(executive.goals) if goal.objective == item.id)
        served = {executive.goals[slot].skill_id for slot in slots}
        counts: dict[str, int] = {}
        for report in skills.outcomes:
            if report.skill_id in served:
                counts[report.outcome.name] = counts.get(report.outcome.name, 0) + 1
        objective_rows.append(ObjectiveRow(item.id, item.text, item.status, item.deadline_tick, slots, counts))
    budget = executive.budget
    executive_row = ExecutiveRow(
        version=executive.version,
        issued_tick=executive.issued_tick,
        goals=goals,
        facts=tuple(executive.facts),
        notes=executive.notes,
        budget={
            "jev_latency_ms": budget.jev_latency_ms,
            "search_expansions": float(budget.search_expansions),
            "executive_calls_per_minute": budget.executive_calls_per_minute,
        },
        squads=tuple((squad.name, tuple(squad.lanes)) for squad in executive.squads),
        objectives=tuple(objective_rows),
    )

    def outcome_row(report: Any) -> OutcomeRow:
        return OutcomeRow(
            tick=report.tick, lane=report.lane, skill_id=report.skill_id, outcome=report.outcome.name,
            duration_ticks=report.duration_ticks, executive_version=report.executive_version, detail=report.detail,
        )

    live_by_lane: dict[int, ActionRequest] = {}
    for request in requests.live:
        live_by_lane[request.agent] = request  # the newest live request per lane wins
    npcs = []
    empty = Situation(None, None, None, None, None, None, None, None, None, tuple(_FEATURE))
    for lane in chosen:
        in_force = decision.in_force[lane] if lane < len(decision.in_force) else None
        goal_row = None
        if in_force is not None:
            entry = executive.goals[in_force.slot] if 0 <= in_force.slot < len(executive.goals) else None
            goal_row = InForceRow(
                skill_id=in_force.skill_id, slot=in_force.slot, executive_version=in_force.executive_version,
                condition=understood(in_force.slot, entry) if entry is not None else "idle", confidence=in_force.confidence,
            )
        request = live_by_lane.get(lane)
        outcomes_here = [report for report in skills.outcomes if report.lane == lane]
        npcs.append(
            NpcRow(
                lane=lane,
                situation=situation(belief.lanes[lane]) if lane < len(belief.lanes) else empty,
                goal=goal_row,
                request=None if request is None else _request_row(request, skills.statuses),
                last_outcome=outcome_row(outcomes_here[-1]) if outcomes_here else None,
                events=tuple(events.active[lane]) if lane < len(events.active) else (),
                squads=tuple(squad.name for squad in executive.squads if lane in squad.lanes),
            )
        )

    questions = tuple(
        QuestionRow(
            tick=item.tick, source=item.source, executive_version=item.executive_version, slot=item.slot,
            lane=item.lane, text=item.text, reading=_reading(item), reason=item.reason,
        )
        for item in decision.information_requests
    )
    rejected = [status for status in skills.statuses if status.kind == "rejected"]
    rejection_rows = tuple(
        RejectionRow(tick=status.tick, lane=status.agent, request_id=status.request_id, reason=status.reason)
        for status in rejected[-rejections:]
    )
    counts = {kind: sum(1 for status in skills.statuses if status.kind == kind) for kind in STATUS_KINDS}

    since = executive.issued_tick if executive.version > 0 else -1
    delta = Delta(
        since_version=executive.version,
        since_tick=executive.issued_tick,
        outcomes=tuple(outcome_row(report) for report in skills.outcomes if report.tick > since),
        questions=sum(1 for item in decision.information_requests if item.tick > since),
        rejections=sum(1 for status in rejected if status.tick > since),
        statuses=sum(1 for status in skills.statuses if status.tick > since),
        lanes_with_goal_from_older_version=tuple(
            lane for lane in chosen
            if lane < len(decision.in_force) and decision.in_force[lane].executive_version < executive.version
        ),
    )
    return Briefing(
        version=version, tick=tick, executive=executive_row, npcs=tuple(npcs), questions=questions,
        rejections=rejection_rows, status_counts=counts, delta=delta,
    )


__all__ = [
    "BEARINGS",
    "Briefing",
    "Delta",
    "ExecutiveRow",
    "GoalRow",
    "InForceRow",
    "NpcRow",
    "ObjectiveRow",
    "OutcomeRow",
    "QuestionRow",
    "RejectionRow",
    "RequestRow",
    "Situation",
    "bearing_word",
    "briefing",
    "condition_text",
    "situation",
]
