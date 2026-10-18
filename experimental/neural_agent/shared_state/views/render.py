"""Render a :class:`Briefing` for the LLM in one of three formats, behind one options object.

The formats carry the same facts (``briefing.py`` decides the content); they differ in shape and cost:
    markdown   headed sections and one table of NPCs: the default, easy to read and to quote back
    json       the briefing's dict, canonical (sorted keys, no spaces): for a model that answers in JSON
               and for machine checks that every rendered fact traces to a field
    compact    one line per block, ``key=value`` pairs: the cheapest in tokens, for long squads
``ViewOptions`` also sets the tick length (seconds are a rendering choice; the briefing keeps ticks), the
lanes to show (a squad), and whether the delta block is included. ``estimate_tokens`` is the budget
measure the ledger records: characters over four, the usual English rule of thumb, stated as an estimate
because no provider tokenizer runs here.

CLI, the information budget: ``python -m experimental.neural_agent.shared_state.views.render --lanes 8``
builds a filled scratchpad (the goal-language generator's executive, synthetic belief, requests, statuses,
outcomes, events) and prints characters and estimated tokens per format.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from .briefing import Briefing, NpcRow, Situation, briefing

ViewFormat = Literal["markdown", "json", "compact"]
FORMATS: tuple[str, ...] = ("markdown", "json", "compact")


@dataclass(frozen=True)
class ViewOptions:
    """How the LLM's view is rendered.

    Attributes:
        format: One of FORMATS.
        tick_seconds: Length of one tick, for the seconds shown beside tick counts (0.045 in the gen-5
            simulator, 1/30 in gen 7).
        lanes: The lanes to show, or None for all (a squad's members when the scratchpad is shared).
        delta: Include the block of what happened since the LLM's own version.
        rejections: How many of the newest rejected requests to show.
    """

    format: str = "markdown"
    tick_seconds: float = 0.045
    lanes: tuple[int, ...] | None = None
    delta: bool = True
    rejections: int = 5

    def __post_init__(self) -> None:
        if self.format not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}, got {self.format!r}")
        if not (isinstance(self.tick_seconds, (int, float)) and self.tick_seconds > 0):
            raise ValueError("tick_seconds must be positive")
        if self.rejections < 0:
            raise ValueError("rejections must be at least 0")


def estimate_tokens(text: str) -> int:
    """Characters over four, rounded up: the estimate the information budget records."""

    return math.ceil(len(text) / 4)


def _seconds(ticks: int | None, tick_seconds: float) -> str:
    return "unknown" if ticks is None else f"{ticks * tick_seconds:.1f} s ago"


def _fraction(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.2f}"


def situation_text(sit: Situation, tick_seconds: float) -> str:
    """One sentence per lane: target, distance and bearing, health, hits, sight."""

    if sit.target_visible is None and sit.distance_blocks is None:
        parts = ["target unknown"]
    else:
        seen = "visible" if sit.target_visible else "not visible"
        where = "" if sit.distance_blocks is None else f" {sit.distance_blocks:.1f} blocks"
        where += "" if sit.bearing is None else f" {sit.bearing}"
        parts = [f"target {seen}{where}"]
    parts.append(f"health self {_fraction(sit.self_health)} target {_fraction(sit.target_health)}")
    parts.append(f"last hit {_seconds(sit.ticks_since_hit, tick_seconds)}")
    parts.append(f"target last seen {_seconds(sit.ticks_since_target_seen, tick_seconds)}")
    if sit.closing_blocks_per_tick is not None:
        parts.append(f"closing {sit.closing_blocks_per_tick / tick_seconds:.1f} blocks/s")
    return "; ".join(parts)


def _goal_text(npc: NpcRow) -> str:
    if npc.goal is None:
        return "none"
    goal = npc.goal
    confidence = "" if goal.confidence is None else f", confidence {goal.confidence:.2f}"
    return f"{goal.skill_id} (slot {goal.slot}, E_t v{goal.executive_version}, {goal.condition}{confidence})"


def _request_text(npc: NpcRow) -> str:
    if npc.request is None:
        return "none"
    req = npc.request
    args = "" if not req.args else " " + json.dumps(req.args, separators=(",", ":"))
    status = "no status yet" if req.status is None else req.status
    if req.signal:
        status += f" ({req.signal})"
    if req.reason:
        status += f": {req.reason}"
    return f"{req.verb}#{req.request_id}{args}, urgency {req.urgency:.2f}, {status}"


def _outcome_text(npc: NpcRow, tick_seconds: float) -> str:
    if npc.last_outcome is None:
        return "none"
    out = npc.last_outcome
    detail = f" ({out.detail})" if out.detail else ""
    return f"{out.skill_id} {out.outcome} after {out.duration_ticks * tick_seconds:.1f} s at tick {out.tick}{detail}"


def to_markdown(b: Briefing, options: ViewOptions) -> str:
    ts = options.tick_seconds
    lines = [f"# Shared state, version {b.version}, tick {b.tick}", ""]
    ex = b.executive
    lines.append(f"## Your goals (E_t version {ex.version}, issued at tick {ex.issued_tick})")
    if not ex.goals:
        lines.append("none")
    else:
        lines.append("| slot | skill | priority | condition | horizon s | deadline tick | intent | when | note |")
        lines.append("|---|---|---:|---|---:|---|---|---|---|")
        for g in ex.goals:
            deadline = "none" if g.deadline_tick is None else str(g.deadline_tick)
            lines.append(
                f"| {g.slot} | {g.skill_id} | {g.priority:g} | {g.condition} | {g.horizon_seconds:g} | {deadline} "
                f"| {g.intent or '-'} | {g.when or '-'} | {g.note or '-'} |"
            )
    lines.append(f"facts: {'; '.join(ex.facts) if ex.facts else 'none'}")
    lines.append(f"notes: {ex.notes or 'none'}")
    lines.append("")
    lines.append(f"## NPCs (belief as of tick {b.tick}; unknown means not known, never zero)")
    if not b.npcs:
        lines.append("none")
    else:
        lines.append("| lane | situation | goal in force | current request | last outcome | events |")
        lines.append("|---:|---|---|---|---|---|")
        for npc in b.npcs:
            events = ", ".join(npc.events) if npc.events else "-"
            lines.append(
                f"| {npc.lane} | {situation_text(npc.situation, ts)} | {_goal_text(npc)} | {_request_text(npc)} "
                f"| {_outcome_text(npc, ts)} | {events} |"
            )
    lines.append("")
    lines.append("## Questions to you from the decision layer (newest last)")
    if not b.questions:
        lines.append("none")
    for q in b.questions:
        where = "no slot" if q.slot is None else f"goal slot {q.slot}"
        lane = "" if q.lane is None else f", lane {q.lane}"
        lines.append(f"- tick {q.tick}, E_t v{q.executive_version}, {where}{lane}, {q.source}: {q.text!r} read as {q.reading}. {q.reason}")
    lines.append("")
    lines.append(f"## Rejected requests (newest {options.rejections})")
    if not b.rejections:
        lines.append("none")
    for r in b.rejections:
        lines.append(f"- tick {r.tick}, lane {r.lane}, request {r.request_id}: {r.reason}")
    counts = ", ".join(f"{kind} {n}" for kind, n in b.status_counts.items() if n)
    lines.append(f"status stream: {counts or 'empty'}")
    if options.delta:
        d = b.delta
        lines.append("")
        lines.append(f"## Since your version {d.since_version} (tick {d.since_tick})")
        lines.append(
            f"{len(d.outcomes)} outcomes, {d.questions} questions, {d.rejections} rejections, {d.statuses} statuses"
            + (f"; lanes still on an older version: {list(d.lanes_with_goal_from_older_version)}" if d.lanes_with_goal_from_older_version else "")
        )
        for out in d.outcomes:
            detail = f" ({out.detail})" if out.detail else ""
            lines.append(f"- tick {out.tick}, lane {out.lane}: {out.skill_id} {out.outcome} after {out.duration_ticks * ts:.1f} s{detail}")
    return "\n".join(lines)


def to_json(b: Briefing, options: ViewOptions) -> str:
    payload = b.to_dict()
    payload["tick_seconds"] = options.tick_seconds
    if not options.delta:
        payload.pop("delta")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def to_compact(b: Briefing, options: ViewOptions) -> str:
    ts = options.tick_seconds
    ex = b.executive
    lines = [f"S_t v{b.version} t{b.tick}; E_t v{ex.version} issued t{ex.issued_tick}"]
    for g in ex.goals:
        words = " ".join(f"{k}={v!r}" for k, v in (("intent", g.intent), ("when", g.when), ("note", g.note)) if v)
        lines.append(f"goal {g.slot}: {g.skill_id} p={g.priority:g} if={g.condition} horizon={g.horizon_seconds:g}s{' ' + words if words else ''}")
    if ex.facts:
        lines.append("facts: " + "; ".join(ex.facts))
    if ex.notes:
        lines.append(f"notes: {ex.notes}")
    for npc in b.npcs:
        lines.append(
            f"lane {npc.lane}: {situation_text(npc.situation, ts)} | goal {_goal_text(npc)} | request {_request_text(npc)} "
            f"| last {_outcome_text(npc, ts)}" + (f" | events {','.join(npc.events)}" if npc.events else "")
        )
    for q in b.questions:
        lines.append(f"question t{q.tick} slot={q.slot} lane={q.lane} {q.source}: {q.text!r} -> {q.reading}; {q.reason}")
    for r in b.rejections:
        lines.append(f"rejected t{r.tick} lane {r.lane} req {r.request_id}: {r.reason}")
    counts = " ".join(f"{kind}={n}" for kind, n in b.status_counts.items() if n)
    lines.append(f"statuses: {counts or 'none'}")
    if options.delta:
        d = b.delta
        lines.append(
            f"since v{d.since_version} t{d.since_tick}: outcomes={len(d.outcomes)} questions={d.questions} "
            f"rejections={d.rejections} statuses={d.statuses}"
            + (f" older_version_lanes={list(d.lanes_with_goal_from_older_version)}" if d.lanes_with_goal_from_older_version else "")
        )
        for out in d.outcomes:
            lines.append(f"  t{out.tick} lane {out.lane}: {out.skill_id} {out.outcome} {out.duration_ticks * ts:.1f}s{(' ' + out.detail) if out.detail else ''}")
    return "\n".join(lines)


_RENDERERS = {"markdown": to_markdown, "json": to_json, "compact": to_compact}


def render(b: Briefing, options: ViewOptions | None = None) -> str:
    """Render ``b`` in ``options.format`` (markdown by default)."""

    options = options or ViewOptions()
    return _RENDERERS[options.format](b, options)


def view(state: Any, options: ViewOptions | None = None) -> str:
    """The LLM's view of ``state``: build the briefing for ``options.lanes`` and render it."""

    options = options or ViewOptions()
    return render(briefing(state, lanes=options.lanes, rejections=options.rejections), options)


# ---------------------------------------------------------------------------------------------- budget CLI


def filled_scratchpad(lanes: int, seed: int = 0) -> Any:
    """A scratchpad with every section populated, for the information budget (deterministic in ``seed``)."""

    import numpy as np

    from experimental.neural_agent.contracts import (
        BELIEF_FEATURE_COUNT, BELIEF_LATENT_WIDTH, GoalReport, SharedBeliefState, SkillOutcome, feature_index,
    )
    from experimental.neural_agent.executive.goal_language import REAL_CATALOGUE, all_wordings, sample_state
    from experimental.neural_agent.requests import ActionRequest, RequestArgs, RequestStatus

    from ..document import SharedState
    from ..sections.belief import BeliefSection
    from ..sections.decision import DecisionSection, GoalInForce, InformationRequest
    from ..sections.events import EventsSection
    from ..sections.requests import RequestsSection
    from ..sections.skills import SkillsSection

    rng = np.random.default_rng(seed)
    state = SharedState()
    section, _ = sample_state(rng, REAL_CATALOGUE, all_wordings(), version=3, issued_tick=40, hint_rate=0.5)
    state.write("executive", section, writer=state.writers["executive"], tick=40)
    features = np.zeros((lanes, BELIEF_FEATURE_COUNT), dtype=np.float32)
    known = np.ones((lanes, BELIEF_FEATURE_COUNT), dtype=np.bool_)
    features[:, feature_index("target_visible")] = rng.integers(0, 2, lanes)
    features[:, feature_index("target_distance_blocks")] = rng.uniform(2, 20, lanes)
    angle = rng.uniform(-np.pi, np.pi, lanes)
    features[:, feature_index("target_bearing_sin")] = np.sin(angle)
    features[:, feature_index("target_bearing_cos")] = np.cos(angle)
    features[:, feature_index("self_health_fraction")] = rng.uniform(0.2, 1.0, lanes)
    features[:, feature_index("target_health_fraction")] = rng.uniform(0.2, 1.0, lanes)
    features[:, feature_index("ticks_since_damage")] = rng.integers(0, 256, lanes) / 256.0
    features[:, feature_index("ticks_since_target_seen")] = rng.integers(0, 256, lanes) / 256.0
    known[:, feature_index("closing_speed_blocks_per_tick")] = False
    belief = SharedBeliefState(features, known, np.zeros((lanes, BELIEF_LATENT_WIDTH), np.float32))
    state.write("belief", BeliefSection.from_belief(belief), writer=state.writers["belief"], tick=60)
    skills = [goal.skill_id for goal in section.goals]
    in_force = tuple(
        GoalInForce(skill_id=skills[lane % len(skills)], slot=lane % len(skills), executive_version=3, confidence=0.8)
        for lane in range(lanes)
    )
    question = InformationRequest(
        tick=52, source="typed_rule", executive_version=3, slot=0, lane=None, text=section.goals[0].when or "when it is close",
        kind="distance_below", argument=3.0, confidence=0.6, reason="the reading is unsure; confirm it by writing the typed condition",
    )
    state.write("decision", DecisionSection(in_force=in_force, information_requests=(question,)), writer=state.writers["decision"], tick=60)
    requests = tuple(
        ActionRequest(
            request_id=100 + lane, agent=lane, issued_tick=55, expires_tick=70, verb="finish" if lane % 2 else "avoid",
            args=RequestArgs(entity=0), urgency=0.5 + 0.05 * lane, confidence=0.9, goal_slot=lane % len(skills), executive_version=3,
        )
        for lane in range(lanes)
    )
    state.write("requests", RequestsSection(live=requests), writer=state.writers["requests"], tick=60)
    statuses = []
    for lane in range(lanes):
        statuses.append(RequestStatus(agent=lane, request_id=100 + lane, kind="accepted", tick=55))
        statuses.append(RequestStatus(agent=lane, request_id=100 + lane, kind="running", tick=60))
    statuses.append(RequestStatus(agent=0, request_id=90, kind="rejected", tick=50, reason="entity_unknown: target not seen"))
    outcomes = tuple(
        GoalReport(tick=45 + lane, lane=lane, skill_id="pursuit", executive_version=2, outcome=SkillOutcome.TIMEOUT,
                   duration_ticks=220, detail="")
        for lane in range(lanes)
    )
    state.write(
        "skills",
        SkillsSection(active=tuple(g.skill_id for g in in_force), outcomes=outcomes, statuses=tuple(statuses)),
        writer=state.writers["skills"], tick=60,
    )
    state.write("events", EventsSection(active=tuple(("DAMAGE_RECEIVED",) if lane == 0 else () for lane in range(lanes))),
                writer=state.writers["events"], tick=60)
    return state


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m experimental.neural_agent.shared_state.views.render",
        description="The information budget of the LLM view: characters and estimated tokens per format.",
    )
    parser.add_argument("--lanes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", choices=FORMATS, default=None, help="also print one rendering")
    args = parser.parse_args(argv)
    print("lanes | format | characters | est. tokens (chars/4) | lines")
    for lanes in args.lanes:
        state = filled_scratchpad(lanes, args.seed)
        for fmt in FORMATS:
            text = view(state, ViewOptions(format=fmt))
            print(f"{lanes} | {fmt} | {len(text)} | {estimate_tokens(text)} | {text.count(chr(10)) + 1}")
        if args.show:
            print(view(state, ViewOptions(format=args.show)))
    return 0


__all__ = [
    "FORMATS",
    "ViewFormat",
    "ViewOptions",
    "estimate_tokens",
    "filled_scratchpad",
    "render",
    "situation_text",
    "to_compact",
    "to_json",
    "to_markdown",
    "view",
]


if __name__ == "__main__":
    raise SystemExit(main())
