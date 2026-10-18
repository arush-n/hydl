"""The shared scratchpad S_t: one versioned document between the LLM, the JEV and the skills.

``docs/SYSTEM_ONE.md`` section 8.10, Consequence 3. The owner (2026-10-17): "think of the state as
the scratchpad between all 3 layers where all models pass their information in between (like a
shared memory) there are different parts for each component ... we have to find an optimal way to
structure it such that each model gets the information it needs and has the proper context to make
better decisions". This module composes the typed records of ``contracts.py`` (``ExecutiveState``,
``GoalSpec``, ``ComputeBudget``, ``SharedBeliefState``, ``GoalReport``, ``EVENT_NAMES``) into that
document, with its JSON Schema and one view per reader. It is not wired into the executive harness
or the session yet; until it is, those keep using ``ExecutiveStateStore`` and ``ReportChannel``.

Sections. Each has exactly one writer. ``SharedState.write`` refuses any other writer with
``PermissionError``, and every accepted write bumps the document version by one::

    section     writer      content
    executive   "llm"       goals as JSON: GoalSpec's typed fields plus ``intent`` and ``when``
                            (and an optional ``note``) in the LLM's words; facts; budget; ``notes``,
                            the LLM's working notes ("the idea it has")
    belief      "belief"    per lane: the ten features keyed by BELIEF_FEATURE_NAMES, their known
                            flags, and a latent summary (L2 norm and the first 8 values)
    decision    "jev"       per lane: the goal in force (skill, params, slot, E_t version) with
                            confidence and entropy when known; the last N decisions (N = 16)
    requests    "jev"       the live ``requests.ActionRequest`` records per agent (lane), sorted by
                            request id, at most REQUEST_LIMIT (16) per agent (section 9.10)
    skills      "skills"    per lane: the active skill; runtime summaries (free-form JSON from the
                            dispatcher); the outcomes as GoalReports (bounded, newest last); the
                            status stream, ``requests.RequestStatus`` records (bounded, newest last)
    events      "runtime"   per lane: the DecisionEvent names active in the last window

The decision layer writes two sections and the dispatcher (the lower layer's combiner) writes the
status stream inside ``skills``. Each section still has exactly one writer.

Who reads what, and why. Each model gets the information it needs and nothing more:

* The LLM reads text. ``view_for_llm`` renders its own last goals, facts, budget and notes, the
  belief per lane, the decision in force with its confidence, the live request count, the status
  counts of the stream (accepted, rejected, ...) with the last LLM_REJECTIONS (5) rejections and
  their reasons, the skill outcomes it has not seen yet, and the recent events. The LLM thinks
  slowly and in words, so it gets a compact rendering: no array is dumped, and the latent appears
  only as its norm.
* The JEV reads typed arrays plus per-goal text. ``view_for_jev`` gives E_t lowered exactly as
  ``contracts.executive_arrays`` lowers it, plus the ``intent`` and ``when`` of each goal aligned
  with those slots (the words its text path encodes per goal, section 8.10 Consequence 1). It also
  gives the belief arrays, the current skill per lane, and per lane the live request ids with the
  latest status of each (the input of its correspondence rules, section 9.10). The JEV decides at
  about 2 Hz under a hard budget, so it gets arrays for a compiled pass and only the words it must
  interpret. A goal's ``note`` is the LLM's own remark and stays out of this view.
* The skills read the goal only. ``view_for_skills`` gives the goal payload fields per lane (skill,
  params, E_t version, slot). A skill reads B_t itself from the hierarchy step and never sees words.

Formulas
--------
::

    latent_norm_b   = sqrt(sum_j z_bj^2)       over the BELIEF_LATENT_WIDTH latent values of lane b
    latent_prefix_b = (z_b0, ..., z_b7)
    bounded list    = the newest ``limit`` entries; the oldest are dropped first (as in ReportChannel)
    live requests   = per agent, the ``limit`` highest request ids (ids are monotonic per agent)
    latest status   : of request (a, i), the last entry of the status stream with agent a and id i
    unseen outcome  : report.tick > executive.issued_tick, or every report while executive.version == 0
    write           : version' = version + 1 and tick' = the tick of the write, under one lock

Bounds. REQUEST_LIMIT = 16 = L x M, a whole behaviour program per agent: L = 4 steps (section 9.9)
times M = 4 concurrent granted requests (section 9.10). STATUS_HISTORY = 256 covers one cadence
interval (11 ticks) of 8 agents x 2 live requests at one status per request per tick (176
entries), so no status falls out of the stream before the middle layer's next read at that load.
How often the lower layer emits a status (every tick, or on change) is the lower layer's choice.

Citations: ``docs/SYSTEM_ONE.md`` section 8.10 (the section table and the rules) and section 9.10
(the request stream); ADR-001 D3 (E_t is versioned) and section 5, seam 3 (outcomes reported
upward). The schema is JSON Schema draft
2020-12 (https://json-schema.org/draft/2020-12/schema). It is checked with the manual validator of
``executive/store.py`` (``validate_against_schema``), so it uses only the keywords that validator
enforces. Semantic rules the schema cannot state (event names, outcome names, the fixed writers)
are checked when a section is built.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from experimental.neural_agent.contracts import (
    BELIEF_FEATURE_COUNT,
    BELIEF_FEATURE_NAMES,
    BELIEF_LATENT_WIDTH,
    EVENT_NAMES,
    EXECUTIVE_STATE_JSON_SCHEMA,
    INT32_MAX,
    MAX_GOALS,
    PARAM_WIDTH,
    SKILL_IDLE,
    ComputeBudget,
    ExecutiveArrays,
    ExecutiveState,
    GoalReport,
    GoalSpec,
    SharedBeliefState,
    SkillOutcome,
    _finite_number,
    _finite_tuple,
    _non_negative_int,
    _text,
    _text_tuple,
    executive_arrays,
)
from experimental.neural_agent.executive.store import validate_against_schema
from experimental.neural_agent.requests import STATUS_KINDS, ActionRequest, RequestArgs, RequestStatus

SECTION_NAMES: tuple[str, ...] = ("executive", "belief", "decision", "requests", "skills", "events")
"""The six sections of S_t, in document order."""
SECTION_WRITERS: Mapping[str, str] = MappingProxyType(
    {
        "executive": "llm",
        "belief": "belief",
        "decision": "jev",
        "requests": "jev",
        "skills": "skills",
        "events": "runtime",
    }
)
"""The one writer of each section. Fixed: ``SharedState.write`` refuses any other writer."""
LATENT_PREFIX_WIDTH: int = 8
"""Latent values kept per lane in the belief summary: the first 8 of BELIEF_LATENT_WIDTH."""
DECISION_HISTORY: int = 16
"""Default bound N of ``DecisionSection.recent``."""
OUTCOME_HISTORY: int = 16
"""Default bound of ``SkillsSection.outcomes``."""
REQUEST_LIMIT: int = 16
"""Default bound of ``RequestsSection.live`` per agent: L = 4 program steps x M = 4 concurrent requests."""
STATUS_HISTORY: int = 256
"""Default bound of ``SkillsSection.statuses`` (the module docstring gives the sizing)."""
LLM_REJECTIONS: int = 5
"""Number of newest rejections that ``view_for_llm`` shows with their reasons."""
EVENT_WINDOW_TICKS: int = 11
"""Default event window: the session's host decision cadence (``SessionConfig.host_cadence_ticks``),
0.495 s at 0.045 s per tick."""

_SPEC_FIELDS: tuple[str, ...] = tuple(item.name for item in dataclasses.fields(GoalSpec))
_TEXT_FIELDS: tuple[str, ...] = ("intent", "when", "note")


def _string(value: Any, name: str) -> str:
    """Return ``value`` when it is a string. Empty is allowed: the LLM's words are optional."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    return value


def _positive_int(value: Any, name: str) -> int:
    """Return ``value`` as an int of at least 1."""

    number = _non_negative_int(value, name)
    if number < 1:
        raise ValueError(f"{name} must be at least 1, got {number}")
    return number


def _bounded(value: Any, name: str, maximum: float | None = None) -> float:
    """Return ``value`` as a finite float in ``[0, maximum]`` (no upper bound when ``maximum`` is None)."""

    number = _finite_number(value, name)
    if number < 0.0 or (maximum is not None and number > maximum):
        upper = "inf" if maximum is None else f"{maximum:g}"
        raise ValueError(f"{name} must be in [0, {upper}], got {number}")
    return number


def _optional_bounded(value: Any, name: str, maximum: float | None = None) -> float | None:
    """Return None for None, else ``_bounded(value, name, maximum)``."""

    return None if value is None else _bounded(value, name, maximum)


def _flag(value: Any, name: str) -> bool:
    """Return ``value`` as a bool. Only bools (Python or NumPy) are accepted."""

    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean, got {type(value).__name__}")
    return bool(value)


def _slot(value: Any, name: str) -> int:
    """Return an executive slot index in ``[0, MAX_GOALS)``, or -1 for no slot."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    slot = -1 if value == -1 else _non_negative_int(value, name)
    if slot >= MAX_GOALS:
        raise ValueError(f"{name} must be -1 (no slot) or below {MAX_GOALS}, got {slot}")
    return slot


def _instances(values: Any, kind: type, name: str) -> tuple[Any, ...]:
    """Return ``values`` as a tuple after checking that every item is a ``kind``."""

    items = tuple(values)
    for position, item in enumerate(items):
        if not isinstance(item, kind):
            raise TypeError(f"{name}[{position}] must be a {kind.__name__}, got {type(item).__name__}")
    return items


def _checked_report(report: Any, name: str) -> GoalReport:
    """Return ``report`` with validated fields and ``outcome`` as a SkillOutcome."""

    if not isinstance(report, GoalReport):
        raise TypeError(f"{name} must be a GoalReport, got {type(report).__name__}")
    if isinstance(report.outcome, bool) or not isinstance(report.outcome, (int, np.integer)):
        raise TypeError(f"{name}.outcome must be a SkillOutcome, got {type(report.outcome).__name__}")
    return GoalReport(
        tick=_non_negative_int(report.tick, f"{name}.tick"),
        lane=_non_negative_int(report.lane, f"{name}.lane"),
        skill_id=_text(report.skill_id, f"{name}.skill_id"),
        executive_version=_non_negative_int(report.executive_version, f"{name}.executive_version"),
        outcome=SkillOutcome(int(report.outcome)),
        duration_ticks=_non_negative_int(report.duration_ticks, f"{name}.duration_ticks"),
        detail=_string(report.detail, f"{name}.detail"),
    )


def _outcome_named(name: str) -> SkillOutcome:
    """Return the SkillOutcome called ``name`` (the JSON form), or raise ValueError."""

    try:
        return SkillOutcome[name]
    except KeyError:
        raise ValueError(
            f"unknown outcome {name!r}; expected one of {[item.name for item in SkillOutcome]}"
        ) from None


# ---- JSON Schema (only the keywords that executive/store.py validate_against_schema enforces) ----


def _count_schema(minimum: int = 0) -> dict[str, Any]:
    """Return the schema of an int32 count of at least ``minimum``."""

    return {"type": "integer", "minimum": minimum, "maximum": INT32_MAX}


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    """Return a closed object schema in which every property is required (exact keys)."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


_GOAL_ENTRY_SCHEMA: dict[str, Any] = copy.deepcopy(EXECUTIVE_STATE_JSON_SCHEMA["properties"]["goals"]["items"])
_GOAL_ENTRY_SCHEMA["required"] = [*_GOAL_ENTRY_SCHEMA["required"], *_TEXT_FIELDS]
_GOAL_ENTRY_SCHEMA["properties"].update(
    intent={"type": "string", "description": "What you want from this goal, in your own words; may be empty."},
    when={"type": "string", "description": "When this goal applies, in your own words; may be empty."},
    note={"type": "string", "description": "Anything else about this goal; may be empty."},
)
_EXECUTIVE_SECTION_SCHEMA = _object_schema(
    {
        **copy.deepcopy(EXECUTIVE_STATE_JSON_SCHEMA["properties"]),
        "goals": {"type": "array", "maxItems": MAX_GOALS, "items": _GOAL_ENTRY_SCHEMA},
        "notes": {"type": "string", "description": "Your working notes, the idea behind the goals; may be empty."},
    }
)
_BELIEF_SECTION_SCHEMA = _object_schema(
    {
        "lanes": {
            "type": "array",
            "items": _object_schema(
                {
                    "features": _object_schema({name: {"type": "number"} for name in BELIEF_FEATURE_NAMES}),
                    "known": _object_schema({name: {"type": "boolean"} for name in BELIEF_FEATURE_NAMES}),
                    "latent_norm": {"type": "number", "minimum": 0},
                    "latent_prefix": {
                        "type": "array",
                        "minItems": LATENT_PREFIX_WIDTH,
                        "maxItems": LATENT_PREFIX_WIDTH,
                        "items": {"type": "number"},
                    },
                }
            ),
        }
    }
)
_DECISION_SECTION_SCHEMA = _object_schema(
    {
        "in_force": {
            "type": "array",
            "items": _object_schema(
                {
                    "skill_id": {"type": "string", "minLength": 1},
                    "slot": {"type": "integer", "minimum": -1, "maximum": MAX_GOALS - 1},
                    "executive_version": _count_schema(),
                    "params": {"type": "array", "maxItems": PARAM_WIDTH, "items": {"type": "number"}},
                    "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                    "entropy": {"type": ["number", "null"], "minimum": 0},
                }
            ),
        },
        "recent": {
            "type": "array",
            "items": _object_schema(
                {
                    "tick": _count_schema(),
                    "lane": _count_schema(),
                    "skill_id": {"type": "string", "minLength": 1},
                    "replace": {"type": "boolean"},
                    "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                }
            ),
        },
        "recent_limit": _count_schema(1),
    }
)
def _triple_schema(item: dict[str, Any]) -> dict[str, Any]:
    """Return the schema of an optional triple: three ``item`` values, or null when absent."""

    return {"type": ["array", "null"], "minItems": 3, "maxItems": 3, "items": item}


_REQUEST_SCHEMA = _object_schema(
    {
        "request_id": _count_schema(),
        "agent": _count_schema(),
        "issued_tick": _count_schema(),
        "expires_tick": _count_schema(1),
        "verb": {"type": "string", "minLength": 1},
        "args": _object_schema(
            {
                "position": _triple_schema({"type": "number"}),
                "direction": _triple_schema({"type": "number"}),
                "entity": {"type": ["integer", "null"], "minimum": 0, "maximum": INT32_MAX},
                "cell": _triple_schema({"type": "integer", "minimum": -INT32_MAX - 1, "maximum": INT32_MAX}),
                "face": {"type": ["string", "null"], "minLength": 1},
                "scalars": {
                    "type": "array",
                    "minItems": PARAM_WIDTH,
                    "maxItems": PARAM_WIDTH,
                    "items": {"type": ["number", "null"]},
                },
            }
        ),
        "mode": {"type": "string", "minLength": 1},
        "urgency": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "priority": {"type": "number"},
        "goal_slot": {"type": "integer", "minimum": -1, "maximum": MAX_GOALS - 1},
        "executive_version": _count_schema(),
        "program_step": _count_schema(),
    }
)
_REQUESTS_SECTION_SCHEMA = _object_schema(
    {"live": {"type": "array", "items": _REQUEST_SCHEMA}, "limit": _count_schema(1)}
)
_STATUS_SCHEMA = _object_schema(
    {
        "agent": _count_schema(),
        "request_id": _count_schema(),
        "kind": {"type": "string", "minLength": 1},
        "tick": _count_schema(),
        "reason": {"type": "string"},
        "conflict_with": {"type": ["integer", "null"], "minimum": 0, "maximum": INT32_MAX},
        "outcome": {"type": ["string", "null"], "minLength": 1},
        "progress": {
            "type": ["array", "null"],
            "items": _object_schema({"name": {"type": "string", "minLength": 1}, "value": {"type": "number"}}),
        },
        "heads": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "signal": {"type": "string"},
    }
)
_SKILLS_SECTION_SCHEMA = _object_schema(
    {
        "active": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "runtime": {"type": "object"},
        "outcomes": {
            "type": "array",
            "items": _object_schema(
                {
                    "tick": _count_schema(),
                    "lane": _count_schema(),
                    "skill_id": {"type": "string", "minLength": 1},
                    "executive_version": _count_schema(),
                    "outcome": {"type": "string", "minLength": 1},
                    "duration_ticks": _count_schema(),
                    "detail": {"type": "string"},
                }
            ),
        },
        "outcome_limit": _count_schema(1),
        "statuses": {"type": "array", "items": _STATUS_SCHEMA},
        "status_limit": _count_schema(1),
    }
)
_EVENTS_SECTION_SCHEMA = _object_schema(
    {
        "window_ticks": _count_schema(1),
        "active": {"type": "array", "items": {"type": "array", "items": {"type": "string", "minLength": 1}}},
    }
)

SHARED_STATE_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SharedState",
    "description": "The shared scratchpad S_t from experimental/neural_agent/shared_state.py "
    "(docs/SYSTEM_ONE.md section 8.10).",
    **_object_schema(
        {
            "version": _count_schema(),
            "tick": _count_schema(),
            "writers": _object_schema({name: {"type": "string", "minLength": 1} for name in SECTION_NAMES}),
            "executive": _EXECUTIVE_SECTION_SCHEMA,
            "belief": _BELIEF_SECTION_SCHEMA,
            "decision": _DECISION_SECTION_SCHEMA,
            "requests": _REQUESTS_SECTION_SCHEMA,
            "skills": _SKILLS_SECTION_SCHEMA,
            "events": _EVENTS_SECTION_SCHEMA,
        }
    ),
}


def validate(payload: Any) -> None:
    """Check a JSON-shaped payload against SHARED_STATE_JSON_SCHEMA, as the executive store checks E_t.

    This is the structural check (keys, JSON types, ranges). ``SharedState.from_dict`` runs it
    first, then the semantic checks of each section.

    Raises:
        ValueError: If ``payload`` breaks the schema.
    """

    validate_against_schema(payload, SHARED_STATE_JSON_SCHEMA)


# ---- executive section: written by the LLM ----


@dataclass(frozen=True)
class GoalEntry:
    """One goal as the LLM writes it: GoalSpec's typed fields plus its own words.

    Attributes:
        skill_id, priority, params, constraints, horizon_seconds, deadline_tick: As in
            :class:`GoalSpec`, and validated by it. ``constraints`` labels (for example
            ``"on_damage"``) stay typed hints beside the words.
        intent: What the LLM wants from this goal, free text. May be empty.
        when: The condition, in the LLM's words. May be empty.
        note: Any other remark on this goal. May be empty.
    """

    skill_id: str
    priority: float = 1.0
    params: tuple[float, ...] = ()
    constraints: tuple[str, ...] = ()
    horizon_seconds: float = 10.0
    deadline_tick: int | None = None
    intent: str = ""
    when: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        """Validate the typed fields through GoalSpec and require strings for the words."""

        spec = GoalSpec(**{key: getattr(self, key) for key in _SPEC_FIELDS})
        for key in _SPEC_FIELDS:
            object.__setattr__(self, key, getattr(spec, key))
        for key in _TEXT_FIELDS:
            _string(getattr(self, key), f"GoalEntry.{key}")


def entry_to_spec(entry: GoalEntry) -> GoalSpec:
    """Return the typed goal of ``entry``: the words are dropped, the constraint labels are kept."""

    if not isinstance(entry, GoalEntry):
        raise TypeError(f"entry must be a GoalEntry, got {type(entry).__name__}")
    return GoalSpec(**{key: getattr(entry, key) for key in _SPEC_FIELDS})


def spec_to_entry(spec: GoalSpec, *, intent: str = "", when: str = "", note: str = "") -> GoalEntry:
    """Return the goal entry for ``spec`` with the given words (empty by default)."""

    if not isinstance(spec, GoalSpec):
        raise TypeError(f"spec must be a GoalSpec, got {type(spec).__name__}")
    return GoalEntry(**{key: getattr(spec, key) for key in _SPEC_FIELDS}, intent=intent, when=when, note=note)


@dataclass(frozen=True)
class ExecutiveSection:
    """The LLM's section: the content of ExecutiveState with goals in words, plus its working notes.

    Attributes:
        version, issued_tick, budget, facts: As in :class:`ExecutiveState` (version is the E_t
            version, not the document version), validated by it.
        goals: At most ``MAX_GOALS`` :class:`GoalEntry` values; goal k is slot k when lowered.
        notes: The LLM's working notes, the idea behind the goals. May be empty.
    """

    version: int = 0
    issued_tick: int = 0
    goals: tuple[GoalEntry, ...] = ()
    budget: ComputeBudget = field(default_factory=ComputeBudget)
    facts: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate through ExecutiveState and normalize the sequences to tuples."""

        goals = _instances(self.goals, GoalEntry, "ExecutiveSection.goals")
        state = ExecutiveState(
            self.version, self.issued_tick, tuple(entry_to_spec(goal) for goal in goals), self.budget, self.facts
        )
        object.__setattr__(self, "version", state.version)
        object.__setattr__(self, "issued_tick", state.issued_tick)
        object.__setattr__(self, "goals", goals)
        object.__setattr__(self, "facts", state.facts)
        _string(self.notes, "ExecutiveSection.notes")

    def to_executive_state(self) -> ExecutiveState:
        """Return E_t without the words: every typed field kept, the text dropped."""

        return ExecutiveState(
            self.version, self.issued_tick, tuple(entry_to_spec(goal) for goal in self.goals), self.budget, self.facts
        )

    @classmethod
    def from_executive_state(cls, state: ExecutiveState) -> ExecutiveSection:
        """Return the section for an E_t from any existing writer; every text field is empty."""

        if not isinstance(state, ExecutiveState):
            raise TypeError(f"state must be an ExecutiveState, got {type(state).__name__}")
        goals = tuple(spec_to_entry(goal) for goal in state.goals)
        return cls(state.version, state.issued_tick, goals, state.budget, state.facts)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-shaped mapping: ExecutiveState's keys, the words per goal, and ``notes``."""

        mapping = json.loads(self.to_executive_state().to_json())
        for item, goal in zip(mapping["goals"], self.goals):
            item.update({key: getattr(goal, key) for key in _TEXT_FIELDS})
        mapping["notes"] = self.notes
        return mapping

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> ExecutiveSection:
        """Parse the JSON-shaped mapping (for example the LLM's reply), schema first.

        Raises:
            ValueError: If the mapping breaks the executive part of the schema or a value fails
                validation.
        """

        validate_against_schema(mapping, _EXECUTIVE_SECTION_SCHEMA, "$.executive")
        typed = {key: mapping[key] for key in EXECUTIVE_STATE_JSON_SCHEMA["required"]}
        typed["goals"] = [{key: goal[key] for key in _SPEC_FIELDS} for goal in mapping["goals"]]
        state = ExecutiveState.from_dict(typed)
        goals = tuple(
            spec_to_entry(spec, **{key: goal[key] for key in _TEXT_FIELDS})
            for spec, goal in zip(state.goals, mapping["goals"])
        )
        return cls(state.version, state.issued_tick, goals, state.budget, state.facts, mapping["notes"])


# ---- belief section: written by the belief module ----


@dataclass(frozen=True)
class BeliefLane:
    """One lane of the belief summary.

    Attributes:
        features: ``BELIEF_FEATURE_COUNT`` finite values, in BELIEF_FEATURE_NAMES order.
        known: ``BELIEF_FEATURE_COUNT`` flags; False means not published, which is not zero.
        latent_norm: The L2 norm of the lane's latent, finite and non-negative.
        latent_prefix: The first ``LATENT_PREFIX_WIDTH`` latent values.
    """

    features: tuple[float, ...]
    known: tuple[bool, ...]
    latent_norm: float
    latent_prefix: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the widths and values and normalize them to Python floats and bools."""

        features = _finite_tuple(self.features, BELIEF_FEATURE_COUNT, "BeliefLane.features")
        known = tuple(_flag(flag, f"BeliefLane.known[{index}]") for index, flag in enumerate(self.known))
        prefix = _finite_tuple(self.latent_prefix, LATENT_PREFIX_WIDTH, "BeliefLane.latent_prefix")
        if len(features) != BELIEF_FEATURE_COUNT or len(known) != BELIEF_FEATURE_COUNT:
            raise ValueError(
                f"BeliefLane needs {BELIEF_FEATURE_COUNT} features and known flags, "
                f"got {len(features)} and {len(known)}"
            )
        if len(prefix) != LATENT_PREFIX_WIDTH:
            raise ValueError(f"BeliefLane.latent_prefix needs {LATENT_PREFIX_WIDTH} values, got {len(prefix)}")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "known", known)
        object.__setattr__(self, "latent_norm", _bounded(self.latent_norm, "BeliefLane.latent_norm"))
        object.__setattr__(self, "latent_prefix", prefix)


@dataclass(frozen=True)
class BeliefSection:
    """The belief module's section: a lane-wise summary of B_t (``lanes[b]`` is lane b)."""

    lanes: tuple[BeliefLane, ...] = ()

    def __post_init__(self) -> None:
        """Normalize ``lanes`` to a tuple of BeliefLane."""

        object.__setattr__(self, "lanes", _instances(self.lanes, BeliefLane, "BeliefSection.lanes"))

    @classmethod
    def from_belief(cls, belief: SharedBeliefState) -> BeliefSection:
        """Summarize B_t per lane: features, known flags, latent L2 norm and the first 8 latent values.

        Raises:
            TypeError: If ``belief`` is not a SharedBeliefState.
            ValueError: If a shape or the known dtype is wrong, or a value is not finite.
        """

        if not isinstance(belief, SharedBeliefState):
            raise TypeError(f"belief must be a SharedBeliefState, got {type(belief).__name__}")
        features = np.asarray(belief.features)
        known = np.asarray(belief.feature_known)
        latent = np.asarray(belief.latent, dtype=np.float64)
        lanes = features.shape[0] if features.ndim == 2 else -1
        expected = ((lanes, BELIEF_FEATURE_COUNT), (lanes, BELIEF_FEATURE_COUNT), (lanes, BELIEF_LATENT_WIDTH))
        if (features.shape, known.shape, latent.shape) != expected or known.dtype != np.bool_:
            raise ValueError(
                f"belief must hold features [B, {BELIEF_FEATURE_COUNT}], feature_known bool "
                f"[B, {BELIEF_FEATURE_COUNT}] and latent [B, {BELIEF_LATENT_WIDTH}]; got {features.shape}, "
                f"{known.shape} {known.dtype} and {latent.shape}"
            )
        norms = np.linalg.norm(latent, axis=1)
        return cls(
            tuple(
                BeliefLane(
                    features=tuple(features[lane]),
                    known=tuple(known[lane]),
                    latent_norm=float(norms[lane]),
                    latent_prefix=tuple(latent[lane, :LATENT_PREFIX_WIDTH]),
                )
                for lane in range(lanes)
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-shaped mapping; features and known flags are keyed by BELIEF_FEATURE_NAMES."""

        return {
            "lanes": [
                {
                    "features": dict(zip(BELIEF_FEATURE_NAMES, lane.features)),
                    "known": dict(zip(BELIEF_FEATURE_NAMES, lane.known)),
                    "latent_norm": lane.latent_norm,
                    "latent_prefix": list(lane.latent_prefix),
                }
                for lane in self.lanes
            ]
        }

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> BeliefSection:
        """Parse the JSON-shaped mapping, schema first."""

        validate_against_schema(mapping, _BELIEF_SECTION_SCHEMA, "$.belief")
        return cls(
            tuple(
                BeliefLane(
                    features=tuple(item["features"][name] for name in BELIEF_FEATURE_NAMES),
                    known=tuple(item["known"][name] for name in BELIEF_FEATURE_NAMES),
                    latent_norm=item["latent_norm"],
                    latent_prefix=tuple(item["latent_prefix"]),
                )
                for item in mapping["lanes"]
            )
        )


# ---- decision section: written by the JEV ----


@dataclass(frozen=True)
class GoalInForce:
    """The goal in force in one lane, with the JEV's confidence and entropy when known.

    Attributes:
        skill_id: The skill of the goal.
        slot: The executive slot it came from, or -1 for none (for example idle).
        executive_version: The E_t version the decision served.
        params: At most ``PARAM_WIDTH`` goal parameters (the GoalPayload params, before padding).
        confidence: The confidence head's probability in [0, 1], or None when not known.
        entropy: The entropy of the skill distribution in nats (>= 0), or None when not known.
    """

    skill_id: str
    slot: int
    executive_version: int
    params: tuple[float, ...] = ()
    confidence: float | None = None
    entropy: float | None = None

    def __post_init__(self) -> None:
        """Validate the fields."""

        _text(self.skill_id, "GoalInForce.skill_id")
        object.__setattr__(self, "slot", _slot(self.slot, "GoalInForce.slot"))
        object.__setattr__(
            self, "executive_version", _non_negative_int(self.executive_version, "GoalInForce.executive_version")
        )
        object.__setattr__(self, "params", _finite_tuple(self.params, PARAM_WIDTH, "GoalInForce.params"))
        object.__setattr__(self, "confidence", _optional_bounded(self.confidence, "GoalInForce.confidence", 1.0))
        object.__setattr__(self, "entropy", _optional_bounded(self.entropy, "GoalInForce.entropy"))


@dataclass(frozen=True)
class DecisionRecord:
    """One past decision: tick, lane, chosen skill, whether it replaced the goal, and its confidence."""

    tick: int
    lane: int
    skill_id: str
    replace: bool
    confidence: float | None = None

    def __post_init__(self) -> None:
        """Validate the fields."""

        object.__setattr__(self, "tick", _non_negative_int(self.tick, "DecisionRecord.tick"))
        object.__setattr__(self, "lane", _non_negative_int(self.lane, "DecisionRecord.lane"))
        _text(self.skill_id, "DecisionRecord.skill_id")
        object.__setattr__(self, "replace", _flag(self.replace, "DecisionRecord.replace"))
        object.__setattr__(self, "confidence", _optional_bounded(self.confidence, "DecisionRecord.confidence", 1.0))


@dataclass(frozen=True)
class DecisionSection:
    """The JEV's section: the goal in force per lane and the last decisions.

    Attributes:
        in_force: One :class:`GoalInForce` per lane.
        recent: The last decisions, newest last. Building the section keeps only the newest
            ``recent_limit`` of them, so the list stays bounded.
        recent_limit: The bound N, at least 1. Default ``DECISION_HISTORY`` (16).
    """

    in_force: tuple[GoalInForce, ...] = ()
    recent: tuple[DecisionRecord, ...] = ()
    recent_limit: int = DECISION_HISTORY

    def __post_init__(self) -> None:
        """Validate the records and keep the newest ``recent_limit`` decisions."""

        limit = _positive_int(self.recent_limit, "DecisionSection.recent_limit")
        object.__setattr__(self, "recent_limit", limit)
        object.__setattr__(self, "in_force", _instances(self.in_force, GoalInForce, "DecisionSection.in_force"))
        recent = _instances(self.recent, DecisionRecord, "DecisionSection.recent")
        object.__setattr__(self, "recent", recent[-limit:])

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-shaped mapping."""

        return {
            "in_force": [dataclasses.asdict(goal) | {"params": list(goal.params)} for goal in self.in_force],
            "recent": [dataclasses.asdict(record) for record in self.recent],
            "recent_limit": self.recent_limit,
        }

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> DecisionSection:
        """Parse the JSON-shaped mapping, schema first."""

        validate_against_schema(mapping, _DECISION_SECTION_SCHEMA, "$.decision")
        return cls(
            in_force=tuple(GoalInForce(**item) for item in mapping["in_force"]),
            recent=tuple(DecisionRecord(**item) for item in mapping["recent"]),
            recent_limit=mapping["recent_limit"],
        )


# ---- requests section: written by the JEV (the decision layer) ----


def _request_to_dict(request: ActionRequest) -> dict[str, Any]:
    """Return the JSON form of a request: absent arguments are null, the mode is its string."""

    args = request.args
    mapping = {item.name: getattr(request, item.name) for item in dataclasses.fields(ActionRequest)}
    mapping["args"] = {
        "position": None if args.position is None else list(args.position),
        "direction": None if args.direction is None else list(args.direction),
        "entity": args.entity,
        "cell": None if args.cell is None else list(args.cell),
        "face": args.face,
        "scalars": list(args.scalars),
    }
    return mapping


def _request_from_dict(item: Mapping[str, Any]) -> ActionRequest:
    """Build a request from its JSON form (already checked against the schema)."""

    args = item["args"]
    return ActionRequest(
        **{key: value for key, value in item.items() if key != "args"},
        args=RequestArgs(
            position=args["position"],
            direction=args["direction"],
            entity=args["entity"],
            cell=args["cell"],
            face=args["face"],
            scalars=tuple(args["scalars"]),
        ),
    )


def _status_to_dict(status: RequestStatus) -> dict[str, Any]:
    """Return the JSON form of a status: the outcome by name, the progress as name/value pairs."""

    return {
        "agent": status.agent,
        "request_id": status.request_id,
        "kind": status.kind,
        "tick": status.tick,
        "reason": status.reason,
        "conflict_with": status.conflict_with,
        "outcome": None if status.outcome is None else status.outcome.name,
        "progress": None if status.progress is None else [{"name": n, "value": v} for n, v in status.progress],
        "heads": list(status.heads),
        "signal": status.signal,
    }


def _status_from_dict(item: Mapping[str, Any]) -> RequestStatus:
    """Build a status from its JSON form (already checked against the schema)."""

    progress = item["progress"]
    return RequestStatus(
        agent=item["agent"],
        request_id=item["request_id"],
        kind=item["kind"],
        tick=item["tick"],
        reason=item["reason"],
        conflict_with=item["conflict_with"],
        outcome=None if item["outcome"] is None else _outcome_named(item["outcome"]),
        progress=None if progress is None else tuple((pair["name"], pair["value"]) for pair in progress),
        heads=tuple(item["heads"]),
        signal=item.get("signal", ""),
    )


@dataclass(frozen=True)
class RequestsSection:
    """The decision layer's live requests to the lower layer (SYSTEM_ONE.md section 9.10).

    Attributes:
        live: The live requests of every agent, stored sorted by (agent, request_id). A request id
            appears once per agent. Building the section keeps the ``limit`` highest ids per agent.
            Every argument number must be finite: S_t is JSON, which holds no NaN. A non-finite
            request is the combiner's to reject (``requests.validate``), and it never reaches S_t.
        limit: The bound per agent, at least 1. Default ``REQUEST_LIMIT`` (16).
    """

    live: tuple[ActionRequest, ...] = ()
    limit: int = REQUEST_LIMIT

    def __post_init__(self) -> None:
        """Validate the requests, sort them and keep the newest ``limit`` per agent."""

        limit = _positive_int(self.limit, "RequestsSection.limit")
        requests = _instances(self.live, ActionRequest, "RequestsSection.live")
        for index, request in enumerate(requests):
            if not all(math.isfinite(value) for value in request.args.numbers()):
                raise ValueError(
                    f"RequestsSection.live[{index}] (agent {request.agent}, request {request.request_id}) holds a "
                    "non-finite argument; S_t holds finite numbers only"
                )
        ordered = sorted(requests, key=lambda request: (request.agent, request.request_id))
        keys = [(request.agent, request.request_id) for request in ordered]
        if len(set(keys)) != len(keys):
            raise ValueError("RequestsSection.live repeats a request id for one agent")
        kept: list[ActionRequest] = []
        for agent in sorted({request.agent for request in ordered}):
            kept.extend([request for request in ordered if request.agent == agent][-limit:])
        object.__setattr__(self, "limit", limit)
        object.__setattr__(self, "live", tuple(kept))

    def for_agent(self, agent: int) -> tuple[ActionRequest, ...]:
        """Return the live requests of ``agent``, oldest id first."""

        return tuple(request for request in self.live if request.agent == agent)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-shaped mapping."""

        return {"live": [_request_to_dict(request) for request in self.live], "limit": self.limit}

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> RequestsSection:
        """Parse the JSON-shaped mapping, schema first."""

        validate_against_schema(mapping, _REQUESTS_SECTION_SCHEMA, "$.requests")
        return cls(live=tuple(_request_from_dict(item) for item in mapping["live"]), limit=mapping["limit"])


# ---- skills section: written by the dispatcher and the skills ----


@dataclass(frozen=True)
class SkillsSection:
    """The skills' section: what runs, its runtime summaries, the outcomes, and the status stream.

    Attributes:
        active: The active skill id per lane.
        runtime: Runtime summaries keyed by skill id, with free-form JSON values from the
            dispatcher. Stored as its JSON form (a deep copy); NaN and non-JSON values are refused.
        outcomes: GoalReports, newest last. Building the section keeps only the newest
            ``outcome_limit`` of them.
        outcome_limit: The bound, at least 1. Default ``OUTCOME_HISTORY`` (16).
        statuses: The status stream of the requests (section 9.10), ``RequestStatus`` records of
            every agent, newest last. Building the section keeps only the newest ``status_limit``.
        status_limit: The bound, at least 1. Default ``STATUS_HISTORY`` (256).
    """

    active: tuple[str, ...] = ()
    runtime: Mapping[str, Any] = field(default_factory=dict)
    outcomes: tuple[GoalReport, ...] = ()
    outcome_limit: int = OUTCOME_HISTORY
    statuses: tuple[RequestStatus, ...] = ()
    status_limit: int = STATUS_HISTORY

    def __post_init__(self) -> None:
        """Validate the fields, store ``runtime`` as JSON values and keep the newest outcomes and statuses."""

        object.__setattr__(self, "active", _text_tuple(self.active, "SkillsSection.active"))
        if not isinstance(self.runtime, Mapping):
            raise TypeError(f"SkillsSection.runtime must be a mapping, got {type(self.runtime).__name__}")
        for key in self.runtime:
            _text(key, "SkillsSection.runtime key")
        object.__setattr__(self, "runtime", json.loads(json.dumps(dict(self.runtime), allow_nan=False)))
        limit = _positive_int(self.outcome_limit, "SkillsSection.outcome_limit")
        object.__setattr__(self, "outcome_limit", limit)
        reports = tuple(
            _checked_report(report, f"SkillsSection.outcomes[{index}]") for index, report in enumerate(self.outcomes)
        )
        object.__setattr__(self, "outcomes", reports[-limit:])
        status_limit = _positive_int(self.status_limit, "SkillsSection.status_limit")
        object.__setattr__(self, "status_limit", status_limit)
        statuses = _instances(self.statuses, RequestStatus, "SkillsSection.statuses")
        object.__setattr__(self, "statuses", statuses[-status_limit:])

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-shaped mapping; an outcome is written by its SkillOutcome name."""

        return {
            "active": list(self.active),
            "runtime": copy.deepcopy(self.runtime),
            "outcomes": [report._asdict() | {"outcome": report.outcome.name} for report in self.outcomes],
            "outcome_limit": self.outcome_limit,
            "statuses": [_status_to_dict(status) for status in self.statuses],
            "status_limit": self.status_limit,
        }

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> SkillsSection:
        """Parse the JSON-shaped mapping, schema first."""

        validate_against_schema(mapping, _SKILLS_SECTION_SCHEMA, "$.skills")
        return cls(
            active=tuple(mapping["active"]),
            runtime=mapping["runtime"],
            outcomes=tuple(
                GoalReport(**{**item, "outcome": _outcome_named(item["outcome"])}) for item in mapping["outcomes"]
            ),
            outcome_limit=mapping["outcome_limit"],
            statuses=tuple(_status_from_dict(item) for item in mapping["statuses"]),
            status_limit=mapping["status_limit"],
        )


# ---- events section: written by the runtime ----


@dataclass(frozen=True)
class EventsSection:
    """The runtime's section: the decision events active in the last window, per lane.

    Attributes:
        window_ticks: The window length in ticks, at least 1. Default ``EVENT_WINDOW_TICKS`` (11).
        active: Per lane, the names of the events seen in the window, from EVENT_NAMES. They are
            stored in EVENT_NAMES order, each name once.
    """

    window_ticks: int = EVENT_WINDOW_TICKS
    active: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        """Validate the window and the names, and order each lane's names as EVENT_NAMES."""

        object.__setattr__(self, "window_ticks", _positive_int(self.window_ticks, "EventsSection.window_ticks"))
        lanes = []
        for lane, names in enumerate(self.active):
            given = set(_text_tuple(names, f"EventsSection.active[{lane}]"))
            unknown = sorted(given - set(EVENT_NAMES))
            if unknown:
                raise ValueError(f"EventsSection.active[{lane}] names unknown events {unknown}; see EVENT_NAMES")
            lanes.append(tuple(name for name in EVENT_NAMES if name in given))
        object.__setattr__(self, "active", tuple(lanes))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-shaped mapping."""

        return {"window_ticks": self.window_ticks, "active": [list(names) for names in self.active]}

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> EventsSection:
        """Parse the JSON-shaped mapping, schema first."""

        validate_against_schema(mapping, _EVENTS_SECTION_SCHEMA, "$.events")
        return cls(window_ticks=mapping["window_ticks"], active=tuple(tuple(names) for names in mapping["active"]))


_SECTION_TYPES: dict[str, type] = {
    "executive": ExecutiveSection,
    "belief": BeliefSection,
    "decision": DecisionSection,
    "requests": RequestsSection,
    "skills": SkillsSection,
    "events": EventsSection,
}


# ---- the document ----


class SharedState:
    """The scratchpad S_t: six sections, each with one writer, and a document version.

    The constructor builds a document (``from_dict`` uses it); after that the document changes
    only through :meth:`write`. Sections are immutable values that a write replaces whole.
    ``write`` and every read of more than one section (``to_dict``, ``view_for_llm``,
    ``view_for_jev``) take one lock, so no reader sees a half-applied write; a one-section read
    (a property, ``view_for_skills``) is a single reference read.

    Args:
        version: Document version, a non-negative int. Default 0.
        tick: Tick of the last write, a non-negative int. Default 0.
        executive, belief, decision, requests, skills, events: The sections. None gives the empty
            section.

    Raises:
        TypeError: If a section has the wrong type, or ``version`` or ``tick`` is not an integer.
        ValueError: If ``version`` or ``tick`` is negative.
    """

    def __init__(
        self,
        *,
        version: int = 0,
        tick: int = 0,
        executive: ExecutiveSection | None = None,
        belief: BeliefSection | None = None,
        decision: DecisionSection | None = None,
        requests: RequestsSection | None = None,
        skills: SkillsSection | None = None,
        events: EventsSection | None = None,
    ) -> None:
        given = {
            "executive": executive,
            "belief": belief,
            "decision": decision,
            "requests": requests,
            "skills": skills,
            "events": events,
        }
        sections: dict[str, Any] = {}
        for name, section in given.items():
            kind = _SECTION_TYPES[name]
            if section is None:
                section = kind()
            elif not isinstance(section, kind):
                raise TypeError(f"the {name} section must be a {kind.__name__}, got {type(section).__name__}")
            sections[name] = section
        self._version = _non_negative_int(version, "SharedState.version")
        self._tick = _non_negative_int(tick, "SharedState.tick")
        self._sections = sections
        self._lock = threading.Lock()

    @property
    def version(self) -> int:
        """The document version: the number of writes since version 0."""

        return self._version

    @property
    def tick(self) -> int:
        """The tick recorded by the last write."""

        return self._tick

    @property
    def writers(self) -> Mapping[str, str]:
        """The fixed mapping section -> writer name (``SECTION_WRITERS``)."""

        return SECTION_WRITERS

    @property
    def executive(self) -> ExecutiveSection:
        """The executive section (writer ``"llm"``)."""

        return self._sections["executive"]

    @property
    def belief(self) -> BeliefSection:
        """The belief section (writer ``"belief"``)."""

        return self._sections["belief"]

    @property
    def decision(self) -> DecisionSection:
        """The decision section (writer ``"jev"``)."""

        return self._sections["decision"]

    @property
    def requests(self) -> RequestsSection:
        """The requests section (writer ``"jev"``)."""

        return self._sections["requests"]

    @property
    def skills(self) -> SkillsSection:
        """The skills section (writer ``"skills"``)."""

        return self._sections["skills"]

    @property
    def events(self) -> EventsSection:
        """The events section (writer ``"runtime"``)."""

        return self._sections["events"]

    def write(self, section_name: str, new_section: Any, *, writer: str, tick: int) -> int:
        """Replace one section as its owner, bump the version by one, record ``tick``.

        Args:
            section_name: One of ``SECTION_NAMES``.
            new_section: The new section value, of that section's type.
            writer: The writer's name; it must be ``SECTION_WRITERS[section_name]``.
            tick: The tick of this write, a non-negative int. It is recorded as given.

        Returns:
            The new document version.

        Raises:
            ValueError: If ``section_name`` is unknown or ``tick`` is negative.
            PermissionError: If ``writer`` does not own the section. Nothing changes.
            TypeError: If ``new_section`` has the wrong type or ``tick`` is not an integer.
        """

        if section_name not in SECTION_WRITERS:
            raise ValueError(f"unknown section {section_name!r}; the sections are {SECTION_NAMES}")
        owner = SECTION_WRITERS[section_name]
        if writer != owner:
            raise PermissionError(
                f"writer {writer!r} may not write the {section_name!r} section; its only writer is {owner!r}"
            )
        kind = _SECTION_TYPES[section_name]
        if not isinstance(new_section, kind):
            raise TypeError(f"the {section_name} section must be a {kind.__name__}, got {type(new_section).__name__}")
        tick = _non_negative_int(tick, "tick")
        with self._lock:
            self._sections[section_name] = new_section
            self._version += 1
            self._tick = tick
            return self._version

    def _snapshot(self) -> tuple[int, int, dict[str, Any]]:
        """Return ``(version, tick, sections)`` read together under the lock."""

        with self._lock:
            return self._version, self._tick, dict(self._sections)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-shaped mapping that SHARED_STATE_JSON_SCHEMA describes."""

        version, tick, sections = self._snapshot()
        mapping: dict[str, Any] = {"version": version, "tick": tick, "writers": dict(SECTION_WRITERS)}
        mapping.update({name: sections[name].to_dict() for name in SECTION_NAMES})
        return mapping

    def to_json(self) -> str:
        """Return canonical JSON text (sorted keys, NaN refused) that :meth:`from_json` parses back."""

        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SharedState:
        """Build a document from a JSON-shaped payload: :func:`validate`, then each section's checks.

        Raises:
            TypeError: If a value has the wrong Python type for its section.
            ValueError: If the payload breaks the schema, the writers differ from
                ``SECTION_WRITERS``, or a value fails a section's checks.
        """

        validate(payload)
        if dict(payload["writers"]) != dict(SECTION_WRITERS):
            raise ValueError(f"writers are fixed at {dict(SECTION_WRITERS)}, got {dict(payload['writers'])}")
        sections = {name: _SECTION_TYPES[name].from_dict(payload[name]) for name in SECTION_NAMES}
        return cls(version=payload["version"], tick=payload["tick"], **sections)

    @classmethod
    def from_json(cls, text: str) -> SharedState:
        """Parse text produced by :meth:`to_json` (or any schema-valid text)."""

        if not isinstance(text, str):
            raise TypeError("text must be a JSON string")
        return cls.from_dict(json.loads(text))


# ---- views: one per reader ----


class JevView(NamedTuple):
    """What the decision layer reads from S_t. B lanes (the belief section's), G = MAX_GOALS.

    Attributes:
        executive: E_t without the words, the record the slow path encodes.
        arrays: ``contracts.executive_arrays(executive, skill_ids, B)``, E_t lowered as today.
        intents: G strings; ``intents[k]`` is the ``intent`` of the goal in slot k, "" in an empty slot.
        whens: G strings, aligned the same way, for ``when``.
        features: float32[B, BELIEF_FEATURE_COUNT].
        feature_known: bool[B, BELIEF_FEATURE_COUNT].
        latent_norm: float32[B].
        latent_prefix: float32[B, LATENT_PREFIX_WIDTH].
        current_skill: B skill ids: the skills section's active skill, or idle in every lane before
            the skills section is written (the carry starts with ``idle_goal``).
        current_skill_index: int32[B], positions of ``current_skill`` in ``skill_ids``.
        live_requests: B mappings; ``live_requests[b]`` maps each live request id of agent b, in id
            order, to its latest ``RequestStatus`` in the stream, or None before its first status.
    """

    executive: ExecutiveState
    arrays: ExecutiveArrays
    intents: tuple[str, ...]
    whens: tuple[str, ...]
    features: np.ndarray
    feature_known: np.ndarray
    latent_norm: np.ndarray
    latent_prefix: np.ndarray
    current_skill: tuple[str, ...]
    current_skill_index: np.ndarray
    live_requests: tuple[dict[int, RequestStatus | None], ...]


class SkillsView(NamedTuple):
    """What the skills read from S_t: the goal payload fields per lane, from the decision section.

    Attributes:
        skill_id: The skill of the goal in force, per lane.
        params: float32[B, PARAM_WIDTH], zero-padded.
        executive_version: int32[B].
        goal_slot: int32[B], -1 where no executive slot is involved.
    """

    skill_id: tuple[str, ...]
    params: np.ndarray
    executive_version: np.ndarray
    goal_slot: np.ndarray


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


def view_for_jev(state: SharedState, skill_ids: Sequence[str]) -> JevView:
    """Return the typed view the decision layer reads; B is the belief section's lane count.

    Args:
        state: The scratchpad.
        skill_ids: The skill catalogue in dispatcher order, idle first. S_t does not hold the
            catalogue, and ``contracts.executive_arrays`` needs it to lower E_t.

    Raises:
        ValueError: If the belief section has no lanes, the skills section reports a different
            number of lanes, a live request names an agent outside the belief's lanes, or
            ``executive_arrays`` refuses the catalogue.
        KeyError: If a goal or an active skill is not in ``skill_ids``.
    """

    _version, _tick, sections = state._snapshot()
    executive: ExecutiveSection = sections["executive"]
    belief: BeliefSection = sections["belief"]
    requests: RequestsSection = sections["requests"]
    skills: SkillsSection = sections["skills"]
    lanes = len(belief.lanes)
    if lanes == 0:
        raise ValueError("the JEV view needs the belief section written for at least one lane")
    outside = sorted({request.agent for request in requests.live if request.agent >= lanes})
    if outside:
        raise ValueError(f"live requests name agents {outside}; the belief section has {lanes} lanes")
    latest = {(status.agent, status.request_id): status for status in skills.statuses}
    record = executive.to_executive_state()
    arrays = executive_arrays(record, skill_ids, lanes)
    catalogue = tuple(skill_ids)
    current = skills.active or (SKILL_IDLE,) * lanes
    if len(current) != lanes:
        raise ValueError(f"the skills section reports {len(current)} lanes; the belief section has {lanes}")
    unknown = sorted({skill for skill in current if skill not in catalogue})
    if unknown:
        raise KeyError(f"active skill(s) {unknown} are not in skill_ids")
    empty = ("",) * (MAX_GOALS - len(executive.goals))
    return JevView(
        executive=record,
        arrays=arrays,
        intents=tuple(goal.intent for goal in executive.goals) + empty,
        whens=tuple(goal.when for goal in executive.goals) + empty,
        features=np.asarray([lane.features for lane in belief.lanes], dtype=np.float32),
        feature_known=np.asarray([lane.known for lane in belief.lanes], dtype=np.bool_),
        latent_norm=np.asarray([lane.latent_norm for lane in belief.lanes], dtype=np.float32),
        latent_prefix=np.asarray([lane.latent_prefix for lane in belief.lanes], dtype=np.float32),
        current_skill=tuple(current),
        current_skill_index=np.asarray([catalogue.index(skill) for skill in current], dtype=np.int32),
        live_requests=tuple(
            {request.request_id: latest.get((lane, request.request_id)) for request in requests.for_agent(lane)}
            for lane in range(lanes)
        ),
    )


def view_for_skills(state: SharedState) -> SkillsView:
    """Return the goal payload fields per lane, from the decision section's goals in force.

    B is the number of lanes in ``decision.in_force`` (0 before the JEV's first write).
    """

    in_force = state.decision.in_force
    params = np.zeros((len(in_force), PARAM_WIDTH), dtype=np.float32)
    for lane, goal in enumerate(in_force):
        params[lane, : len(goal.params)] = goal.params
    return SkillsView(
        skill_id=tuple(goal.skill_id for goal in in_force),
        params=params,
        executive_version=np.asarray([goal.executive_version for goal in in_force], dtype=np.int32),
        goal_slot=np.asarray([goal.slot for goal in in_force], dtype=np.int32),
    )


__all__ = [
    "DECISION_HISTORY",
    "EVENT_WINDOW_TICKS",
    "LATENT_PREFIX_WIDTH",
    "LLM_REJECTIONS",
    "OUTCOME_HISTORY",
    "REQUEST_LIMIT",
    "SECTION_NAMES",
    "SECTION_WRITERS",
    "SHARED_STATE_JSON_SCHEMA",
    "STATUS_HISTORY",
    "BeliefLane",
    "BeliefSection",
    "DecisionRecord",
    "DecisionSection",
    "EventsSection",
    "ExecutiveSection",
    "GoalEntry",
    "GoalInForce",
    "JevView",
    "RequestsSection",
    "SharedState",
    "SkillsSection",
    "SkillsView",
    "entry_to_spec",
    "spec_to_entry",
    "validate",
    "view_for_jev",
    "view_for_llm",
    "view_for_skills",
]
