"""The decision section of S_t: goals in force, recent decisions and the questions to the LLM, written by the decision layer.

Split out of the single ``shared_state.py`` on 2026-10-18 without changes; the package docstring
(``shared_state/__init__.py``) is the design record.
"""

from __future__ import annotations


import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping


from experimental.neural_agent.contracts import (
    PARAM_WIDTH,
    _finite_number,
    _finite_tuple,
    _non_negative_int,
    _text,
)
from experimental.neural_agent.executive.store import validate_against_schema

from .._common import (
    DECISION_HISTORY,
    INFORMATION_HISTORY,
    _bounded,
    _flag,
    _instances,
    _optional_bounded,
    _positive_int,
    _slot,
    _string,
)
from ..schema import (
    _DECISION_SECTION_SCHEMA,
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
class InformationRequest:
    """One question the decision layer asks the LLM (SYSTEM_ONE 9.12): a goal's words it read below
    CLEAR confidence (``jev/typed_rule.py``), or a step the skills kept rejecting (``jev/programs.py``,
    item 8). The LLM answers by writing the typed field or rewording in its next version.

    Attributes:
        tick: The session tick of the request.
        source: Who asks: ``"typed_rule"`` or ``"program"``.
        executive_version: The E_t version concerned.
        slot: The goal slot concerned, or None.
        lane: The lane concerned, or None when it concerns every lane.
        text: What was read or attempted: the goal's ``when`` words, or the rejected verb.
        kind, argument, confidence: The reading, each None when there is none.
        reason: What went wrong and what the LLM should do, in a sentence.
    """

    tick: int
    source: str
    executive_version: int
    slot: int | None
    lane: int | None
    text: str
    kind: str | None
    argument: float | None
    confidence: float | None
    reason: str

    def __post_init__(self) -> None:
        """Validate the fields."""

        object.__setattr__(self, "tick", _non_negative_int(self.tick, "InformationRequest.tick"))
        _text(self.source, "InformationRequest.source")
        object.__setattr__(
            self, "executive_version", _non_negative_int(self.executive_version, "InformationRequest.executive_version")
        )
        for name in ("slot", "lane"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _non_negative_int(value, f"InformationRequest.{name}"))
        _string(self.text, "InformationRequest.text")
        if self.kind is not None:
            _text(self.kind, "InformationRequest.kind")
        if self.argument is not None:
            object.__setattr__(self, "argument", _finite_number(self.argument, "InformationRequest.argument"))
        object.__setattr__(self, "confidence", _optional_bounded(self.confidence, "InformationRequest.confidence", 1.0))
        _string(self.reason, "InformationRequest.reason")


@dataclass(frozen=True)
class GoalReading:
    """How the decision layer understood one goal of an executive version (SYSTEM_ONE 9.13: the decoded
    interpretation is written back so the LLM sees how it was read).

    Attributes:
        executive_version: The E_t version the reading belongs to.
        slot: The goal slot.
        kind: The condition kind read, or None when the slot is ineligible (nothing readable).
        argument: The condition's number, or None.
        source: Where the reading came from: ``typed`` (the LLM's typed field), ``label`` (a constraint
            label), ``words`` (the fallback reader), ``always`` (no condition written).
        confidence: The reader's confidence in [0, 1], 1.0 for typed fields.
    """

    executive_version: int
    slot: int
    kind: str | None
    argument: float | None
    source: str
    confidence: float

    def __post_init__(self) -> None:
        """Validate the fields."""

        object.__setattr__(self, "executive_version", _non_negative_int(self.executive_version, "GoalReading.executive_version"))
        object.__setattr__(self, "slot", _non_negative_int(self.slot, "GoalReading.slot"))
        if self.kind is not None:
            _text(self.kind, "GoalReading.kind")
        if self.argument is not None:
            object.__setattr__(self, "argument", _finite_number(self.argument, "GoalReading.argument"))
        _text(self.source, "GoalReading.source")
        object.__setattr__(self, "confidence", _bounded(self.confidence, "GoalReading.confidence", 1.0))

    @property
    def condition(self) -> str:
        """The reading as one token: ``always``, ``distance_below 3``, or ``unreadable``."""

        if self.kind is None:
            return "unreadable"
        return self.kind if self.argument is None else f"{self.kind} {self.argument:g}"


@dataclass(frozen=True)
class DecisionSection:
    """The JEV's section: the goal in force per lane and the last decisions.

    Attributes:
        in_force: One :class:`GoalInForce` per lane.
        recent: The last decisions, newest last. Building the section keeps only the newest
            ``recent_limit`` of them, so the list stays bounded.
        recent_limit: The bound N, at least 1. Default ``DECISION_HISTORY`` (16).
        information_requests: The questions to the LLM, newest last; only the newest
            ``information_limit`` are kept.
        information_limit: Their bound, at least 1. Default ``INFORMATION_HISTORY`` (8).
        readings: How each goal of the executive version in force was understood, one per slot, in
            slot order; empty before the first read.
    """

    in_force: tuple[GoalInForce, ...] = ()
    recent: tuple[DecisionRecord, ...] = ()
    recent_limit: int = DECISION_HISTORY
    information_requests: tuple[InformationRequest, ...] = ()
    information_limit: int = INFORMATION_HISTORY
    readings: tuple[GoalReading, ...] = ()

    def __post_init__(self) -> None:
        """Validate the records and keep the newest ``recent_limit`` decisions and ``information_limit`` questions."""

        limit = _positive_int(self.recent_limit, "DecisionSection.recent_limit")
        object.__setattr__(self, "recent_limit", limit)
        object.__setattr__(self, "in_force", _instances(self.in_force, GoalInForce, "DecisionSection.in_force"))
        recent = _instances(self.recent, DecisionRecord, "DecisionSection.recent")
        object.__setattr__(self, "recent", recent[-limit:])
        bound = _positive_int(self.information_limit, "DecisionSection.information_limit")
        object.__setattr__(self, "information_limit", bound)
        questions = _instances(self.information_requests, InformationRequest, "DecisionSection.information_requests")
        object.__setattr__(self, "information_requests", questions[-bound:])
        object.__setattr__(self, "readings", _instances(self.readings, GoalReading, "DecisionSection.readings"))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-shaped mapping."""

        return {
            "in_force": [dataclasses.asdict(goal) | {"params": list(goal.params)} for goal in self.in_force],
            "recent": [dataclasses.asdict(record) for record in self.recent],
            "recent_limit": self.recent_limit,
            "information_requests": [dataclasses.asdict(item) for item in self.information_requests],
            "information_limit": self.information_limit,
            "readings": [dataclasses.asdict(item) for item in self.readings],
        }

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> DecisionSection:
        """Parse the JSON-shaped mapping, schema first."""

        validate_against_schema(mapping, _DECISION_SECTION_SCHEMA, "$.decision")
        return cls(
            in_force=tuple(GoalInForce(**item) for item in mapping["in_force"]),
            recent=tuple(DecisionRecord(**item) for item in mapping["recent"]),
            recent_limit=mapping["recent_limit"],
            information_requests=tuple(InformationRequest(**item) for item in mapping["information_requests"]),
            information_limit=mapping["information_limit"],
            readings=tuple(GoalReading(**item) for item in mapping["readings"]),
        )

