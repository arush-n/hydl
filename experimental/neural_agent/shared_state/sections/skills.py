"""The skills section of S_t: active skills, outcomes and the status stream, written by the lower layer and the session.

Split out of the single ``shared_state.py`` on 2026-10-18 without changes; the package docstring
(``shared_state/__init__.py``) is the design record.
"""

from __future__ import annotations


import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


from experimental.neural_agent.contracts import (
    GoalReport,
    _finite_number,
    _non_negative_int,
    _text,
    _text_tuple,
)
from experimental.neural_agent.executive.store import validate_against_schema
from experimental.neural_agent.requests import RequestStatus

from .._common import (
    KNOWN_ENTITY_LIMIT,
    OUTCOME_HISTORY,
    STATUS_HISTORY,
    _checked_report,
    _instances,
    _outcome_named,
    _positive_int,
)
from ..schema import (
    _SKILLS_SECTION_SCHEMA,
)
from ..sections.requests import (
    _status_from_dict,
    _status_to_dict,
)
from .writes import (
    WriteStatus,
    write_status_from_dict,
    write_status_to_dict,
)


# ---- known entities: written by the skill layer each tick ----


@dataclass(frozen=True, kw_only=True)
class KnownEntity:
    """One entity an agent knows, as the skill layer perceives it (``skill_agent/runtime/state_tick.py``,
    ``known_entities``). The decision layer names entities by these ids (the owner's ruling of 2026-10-22).

    Attributes:
        id: The entity handle that requests and remember writes take.
        visible: Whether it was seen this tick.
        age_s: Seconds since it was last seen, at least 0.
        position: Where it was last seen, world (x, y, z) in blocks.
        distance: Blocks from the agent, or None when the agent's position was not given.
        hostile: The skill layer's hostility, 0 to 1 (the decision layer takes it above 0.5).
        attitude: The skill layer's attitude, or None while unknown.
        health: The health fraction.
        alive: Whether it is alive.
        player: Whether it is a player.
        tagged: The skill layer's tag value.

    Raises:
        TypeError: If a flag is not a bool, or a number is not a real number.
        ValueError: If the id, the age or the distance is negative, or a number is not finite.
    """

    id: int
    visible: bool
    age_s: float
    position: tuple[float, float, float]
    distance: float | None
    hostile: float
    attitude: str | None
    health: float
    alive: bool
    player: bool
    tagged: float

    def __post_init__(self) -> None:
        """Check the fields and normalize the numbers to Python types."""

        object.__setattr__(self, "id", _non_negative_int(self.id, "KnownEntity.id"))
        for name in ("visible", "alive", "player"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"KnownEntity.{name} must be a bool, got {type(getattr(self, name)).__name__}")
        age = _finite_number(self.age_s, "KnownEntity.age_s")
        if age < 0:
            raise ValueError(f"KnownEntity.age_s must be at least 0, got {age}")
        object.__setattr__(self, "age_s", age)
        if len(self.position) != 3:
            raise ValueError(f"KnownEntity.position must hold 3 coordinates, got {len(self.position)}")
        object.__setattr__(
            self, "position", tuple(_finite_number(value, "KnownEntity.position") for value in self.position)
        )
        if self.distance is not None:
            distance = _finite_number(self.distance, "KnownEntity.distance")
            if distance < 0:
                raise ValueError(f"KnownEntity.distance must be at least 0, got {distance}")
            object.__setattr__(self, "distance", distance)
        for name in ("hostile", "health", "tagged"):
            object.__setattr__(self, name, _finite_number(getattr(self, name), f"KnownEntity.{name}"))
        if self.attitude is not None:
            _text(self.attitude, "KnownEntity.attitude")


@dataclass(frozen=True, kw_only=True)
class AgentEntities:
    """The entities one agent knows at one tick, visible first, then the most recently seen (``KnownEntity``).

    Attributes:
        agent: The agent (lane).
        tick: The tick the list was read at.
        entities: The known entities, at most ``KNOWN_ENTITY_LIMIT`` (16).

    Raises:
        ValueError: If there are more entities than the limit, or a count is negative.
    """

    agent: int
    tick: int
    entities: tuple[KnownEntity, ...] = ()

    def __post_init__(self) -> None:
        """Check the counts and the bound."""

        for name in ("agent", "tick"):
            object.__setattr__(self, name, _non_negative_int(getattr(self, name), f"AgentEntities.{name}"))
        entities = _instances(self.entities, KnownEntity, "AgentEntities.entities")
        if len(entities) > KNOWN_ENTITY_LIMIT:
            raise ValueError(f"AgentEntities holds {len(entities)} entities; the limit is {KNOWN_ENTITY_LIMIT}")
        object.__setattr__(self, "entities", entities)


def _known_to_dict(record: AgentEntities) -> dict[str, Any]:
    """Return the JSON form of an agent's known entities (a position is a list of three numbers)."""

    entities = [{**asdict(entity), "position": list(entity.position)} for entity in record.entities]
    return {"agent": record.agent, "tick": record.tick, "entities": entities}


def _known_from_dict(item: Mapping[str, Any]) -> AgentEntities:
    """Build an agent's known entities from their JSON form (already checked against the schema)."""

    entities = tuple(KnownEntity(**{**entity, "position": tuple(entity["position"])}) for entity in item["entities"])
    return AgentEntities(agent=item["agent"], tick=item["tick"], entities=entities)


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
        write_statuses: The result of each write of the decision layer (``writes`` section), ``WriteStatus``
            records, newest last. Kept to the newest ``STATUS_HISTORY``.
        known_entities: The known entities of each agent, ``AgentEntities`` records, at most one per agent. The skill
            layer writes them every tick it runs with the agents' positions (``shared_state/skill_writes.py``).
    """

    active: tuple[str, ...] = ()
    runtime: Mapping[str, Any] = field(default_factory=dict)
    outcomes: tuple[GoalReport, ...] = ()
    outcome_limit: int = OUTCOME_HISTORY
    statuses: tuple[RequestStatus, ...] = ()
    status_limit: int = STATUS_HISTORY
    write_statuses: tuple[WriteStatus, ...] = ()
    known_entities: tuple[AgentEntities, ...] = ()

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
        write_statuses = _instances(self.write_statuses, WriteStatus, "SkillsSection.write_statuses")
        object.__setattr__(self, "write_statuses", write_statuses[-STATUS_HISTORY:])
        known = _instances(self.known_entities, AgentEntities, "SkillsSection.known_entities")
        if len({record.agent for record in known}) != len(known):
            raise ValueError("SkillsSection.known_entities holds more than one record of an agent")
        object.__setattr__(self, "known_entities", known)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-shaped mapping; an outcome is written by its SkillOutcome name."""

        return {
            "active": list(self.active),
            "runtime": copy.deepcopy(self.runtime),
            "outcomes": [report._asdict() | {"outcome": report.outcome.name} for report in self.outcomes],
            "outcome_limit": self.outcome_limit,
            "statuses": [_status_to_dict(status) for status in self.statuses],
            "status_limit": self.status_limit,
            "write_statuses": [write_status_to_dict(status) for status in self.write_statuses],
            "known_entities": [_known_to_dict(record) for record in self.known_entities],
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
            write_statuses=tuple(write_status_from_dict(item) for item in mapping["write_statuses"]),
            known_entities=tuple(_known_from_dict(item) for item in mapping["known_entities"]),
        )

