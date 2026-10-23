"""The executive section of S_t: the LLM's goals with its words, written by the LLM through the harness.

Split out of the single ``shared_state.py`` on 2026-10-18 without changes; the package docstring
(``shared_state/__init__.py``) is the design record.
"""

from __future__ import annotations


import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping


from experimental.neural_agent.contracts import (
    EXECUTIVE_STATE_JSON_SCHEMA,
    ComputeBudget,
    Condition,
    ExecutiveState,
    GoalSpec,
    ADDRESSEE_ALL,
    Objective,
    Squad,
)
from experimental.neural_agent.executive.store import validate_against_schema

from .._common import (
    _SPEC_FIELDS,
    _TEXT_FIELDS,
    _instances,
    _string,
)
from ..schema import (
    _EXECUTIVE_SECTION_SCHEMA,
)


# ---- executive section: written by the LLM ----


@dataclass(frozen=True)
class RelationDirective:
    """One relationship directive a goal carries (MIDDLE_LAYER.md section 11): set ``name`` of the relationship with
    the individual ``entity`` to ``value``, for as long as the goal is in the section. The decision layer writes it; the
    rules of ``jev/relations.py`` leave that field alone meanwhile.

    Attributes:
        entity: The perceived entity id the directive is about, a non-negative integer.
        name: The relationship field, for example ``"friend"`` or ``"closeness"``. Checked by the skill layer's host check.
        value: The value, finite. Its range is checked by the skill layer, not here.

    Raises:
        ValueError: If the entity is not a non-negative integer, or the value is not finite.
    """

    entity: int
    name: str
    value: float

    def __post_init__(self) -> None:
        """Check the entity and the value, and normalize the value to a float."""

        if isinstance(self.entity, bool) or not isinstance(self.entity, int) or self.entity < 0:
            raise ValueError(f"RelationDirective.entity must be a non-negative integer, got {self.entity!r}")
        _string(self.name, "RelationDirective.name")
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError(f"RelationDirective.value must be finite, got {self.value!r}")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class GoalEntry:
    """One goal as the LLM writes it: GoalSpec's typed fields plus its own words.

    Attributes:
        skill_id, priority, params, constraints, horizon_seconds, deadline_tick, condition, addressee,
        objective: As in
            :class:`GoalSpec`, and validated by it. ``constraints`` labels (for example
            ``"on_damage"``) stay typed hints beside the words.
        intent: What the LLM wants from this goal, free text. May be empty.
        when: The condition, in the LLM's words. May be empty.
        note: Any other remark on this goal. May be empty.
        relation: The goal's relationship directives (:class:`RelationDirective`). Empty by default. Not carried by
            ``to_dict`` / ``from_dict`` yet: the LLM's schema (``contracts.py``) has no field for it.
    """

    skill_id: str
    priority: float = 1.0
    params: tuple[float, ...] = ()
    constraints: tuple[str, ...] = ()
    horizon_seconds: float = 10.0
    deadline_tick: int | None = None
    condition: Condition | None = None
    addressee: str = ADDRESSEE_ALL
    objective: str | None = None
    intent: str = ""
    when: str = ""
    note: str = ""
    relation: tuple[RelationDirective, ...] = ()

    def __post_init__(self) -> None:
        """Validate the typed fields through GoalSpec and require strings for the words."""

        spec = GoalSpec(**{key: getattr(self, key) for key in _SPEC_FIELDS})
        for key in _SPEC_FIELDS:
            object.__setattr__(self, key, getattr(spec, key))
        for key in _TEXT_FIELDS:
            _string(getattr(self, key), f"GoalEntry.{key}")
        relation = tuple(self.relation)
        if not all(isinstance(item, RelationDirective) for item in relation):
            raise TypeError("GoalEntry.relation must hold RelationDirective values")
        object.__setattr__(self, "relation", relation)


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
        squads, objectives: As in :class:`ExecutiveState` (SYSTEM_ONE 9.15).
    """

    version: int = 0
    issued_tick: int = 0
    goals: tuple[GoalEntry, ...] = ()
    budget: ComputeBudget = field(default_factory=ComputeBudget)
    facts: tuple[str, ...] = ()
    notes: str = ""
    squads: tuple[Squad, ...] = ()
    objectives: tuple[Objective, ...] = ()

    def __post_init__(self) -> None:
        """Validate through ExecutiveState and normalize the sequences to tuples."""

        goals = _instances(self.goals, GoalEntry, "ExecutiveSection.goals")
        state = ExecutiveState(
            self.version, self.issued_tick, tuple(entry_to_spec(goal) for goal in goals), self.budget, self.facts,
            squads=self.squads, objectives=self.objectives,
        )
        object.__setattr__(self, "version", state.version)
        object.__setattr__(self, "issued_tick", state.issued_tick)
        object.__setattr__(self, "goals", goals)
        object.__setattr__(self, "facts", state.facts)
        object.__setattr__(self, "squads", state.squads)
        object.__setattr__(self, "objectives", state.objectives)
        _string(self.notes, "ExecutiveSection.notes")

    def to_executive_state(self) -> ExecutiveState:
        """Return E_t without the words: every typed field kept, the text dropped."""

        return ExecutiveState(
            self.version, self.issued_tick, tuple(entry_to_spec(goal) for goal in self.goals), self.budget, self.facts,
            squads=self.squads, objectives=self.objectives,
        )

    @classmethod
    def from_executive_state(cls, state: ExecutiveState) -> ExecutiveSection:
        """Return the section for an E_t from any existing writer; every text field is empty."""

        if not isinstance(state, ExecutiveState):
            raise TypeError(f"state must be an ExecutiveState, got {type(state).__name__}")
        goals = tuple(spec_to_entry(goal) for goal in state.goals)
        return cls(state.version, state.issued_tick, goals, state.budget, state.facts, squads=state.squads, objectives=state.objectives)

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
        return cls(
            state.version, state.issued_tick, goals, state.budget, state.facts, mapping["notes"],
            squads=state.squads, objectives=state.objectives,
        )

