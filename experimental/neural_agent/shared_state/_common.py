"""Constants and validation helpers shared by every section of S_t.

Split out of the single ``shared_state.py`` on 2026-10-18 without changes; the package docstring
(``shared_state/__init__.py``) is the design record.
"""

from __future__ import annotations


import dataclasses
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from experimental.neural_agent.contracts import (
    MAX_GOALS,
    GoalReport,
    GoalSpec,
    SkillOutcome,
    _finite_number,
    _non_negative_int,
    _text,
)
from experimental.neural_agent.skill_agent.contract import variables



SECTION_NAMES: tuple[str, ...] = ("executive", "belief", "decision", "requests", "writes", "skills", "events")
"""The seven sections of S_t, in document order."""
SECTION_WRITERS: Mapping[str, str] = MappingProxyType(
    {
        "executive": "llm",
        "belief": "belief",
        "decision": "jev",
        "requests": "jev",
        "writes": "jev",
        "skills": "skills",
        "events": "runtime",
    }
)
"""The one writer of each section. Fixed: ``SharedState.write`` refuses any other writer."""
LATENT_PREFIX_WIDTH: int = 8
"""Latent values kept per lane in the belief summary: the first 8 of BELIEF_LATENT_WIDTH."""
DECISION_HISTORY: int = 16
"""Default bound N of ``DecisionSection.recent``."""
INFORMATION_HISTORY: int = 8
"""Default bound of ``DecisionSection.information_requests``: the newest questions the decision layer
asks the LLM (SYSTEM_ONE 9.12)."""
OUTCOME_HISTORY: int = 16
"""Default bound of ``SkillsSection.outcomes``."""
REQUEST_LIMIT: int = 16
"""Default bound of ``RequestsSection.live`` per agent: L = 4 program steps x M = 4 concurrent requests."""
WRITES_PER_AGENT_TICK: int = 8
"""Writes of one agent at one tick (``WritesSection``). Equal to the skill layer's ``WRITE_SLOTS``
(``skill_agent/state/written.py``); a test pins the two equal."""
KNOWN_ENTITY_LIMIT: int = 16
"""Known entities of one agent (``SkillsSection.known_entities``): the skill layer's entity store holds this many by
default (``skill_agent/state/memory.py``, ENTITY_SLOTS)."""
STATUS_HISTORY: int = 256
"""Default bound of ``SkillsSection.statuses`` (the module docstring gives the sizing)."""
LLM_REJECTIONS: int = 5
"""Number of newest rejections that ``view_for_llm`` shows with their reasons."""
EVENT_WINDOW_TICKS: int = 11
"""Default event window: the session's host decision cadence (``SessionConfig.host_cadence_ticks``),
0.495 s at 0.045 s per tick."""

_SPEC_FIELDS: tuple[str, ...] = tuple(item.name for item in dataclasses.fields(GoalSpec))
_TEXT_FIELDS: tuple[str, ...] = ("intent", "when", "note")
DIRECTABLE_RELATIONS: tuple[str, ...] = tuple(
    entry.name for entry in variables.ENTITY_ENTRIES if entry.name not in ("remember", "integrity")
)
"""The relationship fields a goal's relation directive may name (MIDDLE_LAYER.md section 11): the skill layer's entity
fields except ``remember`` (the remember policy's) and ``integrity`` (never written by the decision layer). The same set
as ``jev/relations.py`` WRITABLE."""


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
