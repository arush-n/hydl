"""The skills section of S_t: active skills, outcomes and the status stream, written by the lower layer and the session.

Split out of the single ``shared_state.py`` on 2026-10-18 without changes; the package docstring
(``shared_state/__init__.py``) is the design record.
"""

from __future__ import annotations


import copy
import json
from dataclasses import dataclass, field
from typing import Any, Mapping


from experimental.neural_agent.contracts import (
    GoalReport,
    _text,
    _text_tuple,
)
from experimental.neural_agent.executive.store import validate_against_schema
from experimental.neural_agent.requests import RequestStatus

from .._common import (
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
    """

    active: tuple[str, ...] = ()
    runtime: Mapping[str, Any] = field(default_factory=dict)
    outcomes: tuple[GoalReport, ...] = ()
    outcome_limit: int = OUTCOME_HISTORY
    statuses: tuple[RequestStatus, ...] = ()
    status_limit: int = STATUS_HISTORY
    write_statuses: tuple[WriteStatus, ...] = ()

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
        )

