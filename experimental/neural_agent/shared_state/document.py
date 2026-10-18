"""The S_t document: six sections, one writer each, versioned; the section types and writers table.

Split out of the single ``shared_state.py`` on 2026-10-18 without changes; the package docstring
(``shared_state/__init__.py``) is the design record.
"""

from __future__ import annotations


import json
import threading
from typing import Any, Mapping


from experimental.neural_agent.contracts import (
    _non_negative_int,
)

from ._common import (
    SECTION_NAMES,
    SECTION_WRITERS,
)
from .schema import (
    validate,
)
from .sections.executive import (
    ExecutiveSection,
)
from .sections.belief import (
    BeliefSection,
)
from .sections.decision import (
    DecisionSection,
)
from .sections.requests import (
    RequestsSection,
)
from .sections.skills import (
    SkillsSection,
)
from .sections.events import (
    EventsSection,
)


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

