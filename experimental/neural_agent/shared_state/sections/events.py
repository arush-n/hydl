"""The events section of S_t: the active interrupt events per lane, written by the runtime through the session.

Split out of the single ``shared_state.py`` on 2026-10-18 without changes; the package docstring
(``shared_state/__init__.py``) is the design record.
"""

from __future__ import annotations


from dataclasses import dataclass
from typing import Any, Mapping


from experimental.neural_agent.contracts import (
    EVENT_NAMES,
    _text_tuple,
)
from experimental.neural_agent.executive.store import validate_against_schema

from .._common import (
    EVENT_WINDOW_TICKS,
    _positive_int,
)
from ..schema import (
    _EVENTS_SECTION_SCHEMA,
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

