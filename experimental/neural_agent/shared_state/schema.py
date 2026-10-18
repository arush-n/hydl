"""The JSON Schema of S_t (only the keywords executive/store.py validate_against_schema enforces) and ``validate``.

Split out of the single ``shared_state.py`` on 2026-10-18 without changes; the package docstring
(``shared_state/__init__.py``) is the design record.
"""

from __future__ import annotations


import copy
from typing import Any


from experimental.neural_agent.contracts import (
    BELIEF_FEATURE_NAMES,
    EXECUTIVE_STATE_JSON_SCHEMA,
    INT32_MAX,
    MAX_GOALS,
    PARAM_WIDTH,
)
from experimental.neural_agent.executive.store import validate_against_schema

from ._common import (
    LATENT_PREFIX_WIDTH,
    SECTION_NAMES,
    _TEXT_FIELDS,
)


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
    intent={
        "type": "string",
        "description": (
            "What you want from this goal, in your own words: start with the skill's catalogue id, then a few"
            " words on what the skill is for; may be empty."
        ),
    },
    when={
        "type": "string",
        "description": (
            "When this goal applies, in your own words, as one clause: a feature word, a comparator and a number"
            " with its unit, or 'always'; may be empty."
        ),
    },
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
        "information_requests": {
            "type": "array",
            "items": _object_schema(
                {
                    "tick": _count_schema(),
                    "source": {"type": "string", "minLength": 1},
                    "executive_version": _count_schema(),
                    "slot": {"type": ["integer", "null"], "minimum": 0},
                    "lane": {"type": ["integer", "null"], "minimum": 0},
                    "text": {"type": "string"},
                    "kind": {"type": ["string", "null"]},
                    "argument": {"type": ["number", "null"]},
                    "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                    "reason": {"type": "string"},
                }
            ),
        },
        "information_limit": _count_schema(1),
        "readings": {
            "type": "array",
            "items": _object_schema(
                {
                    "executive_version": _count_schema(),
                    "slot": _count_schema(),
                    "kind": {"type": ["string", "null"]},
                    "argument": {"type": ["number", "null"]},
                    "source": {"type": "string", "minLength": 1},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                }
            ),
        },
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


EXECUTIVE_SECTION_JSON_SCHEMA: dict[str, Any] = _EXECUTIVE_SECTION_SCHEMA
"""The executive section's schema: ExecutiveState's keys plus the words per goal (intent, when, note) and
``notes``. What the LLM replies in (``executive/anthropic_backend.py``)."""
