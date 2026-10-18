"""Versioned executive state store: the E_t writer seam (ADR-001 D3, seam 3).

:class:`ExecutiveStateStore` holds the executive state E_t that Level 3 writes and
the JEV reads. It accepts a proposal only when the proposal is valid JSON for
``EXECUTIVE_STATE_JSON_SCHEMA``, names only catalog skills, and carries the next
version. A rejected proposal leaves the state in force unchanged.

Formulas
--------
Let ``v_t`` be the version in force after ``t`` accepted proposals, with ``v_0 = 0``
(the empty state). A proposal ``p`` is accepted if and only if all three hold::

    p.version == v_t + 1                                    (else ValueError)
    json(p) validates against EXECUTIVE_STATE_JSON_SCHEMA   (else ValueError)
    every goal's skill_id is in the catalog                 (else ValueError)

The history is therefore ``v_0 < v_1 < ...`` with no gaps and no repeats, so a
decision that reports version ``v`` can be matched to exactly one accepted state.

Citations: ADR-001 D3 ("E_t is versioned so the JEV can report which version a
decision served"); ``contracts.EXECUTIVE_STATE_JSON_SCHEMA``;
``contracts.executive_arrays``. The keyword meanings that the manual validator enforces
are those of JSON Schema draft 2020-12 (https://json-schema.org/draft/2020-12/schema),
the dialect named by the schema's ``$schema`` field.
"""

from __future__ import annotations

import json
import math
import threading
from typing import Any, Mapping, Sequence

from experimental.neural_agent.contracts import (
    EXECUTIVE_STATE_JSON_SCHEMA,
    ComputeBudget,
    ExecutiveArrays,
    ExecutiveState,
    executive_arrays,
)

# Keywords that only annotate a schema. They never reject a value.
_ANNOTATION_KEYWORDS = frozenset({"$schema", "title", "description"})
# Constraint keywords that the manual validator enforces. Any other keyword fails closed.
_CONSTRAINT_KEYWORDS = frozenset(
    {
        "type",
        "required",
        "properties",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "minLength",
        "enum",
        "minimum",
        "maximum",
        "exclusiveMinimum",
    }
)
_SUPPORTED_KEYWORDS = _ANNOTATION_KEYWORDS | _CONSTRAINT_KEYWORDS


def _is_number(value: Any) -> bool:
    """Return True for a JSON number: an int or a float, and not a bool."""

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _has_json_type(value: Any, name: str) -> bool:
    """Return True when ``value`` has the JSON Schema type ``name``.

    ``integer`` accepts an int, or a float with no fractional part, as JSON Schema does.
    Bools are never numbers.
    """

    if name == "object":
        return isinstance(value, Mapping)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    if name == "number":
        return _is_number(value)
    if name == "integer":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and value.is_integer()
        )
    raise ValueError(f"unknown JSON type {name!r} in schema")


def _json_kind(value: Any) -> str:
    """Return the JSON type name of ``value``, for error messages."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    return type(value).__name__


def validate_against_schema(value: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    """Check a JSON-shaped ``value`` against the schema subset used by EXECUTIVE_STATE_JSON_SCHEMA.

    The check is manual because ``jsonschema`` is not installed in the repository
    venv. It enforces ``type`` (one name or a list), ``properties``, ``required``,
    ``additionalProperties`` (True or False), ``items``, ``minItems``, ``maxItems``,
    ``minLength``, ``minimum``, ``maximum``, ``exclusiveMinimum`` and ``enum`` (the
    value must be one of the listed values). Annotation
    keywords (``$schema``, ``title``, ``description``) are ignored. Any other keyword
    raises, so a new constraint cannot pass silently.

    Args:
        value: A JSON-shaped value: a mapping, list, str, int, float, bool or None.
        schema: A schema mapping in the form of EXECUTIVE_STATE_JSON_SCHEMA.
        path: Location of ``value`` for error messages.

    Raises:
        ValueError: If ``value`` breaks a constraint, or ``schema`` uses a keyword
            this validator does not enforce.
    """

    unsupported = sorted(set(schema) - _SUPPORTED_KEYWORDS)
    if unsupported:
        raise ValueError(
            f"{path}: schema keywords {unsupported} are not supported by the manual validator"
        )
    additional = schema.get("additionalProperties", True)
    if additional is not True and additional is not False:
        raise ValueError(f"{path}: additionalProperties must be a boolean in this validator")

    allowed = schema.get("type")
    if allowed is not None:
        names = [allowed] if isinstance(allowed, str) else list(allowed)
        if not any(_has_json_type(value, name) for name in names):
            raise ValueError(f"{path} must be {' or '.join(names)}, got {_json_kind(value)}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} must be one of {list(schema['enum'])}, got {value!r}")

    if _is_number(value):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite, got {value}")
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path} must be at least {schema['minimum']}, got {value}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path} must be at most {schema['maximum']}, got {value}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise ValueError(f"{path} must be greater than {schema['exclusiveMinimum']}, got {value}")

    if isinstance(value, str) and "minLength" in schema and len(value) < schema["minLength"]:
        raise ValueError(f"{path} must have at least {schema['minLength']} character(s)")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValueError(f"{path} has {len(value)} items; the minimum is {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValueError(f"{path} has {len(value)} items; the limit is {schema['maxItems']}")
        if "items" in schema:
            for index, item in enumerate(value):
                validate_against_schema(item, schema["items"], f"{path}[{index}]")

    if isinstance(value, Mapping):
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            raise ValueError(f"{path} is missing required keys {missing}")
        properties = schema.get("properties", {})
        if additional is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ValueError(f"{path} has unexpected keys {extra}")
        for key, sub_schema in properties.items():
            if key in value:
                validate_against_schema(value[key], sub_schema, f"{path}.{key}")


def validate_catalog(skill_ids: Sequence[str]) -> tuple[str, ...]:
    """Return the skill catalog as a tuple, after checking it.

    Args:
        skill_ids: Skill ids in dispatcher order. Each must be a non-empty string, and
            no id may repeat. The list must not be empty.

    Returns:
        The same ids as a tuple, in the same order.

    Raises:
        TypeError: If ``skill_ids`` is a bare string, or an entry is not a string.
        ValueError: If the catalog is empty, an id is empty, or an id repeats.
    """

    if isinstance(skill_ids, (str, bytes)):
        raise TypeError("skill_ids must be a sequence of skill id strings, not one string")
    ids = tuple(skill_ids)
    if not ids:
        raise ValueError("skill_ids must name at least one skill")
    for position, skill_id in enumerate(ids):
        if not isinstance(skill_id, str):
            raise TypeError(f"skill_ids[{position}] must be a string, got {type(skill_id).__name__}")
        if not skill_id:
            raise ValueError(f"skill_ids[{position}] must be a non-empty string")
    if len(set(ids)) != len(ids):
        raise ValueError("skill_ids repeats an id; each catalog id must appear once")
    return ids


class ExecutiveStateStore:
    """The versioned E_t history for one skill catalog. One writer and any number of readers.

    The version and history change under a lock, so :meth:`current` always returns a
    complete, accepted state, never a partly written one.

    Args:
        skill_ids: The skill catalog in dispatcher order, validated by :func:`validate_catalog`.

    Raises:
        TypeError: If ``skill_ids`` is a bare string, or an entry is not a string.
        ValueError: If the catalog is empty or repeats an id.
    """

    def __init__(self, skill_ids: Sequence[str]) -> None:
        self._skill_ids = validate_catalog(skill_ids)
        self._known = frozenset(self._skill_ids)
        self._lock = threading.Lock()
        self._history: list[ExecutiveState] = [
            ExecutiveState(version=0, issued_tick=0, goals=(), budget=ComputeBudget(), facts=())
        ]

    @property
    def skill_ids(self) -> tuple[str, ...]:
        """The skill catalog in dispatcher order."""

        return self._skill_ids

    def current(self) -> ExecutiveState:
        """Return the state in force, the last accepted version."""

        with self._lock:
            return self._history[-1]

    def history(self) -> tuple[ExecutiveState, ...]:
        """Return every accepted state, oldest first, starting with version 0."""

        with self._lock:
            return tuple(self._history)

    def propose(self, state: ExecutiveState) -> ExecutiveState:
        """Accept ``state`` as the next version, or raise and keep the state in force.

        Args:
            state: The proposed ExecutiveState. Its version must be the current version plus one.

        Returns:
            The accepted state.

        Raises:
            TypeError: If ``state`` is not an ExecutiveState.
            ValueError: If the version is not current + 1, a goal names a skill that is not
                in the catalog, or the JSON form breaks EXECUTIVE_STATE_JSON_SCHEMA.
        """

        if not isinstance(state, ExecutiveState):
            raise TypeError(f"proposal must be an ExecutiveState, got {type(state).__name__}")
        validate_against_schema(json.loads(state.to_json()), EXECUTIVE_STATE_JSON_SCHEMA)
        unknown = [goal.skill_id for goal in state.goals if goal.skill_id not in self._known]
        if unknown:
            raise ValueError(
                f"proposal names unknown skill id(s) {unknown}; catalog is {list(self._skill_ids)}"
            )
        with self._lock:
            expected = self._history[-1].version + 1
            if state.version != expected:
                raise ValueError(
                    f"proposal version {state.version} must be {expected} (current version plus one)"
                )
            self._history.append(state)
        return state

    def to_arrays(self, batch: int) -> ExecutiveArrays:
        """Lower the current state with :func:`contracts.executive_arrays`.

        Args:
            batch: Number of lanes B, a positive int.

        Returns:
            ExecutiveArrays with ``skill_index`` int32[B, 4], ``priority`` float32[B, 4],
            ``params`` float32[B, 4, 4], ``goal_valid`` bool[B, 4], ``budget`` float32[B, 3]
            and ``version`` int32[B].

        Raises:
            TypeError: If ``batch`` is not an integer.
            ValueError: If ``batch`` is not positive, or the catalog does not start with
                ``"idle"``, because ``contracts.executive_arrays`` requires the no-op skill
                at index 0.
        """

        return executive_arrays(self.current(), self._skill_ids, batch)
