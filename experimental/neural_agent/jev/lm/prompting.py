"""Prompt, JSON Schema and fail-closed parsing for the quick language-model JEV.

ADR-001 D2 says the J-LM reads B_t and E_t and emits one ``ManagedGoal``. This
module turns those records into text (:func:`build_decision_prompt`), defines the
JSON Schema the reply must satisfy (:data:`DECISION_JSON_SCHEMA`), and turns the
reply back into a skill index, four parameters and a continue flag
(:func:`parse_decision`). Nothing here calls a model.

Prompt order follows ADR-001 D3. The stable part (task, skill catalogue, E_t)
comes first and the volatile part (current skill, B_t) comes last, so a backend
that caches a common prefix can reuse it across decisions. That benefit is
inferred from D3. It has not been measured on this path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from experimental.neural_agent.contracts import (
    BELIEF_FEATURE_COUNT,
    BELIEF_FEATURE_NAMES,
    PARAM_WIDTH,
    ExecutiveState,
)

SYSTEM_PROMPT: str = (
    "You are the fast decision model of a game agent. Choose the skill to run next "
    "from the listed skills, using only the belief features and executive goals in "
    "the message. A feature marked unknown was not published and is not zero. "
    "Reply with one JSON object and nothing else."
)

TASK_TEXT: str = (
    "Choose the next skill for the agent. Reply with one JSON object with keys: "
    "skill_id (one of the listed skills), params (at most 4 numbers), "
    "continue_current (true to keep the current skill, false to switch), "
    "and optionally reason (at most a few words; omit it to answer faster)."
)

DECISION_KEYS: frozenset[str] = frozenset(
    {"skill_id", "params", "continue_current", "reason"}
)

# The reply schema. Ollama constrains decoding to it. The Anthropic backend
# sends a copy with the numeric and array limits removed (anthropic_quick.py).
DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["skill_id", "params", "continue_current"],
    "properties": {
        "skill_id": {"type": "string"},
        "params": {
            "type": "array",
            "items": {"type": "number"},
            "maxItems": PARAM_WIDTH,
        },
        "continue_current": {"type": "boolean"},
        "reason": {"type": "string", "maxLength": 60},
    },
}


def validate_skill_ids(skill_ids: Sequence[str]) -> tuple[str, ...]:
    """Return the skill catalogue as a tuple, checked for shape and uniqueness.

    Args:
        skill_ids: Skill ids in dispatcher order. The position of an id is its
            ``skill_index``. The sequence must not be empty.

    Returns:
        The ids as a tuple of non-empty strings, in the same order.

    Raises:
        ValueError: If ``skill_ids`` is a bare string, empty, holds a value that
            is not a non-empty string, or repeats an id.
    """

    if isinstance(skill_ids, (str, bytes)):
        raise ValueError("skill_ids must be a sequence of skill id strings, not one string")
    catalogue = tuple(skill_ids)
    if not catalogue:
        raise ValueError("skill_ids must not be empty")
    for position, skill_id in enumerate(catalogue):
        if not isinstance(skill_id, str) or not skill_id:
            raise ValueError(f"skill_ids[{position}] must be a non-empty string")
    if len(set(catalogue)) != len(catalogue):
        raise ValueError("skill_ids repeats an id; each catalogue id must appear once")
    return catalogue


def build_decision_prompt(
    features_row: Any,
    feature_known_row: Any,
    executive: ExecutiveState,
    skill_ids: Sequence[str],
    current_skill_id: str,
) -> str:
    """Render one lane's decision request as text.

    Stable part first, then volatile part (ADR-001 D3)::

        task text
        Skills: <skill_ids joined by ", ">
        Executive state E_t, version V, issued at tick T.
        Goals: one line per goal (priority, params, constraints, horizon, deadline)
        Facts: one line per fact
        Current skill: <current_skill_id>
        Belief features B_t: one line per feature, "name = value" or "name = unknown"

    Args:
        features_row: One lane of ``SharedBeliefState.features``. Shape
            ``(BELIEF_FEATURE_COUNT,)``, that is ``(10,)``, numeric.
        feature_known_row: One lane of ``SharedBeliefState.feature_known``. Shape
            ``(10,)``, bool. A False entry is printed as ``unknown``, never as zero.
        executive: The ExecutiveState E_t that this decision serves.
        skill_ids: The skill catalogue, in dispatcher order.
        current_skill_id: The skill of the goal in force. Must be in ``skill_ids``.

    Returns:
        The prompt text. Every belief feature name and every skill id appears in it.

    Raises:
        TypeError: If ``executive`` is not an ExecutiveState.
        ValueError: If a shape or dtype is wrong, a known feature is not finite,
            a goal names a skill outside the catalogue, or ``current_skill_id``
            is not in the catalogue.
    """

    if not isinstance(executive, ExecutiveState):
        raise TypeError("executive must be an ExecutiveState")
    catalogue = validate_skill_ids(skill_ids)
    if current_skill_id not in catalogue:
        raise ValueError(
            f"current skill {current_skill_id!r} is not in the skill catalogue {catalogue}"
        )
    for slot, goal in enumerate(executive.goals):
        if goal.skill_id not in catalogue:
            raise ValueError(f"executive goal {slot} names unknown skill {goal.skill_id!r}")
    features = _feature_row(features_row)
    known = _known_row(feature_known_row)

    stable = [
        TASK_TEXT,
        f"Skills: {', '.join(catalogue)}",
        f"Executive state E_t, version {executive.version}, issued at tick {executive.issued_tick}.",
        "Goals (priority is the executive's relative weight):",
        *_goal_lines(executive),
        "Facts:",
        *_fact_lines(executive),
    ]
    volatile = [
        f"Current skill: {current_skill_id}",
        "Belief features B_t (unknown means not published; it is not zero):",
        *_feature_lines(features, known),
    ]
    return "\n".join(stable) + "\n\n" + "\n".join(volatile)


def parse_decision(decision: Any, skill_ids: Sequence[str]) -> tuple[int, list[float], bool]:
    """Validate one reply and return ``(skill_index, params, continue_current)``.

    The reply must be a mapping with exactly the keys ``skill_id``, ``params``,
    ``continue_current`` and ``reason``. ``skill_id`` must name a skill in the
    catalogue. ``params`` must be a JSON array of at most ``PARAM_WIDTH`` finite
    numbers (bools are not numbers). It is zero-padded to ``PARAM_WIDTH``.
    ``continue_current`` must be a bool. ``reason`` must be a string. Nothing is
    coerced or guessed.

    Args:
        decision: The decoded reply, as returned by a DecisionLM.
        skill_ids: The skill catalogue, in dispatcher order.

    Returns:
        ``skill_index`` is the position of ``skill_id`` in ``skill_ids``.
        ``params`` is a list of ``PARAM_WIDTH`` floats. ``continue_current`` is a bool.

    Raises:
        ValueError: On any failed check. Every failure is a ValueError.
    """

    if not isinstance(decision, Mapping):
        raise ValueError(f"decision must be a JSON object, got {type(decision).__name__}")
    keys = set(decision.keys())
    missing = sorted(str(key) for key in DECISION_KEYS - keys)
    unexpected = sorted(str(key) for key in keys - DECISION_KEYS)
    if missing:
        raise ValueError(f"decision is missing keys {missing}")
    if unexpected:
        raise ValueError(f"decision has unexpected keys {unexpected}")

    catalogue = validate_skill_ids(skill_ids)
    skill_id = decision["skill_id"]
    if not isinstance(skill_id, str) or skill_id not in catalogue:
        raise ValueError(f"unknown skill {skill_id!r}; expected one of {catalogue}")
    params = _params(decision["params"])
    continue_current = decision["continue_current"]
    if not isinstance(continue_current, bool):
        raise ValueError("continue_current must be a boolean")
    if "reason" in decision and not isinstance(decision["reason"], str):
        raise ValueError("reason must be a string")
    return catalogue.index(skill_id), params, continue_current


def _feature_row(value: Any) -> np.ndarray:
    """Return one lane of features as float64, shape ``(BELIEF_FEATURE_COUNT,)``."""

    array = np.asarray(value)
    if array.shape != (BELIEF_FEATURE_COUNT,):
        raise ValueError(
            f"features_row must have shape ({BELIEF_FEATURE_COUNT},), got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"features_row must be numeric, got dtype {array.dtype}")
    return array.astype(np.float64)


def _known_row(value: Any) -> np.ndarray:
    """Return one lane of availability flags, bool, shape ``(BELIEF_FEATURE_COUNT,)``."""

    array = np.asarray(value)
    if array.shape != (BELIEF_FEATURE_COUNT,):
        raise ValueError(
            f"feature_known_row must have shape ({BELIEF_FEATURE_COUNT},), got {array.shape}"
        )
    if array.dtype != np.bool_:
        raise ValueError(f"feature_known_row must be bool, got dtype {array.dtype}")
    return array


def _feature_lines(features: np.ndarray, known: np.ndarray) -> list[str]:
    """Render each feature as ``name = value`` (3 decimals) or ``name = unknown``."""

    lines: list[str] = []
    for index, name in enumerate(BELIEF_FEATURE_NAMES):
        if not known[index]:
            lines.append(f"  {name} = unknown")
            continue
        value = float(features[index])
        if not math.isfinite(value):
            raise ValueError(f"feature {name} is marked known but is not finite")
        lines.append(f"  {name} = {value:.3f}")
    return lines


def _goal_lines(executive: ExecutiveState) -> list[str]:
    """Render each E_t goal on one line, or ``none`` when E_t has no goals."""

    if not executive.goals:
        return ["  none"]
    lines: list[str] = []
    for slot, goal in enumerate(executive.goals):
        deadline = "none" if goal.deadline_tick is None else str(goal.deadline_tick)
        lines.append(
            f"  goal {slot}: skill={goal.skill_id} priority={goal.priority:g} "
            f"params={list(goal.params)} constraints={list(goal.constraints)} "
            f"horizon_seconds={goal.horizon_seconds:g} deadline_tick={deadline}"
        )
    return lines


def _fact_lines(executive: ExecutiveState) -> list[str]:
    """Render each E_t fact on its own line, or ``none`` when there are no facts."""

    if not executive.facts:
        return ["  none"]
    return [f"  - {fact}" for fact in executive.facts]


def _params(value: Any) -> list[float]:
    """Return a reply's ``params`` as ``PARAM_WIDTH`` floats, zero-padded."""

    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (list, tuple)):
        raise ValueError("params must be a JSON array of numbers")
    if len(value) > PARAM_WIDTH:
        raise ValueError(f"params has {len(value)} values; the limit is {PARAM_WIDTH}")
    numbers: list[float] = []
    for position, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"params[{position}] must be a number")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"params[{position}] must be finite")
        numbers.append(number)
    numbers.extend([0.0] * (PARAM_WIDTH - len(numbers)))
    return numbers

