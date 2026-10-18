"""Shared contract for the neural agent: the ADR-001 three-level agent as code.

ADR-001 is a local-only record: this clone excludes it through .git/info/exclude and
it is not in the repository.

``experimental/neural_agent`` implements ADR-001 (``agents/design/ADR-001-three-level-agent.md``).
This file is the only interface its modules share. Each module imports
cross-module names from here, so a change to this file changes every module.

The three levels of ADR-001 section 2 use these types:

* Level 3, the LLM executive (asynchronous, event-driven), writes an
  :class:`ExecutiveState` (E_t) and lowers it with :func:`executive_arrays`.
* Level 2, the JEV decision model (about 2 Hz plus interrupts, hard latency
  budget), reads a :class:`SharedBeliefState` (B_t) and the
  :class:`ExecutiveArrays`, and returns a goal: a :class:`GoalPayload` inside an
  ADK ``ManagedGoal``, built with :func:`make_goal` or :func:`idle_goal`. The
  goal's control is the chosen skill's: :func:`control_for` reads it from a
  :class:`SkillControlTable` built from the skill contracts (SYSTEM_ONE.md
  section 8.12, item 1).
* Level 1, the skill policies (every decision step), implement :class:`Skill`
  and return int32 action factors on the 12-head surface, restricted to the
  skill scope and to the runtime action mask. Each :class:`SkillContract`
  declares the commitment, duration and emergency events of its goals.

Conventions:

* Every array has a leading lane axis ``B``, except the catalogue axis ``K``
  of :class:`SkillControlTable`. Records that cross the ADK
  hierarchy core (:class:`SharedBeliefState`, :class:`ExecutiveArrays`,
  :class:`GoalPayload`) are batched JAX pytrees. Host-side records
  (:class:`ExecutiveState`, :class:`GoalSpec`, :class:`ComputeBudget`,
  :class:`SkillContract`, :class:`GoalReport`, :class:`JevBudget`,
  :class:`JevStats`) never enter a jitted core.
* Deployed code reads only :class:`~adk.architecture.inputs.ActorPolicyInput`,
  which carries legal observation products. ``JaxTrainingInput`` is privileged
  and is neither imported nor accepted here (ADR-001 section 2).
* Invalid input raises ``TypeError`` or ``ValueError`` when a host record is
  built. Array builders (``make_goal``) cast dtypes so that every goal has the
  dtypes ``idle_goal`` has; the ADK core's ``lax.cond`` requires identical
  structures on both branches. Whole-number floats are accepted where JSON
  Schema ``integer`` would accept them (an LLM may emit ``1.0``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Mapping, NamedTuple, Protocol, Sequence, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np

from adk.architecture.components import GoalDecision
from adk.architecture.decision_context import DecisionContext, DecisionEvent
from adk.architecture.hierarchy import GoalControl, GoalRuntimeContext, ManagedGoal
from adk.architecture.inputs import ActorPolicyInput

# Maximum number of executive goals in one ExecutiveState, and the goal-slot
# axis of ExecutiveArrays.
INT32_MAX = 2**31 - 1
"""Upper bound for every integer that becomes an int32 array leaf."""

MAX_GOALS: int = 4
# Parameter slots per goal. Shorter parameter vectors are zero-padded.
PARAM_WIDTH: int = 4
# Width of the skill-local state float32[B, SKILL_STATE_WIDTH].
SKILL_STATE_WIDTH: int = 128
# Factored action heads on the published action surface.
ACTION_HEADS: int = 12
# Width of SharedBeliefState.latent, float32[B, BELIEF_LATENT_WIDTH].
BELIEF_LATENT_WIDTH: int = 64
# Id of the no-op skill. By convention it is catalog index 0 (see idle_goal).
SKILL_IDLE: str = "idle"
# Saturation horizon in ticks for the tick-count belief features. The
# normalization formula belongs to the belief module, not to this contract.
TICKS_SATURATION: float = 256.0
# DecisionEvent member names in enum order (14 entries, decision_context.py).
EVENT_NAMES: tuple[str, ...] = tuple(event.name for event in DecisionEvent)

# Shared belief features, in column order of SharedBeliefState.features.
BELIEF_FEATURE_NAMES: tuple[str, ...] = (
    "target_visible",
    "target_distance_blocks",
    "target_bearing_sin",
    "target_bearing_cos",
    "self_health_fraction",
    "target_health_fraction",
    "ticks_since_damage",
    "ticks_since_target_seen",
    "self_speed_blocks_per_tick",
    "closing_speed_blocks_per_tick",
)
BELIEF_FEATURE_COUNT: int = len(BELIEF_FEATURE_NAMES)

_EXECUTIVE_KEYS = frozenset({"version", "issued_tick", "goals", "budget", "facts"})
_GOAL_KEYS = frozenset(
    {
        "skill_id",
        "priority",
        "params",
        "constraints",
        "horizon_seconds",
        "deadline_tick",
        "condition",
    }
)
_CONDITION_KEYS = frozenset({"kind", "argument"})
_BUDGET_KEYS = frozenset(
    {"jev_latency_ms", "search_expansions", "executive_calls_per_minute"}
)


def _finite_number(value: Any, name: str) -> float:
    """Return ``value`` as a float, rejecting bools, non-numbers and non-finite values."""

    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number}")
    if abs(number) > float(np.finfo(np.float32).max):
        raise ValueError(
            f"{name} must fit float32 (|x| <= {float(np.finfo(np.float32).max):.3e}), got {number}"
        )
    return number


def _non_negative_int(value: Any, name: str) -> int:
    """Return ``value`` as an int that is zero or greater; bools are rejected."""

    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    if value > INT32_MAX:
        raise ValueError(f"{name} must fit int32 (<= {INT32_MAX}), got {value}")
    return int(value)


def _text(value: Any, name: str) -> str:
    """Return ``value`` when it is a non-empty string."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _text_tuple(values: Any, name: str) -> tuple[str, ...]:
    """Return ``values`` as a tuple of non-empty strings. A bare string is rejected."""

    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of strings, not one string")
    return tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(values))


def _finite_tuple(values: Any, limit: int, name: str) -> tuple[float, ...]:
    """Return at most ``limit`` finite floats as a tuple. A bare string is rejected."""

    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numbers, not one string")
    items = tuple(values)
    if len(items) > limit:
        raise ValueError(f"{name} has {len(items)} values; the limit is {limit}")
    return tuple(
        _finite_number(item, f"{name}[{index}]") for index, item in enumerate(items)
    )


def _positive_batch(batch: Any) -> int:
    """Return ``batch`` as a positive int."""

    if isinstance(batch, bool) or not isinstance(batch, (int, np.integer)):
        raise TypeError(f"batch must be an integer, got {type(batch).__name__}")
    if batch < 1:
        raise ValueError(f"batch must be positive, got {batch}")
    return int(batch)


def _require_shape(value: jax.Array, expected: tuple[int, ...], name: str) -> None:
    """Raise ValueError unless ``value`` has exactly the ``expected`` shape."""

    if value.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {value.shape}")


def feature_index(name: str) -> int:
    """Return the column of a belief feature in ``SharedBeliefState.features``.

    Args:
        name: One of ``BELIEF_FEATURE_NAMES``.

    Returns:
        The column index, an integer in ``[0, BELIEF_FEATURE_COUNT)``.

    Raises:
        ValueError: If ``name`` is not a belief feature name.
    """

    try:
        return BELIEF_FEATURE_NAMES.index(name)
    except ValueError:
        raise ValueError(
            f"unknown belief feature {name!r}; expected one of {BELIEF_FEATURE_NAMES}"
        ) from None


class SharedBeliefState(NamedTuple):
    """Shared belief B_t, the one object every level reads (ADR-001 D6).

    This is ``BeliefUpdate.belief`` from ``adk.architecture.components``. The
    belief core must derive every field from legal observation products only.

    Shapes, with B lanes:
        features: float32[B, BELIEF_FEATURE_COUNT], columns named by
            ``BELIEF_FEATURE_NAMES``.
        feature_known: bool[B, BELIEF_FEATURE_COUNT]. False marks a value the
            backend did not publish, which is distinct from a real zero.
        latent: float32[B, BELIEF_LATENT_WIDTH], the recurrent belief latent.
    """

    features: jax.Array
    feature_known: jax.Array
    latent: jax.Array


CONDITION_KINDS: tuple[str, ...] = (
    "always",
    "on_damage",
    "target_visible",
    "target_not_visible",
    "target_lost_for",
    "distance_below",
    "distance_above",
    "self_health_below",
    "target_health_below",
)
"""The nine condition kinds a goal may carry (SYSTEM_ONE sections 8.11 and 9.12), ``always`` first.
Which belief feature a kind reads and how it is evaluated live in ``executive/goal_language.py``
(``Predicate``); this module only names the vocabulary the executive writes."""
CONDITION_ARGUMENT_UNITS: dict[str, str] = {
    "target_lost_for": "ticks",
    "distance_below": "blocks",
    "distance_above": "blocks",
    "self_health_below": "fraction",
    "target_health_below": "fraction",
}
"""The kinds that take a numeric argument, with the unit of the number (fraction: 0 to 1 of full health)."""


@dataclass(frozen=True)
class Condition:
    """A goal's typed condition: when the goal applies (host-side record, SYSTEM_ONE 9.12).

    Attributes:
        kind: One of ``CONDITION_KINDS``.
        argument: A finite number for the kinds in ``CONDITION_ARGUMENT_UNITS``, in that unit;
            None for every other kind.

    Raises:
        TypeError: If ``kind`` is not a string or ``argument`` is not a real number.
        ValueError: If ``kind`` is unknown, a numeric kind lacks its argument, or another kind has one.
    """

    kind: str
    argument: float | None = None

    def __post_init__(self) -> None:
        """Validate the kind and that exactly the numeric kinds carry a finite argument."""

        _text(self.kind, "Condition.kind")
        if self.kind not in CONDITION_KINDS:
            raise ValueError(f"unknown condition kind {self.kind!r}; expected one of {CONDITION_KINDS}")
        numeric = self.kind in CONDITION_ARGUMENT_UNITS
        if numeric != (self.argument is not None):
            raise ValueError(
                f"condition {self.kind!r} {'needs an' if numeric else 'takes no'} argument"
            )
        if self.argument is not None:
            object.__setattr__(
                self, "argument", _finite_number(self.argument, "Condition.argument")
            )


@dataclass(frozen=True)
class GoalSpec:
    """One goal the executive asks the JEV to pursue (host-side record).

    Attributes:
        skill_id: Non-empty catalog id of the skill to run.
        priority: Finite relative weight, default 1.0. The JEV interprets it.
        params: At most ``PARAM_WIDTH`` finite values. Zero-padded when lowered.
        constraints: Non-empty constraint labels for the JEV to respect.
        horizon_seconds: Planning horizon in seconds, finite and positive.
        deadline_tick: Simulation tick by which the goal should finish, a
            non-negative int, or None for no deadline.
        condition: The typed condition under which the goal applies, a
            :class:`Condition`, or None when it always applies (SYSTEM_ONE 9.12).

    Raises:
        TypeError: If a field has the wrong Python type.
        ValueError: If a value is empty, non-finite, out of range, or too long.
    """

    skill_id: str
    priority: float = 1.0
    params: tuple[float, ...] = ()
    constraints: tuple[str, ...] = ()
    horizon_seconds: float = 10.0
    deadline_tick: int | None = None
    condition: Condition | None = None

    def __post_init__(self) -> None:
        """Validate the fields and normalize them to their documented types."""

        _text(self.skill_id, "GoalSpec.skill_id")
        object.__setattr__(
            self, "priority", _finite_number(self.priority, "GoalSpec.priority")
        )
        object.__setattr__(
            self,
            "params",
            _finite_tuple(self.params, PARAM_WIDTH, "GoalSpec.params"),
        )
        object.__setattr__(
            self,
            "constraints",
            _text_tuple(self.constraints, "GoalSpec.constraints"),
        )
        horizon = _finite_number(self.horizon_seconds, "GoalSpec.horizon_seconds")
        if horizon <= 0.0:
            raise ValueError(f"GoalSpec.horizon_seconds must be positive, got {horizon}")
        object.__setattr__(self, "horizon_seconds", horizon)
        if self.deadline_tick is not None:
            object.__setattr__(
                self,
                "deadline_tick",
                _non_negative_int(self.deadline_tick, "GoalSpec.deadline_tick"),
            )
        if self.condition is not None and not isinstance(self.condition, Condition):
            raise TypeError(
                f"GoalSpec.condition must be a Condition or None, got {type(self.condition).__name__}"
            )


@dataclass(frozen=True)
class ComputeBudget:
    """Compute limits that the executive grants the JEV (host-side record).

    Attributes:
        jev_latency_ms: Hard per-decision latency budget for the JEV, finite and
            positive. An instance that misses it leaves the previous goal in
            force, and the miss is counted (ADR-001 section 2).
        search_expansions: Search node expansions allowed per decision, a
            non-negative int.
        executive_calls_per_minute: Executive (LLM) calls allowed per minute,
            finite and non-negative.

    Raises:
        TypeError: If a field has the wrong Python type.
        ValueError: If a value is non-positive, non-finite, or negative.
    """

    jev_latency_ms: float = 20.0
    search_expansions: int = 0
    executive_calls_per_minute: float = 2.0

    def __post_init__(self) -> None:
        """Validate the budget and normalize numeric fields to float or int."""

        latency = _finite_number(self.jev_latency_ms, "ComputeBudget.jev_latency_ms")
        if latency <= 0.0:
            raise ValueError(f"ComputeBudget.jev_latency_ms must be positive, got {latency}")
        object.__setattr__(self, "jev_latency_ms", latency)
        object.__setattr__(
            self,
            "search_expansions",
            _non_negative_int(self.search_expansions, "ComputeBudget.search_expansions"),
        )
        calls = _finite_number(
            self.executive_calls_per_minute,
            "ComputeBudget.executive_calls_per_minute",
        )
        if calls < 0.0:
            raise ValueError(
                f"ComputeBudget.executive_calls_per_minute must be >= 0, got {calls}"
            )
        object.__setattr__(self, "executive_calls_per_minute", calls)


@dataclass(frozen=True)
class ExecutiveState:
    """Executive state E_t, written by the LLM executive and read by the JEV.

    Host-side record. It reaches the hierarchy core only through
    :func:`executive_arrays`.

    Attributes:
        version: Non-negative version of this state. A decision reports the
            version it served (ADR-001 D3).
        issued_tick: Non-negative simulation tick at which it was issued.
        goals: At most ``MAX_GOALS`` :class:`GoalSpec` values.
        budget: The :class:`ComputeBudget` in force.
        facts: Non-empty strings the executive judges relevant.

    Raises:
        TypeError: If a field has the wrong Python type.
        ValueError: If a value is out of range or more than ``MAX_GOALS`` goals
            are given.
    """

    version: int
    issued_tick: int
    goals: tuple[GoalSpec, ...]
    budget: ComputeBudget
    facts: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate every field and normalize sequences to tuples."""

        object.__setattr__(
            self, "version", _non_negative_int(self.version, "ExecutiveState.version")
        )
        object.__setattr__(
            self,
            "issued_tick",
            _non_negative_int(self.issued_tick, "ExecutiveState.issued_tick"),
        )
        goals = tuple(self.goals)
        if len(goals) > MAX_GOALS:
            raise ValueError(
                f"ExecutiveState has {len(goals)} goals; the limit is {MAX_GOALS}"
            )
        for position, goal in enumerate(goals):
            if not isinstance(goal, GoalSpec):
                raise TypeError(f"ExecutiveState.goals[{position}] must be a GoalSpec")
        object.__setattr__(self, "goals", goals)
        if not isinstance(self.budget, ComputeBudget):
            raise TypeError("ExecutiveState.budget must be a ComputeBudget")
        object.__setattr__(
            self, "facts", _text_tuple(self.facts, "ExecutiveState.facts")
        )

    def to_json(self) -> str:
        """Serialize to canonical JSON text that matches EXECUTIVE_STATE_JSON_SCHEMA.

        Keys are sorted and NaN or infinity is refused, so the text is stable
        and always parses back to an equal state.

        Returns:
            A JSON object string.
        """

        return json.dumps(_executive_mapping(self), sort_keys=True, allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> ExecutiveState:
        """Parse JSON text produced by :meth:`to_json` (or any schema-valid text).

        Args:
            text: A JSON object string.

        Returns:
            The validated ExecutiveState.

        Raises:
            TypeError: If ``text`` is not a string or is not a JSON object.
            ValueError: If the JSON is malformed or a value fails validation.
        """

        if not isinstance(text, str):
            raise TypeError("text must be a JSON string")
        return cls.from_dict(json.loads(text))

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> ExecutiveState:
        """Build a state from a JSON-shaped mapping with exactly the schema keys.

        Args:
            mapping: A mapping whose keys are exactly ``version``,
                ``issued_tick``, ``goals``, ``budget`` and ``facts``, shaped as
                EXECUTIVE_STATE_JSON_SCHEMA describes.

        Returns:
            The validated ExecutiveState.

        Raises:
            TypeError: If a value has the wrong JSON type.
            ValueError: If a key is missing or unexpected, or a value fails
                validation.
        """

        root = _json_object(mapping, "ExecutiveState")
        _exact_keys(root, _EXECUTIVE_KEYS, "ExecutiveState")
        goals = tuple(
            _goal_from_mapping(item)
            for item in _json_array(root["goals"], "ExecutiveState.goals")
        )
        budget = _json_object(root["budget"], "ExecutiveState.budget")
        _exact_keys(budget, _BUDGET_KEYS, "ExecutiveState.budget")
        return cls(
            version=root["version"],
            issued_tick=root["issued_tick"],
            goals=goals,
            budget=ComputeBudget(
                jev_latency_ms=budget["jev_latency_ms"],
                search_expansions=budget["search_expansions"],
                executive_calls_per_minute=budget["executive_calls_per_minute"],
            ),
            facts=_json_array(root["facts"], "ExecutiveState.facts"),
        )


def _executive_mapping(state: ExecutiveState) -> dict[str, Any]:
    """Return the JSON-shaped mapping for ``state``, the inverse of ``from_dict``."""

    return {
        "version": state.version,
        "issued_tick": state.issued_tick,
        "goals": [
            {
                "skill_id": goal.skill_id,
                "priority": goal.priority,
                "params": list(goal.params),
                "constraints": list(goal.constraints),
                "horizon_seconds": goal.horizon_seconds,
                "deadline_tick": goal.deadline_tick,
                "condition": _condition_mapping(goal.condition),
            }
            for goal in state.goals
        ],
        "budget": {
            "jev_latency_ms": state.budget.jev_latency_ms,
            "search_expansions": state.budget.search_expansions,
            "executive_calls_per_minute": state.budget.executive_calls_per_minute,
        },
        "facts": list(state.facts),
    }


def _json_object(value: Any, name: str) -> Mapping[str, Any]:
    """Return ``value`` when it is a JSON object (a mapping)."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _json_array(value: Any, name: str) -> list[Any]:
    """Return ``value`` when it is a JSON array (a list)."""

    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _exact_keys(mapping: Mapping[str, Any], expected: frozenset[str], name: str) -> None:
    """Raise ValueError unless ``mapping`` has exactly the ``expected`` keys."""

    keys = {str(key) for key in mapping}
    missing = sorted(expected - keys)
    unexpected = sorted(keys - expected)
    if missing:
        raise ValueError(f"{name} is missing required keys {missing}")
    if unexpected:
        raise ValueError(f"{name} has unexpected keys {unexpected}")


def _condition_mapping(condition: Condition | None) -> dict[str, Any] | None:
    """Return the JSON-shaped mapping of a typed condition, or None for no condition."""

    if condition is None:
        return None
    return {"kind": condition.kind, "argument": condition.argument}


def _condition_from_mapping(item: Any) -> Condition | None:
    """Build the typed condition from its JSON object (null means none), with exact key checking."""

    if item is None:
        return None
    entry = _json_object(item, "goal.condition")
    _exact_keys(entry, _CONDITION_KEYS, "goal.condition")
    return Condition(kind=entry["kind"], argument=entry["argument"])


def _goal_from_mapping(item: Any) -> GoalSpec:
    """Build one GoalSpec from its JSON object, with exact key checking."""

    entry = _json_object(item, "ExecutiveState.goals[]")
    _exact_keys(entry, _GOAL_KEYS, "goal")
    return GoalSpec(
        skill_id=entry["skill_id"],
        priority=entry["priority"],
        params=_json_array(entry["params"], "goal.params"),
        constraints=_json_array(entry["constraints"], "goal.constraints"),
        horizon_seconds=entry["horizon_seconds"],
        deadline_tick=entry["deadline_tick"],
        condition=_condition_from_mapping(entry["condition"]),
    )


EXECUTIVE_STATE_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ExecutiveState",
    "description": "Executive state E_t from experimental/neural_agent/contracts.py.",
    "type": "object",
    "additionalProperties": False,
    "required": ["version", "issued_tick", "goals", "budget", "facts"],
    "properties": {
        "version": {"type": "integer", "minimum": 0, "maximum": 2147483647},
        "issued_tick": {"type": "integer", "minimum": 0, "maximum": 2147483647},
        "goals": {
            "type": "array",
            "maxItems": MAX_GOALS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "skill_id",
                    "priority",
                    "params",
                    "constraints",
                    "horizon_seconds",
                    "deadline_tick",
                    "condition",
                ],
                "properties": {
                    "skill_id": {"type": "string", "minLength": 1},
                    "priority": {"type": "number"},
                    "params": {
                        "type": "array",
                        "maxItems": PARAM_WIDTH,
                        "items": {"type": "number"},
                    },
                    "constraints": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "horizon_seconds": {"type": "number", "exclusiveMinimum": 0},
                    "deadline_tick": {
                        "type": ["integer", "null"],
                        "minimum": 0,
                    },
                    "condition": {
                        "type": ["object", "null"],
                        "description": (
                            "When this goal applies: null means always; otherwise kind names the"
                            " condition and argument is its number (for *_below, *_above and"
                            " *_lost_for kinds; null for the others)."
                        ),
                        "additionalProperties": False,
                        "required": ["kind", "argument"],
                        "properties": {
                            "kind": {"type": "string", "enum": list(CONDITION_KINDS)},
                            "argument": {"type": ["number", "null"]},
                        },
                    },
                },
            },
        },
        "budget": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "jev_latency_ms",
                "search_expansions",
                "executive_calls_per_minute",
            ],
            "properties": {
                "jev_latency_ms": {"type": "number", "exclusiveMinimum": 0},
                "search_expansions": {"type": "integer", "minimum": 0},
                "executive_calls_per_minute": {"type": "number", "minimum": 0},
            },
        },
        "facts": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
    },
}


class ExecutiveArrays(NamedTuple):
    """The executive state lowered to batched arrays for the JEV (ADR-001 D3).

    Shapes, with B lanes and G = MAX_GOALS, P = PARAM_WIDTH:
        skill_index: int32[B, G]. Index into the skill catalog, -1 in an empty slot.
        priority: float32[B, G].
        params: float32[B, G, P], zero-padded.
        goal_valid: bool[B, G]. True where a goal occupies the slot.
        budget: float32[B, 3] = [jev_latency_ms, search_expansions,
            executive_calls_per_minute].
        version: int32[B], the ExecutiveState.version.
    """

    skill_index: jax.Array
    priority: jax.Array
    params: jax.Array
    goal_valid: jax.Array
    budget: jax.Array
    version: jax.Array


def _skill_index_map(skill_ids: Sequence[str]) -> dict[str, int]:
    """Map each catalog skill id to its position. Duplicate ids are rejected."""

    if isinstance(skill_ids, (str, bytes)):
        raise TypeError("skill_ids must be a sequence of skill id strings")
    index_of: dict[str, int] = {}
    for position, skill_id in enumerate(skill_ids):
        name = _text(skill_id, f"skill_ids[{position}]")
        if name in index_of:
            raise ValueError(
                f"skill_ids repeats {name!r}; each catalog id must appear once"
            )
        index_of[name] = position
    return index_of


def executive_arrays(
    state: ExecutiveState,
    skill_ids: Sequence[str],
    batch: int,
) -> ExecutiveArrays:
    """Lower an ExecutiveState into the batched arrays that the JEV reads.

    The same state is replicated to every lane. Goal ``g`` of ``state`` fills
    slot ``g``. Unused slots hold ``skill_index`` -1, ``goal_valid`` False and
    zero priority and parameters.

    Args:
        state: The executive state E_t to lower.
        skill_ids: The skill catalog in dispatcher order. The position of an id
            here becomes its ``skill_index``.
        batch: Number of lanes B, a positive int.

    Returns:
        ExecutiveArrays whose shapes are ``skill_index`` int32[B, MAX_GOALS],
        ``priority`` float32[B, MAX_GOALS], ``params`` float32[B, MAX_GOALS,
        PARAM_WIDTH], ``goal_valid`` bool[B, MAX_GOALS], ``budget`` float32[B, 3]
        and ``version`` int32[B].

    Raises:
        TypeError: If ``state`` is not an ExecutiveState, or ``skill_ids`` is
            not a sequence of strings, or ``batch`` is not an integer.
        KeyError: If a goal names a skill id that is not in ``skill_ids``.
        ValueError: If ``skill_ids`` repeats an id, if ``skill_ids[0]`` is not
            ``SKILL_IDLE``, or if ``batch`` is not positive.
    """

    if not isinstance(state, ExecutiveState):
        raise TypeError("state must be an ExecutiveState")
    lanes = _positive_batch(batch)
    index_of = _skill_index_map(skill_ids)
    if index_of and next(iter(index_of)) != SKILL_IDLE:
        raise ValueError(
            f"skill_ids[0] must be {SKILL_IDLE!r} because idle_goal selects index 0; "
            f"got {next(iter(index_of))!r}"
        )

    skill_index = np.full((lanes, MAX_GOALS), -1, dtype=np.int32)
    priority = np.zeros((lanes, MAX_GOALS), dtype=np.float32)
    params = np.zeros((lanes, MAX_GOALS, PARAM_WIDTH), dtype=np.float32)
    goal_valid = np.zeros((lanes, MAX_GOALS), dtype=np.bool_)
    for slot, goal in enumerate(state.goals):
        if goal.skill_id not in index_of:
            raise KeyError(
                f"goal {slot} names unknown skill id {goal.skill_id!r}; "
                "it is not in skill_ids"
            )
        skill_index[:, slot] = index_of[goal.skill_id]
        priority[:, slot] = goal.priority
        params[:, slot, : len(goal.params)] = goal.params
        goal_valid[:, slot] = True

    budget_row = np.asarray(
        [
            state.budget.jev_latency_ms,
            state.budget.search_expansions,
            state.budget.executive_calls_per_minute,
        ],
        dtype=np.float32,
    )
    return ExecutiveArrays(
        skill_index=jnp.asarray(skill_index),
        priority=jnp.asarray(priority),
        params=jnp.asarray(params),
        goal_valid=jnp.asarray(goal_valid),
        budget=jnp.asarray(np.tile(budget_row, (lanes, 1))),
        version=jnp.full((lanes,), state.version, dtype=jnp.int32),
    )


def empty_executive_arrays(batch: int) -> ExecutiveArrays:
    """Return ExecutiveArrays for ``batch`` lanes with no goals.

    Use this before the executive has written any state. ``skill_index`` is -1
    everywhere, ``goal_valid`` is False everywhere, ``budget`` holds the
    ``ComputeBudget()`` defaults and ``version`` is 0.

    Args:
        batch: Number of lanes B, a positive int.

    Returns:
        ExecutiveArrays with the shapes documented on :func:`executive_arrays`.

    Raises:
        TypeError: If ``batch`` is not an integer.
        ValueError: If ``batch`` is not positive.
    """

    empty = ExecutiveState(
        version=0,
        issued_tick=0,
        goals=(),
        budget=ComputeBudget(),
        facts=(),
    )
    return executive_arrays(empty, (), batch)


class GoalPayload(NamedTuple):
    """The goal the JEV commits to, as batched arrays (``ManagedGoal.payload``).

    Shapes, with B lanes and P = PARAM_WIDTH:
        skill_index: int32[B]. Index into the dispatcher's skill catalog.
        params: float32[B, P], zero-padded goal parameters.
        executive_version: int32[B]. The ExecutiveState.version this decision served.
        goal_slot: int32[B]. The ExecutiveArrays slot the goal came from, or -1
            when no executive goal is involved (for example the idle goal).
    """

    skill_index: jax.Array
    params: jax.Array
    executive_version: jax.Array
    goal_slot: jax.Array


def make_goal(
    skill_index: jax.Array,
    params: jax.Array,
    executive_version: jax.Array,
    goal_slot: jax.Array,
    *,
    minimum_commitment_seconds: jax.Array,
    maximum_duration_seconds: jax.Array,
    emergency_interrupt_mask: jax.Array,
) -> ManagedGoal:
    """Build the ADK ``ManagedGoal`` that a JEV returns.

    Every argument is batched. The batch size is taken from ``skill_index``, and
    the other arguments are checked against it. Values are not inspected, so the
    function stays traceable under ``jax.jit``.

    Args:
        skill_index: int32[B], catalog index of the chosen skill.
        params: float32[B, PARAM_WIDTH], goal parameters.
        executive_version: int32[B], executive version served by this decision.
        goal_slot: int32[B], the executive slot used, or -1 for none.
        minimum_commitment_seconds: float32[B], time before an ordinary
            request may replace the goal.
        maximum_duration_seconds: float32[B], time after which the goal is
            forced to end. ``jnp.inf`` means no limit.
        emergency_interrupt_mask: bool[B, len(DecisionEvent)], events that may
            bypass the minimum commitment.

    Returns:
        ManagedGoal with ``payload`` a :class:`GoalPayload` and ``control`` an
        ADK ``GoalControl``.

    Raises:
        ValueError: If any argument does not have the documented shape.
    """

    skill = jnp.asarray(skill_index, dtype=jnp.int32)
    if skill.ndim != 1:
        raise ValueError(f"skill_index must have shape (B,), got {skill.shape}")
    lanes = skill.shape[0]
    param_values = jnp.asarray(params, dtype=jnp.float32)
    version = jnp.asarray(executive_version, dtype=jnp.int32)
    slot = jnp.asarray(goal_slot, dtype=jnp.int32)
    minimum = jnp.asarray(minimum_commitment_seconds, dtype=jnp.float32)
    maximum = jnp.asarray(maximum_duration_seconds, dtype=jnp.float32)
    mask = jnp.asarray(emergency_interrupt_mask, dtype=jnp.bool_)
    _require_shape(param_values, (lanes, PARAM_WIDTH), "params")
    _require_shape(version, (lanes,), "executive_version")
    _require_shape(slot, (lanes,), "goal_slot")
    _require_shape(minimum, (lanes,), "minimum_commitment_seconds")
    _require_shape(maximum, (lanes,), "maximum_duration_seconds")
    _require_shape(mask, (lanes, len(DecisionEvent)), "emergency_interrupt_mask")
    return ManagedGoal(
        payload=GoalPayload(
            skill_index=skill,
            params=param_values,
            executive_version=version,
            goal_slot=slot,
        ),
        control=GoalControl(
            minimum_commitment_seconds=minimum,
            maximum_duration_seconds=maximum,
            emergency_interrupt_mask=mask,
        ),
    )


def idle_goal(batch: int) -> ManagedGoal:
    """Return the no-op goal: skill index 0, no commitment, unlimited duration.

    By convention skill index 0 is ``SKILL_IDLE``, so the catalog passed to
    :func:`executive_arrays` must list ``"idle"`` first. This function cannot
    check that convention, because it receives no catalog.

    Args:
        batch: Number of lanes B, a positive int.

    Returns:
        ManagedGoal with ``skill_index`` int32[B] = 0, ``params`` float32[B, 4]
        = 0, ``executive_version`` int32[B] = 0, ``goal_slot`` int32[B] = -1,
        ``minimum_commitment_seconds`` float32[B] = 0,
        ``maximum_duration_seconds`` float32[B] = inf and
        ``emergency_interrupt_mask`` bool[B, 14] = False.

    Raises:
        TypeError: If ``batch`` is not an integer.
        ValueError: If ``batch`` is not positive.
    """

    lanes = _positive_batch(batch)
    return make_goal(
        jnp.zeros((lanes,), dtype=jnp.int32),
        jnp.zeros((lanes, PARAM_WIDTH), dtype=jnp.float32),
        jnp.zeros((lanes,), dtype=jnp.int32),
        jnp.full((lanes,), -1, dtype=jnp.int32),
        minimum_commitment_seconds=jnp.zeros((lanes,), dtype=jnp.float32),
        maximum_duration_seconds=jnp.full((lanes,), jnp.inf, dtype=jnp.float32),
        emergency_interrupt_mask=jnp.zeros(
            (lanes, len(DecisionEvent)), dtype=jnp.bool_
        ),
    )


def event_mask(*events: DecisionEvent, batch: int) -> jax.Array:
    """Return a bool mask over the ``DecisionEvent`` axis with ``events`` set.

    This is the layout that ``GoalControl.emergency_interrupt_mask`` expects.

    Args:
        *events: ``DecisionEvent`` members to include. With no arguments the
            mask is all False.
        batch: Number of lanes B, a positive int.

    Returns:
        bool[batch, len(DecisionEvent)]. Column ``int(event)`` is True in every
        lane for each given event, and every other column is False.

    Raises:
        TypeError: If an argument is not a ``DecisionEvent`` member. A bare
            integer is rejected.
        ValueError: If ``batch`` is not positive.
    """

    lanes = _positive_batch(batch)
    row = np.zeros((len(DecisionEvent),), dtype=np.bool_)
    for event in events:
        if not isinstance(event, DecisionEvent):
            raise TypeError(
                f"events must be DecisionEvent members, got {type(event).__name__}"
            )
        row[int(event)] = True
    return jnp.asarray(np.tile(row, (lanes, 1)))


class SkillOutcome(IntEnum):
    """Terminal or current status of one skill activation, reported upward."""

    RUNNING = 0
    COMPLETED = 1
    INTERRUPTED = 2
    FAILED = 3
    TIMEOUT = 4


# Goal control that a SkillContract declares when it sets none.
_DEFAULT_MINIMUM_COMMITMENT_SECONDS: float = 0.5
_DEFAULT_MAXIMUM_DURATION_SECONDS: float = 10.0
_DEFAULT_EMERGENCY_EVENTS: tuple[str, ...] = ("DAMAGE_RECEIVED", "TARGET_DIED", "NPC_STUCK")


@dataclass(frozen=True)
class SkillContract:
    """Frozen contract of one Level 1 skill (ADR-001 D1 and D6).

    The skill owns the control of its goals: the last three attributes. A JEV
    instance looks them up by skill index (:func:`control_table`,
    :func:`control_for`) and does not choose them (SYSTEM_ONE.md section 8.12,
    item 1).

    Attributes:
        skill_id: Non-empty catalog id, for example ``"pursuit"``.
        version: Non-empty version label, for example ``"v0001"``. A changed
            skill needs a new version, because a controller experiment pins
            skill versions (ADR-001 D1).
        observation_groups: Non-empty names of the observation groups consumed.
        action_scope_sha256: Non-empty digest of the action scope the skill may
            emit.
        param_names: Names of the goal parameters, at most ``PARAM_WIDTH``.
        termination: Non-empty termination condition labels.
        success_signals: Non-empty labels that mean the skill succeeded.
        failure_signals: Non-empty labels that mean the skill failed.
        checkpoint_sha256: Digest of the pinned checkpoint, or None when no
            checkpoint is pinned.
        state_reset: Must be ``"on_entry"``. Skill-local state resets when the
            skill is entered (ADR-001 D6).
        legal_observation_only: Must be True. A deployed skill reads only legal
            observation products.
        minimum_commitment_seconds: Time a goal for this skill is held before an
            ordinary request may replace it, finite and >= 0. Default 0.5.
        maximum_duration_seconds: Time after which a goal for this skill is
            forced to end, positive; ``math.inf`` means no limit. Default 10.0.
        emergency_events: ``EVENT_NAMES`` entries that may interrupt a goal for
            this skill before its minimum commitment, each at most once; ``()``
            means none. Default ``("DAMAGE_RECEIVED", "TARGET_DIED", "NPC_STUCK")``.

    Raises:
        TypeError: If a field has the wrong Python type.
        ValueError: If a value is empty, too long or out of range, an emergency
            event is unknown or repeated, or ``state_reset`` or
            ``legal_observation_only`` breaks its rule.
    """

    skill_id: str
    version: str
    observation_groups: tuple[str, ...]
    action_scope_sha256: str
    param_names: tuple[str, ...]
    termination: tuple[str, ...]
    success_signals: tuple[str, ...]
    failure_signals: tuple[str, ...]
    checkpoint_sha256: str | None
    state_reset: str = "on_entry"
    legal_observation_only: bool = True
    minimum_commitment_seconds: float = _DEFAULT_MINIMUM_COMMITMENT_SECONDS
    maximum_duration_seconds: float = _DEFAULT_MAXIMUM_DURATION_SECONDS
    emergency_events: tuple[str, ...] = _DEFAULT_EMERGENCY_EVENTS

    def __post_init__(self) -> None:
        """Validate the contract and normalize sequences to tuples."""

        _text(self.skill_id, "SkillContract.skill_id")
        _text(self.version, "SkillContract.version")
        object.__setattr__(
            self,
            "observation_groups",
            _text_tuple(self.observation_groups, "SkillContract.observation_groups"),
        )
        _text(self.action_scope_sha256, "SkillContract.action_scope_sha256")
        param_names = _text_tuple(self.param_names, "SkillContract.param_names")
        if len(param_names) > PARAM_WIDTH:
            raise ValueError(
                f"SkillContract.param_names has {len(param_names)} names; "
                f"the limit is {PARAM_WIDTH}"
            )
        object.__setattr__(self, "param_names", param_names)
        object.__setattr__(
            self,
            "termination",
            _text_tuple(self.termination, "SkillContract.termination"),
        )
        object.__setattr__(
            self,
            "success_signals",
            _text_tuple(self.success_signals, "SkillContract.success_signals"),
        )
        object.__setattr__(
            self,
            "failure_signals",
            _text_tuple(self.failure_signals, "SkillContract.failure_signals"),
        )
        if self.checkpoint_sha256 is not None:
            _text(self.checkpoint_sha256, "SkillContract.checkpoint_sha256")
        if self.state_reset != "on_entry":
            raise ValueError(
                "SkillContract.state_reset must be 'on_entry', "
                f"got {self.state_reset!r} (ADR-001 D6)"
            )
        if self.legal_observation_only is not True:
            raise ValueError(
                "SkillContract.legal_observation_only must be True: deployed skills "
                "read legal observation products only (ADR-001 section 2)"
            )
        minimum = _finite_number(
            self.minimum_commitment_seconds, "SkillContract.minimum_commitment_seconds"
        )
        if minimum < 0.0:
            raise ValueError(
                f"SkillContract.minimum_commitment_seconds must be >= 0, got {minimum}"
            )
        object.__setattr__(self, "minimum_commitment_seconds", minimum)
        maximum = self.maximum_duration_seconds
        if isinstance(maximum, (float, np.floating)) and maximum == math.inf:
            maximum = math.inf
        else:
            maximum = _finite_number(maximum, "SkillContract.maximum_duration_seconds")
            if maximum <= 0.0:
                raise ValueError(
                    f"SkillContract.maximum_duration_seconds must be positive, got {maximum}"
                )
        object.__setattr__(self, "maximum_duration_seconds", maximum)
        events = _text_tuple(self.emergency_events, "SkillContract.emergency_events")
        unknown = [name for name in events if name not in EVENT_NAMES]
        if unknown:
            raise ValueError(
                f"SkillContract.emergency_events has unknown names {unknown}; "
                "expected names from EVENT_NAMES"
            )
        if len(set(events)) != len(events):
            raise ValueError(f"SkillContract.emergency_events repeats a name: {events}")
        object.__setattr__(self, "emergency_events", events)


# The control that the known skills' contracts declare (skill_library/idle.py,
# pursuit.py, evader.py), for callers that hold only skill ids. contracts.py
# imports no sibling module, so this mirrors those declarations, and
# test_na_skills.py pins the mirror to the real contracts.
_DECLARED_CONTROL: dict[str, tuple[float, float, tuple[str, ...]]] = {
    SKILL_IDLE: (0.0, math.inf, ()),
    "pursuit": (0.5, 10.0, _DEFAULT_EMERGENCY_EVENTS),
    "evader": (2.0, 10.0, _DEFAULT_EMERGENCY_EVENTS),
}


class SkillControlTable(NamedTuple):
    """The goal control that each catalogue skill declares, by skill index.

    Build it with :func:`control_table` from the contracts, or with
    :func:`default_control_table` when only the ids are known. A JEV instance
    holds one and gathers rows with :func:`control_for`. The leading axis is
    the catalogue axis K, not the lane axis B, so the table itself never
    crosses the hierarchy core.

    Shapes, with K skills in catalogue order:
        minimum_commitment_seconds: float32[K].
        maximum_duration_seconds: float32[K]. ``inf`` means no limit.
        emergency_interrupt_mask: bool[K, len(EVENT_NAMES)], in the layout of
            :func:`event_mask`.
    """

    minimum_commitment_seconds: jax.Array
    maximum_duration_seconds: jax.Array
    emergency_interrupt_mask: jax.Array


def _control_rows(
    rows: Sequence[tuple[float, float, tuple[str, ...]]],
) -> SkillControlTable:
    """Return the table whose row k is ``rows[k] = (minimum, maximum, event names)``."""

    mask = np.zeros((len(rows), len(DecisionEvent)), dtype=np.bool_)
    for index, (_, _, events) in enumerate(rows):
        for name in events:
            mask[index, int(DecisionEvent[name])] = True
    return SkillControlTable(
        minimum_commitment_seconds=jnp.asarray([row[0] for row in rows], dtype=jnp.float32),
        maximum_duration_seconds=jnp.asarray([row[1] for row in rows], dtype=jnp.float32),
        emergency_interrupt_mask=jnp.asarray(mask),
    )


def control_table(contracts: Sequence[SkillContract]) -> SkillControlTable:
    """Return the control table of a catalogue: row k is ``contracts[k]``'s control.

    Args:
        contracts: The catalogue's SkillContract values in catalogue order, at
            least one (``SkillRegistry.contracts``).

    Returns:
        A SkillControlTable with K = ``len(contracts)`` rows.

    Raises:
        TypeError: If ``contracts`` is not a sequence of SkillContract values.
        ValueError: If ``contracts`` is empty.
    """

    items = tuple(contracts)
    if not items:
        raise ValueError("contracts must hold at least one SkillContract")
    for position, contract in enumerate(items):
        if not isinstance(contract, SkillContract):
            raise TypeError(
                f"contracts[{position}] must be a SkillContract, got {type(contract).__name__}"
            )
    return _control_rows(
        [
            (
                contract.minimum_commitment_seconds,
                contract.maximum_duration_seconds,
                contract.emergency_events,
            )
            for contract in items
        ]
    )


def default_control_table(skill_ids: Sequence[str]) -> SkillControlTable:
    """Return the control table of a catalogue that is known only by its ids.

    The ids ``"idle"``, ``"pursuit"`` and ``"evader"`` get the control their
    contracts declare in ``skill_library``. Any other id gets the
    :class:`SkillContract` defaults. With a registry at hand, prefer
    ``SkillRegistry.control_table()``.

    Args:
        skill_ids: Catalogue ids in catalogue order, at least one.

    Returns:
        A SkillControlTable with K = ``len(skill_ids)`` rows.

    Raises:
        TypeError: If ``skill_ids`` is one string or holds a non-string.
        ValueError: If ``skill_ids`` is empty or holds an empty id.
    """

    ids = _text_tuple(skill_ids, "skill_ids")
    if not ids:
        raise ValueError("skill_ids must name at least one skill")
    default = (
        _DEFAULT_MINIMUM_COMMITMENT_SECONDS,
        _DEFAULT_MAXIMUM_DURATION_SECONDS,
        _DEFAULT_EMERGENCY_EVENTS,
    )
    return _control_rows([_DECLARED_CONTROL.get(skill_id, default) for skill_id in ids])


def control_for(
    table: SkillControlTable, skill_index: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gather the declared control of each lane's skill. Traceable under ``jax.jit``.

    Args:
        table: The catalogue's SkillControlTable.
        skill_index: int32[B], the catalogue index of each lane's skill. Values
            are not inspected, as in :func:`make_goal`: an index outside
            ``[0, K)`` is clipped into range.

    Returns:
        ``(minimum_commitment_seconds, maximum_duration_seconds,
        emergency_interrupt_mask)``: float32[B], float32[B] and
        bool[B, len(EVENT_NAMES)], in ``GoalControl`` order, ready for the
        keyword arguments of :func:`make_goal`.

    Raises:
        TypeError: If ``table`` is not a SkillControlTable.
        ValueError: If ``skill_index`` does not have shape (B,).
    """

    if not isinstance(table, SkillControlTable):
        raise TypeError(f"table must be a SkillControlTable, got {type(table).__name__}")
    index = jnp.asarray(skill_index, dtype=jnp.int32)
    if index.ndim != 1:
        raise ValueError(f"skill_index must have shape (B,), got {index.shape}")
    return (
        jnp.take(table.minimum_commitment_seconds, index, axis=0, mode="clip"),
        jnp.take(table.maximum_duration_seconds, index, axis=0, mode="clip"),
        jnp.take(table.emergency_interrupt_mask, index, axis=0, mode="clip"),
    )


def _require_control_table(table: Any, skills: int) -> SkillControlTable:
    """Return ``table`` after checking that it is a SkillControlTable with ``skills`` rows."""

    if not isinstance(table, SkillControlTable):
        raise TypeError(f"control must be a SkillControlTable, got {type(table).__name__}")
    _require_shape(
        jnp.asarray(table.minimum_commitment_seconds),
        (skills,),
        "control.minimum_commitment_seconds",
    )
    _require_shape(
        jnp.asarray(table.maximum_duration_seconds),
        (skills,),
        "control.maximum_duration_seconds",
    )
    _require_shape(
        jnp.asarray(table.emergency_interrupt_mask),
        (skills, len(DecisionEvent)),
        "control.emergency_interrupt_mask",
    )
    return table


@runtime_checkable
class Skill(Protocol):
    """Level 1 skill policy: factored legal actions under one contract.

    A skill is goal-conditioned through ``params``. It reads only the legal
    ``ActorPolicyInput``. Its factors are already restricted to the skill scope
    and to the runtime ``action_mask``, so the dispatcher can execute them
    without masking again.

    Attributes:
        contract: The frozen :class:`SkillContract` this policy implements.
    """

    contract: SkillContract

    def initial_state(self, batch: int) -> jax.Array:
        """Return the reset skill-local state.

        The dispatcher calls this on every skill entry, because skill-local
        state resets on entry (ADR-001 D6).

        Args:
            batch: Number of lanes B, a positive int.

        Returns:
            float32[batch, SKILL_STATE_WIDTH].
        """

    def __call__(
        self,
        state: jax.Array,
        belief: SharedBeliefState,
        actor_input: ActorPolicyInput,
        params: jax.Array,
        key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Advance the skill by one decision step.

        Args:
            state: float32[B, SKILL_STATE_WIDTH], skill-local state.
            belief: The shared belief B_t, with ``features`` float32[B, 10],
                ``feature_known`` bool[B, 10] and ``latent`` float32[B, 64].
            actor_input: The legal ``ActorPolicyInput``. Privileged input is
                never accepted.
            params: float32[B, PARAM_WIDTH], goal parameters.
            key: A JAX PRNG key for sampling.

        Returns:
            ``(next_state, factors)``, where ``next_state`` is
            float32[B, SKILL_STATE_WIDTH] and ``factors`` is
            int32[B, ACTION_HEADS], restricted to the skill scope and to
            ``actor_input.action_mask``.
        """


class GoalReport(NamedTuple):
    """Host-side report that a skill or the dispatcher sends up on exit.

    This is the outcome report on exit (ADR-001 section 5, seam 1). Its values
    are host integers and strings, not arrays.

    Attributes:
        tick: Simulation tick at which the report was made.
        lane: Lane index in the batch.
        skill_id: Id of the skill that was active.
        executive_version: ``ExecutiveState.version`` that the goal came from.
        outcome: The :class:`SkillOutcome`.
        duration_ticks: Ticks the activation lasted.
        detail: Short free-text reason.
    """

    tick: int
    lane: int
    skill_id: str
    executive_version: int
    outcome: SkillOutcome
    duration_ticks: int
    detail: str


class JevMode(str, Enum):
    """Where a JEV instance runs.

    ``IN_GRAPH`` runs inside the jitted hierarchy step. ``HOST`` runs between
    steps, for example a language-model call. Both are bound by the same
    latency budget. Read ``.value`` for the string; ``str(JevMode.HOST)`` is
    ``'JevMode.HOST'`` on Python 3.12.
    """

    IN_GRAPH = "in_graph"
    HOST = "host"


@dataclass(frozen=True)
class JevBudget:
    """Declared per-decision budget of one JEV instance (ADR-001 D2).

    Attributes:
        latency_ms: Latency budget per decision, finite and positive. It is set
            per deployment target.
        compute_units: Compute units per decision, a non-negative int. The unit
            is an open question in ADR-001 section 12.

    Raises:
        TypeError: If a field has the wrong Python type.
        ValueError: If ``latency_ms`` is non-positive or non-finite, or
            ``compute_units`` is negative.
    """

    latency_ms: float
    compute_units: int = 0

    def __post_init__(self) -> None:
        """Validate the budget and normalize numeric fields."""

        latency = _finite_number(self.latency_ms, "JevBudget.latency_ms")
        if latency <= 0.0:
            raise ValueError(f"JevBudget.latency_ms must be positive, got {latency}")
        object.__setattr__(self, "latency_ms", latency)
        object.__setattr__(
            self,
            "compute_units",
            _non_negative_int(self.compute_units, "JevBudget.compute_units"),
        )


@runtime_checkable
class JevInstance(Protocol):
    """Level 2 JEV decision model: one ManagedGoal per decision within a budget.

    Every instance implements this protocol, so instances can be swapped without
    changing orchestration (ADR-001 D2). Examples are J0 (scripted rules), J1
    (learned reactive network), J-LM (quick language model) and the later
    world-model instances. An instance that misses its budget leaves the
    previous goal in force, and the miss is counted.

    Attributes:
        name: Instance id used in reports, for example ``"J1"``.
        mode: A :class:`JevMode`.
    """

    name: str
    mode: JevMode

    def initial_state(self, batch: int) -> Any:
        """Return the reset instance state for ``batch`` lanes, a batched PyTree.

        Args:
            batch: Number of lanes B, a positive int.

        Returns:
            Any batched PyTree with a leading lane axis B.
        """

    def __call__(
        self,
        state: Any,
        belief: SharedBeliefState,
        current_goal: GoalPayload,
        actor_input: ActorPolicyInput,
        key: jax.Array,
        *,
        decision_context: DecisionContext,
        goal_runtime: GoalRuntimeContext,
        executive: ExecutiveArrays,
    ) -> GoalDecision:
        """Choose the goal for this decision.

        Args:
            state: The instance state from the previous decision, or
                ``initial_state``.
            belief: The shared belief B_t.
            current_goal: The goal currently in force.
            actor_input: The legal ``ActorPolicyInput`` only.
            key: A JAX PRNG key.
            decision_context: The normalized ``DecisionContext`` (events, timing).
            goal_runtime: The ADK commitment and interrupt view.
            executive: The executive state E_t, lowered by
                :func:`executive_arrays`.

        Returns:
            An ADK ``GoalDecision`` whose ``goal`` is a ``ManagedGoal`` with a
            :class:`GoalPayload`. ``terminate`` False means CONTINUE the current
            goal.
        """


class JevStats(NamedTuple):
    """Aggregate statistics of one JEV instance over a run.

    Attributes:
        decisions: Number of decisions made.
        misses: Number of decisions that exceeded the latency budget.
        latency_ms_p50: Median decision latency in milliseconds.
        latency_ms_p95: 95th-percentile decision latency in milliseconds.
        compute_units: Total compute units spent.
    """

    decisions: int
    misses: int
    latency_ms_p50: float
    latency_ms_p95: float
    compute_units: int


__all__ = [
    "ACTION_HEADS",
    "BELIEF_FEATURE_COUNT",
    "BELIEF_FEATURE_NAMES",
    "BELIEF_LATENT_WIDTH",
    "ComputeBudget",
    "EVENT_NAMES",
    "EXECUTIVE_STATE_JSON_SCHEMA",
    "ExecutiveArrays",
    "ExecutiveState",
    "GoalPayload",
    "GoalReport",
    "GoalSpec",
    "Condition",
    "CONDITION_KINDS",
    "CONDITION_ARGUMENT_UNITS",
    "JevBudget",
    "JevInstance",
    "JevMode",
    "JevStats",
    "MAX_GOALS",
    "PARAM_WIDTH",
    "SKILL_IDLE",
    "SKILL_STATE_WIDTH",
    "SharedBeliefState",
    "Skill",
    "SkillContract",
    "SkillControlTable",
    "SkillOutcome",
    "TICKS_SATURATION",
    "control_for",
    "control_table",
    "default_control_table",
    "empty_executive_arrays",
    "event_mask",
    "executive_arrays",
    "feature_index",
    "idle_goal",
    "make_goal",
]
