"""AnthropicExecutiveBackend: Level 3 as a Claude model behind the seam (ADR-001 D3).

Each decision sends one Messages request. The stable part comes first, so the prompt
cache can serve it. The volatile part comes after the cache breakpoint::

    model          claude-opus-5-5 by default (shared/models.md lists it as Active)
    max_tokens     4096
    system         one text block: role, skill catalog, reply rules, JSON Schema (sorted
                   keys), with cache_control {"type": "ephemeral"} (the 5-minute TTL)
    messages       one user turn: tick, required_version, the current E_t JSON, the reports
    output_config  {"effort": effort, "format": {"type": "json_schema",
                   "schema": wire_schema(EXECUTIVE_SECTION_JSON_SCHEMA)}}

No ``thinking`` field is sent. On claude-opus-5-5 thinking is adaptive and always on, and
``output_config.effort`` sets its depth (python README, "Extended Thinking").

Formulas
--------
Usage comes from the response's usage object (source: the Messages API usage fields,
anthropic 1.15.0 types/usage.py, and
https://platform.claude.com/docs/en/build-with-claude/prompt-caching.md)::

    input_tokens          = usage.input_tokens                       (uncached)
    cache_read_tokens     = usage.cache_read_input_tokens or 0
    cache_creation_tokens = usage.cache_creation_input_tokens or 0

An absent cache field counts as zero. Caching engages only when the stable prefix is at
least the model's minimum cacheable length. For claude-opus-5-5 the minimum is 512 tokens
(https://platform.claude.com/docs/en/build-with-claude/prompt-caching.md, "Cache limitations",
re-read live on 2026-10-16). A shorter prefix caches nothing and
reports no error, so ``cache_read_fraction`` stays 0. The smoke script prints the usage so
this can be checked on a real call.

Citations (the URLs are listed under References in ``__init__.py``): the Anthropic docs for
structured outputs (``output_config.format``, and the subset that ``wire_schema`` keeps),
prompt caching (512-token minimum for claude-opus-5-5, ``cache_control`` on a system text
block), effort (``output_config.effort``, default ``medium`` on claude-opus-5-5), adaptive
thinking, and the Messages API usage fields. A local copy of those docs is the claude-api
skill bundle (python/claude-api/README.md, tool-use.md, shared/prompt-caching.md,
shared/models.md). SDK anthropic 1.15.0: types/output_config_param.py,
types/json_output_format_param.py, types/usage.py.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import anthropic

from experimental.neural_agent.contracts import (
    CONDITION_ARGUMENT_UNITS,
    CONDITION_KINDS,
    ExecutiveState,
    GoalReport,
    SkillOutcome,
)
from experimental.neural_agent.executive.baselines import CallUsage
from experimental.neural_agent.executive.reports import EventRecord
from experimental.neural_agent.executive.patterned import PATTERN_DIRECTIONS
from experimental.neural_agent.executive.store import validate_catalog
from experimental.neural_agent.shared_state import EXECUTIVE_SECTION_JSON_SCHEMA, ExecutiveSection, view_for_llm

DEFAULT_MODEL = "claude-opus-5-5"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
MAX_TOKENS = 4096


def build_system_prompt(skill_ids: Sequence[str]) -> str:
    """Return the stable system prompt for a skill catalog.

    The text depends only on the catalog and the schema. It holds no clock, random or
    per-call value, so two calls with the same catalog send the same bytes, and the
    prompt cache can serve them.

    Args:
        skill_ids: The skill catalog in dispatcher order, validated by ``validate_catalog``.

    Returns:
        The prompt as one string.

    Raises:
        TypeError: If the catalog is not a sequence of strings.
        ValueError: If the catalog is empty or repeats an id.
    """

    catalog = validate_catalog(skill_ids)
    skills_json = json.dumps(list(catalog), ensure_ascii=True)
    kinds_json = json.dumps(list(CONDITION_KINDS), ensure_ascii=True)
    units_json = json.dumps(dict(CONDITION_ARGUMENT_UNITS), ensure_ascii=True, sort_keys=True)
    schema_json = json.dumps(
        EXECUTIVE_SECTION_JSON_SCHEMA, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return (
        "You are the executive of a hierarchical game agent (ADR-001 D3). You do not act in the "
        "game. You maintain the executive state E_t: at most four goals, each naming a skill from "
        "the catalog with a priority, parameters, constraints and a planning horizon, plus a compute "
        "budget and the facts you judge relevant. A fast decision model reads E_t on every decision, "
        "so E_t must be complete and self-contained.\n"
        f"Legal skill ids in catalog order: {skills_json}. A goal's skill_id must be one of them.\n"
        "A goal's condition says when the goal applies: null when it always applies, otherwise an "
        f"object with kind one of {kinds_json} and argument its number. The kinds that take a number, "
        f"with the unit of that number, are {units_json} (fraction: 0 to 1 of full health; ticks: decision "
        "ticks); every other kind takes argument null. The decision model evaluates the typed condition "
        "exactly on its belief, so write it whenever a goal is conditional.\n"
        f"{PATTERN_DIRECTIONS}\n"
        "Each goal has an addressee: 'all' (every NPC), 'lane:<n>' (one NPC) or 'squad:<name>' for a squad you "
        "list in squads (name, lanes); a goal is in force only on the lanes it addresses. objectives are your "
        "long-term aims with stable ids (status open, done or dropped); a goal may name the objective it serves, "
        "and outcomes roll up per objective in your view.\n"
        "Each user turn gives tick, required_version, the current E_t, the reports that arrived since your "
        "last call and, when the notepad is shared, your briefing of it: every NPC's situation, the goals in "
        "force with how your words were read, the questions to you, the rejections and what happened since "
        "your version. A report is a goal outcome (COMPLETED, INTERRUPTED, FAILED or TIMEOUT) or an "
        "interrupt event.\n"
        "Reply with the whole new executive section as one JSON object: the typed fields, your words per "
        "goal (intent, when, note) and your notes. Set version to required_version and "
        "issued_tick to tick. Keep every goal that still applies, and change only what the reports "
        "show is wrong. If nothing changed, return the current goals unchanged with the new version "
        "and tick.\n"
        f"Reply format: a JSON object that matches this JSON Schema: {schema_json}"
    )


def _report_payload(item: GoalReport | EventRecord) -> dict[str, Any]:
    """Return one report as a JSON-shaped dict, with a ``kind`` tag."""

    if isinstance(item, GoalReport):
        return {
            "kind": "goal_report",
            "tick": item.tick,
            "lane": item.lane,
            "skill_id": item.skill_id,
            "executive_version": item.executive_version,
            "outcome": SkillOutcome(item.outcome).name,
            "duration_ticks": item.duration_ticks,
            "detail": item.detail,
        }
    if isinstance(item, EventRecord):
        return {"kind": "event", "tick": item.tick, "lane": item.lane, "event": item.event_name}
    raise TypeError(f"report must be a GoalReport or EventRecord, got {type(item).__name__}")


def build_user_turn(
    current: ExecutiveState,
    reports: Sequence[GoalReport | EventRecord],
    tick: int,
    briefing: str | None = None,
) -> str:
    """Return the volatile user turn as JSON with sorted keys.

    The turn holds ``tick``, ``required_version`` (current version plus one), the current
    E_t, the reports and, when given, ``briefing``: the LLM's view of the shared notepad
    (``shared_state.view_for_llm``). It is the only part of the request that changes per call.

    Raises:
        TypeError: If ``current`` is not an ExecutiveState, ``tick`` is not an integer, or a
            report is neither a GoalReport nor an EventRecord.
        ValueError: If ``tick`` is negative.
    """

    if not isinstance(current, ExecutiveState):
        raise TypeError("current must be an ExecutiveState")
    if isinstance(tick, bool) or not isinstance(tick, int):
        raise TypeError(f"tick must be an integer, got {type(tick).__name__}")
    if tick < 0:
        raise ValueError(f"tick must be non-negative, got {tick}")
    payload = {
        "tick": tick,
        "required_version": current.version + 1,
        "current": json.loads(current.to_json()),
        "reports": [_report_payload(item) for item in reports],
    }
    if briefing is not None:
        payload["briefing"] = briefing
    return json.dumps(payload, sort_keys=True, allow_nan=False)


# Keywords outside the structured-output subset. The live docs list the numeric and string
# constraints as unsupported, allow array minItems only as 0 or 1, and state that an
# unsupported feature returns HTTP 400. $schema is not in the subset either, so it goes too.
_WIRE_DROPPED_KEYWORDS: frozenset[str] = frozenset(
    {
        "$schema",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "multipleOf",
    }
)


def wire_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return the copy of ``schema`` that the request sends as ``output_config.format.schema``.

    Every dropped keyword (see ``_WIRE_DROPPED_KEYWORDS``) is removed at every level. ``type``
    lists such as ``["integer", "null"]``, ``title`` and ``description`` are kept. ``schema`` is
    not changed. The system prompt still carries the full schema, and the store and
    ``ExecutiveState.from_dict`` still enforce every dropped constraint on the reply.

    Args:
        schema: A JSON Schema object, such as ``EXECUTIVE_STATE_JSON_SCHEMA``.

    Returns:
        A new dict with the same structure, minus the dropped keywords.

    Raises:
        TypeError: If ``schema`` is not a mapping.
    """

    if not isinstance(schema, Mapping):
        raise TypeError(f"schema must be a JSON Schema object, got {type(schema).__name__}")
    return _wire_node(schema)


def _wire_node(node: Any) -> Any:
    """Return one schema node with the dropped keywords removed, recursively."""

    if isinstance(node, list):
        return [_wire_node(item) for item in node]
    if not isinstance(node, Mapping):
        return node
    kept: dict[str, Any] = {}
    for key, value in node.items():
        if key in _WIRE_DROPPED_KEYWORDS:
            continue
        if key in ("properties", "$defs") and isinstance(value, Mapping):
            # Property names are data, not keywords, so they are never dropped.
            kept[key] = {name: _wire_node(sub) for name, sub in value.items()}
        else:
            kept[key] = _wire_node(value)
    return kept


def _count(value: int | None) -> int:
    """Return an optional usage field as an int. An absent field counts as zero."""

    return 0 if value is None else int(value)


class AnthropicExecutiveBackend:
    """Level 3 backend that asks a Claude model for the next E_t.

    Args:
        model: Model id. Default ``claude-opus-5-5``.
        effort: One of ``EFFORT_LEVELS``. Default ``medium``, which is the default on claude-opus-5-5.
        client: An object with ``messages.create``. None builds ``anthropic.Anthropic()``. The
            constructor needs no key. A call reads ANTHROPIC_API_KEY, and the SDK raises then if it is unset.
        skill_ids: Keyword-only and required. The catalog that the system prompt lists. It
            should be the catalog the store holds.

    Raises:
        TypeError: If ``model`` is not a string, or ``skill_ids`` is not given.
        ValueError: If ``model`` is empty, ``effort`` is not in ``EFFORT_LEVELS``, or the catalog is invalid.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        client: Any = None,
        *,
        skill_ids: Sequence[str],
    ) -> None:
        if not isinstance(model, str):
            raise TypeError(f"model must be a string, got {type(model).__name__}")
        if not model:
            raise ValueError("model must be a non-empty string")
        if effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {EFFORT_LEVELS}, got {effort!r}")
        self.model = model
        self.effort = effort
        self._client = client if client is not None else anthropic.Anthropic()
        self._system = build_system_prompt(skill_ids)
        self._wire = wire_schema(EXECUTIVE_SECTION_JSON_SCHEMA)
        self.shared_state: Any = None  # bound by the harness when the notepad is shared: the briefing goes in the user turn
        self.last_section: ExecutiveSection | None = None  # the section the last accepted reply parsed to, words included

    @property
    def system_prompt(self) -> str:
        """The stable system prompt that every call sends."""

        return self._system

    def propose(
        self,
        current: ExecutiveState,
        reports: Sequence[GoalReport | EventRecord],
        tick: int,
    ) -> ExecutiveState:
        """Return the proposed E_t. The usage is dropped; use :meth:`propose_with_usage` to keep it."""

        state, _usage = self.propose_with_usage(current, reports, tick)
        return state

    def propose_with_usage(
        self,
        current: ExecutiveState,
        reports: Sequence[GoalReport | EventRecord],
        tick: int,
    ) -> tuple[ExecutiveState, CallUsage]:
        """Make one Messages call and return ``(state, usage)``.

        Args:
            current: The state in force.
            reports: The reports drained since the last call.
            tick: The simulation tick of this call.

        Returns:
            The parsed ExecutiveState (the typed part of the section the LLM replied with; the
            whole section, words included, is ``last_section``) and the CallUsage of the call.

        Raises:
            ValueError: If the reply stopped for any reason other than ``end_turn`` (for
                example ``max_tokens`` or ``refusal``), has no text block, or the JSON does
                not parse to an ExecutiveState. Fails closed: nothing is guessed.
            anthropic.APIError: From the SDK, for transport or API failures.
        """

        response = self._client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=[
                {"type": "text", "text": self._system, "cache_control": {"type": "ephemeral"}}
            ],
            messages=[
                {
                    "role": "user",
                    "content": build_user_turn(
                        current, reports, tick,
                        briefing=None if self.shared_state is None else view_for_llm(self.shared_state),
                    ),
                }
            ],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": self._wire},
            },
        )
        if response.stop_reason != "end_turn":
            raise ValueError(
                f"executive reply stopped with {response.stop_reason!r}; expected 'end_turn'"
            )
        text = next((block.text for block in response.content if block.type == "text"), None)
        if text is None:
            raise ValueError("executive reply has no text block")
        try:
            section = ExecutiveSection.from_dict(json.loads(text))
        except (TypeError, ValueError) as error:
            raise ValueError(f"executive reply is not a valid executive section: {error}") from error
        self.last_section = section
        state = section.to_executive_state()
        usage = response.usage
        return state, CallUsage(
            model=self.model,
            effort=self.effort,
            input_tokens=_count(usage.input_tokens),
            output_tokens=_count(usage.output_tokens),
            cache_read_tokens=_count(usage.cache_read_input_tokens),
            cache_creation_tokens=_count(usage.cache_creation_input_tokens),
        )
