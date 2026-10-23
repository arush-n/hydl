"""The shared scratchpad S_t: one versioned document between the LLM, the JEV and the skills.

``docs/SYSTEM_ONE.md`` section 8.10, Consequence 3. The owner (2026-10-17): "think of the state as
the scratchpad between all 3 layers where all models pass their information in between (like a
shared memory) there are different parts for each component ... we have to find an optimal way to
structure it such that each model gets the information it needs and has the proper context to make
better decisions". This module composes the typed records of ``contracts.py`` (``ExecutiveState``,
``GoalSpec``, ``ComputeBudget``, ``SharedBeliefState``, ``GoalReport``, ``EVENT_NAMES``) into that
document, with its JSON Schema and one view per reader. It is not wired into the executive harness
or the session yet; until it is, those keep using ``ExecutiveStateStore`` and ``ReportChannel``.

Sections. Each has exactly one writer. ``SharedState.write`` refuses any other writer with
``PermissionError``, and every accepted write bumps the document version by one::

    section     writer      content
    executive   "llm"       goals as JSON: GoalSpec's typed fields plus ``intent`` and ``when``
                            (and an optional ``note``) in the LLM's words; facts; budget; ``notes``,
                            the LLM's working notes ("the idea it has")
    belief      "belief"    per lane: the ten features keyed by BELIEF_FEATURE_NAMES, their known
                            flags, and a latent summary (L2 norm and the first 8 values)
    decision    "jev"       per lane: the goal in force (skill, params, slot, E_t version) with
                            confidence and entropy when known; the last N decisions (N = 16)
    requests    "jev"       the live ``requests.ActionRequest`` records per agent (lane), sorted by
                            request id, at most REQUEST_LIMIT (16) per agent (section 9.10)
    skills      "skills"    per lane: the active skill; runtime summaries (free-form JSON from the
                            dispatcher); the outcomes as GoalReports (bounded, newest last); the
                            status stream, ``requests.RequestStatus`` records (bounded, newest last)
    events      "runtime"   per lane: the DecisionEvent names active in the last window

The decision layer writes two sections and the dispatcher (the lower layer's combiner) writes the
status stream inside ``skills``. Each section still has exactly one writer.

Who reads what, and why. Each model gets the information it needs and nothing more:

* The LLM reads text. ``view_for_llm`` renders its own last goals, facts, budget and notes, the
  belief per lane, the decision in force with its confidence, the live request count, the status
  counts of the stream (accepted, rejected, ...) with the last LLM_REJECTIONS (5) rejections and
  their reasons, the skill outcomes it has not seen yet, and the recent events. The LLM thinks
  slowly and in words, so it gets a compact rendering: no array is dumped, and the latent appears
  only as its norm.
* The JEV reads typed arrays plus per-goal text. ``view_for_jev`` gives E_t lowered exactly as
  ``contracts.executive_arrays`` lowers it, plus the ``intent`` and ``when`` of each goal aligned
  with those slots (the words its text path encodes per goal, section 8.10 Consequence 1). It also
  gives the belief arrays, the current skill per lane, and per lane the live request ids with the
  latest status of each (the input of its correspondence rules, section 9.10). The JEV decides at
  about 2 Hz under a hard budget, so it gets arrays for a compiled pass and only the words it must
  interpret. A goal's ``note`` is the LLM's own remark and stays out of this view.
* The skills read the goal only. ``view_for_skills`` gives the goal payload fields per lane (skill,
  params, E_t version, slot). A skill reads B_t itself from the hierarchy step and never sees words.

Formulas
--------
::

    latent_norm_b   = sqrt(sum_j z_bj^2)       over the BELIEF_LATENT_WIDTH latent values of lane b
    latent_prefix_b = (z_b0, ..., z_b7)
    bounded list    = the newest ``limit`` entries; the oldest are dropped first (as in ReportChannel)
    live requests   = per agent, the ``limit`` highest request ids (ids are monotonic per agent)
    latest status   : of request (a, i), the last entry of the status stream with agent a and id i
    unseen outcome  : report.tick > executive.issued_tick, or every report while executive.version == 0
    write           : version' = version + 1 and tick' = the tick of the write, under one lock

Bounds. REQUEST_LIMIT = 16 = L x M, a whole behaviour program per agent: L = 4 steps (section 9.9)
times M = 4 concurrent granted requests (section 9.10). STATUS_HISTORY = 256 covers one cadence
interval (11 ticks) of 8 agents x 2 live requests at one status per request per tick (176
entries), so no status falls out of the stream before the middle layer's next read at that load.
How often the lower layer emits a status (every tick, or on change) is the lower layer's choice.

Citations: ``docs/SYSTEM_ONE.md`` section 8.10 (the section table and the rules) and section 9.10
(the request stream); ADR-001 D3 (E_t is versioned) and section 5, seam 3 (outcomes reported
upward). The schema is JSON Schema draft
2020-12 (https://json-schema.org/draft/2020-12/schema). It is checked with the manual validator of
``executive/store.py`` (``validate_against_schema``), so it uses only the keywords that validator
enforces. Semantic rules the schema cannot state (event names, outcome names, the fixed writers)
are checked when a section is built.

Layout (2026-10-18, split from one module without behaviour change): ``_common`` (constants, validation
helpers), ``schema`` (the JSON Schema and ``validate``), ``sections/`` (one module per section: executive,
belief, decision, requests, skills, events), ``document`` (``SharedState``), ``views/`` (``llm``, ``jev``,
``skills``: one view per reader) and ``writers`` (the session's scribe, also reachable as
``experimental.neural_agent.shared_state_writers``). Every public name is re-exported here, so
``from experimental.neural_agent.shared_state import SharedState`` is unchanged.
"""

from __future__ import annotations

from ._common import (
    DECISION_HISTORY,
    EVENT_WINDOW_TICKS,
    INFORMATION_HISTORY,
    LATENT_PREFIX_WIDTH,
    LLM_REJECTIONS,
    OUTCOME_HISTORY,
    REQUEST_LIMIT,
    SECTION_NAMES,
    SECTION_WRITERS,
    KNOWN_ENTITY_LIMIT,
    STATUS_HISTORY,
    WRITES_PER_AGENT_TICK,
)
from .document import (
    SharedState,
)
from .schema import (
    EXECUTIVE_SECTION_JSON_SCHEMA,
    SHARED_STATE_JSON_SCHEMA,
    validate,
)
from .sections.belief import (
    BeliefLane,
    BeliefSection,
)
from .sections.decision import (
    GoalReading,
    DecisionRecord,
    DecisionSection,
    GoalInForce,
    InformationRequest,
)
from .sections.events import (
    EventsSection,
)
from .sections.executive import (
    ExecutiveSection,
    GoalEntry,
    entry_to_spec,
    spec_to_entry,
)
from .sections.requests import (
    RequestsSection,
)
from .sections.writes import (
    WriteRecord,
    WriteStatus,
    WritesSection,
)
from .sections.skills import (
    AgentEntities,
    KnownEntity,
    SkillsSection,
)
from .views.jev import (
    JevView,
    view_for_jev,
)
from .views.briefing import (
    briefing,
)
from .views.render import (
    ViewOptions,
    estimate_tokens,
    render,
)
from .views.llm import (
    view_for_llm,
)
from .views.skills import (
    SkillsView,
    view_for_skills,
)

__all__ = [
    "DECISION_HISTORY",
    "EVENT_WINDOW_TICKS",
    "LATENT_PREFIX_WIDTH",
    "LLM_REJECTIONS",
    "OUTCOME_HISTORY",
    "REQUEST_LIMIT",
    "SECTION_NAMES",
    "SECTION_WRITERS",
    "SHARED_STATE_JSON_SCHEMA",
    "EXECUTIVE_SECTION_JSON_SCHEMA",
    "STATUS_HISTORY",
    "KNOWN_ENTITY_LIMIT",
    "AgentEntities",
    "KnownEntity",
    "WRITES_PER_AGENT_TICK",
    "WriteRecord",
    "WriteStatus",
    "WritesSection",
    "BeliefLane",
    "BeliefSection",
    "DecisionRecord",
    "DecisionSection",
    "INFORMATION_HISTORY",
    "InformationRequest",
    "EventsSection",
    "ExecutiveSection",
    "GoalEntry",
    "GoalInForce",
    "JevView",
    "RequestsSection",
    "SharedState",
    "SkillsSection",
    "SkillsView",
    "entry_to_spec",
    "spec_to_entry",
    "validate",
    "view_for_jev",
    "view_for_llm",
    "GoalReading",
    "ViewOptions",
    "briefing",
    "estimate_tokens",
    "render",
    "view_for_skills",
]
