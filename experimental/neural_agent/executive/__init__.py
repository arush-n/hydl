"""Level 3 executive seam: the E_t writer, the report channel, the scheduler and the API adapter (ADR-001 D3, P5).

This package is the seam that an executive plugs into. It is not the executive. It does not
decide what the agent should do, and it does not measure decision quality. It records what the
executive costs and when the executive ran, and it keeps the state in force for Levels 1 and 2
versioned and valid.

Thesis (ADR-001 section 1 and D3). The research question is how an agent spends time and
compute: when to think, when to act, and when to reuse an existing decision. The executive is an
API language model that edits E_t asynchronously. It is invoked by events, not on every action.
Every call logs four knobs: whether to call (the trigger), the model tier, the effort level, and
how much of the prompt is served from cache. Deterministic baselines run against the same
interface, so an event-driven policy can be compared with them on outcome per call (ADR-001 P5,
hypotheses H8 and H9).

Modules:

* ``store``: ExecutiveStateStore, a versioned E_t with schema and catalog checks, and to_arrays.
* ``reports``: ReportChannel, a bounded drop-oldest queue of GoalReport and EventRecord.
* ``scheduler``: VirtualClock, CallLedger with summary(), and the four triggers.
* ``baselines``: the ExecutiveBackend protocol, CallUsage, OfflinePlanBackend and RuleBackend.
* ``harness``: ExecutiveHarness, with a synchronous step and a worker thread that never blocks its callers.
* ``anthropic_backend``: AnthropicExecutiveBackend, one cache-aware Messages call per decision.
* ``smoke_anthropic``: one real call when ANTHROPIC_API_KEY is set. The tests do not run it.

Formulas. The shared ones are the version rule ``v_t = v_0 + t`` with ``v_0 = 0`` and no gaps
(store; source: the ADR-001 D3 versioning rule, a design definition), the nearest-rank percentile
``x_(ceil(p n / 100))`` (scheduler; source: the secondary Wikipedia definition in References), and
``cache_read_fraction = cache_read / (input + cache_creation + cache_read)`` (scheduler; source:
the Messages API usage fields in References). Each module docstring gives its own formulas and
their sources.

Citations. The ADR-001 sections are cited by number: section 1 (thesis), section 2 (levels; Level 1
never blocks), D3 (the executive), section 5 (seams 3 and 4), P5 (exit gate), H8 and H9. ADR-001 is
an internal record at agents/design/ADR-001-three-level-agent.md. It is local to this clone, has no
URL or DOI, and is located by path and sha256 f5f0fdb9ee56fc43743e668a9e918ca5ee0f26e415d2c57eb52af4a19c73870d.
The external references follow, each with its URL or DOI.

References:

* [R1] Russell and Wefald 1991, "Principles of metareasoning", Artificial Intelligence 49(1-3),
  361-395. https://doi.org/10.1016/0004-3702(91)90015-C. The DOI is confirmed by the Crossref
  record, checked 2026-10-16. ADR-001 section 11 gives no DOI.
* [L6] Anthony, Tian, Barber 2017, Thinking Fast and Slow with Deep Learning and Tree Search, NeurIPS
  2017. https://arxiv.org/abs/1705.08439
* [L7] Christakopoulou, Mourad, Matarić 2024, Agents Thinking Fast and Slow: A Talker-Reasoner
  Architecture, NeurIPS 2024 workshop. https://arxiv.org/abs/2410.08328
* [L8] NVIDIA 2025, GR00T N1, arXiv 2503.14734. https://arxiv.org/abs/2503.14734
* [L11] Horvitz 1987, Reasoning about Beliefs and Actions under Computational Resource Constraints, UAI
  1987. https://arxiv.org/abs/1304.2759
* [L12] Zilberstein 1996, Using Anytime Algorithms in Intelligent Systems, AI Magazine 17(3).
  https://doi.org/10.1609/aimag.v17i3.1232
* [L13] Labiosa and Hanna 2026, When to Plan: Learning to Select Between Reactive Control and
  Deliberative Planning, RLC 2026. https://arxiv.org/abs/2607.16421
* [L14] Yang, Hou, Wei, Bao, Chang 2026, Ares: Adaptive Reasoning Effort Selection for Efficient LLM
  Agents, arXiv 2603.07915. https://arxiv.org/abs/2603.07915
  (The [L6] to [L14] identifiers are those of ADR-001 section 11, which marks each one verified.)
* Nearest-rank percentile: https://en.wikipedia.org/wiki/Percentile, section "The nearest-rank method".
  This is a secondary source, and the page flags that section as possibly original synthesis.
* Python time.perf_counter: https://docs.python.org/3/library/time.html#time.perf_counter (a monotonic
  clock in fractional seconds, checked 2026-10-16).
* JSON Schema draft 2020-12: https://json-schema.org/draft/2020-12/schema
* Anthropic structured outputs: https://platform.claude.com/docs/en/build-with-claude/structured-outputs.md
  (checked live 2026-10-16: an unsupported feature returns HTTP 400).
* Anthropic prompt caching: https://platform.claude.com/docs/en/build-with-claude/prompt-caching.md
  (checked live 2026-10-16: the minimum cacheable prefix for claude-opus-5-5 is 512 tokens).
* Anthropic effort: https://platform.claude.com/docs/en/build-with-claude/effort.md (checked live
  2026-10-16: the default is medium on claude-opus-5-5, and max_tokens is a hard limit on thinking plus
  reply text).
* Anthropic models: https://platform.claude.com/docs/en/about-claude/models/overview.md
* Anthropic SDK, anthropic 1.15.0: https://github.com/anthropics/anthropic-sdk-python (the installed
  types are output_config_param.py, json_output_format_param.py and usage.py).
* Local copy of the Anthropic docs, not a stable reference: the claude-api skill bundle, version 2.1.302.

The package imports no submodules. SKILL.md has the rules, the gates and the open decisions.
"""

__version__ = "0.1.0"
