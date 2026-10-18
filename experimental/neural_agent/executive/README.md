# executive/ — Level 3, the LLM seam

Everything between the top LLM and the rest of the stack: the versioned executive state E_t, the schema the
LLM replies in, the report channel upward, the scheduler that decides when the LLM is called, the harness
that calls it, and the procedural generator that stands in for the LLM's language in experiments.
Standing rule: **no live LLM call until the lower level is finished**; the tests never call a model.

| File | Lines | What it is |
|---|---:|---|
| `store.py` | 308 | The versioned E_t store (the E_t writer seam) and `validate_against_schema`, the manual JSON-Schema validator (type, properties, required, items, bounds, `enum`). |
| `reports.py` | 136 | Upward report channel: goal outcomes and interrupt events from Levels 1 and 2 (`GoalReport`, `EventRecord`). |
| `scheduler.py` | 377 | Virtual clock, call ledger and call triggers: when the executive is consulted. |
| `baselines.py` | 200 | The executive backend protocol, the usage record, two deterministic baselines. |
| `harness.py` | 259 | `ExecutiveHarness`: call the backend on a trigger, time it, log it, store the result, write the scratchpad's executive section. |
| `anthropic_backend.py` | 339 | The Claude adapter behind the seam: `build_system_prompt` (role, catalogue, the condition vocabulary and units, the JSON Schema), `wire_schema` (the structured-output subset). Tested through an injected client only. |
| `smoke_anthropic.py` | 56 | One real call when `ANTHROPIC_API_KEY` is set. Never run by the tests. |
| `export.py` | 61 | E_t history as JSON lines. |
| `patterned.py` | 170 | The patterned language: `PATTERN_DIRECTIONS` (the prose the system prompt carries), the canonical clause per condition kind, and a generator of patterned goals with controlled breakage for P7 (SYSTEM_ONE 9.14). |
| `goal_language.py` | 752 | The goal-language generator and teacher: `INTENT_FAMILIES`, the nine `CONDITION_FAMILIES` with their paraphrases, `sample_state` (an executive section in the LLM's words, with the typed `condition` when hinted), `Predicate.evaluate`, `teacher` (the exact rule M13), `split_wordings` (the held-out wording split every reading experiment uses). |

Gates: `test_na_executive.py`, `test_na_executive_export.py`, `test_na_goal_language.py`.

What the LLM writes (the contract, `../contracts.py`): per goal a typed `skill_id`, `priority`, `params`,
`constraints`, `horizon_seconds`, `deadline_tick` and, since 2026-10-17 late, a typed `condition`
(`{kind, argument}` from the nine kinds, or null for always); plus its own words `intent`, `when`, `note`
in the scratchpad's executive section (`../shared_state.py`). Decisions: `../docs/SYSTEM_ONE.md` sections
8.12, 9.11 to 9.13.

Pending here (plan step 7 of `../docs/MIDDLE_LAYER.md`): the backend writing the full executive section
(words included) rather than `ExecutiveState`; the objectives list; per-squad views.
