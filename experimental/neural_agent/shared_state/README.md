# shared_state/ — the scratchpad S_t

One versioned document between the LLM, the decision layer and the skills: six sections, one writer each,
a closed JSON schema, and one view per reader. The design record is the package docstring
(`__init__.py`); the decisions behind it are `../docs/SYSTEM_ONE.md` sections 8.10 and 9.10 to 9.13.
Import path: `experimental.neural_agent.shared_state` (unchanged by the 2026-10-18 split into a package).

| Module | Lines | What it holds |
|---|---:|---|
| `_common.py` | 152 | Section names and writers, the history bounds (`DECISION_HISTORY`, `INFORMATION_HISTORY`, `STATUS_HISTORY`, ...), the validation helpers every section uses. |
| `schema.py` | 250 | The JSON Schema of the whole document (only the keywords the manual validator enforces) and `validate`. |
| `sections/executive.py` | 162 | `GoalEntry` (the typed goal fields plus the LLM's words), `ExecutiveSection`; `entry_to_spec` / `spec_to_entry`. Writer: the LLM through the harness. |
| `sections/belief.py` | 150 | `BeliefLane`, `BeliefSection.from_belief`. Writer: the session, from the belief module. |
| `sections/decision.py` | 250 | `GoalInForce`, `DecisionRecord`, `InformationRequest` (questions to the LLM), `GoalReading` (how each goal was understood), `DecisionSection`. Writer: the decision layer, through the session's scribe. |
| `sections/requests.py` | 153 | `RequestsSection`: the live typed requests to the skills. Writer: the decision layer. |
| `sections/skills.py` | 114 | `SkillsSection`: active skills, bounded outcomes, the bounded status stream. Writers: the lower layer (statuses) and the session (active, outcomes). |
| `sections/events.py` | 69 | `EventsSection`: active interrupt events per lane. Writer: the runtime through the session. |
| `document.py` | 243 | `SharedState`: the document, `write(section, value, writer=, tick=)`, versions, JSON round trip, `_snapshot`. |
| `views/briefing.py` | 447 | The LLM's view as typed rows: per-NPC situation in words, goal in force with its condition as the decision layer read it, current request with its latest status, last outcome, events; questions; rejections; status counts; the delta since the LLM's version. |
| `views/render.py` | 354 | `ViewOptions(format, tick_seconds, lanes, delta, rejections)`; renderers `to_markdown`, `to_json`, `to_compact`; `estimate_tokens`; the information-budget CLI. |
| `views/llm.py` | 24 | `view_for_llm(state, options=None)`: the briefing rendered (markdown by default). |
| `views/jev.py`, `views/skills.py` | 128, 58 | The typed view the decision layer reads; the payload view the skills read. |
| `writers.py` | 238 | `SessionScribe` (belief, events, skills, decision at the host cadence; outcomes on report), `information_request_record`, `goal_reading`. |

Gates: `test_na_shared_state.py`, `test_na_briefing.py`, `test_na_requests.py` (views of the stream), the
lower layer's `test_sa_scratchpad.py` and `test_sa_joint.py`, `test_na_session.py` (the scribe in the loop).

## The LLM's view

```sh
PYTHONPATH=/Users/arush/hydl JAX_PLATFORMS=cpu .venv/bin/python -m experimental.neural_agent.shared_state.views.render --lanes 1 8 --show markdown
```

| format | 1 lane | 8 lanes | use |
|---|---:|---:|---|
| markdown (default) | 388 est. tokens | 943 | reading and quoting back; one table of NPCs |
| json | 586 | 2,069 | a model that answers in JSON; machine checks that every fact traces to a field |
| compact | 268 | 827 | long squads; one line per block |

Estimated tokens are characters over four (no provider tokenizer runs here). The 8-lane markdown bar in
`test_na_briefing.py` is 1,200 estimated tokens. Seconds in the view come from `ViewOptions.tick_seconds`
(0.045 in the gen-5 simulator, 1/30 in gen 7); the briefing itself keeps ticks.

Rules: every value in a view traces to one section field; an unknown belief feature is named as unknown,
never shown as zero; the latent is never shown; the "since your version" block lists only what has a tick
after the LLM's `issued_tick`; lists are bounded by the section limits, so a long episode does not grow the
view.

## Next (task #46)

Squads: an addressee per goal, one notepad per squad with a lane per NPC, a squad summary row above the NPC
rows. Objectives with stable ids that goals point at, with outcomes rolled up per objective.
