# experimental/neural_agent

The three-level agent of ADR-001 as code: an LLM executive (Level 3) writes goals in language and typed fields
onto a shared scratchpad; a fast decision layer (Level 2) turns them into skill requests on every tick; a skill
library (Level 1) executes them and streams status back. This package holds the middle layer, the seams to the
other two levels, the shared state, the measurement harness, and the lower layer's build (`skill_agent/`,
another session). Map last updated **2026-10-18**; the dated state of everything is
[`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md).

| | |
|---|---|
| Decision record | `agents/design/ADR-001-three-level-agent.md`. Local-only (excluded through `.git/info/exclude`); pinned by sha256 `6d81f8b7fb660fe0e66c9be9555c6eddf57ae3840dc07daf3d88442219768594` in `docs/ARCHITECTURE.md` section 8. |
| Shared interface | [`contracts.py`](contracts.py), sha256 `65a056b0eded18af62910ee30f379a4e145d6f713c5f3f4daac812911dc887a3`. The only module the others import shared names from; the pin is refreshed on every edit. |
| Where things stand | [`docs/MIDDLE_LAYER.md`](docs/MIDDLE_LAYER.md) (the decision layer: measured, built, designed, failed, per gate), [`docs/LEDGER.md`](docs/LEDGER.md) (every measurement with its command), [`skill_agent/docs/BUILD.md`](skill_agent/docs/BUILD.md) (the lower layer). |
| The decisions that shape the design | [`docs/SYSTEM_ONE.md`](docs/SYSTEM_ONE.md) sections 8.12 and 9.9 to 9.13: typed fields are the contract, words are context; behaviour programs; the request stream; how the LLM communicates. |

## The three levels, in one table

| Level | Decides | Cadence | Reads | Writes | Where |
|---|---|---|---|---|---|
| 3 Executive (the top LLM) | goals, priorities, typed conditions, horizons, objectives, when to reason | event-driven, seconds | the scratchpad's LLM view (belief, decisions, statuses, questions) | the scratchpad's executive section | `executive/`, `shared_state.py` |
| 2 Decision layer | which skill, with which arguments, under which condition; programs of steps; requests to the skills | every host cadence tick (0.5 s) plus events; sub-millisecond decisions | B_t, E_t, the status stream | goals, `ActionRequest`s, questions to the LLM | `jev/`, `requests.py`, `session.py` |
| 1 Skills | the action factors for this step | every tick (0.045 s in the gen-5 simulator, 1/30 s in gen 7) | B_t, the request, the legal observation | actions; `RequestStatus`es | `skill_agent/` (the real library, being built), `skill_library/` (stand-ins) |

Rules every level keeps: Level 1 never blocks; a decision slower than its budget is a miss and the previous goal
stands; commitment and duration belong to the skill's contract; legal observation only; nothing external on the
path except the top LLM; no live LLM call until the lower level is finished.

## Package map

Each folder has a `README.md` (purpose, files with line counts, gates, how to run) and, locally, a gitignored
`SKILL.md` with working notes. Tests live in `tests/experimental/neural_agent/` (gitignored by the repository;
present in this clone).

| Path | What it is | Gate |
|---|---|---|
| [`contracts.py`](contracts.py) | The shared contract: constants, `GoalSpec` (typed `Condition`, `addressee`, `objective`), `ExecutiveState` with `Squad`s and `Objective`s and its JSON schema, the per-lane lowering, `ManagedGoal` / `make_goal`, `SkillContract` and the control table, the JEV protocol. | `test_na_contracts.py` |
| [`shared_state/`](shared_state/__init__.py) | The scratchpad S_t as a package: `_common` (constants, validation), `schema` (JSON Schema, `validate`), `sections/` (executive, belief, decision, requests, skills, events: one module each), `document` (`SharedState`), `views/` (`briefing` and `render`: the LLM's briefing in markdown / json / compact behind `ViewOptions`; `jev`, `skills`: the other readers' views), `writers` (the session's scribe; `shared_state_writers.py` is an import shim). The import path `experimental.neural_agent.shared_state` is unchanged. | `test_na_shared_state.py` |
| [`requests.py`](requests.py) | The request stream between Levels 2 and 1: `ActionRequest`, `RequestArgs`, `RequestStatus` (with `signal`), validation with rejection codes, `ArgumentSchema`. | `test_na_requests.py` |
| [`session.py`](session.py) | One continuous run of the stack: executive seam, host deciders (the commitment-honouring seam, misses, `on_tick`), the scribe, the questions channel. | `test_na_session.py` |
| [`belief/`](belief/README.md) | B_t: `SharedBelief` and its 10 features per lane. | `test_na_belief.py` |
| [`executive/`](executive/README.md) | Level 3 seam: E_t store and schema validator, reports, scheduler, harness, the Claude adapter (no live call), the goal-language generator and teacher. | `test_na_executive*.py`, `test_na_goal_language.py` |
| [`jev/`](jev/README.md) | Level 2 instances: J0, the program executor, the exact typed path (`jtyped`), the runner; the J1 line; `lm/` (deprecated). | `test_na_jev_*.py`, `test_na_programs.py`, `test_na_typed_rule.py` |
| [`jev/system_one/`](jev/system_one/README.md) | J-S1, our own System One model, and the reading experiments (interpret, fine-tune pilot, catalogue matching, decoder readers, the deterministic fallback reader). | `test_na_system_one.py`, `test_na_interpret.py`, `test_na_condition_rules.py`, ... |
| [`eval/`](eval/README.md) | The composition driver (`run_composition`), the arm registry, metrics, datasets, DAgger relabelling; `out/` (ignored). | `test_na_eval.py`, `test_na_eager_loop.py`, `test_na_record_*.py` |
| [`tasks/`](tasks/README.md) | The composition task (pursue, evade when damaged, resume) and its arenas. | `test_na_tasks.py`, `test_na_arenas.py` |
| [`skill_library/`](skill_library/README.md) | Level 1 stand-ins: pursuit, evader, idle, the dispatcher and catalogue. | `test_na_skills.py` |
| [`skill_agent/`](skill_agent/README.md) | The real skill library's build (another session): surface, catalogue parser, protocol, combiner, library, freezing, traces, behaviour cloning. | `test_sa_*.py` |
| [`probes/`](probes/README.md) | Early measurement scripts; not a build target; `out/` (ignored). | none |
| [`docs/`](docs/README.md) | Thesis, architecture, formulas, orchestration, references, System One, the middle layer's status, the ledger, the current state. | `test_na_docs.py`, `test_na_docs_folders.py` |

## Running things

```sh
# the whole test tree for this package (CPU, no network; several minutes)
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/experimental/neural_agent -q -p no:cacheprovider

# the standard comparison on the composition task (see eval/README.md)
PYTHONPATH=/Users/arush/hydl JAX_PLATFORMS=cpu .venv/bin/python -m experimental.neural_agent.eval.run_composition \
  --arms j0,jtyped --runner eager --arena combat_open_flat --ticks 512 --batch 8 --episodes 8 --seed 0 --out <dir>

# the fallback condition reader's gate
PYTHONPATH=/Users/arush/hydl JAX_PLATFORMS=cpu .venv/bin/python -m experimental.neural_agent.jev.system_one.condition_rules --seed 0
```

Lint: `ruff check --isolated --no-respect-gitignore --select E4,E7,E9,F <files>` (the package is under an
`include` filter in `pyproject.toml`).

## Where to start

1. [`docs/CURRENT_STATE.md`](docs/CURRENT_STATE.md), then [`docs/MIDDLE_LAYER.md`](docs/MIDDLE_LAYER.md).
2. [`contracts.py`](contracts.py): each record and protocol has a docstring with its shapes.
3. The folder `README.md` of what you are changing, then its `SKILL.md` (local notes), then its gate.
4. [`docs/SYSTEM_ONE.md`](docs/SYSTEM_ONE.md) sections 8 and 9 for why the design is what it is.
5. [`docs/THESIS.md`](docs/THESIS.md), [`docs/FORMULAS.md`](docs/FORMULAS.md), [`docs/ORCHESTRATION.md`](docs/ORCHESTRATION.md), [`docs/REFERENCES.md`](docs/REFERENCES.md) for the research framing, every formula and every citation.

## Rules of work in this package

- Stage only the files you changed, never a directory; commit in batches; never push (the owner pushes).
- `docs/LEDGER.md` is append-only (`>>`); every number comes with its command and date.
- `contracts.py` is pinned by sha256 in `README.md`, `SKILL.md`, `docs/ARCHITECTURE.md`, `jev/SKILL.md` and two tests; refresh the pin on every edit.
- Private inputs under `agent_skills/` are read, never staged or copied into tracked files.
- `skill_agent/` belongs to the lower-layer session; coordinate through its `docs/BUILD.md` and messages rather than editing it.
- Code paths are stable: other sessions import `experimental.neural_agent.{contracts,shared_state,requests,session}`; a split keeps the import path through the package's `__init__` (as `shared_state/` does since 2026-10-18), and a move needs re-export shims and an owner decision.

## Limits

- Composition numbers measure the middle layer's decisions over three stand-in skills; the real skills are not in the loop yet.
- The gen-5 simulator (0.5.7, 0.045 s per tick) is what everything has run on; the gen-7 engine (0.7.2, 1/30 s) is being rendered and verified in `HytaleRL/hytalegym/hytalegym/sim/`.
- No live LLM has been in the loop; the executive is exercised through deterministic baselines, injected clients and the goal-language generator.
