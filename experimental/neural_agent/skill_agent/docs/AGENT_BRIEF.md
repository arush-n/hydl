# Brief for the agent building skill_agent phase A (2026-10-17, night)

You are building the first phase of the lower layer of a three-level game agent in the repository
`/Users/arush/hydl` (branch `main`). Phase A needs no simulator. Everything you build is proved by a
gate before it is claimed, recorded in an append-only ledger, and committed in batches. Nothing is
pushed. Read this whole brief, then `docs/BUILD.md`, before writing code.

## 1. Read first, in this order

1. `experimental/neural_agent/skill_agent/docs/BUILD.md` (whole): the layout (section 1), the resources (2),
   the architecture (3), the gen-7 checklist (4), the proofs E1-E14 (6), the order of work (7).
2. `experimental/neural_agent/docs/MIDDLE_LAYER.md` sections 3 to 5, and `experimental/neural_agent/docs/SYSTEM_ONE.md`
   §9.10: the request stream contract you implement (requests down, status up, the fixed greedy combiner).
3. `agent_skills/training_runs/SKILL_CATALOGUE.md` (the 139 skills, their numbers, instruction -> done),
   `SKILL_LIBRARY.md` §2 and §7 (what every skill needs; the sixteen requirements), `GENERAL_AGENT.md`
   (design rules). This folder is PRIVATE: read it, never `git add` anything under `agent_skills/`,
   never copy its content into tracked files beyond short quotations in tests' inline fixtures.
4. `experimental/neural_agent/contracts.py` (`SkillContract` with the control fields, `GoalPayload`,
   `control_table`, `EVENT_NAMES`, `PARAM_WIDTH`, `MAX_GOALS`), `shared_state.py` (sections, `write`,
   views), `skill_library/` (the proof-of-concept idle, pursuit and evader contracts; the dispatcher
   that runs one skill per lane; `descriptions.py`), `adk/architecture/hierarchy.py` (the loop the
   bodies run in), `HytaleRL/hytalegym/hytalegym/jax/combat/observation/v3/policy/layout.py` (the
   twelve action heads of today and their sizes).
5. `HytaleRL/hytale-plugin-0.7.2/SIMULATOR-0.7.2.md` and `TRAINING-0.7.2.md` (gen-7 status and the
   measured 0.7.2 changes), `HytaleRL/hytale-plugin/src/main/java/com/hytalerlbridge/imitation/`
   (NpcTraceFrame and the recorder) for deliverable A7.
6. `experimental/neural_agent/SKILL.md` and `skill_agent/SKILL.md` (folder rules and traps).

## 2. Rules

- Python: `.venv/bin/python`; set `JAX_PLATFORMS=cpu` for every run; macOS, so no `timeout` command;
  `cmd | tail` hides exit codes (use `${pipestatus[1]}` in zsh or echo `$?` right after).
- Lint: `.venv/bin/ruff check --isolated --no-respect-gitignore --select E4,E7,E9,F <paths>`.
- Tests live in `tests/experimental/neural_agent/test_sa_*.py`; the `tests/` tree is gitignored by
  policy and that is expected. Record the baseline count of the package gate before your first change
  (`JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/experimental/neural_agent -q -p no:cacheprovider`)
  and report every later run as a delta against it.
- Imports across packages only from `experimental.neural_agent.contracts`, `shared_state`, the ADK and
  the gym. New public names in `contracts.py` are allowed; then refresh `CONTRACTS_SHA256` and the
  `__all__` count in `tests/experimental/neural_agent/test_na_docs.py`, and the pin string in
  the package's `README.md`, `SKILL.md`, `docs/ARCHITECTURE.md` and `jev/SKILL.md` (under `experimental/neural_agent/`).
- Every gate run goes into `experimental/neural_agent/docs/LEDGER.md` as an appended entry
  (`## <date> — <what>` / `command:` / `result:` / `verdict:`) before you claim it. Append a dated note
  to `skill_agent/SKILL.md` for every discovery or trap.
- Do not touch (under `experimental/neural_agent/`): `jev/`, `executive/`, `session.py`, `eval/`, `tasks/`, `docs/*.md`
  other than the ledger, the proof-of-concept skills' behaviour, anything under `HytaleRL/hytalegym/hytalegym/jax/` (the
  simulator), or `agents/`. You may extend `shared_state.py` with the two new section types (A3) and `contracts.py` with
  new names.
- No training runs of any kind in this phase. No pushes. Commit each deliverable as its own batch on
  `main` with a message that says what was built and what the gate showed; stage only files you
  changed, never `git add` a directory.
- Owner rulings you must honour: gen-7 is the target; every component is proved before it is final;
  the neural layer follows directions and never parses words; requests are structured (coordinates,
  directions, scalars); conflicts are resolved greedily by urgency then confidence; independent
  actions combine; legal observation only.

## 3. Deliverables, in order (each: build, test, gate, ledger, commit)

**A1 `contract/surface.py`: the gen-7 surface specification.**
Token types as frozen records with documented fields: `EntityToken` (role from a fixed set: enemy,
ally, animal, merchant, player, unknown; relative position and velocity; visible; last-seen age; health
fraction; marked), `BlockCandidateToken` (cell, face, block kind, marked), `InventorySlotToken` (slot,
item id, quantity, durability, marked), `GeometryToken`, `LightToken`, `SelfToken` (health, stamina, mana,
signature energy, oxygen, grounded, yaw, pitch, speed), `KitToken` (combo window, hold thresholds,
charged tiers, costs of the held item), `RouteToken` (next waypoint). The action-head layout for 0.7.2:
today's twelve heads from `layout.py` with their sizes, plus `duration` (hold-for-k-ticks buckets),
`item_move`, `item_drop_give`, `recipe`, `door`, `mount` (ride / glide controls), `rune_set`, `cast`,
and `dodge` reduced to left and right. Six scopes (`locomotion`, `gaze`, `hands`, `posture_defence`,
`interaction`, `abilities_magic`) as a mapping head -> scope; `scope_mask(scope, present_heads)` so the
same code runs on today's twelve heads (gen-5 layout) and on the full gen-7 layout; a JSON export.
Gate (part of E1): every head belongs to exactly one scope; the twelve present heads reproduce
`layout.py`'s names and sizes exactly; JSON round-trips.

**A2 `contract/catalogue.py`: the catalogue parser and the contracts it yields.**
Parse `SKILL_CATALOGUE.md` sections 1 to 14 (markdown tables; columns skill | what a person does |
Hytale mechanic | instruction -> done | kind | gen 5; a cell may hold several ids such as
"walk_to, run_to, sprint_to") into `CatalogueSkill` records: ids, group (section), description (the
"what a person does" text), mechanic text, kind (L / C / A), the gen-5 note (kept as information only),
and the instruction parsed into a typed argument schema: which of position, direction, entity, cell
(+face), region, item, recipe, route it needs, plus named scalars (radius, pace, count, duration ...),
and the done signal. From a record build a `SkillContract` (typed argument schema, head-group scope by
group and verb, termination / success / failure signals from "done"; control-field defaults as today)
and a description string for the middle layer's matching. Section 14's activities become
`ActivityProgram` records (ordered verb steps with exit conditions as written). Provide a CLI that dumps
the parsed catalogue as JSON. Tests use a small inline fixture table (a handful of rows you write in the
test, not copied wholesale) and a full-file test that skips when the private file is absent.
Gate E1: on the real file, every learned row maps to exactly one scope and a complete argument schema
(report the count, expected 86 of 139 with 33 commands and 19 activities; report the exact numbers you
parse and any row you could not parse, verbatim); every row has a description; the three proof-of-concept
skills map onto contracts compatible with `skill_library/` (same ids, same control fields).

**A3 `requests.py` and the scratchpad sections.**
`ActionRequest` (request_id, issued_tick, expires_tick, verb, args with presence flags: position[3],
direction[3], entity, cell + face, scalars[4] named by the verb, mode = parallel | after:id | interrupt,
urgency, confidence, priority, goal_slot, executive_version, program_step) and the status records
(accepted | rejected(reason) | deferred(conflict_with) | running | done(outcome) | aborted(reason),
with ticks; progress signals; per-tick head composition). `validate(request, contract, legal_mask) ->
rejection reason or None` (verb exists; required args present and finite, positions inside the world
bounds given, entity known; scope legal under the mask; `after` predecessor done). Add to
`shared_state.py` a `requests` section (writer `jev`) and extend the `skills` section with the status
stream (writer `skills`), keeping one writer per section, the version bump, the JSON schema and the
views consistent (the LLM view shows accepted / rejected counts and the last rejections with reasons;
the JEV view carries the live request ids and statuses). Gate R1: every invalid request is rejected with
the right reason and valid ones pass; the scratchpad tests still pass; schema round-trip.

**A4 `contract/references.py`: the resolver.**
`to_agent_frame(world_point, agent_position, agent_yaw) -> vector` (and its inverse), `entity_index(
handle, entity_tokens) -> index or None`, `candidate_index(cell, face, block_tokens) -> index or None`,
`mark(tokens, index)`. Gate: rotation and translation correct on known geometry (including yaw wrap),
an unknown handle yields a rejection reason, marking touches exactly one token.

**A5 `runtime/combiner.py`: the fixed greedy combiner, with stub bodies.**
A stub body is a deterministic function from (request, tokens) to factors for its scope's heads. The
combiner, per tick and per agent: take live requests (not expired, not done); validate; sort by
(urgency, confidence, priority, recency); grant a request if every head in its scope is free, else
defer naming the holder; run the granted stub bodies; merge into one factored action over the present
heads with unused heads neutral; emit statuses and the head composition. Also the in-graph form: an
active-request table of one granted request per scope, as arrays, and a test that the table and the
host path produce the same action on the same stream. Gates R2 (no two granted requests share a head),
R3 (the order is exactly (urgency, confidence, priority, recency) and reproducible), R4 (`after` chains
run in order; parallel sets co-run), R6 (compiled table = host), and E7 with stubs: a walk stub and a
look stub co-run on one agent; two movement stubs conflict and the loser is deferred with the holder's
id; an expired request is dropped with a status.

**A6 Parity adapter: today's goals as single requests.**
An adapter that expresses the current dispatcher's input (one goal per lane: skill index, params,
slot, version) as one request per lane and runs the three proof-of-concept skills through the combiner
path on today's twelve heads. Gate: the four-arm composition smoke (`run_composition --arms
flat,j0,shuffled,j1 --ticks 64 --batch 8 --episodes 1 --seed 0`) produces identical non-timing metrics
with and without the adapter (compare `metrics.csv` columns: catches, damage_events,
goal_switches_per_lane_run, interrupts, commitment_violations). This is the owner's "today's behaviour
is the one-request special case".

**A7 Native NPC traces from the live 0.7.2 server.**
Using the bridge (`HytaleRL/hytale-plugin-0.7.2/tools/launch_bridge_072.py`, `smoke_bridge_072.py`) and
the native trace recorder, record Trork and Kweebec behaviour (movement and melee) for as long as the
environment allows, into `agent_skills/training_runs/traces_072/` (private; never staged), and write
`skill_agent/training/traces.py`: a loader from NpcTraceFrame batches to arrays with a documented
schema. Gate: traces on disk with frame counts and durations reported; the loader round-trips one
batch. If the live server cannot be started on this machine, report the exact blocker and stop this
deliverable; do not simulate traces.

**A8 `docs/FIDELITY.md`: the gen-7 fidelity ledger.**
Create `skill_agent/docs/FIDELITY.md`, append-only, seeded with every row of docs/BUILD.md section 4 as
`pending`, with the expected value, the differential that will verify it and the pass bar. Nothing is
marked passed in this phase.

## 4. Gates you run at the end of the phase

```
cd /Users/arush/hydl
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/experimental/neural_agent -q -rf -p no:cacheprovider; echo "pytest exit=$?"
.venv/bin/ruff check --isolated --no-respect-gitignore --select E4,E7,E9,F experimental/neural_agent/skill_agent/ experimental/neural_agent/shared_state.py experimental/neural_agent/contracts.py tests/experimental/neural_agent/; echo "ruff exit=$?"
PYTHONPATH=/Users/arush/hydl JAX_PLATFORMS=cpu .venv/bin/python -m experimental.neural_agent.skill_agent.contract.catalogue --json | head -c 2000
```
The package gate must show your baseline count plus your new tests and no other change; name any
test you changed and why.

## 5. Report (the only thing the main session sees)

Per deliverable: files with line counts and sha256; the gate output verbatim; the before and after
test counts; deviations from this brief and how you resolved each (one line each); anything you could
not make pass, verbatim. At the end: the parsed-catalogue counts, the composition parity table, the
trace inventory or the blocker, and the one claim you would most expect to be wrong. Do not claim
anything you did not run.
