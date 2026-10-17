# experimental/neural_agent: architecture

Status at a snapshot taken on 2026-10-16 at about 20:53 local time, updated 2026-10-17 (UTC; the local clock read 2026-10-16 about 22:15, CDT). The package is being built in parallel, so the file list below is what was on disk at those times. It is not a claim about the final state. Sections 1 to 5 and 7 to 8 are a snapshot, and this document does not verify them, except where a passage names its own check. Section 6 is the checklist for the verifier. It is filled from the main session's gate run of 2026-10-17, and a box is ticked only where its tests passed. This update renames the skill folder to `skill_library/` throughout, adds the session-runner rules (FORMULAS D10 to D12) and records the flaw that the 256-tick composition smoke exposed (section 6).

**Record caveat.** The decision record is ADR-001 (`agents/design/ADR-001-three-level-agent.md`), which is local-only and is not in this repository. PLAN v0.2 and the other design files cited here are local-only as well. Section 8 pins each cited file by SHA-256. Companion documents: `THESIS.md` (research question, hypotheses, metrics), `FORMULAS.md` (every formula, with its label) and `REFERENCES.md` (bibliography).

Labels follow ADR section 0: IMPLEMENTED, VALIDATED, COMMITTED, HYPOTHESIS.

---

## 1. Package map

**`contracts.py`** (present). The one module that other modules import shared names from. It defines the constants (`MAX_GOALS = 4`, `PARAM_WIDTH = 4`, `SKILL_STATE_WIDTH = 128`, `ACTION_HEADS = 12`, `BELIEF_LATENT_WIDTH = 64`, `TICKS_SATURATION = 256.0`, `SKILL_IDLE = "idle"`). It defines the host records `GoalSpec`, `ComputeBudget`, `ExecutiveState`, `SkillContract`, `GoalReport`, `JevBudget` and `JevStats`, and the batched PyTrees `SharedBeliefState`, `ExecutiveArrays` and `GoalPayload`. It also provides the functions `executive_arrays`, `empty_executive_arrays`, `make_goal`, `idle_goal`, `event_mask` and `feature_index`, the protocols `Skill` and `JevInstance`, and the JSON schema `EXECUTIVE_STATE_JSON_SCHEMA`. `__all__` lists 31 names (checked after the 20:00:18 change). Its contract tests are `tests/experimental/neural_agent/test_na_contracts.py`, which `.gitignore` excludes (see section 7).

**`__init__.py`** (present). Imports no submodules. It holds `__version__ = "0.1.0"` and the package docstring.

**`belief/`** (present: `core.py`, `features.py`, `__init__.py`, `SKILL.md`). The producer of B_t. `core.py` defines `SharedBelief`, the ADK `BeliefCore`. It holds the carry, the GRU, the two tick counters, and a `contract` property (line 292). `features.py` reads the eight history-free features from the legal observation. The module reads only `actor_input.legal_observation` and never imports the privileged training input. The belief resets at episode boundaries only (ADR D6). Its `SKILL.md` records the column sources and formulas, which this document does not repeat.

**`skill_library/`** (present: `_common.py`, `idle.py`, `pursuit.py`, `evader.py`, `library.py`, `dispatcher.py`, `__init__.py`, `SKILL.md`). Level 1. `_common.py` holds the shape checks, the head offsets and the mask-legality checks for the 12-head surface. `idle.py` defines `IdleSkill`, which chooses the neutral on every head, restricted to the mask. `pursuit.py` defines `PursuitSkill`, which loads the pinned `pursuit_contact` candidate v0001 through that package's scoped loader. `evader.py` defines `ScriptedEvaderSkill`, a scripted rule. Its docstring says the frozen evader network is not on disk. `library.py` defines `SkillRegistry`, with idle at index 0. `dispatcher.py` defines `SkillDispatcher`, the ADK `LowLevelActor`. **Renamed from `skills`** (2026-10-17). `.gitignore:189` ignores every directory named `skills`, so the old name kept the library out of git. After the rename, `git check-ignore -v` lists neither `skill_library/dispatcher.py` nor `skill_library/library.py`. Only `skill_library/SKILL.md` is ignored, by `.gitignore:176`, like every `SKILL.md`.

**`jev/`** (present: `scripted.py`, `runner.py`, `network.py`, `train_j1_ppo.py`, `kev.py`, `lm/`, `__init__.py`, `SKILL.md`). Level 2 instances behind one protocol. `scripted.py` defines `ScriptedJev`, which is J0. It runs IN_GRAPH in pure JAX and its rule is FORMULAS D9. `runner.py` defines `JevRunner`, which binds a `JevInstance` to the ADK core. It carries E_t, passes it to the instance, and applies the latency budget to HOST-mode calls. On a miss while the core is forcing a switch, it emits the idle goal (FORMULAS D12). `network.py` holds J1, the learned reactive network over B_t and E_t with no model, which the ADR names the primary instance. `train_j1_ppo.py` trains J1 with PPO over goal decisions, treated as a semi-Markov decision process. `lm/` holds J-LM: `instance.py`, `prompting.py`, `protocol.py`, `anthropic_quick.py` and `ollama_qwen.py`, with its own `SKILL.md`. `kev.py` holds a reserved KEV slot. It defines `KevJev`, whose constructor raises `NotImplementedError` until the owner supplies a definition. The abbreviation KEV is not defined in ADR-001, PLAN.md, the neural_agent README, SKILL.md or DEV.md, nor in any Markdown or Python file of the repository (`grep -rnw KEV`, 2026-10-16), so this document assigns it no behaviour. J2 to J4 are not present. The record puts them in the same folder later, behind the same contract, gated by ADR section 8.

**`executive/`** (present: `store.py`, `reports.py`, `scheduler.py`, `baselines.py`, `harness.py`, `anthropic_backend.py`, `smoke_anthropic.py`, `__init__.py`, `SKILL.md`). The seam for Level 3. It is not the executive. `store.py` defines `ExecutiveStateStore`, which holds the versioned E_t. It accepts a proposal only if it validates against `EXECUTIVE_STATE_JSON_SCHEMA`, names catalogue skills, and carries the next version. `reports.py` defines `ReportChannel`, a bounded FIFO that drops the oldest item and counts the drop. `scheduler.py` holds `VirtualClock`, `CallLedger` and the triggers (fixed cadence, per transition, event-driven, never). It logs the four allocation knobs of ADR D3 for each call. `baselines.py` holds the backend protocols and two deterministic backends, `OfflinePlanBackend` and `RuleBackend`. `harness.py` defines `ExecutiveHarness`, which calls a backend on a trigger, times it and stores the result. `anthropic_backend.py` is an adapter for the Anthropic Messages API. It puts the stable part of the prompt first, for caching. `smoke_anthropic.py` makes one real call when an API key is set, and the tests do not run it. The record leaves the provider, the model tiers, the cost and the determinism open (ADR section 12, question 3).

**`tasks/`** (present: `composition.py`, `events.py`, `flat_arena.py`, `__init__.py`, and since 22:14 local `arenas.py`, which lets `eval/run_composition.py` choose `flat_pursuit` or `combat_baseline`). Composition tasks for P2. `composition.py` defines a task in which the executive writes a pursuit goal and an evader goal gated by `on_damage`. The agent pursues, evades when damaged, and then resumes pursuit. `events.py` holds the `DAMAGE_RECEIVED` read helper and an ablation injector. `flat_arena.py` rebuilds the flat two-actor pursuit arena that the pursuit candidate was trained on. The three P2 composition cases in ADR section 8 are sequencing (navigate, then pursue), interruption (evade on `PROJECTILE_THREAT_ENTERED` or `DAMAGE_RECEIVED`) and dependency (interact after arrival). The damage case is covered here, and no file on disk covers the other two.

**`eval/`** (present: `arms.py`, `metrics.py`, `run_composition.py`, `traces.py`, `eager_loop.py`, `__init__.py`). The evaluation of ADR sections 8 and 9. `arms.py` defines the comparison arms: `flat`, `j0`, `shuffled`, `j1` and `j_lm`. `j1` and `j_lm` were placeholders at the snapshot and were wired at 22:16 local. `j1` is `NetworkJev` inside `JevRunner`, a random initialization labelled `j1-untrained` until parameters are supplied. `j_lm` is a host-mode arm that runs only through `session.run_session` (the `arms.py` module docstring). The shuffled arm wraps the J0 manager and draws a fresh random permutation of the skill catalogue at each decision, shared by all lanes. A proposed skill index `s` becomes `perm[s]` (`ShuffledGoalManager.__call__`). `metrics.py` holds pure numpy functions over `[T, B]` arrays. Its docstring refers to `eval/README.md`, which now exists (section 7, item 17). `run_composition.py` runs arms on a composition task and writes `metrics.csv` and `REPORT.md`. `traces.py` writes the trace dataset for the learned J1 from a J0 run. `eager_loop.py` defines `run_eager`, a Python step loop for host-mode decision models. By its docstring it repeats the sequence of `adk.runtime.hierarchical.collect_hierarchical` one step at a time, with the same key splits, and its test checks equivalence on CPU.

**`session.py`** (present, package root). One continuous run of the three-level stack. It drives `run_eager` with a host hook that advances a virtual clock by one tick per environment step (`tick_seconds` is 0.045 s by default) and connects the executive seam, the JEV and the skills. A host-mode decision is accepted only if it returns a goal within `host_budget_ms`. Otherwise the goal in force stands and a miss is counted (FORMULAS D10 and D11).

**`probes/`** (present: `probe_events.py`, `probe_timing.py`, `README.md`, `out/`, and `SKILL.md` since 2026-10-17). Measurement scripts. The ADR does not name them, and the brief does not either. The probes compose `HierarchicalAgentCore` with a pass-through belief, a constant manager and an actor. `probes/README.md` records measurements taken on CPU on 2026-10-16 with one seed. Outputs under `probes/out/` are git-ignored.

**`docs/`** (this folder). `THESIS.md`, `ARCHITECTURE.md`, `FORMULAS.md`, `REFERENCES.md` and `SKILL.md`. Git does not ignore these, except `SKILL.md`, which `.gitignore:176` ignores by name (checked 2026-10-16). None is tracked yet: the whole `experimental/neural_agent/` folder was untracked on that date.

---

## 2. Data flow between levels

```text
LEVEL 3   LLM executive (API model), asynchronous, event-driven
          writes  E_t (ExecutiveState, versioned)  --executive_arrays()-->  ExecutiveArrays
          reads   abstracted B_t, the DecisionEvent set, GoalReport outcomes, memory summaries
             |                                                   ^
             | E_t, lowered (ExecutiveArrays)                    | GoalReport: completed,
             v                                                   | interrupted, failed, timeout
LEVEL 2   JEV instance (J0, J1 or J-LM now; J2 to J4 later), about 2 Hz plus interrupts
          inputs  B_t, E_t, current GoalPayload, ActorPolicyInput (legal only),
                  DecisionContext, GoalRuntimeContext, PRNG key
          output  GoalDecision: ManagedGoal(payload=GoalPayload, control=GoalControl)
          budget  decision accepted iff latency <= L; otherwise the previous goal stands and misses += 1
             |
             | ManagedGoal: skill_index, params, executive_version, goal_slot;
             |              minimum commitment, maximum duration, emergency-interrupt mask
             v
          HierarchicalAgentCore (ADK) applies due, commit, timeout, emergency and switch rules
          (FORMULAS D1 to D4) and passes the active goal down
LEVEL 1   skill, selected by the dispatcher from skill_index
          inputs  state float32[B,128], B_t, ActorPolicyInput, params float32[B,4], key
          output  next state float32[B,128], factors int32[B,12]
                  (inside the skill scope and the runtime action mask M_t)
             |
             | int32[B,12] factors, checked again before execution
             v
ENVIRONMENT  Hytale server (native, authoritative: legality, rules, collision, damage, death)
             HytaleGym (JAX, an approximation; where they disagree the server is right)

BELIEF       belief core: legal O_t, previous action A_{t-1}, elapsed time Delta t  -->  B_t
             (features float32[B,10], feature_known bool[B,10], latent float32[B,64]);
             read by Levels 1, 2 and 3 (abstracted); reset at episode boundaries only
```

---

## 3. The JEV contract

**Inputs.** The instance's `__call__` receives `state` (its own state), `belief` (`SharedBeliefState`), `current_goal` (`GoalPayload`), `actor_input` (`ActorPolicyInput`, legal only), a PRNG `key`, and the keyword arguments `decision_context` (`DecisionContext`), `goal_runtime` (`GoalRuntimeContext`) and `executive` (`ExecutiveArrays`, the lowered E_t). The ADK core calls its manager without the `executive` keyword (`adk/architecture/hierarchy.py:304-312`). `jev/runner.py` supplies E_t, and its docstring says so.

**Output.** An ADK `GoalDecision`, carrying the new instance state, a `ManagedGoal` (payload `GoalPayload`, control `GoalControl`), a `terminate` flag (False means continue the goal in force), and diagnostics. The contract also asks for a declared compute request (`JevBudget.compute_units`).

**Latency budget.** The record requires a budget per deployment target (ADR D2). There are two records of it today. `ComputeBudget.jev_latency_ms` is the executive's grant (default 20.0 ms). `JevBudget.latency_ms` is the instance's declaration, and the runner takes it. The record does not say which governs (open; section 7).

**Rule.** A decision is accepted if its wall-clock latency in milliseconds is at most the budget. Otherwise the previous goal stays in force and `misses` increases by one (FORMULAS D7; ADR section 2). In the current runner, the rule applies to HOST-mode calls only, and IN_GRAPH calls are not timed (FORMULAS D9).

**Timeout fallback.** For a missed HOST call, `jev/runner.py` keeps the previous instance state and the previous goal payload with its previous control, sets `terminate` to False, and marks the diagnostics with `miss: True` (`jev/SKILL.md` F2). When the core is forcing a switch on a lane at that moment (an invalid or timed-out goal, or an actor termination request), the runner emits `idle_goal` on that lane instead, because re-emitting the ended goal would renew it with a fresh age. ADR D2 now records this as a decision of 2026-10-16 (FORMULAS D12; `JevRunner._decide`). When `goal_runtime` is supplied, this also covers the very first decision, when no goal is valid yet.

**Misses and statistics.** `JevStats` holds `decisions`, `misses`, `latency_ms_p50`, `latency_ms_p95` and `compute_units`. Percentiles use linear interpolation in the runner, and nearest rank in the executive ledger (FORMULAS D8). Report the batch size and the device with every latency figure (PLAN §30.4).

**Enforcement.** ADR section 5, seam 8, requires the contract to be "enforced in orchestration so every instance is measured the same way". In the current code, the runner enforces it for HOST-mode instances only. The ADK orchestration does not enforce it yet (ADR section 3 D2: "Orchestration in `adk/` does not enforce the latency budget yet (seam 8)").

**Known gaps.** DEV.md (19:55 version) lists three integration gaps. The snapshot shows the following.

- *The executive argument.* The core calls the manager without `executive`. `jev/runner.py` now binds E_t, so this gap is addressed in code. This document did not test the binding.
- *The skill adapter.* The core calls the actor with the goal payload and expects an ADK `ActorDecision` (`hierarchy.py:374`). `skill_library/dispatcher.py` is documented as doing that mapping. This document did not test it.
- *The `contract` attribute.* The ADK protocols require a `contract` (`adk/architecture/components.py:94-98`). `belief/core.py` defines one (line 292). A grep of `jev/scripted.py`, `jev/runner.py`, `jev/network.py` and `jev/lm/instance.py` found none. The skill modules carry a `contract` field of type `SkillContract`, which is a different type from the ADK's. Whether the ADK protocol checks pass is not verified.

---

## 4. The three memories (ADR D6, COMMITTED)

| Memory | Owner | Lifetime | Reset |
|---|---|---|---|
| Shared recurrent belief: current environmental and temporal state | shared infrastructure; Levels 1 and 2 read it, Level 3 reads an abstraction | episode | episode boundary only; never on a skill switch |
| Skill-local state: specialized short-term execution | the skill, declared in its contract | one goal activation | on skill entry; discarded on exit |
| Persistent task and episodic memory: objectives, completed tasks, observations, inferred relationships, strategic commitments; E_t is its JEV-readable projection. Precedent: the memory stream of Generative Agents [G1] (ADR D3) | Level 3 | across episodes | explicit executive action only |

ADR D6 adds: Level 3's input is an abstraction of legal observation and memory, and B_t is shared between the LLM and the JEV and read by the skills. No level treats it as a private latent. Its implementation status is "Not implemented: cross-skill handover of `actor_state`; any episodic memory store" (ADR D6).

---

## 5. Four-arm evaluation protocol

The protocol is `experimental/hierarchical_planner/DESIGN.md` section 6, adopted by ADR-001 D2 and P2. The table maps each arm to this package's code (the arm names in `eval/arms.py`).

| Arm | DESIGN.md section 6 wording | This package | Question the arm answers (DESIGN.md) |
|---|---|---|---|
| 1 | flat actor, no hierarchy | `flat`: the skill policy alone, with no JEV | "the real baseline" |
| 2 | scripted manager | `j0`: `ScriptedJev` (FORMULAS D9) | "does the seam work; is hierarchy worth anything here?" |
| 3 | transformer manager | `j1`: `NetworkJev` (`jev/network.py`) inside `JevRunner`, wired via `eval/arms.py`; it runs untrained (`j1-untrained`) until a checkpoint is supplied. The ADR names J1 the learned reactive network (D2). | "does learning beat scripting?" |
| 4 | transformer manager, shuffled goals | `shuffled`: the J0 arm with a random permutation of skill indices at each decision | "does the goal content matter, or only the timing?" |

Notes on the mapping.

- DESIGN.md arm 3 says "transformer manager". ADR-001 makes the learned controller J1, a small reactive network with no model, and it does not name a transformer. This document reads arm 3 as the learned controller, which is the wording ADR P2 uses. The record does not reconcile the two.
- In the current code, the shuffled arm permutes the skill indices of the J0 arm. DESIGN.md's control is defined over a learned manager's goals. The record does not say which goals are shuffled, so the arm is not yet the control the ADR specifies. The ADR says each instance must be compared with "the shuffled-goal control" (D2). The arm 4 result should be read with that in mind.
- DESIGN.md section 6 reads: "Arm 4 is the honest control. If 3 ≈ 4, the manager is providing temporal structure rather than meaningful goals, which is a different (and cheaper) result."
- Gates (ADR P2): J0 must complete composite tasks the flat actor cannot, on held-out seeds. J1 must beat J0 on the same tasks with intervals, and beat its own shuffled-goal control. The same test is H16 in ADR section 3 D2.
- Comparison rule (ADR D2 and section 9): each instance is compared with the instance before it, with identical skills and identical observation constraints. Every claim about a level is made against the preceding arm on its axis (ADR section 9).
- `j_lm` is not one of DESIGN.md's four arms. It is the J-LM arm of H15, wired via `eval/arms.py`, and as a host-mode instance it requires `--runner eager` (it runs through `session.run_session`).

---

## 6. What is implemented versus claimed

Each module is IMPLEMENTED only once its tests pass in the Verify stage. A box is ticked only where its tests passed in the gate of record, and the count follows it. A box stays empty when the module has no gate, or when its gate did not pass.

**Gate of record: the main session's whole-tree run of 2026-10-17** (CPU, no network): 371 passed, 2 skipped, 1 failed, and ruff clean. The one failure is `test_na_docs.py::test_adr_pin_matches_the_local_record_when_present`. ADR-001 changed after it was pinned, so the pin needs a refresh, which the main session does last. The per-file counts below are the verifiers' runs, and their passes add up to the 371 when the docs file counts 8 passes and the pin failure. The two skips are the live Ollama test (`NEURAL_AGENT_LIVE_OLLAMA=1`) and the eval refusal path, which runs only while a sibling module is missing.

**After the gate run** (this pass, about 22:12 local on 2026-10-16, which is 2026-10-17 UTC; a collection and a docs run, not a gate re-run): `pytest --collect-only` finds 381 tests, 7 more than the gate's 374. The new ones are `test_na_jev_runner_fallback.py` (3) and four more in `test_na_tasks.py` (34 now), for the runner's idle fallback and the strict success rule. The two docs files gave 9 passed and 2 failed at about 22:05 local: the pin, and the folder test, whose `OWNED_FOLDERS` still named `skills`. That tuple names `skill_library` since 22:07, and after this pass's edits the docs files give 10 passed and 1 failed, the pin.

- [x] `contracts.py`. Shared interface. Gate: `test_na_contracts.py`: 43 passed. Status: present; unchanged since 20:00:18 (its pin in section 8 still matches); `__all__` has 31 names.
- [x] `belief/core.py` and `belief/features.py`. The shared belief B_t, `SharedBelief`. Gate: `test_na_belief.py`: 21 passed. Status: present, with `SKILL.md`; the feature normalization is the belief module's decision (`belief/SKILL.md`; FORMULAS B5).
- [x] `skill_library/_common.py`. Head layout, shape and mask checks. Gate: `test_na_skills.py`: 48 passed (shared by all skill files). Status: present.
- [x] `skill_library/idle.py`. Idle skill. Gate: `test_na_skills.py`: 48 passed (shared). Status: present.
- [x] `skill_library/pursuit.py`. Pinned pursuit candidate v0001 behind its scoped loader. Gate: `test_na_skills.py`: 48 passed (shared). Status: present; the weights live in the gitignored `agent_skills/policies/pursuit_contact/`.
- [x] `skill_library/evader.py`. Scripted evader. Gate: `test_na_skills.py`: 48 passed (shared). Status: present; the frozen evader network is not on disk (module docstring).
- [x] `skill_library/library.py` and `skill_library/dispatcher.py`. Catalogue and dispatcher (ADR seam 1). Gate: `test_na_skills.py`: 48 passed (shared). Status: present; renamed from `skills`, so git no longer ignores them (section 1).
- [x] `jev/scripted.py`. J0 (FORMULAS D9). Gate: `test_na_jev_core.py`: 19 passed (shared with `runner.py`). Status: present; the rule is an implementation with constants the record does not set.
- [x] `jev/runner.py`. Binds a JEV to the core and applies the miss rule to HOST calls (FORMULAS D7). Gate: `test_na_jev_core.py`: 19 passed (shared). Status: present. The idle fallback on a forced switch (FORMULAS D12) and its 3 tests in `test_na_jev_runner_fallback.py` are not in the gate's count, so they came after it (inferred from the counts).
- [x] `jev/network.py`. J1 learned reactive network. Gate: `test_na_jev_network.py`: 24 passed. Status: present; no trained result is on disk.
- [x] `jev/train_j1_ppo.py`. PPO over goal decisions for J1. Gate: `test_na_jev_ppo.py`: 6 passed. Status: present; it trains a local stand-in MLP, not the network in `network.py` (TODO at `train_j1_ppo.py:796`; see `jev/SKILL.md`).
- [x] `jev/lm/`. J-LM (H15): `instance.py`, `prompting.py`, `protocol.py`, `anthropic_quick.py`, `ollama_qwen.py`. Gate: `test_na_jev_lm.py`: 65 passed, 1 skipped (the live Ollama test). Status: present, with its own `SKILL.md`.
- [x] `jev/kev.py`. KEV, defined by the owner on 2026-10-17 as an example instance, not a requirement: "a JEV clone which is just a quantized Qwen model fine-tuned to focus on quick decision making"; "you do not need to use kev, kev was just an example just like jev". Gate: none. Status: a documented stub; the un-finetuned baseline is the `j_lm` arm with `--lm-backend ollama` (`qwen3:4b`, 471 ms p50 per decision, `LEDGER.md`).
- [x] `executive/store.py`, `reports.py`, `scheduler.py`, `baselines.py`, `harness.py`. E_t store, report channel, scheduler hooks, deterministic baselines, call ledger. Gate: `test_na_executive.py`: 59 passed. Status: present.
- [x] `executive/anthropic_backend.py`. API adapter, tested through an injected client. Gate: `test_na_executive.py`: 59 passed (shared). Status: present.
- [ ] `executive/smoke_anthropic.py`. One real call, only when `ANTHROPIC_API_KEY` is set. It is not a test, and no live call has been made. Its no-key path exits 0 (checked 2026-10-16). Status: present.
- [x] `tasks/composition.py`, `tasks/events.py`, `tasks/flat_arena.py`. Composition task and arena. Gate: `test_na_tasks.py`: 30 passed. Status: present; covers the damage case only. The success rule was tightened after the gate run (below).
- [x] `eval/arms.py`, `eval/metrics.py`, `eval/run_composition.py`, `eval/traces.py`. Arms, metrics and trace writer. Gate: `test_na_eval.py`: 36 passed, 1 skipped (the refusal path). Status: present; `j1` and `j_lm` were wired into `eval/arms.py` after the gate run (section 5).
- [x] `eval/eager_loop.py`. Host-mode step loop. Gate: `test_na_eager_loop.py`: 3 passed. Status: present.
- [x] `session.py`. One continuous run of the stack. Gate: `test_na_session.py`: 7 passed. Status: present; its clock and host-decision rules are FORMULAS D10 and D11.
- [ ] `probes/`. Measurement scripts; not a build target. Gate: none. Status: present, with `SKILL.md` (2026-10-17); `probes/README.md` records CPU measurements from 2026-10-16.
- [ ] `docs/`. These documents. Gate: `test_na_docs.py`: 9 tests, 8 passed and 1 failed (the ADR pin, above); `test_na_docs_folders.py`: 2 passed (its `OWNED_FOLDERS` names `skill_library` since 22:07 local). The box is ticked once the pin is refreshed. Status: written.
- [ ] Upstream pieces this package depends on (ADR section 5, "used as-is"): `HierarchicalAgentCore` (`adk/architecture/hierarchy.py`), `manager_decision_due` (`adk/architecture/decision_context.py:415`), `wilson_lower` (`agent_skills/training_runs/recipe.py:85`). Gate: their own tests. Status: present; not re-run in the documentation passes, so the box stays empty.

**Composition smoke at 256 ticks** (part of the main session's gate run; `eval/out/gate256/`, written 2026-10-16 21:59 local; arms `flat`, `j0` and `shuffled`; one collector run of batch 4, seed 0). It is a smoke of the instrument, not a result about the hierarchy.

| Arm | Episodes completed | Catches / episode | Composed successes | Pursue-only | Skill changes / lane-run | Interrupt onsets |
|---|---|---|---|---|---|---|
| flat | 6 | 1.000 | 0/6 | 6 | 0.000 | 0 |
| j0 | 6 | 1.000 | 0/6 | 6 | 0.000 | 0 |
| shuffled | 5 | 0.800 | 3/5 under the loose rule (Wilson lower bound 0.231) | 1 | 4.000 | 0 |

- Flat and J0 caught once per episode and composed nothing. The flat pursuit arena produces no damage, so nothing triggers an evade, and no arm saw an interrupt onset.
- **The flaw the smoke exposed.** The shuffled-goal control scored 3/5 "successes" with no damage at all. The success rule then counted any evade entry before the catch, so a random permutation that selected the evader at any point before a catch passed. The 3/5 is an artifact of the rule. It is not evidence of composition, and the smoke gives no evidence for or against the hierarchy.
- **The fix, in progress.** `tasks/composition.py` `classify` (modified 22:06 local) now requires an evade entry that a damage tick triggered at most `EVADE_TRIGGER_WINDOW_TICKS = 40` ticks before it, and a resumed pursuit between that entry and the catch. A catch after any other evade entry is `composed_loose`, which is reported and never counted as a success. Since 22:20 local, `eval/run_composition.py` records both keys (`_score_run`) and takes `--arena`: `flat_pursuit`, the arena of this smoke, has catches and no damage, and `combat_baseline`, the new default, has damage and no catches. So under the strict rule neither arena composes without the damage injection, which is an ablation (ADR-001 D5). The smoke has not been re-run under the strict rule: `eval/out/` holds nothing newer than 21:59. FORMULAS B5 raises a unit question about the 40-tick window.

---

## 7. Conflicts and open items

Each item is a conflict or gap in the record, the code, or the documents. None is resolved here.

1. **KEV is undefined (resolved 2026-10-17).** The owner defined it as an example: a quantized Qwen fine-tuned for quick decisions, not required. `jev/kev.py` documents that and still raises on construction; the Ollama J-LM arm is the baseline of that instance.
2. **The skill folder name and `.gitignore:189` (resolved 2026-10-17).** The folder is now `skill_library/`, which git does not ignore (section 1). `DEV.md` (lines 13, 141 and 143) still uses the old name, `skills`, and is outside this pass. `test_na_docs_folders.py` names `skill_library` since 22:07 local.
3. **Skill training: one residual question.** The "What is trained here" paragraph of ADR-001 section 2 no longer lists the skill policies, and "Build ownership" says this build treats them as frozen inputs. D1 and H1 still describe training skill heads on the shared trunk, and the record does not say which system trains them (THESIS section 4).
4. **Non-negotiable count (resolved).** The version of ADR-001 read at 20:56:46 says in section 6.4 "the six non-negotiables (§2)", which matches section 2.
5. **Executive conditioning of π_high (resolved).** ADR section 4 now says the executive state E_t conditions π_high (FORMULAS C1).
6. **Belief width.** PLAN §17.2 gives a GRU hidden width of 384 or 512. The contract's belief latent is 64 wide (`BELIEF_LATENT_WIDTH`). The skill GRU is 128 wide (ADR D1; `SKILL_STATE_WIDTH`). The three are not reconciled (FORMULAS B2).
7. **Action factors versus heads.** PLAN §20.4 factors the action into (k, τ, p, d). The record does not map these onto the 12 published heads (FORMULAS C3).
8. **Two latency budgets.** `ComputeBudget.jev_latency_ms` (executive grant) and `JevBudget.latency_ms` (instance declaration) are not reconciled (section 3).
9. **The budget is enforced for HOST-mode instances only.** `jev/runner.py` times HOST calls. IN_GRAPH calls, including J0, are not timed (FORMULAS D7 and D9).
10. **Two percentile methods.** `jev/runner.py` uses linear interpolation and `executive/scheduler.py` uses nearest rank (FORMULAS D8, K2).
11. **The J0 rule is an implementation.** Its window (40 ticks), its commitments (2.0 and 0.5 seconds) and its emergency mask are not in the record. It covers damage only, and it does not respond to `PROJECTILE_THREAT_ENTERED` (FORMULAS D9).
12. **Two of the three P2 composition cases have no file.** Sequencing and dependency are not implemented on disk (section 1, `tasks/`).
13. **The shuffled arm is not the ADR's control.** It permutes the J0 arm's skill indices per decision, shared by all lanes (`ShuffledGoalManager.__call__` in `eval/arms.py`). The record does not define the scheme (section 5).
14. **Arm 3 wording.** DESIGN.md says "transformer manager". The ADR's learned controller is J1 (section 5).
15. **No guard on the Wilson bound.** `wilson_lower` raises for `n = 0` (FORMULAS J1).
16. **ADVISOR cutoff unverified.** The ADR says the released code adds a small-probability cutoff. The paper text as fetched has none. `advisor_losses.py` was not read (FORMULAS F5).
17. **Reference file created after the snapshot (resolved 2026-10-17).** `eval/metrics.py` refers to `eval/README.md`, which was not on disk at the snapshot (about 20:53). It exists now and is maintained with these documents.
18. **No `contract` on JEV instances.** A grep found none on the JEV instances (section 3). The ADK protocol check is not verified.
19. **README and DEV were out of date (resolved by 21:53 on 2026-10-16).** DEV.md's layout table now lists every folder as present, `probes/` included, and `README.md` now carries the lineage entries [L1] to [L14]. DEV.md still uses the old skill folder name (its line 13), and it says that `probes/` and `tests/experimental/neural_agent/` have no `SKILL.md`. Both have one since 2026-10-17.
20. **Stale pins.** `SKILL.md` and `README.md` in the package folder pin the ADR at `363a7196…`. `SKILL.md` pins `contracts.py` at `f8b1dbfe…`. `jev/lm/SKILL.md` pins the current values (`4284c60d…`, `313ad604…`). These files are outside `docs/`, so this document did not edit them. The current values are in section 8.
21. **Action-mask leak (negative, 2026-10-16).** The record's three-world leakage test fails its negative control: a hidden live enemy changes the ability legality, the ability action mask and `defense_f32` (ADR section 3 D5). The gym change awaits the owner's decision. The test file is git-ignored (`HytaleRL/.gitignore:53`), so the verifier should not expect it to be tracked. Re-run on 2026-10-17 (UTC): 6 passed and 2 failed, the same result (THESIS section 5, H12).
22. **The record changed during this work.** ADR-001 was modified at 19:58:30, 20:56:46 and 20:59:34. The documents quote the text as read at those times. The hash in section 8 is the version used for the bibliography and the thesis text.
23. **Test files are git-ignored.** `.gitignore:105` (`**/tests/`) matches every file under `tests/`, including `test_na_contracts.py` and `test_na_docs.py`. Git does not track them, so the gates in section 6 exist only in the working tree (`git check-ignore -v`, 2026-10-16).
24. **Mask legality is checked only on concrete calls.** `skill_library/_common.py` (`require_concrete_true`) skips the check for traced arrays. Under `jax.jit` the dispatcher reports legality only in its `legal` diagnostic (`skill_library/dispatcher.py:184-206`). Idle and evader then emit their first preference on a lane whose neutral choice is masked (`skill_library/idle.py:98-111`, `skill_library/evader.py:174-192`). The pursuit skill is tested under `jit` (`test_na_skills.py`).
25. **Skill-local state resets on a change of skill, not on each goal activation.** `skill_library/dispatcher.py:163-168` resets when the skill index changes. ADR-001 D6 gives the lifetime as "one goal activation". A new goal that names the same skill keeps the old state, because `adk/architecture/hierarchy.py:413` carries the actor state through with no goal-switch reset.
26. **The PRNG key has no lane axis.** The ADK core takes one key, splits it into three (`adk/architecture/hierarchy.py:260`), and passes the manager and the actor their own keys. Per-lane outputs are checked for a leading lane axis (`_validated_decision` and `_require_lane_leaves` in `jev/runner.py`). The ADR text states no lane-axis rule of its own.
27. **Two tick units (read in the code; the runtime effect is not tested).** The belief counts ticks of 1/30 s (`belief/core.py:373`), and the session clock counts environment steps of 0.045 s (`session.py`). J0's 40-tick window is then about 29.6 steps. The strict success rule's `EVADE_TRIGGER_WINDOW_TICKS = 40`, which `tasks/composition.py` calls "the same window J0 uses", is compared with recorded step indices (`run_composition._score_run`), so it is 40 steps (FORMULAS B5).
28. **The loose success rule (found 2026-10-17).** The 256-tick smoke scored the shuffled control 3/5 with no damage, because any evade entry before a catch counted (section 6). `tasks/composition.py` now holds the strict rule, and `eval/run_composition.py` passes the two keys it needs (22:20 local). Neither arena composes without the damage injection (section 6).
29. **A `None` decision counts as a miss.** The `HostDecider` docstring allows `None` to keep the current goal, and `run_session` counts it as a miss (FORMULAS D11; read in the code, not tested).

---

## 8. Provenance

SHA-256 values and sizes taken on 2026-10-16 at the snapshot. A reader with a copy of a local-only file can check it against these values.

| File | SHA-256 | Size and modification time | Used for |
|---|---|---|---|
| `agents/design/ADR-001-three-level-agent.md` | `204c10a7040cec531d4f3e84ef32eff6428021dee1b84458d99c285d34fb3793` | 83,845 bytes, 2026-10-16 20:59:34 | the decision record (local-only); earlier versions `f5f0fdb9…` (19:58:30) and `6b34c6fe…` (20:56:46) are superseded |
| `agents/design/PLAN.md` | `db6413dbd556a714c4193762242c8aabe5723246331f94cf35b62a17bc32a77e` | 114,137 bytes, 2026-08-02 16:46:18 | PLAN v0.2 (local-only) |
| `experimental/hierarchical_planner/DESIGN.md` | `09ee95ef7f81b6205a94c42089195bbf746f18842f02ead6271e804bccf3247d` | not recorded | the four-arm protocol (section 6) |
| `agent_skills/training_runs/recipe.py` | `ad96a4a0b0d8a7cc5d49875ac2cd0afdd154314282fba695480706aad557a0f6` | not recorded | `wilson_lower`, advance rule |
| `HytaleRL/hytalegym/hytalegym/jax/training/ppo.py` | `1825f825d2a309107bb921f429111f9e644a0a24bffbe38c83b53906611186e9` | not recorded | PPO and GAE forms |
| `HytaleRL/hytalegym/hytalegym/jax/training/types.py` | `dad43b7094bca873dc1c2ecd85538f21c2e7181308b0c2adec65b11b3bb1d60a` | not recorded | `PPOConfig` defaults |
| `adk/architecture/hierarchy.py` | `2ca882c5738fd3d925b8170c54ba00205e893c5c69db1bf34aa31ae04464d594` | not recorded | orchestration rules D2 to D4 |
| `adk/architecture/decision_context.py` | `189f5fbad5d7c74ff05b0359f138cf3a1b8a809a03e862c543ded7fdadd8cfa0` | not recorded | `manager_decision_due` (D1) |
| `experimental/neural_agent/contracts.py` | `b0fc994d9f55506ffee29507a5bdfe8b06036b52c96428198679a729bb90dd13` | 44,249 bytes, 2026-10-16 20:00:18 | the shared contract |

Stale pins: see section 7, item 20.

## 2026-10-17 (late): the intended primary instance is our own System One model

"JEV" is TypeSafe AI's Jev, a System One model (typed calibrated decisions in one parallel pass). The owner wants that class of model built here, not called: J-S1 reads the rendered executive state, belief and skill catalogue through a small pretrained backbone and emits skill, replace, params and confidence in one pass, trained supervised, then calibrated, then on outcomes through the semi-MDP harness. J1 keeps the mechanism role, J-LM the knowledge comparison. Full design and references in `SYSTEM_ONE.md`; `jev/system_one.py` does not exist yet.
