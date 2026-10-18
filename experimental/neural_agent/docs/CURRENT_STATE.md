# Current state of experimental/neural_agent (2026-10-18, morning)

A dated snapshot. Numbers and hashes are from the ledger and the commit log; nothing here is inferred unless
it says so. Status words: **measured** (a gate ran, the number is in `LEDGER.md`), **built** (code and tests,
no decision-quality claim), **designed** (written, not built), **failed** (measured below its bar and
recorded), **running** (a background agent or run is in progress).

## 1. The architecture, as it stands

The loop the owner asked for exists end to end on the gen-5 simulator with stand-in skills: the executive
writes goals (typed fields plus words) to the scratchpad; the decision layer reads them once per version,
evaluates typed conditions exactly on the belief every host cadence tick, applies the priority rule, runs
behaviour programs, emits typed `ActionRequest`s into the scratchpad; the lower layer's combiner reads them,
rejects invalid ones with a reason, executes with stub bodies and streams `RequestStatus`es (accepted,
running, done with a stop signal, aborted, rejected) back; the decision layer's correspondence rules react;
unsure readings become questions to the LLM in the decision section. What is **not** yet in the loop: a live
LLM (by rule, not before the lower level is finished), real neural skills (phase C of the lower layer waits
for the gen-7 simulator), and the gen-7 simulator itself (phase 3 of 6 running).

## 2. Level 3, the executive seam (`executive/`, `shared_state.py`)

| Item | Status |
|---|---|
| E_t store, schema, reports, scheduler, harness, Claude adapter behind an injected client | built; 60+ tests; no live call |
| Scratchpad S_t with one writer per section, versioned, closed schema, LLM view | built (`00a8fc7`, grown since) |
| Typed goal `condition` (nine kinds, enum in the schema), the vocabulary and units in the system prompt | built `1b76c0e` |
| Questions from the decision layer to the LLM (`DecisionSection.information_requests`, shown in the LLM view) | built `59d2047`; the LLM's answering loop is not exercised (no live call) |
| The LLM's view as a briefing (per-NPC rows in words, goals with how they were read, questions, rejections, the delta since the LLM's version; markdown / json / compact; 8 lanes in 943 estimated tokens) and `DecisionSection.readings` | built 2026-10-18 (`shared_state/views/`, `shared_state/README.md`) |
| The backend writing the full executive section (words included), objectives list, per-squad views, the group scratchpad | designed (`MIDDLE_LAYER.md` plan step 7; task #31) |

## 3. Level 2, the decision layer (`jev/`, `requests.py`, `session.py`)

| Item | Status |
|---|---|
| J0 scripted rule (baseline) | measured: 43/81 = 0.531 on seed 0 at 512 ticks (Wilson lower 0.423) |
| J1 learned reactive network: behaviour cloning on real J0 traces, online comparison, PPO over goal decisions | measured 2026-10-17; superseded as the primary instance |
| J-S1, our own System One model: latency probe (G2), structured catalogue-agnostic encoder (G3), online composition (G4) | measured: G1 to G3 pass; G4 rate pass on seeds 0, 1, 2 (0.500 / 0.488 / 0.458 against lower bounds 0.423 / 0.465 / 0.337) once it read the contracts' commitments; the 0-miss part waits for an unloaded machine |
| Reading the LLM's words: frozen text in slots (G3-NL), frozen and fine-tuned small encoders as condition readers (P2), description matching for skills (P3) | failed (0.53 to 0.58; 0.71 to 0.82 with polarity lost; 0.42); verdict `SYSTEM_ONE.md` 9.11 to 9.13: typed fields are the contract |
| Exact number reader | measured 1.000 |
| Deterministic fallback condition reader (`condition_rules.py`) | measured: 1.000 kind and argument on held-out wordings, seeds 0 to 2 (caveat: the cue tables know the generator's vocabulary) |
| Decoder LM readers once per version (P6: Qwen3 0.6B, 1.7B, 4B, 8B, 14B, schema-constrained) | running; interim: 4B passes kind 0.975 and number 0.997, fails skill-by-description 0.769; 0.6B / 1.7B fail kind (polarity); 8B / 14B and the restate variant in progress; LoRA blocked (no `peft` installed) |
| Behaviour programs (executor with exits, interrupts, resumption; the host switch rule; cadence gate) | measured P4: 0.512 vs 0.531 (p = 0.875); under the commitment seam 0.474 vs lower bound 0.423, timeouts 1 (were 48 to 81 per lane before the clock fix) |
| The exact typed path `jtyped` (typed condition, legacy label, or words through the rule reader; the priority rule) | measured `59d2047`: equals the teacher on every decision over 300 generator states; online 43/82 = 0.524 against j0 43/81 = 0.531, 0 misses in 1,033 decisions, interrupts within 1 tick |
| Request stream (sections, statuses with `signal`, validation with 8 rejection codes, `any_of` schemas, correspondence rules) | built and measured R1 to R6 with the lower layer (`5ce0456`, `e62f654`, `3439c67`) |
| Host seam honouring the skills' commitments | default since `0cf8c05` |
| Per-tick observer `on_tick` for the combiner; `--host-budget-ms` on the driver | built `ffe11c5`, `8707eec` |
| Joint check of executor and combiner at 512 ticks | measured 2026-10-18: identical at a 200 ms host budget (0 non-timing differences, 0 misses both, 24,680 running statuses through the per-tick hook); at the default 20 ms the joint run diverged through 3 latency misses, not logic; the rejected-request path is still unexercised (0 rejections in 8,088 grants; the lower layer is writing a forced-target-loss test) |
| Learned heads beside the exact path (replace, params, confidence; P1d relabelling), calibration (G5), absent-family evaluation (#32) | designed / deferred until real skills |

## 4. Level 1, the skills (`skill_agent/`, `skill_library/`)

| Item | Status |
|---|---|
| Stand-ins (pursuit candidate v0001, scripted evader, idle) and the dispatcher | built; the composition task runs on them |
| Lower layer phase A (the other session): 12-head surface (+9 for 0.7.2, six scopes), catalogue parser (139 rows, 19 activities), protocol, combiner with stub bodies, parity, library, body contract, 32 NPC traces at 1/30 s | built, 763 tests at hand-off |
| Phase B: behaviour-cloning dataset, freezing (shared encoder plus per-skill adapter; frozen skills bit-identical when a skill is added), joint wiring with the executor, progress names and stop reasons on every contract, bounded done-id window | built `baff062`, `0f17bc8`, `e62f654`, `3439c67`, `39b450e`; 827 tests |
| Phase C: real skill competence tests (E2 to E6 of `skill_agent/BUILD.md`) | blocked on the gen-7 simulator's locomotion phase |
| Asks to the owner from the lower layer | a body-turn label rule (built like the head-turn one); the training observation is too thin (own state plus opponent, no terrain); push, review of the catalogue edits, the uncommitted pursuit fix |

## 5. The simulator port to Hytale 0.7.2 (`HytaleRL/hytalegym/hytalegym/sim/`)

| Phase | Status |
|---|---|
| 1 Content layer: extractors on both versions, typed tables, adapters, the swap plan, `PORTING.md` | built `93418ae` |
| 2a Renderer: overlay spec, `render.py` from the pinned commit, bench; identity gate on gen 5 | measured `e09c496`: bit-identical (2,307 arrays, 0 mismatched), negative control caught, identical HLO, existing suite unchanged; gen-7 skeleton 24 rows applied, 90 pending, 1/30 s clock; **no 0.7.2 fidelity claimed** |
| 3 Locomotion and line-of-sight constructs with their differentials, locomotion audit | measured 2026-10-18: 16 locomotion rows active (value-identical to gen 5), the detail-hitbox line-of-sight construct built; motion on a ledge floor 1.14e-05 m against the 5e-4 bar PASS, geometry compare 8/8 PASS, LOS on full cubes 3/3 MATCH, all against a live 0.7.2 bridge; identity gate on gen 5 still bit-identical. Open: slope/wall limits have no reader in the 0.7.2 server (set to no-op), NPC jump force (gen 7 adopts the native 5.099), partial-block LOS and slope fixtures, production perception LOS hook, Region recapture (not started) |
| 4 to 6 Combat and mana constructs, crafting and source-pin baselines, stamp and cutover | designed (`GEN7-PLAN.md` section 4) |
| Open for the owner | the planned compile and the general compile are not bit-identical (6 of 315 arrays, max 2.4e-7): moving callers to planned is a methodology decision; one opt-in bit-exact speed-up exists (`OPTIMIZED`), four hot spots proposed |

## 6. Running right now

P6 (8B, 14B, restate variant); gen-7 phase 3 agent; the lower-layer session's on_tick commit and
forced-target-loss test.

## 7. Decisions waiting on the owner

1. Push the local commits (nothing is pushed).
2. Install `peft` in the shared venv so P6's LoRA step can run, or skip LoRA.
3. Planned versus general compile for the simulator callers (numbers in `sim/BASELINES.md` section 5).
4. The cleanup deletions proposed on 2026-10-17 (`jev/lm/anthropic_quick.py`, `tasks/events.py`, the sibling-refusal path, unused constants, the synthetic PPO environment).
5. `HytaleRL/.gitignore` exception for `tests/sim/`; `pyproject` package data for `sim/*/data`; the content sha in the contract stamp (uncommitted `publication.py`); the 53 uncommitted `HytaleRL/` files held by other sessions.
6. The lower layer's asks: body-turn label rule; a richer training observation (terrain).
7. Review of the lower layer's private catalogue edits and the uncommitted pursuit fix.

## 8. Commits of 2026-10-17 late to 2026-10-18 (this session)

`0cf8c05` seam default; `1b76c0e` typed condition and the fallback reader; `782a523` B5 parser switch;
`59d2047` the exact typed path and the questions channel; `ffe11c5` `on_tick`; `8707eec` `--host-budget-ms`;
`e09c496` the renderer (phase 2a). Lower layer: `e62f654`, `3439c67`, `0f17bc8`, `39b450e`.

## 9. Layout

The package map is the top-level `README.md`; every folder has a `README.md`; the formulas and citations
that used to sit in the README are `ORCHESTRATION.md`. The scratchpad became a package (`shared_state/`:
sections, document, views, writers) on 2026-10-18 with its import path unchanged; no other module moved.
