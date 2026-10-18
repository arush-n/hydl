# Current state of experimental/neural_agent (2026-10-18, evening)

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
| The patterned language for the LLM's words (a direction in the prompt and the schema, not a grammar) and the pattern reader (skill by catalogue id or alias; the clause by the cue reader); P7 1.000 at no breakage, 0.9925 with every perturbation | built 2026-10-18 (`executive/patterned.py`, `jev/system_one/pattern_reader.py`, SYSTEM_ONE 9.14) |
| Questions from the decision layer to the LLM (`DecisionSection.information_requests`, shown in the LLM view) | built `59d2047`; the LLM's answering loop is not exercised (no live call) |
| The LLM's view as a briefing (per-NPC rows in words, goals with how they were read, questions, rejections, the delta since the LLM's version; markdown / json / compact; 8 lanes in 943 estimated tokens) and `DecisionSection.readings` | built 2026-10-18 (`shared_state/views/`, `shared_state/README.md`) |
| Squads and objectives in the contract (`Squad`, `Objective`, a goal's `addressee` and `objective`), lowered per lane; the briefing shows squads, an objectives table with outcomes rolled up, and renders one squad on request (`ViewOptions(squad=...)`) | built 2026-10-18 (SYSTEM_ONE 9.15; task #31) |
| The Claude adapter replies in the executive section schema (typed fields plus the words), the harness writes the words to the notepad, and the user turn carries the briefing when the notepad is shared | built 2026-10-18 (fake client only; no live call by rule) |

## 3. Level 2, the decision layer (`jev/`, `requests.py`, `session.py`)

| Item | Status |
|---|---|
| J0 scripted rule (baseline) | measured: 43/81 = 0.531 on seed 0 at 512 ticks (Wilson lower 0.423) |
| J1 learned reactive network: behaviour cloning on real J0 traces, online comparison, PPO over goal decisions | measured 2026-10-17; superseded as the primary instance |
| J-S1, our own System One model: latency probe (G2), structured catalogue-agnostic encoder (G3), online composition (G4) | measured: G1 to G3 pass; G4 rate pass on seeds 0, 1, 2 (0.500 / 0.488 / 0.458 against lower bounds 0.423 / 0.465 / 0.337) once it read the contracts' commitments; the 0-miss part waits for an unloaded machine |
| Reading the LLM's words: frozen text in slots (G3-NL), frozen and fine-tuned small encoders as condition readers (P2), description matching for skills (P3) | failed (0.53 to 0.58; 0.71 to 0.82 with polarity lost; 0.42); verdict `SYSTEM_ONE.md` 9.11 to 9.13: typed fields are the contract |
| Exact number reader | measured 1.000 |
| Deterministic fallback condition reader (`condition_rules.py`) | measured: 1.000 kind and argument on held-out wordings, seeds 0 to 2 (caveat: the cue tables know the generator's vocabulary) |
| Decoder LM readers once per version (P6: Qwen3 0.6B, 1.7B, 4B, 8B, 14B, schema-constrained, zero-shot) | measured 2026-10-18: kind 0.92 / 0.90 / 0.97 / 1.00 / 1.00 on held-out wordings (bar 0.95), number 0.98 to 1.00, skill by description 0.36 to 0.88 (bar 0.90, fails at every size); a restate-then-emit second pass lowers kind everywhere; the 8B through local Ollama is the words-only fallback when a decoder is wanted (528 ms per goal on the GPU); LoRA plan recorded, not run |
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
| 3b Renderer anchors fixed (the shim shadowed the package; the contract was right), dagger damage from the 0.7.2 rows (15/15 live), native NPC jump first step exact (arc still off: gravity timing F1c, owner decision), production line of sight (3/3), slope limits recorded as a no-op, content fixes | measured 2026-10-18 (`GEN7-PLAN.md` 2b, FIDELITY-0.7.2.md section h) |
| 4 Combat and mana constructs | blocked on bridge fixtures and readouts (an eligible role for hit interrupt, a displacement harness, a mana readout): phase 4b |
| 5 to 6 Crafting and source-pin baselines, stamp and cutover | designed (`GEN7-PLAN.md` section 4) |
| Open for the owner | the planned compile and the general compile are not bit-identical (6 of 315 arrays, max 2.4e-7): moving callers to planned is a methodology decision; one opt-in bit-exact speed-up exists (`OPTIMIZED`), four hot spots proposed |

## 6. Running right now

P6 (8B, 14B, restate variant); gen-7 phase 3 agent; the lower-layer session's on_tick commit and
forced-target-loss test.

## 7. Decisions taken and still open (2026-10-18, evening)

Resolved today under the owner's delegation ("manage the remaining commits the best way you think"):
- The long-uncommitted work reviewed on 2026-10-18 (86 files, six groups) is committed in dependency order:
  engine edits that change no rollout (`6bdcd20`), tooling and docs (`cb3d08c`), the bridge's world-verb transport
  v5 / facade v4 with the native Python side (`64c8044`), the publication-hygiene entry (`35a5a2d`), the v6 policy
  contract across jvm-agent, adk/deploy and the export tools (`c0d5f9b`), and the 0.7.2 bridge plugin project
  (`cdf258c`). `peft` is installed for later LoRA work; compile-mode differences below the differential bars are
  acceptable (`sim/BASELINES.md` 5); the lower layer has the go-ahead on the body-turn label and the terrain
  observation; the 2026-10-17 cleanup is done (`b58a317`); `tests/sim` is versioned and the sim tables ship as
  package data (`ad220cb`).

Still uncommitted, by design:
- **The value-changing engine edits** (jax/combat `agent.py`, `step.py`, `types.py`, `events.py`, `motion.py`,
  `encoder.py`, `item_programs.py`; worldgen `source_pins.py`, `recertification.py`, `region/__init__.py`; the arena
  curriculum). They change the gen-5 fixed-rollout fingerprints (the review measured: HEAD `0b28448a` / `ace7a94e`,
  working tree `d8477b63` / `b02ee274`, attributable to `agent.py` + `step.py` + `types.py`), and they include the
  native NPC jump fix that gen 7 adopts through its overlay. Every composition number measured this week ran on the
  working tree, so committing them makes HEAD match what was measured. Plan: commit them as one unit with
  `sim/gen5/identity.json` re-recorded and the combat stamps republished, once the phase 4 port agent has reported,
  because that agent renders from HEAD and its overlay patches anchor on the current literals.
- **The arena session's files** (console/**, arena/training/**, worldgen/region/*, jax/training/multi_actor/*,
  HytaleRL/examples/*, worlds/region.py, root .gitignore; dated 2026-10-13 to 17). Asked which are theirs; no answer
  yet. Not touched.
- **In-flight files of live agents**: `sim/gen7/*` (phase 4), `skill_agent/*` (the lower layer), `lm_reader.py` (P6).

Open for the owner: pushing (nothing is pushed); the reviewer's one flagged question in the v6 contract
(`PublicServerEvidenceSource.java` self row now uses head yaw and pitch, unexplained); whether to re-extract
`sim/gen5/data` from a 0.5.7 archive or relabel it 0.5.10.

## 8. Commits of 2026-10-17 late to 2026-10-18 (this session)

`0cf8c05` seam default; `1b76c0e` typed condition and the fallback reader; `782a523` B5 parser switch; `59d2047` the
exact typed path and the questions channel; `ffe11c5` `on_tick`; `8707eec` `--host-budget-ms`; `e09c496` the renderer
(phase 2a); `97539bf` folder documentation and the package map; `ad220cb` packaging; `30c6136` port phase 3;
`83fe6eb` the scratchpad package; `b7d71c2` the briefing view and goal readings; `20c2c0d` the patterned language
(P7); `b58a317` cleanup; `6bdcd20`, `cb3d08c`, `64c8044`, `35a5a2d`, `c0d5f9b`, `cdf258c` the old uncommitted work
in groups. Lower layer: `e62f654`, `3439c67`, `0f17bc8`, `39b450e`, `9b6c571`, `cbd4e00`, `5286797`.

## 9. Layout

The package map is the top-level `README.md`; every folder has a `README.md`; the formulas and citations
that used to sit in the README are `ORCHESTRATION.md`. The scratchpad became a package (`shared_state/`:
sections, document, views, writers) on 2026-10-18 with its import path unchanged; no other module moved.
