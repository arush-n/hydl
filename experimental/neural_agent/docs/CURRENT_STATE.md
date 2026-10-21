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
| Joint check of executor and combiner at 512 ticks | measured 2026-10-18: identical at a 200 ms host budget (0 non-timing differences, 0 misses both, 24,680 running statuses through the per-tick hook); at the default 20 ms the joint run diverged through 3 latency misses, not logic; the rejected-request path is exercised since `9b6c571`: `test_sa_joint.py` forces three entity_unknown rejections, a question to the LLM at the third, then withdrawal; the 20 ms run is identical on `9b6c571` and on `5c523a8` (`7734dcf`; run 1 had one load-flake miss at load 13.5, run 2 alone was identical with 0 misses on both sides) |
| An intent the library cannot serve (#32): the nearest catalogue id serves with a question, descriptions only suggest, the harness leaves an unresolvable goal out of the version instead of refusing it (`jev/system_one/nearest_skill.py`, `SYSTEM_ONE.md` 9.16) | built 2026-10-18; gate: 4 of 7 absent families served from the expected group on all phrasings, 3 asked about; description matching measured as noise |
| Learned heads beside the exact path (replace, params, confidence; P1d relabelling), calibration (G5) | designed / deferred until real skills |

## 4. Level 1, the skills (`skill_agent/`, `skill_library/`)

| Item | Status |
|---|---|
| Stand-ins (pursuit candidate v0001, scripted evader, idle) and the dispatcher | built; the composition task runs on them |
| Lower layer phase A (the other session): 12-head surface (+9 for 0.7.2, six scopes), catalogue parser (139 rows, 19 activities), protocol, combiner with stub bodies, parity, library, body contract, 32 NPC traces at 1/30 s | built, 763 tests at hand-off |
| Phase B: behaviour-cloning dataset, freezing (shared encoder plus per-skill adapter; frozen skills bit-identical when a skill is added), joint wiring with the executor, progress names and stop reasons on every contract, bounded done-id window | built `baff062`, `0f17bc8`, `e62f654`, `3439c67`, `39b450e`, `14db188`, `9b6c571`, `cbd4e00` (body-yaw label, dataset v2), `5286797` (terrain, meters and visibility, dataset v3), `19c642f` (sim_adapter), `e07de22` (an already-expired request is rejected once), `63d85dd` (bench_bodies); package gate 867 passed, 2 skipped, 1 xfailed on the 2026-10-18 tree (baseline 863 plus 4 new) |
| Phase C: real skill competence tests (E2 to E6 of `skill_agent/docs/BUILD.md`) | blocked on three things: the gen-7 locomotion phase (walk resume after landing; the Region library recaptured on 0.7.2), the owner's training go-ahead, and the owner's motion-model choice (the player model has no live oracle; FIDELITY-0.7.2 (e), P2 item 1). Measured (`63d85dd`, ledger): as built, trained bodies cost about 1 ms per agent per skill at batch 8; the combiner calls each body per agent and freezing's verb program reruns the trunk on every call, so p95 is 12.5 ms at 3 skills per agent and 24.5 / 44.9 ms at 6 (shared / distinct verbs), over the 15 ms bar; one trunk pass per tick plus per-verb adapter programs keeps every case within the bar (worst 10.0 ms), a body-contract decision still open |
| Asks to the owner from the lower layer | the body-turn label and the terrain observation are done (`cbd4e00`, `5286797`); remaining: push, review of the catalogue edits, the uncommitted pursuit fix in `arena/training/runs/pursuit_run.py`, the training go-ahead, the motion-model choice, the one-trunk-pass body-contract change |

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

## 6. Running right now (2026-10-18, night)

The simulator lane is this session's since the fidelity session's handoff (GEN7-PLAN 2b, `5c523a8`), and the work runs
as a fleet of Haiku subagents (the owner's ruling: launch Haiku agents, pass the model explicitly, and distil every long
process into runbooks and tools). Committed since: the port runbook and tools (`deb8694`: the version bump checklist at
the top of PORTING.md, BRIDGE-PORTING.md, regen_swap_plan.py with its test, pin_bridge.py, fidelity_index.py and the
generated index in FIDELITY-0.7.2.md); the 13 content values as Redirects (`ec9c6da`, section (w); the shortbow tier-0
row matches live, the iron sword hit tick stays open); the shield bash under a held guard (`fc80707`, pin `56D2B532`,
section (x); the plain-root bash with no guard held is queued as #71); potions and durability (`6b98ee3`, section (y):
the 0.7.2 stamina drink is a 30 s interaction cooldown with no status, the durability table served through a hook).
Since then: the damage-fix jar verified against the previous jar across the native suites, 49 failures on both and no
regression (`7d3c2d5`, (u) addendum); the landing walk resume construct (`580e37b`, section (z): after a jump the gen-7
NPC walks on the third tick after touchdown, within 1.5e-5 m of native on two seeds, closing the locomotion half of the
lower layer's Phase C blocker; guard cooldown, attack start and shortbow reset ammo re-filed as #73 with findings); the
ancient-steel crossbow rows settled as a harness pitch gap and two stale 0.5.7 expectations (`02ff098`, section (aa);
follow-ups #74); the bridge half of the plain-root bash withdrawal (`bdb6245`, pin `BF1D9697`, (x) addendum; the
simulator half waits); bench tiers and the memories level on the simulator with the oracle receipts replayed and the
harness readers (`a92d0a2`, section (ab): the render admits 1,264 recipes, 1,168 at a fresh bench and 1,264 at tier 8
and level 6); the bridge session file split for headroom (`0619b1a`: 9,691 to 8,439 lines, 36 methods moved verbatim
into ten helpers, the full suite unchanged at 421, live smoke, reload and hold probes green, pin `460054A8`). The stale engine-test fixtures
(#62, gitignored test trees, nothing to commit) went 41 to 19 failures: 22 fixtures re-stated in the current
`LearnerArsenalAction` shape; of the 19 left, 5 need `HYTALE_ASSETS_ZIP`, 12 are artefacts keyed to the old bridge
pin `5A54A8F5` (recordings, world-evidence history, traversal and token-contract manifests, the stability constant) to
republish once the pin settles, 2 are expectation questions (#76; receipts under the session `test_fixtures/`,
pitfalls in `sim/PORTING.md` at `03090b0`). Timed bench crafts on the simulator (`ec1f968`, section (ac): construct
`timed_crafting`, switch `crafting_time`; seven CombatState leaves; inputs out on R+1, output on R+90 for 3.0 s, queue one
deep, 7-block reach cancel with refund; all seven stage-3 receipts replay; the render admits 1,485 recipes, 1,446 commit
alone, 1,305 at a fresh bench, all 211 timed commits finish; quick gate 197, identity 4; open: the executor path inside a live
Region step, the processing and diagram actions, the fieldcraft `Weapon_Axe_Crude`). The eleven source-pin audits on 0.7.2 (`d069a3e`, section (ad), phase 5): 0 of 11 pass before and after, now failing closed on
recorded 0.7.2 rows (absent-safe digests; pins keyed by the audited jar's release line in `worldgen/source_pin_baselines.py`
and `source_pin_baselines_072.json`; the 0.5 line byte-identical on the 0.5.10 control); each audit's finding names the
simulator work it implies; the terrain recapture stays gated on all eleven; next is claim re-derivation per audit, the
inventory row first. Landed together in `a21bdc7` (two lanes that shared the overlay files): stage 5 on the simulator (section (ae): processing
benches, progress and fuel at one recipe second per 30 ticks, charcoal on the second fuel item; the Armory click-craft committing
two ticks after the request; all six receipts replay in order, fuel and seconds; the render admits 1,945 recipes) and the three
timing constructs (section (af): guard re-entry 0.35 s from the claim, the attack-start split one drain after the claim, the
shortbow reset ammo 0; native under load reopened one tick earlier in 17 of 24 runs, recorded as jitter with the idle-machine
re-measurement filed as #78); coordinator quick gate 292, identity 4. Landed on the evening of 2026-10-18: `2c311c9`, the joint construct commit (the crossbow
scalar variant, section (ag); the crafting executors run inside a live Region step for the first time and five defects fixed, among
them that none of the three wrappers set `state_commit`, (ah); the shield bash as a held-guard fork, offered on ticks 3 to 12 of a hold as
live 0.7.2 does, (ai); the head-at-reset measurement, (ak); quick gate 314 passed on the final tree, identity gate 4 passed on HEAD
2c311c9 in 16:02 under load); `41981ca`, the Region recapture on 0.7.2 (500 of 500 sections; the exact-digest compare fails on 5 to 6
non-damaging fluid cells while the dry projection is identical, so the 0.5.7 contract methodology applies and passes with a cap of 6;
the terrain differs from the 0.5.7 capture in 46 of 250 sections; `tools/worldgen/recapture_region.py` and its README; section (aj));
`0161a16`, the source-pin audits second pass (1 of 11 verified, inventory with a recorded caveat; 10 fail closed with their gaps named;
`tools/source_pins/rebaseline.py` and README; the gate now passes `HYTALE_ASSETS_ZIP` to the audits; section (al)); `9722621`, the
design record and thirteen probes for the unimplemented switches (`sim/gen7/mechanics/DESIGNS-072.md`). Running: the head-heading
construct lane (#75, under the overlay lock: both heads seeded from `entity_yaw` on the gen-7 render, a no-op when none is given).
Waiting for a quiet machine: the stub measurement lanes in the DESIGNS-072.md 14a order (#69) and the idle timing re-measurement (#78).
The owner's: the library swap and the product rebuild after the recapture (`tools/worldgen/README.md` steps 1 to 5), the 43-row
expectation in `test_region_runtime_provider` (#76: committed code gives 42, the arena session's uncommitted edit 32), the phase-6
training ruling.

Landed on the night of 2026-10-18 to 2026-10-19 (coordinator-verified, none pushed): `893d40b`, the head-heading construct (#75,
section (ak), row 5g: an Arsenal reset given `entity_yaw` seeds both heads from the stored body yaw on the gen-7 render and
rebuilds the observation with the rendered `observe_batch`; quick gate 324, identity 4, fingerprints unchanged); `0756aca` and
`62253eb`, the classification of the failing source-pin audits (section (al) addendum and correction: A 1, explosion_dynamics
accepted as pin-only so 2 of 11 carry a semantic baseline; B 6, perception, entity_geometry, locomotion, doors, mutable_blocks,
explosion_mutation; C 3, navigation, actor_world_state, crafting; the genuine 0.7.2 changes are block health per section with the
`BlockOperations.setBlock` write path, entity scale made physical, the `MovementStates` extra byte and the content censuses, while
the `DeathComponent` exclusion, the 25 `InteractionType` constants and `WorldChunk.breakBlock` were already false on the 0.5.10
control; every claim's verdict in `source_pin_baselines_072.json` and `tools/source_pins/evidence_072/`, the claim-level pin
design in the runbook); `941d6bd`, the Region library loader made version-aware (section (aj) addendum: the chunk API certified
per server version, the `inherited_seed_labels_v1` split method, the staged 0.7.2 library `9dca3c71…` loads and the provider
selects it natively when traversal is not required; the task zoo follows the manifest's method; the swap README's step 3 names the
repository generators). Fleet state: the Haiku weekly limit was reached at 00:13 CDT on 2026-10-19 (resets 2026-10-23, 20:00
Chicago); five lanes stopped mid-work (the loader lane after its edits and gates, four claim lanes mid-analysis), the collector
lane finished at 00:31, and no Haiku lane can run until the reset, so the remaining work (the C leaves of the audits, the 0.7.2
traversal library, the inherited-labels writer, the stub measurements #69, the idle re-measurement #78) waits on the reset or on
an owner ruling to use another model. Still the owner's: the publish gate (re-derive the audits or publish the 0.7.2 library with
the audit state recorded), the arena-side zoo and horizon rules, the phase-6 stamp (#79) and default flip (#80).

Update after the limit lifted (2026-10-19, 00:38 to 01:15 CDT; `2117ab0`): the owner reported the Haiku limit reset, the four
audit claim lanes and the collector were resumed from their transcripts and finished, so all ten failing audits are classified
claim by claim: A 1 (explosion_dynamics), B 7 (perception, entity_geometry, locomotion, doors, mutable_blocks, explosion_mutation,
crafting), C 2 (navigation, actor_world_state, both on repo-side or runtime-only claims with nothing false). Genuine 0.7.2
changes: block health per `BlockHealthSection` cleared on a changing write, entity scale physical, the LOS detail-hitbox test
(already modelled by `line_of_sight`), the bench gate's augment tags (unmodelled; task #82), the `MovementStates` byte (not
reaching the bridge), 28 interaction types against 25 (coordinator-confirmed with javap; the earlier 26 against 29 counted the
`$VALUES` array), and the content censuses. Gen-5 test artifacts (false on the 0.5.10 control too): the `DeathComponent`
exclusion, the sprint fallback 1.65, `WorldChunk.breakBlock`, the bridge `canCraft` output-room omission, the recipe id split.
Bridge follow-up filed as #81 (no-heightmap columns flagged valid). Running: the loader lane's writer for
`inherited_seed_labels_v1`, and the idle timing re-measurement #78, which waits for the peer session's distill run to end (load
10.6 at 00:42).
Committed earlier: the
crafting oracle stage 3 (`274aaec`, timed bench crafts with probe events; pin `5FE609E2`), phase 4j (`a534e83`: the Trork
target no longer gets a headless Player, which the PvP-off filter used to cancel the agent's damage; the iron combo and
signature rows now land; pin `CAFAD9B7`), and the crafting oracle stage 5 (`7374b76`: processing benches advance once the
probe runs the world and the game clock unpaused, Campfire 61.9 game seconds and Furnace about 301; the Armory click-craft;
pin `4A5E5D6D`). Every crafting stage now has its native shape (CRAFTING-ORACLE.md answers; FIDELITY sections (s) to (v)). Committed since the last update: phase 4f diagnosis and 4g (`b4be1fe`, `6fd60a2`, `aedd85e`:
interaction legality from a side-effect-free replay of the engine's admission, the crossbow interrupt offered at the
Repeat window, pin `A4487CF3`; the peer re-measured guard and dodge, no row moved); phase 4h (`cb01a31`, swap plan
`25ad61f`, identity gate 4 passed in 6:41: the accepted root's child from the pre-admission row and the reload's instant
stat-modifier status omitted, the official interrupt test passing with the `Ability4` binding); the crafting oracle
stages 1, 2 and 4 (`76839c5`, pin `A6C310DD`: bench tier and memories fixtures and readouts, structural craft, reset-time
probes in an oracle-only scope, all ten contract scenarios matching; the contract itself is `CRAFTING-ORACLE.md`,
`84f1a83`). Phase 4e is committed (`6d914c8`, pin `B56FD38E`; swap plan `6c30ba9`;
identity gate 4 passed in 6:37 against `6c30ba9`): the actor evidence row carries `head_yaw_degrees` and the decoder
maps `agent_head_yaw` from it, so the native self row is the head frame the policy trained on; the live learner-row
look case passes on both seeds on the new pin (the peer's run, `b2acf2f`). Phase 4d is committed (`9f12f0d`, pin
`32EF5EDD` then): the bridge accepts `Ability4` by name and the 0.7.2 crossbow reload runs live; the peer's content
binding is in (`69422d7`), and the interrupt case (Primary not offered during the reload) stays open. The peer has
also landed F1c (`e8a1fc7`), the 5f dodge timing (`ff51431`, `4661d5c`) and crafting stage 0 (`6aaf21d`, 197 of 197
offered recipes commit on the render) as gen-7 constructs, with fidelity section (o) (`0e45fc6`); tests/sim 114; the
learner rows pass 6 of 14 on the final render against 4 on stock. Phase 4c is committed (`4066417`): the headless entity gets an `AbilitySlots`
component, so the reload interrupt no longer stops the world (pin `8720BC76`); the group yaw finding was withdrawn as a
bridge fault (the camera lever steers the head by design, on 0.5 as on 0.7.2; the group tests need `body_delta_yaw`);
the slot-4 `Ability3` versus `Ability4` mapping is an owner decision (`FIDELITY-0.7.2.md` (k)). The peer session (hydl-1f) owns the fidelity tests under `tests/fidelity` and, since `2b88082`,
the overlay rows 5f and 11 and the learner-row harness repair, each as its own construct with a native differential,
both sim gates before any overlay commit. P6 is concluded; the phase 3, 4 and 4b port agents have reported and their
work is committed.

Later on 2026-10-19 (`fc2d32c`, whose message says "current state updated" though this paragraph landed one commit later): the
writer for the inherited labels landed (`7b4ca63`: `create_region_artifact_library(inherit_split_from=...)` regenerates the staged
0.7.2 manifest byte for byte; no publish step calls it yet), and #78 finished (section (am)): the guard 0.35 s and the bank 1.0 s
stand in the short-tick regime the simulator's 30 Hz step assumes, and the native reopen's dependence on the engine tick period is
recorded as a known deviation. No lane is running. The swap waits on the publish gate (recertification's audit gate stops at
perception; re-word the B claims and accept, or publish with the audit state recorded).

Afternoon of 2026-10-19 (the Haiku limit lifted; the four lanes resumed from their transcripts): landed the per-line claim text
mechanism (`7a260d2`: a 0.7 baseline record carries `claim_texts` keyed by the audit's claim constant, the 0.5 text and hashes
never move), host contracts keyed per server version (#84, `8be0e7c`: the ExplosionUtils and DamageSystems digests and the
mutable-block evidence and cells labels select a 0.5.7 or 0.7.2 entry by the profile; the 0.5.10 control is byte-identical;
`STALE_CLAIM_TEXT_072` is enforced by `pins_for`), and recertification first-capture mode (`02540a4`: per-line fixture roots,
0.7.2 under `fixtures/world/v0_7/`; compare or first_capture decided from four references, a partial set refused; the staged
seed-570057 contract checked against its NPZ and the expected digests `9DCA3C71`/`408D5FB2`; the capacity report produced for
the line; two stationary captures that must agree; the cascade derived last). Running: the collector's acceptance of the eleven
0.7.2 pins (round one accepted nine with their re-worded claim text; round two takes explosion_dynamics, explosion_mutation and
inventory after the #84 digests), and the hit-interrupt probe (#69, early receipts under analysis). Next, in order: the live
first-capture run on 0.7.2 once the gate passes all eleven (launcher on 5556/25574, about 30 minutes; pre-checks pass: the three
references absent, lease free, jar `460054A8`), then the 0.7.2 products and the suites' frozen v7, the pointer flips, and the
stamp #79 and default flip #80 which wait for the owner's training. Nothing is pushed.

Evening of 2026-10-19: all eleven World source audits are accepted on 0.7.2 (`f74968e`; the verifier keys the audit subprocesses
by the audited JAR, `455c7a2`; a per-line pin helper `expected_source_pin`, `2d1e993`). The first live first-capture run on
0.7.2 passed the gate, the seed-570057 projection (digest `408D5FB2` reproduced, 5 fluid-only cells against a cap of 6), the
capacity report, two byte-identical stationary captures and the native graph build (20 graphs), then stopped at the first
world-action probe: the gitignored live compare scripts under `tests/fidelity` gated on the audits' 0.5.7 literal match
instead of the per-line semantic pin. A repair lane routed the twelve scripts through the per-line pin, and the probes now record
the bridge's own server version; the run is being repeated (run 2). Landed beside it: the promotion tool
`tools/worldgen/promote_world_evidence.py` (`895224a`; the manual copy made checkable, keyed by line, refusing overwrites,
writing the receipt) and the diagnosis of the Arsenal Region v6 suite drift (`cf906ee`: the engine batch `879321e` moved the
0.5 bridge pin the arsenal contract embeds; the frozen v6 value is not reproducible from any on-disk input; the 0.7.2 suite v7
will be per line, task #88). Hit interrupt (#69): under the learner-row harness the opponent carries no Eligible until late, and
any observed actor's unknown 0.7.2 status fails the gen-5 observation closed (failure bits 128 and 4, empty ability head), so
the construct must declare the eight Hit_Interrupt_* statuses before a 0.7.2 checkpoint can run live against such NPCs; the
native-AI duel measured the Eligible refresh (about 0.27 s) but its club hits are unflagged, so the eligible-hit rows rest on
asset values unless a flagged NPC attack exists (last probe round). Running: run 2 of the first capture, the pointer keying
lane (#89), the probe's last round. Next: promotion with the tool, then products (#90), suites v7 (#88), pointers (#89).

Night of 2026-10-19: the 0.7.2 World evidence is captured and promoted. Live run 4 of the first capture passed end to end
(after run 2 exposed the pointer keying's effect on the world-token identity, fixed by exporting the staged traversal manifest,
`b9a1d75`, and the tokens module's override and intermediate state, `7ab2139`; and run 3 exposed the block-use probe's unbound
typed requests, now bound from the capture with the bridge's closures recorded, task #91). `promote_world_evidence.py`
promoted it (receipt `460054A8/publication-v12.json`; Region `9DCA3C71`, traversal `CC613B12`, world token `9A216202`), and
the 0.7.2 line recertifies in compare mode. Certification status at 20:20: audits accepted, evidence promoted, pointers keyed,
the Arsenal Region suite v7 frozen per line (`896e31c`, 0.7.2 suite `51BFCC28`; v1 to v6 byte-identical), the task zoo and the
twenty path-distance tables built (`5ba2010`), and the gitignored recorded-evidence tests routed by line (0.7: worldgen
recordings 11 passed and 1 skipped, fidelity replays 5 of 6; the remaining 0.7 failures are 0.5-only chains, five owner
calls listed in the lane's report under `recorded_tests_072/`; the 0.5 line unchanged test by test). The stamp (#79) and the
default flip (#80) wait for the owner's training and a gen-7 checkpoint that passes live. The peer session's seven perception asks are accepted
as a gen-7 extension channel (#92, after certification). Nothing is pushed.

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
- **In-flight files of live agents**: `sim/gen7/*` is this session's (the content-values agent, #64); `skill_agent/*` and `LEDGER.md` have nothing in flight (the lower layer's lane is committed through `63d85dd`).

Follow-ups for whoever owns `tests/fidelity/combat` (from the peer session's section (i) run, 2026-10-18): six
crossbow tests look up the head `world_move_none_plus_compass`, which `ARSENAL_POLICY_ACTION_HEAD_NAMES` has at
no revision, so they raise before measuring; two static placement tests pin the 0.5.7 `Assets.zip`; the
reload-interrupt test gets "World thread is not accepting tasks" from the bridge. Harness problems, not mechanics.

Ruled by the owner on 2026-10-18 (late), one line each ("do whatever has fidelity / your choice / whatever matches
fidelity / whatever is best"), and in progress:
- Slot 4 drives the 0.7.2 crossbow reload under `Ability4`: the bridge accepts and reports it by name (phase 4d agent,
  task #50; the 25-index wire and evidence numbering stay 0.5's); the gen-7 content binding follows in the peer's lane.
- Group facing tests use the body lever (the peer's two-lever and turn tests); no bridge change.
- F1c is a gen-7 construct, built by the peer (`e8a1fc7`): the server's fall-speed update with the previous tick's
  duration and cubic drag; passes jump in place, jump held, jump held with forward and ledge walk live on both seeds.
- The deferred gen-5 engine edits are committed (`ede9a39`; `879321e` added the three worldgen/region files the batch
  imports, after `ede9a39` failed to import). `sim/gen5/identity.json` is re-recorded from a git archive of `879321e`
  (`baacdc9`: batch 8 `d8477b63`, batch 64 `b02ee274`, identical in two runs and equal to the working-tree values of the
  week). The step-clock line anchors moved with the edits, so the swap plan was regenerated from HEAD and the render
  test re-pinned (next commit after `9f12f0d`); gates on that tree: tests/sim green, identity gate 4 passed in 28:17.
  The `npc_jump` construct is a by-value no-op now that gen 5 computes the native jump formula for `npc_walk`. The
  engine's own suite (tests/jax, untracked) ran 86 passed, 1 skipped and one bit-exact geometry test failing by one
  float32 ulp; that test fails identically on an archive of HEAD before the batch, so it predates it (recorded in the
  `ede9a39` message). The hold on `sim/` is released with this commit.
- The head-yaw self row flagged in `c0d5f9b` is correct: the simulator's self row is the head frame (`_self_features`
  reads `agent_head_yaw`); commented in both agent copies (`e4888df`), no behaviour change. The 0.7.2 policy agent copy
  of 2026-10-16 is committed (`31d08be`, 131 files, validation in its PORT-0.7.2.md).
- The fixture-only attitude override of `1159f8f` stays (gated to `hit_interrupt_duel`).
Repo hygiene, left as found: both bridge plugins' Java tests (111 files for 0.7.2) are untracked because
`HytaleRL/.gitignore` ignores `**/test/`.

Open for the owner: pushing (nothing is pushed; the standing rule stands until asked); the reviewer's one flagged question in the v6 contract
(`PublicServerEvidenceSource.java` self row now uses head yaw and pitch, unexplained); whether to re-extract
`sim/gen5/data` from a 0.5.7 archive or relabel it 0.5.10.

## 8. Commits of 2026-10-17 late to 2026-10-18 (this session)

`0cf8c05` seam default; `1b76c0e` typed condition and the fallback reader; `782a523` B5 parser switch; `59d2047` the
exact typed path and the questions channel; `ffe11c5` `on_tick`; `8707eec` `--host-budget-ms`; `e09c496` the renderer
(phase 2a); `97539bf` folder documentation and the package map; `ad220cb` packaging; `30c6136` port phase 3;
`83fe6eb` the scratchpad package; `b7d71c2` the briefing view and goal readings; `20c2c0d` the patterned language
(P7); `b58a317` cleanup; `6bdcd20`, `cb3d08c`, `64c8044`, `35a5a2d`, `c0d5f9b`, `cdf258c` the old uncommitted work
in groups. Lower layer: `e62f654`, `3439c67`, `0f17bc8`, `39b450e`, `9b6c571`, `cbd4e00`, `5286797`.
Night: `1f0af14` identity gate against the repaired HEAD; `ebc2e41` P6; `3cb4416` the peer's fidelity section (i);
`1159f8f` the phase 4b bridge half (readouts, `hit_interrupt_duel`, jar pin `8B09661F`); `235565b` the nearest-skill
resolver (#32, SYSTEM_ONE 9.16); `2b88082` the gen-7 mana pool construct with fidelity section (j) and phase 4c recorded;
`7f56001` this document; `4066417` phase 4c (AbilitySlots fix, pin `8720BC76`, fidelity section (k)); `37fdef4` this
document; `e4888df` the head-frame comment; `31d08be` the 0.7.2 policy agent copy; `d18bf2c` this document;
`ede9a39` and `879321e` the engine batch; `baacdc9` the identity re-record; `9f12f0d` phase 4d (Ability4, pin
`32EF5EDD`, fidelity section (m)); `5cb738c` the swap-plan regeneration; `9ad7c5f` this document; `6d914c8` phase 4e
(head yaw on the wire, pin `B56FD38E`, section (n)); `6c30ba9` its swap-plan regeneration; `b2acf2f` the live
learner-row pass. Peer: `490b1d4` the 5d guard window construct, `4434a48` fidelity section (l), `69422d7` the
crossbow `Ability4` binding, `e8a1fc7` F1c, `ff51431` and `4661d5c` the 5f dodge, `6aaf21d` crafting stage 0, `486388e`
stage 1, `0e45fc6` section (o). Later this session: `8758cb3` docs; `b4be1fe` section (p); `6fd60a2` phase 4g; `aedd85e`
(q) recheck; `84f1a83` CRAFTING-ORACLE.md; `cb01a31` phase 4h; `25ad61f` swap plan; `76839c5` crafting oracle stages 1, 2, 4; `9750283` (s) note closed; `b4be1fe`,
`274aaec` stage 3; `5c523a8` the handoff; `a534e83` phase 4j; `7374b76` stage 5; `dc517df` docs; `deb8694` runbooks and
tools; `8b6e284` the peer's state corrections; `ec9c6da` content values; `fc80707` shield bash; `6b98ee3` potions and
durability; `4034a10` docs; `7d3c2d5` the no-regression verification; `580e37b` landing walk resume; `02ff098` and
`f9e79ba` phase 4n and the index tool; `b461500` docs; `bdb6245` the bash withdrawal (bridge); `a92d0a2` tiers and
memories; `c69d9fd` docs; `0619b1a` the session split. Peer, before stopping: `0218d69` guard
release mask, `025ee62` structural crafting (1,168 recipes), `325f938` type 3 on the 0.7 line, `8b3b2ac` last.

## 9. Layout

The package map is the top-level `README.md`; every folder has a `README.md`; the formulas and citations
that used to sit in the README are `ORCHESTRATION.md`. The scratchpad became a package (`shared_state/`:
sections, document, views, writers) on 2026-10-18 with its import path unchanged; no other module moved.

Status at 23:45 on 2026-10-19 (owner: do not wait for training; continue the other work). The 0.7.2 bridge pins the 0.7.2 host
contracts (b682e33: actor evidence `E93BBEBE`, NPC observation `3D080DE9`; jar pin `6EB44F30`); recertification run 8 passed in
compare mode against it and was promoted with supersede (receipt `world-evidence-history/6EB44F30/publication-v12.json`); the
v7 suite line was re-frozen for the new bridge (`493FD07E`, 8a873fa). The combat arena's 120-degree sector reaches the row gate
(2591899; opt-in keyword on the open-flat fixture, gen 5 bit-identical by default). Lanes launched for the items that need no
training: the combat program's evade gate (#94), the bridge augment-tag readout (#93), the world-action residuals (#91) and the
content identity in the contract stamp (#79). The hit-interrupt probes (#69) wait for a quiet machine. #80 (the default flip)
still needs a gen-7 checkpoint that passes live. Nothing is pushed.

Status at 02:10 on 2026-10-20. The 0.7.2 line is certified modulo training under the bridge `1CB4F651` (the host-contract pins,
the fixture aim, the augment readout and the request-tick fix): recertification run 10 passed on a quiet machine with the seed
projection exactly equal (the cross-reset fluid drift is load-driven: 3, 5, 6, 7 cells under load, 0 quiet) and was promoted with
supersede (receipt `world-evidence-history/1CB4F651/publication-v12.json`); the v7 suite line is `703ED690`; the Combat contract
stamp v2 (generation and content identity) is published for 0.7.2 at `combat-contract-stamp-e5046b63`. The hit-interrupt Part 1
measurement is in the fidelity record (eligibility at tick 42 to 43; an eligible hit adds the 0.7 s stun, 0.5 s flinch and 4 s
reaction A) and its construct is being implemented (#102). For the skill layer: the combat arena's sector reaches the row gate
(2591899), combat_cover (9046a78), the remembered target position in requests with perception_v2 self_position (f808d3f), the
gated combat evader (adopted on the referee's numbers), the bow recipe (#97), the duel arena seam for the league (bc69536,
#101; its health-fraction rescale in progress). Lanes running: combat styles (#100), the guerrilla verbs and program (#99), the
hit-interrupt construct (#102). Filed for the owner, not applied: the fluid-cell cap (an owner knob; runs now wait for a quiet
machine), the legal mask's missing inventory term and the silent geometry freeze (engine design gaps found by the bow lane), the
place feature behind the contract gate (#91), cover at range through the Region geometry route, #80 (the default flip) after a
gen-7 checkpoint that passes live. Nothing is pushed.

Status at 16:31 on 2026-10-20. Decision layer: the combat styles beside P_COMBAT (`0b22cda`; COUNTER, PRESSURE and HIT_AND_RUN
registered under their NA_PROGRAM names, AMBUSH and ZONING as designs; the referee variants wait for the skill layer's charged
holds, since its sword ignores the hold scalar today); the guerrilla verbs hide, shortbow and avoid with their contracts, the
`Report` condition that reads the bodies' five reports from the newest live status with the belief as fallback, and P_GUERRILLA
(`21a7dc3`; avoid while faced or hurt, shortbow from 3.5 to 4.5 blocks, hide while seen, the pursuit step as the search through the
remembered position); the shortbow contract pins `hold_seconds` as the draw time mapped to the authored tiers 0.1, 0.3, 0.6, 0.9,
1.2 s with a hold below the first as tier 0 (`a827072`, confirmed on both lines by the sim's adapter), and the guerrilla hold is
provisional at 0.2 s because gen 7 authors 1, 2, 4, 7, 19 damage per tier (#105: the tier follows the skill layer's per-tier
measurement). The duel for the league: entity 1's observation on its own maxima (`1efe339`), `max_health` (`5871dee`),
`roles=(entity 0 role, entity 1 role)` over the 14 paired CombatParams fields and the statuses (`f2f5957`), and the built-in
target motion off (`fb78247`: it was gated by `target_active`, not the controller mask, and moved entity 1 from its activation
tick). The seat bias the skill layer found (entity 0 wins most decisive gen-5 fights) is diagnosed (#103, two lanes, ledger):
masks and the ruleset attack table ruled out; the one-tick first-hit lag is selector geometry (eye offset with the defender's
box; same-role pairs land together); what remains is the engine's two-actor path, filed as #108 for the owner (only slot 0 takes
the force-pushed knockback branch; the motion kernel runs every row through the agent slot), so the league plays both seats. The
gen-7 render moves both duel bodies alike (#104 measured: no agent-only construct fires in a flat duel; the shared start split
delays both chains one tick and the sword knockback patch stops entity 1 being lifted); its follow-ups are #107 (the landing hold
never arms on knockback landings; the duel's mana reads 0 against the construct's 100 at spawn; target-slot twins for the jump
constructs). Simulator: the hit-interrupt construct part A (`61b8288`: the measured logic and the asset table, 14 tests, hooks
unwired) with part B running in a lane (the eligible flag per entity from `entity_role_ids`, the hit row identity through a
call-site patch under the overlay lock, the flinch verified on the target's motion, the construct's statuses pre-compensated by
half a tick); the stock status clock's 16 ticks for a 0.5 s status (native 15) filed as #106. The skill layer landed L2
(`2b8351a`, the pure-JAX styles under the programs' names, constants pinned to the programs) and marks league numbers
seat-confounded until #108 is decided; its shortbow body comes with the per-tier measurement. Remaining items on this side after
part B: #80 (the default flip, a gen-7 checkpoint that passes live), the fluid-cell cap (owner knob), the legal mask's missing
inventory term and the silent geometry freeze, the bound place behind the contract gate (#91 residual), cover at range through
the Region geometry route, the duel's residual role-scaled normalisers (a target-role gait maximum in the actor-first view, in
another session's dirty file), the hit-interrupt Part 2 probe, the 0.5.7 line items (v6 suite repair, the path-table 255 rule,
the built 0.5 bridge against its pin, the stamp tests on the default line), the other #69 stubs (runes, hookshot, climbing,
gliding, riding, swimming), grants beyond none for the augment differential, the 0.5-only replays on 0.7, and #105 to #108. No
training on either side.


Status at 21:30 on 2026-10-20 (coordinator). Decision layer: interrupts may carry companion requests on disjoint scopes
(`Interrupt.beside`, `5bd3fc1`); GUERRILLA is the referee's pick (`732ed9a`: the clean shot at 0.6 above avoid, keep_range
(6, 10) beside it; the old program never shot a chaser), with the shot window 6 to 10 and the 1.2 s full draw (`102a8cf`);
avoid takes the entity or the remembered position (`2c290fa`); the combat programs strike and punish on the body's
weapon-aware `in_reach` report with the old 3.5-block rule as the belief fallback (`a45bcfc`, bit-identical without reports).
Known limit #114: a goal with a companion ends by its maximum duration, not COMPLETED. Arenas and runner: combat_cover takes
`arrows=` and run_composition `--loadout` / `--arrows` (`dfdceb7`); a death needs the agent's health at 0, with
`terminated_alive` for world-error ends, and combat_cover's front pillars no longer box the NPC (`cd11af1`). Not answered by
any arena yet: whether a bow kites a same-speed melee chaser (combat_cover's 9x9 window ends a retreat past about 4 blocks and
has no NPC navigation provider; on open ground the chaser follows at 3.4 to 3.9 blocks); this needs cover at range (the
Region route). The duel: entity 1's attack facing synced every step (`b5ee120`; the sword mirror's edge for entity 0 fell from
+21.2 to +11.0, equal to the engine patch lane by lane). The seat gaps that remain are engine-only, mapped in #108 for the
owner: the agent-only control lock (measured: the sword lean to entity 1 after the sync, −9.4 → −2.5 with it off), agent-only
regeneration (small), slot 0's body yaw held while knocked back (#115: most of the mace lean, −7.8 → about −2 when slot 0
may turn, but the sword control then moves toward seat 0; native turning under knockback is unknown and needs a live check
first), and entity 1's arrows leaving along its body at level pitch (shoot from entity 0). Simulator: the hit-interrupt
construct is wired on the gen-7 render (`5e68c47`, IMPLEMENTED True; tests/sim 387 passed, the identity gate green, the gen-5
fingerprints unchanged). Package gate at the last change: 1222 passed, 2 skipped, 1 xfailed. Owner items: #108 (and its native
turning check), #80, cover at range, the fluid-cell cap; plus the list in the 16:31 block above.
