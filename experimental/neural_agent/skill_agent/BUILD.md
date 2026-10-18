# The skill agent: build plan, proofs and the gen-7 verification (2026-10-17, night)

This document is the brief for the agent who builds the lower layer of the three-level agent: the
neural skills that execute what the middle layer asks for, on the gen-7 (Hytale 0.7.2) simulator.
It says where the code goes, what it is made of, what already exists and must be reused, what the
gen-7 environment must be verified for, and, before anything is called final, which experiment
proves each component. Read it whole before writing code. Labels: **RULED** is the owner's word,
**MEASURED** a number in a ledger, **BUILT** committed code, **PLANNED** this document's proposal.

The companion documents: `experimental/neural_agent/docs/MIDDLE_LAYER.md` (the middle layer and the
contract this layer speaks), `agent_skills/training_runs/SKILL_CATALOGUE.md` (the 139 skills with the
game's numbers; private folder, never stage it), `agent_skills/training_runs/SKILL_LIBRARY.md`
(requirements for every skill and for the next architecture), `GENERAL_AGENT.md` (design rules),
`TRAINING_NEEDS.md`, `HytaleRL/hytale-plugin-0.7.2/SIMULATOR-0.7.2.md` (the gen-7 simulator plan)
and `TRAINING-0.7.2.md` (the measured 0.7.2 gameplay changes).

## 0. The rulings this build follows

- RULED (2026-10-17, late): the agent operates at 30 Hz, the game's tick (1/30 s); every "45 ms" or
  "0.045 s" figure below is the gen-5 simulator's and is superseded: holds of 0.1 to 3 s are 3 to 90 ticks,
  the host cadence of 0.5 s is 15 ticks, and the per-tick path must finish well inside 33.3 ms with margin
  (p95 <= 15 ms at the deployed batch) so the agent paces itself to the game clock instead of falling behind.
- RULED: training of the neural systems is paused because the current architecture (one PPO MLP with
  a GRU per skill over an 8,272-wide vector) is too weak to train the catalogue; the environment
  exists; everything else is finalized first, then the skills are trained, then the middle layer is
  fine-tuned.
- RULED: "we need to start proving that everything works before we commit to that component being a
  part of the final architecture" (section 6 is that list).
- RULED: the agent trains in the JAX environment; JAX or PyTorch, "whatever works better", as long as
  it runs inside that environment (experiment E11 decides).
- RULED: gen-7 (0.7.2) is the target; gen-5 is deprecated; the skill lineage is retrained on gen-7,
  not carried over (`SIMULATOR-0.7.2.md`, P6).
- RULED: no human traces for now; demonstrations come from native NPC traces only.
- RULED: the neural layer "follows directions clearly but does not think much for itself"; its only
  thinking is low-level (combat, pathfinding). Requests are structured: coordinates, directions,
  scalars. The layer receives a stream of requests, rejects the invalid ones, keeps acting as best it
  can, and streams status back; conflicts are resolved greedily by urgency or confidence; independent
  discrete actions combine (locomotion and attacks); as many concurrent requests as plausible.
- RULED: commitment and duration belong to the skill's contract (BUILT in `contracts.SkillContract`).
- RULED: nothing external except the top LLM; the final architecture must not cheat (legal observation
  only; the privileged simulator mode is opt-in for proof-of-concept runs only).
- RULED: the catalogue is the skill list, and more skills will be added later.
- RULED (2026-10-18, late): the trunk runs once per tick for every agent, not inside every body call ("sure").
  BUILT in `freezing.encode_contexts` / `VerbBody`; the body signature is unchanged
  (`bench_bodies.py`, README 2.7).
- RULED (2026-10-18, late): skills train on the base NPC model, the body the bridge drives ("the base npc model
  is best (npc walking should just be an animation of it)"): the simulator's `npc_walk` motion model, with
  walking as one of its movement states and not a separate model. The `player` model is not a training target.
- RULED (2026-10-18, late): training starts only from a clear plan the owner has approved ("depends, you need
  to have a clear plan"): `TRAINING_PLAN.md`. The dataset smoke (`bc_smoke`, seconds) may be used where it
  helps ("your choice if it helps").

## 1. Where it is built

A new package, `experimental/neural_agent/skill_agent/`, beside the harness it speaks to. The
proof-of-concept skills in `skill_library/` stay frozen until the new bodies match them (E2, E3), then
`skill_library/` is retired. Private material (run configs, checkpoints, suites, evidence) stays in
`agent_skills/training_runs/`, as today; nothing under `agent_skills/` is ever staged.

```
experimental/neural_agent/skill_agent/
  BUILD.md              this document
  SKILL.md              folder notes for agents (dated entries; gitignored by policy)
  surface.py            the gen-7 observation token types and the action-head layout, grouped into
                        scopes; derived from the catalogue; the single layout both the simulator
                        work and the bodies build against (E1)
  catalogue.py          parser of SKILL_CATALOGUE.md rows -> SkillContract (typed argument schema,
                        head-group scope, progress signals), description text (for the middle layer's
                        matching), and activity programs; the catalogue is the source of truth
  references.py         the resolver: a request's world point, entity handle or cell+face -> the
                        agent-frame vector, the entity token index, the marked block candidate
  requests.py           ActionRequest and the status stream records (the middle layer's §9.10
                        contract), validation against a contract
  combiner.py           per tick and per agent: validate, greedy grant by scope, run the granted
                        bodies, merge heads, stream status (R1-R6)
  bodies/
    encoder.py          the set encoder over tokens (entities with roles, block candidates with
                        faces, inventory slots, geometry, self, held-item kit), target marking
    body.py             one goal-conditioned policy per head group: verb + typed args in, its heads
                        out, recurrent memory reset on start, status and progress heads, scoped
                        sampling, per-verb adapters on a frozen trunk
    locomotion.py, gaze.py, hands.py, posture.py, interaction.py, abilities.py
                        the six bodies' verb tables and head scopes (thin; the logic is body.py)
  commands/             exact scripts for the 33 command skills, same request/status envelope
  training/
    recipe.py           the one recipe that must improve every verb: imitation warm start from NPC
                        traces, then PPO with potential-based progress shaping, mixtures over worlds
                        x openings x opponents, selection on fresh openings, report on unseen axes
    curriculum.py       prerequisite chains inside a body, old verbs kept in every batch
    traces.py           native NPC trace ingestion (NpcTraceFrame) -> behaviour-cloning datasets
    cards.py            capability cards per verb: success and time-to-success per environment
                        family, preconditions, failure modes, calibration
  eval/
    suites.py           per-verb gates (Wilson lower bounds, FOV16/32 with occlusion), combination
                        gates, interruption gates, the fidelity gate hook (section 4)
tests/experimental/neural_agent/test_sa_*.py   (gitignored by policy, like the rest)
```

Rules for the folder: import cross-package names only from `experimental.neural_agent.contracts`,
`shared_state`, the ADK and the gym; a dated entry in `SKILL.md` for every discovery; every gate run
recorded in `experimental/neural_agent/docs/LEDGER.md` (append-only) before it is claimed.

## 2. What exists and must be reused

| Resource | Where | Use here |
|---|---|---|
| The skill catalogue, 139 rows with the game's numbers, kinds L / C / A, instruction -> done per row | `agent_skills/training_runs/SKILL_CATALOGUE.md` | the source of contracts, descriptions, activities, physics inputs and acceptance ("done") |
| Requirements for every skill (instruction, context, perception, action surface, report, rules) and the 16 requirements for the next architecture | `SKILL_LIBRARY.md` §2, §7 | the acceptance list for section 5; E1-E13 map onto them |
| Design rules (one I/O contract, goals as relations, outcome contracts, no privileged inputs, generalize by construction, interruptible and composable, capability cards, measured horizons) and the build order | `GENERAL_AGENT.md` | unchanged; this document adds the bodies and the request stream |
| Training needs, pipeline, reward, curriculum, "collectors and mechanics", the gen-7 note | `TRAINING_NEEDS.md` | the recipe's constraints |
| The gen-7 simulator plan (P0-P6, gates, risks) and the measured 0.7.2 gameplay changes | `HytaleRL/hytale-plugin-0.7.2/SIMULATOR-0.7.2.md`, `TRAINING-0.7.2.md` | section 4 |
| The proof-of-concept skills: idle, pursuit (pinned checkpoint, 31/41 catches on validation), evader; contracts with scopes; the dispatcher (one skill per lane); descriptions | `experimental/neural_agent/skill_library/` | the parity baseline for E2 and E3; retired after |
| The harness: contracts (GoalPayload, SkillContract with control fields, control_table), the shared scratchpad with views, the session loop, the composition task and the four-arm evaluation driver | `experimental/neural_agent/contracts.py`, `shared_state.py`, `session.py`, `tasks/`, `eval/` | the request stream plugs in here (§9.10 of SYSTEM_ONE.md) |
| ADK: the belief / manager / actor hierarchy, goal cadence, commitment and interruption rules, action scopes, deployment bundles | `adk/architecture/hierarchy.py`, `adk/deploy/` | the in-graph loop the bodies run in |
| The trainer: PPO, token encoder and candidate encoder, multi-actor, world actions, checkpoints with contract stamps, readiness | `HytaleRL/hytalegym/hytalegym/jax/training/` | the set encoder starts from `token_encoder.py`; the recipe reuses `ppo.py` |
| Arena: curriculum strategy and ladder, `FactoredActionScope`, stage binding, pursuit run with milestones, imitation (behaviour cloning, native trace projection, recurrent replay, checked transfer) | `arena/curriculum/`, `arena/training/skills/`, `arena/training/imitation/` | the imitation warm start and the scopes |
| Navigation: A* and the follower | `HytaleRL/hytalegym/hytalegym/jax/world/navigation/` | demonstrations for walk_to / go_around; route hints as context |
| Region library, path-distance tables, task suites, task zoo, catch-time tool, recipe bench | `agent_skills/training_runs/tools/` | suites and horizons on gen-7 Regions once recaptured |
| The 0.7.2 bridge and the native NPC trace recorder (NpcTraceFrame: tick, dt, state, ...) | `HytaleRL/hytale-plugin-0.7.2/`, `hytale-plugin/src/.../imitation/` | the only demonstration source for now |
| Fidelity: 11 source-pin audits, native differentials (motion, combat), crafting compare | `HytaleRL/hytalegym/tests/fidelity/` | section 4's gates |
| The recipe catalogue, 2,047 recipes | captured by `surrogate_crafting_native_compare.py` | the crafting and cooking commands |
| The legal observation switch (opt-in) | `ArsenalRuntimeConfig.legal_observation` | on for every result that counts |

## 3. The architecture to build

### 3.1 Surfaces (what a body sees and does)

**Observation as token sets, legal only.** Entity tokens with a role (enemy, ally, animal, merchant,
player) and relative position and velocity; block-candidate tokens with faces; inventory-slot tokens;
geometry and light tokens; a self token with health, stamina, mana, signature energy, oxygen,
durability, grounded, yaw and pitch; held-item kit tokens (combo windows, hold thresholds, charged
tiers, costs) straight from the catalogue; a route-hint token set (next waypoints). The instructed
target is marked in the set, which is how one network points at any entity or cell. Field of view and
the 0.7.2 line-of-sight rule apply; the privileged mode is never on for a result that counts.

**Action heads for 0.7.2, grouped into six scopes.** Today's twelve heads (base action, ability,
guard, jump, locomotion with four dodges, camera and body yaw, pitch, hotbar, use, trigger, block
candidate) extend with: a duration head ("hold for k ticks", because holds run 0.1 to 3 s = 2 to 67
ticks at 45 ms, and a per-tick press/release policy is jittery and inhuman), rune selection and cast
through the ability slots with mana as a resource, hookshot aim and trigger, dodge reduced to left and
right, item move / drop / give, recipe choice, doors, riding and gliding controls. The six scopes:
**locomotion** (move, jump, dodge, slide, climb, swim, glide, ride), **gaze** (yaw, pitch, look
targets), **hands** (trigger, duration, hotbar, item moves, block candidate for use), **posture and
defence** (crouch, guard, dodge timing), **interaction** (use, doors, menus, recipe), **abilities and
magic** (ability slot, cast, rune set). The combiner grants one request per scope at a time.

### 3.2 Bodies (the networks)

Six goal-conditioned policies, one per scope, not 86 networks. Each takes the verb and its typed
arguments (from the request, resolved by `references.py`) and the token sets, and emits its scope's
heads. Sprint is walk with a pace; charged frost shot is a ranged verb with a hold duration; climb is
locomotion on a climbable cell. Learned skills are instructions to bodies; commands are scripts in the
same envelope; activities are the middle layer's programs.

- Encoder: attention over the token sets with role embeddings and target marking (requirement 2);
  starts from the trainer's `token_encoder.py`.
- Memory: a recurrent state reset when a verb starts, with remembered context accepted from outside
  (last-known target position from the belief) (requirement 3).
- Interruptible: re-goaled on any tick; trained with mid-episode goal changes (requirement 4).
- Status and progress heads: done, stuck, chance of success and time left, calibrated (requirement 5);
  these are the status stream the middle layer reads.
- Scope enforced at sampling: the body's heads plus the runtime mask (requirement 6).
- Frozen trunk with per-verb adapters, versioned: adding a verb never changes a frozen verb
  (requirement 7; E13 proves it bit-exactly).
- Physics as inputs, not as things to learn: the projectile's launch and gravity give an analytic lead
  and arc that enters as a feature, the network learns the residual (requirement 13); the kit's timing
  tokens (12); resource meters with refill rates (14).

### 3.3 The contract with the middle layer

`requests.py` and `combiner.py` implement §9.10 of `SYSTEM_ONE.md`: a request is verb, typed
arguments with presence flags (position[3], direction[3], entity, scalars[4] named by the verb), mode
(parallel | after id | interrupt), urgency, confidence, priority, goal slot, executive version, program
step, expiry; positions are absolute world coordinates and directions unit vectors in the game's frame.
The combiner, per tick and per agent: validate against the verb's contract and reject invalid requests
with a reason; grant greedily in the order urgency, confidence, priority, recency, a request getting its
scope only if the scope is free, otherwise deferred naming the conflict; run the granted bodies; merge
the heads into one factored action, unused heads neutral; stream status (accepted | rejected(reason) |
deferred(conflict) | running | done(outcome) | aborted(reason)), progress signals, and per tick which
requests drove which heads. Fixed and exact; nothing learned in it. The in-graph form for the compiled
runner is an active-request table of one granted request per scope.

### 3.4 Training recipe (one recipe, every verb)

1. Warm start by behaviour cloning from native NPC traces where a verb has a native counterpart
   (locomotion, melee); the arena's imitation stack already does projection and recurrent replay.
2. PPO on gen-7 with potential-based progress shaping from the catalogue's "done" column; idle costs;
   means are never paid; mixtures over worlds x openings x opponents; selection on fresh openings
   (development seed 66001); reports on unseen axes (seed 77001); never tune on a report suite.
3. Curriculum inside a body by prerequisite (walk_to -> step_and_hop -> go_around -> sneak / sprint),
   old verbs kept in every batch so nothing is forgotten (E3 measures it).
4. Horizons from measured time-to-success (`catch_time.py`), never guessed.
5. Capability cards per verb from the suites: success and time-to-success per environment family,
   preconditions, failure modes, calibration. The cards are what the middle layer's catalogue
   descriptions are checked against.

### 3.5 Framework

The bodies run inside the JAX training loop (RULED). Default JAX, because the simulator, the ADK loop
and the trainer are JAX and collection stays in-graph. PyTorch is allowed if a body trained in PyTorch
inside that loop (DLPack hand-off per step, or a torch collector against the JAX simulator) gives
higher steps per second on this machine for the same body; E11 measures both on the same task before
the choice is final. No other criterion.

## 4. The gen-7 environment and its verification

**Status (2026-10-18, late).** The bridge to the live 0.7.2 server is done. The gen-7 simulator is a
render of HEAD's engine with the `sim/gen7` overlay (`HytaleRL/hytalegym/hytalegym/sim/`, `PORTING.md`),
and the identity gate keeps gen 5 bit-identical. The content layer, the locomotion rows, line of sight,
and constructs for the NPC jump arc, dodge timing, dagger damage, the guard window, the crossbow reload
binding, mana and crafting (1,168 of 2,047 recipes) are in. The rest of `SIMULATOR-0.7.2.md`'s P2 to P6
is open; each mechanic's status is the roll-up in `FIDELITY.md`. Since 2026-10-18 (late) the simulator
and the bridge belong to the peer session (`sim/gen7/GEN7-PLAN.md` section 2b). No skill trains on
gen 7 yet (section 7b says what that waits on). Until then the live server is the environment for
demonstrations and differentials, and sections 1 to 3 are built and proved with stub bodies.

**What this layer needs the simulator to reproduce, and how each is verified.** Every mechanic gets a
native differential on the live 0.7.2 server before it is called calibrated (the plan's own principle;
"a policy replay is not fidelity"). The checklist, with the catalogue's numbers as the expected values:

| Mechanic (catalogue source) | Verification | Pass bar |
|---|---|---|
| Movement: walk 1.65, run 5.5, sprint 7.0, sneak 3.0, jump 11.8, slide, landing roll, step-up | locomotion source audit on 0.7.2 assets and jar; flat and slope motion differential | position error <= 5e-4 per step (P2 gate) |
| Line of sight (0.7.2 passes non-transparent blocks unless a detail hitbox is hit) | LOS differential on an occlusion fixture; the catch rule and the `hidden` opening updated | agreement with native on every fixture ray |
| Terrain: renumbered block ids, changed Region sections, web speed | Region library recaptured on 0.7.2; path tables, suites, biome table, horizons rebuilt; audit worlds stay unseen | 500/500 sections captured; suites regenerated |
| Reset timing (first step after reset 34-68 ticks) | native episode alignment check | documented offset, applied in the session clock |
| Melee: hit interrupt, stun and 0.5 s flinch on 31 weapon attacks (new construct), sword knockback 4.3 at (0, 0.2, -1), dagger +50 % | new simulator construct; combat compare and active-combat compare re-run with 0.7.2 receipts; interrupt/flinch differential | per-hit displacement and flinch timing match native within tolerance |
| Ranged: damage cut 3-4x; bow draw tiers; launch and gravity per projectile (bow full draw 85 / 25, crossbow 40 / 10, Big Arrow 100 / 40) | projectile differential: launch speed, drop, hit distance per tier | range error <= 2 % per tier |
| Mana 100, +1 per 0.3 s; runes (fireball 30 mana, 2 s cooldown, 0.87 s cast, straight flight; wind strike; chain hook to 1.5; ground slam radius 3; charged frost tiers) | mana construct; rune differentials per rune | cost, cooldown, cast time, flight and effect match native |
| Stamina 10 (sprint, guard, dodge), oxygen 100 (-6/s submerged), signature energy, durability tables | resource differentials | rates match native within 5 % |
| Dodge: left and right only, impulse 13, 0.25 s invulnerable, 2 stamina | dodge differential; the locomotion head's four dodges reduced to two | impulse and timing match |
| Climbing (44 climbable blocks), doors, swimming, gliding, riding | new mechanics in the simulator, each with a differential | present and matching, or the verb waits |
| Crafting: 2,047 recipes, 19 benches, cook 2 s | crafting native compare on 0.7.2 | catalogue and timings match |
| The 11 source-pin audits (perception, locomotion, navigation, doors, mutable blocks, explosion mutation and dynamics, crafting, actor world state, inventory, entity geometry) | re-recorded from the 0.7.2 jar and assets | all 11 pass on 0.7.2 (P5 gate) |
| Contracts and checkpoints: new ruleset bytes change the contract stamp | every checkpoint retrained; native transfer `execution_valid` with one skill checkpoint on 0.7.2 (P6 gate) | one skill transfers and acts validly on the live server |
| Legal observation | the opt-in legal mode on for every result that counts; the leakage three-world test | no privileged field reaches a body |

A fidelity ledger (append-only, like the harness's) records every differential with its receipt. A
verb's training result counts only after the mechanics it depends on have passed their rows.

## 5. Acceptance: the sixteen requirements, mapped

| Requirement (SKILL_LIBRARY §7) | Met by | Proved by |
|---|---|---|
| 1 goal as input | request -> resolved typed args into the body | E4 |
| 2 sets, target marked | the set encoder, marking | E2 |
| 3 memory, reset on start, context from outside | recurrent state, last-known position input | E5 |
| 4 interruptible, trained with goal changes | re-goal on any tick, curriculum with mid-episode changes | E5 |
| 5 status and progress, calibrated | status heads, the status stream | E6 |
| 6 scope kept | scoped sampling + runtime mask | E7, R2 |
| 7 frozen skills never change | frozen trunk + per-verb adapters, versioned | E13 |
| 8 legal inputs only, FOV with occlusion | token sets from legal products; legal mode on | E12 |
| 9 every tick, all lanes, measured latency | in-graph bodies; latency probe before adoption | E11 |
| 10 trainable by the recipe and from demonstrations | recipe.py, traces.py | E10 |
| 11 reproducible | seeds, pinned configs, two-seed reruns | every E |
| 12 the held item's kit | kit tokens; duration head | E8 |
| 13 aim with physics | analytic lead/arc feature + residual | E9 |
| 14 resources | meter tokens with refill rates | E8, E9 |
| 15 many creatures with roles | entity tokens with roles | E2 |
| 16 blocks, faces, items, slots as targets | candidate and slot tokens, the resolver | E4 |

## 6. Proofs before a component is final (RULED: prove, then commit)

Each component enters the final architecture only after its experiment passes on gen-7 (or, for
components that do not touch the simulator, on stubs), reproduced on a second seed. Results go to the
ledger with the command and the numbers. "Baseline" means the proof-of-concept pursuit skill on its
own suite (31/41 catches on validation, 25/39 on fresh seeds, MEASURED).

| # | Component | Experiment | Pass bar | If it fails |
|---|---|---|---|---|
| E1 | Surface specification | every learned row of the catalogue maps to heads in one scope and to the token types it needs; no row needs a control the surface lacks | 86/86 rows mapped; the layout committed as the baseline before the simulator work | widen the surface, not the row |
| E2 | Set encoder | the set encoder replaces the flat MLP on the baseline task, same recipe, same suite | catches not below the baseline's lower bound; latency per tick measured | tune depth / width once; else keep the flat encoder for locomotion only |
| E3 | Shared body | one locomotion body learns walk_to, step_and_hop, go_around interleaved | each verb >= its single-skill result; no verb drops when the next is added | per-verb adapters larger; else separate bodies per verb family |
| E4 | Instruction following | typed args honoured: pace within 5 % of the gait asked, arrival radius respected, the marked cell is the one acted on | 95 % of episodes | args as extra tokens instead of a vector |
| E5 | Interruptibility | re-goal mid-episode to a new point or entity | recovery within the contract's commitment; success within 10 % of a fresh start | train with more goal changes per episode |
| E6 | Status and progress | calibration of the success-probability head; stuck detection | ECE <= 0.05; stuck precision >= 0.9 | recalibrate with temperature; else a separate critic |
| E7 | Combiner | two bodies on one agent through the request stream: walk_to a point while look_at an entity | arrive and keep the entity in view >= 90 % of ticks; R1-R6 pass with stub bodies | arbitration order change (R3) is a design change: back to the owner |
| E8 | Duration control | hold-for-k-ticks reproduces the bow's draw tiers and a charged release exactly | tier selected = tier asked in 100 % of trials in the simulator | fall back to per-tick press with a learned release |
| E9 | Physics prior | aim with the analytic lead/arc feature versus learned-only on a moving target at mid and long range | hit rate higher at every range; converges in fewer updates | keep the feature anyway (it is free) |
| E10 | Demonstrations | behaviour-cloning warm start from NPC traces versus from scratch on the locomotion body | reaches the baseline bar in >= 30 % fewer environment steps, or equal success with lower action jitter | RL from scratch; revisit when human traces exist |
| E11 | Framework and training throughput | the same body trained in JAX in-graph and in PyTorch inside the JAX loop; the full training loop (collection at the recipe's batch + the update + resets) measured in lane-steps per second and in wall time per million lane-steps (owner: training in the simulator must be fast as well) | steps per second on this machine decide; latency per tick for all lanes within the budget; the training-throughput number recorded as the baseline every later change must beat | JAX |
| E12 | Legal perception | every body scored under FOV16 and FOV32 with occlusion, legal mode on | success reported per FOV; no privileged field in any input (the legality test) | fix the input, never the test |
| E13 | Freezing | add a verb to a frozen body | frozen verbs' outputs bit-identical on a fixed input set; their suite numbers unchanged | separate networks for that body |
| E14 | Gen-7 fidelity | section 4's differentials for the mechanics a verb depends on | all rows green for that verb | the verb waits |

Order: E1 and E11 first (they fix the layout and the framework), then E2, E3, E4, E5, E6 on the
locomotion body (they need only P2 of the simulator), E7 with stub bodies in parallel (no simulator),
E8, E9 and the combat bodies after P3, E10 as soon as traces are recorded, E12 and E13 on every body.

## 7. Order of work

| Phase | Work | Needs | Gate |
|---|---|---|---|
| A (done) | `surface.py`, `catalogue.py` (contracts, descriptions, activities from the 139 rows), `requests.py`, `references.py`, `combiner.py` with stub bodies; the fidelity ledger; record NPC traces on the live 0.7.2 server | nothing | E1, E7 (stubs), R1-R6, traces on disk |
| B | the set encoder and `body.py`; E11 framework decision; the locomotion body on the baseline task | simulator P2 (locomotion, LOS, Regions) | E2, E3, E4, E5, E6, E12, E13 |
| C | gaze and posture bodies; combination gates with real bodies | B | E7 with real bodies |
| D | hands, abilities, interaction bodies: melee kits, ranged with lead and arc, runes with mana, interrupts, mining and placing | simulator P3 (combat), world edits executor | E8, E9, per-verb gates |
| E | commands (33 scripts) against the recipe catalogue; activities handed to the middle layer as programs | simulator P4 | script preconditions and reports tested |
| F | capability cards for every trained verb; descriptions checked against the cards; the middle layer's fine-tune begins (owner: after the skills) | A-E | cards published |

Per phase: run the gate, record it in the ledger with the command and the numbers, append a dated note
to `SKILL.md`, commit in one batch, never push.

## 7b. Phase B for the skill-agent session (proposed 2026-10-17, late; done 2026-10-18)

Phase A is done (eight commits: catalogue, protocol, combiner, parity, trace loader, fidelity ledger,
library, README; 122 verbs and 19 programs run on stand-in bodies). Nothing below needs the gen-7
simulator; everything below is needed before the first body trains on it. Each step keeps the phase-A
rules: gates recorded before claims, commit per step, never push, stage only changed files.

| # | Step | Gate | Why now |
|---|---|---|---|
| B1 | **Body interface and the token-set encoder.** `body(request, ticks_since_start, observation_tokens, belief_tokens) -> factors` as the one signature (no callback); a JAX set encoder over entity, block, inventory, geometry, self and kit tokens with role embeddings and target marking; shape tests on synthetic token sets; per-tick latency at batch 1, 8, 64 | encoder + combiner + resolver p95 <= 15 ms per tick at batch 8 (30 Hz budget with margin); shapes stable for any number of tokens | the bodies' contract must exist before any training; the latency bar is the owner's 30 Hz ruling |
| B2 | **Behaviour-cloning dataset from the 32 native traces.** Frames are 1/30 s already (no resampling at 30 Hz); map each frame to observation tokens and action factors on the gen-7 layout; held-out traces by world; a tiny BC body on CPU or the Mac GPU as a smoke | the dataset validates; held-out action agreement reported; action jitter reported; the smoke trains in minutes | the only demonstrations we have; proves the trace -> token -> factor pipeline before the simulator exists |
| B3 | **The combiner path becomes the default for today's three skills.** The composition arms run through parity by default; the middle layer's program executor emits `ActionRequest`s into the `requests` section and the combiner consumes them (joint step with this session) | the four arms' metrics unchanged with the new path on; the executor's requests appear in the scratchpad with statuses | closes the loop the owner described: LLM -> translate -> decode -> requests -> combine -> status |
| B4 | **Freezing mechanism (E13 on stubs).** Frozen trunk plus per-verb adapters; adding a verb leaves frozen verbs' outputs bit-identical on a fixed input set | bit-identical | requirement 7 is cheapest to prove before real weights exist |
| B5 | **One catalogue reader.** `skill_agent/catalogue.py` is the reader; `jev/system_one/catalogue_matching.py` imports it instead of parsing again (this session does the import change) | both tests green; the two readers' row-for-row agreement test retired | one source of truth, as the owner asked |
| B6 | **Per-skill progress and termination reasons in the contract.** From the catalogue's "done" column: the typed progress signals and the stop reasons each verb reports into the status stream | every contract carries them; the stub bodies emit them; the scratchpad shows them | the context the upper layers need to know when to ask for more (owner) |
| then | E2-E6 of section 6 on the rendered gen-7 locomotion engine (the set encoder against the flat MLP on the baseline task; one body, three verbs; typed arguments; interruption; calibrated status) | section 6 bars | needs the port's phase 2 (locomotion, LOS) |

**Status (2026-10-18, late).** B1 to B6 are built and committed:
- B1 `11acfc5`;
- B2 `baff062`, with the body-yaw label (`cbd4e00`) and terrain, meters and visibility (`5286797`);
- B3 `14db188`, `e62f654` and `9b6c571` (the combiner on every tick; `test_sa_joint.py` forces the
  rejected path);
- B4 `0f17bc8`;
- B5, done on the middle layer's side (`jev/system_one/catalogue_matching.py` imports
  `catalogue.parse_catalogue`);
- B6 `3439c67`;
- also `sim_adapter.py` (`19c642f`).

The joint check matches at 512 ticks at a 200 ms and at the default 20 ms host budget (ledger,
2026-10-18). The `then` row waits on three things:
- the simulator's locomotion phase: the walk resume after landing, and the Region library recaptured
  on 0.7.2;
- the owner's go-ahead for training runs (phase A's rule: none);
- the owner's choice of the motion model the skills train on. The `player` model has no live oracle
  while the bridge body is an NPC (`FIDELITY-0.7.2.md` (e), P2 item 1).

B2's observation, recorded as the owner asked (ruling 2026-10-17, "go ahead"). The cloned body reads three
token types the body contract already carries:
- **self** (`SelfToken`): health, meters, grounded, facing, speed;
- **entity** (`EntityToken`): the opponent, marked, with visible and last-seen age;
- **geometry** (`GeometryToken`): up to 64 solid, fluid, climbable or damaging cells of the trace's 9x9x9 grid,
  nearest first.

Light, route, kit, inventory and block-candidate tokens are not filled: these NPC traces have no data for them.
The labels add one rule to arena's four, body yaw from body steering.

What the control showed (`skill_agent/README.md` 2.8): geometry helps body turning on the held-out generated world
(+0.079) and hurts the compass on the held-out flat world (-0.079), against a masked-geometry control with the same
network. Next for this observation (not started):
- choose obstacle and height-change cells over the floor (the nearest 64 are mostly floor);
- record traces on more worlds before judging the token.

Owner items outstanding (2026-10-18, late):
- pushing: nothing is pushed, and main is 148 commits ahead of origin across all sessions;
- reviewing the lower layer's edits to the private catalogue (five reworded activity sequences, six
  added programs);
- the uncommitted pursuit fix in `arena/training/runs/pursuit_run.py`;
- for the `then` row, the training go-ahead and the motion-model choice.

## 8. Interfaces to the middle layer

The middle layer gives: a stream of typed requests (3.3). It takes: the status stream, progress, head
composition, outcomes and events through the shared scratchpad; the catalogue's descriptions (for
matching intents to verbs by meaning) and the activities (as behaviour programs). Nothing else crosses.
Its own fine-tuning (SYSTEM_ONE.md §9) starts after the skills train (RULED).

## 9. Risks, stated plainly

- The simulator is the long pole: nothing trains on gen-7 before P2; phase A is designed to need no
  simulator so the proofs that can start now do.
- The interrupt / flinch mechanic is a new construct, not a parameter change; its differential is the
  hardest row of section 4.
- Region recapture changes terrain identity: path tables, suites and horizons must be rebuilt, and the
  audit worlds re-selected and kept unseen.
- NPC traces cover locomotion and melee only; the other bodies start from scratch until human traces
  exist.
- The uncommitted observation-layout change noted in the simulator plan (8,272 vs 8,271) must be
  resolved and committed as the baseline before E1, or no later diff is interpretable.
- Six scopes is a design choice; E7 may show a different grouping (for example gaze inside
  locomotion); change the grouping, not the principle.
