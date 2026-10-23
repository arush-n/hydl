# The middle layer: consolidated design and plan (2026-10-17, night)

This is the entry point for the decision layer of the three-level agent (ADR-001). It consolidates
`SYSTEM_ONE.md` §8-§9, the owner's rulings of 2026-10-17, and the ledger's measurements into one
document. Every statement carries one of four labels:
**MEASURED** (a number in `LEDGER.md`), **BUILT** (committed code), **RULED** (the owner's words),
**DESIGNED** (specified, not built). Detail lives in `SYSTEM_ONE.md`; this file is the map.

## 1. The job, in the owner's words

- RULED: the LLM "reads the state and feeds high level ideas and keeps track of long term goals"; the
  middle layer "reads both the current state and the llm goals to make quick decisions based on what
  actions it thinks are best"; the lower layer "takes those actions and operates in the real world",
  follows directions clearly, and thinks only at the low level (combat, pathfinding).
- RULED: the middle layer's principle is to "give a series of neural actions to create unique human like
  behaviors based on the llm input and state"; the lower layer receives a stream of requests, conflicting,
  sequential or parallel, rejects the invalid ones, keeps acting, and streams status back; the middle
  layer works in correspondence with that stream.
- RULED: speed is key; nothing external except the top LLM; no live LLM run until the lower level is
  finished; the current skills are proof of concept; the final architecture must not cheat.
- RULED (late night): the agent operates at 30 Hz, the game's tick; the whole per-tick path must finish
  well inside 33.3 ms with margin (p95 <= 15 ms at the deployed batch), so the agent paces itself to the
  game clock rather than falling behind; horizons restate in 1/30 s ticks (512 ticks = 17.1 s).
- RULED (night): gen-7 (0.7.2) is the training target, gen-5 is deprecated; training of the neural
  systems is paused until the new lower-layer architecture is proved component by component; the
  skills train next, the middle layer is fine-tuned after. The lower layer's build plan, proofs and
  gen-7 verification: `experimental/neural_agent/skill_agent/docs/BUILD.md`.

## 2. What is established

| Piece | Status | Evidence |
|---|---|---|
| Shared scratchpad S_t: executive / belief / decision / skills / events, one writer each, versioned, JSON schema, a view per reader | BUILT `00a8fc7`; a package with the briefing view (markdown / json / compact, goal readings, the delta since the LLM's version) 2026-10-18 | wired into the harness (LLM writes), the session (belief, events, skills, decisions, outcomes), the eager runner (writes `scratchpad_<arm>.json` and the LLM view), J-S1 (`set_shared_state`); 12 tests |
| Decision speed, per decision | MEASURED | J-S1 0.29 ms at batch 64 (0.21 compiled), J1 0.10 ms, J0 7 us; budget 20 ms; zero text encoded per decision |
| Typed goal reading (structured, catalogue-agnostic encoder): unseen states and unseen catalogue sizes | MEASURED, BUILT `93defae` | 0.9599 / 0.9489 at 4096 generated training states; memorises at 256 (0.79 / 0.63); reads a new executive version in 0.16 ms |
| Goal-language generator: 10 intent families (3 real skills, 7 future), 9 condition families, teacher rule M13, splits by wording / skill family / catalogue size | BUILT `9039f3f` | 29 tests; M13 reproduces the composition rule M10 |
| Natural-language reading with frozen text embeddings in the goal slots | MEASURED FAIL, BUILT `737ea7e`, DEPRECATED in code 2026-10-17 (benchmark floor) | held-out wordings 0.53 (hashed) / 0.58 (ModernBERT-base), families 0.60 / 0.66, sizes 0.78 / 0.84; below the trivial rule (0.63); recorded rows fell to 0.88 |
| Pretrained trunk cost on this CPU, one pass over 330 tokens | MEASURED | ModernBERT-base 72.5 ms; Qwen3-0.6B 245.5 ms; so language models run per executive version, never per tick |
| Composition task (pursue, evade on damage, re-engage) at 512 ticks, legal observation mode | MEASURED | scripted rule 33/64 (Wilson lower bound 0.396), shuffled control 16/64, flat 0/89; the behaviour test the repository already has |
| Executive seam: versioned store, report channel, triggers, async harness, Anthropic adapter, baselines | BUILT | no live API call made; 60+ tests |
| Simulator leak closed, opt-in legal observation mode; our arena opts in | BUILT | privileged default unchanged for every other consumer |
| Skill library today: idle, pursuit, evader; contracts with action scopes; dispatcher runs one skill per lane; factored action space of 12 heads | BUILT (proof of concept) | commitment and duration moving into the contracts (in progress, owner ruling) |

## 3. Architecture

```
   L3  EXECUTIVE (external LLM, asynchronous, seconds)
       reads: the LLM view of S_t (goals, facts, objectives, per-skill aggregates, outcomes since its
              last version, how its words were read, information requests)
       writes: the `executive` section: goals = typed fields + typed target + words (intent, when, note),
               facts, long-term objectives, budget, style notes
                     |                                              ^
                     v   S_t  (one document, one writer per section, versioned, per NPC or per squad)
   L2  DECISION LAYER (our System One model, two timescales)
       slow path, once per executive version (~73 ms CPU, off the tick):
           J-S2 reads goals + catalogue in one pass -> per goal: matched skill, condition kind,
           has-number (exact number reader supplies the number); per intent: a behaviour program
           (<= 4 steps of request sets), sampled, style-conditioned, calibrated
           -> written to the `decision` section (the LLM sees how it was read and what was planned)
       fast path, every cadence tick (~0.3 ms, exact + light heads, no language model):
           evaluate the interpreted predicates on B_t, priority rule with interrupts, advance the
           program, emit requests (verb, typed args, urgency, confidence, mode) into `requests`,
           read the status stream and correspond (re-sample, raise urgency, ask the LLM)
                     |                                              ^
                     v   `requests` section                         |   `skills` status stream
   L1  SKILL LIBRARY (neural verbs with typed arguments; the combiner)
       every tick: validate requests against the verb contracts (reject invalid with a reason),
       grant greedily by (urgency, confidence, priority, recency) over action-head groups, run the
       granted verbs, merge their heads into one factored action, stream status and progress back.
       Skills own their commitment and duration and the low-level thinking (combat, pathfinding).
       The belief module writes `belief` (what each agent knows) for all three layers.
```

What crosses each boundary: downward, only typed records (goals; requests). Upward, only typed status
(outcomes, progress, rejections, head composition, events) plus the decision layer's confidence and
interpretation. No layer sends observations to another; every layer reads the state.

## 4. The decision layer in detail

**Reading (DESIGNED, §9.1-§9.4).** J-S2 is the Jev recipe applied where the CPU has time: one forward pass
of a pretrained backbone fine-tuned on our decision data, every output read in parallel as a typed
classification with a calibrated probability. Input: the LLM's goals as written, facts and notes, and
the catalogue as `id: description` lines, each goal and each entry under a marker token. Per goal, from
the marker states: skill over the K entries (`softmax_j q_k^T W e_j`, no weight depends on K; an unseen
skill is matched by its description; note that for an intent the library cannot serve, "the best
possible choice" is built today as the nearest catalogue id with descriptions as suggestions only, SYSTEM_ONE
9.16, because description matching measured as noise), condition kind over the nine predicates, has-number. Numbers come from an exact
reader, never from the network. Backbone: ModernBERT-base, top 4 of 22 layers trainable; fallbacks a
22M encoder (speed) and a LoRA-tuned Qwen3-0.6B decoder (accuracy, the owner's KEV idea).

**Programs (DESIGNED, §9.9).** Per intent, up to 4 steps; a step is a request set (verbs with typed args),
an exit condition (predicate on B_t, or done, or a timer) and a hold-time bucket. Sampled with a
temperature once per executive version per agent (uniqueness), re-sampled on failure, shifted by the
LLM's style notes. The one-step program is today's single skill call.

**Execution (DESIGNED, §9.9-§9.10).** Per cadence tick, exactly: evaluate the current step's exit on B_t,
advance or hold; the LLM's conditional goals interrupt by priority and the program resumes; emit the
step's requests with urgency and confidence; read the status stream: rejected -> fix args or re-sample,
repeated rejections -> information request to the LLM, done -> advance, stalled progress -> re-sample,
long deferral -> raise urgency or drop. Replace, params, confidence and value heads stay (BUILT).

**Why this is a System One model.** Unstructured state in, typed probabilistic decisions out, in one
pass, calibrated, every field in parallel; the interpretation stage is the "reasoning within", made
explicit, typed and inspectable, and it is where real LLM wording will be learned later.

## 5. The contract with the lower layer (DESIGNED, §9.10)

- `requests` section (writer: the decision layer): `ActionRequest` = id, issued and expiry ticks, verb,
  typed args with presence flags (position[3], direction[3], entity, scalars[4] named by the verb),
  mode (parallel | after id | interrupt), urgency, confidence, priority, goal slot, executive version,
  program step. Positions are absolute world coordinates, directions unit vectors, in the game's frame.
- `writes` section (writer: the decision layer; BUILT 2026-10-22, the owner's rulings 1 to 3): one `WriteRecord` per
  write = id, agent, tick, writer `decision`, name (`remember`), value, one key (an entity id or a world position),
  goal slot, executive version, program step. At most 8 records per agent per tick; a ninth is refused, not dropped.
  The skill layer applies them (`skill_agent/state/written.py`: `lower`, then `apply`); the adapter is
  `shared_state/skill_writes.py` (`writes_for_tick`, `record_write_results`; and `outcomes_for_tick`, the per-tick request outcomes for the skill layer's event log). The session runs the skill layer's state tick when given `run_session(state_binding=StateTickBinding(...))` (opt-in, default off; task #136): every tick, last, the decision's writes on a decided tick and the ended requests every tick. Each write's result comes back in the
  `skills` section as a `WriteStatus`: `write_applied`, or `write_refused(reason)` with the host's reason
  (`unknown field`, `wrong writer`, `out of range`, `bad key`, `two keys`, `too many writes this tick`) or `full`
  (the device table had no free entry). Policy (`jev/remember.py`, `jev/programs.py` item 10): (1) a program step
  whose request targets an entity or a place writes `remember = p` on that key when the step starts, with p the
  goal's priority clipped to [0, 1], the mapping the request's urgency uses; a goal with no slot (priority 0) writes
  nothing, because a write of 0 ejects. (2) The program run that wrote a key ends by done, aborted (a withdrawn request)
  or replaced (a restart), and writes `remember = 0` on each key it wrote, unless `Program.keep` (default False) keeps
  them. (3) On `full`: the lowest-valued entry this layer wrote that no live request refers to is ejected, and the
  refused write is retried at the next decision; when every entry is referenced, the LLM is asked (a
  `LaneInformationRequest` in the decision section, verb of the step, reason `full`).
- Status stream in the `skills` section (writer: the dispatcher): accepted | rejected(reason) |
  deferred(conflict) | running | done(outcome) | aborted(reason), progress signals per the verb's
  contract, and per tick which requests drove which heads.
- Combiner: validate, greedy grant by head-group scope, run, merge, stream back. Fixed and exact;
  nothing learned in it (RULED: independent discrete actions combine; conflicts rejected greedily by
  urgency or confidence; as many concurrent requests as plausible, bounded in-graph by the four head
  groups).
- Contract changes: typed argument schema and explicit head-group scopes per skill; the four-float
  parameter block becomes an eleven-float argument block with flags; an active-request table of 4 per
  agent for the compiled runner; commitment and duration owned by the skill (RULED).

## 6. Training and data

- Stage A, supervised (DESIGNED): the goal-language generator (BUILT) for reading; a behaviour-language
  generator (DESIGNED) for programs: behaviour families with canonical programs over skill families,
  paraphrases, hold times, style modifiers; the recorded traces for the typed path (BUILT). Held out by
  wording, by family, by catalogue size. Minutes per fine-tune on the Mac GPU (RULED available).
- Stage B, calibration (DESIGNED): temperature scaling and a Brier check; ECE <= 0.05.
- Stage C, outcomes (DESIGNED, later): through the harness once the library exists, RLCR-style reward
  with the non-hackable scoring; this is where human-likeness is learned rather than scripted.
- The real LLM's goal texts become the second data source when the live run is allowed (RULED: later).

## 7. Gates

| Gate | Bar | Status |
|---|---|---|
| G1 pipeline agreement with the scripted rule, recorded rows | >= 0.95 | MEASURED 1.000 |
| G2b trunk calls per decision; decision latency | 0; << 20 ms | MEASURED 0; 0.29 ms |
| G3 typed goal reading, unseen states and catalogue sizes | >= 0.90 | MEASURED 0.96 / 0.95 at 4096 |
| I1 reading: condition kind; number; skill id | kind: typed in the schema, BUILT 2026-10-17 late as `condition` (fine-tuned encoders FAILED the 0.95 bar at 0.75-0.82, polarity, SYSTEM_ONE §9.12; deterministic fallback reader MEASURED 1.000 kind and kind+argument on held-out wordings, seeds 0-2, no polarity confusion, weak readings flagged; decoder readers MEASURED, P6: 4B 0.9745, 8B and 14B 1.000 on held-out wordings, 0.6B and 1.7B FAIL on the not-visible polarity); number MEASURED 1.000; skill id typed by schema and read from patterned words at 1.000 on the real catalogue (P7, §9.14; description matching FAILED, §9.11) | BUILT: typed fields are the contract (schema, prompt, generator), words follow a prompted pattern read first (§9.14); the fallback readers pass their bars |
| I2 end to end agreement with M13: typed conditions evaluated exactly, then the priority rule | >= 0.90 on three splits | MEASURED 2026-10-17 late: `jev/typed_rule.py` equals the teacher on every decision over 300 generator states with typed fields, >= 0.99 on words-only goals through the rule reader (test_na_typed_rule); `jtyped` arm online in the ledger |
| I3 a new executive version read; backbone calls per decision | <= 150 ms; 0 | BUILT: the typed path reads a version by schema validation and predicate construction, no model on the path, 0 backbone calls; latency is the composition driver's p50 column |
| I4 calibration of skill and kind probabilities | ECE <= 0.05 | DESIGNED |
| B1-B4 programs: step agreement 0.90 / 0.85; validity 100 %; >= 3 distinct sensible programs per intent; composition not below the scripted bound | as stated | executor MEASURED (P4: the scripted-rule program 0.512 vs 0.531, Fisher p = 0.875, 0 misses, 0.24 ms; a three-step program runs in order; under the host seam that honours the skills' commitments, the default since 2026-10-17 late, 36/76 = 0.474 against J0's lower bound 0.423, timeouts 1 in total once the executor's clock reads the age's increase, 1 miss of 997 under load); the learned program heads DESIGNED, deferred until real skills |
| R1-R6 request stream: invalid rejected with reason; no shared head; reproducible greedy order; chains in order; reaction within one cadence tick; compiled = host | as stated | middle-layer side MEASURED (P5: R1, R5 with a stub combiner; 20-point mutation check); combiner side the other session's phase A (A5 tests report R2, R3, R4, R6) |
| G4 online composition with the learned instance on held-out seeds | not below J0's lower bound, 0 misses | MEASURED rate PASS on seeds 0, 1, 2 (0.500 / 0.488 / 0.458 vs lower bounds 0.423 / 0.465 / 0.337; pooled 0.482 vs 0.514, not significant) once the instance reads the contracts' commitments; 0-miss part deferred to an unloaded machine (misses were latency under six concurrent agents) |
| G5 calibration of the decision heads | ECE <= 0.05 | deferred |

## 8. Plan

| # | Step | Depends on | Effort | Gate |
|---|---|---|---|---|
| 0 | Control fields into the skill contracts (runner and instances read a per-skill table) | - | running (subagent) | composition rows unchanged |
| 1 | Exact number reader; interpretation benchmark with frozen encoders as the floor | - | running (subagent) | reader exact-match >= 0.98 |
| 2 | Request stream: `requests` section, status stream, combiner with stub skills, typed argument schema and head-group scopes, argument block, active-request table | 0 | ~2 days, host side first | R1-R6 |
| 3 | Fine-tuning harness: markers, dataset from the generators, reading heads, program heads emitting request sets, behaviour generator, three-split evaluation, checkpoints | 1, 2 | ~1 day | smoke run learns training wordings |
| 4 | Fine-tune ModernBERT-base on the Mac GPU; iterate | 3 | minutes per run; a day of iteration | I1, B1-B3 |
| 5 | Calibration and latency: temperature scaling, ECE, int8, the probe | 4 | half a day | I3, I4 |
| 6 | Integration: the instance interprets once per version, executes programs, emits requests; decision section shows plan and stream; js1 arm through the composition driver | 2, 5 | ~1 day | I2, B4, G1 unchanged |
| 7 | Offline LLM-side pieces: predicate vocabulary and typed target in the prompt and schema; the backend writes the executive section; objectives list; per-squad views | 6 | DONE 2026-10-17 to 18 (SYSTEM_ONE 9.14 and 9.15): vocabulary, typed target, patterned directions, addressees and squads, objectives, the squad view, and the backend replying in the section schema with the briefing in its user turn | prompt bytes stable; no live call: holds (fake client only) |
| 8 | Fallbacks if 4 misses: more states, more layers, then the decoder path | 4 | as needed | I1 |
| later | Stage C on outcomes; live LLM run; the real skill library replaces the stubs; group scratchpad | the lower level | - | G4, G5 |

Owner decisions taken into this plan: encoder first (recommended), the Mac GPU for training, absolute
coordinates in requests. Owner decision still needed: the go for steps 2 onwards.

## 9. Risks, stated plainly

- Zero-shot composition for unseen behaviour families is the hard case; the bar is 0.85, not 0.90.
- Human-likeness cannot be gated; its ingredients can (composition, timing, interrupts, sampled variety,
  outcome learning later).
- The per-version slow path is 73 ms per NPC on this CPU; squads of many NPCs need the GPU or a smaller
  encoder.
- The request-stream contract touches the ADK core (dispatcher, runner, lowering); it is staged host-side
  first with stubs, compiled second, so the proof-of-concept skills keep working until the library lands.
- All reading data is procedural until the live LLM run; the generator's paraphrases are our best model
  of the LLM's wording, not the real thing.

## 10. Where the detail is

`skill_agent/docs/BUILD.md` for the lower layer (bodies, combiner, catalogue parser, gen-7 verification, proofs E1-E14);
`SYSTEM_ONE.md` §8 (J-S1, the scratchpad §8.10, the generator §8.11, the rulings §8.12, the failure and
the split §8.13), §9 (J-S2, programs §9.9, the request stream §9.10); `FORMULAS.md` M1-M13;
`LEDGER.md` for every number above; ADR-001 (local) for the decisions and the evidence rows.

## 11. Relationship writes (step 9's decision-layer side; DESIGNED and APPROVED 2026-10-23)

The skill layer's step 9 (`skill_agent/docs/VIRTUAL_HUMAN.md` §26) adds 16 relationship fields to each agent's written
entity entries (§26.2: `closeness`, `ability`, `benevolence`, `integrity`, `hostility`, `fear`, `anger`, six role
flags, `ingroup`, `owe_them`, `they_owe`), up to 150 individuals, keyed by the perceived id. The owner ruled that the
decision layer writes every field (§26.5 item 2). This section is how it decides what to write. Built after the skill layer's step 9 lands; its rules read §26.2's interaction facts.

**Inputs that exist today** (each named in S_t or the skill layer's state):
- `skills.known_entities` per agent (#140, fc2b129): id, visible, age, distance, hostile, attitude, health, alive.
- The skill layer's interaction facts (§26.2, `state/social.py`, being built): meetings and time in view, hits taken
  and dealt per individual (inferred until the simulator publishes the attacker, S11), who struck first.
- Request outcomes (skills section statuses) and the executive's goals (what the LLM asked for, and about whom).
- Missing, so their fields wait (§26.5 item 4): speech, item transfers, visible names and each entity's activity.
  `family`, `leader`, `follower`, `trader`, `owe_them` and `they_owe` have no input yet. They are written only when the
  executive (LLM) names one through a goal (below), never by rule.

**The rule set** (in `jev/relations.py`, new). It runs at each host decision per agent over the individuals with new
facts since the last decision. Every write goes through the existing `writes` section (≤ 8 per agent per tick; the
largest changes first; a change under 0.05 is not written). Constants marked [E] are starting values, to be fitted.
- **Fights** (per hit taken from them, `d` = damage / my maximum health): `hostility += 0.5 d` and `anger += 0.8 d`
  when they struck first in the meeting (blame: harm another caused, the appraisal account of anger); `fear += d`
  scaled by how badly I am losing (their damage per second over mine) [E]. A hit I dealt first raises `hostility` by
  0.2 d (rivalry), not anger. `ability` moves toward their measured hit rate on me, scaled to [-1, 1] [E].
- **Peace** (time in view with no harm either way, per meeting): `closeness += 1/30` per meeting that lasted over 60 s
  without a hit, following Humanoid Agents' ±1 on a 0-30 scale per interaction (§26.6). `benevolence += 0.05` when
  they could have struck (in reach and facing me, the bodies' reports) and did not [E].
- **Decay** (written by the decision layer; written values never fade on their own, §19.7): `anger` halves every 60 s
  out of contact, `fear` every 120 s out of view, `hostility` every 30 min [E]. `closeness`, trust parts and roles do
  not decay; Humanoid Agents' closeness does not.
- **Role flags by rule:** `rival` after 3 meetings with fights; `friend` when closeness > 1/2 and hostility < 0.1;
  `ingroup` from the scenario's team, when it publishes one. Cleared by the same rules in reverse.
- **The LLM's say:** an executive goal may carry a relationship directive ("treat 12 as a friend", "trust 7 less").
  The decision layer writes it, and the rules do not overwrite a directed field for that goal's lifetime. The LLM has
  no write path of its own (§26.5 item 2).
- **Full table:** a 151st individual is refused "full"; the decision layer ejects the record with the lowest closeness,
  no role flag and the oldest last contact, then retries. This is the eject-and-retry rule of #135.

**Grounding:** the fields, scales and turnover come from §26.6 (Humanoid Agents' closeness steps; Mayer, Davis and
Schoorman's trust parts; Dunbar's layers). The blame-to-anger and threat-to-fear split follows appraisal theory
(Lazarus 1991; Ortony, Clore and Collins 1988), from memory and not re-read here. The rates are engineering choices,
marked [E].

**Verification:** unit tests per rule; a scripted duel where the opponent strikes first, so hostility and anger rise
and anger halves after 60 s apart; a peaceful co-presence run where closeness rises by 1/30 per qualifying meeting;
a directive that holds against the rules; a 151st individual causing one ejection and a retry.

**Decisions (RULED 2026-10-23):** (1) the rates and half-lives above, as starting values marked [E]: "use these";
(2) `ability` and `benevolence` move by rule now, `integrity` waits: "yes, by rule now"; (3) role flags `rival` and
`friend` by rule, any role by LLM directive, and a directive holds against the rules: "by rule + LLM".

**Built (2026-10-23).**
- **Code.** The rules are in `jev/relations.py`. They are called at each host decision from `ProgramExecutor._issue_requests` (`jev/programs.py`), and the results settle at the next decision (`_settle_relations`). `session.py` binds `agent_state` to the skill layer's state, so with no binding the decision layer writes nothing. Elapsed time is in seconds: the device clock times `tick_seconds`. Relationship writes share the 8-per-agent budget after the remember writes. A goal carries an optional `relation` (`RelationDirective`, `shared_state/sections/executive.py`).
- **Tests.** `test_na_relations.py`, 18 tests, one or more per rule: a fight where the opponent struck first, with anger halving after 60 s; fear and hostility half-lives; peace and spared chances; rival, friend, and that ingroup and integrity are never written; a directive holding against the rules; the budget; and a 151st individual causing one ejection and a retry.

**Readings of the rule set made in the build (accepted; each marked where it is in the code):**
1. The 0.05 floor applies per individual, using the largest change; the smaller changes for that individual go with it.
2. The peace credit of 1/30 is written although it is under 0.05. Without that exception closeness never rises from peace.
3. "Out of contact" and "out of view" both read as "not perceived this tick". The records have no separate contact signal.
4. `ability` moves toward `2 × their share of hits − 1` at rate 0.1 [E].
5. Rivalry (+0.2 d) counts the damage I dealt while I struck first in the meeting.
6. Ejection skips entries holding a `remember` value.
7. Fear's ratio is the meeting's damage taken over damage dealt, clamped to 2 [E].

**Open.**
- The LLM's goal may carry an optional `relation` list ({entity, name, value}), read into `GoalEntry.relation` by `ExecutiveSection.from_dict`; `integrity` and `remember` are refused there. Its rule-side hold is tested in `test_na_relation_directive.py`.
- A refused write waits when no record can be ejected. The remember path asks the LLM in that case; this path does not yet.
- The skill layer's `views.facts(dt_seconds)` has no production caller to pass seconds to.
