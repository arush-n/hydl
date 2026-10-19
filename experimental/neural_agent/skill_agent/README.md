# The skill agent: how to use it, and why it is built this way (2026-10-17, base 691e854)

`docs/BUILD.md` is the plan: what to build, the gen-7 checklist and the proofs. This file covers two other
things. First, the guide to creating, registering and using a skill (section 1). Second, the reasons for
each component, with the research each one follows (section 2). Two handoffs follow: one for the agent
building the gen-7 simulator (section 3), and one for the agent that will be built on top of this layer
(section 4).

Evidence markers:

- **[test: name]** — a test in `tests/experimental/neural_agent/test_sa_*.py` proves the claim.
- **[ledger]** — a recorded gate in `experimental/neural_agent/docs/LEDGER.md` proves it.
- **(inferred)** — the claim is a reason, not a measurement.

Each reference in section 6 says how it was checked, in the vocabulary of
`experimental/neural_agent/docs/REFERENCES.md`. *Recalled* means cited from memory and not reopened. *Search summary
2026-10-17* means only a search engine's summary was read; three entries were checked this way: MineDojo's action space,
Pirjanian's report number, and ros_control's exclusive resource claims.

```
middle layer (JEV)                lower layer (this folder)                              environment
ActionRequest(verb, typed args, ──► check ──► grant per scope ──► bodies ──► merge ──► 21 heads ──► gen-7 sim / server
  mode, urgency, ..., expiry)
RequestStatus(accepted | ...    ◄───────────────── status stream (heads driven, progress, reasons) ◄──┘
  | done(outcome))
```

## Layout

The modules are sorted into subfolders by purpose (2026-10-18); each subfolder's README lists its files, gates and
commands. Paths in this package's documents are relative to `skill_agent/`; files elsewhere are cited by repo path.
Import a module by its subfolder: `from experimental.neural_agent.skill_agent.runtime.library import SkillLibrary`.

| Folder | What it holds |
|---|---|
| [`contract/`](contract/README.md) | What a skill is and what it speaks: the gen-7 surface, the catalogue's verb contracts, the request protocol, the resolver. |
| [`runtime/`](runtime/README.md) | Running skills: the registry, the combiner, programs, the combiner on S_t, the simulator head adapter, the parity and joint checks. |
| [`model/`](model/README.md) | The networks: the token-set trunk, per-verb adapters, the request-conditioned generalist, the legal-observation adapter. |
| [`teaching/`](teaching/README.md) | Scripted teachers and their certificates. |
| [`tasks/`](tasks/README.md) | The episodes skills are trained and measured on. |
| [`training/`](training/README.md) | Native traces, the behaviour-cloning dataset, the trainers. |
| [`bench/`](bench/README.md) | Per-tick latency benches (the 30 Hz gate). |
| [`docs/`](docs/README.md) | The design record: `BUILD.md` (the plan), `TRAINING_PLAN.md`, `ARCHITECTURE_RESEARCH.md`, `FIDELITY.md`, `AGENT_BRIEF.md`. |

## 1. Create, register, use (the guide)

The code below runs as written. Its output is shown in section 4.

```python
from experimental.neural_agent import requests as middle
from experimental.neural_agent.skill_agent.contract.catalogue import ActivityProgram, ActivityStep
from experimental.neural_agent.skill_agent.runtime.combiner import AgentContext, BodyStep
from experimental.neural_agent.skill_agent.runtime.library import SkillLibrary, script
from experimental.neural_agent.skill_agent.runtime.programs import ProgramRunner

library = SkillLibrary.from_catalogue()                     # 122 verbs and 19 programs, one call
context = AgentContext(world_bounds=((-100.0, -64.0, -100.0), (100.0, 320.0, 100.0)),
                       known_entities=frozenset({7}))

# A learned skill: its body is (request, ticks since start, observation tokens, belief tokens)
# -> BodyStep(factors, done, outcome, progress).
def wave_body(request, ticks_since_start, observation_tokens, belief_tokens):
    return BodyStep({"yaw_delta_bins": 8}, done=ticks_since_start >= 3)

library.create("wave_at", description="turn to someone and wave", scopes=("gaze",),
               required=("entity",), done="waved", body=wave_body)

# A command: a script in the same envelope, one frame per tick after the grant.
library.create("open_gate", description="open the door ahead", scopes=("interaction",),
               done="opened", kind="C", body=script([{"door": 1}]))

# A frozen skill changes only under a new version.
library.create("wave_at", description="turn to someone and wave, faster", scopes=("gaze",),
               required=("entity",), done="waved", version="v2", body=wave_body, replace=True)

# A program: stages run in order, steps of one stage run together, (for each ...) repeats.
library.register_program(ActivityProgram("greet_and_open", (
    ActivityStep("wave_at(friend)", ("wave_at",), "friend", stage=0),
    ActivityStep("open_gate", ("open_gate",), "", stage=1),
), "who the friend is"))

# Use. Build the combiner after registering: it is a snapshot, and it holds the grant state.
combiner = library.combiner("gen7")
walk = library.request("walk_to", agent=0, request_id=1, tick=0,
                       position=(4.0, 64.0, -2.0), scalars={"radius": 1.0, "pace": 1.0})
step = combiner.step(0, 0, [walk], context)     # step.factors: 21 choices; step.statuses

def binder(step, verb, iteration):              # where each step acts: from belief and world memory
    return middle.RequestArgs(entity=7) if verb == "wave_at" else middle.RequestArgs()

runner = ProgramRunner(library, "greet_and_open", agent=1, binder=binder)
for tick in range(20):
    runner.observe(combiner.step(tick, 1, runner.requests(tick), context).statuses)
    if runner.state != "running":
        break
```

Rules the registry enforces, each covered by a test:

- **Unique verb ids.** A verb id is lowercase with underscores and is registered once. A replacement
  needs `replace=True` and a new contract version. A skill has at most `PARAM_WIDTH` (4) named scalars,
  and its scopes must be distinct and real.
  [test: test_registration_refuses_duplicates_and_same_version_replacements]
- **Programs name registered things.** A program names only registered verbs or programs, a nested
  program is a stage of its own, and there are no cycles.
  [test: test_programs_must_name_registered_verbs_and_must_not_loop]
- **Placeholders are per layout.** A verb registered without an executor gets a placeholder. It is
  built for the layout of the combiner that asks, so one registry serves the gen-5 and gen-7 combiners.
  [test: test_skills_run_through_the_librarys_combiner_and_the_command_script_sets_the_hotbar runs
  gen 5; test_every_verb_is_accepted_drives_only_its_own_heads_and_finishes runs gen 7]
- **Scalars by name.** `request` takes scalars by name, rejects a name the verb does not have, and sets
  the expiry from the contract: `expires = tick + max(1, ceil(T / tick))`. T is the contract's
  `maximum_duration_seconds`, or 60 s when that is infinite, and the tick is the layout's: 1/30 s on
  gen 7 (the owner's 30 Hz ruling), 0.045 s on gen 5.
  [test: test_the_request_helper_names_scalars_and_sets_the_expiry_from_the_contract]
- **Cancelling.** To cancel a request, stop sending it. The next step aborts it with
  "withdrawn". [test: test_a_request_that_leaves_the_stream_is_aborted_as_withdrawn]

**How a body sees (decided 2026-10-17 with the architecture review).** Every body is called as
`body(request, ticks_since_start, observation_tokens, belief_tokens)` and returns a `BodyStep`.

- **Where the tokens come from.** The caller builds the agent's `AgentContext` each tick with its
  observation tokens (the surface token types) and its belief tokens. The combiner hands both to every
  body it runs for that agent, untouched. [test: test_bodies_receive_the_agents_tokens_unchanged_and_count_ticks_from_the_grant]
- **The trunk runs once per tick (owner ruling 2026-10-18).** When trained verbs are in the library,
  the caller passes the tick's contexts through `freezing.encode_contexts(network, contexts)` before
  the combiner runs. Each agent's observation tokens then arrive as an `EncodedObservation`: the
  tokens, the agent's context row and the trunk that produced it.
  - A trained verb's body (`freezing.VerbBody`) runs only its adapter on that row.
  - On raw tokens the same body runs the whole verb program, so the trunk runs once per body call.
  - The signature is unchanged; placeholders and scripts ignore the tokens either way.
  [test: test_the_encoded_body_chooses_what_the_raw_body_chooses; test_a_body_refuses_an_observation_encoded_by_another_trunk]
- **Why arguments, not a callback.**
  - The owner ruled that the state carries everything the neural agent needs.
  - An argument keeps a body a pure function: the compiled runner can trace it, where a callback has no
    place, and tests can feed it fixtures.
  - It matches the request contract: the resolver has already turned references into agent-frame
    vectors and marked tokens.
- **Memory.** A body keeps its recurrent state per `request_id` and resets it when
  `ticks_since_start == 0`, because a new grant is a new run of the verb (docs/BUILD.md §3.2).
- **Why `BodyStep`, not bare choices.** A body returns its choices together with done, outcome and
  progress, because the status stream reports them.

## 2. Why each piece is the way it is

### 2.1 A skill is an option with a contract (`contract/catalogue.py`, `runtime/library.py`)

An option is a triple (I, π, β): where it may start, its policy, and when it ends (Sutton, Precup and
Singh 1999). Here the three parts are:

```
I  the check: requests.validate through protocol.check, in the order
   expired, verb_unknown, missing_argument, non_finite_argument, out_of_bounds, entity_unknown,
   scope_illegal, predecessor_not_done
π  the executor: a Body (learned network, placeholder, or command script)
β  the contract's done signal, its failure signals (stuck, timeout, precondition_failed),
   and maximum_duration_seconds
```

Design by contract (Meyer 1992) maps the same way:

- **Precondition:** the check. The cheapest and most basic conditions come first, and the first one that
  fails is reported, so a rejection names one reason the caller can act on.
- **Postcondition:** the done and failure signals.
- **Frame condition:** the scope set, which says what the skill may touch. It is enforced at run time,
  not trusted: a body that writes a head outside its scopes raises.
  [test: test_bodies_must_stay_inside_their_scopes]

**The catalogue is the source of truth.** The owner edits one file. The parser stores rules, never
catalogue text. [test: test_the_parser_source_holds_no_catalogue_text]

All 139 rows parse. This includes 86 learned rows, each with a scope and a complete schema, and 10 of
them own a second scope. [test: test_the_real_catalogue_parses_completely]

**Commands share the learned skills' envelope.** A command uses the same request, check, grant and
status stream. The middle layer therefore cannot tell a script from a network. A script can stand in
until a network exists, and the network can replace it under a new version without any change above.
SayCan (Ahn et al. 2022) and Code as Policies (Liang et al. 2023) treat skills the same way: a
described, callable unit whose implementation the planner does not see.

**Versions are frozen.** A replacement needs a new version (ADR-001 D1; SKILL_LIBRARY requirement 7).
Programs, the middle layer's matching and the capability cards all depend on a skill's behaviour, so
that behaviour may change only under a new name (inferred).

**Descriptions are the matching text.** Voyager (Wang et al. 2023) keeps a growing library of verified
skills and retrieves them by their descriptions. `library.descriptions()` gives the text for every verb
and program. The owner's ruling (commit e865049) keeps conditions typed; the words are context only.

### 2.2 Factored heads in six scopes (`contract/surface.py`)

**Factored heads.** The action is a set of heads that one torso emits together:

- action branching (Tavakoli, Pardo and Kormushev 2018);
- AlphaStar's action type with argument heads (Vinyals et al. 2019);
- MineDojo's `MultiDiscrete([3, 3, 4, 25, 25, 8, 244, 36])`: movement, camera, one functional action
  (noop, use, drop, attack, craft, equip, place, destroy) and its craft and inventory argument heads
  (Fan et al. 2022).

Gen 5 is exactly the gym's twelve native heads [test: test_the_gen5_layout_reproduces_the_gym_heads_exactly].
Acting on the native surface is VPT's choice (Baker et al. 2022): nothing is lost to a hand-made
abstraction. Gen 7 keeps those twelve first and adds nine planned heads (section 3).

**Scopes partition the heads,** so whether two requests can run at once is a disjointness test:

```
heads(r) = ∪ over s in scopes(r) of H_s,   H_s ∩ H_s' = ∅ for s ≠ s',   r and r' co-run  ⇔  scopes(r) ∩ scopes(r') = ∅
```

[test: test_every_head_belongs_to_exactly_one_scope, test_scope_masks_partition_the_present_heads]

This is ros_control's resource model: a controller claims hardware interfaces, and a resource claimed
by two controllers is a conflict, refused at switch time (`RobotHW::checkForConflict`; Chitta et al.
2017). Heads no request owns are neutral, the "do nothing" choice, so a request never acts outside its
scopes. [test: test_neutral_choices_follow_the_proof_of_concept_rule]

**The duration head.** It says "hold for k ticks" instead of a per-tick press-and-release policy.
FiGAR (Sharma, Lakshminarayanan and Ravindran 2017) showed that a learned repetition length helps
policies whose actions must last. Holds in 0.7.2 run 0.1 to 3 s.

- **The buckets:** 0, 1, 2, 3, 4, 6, 8, 11, 16, 22, 31, 44, 62 and 90 ticks, up to 3.0 s at 30 Hz: the
  ruling's range of 0.1 to 3 s is 3 to 90 ticks, and a longer action (eating, 4 s) chains two holds.
  Above 4 ticks, each bucket is about √2 times the one before.
- **The rounding bound:** rounding a hold to the nearest bucket is off by at most (r − 1)/(r + 1) of its
  length. For the widest ratio, r = 1.5, that is 20 %.
- **What the test checks:** every shortbow draw tier is reachable.
  [test: test_hold_buckets_reach_every_shortbow_draw_tier]

**Dodge in gen 7.** The 0.7.2 assets allow dodging left and right only. Gen 7 therefore disables the
gen-5 dodges (locomotion choices 33 to 36) and dodges through the `dodge` head (none, left, right).
[test: test_gen7_disables_exactly_the_four_gen5_dodges_and_keeps_two_in_the_dodge_head]

**Observations are token sets, and the target is marked inside its set.** One network can then point at
any entity, cell or slot without a fixed slot per target. The tokens are permutation-invariant set
inputs (Deep Sets, Zaheer et al. 2017; Set Transformer, Lee et al. 2019). Pointing by marking is a
pointer network's output over its inputs (Vinyals, Fortunato and Jaitly 2015).

**Positions are in the agent frame,** so a policy learned at one place and heading applies at every
other (inferred: equivariance by construction, not measured):

```
forward = (−sin yaw, 0, −cos yaw),   right = (cos yaw, 0, −sin yaw),   up = y      (gym reset.py:513-516)
```

[test: test_agreement_with_the_gym_formula_on_a_seeded_batch]

### 2.3 Typed requests and their statuses (`contract/protocol.py` over the middle layer's `requests.py`)

**The request.** It carries a verb and four typed argument kinds, each with a presence flag: position
(world), direction (unit), entity handle, and cell with face. It also carries up to four named scalars,
a mode (parallel, after:<id> or interrupt), and the arbitration fields. The owner ruled for typed
conditions after fine-tuned encoders failed to read them reliably from text (commit e865049).

**What the check cannot see.** The kinds region, item, recipe and route stay context: the check
cannot validate them, so they reach a body as context, not as checked arguments.
`protocol.verb_spec` turns each catalogue schema into the middle layer's `ArgumentSchema`. The
"x, y or z" lists become one any-of group. [test: test_one_of_arguments_are_checked_in_the_middle_layers_order]

**The statuses** follow ROS actionlib's goal states, a long-used protocol for preemptible goals with
feedback. The mapping (ours):

| actionlib | here |
|---|---|
| PENDING | deferred(conflict_with) |
| ACTIVE | accepted, then running |
| SUCCEEDED | done(COMPLETED) |
| REJECTED | rejected(reason) |
| ABORTED | aborted(reason), or done(another outcome) |
| PREEMPTED | aborted("interrupted: by request <id>") |

Each accepted or running status names the heads it drove that tick, so the middle layer can see which
request moved the body.

### 2.4 The combiner: arbitration per scope, atomic grants (`runtime/combiner.py`)

**Arbitration, not fusion.** Pirjanian's survey (1999) divides behaviour coordination into arbitration
(one behaviour wins a resource, as in Brooks's 1986 subsumption) and command fusion (votes are
combined, as in DAMN, Rosenblatt 1997). Fusion needs commands that can be averaged. Discrete head
choices from two skills cannot be: there is no midpoint between "attack" and "use". So every head has
exactly one driver. [test: test_no_head_is_ever_shared_and_the_compiled_table_gives_the_host_action]

**Order.** Requests are taken in descending order of

```
key(r) = (urgency, confidence, priority, issued_tick, request_id)
```

- urgency comes first: emergencies win;
- recency breaks ties, so the newest intent wins;
- the id makes the order total, so a run is reproducible.

[test: test_the_greedy_order_is_urgency_confidence_priority_recency_and_reproducible]

Grants are greedy in this order. That gives the lexicographically best set: no lower-key request ever
displaces a compatible higher-key one. Maximising a weighted sum instead would be weighted set packing,
which is NP-hard, and its trades would be hard to explain.

**Atomic grants cannot deadlock.** A request is granted only when every head of every one of its scopes
is free. Otherwise it is deferred and names the holder of the first taken head.
[test: test_two_movement_requests_conflict_and_the_loser_waits_for_the_holder]

A deadlock needs four conditions at once: mutual exclusion, hold-and-wait, no preemption and circular
wait (Coffman, Elphick and Shoshani 1971). A request here never holds part of its scopes while it waits
for the rest, so hold-and-wait is absent and the combiner cannot deadlock. This follows from the code's
structure; no test exercises it directly.

A strict priority order can starve a request. The middle layer bounds starvation with aging: a step
deferred past its hold gains 0.25 urgency and is dropped at 1.0 (`jev/correspondence.py`). Aging is the
standard remedy in priority scheduling.

**Holders keep their scopes until they finish.** A granted request keeps its scopes until it is done,
aborted, withdrawn or expired. It is re-checked every tick, and a failed check aborts it. Keeping the
grant stops the body flipping between two requests with nearly equal keys (inferred). This is the same
reason the proof-of-concept skills carry a minimum commitment.

A verb that owns no scope (idle, which drives nothing) is granted without holding a head. It runs and
streams its statuses like any request, and never blocks another.
[test: test_a_request_whose_verb_owns_no_scope_runs_and_never_blocks]

Interrupt mode is the explicit preemption. An interrupt takes heads only from requests granted on an
earlier tick, so two requests issued on the same tick are ordered by their key alone.
[test: test_an_interrupt_pre_empts_an_earlier_holder_but_not_a_grant_of_the_same_tick]

Mode `after:<p>` defers a request until request p is done; this is skill chaining's link, where one
option's termination is the next one's start (Konidaris and Barto 2009).
[test: test_after_chains_run_in_order_and_parallel_sets_co_run]
A done predecessor is remembered for `done_memory_ticks`, counted back from the oldest live request (default
900 ticks, 30 s at 30 Hz). A finished request that has left the stream is forgotten. So the combiner's
memory and work per tick follow the live requests, however long the episode.
[test: test_done_ids_are_remembered_for_the_configured_window_only]

**The in-graph form.** The active-request table (B agents × S scopes: request id, verb, typed arguments,
ticks since grant) lowers the host's grants to arrays, so a compiled runner gives the same action as
the host. [test: test_no_head_is_ever_shared_and_the_compiled_table_gives_the_host_action, gate R6]

### 2.5 Programs: sequence, parallel, repeat (`runtime/programs.py`)

**The three forms.** A program is a list of stages. Stage k + 1 starts only when every step of stage k
is done. The steps of one stage run together, and a `(for each ...)` step repeats. These are the
sequence and parallel nodes of behaviour trees (Colledanchise and Ögren 2018; Halo 2's behaviour DAG,
Isla 2005), ordered HTN methods (Erol, Hendler and Nau 1994) and policy sketches (Andreas, Klein and
Levine 2017).

**The binder supplies places.** A program names which verbs run and in what order. Where each one acts
comes from a binder: `binder(step, verb, iteration) -> RequestArgs | None`. The middle layer's binder
reads the belief and the world memory. Code as Policies also separates the two, with a program calling
perception functions for its arguments.

**Failure goes up.** A rejected or aborted step fails the program, and the failure names the step and
the reason. There is no fallback node. Recovery has one owner, the middle layer, which already
re-samples with reasons (SYSTEM_ONE §9.9; the correspondence rules), so a second recovery path would
only compete with it. [test: test_a_failing_step_fails_the_program_with_its_reason]

**Parallel steps own disjoint scopes.** Steps of one stage must not share a scope. If they did, the
combiner would defer one of them, and "together" would quietly become "one after the other". The
catalogue's 19 programs meet this rule. [test: test_the_steps_of_a_parallel_stage_own_disjoint_scopes]

**Limit: no background node.** `fight_group` keeps range and guards in its first stage and attacks
each threat in its second. The attacks therefore start only after keep_range and guard have both
finished. A step that should last across stages is the middle layer's to issue as a request of its own.

### 2.6 Parity, traces and fidelity (`runtime/parity.py`, `training/traces.py`, `docs/FIDELITY.md`)

**Parity is differential testing** (McKeeman 1998): the same inputs go through the old and the new path,
and every non-timing metric is compared.

- Today's one goal per lane, run as one request through the combiner path, gives identical metrics on
  the four-arm composition smoke.
- A negative control shows the adapter really is in the loop. [ledger: A6]
- Parity is gen-5 only: it wraps today's dispatcher on the twelve heads.

**Native NPC traces warm-start the bodies by behaviour cloning** (Pomerleau 1989). Cloning alone drifts
off the demonstrated states, and DAgger corrects that drift (Ross, Gordon and Bagnell 2011); this project
already needed it once [ledger: P1d]. Every trace carries a receipt (server version, bridge sha256,
world, seed), so it is never used against the wrong ruleset.
[test: test_arrays_round_trip_through_npz_and_report_their_receipt]

**The fidelity ledger closes the reality gap one mechanic at a time** (Jakobi, Husbands and Harvey
1995). Each mechanic gets a native differential on the live 0.7.2 server; replaying a policy does not
count as fidelity. Randomising a gap (Tobin et al. 2017) is allowed only after a differential has named
that gap.

### 2.7 The trunk: a token-set encoder (`model/encoder.py`, B1)

**What it reads.** Every body reads its agent's token sets through one trunk: entity, block candidate,
inventory slot, geometry, self and kit tokens (`surface.TOKEN_TYPES` names), any number of each.

```
h      = MLP_t(features) + type_t + marked · m      one small MLP per token type; learned type and mark vectors
X      = pre-LayerNorm transformer blocks over all valid tokens (masked multi-head attention, then an MLP)
context = Dense(concat(mean, max, marked mean, self token))     tokens[t] = the per-token embeddings
```

**Why this shape.**
- **Sets, not slots.** The tokens are a set, so the trunk is permutation-invariant and its parameters
  never depend on how many tokens there are (Deep Sets, Zaheer et al. 2017; Set Transformer, Lee et al.
  2019). Attention lets each token see the others (Vaswani et al. 2017); pre-LayerNorm blocks train stably
  without warm-up tricks (Xiong et al. 2020).
- **Marking is how one network points.** The instructed target is marked inside its set, and the marked
  mean is a separate readout, so the target is never averaged away. A pointer head indexes `tokens[t]`
  (Pointer Networks, Vinyals et al. 2015).
- **Masking is exact.** A padded slot is zeroed before the first layer and is never attended to, so
  nothing a padded slot holds, even NaN, changes an output.
  [test: test_sa_encoder.py, padding, permutation and capacity invariance]

**The 30 Hz budget.** The owner set p95 <= 15 ms for the whole per-tick path, inside the 33.3 ms tick.
`bench/bench_tick.py` times the lower layer's share: the resolver, the combiner and the encoder.

| batch | p95 per tick |
|---|---|
| 1 | 0.65 ms |
| 8 | 2.29 ms |
| 64 | 12.1 ms (one 40 ms spike) |

These were measured on 2026-10-17 at a machine load of about 21 on 18 cores [ledger: B1]. The bodies
were stand-ins, so a trained body's own heads come on top of the encoder's cost.

**Trained bodies with the trunk in every call cost about 1 ms per agent per skill** (`bench/bench_bodies.py`,
2026-10-18 [ledger]). That was the design until the owner's ruling later that day. A trained body was a
freezing verb (section 2.9): the trunk and the verb's adapter in one compiled program. The combiner calls
each granted body once per agent per tick, so every call ran the trunk again on one agent. The bench times
that path against two batched layouts that choose the same factors [test: test_sa_bench_bodies.py].
Batch 8, p95 per tick in ms, with every agent holding the same skills (shared) or its own (distinct):

| skills per agent | trunk per call (shared / distinct) | trunk once per tick, adapter per agent and skill | trunk once, adapters batched per skill (shared) |
|---|---|---|---|
| 1 | 7.1 / 7.7 | 2.8 + 0.3 | 2.7 + 0.3 |
| 2 | 8.4 / 8.6 | 4.4 + 0.3 | 1.5 + 0.4 |
| 3 | 12.5 / 13.6 | 5.8 + 0.5 | 1.7 + 0.5 |
| 6 | 24.5 / 44.9 | 9.0 + 1.0 | 1.9 + 1.0 |

The first column includes the combiner. The other two are the bodies' work plus the combiner's
bookkeeping alone (`StubBody`), measured separately and summed.
- **With the trunk per call, the bar holds up to three skills per agent at 8 agents**, and leaves little
  room for the belief and the decision. At six it fails, and with distinct skills it exceeds the 33.3 ms
  tick.
- **One trunk pass per tick keeps every measured case within the bar** (worst 10.0 ms). Each verb's
  adapter stays compiled alone and never stacked, so freezing's guarantee is unchanged.
- **Changed by the owner's ruling (2026-10-18, late).** The trunk runs once per tick:
  `freezing.encode_contexts`, then `VerbBody` runs the adapter alone (section 1). The bench's `encoded`
  layout times that path through the combiner.
  - Its first run was at a machine load of 20 to 59, which inflated every layout about tenfold, so its
    absolute numbers are not used.
  - In that run, at 8 agents the encoded path's median was 2.5 to 7.5 times below the trunk-per-call
    path's: 4.9 to 7.5 times at 3 and 6 skills (6 skills, shared: 25.9 against 171.3 ms).
  - With one agent holding one skill it was slightly slower (0.8 times as fast), because it makes two
    calls where the old path made one.
  - On a quieter machine (load about 10), batch 8 p95 through the combiner is 5.78 / 4.60 ms at 3 skills
    (shared / distinct) and 6.32 / 6.50 ms at 6, against 24.4 / 21.1 and 39.5 / 39.2 ms for the trunk per call
    in the same run. That is within the bar at 6 skills per agent.

Caveats: freezing's default sizes (trunk width 64, two layers, context 128; adapter hidden 64), random
weights, CPU, one run at a load of about 6 on 18 cores.

### 2.8 Behaviour cloning from the native traces (`training/bc_dataset.py`, `training/bc_smoke.py`, B2)

**The dataset.** The 32 native 0.7.2 traces become 21,088 rows, one per native-controlled frame, at the
game's 1/30 s (11,104 on flat, 9,984 on the generated world). Each row holds:
- the NPC's self token and the opponent as a marked enemy token, both in the agent frame;
- labels on the 21-head gen-7 layout.

**Labels match arena's own projection exactly.** They follow `arena.training.imitation.native_npc_demonstrations`
(`project_attack_execution=True`), mirrored here without importing arena. A differential test finds every one
of the 84,352 labelled cells equal on all 32 traces [test: test_sa_bc_dataset.py]. Four heads carry labels:
- the movement compass, from body steering;
- yaw and pitch deltas, from head steering;
- attack, only from an executed Primary interaction.
Every other head is neutral and unsupervised.

**The smoke** trains the B1 trunk plus one linear head per labelled head on one world, and reports on the
other. 1,500 steps take about 6 s on CPU. Held-out agreement against the majority-class baseline:

| head | gen held out, seed 0 | seed 1 | flat held out | native jitter |
|---|---|---|---|---|
| locomotion compass | 0.648 (base 0.593) | 0.670 | 0.771 (base 0.745) | 0.047 / 0.023 |
| yaw delta | 0.891 (base 0.905) | 0.880 | 0.908 (base 0.885) | 0.039 / 0.032 |
| pitch delta | 0.927 (base 0.934) | 0.922 | 0.981 (base 0.983) | 0.055 / 0.030 |
| attack | 0.988 (base 0.988) | 0.988 | 0.988 (base 0.989) | 0.024 / 0.022 |

**Reading.**
- **Plumbing:** run as a skill's body through the combiner, the trained network gives exactly its own
  choices (0 mismatches in every run).
- **Competence:** not shown. Train agreement is about 0.97 to 0.99, but held out it stays near the
  baselines. Only the compass beats its baseline every time, by 2.6 to 7.7 points. Attack, 1.1 % of the
  rows, is never learned.
- **Why (v1, v2):** a self token and an opponent token cannot show terrain or the NPC's own internal state.
  v3 adds terrain; see below.

**Terrain, meters and visibility (v3, 2026-10-17, with the owner's go-ahead).** Each row now also carries
what the traces record about the NPC's surroundings:
- **Terrain:** up to 64 geometry tokens, the solid, fluid, climbable or damaging cells of the 9x9x9 grid
  around the NPC nearest to it. They come from each frame's geometry record (`hytalegym.geometry.parse_geometry`).
  The grid's origin is the NPC's floored position, so a cell's centre is origin + offset + 0.5.
  - The features are the agent-frame offset, solid, climbable, a fluid one-hot (the gym has no
    fluid-id-to-name table, so the kind is left unknown), hazard and top_walkable.
- **Meters:** stamina, mana, signature energy and oxygen on the self token, from the frame's actor evidence.
- **Visibility:** visible and last-seen age on the opponent token, from `target_perceptible`.

The labels are identical to v2. In these traces the meters never change (full, or 0 for signature energy),
the target is perceptible on 99.6 % of rows, and every frame has more than 64 candidate cells (mean 318),
all plain solid ground. So the terrain tokens describe the floor more than any obstacles.

The control: the same network and initial parameters, with the geometry tokens present or masked out
(`bc_smoke --mask-type geometry`). Mean held-out agreement, with geometry / masked / baseline:

| held out | compass | yaw | body yaw | pitch |
|---|---|---|---|---|
| generated world (2 seeds) | 0.641 / 0.632 / 0.593 | 0.901 / 0.891 / 0.905 | 0.793 / 0.714 / 0.772 | 0.917 / 0.897 / 0.934 |
| flat (3 seeds) | 0.686 / 0.765 / 0.745 | 0.897 / 0.910 / 0.885 | 0.817 / 0.838 / 0.792 | 0.983 / 0.980 / 0.983 |

Reading:
- **Terrain is information.** On the generated world it lifts body turning by 0.079, above its baseline.
- **It also costs.** Trained on varied terrain and tested on a uniform floor, the compass drops by 0.079,
  with a wide seed spread.
- **The likely weakness is the token selection** (inferred): the 64 nearest cells are mostly floor. A
  selection of obstacles and height changes, and more worlds, are the next things to try. docs/BUILD.md 7b
  records the choice.

**The compass was labelled in the wrong frame (v4, 2026-10-18).**
- *The finding.* The traces' steering translation is a world heading: the NPC's next velocity points along it (median
  0.0 degrees apart on the 5,445 moving native frames). The gym's compass is body-relative: choice d moves along the
  body angle (d - 1) x 45 degrees (`apply_world_move_override`, `step.py`).
- *The bug.* Arena's rule, which v1 to v3 mirror, takes the world heading's sector, and that is right only at body yaw
  0. The NPCs' median body yaw while moving is 107 degrees, so 90.6 % of the moving frames had the wrong label.
- *The fix.* v4 takes the sector in the commanded body frame: the body yaw plus the body-yaw label's decoded delta.
- *The proof.* Decoded through the engine's rule, v4's labels land within 45 degrees of where the NPC went on 93.6 %
  of those frames (median error 0.1 degrees). Arena's rule reaches 15.6 % (median 90.7 degrees).
  [test: test_the_compass_label_reproduces_the_native_heading_through_the_engines_rule]
- *The effect.* Same smoke, three seeds, held-out agreement of the compass against the majority baseline:

| held out | v3 (world frame) | v4 (body frame) | baseline |
|---|---|---|---|
| generated world | 0.642 | 0.764 (0.759, 0.770, 0.763) | 0.593 |
| flat | 0.686 | 0.766 (0.670, 0.858, 0.772) | 0.745 |

So the compass was held back by its labels as well as by the data. Arena's own imitation stack has the same rule
(`arena/training/imitation/native_imitation.py`, `_compass_choice`), which is outside this folder and reported to the
owner. `--arena-only` (`body_frame_compass=False`) keeps arena's labels exactly for the parity test.

**The head's yaw offset joins the self token (v5, 2026-10-18 late).** The traces carry the head yaw, and the
generalist needs it (2.16), so the cloning dataset carries it too. v5 equals v4 apart from that column. The control
is v5 with the column zeroed: the same network and initial parameters. Held-out agreement, three seeds, offset /
zeroed / baseline:

| held out | compass | yaw | body yaw | pitch |
|---|---|---|---|---|
| generated world | 0.761 / 0.765 / 0.593 | 0.896 / 0.888 / 0.905 | 0.788 / 0.796 / 0.772 | 0.920 / 0.904 / 0.934 |
| flat | 0.814 / 0.779 / 0.745 | 0.918 / 0.903 / 0.885 | 0.836 / 0.834 / 0.792 | 0.982 / 0.980 / 0.983 |

On the flat world the offset helps the yaw head on every seed (+0.015) and the compass on every seed (+0.035, with a
wide spread). On the generated world the differences sit inside the seed spread. The trained body still runs through
the combiner with 0 mismatches. The v4 file no longer loads, because its self tokens are 12 wide.

Known limit: the opponent's position comes from its true state even on the 0.4 % of rows where it is not
perceptible. The legal observation mode would hide it.

**Body turning (v2, 2026-10-17, with the owner's go-ahead).** Arena's projection has no rule for the
body's own turn, though every frame steers the body and 13,593 of 21,088 request a body yaw. The dataset
now adds one, built like the head-yaw rule: on a body-steering row, the bin nearest the requested body yaw
minus the current one, else the centre. The gym decodes both heads with one codebook.
- `build_dataset(body_yaw=False)` still gives exactly arena's labels, and the differential test checks the
  other heads against arena as before.
- The label is supervised on all 21,088 rows; the centre bin is 78 %, and native jitter is 0.054.
- Held out, the smoke's body-yaw agreement is 0.733 and 0.725 (two seeds, base 0.772) with the generator
  world held out, and 0.864 (base 0.792) with flat held out.
- Like the other heads, it learns the training world and not the other one.
[test: test_the_body_yaw_rule_labels_the_requested_body_turn; test_labels_equal_arena_projection_on_every_trace]

### 2.9 Freezing: adding a skill never changes a frozen one (`model/freezing.py`, B4, proof E13)

**The mechanism.** One shared trunk (the B1 encoder) and one small adapter per verb:
`logits_v = Adapter_v(Encoder(x))`. A verb's compiled program takes only the trunk and its own adapter as
arguments, and is cached per verb. Adapters are never stacked across verbs, so a verb's program does not
depend on how many verbs exist.
- Training a verb differentiates only its adapter, plus the trunk while the trunk is not frozen.
- A frozen verb cannot be trained.
- `params_sha256` fingerprints the weights.

This is the weights half of requirement 7; `runtime/library.py`'s version rule is the contract half. It is the
frozen-trunk-plus-adapters pattern of parameter-efficient transfer (Houlsby et al. 2019). Keeping a frozen
skill's weights untouched is the strongest guard against catastrophic forgetting (McCloskey and Cohen 1989).

**The proof.**
- **The gate:** walk and look are trained, then frozen with the trunk. Swing is added and trained 50 steps.
  Walk's and look's logits on a fixed input set are bit-identical (largest difference exactly 0.0), and
  their weights' and the trunk's fingerprints are unchanged.
- **The negative control:** the same run with the trunk not frozen moves the frozen verbs' logits by up to
  3.87 through the trunk, while their adapters stay unchanged. So the gate can fail.
- **The encoded path (trunk once per tick, 2026-10-18).**
  - With the trunk frozen, adding and training swing leaves walk's and look's adapter logits on the
    encoded contexts bit-identical, and the contexts themselves too.
  - The encoded and raw paths choose the same factors on all 64 agents, with logits within 1e-5.
  - A body refuses a context that another trunk encoded.
- [test: test_sa_freezing.py]

**Assumption.** Bit-identity is shown on CPU for these shapes. Another backend or JAX version should be
re-checked with the same test.

### 2.10 Progress and stop reasons in every contract (B6)

**What a verb reports.** Every verb contract lists the typed progress signals the verb reports into the
status stream:
- `progress`: the fraction toward its done condition, 0 to 1. The middle layer's stall rule reads this.
- `elapsed_s`: seconds since the grant.
- One measure by the done text's family, matched on whole words: `distance_m` (arrive, reach), `hits`
  (hit, land, down), `count` (gained, placed, crafted) or `angle_deg` (centred, facing, found).

On the catalogue, 72 of 122 verbs get a family measure (hits 27, distance 22, count 20, angle 3), and 50
report the universal pair only.

**Why a verb stopped.** A body that finishes reports `BodyStep.signal`, one of its contract's termination
labels: the success label from the done column, or `stuck`, `timeout` or `precondition_failed`. The
combiner refuses any other label, and writes the label into `RequestStatus.signal`, the middle layer's
field. Two verbs that end with the same outcome class can still be told apart: arrived is not timed out.
Stand-in bodies report their contract's progress names and, on done, the success label.
[test: test_progress_signals_follow_the_done_texts_family; test_a_done_status_carries_the_bodys_termination_signal_and_a_foreign_signal_is_refused; every verb in test_sa_execute_all]

### 2.11 Scripted teachers for the simple track (`teaching/teachers.py`, 2026-10-18)

**What they are.** Pure JAX functions of the simulator's full state and the request. They return choices on the gym's
12 heads plus a done flag, for every lane at once (TRAINING_PLAN 2d). They will label the student's data.
- `walk_to_choices`: steers the body toward the target by one codebook step, then takes the compass sector in the
  commanded body frame that results. Gait comes from the request; sprint falls back to run off the forward direction.
  It stops inside the radius.
- `look_at_choices`: aims the head at the target point within the ±45° window around the body. It wraps across the
  ±180° seam before clamping, and reports done when the actual head is within tolerance.
- `TEACHERS` registers each with its verbs, kind, the fields it reads, its parameters, version and certificate
  (none yet).

**Measured** (flat world, `npc_walk`, 64 lanes):
- walk_to: 64 of 64 arrive within the horizon, median 54 ticks (1.99 s);
- look_at: 64 of 64 done within 90 ticks, median 11, p95 20;
- no illegal decision on either.

This is T1 and T3 of the certificate on a small sample, not the full certificate.
[test: test_sa_teachers.py, 13 tests; the open-loop factors are decoded by the engine's own decoder]

**Learned while building them:**
- The look codebook is regenerated for each `maximum_turn_degrees`, not scaled, so a teacher quantises against the
  decoder's own table.
- The 5-bin pitch head steps ±1° or ±45°, so pitch overshoots by up to 21.5° before settling.
- Done is per tick. A head swinging through the target can be "done" for one tick (1 of 64 lanes), so a data pipeline
  needs done for k consecutive ticks.

### 2.12 Flat tasks with the NPC body (`tasks/simple_tasks.py`, 2026-10-18)

**What it is:**
- `make_flat_npc_environment` builds the arsenal environment on a flat world with `npc_walk`, the body the bridge
  drives, and explicit spawns through reset's entity form.
- `WalkToTasks` and `LookAtTasks` sample goals with their horizons and success rules.
- `run_episode` rolls any policy that returns the 12 factors.

**Measured.** Run-forward moves at exactly 5.0 blocks/s along the body's forward vector after the ramp (the speed is
min(5, 10 k / 30) on tick k, so it reaches 5.0 at tick 15). [test: test_sa_simple_tasks.py, 10 tests]

**Line of sight: open-flat by default (2026-10-18).** The environment's own default capabilities are fail-closed:
they deny line of sight, so under the legal observation the other entity is never perceptible, even 5 blocks in
front. The flat tasks therefore default to `open_flat_arsenal_world_capabilities`, which is right for a world with no
blocks and is what the neural_agent's legal open-flat arena uses. `world_capabilities="fail_closed"` keeps the old
behaviour. The pair path applies no field of view, a simulator gap now in the simulator lane as an opt-in
(TRAINING_PLAN 2e; BUILD E12 defines FOV16 and FOV32).

**Two settings the defaults would get wrong:**
- The default motion-timing band gives 0.045 s ticks on two of three profiles (gen 5's loaded tick), so the
  environment pins profile 0, the 1/30 s tick of the 30 Hz ruling.
- Explicit spawns compose the carry from `reset_arsenal_batch`'s entity form. At the default spawn the composed carry
  equals the environment's own reset on all 342 leaves; other spawns are checked through the behaviour tests.

**Simulator note** (passed to the simulator lane): reset sets the head to yaw 0, not to the spawn's body heading,
despite its comment (`runtime/reset.py`:158-161). The head catches up in about 2 ticks.

### 2.13 The request-conditioned generalist (`model/generalist.py`, `bench/bench_generalist.py`, 2026-10-18)

**What it is.** The working design of `docs/ARCHITECTURE_RESEARCH.md`, built and untrained: one policy for every learned
verb.
- *Inputs.* Each granted request becomes a token: a verb embedding, plus its typed arguments (agent-frame position and
  direction, 4 scalars, mode, urgency, age), with absent argument groups masked, plus the granted scope's embedding,
  plus the trunk tokens its entity or block pointer names. Six learned scope queries join them.
- *Layers.* Each block is self-attention over [requests; queries], with invalid requests masked as keys, then
  cross-attention into the trunk's token sets, then an MLP.
- *Outputs.* Every head's logits come from its scope's query; status (done, stuck, success) and progress come from
  each request token.
- *Size.* 125,900 parameters, beside the trunk's 131,136.

**Proved** [test: test_sa_generalist.py, 21 tests, checked with mutants]:
- invalid requests and absent arguments change nothing (bit-identical);
- removing a padded request slot changes nothing (within 1e-5);
- pointers gather the named token;
- outputs are deterministic.

**Speed (A1): PASS.** At a machine load of about 10 on 18 cores, batch 8 p95 of trunk plus generalist is 3.56 ms
against the 15 ms bar (trunk 2.60, generalist 0.98; median 2.19). Batch 64 is 8.99 ms; batch 1 is 3.51 ms. The first
run's 15.08 ms (load 53 to 75) was the load.

### 2.14 The legal observation adapter (`model/observation_adapter.py`, 2026-10-18)

**What it is.** `legal_tokens(observation, *, capacities, body_yaw_degrees)` turns the environment's legal observation
into the trunk's token sets. It never reads the simulator state; the signature takes none.
- *Self:* meters, grounded, body yaw, the head's yaw offset from the body (read from the self row, whose yaw is
  the head's), pitch and speed.
- *Entity:* the opponent in slot 0 when the observation marks it perceptible.
- *Other types:* empty for now (no blocks on the flat world; inventory and kit come later).

**Frames, measured** (body 90°, head 45°):
- The opponent row is already in the body frame (`observation/v1/runtime/encoder.py`:108, :178), so the adapter does
  not rotate it.
- The self row's yaw is the head yaw (:323). The body yaw is not published as a value, so the caller passes it: it is
  the agent's own state. Omitting it is an error, never a silent head-yaw fallback.

**Leakage.** Two worlds whose hidden opponent sits in different places give identical tokens, and moving a visible
opponent changes them (the positive control). Perceptibility here means:
- alive;
- line of sight;
- within 16 blocks planar and 24 blocks in 3-D.

The pair path's field of view and sensor range are opt-in since c613d22 (docs/BUILD.md E12); the flat tasks leave
them off.
[test: test_sa_observation_adapter.py, 13 tests; the first 10 mutation-checked]

**What the legal observation does not provide:**
- roles other than hostile;
- a last-seen position or age (hidden means an empty slot);
- held-item durability;
- the speed units in blocks per second.

Positions are quantised to 0.1 block.

### 2.15 Teacher certificates (`teaching/certificates.py`, 2026-10-18)

**What it is.** `certify_walk_to` and `certify_look_at` run a teacher in closed loop on the flat tasks and write a
JSON certificate (TRAINING_PLAN 2c):
- T1, success with Wilson intervals;
- T2, time against the kinematic minimum d / v + v / (2a);
- T3, pace, radius and legality;
- T7-lite, failure classes and done hysteresis.

The CLI is `python -m experimental.neural_agent.skill_agent.teaching.certificates --teacher walk_to --seeds
77001,... --out PATH --date YYYY-MM-DD`. [test: test_sa_certificates.py, 7 tests; a deliberately idle teacher fails T1]

**First certificates (4 seeds x 256 episodes, flat `npc_walk`):**

| teacher | T1 success | T2 | T3 | failures |
|---|---|---|---|---|
| walk_to v1 | 1,024 / 1,024 (Wilson [0.9963, 1]) | median 1.0004 x the minimum, p90 1.006 | pace within 5 % on all 1,002 steady lanes; 0 illegal of 207,561 decisions | none; done never blinks |
| look_at v1 | 1,024 / 1,024 | median 13 ticks, p95 20 (no bar: the codebook sets the pace) | tolerance at done on all lanes; 0 illegal of 92,160 | none |

**What they do not show:**
- On a flat world with point goals, the scripted teachers are essentially perfect, so T1 cannot separate them, and
  T2's reference is the engine's own kinematics, not a native NPC.
- The checks that test a teacher are still to come:
  - T4, the live replay;
  - T5, whether a student can follow it;
  - harder distributions.
- By the owner's bar (TRAINING_PLAN 1b), a student can only match these scripts on easy solo tasks; it has to win on
  combined and disturbed ones.

**T4, the live replay, for walk_to v1: PASS** (`teaching/certify_live.py`, 2026-10-18, jar 460054A8). The teacher ran in
closed loop on the live 0.7.2 server, reading the native position and tracking its commanded body yaw, and the
simulator teacher ran from the same start and goal:
- 24 of 24 live and 24 of 24 simulated, 0 illegal decisions;
- the live runs end inside the radius;
- live runs finished a median 2 ticks sooner (at most 8).

The live per-tick step is 0.1667 blocks (the 1/30 s step) on its quietest ticks and 0.1748 at the median, and the
ramp matches an acceleration of about 10 blocks/s^2. So the physics agree, and some live ticks ran longer than 1/30 s
under load. look_at's T4 is not run yet.

### 2.16 DAgger distillation into the generalist (`training/distill.py`, 2026-10-18)

**What it is.** The trainer for A2, the first student: the generalist (2.13) learns walk_to and look_at from the two
scripted teachers (2.11) on the flat tasks (2.12), and reads only the legal observation (2.14). The teachers read the
full simulator state, but they only label; the student never sees that state.

**How it trains.** DAgger (Ross, Gordon and Bagnell 2011):
- each iteration runs a batch of episodes, the teachers label every state visited, the labels join one aggregate
  dataset, and Adam fits the student on the whole aggregate;
- iteration 0 runs the teachers (beta = 1). From iteration i on, a lane follows the teacher for its whole episode with
  probability 0.5^i and the student otherwise, so later data comes from the student's own mistakes;
- each lane runs one mode: walk only, look only, or both at once (the composition the owner's bar asks for);
- the requests are rebuilt at every decision from the world goal and the agent's own pose, in the body frame
  (positions over 16 blocks): the agent may know its own pose;
- the loss is the cross-entropy on the heads of each request's scope (walk: locomotion; look: gaze), plus 0.5 x the
  binary cross-entropy of each request's done logit.

**What is held out.** Two argument regions are never trained on: walk targets in the rear-left quadrant of the spawn
body (bearing 180 to 270 degrees) and look points 25 to 40 degrees off the body. The evaluation runs the student and
the teachers on the same seeds, seen and held out, and reports success per mode, walk pace (within 5 %), radius and
illegal decisions. [test: test_sa_distill.py, 9 tests]

**Status: built; the two prep changes are in; A2 has not run.** A 2-iteration smoke (32 lanes, 50 steps per iteration,
10,191 samples) ran end to end with 0 illegal decisions, before the changes below. Its student was untrained by
design: walk 0.23 and look 0.0, against the teachers' 1.0 and 1.0. [test: test_sa_distill.py, 11 tests]
- **The head's offset from the body is in the self token** (`SelfToken.head_yaw_offset`, 13 features, 2026-10-18
  late). Without it, look_at could not tell when its gaze was on target: the self token carried the body yaw, and the
  head moves up to 45 degrees away from it. The adapter reads the head yaw from the self row's sin and cos (body 90 and
  head 45 give -pi/4) [test: test_sa_observation_adapter.py]; the cloning dataset reads it from the traces (v5).
- **The training batches drop the four token types the flat world never fills.** The loss is unchanged (gap 0.0 on a
  fixed batch; largest gradient gap 7.5e-9), and a step at batch 1024 takes 0.045 s instead of 0.43 s
  (9.6 times faster; one process, interleaved, load about 15). [test: test_loss_and_gradients_do_not_change_when_the_empty_token_types_are_dropped]
- **The report records its provenance:** the commit, uncommitted changes in this package, the command line, the load
  at the start and end, and the field-of-view settings (off here).

## 3. For the agent building the gen-7 simulator

**What you receive.** Each tick you get one choice per head per agent, in this order. A head no request
owns carries its neutral choice. The first twelve heads are the gym's own, with the same names, sizes
and order. The nine planned heads, with their choices from `contract/surface.py`, are:

| head | size | scope | choices |
|---|---|---|---|
| `duration` | 14 | hands | no_hold, then hold k ticks for k in 1, 2, 3, 4, 6, 8, 11, 16, 22, 31, 44, 62, 90 |
| `item_move` | 5 | hands | none, to_hotbar, to_bag, to_container, from_container |
| `item_drop_give` | 4 | hands | none, drop_one, drop_stack, give |
| `recipe` | 17 | interaction | none, candidate_1..16 |
| `door` | 4 | interaction | none, open, close, toggle |
| `mount` | 6 | locomotion | none, mount, dismount, glider_deploy, glider_stow, glider_dive |
| `rune_set` | 9 | abilities_magic | none, rune_slot_1..8 |
| `cast` | 14 | abilities_magic | none, then cast-hold k ticks with the duration buckets |
| `dodge` | 3 | posture_defence | none, left, right |

The full layout is 21 heads and 200 choices. Locomotion choices 33 to 36 (the gen-5 dodges) are illegal
on gen 7. One tick is 1/30 s, the game's own step (the owner's 30 Hz ruling, 2026-10-17;
`surface.tick_seconds("gen7")`); 0.045 s is the gen-5 simulator's tick. The whole per-tick path (belief,
decision, combiner, bodies) has a budget of p95 <= 15 ms at the deployed batch, inside the 33.3 ms tick.

**What the lower layer needs back, each tick and for each agent:**

1. **A per-choice legality mask for every head you implement.** `protocol.legal_scopes` derives the
   legal scopes from it: a scope is legal when it has at least one present head and every present head
   has a legal choice.
   - Leave a head out of the mask until it exists. If a scope has no present head, its requests are
     rejected with scope_illegal (this is the case for interaction on gen 5).
   - Do not stand in for a missing mechanic with a present head whose only legal choice is neutral.
     That pattern means "a cooldown": the scope stays legal and the request waits.
   - [test: test_legal_scopes_follow_the_action_mask]
2. **The token sets, with their `FEATURES` from `contract/surface.py`.**
   - Fill every position in the agent frame (section 2.2), and keep entity handles stable across ticks:
     requests point at entities by handle.
   - Block candidates are matched on (cell, face) exactly; the faces are Hytale's six BlockFace names.
   - The token types are entity, block candidate, inventory slot, geometry, light, self, kit and route.
   - They reach the bodies as `AgentContext.observation_tokens`, with the belief tokens beside them
     (section 1).
3. **The world bounds and the known entity handles** (`AgentContext`). The check rejects points outside
   the bounds and handles it does not know.

**When a planned head lands:** turn its `HeadSpec.status` to `published` in the same commit that adds its
`docs/FIDELITY.md` row with the differential's receipt. The pass bars are in `docs/BUILD.md` §4.

**The adapter between the combiner and the simulator** (`sim_adapter.HeadAdapter`, 2026-10-17).
- Build it from the simulator's mask (`HeadAdapter.from_mask`): the heads the mask names are the
  implemented ones.
- `adapt`, or `adapt_batch` for B agents, writes neutral for every head the simulator does not implement.
  It counts each non-neutral choice it drops in `dropped`, per head, so a skill that depends on a missing
  head shows up instead of looking like a skill that fails.
- `HeadAdapter.legal_scopes(mask)` gives the scopes legal under the mask, by the rule above.
- Implemented heads pass through unchanged; checking each choice against the mask is still to do.
[test: test_sa_sim_adapter.py, including a combiner round trip in which walk_to's placeholder fills the
planned mount head and the gen-5 mask drops it]

**What still speaks the old contract:**

- `runtime/parity.py` and the ADK proof-of-concept skills (idle, pursuit, evader): twelve heads.
- The JAX environment's action decoder: twelve heads, on gen 5 and on the gen-7 render alike. The overlay
  adds none of the nine planned heads, its dodge-head construct is a stub, and the policy layout publishes
  no recipe head (`observation/v3/policy/layout.py`: "Crafting is out of scope for a combat agent").
- The 0.7.2 NPC traces: these use the server's NPC state (14 columns) and control (9 columns) layouts,
  not policy heads.

## 4. For the agent built on top: what it can execute today

**Every verb runs.** All 122 catalogue verbs are accepted on the gen-7 layout, drive exactly the heads
of their own scopes, and report done. [test: test_every_verb_is_accepted_drives_only_its_own_heads_and_finishes]

**Every program runs.** All 19 programs run to done through the same path, with nested programs inlined
and each repeat run for two iterations. [test: test_every_activity_program_runs_to_done]

**The scratchpad carries everything.** Every request and its statuses survive the JSON scratchpad, in
the requests section and the skills section.
[test: test_every_request_and_its_statuses_round_trip_through_the_scratchpad]

**A deliberate break is caught.** Each of these four makes one of the tests fail: a placeholder that
fills nothing, requests with no arguments, a repeat that never advances, and an un-inlined nested
program [ledger].

**Running is not competence.** Of the 122 verbs:

- 121 are placeholders: a deterministic stand-in that fills the verb's heads and reports done after one
  tick. A placeholder proves the plumbing, not the skill.
- 1 has a real script: `switch_item`, which sets the hotbar.
- The trained proof-of-concept skills run on gen 5 through the in-graph path (A6).

`library.placeholders()` lists the stand-ins, so the agent can always tell which verbs are real.

**What the agent reads:**

- `library.descriptions()` — the matching text for every verb and program.
- `protocol.request_schemas(library.specs())` — the `schemas=` argument for `requests.validate`.
- `library.example_args(verb)` — an argument set that satisfies the verb's schema.
- `library.scope_heads(verb, layout)` — the heads a verb's request owns.
- The status stream: the reasons, the conflicts and the heads driven.

**Through the scratchpad (B3, with the middle layer).**
- `scratchpad.combine_from_scratchpad(state, combiner, tick, contexts)` runs the combiner on S_t. It reads
  each agent's live requests from the `requests` section (`state.requests.for_agent`) and appends every
  status to the `skills` section's stream, in one write per tick as writer "skills".
  [test: test_requests_from_the_section_are_combined_and_their_statuses_land_in_the_skills_section]
- `SkillLibrary.from_catalogue(proof_of_concept=True)` adds idle, pursuit and evader the way the
  middle layer's program executor requests them.
  - Their timing and commitment come from the catalogue.
  - Their arguments and scopes come from the request registry, which the in-graph parity path also
    grants: pursuit and evader take an entity, idle takes nothing.
  - Their stand-in bodies never report done, because the skills themselves run in-graph and end there.
  - [test: test_the_proof_of_concept_skills_register_as_the_executor_requests_them]
- **Wiring, with both halves on.** `scratchpad.CombiningDecider` wraps the middle layer's program
  executor (the `jprog` arm). The session binds its attributes as it does on the executor. Each decision
  returns the executor's goal unchanged, then runs the combiner on S_t at the session's clock tick.
  - Each lane's context marks the belief's single target (entity 0) known when the belief has seen it
    within saturation, the executor's own rule.
  - It also implements the middle layer's per-tick hook: the session calls a decider's `on_tick` on every
    tick. So the combiner runs on every tick, once per tick even when a decision happened on it, and
    `running` statuses reach the stream between decisions. A session without the hook combines on
    decision ticks only. [test: test_the_per_tick_hook_combines_every_tick_once_and_streams_running_between_decisions]
  - Once the hook has been seen, the decision itself does no combining, so the combiner's time stays out of
    the session's timed decision budget. When the combiner ran inside the decision, a 512-tick run at the
    20 ms budget had 3 misses where the baseline had none, and the run diverged.
    [test: test_once_the_hook_is_seen_decide_leaves_the_combining_to_it]
  - [test: test_the_combining_decider_forwards_the_session_bindings_and_runs_the_combiner_after_each_decision]
- **The joint check** (`runtime/joint_check.py`, B3's gate). It runs the composition arms flat, j0, shuffled and
  jprog-J0 (64 ticks, batch 8, seed 0) with and without the wrapper.
  - Every non-timing metric is identical.
  - In the jprog run the combiner wrote 232 statuses: 120 accepted (one per lane per decision) and 112
    withdrawn (each decision replaces the last request).
  - There were no rejections, so the executor had nothing to correct [ledger: B3 joint].
- **The rejected path, forced** (`test_sa_joint.py`). It uses the middle layer's real program executor
  behind the wrapper, with a belief that shows the target but has not seen it within saturation.
  - The combiner rejects each pursuit request as entity unknown.
  - The executor re-samples the step twice, asks the LLM once on the third rejection, and withdraws.
  - With the target seen, the same loop grants every request.

The guide in section 1 prints:

```
21 [('accepted', ('jump_off_on', 'locomotion_gait_compass', 'body_yaw_delta_bins', 'mount')), ('done', ())]
done [(0, 'wave_at(friend)', 0, 'accepted'), (1, ..., 'running'), (2, ...), (3, ..., 'running'),
      (3, 'wave_at(friend)', 0, 'done'), (4, 'open_gate', 1, 'accepted'), (4, 'open_gate', 1, 'done')]
```

## 5. Known limits

- **Scripts.** Of 33 commands, only `switch_item` has a script so far; each other command gets its
  script when its mechanic passes a fidelity row.
- **Programs.** There is no background node (section 2.5).
- **Context kinds.** Region, item, recipe and route are carried, not checked.
- **Bodies are per layout.** A real body is trained against one layout. On the other layout, its first
  write to a head outside that layout raises.
- **The combiner is a snapshot.** Register skills first, then build one combiner per episode; it holds
  the grant state.
- **Starvation.** It is bounded only by the middle layer's aging rule.
- **Gen 7 is partly proved.** The gen-7 simulator is a render of HEAD's engine with the `sim/gen7`
  overlay. Since 2026-10-18 (late) it and the bridge belong to the peer session
  (`sim/gen7/GEN7-PLAN.md` section 2b). In the fidelity ledger's current status (`docs/FIDELITY.md`) only
  Dodge passes, so a verb that needs any other mechanic cannot count a gen-7 training result yet (E14).
- **Trace timing.** The 32 native NPC traces from the live 0.7.2 server (A7) have one frame per 1/30 s
  (120 TPS × 4 dilation), the gen-7 tick under the 30 Hz ruling, so they need no resampling for gen 7.
  A run at the gen-5 simulator's 0.045 s tick must resample them and record the factor.

## 6. References

- Ahn, M., et al. (2022). Do as I can, not as I say: Grounding language in robotic affordances. CoRL. arXiv:2204.01691. — recalled.
- Andreas, J., Klein, D., & Levine, S. (2017). Modular multitask reinforcement learning with policy sketches. ICML. — recalled.
- Baker, B., et al. (2022). Video PreTraining (VPT): Learning to act by watching unlabeled online videos. NeurIPS. — recalled.
- Brooks, R. A. (1986). A robust layered control system for a mobile robot. IEEE J. Robotics and Automation 2(1), 14-23. — recalled.
- Chitta, S., et al. (2017). ros_control: A generic and simple control framework for ROS. JOSS 2(20), 456. Resource conflicts: `hardware_interface::RobotHW::checkForConflict`, https://docs.ros.org/en/noetic/api/hardware_interface/html/c++/robot__hw_8h_source.html — search summary 2026-10-17 for the `checkForConflict` behaviour; the JOSS entry recalled.
- Coffman, E. G., Elphick, M. J., & Shoshani, A. (1971). System deadlocks. ACM Computing Surveys 3(2), 67-78. — recalled.
- Colledanchise, M., & Ögren, P. (2018). Behavior Trees in Robotics and AI: An Introduction. CRC Press. — recalled.
- Erol, K., Hendler, J., & Nau, D. S. (1994). HTN planning: Complexity and expressivity. AAAI. — recalled.
- Fan, L., et al. (2022). MineDojo: Building open-ended embodied agents with internet-scale knowledge. NeurIPS Datasets and Benchmarks. Action space: https://docs.minedojo.org/sections/core_api/action_space.html — search summary 2026-10-17 (the action-space page).
- Houlsby, N., et al. (2019). Parameter-efficient transfer learning for NLP. ICML. — recalled.
- Isla, D. (2005). Handling complexity in the Halo 2 AI. Game Developers Conference. — recalled.
- Jakobi, N., Husbands, P., & Harvey, I. (1995). Noise and the reality gap: The use of simulation in evolutionary robotics. ECAL. — recalled.
- Konidaris, G., & Barto, A. (2009). Skill discovery in continuous reinforcement learning domains using skill chaining. NeurIPS. — recalled.
- Lee, J., et al. (2019). Set Transformer. ICML. — recalled.
- Liang, J., et al. (2023). Code as Policies: Language model programs for embodied control. ICRA. arXiv:2209.07753. — recalled.
- McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. Psychology of Learning and Motivation 24, 109-165. — recalled.
- McKeeman, W. M. (1998). Differential testing for software. Digital Technical Journal 10(1), 100-107. — recalled.
- Meyer, B. (1992). Applying "design by contract". IEEE Computer 25(10), 40-51. — recalled.
- Pirjanian, P. (1999). Behavior coordination mechanisms: State-of-the-art. Tech. Report IRIS-99-375, Institute for Robotics and Intelligent Systems, University of Southern California. https://robotics.usc.edu/publications/319 — search summary 2026-10-17 (the USC publications page).
- Pomerleau, D. A. (1989). ALVINN: An autonomous land vehicle in a neural network. NIPS 1. — recalled.
- Rosenblatt, J. K. (1997). DAMN: A distributed architecture for mobile navigation. J. Experimental and Theoretical AI 9(2-3), 339-360. — recalled.
- Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. AISTATS. — recalled.
- Sharma, S., Lakshminarayanan, A. S., & Ravindran, B. (2017). Learning to repeat: Fine grained action repetition for deep reinforcement learning. ICLR. — recalled.
- Sutton, R. S., Precup, D., & Singh, S. (1999). Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. Artificial Intelligence 112(1-2), 181-211. — recalled.
- Tavakoli, A., Pardo, F., & Kormushev, P. (2018). Action branching architectures for deep reinforcement learning. AAAI. — recalled.
- Tobin, J., et al. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. IROS. — recalled.
- Vaswani, A., et al. (2017). Attention is all you need. NeurIPS. — recalled.
- Vinyals, O., Fortunato, M., & Jaitly, N. (2015). Pointer networks. NeurIPS. — recalled.
- Vinyals, O., et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning. Nature 575, 350-354. — recalled.
- Wang, G., et al. (2023). Voyager: An open-ended embodied agent with large language models. arXiv:2305.16291. — recalled.
- Xiong, R., et al. (2020). On layer normalization in the transformer architecture. ICML. — recalled.
- Zaheer, M., et al. (2017). Deep Sets. NeurIPS. — recalled.
- ROS actionlib goal states: http://wiki.ros.org/actionlib/DetailedDescription. — recalled.
