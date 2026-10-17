# experimental/neural_agent: thesis

Status on 2026-10-16; updated 2026-10-17 (UTC; the local clock read 2026-10-16 evening, CDT) with the leakage re-run (section 5, H12), the belief normalization (section 3) and the lineage citations (section 2). This document states the research question of the three-level agent in `experimental/neural_agent/`, the levels and the two state objects, what the project trains and what it does not, the hypotheses with their experiments, the evaluation metrics, and the ablation matrix. It makes no claim of learned competence. Section 9 lists the evidence the record holds.

**Record caveat.** The decision record quoted here is ADR-001, `agents/design/ADR-001-three-level-agent.md`. Its header says it is local-only (excluded through `.git/info/exclude`), and it is not in this repository. `agents/design/PLAN.md` (PLAN v0.2) and the other design files cited here are local-only too. `ARCHITECTURE.md` section 8 pins each cited local file by SHA-256, so a reader with a copy can check it. The ADR text quoted here was checked against the version with SHA-256 `8bd58354393c3e61b3ad4123be4f0d2bf6db94d3573186748fb22009ca8ca6d3` (modified 2026-10-16 20:59:34). The ADR changed three times during this work: at 19:58:30 (SHA-256 `f5f0fdb9…`), at 20:56:46 (SHA-256 `6b34c6fe…`) and at 20:59:34. The SHA-256 `363a7196…` pinned in `experimental/neural_agent/SKILL.md` and `README.md` predates all three, so those pins are stale.

Labels follow ADR section 0: **IMPLEMENTED** (code exists and its tests ran), **VALIDATED** (a measured result, with its scope), **COMMITTED** (a design decision of the record), **HYPOTHESIS** (not yet measured, with the experiment that tests it). Formulas carry their own labels in `FORMULAS.md`.

## 0. Glossary

- **Legal observation `O_t`**: what the agent could legally perceive (field of view, line of sight, sound, contact, legal memory). Deployed levels read only this.
- **Privileged state `P_t`**: the full local scene, including hidden objects. Only teachers, critics, diagnostics and oracle ablations may read it.
- **B_t, shared belief**: one recurrent latent built from legal observation. Read by the executive abstraction, the JEV and the skills.
- **E_t, executive state**: objectives, priorities, constraints, horizons, budgets and facts, written by the LLM executive and read by the JEV.
- **JEV**: "a very efficient decision model that reads and manages state and produces an output very quickly" (ADR section 2). It is defined by its interface and latency budget, not by its internals. A small learned network (J1) is the primary instance and a scripted rule set (J0) the baseline. A quick language model (J-LM) is an arm, and a JEPA-style world model is a later, gated family (J2–J4).
- **Skill**: a Level 1 policy with an explicit contract (observation groups, action scope, parameters, termination, success and failure signals).
- **ManagedGoal**: the ADK object a JEV returns. Its payload names a skill and its parameters; its control holds the minimum commitment, the maximum duration and the emergency-interrupt mask.
- **DecisionEvent**: one of 14 interrupt events. Each has an active bit and a known bit.
- **Lane**: one batch element. Every per-lane array carries a leading lane axis `B`. The PRNG key is the exception: the ADK core takes one key and splits it into three per-call keys (`adk/architecture/hierarchy.py:260`), so it has no lane axis.
- **12-head surface**: the factored action heads that Level 1 emits, `int32[B, 12]`.
- **ADK**: the agent development kit in `adk/`. It holds the hierarchy orchestration, the decision context and counterfactual imagination.
- **HytaleGym**: the JAX simulator in `HytaleRL/hytalegym/`. Where it disagrees with the native Hytale server, the server is right (ADR section 2).
- **Dawn**: a local-only agent project in `agents/dawn/` with structural tests and MiniGrid results. It has never been trained on Hytale.

- **JEV (corrected 2026-10-17).** Not shorthand: TypeSafe AI's Jev, a System One model (state in, typed calibrated decisions out, in one parallel pass, 70-500 ms). The owner's middle layer is a JEV-like model built here, J-S1 (`SYSTEM_ONE.md`); J0 and J1 are its baseline and mechanism proof.

## 1. Research question

ADR-001 section 1 (COMMITTED). The first paragraph is quoted verbatim below. The two paragraphs that follow it in the ADR, the end goal and the PLAN goal, are paraphrased in the next two paragraphs:

> The system is a hierarchical agent that operates continuously in a simulated world while allocating its reasoning time and compute intelligently. The research question is how an intelligent agent uses time and compute: when to think, when to act, and when to reuse an existing decision. Task completion is the instrument, not the question. Final evaluation therefore measures success *relative to* reasoning compute, wall-clock latency, virtual time, task concurrency, and the ability to react to unexpected events (§9). This project answers the question for the JEV, and builds the substrate and measurement harness on which others answer it for the LLM executive (D3); the skills it consumes are validated by a separate system (§2). At the JEV, J0 and J1 answer it through cadence, commitment, interrupts, and decision reuse; adaptive compute allocation (J4) is one later arm (D2).

The record adds an end goal, also COMMITTED: large labs use the environment, the JEV and the skills as a testbed for LLM reinforcement learning with self-play. That work is theirs, not this project's, and this project keeps the Level 3 interface and metrics stable so that it is possible (ADR section 1 and D3).

The record also keeps PLAN v0.2's goal of NPCs that are "measurably stronger, more adaptable and less predictable than conventional scripted behavior" as a necessary condition, not as the headline (ADR section 1).

## 2. The three levels

ADR section 2 (COMMITTED). Level 3 is asynchronous and event-driven. Level 2 runs at about 2 Hz plus interrupts under a hard latency budget. Level 1 runs at every decision step, 10 to 20 Hz.

| Level | Decides | Cadence | Reads | Writes | May never see |
|---|---|---|---|---|---|
| 3 Executive (LLM, API model) | objectives, priorities, constraints, horizons, budgets; when to reason at all | event-driven; no per-action or per-transition requirement | abstracted B_t, events, outcomes, its own memory | E_t; task and episodic memory | privileged simulator state; raw observation dumps |
| 2 JEV (decision model) | which skill, with which parameters, for how long; whether to search and how much (J3–J4 only) | about 2 Hz plus `DecisionEvent` interrupts (PLAN §36; §19.2 allows 1–4 Hz); hard latency budget | B_t, E_t, decision context, goal runtime | `ManagedGoal`; its own model and search state (J3–J4) | privileged state at deployment |
| 1 Skills (policies) | the action factors for this decision step | every decision step | B_t, goal payload, legal observation, action mask | actions; skill-local state | privileged state; anything outside the skill contract |

Non-negotiables (ADR section 2, COMMITTED). The record lists six. ADR section 6.4 now says "the six non-negotiables (§2)", which matches.

1. **Level 1 never blocks.** On a missing or stale JEV decision it continues the current valid goal, falls back, or stops safely.
2. **The JEV answers within its latency budget or not at all.** An instance that misses its budget leaves the previous goal in force and the miss is counted. This is what makes the small learned network (J1, the primary instance), a scripted rule set, a quick language model, and a world-model search comparable as instances.
3. **Legal observation only, from the start.** Deployed Levels 1 and 2 consume `ActorPolicyInput` products only. Oracle-state runs are ablations and are reported as such (D5). On the current surface the record's leakage test fails its negative control, so this non-negotiable is not yet met (section 9).
4. **Level 3 receives an abstraction** of B_t and memory, never a simulator dump.
5. **Skills are frozen and version-pinned** during any JEV or executive experiment (D1). HIRO found pre-training then freezing the lower level detrimental on harder navigation tasks [16]; section 10 of the ADR carries that as a risk.
6. **Hytale is right by definition** where JAX and Hytale disagree.

**Lineage** (ADR section 3 D2 and D3, VALIDATED external; their related-work paragraphs, paraphrased). Three-layer robot architectures ran on real robots [L1][L2][L3], with tiers the 3T paper names reactive skills, a sequencer and a deliberative planner [L2]. Subsumption keeps lower layers acting while higher layers suppress or inhibit them [L4], and blackboard control separates control reasoning from domain problem solving [L5]. The JEV is the sequencer of that lineage, and non-negotiable 1 is Brooks's layering. The lineage shows that the shape works on hardware, not that a learned sequencer helps, which is what J1 must show. Fast-and-slow agents pair a fast network or agent with slow search or planning [L6][L7]. GR00T N1 reports its latency, on robots [L8], and two Minecraft systems put a language-model planner over skills with concurrent, interruptible execution and report no latency [L9][L10]. Resource-bounded reasoning [L11] and anytime algorithms [L12] are the classical basis for when to think, and two 2026 papers learn that choice: a selector between reactive control and planning [L13], and a per-step router of reasoning effort for language-model agents [L14]. None of them runs over a real-time learned controller in a game (ADR D3).

## 3. The two state objects

- **B_t, the shared belief.** Produced by the shared encoder and belief module from legal observation only. It is a separate object, not any level's private latent. The contract types it as `SharedBeliefState` with `features float32[B, 10]`, `feature_known bool[B, 10]` and `latent float32[B, 64]`. Its ten features are `target_visible`, `target_distance_blocks`, `target_bearing_sin`, `target_bearing_cos`, `self_health_fraction`, `target_health_fraction`, `ticks_since_damage`, `ticks_since_target_seen`, `self_speed_blocks_per_tick` and `closing_speed_blocks_per_tick`. Their normalization is decided by the belief module, not open: the two tick counters are published as counter / `TICKS_SATURATION` (`belief/core.py:400`, `:402`), and the other columns follow `belief/SKILL.md` (FORMULAS B5).
- **E_t, the executive state.** Objectives, priorities, constraints, planning horizons, compute budgets, deadlines and the facts the executive judges relevant. It is written by Level 3 and read by the JEV, which is how the LLM manages the state the JEV works in. It is versioned, so every decision can report the version it served (ADR D3).

Memory ownership (ADR D6, COMMITTED). The shared belief resets at episode boundaries only, never on a skill switch. Skill-local state resets on skill entry. Persistent task and episodic memory belongs to Level 3 and changes only through explicit executive action.

## 4. What this project trains, and what it does not

**Trains (ADR section 2, COMMITTED).** The shared encoder and belief B_t, and every learned JEV instance that reaches its milestone: the reactive network (J1), then the world model and its heads, the search scoring and the allocation policy (J2–J4). J0 is scripted, and J-LM is off the shelf unless stated. **Skill policies are not trained here** (same section, build ownership, COMMITTED): they are "produced and validated by a separate system and enter this stack as frozen inputs behind the `Skill` contract."

**Open question within the record.** The "What is trained here" paragraph no longer lists the skill policies (ADR-001 line 82, SHA-256 `4284c60d…`). Two passages still describe training skill heads: D1, "The shared token encoder is rebuilt before a second skill is trained on the shared trunk", and H1, "freeze the trunk, train an evasion head". The "Build ownership" paragraph says this build does not retrain skills. The record does not say which system trains a new head. This document does not resolve it. `ARCHITECTURE.md` section 7 lists the question as open.

**Does not train.** The LLM executive. It is an API model, "not trained, developed, or tuned in this project" (D3; owner: "we will not touch the LLM side"). This project builds the seam the executive plugs into: the E_t record, the report channel, the scheduler hooks, the measurement harness and the deterministic baselines. One API-backed executive run may prove that the seam carries an executive. JEV and skill self-play are later work (ADR section 12, question 8). LLM self-play is external (D3).

**J-LM** (a quick small language model reading B_t and E_t) is an arm to compare, not the primary instance. It is compared under the same contract and latency budget. The record does not say whether that arm is trained or prompted.

## 5. Hypotheses H1 to H16

Copied from ADR section 3 (D1 to D6) and section 6.6 of the current version. The milestone in parentheses is the one the ADR names. Each hypothesis is HYPOTHESIS until its experiment promotes or kills it.

- **H1** (P1). A trunk trained on pursuit transfers to evasion, navigation, and interaction heads without retraining. *Experiment:* freeze the trunk, train an evasion head, compare against a from-scratch evasion network at equal budget on the fixed evaluator.
- **H2** (P1). Goal-conditioned heads suffice for pursuit, evasion, and navigation; specialized policies are needed only where scopes differ, such as world interaction. *Experiment:* per-skill head-versus-policy ablation under D5 constraints.
- **H3** (P3). Skill-level action-conditioned latent dynamics, trained on JAX transitions and `imagine` branches, reach pre-registered thresholds on action ranking (rank correlation with the exact-dynamics ceiling), Brier and expected-calibration-error calibration, and action-difference error below an action-blind model's. 1/2/4/8-decision prediction error is reported, not gated (the M3 lesson, D2.1). *Experiment:* P3 in §8.
- **H4** (P4). Bounded search (J3) improves outcome per unit of compute over J1 with identical skills and observation constraints, across three seeds with intervals, with no death or damage regression. *Experiment:* P4.
- **H5** (P4). Adaptive allocation (J4) beats the best fixed search budget on the same metric. *Experiment:* P4 extension.
- **H6** (P3). Separate fast and slow representations beat a single shared one at matched parameters. *Experiment:* J2 ablation.
- **H7** (P1). The ADVISOR auxiliary-policy weighting [3] is preferable to PLAN §24.3's reconstructed-privileged-state weighting for privileged distillation at Levels 1 and 2. *Experiment:* side-by-side on one skill. PLAN §24.3 already labels its formula a proposal.
- **H8** (external, P5 harness). Event-driven invocation reaches at least the outcomes of per-transition invocation at a fraction of the calls. *Experiment:* external, run on the P5 harness; three trigger policies.
- **H9** (external, P5 harness). A value-of-reasoning trigger beats fixed-cadence triggers on outcome per call. *Experiment:* external, run on the P5 harness.
- **H10** (external, P5 harness). A legal-only abstraction of B_t and memory is sufficient for strategic decisions. *Experiment:* external, run on the P5 harness, with a privileged-summary ablation reported separately.
- **H11** (P2). Four skills are enough to expose the composition failure modes the JEV must handle: switch thrash, state handover, interruption, dependency. *Experiment:* P2's composition tasks. If a failure mode needs a fifth skill, add it there and record why.
- **H12** (P1). Skills trained under legal observation reach the oracle arms' success within a stated margin at matched budget. *Experiment:* paired arms in P1. The margin is the measured cost of partial observability and is reported, not hidden.
  - *Status: VALIDATED local, negative* (ADR section 3 D5, 2026-10-16; re-run 2026-10-17). The legal surface this hypothesis needs is not leak-free yet. The three-world test is `HytaleRL/hytalegym/tests/jax/combat/arsenal/tests/agents/ppo/test_leakage_three_world.py`. Its positive control passes: a legally visible enemy changes the flat policy observation (16 columns, as the ADR records; the test asserts more than one). Its negative control fails: a live enemy hidden behind a wall and no live enemy differ on `ability_legal`, the ability action mask and `defense_f32` (4 flat columns and 4 mask entries), eagerly and under `jax.jit`, on both seeds. The cause is in the gym. Ability legality is gated by the privileged selector line-of-sight flag: `REQUIRE_LINE_OF_SIGHT` is satisfied by `world.selector_line_of_sight_valid` (`HytaleRL/hytalegym/hytalegym/jax/combat/arsenal/programs/ability.py:1396-1397`). The test also reports a seam limit: the reset seam carries only two positions, so World B zeroes the enemy's health instead of omitting it. Re-run on 2026-10-17 (UTC), CPU: 6 passed and 2 failed, the two negative-control cases (seeds 570057 and 913771), in 54.6 s. Until the gym gates that legality on legal target visibility, the surface is not leak-free, and no arm on it is legal by PLAN §12.3's definition. This document's inference: H12's margin cannot be measured as stated until then. The test file is git-ignored (`HytaleRL/.gitignore:53`).
- **H13** (P1). A shared belief with stateless or nearly stateless skill heads loses nothing against today's per-skill GRU-128. *Experiment:* pursuit with shared belief and an MLP head versus the current candidate, same evaluator.
- **H14** (P2). Reset-on-entry skill state beats carried state across switches. *Experiment:* P2 switching tasks, both variants.
- **H15** (P4). A quick language-model instance (J-LM) meets the JEV latency budget on the deployment target and matches J1 on outcome. *Experiment:* P4, compared with J1, and with J3 if it exists, on the same tasks and budget; latency misses counted against it.
- **H16** (P2). J1 beats J0, the flat actor, and its shuffled-goal control on the composition tasks. *Experiment:* P2, the four-arm protocol on held-out seeds, with intervals. *Status 2026-10-17:* J1 cloned from J0 matches J0 (16 vs 17 of 32) and beats the control (2 of 33) on a held-out seed; superiority untested (`LEDGER.md`).

## 6. Milestones and exit gates

ADR section 8 (COMMITTED). The gates are pre-registered before the run they gate. The table is a summary; the ADR text governs.

| Milestone | Delivers | Exit gate (summary) |
|---|---|---|
| P0 reliable native execution, encoder | native movement diagnosis; shared token encoder (permutation-invariant [11][12]); three-world enemy leakage test | pursuit bundle passes the runtime movement gate under legal observation; the new encoder's Wilson lower bound at or above the incumbent's by the recipe's advance rule; `exact_divergences == 0`; the three-world test passes on two seeds |
| P1 composable skills, shared belief | the four D4 skills on a shared trunk, with contracts; the skill contract runtime fields (seam 7); a dispatcher; H7 side by side on one skill | each skill passes `PromotionGate` on the held-out split with at least 64 episodes per arm and three seeds; an oracle twin is reported (H12); the P0 native gate is re-run for pursuit and navigation |
| P2 scripted and reactive JEV (J0, J1, then J-S1 as the intended primary learned instance; gates G1-G5 in `SYSTEM_ONE.md`), composition | J0 rules over frozen skills; J1, the learned reactive network, behind the same contract; the JEV contract; the first part of the E_t record (seam 3); composition tasks | J0 completes composite tasks the flat actor cannot, on held-out seeds; J1 beats J0 with intervals and beats its shuffled-goal control (H16) |
| P3 world model, offline | J2 latent dynamics with outcome, value, reachability and uncertainty heads, and a decision-aware loss | gated on action ranking, calibration (Brier and ECE within declared thresholds) and action-difference error below an action-blind model's; 1/2/4/8-decision error is reported, not gated; no runtime use |
| P4 bounded search, adaptive allocation, quick-LM arm | J3, J4, J-LM; starts only after P3's gates pass | J3 at or above J1 on outcome per unit of compute, three seeds, intervals, no death or damage regression; J4 at or above the best fixed-budget J3; J-LM compared with J1 and J3 with latency misses counted |
| P5 executive seam, scheduler hooks, baselines | the report channel; scheduler hooks; logging of the four allocation knobs per call; the episodic memory interface; the asynchronous runtime; three deterministic baselines; one API-backed run | a harness check, not an executive result: the baselines run end to end through the seam, the knobs are logged per call, and the API run replays from fixed transcripts; H8–H10 are external and run on this harness |
| P6 final evaluation | the arms this project builds (every arm not marked external) | per section 9 on held-out worlds and seeds, with native runs for every skill-level claim |

## 7. Evaluation metrics

ADR section 9 (COMMITTED) names six metric families. Each metric below has a formula. The same formulas appear in `FORMULAS.md` section K, where each one carries its label (*COMMITTED* family, ADR section 9; the formula is this document's); the two must agree. Notation: episodes `i = 1..N`, decision index `t`, ticks, virtual time `T_v`.

| Family (ADR section 9) | Measures |
|---|---|
| Outcome | task success; reward components per PLAN §31, logged separately; deaths; damage dealt and taken |
| Reasoning compute | language-model calls and tokens; search node expansions; model forward passes; per decision and per episode |
| Wall-clock latency | per level, p95, against PLAN §30.4's absolute budgets at a stated batch size and device |
| Virtual time | simulation ticks to objective; ticks spent on stale or fallback decisions |
| Task concurrency | outstanding objectives progressed per unit of virtual time |
| Reaction to unexpected events | ticks from an injected `DecisionEvent` to goal change, and to the first changed skill action |

**7.1 Outcome.** Task success rate over `N` episodes, reported with the Wilson lower bound of `FORMULAS.md` J1 for each arm:

$$\hat S = \frac{1}{N}\sum_{i=1}^{N}\mathbf 1[\text{success}_i]$$

Deaths and damage are per-episode counts, dealt and taken, and reward components are logged separately (PLAN §31).

**7.2 Reasoning compute.** Per episode, the compute vector is `c_i = (calls_i, tokens_i, expansions_i, forward_i)`. Per decision, divide each entry by the number of decisions. The record leaves the compute unit open (ADR section 12, question 2), so the vector is reported, and a scalar is reported only with weights declared before the run.

**7.3 Outcome per unit of reasoning compute.** This is the headline for H4, H5 and H9.

$$\mathrm{OPC} = \frac{\hat S}{\bar C}, \qquad \bar C = \frac{1}{N}\sum_{i=1}^{N}\sum_{k} w_k\, c_{k,i}$$

Here `k` indexes calls, tokens, expansions and forward passes, and the weights `w_k` are declared before the run. For the trigger comparisons (H8, H9) the per-call form is also reported: `OPC_call = Ŝ / mean calls per episode`. The ADR names "hierarchical intervals" for H4 and P4 without defining the method. The method must be declared before the run.

**7.4 Wall-clock latency, p95 per level.** For a level with decision latencies `ℓ_(1) ≤ … ≤ ℓ_(n)` in milliseconds, `p95 = ℓ_(⌈0.95 n⌉)` (nearest rank, the method of `executive/scheduler.py`; the ADR does not name a method). `jev/runner.py` uses linear interpolation, so the repository has two methods, and the owner must choose one before the run. Report the batch size and device (PLAN §30.4). The JEV miss rate is `misses / decisions` per instance.

**7.5 Virtual time to objective.**

$$T_{\mathrm{obj}} = \min\{t : \mathrm{success}(s_t)\} - t_0 \quad\text{(ticks; censored at } T_{\max}\text{ if never reached)}$$

Ticks spent on stale or fallback decisions are `T_stale = Σ_t 1[decision t is stale or a fallback]`, reported as the fraction `T_stale / T_obj`.

**7.6 Task concurrency.** With a progress measure `P_o(t) ∈ [0, 1]` for each objective `o` (the record does not define it; it must be declared per task):

$$\kappa = \frac{1}{T_v}\sum_{o}\sum_{t=1}^{T_v}\mathbf 1[P_o(t) > P_o(t-1)]$$

This is the number of objective progress events per tick of virtual time. A value above 1 means that several objectives progressed within one tick.

**7.7 Reaction to unexpected events, in ticks.** For an injected `DecisionEvent` `e` at tick `t_e`, with `G` the goal and `A^skill` the skill action factors:

$$R^{\mathrm{goal}}_e = \min\{t \ge t_e : G_t \ne G_{t_e-1}\} - t_e, \qquad R^{\mathrm{act}}_e = \min\{t \ge t_e : A^{\mathrm{skill}}_t \ne A^{\mathrm{skill}}_{t_e-1}\} - t_e$$

Both are censored at a declared horizon `H_r`. Report the median and p95 over injected events. Caveat (this document's inference): an action can change without any reaction to the event, so `R^act` is only interpretable paired with a no-injection run at the same seed. The ADR states the metric but not this control.

## 8. Ablation matrix

ADR section 9 (COMMITTED), current text: "Every claim about a level is made against the preceding arm on its axis."

| Axis | Arms |
|---|---|
| JEV | J0 scripted · J1 reactive · **J-S1 our own System One model (pretrained trunk, parallel typed heads; `SYSTEM_ONE.md` §8, added 2026-10-17)** · J2 reactive plus model (offline only) · J3 search · J4 adaptive · J-LM quick language model · shuffled-goal control |
| Executive | none · scripted E_t writer · offline plan · external LLM |
| Trigger | fixed cadence · per transition · event-driven (external; harness only) · value-of-reasoning (external; harness only) |
| Observation | legal (the reported number) · oracle (ablation, reported separately) |
| Belief | shared · per-skill |
| Skills | frozen pinned versions; any change is a new experiment identity |

The primary success measure remains performance in held-out real Hytale scenarios, not simulated or imagined return (PLAN §32.4), per the ADR's section 9 text.

## 9. Evidence status

What the record establishes today, quoted or paraphrased from ADR-001 (version of 2026-10-16 20:59:34). Nothing here is a Hytale result for the hierarchy.

**Update 2026-10-17 (main session; every number below is an entry in `LEDGER.md`).** The first learned-JEV result exists, and it is a simulator result for the middle level over frozen skills:

- **Horizon.** On `combat/open_flat_control` the 256-tick horizon (11.5 s) rarely completes a re-engagement after an evade (J0 composed 1 of 4 lanes). At 512 ticks, 64 lane-runs per arm, seed 0: J0 composed 33/64 (Wilson lower 0.396), the shuffled control 16/64 (0.160), the flat actor 0/89; two-sided Fisher exact J0 vs control p = 0.0034. Both JEV arms survived every episode; the flat actor died 25 times in 89 episodes. The composition CLI's default horizon is 512 from this date.
- **J1 cloned from J0 on real traces.** 2,966 decision rows from 16 seeds at 256 ticks; held-out seeds 100-103, 737 rows: skill accuracy 0.973 after 400 updates, 0.996 after 2,000 (`jev/eval_j1.py`). Online on held-out seed 100, 512 ticks, 32 lane-runs per arm: J1 16/32 composed (v2; 15/32 for v1), J0 17/32, shuffled 2/33, flat 0/45. J1 vs control Fisher p = 0.00019; J0 vs J1 p = 0.80.
- **Direction (2026-10-17, late).** The owner defined JEV precisely (TypeSafe's System One model) and the target instance: our own System One decision model for goal reading and decision making, no external component except the top LLM. Survey, design and build order: `SYSTEM_ONE.md`.
- **H16 status.** "J1 beats J0" is not shown and cannot follow from cloning J0. J1 matches J0 and beats its control on seeds and a horizon it never saw. Superiority needs the PPO stage on real transitions; `jev/train_j1_ppo.py` trains on a synthetic environment today.

- **No learned JEV result exists in the record (as of 2026-10-16; see the 2026-10-17 update above).** The record's IMPLEMENTED status (section 3, D2) lists the orchestration, the interrupt vocabulary and the contract records (43 contract tests passing, as of its date), and it says that "Orchestration in `adk/` does not enforce the latency budget yet (seam 8)". The repository has a J0 module and a J1 network on disk (ARCHITECTURE section 1). Their tests pass, and neither holds a learned result (ARCHITECTURE section 6).
- **Level 3 is confirmed only at the record level.** The record confirms no language layer, no E_t record, no scheduler hooks, and no virtual-time or compute-budget code in `adk/`, `arena/` or `agents/dawn/` (section 3, D3). The seam exists in `experimental/neural_agent/executive/` on disk, and nothing in it is measured.
- **Legal observation is available and leak-free since 2026-10-17 (VALIDATED local; the 2026-10-16 negative finding below is superseded); it is opt-in, and this project's arenas opt in.** The three-world gate passes 10 of 10 in that mode: ability legality gates on the perception ray to a living target, the per-entity rows of entities the actor cannot perceive or that are not living are blank, and `ArsenalRuntimeConfig.legal_observation` (default False after the attribution run: four simulator tests encode the privileged semantics) keeps the old behaviour for agents allowed to cheat, with a test showing that mode still leaks (`LEDGER.md`). The earlier text, kept for the record: **Legal observation is not yet leak-free (VALIDATED local, negative, 2026-10-16).** The record's three-world test has a positive control that passes and a negative control that fails. A live enemy hidden behind a wall differs from no live enemy on `ability_legal`, the ability action mask and `defense_f32`. The record says "The action mask therefore reveals an unseen live enemy", that "no policy on this surface is leak-free by PLAN §12.3's definition", and that P0 exit (d) "is failing, not pending". It adds that "The gym change is high-blast and awaits the owner's decision" (ADR section 3 D5). Re-run on 2026-10-17 (UTC): the same result, 6 passed and 2 failed (section 5, H12).
- **Native transfer is failing.** On Hytale 0.5.9 the trained pursuit bundle ran 457 legal ticks, travelled 0.0186 blocks, and died, while a JAX replay matched the Java policy factors on all 457 rows (ADR section 3, D4, VALIDATED local negative).
- **The pursuit candidate reached 31/41 validation and 25/39 held-out catches** on an authored flat terrain with a walking evader and field-of-view perception. The record says this is not terrain, not native, and not composition (ADR section 6.5).
- **Local lessons from Dawn and the Clash Royale sibling repository.** A learned planner lost to reactive control (49.22% vs 54.17% success, lost on every seed). Endpoint error improved while action ranking did not (LavaCrossing root cause; the Clash Royale M3 result). Executed-action DAgger [24] beat matched behaviour cloning (73.96% vs 29.95%). All are MiniGrid or Clash Royale results and do not transfer to Hytale (ADR section 3, D2).
- **Novelty claim.** "No published result found on 2026-10-16 measures an asynchronous language-model executive over a real-time learned controller by outcome per unit of reasoning compute" (ADR section 3, D3). This is a search result, not a proof, and the record makes deterministic baselines mandatory for that reason. The nearest published systems put a language model over skills, synchronously or turn by turn, and measure no compute allocation [S1][Plan4MC][DECKARD][Voyager] (ADR section 3 D3 and its section 6.2 table).

## 10. Open questions

ADR section 12 (not decided by the record). Each blocks a named milestone.

1. **Virtual-time semantics** (P5): the unit of virtual time, the map from language-model wall-clock to ticks, and reproducibility under latency variance.
2. **Compute-budget units** (P4, P5): tokens, calls, expansions, forward passes, or a common currency, and how a Level 3 budget becomes something Level 2 can obey and report against.
3. **Provider, model tiers, cost and determinism** of the API executive (P5).
4. **Skill parameter space** (P2, P3): discrete, continuous or mixed. This decides between MuZero-style [MuZero] and Sampled-MuZero-style [SampledMZ] search at Level 2.
5. **First world-interaction verb** (P1): Use, Place, Break or Craft.
6. **Episodic memory store** (P5): format, retrieval scoring, and the audit that keeps privileged facts out.
7. **Where Level 2 runs natively** (P2): the Python inference service of PLAN §30.1, or a Java port of the JEV. The Java agent today executes only the dense policy.
8. **Multi-agent sequencing** (no milestone yet): self-play and the league (PLAN §28 Phase 10) relative to P5, and the first market, war or diplomacy environment.
9. **Clash Royale reference** (P2, P3): whether the wm31 hybrid world-model architecture and the M3.5 Wait-referenced residual objective port to Hytale's skill-level transitions, and whether the T1 timing-manager design is the right shape for J1.
