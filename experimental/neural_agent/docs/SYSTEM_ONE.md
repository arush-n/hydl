# Our own System One decision model for goal reading and decision making

> Consolidated overview and plan (2026-10-17 night): [`MIDDLE_LAYER.md`](MIDDLE_LAYER.md). This file holds the detail it points to.

Written 2026-10-17 (main session) after the owner's clarifications of that day: the middle layer
is a JEV-like System One model; nothing external is used except the top LLM; we build our own.
Research question (ADR-001 §1) unchanged. Every number below is either quoted from the cited source
(self-reported unless stated) or from `LEDGER.md`.

## 1. What "JEV" actually is

Jev is TypeSafe AI's first "System One model" ([JEV]). Their description: "unstructured state in,
typed probabilistic decisions out"; the possible outputs are defined in advance by a schema; every
answer carries calibrated probabilities; all outputs are produced in parallel in one query rather than
token by token; 70-500 ms end to end; trained with "Reinforcement Learning for Calibrated Decisions"
([RLCD], described only in their blog, no paper); intended for classify / route / score / branch
("smart if-statements"), not text generation, planning or tool use. Weights, size, architecture and
training data are unpublished ([JEV-STATUS]). It is named after W. S. Jevons, not an acronym.

Before this date our documents used "JEV" as the owner's shorthand for "fast decision model". The
shorthand pointed at the right thing; the definition above is the precise one.

## 2. What exists in the open

| Project | What it is | Relevance |
|---|---|---|
| JuL ([JUL]) | PyPI runtime with Jev's typed-decision interface over any local LLM (MLX or PyTorch), turned into a calibrated decision head; accuracy reported with wide error bars (n=200, SE 3.5 points) | the interface shape and the "local LLM + decision head" pattern |
| AnyJev ([ANYJEV]) | Nokia applied-research repo: prompt layer, calibration core, interchangeable backends; documents that heads do not transfer across questions or models | honest negative: a head is per question; ours is one question (which skill now) |
| Laya ([LAYA]) | checkpoints on ModernBERT-large and mmBERT-base encoders, "about 33 ms" per calibrated decision on one GPU (author's claim), Apache 2.0 | encoder backbone, single forward pass |
| JevK5-Lite ([JEVK5]) | Hugging Face preview; 0.587 head accuracy vs 0.637 for GLiNER2.5-Decide on its own split | shows small decision heads exist, quality varies |

None of these is used here (owner: no external components); they fix the design space.

## 3. The published pieces a System One model is made of

- **A frozen pretrained backbone with trained adapters and an action head, trained by RL**: LLaRP
  ([LLaRP], Apple, ICLR 2024). Frozen LLaMA-7B, text instruction plus observations in, actions out of
  an action decoder head, PPO, 42% success on 1,000 unseen tasks (1.7x the baselines), robust to
  paraphrased instructions. This is the closest published design to "general pretrained decision model
  fine-tuned on outcomes".
- **Calibration as a reward**: RLCR ([RLCR], MIT, ICLR 2026): correctness plus a Brier score as the RL
  reward; proper scoring rules provably yield accurate and calibrated models; calibration error down
  by up to 90% at equal accuracy (MIT write-up). A follow-up ([RLCR-HACK]) shows unconstrained Brier
  or log-loss rewards can be hacked (the model answers wrongly to be confidently wrong) and proposes
  non-hackable schemes. This is the open analogue of RLCD.
- **Parallel heads on a shared trunk instead of token-by-token output**: multi-token prediction
  ([MTP], ICML 2024): n independent heads on one trunk; parallel heads beat causally chained ones;
  up to 3x faster inference. Non-autoregressive prediction in general is up to m times faster than
  m decoding steps but assumes output fields are independent ([NAR]); with a handful of typed fields
  (skill, replace, params, confidence) the assumption is mild, and a two-stage scheme covers large
  output cardinalities (Jev itself uses two stages above 255 choices).
- **Fast encoder backbones**: ModernBERT ([ModernBERT], 2024): 8,192-token context, about 2x faster
  than DeBERTaV3 on short inputs, best GLUE encoder of its size; small encoders run in 8-28 ms on CPU
  in a 2026 distillation study, against about 128 ms for an 8-bit Qwen2.5-0.5B generative student
  ([CPU-DISTILL]).
- **Dual-system agents**: GR00T N1 and Helix ([DUAL]): a slow vision-language planner hands a latent
  intent to a fast policy; Helix 02 reportedly adds a "System 0" for balance. Our interface is E_t,
  a typed state the LLM writes, rather than a latent.

## 4. Where our instances stand

| Instance | Right about | Wrong about |
|---|---|---|
| J0 rule | the slot works without learning; 7 us | fixed |
| J1 small network | the mechanism: state in, all heads in one pass, 0.1 ms, 0.996 agreement with J0 held out | no pretrained knowledge, no calibration, cannot read a new goal written in words |
| J-LM prompted LLM | the knowledge: reads goals and skills as text | the mechanism: 40 tokens one by one, 471 ms (`LEDGER.md`) |

The owner's intended instance is the combination. Name here: **J-S1**, our own System One model.

## 5. Design of J-S1

**Input.** The stable rendering (skill catalogue with descriptions, the executive state E_t: goals with
priority, params, constraints, horizon, deadline; facts), produced by the stable part of
`jev/lm/prompting.py`'s renderer, one per executive version, not per lane and not per tick; plus, per lane
and per decision, the numeric belief vector and the current skill (revised 2026-10-17; the earlier per-lane
text rendering of the belief was measured too slow, see "Two timescales").

**Two timescales (revised 2026-10-17 after G2).** Measured on this machine (CPU, contended by three test
suites): one pass of ModernBERT-base over the 330-token rendering takes about 0.5 s per lane, Qwen3-0.6B about
1.5 s (`LEDGER.md`). Reading the text every decision is therefore never the fast path. J-S1 splits in two:
the *slow path* encodes the stable rendering, the skill catalogue with descriptions and the executive state
E_t, once per executive version, and each skill description once per catalogue change (`h_E = f_theta(render(skills, E_t))`,
`e_j = f_theta(desc_j)`, both cached, computed between steps when the LLM writes); the *fast path* runs every
decision over the cached `h_E`, the numeric belief `u_i` through the adapter, and the current skill, through
the heads only, which is J1's cost (about 0.1 ms for 64 lanes). The belief enters numerically, not as text.
This is exactly "share a state with the LLM": the LLM's state is digested into an embedding when it changes,
and decisions are instant. Gate G2 is therefore split: G2a the slow path per E_t version (seconds are fine,
it runs off the tick), G2b the fast path per decision (must match J1's order of magnitude).

**Backbone.** A small pretrained encoder run in process on CPU (Metal later), frozen at first, for the slow path only.
Two candidates to measure before choosing: an encoder (ModernBERT-base class, about 150M) and a
small decoder used as an encoder by pooling (Qwen3-0.6B class). The measurement is one forward pass on
the real rendering length, batch 1 and batch 8, on this machine. Target: well under the 0.5 s decision
cadence in host mode; stretch: inside the 20 ms budget.

**Heads, all in one pass (System One).** On the pooled trunk output: skill logits masked to the
executive's allowed skills; keep-or-replace logit; parameter regression; a confidence head; and,
behind a flag, a short plan head that emits the next K skill calls, which is the "several decisions
at once" the owner asked for and amortises the pass.

**Output contract.** Exactly today's `GoalPayload` and the J-LM schema, so `JevRunner`, the miss and
fallback rules, the dispatcher and every arm stay unchanged; J-S1 is one more `JevInstance`.

**Training, three stages, all on data and harnesses that exist.**
1. Supervised: the recorded decision rows (`eval/record_traces.py`; 2,966 train, 737 held out, labels
   equal to J0's rule) plus decisions written by the top LLM through the executive harness, rendered as
   above. Fit the heads (and LoRA on the top layers if needed). Gate: held-out agreement with J0 and with
   the LLM's decisions on prompts J0 never saw.
2. Calibration: proper-scoring-rule loss on the confidence head, held-out reliability diagram, with a
   non-hackable form ([RLCR-HACK]).
3. Outcomes: reinforcement learning on real transitions through the semi-MDP harness
   (`jev/sim_goal_env.py`, `train_j1_ppo.py --env sim`), reward from the arena plus a bounded
   calibration term. This is where J-S1 can beat J0 (H16). Running it is training work, outside this
   session's scope by the owner's instruction; the harness is in scope.

**Evaluation.** The existing four-arm protocol on held-out seeds at 512 ticks, plus: decision latency
(`probes/probe_jev_latency.py` extended), agreement with the LLM on unseen goal text, reliability
(expected calibration error), and the ablations: heads without the pretrained trunk (that is J1),
trunk without the goal text, autoregressive decoding instead of parallel heads (that is J-LM).

## 6. Risks, stated

- Latency: a 0.6B pass on CPU may land near 100 ms (unmeasured here); the encoder class is the fallback.
- Independence of parallel heads: mild for four fields; the plan head needs care (predict K small).
- Calibration reward hacking ([RLCR-HACK]); use the bounded, non-hackable form.
- Heads do not transfer across questions ([ANYJEV]); ours is one question, re-fit when the catalogue changes.
- Data: 2,966 J0 rows teach J0's rule; the LLM-written decisions are what teach goal reading.
- One dependency decision for the owner: loading pretrained weights needs `transformers` (or an
  equivalent loader) in the venv; `torch` 2.14 is present, `transformers` is not.

## 7. Build order proposed

1. Latency probe of both backbone classes on the real rendering, in process, CPU: numbers to the ledger.
2. `jev/system_one.py`: the J-S1 instance (backbone, heads, `JevInstance` surface, host mode), tests.
3. Supervised stage on the recorded rows plus LLM-written decisions; offline gates; the four-arm run.
4. Calibration stage and the reliability gate.
5. Hand-off: the outcome stage runs on the PPO harness by whoever owns training.

## 8. Design specification of J-S1 (2026-10-17, owner: "start designing it now")

Owner's statement of the goal (2026-10-17): "the simple goal is that the model can share a state with llm / read the inputs and be able to create outputs very quickly for the neural components, note that the library will increase over time". Every design choice below serves those three clauses: shared state (E_t and B_t as inputs), speed (one pass, parallel heads), and a growing library (catalogue-agnostic scoring).

Labels as in ADR-001: IMPLEMENTED / COMMITTED / HYPOTHESIS. Everything in this section is COMMITTED
unless marked; nothing is IMPLEMENTED until `jev/system_one/` exists and its tests pass.

### 8.1 Position in the stack

J-S1 is one more `JevInstance` (`contracts.py`), host mode, behind `JevRunner`: the same inputs
(`SharedBeliefState` B_t, `GoalPayload` current goal, `ExecutiveState` E_t, decision context, goal
runtime), the same output (`GoalDecision` with a `ManagedGoal`), the same latency budget, miss rule and
idle fallback. The dispatcher, the arenas, the five arms and the composition task do not change. The
decision layer reads from the LLM layer through E_t and calls the neural layer through the goal; the
outcomes return as `GoalReport`s (owner ruling, ADR D3).

### 8.2 Inputs, per lane i

- Text rendering (exists: `jev/lm/prompting.py build_decision_prompt`):
  `x_i = render(skills with descriptions, E_t, current_skill_i, B_t^i)`; stable part first (skills, E_t),
  volatile part last (current skill, belief lines), so a cached prefix is possible later.
- Numeric belief vector (exists as J1's input pieces, `jev/network.py j1_inputs`):
  `u_i = [features_i * known_i ; known_i ; latent_i]`, width 10 + 10 + 64 = 84.
- Allowed-skill mask from E_t (exists: `j1_allowed_skills`): `m_i in {0,1}^K`.

### 8.3 Model

Trunk, frozen at first (COMMITTED; the backbone class is an open measurement):
  `h_i = f_theta(x_i) in R^D`           pooled representation of the rendering (mean or last token).
Belief adapter (trained):
  `a_i = W_u u_i + b_u in R^{D_a}`      LLaRP-style observation adapter, D_a = 64.
Fusion (trained):
  `z_i = LayerNorm([h_i ; a_i ; onehot(current_skill_i)])`, then one hidden layer `g_i = GELU(W_g z_i + b_g)`, width 256.
Heads, all evaluated in the same pass (System One):
  skill      `s_i = W_s g_i + b_s in R^K`,  masked `s~_i = where(m_i, s_i, -1e9)`
  replace    `r_i = w_r . g_i + b_r`
  params     `p_i = W_p g_i + b_p in R^{PARAM_WIDTH}`
  confidence `c_i = sigmoid(w_c . g_i + b_c)`      probability that the chosen skill is the right one
  value      `V_i = w_v . g_i + b_v`                critic for stage C only
  plan (flag, HYPOTHESIS) `s_i^{(k)} = W_k g_i + b_k`, k = 1..K_plan, the next K_plan skill calls, independent heads as in [MTP].
**Catalogue-agnostic skill scoring (owner, 2026-10-17: "the library will increase over time").** The
fixed softmax head `W_s` ties the model to the training-time catalogue size K. The primary skill head is
therefore a matching head: each skill j has a description embedding `e_j = f_theta(desc_j)` from the same
trunk (computed once per catalogue change and cached), and
  `s_ij = (W_q g_i) . (W_k e_j) / sqrt(d_k)`,  masked by `m_ij` as before.
Adding a skill adds a row `e_j`; no head weight changes shape and no retraining is required for the
head to produce a score for the new skill (its quality on the new skill is a separate question, gate G3b
below). The fixed head stays available behind a flag as the ablation.

Decision: `skill_i = argmax s~_i`, `replace_i = r_i > 0`, control constants as J0 (`jev/scripted.py`:
commitment, horizon, emergency mask). Diagnostics carry `c_i` and the entropy of `softmax(s~_i)`.

Why this shape: the trunk supplies goal reading (words in E_t and skill descriptions); the adapter
supplies the numeric precision J1 had; the heads make it one pass; the confidence head is what makes
it a System One model rather than a classifier.

### 8.4 Training

Stage A, supervised (COMMITTED). Data rows `(x_i, u_i, m_i) -> (y_skill, y_replace, y_params)` from
two sources: (1) the recorded decision rows (`eval/record_traces.py`, labels = J0's decisions),
(2) decisions under *varied executive states*: E_t variants with reordered priorities, different goal
sets, different constraints and horizons, rendered through the same renderer, labelled by the rule the
variant implies (and, when the owner runs the top LLM, by the LLM's own choice). Source (2) is what
teaches goal reading; source (1) alone teaches J0. The LLM-written states come out of the executive
store through `executive/export.py` (`export_history` / `load_history`, JSON lines, added 2026-10-17).
  `L_A = CE(s~, y_skill) + BCE(r, y_replace) + lambda_p ||p - y_params||^2 [+ sum_k CE(s~^{(k)}, y^{(k)})]`
Stage B, calibration (COMMITTED). With the heads frozen, fit the confidence head on held-out-like rows:
  `L_B = (c - 1[argmax s~ = y_skill])^2`   (Brier, a proper scoring rule, [RLCR]); report ECE = sum_b (n_b/N) |acc_b - conf_b|.
Freezing the heads during B is the non-hackable form: confidence cannot improve by changing answers ([RLCR-HACK]).
Stage C, outcomes (COMMITTED as a harness, HYPOTHESIS as a result). PPO over goal decisions on
`SimGoalDecisionEnv` with `apply_fn(params, obs) -> (s~, V)`; the trunk stays frozen, adapter and heads
train; reward = the arena's reward (`jev/sim_goal_env.py`). Running it is the training side's work.

### 8.5 Speed budget

`T_decide = T_render + T_trunk + T_heads` per batch of lanes, host mode. Today: J1 0.1 ms (heads only,
compiled); J-LM 471 ms (token by token). Targets: T_trunk measured first (G2 below); the decision
cadence is 0.5 s, the host budget 20 ms (`eval/arms.py J0_LATENCY_BUDGET_MS`), a miss keeps the goal
(`jev/runner.py`). The plan head amortises one pass over K_plan decisions.

### 8.6 Gates

| Gate | Pass condition | Depends on |
|---|---|---|
| G1 pipeline | with a deterministic fake trunk (hashed bag of tokens, no weights), stage A reaches >= 0.95 skill agreement with J0 on the 737 held-out rows (J1 reached 0.996 with numeric input only) | nothing new |
| G2a slow path | one forward pass of the encoder over the stable rendering per executive version, CPU, in process; measured 2026-10-17 under contention: ModernBERT-base 544 ms, Qwen3-0.6B 1,529 ms at batch 1 (`LEDGER.md`); uncontended rerun pending | `transformers` (installed 2026-10-17) |
| G2b fast path | per-decision heads over the cached embedding and the numeric belief: p50 under 1 ms for 64 lanes compiled, the same order as J1 | the scaffold |
| G3 goal reading | on a held-out split of E_t variants (goal order, sets, constraints never seen in training), skill agreement >= 0.90 with the rule the variant implies; J1 is the control and is expected to fail this split | stage A on source (2) |
| G4 online | four arms plus `js1` at 512 ticks on held-out seeds: J-S1 composed rate not below J0's Wilson lower bound; interrupts answered; 0 misses at the measured latency | G1-G3 |
| G5 calibration | ECE <= 0.05 on the held-out rows after stage B | G1 |
| G3b growing library | with the matching head, a skill added after training (unseen description) receives sensible scores: on rows where the rule selects it, agreement >= 0.80 without any retraining of the head; the fixed softmax head cannot be evaluated here at all | G1 |

### 8.7 Files (to build)

```
experimental/neural_agent/jev/system_one/
  __init__.py
  trunk.py        TrunkEncoder protocol: encode(list[str]) -> float32[B, D]; HashedBagTrunk (deterministic, no weights);
                  TorchTransformerTrunk adapter (imports transformers lazily; refused with a clear message when absent)
  heads.py        JAX: init_heads(key, D, K), apply_heads(params, h, u, current) -> (skill_logits, replace_logit, params, confidence, value[, plan])
  instance.py     SystemOneJev(JevInstance, HOST): renders, encodes, applies heads, builds the ManagedGoal, diagnostics
  data.py         rows from a decision dataset + an ExecutiveState -> (texts, u, mask, labels); executive variants generator (templated)
  train_supervised.py   stage A (optax, like train_j1.py); writes heads params .npz and metrics.csv
  calibrate.py    stage B; writes the confidence head and ECE
  probe_latency.py      G2
tests/experimental/neural_agent/test_na_system_one.py
```
`eval/arms.py` gains the `js1` arm (host decider, like `j_lm`), `run_composition.py` the choice.
`record_traces.py` writes the run's `executive.json` next to the traces so rows can be re-rendered.

### 8.7b Status after the scaffold (2026-10-17, main-session gates)

IMPLEMENTED: `jev/system_one/` (`trunk.py` with `HashedBagTrunk` and the lazy `TorchTransformerTrunk`,
`heads.py`, `data.py`, `instance.py` (`SystemOneJev`, host mode, two timescales, version-keyed cache),
`train_supervised.py`, `calibrate.py`, `probe_latency.py`), the `js1` arm and `--js1-heads`, 24 tests in
`test_na_system_one.py` (`75 passed, 1 skipped` with `test_na_eval.py`), ruff clean. Built by an Opus
subagent; every gate below was re-run by the main session (`LEDGER.md`).

| Gate | Status |
|---|---|
| G1 pipeline | PASS: held-out skill agreement 1.0000 (737 of 737) with the matching head and the hashed trunk; train 0.9993; replace agreement 0.9932; seeds 1 and 2 (subagent) 0.9973 and 0.9986; fixed-head ablation 1.0000 |
| G2a slow path | PASS for the two-timescale design: uncontended, ModernBERT-base 72.5 ms p50 per pass over the 330-token rendering at batch 1 (462.9 ms at batch 8), Qwen3-0.6B 245.5 ms (1,527.6 at batch 8); the contended run had read 544 / 1,529 ms. About 73 ms per executive version, off the tick; never per decision |
| G2b fast path | PASS: hashed trunk, no trunk call per decision (counted: 0); full decision p50 0.175 / 0.206 / 0.292 ms at batch 1 / 8 / 64, compiled pass alone 0.047 / 0.141 / 0.208 ms |
| G3 goal reading | NOT MET with either trunk on 10 held-out executive variants: hashed 0.8069, ModernBERT-base (frozen, mean-pooled, ids as descriptions) 0.7843, and the pretrained trunk also lost 3 points on the recorded rows (0.9701 vs 0.9986). A mean-pooled text embedding carries priorities, goal sets and constraints poorly, and 22 training variants are few. Next: a structured, catalogue-agnostic executive encoder in the fast path (per slot: skill description embedding, priority, validity, params; pooled over slots) beside the text embedding; more variants; LoRA on the trunk later |
| G3b growing library | mechanism shown (a skill appended after initialisation is scored and can be chosen, test (b)); the quality threshold waits for the pretrained trunk |
| G4 online | not run |
| G5 calibration | degenerate: stage A is 99.9% correct on the fit rows, so the confidence head saturates at 1.0 and ECE 0.0 says nothing; needs harder rows (variants, outcomes) |

Design decisions taken in the scaffold and kept: (1) the current skill enters the fast path as its
description embedding, not a one-hot, so no weight depends on K (FORMULAS M2 amended); (2) the stable text
excludes the executive's version and issue tick, so an identical state re-issued keeps its encoding; (3)
every non-idle goal gets J0's 0.5 s commitment, 10 s horizon and emergency mask; J0's 2 s evade commitment
is not reproduced (known difference against J0 in the online comparison; follow-up: per-skill controls);
(4) the replace label is the recorded core switch, as for J1 (61 of 2,966 rows differ from J0's proposal
because a commitment held the goal). No skill description texts exist yet, so skills are described by
their ids; writing descriptions is the first step towards G3b quality. The subagent's own caveat stands:
its latency figures were taken on a loaded machine; the main session's rerun above was quieter.

### 8.9 Goal reading first: the structured executive encoder (owner refocus, 2026-10-17)

Owner: "the goal is that this fast middle layer is just tuned to reading the state/goals fast and to
be able to query the lower level neural actions faster, right now we need to focus on making it able to
read state and llm inputs fast as we have not finalized or gotten close to finishing the neural library".
So the open gate that matters is G3, goal reading, and it must not depend on which skills exist. Skills
are placeholders with ids and short descriptions; the catalogue is a variable K.

**Why the text embedding alone failed G3 (0.78 / 0.81).** E_t is structured data: per goal slot a skill,
a priority, params, constraint labels, a horizon, a deadline. A mean-pooled embedding of its rendering
carries the words but blurs the numbers and the slot structure. J1 read the lowered arrays directly and
reached 0.996 on recorded rows, but its one-hot slots tie it to K.

**Structured, catalogue-agnostic encoding (COMMITTED).** For each goal slot k = 1..MAX_GOALS:
  `q_k = W_slot [ e_{skill_k} ; priority_k ; valid_k ; params_k ; horizon_k / H ; has_deadline_k ; c_k ] + b_slot`
with `e_{skill_k}` the cached description embedding of the slot's skill (zeros for an empty slot), `c_k`
the embedding of the slot's constraint labels (cached per distinct label; zeros for none), H a fixed
horizon scale (30 s). Pooled over slots with validity-masked attention:
  `alpha_k = softmax_k( w_a . q_k + log(valid_k) )`,   `g_E = sum_k alpha_k q_k`,
plus the top-priority valid slot's vector `q_top`. Fast-path fusion becomes
  `z_i = LayerNorm([ h_E ; g_E ; q_top ; a_i ; e_current_i ])`
where `h_E` (the text embedding) is kept behind a flag for the ablation "text off". No weight shape
depends on K or on the number of distinct constraint labels. Everything here changes only when E_t
changes, so it belongs to the slow path's cache; the per-decision cost stays the heads.

**Goal-reading benchmark (COMMITTED).** A generator of executive states that is independent of the
real skill library: a placeholder catalogue of K skills (K sampled in 3..8, ids like `skill_a`, with
one-line generated descriptions), 1..MAX_GOALS goals with distinct skills, random priorities, params,
horizons, deadlines, and constraint labels drawn from a small vocabulary that includes `on_damage`.
The teacher is the generic form of J0's rule, labelling any such state together with a belief row:
  if the damage window is open (ticks_since_damage known and < 40) and a valid goal carries `on_damage`,
  choose that goal (highest priority among them); else the highest-priority valid goal; else idle.
`target_replace` = chosen skill differs from the current skill. Splits are by executive state, and a
second split holds out catalogue sizes (train K <= 5, test K in {6, 7, 8}) to show K-independence.
Belief rows come from the recorded datasets (real belief distributions), the current skill from the row.

**Gates.** G3: skill agreement >= 0.90 on held-out states, K-held-out split reported separately; J1
(fixed one-hot slots) as the control that cannot run on the K-held-out split. G2b unchanged: fast
path per decision within an order of magnitude of J1 (measured 0.29 ms at batch 64). G3 reading
latency: time from a new E_t to a ready cache (structured path alone, text path alone, both).

**Status 2026-10-17 (evening), main-session verification.** Built as specified in `jev/system_one/`
(`heads.py` slot attention and `encode_executive`, `data.py` goal-reading states/rows/split and the generic
rule, `instance.py` slow path once per version, `train_supervised.py --goal-reading N --text-path on|off`,
`probe_latency.py` reading latency). G3 FAILS at N = 256 training states (the encoder memorises the 101
states that survive the K <= 5 split: 0.79 held-out states / 0.63 held-out catalogue sizes, same on every
trunk and text setting, no regulariser or schedule fixed it) and PASSES at N = 4096 (0.9599 / 0.9489 text
off, 0.9608 / 0.9457 text on, recorded rows 1.000; reproduced by the main session to four digits). The gate
is therefore defined at N = 4096, which the generator produces for free and trains in about 15 s. The
per-decision path makes no trunk call (probe); reading a new E_t on the structured path takes 0.16 ms.
Ledger: "G3 goal reading with the structured executive encoder".

### 8.10 The shared scratchpad and natural-language goals (owner answers, 2026-10-17)

Owner's answers, verbatim where it matters:
1. Goals: "the goal will likely be a json (jev like) formatted structure that contains some unique
   natural language, it is yet to be determined but it will always be in context of the state and some
   idea that the model has, it is up to us to interpret it properly".
2. Conditions: "we dont know, it will be natural language we have to figure it out, the job of the llm
   is to communicate it cleanly and our job to interpret it".
3. The state: "the llm will effectively be modifying the same state as it goes, think of the state as
   the scratchpad between all 3 layers where all models pass their information in between (like a
   shared memory) there are different parts for each component but this is primarily where information
   is shared, we have to find an optimal way to structure it such that each model gets the information
   it needs and has the proper context to make better decisions".
4. No live LLM run until the lower level is finished.

**Consequence 1: the text path is not optional.** A goal carries typed fields and unique natural
language, so the decision layer must interpret words, not only numbers. The structured encoder of §8.9
stays for the typed fields; the text enters *per goal*, not as one pooled rendering of everything:
  `e_text_k = f_theta(text_k)` for the natural-language fields of slot k (intent, condition), cached per
  distinct text, entering `q_k` beside `e_skill_k` and `c_k`.
A frozen encoder's phrase embeddings separate paraphrases to a degree; where they do not, the trunk is
fine-tuned on the goal language (contrastive paraphrase-to-predicate), which is the first use of a
trainable trunk in this design. This replaces the "text off" ablation as the default: text is on, per goal.

**Consequence 2: conditions are interpreted, not enumerated.** Nobody fixes the vocabulary. The decision
layer learns to map condition text to predicates over the belief it can evaluate (target visible,
distance band, own health fraction, damage window, time since the target was seen, closing speed) and
over the executive's own facts. The teacher for this is procedural at first: a goal-language generator
writes conditions and intents as paraphrase families with known semantics ("if you get hit", "when
damaged", "after taking a hit" -> the damage window), labels every state with the rule those semantics
imply, and holds out whole paraphrase families, so the gate measures reading of wording never seen.
Gate G3-NL: skill agreement >= 0.90 on held-out paraphrase families and on held-out catalogue sizes.
When the top LLM is allowed to run (answer 4: later), its real goal texts become the second source.

**Consequence 3: one shared scratchpad, sectioned by writer (COMMITTED as a design; code to follow).**
Today the three records are separate: E_t (LLM to JEV), B_t (belief, shared), GoalReports (skills to LLM
through the harness). They become sections of one versioned document, the scratchpad S_t:

| Section | Writer | Readers | Content |
|---|---|---|---|
| `executive` | the LLM, through the harness | JEV, LLM (its own last version) | goals as JSON with typed fields and natural-language `intent` and `when`; facts; budget; the LLM's working notes ("the idea it has") |
| `belief` | the belief module | JEV, LLM (rendered), skills (numeric) | B_t features, known flags, latent summary |
| `decision` | the JEV | LLM, skills | the goal in force, confidence, entropy, the last few decisions with their reasons as typed fields |
| `skills` | the dispatcher and skills | JEV, LLM | active skill, its runtime state summary, outcomes (today's GoalReports) |
| `events` | the runtime | all | the decision events of the last window |

Rules: each section has one writer; every write bumps the document version; each reader gets a *view*,
a rendering sized for it (the LLM a compact text, the JEV typed arrays plus per-goal text, the skills
the goal payload only), so no model receives more than it needs and every model receives the context
of the others' last state. The existing records stay as the typed building blocks; `shared_state.py`
composes them, carries the JSON schema and the views, and is the single thing the executive harness
reads and writes. This is the "shared memory" the owner described, with the ownership that keeps it
consistent.

**Build order from here.** (a) The structured encoder (§8.9, in progress). (b) Per-goal text in the
slot vector and the goal-language generator with paraphrase families; G3-NL. (c) `shared_state.py` with
the sections, the views and the schema, wired into the harness and the J-S1 slow path; the ADR's E_t
becomes the `executive` section. (d) Trunk fine-tuning on the goal language only if (b) fails its gate.
The live LLM run waits for the lower level, as instructed.

**Status 2026-10-17 (evening).** `shared_state.py` exists and passes its module gate (9 tests; ledger entry
"shared scratchpad S_t built"): `GoalEntry` = GoalSpec + `intent`/`when`/`note`, the five sections with
one writer each, `SharedState.write` (ownership, version bump), the closed JSON schema, and the three views.
Wired the same evening (ledger "scratchpad wired"): `ExecutiveHarness(shared_state=...)` writes every
accepted proposal into the `executive` section as the LLM's writer; `run_session(shared_state=...)` runs a
`SessionScribe` (`shared_state_writers.py`) that writes belief, events, skills and decision at the host
cadence and each GoalReport into the skills outcomes. The J-S1 slow path reading the `executive` section
and its per-slot texts waits for §8.9.

### 8.11 The goal-language generator: the teacher for natural-language goal reading (2026-10-17)

**Why.** G3-NL (M12) needs labelled goal language before the top LLM is allowed to write any (owner answer
4). The generator writes goals the way the LLM will, typed fields plus unique words, with the meaning known
by construction, and it is the only source of skill descriptions for *synthetic* catalogues, so held-out
catalogue sizes and held-out skills come from one mechanism.

**Families.** Two kinds, each a `family_id`, a meaning and at least twelve paraphrases (imperative,
descriptive, with and without the object noun, short and long, synonyms; at least a third of a family's
paraphrases share no content word with its canonical description, so a bag of words cannot solve the gate).
- *Intent families*, one per thing a skill does. Each carries a canonical `description` in the register of
  `skill_library/descriptions.py` (the matching head must meet the same kind of text as a real catalogue
  entry) and the real skill ids it names today: pursue -> `pursuit`, evade -> `evader`, hold -> `idle`.
  Further families (reach a point, patrol an area, retreat to cover, guard a position, search for the
  target, ...) name no real skill yet; they stand in for skills the library will gain and populate the
  synthetic catalogues.
- *Condition families*, one per predicate over B_t (`BELIEF_FEATURE_NAMES`) the decision layer can
  evaluate: `always` (no condition, in words or empty), `on_damage` (ticks since damage known and below
  40, the J0 window), `target_visible`, `target_not_visible`, `target_lost_for(ticks)`,
  `distance_below(blocks)`, `distance_above(blocks)`, `self_health_below(fraction)`,
  `target_health_below(fraction)`. A numeric family renders its argument in several surface forms (digits,
  words, with and without a unit) and the meaning carries the number. An unknown feature (known flag
  False) makes a predicate False.

**Catalogues.** A catalogue is `skill_id -> description`. The real one is `SKILL_DESCRIPTIONS`. A
synthetic one of size K has `idle` first (its real description) and K-1 ids `skill_<n>` whose descriptions
are the canonical descriptions of distinct intent families. An intent family *resolves* in a catalogue to
the id it names, or to the id whose description is the family's; an intent that resolves to nothing is not
a valid goal for that catalogue, and the generator does not emit it there.

**Goals and states.** A goal is a `GoalEntry`: `skill_id` typed (the contracts require it today; see the
follow-up below), `priority`, `horizon_seconds`, optional `deadline_tick`, `intent` from the resolved
family, `when` from a condition family, `note` from a small pool of neutral remarks (distractors) in a
fraction of goals, and `constraints` carrying the predicate label as a typed hint only at a configurable
hint rate (default 0 for the gate: the gate measures reading of words). A state has 1 to `MAX_GOALS` goals
in an `ExecutiveSection`, with a `SlotMeaning(intent_family, skill_id, predicate)` per slot.

**Teacher (M13).** `S_cond = {k : v_k and c_k != always and c_k(B_t)}`; the chosen slot is the
highest-priority member of `S_cond` when it is non-empty, else the highest-priority slot among
`{k : v_k and c_k = always}`, else idle; `y_replace = [chosen != current]`. With the composition state
(pursue/always at 1.0, evade/on_damage at 0.5) this is exactly M10.

**Splits.** (i) *Held-out wordings*: a fraction of each family's paraphrases never appears in training, the
reader meets seen meanings in unseen words. (ii) *Held-out skills*: whole intent families (with their
descriptions) never appear in training catalogues, the reader must match unseen intents to unseen
descriptions, which only the catalogue-agnostic head can do. (iii) *Held-out catalogue sizes*, as M11.
Gate G3-NL: skill agreement with M13 >= 0.90 on each.

**Reading modes for the gate.** *Typed*: the slot shows the skill's description embedding `e_skill_k`
and the texts; the reader must read the condition. *Text-only*: `e_skill_k` is masked to zero and the
reader must resolve the skill from the intent text through the matching head. Both are reported.

**Follow-up, not now.** The contracts require a typed `skill_id` per goal. The owner's answers imply the
LLM may name what it wants without knowing the library's ids; the eventual contract is a slot with an
unresolved skill (lowered index -1) that the matching head resolves. That change touches `contracts.py`,
the lowering, J0 and the runner, and waits until the text-only mode passes G3-NL on the generator.

Files: `executive/goal_language.py` (families, predicates, catalogues, the sampler, the teacher, the
splits, a JSONL dump CLI), `tests/.../test_na_goal_language.py`. The row builder that joins generator
states with belief rows and the per-goal text embedding (M12) lives in `jev/system_one/` and follows §8.9.

**Status 2026-10-17 (night).** Built and verified (ledger "goal-language generator built"; commit 9039f3f):
10 intent families (pursue, evade, hold name today's skills; reach_point, patrol, retreat_to_cover,
guard_position, search_target, flank, follow_ally stand in for future skills and populate the synthetic
catalogues, K up to 10) with 15-16 paraphrases each, 53-67 % of them sharing no content word with the
family's description; 9 condition families with 14 templates each, numeric arguments on grids (blocks 2-12,
ticks 20-200, fractions 0.20-0.80) rendered as digits, words, units or percentages. A slot is unconditional
with probability 1/3; `always` is written as "" half the time; `target_lost_for` is written in ticks only
(an LLM may say seconds: known limit). Unit trap for the row builder: the teacher takes raw ticks while the
belief publishes the tick columns as counter / 256. The G3-NL stage (text columns in the slot vectors, rows,
modes, splits; task #28) is built in `jev/system_one/` against this section and FAILS its gate in
every mode and trunk (ledger "G3-NL ... FAIL"): held-out wordings 0.53-0.58, held-out families 0.52-0.66,
held-out sizes 0.76-0.84 (typed and text-only; hashed and ModernBERT-base). See §8.13.

### 8.12 Owner rulings on the open questions (2026-10-17, night)

Asked after the scratchpad and the generator landed; answers verbatim where short.
1. *Who owns commitment and duration?* "stay with skill, you only call the skill and the feedback to the
   shared state when done." Consequence: the decision layer emits skill, params and replace only. The
   skill's contract owns `minimum_commitment_seconds`, `maximum_duration_seconds` and the emergency
   events that may interrupt it; the runner and every JEV instance take the control fields from a
   per-skill table built from the contracts, not from the JEV's constants (today J0's constants in
   `jev/scripted.py`, copied by `network.py` and `system_one/instance.py`). The skill reports when it is
   done; that report is the feedback into the scratchpad (already: `GoalReport` -> `skills.outcomes`).
2. *One NPC or a group per LLM?* "there are both options." Consequence: a goal needs an addressee (the
   agents it applies to; default all), the lowering needs a per-lane allowed mask from it, and the LLM
   view needs per-skill aggregation across agents plus per-agent detail on request. Design to follow;
   nothing in reading depends on it.
3. *Can the LLM be told the condition vocabulary?* "yes." Consequence: the system prompt lists the
   predicates the layer can evaluate (the condition families of §8.11, one example each) and tells the
   LLM to write conditions in its own words using them; the reader still handles any wording. Offline
   change to `executive/anthropic_backend.build_system_prompt`; no live call.
4. *An intent that names something the library cannot do?* "then we have to choose the best possible
   choice." Consequence: no refusal and no fall-through. The matching head always picks the nearest
   skill by description; the confidence head's value, written to the `decision` section, is the LLM's
   signal that the match was weak. The generator gets a nearest-real-skill teacher for intents whose
   family is absent from the catalogue (an "absent families" evaluation beside the three splits of
   §8.11); until it exists, this is a design commitment, not a measured property.

### 8.13 Reading words: interpret, then evaluate (proposal after the G3-NL failure, 2026-10-17 night)

**What failed.** Putting each goal's text embedding into its slot and letting slot attention + an MLP learn
the rule end to end (M9 + M12) does not generalise: on unseen wording the trained reader scores below
the rule "take the top-priority unconditional goal" (0.53 vs 0.63), agreement falls with every added
conditional goal even on training rows (0.99 -> 0.90 for 1 -> 4 goals), and mixing goal-language rows in
dragged the recorded-rule agreement under G1's bar (0.88). Two causes, both structural: (1) a pooled slot
code never meets the belief slot by slot, so "which condition holds now" has to be re-derived after
pooling; (2) a frozen pooled text embedding (hashed bag, or ModernBERT mean pool) neither carries an unseen
paraphrase to its predicate nor preserves a number ("seventy-five percent", "7 blocks") precisely enough
to compare with the belief. More data would not fix (1); no frozen pooled trunk fixes (2).

**Proposal.** Split reading into the two things the owner named: *interpret* ("our job to interpret it")
and *decide fast*.
- *Interpret, once per executive version (slow path, off the tick).* For each goal, read the words into
  the vocabulary the layer can evaluate: `when` -> (predicate kind from the §8.11 families, numeric
  argument), `intent` -> the matching skill (the catalogue-agnostic head of M3b, on the intent text and
  the skill descriptions). The kind comes from a text encoder plus a small classifier; the argument from
  a deterministic number reader (digits, number words, units, percentages) beside the encoder, because
  numbers must be exact. The interpretation is typed, so it is written into the `decision` section and
  the LLM can see how its words were read (and correct them in its next version).
- *Evaluate, per decision (fast path).* The predicates are evaluated on B_t exactly, the priority rule
  (M13) picks the slot, and the learned heads keep replace, params, confidence and value over the typed
  slot code of M9. Nothing about the words happens on the tick, so the 0.3 ms decision stands.

**Why this is still a System One model.** The LLM writes free text; the layer produces typed,
calibrated decisions in one pass per tick. The interpretation stage is the "reasoning within" the owner
asked for, made explicit and inspectable, and it is the part that benefits from learning on real LLM
wording later. It is also what the owner's rulings assume: the LLM is told the predicate vocabulary
(§8.12 item 3), and an intent that matches nothing is read as the nearest skill (item 4).

**Gate for the interpret stage (replaces G3-NL's end-to-end form).** On held-out wordings and held-out
families from the generator: predicate-kind accuracy >= 0.95, argument exact-match >= 0.98 where the
family is numeric, intent -> skill accuracy >= 0.90 (held-out families: nearest description); then the
end-to-end agreement with M13 follows from these and is reported for the record.

**The owner's call: the text encoder for the kind and the skill match.**
(a) A frozen sentence-embedding model (a small open-weights encoder trained for sentence similarity,
    about 20-30M parameters, a few ms per sentence on CPU) with a trained classifier on top; local,
    no fine-tuning, the quickest test of whether paraphrases land together.
(b) Fine-tuning ModernBERT-base (or the small encoder) contrastively on the generator's
    paraphrase-to-predicate and intent-to-description pairs, M12's own fallback: the first trainable
    trunk, more work, most likely to pass, and the natural place to add the real LLM's wording later.
Recommendation: (a) first because it is a one-day test with the harness already built; (b) if (a) misses
the kind-accuracy bar on held-out wordings. Both keep the two-timescale design: encode once per
version, never per decision.

**Measured the same night (ledger "interpret stage ... frozen encoders"; `jev/system_one/interpret.py`,
`interpret_benchmark.py`).** The number reader passes (argument exact-match 1.0000 on every generated
surface form). No frozen encoder meets the kind bar: hashed 0.41, ModernBERT-base mean-pooled 0.68,
MiniLM-L6-v2 0.71 on held-out wordings, with systematic polarity errors (every "not visible" text read as
distance-below; below versus above near chance), the known weakness of pooled frozen embeddings with
negation. Intent -> skill: MiniLM with a fitted map 0.91 on held-out wordings (the one passing cell),
zero-shot 0.85 on held-out families (the best unseen-skill reader; only two families held out, so
fragile). Option (a) is therefore out; (b), fine-tuning, is the path, which is J-S2 (§9). MiniLM
(22M, 0.45 s per 1,000 texts, 5x faster than ModernBERT-base) joins the backbone candidates of §9.2.

### 8.8 What is deliberately not in J-S1

No text generation, no tool use, no chain of thought at decision time (System One). No external
model or service: the only external component is the top LLM. No change to the goal contract: the
"series of quick actions" is the sequence of decisions, plus the optional plan head.

## 9. J-S2: the fine-tuned System One interpreter (design, 2026-10-17 night; awaiting the owner's go)

**Position.** J-S2 is the Jev recipe applied where our CPU has time for it. One forward pass of a
pretrained backbone fine-tuned on our decision data, every output field read in parallel as a small
typed classification with a calibrated probability, run once per executive version. The per-tick path
is unchanged: the typed structured encoder and light heads of §8 (0.3 ms), with the predicates J-S2 read
evaluated exactly on B_t. J-S1's frozen-trunk text path is retired to the benchmark floor.

### 9.1 One query: input and outputs

Input, rendered once per executive version (and re-rendered when the catalogue changes): the executive
section's goals as the LLM wrote them (typed fields plus `intent`, `when`, `note`), its facts and notes,
and the skill catalogue as `id: description` lines. Every goal and every catalogue entry gets a marker
token, so the backbone's token states at the markers are the span representations the heads read.
No belief in the input: J-S2 interprets, it does not decide per tick (§9.5 for the optional decision mode).

Outputs, all from the same pass, per goal slot k (up to MAX_GOALS):
- `skill_k`: a distribution over the K catalogue entries, `softmax_j (q_k^T W e_j)` with `q_k` the goal
  marker's state and `e_j` the catalogue entry's marker state from the same pass. No weight depends on K;
  an unseen skill is matched through its description (owner ruling: always the best choice, §8.12).
- `kind_k`: a distribution over the predicate kinds of §8.11 (nine, from the `when` span).
- `has_argument_k`: whether the condition carries a number; the number itself comes from the exact
  reader (`interpret.read_number`), never from the network.
- Later, per lane and only in the decision mode: `skill_now`, `replace`, `confidence`.
Every head is a softmax, so the probabilities are calibratable (Brier, temperature) and the typed value
is the argmax: typed by construction, parallel, one pass.

### 9.2 Backbone and heads

Primary: ModernBERT-base (149M; 72.5 ms per pass over 330 tokens on this CPU, uncontended, measured
2026-10-17), the Laya precedent. Heads: the bilinear matcher `W` (d x d), the kind classifier
(d -> 9), the argument flag (d -> 1), on marker states. Fine-tuning: the top 4 of 22 layers plus the
heads trainable, the rest frozen (backward cost about a fifth of a full fine-tune; unfreeze more if the
gate misses). Fallbacks, in order: a 22M six-layer encoder (MiniLM-L6-v2, the best frozen reader in the §8.13 benchmark; about 8 ms per pass) if speed binds; the
decoder path, Qwen3-0.6B with field readouts at the marker positions and LoRA (245 ms per pass measured;
int4 later), the owner's KEV idea, if the encoder misses accuracy.

### 9.3 Training

Data: the goal-language generator, 32k states over real and synthetic catalogues (free; states are the
unit, not rows: no belief is needed to label an interpretation), labels per slot = the generator's
meanings (skill, kind, has-argument). Held out exactly as §8.11: wordings, skill families, catalogue
sizes. Stage A: supervised, cross-entropy on the three heads, 3 epochs, batch 8, AdamW; on this CPU
about 4-6 hours with the top layers trainable (one overnight run), under an hour for the 22M encoder.
Stage B: temperature scaling and a Brier check on held-out wordings. Stage C (later, with the lower
level): outcomes through the harness, RLCR-style with the non-hackable scoring.

**Learner-visited states (added 2026-10-17 night, from the G4 diagnosis).** Behaviour cloning on the
teacher's own rollouts agreed 1.000 on recorded held-out rows and still composed worse online (0.40
against the rule's 0.53): the learner under-reacted to the evade transition, then sat in states the
teacher's traces never contain, because the teacher leaves them on the first tick (ledger "P1d
shadow"). Stage A therefore trains on the learner's own visited decision states labelled by the teacher
rule, merged with the teacher's rollouts, in rounds (DAgger), until the online rate meets the bar. The
teacher is a rule, so the labels are free; the same applies to J-S2's programs once the behaviour
generator's canonical programs label a learner's states.

### 9.4 Runtime placement and speed

Slow path, per executive version: render (0.1 ms) + one pass (73 ms, batch 1; int8 dynamic quantisation
to be measured, expected about 2x) + the number reader -> a typed `GoalInterpretation` per slot, written
to the `decision` section (the LLM sees how its words were read and can correct them). Per tick: the
interpreted predicates evaluated exactly on B_t, the priority rule (M13), and the §8 heads for replace,
params and confidence; 0.3 ms, zero backbone calls (G2b holds). With one to four NPCs per scratchpad the
slow path fits inside the LLM's own cadence many times over.

### 9.5 Optional decision mode (not built first)

With the belief summary appended to the input and `skill_now`/`replace`/`confidence` heads per lane,
J-S2 is a full Jev-style decider at the host cadence (0.5 s). On this CPU it fits only for a handful of
lanes (batch 8 was 463 ms); on a GPU it fits all lanes. Built only if the interpret mode passes and the
owner wants decisions from the backbone itself.

### 9.6 Gates

- I1 interpretation: kind accuracy >= 0.95 on held-out wordings; argument exact-match >= 0.98 (reader);
  intent -> skill >= 0.90 on held-out wordings, held-out families and held-out sizes.
- I2 end to end: agreement with M13 >= 0.90 on the three splits with interpret + exact evaluation
  (this replaces G3-NL).
- I3 speed: a new executive version read in <= 150 ms at batch 1 on this CPU; zero backbone calls per
  decision; the recorded-row agreement (G1) stays >= 0.95 because the per-tick heads are untouched.
- I4 calibration: ECE <= 0.05 for `skill_k` and `kind_k` on held-out wordings after temperature scaling.
- I5 no regression: the composition rows on the legal open-flat scene unchanged for J0; the js1 arm
  with J-S2 interpretation not below J0's lower bound (G4, when run).

### 9.7 Build order (each step gated before the next)

1. Floor and reader (running): `interpret.py` number reader and the interpretation benchmark; the frozen
   encoders become the floor row. Gate: reader exact-match >= 0.98 on every generated surface form.
2. Fine-tuning harness (`jev/system_one/finetune.py`, PyTorch): rendering with markers, dataset from the
   generator, the heads, the training loop with the top-N-layers switch, evaluation on the three splits,
   checkpoint save/load. Gate: a 2k-state smoke run learns the training wordings (>= 0.99) in minutes.
3. The first real fine-tune of ModernBERT-base (overnight on CPU). Gate: I1.
4. Calibration and speed: temperature scaling, ECE, int8 latency, the probe. Gates: I3, I4.
5. Integration: `SystemOneJev.set_shared_state` calls the interpreter once per version and writes the
   interpretation to the `decision` section; the per-tick evaluator uses it; end-to-end test; the js1 arm
   through the composition driver. Gates: I2, I5.
6. If I1 misses: more states, more trainable layers, then the decoder path (§9.2), in that order.

### 9.8 What is reused and what is retired

Reused: the generator and teacher (§8.11), the splits, the number reader, the benchmark, the scratchpad
and its views, the typed structured encoder and heads for the per-tick path, the two-timescale instance
code and caches, the LLM prompt vocabulary (§8.12). Retired as a mechanism: per-goal frozen text
embeddings in the slot vectors (`--goal-language` rows, `--language-mode`), kept only as the floor row of
the benchmark and as the record of why the design changed.

Deprecation in code (2026-10-17 night, cleanup): the pooled text path (`--text-path on`) and the per-goal
text embeddings carry `DEPRECATED (2026-10-17)` markers in the docstrings of `train_supervised.py`,
`data.py` and `heads.py` and in `train_supervised.py --help`; behaviour unchanged, the benchmark floor
still runs. `jev/lm/` (J-LM over a local model) is marked the same way: an experiment arm, not a component.
`jev/kev.py` was removed.

### 9.9 Behaviour programs: the series of actions (owner principle, 2026-10-17 night)

Owner: "the principle idea of the middle layer is that it can give a series of neural actions to create
unique human like behaviors based on the llm input and state." This amends §9.1, §9.4, §9.6 and §9.7.

**What changes.** The decision model's output per intent is a *program*, not one skill: up to L = 4
steps, each step a skill chosen over the catalogue by description, an exit condition (a predicate kind
of §8.11 over B_t, or "done" when the skill reports completion, or a timer) and a hold-time bucket
(0.5, 1, 2, 4, 8 s) for human-like timing. A one-step program is today's single call. The LLM may write
a high-level intent ("ambush them at the door", "harass them but don't die"); composing the steps from the
library is the middle layer's job, which is where the "reasoning within" lives. All program fields are
typed and read in parallel from the same pass (multi-token-prediction style heads), so the recipe holds.

**Where "unique" comes from.** Programs are *sampled* from the calibrated distributions (temperature
tau) once per executive version per agent, so two agents given the same intent behave differently; a
program is re-sampled when a step's skill reports failure. The LLM's notes (style words: cautious,
aggressive, patient) are in the input, so the model learns to shift the program with them. Human-likeness
itself cannot be gated procedurally; its ingredients can: composition with timing, reactive interrupts,
sampled variety under validity, and later outcome learning (stage C).

**Execution (fast path, per cadence tick, exact, no backbone).** Per agent: evaluate the current step's
exit condition on B_t (or the skill's completion report, or the timer); advance or hold. The LLM's
conditional goals ("if hit, evade") act as interrupts by priority: a conditional goal that holds now
pre-empts the program (M13 over the goals that hold versus the current step); when it ends, the program
resumes at its step. The selection crossing to the neural layer stays one skill call plus params; the
skill's contract owns its commitment (§8.12). Replace, confidence and value heads stay. The decision
section holds the program, the current step and the sampled temperature, so the LLM sees the plan it
caused; the skills section reports step completions.

**Teacher for stage A: a behaviour-language generator** beside §8.11's: behaviour families (ambush,
kite / hit-and-run, patrol-then-engage, hunt, hold-and-defend, flee-and-regroup, harass, ...) each with
at least twelve paraphrases, a canonical program over *skill families* (so synthetic catalogues work:
reach_point -> hold -> pursue-when-within-4-blocks is "ambush" in any catalogue that has those three),
hold-time ranges, and style modifiers that change the canonical program in a known way (cautious:
longer waits, earlier exits). Held out as before: wordings, behaviour families, catalogue sizes.

**Gates added.** B1 program agreement (step skills and exit kinds, step-wise) >= 0.90 on held-out
wordings and >= 0.85 on held-out behaviour families (zero-shot composition is the hard case; the bar is
honest). B2 validity: 100 % of sampled programs name catalogue skills and evaluable exits. B3 variety:
at tau = 1, at least 3 distinct valid programs per intent in 16 samples, each with step agreement >= 0.7
to the canonical (varied but sensible). B4 online: on the composition task the intent "catch the target
but do not get hurt" composes pursuit and evasion not below J0's lower bound at 512 ticks (the
human-like behaviour test the repository already has).

**Plan deltas.** Step 2 gains the program heads and the behaviour generator (about half a day more);
step 5 gains the program executor in the fast path (a small exact state machine, about a day); the Mac
GPU makes every fine-tune a matter of minutes, so iterations on the behaviour head are cheap.

### 9.10 The request stream: how the middle and lower layers correspond (owner answers, 2026-10-17 night)

Owner's answers, condensed: the neural layer receives a *stream* of requests from the state, conflicting,
sequential or parallel; it interprets them, rejects the invalid ones and keeps acting as best it can, and
streams its status back into the state; the middle layer works in correspondence with that stream.
Conflicts are resolved greedily by an urgency or confidence metric, one request rejected. The combiner
should combine independent discrete actions (locomotion and attacks do not depend on each other). Requests
are structured: coordinates, directions and scalars are passed; the neural layer follows directions
clearly and thinks only at the low level (better combat, pathfinding). As many concurrent requests as
plausible.

**Requests (middle -> lower), a new scratchpad section `requests`, writer: the decision layer.** Per agent
an ordered list of `ActionRequest`: `request_id` (monotonic per agent), `issued_tick`, `expires_tick`,
`verb` (a skill id of the library), typed `args` (`target_position[3]`, `target_direction[3]`,
`target_entity`, `scalars[4]` named by the verb's contract, each with a presence flag), `mode`
(`parallel`, `after: request_id`, `interrupt`), `urgency` and `confidence` in [0, 1], `priority`,
`goal_slot`, `executive_version`, `program_step`. Coordinates travel typed, never in prose: the LLM's goal
schema gains an optional typed `target` (position, direction or entity) beside the words, and the belief
section carries the agent's and the target's positions, so the middle layer can fill `args` from the
state and the behaviour programs of §9.9 can name places.

**Status (lower -> state), the `skills` section extended, writer: the dispatcher.** Per request:
`accepted | rejected(reason) | deferred(conflict with id) | running | done(outcome) | aborted(reason)`
with ticks, the progress signals of the verb's contract while it runs, and per tick which requests drove
which action heads (the composition of the real action). This is the context the owner asked for: the
upper layers see what was actually done, what was refused and why.

**The combiner (lower layer, in the dispatcher), per tick and per agent, fixed and exact:**
1. Take the live requests (not expired, not done).
2. Validate each against the verb's contract: the verb exists; required args present and in range
   (positions inside the world, entity known to the belief); the verb's action scope is legal under the
   current action mask; `after` predecessors are done. Invalid -> `rejected(reason)`.
3. Greedy arbitration: sort by (urgency, confidence, priority, recency); grant a request if every action
   head in its scope is free, else `deferred(conflict with id)` until the holder finishes or expires.
   Independent heads co-run (locomotion with attack), dependent ones conflict and the loser waits.
4. Run the granted verbs' policies, each emitting factors for its heads only; merge into one factored
   action, unused heads neutral.
5. Stream the status back. The combiner never thinks beyond this; the thinking inside a verb (combat
   micro, pathfinding) is the skill's own.

**Correspondence (the middle layer, every cadence tick).** Read the status stream: `rejected` -> fix the
args or choose another verb (the program re-samples the step); repeated rejections of the same step ->
an information request to the LLM (§8.13 gaps); `done` -> advance the program; a stalled progress signal
-> re-sample; `deferred` beyond the step's hold time -> raise urgency or drop the request. The decision
section records the live program, its step and the requests it issued, so the LLM sees the plan and the
stream in one view.

**Contracts.** `SkillContract` gains a typed argument schema (which of position / direction / entity it
needs, its scalar names) and an explicit action scope by head group (locomotion, look, hands, posture);
`PARAM_WIDTH` 4 grows to an `ARG_WIDTH` block (3 + 3 + 1 + 4 = 11 floats plus presence flags); the
in-graph lowering for the compiled runner becomes an active-request table per agent of M concurrent
granted requests (M = 4, one per head group; the host-side queue is unbounded, "as many as plausible").
These replace the "one skill per lane" assumption of today's dispatcher and runner; the typed goal slots
of the executive section stay as they are (the LLM's goals), the requests are what the middle layer
derives from them.

**Gates.** R1 every invalid request is rejected with a reason and the valid ones still run (stub skills).
R2 no two granted requests ever share an action head. R3 the greedy order is (urgency, confidence,
priority, recency) and is reproducible. R4 `after` chains execute in order and parallel sets co-run.
R5 the middle layer reacts to a rejection within one cadence tick. R6 the compiled active-request table
produces the same actions as the host combiner on the same stream. Composition rows unchanged with
single-request programs (today's behaviour is the one-request special case).

**Plan deltas.** A new step before the J-S2 program heads: the request and status sections, the combiner
with stub skills, the contract and lowering changes, R1-R6 (about two days, host side first; the
compiled table follows with the library). The J-S2 program heads then emit request sets with typed args;
the behaviour generator writes canonical programs as request sets. Reading the LLM's words (§9.1-§9.4)
is unchanged.

### 9.11 Selecting skills: the typed id is the selector (measured, 2026-10-17 night)

Measured (ledger "P3"): matching an intent to a skill by its description over the owner's real catalogue
fails for every encoder tried: MiniLM-L6-v2 zero-shot top-1 0.42 (top-3 0.62) over the 119 learned and
command rows, 0.44 with a fitted map, about 0.05 with whole catalogue sections held out; ModernBERT-base
and the hashed floor lower; the fine-tuned pilot weights add 3-4 points. The size curve is the fact that
matters: 0.86 at four candidates, 0.60 at thirty-two, 0.42 at 119. The earlier 0.85 on synthetic
held-out families was the small-catalogue regime. The catalogue itself is not separable by one sentence:
near-synonym rows, parameters written as rows (walk / run / sprint share a description), terse command
descriptions, one identical pair.

Decision. The LLM's typed `skill_id` is the selector. The LLM is told the catalogue with ids and
descriptions in its prompt, and the executive schema already requires the id, so the middle layer must
not re-derive what the LLM knows. Matching by description is demoted to the fallback for an unknown or
absent id: it returns the top three candidates and writes a low-confidence flag into the `decision`
section so the LLM corrects the id in its next version; that is "the best possible choice" of §8.12
item 4 without pretending accuracy the measurements refute. Consequences:
- §9.1's `skill_k` head over the catalogue is retired as a selector and kept only as the fallback and
  for synthetic benchmarks; the interpret stage reads conditions (nine kinds, the natural-language part
  the layer must learn) and numbers.
- §9.9's programs name skills by id; the catalogue's nineteen activities are the canonical program seeds;
  a learned program head classifies over the current catalogue's ids, with description embeddings as an
  input for ids it has not seen, not zero-shot by description.
- Library growth: adding a skill is adding a catalogue row (contract and description); the LLM can use it
  in its next version with no retraining of the middle layer for selection; only programs that involve
  the new skill need data.
- Gate I1 changes: "intent -> skill >= 0.90" becomes: typed id validity 100 % by schema; the fallback's
  top-3 containment and flag precision are reported, not gated.

### 9.12 Reading words: what three experiments decided (2026-10-17 night)

Measured, in order: G3-NL (frozen text embeddings in the goal slots) failed on unseen wording (0.53-0.58);
the frozen interpret stage failed the condition-kind bar (0.71 at best, polarity errors); the fine-tune
pilot (ledger "P2") failed it too (0.75 / 0.82 on two held-out-family draws) while reading the *feature*
at 0.99: what a sentence embedding loses, frozen or fine-tuned on this data, is the side of a
comparison and the presence of a negation ("if it's concealed" has no negation word); and matching
intents to skills by description failed over the real catalogue (§9.11). The exact number reader passed
(1.000).

Decision. Structure where exactness matters; words as context.
- The goal schema gains a typed condition beside the words: `condition: {kind, argument}` with `kind`
  from the nine-predicate vocabulary the LLM is told (§8.12 item 3) and the argument a number. The
  typed `skill_id` is the selector (§9.11). `intent`, `when` and `note` stay as the LLM's words: context
  for the LLM itself, the record, and the fallback.
- The middle layer evaluates typed conditions exactly on B_t and applies the priority rule (M13); no
  learned reader is on the path when the LLM writes typed fields, which is the case the design asks for.
- Fallback for a goal that arrives with words only: a deterministic reader beside the number reader,
  feature words (visible, distance, health, damage, lost) plus comparator and negation words (under,
  below, within, closer / over, above, past, farther / not, can't, hidden, out of sight), producing
  (kind, argument, confidence); a low-confidence reading is written to the `decision` section as an
  information request so the LLM confirms or writes the typed field. Gate: agreement with the generator's
  meanings >= 0.95 on held-out wordings for the deterministic reader, reported, since the typed path is
  the contract.
- J-S2 as a fine-tuned reading backbone (§9.1-§9.4) is not built. What remains to learn in the middle
  layer: the per-tick replace, params and confidence heads (P1d relabeling), and the composition of
  behaviour programs from typed goals and the catalogue's activities (§9.9: the executor first, P4;
  learning from outcomes later). Reading a new executive version therefore costs a schema validation and
  a handful of predicate constructions, well under a millisecond, with the interpret stage gone.

What this keeps of the owner's intent: the LLM writes freely and keeps its ideas in words; the layer
below acts on the typed fields the LLM also writes, as Jev's schema-defined outputs do on the way out.

## References

- [JEV] TypeSafe AI, "Introducing System One Models & Jev", 2026. https://typesafe.ai/blog/introducing-system-one-models-and-jev
- [JEV-STATUS] LLM Reference, TypeSafe AI provider page (no published weights, size or docs at launch). https://www.llmreference.com/provider/typesafe-ai
- [RLCD] S. Dubois, "Reinforcement Learning for Calibrated Decisions (RLCD)". https://www.dsebastien.net/reinforcement-learning-for-calibrated-decisions-rlcd/
- [JUL] JuL on PyPI. https://pypi.org/project/jul/
- [ANYJEV] "AnyJev: Turn Any LLM Into A Calibrated Decision Model". https://pyshine.com/AnyJev-Turn-Any-LLM-Into-A-Calibrated-Decision-Model/
- [LAYA] "Laya, the 33 ms open-source decision model" (author's write-up). https://mer.vin/news/laya-the-33ms-open-source-decision-model-beating-jev/
- [JEVK5] JevK5-Lite (preview). https://huggingface.co/alibiserikbay/JevK5-Lite
- [LLaRP] Szot et al., "Large Language Models as Generalizable Policies for Embodied Tasks", ICLR 2024. https://arxiv.org/abs/2310.17722 ; code https://github.com/apple/ml-llarp
- [RLCR] Damani et al., "Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty", ICLR 2026. https://arxiv.org/abs/2507.16806
- [RLCR-HACK] "On the effectiveness of reward functions in reinforcement learning for confidence calibration of large language models". https://arxiv.org/abs/2607.04332
- [MTP] Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction", ICML 2024. https://arxiv.org/abs/2404.19737
- [NAR] Gu et al., "Non-Autoregressive Neural Machine Translation", 2017. https://arxiv.org/abs/1711.02281
- [ModernBERT] Warner et al., "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder", 2024. https://arxiv.org/abs/2412.13663
- [CPU-DISTILL] "A Reproducible, License-Aware Distillation Recipe for CPU-Deployable Safety Classification", 2026. https://arxiv.org/abs/2608.21570
- [DUAL] "Slow Token, Fast Action" (GR00T N1 and Helix dual-system commentary; secondary). https://atomsfrontier.substack.com/p/slow-token-fast-action
- [JEPA-LIB] Meta FAIR, "A Lightweight Library for Energy-Based Joint-Embedding Predictive Architectures", ICLR 2026 (for the belief side, later). https://arxiv.org/abs/2602.03604
