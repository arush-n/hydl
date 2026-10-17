# Our own System One decision model for goal reading and decision making

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

**Input.** The rendering the J-LM arm already produces (`jev/lm/prompting.py`): skill catalogue with
descriptions, the executive state E_t (goals with priority, params, constraints, horizon, deadline;
facts), the current skill, the belief B_t feature lines. One rendering per lane. Later: raw belief
latent as extra tokens through a small adapter, as LLaRP feeds observations.

**Backbone.** A small pretrained transformer run in process on CPU (Metal later), frozen at first.
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
| G2 latency | one forward pass of each backbone class on the real rendering, batch 1 and 8, CPU, in process; numbers in `LEDGER.md` | a weight loader (`transformers`; owner's call) |
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

### 8.8 What is deliberately not in J-S1

No text generation, no tool use, no chain of thought at decision time (System One). No external
model or service: the only external component is the top LLM. No change to the goal contract: the
"series of quick actions" is the sequence of decisions, plus the optional plan head.

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
