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
