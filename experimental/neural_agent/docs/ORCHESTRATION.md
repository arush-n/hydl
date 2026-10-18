# experimental/neural_agent: orchestration formulas and citations (moved from README.md, 2026-10-18)

These sections were the top-level README's "Formulas" and "Citations" (written 2026-10-16) and are kept
verbatim here so the README can be the package map. Formulas 2 to 6 also appear in `FORMULAS.md` section D;
1 (lowering E_t), 7 (the JEV latency budget) and 10 (auxiliary-policy weighting, a hypothesis) appear only here.
`[confirmed]` means read in the cited source; `[inferred]` means derived here, with the reason given.

## Formulas

These are the rules the contract and the ADK orchestration apply. `[confirmed]`
means read in the cited source. `[inferred]` means derived here, with the reason
given. The belief module's formulas are in [`belief/SKILL.md`](belief/SKILL.md).

### 1. Lowering E_t (`contracts.executive_arrays`) [confirmed]

For lane `b` in `[0, B)` and goal slot `g` in `[0, 4)`:

```text
goal_valid[b, g]      = [g < |goals|]
skill_index[b, g]     = position of goal_g.skill_id in the catalog, or -1 if empty
priority[b, g]        = goal_g.priority, or 0 if empty
params[b, g, :]       = zero-padded goal_g.params (width 4), or 0 if empty
budget[b, :]          = [jev_latency_ms, search_expansions, executive_calls_per_minute]
version[b]            = E_t.version
```

An unknown skill id raises `KeyError`. A catalog whose first entry is not `"idle"` raises `ValueError` (`contracts.py:655-657`).

### 2. Manager decision due (`adk/architecture/decision_context.py:415`) [confirmed]

```text
due_b = [tau_b >= T]  or  g_b  or  OR_e ( a_{b,e} and k_{b,e} )
```

`tau_b` is the time since the last manager decision, after adding `Delta`. `T` is
`manager_cadence_seconds`. `g_b` is true when a termination is pending, the goal
has timed out, or no goal is valid. `a_{b,e}` is the value of DecisionEvent `e`,
and `k_{b,e}` is its known bit. Only an event that is both active and known can
trigger a decision.

### 3. Elapsed time (`adk/architecture/hierarchy.py:244`) [confirmed]

```text
Delta_b = max(delta_b, 0)  if delta_b is known, else 0
age_goal <- age_goal + Delta ;   tau <- tau + Delta
```

Unknown time never advances the clocks.

### 4. Commitment, timeout and emergency (`hierarchy.py:273-282`) [confirmed]

```text
commit_b    = [age_goal >= m_min]
timeout_b   = valid and [age_goal >= m_max]              (m_max may be +inf)
emergency_b = valid and OR_e ( a_{b,e} and k_{b,e} and M_{b,e} )
```

`M` is `emergency_interrupt_mask` of the goal currently in force.

### 5. Goal switch (`hierarchy.py:347-356`) [confirmed]

```text
allowed   = not valid  or  commit  or  timeout  or  emergency
requested = not valid  or  timeout  or  pending_termination  or  (due and term_proposed)
switch    = due and allowed and requested
```

Reading: an ordinary replacement request waits for the minimum commitment. A
matching emergency event bypasses that minimum. The maximum duration always
forces a replacement.

### 6. Action age (`hierarchy.py:399`) [confirmed]

```text
action_age' = action_age + Delta   if action_valid and same_factors and not lifecycle_ended
            = 0                    otherwise
```

### 7. JEV latency budget (ADR section 2 and D2; `JevBudget`, `JevStats`) [inferred]

For instance `i` with budget `L_i` and measured latency `l_{i,t}` in milliseconds,
decision `t`:

```text
miss_{i,t} = [ l_{i,t} > L_i ]
if miss_{i,t}:  goal_t = goal_{t-1}      (the previous goal stays in force)
misses_i = sum_t miss_{i,t}              decisions_i = T
latency_ms_p50, latency_ms_p95 = percentiles of l_{i,.}
```

ADR section 2 says that a missed budget "leaves the previous goal in force and
the miss is counted", but it does not say how equality is treated. The strict
`>` above is this package's reading. Confirm it when the instance runtime is written.
`jev/runner.py` applies the rule to HOST-mode calls only.

### 8. Belief features [defined in `belief/SKILL.md`; the normalization is the belief module's decision, 2026-10-16]

The contract fixes the 10 feature names, their order, and `TICKS_SATURATION =
256.0`. The belief module defines each feature's normalization: the distance is
24 times the encoder value, the two speeds are planar and in blocks per tick, the
closing speed is the planar radial rate, and the two counters are carried ticks
divided by 256. Each formula, the source of each column, and the rule that closing
is unknown when an encoder input is clipped are in `belief/SKILL.md` (S.8, S.9, and
the counters).

### 9. Standard metrics named in ADR sections 8 and 9 [standard definitions]

The ADR names these metrics without restating them. The forms below are the
standard definitions.

```text
Brier score:                BS  = (1/N) * sum_i (p_i - o_i)^2,        o_i in {0, 1}
Expected calibration error: ECE = sum_b (|S_b| / N) * | acc(S_b) - conf(S_b) |
                            over M equal-width confidence bins S_b
Value of computation:       VOC(c) = E_c[ max_a Q(s, a) | c ] - max_a Q(s, a) - cost(c)
```

VOC is the expected improvement in decision quality minus the cost of the
computation (Russell and Wefald, [R1] below). The ADR uses it as the trigger for
"expected value of additional reasoning" (D3). "Outcome per unit of compute" is
open. The unit is ADR section 12, question 2, and it is not fixed here.

### 10. Auxiliary-policy weighting (ADR H7, citing ADVISOR [3]) [hypothesis, not implemented]

```text
w(s) = exp( -alpha * KL( pi_teacher(. | s) || pi_aux(. | s) ) )
```

The ADR states that the released code adds a small-probability cutoff. H7 says
this weighting should be preferred to PLAN section 24.3 for privileged
distillation at Levels 1 and 2. Nothing here implements it.

The full formula set, with a label on every entry, is in [`docs/FORMULAS.md`](docs/FORMULAS.md).

## Citations

Copied from ADR-001 section 11, as recorded there. The ADR marks each entry
"verified" or "recalled" as of 2026-10-16. The DOIs marked "added" were checked
by the documentation pass; the full bibliography with each check is
[`docs/REFERENCES.md`](docs/REFERENCES.md).

**Reused from PLAN v0.2.**

- [2] Chen, Zhou, Koltun, Krähenbühl. *Learning by Cheating.* CoRL 2019. https://arxiv.org/abs/1912.12294 — verified.
- [3] Weihs et al. *Bridging the Imitation Gap by Adaptive Insubordination (ADVISOR).* NeurIPS 2021. https://arxiv.org/abs/2007.12173 — verified, including the weight: an auxiliary actor head sharing the student's representation is trained by imitation only, and the per-sample weight is exp(−α·KL(teacher ‖ auxiliary)) with a small-probability cutoff in the released code.
- [11] Zaheer et al. *Deep Sets.* NeurIPS 2017. https://arxiv.org/abs/1703.06114 — verified.
- [12] Lee et al. *Set Transformer.* ICML 2019. https://arxiv.org/abs/1810.00825 — verified.
- [16] Nachum et al. *Data-Efficient Hierarchical Reinforcement Learning (HIRO).* NeurIPS 2018. https://arxiv.org/abs/1805.08296 — verified.
- [19] Janner, Fu, Zhang, Levine. *When to Trust Your Model (MBPO).* NeurIPS 2019. https://arxiv.org/abs/1906.08253 — verified.
- [20] Yu et al. *MOPO.* NeurIPS 2020. https://arxiv.org/abs/2005.13239 — verified.
- [21] Baisero, Amato. *Unbiased Asymmetric Reinforcement Learning under Partial Observability.* AAMAS 2022. https://arxiv.org/abs/2105.11674 — verified.
- [24] Ross, Gordon, Bagnell. *DAgger.* AISTATS 2011. https://proceedings.mlr.press/v15/ross11a — verified.

**Reused from Dawn's research notes** (`agents/dawn/docs/research/RESEARCH.md`).

- [N1] Ni, Eysenbach, Salakhutdinov. *Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs.* ICML 2022. https://arxiv.org/abs/2110.05038
- [N2] Morad et al. *POPGym.* ICLR 2023. https://arxiv.org/abs/2303.01859
- [N3] Lu et al. *Rethinking Transformers in Solving POMDPs.* ICML 2024. https://arxiv.org/abs/2405.17358
- [A1] Zhu, Li, Poupart, Miao. *On Improving Deep RL for POMDPs (ADRQN).* arXiv preprint 2017, no venue found. https://arxiv.org/abs/1704.07978
- [A2] Che et al. *GRU-D.* Scientific Reports 2018. The ADR gives no URL or DOI. DOI added 2026-10-16: https://doi.org/10.1038/s41598-018-24271-9 (Crossref record: Che, Purushotham, Cho, Sontag, Liu, *Recurrent Neural Networks for Multivariate Time Series with Missing Values*, Scientific Reports 8, 2018).
- [B1] Gangwani et al. *Learning Belief Representations for Imitation Learning in POMDPs.* UAI 2019 (PMLR v115 is dated 2020). https://arxiv.org/abs/1906.09510
- [B2] Lambrechts, Bolland, Ernst. *Recurrent networks, hidden states and beliefs in partially observable environments.* TMLR 2022. https://arxiv.org/abs/2208.03520
- [W1] Warrington et al. *Robust Asymmetric Learning in POMDPs.* ICML 2021. https://arxiv.org/abs/2012.15566
- [K1] Huang, Ontañón. *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms.* arXiv 2020; FLAIRS 2022. https://arxiv.org/abs/2006.14171
- [O2] Bacon, Harb, Precup. *The Option-Critic Architecture.* AAAI 2017. https://arxiv.org/abs/1609.05140
- [V2] Farahmand, Barreto, Nikovski. *Value-Aware Loss Function for Model-based RL.* AISTATS 2017. https://proceedings.mlr.press/v54/farahmand17a

**Lineage, verified 2026-10-16 by web search** (venue and identifier checked).

- [L1] Gat. *On Three-Layer Architectures.* In *Artificial Intelligence and Mobile Robots*, AAAI Press 1998, pp. 195–210. http://www.flownet.com/gat/papers/tla.pdf — chapter and year verified; the controller/sequencer/deliberator labels are attributed to it by secondary sources, read before quoting.
- [L2] Bonasso, Firby, Gat, Kortenkamp, Miller, Slack. *Experiences with an Architecture for Intelligent, Reactive Agents (3T).* JETAI 9(2–3):237–256, 1997. https://doi.org/10.1080/095281397147103 — verified.
- [L3] Firby. *An Investigation into Reactive Planning in Complex Domains.* AAAI-87, pp. 202–206. https://mlanthology.org/aaai/1987/firby1987aaai-investigation — verified.
- [L4] Brooks. *A Robust Layered Control System for a Mobile Robot.* IEEE J. Robotics and Automation RA-2(1):14–23, 1986. https://doi.org/10.1109/JRA.1986.1087032 — verified.
- [L5] Hayes-Roth. *A Blackboard Architecture for Control.* Artificial Intelligence 26(3):251–321, 1985. https://doi.org/10.1016/0004-3702(85)90063-3 — verified.
- [L6] Anthony, Tian, Barber. *Thinking Fast and Slow with Deep Learning and Tree Search.* NeurIPS 2017. https://arxiv.org/abs/1705.08439 — verified.
- [L7] Christakopoulou, Mourad, Matarić. *Agents Thinking Fast and Slow: A Talker-Reasoner Architecture.* NeurIPS 2024 Open-World Agents workshop. https://arxiv.org/abs/2410.08328 — verified.
- [L8] NVIDIA. *GR00T N1.* arXiv 2503.14734, 2025. https://arxiv.org/abs/2503.14734 — verified, including the 10 Hz / 120 Hz figures.
- [L9] Li et al. *Optimus-2.* CVPR 2025. https://arxiv.org/abs/2502.19902 — verified; no latency reported.
- [L10] Li, Liu, Zheng, Sun, Song. *Parallelized Planning-Acting for Efficient LLM-based Multi-Agent Systems.* arXiv 2503.03505, 2025, revised 2026. https://arxiv.org/abs/2503.03505 — verified; no timing reported.
- [L11] Horvitz. *Reasoning about Beliefs and Actions under Computational Resource Constraints.* UAI 1987. https://arxiv.org/abs/1304.2759 — verified.
- [L12] Zilberstein. *Using Anytime Algorithms in Intelligent Systems.* AI Magazine 17(3), 1996. https://doi.org/10.1609/aimag.v17i3.1232 — verified by DOI record.
- [L13] Labiosa, Hanna. *When to Plan: Learning to Select Between Reactive Control and Deliberative Planning.* RLC 2026. https://arxiv.org/abs/2607.16421 — verified from the abstract page.
- [L14] Yang, Hou, Wei, Bao, Chang. *Ares: Adaptive Reasoning Effort Selection for Efficient LLM Agents.* arXiv 2603.07915, 2026. https://arxiv.org/abs/2603.07915 — verified from the abstract page.

**New in ADR-001.**

- [O1] Sutton, Precup, Singh. *Between MDPs and semi-MDPs: a framework for temporal abstraction.* Artificial Intelligence 112(1–2):181–211, 1999 — verified. DOI added 2026-10-16: https://doi.org/10.1016/S0004-3702(99)00052-1 (Crossref record checked: title, the three authors, *Artificial Intelligence* 112(1–2), pages 181–211, 1999).
- [MuZero] Schrittwieser et al. *Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero).* Nature 588, 2020. https://arxiv.org/abs/1911.08265 — verified.
- [SampledMZ] Hubert et al. *Learning and Planning in Complex Action Spaces (Sampled MuZero).* ICML 2021. https://arxiv.org/abs/2104.06303 — verified.
- [StochMZ] Antonoglou et al. *Planning in Stochastic Environments with a Learned Model (Stochastic MuZero).* ICLR 2022. https://iclr.cc/virtual/2022/poster/6832 — verified.
- [V1] Grimm, Barreto, Singh, Silver. *The Value Equivalence Principle for Model-Based RL.* NeurIPS 2020. https://arxiv.org/abs/2011.03506 — verified.
- [TDMPC2] Hansen, Su, Wang. *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR 2024. https://arxiv.org/abs/2310.16828 — verified, including the planner: MPPI over the latent model with a fraction of samples from the learned policy prior.
- [Director] Hafner, Lee, Fischer, Abbeel. *Deep Hierarchical Planning from Pixels (Director).* NeurIPS 2022. https://arxiv.org/abs/2206.04114 — verified.
- [HJEPA] Zhang, Terver, Rabbat, LeCun, Balestriero. *H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning.* arXiv 2610.06805, 2026-10-05. https://arxiv.org/abs/2610.06805 — abstract verified.
- [DJEPA] Liu et al. *D-JEPA: A Decision-Aligned Latent World Model.* arXiv 2609.24749, 2026-09-21. https://arxiv.org/abs/2609.24749 — abstract verified.
- [R1] Russell, Wefald. *Principles of Metareasoning.* Artificial Intelligence 49, 1991 — verified through secondary sources. DOI added 2026-10-16: https://doi.org/10.1016/0004-3702(91)90015-C (Crossref record: *Artificial Intelligence* 49(1–3):361–395, 1991). The paper itself was not read.
- [R2] *Rational Metareasoning for Large Language Models.* arXiv 2410.05563, 2024. https://arxiv.org/abs/2410.05563 — verified through search summary only.
- [G1] Park et al. *Generative Agents: Interactive Simulacra of Human Behavior.* UIST 2023. https://arxiv.org/abs/2304.03442 — verified.
- [S1] Ahn et al. (Ichter et al. in the proceedings). *Do As I Can, Not As I Say (SayCan).* CoRL 2022. https://arxiv.org/abs/2204.01691 — verified.
- [Plan4MC] Yuan et al. *Plan4MC: Skill Reinforcement Learning and Planning for Open-World Minecraft Tasks.* 2023. https://arxiv.org/abs/2303.16563 — verified.
- [DECKARD] Nottingham et al. *Do Embodied Agents Dream of Pixelated Sheep (DECKARD).* ICML 2023. https://arxiv.org/abs/2301.12050 — verified.
- [Voyager] Wang et al. *Voyager: An Open-Ended Embodied Agent with Large Language Models.* 2023. https://arxiv.org/abs/2305.16291 — verified.

