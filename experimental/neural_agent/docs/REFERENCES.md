# experimental/neural_agent: references

Status on 2026-10-16; updated 2026-10-17 (UTC; the local clock read 2026-10-16 evening, CDT). A complete bibliography for `THESIS.md`, `FORMULAS.md` and `ARCHITECTURE.md`. It has four parts. Section 1 copies the entries of ADR-001 section 11, verbatim, from the version pinned below, with the status that section records. Section 2 gives the lineage list supplied for this document. Section 3 gives the references this document adds for formulas the ADR does not cite, and [BCD2001], which `tasks/SKILL.md` cites. Section 4 lists the keys each document cites, and the entries kept as background.

**Record caveat.** The decision record is ADR-001, `agents/design/ADR-001-three-level-agent.md`. It is local-only and is not in this repository. Section 1 was copied by a script from that file at SHA-256 `73d8779ed5200f93f03b17f2e7363e71ef621a47d0ef98e48fe599b007d1d76d` (modified 2026-10-16 20:59:34). The ADR changed during this work, so a later version may differ. The copy is verbatim: each entry line is the ADR line, and each *Supports* line is this document's note.

**How each entry was checked.** The ADR's own wording, as recorded at the pinned version: "verified" means the abstract or landing page was read on 2026-10-16; "abstract verified" and "verified from the abstract page" mean the same; "verified from the full text" means the paper was read; "verified by DOI record" means only the DOI metadata was checked; "verified through secondary sources" means only other works' accounts were read; "verified through search summary only" means only a search engine's summary was read. Entries with no mark are read under the section header's rule, "all verified on 2026-10-16 unless marked".

Entries in section 3 carry their own checks. "Fetched 2026-10-16" means the fetch tool returned the abstract or an HTML rendering, and the note says what was read. "Recalled" means cited from memory and not reopened. "Resolved 2026-10-17" means the DOI was followed at doi.org to the publisher's page (HTTP 302) and the Crossref record was read; the paper itself was not.

A *Supports* line names where the documents cite the key. An entry marked **Background** is kept for the record but is not cited by key in `THESIS.md`, `FORMULAS.md` or `ARCHITECTURE.md`.

---

## 1. ADR-001 section 11, copied

**Reused from PLAN v0.2.**
- [2] Chen, Zhou, Koltun, Krähenbühl. *Learning by Cheating.* CoRL 2019. https://arxiv.org/abs/1912.12294 — verified.
  *Supports:* teacher and student split, $\pi_T(\cdot \mid P_t, G_t)$ and $\pi_S(\cdot \mid B_t, G_t)$ (FORMULAS E2). Scope caveat: driving, from pixels.
- [3] Weihs et al. *Bridging the Imitation Gap by Adaptive Insubordination (ADVISOR).* NeurIPS 2021. https://arxiv.org/abs/2007.12173 — verified from the full text, including the weight: an auxiliary actor head sharing the student's representation is trained by imitation only, and the per-sample weight is exp(−α·KL(teacher ‖ auxiliary)) with a small-probability cutoff in the released code.
  *Supports:* ADVISOR weight and loss (FORMULAS F2 to F5); hypothesis H7 (THESIS section 5). Checked 2026-10-16 from an arXiv HTML rendering through a summarizing fetch, not the PDF; the released-code cutoff is not in the fetched paper text.
- [11] Zaheer et al. *Deep Sets.* NeurIPS 2017. https://arxiv.org/abs/1703.06114 — verified.
  *Supports:* permutation-invariant token encoder for the shared encoder (THESIS section 6, the P0 row; ADR D1).
- [12] Lee et al. *Set Transformer.* ICML 2019. https://arxiv.org/abs/1810.00825 — verified.
  *Supports:* as [11] (THESIS section 6, the P0 row).
- [16] Nachum et al. *Data-Efficient Hierarchical Reinforcement Learning (HIRO).* NeurIPS 2018. https://arxiv.org/abs/1805.08296 — verified from the full text, including the ablation in which pre-training then freezing the lower level was detrimental.
  *Supports:* manager–worker non-stationarity and the freeze risk (ADR section 10; PLAN §19.4; THESIS section 2, non-negotiable 5).
- [19] Janner, Fu, Zhang, Levine. *When to Trust Your Model (MBPO).* NeurIPS 2019. https://arxiv.org/abs/1906.08253 — verified.
  *Supports:* short branched rollouts for the dream score of H1 (FORMULAS H2).
- [20] Yu et al. *MOPO.* NeurIPS 2020. https://arxiv.org/abs/2005.13239 — verified.
  *Supports:* uncertainty penalty in the dream score of H1 (FORMULAS H2).
- [21] Baisero, Amato. *Unbiased Asymmetric Reinforcement Learning under Partial Observability.* AAMAS 2022. https://arxiv.org/abs/2105.11674 — verified.
  *Supports:* history-state critic $V(h, s)$ (FORMULAS G2). Checked 2026-10-16 against the abstract only; the abstract does not state the critic's conditioning.
- [24] Ross, Gordon, Bagnell. *DAgger.* AISTATS 2011. https://proceedings.mlr.press/v15/ross11a — verified.
  *Supports:* on-policy correction; the executed-action DAgger result (THESIS section 9).

**Reused from Dawn's research notes** (`agents/dawn/docs/research/RESEARCH.md`; all verified on 2026-10-16 unless marked).
- [N1] Ni, Eysenbach, Salakhutdinov. *Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDPs.* ICML 2022. https://arxiv.org/abs/2110.05038
  *Supports:* a GRU before a Transformer for belief memory (FORMULAS B2).
- [N2] Morad et al. *POPGym.* ICLR 2023. https://arxiv.org/abs/2303.01859
  *Supports:* as [N1] (FORMULAS B2).
- [N3] Lu et al. *Rethinking Transformers in Solving POMDPs.* ICML 2024. https://arxiv.org/abs/2405.17358
  *Supports:* as [N1] (FORMULAS B2).
- [A1] Zhu, Li, Poupart, Miao. *On Improving Deep RL for POMDPs (ADRQN).* arXiv preprint 2017, no venue found. https://arxiv.org/abs/1704.07978
  *Supports:* previous action in the recurrence (FORMULAS B2; ADR D6).
- [A2] Che et al. *GRU-D.* Scientific Reports 2018.
  *Supports:* elapsed time in the recurrence (FORMULAS B2; ADR D6). The ADR gives no URL or DOI for this entry. DOI added by the second pass, 2026-10-16: https://doi.org/10.1038/s41598-018-24271-9 (Crossref record checked: Che, Purushotham, Cho, Sontag, *Recurrent Neural Networks for Multivariate Time Series with Missing Values*, Scientific Reports 8, 2018).
- [B1] Gangwani et al. *Learning Belief Representations for Imitation Learning in POMDPs.* UAI 2019 (PMLR v115 is dated 2020). https://arxiv.org/abs/1906.09510
  *Supports:* task-aware, distributional auxiliary targets for hidden-state prediction (FORMULAS E3).
- [B2] Lambrechts, Bolland, Ernst. *Recurrent networks, hidden states and beliefs in partially observable environments.* TMLR 2022. https://arxiv.org/abs/2208.03520
  *Supports:* a recurrent state is not automatically a calibrated belief (FORMULAS E3).
- [W1] Warrington et al. *Robust Asymmetric Learning in POMDPs.* ICML 2021. https://arxiv.org/abs/2012.15566
  *Supports:* asymmetric actor–critic learning (FORMULAS G1).
- [K1] Huang, Ontañón. *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms.* arXiv 2020; FLAIRS 2022. https://arxiv.org/abs/2006.14171
  *Supports:* masking inside the policy law (FORMULAS C4).
- [O2] Bacon, Harb, Precup. *The Option-Critic Architecture.* AAAI 2017. https://arxiv.org/abs/1609.05140
  *Supports:* learned termination of skills as options (FORMULAS E8).
- [V2] Farahmand, Barreto, Nikovski. *Value-Aware Loss Function for Model-based RL.* AISTATS 2017. https://proceedings.mlr.press/v54/farahmand17a
  *Supports:* value-aware model learning: a model is judged by the Bellman updates it supports (FORMULAS I3).

**Lineage, verified 2026-10-16 by web search** (venue and identifier checked; each entry names its level per the legend above).
- [L1] Gat. *On Three-Layer Architectures.* In *Artificial Intelligence and Mobile Robots*, AAAI Press 1998, pp. 195–210. http://www.flownet.com/gat/papers/tla.pdf — chapter and year verified; the controller/sequencer/deliberator labels are attributed to it by secondary sources, read before quoting.
  *Supports:* the three-layer lineage (THESIS section 2, lineage paragraph). Read the paper before quoting its labels.
- [L2] Bonasso, Firby, Gat, Kortenkamp, Miller, Slack. *Experiences with an Architecture for Intelligent, Reactive Agents (3T).* JETAI 9(2–3):237–256, 1997. https://doi.org/10.1080/095281397147103 — verified.
  *Supports:* the 3T tiers (THESIS section 2, lineage paragraph).
- [L3] Firby. *An Investigation into Reactive Planning in Complex Domains.* AAAI-87, pp. 202–206. https://mlanthology.org/aaai/1987/firby1987aaai-investigation — verified.
  *Supports:* reactive action packages, the lineage of skills (THESIS section 2, lineage paragraph).
- [L4] Brooks. *A Robust Layered Control System for a Mobile Robot.* IEEE J. Robotics and Automation RA-2(1):14–23, 1986. https://doi.org/10.1109/JRA.1986.1087032 — verified.
  *Supports:* subsumption; Level 1 never blocks (THESIS section 2, lineage paragraph, which ties it to non-negotiable 1).
- [L5] Hayes-Roth. *A Blackboard Architecture for Control.* Artificial Intelligence 26(3):251–321, 1985. https://doi.org/10.1016/0004-3702(85)90063-3 — verified.
  *Supports:* control reasoning kept apart from domain problem solving (THESIS section 2, lineage paragraph; B_t is not claimed to be a blackboard).
- [L6] Anthony, Tian, Barber. *Thinking Fast and Slow with Deep Learning and Tree Search.* NeurIPS 2017. https://arxiv.org/abs/1705.08439 — verified.
  *Supports:* a fast network paired with slow search (THESIS section 2, lineage paragraph).
- [L7] Christakopoulou, Mourad, Matarić. *Agents Thinking Fast and Slow: A Talker-Reasoner Architecture.* NeurIPS 2024 Open-World Agents workshop. https://arxiv.org/abs/2410.08328 — verified.
  *Supports:* a fast agent paired with a slow planner (THESIS section 2, lineage paragraph).
- [L8] NVIDIA. *GR00T N1.* arXiv 2503.14734, 2025. https://arxiv.org/abs/2503.14734 — verified, including the 10 Hz / 120 Hz figures.
  *Supports:* the fast/slow precedent with measured latency, on robots (THESIS section 2, lineage paragraph).
- [L9] Li et al. *Optimus-2.* CVPR 2025. https://arxiv.org/abs/2502.19902 — verified; no latency reported.
  *Supports:* a language-model planner over skills, with no latency reported (THESIS section 2, lineage paragraph).
- [L10] Li, Liu, Zheng, Sun, Song. *Parallelized Planning-Acting for Efficient LLM-based Multi-Agent Systems.* arXiv 2503.03505, 2025, revised 2026. https://arxiv.org/abs/2503.03505 — verified; no timing reported.
  *Supports:* concurrent, interruptible planning and acting (THESIS section 2, lineage paragraph).
- [L11] Horvitz. *Reasoning about Beliefs and Actions under Computational Resource Constraints.* UAI 1987. https://arxiv.org/abs/1304.2759 — verified.
  *Supports:* resource-bounded reasoning; the basis of the value-of-reasoning trigger (THESIS section 2, lineage paragraph; the topic of FORMULAS J6).
- [L12] Zilberstein. *Using Anytime Algorithms in Intelligent Systems.* AI Magazine 17(3), 1996. https://doi.org/10.1609/aimag.v17i3.1232 — verified by DOI record.
  *Supports:* anytime computation and outcome per unit of compute (THESIS section 2, lineage paragraph).
- [L13] Labiosa, Hanna. *When to Plan: Learning to Select Between Reactive Control and Deliberative Planning.* RLC 2026. https://arxiv.org/abs/2607.16421 — verified from the abstract page.
  *Supports:* an RL-trained selector between reactive and planning control (THESIS section 2, lineage paragraph).
- [L14] Yang, Hou, Wei, Bao, Chang. *Ares: Adaptive Reasoning Effort Selection for Efficient LLM Agents.* arXiv 2603.07915, 2026. https://arxiv.org/abs/2603.07915 — verified from the abstract page.
  *Supports:* per-step reasoning-effort routing (THESIS section 2, lineage paragraph).

**New in this record.**
- [O1] Sutton, Precup, Singh. *Between MDPs and semi-MDPs: a framework for temporal abstraction.* Artificial Intelligence 112(1–2):181–211, 1999 — verified.
  *Supports:* skills as temporally extended, terminating options (FORMULAS E8). The ADR gives no URL or DOI for this entry. DOI added by the second pass, 2026-10-16: https://doi.org/10.1016/S0004-3702(99)00052-1 (Crossref record checked: title, authors, *Artificial Intelligence* 112(1–2):181–211, 1999). Resolved 2026-10-17: the DOI redirects to Elsevier PII S0004370299000521, and the Crossref record, re-read, matches the title, the three authors, volume 112(1–2) and pages 181–211.
- [MuZero] Schrittwieser et al. *Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero).* Nature 588, 2020. https://arxiv.org/abs/1911.08265 — verified.
  *Supports:* latent search trained on values, rewards and policies (THESIS section 10, question 4; the J2 and J3 design).
- [SampledMZ] Hubert et al. *Learning and Planning in Complex Action Spaces (Sampled MuZero).* ICML 2021. https://arxiv.org/abs/2104.06303 — verified.
  *Supports:* search over sampled actions, relevant to skill × parameters (THESIS section 10, question 4).
- [StochMZ] Antonoglou et al. *Planning in Stochastic Environments with a Learned Model (Stochastic MuZero).* ICLR 2022. https://iclr.cc/virtual/2022/poster/6832 — verified.
  **Background.** Chance nodes for opponent responses, relevant to the J3 scope (ADR section 3 D2).
- [V1] Grimm, Barreto, Singh, Silver. *The Value Equivalence Principle for Model-Based RL.* NeurIPS 2020. https://arxiv.org/abs/2011.03506 — verified.
  *Supports:* the value-equivalence definition (FORMULAS I1 to I3). Checked 2026-10-16 from the abstract; the HTML version returned 404.
- [TDMPC2] Hansen, Su, Wang. *TD-MPC2: Scalable, Robust World Models for Continuous Control.* ICLR 2024. https://arxiv.org/abs/2310.16828 — verified from the full text, including the planner: MPPI over the latent model with a fraction of samples from the learned policy prior.
  **Background.** Decoder-free latent planning with MPPI, the closest published J2 reference (ADR section 3 D2).
- [Director] Hafner, Lee, Fischer, Abbeel. *Deep Hierarchical Planning from Pixels (Director).* NeurIPS 2022. https://arxiv.org/abs/2206.04114 — verified.
  **Background.** A manager proposing goals in a latent space (ADR section 3 D2; PLAN §19.4).
- [HJEPA] Zhang, Terver, Rabbat, LeCun, Balestriero. *H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning.* arXiv 2610.06805, 2026-10-05. https://arxiv.org/abs/2610.06805 — abstract verified.
  **Background.** Multi-timescale latent planning, the subject of H6, which THESIS section 5 states without citing it (ADR section 3 D2). Pixels and robots, not a game.
- [DJEPA] Liu et al. *D-JEPA: A Decision-Aligned Latent World Model.* arXiv 2609.24749, 2026-09-21. https://arxiv.org/abs/2609.24749 — abstract verified.
  *Supports:* decision-aligned latent distances (FORMULAS I3).
- [R1] Russell, Wefald. *Principles of Metareasoning.* Artificial Intelligence 49, 1991 — verified through secondary sources.
  *Supports:* the value-of-computation formula (FORMULAS J6). Secondary sources only. DOI record added by the second pass, 2026-10-16: https://doi.org/10.1016/0004-3702(91)90015-C (Crossref record checked: *Artificial Intelligence* 49(1–3):361–395, 1991). The paper was not read.
- [R2] *Rational Metareasoning for Large Language Models.* arXiv 2410.05563, 2024. https://arxiv.org/abs/2410.05563 — verified through search summary only.
  *Supports:* value of reasoning for language models (FORMULAS J6; the topic of H9). Search summary only.
- [G1] Park et al. *Generative Agents: Interactive Simulacra of Human Behavior.* UIST 2023. https://arxiv.org/abs/2304.03442 — verified.
  *Supports:* the persistent memory stream, the precedent for Level 3 memory (ARCHITECTURE section 4).
- [S1] Ahn et al. (Ichter et al. in the proceedings). *Do As I Can, Not As I Say (SayCan).* CoRL 2022. https://arxiv.org/abs/2204.01691 — verified.
  *Supports:* a language-model planner grounded by skill value functions as affordances (THESIS section 9, novelty claim; PLAN §25.3).
- [Plan4MC] Yuan et al. *Plan4MC: Skill Reinforcement Learning and Planning for Open-World Minecraft Tasks.* 2023. https://arxiv.org/abs/2303.16563 — verified.
  *Supports:* an offline skill graph, rejected as the final executive form (THESIS section 9, novelty claim; ADR D3).
- [DECKARD] Nottingham et al. *Do Embodied Agents Dream of Pixelated Sheep (DECKARD).* ICML 2023. https://arxiv.org/abs/2301.12050 — verified.
  *Supports:* language models over RL skills, synchronous (THESIS section 9, novelty claim; ADR D3).
- [Voyager] Wang et al. *Voyager: An Open-Ended Embodied Agent with Large Language Models.* 2023. https://arxiv.org/abs/2305.16291 — verified.
  *Supports:* language models over code skills, synchronous (THESIS section 9, novelty claim; ADR D3).

---

## 2. Lineage list supplied for this document

The supplied list was verified on 2026-10-16 by web search, as the supplier states. Its venues and identifiers (arXiv numbers, DOIs and URLs) are reproduced as supplied. Section 1 gives the URL for each of these keys. Most of its entries are already in ADR section 11 as [L1] to [L14]. The "Same entry as" column gives that key. Only [L7] differs in spelling: the supplied list reads "Mataric", and the ADR reads "Matarić". Section 1 keeps the ADR spelling.

| Supplied entry | Same entry as | Supports in this document |
|---|---|---|
| Gat, On Three-Layer Architectures, in Artificial Intelligence and Mobile Robots, AAAI Press 1998, pp. 195-210, http://www.flownet.com/gat/papers/tla.pdf (controller/sequencer/deliberator labels attributed to Gat by secondary sources; read before quoting) | [L1] | THESIS section 2, lineage paragraph |
| Bonasso, Firby, Gat, Kortenkamp, Miller, Slack, Experiences with an Architecture for Intelligent, Reactive Agents, JETAI 9(2-3):237-256, 1997, doi 10.1080/095281397147103 (the 3T architecture) | [L2] | THESIS section 2, lineage paragraph |
| Firby, An Investigation into Reactive Planning in Complex Domains, AAAI-87 pp. 202-206, https://mlanthology.org/aaai/1987/firby1987aaai-investigation (reactive action packages) | [L3] | THESIS section 2, lineage paragraph |
| Brooks, A Robust Layered Control System for a Mobile Robot, IEEE J. Robotics and Automation RA-2(1):14-23, 1986, doi 10.1109/JRA.1986.1087032 (subsumption) | [L4] | THESIS section 2, lineage paragraph |
| Hayes-Roth, A Blackboard Architecture for Control, Artificial Intelligence 26(3):251-321, 1985, doi 10.1016/0004-3702(85)90063-3 | [L5] | THESIS section 2, lineage paragraph |
| Anthony, Tian, Barber, Thinking Fast and Slow with Deep Learning and Tree Search, NeurIPS 2017, arXiv 1705.08439 (expert iteration) | [L6] | THESIS section 2, lineage paragraph |
| Christakopoulou, Mourad, Mataric, Agents Thinking Fast and Slow: A Talker-Reasoner Architecture, NeurIPS 2024 OWA workshop, arXiv 2410.08328 | [L7] | THESIS section 2, lineage paragraph |
| NVIDIA GR00T N1, arXiv 2503.14734, 2025 (System 2 VLM at 10 Hz over a System 1 diffusion policy at 120 Hz; the one 2025-2026 fast/slow paper with measured latency, robots not games) | [L8] | THESIS section 2, lineage paragraph |
| Optimus-2, arXiv 2502.19902, CVPR 2025 (Minecraft, MLLM planner over a goal-conditioned policy, no latency reported) | [L9] | THESIS section 2, lineage paragraph |
| Li et al., Parallelized Planning-Acting for Efficient LLM-based Multi-Agent Systems, arXiv 2503.03505 (concurrent planning and acting threads with interruptible execution, Minecraft, no timing) | [L10] | THESIS section 2, lineage paragraph |
| Horvitz, Reasoning about Beliefs and Actions under Computational Resource Constraints, UAI 1987, arXiv 1304.2759 | [L11] | THESIS section 2, lineage paragraph |
| Zilberstein, Using Anytime Algorithms in Intelligent Systems, AI Magazine 17(3), 1996, doi 10.1609/aimag.v17i3.1232 | [L12] | THESIS section 2, lineage paragraph |
| Labiosa and Hanna, When to Plan: Learning to Select Between Reactive Control and Deliberative Planning, arXiv 2607.16421, RLC 2026 (an RL-trained selector between a fast reactive policy and slower planning) | [L13] | THESIS section 2, lineage paragraph |
| Yang et al., Ares: Adaptive Reasoning Effort Selection for Efficient LLM Agents, arXiv 2603.07915, 2026 (a learned router picks per-step reasoning effort) | [L14] | THESIS section 2, lineage paragraph |

---

## 3. Added by this document

These entries cover formulas that ADR section 11 does not cite. Each one says what was checked.

- **[PPO]** Schulman, Wolski, Dhariwal, Radford, Klimov. *Proximal Policy Optimization Algorithms.* arXiv:1707.06347 [cs.LG], 2017. https://arxiv.org/abs/1707.06347 — *fetched 2026-10-16:* authors and identifier read from the abstract page. The equations were not retrieved; the HTML rendering returned none. The operative form is read from `HytaleRL/hytalegym/hytalegym/jax/training/ppo.py`.
  *Supports:* the clipped surrogate and the value and entropy coefficients (FORMULAS E9).
- **[GAE]** Schulman, Moritz, Levine, Jordan, Abbeel. *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* arXiv:1506.02438, 2015; ICLR 2016 (venue from the DBLP key `conf/iclr/iclr2016`, as the fetch reported; the proceedings page was not opened). https://arxiv.org/abs/1506.02438 — *fetched 2026-10-16:* authors and identifier read from the abstract page. The title is as recalled, because the fetch did not print it. The equations were not retrieved; the recursion is read from `ppo.py`.
  *Supports:* the advantage estimator (FORMULAS E10).
- **[ECE]** Guo, Pleiss, Sun, Weinberger. *On Calibration of Modern Neural Networks.* ICML 2017; arXiv:1706.04599. https://arxiv.org/abs/1706.04599 — *fetched 2026-10-16:* authors and venue read from the abstract page. Eq. (3), the ECE definition, was read from the arXiv HTML v2. The title is as recalled.
  *Supports:* expected calibration error for H3 (FORMULAS J4).
- **[BRIER]** G. W. Brier. Verification of forecasts expressed in terms of probability. *Monthly Weather Review* 78(1):1–3, 1950. https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2 — DOI record checked on Crossref, 2026-10-16 (metadata only).
  *Supports:* the Brier score for H3 (FORMULAS J3).
- **[WILSON]** E. B. Wilson. Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association* 22(158):209–212, 1927. https://doi.org/10.1080/01621459.1927.10502953 — DOI record checked on Crossref, 2026-10-16 (metadata only). `recipe.py` cites no source for its bound.
  *Supports:* the Wilson lower bound and the advance rule (FORMULAS J1, J2).
- **[KL]** S. Kullback, R. A. Leibler. On information and sufficiency. *Annals of Mathematical Statistics* 22(1):79–86, 1951. https://doi.org/10.1214/aoms/1177729694 — DOI record checked on Crossref, 2026-10-16 (metadata only).
  *Supports:* the KL divergence in FORMULAS E2, F1, F2 and J5.
- **[GRU]** K. Cho et al. Learning phrase representations using RNN encoder–decoder for statistical machine translation. EMNLP 2014. https://doi.org/10.3115/v1/D14-1179 and https://arxiv.org/abs/1406.1078 — DOI record and arXiv abstract page checked 2026-10-16 (title and authors; the paper was not read).
  *Supports:* the GRU cell in FORMULAS B2.
- **[HUBER]** P. J. Huber. Robust estimation of a location parameter. *Annals of Mathematical Statistics* 35(1):73–101, 1964. https://doi.org/10.1214/aoms/1177703732 — DOI record checked on Crossref, 2026-10-16 (metadata only).
  *Supports:* the Huber loss in FORMULAS E4.
- **[BCD2001]** L. D. Brown, T. T. Cai, A. DasGupta. Interval estimation for a binomial proportion. *Statistical Science* 16(2):101–133, 2001. https://doi.org/10.1214/ss/1009213286 — resolved 2026-10-17: the DOI redirects to Project Euclid (*Statistical Science* 16(2), "Interval Estimation for a Binomial Proportion"), and the Crossref record gives the title, the authors Lawrence D. Brown, T. Tony Cai and Anirban DasGupta, the pages 101–133 and the year 2001. The paper was not read.
  **Background** for these documents. `tasks/SKILL.md` F8 cites it for the caveat that the Wald interval's coverage is erratic, most of all for small samples. THESIS, FORMULAS and ARCHITECTURE report the Wilson bound [WILSON] instead.

---

## 4. Keys used, by document

Generated on 2026-10-17 (UTC) from the three documents with the citation pattern of `tests/experimental/neural_agent/test_na_docs.py`, which checks that every key cited in them and in `docs/SKILL.md` has an entry in section 1 or 3.

- `FORMULAS.md` (29 keys): [2], [3], [19], [20], [21], [N1], [N2], [N3], [A1], [A2], [B1], [B2], [W1], [K1], [O2], [V2], [O1], [V1], [DJEPA], [R1], [R2], [PPO], [GAE], [ECE], [BRIER], [WILSON], [KL], [GRU], [HUBER].
- `THESIS.md` (25 keys): [3], [11], [12], [16], [24], [L1], [L2], [L3], [L4], [L5], [L6], [L7], [L8], [L9], [L10], [L11], [L12], [L13], [L14], [MuZero], [SampledMZ], [S1], [Plan4MC], [DECKARD], [Voyager].
- `ARCHITECTURE.md` (3 keys): [L1], [L14], [G1]. [L1] and [L14] appear in section 7, item 19, which names the lineage range.
- **Background** (5 entries, not cited by key in these documents): [StochMZ], [TDMPC2], [Director], [HJEPA], [BCD2001]. Each is marked in its entry above.

Section 1 and section 3 hold 59 entries, and each one is either cited above or marked background.
