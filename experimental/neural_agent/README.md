# experimental/neural_agent

ADR-001's three-level agent, as code. Last updated **2026-10-16**.

| | |
|---|---|
| Decision record | [`agents/design/ADR-001-three-level-agent.md`](../../agents/design/ADR-001-three-level-agent.md). Local-only: this clone excludes it through `.git/info/exclude`, and it is not in the repository. Accepted by the project owner on 2026-10-16. sha256 `cc7747750f4063d62877e85508b5409024438def121abd839374176166db2219` (modified 20:59:34). The earlier pin `363a7196…` is stale. |
| Shared interface | [`contracts.py`](contracts.py), sha256 `b0fc994d9f55506ffee29507a5bdfe8b06036b52c96428198679a729bb90dd13` (modified 20:00:18). The only module that other modules in this package may import from. |
| System One decision model | [`docs/SYSTEM_ONE.md`](docs/SYSTEM_ONE.md): what JEV is (TypeSafe's System One model), the open clones and precedents, and the design, formulas (FORMULAS M1-M8) and gates G1-G5 of J-S1, our own System One model for goal reading and decision making. Owner (2026-10-17): nothing external except the top LLM. |
| Status (2026-10-16) | Every module with a gate passes it. `probes/` and `jev/kev.py` have no gate. Across `tests/experimental/neural_agent/`: 372 passed, 2 skipped (one run, CPU, no network, exit 0). The table under "Modules and gates" gives the count for each module. No module here is measured on decision quality. See Limits. |

## Thesis

ADR-001 section 1 (status **COMMITTED**):

> The system is a hierarchical agent that operates continuously in a simulated
> world while allocating its reasoning time and compute intelligently. The
> research question is how an intelligent agent uses time and compute: when to
> think, when to act, and when to reuse an existing decision. Task completion is
> the instrument, not the question.

Working claims that this package has to respect:

- **Three levels** (ADR section 2). An LLM executive (Level 3) is asynchronous
  and event-driven. A JEV decision model (Level 2) runs at about 2 Hz plus
  interrupts under a hard latency budget. Skill policies (Level 1) run at every
  decision step and never wait for a JEV search or an LLM call.
- **Two state objects cross the levels.** B_t is the shared belief, built from
  legal observation only. E_t is the executive state, written by Level 3 and
  read by the JEV. B_t is not any level's private latent.
- **Non-negotiables** (ADR section 2, COMMITTED). Level 1 never blocks. The JEV
  answers within its latency budget or not at all. Legal observation only, from
  the start. Level 3 receives an abstraction of B_t and memory, never a simulator
  dump. Skills are frozen and version-pinned during any JEV or executive
  experiment. Hytale is right by definition where JAX and Hytale disagree.
- **Evidence rule** (ADR section 0). A structural implementation is not evidence
  that the hierarchy works. The tests here establish record validity and shape.
  They do not establish decision quality (ADR section 6.5).

What the decision record said was absent (ADR section 3, D2 and D3): no JEV
instance, no enforced latency budget, no E_t store or executive runtime. This
package now supplies each of those as code. The code is structural. Its
measured behaviour is not established (see Limits).

## The three levels

From ADR section 2. The right-hand column names the contract types in `contracts.py`.

| Level | Decides | Cadence | Reads | Writes | Contract types |
|---|---|---|---|---|---|
| 3 Executive | objectives, priorities, constraints, horizons, budgets; when to reason at all | event-driven | abstracted B_t, events, outcomes, its own memory | E_t; task and episodic memory | `ExecutiveState` (host), lowered by `executive_arrays` to `ExecutiveArrays` |
| 2 JEV | which skill, with which parameters, for how long; whether and how much to search | about 2 Hz plus `DecisionEvent` interrupts; hard latency budget | B_t, E_t, decision context, goal runtime | `ManagedGoal` | `SharedBeliefState`, `ExecutiveArrays`, `GoalPayload` via `make_goal`, `JevInstance` |
| 1 Skills | the action factors for this step | every decision step (10 to 20 Hz) | B_t, goal payload, legal observation, action mask | int32 factors `[B, 12]`; skill-local state | `Skill`, `SkillContract`, `SkillOutcome` |

Vocabulary (ADR section 2). A **JEV** is defined by its interface and latency
budget, not by its internals. A JEPA-style world model is one instance of a JEV.
A quick small language model is another. A scripted rule set is the baseline.
A JEV is not the same thing as a JEPA model, and the record never uses the two
words interchangeably. The instance ladder is J0 (scripted rules), J1 (learned
reactive network, the primary instance), J2 to J4 (world model, bounded search,
adaptive allocation), and J-LM (quick language model).

## Modules and gates

Counts are from one run per file on 2026-10-16 (CPU, no network). Each folder has
its own `SKILL.md` with the local facts, gates and gotchas.

| Path | Status | Gate (in `tests/experimental/neural_agent/`) | Result 2026-10-16 | Local context |
|---|---|---|---|---|
| `contracts.py` | present | `test_na_contracts.py` | 43 passed | [`SKILL.md`](SKILL.md) |
| `belief/` (B_t, `SharedBelief`) | present | `test_na_belief.py` | 21 passed | [`belief/SKILL.md`](belief/SKILL.md) |
| `skills/` (Level 1, dispatcher) | present | `test_na_skills.py` | 48 passed | [`skills/SKILL.md`](skills/SKILL.md) |
| `jev/` J0 scripted rule, runner | present | `test_na_jev_core.py` | 19 passed | [`jev/SKILL.md`](jev/SKILL.md) |
| `jev/network.py` (J1 network) | present; no trained result | `test_na_jev_network.py` | 24 passed | [`jev/SKILL.md`](jev/SKILL.md) |
| `jev/train_j1_ppo.py` (J1 trainer) | present; trains a stand-in MLP on a synthetic environment | `test_na_jev_ppo.py` | 6 passed | [`jev/SKILL.md`](jev/SKILL.md) |
| `jev/lm/` (J-LM) | present; live Ollama check gated | `test_na_jev_lm.py` | 65 passed, 1 skipped | [`jev/lm/SKILL.md`](jev/lm/SKILL.md) |
| `jev/kev.py` | reserved stub; raises `NotImplementedError` | none | not applicable | [`jev/SKILL.md`](jev/SKILL.md) |
| `executive/` (E_t store, reports, scheduler, baselines, harness, API adapter) | present | `test_na_executive.py` | 59 passed | [`executive/SKILL.md`](executive/SKILL.md) |
| `tasks/` (composition task, flat arena) | present; damage case only | `test_na_tasks.py` | 30 passed | [`tasks/SKILL.md`](tasks/SKILL.md) |
| `eval/` (arms, metrics, `run_composition` CLI) | present; all five arms wired (`j1` runs untrained until `--j1-params`; `j_lm` needs `--runner eager`) | `test_na_eval.py` | 36 passed, 1 skipped | [`eval/SKILL.md`](eval/SKILL.md) |
| `eval/eager_loop.py` | present | `test_na_eager_loop.py` | 3 passed | [`eval/SKILL.md`](eval/SKILL.md) |
| `session.py` (one continuous run) | present | `test_na_session.py` | 7 passed | [`SKILL.md`](SKILL.md) |
| `probes/` (measurement scripts) | present; not a build target | none | not applicable | [`probes/README.md`](probes/README.md); no `SKILL.md` yet |
| `docs/` (design documents) | present | `test_na_docs.py`, `test_na_docs_folders.py` | `test_na_docs.py`: 9 passed; `test_na_docs_folders.py`: 2 passed | [`docs/SKILL.md`](docs/SKILL.md) |

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

## Where to start

1. [`contracts.py`](contracts.py). Each record and protocol has a docstring with its shapes.
2. [`../../tests/experimental/neural_agent/test_na_contracts.py`](../../tests/experimental/neural_agent/test_na_contracts.py). Executable form of the contract. Note that `.gitignore` excludes `tests/` (see DEV.md).
3. ADR-001 section 2 (levels and non-negotiables) and section 3 (D2 JEV, D6 belief ownership), if you have the local record.
4. [`SKILL.md`](SKILL.md) for rules and gates. [`DEV.md`](DEV.md) for layout, commands, the CLIs and the known problems. Then the `SKILL.md` of the folder you are changing.
5. [`docs/`](docs/): `THESIS.md`, `FORMULAS.md`, `REFERENCES.md` and `ARCHITECTURE.md` (the package map and the implemented-versus-claimed checklist).

## Limits

- Contract tests show that records validate and that arrays have the documented shapes. They show nothing about learned behaviour.
- The J0 rule is a scripted rule with constants the record does not set (`jev/SKILL.md`). It is not tuned, and its outcome on any task is not measured.
- The J1 network (`jev/network.py`) has no trained result. The trainer (`jev/train_j1_ppo.py`) trains a local stand-in MLP on a synthetic environment, not the network in `network.py` (its TODO is at line 796).
- The `j1` and `j_lm` arms are wired (2026-10-17): `j1` runs `NetworkJev` untrained unless `--j1-params` supplies a checkpoint; `j_lm` runs as a host decider through `session.run_session` and requires `--runner eager`.
- The live checks were not run in the documentation pass: the Ollama test needs `NEURAL_AGENT_LIVE_OLLAMA=1`, and the Anthropic smoke makes one paid call when `ANTHROPIC_API_KEY` is set (DEV.md).
- No Hytale or simulator result is claimed here. ADR section 6.5 lists what the existing evidence does not establish.
