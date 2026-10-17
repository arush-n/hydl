# experimental/neural_agent: formulas

Status on 2026-10-16; third pass 2026-10-17 (UTC; the local clock read 2026-10-16 evening, CDT), see the verification notes. Every formula the three-level design relies on, with one line on what it means and where it comes from. Sources are the decision record (ADR-001, local-only, SHA-256 `03f8ef2eab3406afee0120641833f4220e1d6a9dfd3df1b702bb69c58193231c`, modified 20:59:34), PLAN v0.2 (`agents/design/PLAN.md`, local-only, SHA-256 `db6413db…`), the repository code named in each entry, and the papers listed in `REFERENCES.md`.

**Labels.** Each formula carries one of four labels.

- **COMMITTED**: a decision of ADR-001, or code that the record names as implemented. Code locations are given and were read in this run unless the entry says otherwise.
- **RETAINED**: from PLAN v0.2. The ADR keeps it unless a row of ADR section 4 changes it.
- **EXTERNAL**: from a cited paper. Its key is in `REFERENCES.md`. The entry says whether the paper's equation was read, or whether the form is the standard one written here.
- **CODE**: the repository's behaviour, where neither the record nor a paper gives the formula. The entry names the file and says what was read. It is not a commitment. Used for (D8) and (D9).

**OPEN** is appended where the record leaves part of a formula undefined. It is a status on top of these labels, not a label of its own. Nothing marked OPEN is a commitment.

The notation of PLAN §4 is used where it differs from the brief. For example, PLAN §4 writes the legal history as $H_t^{\text{legal}}$.

---

## A. Observation and privileged state

**(A1)** $O_t = g_{\mathrm{legal}}\!\left(S_t,\ H_t^{\mathrm{legal}},\ c\right)$. The legal actor observation: what the agent can perceive from the authoritative state $S_t$, its legal history $H_t^{\mathrm{legal}}$ and the perception configuration $c$. *RETAINED* (PLAN §4).

**(A2)** $P_t = g_{\mathrm{priv}}(S_t)$. The privileged training scene, including the 360-degree scene and hidden objects. *RETAINED* (PLAN §4). Status on the current surface (ADR section 3 D5, 2026-10-16): the legal observation does not meet this definition for the action mask, because a hidden live enemy changes the ability legality. The deployment restriction is *COMMITTED* (ADR section 2, D5): privileged state reaches only teachers, critics, diagnostics and counterfactual evaluators.

## B. Belief

**(B1)** $B_t = f_\phi\!\left(B_{t-1},\ O_t,\ A_{t-1},\ \Delta t\right)$. The recurrent belief as PLAN §4 states it. *RETAINED*. PLAN §17.2 replaces it with (B2).

**(B2)** $h_t = \mathrm{GRU}\!\left(h_{t-1},\ x_t,\ A_{t-1},\ G_{t-1},\ \Delta t\right)$, where $x_t$ is the fused legal observation encoding. The belief inputs are the previous action, the previous goal and the elapsed time. *RETAINED* (PLAN §17.2). The recurrence over action and elapsed time is supported by [A1][A2] (ADR section 3 D6). The GRU cell is *EXTERNAL* [GRU]. A recurrent belief is tried before a Transformer one [N3], and recurrent model-free policies are strong POMDP baselines [N1][N2] (ADR section 3 D6 and D5). **OPEN**: PLAN §17.2 gives a hidden width of 384 or 512. `SharedBeliefState.latent` is `float32[B, 64]` (`BELIEF_LATENT_WIDTH = 64`), and the skill GRU is 128 wide (ADR section 3 D1). The record does not reconcile these widths.

**(B3)** Prior: $z_t^- \sim p_\theta(z_t \mid h_t)$. The belief predicted before the new observation. *RETAINED* (PLAN §17.3).

**(B4)** Posterior: $z_t^+ \sim q_\phi(z_t \mid h_t, O_t)$. The belief corrected by the new observation. *RETAINED* (PLAN §17.3).

**(B5) Belief feature normalization.** *CODE*: decided by the belief module, not OPEN (`belief/SKILL.md`, "Column sources", "Counters" and decisions 1 to 3). The contract fixes the ten names, their order and $\mathrm{SAT} =$ `TICKS_SATURATION` $= 256.0$ (*COMMITTED*, `contracts.py`). The two tick counters are published divided by SAT:

$$f_6 = \frac{c^{\mathrm{dmg}}_t}{\mathrm{SAT}},\qquad f_7 = \frac{c^{\mathrm{seen}}_t}{\mathrm{SAT}},\qquad c_t = \begin{cases}0 & \text{if the counter's event is observed at } t\\ \min\!\big(c_{t-1} + 30\,\Delta t_t,\ \mathrm{SAT}\big) & \text{otherwise}\end{cases}$$

The event is an observed health drop for $c^{\mathrm{dmg}}$ and a legal sighting of the target for $c^{\mathrm{seen}}$. Both counters start at SAT, so $f_6, f_7 \in [0, 1]$. Code: `belief/core.py:400` and `:402` divide by SAT, `:372-373` turn the elapsed seconds $\Delta t \ge 0$ into ticks with `HYTALE_TICKS_PER_SECOND` $= 30$, and `:316-317` start both counters at SAT. J0 multiplies $f_6$ back by SAT before its 40-tick test (D9). The other eight columns are normalized as `belief/SKILL.md` "Column sources" states (the distance is 24 times the encoder value; the speeds are in blocks per tick, formulas S.8 and S.9).

*Unit note (read in the code; the runtime effect is not tested).* A belief tick is 1/30 s. The session clock (D10) advances 0.045 s per environment step, so one step adds 1.35 belief ticks, and J0's 40-tick window is about 29.6 steps, or 1.33 s. `tasks/composition.py` sets `EVADE_TRIGGER_WINDOW_TICKS = 40` as "the same window J0 uses", and `run_composition._score_run` passes `classify` recorded step indices, so that window is 40 steps, or 1.8 s. Confirm by logging, for one lane, the step of an observed health drop and the step at which $f_6 \cdot \mathrm{SAT}$ reaches 40.

## C. Goal selection and action factorization

**(C1)** $G_t \sim \pi_{\mathrm{high}}(G \mid B_t, E_t)$, with $G_t = (k_t, \theta_t)$: the skill and its parameters. *COMMITTED* (ADR D2 extends PLAN §4 with $E_t$; ADR section 4 says "π_high now selects (skill, parameters)"). PLAN §4 writes $\pi_{\mathrm{high}}(G_t \mid B_t)$. ADR section 4 (current text) says the executive state $E_t$ conditions $\pi_{\mathrm{high}}$, so the earlier $I_t$ wording is no longer in the record. This document uses $E_t$.

**(C2)** $A_t \sim \pi_{\mathrm{low}}(A_t \mid B_t, G_t, M_t)$, with $M_t$ the authoritative legal-action mask. *RETAINED* (PLAN §4). Level 1 emits `int32[B, 12]` factors (ADR section 2; `ACTION_HEADS = 12`).

**(C3)** $\pi(A_t \mid B_t, G_t, M_t) = \pi(k_t)\,\pi(\tau_t \mid k_t)\,\pi(p_t \mid k_t, \tau_t)\,\pi(d_t \mid k_t, \tau_t, p_t)$, where $k$ is the action type, $\tau$ the target, $p$ the position, direction, range or charge, and $d$ the duration or persistence. *RETAINED* (PLAN §20.4). **OPEN**: the record does not map these four factors onto the 12 published heads. PLAN §20.2 says only "factorized action heads".

**(C4)** Masked sampling:

$$\tilde\pi(a \mid \cdot) = \frac{\pi(a \mid \cdot)\,\mathbf 1[a \in M_t]}{\sum_{a' \in M_t}\pi(a' \mid \cdot)}$$

*RETAINED* (PLAN §20.5: "Invalid actions are removed before sampling rather than discouraged only through negative reward"). The renormalization is this document's reading of "removed before sampling". The masking law is studied in [K1] (*EXTERNAL*). The mask $M_t$ is not leak-free on the current surface (ADR section 3 D5, 2026-10-16): see (A1).

## D. Orchestration and the JEV contract

Items D1 to D6 are the orchestration rules of ADK, *COMMITTED* in ADR section 3 D2 and implemented. They were read in the repository code for this document. Item D7 is the JEV rule itself, and D10 to D12 are the session runner's rules.

**(D1) Manager decision due.** *COMMITTED.*

$$\mathrm{due}_b = [\tau_b \ge T]\ \lor\ g_b\ \lor\ \bigvee_{e}\big(a_{b,e} \land k_{b,e}\big)$$

with $T$ the manager cadence, $g_b$ true on a pending termination, a timeout, or no valid goal, $a_{b,e}$ the active bit of `DecisionEvent` $e$, and $k_{b,e}$ its known bit. Only an event that is both active and known can trigger a decision. Confirmed in code: `adk/architecture/decision_context.py`, function `manager_decision_due` (line 415). Its body computes `critical = any(values & known)` and returns `(seconds >= cadence) | goal_terminated | critical`.

**(D2) Elapsed time.** *COMMITTED.*

$$\Delta_b = \begin{cases}\max(\delta_b, 0) & \text{if } \delta_b \text{ is known}\\ 0 & \text{otherwise}\end{cases}$$

Ages advance by $\Delta_b$: $\mathrm{age}^{\mathrm{goal}} \leftarrow \mathrm{age}^{\mathrm{goal}} + \Delta$ and $\tau \leftarrow \tau + \Delta$. Unknown time never advances the clocks. Confirmed in code: `adk/architecture/hierarchy.py`, line 244 (`delta = jnp.where(delta_known, jnp.maximum(raw_delta, 0.0), 0.0)`).

**(D3) Commitment, timeout and emergency.** *COMMITTED.*

$$\mathrm{commit}_b = [\mathrm{age}_b \ge m^{\min}_b],\qquad \mathrm{timeout}_b = \mathrm{valid}_b \land [\mathrm{age}_b \ge m^{\max}_b],\qquad \mathrm{emerg}_b = \mathrm{valid}_b \land \bigvee_{e}\big(a_{b,e} \land k_{b,e} \land M_{b,e}\big)$$

$m^{\max}$ may be $+\infty$, and $M$ is the emergency-interrupt mask of the goal in force. Confirmed in code: `hierarchy.py` (`commitment_elapsed`, `timed_out`, `emergency_events`, `emergency_interrupt`), read at lines 240 to 290.

**(D4) Goal switch.** *COMMITTED.*

$$\mathrm{switch}_b = \mathrm{due}_b \land \mathrm{allowed}_b \land \mathrm{requested}_b$$

$$\mathrm{allowed}_b = \lnot\mathrm{valid}_b \lor \mathrm{commit}_b \lor \mathrm{timeout}_b \lor \mathrm{emerg}_b,\qquad \mathrm{requested}_b = \lnot\mathrm{valid}_b \lor \mathrm{timeout}_b \lor \mathrm{pending}_b \lor (\mathrm{due}_b \land \mathrm{term}_b)$$

An ordinary replacement waits for the minimum commitment. A matching emergency event bypasses that minimum. The maximum duration always forces a replacement. Confirmed in code: `hierarchy.py`, lines 347 to 356 (`switch_allowed`, `switch_requested`, `goal_switched = manager_due & switch_allowed & switch_requested`).

**(D5) Action age.** *COMMITTED.* $\mathrm{age}^{a\prime} = \mathrm{age}^{a} + \Delta$ if the action is valid, has the same factors, and its lifecycle has not ended; otherwise $0$. From `experimental/neural_agent/README.md` formula 6, citing `hierarchy.py:399`. The second pass of 2026-10-16 re-read `hierarchy.py:399-402`, and the rule matches.

**(D6) Executive lowering** (`executive_arrays`, `experimental/neural_agent/contracts.py`). For lane $b$ and goal slot $g \in \{0,1,2,3\}$ (`MAX_GOALS = 4`):

$$\mathrm{skill\_index}[b,g] = \mathrm{idx}(\mathrm{skill\_id}_g)\ \text{or}\ -1,\quad \mathrm{goal\_valid}[b,g] = [g < |G|],\quad \mathrm{budget}[b,:] = (\mathrm{jev\_latency\_ms},\ \mathrm{search\_expansions},\ \mathrm{executive\_calls\_per\_minute}),\quad \mathrm{version}[b] = E_t.\mathrm{version}$$

Priority and parameters are zero-padded to width 4. An unknown skill id raises `KeyError`. *COMMITTED* (ADR section 5, seam 3).

**(D7) JEV latency-budget rule.** For instance $i$ with budget $L_i$ in milliseconds, and decision $t$ with wall-clock latency $\ell_{i,t}$ in milliseconds:

$$\mathrm{accept}_{i,t} = [\ell_{i,t} \le L_i],\qquad G_t = \begin{cases} J_i(B_t, E_t) & \text{if } \mathrm{accept}_{i,t} \\ G_{t-1} & \text{otherwise} \end{cases},\qquad \mathrm{misses}_i = \sum_{t}\mathbf 1[\ell_{i,t} > L_i]$$

*COMMITTED* (ADR section 2, non-negotiable 2; ADR section 3 D2, "timeout semantics: the previous goal stands and the miss is counted"). The boundary $\le$ is the rule as this document states it, and it matches `experimental/neural_agent/README.md` formula 7. The ADK orchestration does not enforce it yet (ADR section 3 D2: "Orchestration in `adk/` does not enforce the latency budget yet (seam 8)"). **OPEN**: `ComputeBudget.jev_latency_ms` (the executive's grant, default 20.0) and `JevBudget.latency_ms` (the instance's declaration) are two records of one budget, and the session runner adds a third, `SessionConfig.host_budget_ms` (default 20.0; D11). The record does not say which one governs.

*Where the rule is enforced today.* `JevRunner.__call__` (`jev/runner.py`; `jev/SKILL.md` section F2) times HOST-mode calls only and counts a miss when `latency_ms > budget.latency_ms`. `JevRunner._decide` then applies the fallback of (D12). An IN_GRAPH instance, such as the J0 instance `ScriptedJev`, is not timed: its latency statistics are NaN (`jev/SKILL.md` F5). So the rule does not run for J0. `session.py` applies the same boundary to host-mode deciders (D11).

**(D8) Instance statistics.** *CODE.* `JevStats` reports decisions, misses, `latency_ms_p50`, `latency_ms_p95` and `compute_units`. `jev/runner.py` computes the percentiles with `numpy.percentile` and linear interpolation (`JevRunner.stats`). The executive ledger, `executive/scheduler.py`, uses nearest rank (`nearest_rank`: rank $\lceil p\,n/100\rceil$ in integer arithmetic, no interpolation). Both files were being edited by other sessions when this pass re-read them, so they are cited by symbol. The record names neither method (**OPEN**). The two disagree, and THESIS section 7.4 uses nearest rank.

**(D9) J0 scripted rule.** *CODE* (implemented; the record does not fix it). ADR section 3 D2 and section 8 (P2) say only that J0 is "scripted rules over frozen skills" for the composition cases. The rule below is `jev/scripted.py`, as `jev/SKILL.md` section F1 states it. Its constants are `EVADE_DAMAGE_WINDOW_TICKS = 40.0` and `EVADER_SKILL_ID = "evader"`. Let $\tau_b$ be the tick count of `ticks_since_damage` (belief column 6). The belief publishes that column as its counter divided by `TICKS_SATURATION` (`belief/core.py:400`; B5), and J0 multiplies it back, so $\tau_b$ is the published value times `TICKS_SATURATION` (`ScriptedJev.__call__`, `jev/scripted.py:188`; the window test is at `:193`). The counter is in belief ticks of 1/30 s (B5). Let $\kappa_b$ be its known bit, and let $\mathrm{is\_evader}_{b,g} = \mathrm{goal\_valid}_{b,g} \land (\mathrm{skill\_index}_{b,g} = \mathrm{evader})$.

$$\mathrm{evading}_b = \kappa_b \land [\tau_b < 40] \land \bigvee_{g}\mathrm{is\_evader}_{b,g}$$

$$\mathrm{slot}_b = \begin{cases}\arg\max_g\{\mathrm{priority}_{b,g} : \mathrm{is\_evader}_{b,g}\} & \text{if } \mathrm{evading}_b \\ \arg\max_g\{\mathrm{priority}_{b,g} : \mathrm{goal\_valid}_{b,g}\} & \text{otherwise}\end{cases}$$

Ties go to the lowest slot. Let $\mathrm{has}_b = \mathrm{evading}_b \lor \bigvee_g \mathrm{goal\_valid}_{b,g}$. The output is:

- skill: $\mathrm{skill\_index}_{b,\mathrm{slot}_b}$ if has, else 0 (idle);
- goal slot: $\mathrm{slot}_b$ if has, else $-1$; parameters: $\mathrm{params}_{b,\mathrm{slot}_b}$ if has, else $0$;
- minimum commitment: $2.0$ if evading, $0.5$ for another goal, $0.0$ if none;
- maximum duration: $10.0$ if has, else $+\infty$ (`jev/SKILL.md` writes $\min(10.0, 30.0)$, which equals $10.0$);
- emergency mask: $\{\mathrm{DAMAGE\_RECEIVED}, \mathrm{TARGET\_DIED}, \mathrm{NPC\_STUCK}\}$ if has, else the empty set;
- terminate: true when the chosen skill differs from the current skill.

Summarized as a piecewise definition:

$$J_0(B_t, E_t) = \begin{cases} \text{highest-priority evader goal} & \text{if } \kappa_b \land [\tau_b < 40] \text{ and an evader goal is valid} \\ \text{highest-priority valid goal} & \text{if any goal is valid} \\ \text{idle (skill 0)} & \text{otherwise}\end{cases}$$

Comparison with the P2 list (this document's reading of the rule). The rule responds to damage inside a 40-tick window. It does not respond to `PROJECTILE_THREAT_ENTERED`, and it has neither sequencing (navigate, then pursue) nor dependency (interact after arrival). The window and the commitments are implementation values that the record does not set. **OPEN**.

**(D10) Session virtual clock.** *COMMITTED*: ADR section 6.1 lists the session runner as implemented with a "0.045 s virtual clock", and ADR D3 commits to explicit virtual-time hooks. Code: `experimental/neural_agent/session.py` (`TICK_SECONDS = 0.045`; `run_session`).

$$t_v = n \times 0.045\ \mathrm{s}$$

Here $n$ is the number of environment steps the session's host hook has seen; it increases by one per step. `SessionConfig.tick_seconds` replaces 0.045 and must be finite and positive. Time spent in a host decision does not advance $t_v$. **OPEN** (ADR section 12, question 1): the unit of virtual time in general, and the map from language-model wall-clock to ticks.

**(D11) Host-mode decision in the session.** *COMMITTED*: ADR section 2, non-negotiable 2, and ADR D2, "the previous goal stands and the miss is counted". Code: `session.py`, the host hook in `run_session` (its step 4). With $s$ the step index, $K$ = `host_cadence_ticks` (default 11) and `host_on_events` (default True), a decision is due when

$$\mathrm{due}_s = [\,s \bmod K = 0\,] \lor \big(\texttt{host\_on\_events} \land \text{some event is active on some lane}\big)$$

For a due decision with returned goal $g_s$, wall-clock latency $w_s$ in milliseconds (`time.perf_counter` around `HostDecider.decide`) and budget $L$ = `host_budget_ms` (default 20.0):

$$\mathrm{accept}_s = [\,g_s \ne \varnothing\,] \land [\,w_s \le L\,],\qquad \mathrm{misses} \leftarrow \mathrm{misses} + \mathbf 1[\lnot\,\mathrm{accept}_s]$$

On acceptance the session writes $g_s$ into the carried state for every lane, with goal age 0 and `goal_valid` True. On a miss it writes nothing, so the goal in force stands. An exception from the decider counts as $g_s = \varnothing$ and is also counted in `errors`. Reading of the code, not tested here: a decider that returns `None` to keep the current goal, which the `HostDecider` docstring allows, is counted as a miss too.

**(D12) Miss fallback while the core forces a switch.** *COMMITTED*: ADR D2, the JEV contract paragraph: "When the core is forcing a switch at that moment (the previous goal timed out, or the actor requested termination after its commitment), "stands" is impossible, and the fallback is the idle goal rather than a renewed copy of the ended goal (decision 2026-10-16, from the runner verifier's finding that a renewed goal would reset its age)." That paragraph was read in the ADR as modified at 21:55 on 2026-10-16, which is newer than the version pinned at the top of this file. Code: `JevRunner._decide` in `jev/runner.py`, not `session.py`. On a miss, for lane $b$:

$$\mathrm{forced}_b = \lnot\,\mathrm{valid}_b \lor \mathrm{timed\_out}_b \lor \mathrm{actor\_termination\_requested}_b,\qquad G_b = \begin{cases}\texttt{idle\_goal} & \text{if } \mathrm{forced}_b \\ G_{t-1,b} & \text{otherwise}\end{cases}$$

`terminate` is False and the instance state is kept. Without `goal_runtime` the forced lanes are unknown, and the previous goal is re-emitted on every lane. The code also counts an invalid goal as forcing, which the ADR's parenthesis does not name. The session's host path (D11) has no idle fallback of its own: on a miss it leaves the in-graph manager's goal in force.

## E. Losses and training

**(E1) Behaviour cloning.** $\mathcal L_{\mathrm{BC}} = -\log \pi_S(A_t^{\mathrm{expert}} \mid B_t, G_t)$. *RETAINED* (PLAN §27.1).

**(E2) Distillation from a privileged teacher.** $\mathcal L_{\mathrm{distill}} = w_t\,D_{\mathrm{KL}}\!\big(\pi_T(\cdot \mid P_t, G_t)\ \|\ \pi_S(\cdot \mid B_t, G_t)\big)$. *RETAINED* (PLAN §27.2). The weight $w_t$ is set by (F1) or (F2), which is what H7 compares. The teacher–student split is that of Learning by Cheating [2] (*EXTERNAL*; ADR section 3 D5).

**(E3) Hidden-state prediction.** $\mathcal L_{\mathrm{hidden}} = -\log p_\theta\big(P_t^{\mathrm{relevant\ hidden}} \mid O_{\le t}, A_{<t}\big)$. Use distributions, not point targets, for uncertain facts. *RETAINED* (PLAN §27.3). ADR section 3 D6 gives the reason: a recurrent state is not automatically a calibrated belief, so auxiliary targets stay task-aware and distributional [B1][B2] (*EXTERNAL*).

**(E4) Dynamics.** Continuous quantities: $\mathcal L_{\mathrm{dynamics}} = \mathrm{Huber}(\Delta\hat x, \Delta x)$ (*EXTERNAL* [HUBER]). Discrete events: $\mathcal L_{\mathrm{events}} = -\sum_e y_e \log\hat p_e$. *RETAINED* (PLAN §27.4).

**(E5) Reveal correction.** $\mathcal L_{\mathrm{reveal}} = D\big(B_t^{\mathrm{prior}}, P_t^{\mathrm{newly\ revealed}}\big)$. *RETAINED* (PLAN §27.5). **OPEN**: PLAN does not define the divergence $D$.

**(E6) Multi-step prediction.** $\mathcal L_{\mathrm{multi\text{-}step}} = \sum_{k \in \{1,2,4,8\}}\lambda_k\,D\big(\hat P_{t+k}, P_{t+k}\big)$. *RETAINED* (PLAN §27.6).

**(E7) Total objective.**

$$\mathcal L_{\mathrm{total}} = \lambda_{\mathrm{BC}}\mathcal L_{\mathrm{BC}} + \lambda_{\mathrm{KD}}\mathcal L_{\mathrm{distill}} + \lambda_{\mathrm{belief}}\mathcal L_{\mathrm{hidden}} + \lambda_{\mathrm{dyn}}\mathcal L_{\mathrm{dynamics}} + \lambda_{\mathrm{event}}\mathcal L_{\mathrm{events}} + \lambda_{\mathrm{reward}}\mathcal L_{\mathrm{reward}} + \lambda_{\mathrm{value}}\mathcal L_{\mathrm{value}} + \lambda_{\mathrm{reveal}}\mathcal L_{\mathrm{reveal}} + \lambda_{\mathrm{roll}}\mathcal L_{\mathrm{multi\text{-}step}} + \lambda_{\mathrm{KL}}\mathcal L_{\mathrm{latent}}$$

*RETAINED* (PLAN §27). ADR section 4 adds a value- or decision-aware model loss. **OPEN**: no $\lambda$ weights are set in the record.

**(E8) J1 behaviour cloning with a termination head.**

$$\mathcal L_{J1} = -\log\pi_{J1}(k^{\ast} \mid x) - \Big[r^{\ast}\log\sigma(l) + (1 - r^{\ast})\log\big(1 - \sigma(l)\big)\Big]$$

Here $k^{\ast}$ is the expert skill index, $x$ the J1 input (B_t and E_t), $r^{\ast} \in \{0,1\}$ the expert's termination label, $\sigma$ the logistic function, and $l$ the termination logit. The first term is the *RETAINED* behaviour-cloning term (PLAN §27.1). The second is a binary cross-entropy in its standard form (*EXTERNAL*, written here; no citation). PLAN §17.2 (option termination head) and §19.4 (learned termination) name the idea, but neither writes this equation, and the brief gives it. The idea is the options framework: skills as terminating options [O1] whose termination can be learned [O2] (*EXTERNAL*; ADR section 3 D1). **OPEN**: the record does not define $r^{\ast}$ or its source.

**(E9) PPO clipped surrogate, as used by the repository.** *EXTERNAL* [PPO] gives the surrogate. The operative code is `HytaleRL/hytalegym/hytalegym/jax/training/ppo.py`.

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)},\qquad L^{\mathrm{CLIP}}(\theta) = \hat{\mathbb E}_t\Big[\min\big(r_t(\theta)\,\hat A_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,\hat A_t\big)\Big]$$

with $\epsilon = $ `clip_epsilon` $= 0.2$ (`types.py:34`). The repository minimizes (`ppo.py:868-905`):

$$\mathcal L = -L^{\mathrm{CLIP}}(\theta) + c_v L^{VF} - c_e\,\hat{\mathbb E}[H]$$

with $c_v = $ `value_coefficient` $= 0.5$ (`types.py:35`) and $c_e = $ `entropy_coefficient` $= 0.01$ (`types.py:36`). The value loss clips the value at the same $\epsilon$:

$$L^{VF} = \tfrac12\,\hat{\mathbb E}\Big[\max\big((V_\theta - R)^2,\ (V^{\mathrm{clip}} - R)^2\big)\Big],\qquad V^{\mathrm{clip}} = V_{\mathrm{old}} + \mathrm{clip}(V_\theta - V_{\mathrm{old}}, -\epsilon, \epsilon)$$

(`ppo.py:889-898`). The repository also drops samples whose $|\log r_t| > 20$ (`_MAXIMUM_COMPARABLE_LOG_RATIO`), averages over the rest, and reports the dropped fraction. That guard is for sentinel log-probabilities. It is not part of PPO. Other `PPOConfig` values (`types.py:27-37`): 4 update epochs, 8 minibatches, learning rate $3\times10^{-4}$, $\gamma = 0.99$, GAE $\lambda = 0.95$, global gradient-norm clip $0.5$. The paper's typeset equations were not retrieved; the form above is confirmed from the code.

**(E10) Generalized advantage estimation.** *EXTERNAL* [GAE]. The repository recursion (`ppo.py:318-336`):

$$\delta_t = r_t + \gamma\big(\mathrm{nonterminal}_t\,V_{t+1} + \mathrm{timeout}_t\big) - V_t,\qquad \hat A_t = \delta_t + \gamma\lambda\,\mathrm{nonterminal}_t\,\hat A_{t+1}$$

Without terminations this equals $\hat A_t = \sum_{l \ge 0}(\gamma\lambda)^l\,\delta_{t+l}$. Here $\mathrm{timeout}_t$ is `time_limit_bootstrap_values[t]`, which is zero when the argument is omitted (`ppo.py:314`, `321–325`, `334`). It is a value, not a flag. The paper's equations were not retrieved; the recursion is confirmed from the code.

**(E11) Decision-level return of J1's semi-MDP PPO.** *CODE*: `jev/train_j1_ppo.py` (module formulas, `decision_advantages`) and `jev/sim_goal_env.py` (`SimGoalDecisionEnv.step`), read 2026-10-17. Discounting a $k$-tick decision by $\gamma^{k}$ is the semi-MDP form of [O1] (*EXTERNAL*; recalled, not re-read on 2026-10-17). A decision $t$ that lasts $k_t \ge 1$ ticks earns

$$R_t = \sum_{i=0}^{k_t-1}\gamma^{i}\,r_{t,i},\qquad \delta_t = R_t + \gamma^{k_t}(1-d_t)\,V(s_{t+1}) - V(s_t),\qquad \hat A_t = \delta_t + (\gamma\lambda)^{k_t}(1-d_t)\,\hat A_{t+1}$$

with $\gamma = 0.99$ per tick, $\lambda = 0.95$, and $d_t$ the arena's `EpisodeBoundary.done`: a death, a world error or the episode limit. The protocol has no truncation channel, so the limit counts as terminal and is not bootstrapped. With every $k_t = 1$ this is (E10) without time limits. On `SimGoalDecisionEnv`, $k_t$ is the first $k \ge 1$ that meets one of three conditions: $k$ equals `cadence_ticks` (default 11, the host cadence of D11); the episode ends ($d$); or an event in the goal's emergency mask is active at tick $t + k$ (the core's test, D3). The per-tick reward $r$ is the arena's native combat reward, with no shaping:

$$r = 2.0\,D^{\mathrm{dealt}} - 1.0\,D^{\mathrm{received}} + 50\cdot\mathbf 1\{\mathrm{completion}\} - 20\cdot\mathbf 1\{\mathrm{death}\}$$

$D^{\mathrm{dealt}}$ is the damage the agent dealt to opponents on that tick, and $D^{\mathrm{received}}$ the damage it received. Completion is awarded once, when every opponent's health is at or below 0, and death once, when the agent's health is. $r = 0$ on an invalid tick. *CODE*: `compose_reward` in `HytaleRL/hytalegym/hytalegym/jax/combat/types.py`, with the `reward` block of `rulesets/hytale_0_5_7/kweebec_razorleaf_vs_trork_brawler_v9.json`. The four scales were confirmed on the built `combat_open_flat` arena's `combat_params` on 2026-10-17.

## F. Imitation weights (hypothesis H7)

**(F1) PLAN §24.3 weight.** *RETAINED.* PLAN labels it "an implementation proposal inspired by imitation-gap research, not a claim that the exact formula is established".

$$w_t = \exp\!\Big(-\alpha\,D_{\mathrm{KL}}\big[\pi_T(\cdot \mid P_t)\ \|\ \pi_T(\cdot \mid \hat P_t)\big]\Big),\qquad \mathcal L_{\mathrm{IL}} = -w_t\log\pi_S(A_t^{T} \mid B_t)$$

$\hat P_t$ is the student's reconstruction of the privileged state. The weight is high when legal history supports the teacher's action.

**(F2) ADVISOR weight.** *EXTERNAL* [3]. The paper's Eq. (3) (read 2026-10-16 from arXiv HTML v3) gives $w(s) = m_\alpha\big(d(\pi^{\mathrm{teach}}, \pi^{\mathrm{aux}})(s)\big)$ with $m_\alpha(x) = e^{-\alpha x}$. Sec. 3.2 uses the KL divergence for $d$. So

$$w(s) = \exp\!\Big(-\alpha\,\mathrm{KL}\big(\pi^{\mathrm{teach}}(\cdot \mid s)\ \|\ \pi^{\mathrm{aux}}(\cdot \mid s)\big)\Big)$$

which matches the record's citation of [3] and H7. The paper text does not fix the argument order of the KL. The record's order, teacher first, is used here.

**(F3) ADVISOR student loss.** *EXTERNAL* [3], paper Eq. (2), with the cross-entropy of its Eq. (1):

$$\mathcal L^{\mathrm{ADV}}(\theta) = \mathbb E_\mu\Big[w(S)\,\mathrm{CE}\big(\pi^{\mathrm{teach}}(S), \pi_f(S, \theta)\big) + \big(1 - w(S)\big)\,\mathcal L(\theta, S)\Big],\qquad \mathrm{CE}(\pi^{\mathrm{teach}}, \pi_f)(s) = -\sum_{a}\pi^{\mathrm{teach}}(a \mid s)\log\pi_f(a \mid s)$$

Here $\mathcal L$ is the reward-based RL loss. The equations were read from the arXiv HTML rendering through a summarizing fetch. Re-check them against the PDF before quoting.

**(F4) Auxiliary head.** *EXTERNAL* [3]. Trained by imitation only (paper Fig. 3 caption: "trained only using IL"), as a second actor head on the shared representation (Sec. 3.3). The paper does not write the auxiliary objective out. Taking $\pi_f = \pi^{\mathrm{aux}}$ in the cross-entropy of (F3) is this document's inference.

**(F5) What the H7 experiment must separate.** *RETAINED* (PLAN §24.3) and *EXTERNAL* [3]. PLAN §24.3 weights an imitation log-likelihood by a KL between the teacher under true and reconstructed privileged state, and has no RL term. ADVISOR weights its imitation term by a KL to a legal auxiliary head and mixes it with the RL loss through $(1-w)$. The ADR says the released code adds a small-probability cutoff. The paper text as fetched has none in $m_\alpha$. The released code (`advisor_losses.py` in `allenai/advisor`) was not read. **OPEN**.

---

## G. Critics

**(G1) Training critic and actor.** The training critic is $V_\psi(B_t, P_t, G_t)$ and the actor is $\pi_\phi(A_t \mid B_t, G_t)$. *RETAINED* (PLAN §24.4). PLAN warns that "some state-critic formulations produce biased policy gradients". The record cites robust asymmetric learning [W1] for this setting (*EXTERNAL*; ADR section 3 D5).

**(G2) History-state critic.** $V_\psi(h_t, s_t)$ conditions on the actor's history $h_t$ and the state $s_t$. A state-only critic $V(s_t)$ does not. The record states that "the history-state critic V(h, s) is unbiased where a state-only critic is not" (ADR section 3 D5 and section 6.2 [21]). *EXTERNAL* [21], partly checked on 2026-10-16 against the arXiv abstract. The abstract says the state-only variant has "fundamental issues which undermine the validity of a common variant". It says the proposed variant maintains "the validity of the policy gradient theorem" and introduces "no bias". It does not say what the proposed critic conditions on, so the history-conditioning claim is not confirmed from the abstract. The full text was not retrieved. Inferred mechanism, not checked: a critic over $s$ alone averages over hidden state with a different weighting from the one the history-conditioned actor implies, which biases the gradient.

**(G3) Critic of J1's PPO.** *CODE*: `jev/j1_actor_critic.py`, read 2026-10-17. J1 has no value head, so the PPO of (E9) and (E11) pairs it with a separate critic on the same input $x$ (`j1_inputs`: $B_t$, $E_t$ and the goal in force):

$$h_1 = \mathrm{relu}(x W_1 + b_1),\qquad h_2 = \mathrm{relu}(h_1 W_2 + b_2),\qquad V_\psi(x) = h_2\,w_v + b_v,\qquad W_1 \in \mathbb R^{D \times 64},\ W_2 \in \mathbb R^{64 \times 64}$$

The initialization is He normal for $W_1$ and $W_2$, std $\sqrt{1/64}$ for $w_v$, and zero biases, as in the trainer's stand-in (`init_mlp_policy`). The critic reads only the actor's legal input. It is neither the privileged critic of (G1) nor the history-state critic of (G2). The loss reads J1's skill logits only, so J1's replace and params heads are not trained. Only the actor is deployed: `save_j1_params` writes the ten `J1Params` arrays that `eval/arms.py::load_j1_params` reads.

## H. Dream score and uncertainty

**(H1) Dream score.** *RETAINED* (PLAN §23.3).

$$J(t) = \sum_{k=0}^{H-1}\gamma^k\,\hat r_{t+k} + \gamma^{H}\,V\big(\hat B_{t+H}, G_t\big) - \beta\,U_t$$

$U_t$ is the predictive uncertainty. PLAN writes the value term as $V_{\mathrm{tactical}}$; ADR section 3 D2 names the JEV's value head. The horizon $H$ is PLAN's 4 to 8 low-level decisions. ADR D2 (COMMITTED) measures the horizon at the skill level, in goal decisions, and keeps PLAN's 4 to 8 for any later per-tick model.

**(H2) Uncertainty terms.** Version one, *RETAINED* (PLAN §23.4): predicted variance, event entropy and an out-of-distribution detector. An uncertainty penalty discourages exploiting unreliable learned dynamics (MOPO [20], *EXTERNAL*). Short branched rollouts from real states are supported by MBPO [19] (*EXTERNAL*).

## I. Value equivalence

**(I1) Policy-evaluation Bellman operator** for a model $m = (r_m, P_m)$. *EXTERNAL* [V1]; the operator is written here in its standard form.

$$(T^{\pi}_m v)(s) = \sum_a \pi(a \mid s)\Big[r_m(s,a) + \gamma\sum_{s'}P_m(s' \mid s,a)\,v(s')\Big]$$

**(I2) Value equivalence.** *EXTERNAL* [V1]. Models $m$ and $\tilde m$ are value equivalent with respect to a set of policies $\Pi$ and a set of value functions $\mathcal V$ when

$$T^{\pi}_m v = T^{\pi}_{\tilde m} v \qquad \text{for all } \pi \in \Pi,\ v \in \mathcal V$$

The abstract of [V1], read 2026-10-16, says: "two models are value equivalent with respect to a set of functions and policies if they yield the same Bellman updates". The typeset definition was not retrieved. The operator form in (I1) is this document's transcription.

**(I3) Use in the record** (COMMITTED, ADR section 2 and D2; a statement, not a formula). A model can have a small average state error and still rank actions wrongly. The LavaCrossing root cause shows this (ADR section 3 D2). The record therefore gates J2 on action ranking and decision-aware losses rather than on endpoint error (ADR D2.1, lesson 1, which cites [V1] and [DJEPA]). The ADR also cites value-aware model learning [V2] for the point that a model is judged by the Bellman updates it supports (ADR section 3 D2).

## J. Evaluation statistics

**(J1) Wilson lower bound.** *COMMITTED* in code: `agent_skills/training_runs/recipe.py`, function `wilson_lower(result, z=1.96)`, lines 85 to 91. This is the exact computation:

$$\underline p(n, p, z) = \frac{p + \dfrac{z^2}{2n} - z\sqrt{\dfrac{p(1-p) + \dfrac{z^2}{4n}}{n}}}{1 + \dfrac{z^2}{n}},\qquad z = 1.96,\quad n = \texttt{result.episodes},\quad p = \texttt{result.rate}$$

**OPEN**: the function has no guard for $n = 0$, which raises `ZeroDivisionError`. Callers must ensure $n \ge 1$.

**(J2) Advance rule.** *COMMITTED* in code: `recipe.py`, lines 342 to 348. With $\theta_{\mathrm{adv}}$ = `config["advance_lower_bound"]` and the preferred candidate's development results indexed by task $\tau$:

$$\text{verdict} = \begin{cases}\texttt{advance} & \text{if}\ \min_{\tau}\ \underline p\big(n_\tau, p_\tau, 1.96\big) \ge \theta_{\mathrm{adv}} \\ \texttt{hold} & \text{otherwise}\end{cases}$$

The code builds the `bounds` dictionary with `wilson_lower` for each task and then compares its minimum with the threshold. The helper `preferred_candidate` was not read.

**(J3) Brier score.** *EXTERNAL* [BRIER] (Brier 1950, recalled; the form is as README formula 9 states it).

$$\mathrm{BS} = \frac{1}{N}\sum_{i=1}^{N}(p_i - o_i)^2,\qquad o_i \in \{0, 1\}$$

**(J4) Expected calibration error.** *EXTERNAL* [ECE], Eq. (3) of Guo et al. (2017), read from the arXiv HTML on 2026-10-16. The paper's reported values use $M = 15$ bins.

$$\mathrm{ECE} = \sum_{m=1}^{M}\frac{|B_m|}{n}\,\big|\mathrm{acc}(B_m) - \mathrm{conf}(B_m)\big|$$

$$B_m = \{i : \hat p_i \in I_m\},\quad I_m = \Big(\tfrac{m-1}{M}, \tfrac{m}{M}\Big],\quad \mathrm{acc}(B_m) = \frac{1}{|B_m|}\sum_{i \in B_m}\mathbf 1[\hat y_i = y_i],\quad \mathrm{conf}(B_m) = \frac{1}{|B_m|}\sum_{i \in B_m}\hat p_i$$

**(J5) Kullback–Leibler divergence.** *EXTERNAL* [KL] (standard; recalled).

$$D_{\mathrm{KL}}(p \,\|\, q) = \sum_a p(a)\log\frac{p(a)}{q(a)}$$

**(J6) Value of computation.** *EXTERNAL* [R1]. The form is as README formula 9 states it. It has not been checked against the 1991 text, which the ADR cites through secondary sources only.

$$\mathrm{VOC}(c) = \mathbb E_c\big[\max_a Q(s,a)\ \big|\ c\big] - \max_a Q(s,a) - \mathrm{cost}(c)$$

The ADR uses VOC as the trigger for "expected value of additional reasoning" (D3), and cites a 2024 preprint that applies the same framing to language models [R2] (search summary only). The record gives no practical estimator. **OPEN**.

## K. Metrics (ADR section 9; definitions in THESIS section 7)

The families are *COMMITTED* in ADR section 9. The formulas below are this document's, and the method choices they leave open are marked.

**(K1) Success and outcome per unit of reasoning compute.** *COMMITTED* family (ADR section 9); the formula is this document's.

$$\hat S = \frac{1}{N}\sum_{i=1}^{N}\mathbf 1[\text{success}_i],\qquad \mathrm{OPC} = \frac{\hat S}{\bar C},\qquad \bar C = \frac{1}{N}\sum_{i=1}^{N}\sum_{k}w_k\,c_{k,i},\qquad \mathrm{OPC}_{\mathrm{call}} = \frac{\hat S}{\overline{n^{\mathrm{calls}}}}$$

$c = (\text{calls}, \text{tokens}, \text{expansions}, \text{forward passes})$, and the weights $w_k$ are declared before the run. **OPEN**: the compute unit (ADR section 12, question 2) and the interval method ("hierarchical intervals", ADR P4).

**(K2) Latency p95.** *COMMITTED* family (ADR section 9); the formula is this document's. Sorted decision latencies $\ell_{(1)} \le \dots \le \ell_{(n)}$ in milliseconds. Nearest rank:

$$p_{95} = \ell_{(\lceil 0.95\,n\rceil)}$$

**OPEN**: the ADR does not name a percentile method, and the repository uses two: nearest rank in `executive/scheduler.py`, and linear interpolation in `jev/runner.py` (D8). Batch size and device must be stated (PLAN §30.4).

**(K3) Virtual time.** *COMMITTED* family (ADR section 9); the formula is this document's.

$$T_{\mathrm{obj}} = \min\{t : \mathrm{success}(s_t)\} - t_0\ \ \text{(ticks; censored at } T_{\max}\text{ if never reached)},\qquad T_{\mathrm{stale}} = \sum_t\mathbf 1[\text{decision } t \text{ is stale or a fallback}]$$

**(K4) Task concurrency.** *COMMITTED* family (ADR section 9); the formula is this document's. With $P_o(t) \in [0,1]$ a progress measure declared per objective $o$ (**OPEN**: not defined in the record):

$$\kappa = \frac{1}{T_v}\sum_{o}\sum_{t=1}^{T_v}\mathbf 1\big[P_o(t) > P_o(t-1)\big]$$

**(K5) Reaction latency in ticks.** *COMMITTED* family (ADR section 9); the formula is this document's. For an injected `DecisionEvent` $e$ at tick $t_e$:

$$R^{\mathrm{goal}}_e = \min\{t \ge t_e : G_t \ne G_{t_e-1}\} - t_e,\qquad R^{\mathrm{act}}_e = \min\{t \ge t_e : A^{\mathrm{skill}}_t \ne A^{\mathrm{skill}}_{t_e-1}\} - t_e$$

Both are censored at a declared horizon $H_r$. This document's inference: `R^act` needs a paired no-injection control at the same seed. The ADR states the metric without this control.

**(K6) JEV miss rate.** *COMMITTED* family (ADR section 9); the formula is this document's. $\mathrm{miss\_rate}_i = \mathrm{misses}_i / \mathrm{decisions}_i$, from (D7).

## L. Contract shapes

COMMITTED. From `experimental/neural_agent/contracts.py` as read on 2026-10-16. Line numbers are as of the file's modification at 20:00:18 and may move.

| Object (line) | Fields and shapes, with `B` lanes |
|---|---|
| `SharedBeliefState` (207) | `features float32[B, 10]`, `feature_known bool[B, 10]`, `latent float32[B, 64]` |
| `ExecutiveArrays` (583) | `skill_index int32[B, 4]`, `priority float32[B, 4]`, `params float32[B, 4, 4]`, `goal_valid bool[B, 4]`, `budget float32[B, 3]`, `version int32[B]` |
| `GoalPayload` (722) | `skill_index int32[B]`, `params float32[B, 4]`, `executive_version int32[B]`, `goal_slot int32[B]` (−1 when no executive goal) |
| `GoalControl` via `make_goal` (739) | `minimum_commitment_seconds float32[B]`, `maximum_duration_seconds float32[B]` (+∞ allowed), `emergency_interrupt_mask bool[B, 14]` |
| `Skill` (972) | `initial_state(batch) -> float32[B, 128]`; `__call__(state float32[B, 128], belief, actor_input, params float32[B, 4], key) -> (float32[B, 128], int32[B, 12])` |
| `JevInstance` (1098) | `__call__(state, belief, current_goal, actor_input, key, *, decision_context, goal_runtime, executive) -> GoalDecision` |
| `idle_goal` (806) | skill index 0 (`"idle"` by convention), params 0, version 0, goal_slot −1, minimum 0, maximum +∞, mask all False |

Constants: `MAX_GOALS = 4`, `PARAM_WIDTH = 4`, `SKILL_STATE_WIDTH = 128`, `ACTION_HEADS = 12`, `BELIEF_LATENT_WIDTH = 64`, `TICKS_SATURATION = 256.0`, `SKILL_IDLE = "idle"`, `BELIEF_FEATURE_COUNT = 10`. `EVENT_NAMES` has 14 entries: `ATTACK_TELEGRAPH_BEGAN`, `PROJECTILE_THREAT_ENTERED`, `DAMAGE_RECEIVED`, `TARGET_DIED`, `LINE_OF_SIGHT_CHANGED`, `PATH_INVALID`, `ABILITY_COMPLETED`, `ABILITY_INTERRUPTED`, `NPC_STUCK`, `GOAL_IMPOSSIBLE`, `ACTION_COMPLETED`, `ACTION_CANCELLED`, `ACTION_FAILED`, `ACTION_REJECTED`. The `__all__` list has 31 names.

---

## Verification notes

- Read in the repository code during this run: `contracts.py` (signatures, constants, lowering, budget and goal records, read before its 20:00:18 change; after the change, the field and signature lines were re-checked and the import count was re-run), `ppo.py` (loss, GAE, clip, value and entropy coefficients), `types.py` (`PPOConfig` defaults), `recipe.py` (`wilson_lower`, advance rule), `hierarchy.py` and `decision_context.py` (the orchestration rules D1 to D4), `jev/SKILL.md` section F and `jev/scripted.py` (the J0 rule, matched against the code by grep), and `eval/arms.py` (the shuffled arm, matched by grep).
- Second pass, 2026-10-16: `hierarchy.py:399-402` (D5), `jev/runner.py` (D7, D8), `jev/scripted.py` (D9), the PPO loss in `ppo.py` (E9) and its GAE recursion (E10) were read again. The second pass corrected D9 (belief scaling) and E10 (the timeout term), and added labels to D1 to D5, D8, E8, F4, F5, I2 and K1 to K6.
- Fetched in this run: the arXiv abstracts of [PPO], [GAE], [V1], [21] and [ECE]; the arXiv HTML of [3] (Eq. (1) to (3)) and of [ECE] (Eq. (3)). The fetch gave the PPO and GAE abstracts only, and no equations for them. The value-equivalence HTML returned 404.
- DOI records checked on Crossref, 2026-10-16, for [BRIER], [KL], [HUBER], [WILSON] and [GRU] (the arXiv abstract page for [GRU]). Their papers were not read. [WILSON] is the 1927 source of the bound that `recipe.py` implements.
- Third pass, 2026-10-17 (UTC): every file and line reference in this file was re-located against the files on disk at about 22:10 local time on 2026-10-16. D9 cited `jev/scripted.py:18` for the multiply, which is at `:188` (fixed). The `hierarchy.py`, `decision_context.py`, `ppo.py`, `types.py`, `recipe.py`, `belief/core.py` and `contracts.py` lines still match. `jev/runner.py` and `executive/scheduler.py` were being edited by other sessions, so D7, D8 and D12 cite them by symbol. Added: B5 (the belief normalization, decided in the belief module), D10 to D12 (session clock, host decision, miss fallback), and the citations [2], [B1], [B2], [N1] to [N3], [O1], [O2], [W1], [V2], [DJEPA] and [R2], each where ADR-001 cites that paper for the same claim.
- 2026-10-17, J1 sim-PPO pass: added (E11) and (G3). Read for them: `jev/train_j1_ppo.py`, `jev/sim_goal_env.py`, `jev/j1_actor_critic.py`, `adk/architecture/hierarchy.py` (the emergency test of D3), `compose_reward` in `HytaleRL/hytalegym/hytalegym/jax/combat/types.py`, completion and death in `jax/combat/arsenal/runtime.py`, and the ruleset's `reward` block. No other entry was changed.
