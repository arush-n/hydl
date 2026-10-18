# Phase C: the plan for training the first skill bodies (PROPOSED 2026-10-18, late; needs the owner's approval)

**Provisional (owner, 2026-10-18 late).** The architecture may move away from pure PPO with per-verb adapters, toward
one generalist, request-conditioned policy, "similar to a vla model". Section 2's body and algorithm are therefore
provisional until the architecture research lands. The method (section 2b), the staging, the gates and the
dependencies stand either way.

Nothing in this plan runs until the owner approves it (BUILD.md section 0). It answers the owner's condition for
training: what trains, on what, for how long, what counts as success, what happens on failure, and what each step
waits on. Approval is staged, so compute is never spent ahead of evidence:
- **Stage 1** asks for steps S1 to S4 (building, no training) and C1 (a throughput measurement whose policies are
  thrown away).
- **Stage 2** comes back with C1's measured numbers and asks for the budget of C2 and C3.
- Later stages are asked for the same way.

## 1. The question and the hypotheses

**Question.** Does one recipe train competent, single-competency skill bodies on the gen-7 NPC model, from
native-trace warm starts, inside the 30 Hz budget? And can the library grow without changing a frozen skill?

| # | Hypothesis | Proof (BUILD.md 6) | Pass bar | If it fails (BUILD.md 6) |
|---|---|---|---|---|
| H1 | The set encoder does at least as well as the flat MLP on the baseline task | E2 | catches not below the baseline's lower bound, same recipe and suite | tune depth and width once; else keep the flat encoder for locomotion only |
| H2 | One locomotion body learns walk_to, step_and_hop and go_around, each added without losing the earlier ones | E3 | each verb at or above its single-skill result | larger per-verb adapters; else one body per verb family |
| H3 | Typed arguments are honoured | E4 | pace within 5 % of the gait asked, radius respected, in 95 % of episodes | arguments as extra tokens instead of a vector |
| H4 | A re-goal mid-episode is recovered | E5 | within the contract's commitment; success within 10 % of a fresh start | more goal changes per training episode |
| H5 | Status is calibrated | E6 | ECE <= 0.05; stuck precision >= 0.9 | temperature recalibration; else a separate critic |
| H6 | Cloning from the native traces saves steps | E10 | the baseline bar in >= 30 % fewer steps, or equal success with lower jitter | RL from scratch |
| H7 | The framework is chosen by measured speed | E11 | lane-steps per second on this machine decide | JAX (the default) |
| H8 | Results hold under legal perception | E12 | success reported at FOV16 and FOV32 with occlusion; no privileged input | fix the input, never the test |
| H9 | A frozen verb stays bit-identical with real weights | E13 | outputs bit-identical, suite numbers unchanged | separate networks for that body |

## 2. Fixed choices

- **Simulator:**
  - the gen-7 render (HEAD's engine plus the `sim/gen7` overlay), mounted per run with
    `hytalegym.sim.common.render.redirect_imports`;
  - every result records the render's commit and overlay hash;
  - the gen-7 lane is the peer session's (GEN7-PLAN 2b): this plan consumes the render and never edits it.
- **Body model:** `npc_walk`, the NPC the bridge drives (owner ruling 2026-10-18). Walking is one of its movement
  states. Player-only movement (sprint transitions, slide, landing roll) cannot be verified on this body, so those
  verbs wait.
- **Clock:** 1/30 s per tick. Horizons are in ticks, from measured time to success (`catch_time.py`), never
  guessed (TRAINING_NEEDS 3).
- **Observation:** legal mode on for every result that counts. A privileged run is labelled an ablation.
- **The body (one per scope; locomotion first).** Each tick, for agent i:

      c_t   = Trunk(tokens_t)                    once per tick for all agents (freezing.encode_contexts)
      x_t   = [c_t ; a]                          a = the request's resolved arguments (target vector, pace, radius)
      h_t   = GRU_v(Dense_v(x_t), h_{t-1})       h reset to 0 on the tick the request is granted
      pi_v  = softmax(W_h h_t) for each head h of the verb's scope;  V_v = w_V h_t
      status = sigmoid(W_s h_t): done, stuck, P(success)   (trained in C7)

  The GRU and the heads belong to the verb's adapter, so freezing's rules hold: compiled per verb, never stacked,
  trunk frozen before the body grows. C9 re-proves the bit-identity with recurrent adapters. The trunk is
  encoder.py at its B1 sizes (width 64, two layers, context 128).
- **Algorithm:** PPO (Schulman et al. 2017) with GAE (Schulman et al. 2016) over the verb's factored heads:

      L(theta) = E_t[ min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t) ]      r_t = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
      log pi(a_t|s_t) = sum over the verb's heads h of log pi_h(a_{t,h}|s_t)    (other heads are neutral, not sampled)
      A_t = sum_l (gamma lambda)^l delta_{t+l}                         delta_t = r_t + gamma V(s_{t+1}) - V(s_t)

- **Reward:** the catalogue's done rule, plus potential-based shaping, which leaves the optimal policy unchanged
  (Ng, Harada and Russell 1999):

      F(s, s') = gamma Phi(s') - Phi(s)        walk_to: Phi(s) = - d(s) / d_0   (d: distance to the point)

  It keeps the facing, sight and aim terms in every locomotion template (TRAINING_NEEDS 6). Idle costs; means are
  never paid (BUILD.md 3.4).
- **Seeds and decisions:**
  - selection on development seed 66001 and reports on 77001 (BUILD.md 3.4);
  - at least 4 development seeds and 2 training seeds, with the seed spread reported (TRAINING_NEEDS 4 and 5;
    Henderson et al. 2018);
  - a recipe or arm is adopted when its mean effect exceeds 2 SE and no skill falls below -2 SE;
  - intervals are reported, not only means (Agarwal et al. 2021).
- **Every run:**
  - a smoke first, which must reach a success branch (TRAINING_NEEDS 2);
  - then a bounded run, wall = U x L x T / S (U updates, L lanes, T steps per update, S measured lane-steps per
    second);
  - stop at the budget, or when the development success has not improved over 4 evaluations;
  - a ledger entry with the command and numbers;
  - findings into TRAINING_NEEDS.md;
  - policies are throwaway unless a stage says otherwise.

## 2b. How the training is made to work (the method)

Running PPO on a skill is what failed before. The reach, hop and detour runs of 2026-10-17 learned little, and one
bad recipe destroyed a working policy. Each rule below answers one of those findings (TRAINING_NEEDS item numbers in
brackets).

1. **Imitate first, then reinforce.**
   - *Teacher A, allowed now: native NPCs.* The 32 traces are fights, but each holds native pursuit with the server's
     own pathfinding. More are recorded on varied worlds through the bridge; that is recording, not training.
   - *Teacher B, which needs the owner (the ruling says native NPC traces only): a scripted planner in the simulator
     that sees the whole map* (the path tables exist). It gives unlimited walk_to and go_around demonstrations.
     - The student learns from the legal observation only.
     - DAgger relabels the states the student itself visits (Ross et al. 2011). The middle layer's P1d used the same
       loop.
     - This privileged-teacher pattern is "learning by cheating" (Chen et al. 2019).
   - *Then PPO starts from the cloned policy,* with a decaying pull back toward it, so one bad update cannot erase
     the start (Rajeswaran et al. 2018):

         L(theta) = L_PPO(theta) + beta_t KL(pi_clone || pi_theta),     beta_t -> 0 over training

2. **Check each piece in minutes before any long run** [2]:
   - a scripted controller succeeds on the task, so the environment and the reward are right;
   - the learner reaches about 100 % on 8 fixed openings, so the trainer is right;
   - a smoke on the real distribution reaches a success branch.
3. **Make the task learnable by construction:**
   - one verb per adapter, each starting from its prerequisite: walk_to, then step_and_hop, then go_around [9];
   - a success-gated curriculum (Narvekar et al. 2020). The goal distance grows, and for go_around the ratio of
     path length to straight-line distance grows, only when development success on the current level passes
     70 % [10];
   - shaping that keeps the facing, sight and aim terms [6];
   - horizons from measured time to success [3];
   - wasted episodes removed first: openings filtered to terrain the simulator supports [7], and lessons aimed at
     the terrain where agents stall [8].
4. **Guard every run so a failure shows within minutes.**
   - Logged every update: entropy, approximate KL, clip fraction, value explained variance, and success on fixed
     development openings.
   - An alarm (entropy collapse, a KL spike, success falling over 3 evaluations) stops the run and keeps the best
     checkpoint.
   - PPO's standard stabilisers: normalised advantages and observations, gradient clipping, early stopping on a
     target KL, annealed learning rate and entropy (Engstrom et al. 2020; Andrychowicz et al. 2021).
5. **Measure enough to decide** [4, 5]:
   - 256 or more episodes per evaluation, 4 development seeds, 2 training seeds;
   - effects judged against 2 SE: at 16 updates and one seed, effects of 0.01 to 0.03 drowned in SEs of 0.017
     to 0.027.
6. **Throughput decides how much of this is affordable.** C1 measures lane-steps per second. If that is too low,
   the environment step is profiled and sped up, with bit-exact changes only, before any long run, rather than
   running longer.
7. **One change per run, and short runs:** one axis per experiment, a hypothesis and stop rule stated before it
   runs, and a ledger entry with its numbers.

## 3. Build before any run (stage 1; no training)

| Step | What | Gate |
|---|---|---|
| S1 | The observation adapter: the render's in-graph state to the skill tokens (entity, geometry, self, kit; block and inventory later), batched in JAX, from the legal observation only. Starts from `hytalegym/jax/training/token_encoder.py`'s slicing of the dense policy row. | equal to the host path (`encoder.pack_tokens`) on recorded states; the three-world leakage test on skill inputs passes (E12's legality half, ledger row 14) |
| S2 | The task environment: walk_to on flat with `npc_walk`, and the pursuit baseline task on flat, both on the render. Certified today on `npc_walk`: the flat walk (1.1e-5 m), the ledge walk (1.9e-5 m), and on the render jump in place, jump held and jump held with forward (<= 1.5e-5 m). Single jump then forward still fails | a scripted controller reaches the success branch; horizons from measured time to success |
| S3 | The trainer (BUILD.md 1: `training/recipe.py`): PPO over a body (section 2), in JAX in-graph, and the same body with a PyTorch learner inside the JAX loop for E11 | gradients and masking tests; a 1-update smoke each way |
| S4 | Evaluation: development and report suites, the capability-card fields (BUILD.md 3.4.5) | suites reproducible from their seeds |

## 4. The experiments, in order

| Run | Proof | What trains | Needs | Ask |
|---|---|---|---|---|
| C1 | E11 | the same walk_to body, JAX vs PyTorch-in-loop, about 15 min each, throughput only | S1-S3 | stage 1 |
| C2 | E2 | pursuit on flat: set-encoder body vs flat-MLP body, same recipe, both fresh on gen 7 | C1 | stage 2, with C1's numbers |
| C3 | E10 | walk_to / pursuit warm-started from the native-trace dataset vs from scratch | C2 | stage 2 |
| C4 | E3 | walk_to, then step_and_hop, then go_around in one body | the simulator's walk resume after landing (step_and_hop) and the Region library recaptured on 0.7.2 (go_around) | stage 3 |
| C5 | E4 | typed pace and radius on walk_to | C2 | stage 3 |
| C6 | E5 | re-goals mid-episode | C2 | stage 3 |
| C7 | E6 | the status heads (done, stuck, P(success)); ECE = sum_m (n_m / n) abs(acc_m - conf_m) (Guo et al. 2017) | C2 | stage 3 |
| C8 | E12 | every passed body at FOV16 and FOV32 with occlusion | S1 | stage 3 |
| C9 | E13 | freeze walk_to, add step_and_hop, check bit-identity and the suites | C4 | stage 3 |

## 5. What it waits on

- **The simulator (peer session):**
  - the walk resume after landing (C4's step_and_hop);
  - the Region library recaptured on 0.7.2 (go_around and the terrain suites);
  - a slope and stairs fixture for `npc_walk`;
  - enough geometry-sweep capacity, or openings filtered to supported terrain (TRAINING_NEEDS 7).
- **The owner:** approval of stage 1 now, then of each stage's budget when its evidence is in.
- **This session:** S1 to S4.

## 6. Risks

- CPU throughput may make C2 to C4 slow; C1 measures it before any budget is asked for.
- Other projects load this machine (load 20 to 59 on 2026-10-18), which inflates timings: latency and
  throughput numbers are taken in quiet windows, with the load recorded.
- The NPC model cannot verify player-only movement, so those verbs stay untrained or scripted.
- Cloning has 32 traces on two worlds (README 2.8): H6 may fail on data, not on method. More worlds are recorded
  first if it does.

## References

- Agarwal, R., et al. (2021). Deep reinforcement learning at the edge of the statistical precipice. NeurIPS.
  arXiv:2108.13264.
- Andrychowicz, M., et al. (2021). What matters for on-policy deep actor-critic methods? A large-scale study. ICLR.
  arXiv:2006.05990.
- Chen, D., Zhou, B., Koltun, V., and Krähenbühl, P. (2019). Learning by cheating. CoRL. arXiv:1912.12294.
- Engstrom, L., et al. (2020). Implementation matters in deep policy gradients: a case study on PPO and TRPO. ICLR.
  arXiv:2005.12729.
- Narvekar, S., et al. (2020). Curriculum learning for reinforcement learning domains: a framework and survey.
  JMLR 21. arXiv:2003.04960.
- Rajeswaran, A., et al. (2018). Learning complex dexterous manipulation with deep reinforcement learning and
  demonstrations. RSS. arXiv:1709.10087.
- Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine
  translation (the GRU). arXiv:1406.1078.
- Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML.
  arXiv:1706.04599.
- Henderson, P., et al. (2018). Deep reinforcement learning that matters. AAAI. arXiv:1709.06560.
- Lee, J., et al. (2019). Set Transformer. ICML. arXiv:1810.00825.
- Ng, A. Y., Harada, D., and Russell, S. (1999). Policy invariance under reward transformations: theory and
  application to reward shaping. ICML, 278-287.
- Ross, S., Gordon, G., and Bagnell, D. (2011). A reduction of imitation learning and structured prediction to
  no-regret online learning (DAgger). AISTATS. arXiv:1011.0686.
- Schulman, J., et al. (2016). High-dimensional continuous control using generalized advantage estimation. ICLR.
  arXiv:1506.02438.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
- Sutton, R. S., Precup, D., and Singh, S. (1999). Between MDPs and semi-MDPs: a framework for temporal
  abstraction in reinforcement learning (options). Artificial Intelligence 112, 181-211.
  doi:10.1016/S0004-3702(99)00052-1.
