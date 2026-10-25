# Phase C: the plan for training the first skill bodies (2026-10-18, late; approved in stages)

**Status (2026-10-18, late).** The owner approved the plan in stages: the simple track (sections 2c to 2e, "you have
the green light"), then A2, the first student training ("ok continue"). The architecture moved away from pure PPO
with per-verb adapters, toward one generalist, request-conditioned policy ("similar to a vla model"):
`docs/ARCHITECTURE_RESEARCH.md` recommends it, trained by teacher distillation (DAgger) and KL-anchored RL, and the
owner's answers made it the working design (docs/BUILD.md section 0). The method (section 2b), the staging, the
gates and the dependencies stand. 2026-10-19: A2 passes with run C' (README 2.16). Each skill's own plan is now in its
folder, `neural_skills/<skill>/TRAINING.md`; this file keeps the shared method, the hypotheses and the staging. Owner,
2026-10-19: each skill trains its own module by its own strategy (`neural_skills/<skill>/train.py`) on a shared frozen
trunk, and modules read the requests of the skills they trained beside (README 2.17); the jointly trained generalist of
A2 to C' is the reference those modules are measured against.

Nothing in this plan runs until the owner approves it (docs/BUILD.md section 0). It answers the owner's condition for
training: what trains, on what, for how long, what counts as success, what happens on failure, and what each step
waits on. Approval is staged, so compute is never spent ahead of evidence:
- **Stage 1** asks for steps S1 to S4 (building, no training) and C1 (a throughput measurement whose policies are
  thrown away).
- **Stage 2** comes back with C1's measured numbers and asks for the budget of C2 and C3.
- Later stages are asked for the same way.

## 1. The question and the hypotheses

**Question.** Does one recipe train competent, single-competency skill bodies on the gen-7 NPC model, from
native-trace warm starts, inside the 30 Hz budget? And can the library grow without changing a frozen skill?

| # | Hypothesis | Proof (docs/BUILD.md 6) | Pass bar | If it fails (docs/BUILD.md 6) |
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
| H10 | The student is smarter than the heuristic: it beats the scripted teachers on combined requests, disturbances and unseen worlds | section 1b | higher success, or equal success at lower cost, by 2 SE | the heuristic ships for that skill |

## 1b. Why neural at all: the bar a neural skill must clear (owner, 2026-10-18)

The owner: "the goal of having these neural agents instead of normal heuristic actions is that it can combine these
common behaviors to be more complex, otherwise we can just use a simple heuristic; the goal of the low level neural is
that it is smarter in a sense."

So:
- **Teachers are scaffolding.** A scripted teacher gives a fast, correct start. A student that only matches its
  script has not justified itself, because the script could ship instead.
- **H10, the bar.** On suites the scripts were not built for, the student beats the heuristic: the scripted
  teachers run directly through the combiner. The suites:
  - (a) combined requests, several at once, where scripts ignore or fight each other: walk while looking while
    avoiding a hazard; strafe while aiming; chase while dodging;
  - (b) disturbances: a target that moves or hides, an interruption, a push, an obstacle appearing mid-path, an enemy
    arriving;
  - (c) unseen worlds and openings;
  - (d) cost: time to success, damage taken, stalls.
- **Pass (confirmed by the owner, "beat it on hard cases"):** higher success, or equal success at a lower cost, on
  (a) to (c), separated by 2 SE. Matching the script on easy solo tasks is fine. A skill whose student
  does not beat its heuristic ships the heuristic, and the finding goes to TRAINING_NEEDS.
- **Where the gain comes from:**
  - one network learning every skill together;
  - RL practice on combined and disturbed tasks (stage 3), where the student must get ahead of its teachers;
  - variety of worlds and requests.

  Distillation only gets the student started. The combined and disturbed suites are built in S4.

## 2. Fixed choices

- **Simulator:**
  - the gen-7 render (HEAD's engine plus the `sim/gen7` overlay), mounted per run with
    `hytalegym.sim.common.render.redirect_imports`;
  - every result records the render's commit and overlay hash;
  - the gen-7 lane is the peer session's (GEN7-PLAN 2b): this plan consumes the render and never edits it.
- **Compute (owner, 2026-10-18: "you can use gpu if it is available use the cpu as well").** This machine is an
  Apple M5 Max: 18 CPU cores, a 40-core GPU (Metal 4) and 128 GB of unified memory.
  - JAX 0.11.2 sees the CPU only. The experimental `jax-metal` plugin is not installed, and is not to be installed in
    the shared environment without asking.
  - PyTorch 2.14 reaches the GPU through MPS: a 4096 x 4096 float32 matmul measured 14.9 TFLOP/s on the GPU against
    1.7 on the CPU, under load.
  - The simulator stays JAX on the CPU. The model is either JAX on the CPU, trained in-graph with the simulator, or
    PyTorch on the GPU, fed by the JAX simulator through the shared memory. E11 (C1) measures both on the same task
    and decides (BUILD 3.5's ruling).
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
  model/encoder.py at its B1 sizes (width 64, two layers, context 128).
- **Algorithm:** PPO (Schulman et al. 2017) with GAE (Schulman et al. 2016) over the verb's factored heads:

      L(theta) = E_t[ min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t) ]      r_t = pi_theta(a_t|s_t) / pi_old(a_t|s_t)
      log pi(a_t|s_t) = sum over the verb's heads h of log pi_h(a_{t,h}|s_t)    (other heads are neutral, not sampled)
      A_t = sum_l (gamma lambda)^l delta_{t+l}                         delta_t = r_t + gamma V(s_{t+1}) - V(s_t)

- **Reward:** the catalogue's done rule, plus potential-based shaping, which leaves the optimal policy unchanged
  (Ng, Harada and Russell 1999):

      F(s, s') = gamma Phi(s') - Phi(s)        walk_to: Phi(s) = - d(s) / d_0   (d: distance to the point)

  It keeps the facing, sight and aim terms in every locomotion template (TRAINING_NEEDS 6). Idle costs; means are
  never paid (docs/BUILD.md 3.4).
- **Seeds and decisions:**
  - selection on development seed 66001 and reports on 77001 (docs/BUILD.md 3.4);
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

## 2c. How a teacher is shown to be good (the teacher certificate)

A teacher teaches only after it passes these checks, and each result is recorded in the ledger. The bars are starting
proposals, to be reset from measured data (the first measurement is the scripted planner's own T1).

| Check | What it measures | Starting bar |
|---|---|---|
| T1 Success | the skill's own done rule (the catalogue's "done" column) on unseen test worlds (report seeds), >= 256 episodes x 4 seeds, with intervals | navigation: >= 95 % of the goals the path tables say are reachable; other families set from their references |
| T2 Efficiency | time to success against a reference: the shortest-path time at the pace asked (navigation), the native NPC traces (pursuit, combat) | median <= 1.2 x the reference |
| T3 Obedience | the request's arguments honoured (pace, radius, target), only legal actions | 100 % legal; E4's bars |
| T4 Reality | a sample of its episodes replayed on the live 0.7.2 server through the bridge | live success within 5 points of simulated, end positions within tolerance |
| T5 Imitability | a small student trained on its labels, legal observation only | teacher minus student <= 10 points; otherwise restrict what the teacher sees |
| T6 Recovery | the teacher started from states the student actually visits | success >= T1's bar minus 5 points |
| T7 Inspection | a sample of episodes rendered, and failures sorted (stuck, timeout, fall, unreachable) | no failure class caused by a teacher bug |

What each check catches:
- T4 catches a teacher that wins only because the simulator is wrong (today's known gaps: the walk resume after
  landing, unsupported terrain). RL teachers exploit simulator errors.
- T5 catches a teacher that relies on information the student never has. For example, it walks straight to a target
  hidden behind a wall (the imitation gap). The fix is to show the teacher less, or to give the student the memory
  that would let it infer the same thing (the belief's last-seen position).
- T5 also catches a teacher whose decision sits exactly on the success boundary. walk_to stops the moment it is inside
  the radius, so a student a few centimetres off parks just outside and, acting by argmax on a state that no longer
  changes, never restarts (A2: 64 of 66 missed walks, a median 0.02 blocks short; README 2.16). The proposed fix,
  not yet tested, is a margin: the teacher acts well inside the bar it is judged by.
- T6 makes DAgger's labels trustworthy in the places the student actually drifts to.

Status (2026-10-18, late; `teaching/certificates.py`, README 2.15):
- walk_to v1 and look_at v1 pass T1, T2 and T3 on flat `npc_walk`: 1,024 of 1,024 each, walk_to at 1.0004 x the
  kinematic minimum, 0 illegal decisions.
- walk_to v1 passes T4 live: 24 of 24 on the 0.7.2 server against 24 of 24 simulated, 0 illegal
  (`teaching/certify_live.py`). look_at's T4 is not run yet. T5 and T6 need a student.
- T5 for v1 fails by measurement: 28 % of look_at's yaw labels depend on the commanded head yaw, which the student
  cannot see, and its student completed no look. v2 reads the observed state and passes T1 to T3 (walk_to 1.037 x
  the minimum, look_at median 13 ticks, 0 illegal).
- T5 for v2 fails in A2 (README 2.16): the teachers lead the student by 16.3 points on walk, 12.9 on look and 23.5 on
  both. The labels no longer depend on hidden state (v2 reads only observable fields); the misses sit at the two
  decision boundaries instead. Missed walks park a median 0.02 blocks outside the radius, and missed looks end a
  median 5.2 degrees off against a 3 degree tolerance. The next changes, one at a time: walk_to v3 with a stopping
  margin, and a look request that carries the direction to its target.
- Runs W and L (README 2.16): with walk_to v3, walk_to's T5 gap is 0 points, and with the look direction, look_at's
  is 9.4 (a pass, at the edge of the bar). The margin is now the measured fix for a teacher that decides on the
  success boundary.
- Runs C and C' (README 2.16, 2026-10-19): with both changes and the look direction measured in the head's frame
  (C'), every T5 passes: walk 0, look 1.2 and both 2.4 points, and 1.6 and 1.2 on 256 fresh lanes per mode. L's
  look-only pass did not hold on those lanes (12.1 points), so a bar decision near its edge needs more lanes. The head
  frame is the measured fix for a request input that leaves the trained range: a held-out look's direction stays out
  of range in the body frame and comes back into it in the head frame (held-out look 0.576 to 1.000, held).

A teacher need not be perfect, because stage 3's practice can go past it. But a systematic teacher error is copied
into the student, which is why nothing teaches before its certificate.

## 2d. How the teachers are configured

**What a teacher is.** A pure JAX function, `teacher(full_state, request) -> choices on the 21 heads`, filling only the
request's scope heads, plus its done signal. It runs inside the simulator for every lane at once, so it can label
millions of ticks, including the states the student visits (DAgger).

Each teacher is one entry in a registry, as skills are in `runtime/library.py`. The entry records:
- the verbs it teaches;
- its kind (scripted, trained or native);
- the privileged information it reads;
- its parameters;
- its version and its certificate (section 2c).

**Three kinds, by skill family.**

| Kind | Used where | How it decides |
|---|---|---|
| Scripted planner and controller | the game's rules give an exact answer: navigation (walk_to, run_to, sneak_to, step_and_hop, go_around, come_here); gaze (look_at, track); ranged aim and draw; world interaction (mine, place, use) | navigation: a shortest path over the native traversal graph (`worldgen/region/traversal.py`, every edge certified by the live server's NPC walk controller), then each tick steer at a waypoint ahead, pick the nearest compass direction, take the gait from the request's pace, jump at step-up edges (timing from the measured NPC jump arc), and stop inside the radius. Gaze: exact yaw and pitch from true positions, with lead. Ranged: a ballistic solution from the kit's launch speed and gravity, held for the tier asked. Interaction: move into reach, aim at the marked face, hold for the break time |
| Trained in the simulator's cheat mode | no exact rule: fighting decisions (spacing, when to attack, guard and dodge timing), pursuit of an evader, evasion | PPO with `legal_observation=False`, the simulator's documented privileged mode "for a Learning by Cheating teacher" (`jax/combat/arsenal/schema/types.py`): exact opponent state and wind-up timers, dense rewards, one skill each, with the recipe-bench lessons |
| Native NPC recordings | how real game creatures move and fight | the reference for T2 and a warm start; not the main teacher, because an NPC is not optimal at our requests |

**Settings, and how they are chosen.**
- *What it may see:* the full state by default. It is restricted when T5 shows the student cannot follow: search_for's
  teacher may not know a hidden target's position, so it scans from the last-seen position instead. Applied
  2026-10-18: walk_to and look_at v2 read the actual head and body yaw, not the commanded ones (README 2.11).
- *The request:* it reads the same request as the student (target, pace, radius, entity), so every label belongs to
  one request.
- *Parameters* (waypoint distance, jump trigger, aim lead, attack spacing): tuned on development worlds only, judged by
  T1 and T2, never on the report worlds.
- *Variety for data:* small random action noise and parameter jitter while generating data (Laskey et al. 2017,
  DART), so the data shows recoveries and more than one valid style.
- *Versions:* changing a teacher makes a new version that needs a new certificate. Both go in the ledger.

**Order.**
1. walk_to on flat (a straight line, then stop inside the radius) and look_at: proof A2's two teachers.
2. step_and_hop and go_around: these wait for the traversal graph rebuilt on 0.7.2 (today's graphs come from the 0.5.7
   Region captures) and for the walk resume after landing.
3. Pursuit: the planner plus target prediction, then RL against the scripted evader.
4. The combat teachers, after the simulator's combat rows pass.

## 2e. Complex actions: what changes, and what we watch for (owner, 2026-10-18: "for more complex actions we need to be aware")

The scripted-teacher recipe fits simple actions (navigation, gaze, aim), where the rules give an exact answer. Complex
actions break that assumption in the ways below. Each complex family gets its own teacher plan and certificate
extension before it trains.

| What gets harder | Example skills | How it is handled | Added check |
|---|---|---|---|
| No exact teacher exists: the right answer depends on an adaptive opponent | melee and ranged duels, guard and dodge, evade, group fights | trained teachers in cheat mode against a pool of opponents (scripted NPC AI, earlier checkpoints, clones of native behaviour), not one fixed opponent, which a teacher learns to exploit | win and damage rates against held-out opponents, including native NPC AI on the live server |
| Several valid strategies (left or right flank, different combos); a student copying two of them can average into a bad middle action | combat, go_around around wide obstacles, hide | one teacher strategy per episode, history in the policy's input, and a strategy latent or action chunks if averaging shows up (MaskedMimic's latent, ACT's chunks) | per-head and per-episode consistency of the student's choices |
| Timing on cues: the teacher knows the exact hit tick, a player sees only the wind-up | guard, dodge, parry, punish, combo windows | the teacher reads only cues a player could see (the enemy's visible state), or the cue is added to the legal tokens | T5 on timing; requires the simulator's combat timing rows to pass (row 5f) |
| Hidden information and memory: the target is out of sight | search_for, track, hide, ambush | the student gets belief inputs (last-seen positions, from the shared belief) and attention history; the teacher may use only the same information | T5 under FOV16 and FOV32 |
| Long horizons: minutes, many steps | build a structure, crafting chains, mine-then-place | not learned as one skill. Activities stay the middle layer's programs over one-dimensional skills (owner ruling), and each learned skill stays seconds long | each program step's skill certified on its own |
| Several body parts at once (move, aim, attack) | a melee approach, strafing fire | one generalist conditioned on all granted requests outputs every scope in one pass, so coordination is learned and the combiner still owns the grants | the composition checks of proof A2, extended per family |
| Other agents' behaviour changes over time | squads, cooperation, PvP-like play | local reaction at the low level; coordination stays in the middle layer (squads and addressees in the executive contract) | evaluation against opponent pools, not one script |
| Lower simulator fidelity for complex mechanics | combat timing (5f), ranged hits, crafting tiers, runes (absent) | a family trains only when its fidelity rows pass (E14), and the live replay T4 is mandatory | T4 with a larger sample |
| Reward hacking: a degenerate winning trick | corner camping, an AI pathing bug | T7 inspection, live replay, and adversarial evaluation against new opponents | T7 by family |

The order follows from this. First, simple skills with scripted teachers, which prove the whole pipeline. Next,
reactive skills with trained teachers against simple opponents (pursuit, evade). Last, combat, with opponent pools,
after its fidelity rows pass.

## 3. Build before any run (stage 1; no training)

| Step | What | Gate |
|---|---|---|
| S1 | The observation adapter: the render's in-graph state to the skill tokens (entity, geometry, self, kit; block and inventory later), batched in JAX, from the legal observation only. Starts from `hytalegym/jax/training/token_encoder.py`'s slicing of the dense policy row. | equal to the host path (`encoder.pack_tokens`) on recorded states; the three-world leakage test on skill inputs passes (E12's legality half, ledger row 14) |
| S2 | The task environment: walk_to on flat with `npc_walk`, and the pursuit baseline task on flat, both on the render. Certified today on `npc_walk`: the flat walk (1.1e-5 m), the ledge walk (1.9e-5 m), and on the render jump in place, jump held and jump held with forward (<= 1.5e-5 m). Single jump then forward still fails | a scripted controller reaches the success branch; horizons from measured time to success |
| S3 | The trainer (docs/BUILD.md 1: `training/recipe.py`): PPO over a body (section 2), in JAX in-graph, and the same body with a PyTorch learner inside the JAX loop for E11 | gradients and masking tests; a 1-update smoke each way |
| S4 | Evaluation: development and report suites, the capability-card fields (docs/BUILD.md 3.4.5) | suites reproducible from their seeds |

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

## 7. Stage L: combat skill through a league (owner, 2026-10-20; L4 and L5 wait on the owner's yes)

The owner, after watching the referee's replays in the Console: "these are very simple training strategies which are a
good proof of concept but it needs to be a lot more complex than this". Then: start with combat skill through a
league, and reopen training now, with the richer recipe.

**Goal.** Combat skills that show skill against varied opponents: reading the opponent, choosing among the weapon's
moves, spacing, adapting. They are measured against a league of opponents rather than one scripted opponent, and
they move naturally. The single-rule scripted bodies of 2026-10-19 are the floor, not the target.

| Step | What | Gate |
|---|---|---|
| L1 | A two-actor harness: both entities under policy control in one arena, through the gym's multi-actor path (a policy-controlled entity mask, per-actor observation and decoding). The skill layer (the referee and its bodies or modules) drives the agent; a league member drives the opponent. Both perceive legally. | A scripted body against the production opponent through the harness reproduces today's numbers within noise. |
| L2 | A league of scripted styles: counter-puncher, rusher, kiter and turtler, from the style knobs (`IDEAS.md` idea 2), plus the production opponent; Elo ratings (`arena/training/selfplay/league.py`). | The styles beat each other in a cycle, not in a line: no single style dominates. |
| L3 | A lookahead teacher (privileged). Every K ticks it forks the simulator's state and rolls each candidate move H ticks forward against the opponent's policy, all candidates at once: each ability slot, charge, guard, a step in, out or aside, wait. It takes the best by damage dealt minus damage taken plus a spacing term. | It beats every scripted style by a clear margin (2 SE), or it is not a teacher. |
| L4 | The combat cohort distilled from the teacher by DAgger on the student's own states. Modules on the frozen trunk (v2 tokens, the perception channel on gen 7) for sword (its move choice across the slots, and charge), guard, punish, keep_range, evader and avoid; the style scalars as inputs. | The student keeps the teacher's win rate against the league within 2 SE, on legal perception. |
| L5 | PPO from the distilled cohort, with a decaying pull back to it (section 2b), against the league: the scripted styles, the production opponent, and frozen snapshots of the student added as it improves, under rating-aware matchmaking. A motion prior from the native traces keeps it natural (AMP, Peng et al. 2021). | Its rating against the league rises over snapshots; it wins on held-out styles; it does not collapse (the section 2b guards). |
| L6 | The game's own NPC combat: the native fight traces cloned into a behaviour policy (`arena/training/imitation/native_replay.py`, `install_native_teacher_policy`), as a league opponent and the source of the motion prior. | The cloned policy reproduces the traces' move statistics. |

**L1 status (2026-10-20).** The coordinator built the two-actor arena (`experimental/neural_agent/tasks/duel_arena.py`,
bc69536; equal maximum health at 5871dee). The duel has no built-in AI, so the gate is restated with the coordinator:
the scripted priority on both bodies, from placements symmetric under a half turn, each entity after its own random
start delay of 0 to 20 ticks, with wins and damage at parity within 2 SE (`tasks/duel.py`). Without the delays the
twins swing on the same tick, each guards the other's wind-up, and nothing lands. Measured over 64 lanes and 600
ticks: on gen 7 it fails. With swords entity 0 wins 51 of 63 decided bouts (damage +5.9 ± 2.1), and with maces
entity 1 wins 9 of 9 (−4.5 ± 1.7). Gen 5's priority mirror held parity only because 55 to 64 of 64 bouts ended
undecided; with decisive mirrors gen 5 fails too: PRESSURE on both bodies, entity 0 wins 47 to 3 (damage +25.2 ±
2.4), and HIT_AND_RUN on both, 46 to 18 (+17.6 ± 2.9). In fixed probes entity 1's follow-up attacks re-queue on
different ticks than entity 0's (first hits coincide, follow-ups differ in both directions). Movement, turning and
perception are symmetric: both entities read the other's distance within 0.065 blocks and its bearing within 1.1
degrees. The coordinator is tracing the cause (#103). Until the seats are equal, the league plays every pair in both
seats. Two findings for the bodies: the strike reach is the sword's 3.5
blocks for every weapon, but daggers land only within 2.0, so dagger bodies never hit; and against a guard on every
wind-up the slow battleaxe and mace almost never land.

**L2 status (2026-10-20).** The coordinator's style programs landed at 0b22cda. Their controllers are
`tasks/league.py`: COUNTER, PRESSURE and HIT_AND_RUN, with the scripted priority as PRIORITY, closing in by
keep_range with the pursuit body's search turn, and no charged holds yet. The round robin plays 32 placements per
ordered pair in both seats, 768 bouts, as one batch of 384 lanes in about 25 s. Its gate cannot be judged while the
seats are unequal: seat 0 scores 0.91 on gen 5 and 0.81 on gen 7, so a pair played in both seats scores near one half
whatever the styles. On gen 5 no style beats another by 2 SE. On gen 7 PRESSURE and PRIORITY each beat HIT_AND_RUN
(0.66 ± 0.06), there is no cycle, and no style dominates. It is re-run when #103 lands.

**L3 status (2026-10-20).** The lookahead teacher is `teaching/lookahead.py`; its CLI reproduces the scratchpad
prototype's measurement exactly on gen 5 and gen 7. On gen 7 it meets L3's gate against the opponents that exist
today, the production opponent with four weapons; the scripted styles of L2 are not built yet. The teacher forks the
simulator every 4 ticks and rolls 10 candidate moves 24 ticks ahead: wait, strike with each of the sword's 4
abilities, guard, step in, out, left or right; head and body face the target. On gen 7, 16 lanes, opponents at 4
times their health, 600 ticks, it won 64 of 64 fights against the production opponent with four weapons (sword,
battleaxe, mace, daggers) and never died. The scripted priority over the skill layer's bodies won 31 and died 32
times; it is P_COMBAT's rules without P_COMBAT's evade gate, and P_COMBAT itself has not run on this setup. The teacher mostly strikes with the sword's third ability, and against the sword and the mace it steps out of
reach between strikes. A plan costs about 0.1 s for 16 lanes. On gen 5 the teacher had 49 kills, 0 deaths and 15
time-outs, all 15 against the sword; the scripted priority had 16 kills and 32 deaths.
Console replays of the mace fights: 20261020T062617Z-11750ee7 (scripted), 20261020T062640Z-e8823f3b (teacher).

**Bars, every run.**
- Win rate, and damage dealt over damage taken, against each league member, held-out styles included.
- Skill, not damage alone (`IDEAS.md` idea 10): hits per swing, punishes and blocks landed, the variety of moves used
  (their entropy), the spacing held.
- Legality: the view sector, occlusion, the sensor range.
- Naturalness against the native traces.
- A comparison with the scripted bodies and with the lookahead teacher.

**Order and asks.**
- L1 to L3 are build and measurement, with no learning and no budget: they start now.
- L4 and L5 are the training runs. Each starts on the owner's yes to its budget, after a smoke that reaches a success
  branch (section 2b, rule 2).
- L6 waits on an inventory of the native traces (`agent_skills/training_runs/traces_072`,
  `HytaleRL/artifacts/native-trace-*`).

**Risks.**
- The multi-actor arena is the gym's. Plugging the skill layer in as one actor's controller may need a seam from the
  simulator side.
- The local geometry window (9 blocks) keeps this stage to melee; ranged weapons and cover wait on the Region route.
- The lookahead teacher multiplies the simulation cost by candidates times horizon. It runs every K ticks, and its cost
  is measured first.
- A self-play league can cycle, or collapse onto one strategy (Balduzzi et al. 2019). The scripted styles stay in the
  pool as anchors, as AlphaStar kept its exploiters (Vinyals et al. 2019).

## 8. Training on v3: the state layer and the trunk's memory (PROPOSED 2026-10-23; V0 APPROVED the same day, 8.4; the sequence machinery BUILT, 8.7; V1 built, smoke and short pass run, 8.8 and 8.9; V1 PASSES, 8.10; V2 run, 8.11; the longer look_at schedule and come_here's runs, 8.12)

`VIRTUAL_HUMAN.md` section 13: "Moving the trainers onto `runtime/state_tick.py` is the first task when training
resumes." The encoder and the state layer are now built through step 9 (with the tick fix and S5), so this section says
how the move is made. It changes no plan above; it is the input side under all of them, and stage L4 too (its "v2
tokens").

### 8.1 What exists (checked 2026-10-23)

- **The trainers read v1 tokens and no memory.** `training/blocks.py` `token_kernel` builds v1 `legal_tokens` at the
  `bench_tick` capacities; `training/skill_dagger.py` `rollout` and `training/distill.py` step the flat world in a host
  loop, record those tokens per decision into one aggregate dataset, and fit on it. Their trunk has
  `memory_slots=0`: each decision stands alone.
- **The v3 trunk is recurrent.** It carries 8 latent memory slots, a 16-slot background summary and an object latent
  per entity and block, tick to tick (`VIRTUAL_HUMAN.md` 18), and the state layer carries the stores, the places, the
  event log, the written entries and the social records per lane (`state/layer.AgentState`). `state_tick.step_lanes`
  runs both for B lanes, with per-lane `reset` and `respawn`.
- **v3 tokens are 27 times larger:** 42,171 floats a decision (165 KB) against v1's 1,545 (6 KB), at the default
  capacities; geometry is 56% of it, the inventory 19%. A2's first run aggregated 123,398 samples: about 20 GB as v3
  tokens, 0.7 GB as v1. Most slots should be empty on the flat training world (no blocks, little terrain, a
  near-empty inventory; inferred, V0 measures it).
- **A skill module reads every token's trunk embedding** (`apply_generalist` concatenates `encoder_output.tokens`): 356
  x 64 floats and the context, so storing the trunk's outputs instead of the tokens saves only about half.
- **The cost of a v3 tick is measured** (E8; section 16.3's per-agent run, in progress): a few milliseconds per lane.

### 8.2 The design

1. **The kernel.** `state_tick.step_lanes` replaces `token_kernel` and the trunk call in every rollout: the
   `AgentState` joins the rollout's carry, an episode start is a `reset`, and `tick_seconds` is the world's (27.7).
   Teachers and requests are unchanged; the student reads v3.
2. **Two kinds of training, two kinds of data.**
   - **The trunk** (once, as the A2 reference was): the memory must learn what to keep, so its gradients flow through
     time. Rollouts are cut into sequences of L decisions within an episode, stored with the memory at their first
     decision; training unrolls the trunk over each, the first B decisions only rebuilding the memory (no loss), then
     the loss on the rest: R2D2's stored state with burn-in (Kapturowski et al. 2019; it stored 80-step sequences, with
     burn-in of 20 or 40). L and B are set in V0 from the tasks' horizons.
     What the trunk carries is three things (`runtime/state_tick.tick`): the latent memory, the background summary,
     and each remembered object's latent, held in the entity and block stores and read and written through each
     token row's store slot. A new object's latent starts at zero (`state/memory.py`). So a sequence stores, at its
     first decision, the memory, the background and the two stores' latents; and per decision the tokens, each
     row's store slot and which slots took a new object. The unroll then repeats the rollout's reads and writes
     exactly, with the trunk's current parameters; the state layer's rules have none and need no re-run.
   - **A skill's module on the frozen trunk** (the owner's 2026-10-19 ruling): the trunk does not change, so the memory
     is a fixed function of the history and the module needs no gradient through time. Its dataset stays per decision,
     as today; the trunk runs in the rollout and the module trains on its outputs.
3. **The data's size.** Tokens are stored as their valid rows only, in float16, per decision; the aggregate dataset of
   DAgger becomes a bounded buffer when that does not fit (R2D2's replay, sized in memory, newest kept). V0 measures the
   real size on each training world before anything is chosen.
4. **The teacher sees more than the student.** On v1 both lived in the present; on v3 the student remembers, but a
   privileged teacher can still ask for what the student cannot know (a target behind a wall it never saw). Known to
   make plain DAgger fail (Warrington et al. 2021; Messikommer et al. 2025). Measured, not assumed: each skill's
   teacher-student gap, split by whether the target was ever perceived.
5. **The training hooks of `VIRTUAL_HUMAN.md` 11.8,** built with the trunk:
   - an auxiliary head that predicts hidden entities' positions from the memory, supervised by the simulator during
     training only (auxiliary tasks speed representation learning: Jaderberg et al. 2017);
   - full-state tokens that only the critic reads in the RL stage, the actor staying legal (asymmetric actor-critic,
     Pinto et al. 2018);
   - input noise, latency and dropout randomisation (`PERCEPTION.md` N5).
6. **One model per agent** (16.3): training shares one set of weights across lanes, an optimisation; each deployed
   agent gets its own copy, and agents with different trunks need their own modules (14.5).

### 8.3 The stages (each asked for in turn, as in section 0)

| Stage | What | Gate |
|---|---|---|
| V0 | No training. The v3 rollout on the flat world and the Region: decisions per second per core, the `AgentState` bytes per lane, the stored size of a decision (valid rows, float16) on each world; the sequence length from the tasks' horizons | the numbers, and a budget proposal from them |
| V1 | The v3 reference: walk_to and look_at distilled into a fresh v3 trunk and generalist with memory (A2's task and evaluation), plus the same run without memory as the control | at least run C''s rates on A2's evaluation; the memory run not worse than its control |
| V2 | The skills' modules re-trained on the frozen v3 trunk by their own strategies: walk_to and look_at have trainers today, the other five folders are plans | each skill's own gate, against its v1 module |
| V3 | Stage L4 and L5 on v3 | section 7's gates |

### 8.4 Decisions for the owner (RULED 2026-10-23: "continue", taken as the three recommendations)

The ruling: V0 now (measurements, no training); V1, when it is asked for, trains the trunk with its memory from the
start, with the memory-less control; the data's format follows V0's sizes. V1 itself is asked for with V0's numbers.


1. **Start V0 now** (measurements only, no training)? Recommended: yes. It costs an afternoon of CPU and turns the
   open sizes above into numbers.
2. **The trunk with memory from V1,** trained through time on stored sequences, with the memory-less run as its
   control? Recommended: yes. The alternative, v3 without memory first, repeats what A2 already shows and leaves the
   hard part for later.
3. **Valid rows in float16, and a bounded buffer if needed,** instead of the full aggregate? Recommended: yes, decided
   by V0's sizes; the alternative is to store the trunk's outputs, which saves only about half.

### 8.5 What grounds it

- The code as it stands (8.1, with file and function names) and the sizes computed from the token contracts.
- Recurrent replay: Kapturowski et al. 2019 (R2D2), stored state and burn-in (read through secondary sources).
- Privileged teachers and partially observed students: Warrington et al. 2021 (asymmetric DAgger), Messikommer et al.
  2025 (student-informed teachers); the recurrent-student setup of Lee et al. 2020 (recalled; its architecture not
  checked here).
- Auxiliary tasks: Jaderberg et al. 2017 (UNREAL). Asymmetric actor-critic: Pinto et al. 2018.

### 8.6 V0 measured (2026-10-23; no training)

Two tools, both built by Haiku subagents from written specs and checked here: `bench/v3_rollout.py` (the rollout's
speed, v1 against v3, and the state's size) and `bench/v3_data_size.py` (what a decision costs to store, and the
horizons). Each ran plainly and under the gen-7 render.

**What a decision costs to store** (valid rows only, float16, one mask byte a slot; 4 agents x 300 ticks a world, the
duel 3 weapons x 300 ticks):

| World | Channel | Bytes a decision: mean (p95, max) | Of dense float32 (168,684) | Largest share |
|---|---|---|---|---|
| Flat (the trainers' world) | no | 2,805 (2,840, 2,840) | 1.7% | the act token 40% |
| Flat, render | yes | 3,026 (3,052, 3,052) | 1.8% | the act token 37% |
| Region | no | 19,483 (22,392, 23,938) | 11.6% | geometry 85% |
| Region, render | yes | 22,152 (25,004, 28,702) | 13.1% | geometry 81% |
| Duel | no | 3,034 (3,380, 3,614) | 1.8% | the act token 37% |

- **float16 costs nothing that matters:** the only values it moves by more than 1e-3 of their size are below 2^-14
  (6.1e-5, essentially zero), 0.01 to 0.02% of the values; none overflows; in the normal range the error is at most
  2^-11 (4.9e-4).
- So A2's first run (123,398 decisions) is about 0.37 GB on the flat world and 2.7 GB on Region: the whole aggregate
  fits, and no bounded buffer is needed (8.2 item 3).
- **One lane's state** is about 99 KB (the spatial stores 53 KB, the written tables 14 KB, the places 12 KB, the
  social records 11 KB): a sequence's stored starting state is a few decisions' worth.

**The horizons:** walk_to's (`WalkToTasks.sample`, 4,096 draws) p50 162 ticks (5.4 s), p95 428 (14.3 s), max 480
(16.0 s); look_at's 90 ticks (3.0 s).

**The rollout's speed** (a decision tick on the trainers' flat world: the environment, then the path; p50 over 90
ticks a row; two rounds, plainly and under the render; load 15 to 23 on 18 cores, so these are a shared machine's
numbers): v1 takes about 10 ms a tick at batch 64, 15 to 17 at 128 and 25 to 30 at 256 (5,900 to 10,400 decisions a
second); v3 takes 29 to 33, 50 to 55 and 92 to 95 ms (1,950 to 2,800 a second), 2.8 to 3.8 times v1. The v3 trunk is
most of it (about 67 of the 92 ms at batch 256, taken as the tick less the state layer alone, so approximate; the state
layer 12 to 15 ms): it has 335,616 parameters against v1's 131,200 and attends over 356 tokens. The environment's step
is the same for both (2.3 to 4.9 ms).

**What it sets for V1** (the owner's ruling, 8.4: the data's format follows V0):
- **Collecting the data is cheap:** A2's 123,398 decisions take about a minute of rollout at v3's rate. V1's cost is
  training the trunk through time.
- **Storage:** valid rows in float16, the whole DAgger aggregate kept (0.37 GB at A2's scale on the flat world).
- **Sequences:** L = 90 decisions (3 s: a whole look_at episode, about a third of a long walk), with a burn-in of
  B = 30 (1 s) for a sequence that starts mid-episode; one that starts at an episode's start begins from the exact
  initial state and needs none. R2D2 used 80 with 20 or 40.
- **Training through time is not measured** (V0 trains nothing). Estimated (inferred: the trunk's forward cost a
  decision at batch 256, a backward pass about twice the forward, and an unroll that is sequential over the 90
  decisions): 1 to 2 s an Adam step on A2's 1,024 decisions (about 11 sequences), so 0.5 to 1 hour an iteration and 3
  to 7 hours for A2's six iterations on this machine. So V1 starts with a timed smoke (one iteration, a few hundred
  steps) before its full budget is asked for.

### 8.7 The sequence machinery built (G1, 2026-10-23; no training)

**What was built.** 8.2 item 2's machinery, by a Haiku subagent from a written spec, checked here:
- `training/sequences.py`, with `record`, `cut`, `step`, `unroll`, `pack` and `unpack`;
- `runtime/state_tick.trunk_step`, `tick`'s trunk half moved out unchanged, so `tick` is `layer.step` then
  `trunk_step`.

Nothing trains here.

**The test rollout** (`test_sa_sequences.py`): 12 decisions at batch 2 on the trainers' flat world, through `tick`
with a fresh v3 trunk (`bench/agent_models.CONFIG`). The cut is L = 6, and lane 1 resets at decision 4. It runs on two
channels:
- none, as the flat world publishes, so the object stores stay empty;
- the agent's own pose (`own_pose_channel`), so the entity store fills and the reset empties it.

**What it showed:**
- **Exact, one decision at a time.** Replayed one stored sequence at a time, one decision at a time, the trunk
  reproduces every output of the rollout bit for bit: the context, the memory, the background, every token type and
  the object latents, across the reset and the sequence boundaries.
- **Batched under `lax.scan`,** it is within 1.91e-6 on the flat run and 1.67e-6 with the own pose. This is float32
  rounding across program shapes. The jitted rollout differs from the eager one at the same tick by the same 1.91e-6,
  and the unroll is identical jitted and eager on the same inputs (the builder's log, rerun here). The scan test's
  bound is 1e-5, about five times that.
- **float16 storage** moves the outputs by at most 4.65e-4 (the act token; the context 2.1e-4, the memory 1.1e-4).
- **Gradients through time** are finite, and the last decision's context reaches the memory's initial slots.
- **The cut.** A sequence never crosses a reset; its stored start is the rollout's carried state; and pack then
  unpack restores the valid rows exactly in float32.
- **The restart mask is exact under changed parameters.** The mask marks the slots whose latent the state layer
  zeroed. The layer only ever keeps a slot's latent or zeroes it, for a new thing or a reset (`state/memory.py`
  line 184); it never moves one between slots.
- **Not exercised:** a restart in the middle of a sequence, a new thing taking a slot after the first decision.
  Neither rollout has one. It needs a world where things come and go (the duel, the Region).

**What V1 needs beyond it.**
- **Memory.** `record` keeps every decision dense in float32, with its carried state. Dense tokens are about 169 KB a
  lane decision (8.6), so one 128-lane rollout of 480 decisions would hold about 10 GB. V1 packs at record time and
  keeps the carried state only where sequences start.
- **The cut.** It needs a stride, so that sequences overlap by the burn-in (R2D2), and each lane's end (its horizon).

**Gate** (two passes, against the 6acadad baseline): pass 1 the same 11 failing names, with 1,602 passed (the new sequence tests among them); pass 2 the same 30.

### 8.8 V1 built, and its timed smoke (2026-10-23; the owner approved the smoke, not the full run)

**What was built.** `training/distill_v3.py`, by a Haiku subagent from a written spec, checked and extended here. It is
A2's DAgger loop (`distill.py`, unchanged) on v3:
- **The rollout** runs `layer.step` with the agent's own pose, then the trunk's step with the carried state, then the
  head. It records each decision, packed: the valid rows in float16, with the carried state kept only where a sequence
  starts.
- **The data:** sequences of L = 90 decisions that overlap by the burn-in B = 30 (R2D2). A sequence that starts at an
  episode's reset carries the loss on every decision; a mid-episode one only after its burn-in. Every decision inside
  a horizon carries exactly one loss term.
- **The loss** re-runs the trunk over each sequence (`sequences.unroll`), applies the head, and takes A2's
  cross-entropy plus the done term.
- **The control** is the same run with the trunk's memory, background and object latents removed.

`sequences.py` gained the keywords that make this fit in memory: `record(dtype=, carry_at=)`, `cut(stride=, ends=)`
and `take`. Its defaults are unchanged.

**Run C''s options, missed at first.** V1's gate names run C' (README 2.16), which passed only with three options:
walk_to v3's stop margin, the look request's direction, and that direction measured in the head's frame. The spec said
"A2's task", the builder left the options out, and the review here did not catch it.

So the smoke and the overnight pass (8.9) ran with A2's original options. Their rates compare with each other, memory
against control, but not with C'.

Since 2026-10-24, `distill_v3` runs with C''s options by default (`DistillV3Config.walk_stop_margin`,
`look_direction`, `look_frame`). `--walk-stop-margin 0 --no-look-direction` gives A2's options back.

**Checked** (`test_sa_distill_v3.py` 17 passed, 2 skipped by design; `test_sa_sequences.py` 17 passed, 1 skipped):
- with the parameters unchanged, the loss path's head logits equal the rollout's within 1.7e-6 (memory) and 1.43e-6
  (control);
- each decision in a horizon carries exactly one loss term;
- a reset-start sequence trains the initial memory, and a mid-episode one does not;
- twenty steps on one batch lower the loss;
- a batch cut to the filled slots (below) gives the full batch's loss and gradient to float32 rounding, and the cut is
  refused when a decision's valid rows do not come first;
- DAgger's aggregate of two iterations reads each part back exactly;
- V1 runs with run C''s options by default: the 'both' lane's look request carries its direction, the walk
  request's stays empty, and the head frame without the direction is refused.

A two-iteration run of each arm checked what the one-iteration smoke cannot (4 lanes, 3 steps an iteration):
- the aggregate grows from 9 to 20 sequences;
- the student drives half the lanes in the second iteration;
- the cut holds;
- `--params-out` saves parameters that load back.

**The world's tick, measured:** 1/30 s on every lane (the trainers' flat world, `combat.last_motion_delta_seconds`), not
the 0.045 s of the loaded pursuit arena.

**The smoke as first run** (one iteration of 128 lanes, beta 1, 300 Adam steps on 12 sequences each; load 80 to 120 on
18 cores, mostly another project's training; every token slot computed):

| | Memory | Control |
|---|---|---|
| decisions, sequences, stored | 19,422; 381; 107 MB | the same episodes; 94 MB |
| collect | 126 s | 89 s |
| step p50 / p95 | 11.2 / 15.5 s | 10.8 / 20.0 s |
| 300 steps | 3,463 s (58 min) | 3,490 s |
| peak RSS | 21.0 GB | 18.8 GB |
| loss, steps 0 / 100 / 200 / 299 | 2.89 / 0.26 / 0.14 / 0.26 | 3.12 / 0.25 / 0.14 / 0.25 |
| eval seen: walk, look, both | 24%, 2%, 0% | 8%, 4%, 0% |
| eval held out: walk, look, both | 3%, 5%, 1% | 5%, 4%, 0% |
| illegal decisions | 0 | 0 |

After 300 steps on the teachers' data alone, the rates are far from run C''s. That is expected: A2 trained 6 x 2,000
steps with DAgger. The smoke measures the cost and checks the pipeline; it is not the gate.

**Where the time goes, measured.** Building a batch on the host takes 5 to 150 ms, and the compiled step the rest: the
forward pass about 2.5 s, the backward and Adam about 9.

The trunk computes every token slot, filled or not. On this world only 39 of the 356 slots ever hold a row:
- all 21 act rows, 5 kit, 1 entity, and the self, affect, persona and status;
- no geometry, inventory, block candidates or other types.

Empty slots are masked. Valid rows always come first in each type (checked), so a batch cut to the slots the data fills
computes the same values:
- the loss agrees to 7 digits;
- the largest gradient gap is 1.2e-7, against gradients up to 0.6;
- the step is about 13 times faster: 1.9 to 2.6 s against 28 to 33 s in the same process and conditions.

`distill_v3` now cuts every batch that way (`_filled`):
- it refuses data whose rows are not first, and keeps every row a request points at;
- the key test runs on the cut batch.

On the Region, geometry fills, so the gain there is smaller (not measured).

**The smoke with the cut** (the same command, the same seeds; load 125 to 140):

| | Memory | Control |
|---|---|---|
| slots computed | 39 of 356 | 39 of 356 |
| step p50 / p95 | 1.83 / 2.20 s (was 11.2 / 15.5) | 1.36 / 1.70 s (was 10.8 / 20.0) |
| 300 steps | 561 s (was 3,463) | 430 s (was 3,490) |
| peak RSS | 3.5 GB (was 21.0) | 2.7 GB (was 18.8) |
| loss at every logged step | the uncut run's to 6 digits (0.256438 against 0.256437 at step 299) | identical to 6 digits |
| evaluation | identical rates, lane for lane | identical |
| the whole smoke | 20 min (was 65) | 17 min (was 66) |

- **The cut is the same training:** after 300 Adam steps the two paths still agree to 6 digits, and every evaluation
  outcome is the same.
- **The step is 6 to 8 times faster under this load.** In the profile, same process and conditions, it was 13 times.
- **The evaluations now take the largest share,** 3 to 5 minutes each. They run the trunk at full capacity in the
  rollout, which is not worth cutting at two evaluations a run.
- **Memory against control, at this stage:**
  - after step 0, where the initial parameters differ (2.89 against 3.12), the two loss curves agree within 0.02 at
    every logged step, since both learn the same teacher labels on the same episodes;
  - the memory run walks to seen goals more often: 21 of 86 lanes against 7;
  - held out, the two are level.

  This is one seed after 300 steps on teacher data only, so it is not evidence either way. V1's gate is the full
  run's.

**What it sets for the full run** (A2's schedule: 6 iterations of 2,000 steps, DAgger beta halving; inferred from the
cut smoke's numbers at load 125 to 140, not run):
- **Training:** 12,000 steps at 1.4 to 1.8 s is 4.5 to 6.1 hours an arm. Uncut, it would have been about 37.
- **Collection:** about 2 to 3 minutes per iteration. The aggregate grows to about 117,000 decisions and 0.6 GB.
- **Memory:** a peak near 3 to 4 GB; the step's memory does not grow with the aggregate.
- **The two arms** run one after the other, so about 10 to 13 hours on today's shared machine. A quiet machine should
  be about 3 times faster (8.9: 0.54 s a step at load 25 to 40 against 1.7 s under load; measured once).
- **The ask to the owner,** with these numbers: the full V1 run of both arms, or a shorter schedule first, for
  example 6 x 500 steps (about 3.5 hours for both).
- **RULED 2026-10-24:** "do shorter passes overnight", and "ensure that it is proper before committing". Both arms run
  at 6 x 500 steps, in parallel, with the parameters saved; its results are recorded in 8.9 once read. The full
  schedule waits on them.

**Gate** (two passes, against the 6acadad baseline, on 1e52c83 with V1's files):
- pass 1: 1,636 passed. Of its 12 failing names, 11 are the baseline's, and one is 6101ddd's intended relations
  test (VIRTUAL_HUMAN.md 26.8). None comes from V1;
- pass 2: the same 30.

### 8.9 The short pass, memory against control (2026-10-24; the owner: "do shorter passes overnight")

**What ran.** Both arms with the same command, seeds and data:
- 6 DAgger iterations of 500 steps, beta 1 then halving;
- 128 lanes, the batch cut to the filled slots;
- evaluation on seed 77001, 256 lanes, seen and held out.

They ran in parallel, at load 85 to 140. The code that ran is kept beside the reports (`distill_v3_as_run.py`, hashes in
`code.txt`).

**One caveat.** It ran with A2's original options, before run C''s were made V1's default (8.8): walk_to v2, and no
look direction. So these rates compare memory against control, not either against C'.

| | Memory | Control |
|---|---|---|
| aggregate after 6 iterations | 2,445 sequences | 2,445 sequences |
| step p50 by iteration | 1.66 to 2.01 s; 0.54 s in the last, at load 25 to 40 | 1.04 to 1.33 s |
| the whole run | 94 min | 80 min |
| illegal decisions (training, evaluation) | 0, 0 | 0, 0 |
| seen, held success: walk, look, both | 0.767, 0.082, 0.118 | 0.872, 0.047, 0.094 |
| seen, first hit: walk, look, both | 0.767, 0.671, 0.294 | 0.872, 0.435, 0.341 |
| held out, held: walk, look, both | 0.733, 0.035, 0.176 | 0.686, 0.012, 0.129 |
| held out, first hit: walk, look, both | 0.733, 0.188, 0.388 | 0.686, 0.247, 0.376 |

**What it shows** (85 or 86 lanes per mode; one seed an arm):
- **Training through time works over DAgger's iterations.**
  - The aggregate grows to 2,445 sequences.
  - The student drives 121 of the 128 new lanes in the last iteration.
  - Each iteration's loss falls.
  - No decision is illegal, in training or in either evaluation (79,669 decisions an arm).
- **Walking is learnt:** 0.77 and 0.87 seen, 0.73 and 0.69 held out. A walk that arrives also holds.
- **Looking is reached but not held:** held success is at most 0.08 in either arm, while first hits reach 0.19 to
  0.67. This is the failure run C' fixed with the look direction in the head's frame (README 2.16), and these runs
  did not have that option.
- **No consistent difference between memory and control.**
  - On seen goals, the control walks better: 75 of 86 lanes against 66, about 1.8 standard errors.
  - On seen goals, the memory run reaches a look target more often: 57 of 85 against 37, about 3.2 standard errors.
  - Held out, the look difference reverses (0.188 against 0.247), and the walk difference reverses too (0.733
    against 0.686).

  This task needs no memory, since each request carries its target. So "not worse" is the question V1 asks, and one
  seed cannot answer it.
- **The gradient's peaks grow with the iterations:** the largest norm goes from 10 to 37 (memory) and from 11 to 29
  (control). There is no clipping. The loss still falls every iteration, but a full run should log the spikes and be
  ready to clip.
- **Speed.** The memory run's last iteration ran when the machine quieted (load 25 to 40) and took 0.54 s a step,
  against 1.7 s under load 90 to 140: measured once.

**Next, for the owner:** the same short pass with run C''s options, now V1's default, so that V1's bars against C'
can be read. It is about 1.5 hours an arm in parallel at today's load, and less when the machine is quiet. A second
seed would make the memory comparison readable.

### 8.10 V1 with run C''s options: the gate passes (2026-10-24; the owner approved the pass)

**What ran.** The same short pass as 8.9 (6 iterations of 500 steps, 128 lanes, the batch cut to the filled slots),
now with run C''s options (V1's default since 8.8): walk_to v3's stop margin, and the look direction in the head's
frame.
- Two seeds, memory and control: four runs in parallel.
- They ran from a clean worktree at ec87aa9, at load 70 to 110.
- The reports and parameters are in the session scratchpad (`v1_cprime/`), not in the repo.

**The bars** (A2's, `training/bars.py`'s rules):
- Held success, against the teachers' seen rates from run C' (README 2.16: walk 1.000, look 1.000, both 0.906).
  The teachers' rates do not depend on the student: the same seeds, world and options.
- T5: the teacher less the student is at most 0.10.
- Held out: at least 0.90 of seen.
- E4: pace within 5% on at least 95% of lanes.

| Run | Walk, seen / held out | Look | Both | Pace | Bars |
|---|---|---|---|---|---|
| memory, seed 0 | 1.000 / 0.942 | 0.953 / 0.953 | 0.835 / 0.859 | 0.967 | all pass |
| memory, seed 1 | 1.000 / 0.988 | 0.988 / 0.988 | 0.859 / 0.788 | 0.953 | all pass |
| control, seed 0 | 1.000 / 1.000 | 0.976 / 0.824 | 0.859 / 0.776 | 0.980 | held-out look 0.843 fails |
| control, seed 1 | 1.000 / 1.000 | 1.000 / 0.953 | 0.859 / 0.859 | 0.974 | all pass |
| run C' (v1, the bar) | 1.000 / 1.000 | 0.988 / 1.000 | 0.882 / 0.847 | 0.980 | all pass |

No decision was illegal, in training or in any evaluation.

**The gate of 8.3, V1:**
- **At least C''s rates:** the memory run passes every bar on both seeds.
- **The memory run not worse than its control:** it passes on two seeds of two, the control on one of two.

V1 passes. With C''s options, the short schedule (6 x 500 steps, a quarter of A2's) is enough, so the full schedule
is not needed for the gate.

**Cost.** A memory run took 143 minutes and a control run 107, all four in parallel at load 70 to 110. The memory
trunk's step is about 1.4 times the control's.

**What it sets.**
- The memory trunk is the shared frozen trunk for the skills' modules (V2, 8.3), from either seed.
- Its walk_to and look_at heads already serve those two verbs as skill bodies: the architecture test (8.11, being
  built).
- V2's per-skill trainer on the frozen v3 trunk is being built (`training/skill_dagger_v3.py`). Its full runs wait on
  the owner's approval.

### 8.11 V2: each skill its own module on the frozen V1 trunk, and the trained skills as bodies (2026-10-24)

**The rulings.**
- The owner (2026-10-24): "we want to get everything done asap so that we can start training the neural policies and
  actually start to test the architecture".
- V2 was approved: "Approve V2 now": walk_to and look_at as their own modules on V1's frozen memory trunk, 6 x 500
  steps.
- The owner's ruling of 2026-10-19 applies: each skill is its own module on a shared frozen trunk, trained by its own
  strategy.

**The trainer** (`training/skill_dagger_v3.py`, built by a Haiku subagent from a written spec and checked here). It is
`skill_dagger`'s method on v3:
- the rollout runs the state layer and the frozen trunk with its carried state;
- the dataset keeps each decision's trunk outputs, packed as valid rows;
- the module (a generalist head) is fitted per decision. The trunk is frozen, so there is no gradient through time.

**Checked:**
- the stored outputs reproduce the module's rollout logits (1.4e-6);
- the trunk's parameters do not change, and only the module's take a gradient;
- twenty steps on one batch lower the loss;
- the report scores on `bars.py --skill`.
- **The decisive check** (no training): V1's own heads run as V2 modules reproduce V1. On the same 64 held-out
  episodes, with matched policies, positions, hits, holds and decisions are identical. So the V2 rollout, lowering and
  evaluation are V1's.

**The runs** (both skills in parallel, 42 minutes each at load 50 to 120):
- the trunk: `memory_s1`'s, frozen, sha e6800023cca5;
- 6 iterations x 500 steps at 128 lanes; evaluation on seed 77001, 256 lanes;
- a step is about 0.2 s, since only the module trains;
- look_at's run stored 595 MB of trunk outputs (float16).

| Held success, module (teacher) | Seen | Held out | Bars |
|---|---|---|---|
| walk_to alone | 1.000 (1.000) | 1.000 (1.000) | all pass; pace 97.8%, radius 100% |
| walk_to beside look_at | 1.000; look_at beside it 0.852 (0.844 beside the teacher) | 1.000; look_at 0.883 (0.867) | all pass |
| look_at alone | 0.992 (1.000) | 0.992 (1.000) | all pass |
| look_at while walking | 0.906 (0.906) | 0.812 (0.859) | T5 passes; held out / seen 0.897, bar 0.90: fails by 0.003 |

- **walk_to's module passes every bar.**
- **look_at's module passes every bar but one,** by 0.003: the held-out ratio of looking while walking. That is the
  bar V1's own head missed when run as a module (0.860, in the reproduction check), and the one the teacher itself
  drops on most (0.906 to 0.859).
- **A second look_at seed** (seed 1, 29 minutes) gives the same 0.897:
  - looking while walking: seen 0.914 (the teacher 0.906), held out 0.820 (0.859);
  - looking alone: 1.000 both.

  So the gap is real, not noise: looking toward the held-out band while walking generalises about 4 points worse
  than the teacher on both seeds, though T5 holds everywhere. V1's generalist shows the same lean on one of its two
  seeds (0.917 in 8.10). The likely fix is a longer look_at schedule (A2 trained 4 times as many steps); not run.

**The trained skills as bodies** (`training/export_v1.py`, built by a Haiku subagent from a written spec and checked
here). V1's runs are exported to `scratchpad/v1_bodies/<run>/`:
- a trunk with its token layout;
- the walk_to and look_at modules;
- a manifest.

Through `state_tick.ModuleBody`, as the combiner executes a skill:
- on 32 seen and 32 held-out lanes per skill, the bodies succeed and hold on every lane;
- their first success tick equals V1's own student on every lane;
- no executed action is illegal;
- the memory arm, with its state carried across ticks by `step_lanes`, also matches V1 tick for tick.

Four seam defects found on the way were fixed:
- the loaded trunk's config was unhashable;
- a body drove a head it was never trained on (`mount`);
- there was no production lowering: the request's arguments now come from the agent's pose and tick (`AgentView`);
- the legal mask was not applied.

One more appeared when the hearing slice added the `sound` token type (VIRTUAL_HUMAN.md 28.8): the export rebuilt
V1's configuration from today's schema, `sound` included, and could no longer load V1's runs. It now uses V1's
configuration as trained (`export_v1.as_trained`), and the saved layout covers the trunk's own types.

**What is left for the architecture test.** The decision layer must drive these bodies on gen 7: goals become walk_to
and look_at requests, and the combiner executes them through `ModuleBody`. That is the coordinator's lane, asked in the
ledger on 2026-10-24 (two messages expired undelivered). The skill side's pieces are in place:
- the bodies;
- the lowering;
- `step_lanes`, compiled at 20.6 ms a tick (29).

**Gate:** two passes on dc9037a with all six of 2026-10-24, against the step-9 baseline (6acadad). Pass 1: the baseline's 11 failing names and 18 more, each explained. `test_sa_export_v1.py`'s 11 and `test_sa_integration_gen7.py`'s 5 were this lane's (the export rebuilt V1's configuration with `sound`; the check's capture under the compiled tick), fixed and re-run: export 11 passed, integration 5 passed and 1 strict xfail (the Region carries no construct since 6d9307f). `test_sa_drives.py`'s real-simulator clock test (2) fails on a clean dc9037a too, bisected to the coordinator's 6d9307f, now a strict xfail (drives 20 passed, 2 xfailed). 1,693 passed. Pass 2: 1 failing name, a baseline one (29 of the baseline's 30 now pass). Only their own tests import the two files fixed after the gate. Landed on e826fa7 (the coordinator's live hearing; no shared files).

### 8.12 The longer look_at schedule, come_here's, track's and follow's first runs, and the V2 bodies (2026-10-24 and 25, night)

**The rulings.** The owner, 2026-10-24, on the asks of 8.11: "go ahead", taken as:
- come_here's smoke, then its run;
- the longer look_at schedule, on both seeds.

The same night they ruled track (gaze only, scored within the head's reach on paths that cross in front) and follow
(sprint; trained on walking and sneaking people; runners parked). Both are in their TRAINING.md section 8.

All runs use the frozen V1 trunk `memory_s1` (sha e6800023cca5) at 128 lanes. They are evaluated on seed 77001, 256
lanes per region. The trainer is `training/skill_dagger_v3.py`, unchanged from 8.11 except come_here's two gaps
(entity 1 placed per lane, the pointer resolved per decision).

**look_at at 6 x 2,000 steps** (4 times 8.11's schedule, A2's length; 78 minutes each at load 85 to 120). **Every bar
passes, on both seeds:**

| Held success, module (teacher) | Seed 0 | Seed 1 |
|---|---|---|
| alone, seen / held out | 1.000 / 0.992 (1.000 / 1.000) | 0.992 / 1.000 (1.000 / 1.000) |
| while walking, seen / held out | 0.898 / 0.820 (0.906 / 0.859) | 0.883 / 0.836 (0.906 / 0.859) |
| T5 while walking (bar 10 points) | +0.8 | +2.3 |
| held out / seen while walking (bar 0.90) | 0.913 | 0.947 |
| walk_to beside it | 1.000 | 1.000 |

The held-out ratio that failed by 0.003 at 6 x 500 (0.897 on both seeds) passes once trained longer. As 8.11 inferred,
the gap was undertraining, not the method.

**come_here.**
- **The smoke** (one iteration of 200 steps, 21 minutes): end to end, 0 illegal decisions. The module held 0.03, as
  walk_to's smoke held 0.06 at the same size.
- **The short pass** (6 x 500, 50 minutes) **does not pass:**

| Held success, module (teacher) | Seen | Held out |
|---|---|---|
| alone | 0.828 (1.000) | 0.906 (1.000) |
| beside look_at | 0.727 (1.000) | 0.758 (1.000) |

  - T5 fails by 17 and 27 points, against a bar of 10.
  - E4 pace is 87%, against 95%.
  - Held out over seen passes (1.09 and 1.04), and there are 0 illegal decisions.
  - Its final loss rose across iterations (0.068 to 0.087) while walk_to's fell to about 0.02, which suggests
    undertraining.
- **The pointer ablation** (come_here's plan, section 8: clear the pointer and evaluate again).
  - As trained, the evaluation reproduces the report exactly (0.828 / 0.727 seen, 0.906 / 0.758 held out).
  - With the pointer cleared (-1 every decision), come_here's held success falls to 0.000 alone and 0.008 beside look_at,
    seen and held out.

  So the module finds its target through the pointer, as designed. The miss is not the pointer: it is undertraining,
  as look_at's was.
- **The run at 6 x 2,000** (A2's size, as the owner's approval of the run allows; 49 minutes at load 60 to 80). Much
  better, but not every bar:

| Held success, module (teacher) | Seen | Held out | T5 (bar 10 points) | Held out / seen |
|---|---|---|---|---|
| alone | 0.922 (1.000) | 0.938 (1.000) | +7.8 pass | 1.017 pass |
| beside look_at | 0.867 (1.000) | 0.812 (1.000) | +13.3 FAIL | 0.937 pass |

  - E4 pace 89.9%, against 95%: fails. The radius is kept on every success, and there are 0 illegal decisions.
  - look_at beside it held 0.844 seen and 0.891 held out, against 0.844 and 0.867 beside the teacher.
  - Its final loss settled at about 0.045, against walk_to's 0.02.
  - So come_here alone passes T5 and the held-out bar, but not beside look_at, and not pace.
  - **The next lever was the recipe's own rule:** a dependent skill starts from its prerequisite.
- **From walk_to's V2 module** (owner, 2026-10-24: "Start from walk_to's module"; `--init-module`, built for it, see
  `training/README.md`; 6 x 2,000, 33 minutes). It does not help:

| Held success, module (teacher) | Seen | Held out | T5 | Held out / seen |
|---|---|---|---|---|
| alone | 0.906 (1.000) | 0.953 (1.000) | +9.4 pass | 1.052 pass |
| beside look_at | 0.789 (1.000) | 0.766 (1.000) | +21.1 FAIL | 0.970 pass |

  - Pace 88.8%.
  - The loss starts lower (0.013 after the first iteration) but rises to 0.055 by the last.
  - With three schedules and two starts, come_here alone passes T5 and held out. Beside look_at it misses by 13 to 21
    points, and pace by 5 to 6.
  - The cause is not the pointer (cleared, success falls to 0) and not plain undertraining (4 times the steps closed
    look_at's gap, not this one).
  - The entity token's position is in the body frame (`state/frames.py`), so a head that turns does not move the
    target's position columns. It does move the token's head-bearing columns.
  - **A diagnostic:** those columns set to the body bearing at evaluation, on the 6 x 2,000 module from scratch.
    - Alone, held out falls from 0.938 to 0.781; beside look_at, seen falls from 0.867 to 0.734 (held out 0.812 to
      0.781).
    - So the module uses the head-bearing columns, and removing them hurts. The miss beside look_at is not a
      head-frame shortcut that removing them would fix.
  - **How the failed lanes end** (the same module, the report's lanes). Every failed lane stopped short and stayed:
    - it never came within the radius, and its closest point was its last;
    - in the last second its median speed is 0 and it made no progress (0 to 3 lanes per group still moving).

    Alone it stops about 1.5 blocks from the person, just outside the radius. Beside look_at it stops 3.3 to 4.1
    blocks short. So the module decides it has arrived too early, and much earlier when look_at moves the head.
  - **A second diagnostic** (the entity's head elevation recomputed with the head level, everything else as observed).
    - Alone it is unchanged (0.922 and 0.938), as it should be: the head is level there.
    - Beside look_at it falls (seen 0.867 to 0.664, held out 0.812 to 0.625).
    - So the module reads the pitched elevation correctly, and that is not the cause either.
  - **A third diagnostic** (look_at's request position and direction hidden from come_here's module only; look_at's
    teacher unchanged). Beside look_at it falls (seen 0.867 to 0.734, held out 0.812 to 0.672). The module uses
    look_at's target as a helpful cue, since it lies near the person's bearing; it does not mistake it for its own.
  - **Reading the three together:** every cue removed hurts, so the miss is not one misread input. The module has not
    learned the composed case well enough.
    - The likely reason is that come_here's target arrives through the entity token, whose head-frame columns move
      whenever look_at turns the head.
    - walk_to's target arrives as a body-frame position in its request, which head motion does not touch. That would
      be why walk_to composes and come_here does not. This is inferred, not proven.
  - **Options, none run:**
    - more composed lanes in training;
    - a larger module;
    - lowering the person's body-frame position into come_here's request at runtime from the agent's own perception,
      which makes come_here walk_to with a perceived target, a design change.

    The owner (2026-10-25): use come_here alone for now.

**track and follow** (owner, 2026-10-25: "yes for them all": their smokes, then runs; the holding-skill trainer, its
share bars, and track's and follow's builds are in `training/README.md` and their TRAINING.md section 8).
- **The smokes** (one iteration of 200 steps) ran end to end with 0 illegal decisions.
- **The runs** (6 x 2,000, 67 and 59 minutes) do not pass:

| Share in view / in band, module (teacher) | Seen | Held out | T5 | Held out / seen |
|---|---|---|---|---|
| track alone | 0.239 (0.933) | 0.255 (0.933) | +69 FAIL | 1.07 pass |
| track beside walk_to | 0.121 (0.516) | 0.119 (0.507) | +40 FAIL | 0.98 pass |
| follow alone | 0.565 (0.935) | 0.547 (0.945) | +37 FAIL | 0.97 pass |
| follow beside track | 0.790 (0.930) | 0.815 (0.943) | +14 FAIL | 1.03 pass |

- Final losses: track 0.18, follow 0.24, against walk_to's 0.02.
- track's teacher beside walk_to holds only 0.52: when walk_to turns the body, the gaze teacher keeps the person in
  view about half the reachable time. That teacher needs work regardless of the student.
- **A pattern across the three entity skills** (inferred, not proven). come_here, track and follow take their target
  from the entity token, and all three fall well short. walk_to and look_at take theirs from the request's arguments,
  and pass.
  - The frozen V1 trunk was trained with entity 1 inert at one fixed spot (`blocks.OTHER_XYZ`), so its entity
    encoding has never seen a person who is placed elsewhere or who moves. Its geometry weights are likewise at their
    initial values: it never saw terrain.
  - follow does better beside track, whose teacher keeps the head on the person, than alone. That fits a trunk whose
    entity cues are clean only when looked at.
  - **If it holds,** the new trunk the owner approved (teachers first, then a terrain trunk) should be trained on
    people who stand anywhere and move (come_here, track, follow lanes) as well as on terrain, before the entity skills
    are trained again.
  - A cheap check, not yet run: fit a linear read-out from the trunk's entity rows to the person's true position, for
    a person at the fixed spot against one placed or moving elsewhere.

**The V2 bodies** (`scratchpad/v2_bodies/memory_s1/`: V1's trunk, V2's walk_to module, and a V2 look_at module).
They were driven through `state_tick.ModuleBody` as 8.11's check drives V1's, on 32 seen and 32 held-out lanes per
skill.
- With look_at's 6 x 500 seed-1 module:
  - both skills succeed and hold on every lane, seen and held out;
  - 0 illegal actions;
  - the production lowering checked on every decision.
- With the 6 x 2,000 seed-1 module, which the folder now holds: the same.
  - Every lane succeeds and holds, seen and held out, for both skills.
  - The median first success is tick 8 seen and 16 held out for look_at, and 102 and 92 for walk_to.
  - 0 illegal actions; the lowering checked on all 10,018 and 9,112 decisions.
- **A precision to 8.11.** Its 32 + 32 lane check, where every lane succeeds and holds, with first successes equal to
  V1's own student on all 32, ran on the control arm's bodies, which `test_sa_export_v1.py`'s export fixture exports.
  Re-run on today's code (2026-10-24), it holds exactly. The memory arm was checked there on two lanes.
  - Run the same way on 32 + 32, the memory arm's bodies (`v1_bodies/memory_s1`, the trunk V2 builds on) hold on
    every seen lane and on 31 of 32 held-out lanes per skill.
  - The same holds with the tick run eagerly as before its compile (so the compile is not the cause). It is within
    that arm's own held-out rates (0.94 to 0.99, 8.10).

**Gate:** two passes on a56f434 with holding skills and track, follow and --init-module, and the ModuleBody fix, against the step-9 baseline (6acadad). Pass 1: the baseline's 11 failing names, none new (1,766 passed, 2 xfailed). Pass 2: 1 failing name, a baseline one. Landed on 842180e (the coordinator's 812c2ef..842180e, no shared files but the ledger).

### 8.13 The trainer's default path, pace in-episode, and the first terrain teacher (2026-10-25; no training)

**The rulings.** The owner, 2026-10-25:
- on speed, "Both, slot cut first", then "Make A the default";
- on pace, "Fix pace to in-episode ticks, keep B opt-in";
- on terrain, "Teachers first, then a terrain trunk".

**The V2 trainer's default path changed** (`training/skill_dagger_v3.py`; its README row has the details).
- **The slot cut is now the default.** The trunk runs on each token type's filled slots: 39 of 356 on the flat world,
  as V1's own trainer cuts its batches.
- **Bit-exact shortcuts skip what no output reads:**
  - the trunk and module on teacher-only evaluations;
  - the module on iteration 0's collection.
- **The runs of 8.11 and 8.12 used the exact path,** which `--no-slot-cut` keeps. A re-run at the default differs from
  them by float32 rounding: stored trunk outputs within 1e-5, with no executed action and no evaluation number changed on
  the walk_to and track probes.
- **Measured:**
  - end to end, 817 s to 591 s from the bit-exact part alone (walk_to and track, 1 x 50 steps, loaded machine);
  - per batch, collecting 128 lanes 115 s to 38 s and evaluating 256 lanes 161 s to 75 s with the cut.
- **Lane compaction** (`--compact-lanes`) stays opt-in.

**Pace (E4) counts only decisions inside each lane's horizon.** Before, a lane's walk after its horizon, which the rollout
still runs, was scored. Re-scored, no saved module's verdict changes:
- V2 walk_to: 0.978 and 1.000;
- come_here: 0.899 and 0.866, still failing as in 8.12.

**The skill bodies, batched.** Each skill lowers its agents' requests in one numpy call (`lowering_batch`). With
`ModuleBatch` (f4d1c4f), one tick of both verbs is 7 ms for 32 agents and 13 ms for 96, against 150 to 640 ms per agent
before. Choices and done are equal to the per-agent bodies.

**The first terrain teacher** (`teaching/path_teacher.py`, bench `bench/region_path_teacher.py`). It follows the shortest
path on the Region's traversal graph: walk, climb up to 1.3 blocks, drop up to 3.0.

| Reach, batch 32, 20 s | 3 to 6 blocks | 6 to 10 | 10 to 15 |
|---|---|---|---|
| straight teacher, gen 5 | 65.6% | 46.9% | 28.1% |
| straight teacher, gen 7 | 78.1% | 81.3% | 59.4% |
| path teacher, gen 7 | 84.4% | 81.3% | 65.6% |

- Gen 7's step-up is most of the gain. The path teacher adds 0 to 2 lanes, within noise.
- Most gen-7 failures stop at rest on flat ground or at a drop lip, not at a climb, and none leaves the graph.
- **So no terrain teacher is certified yet.** The stalls are being diagnosed (stage T2): a teacher deadlock, body
  contact or the engine. That comes before a terrain trunk is planned.
- **The frozen trunk's entity row, probed** (stage P2; a linear read-out of entity 1's body-frame position, 256 lanes per
  set, split by lane; measured). The raw entity token gives test R² of at least 0.998 for the forward and right
  offsets on every set, and the trunk's row much less:

| Lane set | Forward R², trunk row | Right R², trunk row |
|---|---|---|
| A: the person at the fixed spot (walk_to's lanes) | 0.63 | 0.90 |
| B: placed per lane (come_here's) | 0.46 | 0.23 |
| C: moving (track's) | 0.84 | 0.81 |

  A read-out fitted on A fails on B (R² below 0) and transfers only partly to C (0.30 and 0.20), while the raw token
  transfers fully. This supports 8.12's inference: the next trunk should train on placed and moving people. The deficit
  is already there on A, so placement alone may not be enough (inferred). A nonlinear probe, still running, will show
  whether the position is there but not linearly readable.

## References

- Agarwal, R., et al. (2021). Deep reinforcement learning at the edge of the statistical precipice. NeurIPS.
  arXiv:2108.13264.
- Andrychowicz, M., et al. (2021). What matters for on-policy deep actor-critic methods? A large-scale study. ICLR.
  arXiv:2006.05990.
- Balduzzi, D., et al. (2019). Open-ended learning in symmetric zero-sum games. ICML.
- Bansal, T., Pachocki, J., Sidor, S., Sutskever, I., Mordatch, I. (2018). Emergent complexity via multi-agent competition. ICLR.
- Chen, D., Zhou, B., Koltun, V., and Krähenbühl, P. (2019). Learning by cheating. CoRL. arXiv:1912.12294.
- Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine
  translation (the GRU). arXiv:1406.1078.
- Elo, A. E. (1978). The Rating of Chessplayers, Past and Present. Arco.
- Engstrom, L., et al. (2020). Implementation matters in deep policy gradients: a case study on PPO and TRPO. ICLR.
  arXiv:2005.12729.
- Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML.
  arXiv:1706.04599.
- Henderson, P., et al. (2018). Deep reinforcement learning that matters. AAAI. arXiv:1709.06560.
- Jaderberg, M., Mnih, V., Czarnecki, W. M., Schaul, T., Leibo, J. Z., Silver, D., Kavukcuoglu, K. (2017). Reinforcement
  learning with unsupervised auxiliary tasks. ICLR. arXiv:1611.05397.
- Kapturowski, S., Ostrovski, G., Quan, J., Munos, R., Dabney, W. (2019). Recurrent experience replay in distributed
  reinforcement learning. ICLR.
- Laskey, M., Lee, J., Fox, R., Dragan, A., and Goldberg, K. (2017). DART: noise injection for robust imitation
  learning. CoRL. arXiv:1703.09327.
- Lee, J., et al. (2019). Set Transformer. ICML. arXiv:1810.00825.
- Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V., Hutter, M. (2020). Learning quadrupedal locomotion over challenging
  terrain. Science Robotics 5(47). arXiv:2010.11251.
- Messikommer, N., et al. (2025). Student-informed teacher training. ICLR. arXiv:2412.09149.
- Narvekar, S., et al. (2020). Curriculum learning for reinforcement learning domains: a framework and survey.
  JMLR 21. arXiv:2003.04960.
- Ng, A. Y., Harada, D., and Russell, S. (1999). Policy invariance under reward transformations: theory and
  application to reward shaping. ICML, 278-287.
- Peng, X. B., Ma, Z., Abbeel, P., Levine, S., Kanazawa, A. (2021). AMP: Adversarial motion priors for stylized physics-based character control. ACM TOG (SIGGRAPH).
- Pinto, L., Andrychowicz, M., Welinder, P., Zaremba, W., Abbeel, P. (2018). Asymmetric actor critic for image-based
  robot learning. RSS. arXiv:1710.06542.
- Rajeswaran, A., et al. (2018). Learning complex dexterous manipulation with deep reinforcement learning and
  demonstrations. RSS. arXiv:1709.10087.
- Ross, S., Gordon, G., and Bagnell, D. (2011). A reduction of imitation learning and structured prediction to
  no-regret online learning (DAgger). AISTATS. arXiv:1011.0686.
- Schulman, J., et al. (2016). High-dimensional continuous control using generalized advantage estimation. ICLR.
  arXiv:1506.02438.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
- Sutton, R. S., Precup, D., and Singh, S. (1999). Between MDPs and semi-MDPs: a framework for temporal
  abstraction in reinforcement learning (options). Artificial Intelligence 112, 181-211.
  doi:10.1016/S0004-3702(99)00052-1.
- Vinyals, O., et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning. Nature 575.
- Warrington, A., Lavington, J. W., Scibior, A., Schmidt, M., Wood, F. (2021). Robust asymmetric learning in POMDPs.
  ICML. arXiv:2012.15566.
