# The low-level architecture: research and a recommendation (2026-10-18, late)

Status: RESEARCH, for the owner's decision. Nothing here is built yet.

**The owner's direction.** "We might change the architecture from just pure PPO, because the goal for the agent is that
it is somewhat generalized at the low level: given some request it activates and knows what to do, similar to a VLA
model."

**Question.** What is the best low-level architecture and training recipe for a policy that takes our request and our
observation tokens, acts at 30 Hz, and generalises across skills?

**Short answer.** One request-conditioned generalist policy, a structured-command VLA without the vision and language
parts:
- **Inputs:** the request as typed tokens, with absent arguments masked; the observation as the token sets we
  already encode.
- **Outputs:** one output query per granted scope, factored categorical heads, and a request-completion head.
- **Cost:** one batched forward per tick for all agents and all their active requests.
- **Training:** it is not trained by PPO from scratch. Privileged teachers in the simulator generate the data, DAgger
  distils them into the legal-observation generalist, and RL fine-tunes it with a KL anchor.

This is the recipe of the physics-based character controllers that already solve "one controller, many command
types": HOVER, MaskedMimic, and the legged-locomotion teacher-student line. It matches what the game-agent and VLA
literature found about one network over many tasks.

Evidence tags:
- [read] means this session read the primary abstract or text.
- [agent] means one of four Haiku research agents read it, with a link and no replication. Agent-reported numbers
  that a re-check could not find on the abstract page are not used.
- [I] is an inference for our setting.

## 1. What we need

- **Input:** a structured request (verb, typed arguments, mode, urgency, expiry; README 2.3) and legal token
  observations (entities, block candidates, inventory, geometry, self, kit).
- **Concurrency:** several requests at once on disjoint scopes; the combiner decides which are granted.
- **Output:** 21 factored discrete heads in six scopes.
- **Rate:** 30 Hz, p95 <= 15 ms per tick for 8 agents on this CPU.
- **Data:** a fast JAX simulator; 32 native NPC traces; possibly a scripted planner (the owner's OK is pending).
- **Growth:** the catalogue grows, and a frozen skill must not silently change (requirement 7).
- **What failed before:** PPO per skill on an MLP+GRU (TRAINING_NEEDS), and cloning from the 32 traces, which barely
  beat the majority class.

## 2. What the field does, by design question

### 2.1 One network or many?

- **One network.** Every multi-task system the agents read (Gato, SIMA, AdA, NitroGen, Lumine) uses one network
  across tasks [agent].
- **A distilled generalist beat the specialists.** HOVER's distilled multi-mode controller beat per-mode specialists
  on at least 7 of 12 metrics per mode. It also beat a multi-mode policy trained by RL from scratch on all 32
  metric-mode combinations [read].
- **Distilled multi-task students beat their teachers.** On Atari, a multi-task student beat its single-task teachers
  (Rusu et al. 2015) [agent].
- **Two-speed systems.** The VLA line converged on a slow semantic system feeding a fast action system: Helix runs
  an 80M policy at 200 Hz under a 7B model at 7 to 9 Hz, and GR00T N1 runs actions at 120 Hz [agent]. Our LLM and
  decision layers are already the slow system, so the skill layer is the fast "System 1".

### 2.2 How does the request reach the policy?

- **As typed tokens, with absent parts masked.**
  - MaskedMimic gives each goal modality its own encoder, and "the token masks prevent the transformer from
    attending to unspecified inputs" [read].
  - HOVER masks its command vector by mode and sparsity, with bits sampled per episode [read].
  - Octo blanks its task tokens at random during training [agent].
  - [I] For us: a verb token, one token per typed argument, and absent arguments masked.
- **Structured goals over free text at the low level.**
  - XLand and AdA condition on predicate goals used as attention queries [agent].
  - ROCKET-1 has a VLM planner hand the fast policy typed visual targets (segmentation masks), and reports a 76%
    absolute improvement in open-world interaction [read].
  - [I] Our requests are already structured, so no language model belongs on the tick path.
- **Entity and block references as pointers.** AlphaStar pairs an entity transformer with a pointer-network head
  (Vinyals et al. 2019) [agent]. Our resolver already marks the target token (`references.mark`) [read]. [I]
  Entity handles and cells become marked tokens or pointer scores, so a new entity needs no new weights.
- **Train with stale requests.** A 2026 paper trains a fast action loop with "time-misalignment training", so it
  stays robust when the slow module's conditions are stale [read]. [I] Our requests can be up to 15 ticks old:
  sample request ages in training.
- **Steering by context.** π0.7 conditions on a prompt plus context, such as metadata about task performance, so it
  can learn from varied and imperfect data [read]. [I] Request fields such as urgency, mode and the teacher's quality
  can be conditioning inputs too.

### 2.3 How are actions produced fast?

- **No multi-step heads in the tick loop.** Diffusion Policy needs 10 inference steps (0.1 s on an RTX 3080).
  OpenVLA-OFT measured parallel decoding with an L1 head at 109.7 action-Hz, against 10.1 for diffusion. Token-by-token
  decoding (OpenVLA about 6 Hz, π0-FAST about 750 ms per chunk) is far too slow [agent]. [I] Our heads are discrete,
  so one softmax per head is the cheapest and exact form.
- **Output queries per output group.** Perceiver IO's output queries and Octo's readout tokens give one query per
  output group [agent]. [I] One query per scope gives concurrent skills their own read of the request.
- **Chunking is the second lever.** ACT, π0, GR00T N1 and NitroGen predict chunks of actions, but ACT's authors call
  naive chunking sub-optimal [agent]. [I] Start per tick; chunk only if the measured cost forces it.
- **Small is enough.** TurboVLA reaches 97.6% average success at 0.2B parameters and 31.2 ms on an RTX 4090 [read].
  [I] With no vision encoder, our policy can be far smaller. No source reports a CPU timing, so ours must be
  measured.

### 2.4 How is it trained?

- **Teacher first, then distil.**
  - HOVER trains a privileged teacher with PPO, then a student by DAgger with an MSE loss to the teacher's actions
    [read].
  - MaskedMimic trains an RL motion tracker, then distils it by DAgger into the masked-goal controller [read].
  - Learning by Cheating's privileged teacher reached 100% on the original CARLA benchmark [agent].
  - Lee et al. (2020) trained a privileged terrain teacher, then a proprioceptive student by DAgger [agent].
  - RLDG has RL specialists generate the data for a generalist: up to 40% higher success than human-demonstration
    generalists, from "optimized action distributions and improved state coverage" [read].
- **Variety over volume.** "The diversity of environments and objects is far more important than the absolute number
  of demonstrations" [read]. AdA's 25B-task pool beat a 200M pool [agent].
- **Hindsight relabeling.** Any trajectory becomes data for the request it actually satisfied (HER; GCSL; STEVE-1 at
  USD 60 of compute) [agent].
- **RL fine-tuning, anchored.**
  - VPT's RL stalled without a KL term to the cloned policy [agent].
  - DAPG's decaying demonstration term gave up to 30 times better sample efficiency [agent].
  - Dreamer 4 keeps a KL to a behaviour prior [agent].
  - Large-VLA online RL was unstable on local hardware (iRe-VLA) [agent].
- **Selection by rollouts, not loss.** VPT found the training loss did not track downstream results [agent].

### 2.5 How does the library grow?

- **Zero-initialised additions.** VLA-Adapter's bridge gate and FLOWER's per-action-type modulation start at zero,
  and RT-1's FiLM starts at identity [agent]. Adding them leaves the policy's behaviour unchanged at the start.
- **Insulate the trunk.** "Knowledge insulation" stops the action expert's gradient into the backbone, which protects
  it and trains faster [agent].
- **Rehearse when the shared part trains.** Replay substantially reduces forgetting (Rolnick et al. 2019) [agent].
- **New tasks need data, not just prompts.**
  - Gato's prompting gave no significant gain; 500 demonstrations took a new task to 60% [agent].
  - BC-Z reached 44% zero-shot on unseen tasks [agent].
  - MineDojo's unseen task went from 1.3 zero-shot to 46.0 after RL fine-tuning [agent].

### 2.6 What generalisation to expect

- [I] Within a verb, across arguments, worlds and openings: this is what typed requests, relative frames, pointers
  and simulator variety buy, and it is where the cited systems succeed.
- [I] Across concurrent requests: one query per scope makes walk-while-looking a single forward pass.
- [I] Truly new verbs: expect partial zero-shot at best. A new verb needs a teacher or a data batch and a short
  fine-tune, which the catalogue's teacher-per-skill-family plan provides.

## 3. The recommendation

**The policy.** For each agent, each tick:

    E_t   = Trunk(O_t)                                  once per tick for all agents (freezing.encode_contexts)
    R_t   = [ verb_j, args_j (masked), mode_j, age_j ]  one token group per granted request j (<= 6); cached per request
    Q     = [ q_locomotion, q_gaze, q_hands, q_posture, q_abilities, q_interaction ]   learned scope queries
    Z_t   = Transformer([R_t ; Q ; H_t], cross-attending to E_t)                       H_t: the last K steps' summaries
    pi(a_t^h | .) = softmax(W_h Z_t[q_scope(h)])        for each head h of a granted scope; ungranted scopes stay neutral
    status_j = sigmoid(W_s Z_t[r_j]): done, stuck, P(success); progress_j                (the completion objective)

- The trunk is today's set encoder.
- Pointer arguments are marked observation tokens.
- Positions are in the agent frame, with Fourier features only if an ablation says so.
- K steps of attention history replace the GRU (AdA, HOVER's 25-step history).
- One batched forward covers all agents and all their requests, so the cost no longer grows with the number of
  skills.

**The recipe.**
1. *Teachers, the data engine.*
   - Privileged RL experts per skill family in the simulator (PPO on full state and the map, with the recipe-bench
     lessons).
   - Scripted planners, if the owner allows.
   - Native NPC traces for realism and a warm start.
2. *Distillation.* DAgger into the generalist from the legal observation. Every batch mixes verbs, and every batch
   samples argument masks and request ages. Relabel each rollout's achieved goals.
3. *Anchored RL.* PPO in the simulator, with L = L_PPO + beta_t KL(pi_distilled || pi_theta) and returns normalised
   per skill (PopArt). It is used where a teacher is weak, and checkpoints are selected by rollouts.
4. *Variety.* Procedural worlds, request combinations sampled with some held out, and curriculum by measured success
   (TRAINING_PLAN 2b).
5. *Growth.* A new verb means a new teacher, then continued distillation with rehearsal, a versioned release and
   regression suites. A skill marked frozen keeps strict bit-identity: a zero-initialised verb embedding and adapter,
   with no gradient into the trunk (the machinery of `freezing.py`).

**What carries over from today's code:** the surface, catalogue and contracts; the request protocol; the combiner
(it still decides grants); the resolver (pointers, agent frame); the trunk once per tick; freezing (for strict
skills); the simulator adapter; the benches.

**What changes:**
- The body: one generalist called once per agent per tick with all granted requests, instead of one body per
  request. That means a two-phase combiner step (grant, then one policy call).
- TRAINING_PLAN's algorithm: teachers, DAgger and anchored RL instead of PPO per verb.
- BUILD 3.2's "six policies, one per scope" becomes one policy with six scope queries.

## 4. Prove it before committing (cheap, in this order)

| # | Proof | Pass bar |
|---|---|---|
| A1 | Latency: one batched forward at 8 agents x 6 requests at the real token capacity, on this CPU, on a quiet machine | full per-tick path p95 <= 15 ms |
| A2 | Toy distillation on flat `npc_walk`: scripted walk_to and look_at teachers, then one generalist by DAgger | arguments honoured (E4's bars); walk-while-look composes; held-out argument combinations within 10 % of the seen ones |
| A3 | Growth: add a third verb by continued distillation with rehearsal, and separately by the strict frozen path | old verbs within tolerance (shared path) or bit-identical (frozen path) |
| A4 | Anchored RL on one verb whose teacher is imperfect | beats the distilled start without losing the other verbs |

Then TRAINING_PLAN's C-steps run on the generalist instead of on per-verb bodies.

Status (2026-10-18, late):
- The generalist is built (`generalist.py`, README 2.13).
- A1 passes: 3.56 ms p95 at batch 8 at load about 10 (15.08 ms at load 53 to 75 was the load).
- A2's two teachers exist (`teachers.py`), and so do its flat tasks (`simple_tasks.py`).
- The observation adapter is built (README 2.14). The teachers pass T1, T2 and T3, and walk_to passes T4 live (README
  2.15). A2's student training is next, after the owner sees these results.

## 5. Decisions for the owner (answered 2026-10-18, late; BUILD.md section 0)

1. **The request-conditioned generalist** is the working design. The owner gave the green light, on the grounds that
   neural skills exist to combine behaviours and be smarter than heuristics. Proofs A1 to A4 confirm or refute it.
2. **Scripted teachers** are approved for the simple track (navigation, gaze, aim). Complex families get their own
   teacher plans (TRAINING_PLAN 2e).
3. **The growth rule:** versioned releases with regression suites. Strict bit-identity only for skills the owner
   marks frozen.

## 6. Not adopted, and why

- **A web-scale VLM backbone or image tokenisers:** we have no pixels and a closed catalogue, and their speed is
  1 to 15 Hz on GPUs [agent].
- **Autoregressive or FAST token decoding:** about 750 ms per chunk for π0-FAST [agent].
- **Multi-step diffusion or flow heads in the tick loop:** 10 steps per action [agent].
- **World-model training at Dreamer 4's scale:** 256 to 1,024 TPU-v5p chips for 0.7% diamonds [agent]. A world model
  stays a later research objective (ADR-001).
- **Pure PPO per skill:** it failed here, and HOVER's multi-mode RL from scratch lost to distillation on all 32
  comparisons [read].

## References

Read this session:
- HOVER: He, T., et al. (2024). HOVER: versatile neural whole-body controller for humanoid robots. ICRA 2025.
  arXiv:2410.21229.
- MaskedMimic: Tessler, C., et al. (2024). MaskedMimic: unified physics-based character control through masked
  motion inpainting. SIGGRAPH Asia / ACM TOG. arXiv:2409.14393.
- π0.7: Physical Intelligence (2026). A steerable generalist robotic foundation model with emergent capabilities.
  arXiv:2604.15483.
- RLDG: Xu, C., Li, Q., Luo, J., and Levine, S. (2024). Robotic generalist policy distillation via reinforcement
  learning. arXiv:2412.09858.
- Lin, F., et al. (2024, rev. 2026). Data scaling laws in imitation learning for robotic manipulation.
  arXiv:2410.18647.
- ROCKET-1: Cai, S., et al. (2024). Mastering open-world interaction with visual-temporal context prompting.
  arXiv:2410.17856.
- Asynchronous semantic-action decoupling (2026). arXiv:2606.15285.
- TurboVLA: Xie, H., et al. (2026). Real-time vision-language-action model at 32 Hz. arXiv:2607.27205.
- NitroGen (2026). An open foundation model for generalist gaming agents. CVPR 2026. arXiv:2601.02427.
- SIMA 2 (2025). A generalist embodied agent for virtual worlds. arXiv:2512.04797.

Read by the research agents (links checked by them; numbers not replicated):
- AdA: arXiv:2301.07608.
- XLand: arXiv:2107.12808.
- SIMA: arXiv:2404.10179.
- Gato: arXiv:2205.06175.
- VPT: arXiv:2206.11795.
- STEVE-1: arXiv:2306.00937.
- MineDojo: arXiv:2206.08853.
- Dreamer 4: arXiv:2509.24527.
- Octo: arXiv:2405.12213.
- OpenVLA: arXiv:2406.09246.
- OpenVLA-OFT: arXiv:2502.19645.
- π0: arXiv:2410.24164.
- π0-FAST: arXiv:2501.09747.
- GR00T N1: arXiv:2503.14734.
- VLA-Adapter: arXiv:2509.09372.
- Knowledge insulation: arXiv:2505.23705.
- Helix: https://www.figure.ai/news/helix.
- ACT: arXiv:2304.13705.
- Diffusion Policy: arXiv:2303.04137.
- Perceiver IO: arXiv:2107.14795.
- AlphaStar: doi:10.1038/s41586-019-1724-z.
- Learning by Cheating: arXiv:1912.12294.
- Lee et al. 2020: arXiv:2010.11251.
- DAgger: arXiv:1011.0686.
- DAPG: arXiv:1709.10087.
- HER: arXiv:1707.01495.
- GCSL: arXiv:1912.06088.
- PopArt: arXiv:1809.04474.
- Policy distillation: arXiv:1511.06295.
- Replay for continual RL: arXiv:1811.11682.
- BC-Z: arXiv:2202.02005.
- iRe-VLA: arXiv:2501.16664.
