# Ideas: the core layer that makes anything possible (2026-10-19, night)

Status: IDEAS, written down at the owner's request ("write those down as ideas"). Nothing here is approved or built.
An idea moves into `BUILD.md` or `TRAINING_PLAN.md` only on the owner's yes.

## The direction (owner, 2026-10-19 night)

On the bar for a neural skill: "the idea is not that it solely beats heuristics it is that it does it naturally and it
can combine multiple actions naturally and be able to really match the intent, the issue with heuristics is that they
are one dimensional and when instructions are vague or poor the model does not really adapt or know what to do
furthermore the idea is that the agent cannot cheat and heuristics to a degree have to cheat so by making the networks
more cooperative and capable the idea is that it can with proper instructions do anything that a human can do."

On the layers: the LLM does the thinking and the long-term behaviour. The controller (the decision layer) and the
neural layer control the agent itself. The decision layer, with proper context from the LLM, creates its own
workflows. The neural layer does the basics and combines them very well, so it enables whatever the decision layer
decides, and it behaves like a human, each agent unique: "one thing that we might need to implement is some scalars
that changes behavior a bit".

The owner's example: "the agent could use a guerrilla tactic so the agent should know how to effectively navigate the
environment, know where to hide, know when to shoot, when to retreat ... and compose them all together based on the
current input at a time so it flows seamlessly."

## The design test: guerrilla, layer by layer

| Layer | Its part |
|---|---|
| LLM | The intent and the persona: "harass their camp, hit and run, never get pinned"; a cautious, stealthy, moderately aggressive style. |
| Decision layer | The workflow it writes from that: reach cover near the target, wait for an opening, shoot, fall back to the next cover when spotted or hit. Its triggers: spotted, damaged, the target exposed. |
| Neural layer, every tick | Several skills at once on their own body parts: feet along a route out of the enemy's sight, eyes on the threat, hands firing when the shot is clean, posture low. Each sees the others' requests, so retreating while watching while taking a parting shot is one motion. |

## Idea 0: where each layer's judgement ends (this session's reading, for the owner to correct)

The layers split by time scale. The neural layer owns execution and micro-timing, ticks to about a second: when the
shot is clean, which step keeps the agent in cover, how to blend two requests. The decision layer owns what to do
next, over seconds: hide, shoot, fall back. The LLM owns why, over minutes and longer: the plan, the persona, the
workflow's shape. A skill never decides to start a fight; a program never aims an arrow.

## Idea 1: the bar is five properties, and heuristics never ship

Replaces H10's fallback ("a skill whose student does not beat its heuristic ships the heuristic",
`TRAINING_PLAN.md` section 1b) if approved. A neural skill is judged on:

1. **Intent**: under vague requests (no target named, an area instead of a point, a described target), does the
   outcome match what was meant?
2. **Combination**: several requests at once (walk + look + dodge, strafe + aim), scored on success and smoothness.
3. **Naturalness**: turn rates, gait changes, jitter and reaction times against native NPC traces (and human play,
   if recorded).
4. **Legality**: every reported number with the view sector and occlusion on; no privileged input.
5. **Robustness**: disturbances and unseen worlds (the existing suites).

Scripted bodies and privileged teachers stay, as teachers and baselines only. The plan already teaches with
privileged scripts and keeps the student on legal perception ("learning by cheating", Chen et al. 2019).

## Idea 2: behaviour scalars (a style vector)

- **What**: a few scalars per agent, z in [-1, 1]^k with k about 5. Proposal: aggression, caution, stealth, patience,
  precision.
- **Who sets them**: the LLM sets z as the agent's persona; the decision layer can nudge it per request ("careful
  here").
- **How it enters the network**: every skill module reads z. Its request token gains one more term (the generalist's
  request token, `model/generalist.py`):

      q = Embed_verb(verb) + Dense(args * present) + Embed_scope(scope) + T_entity[pointer] + T_cell[pointer] + Dense_z(z)

  z does not change perception, so the shared trunk stays frozen.
- **How it is learned**: each teacher's own knobs (reaction delay, distance kept, exposure tolerated, aim time) are
  drawn per episode as theta = g(z), with g a fixed monotone map per knob, and the student sees z. One network then
  learns the whole range of styles instead of one teacher's.
- **Natural at every setting**: a motion prior from native traces keeps any style human-like. Adversarial Motion
  Priors (Peng et al. 2021) train a discriminator D on (s, s') transitions from the traces, with a least-squares loss,
  and add a style reward:

      r_t = w_task * r_task + w_style * r_style,     r_style = max(0, 1 - 0.25 * (D(s_t, s_{t+1}) - 1)^2)

  Adversarial Skill Embeddings (Peng et al. 2022) learn reusable, controllable skills the same way, with the
  cross-entropy form r_style = -log(1 - D(s_t, s_{t+1})).
- **Uniqueness**: no two agents need the same z.

## Idea 3: vague requests

Today the combiner rejects an under-specified request: sword without a target, or keep_range without its band, comes
back `missing_argument` (`requests.validate`). For intent matching:

- arguments become optional, and a missing scalar takes a learned default instead of a rejection;
- a target can be described rather than named: a 64-wide description vector, matched by the module against the
  description vectors of what it sees. The registry already embeds every item and creature this way
  (`perception/registry.py`); creatures match only once the perception channel publishes role ids (the role seam).

The request format is shared with the decision layer, so this is a change agreed with the coordinator.

## Idea 4: train skills in cohorts

A module sees only the requests of skills it was trained beside (`model/skill_module.py`, the visibility rule). How
naturally skills combine therefore depends on training them together: the guerrilla core (locomotion, gaze, ranged,
cover) as one cohort, with vague and multi-request episodes in the data from the start, not only in the last
reinforcement stage.

## Idea 5: prove guerrilla with scripted teachers first

As the combat bodies were proved on 2026-10-19 (`runtime/scripted_bodies.py`; 11 of 20 fights won on gen 7):
- scripted take_cover, peek_and_shoot and retreat_to_cover bodies, driving the body through the referee;
- a guerrilla program from the decision layer (the coordinator's);
- gen 7, legal perception.

This proves the composition end to end before any network is trained, and the scripts become the cohort's teachers.

## Idea 6: perception for cover

"Knowing where to hide" means cover: places that break a threat's line of sight.
- Geometry tokens in the adapter (none today, `neural_skills/README.md`, code "terrain").
- Cover candidates from the simulator side, as block candidates were asked for (`PERCEPTION_ASKS.md`).
- The threats' last-seen positions from the belief, so the agent can act on an enemy it no longer sees.

The agent can already tell whether a creature faces it: the entity row publishes `facing_error` and
`head_facing_error`.

## Idea 7: capability cards, the decision layer's vocabulary

For the decision layer to write its own workflows, it needs to know what each skill does, when it applies and how
well it works. That is the capability card per verb `BUILD.md` section 3.4 already names: success and time to success
per environment family, preconditions, failure modes, calibration.

## A proposed order, if approved (architecture only; training stays paused)

1. Behaviour scalars in the request format and the module inputs, with the teacher knobs.
2. The scripted guerrilla set through the referee under a guerrilla program, on gen 7.
3. The perception asks for cover candidates and geometry tokens.
4. Vague requests and capability cards, with the coordinator.
5. When training reopens: the core set trained together, style-randomized, on legal perception only.

## References

- Chen, D., Zhou, B., Koltun, V., Krähenbühl, P. (2019). Learning by Cheating. CoRL.
- Peng, X. B., Ma, Z., Abbeel, P., Levine, S., Kanazawa, A. (2021). AMP: Adversarial Motion Priors for Stylized
  Physics-Based Character Control. ACM Transactions on Graphics (SIGGRAPH).
- Peng, X. B., Guo, Y., Halper, L., Levine, S., Fidler, S. (2022). ASE: Large-Scale Reusable Adversarial Skill
  Embeddings for Physically Simulated Characters. ACM Transactions on Graphics (SIGGRAPH).
