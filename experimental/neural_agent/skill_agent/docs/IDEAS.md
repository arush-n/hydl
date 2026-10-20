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
| Neural layer, every tick | Several skills at once on their own body parts: feet along a route out of the enemy's sight, eyes on the threat, hands firing when the shot is clean, posture low. Each sees the others' requests, so retreating while watching while taking a parting shot is one motion. It writes back what it finds: spotted, no clean shot from here, cover lost, path blocked. |
| Around the blackboard | A skill's "spotted" changes the decision layer's next step (fall back). The LLM marking "their scout is alone" changes the workflow's target, and the skills read the mark. A request changes what the skills do. No layer only gives orders and none only takes them. |

## Idea 0: every layer reads and writes one shared blackboard, in every direction (owner, 2026-10-19 night)

The owner, correcting this session's first reading (a top-down split of judgement by time scale): "the idea is through
the shared blackboard or information layer. all the models feed to and read from that layer so if the llm sees
something that it marks as important or the state that the neural layer feed back is key to the decision layer
information/instructions change; it is not one way rather it is multi directional."

The blackboard is S_t (`experimental/neural_agent/shared_state/`): one versioned document of six sections, each with
one writer, and one view per reader. The LLM writes `executive`, the decision layer `decision` and `requests`, the
skill layer `skills`, the belief module `belief`, the runtime `events`. Control is not a chain of command. The LLM
marking something important changes the decision layer's instructions; a skill reporting what it found does too; the
decision layer's requests change what the skills do; and around again.

Today the neural layer takes a small part. It writes request statuses (accepted, rejected, deferred, running, done,
aborted) to `skills`, and reads only its requests: the skills' view (`shared_state/views/skills.py`) is the goal
payload per lane. Idea 8 widens both.

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

## Idea 8: the neural layer on the blackboard, reading and writing

- **Reads**: the entries that matter for acting, beside the trunk's tokens and not only through requests: what the
  LLM marked (an entity, a place, a danger), the belief's last-seen positions, the active events, and the persona
  (idea 2).
- **Writes**: what it perceives and judges, beyond statuses:
  - events: spotted, exposed, target lost (and where), no clean shot, cover lost, path blocked, low on ammunition;
  - its calibrated chance of success and time left, from the status heads `BUILD.md` section 3.2 names.
- **The one-writer rule holds**: the skill layer writes `skills`. Reading the LLM's marks means widening the skills'
  view, which is the coordinator's file.

## A proposed order, if approved (architecture only; training stays paused)

1. The neural layer on the blackboard (idea 8): richer reports into `skills`, and the marks and the belief in the
   skills' view (the view with the coordinator).
2. Behaviour scalars in the request format and the module inputs, with the teacher knobs.
3. The scripted guerrilla set through the referee under a guerrilla program, on gen 7, with the skills' reports
   driving the program's switches.
4. The perception asks for cover candidates and geometry tokens.
5. Vague requests and capability cards, with the coordinator.
6. When training reopens: the core set trained together, style-randomized, on legal perception only.

## References

- Chen, D., Zhou, B., Koltun, V., Krähenbühl, P. (2019). Learning by Cheating. CoRL.
- Peng, X. B., Ma, Z., Abbeel, P., Levine, S., Kanazawa, A. (2021). AMP: Adversarial Motion Priors for Stylized
  Physics-Based Character Control. ACM Transactions on Graphics (SIGGRAPH).
- Peng, X. B., Guo, Y., Halper, L., Levine, S., Fidler, S. (2022). ASE: Large-Scale Reusable Adversarial Skill
  Embeddings for Physically Simulated Characters. ACM Transactions on Graphics (SIGGRAPH).
