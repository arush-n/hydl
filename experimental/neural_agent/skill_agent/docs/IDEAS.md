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
| The flows | The LLM marking "their scout is alone" changes the decision layer's workflow, which sends the skills the scout as their new target; the skills never read the mark. A skill's "spotted" reaches the decision layer at once (fall back) and the LLM as an event (rethink the approach). Requests go down; reports go up to both. |

## Idea 0: one shared blackboard; the LLM and the decision layer feed each other, the neural layer takes only the decision layer's requests and reports back to both (owner, 2026-10-19 night)

The owner, correcting this session's first reading (a top-down split of judgement by time scale): "the idea is through
the shared blackboard or information layer. all the models feed to and read from that layer so if the llm sees
something that it marks as important or the state that the neural layer feed back is key to the decision layer
information/instructions change; it is not one way rather it is multi directional." And then, on the shape: "the llm
feeds to the middle layer and those two feed to one another, the middle layer feeds to the neural layer and the neural
layer just writes back ... one way down to the neural layer but not up ... decision and llm feed to one another,
neural only gets the decision layer but feeds information back to llm and middle. think of the utility of the
information and where it is needed."

                 reads every section                         reads every section
    LLM  <------------------------->  BLACKBOARD S_t  <------------------------->  decision layer
          writes `executive`                ^    |              writes `decision`, `requests`
                                            |    | `requests` (the only section it reads)
                         writes `skills`    |    v
                         (its reports)  +---+----------+
                                        | neural layer |
                                        +--------------+

The owner: "the llm reads the blackboard and information on the blackboard so does the decision layer". The LLM and
the decision layer each read the whole blackboard through their own views (`views/llm.py` and `views/briefing.py` for
the LLM, `views/jev.py` for the decision layer), so the neural layer's reports reach both. The neural layer reads
only the `requests` the decision layer writes.

The blackboard is S_t (`experimental/neural_agent/shared_state/`): one versioned document of six sections, each with
one writer, and one view per reader. The LLM writes `executive`, the decision layer `decision` and `requests`, the
skill layer `skills`, the belief module `belief`, the runtime `events`.

Each piece of information goes where it is useful, in the form that reader can use, at the rate it needs:

| Flow | What | Why there | Rate |
|---|---|---|---|
| LLM to decision layer | goals, priorities, marks, the persona | the decision layer builds its workflows from the intent | the LLM's cadence |
| decision layer to LLM | how each goal was read, decisions, questions, outcomes | the LLM plans on what actually happened, and answers what the decision layer cannot settle | the decision layer's cadence |
| decision layer to neural layer | requests: verb and typed arguments (target, position, region, a remembered last-seen position), the style scalars, urgency | everything a skill needs beyond its own perception: what to do, on what, how | each decision; a re-goal on any tick |
| neural layer to decision layer | statuses, progress, chance of success, events (spotted, no clean shot, cover lost, path blocked) | the fast loop: switch, re-goal or give up on them | every tick |
| neural layer to LLM | the salient events and outcomes, summarised by its view | what changes a plan (pinned down, target escaped, a skill failing again and again), not tick-level telemetry | the LLM's cadence |

Today the neural layer takes a small part. It writes request statuses (accepted, rejected, deferred, running, done,
aborted) to `skills`, and reads only its requests: the skills' view (`shared_state/views/skills.py`) is the goal
payload per lane. Idea 8 widens what it writes, and what the decision layer passes down.

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
- **Growth** (owner: "5 names for now but in the future there should be many"): the names are an append-only list, as the verb ids are (`neural_skills/spec.py`); a module trained before a name existed reads it as 0 (neutral).

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

## Idea 8: the neural layer's inputs and reports

- **In, from the decision layer only.** A skill's inputs are its own perception and its requests; it never reads the
  LLM's section. Everything else it needs reaches it as request fields the decision layer chose to pass:
  - what to do and on what: the verb and its typed arguments;
  - remembered context: a last-seen position from the belief;
  - the persona's style scalars (idea 2), set by the LLM and forwarded;
  - urgency.

  The LLM's marks reach a skill only when the decision layer turns them into such a field. So the request format
  widens, not the skills' view.
- **Out, to both.**
  - To the decision layer, every tick: statuses, progress, calibrated chance of success and time left (the status
    heads `BUILD.md` section 3.2 names), and events: spotted, exposed, target lost (and where), no clean shot, cover
    lost, path blocked, low on ammunition.
  - To the LLM: the same reports, summarised by its view (`shared_state/views/briefing.py`) into the salient changes.
  - The skill layer writes only `skills` (the one-writer rule).
- **Inputs only, never text** (owner: "the neural layer has to feed from inputs so it cannot read natural language
  but the idea is there"). The LLM's words reach a skill only as typed fields:
  - the executive section already holds them beside the words (`GoalEntry`: the typed goal fields plus the LLM's
    words, `shared_state/sections/executive.py`);
  - the decision layer lowers them into request arguments;
  - a described target (idea 3) arrives as a vector from the text encoder and projection the registry uses for its
    own descriptions, so matching it against what the agent sees is arithmetic, not reading.

  Upward the same holds: the skills write events and numbers, and the views turn them into words for the LLM.

**First slice built (2026-10-19 night).**
- What: every verb's body reports `scripted_bodies.REPORTS` each tick: seen, distance, facing_me (the target's head
  within 60 degrees of the bearing to the agent), in_reach, acting. The granted actor carries them as
  `diagnostics["reports"]`, and `CombiningDecider` writes them into each granted request's status (`progress`) on
  `skills`, with no change to the blackboard's schema.
- Checked: in a COMBAT referee run every granted status carried them.
- Shown at once: with the legal view sector on, an evader that loses sight of the threat stands still (acting 0,
  seen 0). The remembered last-seen position in its request is what it needs.

## Idea 9: inputs as wide as legality allows, each module using what it needs

The owner: "we can expand the information that the model takes in, the more legal information the better, not every
neural layer will activate each input but if needed it should."

The architecture already allows it. The trunk attends over token sets, and each skill's module cross-attends to the
trunk's tokens (`model/generalist.py`). A module that never needs a channel learns to ignore it, and one that does
can find it, so a new legal channel costs a module nothing.

Two bounds:
- **Legality**: every input passes the legal-observation gate.
- **Time**: the per-tick cost stays within 30 Hz, measured whenever the schema widens (`bench/bench_perception.py`).

Candidates, beside what is read today:
- the remembered last-seen positions of what left the agent's sight;
- cover at range;
- block candidates (ask 4);
- fluid ids (ask 6);
- role ids (ask 1);
- the request's own remembered context (idea 8).

## A proposed order, if approved (architecture only; training stays paused)

**The owner's picks (2026-10-19 night):** build first the skills' reports (idea 8) and the guerrilla proof (idea 5);
the ideas stay ideas while they are prototyped, each moving into `BUILD.md` or `TRAINING_PLAN.md` once proven; the
behaviour scalars start with five names, and "in the future there should be many", so the list is append-only like
the verb ids; training reopens after the guerrilla proof, with the first cohort trained together.


1. The neural layer's reports (idea 8): events and calibrated chances into `skills`, routed per reader by utility;
   and the request fields the decision layer passes down (remembered positions, style, urgency), with the
   coordinator.
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
