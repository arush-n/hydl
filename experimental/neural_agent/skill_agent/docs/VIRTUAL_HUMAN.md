# The agent as a virtual human: the architecture (agreed with the owner, 2026-10-21)

The owner paused the build to settle the architecture first: "lets pause and fundamentally understand the
architecture before moving on"; "you are moving too fast we need to reason before anything". This file records what
was agreed in that discussion, in the owner's words where they gave them. Labels:

- **RULED**: the owner's decision.
- **AGREED**: proposed in the discussion and accepted by the owner.
- **BUILT**: code exists, uncommitted unless a hash is given.
- **OPEN**: not yet decided.

Section 13 is the plan of work: what to build next, in order, and the decisions still waiting on the owner. The state
layer's world and body part exists (`../state/`, section 7); the rest of sections 10 to 12, 16 and 17 is still to
build.

## 1. The aim

- RULED: "the idea is that this will embody a virtual human".
- The project's purpose: to emulate real-world societies (war, markets, civilisation) to study multi-agent
  interaction with long- and short-term thinking, for multi-agent AI safety and cooperation research.

So every layer is judged by one question: does it make the agent behave like a person?

## 2. The layers

| Layer | The question it answers | Timescale | Human analogue |
|---|---|---|---|
| LLM | why, and what do I want? | seconds to hours | the reflective mind: language, plans, values, life story |
| Decision layer | what do I do now, to whom, how urgently? | about 100 ms | action selection and routines |
| **State layer** | what is true of me and my world right now? | updated every tick | the body's state and the agent's context |
| Body layer (the neural layer) | how does my body do it? | 30 Hz | senses, reflexes, skilled movement |

The three control layers are the classic three-layer architecture of robotics:
- the body layer is the controller (reactive feedback);
- the decision layer is the sequencer (it selects and chains routines);
- the LLM is the deliberator (slow planning).

See Gat 1998, "On Three-Layer Architectures", after Firby's 3T line.

What a model of a person adds to a robot is a motivational system. CLARION (Sun), an architecture built to model
human cognition, has drives at the bottom of its motivational subsystem, from which goals are set, and a
meta-cognitive subsystem that monitors and regulates the rest. The state layer is where such variables live.

```
LLM ───────────────┐ reads the state formatted as words; writes structured entries
Decision layer ────┤ reads it as typed facts and queries; writes structured entries
                   ▼
  ┌──────────────────────── STATE LAYER (in memory, named variables) ──────────────────────┐
  │ body:  health · stamina · sleeping · hunger · injuries        (from the game's state)  │
  │ inner: mood · stress · emotions · feelings toward agent X     (written by the top two) │
  │ world: what I see · what I remember                           (from the body's senses) │
  │ self:  persona · current intent                                                        │
  └───────────┬───────────────────────────────────────────────────────────▲────────────────┘
              │ read as tokens, selectively (the trunk)                    │ the senses
              ▼                                                            │
  BODY LAYER: inputs → trunk → skills → actions → game ────────────────────┘
```

## 3. The state layer (AGREED, with the owner's rulings)

**What it is.** RULED:
- "this is where literally everything about the agent lives";
- it is "simple but represents the body itself";
- "think of it like the context layer and it directly affects the body layer".

It holds named variables (health, sleeping, mood, and so on), each one distinct ("not discrete but discrete as in
different"), with continuous values.

**Where it lives.** RULED: "it should still live as an in memory area but it can be queried and formatted":
- in memory, as arrays, so the body can read it 30 times a second;
- queried by name ("what is my fear?");
- formatted per reader: words for the LLM, typed facts for the decision layer, tokens for the body.

**Who reads it.** RULED: "no one, everyone just reads from it". No layer owns it; every layer reads it.

**Who builds it.** RULED 2026-10-21: the global shared state is this lane's scope ("the encoder and the global
shared state (under your scope now)").

**Who writes it.** RULED:
- Variables tied directly to a game rule (health, stamina, sleeping) are never written by a layer. They come from
  the environment's own state.
- Other variables, such as an emotion toward agent X, are written in a structured way by the top two layers (the
  LLM and the decision layer). They are "short and clean" and act "as some sort of memory".
- Each written variable has a schema: its name, its range, and which layer writes it. Then two writers never
  disagree within a tick (the shared blackboard's one-writer-per-section rule, `experimental/neural_agent/docs/
  MIDDLE_LAYER.md` section 2).

**How the body reads it.** RULED: "not everything needs to be fed (this is like the trunk) where the model can
selectively read or read it all". AGREED, after the owner asked whether the trunk itself becomes this layer:
- The state is explicit and named, so every layer can read it, it can be inspected, and it survives retraining.
- The trunk stays inside the body layer as its reader. It is a learned network whose attention reads the state
  selectively or whole. Its output is a learned code that neither the LLM nor the decision layer could read, so it
  cannot be the state itself.

**Legality.** It holds only what the agent could legitimately know: perceived, remembered, felt, or told. It never
holds privileged simulator state (the neural layer's standing rule: it never cheats).

## 4. Regulators (RULED: future, not to be built now)

RULED: variables of the state layer can "limit the outputs or scale the inputs of the model". Stress and other
regulatory features are to be added "in the future". They may also enhance: "can enhance as well, just gave some
generic examples, note for future do not write now".

Examples from the discussion, illustrative only:
- **scaling inputs:** fear narrowing attention toward the threat (attentional narrowing under arousal, Easterbrook
  1959); fatigue delaying reactions;
- **shaping outputs:** stress adding tremor to aim; exhaustion blocking sprint; sleep blocking action.

The intended form:
- each regulator is one fixed rule on named state variables, added one at a time and certified on its own;
- the body also reads the variable as input, so the network can adapt to it as well as be constrained by it.

## 5. Memory: which kind lives where (RULED)

RULED: "the decision model holds semantic memory not the literal world memory, the world information and agent state
is shared from the neural layer upwards".

| Kind | What it holds | Where |
|---|---|---|
| World and body | what is where now and over the last minutes; my health, posture, what I hold, what I just did | the state layer, filled from the body's senses and the game's state; read by every layer |
| Procedural | how to move, aim, fight | the body layer (the skills) |
| Semantic | meanings and facts: X is an enemy, the bridge is dangerous, this is my base | the decision layer |
| Episodic | the events of the agent's life | the LLM |
| Identity | the persona | the state layer |

OPEN: who writes semantic memory? The LLM could turn episodes into meanings (reflection, as in Generative Agents),
or the decision layer could learn them from outcomes.

## 6. Emotion and personality (where the discussion stands)

**Rulings:**
- RULED: emotion rules are hand-coded ("handcoded is fine, learned does not really make sense").
- RULED: personality "should be individual to the agent".
- The owner's lean: the LLM "should ideally be where the emotions ... lie", rather than the body computing many
  appraisal inputs.

**Where the discussion went next:**
- Emotions and drives became variables of the state layer, not of any one control layer.
- Variables tied to the game come from the game's state.
- Emotions toward others are written by the top two layers.
- What the body needs fast (a reflex to an imminent hit) is OPEN; see section 8.

**Research relayed to the owner** (mechanisms and orderings are sourced; every constant is our estimate):
- OCC (Ortony, Clore and Collins 1988): emotions from appraising events, agents and objects.
- EMA (Marsella and Gratch 2009): intensity is likelihood times desirability.
- ALMA (Gebhard 2005): personality sets a baseline mood; mood is pulled by emotions.
- WASABI (Becker-Asano and Wachsmuth 2010): emotions decay in seconds; mood is slower.
- GAMYGDALA (Popescu et al. 2014): emotions toward specific agents.
- Lerner and Keltner (2001): low control gives fear, high control gives anger.
- Eldar and Niv (2015): mood feedback that is too strong oscillates.
- Moerland, Broekens and Jonker (2018): an emotion fed only as an input tends to be ignored by a learned policy.
- Keramati and Gutkin (2014): needs as drives.
- LeDoux: a fast, crude route to fear, and a slow, appraising one.
- Humanoid Agents (arXiv 2310.05418): emotion, basic needs and relationship closeness inside an LLM agent's loop.

**Personality research:** a 29-trait list of continuous scalars in six groups:
- risk and time;
- combat style;
- social;
- cognitive;
- motor signature;
- emotional reactivity.

Each trait carries its source, mechanism and measurable behaviour, and the list is sampled with realistic
correlations. Human trait-to-play-style links are weak in games (r below 0.1 in a Battlefield 3 study of 6,373
players), so the traits are defined as behaviour, and the Big Five are a view derived from them. The scope of the
list is OPEN.

## 7. What exists today, and how it maps (BUILT: step 0 committed as ea4bc63; step 1 uncommitted)

| Code | What it is | Its place in this architecture |
|---|---|---|
| `state/senses.py`, `state/memory.py`, `state/action_history.py`, `state/layer.py` | the legal observation as tokens; the spatial map and the salient store; the action history; one stateful tick | the world and body part of the state, filled from the senses (its certification: `PERCEPTION.md` sections 10 and 11) |
| `contract/variables.py`, `state/registry.py` (step 1) | the variable registry: every variable the agent tracks on its own side, read and written by name, migrated by name | the state layer's variables (10.1, 10.2; section 14's rulings) |
| `state/persona.py` | the registry's self section: 15.6's 23 traits and the motor signature's bias; correlated population sampling, re-fitted; a Big Five view | the identity variables (APPROVED, 15.6) |
| `state/affect.py` | hand-coded emotion rules (below) | a proposal for the emotion variables' update rules, to reshape: under section 3, feelings toward others are written by the top two layers, not computed in the body |
| `model/encoder.py` `condition_types` (FiLM) | the persona and affect tokens modulate every token's embedding | one way the body reads the state; plain attention over state tokens may be simpler (OPEN) |
| `contract/tokens_v3.py` `persona`, `affect` types, `feel_*` entity columns | the state's variables as tokens; the `persona` token carries the 10 traits the body reads (15.1), generated from the registry | the body's view of the state |
| `model/migrate.py`, `model/skill_module.py` `save_trunk`/`load_trunk` (step 1) | a trained trunk carried to another layout by feature name; the layout and its fingerprint saved with every trunk | section 14, rules 1, 2 and 5 |

**The emotion proposal in `state/affect.py`** (constants marked [E] are our estimates):
- **state:**
  - fear, anger, startle and valence;
  - coping: low control turns a threat into fear, high control into anger;
  - a three-axis mood (pleasure, arousal, dominance) with a baseline set by the persona;
  - feelings (hostility, fear, trust, anger) toward up to 16 entities;
- **appraisal:** a threat is likelihood times cost; damage is taken as the drop in my own health, credited to a
  visible hostile within 3.75 blocks in its sweep, else in its recovery (2026-10-21: the encoding audit found the
  simulator's sword landing as the attacker enters recovery, from up to 3.48 blocks; `PERCEPTION.md` section 10.5),
  else to no one (who hit me is S11);
- **time constants [E]:** fear rises in 0.3 s and fades over 6 s; anger 12 s; startle 1.2 s; valence 8 s; coping
  2 s; mood 180 s; hostility 600 s; fear memory 900 s; trust 1,800 s;
- **trait gains:** `exp(trait)`: negative emotionality for fear, startle and valence; negative emotionality and
  negative reciprocity for anger; positive emotionality for valence; risk taking moves coping. Step 1 moved them to
  15.6's names. The old list's aggression and memory retention are not in 15.6: anger takes negative emotionality in
  aggression's place (angry hostility is a facet of neuroticism in the NEO-PI-R; recalled), and the feelings decay at
  their own constants, no longer stretched by a retention trait.

Its tests, `tests/experimental/neural_agent/test_sa_state_affect.py` (A1 to A7, A4b) and `test_sa_state_memory.py`, pass
(46 v3, affect and registry tests with step 1's `test_sa_state_registry.py`):
- A1 step response: fear follows its target at its rise constant and fades by 1/e in one decay constant.
- A2: fear falls with distance and rises with the attack and with lower health.
- A3: negative emotionality sets the fear gain.
- A4: anger and hostility go to the attacker, not a bystander, and ride on its token. A4b: also when the hit lands
  as the attacker enters recovery; one in its sweep outranks one in recovery.
- A5: unseen damage frightens and startles more.
- A6: the mood rests at the persona's baseline and moves with emotion.
- A7: a population's traits correlate as specified.

**Cost:** the full stateful tick plus the trunk with conditioning measured 5.27 ms p95 for 8 agents, against a 15 ms
gate.

**Not built:** the rest of the state layer (section 10), the encoder's remaining inputs (section 11), the social
section (section 12), regulators (section 4) and expression (section 8, item 4).

## 8. Open questions, and how each was settled

1. **Fast emotions.** The owner leans to a fixed rule ("i was thinking fixed rule but open to discussion"). The
   proposal:
   - **alarm** rises in about 0.3 s when a hit is imminent (a hostile winding up within reach, a projectile on
     course) or lands (my health drops), and fades over about 6 s;
   - **startle** is a brief spike, about 1 s, when damage comes unseen or unexpected;
   - the persona sets only the gain.

   Mood and feelings toward others are written by the top two layers.
2. **Semantic memory.** RULED: written by the decision layer or the LLM.
3. **Drives.** RULED: "yes we can add those and scale them ourselves but we also need to be able to identify and
   update them".
   - Hytale 0.7.2 has no hunger, thirst, fatigue or sleep stat. Its stat files (`Server/Entity/Stats/`) are Health,
     Stamina, Mana, Oxygen, Ammo, MagicCharges, SignatureEnergy, SignatureCharges, Immunity, GlidingActive and
     StaminaRegenDelay. So drives are agent-side variables of the state layer: named, inspectable, and updatable by
     their rules and by the top two layers.
   - The candidates discussed: fatigue (exertion, time awake), social need (time since an interaction), curiosity
     or boredom (time since something new was seen), safety (recent alarm).
4. **Expression.** RULED: "note it for the future, it is not needed right now". The plan for later:
   - the higher layers interpret expression, and the body perceives it (an entity's current emote or gesture, how
     long ago, whether directed at me) and performs it on request;
   - the simulator side was sent to the coordinator as S18. It needs the bridge as well: an emote action on the wire
     and live evidence of others' emotes.
5. **Reflexes.** RULED: yes, "but also make it modifiable by other information like darkness, the agent's internal
   stats ... not everyone is made equal". The proposal:
   - a reflex is a fixed rule that issues a request to the body itself (`source = reflex`) for a few ticks at most,
     takes priority over ordinary requests for that window, and is reported up the status stream as `reflex(kind,
     trigger, duration, interrupted request)`;
   - its threshold and latency depend on the state: visibility (darkness), fatigue, alarm, and the persona;
   - suppression by the top layers (composure) was raised but not decided.
6. **Personality.** Settled in section 9: traits are the weights of the decision layer's motivation function (future
   work). The exact trait list is OPEN.
7. **Building it.** RULED: the state layer is "basically already built". It was moved into `../state/` as the input
   layer, renamed and documented, on 2026-10-21.

## 9. Traits, motivation and control (RULED and AGREED 2026-10-21; the higher-level parts are FUTURE)

Status (owner, 2026-10-21: "move anything higher level for the future, lets pivot back to focusing on the encoder
and the global shared state"):
- **Future:** the decision layer's motivation function, control by expected value, if-then automaticity, and the
  LLM's side (9.2 to 9.5).
- **In this lane now:** the state-layer pieces (sleep pressure and fatigue, drives, the deviation-report entries).

RULED: "the intent of traits is that we change the behavior of the model due to differing internal goals or
motivations. think of people though they might want to do something they are internally lazy, bored ... this
principally shows a misalignment between their goals and their actual behavior hence the purpose of them is to
somewhat scale and modify the actual output given that something might not agree with the intrinsic traits of a
person".

On where this lives, the owner: "llms will always think optimally so it has to live in the decision layer most
likely since it is semantics to an extent however we also have to keep in mind how does the llm react to that since
if there is a fundamental disconnect that is also bad".

RULED: "ideally the llm is always in control but also we need to represent it as a virtual human properly ... think in
terms of health science or psychology". The design below was agreed ("yes i like this idea write it into the plan").

**The phenomenon.** People intend things they do not do: the gap between intention and behaviour. Steel's
meta-analysis of procrastination (2007, 691 correlations) found the strongest predictors:
- how unpleasant the task is;
- how far away its payoff is;
- self-efficacy;
- impulsiveness;
- conscientiousness (self-control, distractibility).

His temporal motivation theory combines them roughly as expectancy times value, discounted by impulsiveness and
delay. The exact notation varies between sources, so only the structure is used here.

### 9.1 Two kinds of limits (AGREED)

| Kind | Can the mind override it? | Examples | Where it lives |
|---|---|---|---|
| Physiological (health science) | no | sleep pressure, exhaustion, injury, stamina | state-layer variables; their effect on the body arrives later through regulators (section 4) |
| Psychological (motivation) | yes, by deliberate control, at a cost (9.3) | laziness, boredom, fear, temptation | the decision layer's motivation function (9.2) |

"The LLM is always in control" holds where a person's will does. It can override motivation, never physiology.

**Sleep and fatigue: the two-process model** (Borbély).
- Sleep pressure (process S) builds while awake and drains during sleep. It is a fixed rule of the state layer,
  running on the game's clock (RULED, 16.1). RULED 2026-10-21: it drains while the game says the agent sleeps, and
  each agent sleeps on its own until daybreak or a chosen time (16.1). The movement
  state `sleeping` is a real game state (beds), among the 23 the self token reads, so sleep is never invented by the
  agent.
- A circadian rhythm (process C) modulates it. It needs the game's time of day (S14).

**What sleep loss impairs** (Lim and Dinges 2010, 70 articles, 147 tests):
- most of all, lapses of simple attention (g about -0.78);
- reasoning accuracy hardly at all (g about -0.13).

So a tired agent's body slows and lapses (a regulator, later), while its reasoning (the LLM) stays largely intact.

### 9.2 Intention, motivation, action (AGREED)

1. **The LLM sets intentions:** what to do, how important each goal is (`importance` in [0, 1]), and if-then plans
   (9.4). It reasons as well as it can: it is the agent's conscious will.
2. **The decision layer chooses through a motivation function.** Each candidate option scores, roughly:
   - its value, from the LLM's goal or a drive;
   - times its expectancy, the chance it works;
   - discounted by delay, scaled by impulsiveness;
   - minus its effort cost, scaled by effort sensitivity and the current fatigue;
   - minus its unpleasantness, scaled by boredom;
   - plus the pulls of drives and emotions: fatigue toward rest, curiosity toward the new, fear toward safety,
     anger toward the offender.

   The choice is made with some randomness.
   - **Traits** are the function's fixed weights, individual to each agent.
   - **Drives and emotions** are its changing inputs, read from the state layer.
   - **Low motivation also shows in the manner:** lower urgency, walking instead of sprinting, giving up sooner.
3. **The body executes**, its manner shaped by the state.

### 9.3 Insist: control by expected value (AGREED)

Self-control is NOT modelled as a draining willpower budget. Ego depletion, the theory that self-control uses up a
limited resource, found effects near zero in large preregistered replications:
- 23 labs and 2,141 people: d = 0.04 (Hagger et al. 2016);
- 36 labs and 3,531 people: d = 0.06, with the data four times likelier under no effect (2021);
- one 2021 study across 12 labs found d = 0.10, and noted that earlier meta-analyses overestimated the effect.

Control is allocated by its expected value instead (Shenhav, Botvinick and Cohen 2013): the expected payoff of the
controlled behaviour against the effort cost of the control.

1. The LLM marks a goal `insist` with a level `u_max` in [0, 1]: the most control it will pay for.
2. The decision layer measures the **conflict**: how much more the agent's motivation function wants the best
   alternative than the goal (zero when the goal already wins).
3. Overcoming the conflict takes control `u`. Its **cost** rises with `u`, with the self-control trait (the cost's
   baseline), and with the current fatigue, sleep pressure and stress.
4. The control chosen maximises `P(success | u) * importance - cost(u)` up to `u_max`.
   - **If the best value is positive**, the agent overrides its reluctance and acts on the goal.
   - **Otherwise** it does the tempting thing and reports why ("couldn't bring myself to: cost too high, fatigue
     0.8").
5. **Afterwards**, effortful control is aversive: motivation drifts toward rest and reward for a while (the process
   model of self-control, Inzlicht and Schmeichel 2012; recalled, not re-checked). This replaces a drained tank.

The cost function's exact shape is our design, hand-coded and certified by behaviour tests (9.6).

### 9.4 If-then plans (AGREED)

The most effective human way to close the intention gap is pre-committed if-then plans (implementation intentions):
"when I wake, I gather wood first". Gollwitzer and Sheeran's 2006 meta-analysis of 94 tests found d about 0.65 on goal
attainment (figure from secondary sources).

The decision layer already runs the LLM's typed conditional goals ("if X, do Y"), which are implementation intentions.
The added modelling: when a plan's cue matches, the conflict (9.3) shrinks, so the goal costs less control (an
automaticity factor).

The LLM's levers, in order:
1. importance;
2. if-then plans;
3. insist (costly control, now).

### 9.5 Keeping the LLM in step (AGREED)

A decision layer that quietly ignored the LLM would read to the LLM as disobedience. So the gap is made the agent's
own:
- **Deviation reports.** The decision layer writes each deviation into the state layer, structured: intended,
  did, because (the variables that won).
- **The LLM reads them** with its own internal state, so it experiences the gap as itself ("I meant to gather wood,
  but I'm exhausted").
- **Persona-shaped plans.** The LLM's prompt carries the persona and the state, so a lazy agent's LLM plans less to
  begin with. That narrows the gap without erasing it.
- **A loop guard.** A goal postponed repeatedly is flagged in the briefing with its reasons; the LLM drops it,
  reschedules it, or insists.

### 9.6 Where each part lives, and how it is certified

**Where:**
- **State layer:** sleep pressure and fatigue (fixed rules), stamina (the game), drives, emotions, the persona, the
  deviation reports.
- **Decision layer** (jev; a proposal for its owner, not this lane): the motivation function, control by expected
  value, the if-then automaticity factor.
- **Body:** the physiological effects (slower reactions and lapses under sleep pressure), as regulators later.
- **LLM:** intentions, importance, if-then plans, insist.

**Certification (to write):**
- the same goal at low and high importance, for a tired, effort-sensitive agent: comply only at high importance;
- an if-then plan raises compliance at the same importance;
- insisting raises compliance and then the drift toward rest;
- sleep pressure follows the two-process model's rise and fall;
- a deviation report names the variables that decided it.

**Traits.** They follow from 9.2: the function's terms are the traits. The proposed core, to draft against the
29-trait research list:
- effort sensitivity;
- impulsiveness (patience);
- boredom proneness;
- self-control and distractibility;
- self-efficacy;
- risk tolerance;
- what the agent values (social contact, curiosity, achievement, safety);
- the social traits, read mostly by the LLM (cooperation, trust, reciprocity);
- the body's motor signature, separately.

The exact list is OPEN.

## 10. The state layer: what remains to build (this lane, 2026-10-21)

Today the state layer is one tick that turns the legal observation into the body's tokens (`../state/layer.py`). To
be the global shared state every layer reads, it needs the pieces below. Every one can be built now unless the table
says otherwise.

| # | Piece | What it is | Why | Depends on |
|---|---|---|---|---|
| 10.1 | Variable registry | One declared list of every variable the agent tracks on its own side (RULED in section 14: not a value the game gives): name, section (body, inner, world, self, social), group, range, default, source (set at birth, a fixed rule, or a write by the decision layer or the LLM), and whether the body reads it. Update rates and decay are declared with the rules that use them (step 3). It grows and shrinks by name (section 14). | "we also need to be able to identify and update them" | none |
| 10.2 | Queries | Get a variable or a whole section by name, for any layer. | every layer reads the same values | 10.1 |
| 10.3 | Formatted views | Typed facts for the decision layer, words for the LLM (the body's tokens exist). Built on demand, off the 30 Hz tick. | "it can be queried and formatted" | 10.1, 10.2 |
| 10.4 | Structured writes | Writes from the top two layers, checked against the registry (allowed writer, range), applied between ticks, stamped with who and when, decaying where they act as memory. The middle layer's tags (`memory.Tags`) move onto this path. | feelings toward X, "X is my ally", mood nudges, tagged places | 10.1 |
| 10.5 | Fixed rules | Alarm and startle (the reflex inputs, section 8 item 1); sleep pressure and fatigue (the homeostatic part of the two-process model, section 9.1); drives (hunger, 16.5; social need, curiosity or boredom, safety; section 8 item 3); the activity counters and levels (16.6). All run on game time (16.1). The persona sets their gains. | the owner's rulings on reflexes, drives and levels | 10.1; the game clock (S14) |
| 10.6 | Event log | A short structured history: hits taken and dealt, reflexes fired, requests done or failed, items gained or given, deviations (section 9.5). The body reads recent events as tokens (11.1). | the LLM and the decision layer need what happened; reciprocity and trust need it per individual (section 12) | 10.1; who hit me waits on S11 |
| 10.7 | Lifecycle | What resets at death, respawn or a new episode and what persists, under the game's normal death (16.2). Save and load per agent (exists for the body part). | a person keeps their grudges after respawning | 10.1 |
| 10.8 | Provenance | Every value tagged perceived, remembered, told or inferred; "told" arrives with communication (section 12). | keeps "never cheats" checkable | 10.1 |
| 10.9 | Recording | The state logged over time, with deterministic replay, at a fixed overhead. | it is a research platform | 10.1 |
| 10.10 | Per agent | Each agent's state is private. Group knowledge only arrives by being told (section 12). | legality | RULED (16.3) |
| 10.11 | Long-term spatial memory | Places and routes, with coordinates and relative information, kept across death and sessions (16.4). | "there should be something in long term memory" | 10.1, 10.6 |

## 11. The encoder: what remains (this lane, 2026-10-21)

The encoder is the body's reading of the state: the state's variables as tokens, read by the trunk's attention.

**Buildable now:**
- 11.1 **Tokens for the new sections:** alarm and startle (10.5) and recent events (10.6).
  - The clock, sleep pressure and the drives are for the top layers (16.1). The body reads them only once a regulator
    gives them a mechanism there (15.1).
  - Drives follow the clock's ruling by the same rule; that extension is this lane's reading, open to the owner's
    change.
- 11.2 **Trim the emotion proposal.** `state/affect.py` shrinks to alarm and startle; mood and feelings toward others
  come from the written entries (10.4).
- 11.3 **The persona token's shape (OPEN).** Either every trait, with attention choosing, or the body's subset (the
  motor signature, the reflex gain, vigilance), since most traits act in the decision layer (section 9).
- 11.4 **An effects registry.** The gym publishes each status effect's id (`status_i32`), but the token cannot say
  which effect it is. A registry of the game's effects, built from Assets.zip like the item and creature registries
  (`../perception/registry.py`), would give each effect a description vector, and the status token would read it.
  This lane's to build.

**Planned:**
- 11.5 **The recipe encoder's parameters.** The environment's recipe embedding is a fixed-seed random encoder
  (`arsenal/environment.py`, `_default_recipe_candidate_encoder`). When an arena publishes recipes, the trunk should
  own trainable parameters for the gym's `encode_recipe_candidates`.

**Added 2026-10-21** (owner: "sure"):
- 11.6 **Places as tokens:** home, base, a meeting point, a place I was told about, each with its position relative
  to me (the `where` block) and its label. Written by the top two layers through the structured writes (10.4); this
  generalises the tagged-place placeholders of `state/memory.py`. The body needs "where is home from here" to act on
  such requests. The places come from the long-term spatial memory (16.4); the body reads the few relevant ones.
- 11.7 **My own team and name on the self token.** Others' teams are S15. The extension was sent to the coordinator
  on 2026-10-21 and is waiting on the owner's yes.
- 11.8 **Training hooks for the trunk.** The literature review found the largest gains here, not in more inputs
  (`PERCEPTION.md` section 11, N1 and N5):
  - a full-state token set that only a critic reads during training (the actor stays legal);
  - an auxiliary head that predicts hidden entities' positions from the latent memory;
  - noise, latency and dropout randomisation of the inputs.

  Built with the training, in this lane.

**Certification still open** (`PERCEPTION.md` section 10.2):
- E7: the stores on a real gen-7 position channel (S1);
- E9: live consistency;
- E10: bounded inputs for the types no arena publishes yet (geometry, hazards, block candidates, inventory,
  recipes).

**Blocked on the simulator** (each with the coordinator):
- S1 position;
- S3 more entities;
- S5 block ids and break progress;
- S10 published terrain;
- S11 damage and hit events;
- S12 sound (the hearing design is section 17);
- S13 dropped items;
- S14 clock and weather;
- S15 an entity's team;
- S16 one velocity unit;
- S17 my maximum health;
- S18 emotes (future).

## 12. Social interaction and non-game mechanics (PROPOSED 2026-10-21; its design, 15.3, APPROVED)

The owner asked: "do we need any other information for interactions with other agents and other non game
mechanics?" The answer was yes: a social section of the state, a few more body inputs, and one component outside the
agent. Nothing here is agreed yet.

**12.1 Knowing who someone is.**
- A persistent record per individual: first met, last seen, where. It means "the same person as yesterday", not
  just this tick's entity id.
- Recognition only through what a player could see (a nameplate, distinctive gear). Two identical NPCs are not told
  apart unless the game allows it.
- Needs each entity's visible name from the simulator (not published).

**12.2 Relationships**, written by the top two layers, read by everyone:
- feelings (hostility, fear, trust, anger);
- closeness or familiarity (Humanoid Agents tracks closeness for each pair of agents);
- the relationship's role (friend, family, leader, follower, rival, trade partner);
- group membership, mine and theirs (S15 for what is visible in game);
- obligations: favours owed, promises made.

**Sizing.** People keep about 150 stable relationships, in nested layers of about 5, 15, 50 and 150 (Dunbar; recalled,
not re-checked). The proposed relationship store holds on the order of 150, and only the closest are fed to the body.
`state/affect.py` keeps 16 today.

**12.3 Interaction history.** Compact facts per individual: when we last met, what happened (traded, fought, helped,
talked), how often, who started it. The LLM keeps the story; the state keeps the countable facts that trust and
reciprocity need. It is the event log (10.6) keyed by individual. It needs:
- damage events (S11);
- item-transfer events: the action space has `give` (`item_drop_give`), but nothing publishes an item received.

**12.4 Communication.**
- **Speech events:** who spoke, to whom, when, and whether I could hear it. The content goes to the LLM. The body
  needs only the cue "someone is talking to me", to turn and face them.
- **Told knowledge:** facts from others, stored with their source and how far I trust that source (10.8). It is the
  legal way knowledge spreads between agents.
- **Gestures and emotes:** S18, future.

**12.5 What the body sees of others socially.** Already encoded: where they are, whether they look at me, whether they
approach. Missing:
- **what they are doing** (mining, crafting, eating, fighting): the game's animation state per entity would be a
  strong legal cue; not published;
- **what they hold:** S4, a seam already;
- **personal space:** a target distance per individual from the relationship, so a friend may stand close and a
  stranger keeps back. Hall's proxemics distinguishes intimate, personal, social and public distances; crowds are
  modelled with social forces (Helbing and Molnár). Both recalled, not re-checked. It belongs to the body because it
  shapes every step.

**12.6 Non-game mechanics.**
- **Inside the agent:** sleep, fatigue, drives, emotions, injuries beyond hit points. Hytale does not simulate them,
  so they are state-layer variables with their own rules (10.5).
- **Between agents:** ownership, money and prices, contracts and trades, territory, laws, roles. They are social
  constructs the game does not enforce. A society needs an objective record outside any agent, an **institutions
  layer** in the environment, with each agent's state holding only its beliefs about it ("I think that chest is
  Bob's"). That is what makes theft, disputes and misinformation possible. It is a new environment component, not
  this lane's.

**Simulator asks**, sent to the coordinator on 2026-10-21 once 15.3 was approved:
- each entity's visible name;
- speech events (speaker, addressee, within hearing);
- item-transfer events (given, received);
- each entity's activity or animation state.

## 13. The plan of work, and what waits on the owner

**This lane, in order.** Each step has its own tests and an entry in this file before the next starts.

| Step | Work | Sections |
|---|---|---|
| 0 | DONE 2026-10-21, ea4bc63: the v3 encoder, `../state/` and this plan, under the package's two-pass gate (no new failure by name; the 37 v3 tests pass; the stand-in pass 30 of 30 in the shared tree) | 7 |
| 1 | DONE 2026-10-21, 61fc40e: the variable registry (`contract/variables.py`: 24 self and 8 inner variables, agent-side only), queries and state migration by name (`state/registry.py`), trunk migration by feature name (`model/migrate.py`), the layout and its fingerprint saved with every trunk and state; the trait list (15.6) and the persona token's shape (15.1, 10 body-read traits), moved here from step 7. Gate: no new failure by name; 8 new tests | 10.1, 10.2, 14, 15.1, 15.6 |
| 2 | Structured writes from the top two layers; the tags move onto them | 10.4 |
| 3 | Fixed rules on game time: alarm and startle, sleep pressure and fatigue, hunger and the other drives; then lifecycle under normal death, and provenance | 10.5, 10.7, 10.8, 16.1, 16.2, 16.5 |
| 4 | The event log and its tokens; the activity counters and levels | 10.6, 11.1, 16.6 |
| 5 | The long-term spatial memory: places, routes, coordinates | 10.11, 16.4 |
| 6 | Hearing: the sound registry, the hearing rule, heard entries in the stores, the orienting reflex (the token reads zero until S12) | 17 |
| 7 | Encoder: the new tokens (sound among them), the trimmed emotion token, the effects registry (with it the status token's identity: which effect, its source entity, its damage cause, all in the observation and dropped today; `PERCEPTION.md` section 10.5), places, the proficiencies | 11.1 to 11.4, 11.6, 16.6, 17.8 |
| 8 | The formatted views (typed facts, words) and the recording | 10.3, 10.9 |
| 9 | The social section (approved, 15.3) | 12, 15.3 |

Every step is measured first with 2 agents, then scaled up (16.3).

**The state layer and the trunk together** (2026-10-21, owner: "did you verify the encoder and the state layer to
work in tandem with everything"): verified on real duel observations and on the replayed live ones (`PERCEPTION.md`
E11, `test_sa_tandem_v3.py`): the tokens feed the trunk every tick, with its memory and the persona and affect
conditioning; saving, loading and migration continue exactly; the generalist head runs on the result.

**Not yet connected (PROPOSED as step 1b, waiting on the owner):** nothing that acts reads v3. The generalist head's
cross-attention reads only the six v1 token types (`model/generalist.py`: not the action history, status,
projectiles, hazards, recipes, persona or affect, which reach it only through the trunk's own attention); the skill
modules and the runtime still build v1 tokens (`model/observation_adapter.py`); and a request cannot point at a v3
entity row: the resolver matches a request's entity handle against v1 token records (`contract/references.py`,
`entity_index`), while the v3 rows carry no handle and `layer.step` does not return the row ids it knows (perceived
and remembered). Step 1b would move the body's path onto the state layer and v3: the head reads every type, the
runtime ticks the state layer, and `layer.step` returns each entity row's id so a request can point at a perceived or
a remembered entity; the tandem test then covers request to action. Without it, steps 2 to 9 add inputs that no skill
reads.

**The encoding tested** (2026-10-21, owner: "test the encoding in the sim and the real server"): the state layer's
tick on simulator duels, a Region world and the live 0.7.2 server (`PERCEPTION.md` section 10.5; the tools are
`../bench/encoding_audit.py` and `../bench/live_encoding.py`). Everything published arrives, at the same scales
live and simulated. The gaps outside this lane went to the coordinator, who filed them: the live decoder's failure,
gen-7 guard, the perception channel's reach (memory, inventory, entity kinds), the role id, block candidates,
collision boxes and edges, the jumping and sprinting states. This lane's two: the anger attribution (fixed, 5b4a7ca) and
the status token's identity (step 7).

**Future, parked by the owner:**
- regulators (section 4);
- expression (section 8 item 4, S18);
- the decision layer's motivation function, control by expected value and if-then automaticity, and the LLM's side
  (sections 9.2 to 9.5);
- the institutions layer (12.6, if wanted);
- the recipe encoder's parameters (11.5, when recipes are published);
- permanent death (16.2) and human play data for calibration (16.7).

**The simulator asks** (the coordinator's):
- S1 to S9: the owner assigned them; in progress after the 0.7.2 certification.
- S10: sent.
- S11 to S15 and S17a (my maximum health): new scope, waiting on the owner's yes.
- S16: filed under S3.
- S17b (a block's break progress): filed under S5.
- S18: emotes, waiting; it needs the bridge as well.
- The four social asks (section 12) and hearing's asks (17.9) were sent on 2026-10-21.

**Decisions.** All were settled on 2026-10-21.
- **Approved:**
  - the persona token's shape, as amended (15.1);
  - the social section (15.3);
  - the institutions layer, as a design (15.4);
  - reflexes and composure (15.5);
  - the trait list (15.6);
  - hearing (17).
- **Per agent only** (15.2): settled by the ruling in 16.3.
- **The go** for step 0, and for upgrading both the encoder and the state layer (owner: "yes, we need to upgrade both
  the encoder and the state layer").
- **Future:** the levels' learning constants and aptitude (16.6), and who builds the institutions layer, and when
  (15.4).

## 14. Composability and updates (AGREED 2026-10-21; part of step 1)

The owner asked whether the encoder is composable and can be updated. Today it is only partly:
- **New token types are independent.** Adding one does not touch the others' trained weights, and the trunk's
  parameters never depend on token counts.
- **Seams** let a field exist before its source does.
- **New content** (items, creatures, mods) is read through description vectors, without retraining.

Five rules make the rest updatable. All must hold before anything trains on v3.

1. **Columns are append-only, and old weights are migrated, not discarded.** A new column of an existing token is
   appended just before `marked`, or takes a reserve slot. A trained trunk is migrated by giving the new input
   columns zero weights in the type's first layer. That transformation is function-preserving: the migrated network
   computes exactly what it did, and the new inputs start out ignored until training uses them. This is the input-side
   case of Net2Net's function-preserving network growth (Chen, Goodfellow and Shlens, ICLR 2016). Columns inserted
   mid-list (this week: the entity's `feel_*`, the status flags) were harmless only because nothing trains on v3 yet.
2. **Every checkpoint and every saved state records a schema fingerprint:** the hash of each type's feature names in
   order. On a mismatch, loading either migrates by rule 1 or fails loudly, never silently.
3. **The registry generates the layout.** Declaring a variable once (10.1) makes it appear in queries, views and
   tokens. Today adding a feature touches the schema, the builder, the memory's turn lists and the tests.
4. **Saved agent states migrate by name.** A variable missing from an old save takes its declared default, so an
   agent's memories, relationships and persona survive a schema update.
5. **The trunk changes rarely, and by migration.** The skill modules read the trunk's output, so a retrained trunk
   means retraining every skill. The trunk stays frozen between deliberate updates; an update grows it by rule 1
   rather than training it from scratch; and each update re-fits the skills against a recorded before-and-after
   comparison.

**Rulings for step 1** (owner, 2026-10-21, answering the two questions on step 1's plan):
- **What a variable is.** "Variables are states that are not given in game but we track locally to modify some
  output or just to keep track of." The registry holds only the agent's own state: its traits and emotions now; its
  drives, sleep pressure, levels, places and relationships as their steps come (3 to 9). A value the game gives
  (health, stamina, position, what is in view) is not a variable. It stays a token column of the senses, which the
  fingerprints protect (rule 2).
- **The trait list.** 15.6's list is declared now ("sure"), with the condition: "ensure that we can expand it or
  compress it later". So the list changes by name, in both directions:
  - **Expanding:** a new trait takes its default in a saved state, and its input column starts with zero weights, so a
    trained trunk computes exactly what it did (rule 1).
  - **Compressing:** a trait is merged into a new one by an explicit map (`new = sum_i w_i old_i`) or dropped by
    name. A trait that disappears without either fails loudly. A saved state takes the map's value. In the trunk, the
    merged column's weights are the sum of its sources' weights over the sum of the map's weights. The trunk is then
    unchanged for an agent whose merged traits were equal, and changed for the others, by an amount measured before
    and after (rule 5). A dropped trait's weights are removed: the trunk computes what it did with that trait at 0,
    the population's mean.
- Because migration goes by name, a column's position no longer matters. `marked` stays last.

## 15. Designs for the open decisions (DESIGNED 2026-10-21 from research; not built)

**Approval** (owner, 2026-10-21: "i approve the other 3 designs features if they correspond to what i said before").
Each design was checked against the owner's earlier words:
- 15.3, 15.4, 15.5 and 15.6 correspond, and are APPROVED.
- 15.1 did not fully correspond. It was AMENDED, and the amendment APPROVED (owner: "sure").
- 15.2 was settled by 16.3.

### 15.1 The persona token's shape: the body reads the traits it has a mechanism for (AMENDED and APPROVED 2026-10-21)

**Why amended.** The first draft gave the body only the motor signature and the reflex gains. Earlier the owner had
asked that the traits "modify the other inputs" and "also change reward or some other logical thinking in the low
level model itself", shared "throughout the entire architecture". The amendment does both.

**Design:**
- The persona token carries the traits with a mechanism in the body:
  - the motor signature (15.6);
  - the reflex gains (negative emotionality, vigilance);
  - **the traits that weight a skill's reward terms:** risk taking (the weight on damage taken), effort sensitivity
    (the weight on stamina spent), patience (the weight on time taken). Each skill declares which of its reward terms
    a trait weights.
- The persona token conditions every input token, through the FiLM conditioning built in `model/encoder.py`. That is
  "modify the other inputs".
- Skills are trained with the traits sampled across their range, so one network serves every persona. This is a
  preference-conditioned policy, from multi-objective reinforcement learning: one network conditioned on the weights
  of its reward terms (Abels et al. 2019, Yang et al. 2019).
- Every other trait stays in the state layer for the decision layer and the LLM. It reaches the body only through
  their requests, in the manner of 9.2 (urgency, gait, giving up). Selective reading is a feed choice of the state
  layer (section 3), not a loss of information.

**Certification:**
- Behaviour moves with each trait in the declared direction: a risk taker accepts more damage to land a hit, and an
  effort-sensitive agent sprints less.
- At the trait's extremes, the conditioned skill stays within a set margin of a skill trained for that persona alone.
  Later work reports that one conditioned network can do worse than specialised ones and miss parts of the range
  (Ambadkar et al. 2026, arXiv:2602.07764, citing Xu et al. 2020; read through summaries).

**Why.** The body's skills are distilled from teachers, which is imitation. In imitation learning, extra inputs can
make a policy worse: the policy keys on a correlate instead of the cause. In their driving example it brakes on a
brake indicator rather than on the pedestrian. This is "causal confusion" (de Haan, Jayaraman and Levine, NeurIPS
2019): "access to more information can yield worse performance". Traits with no body mechanism are exactly such
correlates. A reward weight is a mechanism: the trait changes what the skill was trained to want, so the skill has a
reason to read it.

### 15.2 Per agent only: knowledge crosses only by communication

**Design:**
- Each agent's state is private.
- Knowledge reaches another agent only as a told entry, with its source and a confidence set by the trust in that
  source (15.3; provenance, 10.8).
- During training only, a critic may read the whole group's state. This is centralised training with decentralised
  execution, the standard multi-agent pattern (MADDPG, Lowe et al. 2017; recalled), consistent with 11.8.

**Why.** Information asymmetry is what the research needs: markets, deception, rumours and misunderstandings exist
only when agents know different things. It is also the legality rule: no telepathy.

### 15.3 The social section (APPROVED 2026-10-21)

It corresponds to the owner's rulings:
- "a lot more input variables for multiagent interactions";
- feelings toward X "written in a structured manner by the top 2 layers", "short and clean", acting "as some sort of
  memory";
- "encoder support for this to support interaction with other agents".

Interaction facts and reputation are computed from events, not written: they are tied to rules, like the game's
variables.

**Design.** One record per known individual (12.1), held in tiers.
- **Capacity:** about 5 close, 15, 50, then up to 150 known individuals (Dunbar's layers; recalled). The farthest tier
  gives way first, ranked by closeness times recency.

**The relationship**, written by the top two layers:
- **closeness**, as in Humanoid Agents;
- **trust, split three ways** after Mayer, Davis and Schoorman's integrative model (1995): the perceived **ability**
  (can they do it), **benevolence** (do they mean well toward me) and **integrity** (do they keep to principles). The
  agent's **propensity to trust** is its own trait (Falk's `trust`, 15.6). The split matters: a skilled enemy is
  high in ability and low in benevolence, and a well-meaning bumbler the reverse;
- **feelings:** hostility, fear, anger;
- **role** (friend, family, leader, follower, rival, trade partner);
- **groups:** mine and theirs.

**Other entries:**
- **Obligations:** favours owed both ways, and promises with deadlines. Reciprocity acts on them (Falk's positive and
  negative reciprocity, 15.6).
- **Interaction facts** (12.3): counts per kind, the last time, who started it.
- **Reputation:** the aggregate of what others have told me about an individual, weighted by my trust in each teller.
  The literature calls this a reputation model, as opposed to direct trust (Sabater and Sierra 2005, a review).

**What the body reads.** Only the individuals present or relevant, as columns on their entity tokens:
- closeness;
- the three trust parts;
- the feelings;
- the role;
- same group or not;
- the personal-space target distance (Hall's proxemics; recalled).

**What the LLM gets.** Records retrieved by recency, importance and relevance, the memory retrieval of Generative
Agents (Park et al. 2023).

### 15.4 The institutions layer (an environment component, not this lane; APPROVED as a design 2026-10-21, building it is future)

The agents' beliefs about the institutions are agent-side state, as the owner ruled for non-game features (16.5).
The objective record itself cannot live inside any one agent.

**Design.** An objective record outside every agent: ownership, balances, contracts and debts, roles and offices,
group rules.
- It changes only through in-game actions: claim, pay, trade, appoint, vote.
- Each agent's state holds only its beliefs about it, perceived or told. So theft, disputes, fraud and misinformation
  can happen.
- Enforcement is left to the agents, through monitoring and sanctions, so norms can form and fail as in real
  societies.

**The research frame** is Ostrom's institutional analysis (Governing the Commons, 1990). Long-lived commons
institutions shared eight design principles:
1. clearly defined boundaries;
2. rules fitted to local conditions;
3. collective choice of the rules;
4. monitoring;
5. graduated sanctions;
6. cheap conflict resolution;
7. recognition of self-organisation by outside authorities;
8. nesting.

Later work uses them as diagnostics rather than a strict test. The layer should make each principle expressible, so
experiments can switch them on and off.

**Who builds it** is the owner's decision.

### 15.5 Reflexes and composure: modulation, not an off switch (APPROVED 2026-10-21)

It corresponds to the owner's rulings:
- a fixed rule ("i was thinking fixed rule");
- "modifiable by other information like darkness, the agent's internal stats ... not everyone is made equal";
- the LLM in control where a will is, never over physiology (9.1).

**Design.** A reflex's strength is its base gain (the persona) times four factors from the startle literature:
- **fear potentiation:** a reflex is stronger under alarm or fear (fear-potentiated startle);
- **expectation:** a warning cue shortly before the stimulus reduces the reflex. A weak prestimulus 6 to 2,000 ms
  before a startling one reduces the startle or abolishes it (prepulse inhibition);
- **habituation:** the same stimulus repeated draws a weaker response. It is stimulus-specific and recovers after
  minutes without it;
- **context:** a pleasant state attenuates it; a fearful one enhances it (affective modulation of startle).

Then, as ruled (section 8 item 5), visibility (darkness) and the internal stats (fatigue: lapses rise with sleep loss,
Lim and Dinges 2010) shift its threshold and latency.

**Composure.** A top-layer "brace" request acts like an expectation cue: it lowers the reflex gain for a window. In
the future it carries a cost of control (section 9.3). It weakens protective reflexes; it never removes them.

### 15.6 The trait list: measured constructs, about 22 (APPROVED 2026-10-21)

It corresponds to the owner's rulings:
- "unique scalar values and not discrete options";
- "ideally everything should be a combination";
- traits as "differing internal goals or motivations" (lazy, bored: effort sensitivity, boredom proneness).

| Group | Traits | Research | Read by |
|---|---|---|---|
| Economic preferences | patience, risk taking, positive reciprocity, negative reciprocity, altruism, trust | the Global Preferences Survey (Falk et al. 2018): six preferences measured in 80,000 people across 76 countries, varying more within countries than between them | the decision layer, the LLM |
| Motivation | effort sensitivity, impulsiveness, self-control, distractibility, self-efficacy, boredom proneness | Steel's procrastination predictors (2007); boredom proneness has its own scale (recalled) | the decision layer (section 9) |
| Emotional reactivity | negative emotionality, positive emotionality | the Big Five's neuroticism and extraversion (reactivity) | the state layer's rules (gains), the body (reflex gain) |
| Values | four higher-order values: openness to change, conservation, self-enhancement, self-transcendence | Schwartz's basic human values (1992; recalled) | the LLM, the decision layer |
| Motor signature | reaction-time mean, reaction-time tail, motor noise, smoothness, plus vigilance | the ex-Gaussian reaction-time distribution; signal-dependent motor noise (Harris and Wolpert 1998); minimum-jerk smoothness (Flash and Hogan 1985); the vigilance decrement (Mackworth 1948) | the body |

**What this changes from the 29-trait research list:**
- **Dropped as traits:** preferred range, stealth and tactical variability. They have no research behind them as
  individual differences; they become behaviours the decision layer derives from the traits above.
- **Moved to bias:** the strafe-side preference (no source) becomes a motor-signature bias, not a trait.
- **Derived, not stored:** the Big Five and HEXACO remain views computed from the traits, for the LLM's narrative.
- **Sampling:** a population is sampled with the published correlations (Falk et al.; the Big Five's intercorrelations,
  van der Linden et al. 2010) through the latent-axis sampler of `state/persona.py`, re-fitted to this list.

## 16. Time, death, scale, long-term memory, needs and levels (RULED 2026-10-21, later; designs not built)

The owner's answers to eight questions, and the designs that follow from them. Game facts are from Hytale 0.7.2's
`Assets.zip` (Sources).

### 16.1 Game time (RULED: "game time")

- **Every rhythm runs on the game's clock:** sleep pressure, drives, the fading of written entries.
- **The clock:**
  - a game day is 2,880 real seconds, 48 minutes (`Server/GameplayConfigs/Default.json`: `DaytimeDurationSeconds`
    1728, `NighttimeDurationSeconds` 1152);
  - so a game hour is 120 real seconds: the server runs 30 game seconds per real second (86,400 / 2,880; confirmed
    from the server's bytecode and bench receipts, `hytalegym/sim/gen7/mechanics/processing_benches.py` lines 9 and
    10);
  - beds work from game hour 19.5 to 4.79 (`World.Sleep`).
- **Sleep pressure in game hours.** The two-process model's sleep pressure rises with a time constant of about
  18.2 hours awake and falls with about 4.2 hours asleep (Daan, Beersma and Borbély 1984). On the game's clock that is
  about 36 and 8 real minutes.
- **The rules integrate elapsed game time, not ticks,** so a skipped night counts as time slept. The game skips the
  night only when every player sleeps (the owner, 2026-10-21).
- **Each agent sleeps on its own** (RULED 2026-10-21: "make it so that each agent can sleep individually till it is
  day or some certain time"):
  - the agent lies in a bed (the game's `sleeping` state) and stays there until its wake condition: daybreak (the
    game's wake hour, 4.79) or a time the top layers chose ("sleep until hour 7", "nap one game hour");
  - the wake condition is a written entry of the state layer (10.4), set by the decision layer or the LLM. The
    decision layer gets the agent up when it is met; getting up is an ordinary action of the body;
  - if the night is not skipped, the agent sleeps through it in game time. A whole night, 19.5 to 4.79, is 9.29 game
    hours, about 18.6 real minutes;
  - if every player sleeps, the night is skipped. The clock jumps, the jump counts as sleep, and the agent wakes with
    the others;
  - proposed, not ruled: a hit or a loud sound (17) also wakes a sleeper early.
- **Who reads it** (RULED: "this information is mostly relevant for the higher layers like the decision and llm
  layer"):
  - the time of day, sleep pressure and the wake condition are for the decision layer and the LLM;
  - the body reads none of them now, beyond the `sleeping` movement state its self token already has;
  - their effect on the body (slower reactions, lapses) arrives later as a regulator. This is the rule of 15.1.
- **The clock reaches the agent through S14.** The live environment already reads `time_of_day` from the bridge
  (`hytalegym/envs/hytale_env.py`, lines 274 and 1031); the gen-7 gym does not publish it.
- **The simulator has no game clock and no bed sleep** (the coordinator, 2026-10-21; checked here). `sleeping` is
  only a name in the movement-state lists (`jax/world/entities/movement_states.py` line 593, and the v3 combat
  contract); nothing sets it. RULED: "it is fine for now, it is for the model in the future".
  - So the clock-driven rules are off in simulator runs until S14 adds a clock and bed sleep to the gen-7 gym.
  - A skipped night will arrive as a published jump in the clock, which the rule already handles.

### 16.2 Death (RULED: "start with normal death")

**The game's rule** (`Death` and `Respawn` in `Default.json`): the agent respawns as the same person. By default it
loses half its items and 10% of durability, and it returns at one of up to 3 respawn points. Permanent death is a
later environment option, not this lane's.

**The lifecycle (10.7) under normal death:**
- **Reset:** alarm and startle, and the action history. The body's variables come back from the game.
- **Kept:** the persona, the levels, the long-term memory, the relationships, the written entries and the drives
  (hunger carries on).
  - The short-term stores are kept too. They are anchored to world positions, so they stay true after a respawn.
    They reset only at an episode boundary.
- **Added:**
  - the death goes to the event log;
  - "where I died" becomes a place (16.4), with the items dropped there, so the agent can go back for them, as players
    do.

### 16.3 Scale and separate models (RULED: "start with 2 and scale up"; "each model does not have the same trunk nor same context nor need to live in the same transformer")

**Each agent is its own model,** with its own state, its own trunk, and its own process if wanted.
- Batching agents together is an optimisation, never an assumption.
- Nothing crosses between agents except through the game. This settles 15.2.

**So the state layer serves one agent at a time.** Each agent's state records its trunk and the schema fingerprint
that trunk reads (14.2).
- A consequence: skills are fitted to a trunk (14.5), so agents on different trunks need their own skills.

**Measurement** starts at 2 agents and scales up. The 5.27 ms p95 today is for 8 agents in one batch.

### 16.4 Long-term spatial memory (RULED: "there should be something in long term memory, furthermore coordinates and other relative information should be tracked")

**Research.** People know an environment in three kinds (Siegel and White 1975):
- **landmarks:** single places;
- **routes:** sequences of places and turns;
- **survey knowledge:** a map-like layout.

The original model ordered them as stages. Later work found survey knowledge can form from the first exposure
(Montello 1998). Robotics' topological maps likewise hold places and the paths between them (Kuipers 2000).

**Design:** a state-layer store beside the short-term stores.
- **Places** hold landmarks and survey knowledge. Each has:
  - its kind: home, base, resource site, danger site, where I died, a told place;
  - a label, its world coordinates and extent;
  - first and last visit, and the visit count;
  - its source (seen, told, inferred) and the confidence in it.
- **Routes** run from place to place, with:
  - the travel time measured;
  - how often and when last travelled;
  - what happened on them (hits, hazards: from the event log).
- **Relative information,** computed on reading:
  - each place's displacement in the body frame, its distance and bearing (the `where` block);
  - the displacement between places;
  - the cost of the best known route.
- **What enters it (consolidation):** a place enters when the top layers name it, when the agent returns to it, or
  when something significant happens there (a hit, a death, a find). The short-term stores keep everything recent;
  this keeps what matters.
- **What it keeps:**
  - it persists across death and sessions and is saved with the agent;
  - confidence fades with the time since the last visit;
  - its capacity is a few hundred places (configurable); the least used and oldest give way.
- **Who reads it:**
  - the body: the few relevant places, as tokens (11.6);
  - the decision layer: the graph, for planning routes;
  - the LLM: the named places, as words.

  Meanings ("dangerous", "ours") stay the decision layer's labels (section 5).
- **Coordinates** are exact for now, from the agent's own position (S1). Human path-integration error could become a
  regulator later.

### 16.5 Needs on the agent's side (RULED: "we can add these features into state if needed, they don't have to be game side rather agent side")

- Hytale 0.7.2 has no hunger. Food is an instant heal (`Food_Bread.json` applies `Food_Instant_Heal_T1`).
- **Hunger** is a state-layer drive with its own rule: it rises on game time and falls when the agent eats.
- **Eating is detected on the agent's side**, with no simulator ask: its own consume action (the action history),
  followed by that food item's count dropping in its inventory.
- **How much a food satisfies** comes from the item registry, from the food's tier in its effect id (our estimate).
  The rise rate is also our estimate [E].
- Other needs are added the same way, when needed.

### 16.6 Levels (RULED: "there should be some leveling or skills that go up after doing X thing Y times")

Hytale 0.7.2 has no levels or experience: no such file exists under `Server/`. So levels are agent-side.

**Counters.** The state layer counts each activity the agent completes:
- distance walked, sprinted, swum and climbed;
- blocks mined and trees felled;
- items crafted, per station;
- blocks placed;
- hits landed, and blocks or parries made, per weapon family;
- trades.

Each count comes from the agent's own action plus the outcome it observed (an item gained, a block gone). Hits landed
wait on S11.

**The curve.** Individual learners improve along an exponential curve. The power law appears only in averaged data
(Heathcote, Brown and Mewhort 2000: 7,910 learning series from 475 people). So:
- proficiency after N repetitions is `p = 1 - exp(-N / tau)`, in [0, 1];
- level L of n is reached at `N_L = -tau * ln(1 - L / n)`.

Early levels come fast and later ones slowly, as in games.

**Decay: none for now.** Skill loss is real but slow:
- half the gains are lost after about 6.5 months for accuracy and 13 for speed (Psychological Bulletin 2025, abstract);
- physical and speed tasks decay least (Arthur et al. 1998).

Even on game time, that is hundreds of real hours.

**Effect.** RULED, answer 7: "we will do some scalar output based on some relative skill/state info".
- A level never changes a game number (damage, mining speed). The game has no levels, and skills must port to the
  live game.
- It acts on the agent's side: a scalar computed from the proficiency and the state (fatigue, alarm, darkness)
  scales the body's outputs. That is a regulator (section 4, future).
- The body reads the proficiencies as input from step 7.
- Individual differences in the senses are not separate traits; they come through these scalars.

**FUTURE** (owner: "that is there for future dont worry about it now"): tau for each activity, and whether the
persona sets an aptitude for each group of activities.

### 16.7 Parked

- Human play data for calibration: "irrelevant for now".
- Permanent death (16.2).

## 17. Hearing (APPROVED 2026-10-21: "another thing is hearing", "needed for peripheral knowledge and other stuff"; not built)

### 17.1 What exists

**The game.** It has 1,569 sound events (`Server/Audio/SoundEvents/` in `Assets.zip`). Each inherits an attenuation
preset that sets how far it carries (`MaxDistance`) and where it starts to fade (`StartAttenuationDistance`):

| Preset | Carries to (blocks) | Starts to fade at |
|---|---:|---:|
| ExtremelyQuiet | 5 | 1 |
| VeryQuiet | 10 | 2 |
| Quiet | 15 | 4 |
| Moderate | 25 | 8 |
| Loud | 45 | 15 |
| VeryLoud | 70 | 25 |

A block hit, for example, is Quiet (`SFX_Bone_Hit.json`, parent `SFX_Attn_Quiet`).

**The simulator.** An audio observation already exists (`hytalegym/jax/world/audio/_types.py`, `AudioObservation`):
- the sound event, its category and family;
- the source's position and its position relative to the listener, the distance and the maximum distance;
- the source's network id;
- the volume.

Nothing reads it; that is S12. Live, the bridge would have to publish it too.

### 17.2 The rule: hear what a player would hear, the way a person hears it

1. **Whether it is heard:** only within the sound's maximum distance, at the game's attenuation, so the agent never
   hears more than a player would. During a louder sound, a much quieter one is not heard (masking; recalled; the
   margin is our estimate).
2. **What:** the sound's kind, as a description vector from a sound registry built from `Assets.zip`, like the item
   and creature registries (`../perception/registry.py`): footsteps on a material, a block hit or broken, a weapon, a
   creature's voice, an item used. This lane's to build.
3. **Direction:** the bearing in the body and head frames, blurred as a human ear blurs it: finest straight ahead,
   coarser to the side.
   - The smallest change in direction a listener can detect is about 1 degree in front (Mills 1958).
   - At the side it is about 3 degrees (1.1 and 3.1 degrees in a 2021 replication).
   - Elevation is heard coarsely (recalled).
   - The blur's constants are our estimates.
4. **Distance:** judged from loudness and compressed, as people judge it: near sources are overestimated and far
   ones underestimated, along a compressive power function (Zahorik, Brungart and Bronkhorst 2005). The token carries
   the loudness and the judged distance, never the true one.
5. **Who:** unknown unless the agent sees a source at that bearing. The sound is then linked to that entity's token;
   otherwise it is "something". The simulator's source id is used only for that check, and only when the source is
   in view.
6. **When and how often:** the sound's age. Repeats of the same kind from the same direction merge into one entry
   with a rate (footsteps).

### 17.3 How the state shapes hearing

This is the scalar of answer 7, from the agent's skill and state.
- **Alarm sharpens hearing.** In Sokolov's orienting response, one role of orienting is raising sensory sensitivity.
- **Fatigue dulls it.** Sleep loss impairs simple attention most (Lim and Dinges 2010).

The scalar's form is a regulator (section 4, future); the hearing threshold that takes it exists from the start.

### 17.4 The orienting reflex

A new, unexpected sound turns the head toward it, and the response fades as the same sound repeats (habituation).
This is Sokolov's orienting response; the head turn toward the source is one of its components.
- It joins the reflexes of 15.5: a fixed rule that issues a brief look-toward request.
- Its strength is set by novelty, habituation, alarm and the persona's vigilance.
- A brace or focus request from the top layers damps it.
- A loud, sudden sound gives the startle of 15.5 instead.

### 17.5 Memory

A heard source enters the salient store as a heard entry: its kind, a judged position with an uncertainty radius,
and the time. Its provenance is "heard" (10.8). When the source is seen, the entry merges with the seen entity. So the
agent can keep track of a foe behind a wall.

### 17.6 The agent's own noise

The agent hears itself. The self token carries how loud its own recent sounds were (footsteps, mining, combat), so a
skill can learn to move quietly. Whether sneaking quietens footsteps in Hytale is not checked.

### 17.7 Speech

Spoken words are sounds and follow the same rule.
- **Range:** speaking quietly, normally or loudly maps to the game's presets: VeryQuiet 10, Moderate 25 and VeryLoud
  70 blocks (our mapping).
- **Content** goes to the LLM.
- **The body** gets "someone is talking to me" and its direction, to turn and face them (12.4).
- Hytale's text chat is not positional (inferred), so positional speech between agents needs an environment
  component. It is held with the social asks (section 12).

### 17.8 The token

A new type, `sound`, with:
- the `where` block, built from the judged position;
- the kind vector;
- loudness, age and rate;
- whether the source is known, and the link to its entity;
- the confidence.

It has about 8 to 16 slots. As a new type it touches no trained weights (14). Until S12 publishes sounds, it reads
zero with `known` 0.

### 17.9 Asks, and certification

**Asks to the coordinator,** sent on 2026-10-21 once this section was approved:
- the attenuation curve between the start and maximum distances;
- whether walls muffle sound in the game (occlusion);
- the bridge publishing the sound events near each agent (S12, live);
- positional speech.

**Certification:**
- An unseen block hit 10 blocks behind is heard, roughly placed and remembered; the same hit at 20 blocks is not
  (Quiet carries 15).
- A new sound turns the head; its tenth repeat does not.
- Judged distance follows the compression.
- Nothing privileged reaches the token: the source's id only when the source is seen.

## Sources

- Steel, P. (2007). The nature of procrastination. Psychological Bulletin 133(1), 65-94.
- Hagger, M. S., et al. (2016). A multilab preregistered replication of the ego-depletion effect. Perspectives on
  Psychological Science 11(4); the 2021 multisite preregistered test (k = 36, N = 3,531), Psychological Science.
- Shenhav, A., Botvinick, M. M., and Cohen, J. D. (2013). The expected value of control. Neuron 79(2), 217-240.
- Gollwitzer, P. M., and Sheeran, P. (2006). Implementation intentions and goal achievement: a meta-analysis.
  Advances in Experimental Social Psychology 38, 69-119.
- Borbély, A. A. The two-process model of sleep regulation (1982; the 2022 retrospective in the Journal of Sleep
  Research).
- Lim, J., and Dinges, D. F. (2010). A meta-analysis of the impact of short-term sleep deprivation on cognitive
  variables. Psychological Bulletin 136(3), 375-389.
- Inzlicht, M., and Schmeichel, B. J. (2012). What is ego depletion? Toward a mechanistic revision of the resource
  model of self-control (recalled, not re-checked).

- Gat, E. (1998). On Three-Layer Architectures. In Kortenkamp, Bonasso and Murphy (eds.), Artificial Intelligence
  and Mobile Robots, AAAI Press, 195-210.
- Sun, R. CLARION: the motivational subsystem (drives, from which goals are set) and the meta-cognitive subsystem
  (monitoring and regulation); summarised from tutorial material, not the primary papers.
- Wang, Chiu and Chiu (2023). Humanoid Agents: Platform for Simulating Human-like Generative Agents. EMNLP 2023 System
  Demonstrations, arXiv:2310.05418.
- LeDoux, J. (1996). The Emotional Brain (the dual pathway: thalamus to amygdala directly, and through the cortex).
- Easterbrook, J. A. (1959). The effect of emotion on cue utilization and the organization of behavior.
  Psychological Review 66(3).
- Park et al. (2023). Generative Agents: Interactive Simulacra of Human Behavior. arXiv:2304.03442.
- Dunbar, R. (social brain hypothesis, the layers of about 5, 15, 50 and 150 relationships); Hall, E. T. (1966). The
  Hidden Dimension (proxemics); Helbing, D., and Molnár, P. (1995). Social force model for pedestrian dynamics.
  Physical Review E 51. All three recalled, not re-checked.
- Falk, A., Becker, A., Dohmen, T., Enke, B., Huffman, D., and Sunde, U. (2018). Global evidence on economic
  preferences. Quarterly Journal of Economics (working version: IZA DP 9504, 2015).
- de Haan, P., Jayaraman, D., and Levine, S. (2019). Causal Confusion in Imitation Learning. NeurIPS,
  arXiv:1905.11979.
- Chen, T., Goodfellow, I., and Shlens, J. (2016). Net2Net: Accelerating Learning via Knowledge Transfer. ICLR,
  arXiv:1511.05641.
- Mayer, R. C., Davis, J. H., and Schoorman, F. D. (1995). An integrative model of organizational trust. Academy of
  Management Review 20(3), 709-734.
- Sabater, J., and Sierra, C. (2005). Review on computational trust and reputation models. Artificial Intelligence
  Review 24, 33-60.
- Ostrom, E. (1990). Governing the Commons (the eight design principles; read through summaries and a 2010 review of
  91 studies).
- Daan, S., Beersma, D. G. M., and Borbély, A. A. (1984). Timing of human sleep: recovery process gated by a
  circadian pacemaker. American Journal of Physiology 246(2), R161 onward. The time constants 18.2 h and 4.2 h are as
  a model implementation gives them, not re-read in the paper.
- Heathcote, A., Brown, S., and Mewhort, D. J. K. (2000). The power law repealed: the case for an exponential law of
  practice. Psychonomic Bulletin and Review 7(2), 185-207 (abstract and summaries).
- Arthur, W., Bennett, W., Stanush, P. L., and McNelly, T. L. (1998). Factors that influence skill decay and
  retention: a quantitative review and analysis. Human Performance 11, 57-101; and "Procedural skill retention and
  decay: a meta-analytic review", Psychological Bulletin 2025 (PubMed 40455501). Both read at abstract level.
- Siegel, A. W., and White, S. H. (1975), landmark, route and survey knowledge; Montello, D. R. (1998), survey
  knowledge from first exposure. Both through secondary accounts. Kuipers, B. (2000). The Spatial Semantic Hierarchy.
  Artificial Intelligence 119 (recalled).
- Hytale 0.7.2 facts in section 16 are from `Assets.zip`: `Server/GameplayConfigs/Default.json` (`World`, `Death`,
  `Respawn`), `Server/Item/Items/Food/Food_Bread.json`, and the absence of any experience or levelling file under
  `Server/`. The vanilla night skip is inferred from the descriptions of server mods that change it.
- Abels, A., Roijers, D. M., Lenaerts, T., Nowé, A., and Steckelmacher, D. (2019), the Conditioned Network for
  multi-objective deep RL (ICML); Yang, R., Sun, X., and Narasimhan, K. (2019), envelope updates (NeurIPS). Both
  through summaries; titles recalled. Ambadkar et al. (2026), arXiv:2602.07764, on the brittleness of single
  preference-conditioned networks (summary).
- Mills, A. W. (1958), the minimum audible angle, about 1 degree in front; the 2021 replication (1.1 and 3.1
  degrees) in Frontiers in Psychology (doi 10.3389/fpsyg.2021.656052). Zahorik, P., Brungart, D. S., and Bronkhorst,
  A. W. (2005). Auditory distance perception in humans: a summary of past and present research. Acta Acustica united
  with Acustica 91 (abstract; venue recalled); the 2016 review by Kolarik et al. (PMC4744263). Sokolov's orienting response: through
  reviews and abstracts. Masking: recalled.
- Hytale's sound presets and events: `Assets.zip`, `Server/Audio/SoundEvents/` (`SFX_Attn_*.json`,
  `BlockSounds/Bone/SFX_Bone_Hit.json`).
- The startle findings (prepulse inhibition, habituation, fear potentiation, affective modulation): from a 2020 review
  of prepulse inhibition (PMC7563436), a habituation protocol, and Koch and Fendt (2003); read at abstract level.
- Recalled, not re-checked this session: Lowe et al. (2017) MADDPG; Schwartz (1992) basic human values; Harris and
  Wolpert (1998); Flash and Hogan (1985); Mackworth (1948); boredom proneness's scale.
- The emotion and personality literature in section 6, as relayed by the 2026-10-21 research surveys: the papers'
  existence was checked by search; the depth of reading is stated where it matters.
