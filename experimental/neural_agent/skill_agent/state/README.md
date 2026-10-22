# skill_agent/state/ — the state layer

Everything about the agent, held in memory and read by every layer (`../docs/VIRTUAL_HUMAN.md` section 3, agreed with
the owner on 2026-10-21: "this is where literally everything about the agent lives"; "think of it like the context
layer and it directly affects the body layer").

- **Its world and body part** comes from the senses: this tick's legal observation, and the agent's own stores of
  what it has seen.
- **Its variables** are what the agent tracks on its own side (owner, 2026-10-21: "states that are not given in game
  but we track locally to modify some output or just to keep track of"). They are declared once in
  `../contract/variables.py` and read and written by name through `registry.py`. Today they are the persona (its self
  section) and the emotional state the affect rules publish (its inner section, a proposal, `affect.py`). Feelings
  toward others will be written by the decision layer or the LLM (step 2), not built yet.

The body reads the state as `skill_tokens_v3` tokens (the schema is `../contract/tokens_v3.py`) through its trunk
(`../model/encoder.py`), selectively or whole. The `persona` and `affect` tokens' columns are the body-read variables,
generated from the registry.

Moved here from `../perception/` on 2026-10-21 (owner: "move all that stuff that we already have built to its own
folder and rename it properly and add documentation"). The content registry and the v1/v2 adapters stay in
`../perception/`; this layer reads the registry from there.

| Module | Lines | What it holds |
|---|---:|---|
| `layer.py` | 169 | One tick of the state layer. `AgentState`: the action history, the stores, the variables (float32[B, V]), the affect rules' working state, and the trunk's latent memory, which the trunk call fills and this module only carries. `initial_state`; `step(observation, state, last_action, reset, tags, ...)`, which updates the state from the senses and returns the body's token sets; `save` and `load` (one `.npz`, with the variables' names and the token layout, so a save migrates by name; `../docs/VIRTUAL_HUMAN.md` section 14). |
| `registry.py` | 60 | The variables, by name: `get`, `take`, `section` and `put` on the array, `defaults`, and `migrate` (a save made under another variable list: a new variable takes its default; a gone one must be merged by a map or dropped by name). |
| `senses.py` | 572 | Stateless: the legal observation, plus the actor's legal views (block and recipe candidates, light rows), as this tick's rows of every type. One builder per type: self, kit, status, entity, projectile, hazard, geometry, block candidate, inventory, recipe. It also holds `pack` and `legal_tokens_v3` (one tick without memory). It applies the view sector where the gym does not, and reads nothing privileged. |
| `frames.py` | 75 | Space, defined once. `where` is every spatial token's block: position in the body frame, log distance, bearing in the body and head frames, elevation against the head's pitch. `world_to_body` and `body_to_world` are the gym's frame. |
| `action_history.py` | 72 | What the body last did: for each gen-7 action head, the last choice, whether it changed, and how many ticks it has been held (the `act` tokens). In neuroscience terms, an efference copy. |
| `memory.py` | 267 | What the body remembers of where things are. The spatial map holds geometry cells, kept by recency. The salient store holds entities by id and blocks by cell, kept by tag, then salience, then recency. Both are anchored to the world by the agent's position. Entity velocity and facing, and cell edges, are stored in world axes and turned into the current body frame on reading. Tags come from the middle layer. |
| `persona.py` | 120 | Identity: the registry's self section (`TRAITS`), 15.6's 23 traits and the motor signature's bias, each in [-1, 1] (APPROVED, `../docs/VIRTUAL_HUMAN.md` section 15.6). `sample_personas` draws correlated populations; `big_five` is a derived narrative view. The body reads 10 of them (15.1). |
| `affect.py` | 312 | A proposal for the emotion variables: hand-coded rules for fear, anger, startle, valence, coping, a three-axis mood, and feelings toward up to 16 entities. Their constants are estimates. To reshape under the rulings: feelings toward others become written by the top two layers, and a fast reflex response stays a fixed rule (open question 1). |

**One tick (`layer.step`):**
1. the action history takes last tick's action;
2. the senses read this tick's rows;
3. the affect rules appraise them, and publish the emotional state to the variables;
4. the stores take them, with the middle layer's live tags;
5. the stores give back what is remembered but not perceived;
6. each entity token gets the feelings toward it;
7. everything is packed into the token sets, with the persona and affect tokens (read from the variables).

**Cost.** The full tick plus the trunk with conditioning measured 5.27 ms p95 for 8 agents, against a 15 ms gate
(`../docs/VIRTUAL_HUMAN.md` section 7).

**Gates:**
- `test_sa_tokens_v3.py`: frames, schema, the trunk, latent memory, conditioning.
- `test_sa_state_senses.py`: decoding and legality, E5 and E6.
- `test_sa_state_memory.py`: the stores, the tick, and the persona and feelings on the tokens.
- `test_sa_state_affect.py`: the emotion rules, A1 to A7 and A4b (a hit landing as the attacker enters recovery).
- `test_sa_state_registry.py`: the registry and the composability rules (section 14): only agent-side variables, the
  trait list, queries, migration of a saved state by name, and a trunk that grows exactly or is compressed by a map.
- `test_sa_tandem_v3.py`: this layer and the trunk together on real duel observations (`../docs/PERCEPTION.md` E11):
  memory, conditioning, padding, reset, save and load, migration, and the generalist head on the result.

The certification table is `../docs/PERCEPTION.md` section 10.2.

**Not built yet.** The plan of work is `../docs/VIRTUAL_HUMAN.md` section 13. Section 10 covers the rest of this
layer: registry, queries, views, structured writes, fixed rules, event log, lifecycle, provenance, recording. Section 11
covers the encoder's remaining inputs, and section 12 the proposed social section. Section 16 holds the rulings on
game time, normal death, one model per agent, the long-term spatial memory, agent-side needs and levels; section 17
designs hearing. Regulators and expression are future.
