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
  toward others will be written by the decision layer or the LLM through the structured writes (`written.py`, step 2,
  built 2026-10-22); today the only written field is `remember`, the decision layer's mark on an entity or a place.

The body reads the state as `skill_tokens_v3` tokens (the schema is `../contract/tokens_v3.py`) through its trunk
(`../model/encoder.py`), selectively or whole. The `persona` and `affect` tokens' columns are the body-read variables,
generated from the registry.

Moved here from `../perception/` on 2026-10-21 (owner: "move all that stuff that we already have built to its own
folder and rename it properly and add documentation"). The content registry and the v1/v2 adapters stay in
`../perception/`; this layer reads the registry from there.

| Module | Lines | What it holds |
|---|---:|---|
| `layer.py` | 324 | One tick of the state layer. `AgentState`: the action history, the stores, the variables (float32[B, V]), the affect rules' working state, the trunk's latent memory, which the trunk call fills and this module only carries, and the written entries (`written`, step 2). `initial_state`; `step(observation, state, last_action, reset, writes, ...)`, which updates the state from the senses and returns the body's token sets, and keeps each entity token row's id (`entity_ids`, perceived or remembered) so a request can point at it, and each important row's store slot (`object_slots`), and carries the trunk's background summary (`background`) beside its latent memory (2026-10-22); `save` and `load` (one `.npz`, with the variables' names and the token layout, so a save migrates by name, the written entries' fields included; `../docs/VIRTUAL_HUMAN.md` section 14). |
| `written.py` | 297 | The top two layers' structured writes (`../docs/VIRTUAL_HUMAN.md` section 19, step 2, 2026-10-22). A `Write` (writer, field, value, and an entity id or a world position as its key) is checked on the host by `lower` against the registry's declared fields (`../contract/variables.py` `ENTITY_ENTRIES`, `PLACE_ENTRIES`) and refused with a reason (unknown field, wrong writer, out of range, bad key, two keys, too many writes this tick). `apply` runs at the start of each tick: up to 8 writes per agent in order (a later write wins) into 16 entity entries and 16 place entries, each stamped with its writer and tick. An entry never fades and is never evicted (the owner's ruling, 19.7): a write to a full table is refused (`Written.applied`), and writing every field back to its default ejects it. `tags` gives the live `remember` entries to the stores as their tags; `entity_entry` and `place_entry` read an entry by key. |
| `events.py` | 329 | The event log and the activity counters (`../docs/VIRTUAL_HUMAN.md` section 21, step 4, 2026-10-22). A ring of 32 events per agent (damage taken and dealt, items gained and lost, eating, request outcomes), each with its tick, the other party, an amount, a detail (item or verb), where it happened, and an **observed** or **inferred** mark (source monitoring, 21.5: an attributed hit is inferred until the simulator says who hit whom, S11). The detectors compare last tick's legal observation with this tick's; a request outcome is logged on any tick. The counters are lifetime totals (distance walked, sprinted, swum and climbed; damage dealt and taken; items gained; food eaten), published each tick to the registry's `self` variables (`rule`, not read by the body yet). The log is not memory (the owner's ruling): the memories keep what they need from it. |
| `places.py` | 315 | The long-term spatial memory (`../docs/VIRTUAL_HUMAN.md` section 22, step 5, 2026-10-22). Up to 128 places per agent (kinds: stay, danger, death, find, marked; a strength, first and last visit, visits) and 256 routes between them (best and last travel time, count). Places form from a stay (3 blocks for 10 s, stay-point detection), a hit taken (strength from the alarm), a respawn (where I died), an item gained, and the decision layer's live place marks (a marked place outlives its mark). Revisits restore a place; unmarked places fade, routes more slowly; the weakest unmarked place gives way when full; marked places never fade nor go (the owner's rulings, 22.5). It survives a respawn and save and load, and restarts at an episode boundary. `relative` gives each place in the body frame for step 7's tokens. Needs the agent's position (a perception channel or the host's own pose). |
| `registry.py` | 60 | The variables, by name: `get`, `take`, `section` and `put` on the array, `defaults`, and `migrate` (a save made under another variable list: a new variable takes its default; a gone one must be merged by a map or dropped by name). |
| `senses.py` | 584 | Stateless: the legal observation, plus the actor's legal views (block and recipe candidates, light rows), as this tick's rows of every type. One builder per type: self, kit, status, entity, projectile, hazard, geometry, block candidate, inventory, recipe. It also holds `pack` and `legal_tokens_v3` (one tick without memory). It applies the view sector where the gym does not, and reads nothing privileged. |
| `frames.py` | 75 | Space, defined once. `where` is every spatial token's block: position in the body frame, log distance, bearing in the body and head frames, elevation against the head's pitch. `world_to_body` and `body_to_world` are the gym's frame. |
| `action_history.py` | 72 | What the body last did: for each gen-7 action head, the last choice, whether it changed, and how many ticks it has been held (the `act` tokens). In neuroscience terms, an efference copy. |
| `memory.py` | 322 | What the body remembers of where things are. The spatial map holds geometry cells, kept by recency (32 nearby cells since 2026-10-22, the background's cache beside the trunk's summary slots). The salient store holds entities by id and blocks by cell, kept by tag, then salience, then recency, each with its object latent (`docs/VIRTUAL_HUMAN.md` section 18: kept while the slot holds the same thing, zero for a new one). Both are anchored to the world by the agent's position, and written in one parallel pass a tick (2026-10-22: 7 to 13 times faster than the serial write, equal to it slot for slot when there is room; when candidates compete, the most salient win and pinned or refreshed slots stay). `object_slots` maps each important token row to its slot. Entity velocity and facing, and cell edges, are stored in world axes and turned into the current body frame on reading. Its tags are the decision layer's `remember` entries (`written.py`). |
| `persona.py` | 121 | Identity: the registry's self section (`TRAITS`), 15.6's 23 traits and the motor signature's bias, each in [-1, 1] (APPROVED, `../docs/VIRTUAL_HUMAN.md` section 15.6). `sample_personas` draws correlated populations; `big_five` is a derived narrative view. The body reads 10 of them (15.1). |
| `affect.py` | 385 | A proposal for the emotion variables: hand-coded rules for fear, anger, startle, valence, coping, a three-axis mood, and feelings toward up to 16 entities. Their constants are estimates. Since step 3 (2026-10-22, `../docs/VIRTUAL_HUMAN.md` 20.7) the startle follows 15.5: fear potentiation, prepulse inhibition (a warning 30 to 500 ms before), habituation, the unseen doubling and darkness; a respawn restarts alarm and startle. It also exposes the entity a hit was credited to (`attributed`, for the event log). To reshape under the rulings: feelings toward others become written by the top two layers, and a fast reflex response stays a fixed rule (open question 1). |

**One tick (`layer.step`):**
1. the action history takes last tick's action;
2. the senses read this tick's rows;
3. the affect rules appraise them, and publish the emotional state to the variables;
4. the event log registers this tick's changes (a respawn logs a death), and its counters are published to the
   variables (`events.py`);
5. the top layers' writes are applied, and the live `remember` entries become the stores' tags;
6. the stores take this tick's rows;
7. the long-term places take this tick's stays, hits, deaths, finds and marks, and the routes between them
   (`places.py`);
8. the stores give back what is remembered but not perceived;
9. each entity token gets the feelings toward it;
10. everything is packed into the token sets, with the persona and affect tokens (read from the variables).

**Provenance** (`../docs/VIRTUAL_HUMAN.md` 10.8 and 20.2): every value says where it came from, by construction.

| Value | How its origin is marked |
|---|---|
| A token row | `remembered` 0 (perceived this tick) or 1 (from the stores), with its `age` |
| A written entry | the writer (`decision` or `llm`) and the tick of its last write (`written.py`) |
| An event | observed or inferred (`events.py`): an attributed hit or eating is inferred until the simulator says so |
| A variable | its source in the registry: `birth`, `rule` or `written` (`../contract/variables.py`) |

"Told" (from another agent) arrives with communication (section 12). `test_sa_state_step3.py` pins the first three.

**Respawn** (16.2): `layer.step(..., respawn=)` restarts the alarm, the startle and the action history on those lanes;
everything else is kept, since the agent is the same person after a normal death. `reset` stays the episode boundary.

**Cost.** The full tick plus the trunk with conditioning measured 5.27 ms p95 for 8 agents, against a 15 ms gate
(`../docs/VIRTUAL_HUMAN.md` section 7).

**Gates:**
- `test_sa_tokens_v3.py`: frames, schema, the trunk, latent memory, conditioning.
- `test_sa_state_senses.py`: decoding and legality, E5 and E6.
- `test_sa_state_memory.py`: the stores, the tick, and the persona and feelings on the tokens; a remembered place and
  a remembered entity given as writes.
- `test_sa_state_written.py`: the writes: every refusal, order, the full table, ejecting, reset, the derived tags,
  and save and load (a save from before step 2 loads with no entries).
- `test_sa_state_events.py`: the event log's detectors, the ring, reset, jit, and through `layer.step` (counters in the
  variables, request outcomes, save and load).
- `test_sa_state_step3.py`: each startle factor's direction, respawn, and provenance.
- `test_sa_state_places.py`: each way a place forms, merging, fading and revisits, routes, eviction with marked
  places kept, reset, the body-frame read, and through `layer.step` a death place and save and load.
- `test_sa_state_affect.py`: the emotion rules, A1 to A7 and A4b (a hit landing as the attacker enters recovery).
- `test_sa_state_memory_write.py`: the parallel write against the serial one it replaced, and its rules when
  candidates compete. `test_sa_object_slots.py`: object latents follow their thing's identity through the tick.
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
